#!/usr/bin/env python3
"""Phase 0 checkpoint-backed *isolated layer* diagnostics for WanGP ConvRot.

Uses a handful of actual INT8 checkpoint tensors and stored FP32 row scales.
Input activations are explicitly synthetic; this cannot determine actual
model dispatch or denoising performance. No permanent expanded weight sets.

R1: FP32 rotation on the *BF16-rounded* input + FP32 quantized-weight GEMM
    with scale and bias applied in FP32, then BF16 rounding (also retain FP32).
R0-leaf: attempt the real WanGP QLinearInt8ConvRot's leaf F.linear path with
    Int8ConvRotWeightTensor; record failure rather than silently substituting.
Surrogate: direct BF16-dequantized F.linear, if R0-leaf is unavailable.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open

from phase0_probe import github_source_rotation

FILE = Path("/cache/ltx25/checkpoints/ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors")
OUT = Path("/cache/ltx25/results")
# Four distinct high-frequency checkpoint weight families, bounded test sizes.
FAMILIES = ((2048, 2048), (4096, 4096), (2048, 4096), (4096, 16384),
            (16384, 4096), (2048, 8192), (8192, 2048), (4096, 2048))
MS = (1, 16)
TRIALS = 5
SAFE_FREE_BYTES = 1_500_000_000


def describe_tensor(t):
    return {"shape": list(t.shape), "dtype": str(t.dtype), "strides": list(t.stride())}


def stats(y, y_ref):
    a = y.float()
    r = y_ref.float()
    delta = a - r
    abs_err = delta.abs()
    scaled = abs_err / r.abs().clamp_min(1.0)
    yf = a.flatten()
    rf = r.flatten()
    result = {
        "p99_abs": float(torch.quantile(abs_err.flatten(), 0.99).item()),
        "max_abs": float(abs_err.max().item()),
        "p99_normalized": float(torch.quantile(scaled.flatten(), 0.99).item()),
        "max_normalized": float(scaled.max().item()),
        "rmse": float(torch.sqrt(torch.mean(delta.square())).item()),
        "cosine": float(F.cosine_similarity(yf.unsqueeze(0), rf.unsqueeze(0)).item()),
        "nonfinite": int((~torch.isfinite(a)).sum().item()),
    }
    return result


def try_actual_qweight(q, s):
    try:
        from shared.qtypes.int8_convrot import (
            Int8ConvRotWeightTensor, _QINT8_CONVROT_QTYPE,
        )
        wrapped = Int8ConvRotWeightTensor.create(
            _QINT8_CONVROT_QTYPE, 0, q.shape, q.stride(), q, s,
            compute_dtype=torch.bfloat16,
        )
        return wrapped, None
    except Exception as exc:
        return None, {"exception": type(exc).__name__, "message": str(exc)[:500]}


def benchmark_cuda(call, *, n=TRIALS):
    # No profiling of checkpoint loading or R1; sync only at trial boundaries.
    for _ in range(2):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(n):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return {"median_ms": statistics.median(samples), "trial_ms": samples}


def tested_shapes(f):
    families = {}
    for key in f.keys():
        if not key.endswith(".weight"):
            continue
        shape = f.get_slice(key).get_shape()
        if len(shape) != 2 or tuple(shape) not in FAMILIES:
            continue
        if f.get_slice(key).get_dtype() != "I8":
            continue
        base = key[:-7]
        skey = base + ".weight_scale"
        if skey not in f.keys() or f.get_slice(skey).get_dtype() != "F32":
            continue
        families.setdefault(tuple(shape), (base, key, skey))
    return [families[family] for family in FAMILIES if family in families]


def test_one(f, triple, rotate, free_bytes):
    base, qkey, skey = triple
    q_cpu = f.get_tensor(qkey)
    s_cpu = f.get_tensor(skey)
    result = {
        "layer": base, "N": q_cpu.shape[0], "K": q_cpu.shape[1],
        "group_size_candidate": 256,
        "scale_shape": list(s_cpu.shape), "scale_dtype": str(s_cpu.dtype),
        "weight_dtype": str(q_cpu.dtype),
        "activation_source": "synthetic BF16 (not real LTX inputs)",
        "R0_kind": "WanGP Int8ConvRotWeightTensor leaf F.linear if importable",
        "actual_runtime_dispatch_observed": False,
        "scratch_cap_bytes": 128 * 1024 * 1024,
    }
    if free_bytes < SAFE_FREE_BYTES:
        result.update(status="SKIPPED", reason="Insufficient free VRAM for isolated diagnostics")
        return result
    try:
        assert s_cpu.numel() in (1, q_cpu.shape[0])
        with torch.inference_mode():
            q = q_cpu.to(device="cuda", non_blocking=False)
            s = s_cpu.to(device="cuda", dtype=torch.float32).reshape(-1)
            if s.numel() == 1:
                s = s.expand(q.shape[0])
            weight, wrap_error = try_actual_qweight(q, s[:, None].contiguous())
            result["real_wrapper_init"] = "PASS" if weight is not None else "ERROR"
            if wrap_error:
                result["real_wrapper_error"] = wrap_error

            # Bias is optional and omitted for this first pass. It is explicitly
            # NOT claiming full-model or bias-on-path validation.
            per_m = []
            torch.manual_seed(20261112)
            for m in MS:
                x = torch.randn(m, q.shape[1], device="cuda", dtype=torch.float32).bfloat16()
                # FP32 reference must use exactly the BF16 input seen by R0.
                xr32 = rotate(x.float(), 256)
                c_ref = (xr32 @ q.float().T) * s[None, :]
                ref_bf16 = c_ref.bfloat16()
                xr_bf16 = rotate(x, 256)
                item = {"M": m, "input_dtype": str(x.dtype), "R1_dtype": str(c_ref.dtype),
                        "reference_nonfinite": int((~torch.isfinite(c_ref)).sum().item())}
                if weight is not None:
                    try:
                        def leaf():
                            return F.linear(xr_bf16, weight)
                        y = leaf()
                        item["R0_leaf"] = {
                            "status": "PASS", "output_dtype": str(y.dtype),
                            "vs_R1_fp32": stats(y, c_ref),
                            "timing": benchmark_cuda(leaf),
                        }
                    except Exception as exc:
                        item["R0_leaf"] = {
                            "status": "ERROR", "exception": type(exc).__name__,
                            "message": str(exc)[:600],
                        }
                # Surrogate: BF16 dequant + BF16 linear. Useful if actual Quanto
                # fallback crashes. Not to be mistaken for measured R0.
                try:
                    dense_bf16 = (q.float() * s[:, None]).bfloat16()
                    def surrogate():
                        return F.linear(xr_bf16, dense_bf16)
                    yy = surrogate()
                    item["bf16_dense_surrogate"] = {
                        "status": "PASS", "vs_R1_fp32": stats(yy, c_ref),
                        "timing": benchmark_cuda(surrogate),
                    }
                    del dense_bf16, yy
                except Exception as exc:
                    item["bf16_dense_surrogate"] = {
                        "status": "ERROR", "exception": type(exc).__name__, "message": str(exc)[:600],
                    }
                item["R1_bf16_rounding_error"] = stats(ref_bf16, c_ref)
                per_m.append(item)
                del x, xr32, xr_bf16, c_ref, ref_bf16
            result["tests"] = per_m
            result["status"] = "PASS"
            del q, s, weight
            torch.cuda.synchronize()
    except Exception as exc:
        result.update(status="ERROR", exception=type(exc).__name__, error=str(exc)[:800])
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sha = os.environ.get("GITHUB_SHA", "local")
    report = {
        "status": "UNTESTED", "source_sha": sha,
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint": str(FILE), "checkpoint_sha256": "ae777a7d5ea1830a9d0acdf784a1af278bfe421f109a8a8e2f0e1f2b03af34ef",
        "checkpoint_hash_method": "verified previously by independent full-file SHA256 command",
        "real_checkpoint_weights": True, "real_model_activations": False,
        "actual_fallback_dispatch_confirmed": False,
        "production_candidate_tested": False, "no_persistent_expanded_weights": True,
        "torch": str(torch.__version__),
    }
    if not FILE.is_file():
        report.update(status="BLOCKED", error="Checkpoint not visible")
    else:
        try:
            free, total = torch.cuda.mem_get_info()
            report["gpu_free_bytes_at_start"] = free
            report["gpu_total_bytes"] = total
            rotate = github_source_rotation()
            with safe_open(str(FILE), framework="pt", device="cpu") as f:
                selected = tested_shapes(f)
                report["selected_matrix_families"] = len(selected)
                report["layers"] = []
                for triple in selected:
                    free, _ = torch.cuda.mem_get_info()
                    item = test_one(f, triple, rotate, free)
                    report["layers"].append(item)
                    print("WEIGHT_LAYER " + json.dumps(item, separators=(",", ":"))[:5000], flush=True)
            report["status"] = "PASS" if all(x["status"] == "PASS" for x in report["layers"]) else "PARTIAL"
        except Exception as exc:
            report.update(status="ERROR", error=str(exc)[:1000], exception=type(exc).__name__)
    path = OUT / f"phase0_weights_{sha[:12]}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("WEIGHT_TEST_RESULT " + json.dumps({k: v for k, v in report.items() if k != "layers"}), flush=True)
    print(f"WEIGHT_REPORT_PATH={path}", flush=True)
    # Fail the run for unexpected script errors, but not the documented
    # incompatible real Quanto fallback (recorded per layer instead).
    if report["status"] == "ERROR":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
