#!/usr/bin/env python3
"""Candidate E exploratory benchmark: actual INT8 weights, synthetic BF16 activations.

No WanGP model invocation, no real activation distribution, no LoRA, no graph
capture, no production fallback. Never enable auto from these results.
The FP16 weight scratch is *reused* and bounded at 128 MiB per layer. The
cuBLAS call produces true FP32 C (not torch FP16 matmul rounded output).
"""
from __future__ import annotations

import ctypes
import datetime as dt
import json
import math
import os
from pathlib import Path
import statistics
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

from phase0_probe import github_source_rotation
from phase0_weights import FILE, OUT, FAMILIES, MS, stats, tested_shapes, try_actual_qweight

SCRATCH_CAP = 128 * 1024 * 1024
LIB = os.environ.get("LTX25_CUBLAS_E_LIB", "")


def load_library():
    if not LIB or not Path(LIB).is_file():
        raise RuntimeError("LTX25_CUBLAS_E_LIB not set to compiled smoke extension")
    dll = ctypes.CDLL(LIB)
    fn = dll.ltx25_cublas_gemm_f16_f32
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    return fn


def timer(call):
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    result = call()
    en.record()
    en.synchronize()
    return st.elapsed_time(en), result


def one(f, triple, rotate, gemm):
    base, qkey, skey = triple
    result = {"name": base, "N": None, "K": None,
              "real_checkpoint_weights": True, "synthetic_activations": True,
              "full_pipeline": False, "tensor_core_instruction_verified": False,
              "candidate_production_eligible": False,
              "weight_scale_application": "FP32 after FP32-output cuBLAS",
              "scratch_cap_mib": SCRATCH_CAP // 2**20}
    try:
        qcpu = f.get_tensor(qkey)
        scpu = f.get_tensor(skey)
        n, k = map(int, qcpu.shape)
        result.update(N=n, K=k)
        scratch_bytes = qcpu.numel() * 2
        result["scratch_bytes"] = scratch_bytes
        if scratch_bytes > SCRATCH_CAP:
            result.update(status="SKIPPED", reason="Weight scratch exceeds cap; chunking not implemented")
            return result
        free, total = torch.cuda.mem_get_info()
        if free < 1_500_000_000:
            result.update(status="SKIPPED", reason="Less than 1.5 GB VRAM free")
            return result
        q = qcpu.to("cuda")
        s = scpu.to("cuda", dtype=torch.float32).reshape(-1)
        if s.numel() == 1:
            s = s.expand(n)
        assert s.numel() == n
        qweight, wrap_error = try_actual_qweight(q, s[:, None].contiguous())
        if qweight is None:
            result.update(status="BLOCKED", wrapper_error=wrap_error)
            return result

        scratch_w = torch.empty((n, k), device="cuda", dtype=torch.float16)
        result["storage_q_int8_bytes"] = q.numel()
        result["scratch_w_fp16_bytes"] = scratch_w.numel() * scratch_w.element_size()
        result["per_call_persistent_fp16_weight_copy"] = False
        layers = []
        with torch.inference_mode():
            extended = (1, 16, 256, 1024) if (n, k) in ((2048, 2048), (4096, 4096), (4096, 16384)) else (1, 16, 256)
            for m in extended:
                torch.manual_seed(20261112 + m)
                x = torch.randn((m, k), dtype=torch.float32, device="cuda").bfloat16()
                scratch_x = torch.empty((m, k), device="cuda", dtype=torch.float16)
                scratch_c = torch.empty((m, n), device="cuda", dtype=torch.float32)
                output = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
                ref_fp32 = (rotate(x.float(), 256) @ q.float().T) * s.reshape(1, -1)
                rot = rotate(x, 256)
                xr32 = rot.float()
                absx = xr32.abs()
                nonzero = absx[absx > 0]
                e_input_range = {
                    "post_rotation_max_abs": float(absx.max().item()),
                    "post_rotation_min_nonzero_abs": float(nonzero.min().item()) if nonzero.numel() else 0.0,
                    "fp16_overflow_count": int((absx > 65504).sum().item()),
                    "fp16_subnormal_count": int(((absx > 0) & (absx < 2**-14)).sum().item()),
                    "fp16_underflow_to_zero_count": int(((absx > 0) & (absx < 2**-24)).sum().item()),
                }

                def r0_call():
                    return F.linear(rotate(x, 256), qweight)

                stream = torch.cuda.current_stream()
                stream_ptr = ctypes.c_void_p(stream.cuda_stream)

                def e_compute(fp32_rotation: bool):
                    # The FP32-rotation variant is a separately labeled
                    # experiment, not Candidate E's unchanged-rotation path.
                    rotated = rotate(x.float() if fp32_rotation else x, 256)
                    scratch_w.copy_(q)          # INT8->FP16 exact, every call
                    scratch_x.copy_(rotated)    # BF16/FP32->FP16, range-gated later
                    err = gemm(ctypes.c_void_p(scratch_w.data_ptr()),
                               ctypes.c_void_p(scratch_x.data_ptr()),
                               ctypes.c_void_p(scratch_c.data_ptr()),
                               n, m, k, stream_ptr)
                    if err:
                        raise RuntimeError(f"cuBLAS status {err}")
                    scratch_c.mul_(s.reshape(1, -1))  # FP32 epilogue
                    output.copy_(scratch_c)          # BF16 final output
                    return output

                def e_call():
                    return e_compute(False)

                def e_fp32rot_call():
                    return e_compute(True)

                for _ in range(3):
                    r0_call()
                    e_call()
                    e_fp32rot_call()
                torch.cuda.synchronize()
                r0_y = r0_call().clone()
                e_y = e_call().clone()
                er_y = e_fp32rot_call().clone()
                torch.cuda.synchronize()
                r0_stats = stats(r0_y, ref_fp32)
                e_stats = stats(e_y, ref_fp32)
                er_stats = stats(er_y, ref_fp32)
                tol = 1e-8
                def gate(candidate):
                    return bool(
                        candidate["nonfinite"] == 0
                        and candidate["p99_normalized"] <= r0_stats["p99_normalized"] + tol
                        and candidate["max_normalized"] <= r0_stats["max_normalized"] + tol
                    )
                times = {"r0_ms": [], "e_ms": [], "efp32_ms": []}
                funcs = [("r0_ms", r0_call), ("e_ms", e_call), ("efp32_ms", e_fp32rot_call)]
                for t in range(6):
                    # Rotate order to mitigate clock and thermal drift.
                    order = funcs[t % 3:] + funcs[:t % 3]
                    for label, call in order:
                        ms, _ = timer(call)
                        times[label].append(float(ms))
                r0_med = statistics.median(times["r0_ms"])
                e_med = statistics.median(times["e_ms"])
                er_med = statistics.median(times["efp32_ms"])
                layers.append({
                    "M": m, "K": k, "N": n, "group_size": 256,
                    "R0": r0_stats, "E": e_stats, "E_fp32rot": er_stats,
                    "precision_gate_pass": gate(e_stats),
                    "fp32rot_precision_gate_pass": gate(er_stats),
                    "R0_full_layer_median_ms": r0_med,
                    "E_full_layer_median_ms": e_med,
                    "E_fp32rot_full_layer_median_ms": er_med,
                    "speedup_ratio_R0_over_E": (r0_med/e_med) if e_med > 0 else None,
                    "speedup_ratio_R0_over_fp32rot": (r0_med/er_med) if er_med > 0 else None,
                    "R0_trials_ms": times["r0_ms"],
                    "E_trials_ms": times["e_ms"],
                    "E_fp32rot_trials_ms": times["efp32_ms"],
                    "range": e_input_range,
                    "note": "E includes unchanged BF16 rotation; E_fp32rot changes it and is diagnostic only. Both include Q casting, cuBLAS, FP32 scale and BF16 output. No H2D or output bias.",
                })
                del x, scratch_x, scratch_c, output, ref_fp32, r0_y, e_y, er_y, rot, xr32, absx, nonzero
                torch.cuda.synchronize()
        result["tests"] = layers
        result["status"] = "PASS"
        result["gpu_peak_allocated_mib"] = round(torch.cuda.max_memory_allocated()/2**20, 2)
        del q, qweight, scratch_w, s
    except Exception as exc:
        result.update(status="ERROR", error_type=type(exc).__name__, error=str(exc)[:700])
    return result


def main():
    sha = os.environ.get("GITHUB_SHA", "local")
    report = {
        "source_sha": sha, "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "kind": "Candidate E checkpoint-backed isolated research benchmark",
        "CUBLAS_COMPUTE_32F": True, "FP32_output": True, "FP32_postscale": True,
        "all_activations_synthetic": True, "candidate_auto_eligible": False,
        "precision_gate_from_spec": "E must not exceed per-shape R0 p99 and max normalized error",
        "weights_expanded_whole_model": False, "scratch_bytes_cap": SCRATCH_CAP,
        "full_pipeline_measured": False, "tensor_core_instruction_verified": False,
    }
    try:
        fn = load_library()
        rotate = github_source_rotation()
        report["results"] = []
        with safe_open(str(FILE), framework="pt", device="cpu") as f:
            shapes = tested_shapes(f)
            for triple in shapes:
                item = one(f, triple, rotate, fn)
                report["results"].append(item)
                print("CUBLAS_E_LAYER " + json.dumps(item, separators=(",", ":"))[:5000], flush=True)
        report["status"] = "PASS" if all(x["status"] == "PASS" for x in report["results"]) else "PARTIAL"
    except Exception as exc:
        report.update(status="ERROR", error_type=type(exc).__name__, error=str(exc)[:700])
    path = OUT / f"phase1_cublas_e_{sha[:12]}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("CUBLAS_E_RESULT " + json.dumps({k:v for k,v in report.items() if k!="results"}), flush=True)
    print(f"CUBLAS_E_REPORT_PATH={path}", flush=True)
    if report["status"] == "ERROR":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
