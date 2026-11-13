#!/usr/bin/env python3
"""Profile PyTorch SDPA kernel dispatch and 1CatAI direct BHMD on V100.

GPU-only diagnostic. The authorized GitHub Actions workflow alone executes it.
Uses the previously validated ABI-locked extensions, no model weights.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile, record_function

OUTPUT = Path(os.environ.get("LTX25_KERNEL_PROBE_OUT", "/tmp/ltx25-sdpa-kernel-probe"))
OUTPUT.mkdir(parents=True, exist_ok=True)
CASES = [
    ("real-video-315-bf16", 315, 32, torch.bfloat16),
    ("real-video-315-fp16", 315, 32, torch.float16),
    ("video-1024-fp16", 1024, 8, torch.float16),
    ("video-4096-fp16", 4096, 8, torch.float16),
    ("video-8192-fp16", 8192, 32, torch.float16),
]
D, B = 128, 1
MAX_ERROR, MAX_RMS = 0.05, 0.005
ROUNDS, ITERATIONS, WARMUP = 3, 12, 4


def sdpa_bmhd(q, k, v):
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        dropout_p=0.0, is_causal=False,
    ).transpose(1, 2)


def sdpa_bhmd(q, k, v):
    return F.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, is_causal=False,
    )


def make_paths(q, k, v, cat):
    """Label prepack costs, never hide them inside kernel-only timings."""
    q16, k16, v16 = (x.to(torch.float16) for x in (q, k, v))
    qbh, kbh, vbh = (x.transpose(1, 2).contiguous() for x in (q16, k16, v16))

    # BMHD inputs: compare with WanGP's original SDPA, and two 1Cat APIs.
    paths = {
        "sdpa_bmhd": lambda: sdpa_bmhd(q, k, v),
    }
    if q.dtype is torch.float16:
        paths["cat_public_bmhd"] = lambda: cat.flash_attn_func(
            q, k, v, dropout_p=0.0, causal=False,
        )
        paths["cat_direct_bhmd_views"] = lambda: cat.flash_attn_bhmd_func(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            dropout_p=0.0, causal=False,
        ).transpose(1, 2)
    else:
        # Real LTX-2.5 BF16 input: GPU conversions are inside the timed path.
        paths["sdpa_bf16_through_fp16"] = lambda: sdpa_bmhd(
            q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
        ).to(torch.bfloat16)
        paths["cat_public_cast_bf16"] = lambda: cat.flash_attn_func(
            q.to(torch.float16), k.to(torch.float16), v.to(torch.float16),
            dropout_p=0.0, causal=False,
        ).to(torch.bfloat16)
        paths["cat_direct_views_cast_bf16"] = lambda: cat.flash_attn_bhmd_func(
            q.to(torch.float16).transpose(1, 2),
            k.to(torch.float16).transpose(1, 2),
            v.to(torch.float16).transpose(1, 2),
            dropout_p=0.0, causal=False,
        ).transpose(1, 2).to(torch.bfloat16)

    # The following are KERNEL-ALIGNED comparisons: identical already-contiguous
    # BHMD Q/K/V. Materialization is outside timing for *both* implementations.
    prepacked = {
        "sdpa_prepacked_bhmd": lambda: sdpa_bhmd(qbh, kbh, vbh),
        "cat_direct_prepacked_bhmd": lambda: cat.flash_attn_bhmd_func(
            qbh, kbh, vbh, dropout_p=0.0, causal=False,
        ),
    }
    reference = sdpa_bmhd(q, k, v)
    reference_bhmd = sdpa_bhmd(qbh, kbh, vbh)
    return paths, prepacked, reference, reference_bhmd


def accuracy(got, ref):
    diff = (got.float() - ref.float()).abs()
    error, rms = float(diff.max().item()), float(diff.square().mean().sqrt().item())
    finite = bool(torch.isfinite(got).all().item())
    return {
        "max_abs": round(error, 7),
        "rms": round(rms, 7),
        "finite": finite,
        "pass": finite and error <= MAX_ERROR and rms <= MAX_RMS,
    }


def bench(fn):
    for _ in range(WARMUP):
        y = fn()
        del y
    torch.cuda.synchronize()
    samples = []
    for i in range(ROUNDS):
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        cpu_start = time.perf_counter()
        start.record()
        for _ in range(ITERATIONS):
            y = fn()
            del y
        end.record()
        end.synchronize()
        samples.append({
            "round": i,
            "cuda_ms": start.elapsed_time(end) / ITERATIONS,
            "wall_ms": ((time.perf_counter() - cpu_start) * 1000) / ITERATIONS,
            "extra_peak_mib": max(0, torch.cuda.max_memory_allocated() - baseline) / 1048576,
        })
    return {
        "median_cuda_ms": round(statistics.median(x["cuda_ms"] for x in samples), 5),
        "median_wall_ms": round(statistics.median(x["wall_ms"] for x in samples), 5),
        "peak_extra_mib": round(max(x["extra_peak_mib"] for x in samples), 3),
        "measurements": samples,
    }


def profile_sdpa(q, k, v, label, backend=None):
    """Return aten operators AND actual GPU kernel names from Kineto events."""
    # Warm up before tracing to exclude lazy context/library initialization.
    for _ in range(3):
        y = sdpa_bhmd(q, k, v)
        del y
    torch.cuda.synchronize()
    try:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=False, profile_memory=False, with_stack=False) as p:
            with record_function("ltx25_sdpa_" + label):
                for _ in range(2):
                    y = sdpa_bhmd(q, k, v)
                    del y
            torch.cuda.synchronize()
        events = p.events()
        aten = {}
        cuda = {}
        for e in events:
            label_name = str(e.name)
            is_cuda = ("CUDA" in str(e.device_type).upper() and
                       "CPU" not in str(e.device_type).upper())
            if is_cuda:
                row = cuda.setdefault(label_name, {"name": label_name, "count": 0, "device_us": 0.0})
                row["count"] += 1
                row["device_us"] += float(e.device_time_total)
            elif label_name.startswith("aten::") and (
                    "scaled_dot_product" in label_name or "attention" in label_name):
                row = aten.setdefault(label_name, {"name": label_name, "count": 0})
                row["count"] += 1
        path = OUTPUT / f"sdpa-{label}.json"
        path.write_text(json.dumps({
            "aten_ops": list(aten.values()),
            "cuda_kernels": sorted(cuda.values(), key=lambda x: -x["device_us"]),
        }, indent=2) + "\n")
        p.export_chrome_trace(str(OUTPUT / f"sdpa-{label}-trace.json"))
        return {
            "status": "PASS",
            "aten_ops": list(aten.values()),
            "cuda_kernels": sorted(cuda.values(), key=lambda x: -x["device_us"])[:18],
            "all_cuda_kernel_types": len(cuda),
        }
    except Exception as exc:
        return {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-2000:]}


def probe_forced_backends(q, k, v):
    """Attempt to force each backend on the 315-token case only."""
    result = {}
    modes = (
        ("flash", SDPBackend.FLASH_ATTENTION),
        ("efficient", SDPBackend.EFFICIENT_ATTENTION),
        ("math", SDPBackend.MATH),
        ("cudnn", SDPBackend.CUDNN_ATTENTION),
    )
    for name, mode in modes:
        try:
            with sdpa_kernel(mode):
                got = sdpa_bhmd(q, k, v)
                torch.cuda.synchronize()
            result[name] = {
                "supported": True,
                "accuracy": accuracy(got, sdpa_bhmd(q, k, v)),
            }
        except Exception as exc:
            result[name] = {
                "supported": False, "error": f"{type(exc).__name__}: {exc}"[:500],
            }
    return result


def main():
    assert torch.__version__.startswith("2.10.0+cu128"), torch.__version__
    assert torch.cuda.get_device_capability() == (7, 0), "SM70 V100 required"
    import flash_attn_v100 as cat
    assert callable(cat.flash_attn_bhmd_func), "Required direct BHMD function missing"
    torch.manual_seed(20261113)
    report = {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "commit": os.environ.get("GITHUB_SHA"),
        "note": "Real workload shape (T315 BF16) and synthetic FP16 shapes; raw BHMD "
                "excludes QKV materialization, while BMHD paths include it.",
        "runtime_sdpa_switches": {
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
            "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
        },
        "cases": [], "profiler": {},
    }
    errors = []
    with torch.inference_mode():
        for name, tokens, heads, dtype in CASES:
            q, k, v = (torch.randn((B, tokens, heads, D), device="cuda", dtype=dtype)
                       * 0.125 for _ in range(3))
            entry = {"name": name, "tokens": tokens, "heads": heads,
                     "head_dim": D, "dtype": str(dtype), "paths": {}, "prepacked": {}}
            report["cases"].append(entry)
            paths, packed, ref, ref_packed = make_paths(q, k, v, cat)
            for group, cases, expected in (
                ("paths", paths, ref), ("prepacked", packed, ref_packed)
            ):
                for label, fn in cases.items():
                    try:
                        got = fn()
                        acc = accuracy(got, expected)
                        del got
                        if not acc["pass"]:
                            raise AssertionError(f"{label} failed numerical check: {acc}")
                        entry[group][label] = {"status": "PASS", "accuracy": acc, **bench(fn)}
                    except Exception as exc:
                        entry[group][label] = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
                        errors.append(f"{name}/{label}: {exc}")
                    print("TEST", name, label, json.dumps(entry[group][label])[:600], flush=True)
            if name == "real-video-315-bf16":
                qb, kb, vb = (z.transpose(1, 2) for z in (q, k, v))
                report["profiler"]["sdpa_315_bf16"] = profile_sdpa(qb, kb, vb, "315-bf16")
                report["profiler"]["forced_315_bf16"] = probe_forced_backends(qb, kb, vb)
            if name == "video-4096-fp16":
                qb, kb, vb = (z.transpose(1, 2) for z in (q, k, v))
                report["profiler"]["sdpa_4096_fp16"] = profile_sdpa(qb, kb, vb, "4096-fp16")
            del q, k, v, ref, ref_packed, paths, packed
            torch.cuda.empty_cache()
            (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["profiler"].get("sdpa_315_bf16", {}).get("cuda_kernels"):
        errors.append("No CUDA kernel names obtained for SDPA 315 BF16")
    if not report["profiler"].get("sdpa_4096_fp16", {}).get("cuda_kernels"):
        errors.append("No CUDA kernel names obtained for SDPA 4096 FP16")
    report["errors"] = errors
    report["status"] = "PASS" if not errors else "FAIL"
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("SDPA_PROFILER_RESULTS", json.dumps(report["profiler"]), flush=True)
    for case in report["cases"]:
        print("BENCH_SUMMARY", case["name"],
              {label: value.get("median_cuda_ms") for label, value in case["paths"].items()},
              {label: value.get("median_cuda_ms") for label, value in case["prepacked"].items()},
              flush=True)
    if errors:
        raise SystemExit("Kernel probe failed: " + "; ".join(errors[:7]))


if __name__ == "__main__":
    main()
