#!/usr/bin/env python3
"""Benchmark FP32 Hadamard H256 (torch matmul versus radix-4 SM70).

No model checkpoint, no weight conversions, no production integration.
Compares the exact WanGP Kronecker H4 sign/order with CUDA radix-4.
50 same-stream CUDA-event repetitions per trial shape. CUDA sync only at
trial boundaries, never inside native entry point or a single conversion.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.qtypes.int8_convrot import _regular_hadamard

OUT = Path("/cache/ltx25/results")
LIB = os.environ.get("LTX25_HADAMARD_LIB", "/tmp/ltx25_hadamard_sm70.so")
SHAPES = [(64, 2048), (64, 4096), (128, 4096), (256, 2048),
          (256, 4096), (512, 4096), (1024, 4096), (256, 16384)]
TRIALS = 50
WARMUP = 8


def timed(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    starts, ends = [], []
    for _ in range(TRIALS):
        a, z = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        z.record()
        starts.append(a)
        ends.append(z)
    torch.cuda.synchronize()
    v = sorted(float(a.elapsed_time(z)) for a, z in zip(starts, ends))
    return {
        "median_ms": round(statistics.median(v), 5),
        "p90_ms": round(v[math.ceil(0.9 * len(v)) - 1], 5),
        "min_ms": round(min(v), 5),
        "trials": TRIALS,
    }


def main():
    assert torch.cuda.get_device_capability() == (7, 0)
    torch.manual_seed(20261113)
    dll = ctypes.CDLL(LIB)
    native = dll.ltx25_hadamard256_f32
    ptr = ctypes.c_void_p
    native.argtypes = [ptr, ptr, ctypes.c_size_t, ptr]
    native.restype = ctypes.c_int
    stream = ptr(torch.cuda.current_stream().cuda_stream)
    H = _regular_hadamard(256, "cuda", torch.float32)
    assert H.shape == (256, 256)
    result = {
        "status": "ERROR",
        "source_sha": os.environ.get("GITHUB_SHA", "local"),
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(),
        "scope": "Synthetic FP32 rotated activations, standalone exact H256 ordering",
        "mode": "isolated_same_stream_cuda_events",
        "production_eligible": False,
        "model_generation": False,
        "trials": TRIALS,
        "shape_results": [],
    }
    for m, k in SHAPES:
        # BF16 input was the common runtime activation; the validated E path
        # does FP32 conversion BEFORE its existing Hadamard torch.matmul.
        for dtype in (torch.bfloat16, torch.float16):
            x = torch.randn((m, k), device="cuda", dtype=torch.float32).to(dtype).float()
            ref = torch.empty_like(x)
            candidate = torch.empty_like(x)

            def torch_path():
                torch.matmul(x.view(-1, k // 256, 256), H,
                             out=ref.view(-1, k // 256, 256))
                return ref

            def cuda_path():
                rc = native(ptr(x.data_ptr()), ptr(candidate.data_ptr()),
                            x.numel(), stream)
                if rc:
                    raise RuntimeError(f"Radix4 H256 native status {rc}")
                return candidate

            # Compare results only AFTER both kernels have fully completed.
            torch_path()
            cuda_path()
            torch.cuda.synchronize()
            assert bool(torch.isfinite(candidate).all().item())
            diff = (ref - candidate).abs()
            norm = diff / torch.maximum(ref.abs(), torch.ones_like(ref))
            rel_rms = float((diff.square().mean().sqrt() /
                             ref.square().mean().sqrt().clamp_min(1e-6)).item())
            max_norm = float(norm.max().item())
            p99_norm = float(torch.quantile(norm.flatten(), 0.99).item())
            exact_frac = float((ref == candidate).float().mean().item())
            # This stage is NOT bit-exact to dense GEMM: rounding order differs.
            assert max_norm < 1e-4 and rel_rms < 1e-5, (
                f"Numerical gate failed M={m},K={k},dtype={dtype}: "
                f"max_norm={max_norm} RMS={rel_rms}"
            )
            a = timed(torch_path)
            z = timed(cuda_path)
            speedup = a["median_ms"] / z["median_ms"]
            row = {
                "M": m, "K": k, "dtype": str(dtype),
                "torch_matmul": a, "radix4_cuda": z,
                "max_normalized_error": max_norm,
                "p99_normalized_error": p99_norm,
                "relative_rms_error": rel_rms,
                "exact_fp32_element_fraction": exact_frac,
                "speedup": round(speedup, 4),
            }
            result["shape_results"].append(row)
            print("LTX25_HADAMARD_RESULT " + json.dumps(row, sort_keys=True), flush=True)
            del x, ref, candidate
    result["status"] = "PASS"
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"hadamard_microbench_{result['source_sha'][:12]}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print("LTX25_HADAMARD_SUMMARY " + json.dumps({
        "status": result["status"], "cases": len(result["shape_results"]),
        "faster_cases": sum(row["speedup"] > 1.0 for row in result["shape_results"]),
        "max_normalized_error": max(x["max_normalized_error"] for x in result["shape_results"]),
    }), flush=True)


if __name__ == "__main__":
    main()
