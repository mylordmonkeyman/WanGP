#!/usr/bin/env python3
"""V100 SM70 exact INT8->FP16 conversion microbench, no model generation.

Isolated GPU allocations of fake benchmark INT8 weights (not checkpoint
copies); compare PyTorch's normal q.to(float16), PyTorch copy_ into a
preallocated FP16 tensor, and two separate native kernels. Exact int8->FP16
equality is required. CUDA events synchronize only at trial boundaries,
never during conversion.

Do not infer video end-to-end speedup from this microbenchmark.
"""
from __future__ import annotations

import ctypes
import datetime as dt
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

import torch

OUT = Path("/cache/ltx25/results")
SHA = os.environ.get("GITHUB_SHA", "local")
LIBRARY = Path(os.environ.get("LTX25_CAST_BENCH_LIB", "/tmp/ltx25_cast_bench.so"))
SHAPES = [
    (2048, 2048), (4096, 4096), (2048, 4096),
    (16384, 4096), (4096, 16384), (8192, 2048),
]
TRIALS = 50
WARMUPS = 6


def load():
    dll = ctypes.CDLL(str(LIBRARY))
    cast = dll.ltx25_cast_int8_half
    cast.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                     ctypes.c_int, ctypes.c_void_p]
    cast.restype = ctypes.c_int
    return dll, cast


def benchmark(fn, device):
    for _ in range(WARMUPS):
        fn()
    torch.cuda.synchronize(device)
    events = []
    for _ in range(TRIALS):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        events.append((start, end))
        # Result may own an allocated tensor; memory is handled by PyTorch.
        del result
    torch.cuda.synchronize(device)
    values = [float(a.elapsed_time(b)) for a, b in events]
    values.sort()
    return {
        "median_ms": round(statistics.median(values), 5),
        "min_ms": round(min(values), 5),
        "p90_ms": round(values[min(len(values) - 1, math.ceil(0.9 * len(values)) - 1)], 5),
        "trials": TRIALS,
    }


def main():
    assert torch.cuda.device_count() == 1
    assert torch.cuda.get_device_capability(0) == (7, 0)
    device = torch.device("cuda:0")
    _, cast = load()
    stream = ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream)
    report = {
        "kind": "INT8-to-FP16 isolated SM70 conversion microbenchmark",
        "source_sha": SHA,
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "production_eligible": False,
        "model_generation": False,
        "input": "synthetic checkpoint-shaped int8 tensor (no checkpoint loaded)",
        "device": str(torch.cuda.get_device_name(0)),
        "shape_results": [],
        "status": "ERROR",
        "trials_per_variant": TRIALS,
    }
    torch.manual_seed(20261113)
    for n, k in SHAPES:
        q = torch.randint(-128, 128, (n, k), device=device, dtype=torch.int8)
        out = torch.empty((n, k), device=device, dtype=torch.float16)
        torch.cuda.synchronize()
        def native(kind):
            code = cast(ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(out.data_ptr()),
                        q.numel(), kind, stream)
            if code:
                raise RuntimeError(f"CUDA cast native variant {kind} returned {code}")
            return out
        expected = q.to(torch.float16)
        native(0)
        torch.cuda.synchronize()
        assert torch.equal(expected, out), f"scalar cast mismatch: {n}x{k}"
        native(4)
        torch.cuda.synchronize()
        assert torch.equal(expected, out), f"vector4 cast mismatch: {n}x{k}"
        results = {
            "shape": [n, k], "int8_bytes": q.numel(), "fp16_bytes": q.numel() * 2,
            "exact_int8_to_fp16": True,
        }
        calls = {
            "torch_to_alloc": lambda: q.to(torch.float16),
            "torch_copy_preallocated": lambda: out.copy_(q),
            "cuda_scalar_preallocated": lambda: native(0),
            "cuda_vector4_preallocated": lambda: native(4),
        }
        # Interleave A/B methods by shape, repeat each method in blocks.
        for name, fn in calls.items():
            results[name] = benchmark(fn, device)
        t0 = results["torch_to_alloc"]["median_ms"]
        t1 = results["torch_copy_preallocated"]["median_ms"]
        t2 = results["cuda_vector4_preallocated"]["median_ms"]
        results["vector4_vs_torch_to_speedup"] = round(t0 / t2, 4) if t2 else None
        results["vector4_vs_torch_copy_speedup"] = round(t1 / t2, 4) if t2 else None
        report["shape_results"].append(results)
        print("LTX25_CAST_RESULT " + json.dumps(results, sort_keys=True), flush=True)
        del q, out, expected
    report["status"] = "PASS"
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / f"cast_microbench_{SHA[:12]}.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print("LTX25_CAST_SUMMARY " + json.dumps({
        "status": report["status"], "shapes": len(report["shape_results"]),
        "source_sha": SHA,
        "faster_shapes_vs_torch_alloc": sum(
            r["vector4_vs_torch_to_speedup"] > 1.0 for r in report["shape_results"])
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
