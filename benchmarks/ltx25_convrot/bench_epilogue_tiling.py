#!/usr/bin/env python3
"""Opt-in tiled Candidate E epilogue: same-CUDA-stream parity and latency.

Synthetic edge-case tensors use REAL 73f Stage1/Stage2 observed matrix dimensions.
This is NOT a generation-quality test; full-video A/B still required.
"""
from __future__ import annotations
import ctypes
import json
import os
from pathlib import Path
import statistics
import time

import torch

LIB = os.environ["LTX25_CUBLAS_E_LIB"]
OUTPUT = Path(os.environ.get("LTX25_EPILOGUE_BENCH_JSON", "/tmp/ltx25-epilogue-tiled-microbench.json"))
PTR = ctypes.c_void_p
SEED = 20261115
# Rows in actual 73f video/audio calls are 76 / 1440 / 5760, though
# only the narrow M<=512 shape family is enabled for the experiment.
CASES = [
    (76, 2048, 2048, "safe", True),
    (76, 4096, 2048, "safe", False),
    (76, 16384, 4096, "safe", True),
    (76, 16384, 4096, "one_slow", True),
    (76, 16384, 4096, "some_slow", False),
    (1440, 4096, 2048, "safe", True),  # ineligible for integrated tiled dispatch
    (512, 8192, 4096, "safe", True),
]


def setup(fn, lib):
    f = getattr(lib, fn)
    f.argtypes = [PTR] * 8 + [ctypes.c_int] * 3 + [PTR]
    f.restype = ctypes.c_int
    return f


def run_case(lib, m, n, k, mode, use_bias):
    torch.manual_seed(SEED + n + m + k)
    c32 = torch.randn((m, n), dtype=torch.float32, device="cuda")
    x32 = (torch.randn((m, k), dtype=torch.float32, device="cuda") * 0.05).contiguous()
    q = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda")
    scale = (torch.rand(n, dtype=torch.float32, device="cuda") * 0.01 + 0.001).contiguous()
    bias = torch.randn(n, dtype=torch.float32, device="cuda") * 0.05 if use_bias else None
    exponents = torch.randint(-3, 4, (m,), dtype=torch.int32, device="cuda")
    slow = torch.zeros(m, dtype=torch.int32, device="cuda")
    if mode == "one_slow":
        slow[0] = 1
    elif mode == "some_slow":
        slow[:2] = 1
    legacy = setup("ltx25_cublas_epilogue_f32_to_bf16", lib)
    tiled = setup("ltx25_cublas_epilogue_f32_to_bf16_tiled", lib)
    a = torch.empty((m,n),dtype=torch.bfloat16,device="cuda")
    b = torch.empty_like(a)
    stream = PTR(torch.cuda.current_stream().cuda_stream)

    def invoke(fn, output):
        args = [PTR(z.data_ptr()) for z in (c32, output, x32, q, scale)]
        args += [PTR(bias.data_ptr() if bias is not None else 0),
                 PTR(exponents.data_ptr()), PTR(slow.data_ptr())]
        result = fn(*args, m, n, k, stream)
        if result:
            raise RuntimeError(f"Epilogue call error {result} {m,n,k,mode}")
    for _ in range(2):
        invoke(legacy,a)
        invoke(tiled,b)
    torch.cuda.synchronize()
    equal = bool(torch.equal(a,b))
    maxabs = float((a.float()-b.float()).abs().max().item())
    if not equal:
        raise AssertionError(f"Tiled epilogue differs from default on {m,n,k,mode} max={maxabs}")

    times={"legacy":[],"tiled":[]}
    reps = 12 if mode == "safe" else 3
    for label in ("legacy","tiled","tiled","legacy",
                  "legacy","tiled","tiled","legacy"):
        fn = legacy if label=="legacy" else tiled
        dest = a if label=="legacy" else b
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            invoke(fn,dest)
        end.record()
        end.synchronize()
        times[label].append(float(start.elapsed_time(end)) / reps)
    return {
        "shape": [m,n,k], "mode":mode, "bias":use_bias,
        "bit_identical":equal,"max_absolute_error":maxabs,
        "per_call_gpu_event_ms": times,
        "legacy_median_ms":round(statistics.median(times["legacy"]),6),
        "tiled_median_ms":round(statistics.median(times["tiled"]),6),
        "speed_ratio_legacy_over_tiled":round(
            statistics.median(times["legacy"])/statistics.median(times["tiled"]),4),
        "extra_persistent_weight_copies":0,
    }


def main():
    assert torch.cuda.get_device_capability(0)==(7,0), "Requires Volta sm70"
    torch.backends.cuda.matmul.allow_tf32 = False
    lib=ctypes.CDLL(LIB)
    results=[]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for spec in CASES:
        row=run_case(lib,*spec)
        print("EPILOGUE_TILED_CASE", json.dumps({k:v for k,v in row.items() if k!="per_call_gpu_event_ms"}), flush=True)
        results.append(row)
        OUTPUT.write_text(json.dumps({"status":"IN_PROGRESS","results":results},indent=2)+"\n")
    OUTPUT.write_text(json.dumps({
        "status":"PASS", "gpu":torch.cuda.get_device_name(0),
        "seed":SEED,"source":"real 73f-observed dimensions but synthetic values",
        "strict_output_requirement":"torch.equal BF16 output including device-selected exceptional FP32 rows",
        "interpretation":"CUDA event latencies include CPU launch gaps and cannot predict whole-video wall benefit",
        "cases":results
    },indent=2)+"\n")
    print("EPILOGUE_TILED_MICROBENCH_PASS",str(OUTPUT),flush=True)


if __name__=="__main__":
    main()
