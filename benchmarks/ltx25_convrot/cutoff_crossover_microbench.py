#!/usr/bin/env python3
"""V100 cutoff crossover: full Candidate E ConvRot vs Stage2 FP16 Quanto.

Synthetic *real* M,N,K and dtypes. Includes activation Hadamard rotation,
temporary dequant/casts, cuBLAS, epilogue and output conversion. One INT8
weight set per shape. A-B-B-A paired CUDA-event and synchronized wall timings.
Not a whole-video performance/quality claim: follow with live A/B tests.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch
import optimum.quanto.library.qbytes_mm  # noqa: F401

from shared.kernels import ltx25_cublas_sm70 as candidate
from shared.kernels import ltx25_selective_fp16_gemm as fp16
from shared.qtypes.int8_convrot import _rotate_activation

OUTPUT = Path(os.environ.get(
    "LTX25_CUTOFF_MICRO_OUT", "/cache/ltx25/results/cutoff-micro-manual.json"
))
GROUPS = (
    (2048, 2048), (2048, 8192), (8192, 2048), (4096, 4096)
)
SMALL = (18, 32, 48, 51, 63, 64, 65, 80, 96, 128)
LARGE = (4096, 6144, 8192, 8193, 10240, 12288, 16384)
# M=18 represents the 17-frame token geometry; M≈65+ represents 73f.
# >8192 is specifically relevant to longer 241f-style token counts.
REPS = 3


def require(p, why):
    if not p:
        raise RuntimeError(why)


def timed(fn, count=REPS):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    gpu_ms = []
    wall_ms = []
    for _ in range(count):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        output = fn()
        stop.record()
        stop.synchronize()
        gpu_ms.append(float(start.elapsed_time(stop)))
        wall_ms.append((time.perf_counter() - wall_start) * 1000)
        del output
    return {
        "gpu_median_ms": round(statistics.median(gpu_ms), 4),
        "wall_median_ms": round(statistics.median(wall_ms), 4),
    }


def compare(m, n, k):
    torch.manual_seed(20261115 + m + n + k)
    q = torch.randint(-32, 33, (n, k), device="cuda", dtype=torch.int8)
    scale32 = torch.rand((n, 1), device="cuda", dtype=torch.float32) * .002 + .001
    scale16 = scale32.to(torch.bfloat16)
    weight = SimpleNamespace(_data=q, _scale=scale32)
    x = (torch.randn((m, k), device="cuda", dtype=torch.float32) * .4).to(torch.bfloat16)
    # Keep rows in the real numerical range; candidate's exceptional-row
    # fallback is nevertheless exercised where real signals demand it.
    op = torch.ops.quanto.qbytes_mm.default
    direct = fp16._SelectiveMode("stage2")
    def run_candidate():
        return candidate.maybe_linear(x, weight, None, 256)
    def run_fp16():
        # Matches ConvRot fallback's rotation followed by the Stage2
        # selective direct Quanto qbytes_mm conversion, with same INT8 q.
        rotated = _rotate_activation(x, 256)
        return direct.__torch_dispatch__(op, (), (rotated, q, scale16), {})
    candidate.MIN_M, candidate.MAX_M = 1, 65536
    a = run_candidate()
    b = run_fp16()
    require(a is not None and b is not None, f"No execution for {m,n,k}")
    torch.cuda.synchronize()
    require(a.shape == (m, n) and b.shape == (m, n), "Wrong output shape")
    a32, b32 = a.float(), b.float()
    require(bool(torch.isfinite(a32).all().item()) and bool(torch.isfinite(b32).all().item()),
            f"Nonfinite {m,n,k}")
    rms_diff = (a32-b32).square().mean().sqrt()
    rms_ref = b32.square().mean().sqrt().clamp_min(1e-12)
    rel_rms = round(float((rms_diff/rms_ref).item()), 7)
    # Full-path speed, no dequantized cached FP16 weights between calls.
    e1 = timed(run_candidate)
    f1 = timed(run_fp16)
    f2 = timed(run_fp16)
    e2 = timed(run_candidate)
    e_ms = (e1["gpu_median_ms"]+e2["gpu_median_ms"])/2
    f_ms = (f1["gpu_median_ms"]+f2["gpu_median_ms"])/2
    e_wall = (e1["wall_median_ms"]+e2["wall_median_ms"])/2
    f_wall = (f1["wall_median_ms"]+f2["wall_median_ms"])/2
    result = {
        "M":m,"N":n,"K":k,
        "candidate_gpu_ms":round(e_ms,4),
        "fp16_gpu_ms":round(f_ms,4),
        "candidate_wall_ms":round(e_wall,4),
        "fp16_wall_ms":round(f_wall,4),
        "candidate_minus_fp16_gpu_ms":round(e_ms-f_ms,4),
        "candidate_minus_fp16_wall_ms":round(e_wall-f_wall,4),
        "faster_gpu":"candidate_e" if e_ms<f_ms else "stage2_fp16",
        "faster_wall":"candidate_e" if e_wall<f_wall else "stage2_fp16",
        "candidate_over_fp16_gpu":round(e_ms/f_ms,3),
        "relative_rms_error":rel_rms,
        "finite":True,
        "memory_bytes":{
            "weights_int8":n*k,
            "candidate_c32":m*n*4,
            "activation_bf16":m*k*2,
        },
    }
    del a,b,a32,b32,x,q,scale32,scale16,weight,direct
    torch.cuda.empty_cache()
    return result


def main():
    require(torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (7,0),
            "V100 SM70 required")
    require(candidate.MODE == "fp32rot" and candidate.ROT_MODE == "radix4"
            and candidate.CAST_MODE == "vector8" and candidate.OUTPUT_MODE == "fused",
            "Candidate E defaults unexpectedly changed")
    require(candidate.MIN_M == 64 and candidate.MAX_M == 8192,
            "Microbenchmark must start from unmodified 64/8192 defaults")
    require(fp16.ENABLED and fp16.BACKEND == "direct"
            and "stage2:balanced" in fp16.GROUPS and "qbytes_mm" in fp16.OPS,
            "Selective Stage2 FP16 not enabled as expected")
    assert (2048, 2048) in candidate.PASS_SHAPES
    # Verify exact current dispatch decisions before timing any alternative:
    q=torch.zeros((2048,2048),dtype=torch.int8,device="cuda")
    sc=torch.ones((2048,1),dtype=torch.float32,device="cuda")*.001
    w=SimpleNamespace(_data=q,_scale=sc)
    for m,expected in ((18,False),(63,False),(64,True),(8192,True),(8193,False)):
        x=torch.randn((m,2048),dtype=torch.bfloat16,device="cuda")*.1
        observed=candidate.maybe_linear(x,w,None,256) is not None
        require(observed==expected, f"Current dispatch mismatch M={m}")
        print("CUTOFF_DISPATCH_PASS",m,observed,flush=True)
        del x
    del w,q,sc
    torch.cuda.empty_cache()
    results=[]
    # Benchmark all relevant M against shapes eligible for Candidate E. N,K
    # weight family (4096,2048) remains numerically excluded.
    for n,k in GROUPS:
        for m in SMALL+LARGE:
            row=compare(m,n,k)
            results.append(row)
            print("CUTOFF_MICRO_CASE",json.dumps(row,sort_keys=True),flush=True)
            OUTPUT.parent.mkdir(parents=True,exist_ok=True)
            OUTPUT.write_text(json.dumps({"status":"PARTIAL","rows":results},indent=2)+"\n")
    by_shape={}
    for n,k in GROUPS:
        rows=[x for x in results if x["N"]==n and x["K"]==k]
        by_shape[f"{n}x{k}"]={
            "below_64":[(x["M"],x["faster_gpu"],x["candidate_over_fp16_gpu"])
                        for x in rows if x["M"]<64],
            "over_8192":[(x["M"],x["faster_gpu"],x["candidate_over_fp16_gpu"])
                         for x in rows if x["M"]>8192],
        }
    data={
        "status":"PASS","kind":"synthetic full-operation GPU cutoff crossover, not video proof",
        "device":torch.cuda.get_device_name(0),
        "baseline_cutoffs":{"min_m":64,"max_m":8192},
        "candidate_e_mode":"fp32rot, vector8, radix4, fused",
        "fp16_fallback":"ConvRot BF16 Hadamard then actual _SelectiveMode stage2 qbytes_mm",
        "sample_method":"CUDA events and wall synchronization; ABBA interleaved, medians of 3 per arm",
        "rows":results,"summary":by_shape,
        "disclaimers":[
            "Synthetic BF16 activations and int8 weights, not captured live layer values.",
            "Candidate E rotates in FP32 and Quanto fallback rotates in BF16: arithmetic is not identical.",
            "Large-M CUDA scratch/workspace plus MMGP pressure and concurrent operations can change crossover.",
            "Follow with matched real 17/73/241 frame video runs and visual quality review.",
            "No persistent FP16 weight copies. No default change.",
        ],
    }
    OUTPUT.write_text(json.dumps(data,indent=2)+"\n")
    print("LTX25_CUTOFF_MICRO_PASS",json.dumps({
        "cases":len(results),"file":str(OUTPUT),"summary":by_shape
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
