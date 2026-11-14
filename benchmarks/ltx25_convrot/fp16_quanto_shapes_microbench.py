#!/usr/bin/env python3
"""V100 BF16 Quanto vs transient FP16 Tensor Core GEMM per real T2V shape.

Measured complete op (dequantization, casting, GEMM, output cast). Synthetic
BF16/INT8 operands match real dtypes and dimensions, not activation statistics.
Timing is CUDA-stream elapsed; final quality and wall-time require real video.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import statistics
import time
import torch
import optimum.quanto.library.qbytes_mm  # noqa: F401 register operator

# Exact M,N,K and call counts from real 49-frame T2V shape census (run 43607514390).
WORKLOADS = (
    ("stage1:main", 1008, 4096, 2048, 384),
    ("stage2:main", 4032, 4096, 2048, 144),
    ("stage1:square", 51, 2048, 2048, 3840),
    ("stage2:square", 51, 2048, 2048, 1440),
    ("stage1:expand", 51, 8192, 2048, 384),
    ("stage2:expand", 51, 8192, 2048, 144),
    ("stage1:contract", 51, 2048, 8192, 384),
    ("stage2:contract", 51, 2048, 8192, 144),
)


def native(a, weights, scales):
    return torch.ops.quanto.qbytes_mm(a, weights, scales)


def fp16(a, weights, scales):
    # Exact experimental codepath: BF16 dequant, FP16 cast, GEMM, BF16 output.
    b16 = (scales * weights).to(torch.float16)
    a16 = a.to(torch.float16)
    return torch.mm(a16, b16.t()).to(torch.bfloat16)


def sample(fn, a, weights, scales, rep):
    for _ in range(2):
        fn(a,weights,scales)
    torch.cuda.synchronize()
    values=[]
    for _ in range(rep):
        start=torch.cuda.Event(enable_timing=True)
        end=torch.cuda.Event(enable_timing=True)
        start.record()
        out=fn(a,weights,scales)
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
        del out
    return statistics.median(values)


def main():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (7,0):
        raise SystemExit("Tesla V100 sm70 required")
    torch.manual_seed(20261114)
    rows=[]
    for name,m,n,k,calls in WORKLOADS:
        a = torch.randn((m,k),device="cuda",dtype=torch.bfloat16) / 64
        weights=torch.randint(-18,18,(n,k),device="cuda",dtype=torch.int8)
        scales=torch.full((n,1),1/512,device="cuda",dtype=torch.bfloat16)
        ref=native(a,weights,scales)
        got=fp16(a,weights,scales)
        torch.cuda.synchronize()
        delta=got.float()-ref.float()
        nrmse=float((delta.square().mean().sqrt()/ref.float().square().mean().sqrt().clamp_min(1e-12)).item())
        finite=bool(torch.isfinite(got).all().item())
        # Interleave to reduce thermal / cache drift, compare medians of 6
        # samples for each arm. This is only a microbench (not a speedup claim).
        first_native=sample(native,a,weights,scales,3)
        first_fp16=sample(fp16,a,weights,scales,3)
        last_fp16=sample(fp16,a,weights,scales,3)
        last_native=sample(native,a,weights,scales,3)
        native_ms=(first_native+last_native)/2
        fp16_ms=(first_fp16+last_fp16)/2
        row={
            "name":name,"M":m,"N":n,"K":k,"calls":calls,
            "native_ms":round(native_ms,4),"fp16_ms":round(fp16_ms,4),
            "fp16_to_native_ratio":round(fp16_ms/native_ms,4),
            "projected_native_seconds":round(calls*native_ms/1000,2),
            "projected_fp16_seconds":round(calls*fp16_ms/1000,2),
            "projected_savings_seconds":round(calls*(native_ms-fp16_ms)/1000,2),
            "synthetic_relative_rms_error":round(nrmse,7),
            "output_finite":finite,
        }
        rows.append(row)
        print("QUANTO_SHAPE_RESULT "+json.dumps(row,sort_keys=True),flush=True)
        del a,weights,scales,ref,got,delta
        torch.cuda.empty_cache()
    result={
        "kind":"isolated real-shape synthetic BF16/INT8 Quanto vs FP16 GPU microbench",
        "valid_for_full_video_speedup_claims":False,
        "actual_model_data":False,
        "workloads":rows,
        "estimated_total_native_seconds":round(sum(x["projected_native_seconds"] for x in rows),3),
        "estimated_total_fp16_seconds":round(sum(x["projected_fp16_seconds"] for x in rows),3),
        "estimated_savings_seconds":round(sum(x["projected_savings_seconds"] for x in rows),3),
        "warning":"No model/attention/MMGP/synchronization contention. Per-shape projections are hypotheses only.",
    }
    file=Path(os.environ.get("LTX25_FP16_MICROBENCH_PATH","/tmp/fp16_quanto_shape_microbench.json"))
    file.parent.mkdir(parents=True,exist_ok=True)
    file.write_text(json.dumps(result,indent=2)+"\n")
    print("QUANTO_SHAPE_MICROBENCH_FINISHED "+json.dumps({
        "file":str(file),"estimated_savings_seconds":result["estimated_savings_seconds"]}),flush=True)


if __name__=="__main__":
    main()
