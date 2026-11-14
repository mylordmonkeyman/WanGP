#!/usr/bin/env python3
"""V100 synthetic GEMM microbench: non-Quanto BF16 shapes observed in real LTX-2.5.

CUDA Event timing of native BF16 GEMMs vs transient FP16 operands + FP16
Tensor Core GEMM + BF16 output, retaining original input BF16 tensors.
DOES NOT patch production path, mutate model weights or claim video fidelity.
Native timings contain torch.mm / torch.addmm (not Quanto).
Observations/counts are from measured LTX-2.5 49-frame one-forward census.
"""
import csv,json,math,os,time
from pathlib import Path
from collections import defaultdict

import torch

OUT=Path(os.environ.get("LTX25_NONQUANTO_BENCH_ROOT","/cache/ltx25/results/nonquanto-fp16-microbench-manual"))
SHAPES=[
 # phase,operator,M,N,K,measured calls per warmed forward
 ("stage1","mm",408,256,256,528),
 ("stage1","mm",8064,256,256,48),
 ("stage1","mm",1632,256,256,48),
 ("stage1","addmm",1008,32,4096,144),
 ("stage1","addmm",51,32,2048,144),
 ("stage2","mm",408,256,256,528),
 ("stage2","mm",32256,256,256,48),
 ("stage2","mm",1632,256,256,48),
 ("stage2","addmm",4032,32,4096,144),
 ("stage2","addmm",51,32,2048,144),
]
def main():
    torch.manual_seed(20261112)
    assert torch.cuda.is_available() and torch.cuda.get_device_capability(0)==(7,0)
    torch.backends.cuda.matmul.allow_tf32=False
    OUT.mkdir(parents=True,exist_ok=True)
    data=[]
    for phase,op,m,n,k,calls in SHAPES:
        print("BENCH_START",phase,op,m,n,k,calls,flush=True)
        # Values bounded to avoid FP16 overflow; quantization error exists.
        a=(torch.randn((m,k),device="cuda",dtype=torch.float32)*0.02).bfloat16()
        b=(torch.randn((k,n),device="cuda",dtype=torch.float32)*0.02).bfloat16()
        bias=torch.randn((n,),device="cuda",dtype=torch.float32).bfloat16()
        def native():
            return torch.mm(a,b) if op=="mm" else torch.addmm(bias,a,b)
        def transient_fp16():
            out=torch.mm(a.half(),b.half())
            if op=="addmm":
                return (out.float()+bias.float()).bfloat16()
            return out.bfloat16()
        for _ in range(8):
            native()
            transient_fp16()
        torch.cuda.synchronize()
        reps=50 if m>10000 else 100
        # Alternate A B B A to minimize thermal drift in synthetic timings.
        def time_kernel(fn):
            start=torch.cuda.Event(enable_timing=True)
            stop=torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(reps):fn()
            stop.record()
            stop.synchronize()
            return float(start.elapsed_time(stop))/reps
        native_ms_1=time_kernel(native)
        fp16_ms_1=time_kernel(transient_fp16)
        fp16_ms_2=time_kernel(transient_fp16)
        native_ms_2=time_kernel(native)
        native_ms=(native_ms_1+native_ms_2)/2
        fp16_ms=(fp16_ms_1+fp16_ms_2)/2
        ref=native().float()
        out=transient_fp16().float()
        norm=float(torch.sqrt(ref.square().mean()).item())
        err=out-ref
        delta_rms=float(torch.sqrt(err.square().mean()).item())
        finite=bool(torch.isfinite(out).all().item())
        row={
            "phase":phase,"op":op,"M":m,"N":n,"K":k,
            "observed_calls_per_forward":calls,
            "native_bf16_ms_per_call":round(native_ms,6),
            "transient_fp16_ms_per_call":round(fp16_ms,6),
            "delta_ms_per_call":round(native_ms-fp16_ms,6),
            "estimated_ms_per_forward_saved":round(calls*(native_ms-fp16_ms),4),
            "speed_ratio":round(native_ms/fp16_ms,4),
            "normalized_rms_error":round(delta_rms/max(norm,1e-12),8),
            "max_abs_error":round(float(err.abs().max().item()),6),
            "finite":finite,
            "observed_calls_are_from":"2026-11-14 matched real V100 per-stage GEMM census",
        }
        data.append(row)
        print("NONQUANTO_GEMM_BENCH "+json.dumps(row,sort_keys=True),flush=True)
        del a,b,bias
        torch.cuda.empty_cache()
    with (OUT/"microbench.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]))
        writer.writeheader();writer.writerows(data)
    phase_totals={p:round(sum(x["estimated_ms_per_forward_saved"] for x in data if x["phase"]==p),3)
                  for p in ("stage1","stage2")}
    # Default 8 Stage 1 transformer forwards + 3 Stage 2 forwards.
    whole_video_ms=(phase_totals["stage1"]*8+phase_totals["stage2"]*3)
    report={
        "phase_saved_gpu_ms_per_forward":phase_totals,
        "all_49_frame_estimated_gpu_ms_saved_if_all_covered":round(whole_video_ms,3),
        "all_49_frame_estimated_gpu_s_saved_if_all_covered":round(whole_video_ms/1000,3),
        "stage1_forward_count":8,"stage2_forward_count":3,
        "method":"Independent synthetic tensors with real model observed shapes and counts; FP16 temporary conversion included",
        "limitations":[
            "Not live model tensors. Numerics do not establish acceptable generated-video quality.",
            "CUDA Event kernel elapsed times omit Python launch dispatch overhead.",
            "Measured FP16 path uses transient conversion and restores BF16 outputs; bias at FP32 for addmm.",
            "In-model op interception could add expensive overhead, especially at tiny M/N.",
            "Only 10 common stage-and-shape cases tested, not all 940 non-Quanto ops.",
            "Stage totals assume 8+3 transformer forwards and full coverage of each sampled shape.",
            "CUDA-only savings are not equivalent to wall-to-wall speedup; memory transfers may dominate.",
        ],
        "rows":data,
    }
    (OUT/"microbench.json").write_text(json.dumps(report,indent=2)+"\n")
    print("NONQUANTO_GEMM_MICROBENCH_PASS "+json.dumps({
        "phase_saved_ms":phase_totals,"49frame_estimated_gpu_s":whole_video_ms/1000}),flush=True)

if __name__=="__main__":main()
