#!/usr/bin/env python3
"""SM70 baseline vector4 INT8 cast + cuBLAS vs tiled INT8->WMMA-fused GEMM.

Measures the SAME matrix operation: q[N,K] int8 @ x[M,K] fp16 -> C[M,N] fp32.
The fusion expands q only into 64x32 CTA shared tiles; it does not retain
FP16 weight copies or change the production WanGP dispatcher.
Synthetic test only; accuracy and speed must be measured before integration.
"""
from __future__ import annotations
import ctypes
import datetime as dt
import json
import math
import os
from pathlib import Path
import statistics

import torch

OUT=Path("/cache/ltx25/results")
LIB=os.environ.get("LTX25_FUSED_WMMA_LIB", "/tmp/ltx25_fused_i8_wmma_sm70.so")
# (N,K,M) real eligible shape families, with WMMA row multiples 32.
SHAPES=((16384,4096,64),(4096,16384,64),(4096,4096,128),
        (2048,4096,128),(16384,4096,1024))
WARMUP=4
TRIALS=22


def samples(fn):
    for _ in range(WARMUP):
        assert fn()==0, "warmup failed"
    torch.cuda.synchronize()
    times=[]
    for _ in range(TRIALS):
        a,z=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        status=fn()
        if status:
            raise RuntimeError(f"kernel status={status}")
        z.record()
        times.append((a,z))
    torch.cuda.synchronize()
    timings=[float(a.elapsed_time(z)) for a,z in times]
    return round(statistics.median(timings),5),round(min(timings),5)


def main():
    assert torch.cuda.get_device_capability()==(7,0), "Experiment requires SM70 V100"
    torch.manual_seed(20261113)
    dll=ctypes.CDLL(LIB)
    ptr=ctypes.c_void_p
    fused=dll.ltx25_fused_i8_wmma
    gemm=dll.ltx25_baseline_gemm
    cast=dll.ltx25_baseline_cast4
    for fn in (fused,gemm):
        fn.argtypes=[ptr,ptr,ptr,ctypes.c_int,ctypes.c_int,ctypes.c_int,ptr]
        fn.restype=ctypes.c_int
    cast.argtypes=[ptr,ptr,ctypes.c_size_t,ptr]
    cast.restype=ctypes.c_int
    stream=ptr(torch.cuda.current_stream().cuda_stream)
    result=[]
    with torch.inference_mode():
        for n,k,m in SHAPES:
            q=torch.randint(-127,128,(n,k),dtype=torch.int8,device="cuda")
            x16=(torch.randn((m,k),dtype=torch.float32,device="cuda")*0.2).half()
            # Only the historical baseline allocates full ephemeral FP16
            # weights; the fused call receives original INT8 directly.
            w16=torch.empty((n,k),dtype=torch.float16,device="cuda")
            baseline=torch.empty((m,n),dtype=torch.float32,device="cuda")
            target=torch.empty_like(baseline)

            qptr,xptr,wptr=ptr(q.data_ptr()),ptr(x16.data_ptr()),ptr(w16.data_ptr())
            bptr,tptr=ptr(baseline.data_ptr()),ptr(target.data_ptr())
            def cast_call():
                return cast(qptr,wptr,q.numel(),stream)
            def gemm_call():
                return gemm(wptr,xptr,bptr,n,m,k,stream)
            def baseline_call():
                rc=cast_call()
                return rc or gemm_call()
            def fused_call():
                return fused(qptr,xptr,tptr,n,m,k,stream)

            assert baseline_call()==0
            assert fused_call()==0
            torch.cuda.synchronize()
            diff=(target-baseline).abs()
            norm=diff/torch.maximum(torch.ones_like(baseline),baseline.abs())
            rel=float((diff.square().mean().sqrt()/
                       baseline.square().mean().sqrt().clamp_min(1e-8)).item())
            mx=float(norm.max().item())
            p99=float(torch.quantile(norm.flatten(),0.99).item())
            identical=float((baseline==target).float().mean().item())
            finite=bool(torch.isfinite(target).all().item())
            numeric_ok=finite and rel<0.003 and mx<0.03
            # Alternate baseline and fused timings; both operate on
            # warmed allocations with the same stream and GPU.
            b1,bmin1=samples(baseline_call)
            f1,fmin1=samples(fused_call)
            f2,fmin2=samples(fused_call)
            b2,bmin2=samples(baseline_call)
            bmed=round(statistics.median((b1,b2)),5)
            fmed=round(statistics.median((f1,f2)),5)
            cast_ms,_=samples(cast_call)
            gemm_ms,_=samples(gemm_call)
            row={
                "n":n,"k":k,"m":m,
                "baseline_cast_plus_cublas_ms":bmed,
                "fused_int8_wmma_ms":fmed,
                "baseline_cast_only_ms":cast_ms,
                "baseline_gemm_only_ms":gemm_ms,
                "speedup_fused_vs_baseline":round(bmed/fmed,4),
                "baseline_rounds_ms":[b1,b2],
                "fused_rounds_ms":[f1,f2],
                "numeric_pass":numeric_ok,
                "fp32_exact_fraction":identical,
                "fp32_relative_rms_error":rel,
                "fp32_max_normalized_error":mx,
                "fp32_p99_normalized_error":p99,
                "fp32_finite":finite,
                "persistent_fp16_weights_in_fused_path":False,
                "baseline_temp_fp16_weight_mib":round(2*n*k/(1<<20),1),
            }
            result.append(row)
            print("LTX25_FUSED_INT8_WMMA_SHAPE "+json.dumps(row,sort_keys=True),flush=True)
            del q,x16,w16,baseline,target,diff,norm
            torch.cuda.empty_cache()
    report={
        "status":"PASS" if len(result)==len(SHAPES) else "INCOMPLETE",
        "source_sha":os.environ.get("GITHUB_SHA","local"),
        "run_id":os.environ.get("GITHUB_RUN_ID"),
        "utc":dt.datetime.now(dt.timezone.utc).isoformat(),
        "gpu":torch.cuda.get_device_name(),
        "kind":"isolated SM70 tiled INT8->WMMA fused GEMM vs exact vector4+cuBLAS pipeline",
        "production_eligible":False,
        "synthetic_only":True,
        "no_model_or_video_test":True,
        "shapes":result,
    }
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/f"fused_int8_wmma_{report['source_sha'][:12]}.json"
    p.write_text(json.dumps(report,indent=2)+"\n")
    print("LTX25_FUSED_INT8_WMMA_SUMMARY "+json.dumps({
        "status":report["status"],
        "shape_count":len(result),
        "numeric_pass_count":sum(x["numeric_pass"] for x in result),
        "speedup_gt_1_count":sum(x["speedup_fused_vs_baseline"]>1 for x in result),
        "report":str(p)
    }),flush=True)


if __name__=="__main__":
    main()
