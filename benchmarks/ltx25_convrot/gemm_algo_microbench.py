#!/usr/bin/env python3
"""SM70 cublasGemmEx algorithm sweep, exact FP32 output requirement.

Synthetic checkpoint-sized INT8->FP16 temporary weights and FP16 operands.
No model loaded, no production code edited, no persistent second weight set.
Selectors -1 default tensor op, -2 default, 0..23 explicit tensor-op.
Unsupported selectors are reported instead of silently falling back.
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
LIB=os.environ.get("LTX25_GEMM_PROBE_LIB","/tmp/ltx25_gemm_algo_sm70.so")
SHAPES=[
    (2048,2048,128),
    (4096,4096,128),
    (16384,4096,64),
    (16384,4096,144),
    (16384,4096,1024),
    (4096,16384,64),
    (4096,16384,144),
    (4096,16384,1024),
]
SELECTORS=[-1,-2,*range(24)]
WARMUP=5
TRIALS=30


def time_cuda(fn):
    for _ in range(WARMUP):
        rc=fn()
        if rc: raise RuntimeError(f"warmup cuBLAS status {rc}")
    torch.cuda.synchronize()
    events=[]
    for _ in range(TRIALS):
        a,z=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        rc=fn()
        if rc: raise RuntimeError(f"trial cuBLAS status {rc}")
        z.record()
        events.append((a,z))
    torch.cuda.synchronize()
    times=sorted(float(a.elapsed_time(z)) for a,z in events)
    return {
        "median_ms":round(statistics.median(times),5),
        "min_ms":round(times[0],5),
        "p90_ms":round(times[math.ceil(0.9*len(times))-1],5),
        "trial_count":TRIALS,
    }


def main():
    assert torch.cuda.get_device_capability()==(7,0)
    torch.manual_seed(20261113)
    dll=ctypes.CDLL(LIB)
    fn=dll.ltx25_cublas_gemm_algo_probe
    ptr=ctypes.c_void_p
    fn.argtypes=[ptr,ptr,ptr,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ptr]
    fn.restype=ctypes.c_int
    stream=ptr(torch.cuda.current_stream().cuda_stream)
    results=[]
    with torch.inference_mode():
        for n,k,m in SHAPES:
            # Representable INT8 weights expanded into ONE temporary FP16
            # allocation; same as validated Candidate E. No model checkpoint.
            q=torch.randint(-127,128,(n,k),device="cuda",dtype=torch.int8)
            q16=q.to(torch.float16)
            x16=torch.randn((m,k),device="cuda",dtype=torch.float32).half()
            reference=torch.empty((m,n),device="cuda",dtype=torch.float32)
            target=torch.empty_like(reference)
            def call(sel,out):
                return fn(ptr(q16.data_ptr()),ptr(x16.data_ptr()),ptr(out.data_ptr()),
                          n,m,k,sel,stream)
            rc=call(-1,reference)
            assert rc==0,f"existing CUBLAS_GEMM_DEFAULT_TENSOR_OP failed {rc}"
            torch.cuda.synchronize()
            base_time=time_cuda(lambda:call(-1,reference))
            candidates=[]
            for sel in SELECTORS:
                rc=call(sel,target)
                if rc:
                    candidates.append({"selector":sel,"status":"UNSUPPORTED","cublas_code":rc})
                    continue
                torch.cuda.synchronize()
                identical=bool(torch.equal(reference,target))
                diff=(reference-target).abs()
                maximum=float((diff/torch.maximum(reference.abs(),torch.ones_like(reference))).max().item())
                frac=float((reference==target).float().mean().item())
                try:
                    timing=base_time if sel==-1 else time_cuda(lambda sel=sel:call(sel,target))
                except RuntimeError as exc:
                    candidates.append({"selector":sel,"status":"ERROR","error":str(exc)[:400]})
                    continue
                row={
                    "selector":sel,
                    "status":"PASS",
                    "bit_exact_fp32":identical,
                    "exact_element_fraction":frac,
                    "max_normalized_error":maximum,
                    "median_ms":timing["median_ms"],
                    "speedup_vs_reference":round(base_time["median_ms"]/timing["median_ms"],4),
                }
                candidates.append(row)
            exact=[c for c in candidates if c["status"]=="PASS" and c["bit_exact_fp32"]]
            faster=[c for c in exact if c["speedup_vs_reference"]>1.02]
            shape={
                "N":n,"K":k,"M":m,
                "baseline_median_ms":base_time["median_ms"],
                "supported":sum(c["status"]=="PASS" for c in candidates),
                "exact":len(exact),
                "faster_exact":len(faster),
                "best_bit_exact":min(exact,key=lambda x:x["median_ms"])["selector"] if exact else None,
                "candidates":candidates,
            }
            results.append(shape)
            print("LTX25_GEMM_SWEEP_SHAPE "+json.dumps({
                k:v for k,v in shape.items() if k!="candidates"
            },sort_keys=True),flush=True)
            del q,q16,x16,reference,target
    report={
        "status":"PASS",
        "source_sha":os.environ.get("GITHUB_SHA","local"),
        "run_id":os.environ.get("GITHUB_RUN_ID"),
        "utc":dt.datetime.now(dt.timezone.utc).isoformat(),
        "device":torch.cuda.get_device_name(),
        "kind":"SM70 isolated cublasGemmEx algorithm exact-output sweep",
        "bit_exact_required_for_promotion":True,
        "production_eligible":False,
        "synthetic_input_only":True,
        "model_generation":False,
        "shapes":results,
    }
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/f"gemm_algo_sweep_{report['source_sha'][:12]}.json"
    p.write_text(json.dumps(report,indent=2)+"\n")
    print("LTX25_GEMM_SWEEP_SUMMARY "+json.dumps({
        "status":report["status"],
        "shape_count":len(results),
        "exact_faster_shapes":sum(x["faster_exact"]>0 for x in results),
    }),flush=True)


if __name__=="__main__":
    main()
