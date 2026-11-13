#!/usr/bin/env python3
"""Isolated SM70 cuBLASLt heuristic candidate versus exact cuBLAS GEMM.

Real *shapes* from the V100 49-frame video; random FP16 checkpoint-like
operands. No real model, checkpoint mutation, or production dispatch change.
Only bit-exact FP32 candidates are possible follow-on validation candidates.
Use pre-created descriptors/heuristics outside GPU timing and same-stream
cuBLASLt calls. Startup and heuristic overhead are NOT included in timings.
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
LIB=os.environ.get("LTX25_CUBLASLT_PROBE_LIB","/tmp/ltx25_cublaslt_sm70.so")
# Current 49-frame actual shape data, favoring the highest call counts.
SHAPES=[
    (16384,4096,64,1008),
    (16384,4096,144,2688),
    (16384,4096,512,1008),
    (4096,16384,144,2688),
    (4096,16384,512,1008),
    (4096,4096,1008,2304),
    (4096,4096,1024,1088),
    (4096,4096,4032,864),
    (2048,4096,1008,1152),
    (2048,2048,1024,1088),
]
WORKSPACE_MIB=(0,8)
MAX_HEURISTICS=12
WARMUP=3
TRIALS=16


def time_cuda(fn):
    for _ in range(WARMUP):
        rc=fn()
        if rc:
            raise RuntimeError(f"cuBLAS call failed in warmup: {rc}")
    torch.cuda.synchronize()
    events=[]
    for _ in range(TRIALS):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        rc=fn()
        if rc:
            raise RuntimeError(f"cuBLAS call failed during measured trial: {rc}")
        end.record()
        events.append((start,end))
    torch.cuda.synchronize()
    times=sorted(float(a.elapsed_time(b)) for a,b in events)
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
    p=ctypes.c_void_p
    create=dll.ltx25_lt_probe_create
    create.argtypes=[ctypes.POINTER(p),ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_size_t,ctypes.c_int]
    create.restype=ctypes.c_int
    count=dll.ltx25_lt_probe_count
    count.argtypes=[p];count.restype=ctypes.c_int
    workspace_size=dll.ltx25_lt_probe_workspace
    workspace_size.argtypes=[p,ctypes.c_int,ctypes.POINTER(ctypes.c_size_t)]
    workspace_size.restype=ctypes.c_int
    run=dll.ltx25_lt_probe_run
    run.argtypes=[p,ctypes.c_int,p,p,p,p,ctypes.c_size_t,p]
    run.restype=ctypes.c_int
    destroy=dll.ltx25_lt_probe_destroy
    destroy.argtypes=[p];destroy.restype=None
    ref_fn=dll.ltx25_cublas_gemm_algo_probe
    ref_fn.argtypes=[p,p,p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,p]
    ref_fn.restype=ctypes.c_int
    stream=p(torch.cuda.current_stream().cuda_stream)
    results=[]
    with torch.inference_mode():
        for n,k,m,model_calls in SHAPES:
            q=torch.randint(-127,128,(n,k),dtype=torch.int8,device="cuda")
            q16=q.to(torch.float16)
            x16=torch.randn((m,k),dtype=torch.float32,device="cuda").half()
            baseline=torch.empty((m,n),dtype=torch.float32,device="cuda")
            target=torch.empty_like(baseline)
            def ref_call():
                return ref_fn(p(q16.data_ptr()),p(x16.data_ptr()),
                    p(baseline.data_ptr()),n,m,k,-1,stream)
            assert ref_call()==0,"Default GEMM failed"
            torch.cuda.synchronize()
            base_time=time_cuda(ref_call)
            attempts=[]
            for budget_mib in WORKSPACE_MIB:
                ctx=p()
                available=budget_mib*1024*1024
                rc=create(ctypes.byref(ctx),n,m,k,available,MAX_HEURISTICS)
                if rc:
                    attempts.append({
                        "workspace_mib":budget_mib,"status":"UNSUPPORTED",
                        "create_code":rc})
                    continue
                try:
                    num=count(ctx)
                    if num<=0:
                        attempts.append({
                            "workspace_mib":budget_mib,"status":"NO_HEURISTICS",
                            "heuristic_count":num})
                        continue
                    ws=(torch.empty((available,),dtype=torch.uint8,device="cuda")
                        if available else None)
                    wp=p(ws.data_ptr()) if ws is not None else p()
                    for index in range(num):
                        required=ctypes.c_size_t()
                        rc=workspace_size(ctx,index,ctypes.byref(required))
                        if rc:
                            attempts.append({
                                "workspace_mib":budget_mib,
                                "index":index,"status":"WORKSPACE_ERROR","code":rc})
                            continue
                        def candidate_call():
                            return run(ctx,index,p(q16.data_ptr()),p(x16.data_ptr()),
                                p(target.data_ptr()),wp,available,stream)
                        rc=candidate_call()
                        if rc:
                            attempts.append({
                                "workspace_mib":budget_mib,"index":index,
                                "status":"UNSUPPORTED","code":rc})
                            continue
                        torch.cuda.synchronize()
                        # Full FP32 compare *before* timing, all same operands.
                        exact=bool(torch.equal(target,baseline))
                        finite=bool(torch.isfinite(target).all().item())
                        rel=(target-baseline).abs()/torch.maximum(
                            baseline.abs(),torch.ones_like(baseline))
                        maximum=float(rel.max().item())
                        fraction=float((target==baseline).float().mean().item())
                        try:
                            timing=time_cuda(candidate_call)
                        except RuntimeError as exc:
                            attempts.append({
                                "workspace_mib":budget_mib,"index":index,
                                "status":"TIMING_ERROR","detail":str(exc)[:250]})
                            continue
                        attempts.append({
                            "workspace_mib":budget_mib,"index":index,
                            "status":"PASS","required_workspace_bytes":required.value,
                            "bit_exact_fp32":exact,"finite":finite,
                            "max_normalized_error":maximum,
                            "exact_element_fraction":fraction,
                            "median_ms":timing["median_ms"],
                            "speedup_vs_cublas":round(
                                base_time["median_ms"]/timing["median_ms"],4),
                        })
                finally:
                    destroy(ctx)
            valid=[z for z in attempts if z.get("status")=="PASS" and z.get("bit_exact_fp32") and z.get("finite")]
            fastest=min(valid,key=lambda z:z["median_ms"]) if valid else None
            row={
                "N":n,"K":k,"M":m,"live_model_calls":model_calls,
                "baseline_cublas_ms":base_time["median_ms"],
                "supported":sum(z.get("status")=="PASS" for z in attempts),
                "exact_supported":len(valid),
                "best_exact":({
                    "workspace_mib":fastest["workspace_mib"],
                    "heuristic_rank":fastest["index"],
                    "median_ms":fastest["median_ms"],
                    "speedup_vs_cublas":fastest["speedup_vs_cublas"],
                } if fastest else None),
                "candidates":attempts,
            }
            results.append(row)
            print("LTX25_CUBLASLT_SHAPE "+json.dumps({
                key:value for key,value in row.items() if key!="candidates"
            },sort_keys=True),flush=True)
            del q,q16,x16,baseline,target
    report={
        "status":"PASS","utc":dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_sha":os.environ.get("GITHUB_SHA","local"),
        "run_id":os.environ.get("GITHUB_RUN_ID"),
        "kind":"SM70 isolated cublasLtMatmul heuristic sweep, FP16 operand FP32 output",
        "real_model_shapes_only":True,"synthetic_data":True,
        "same_exact_fp32_output_required_for_candidate":True,
        "workspaces_mib":WORKSPACE_MIB,
        "new_production_dispatch":False,"model_generation":False,
        "production_eligible":False,
        "shapes":results,
    }
    OUT.mkdir(parents=True,exist_ok=True)
    loc=OUT/f"cublaslt_heuristic_{report['source_sha'][:12]}.json"
    loc.write_text(json.dumps(report,indent=2)+"\n")
    print("LTX25_CUBLASLT_SUMMARY "+json.dumps({
        "status":report["status"],"shapes":len(results),
        "exact_faster_5pct":sum(z["best_exact"] is not None and
            z["best_exact"]["speedup_vs_cublas"]>=1.05 for z in results),
        "no_heuristic_shapes":sum(z["supported"]==0 for z in results),
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
