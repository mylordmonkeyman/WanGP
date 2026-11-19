#!/usr/bin/env python3
"""P5 isolated Hunyuan 1.5 V100 attention kernel/layout screen.

Uses the EXACT 480p forward shape from the P0 request trace:
B=1, S=14159, H=16, D=128, all Q/K tokens valid; also tests a
720p-sized exploratory shape (S=32519, not yet instrumented).
Tests GPU elapsed including layout conversions. No production patch,
external dependencies, or GPU model loading. Control SDPA remains default.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import statistics
import traceback


SHAPES = ((14159, "480p_trace_exact"), (32519, "720p_geometry_estimate"))
HEADS = 16
DIM = 128
SEED = 20261112


def benchmark(torch, name, call, reps=8, warmups=3):
    for _ in range(warmups):
        result=call()
    torch.cuda.synchronize()
    times=[]
    for _ in range(reps):
        start=torch.cuda.Event(enable_timing=True)
        end=torch.cuda.Event(enable_timing=True)
        start.record()
        result=call()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return {"name":name,"median_ms":statistics.median(times),
            "min_ms":min(times),"samples_ms":times,"last_shape":list(result.shape)}


def run_one(torch,seq,label,enable_xformers):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import torch.nn.functional as F
    torch.manual_seed(SEED)
    device=torch.device("cuda:0")
    dtype=torch.float16
    q=torch.randn((1,seq,HEADS,DIM),device=device,dtype=dtype)*0.25
    k=torch.randn_like(q)*0.25
    v=torch.randn_like(q)*0.25
    qh=q.transpose(1,2)
    kh=k.transpose(1,2)
    vh=v.transpose(1,2)

    def sdpa_strided():
        return F.scaled_dot_product_attention(qh,kh,vh,dropout_p=0).transpose(1,2)
    def sdpa_contiguous():
        # Includes all three copies; never measure only the faster kernel
        # while omitting the cost of the required layout transformation.
        return F.scaled_dot_product_attention(
            qh.contiguous(),kh.contiguous(),vh.contiguous(),dropout_p=0).transpose(1,2)

    ref=sdpa_strided()
    torch.cuda.synchronize()
    variants=[("sdpa_strided",sdpa_strided),("sdpa_contiguous_with_copies",sdpa_contiguous)]
    if enable_xformers:
        try:
            import xformers.ops as xf
            variants.append(("xformers_memory_efficient",lambda:xf.memory_efficient_attention(q,k,v)))
        except Exception as e:
            print("P5_XFORMERS_UNAVAILABLE "+str(e),flush=True)

    results={}
    for name,fn in variants:
        try:
            with torch.inference_mode():
                output=fn()
                torch.cuda.synchronize()
                max_abs=(output.float()-ref.float()).abs().max().item()
                # Ref output for our synthetic FP16 workload. A speedup is not
                # eligible for production until real-video parity is verified.
                scale=max(0.03,float(ref.abs().max())*0.02)
                finite=bool(torch.isfinite(output).all().item())
                if not finite or max_abs>scale:
                    raise RuntimeError(f"Numerics fail: max_abs={max_abs}, limit={scale}")
                timing=benchmark(torch,name,fn)
                timing["max_abs_error_fp16"]=max_abs
                timing["reference_tolerance"]=scale
                timing["peak_allocated_gib"]=torch.cuda.max_memory_allocated()/2**30
                results[name]=timing
                print("P5_ATTENTION_LAYOUT_CASE "+json.dumps(
                    {"seq":seq,"label":label,**timing},sort_keys=True),flush=True)
        except Exception as e:
            traceback.print_exc()
            results[name]={"status":"failed","error":str(e)}
            print("P5_ATTENTION_LAYOUT_FAILURE "+json.dumps(
                {"seq":seq,"variant":name,"error":str(e)}),flush=True)
        finally:
            torch.cuda.reset_peak_memory_stats()
    base=results.get("sdpa_strided",{})
    if "median_ms" not in base:
        raise RuntimeError("Essential baseline SDPA failed")
    for name,row in results.items():
        if "median_ms" in row:
            row["speedup_percent"]=100*(base["median_ms"]/row["median_ms"]-1)
    del q,k,v,qh,kh,vh,ref
    torch.cuda.empty_cache()
    return {"seq":seq,"label":label,"head_dim":DIM,"heads":HEADS,"batch":1,
            "dtype":"torch.float16","backend_results":results}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=Path("p5-attention-layout-report.json"))
    p.add_argument("--include-720-estimate",action="store_true")
    args=p.parse_args()
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)!=(7,0):
        raise RuntimeError("This experiment requires the SM70 V100")
    print("P5_ATTENTION_TORCH "+json.dumps({"torch":torch.__version__,
        "cuda_runtime":torch.version.cuda,
        "gpu":torch.cuda.get_device_name(0)}),flush=True)
    rows=[run_one(torch,*SHAPES[0],True)]
    if args.include_720_estimate:
        rows.append(run_one(torch,*SHAPES[1],True))
    result={"status":"P5_EXPLORATORY_ONLY_NO_DEFAULT_CHANGE","rows":rows,
       "provenance":"S=14159,H=16,D=128 from 2026-11-17 run 44019009553 request trace; 720 S estimate not traced",
       "limitations":["Synthetic tensors are not real Q/K/V; no video fidelity measurement",
                      "GPU-event measurements include optional contiguous copies",
                      "Only PyTorch SDPA and preinstalled xFormers considered",
                      "Do not promote without real Hunyuan FP16 end-to-end test"]}
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print("P5_ATTENTION_LAYOUT_REPORT "+json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
