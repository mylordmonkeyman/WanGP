#!/usr/bin/env python3
"""Short real-shape V100 attention experiment; no model/video generation."""
import json, os, statistics, time
from pathlib import Path
import torch
import torch.nn.functional as F

OUT=Path(os.environ.get("LTX25_REAL_SHAPE_OUT","/tmp/ltx25-real-shape"))
OUT.mkdir(parents=True,exist_ok=True)
torch.manual_seed(20261113)
assert torch.cuda.get_device_capability()==(7,0)
assert torch.__version__.startswith("2.10.0+cu128")
import flash_attn_v100 as cat
import vllm_flash_attn as zhin
device="cuda"
B,T,H,D=1,315,32,128
q,k,v=[torch.randn((B,T,H,D),device=device,dtype=torch.bfloat16)*0.125 for _ in range(3)]
cu=torch.tensor([0,T],dtype=torch.int32,device=device)
def sdpa(q,k,v):
    return F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),dropout_p=0.0).transpose(1,2)
def onecat(q,k,v):
    return cat.flash_attn_func(q,k,v,dropout_p=0.0,causal=False)
def onecat_bhmd(q,k,v):
    out=cat.flash_attn_bhmd_func(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),dropout_p=0.0,causal=False)
    return out.transpose(1,2)
def zhinianqin(q,k,v):
    return zhin.flash_attn_varlen_func(q=q.reshape(B*T,H,D),k=k.reshape(B*T,H,D),v=v.reshape(B*T,H,D),cu_seqlens_q=cu,cu_seqlens_k=cu,max_seqlen_q=T,max_seqlen_k=T,dropout_p=0.0,causal=False,window_size=(-1,-1)).reshape(B,T,H,D)
def fp16_then(fn):
    def run(q,k,v):
        return fn(q.to(torch.float16),k.to(torch.float16),v.to(torch.float16)).to(torch.bfloat16)
    return run
cases={"sdpa_bf16":sdpa,"sdpa_fp16_convert":fp16_then(sdpa),"1cat_fp16_convert":fp16_then(onecat),"1cat_bhmd_fp16_convert":fp16_then(onecat_bhmd),"zhinianqin_fp16_convert":fp16_then(zhinianqin)}
reference=sdpa(q,k,v)
torch.cuda.synchronize()
report={"shape":{"batch":B,"tokens":T,"heads":H,"head_dim":D,"dtype":"bf16"},"torch":torch.__version__,"gpu":torch.cuda.get_device_name(),"cases":{}}
def bench(fn):
    for _ in range(5): fn(q,k,v)
    torch.cuda.synchronize()
    times=[]; peaks=[]
    for round_ in range(5):
        torch.cuda.synchronize()
        before=torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(30):
            out=fn(q,k,v)
        end.record();end.synchronize()
        times.append(start.elapsed_time(end)/30)
        peaks.append(max(0,torch.cuda.max_memory_allocated()-before)/1048576)
        del out
    return round(statistics.median(times),6),round(max(peaks),3)
for name,fn in cases.items():
    entry={}
    try:
        with torch.inference_mode():
            out=fn(q,k,v)
            err=(out.float()-reference.float()).abs()
            entry["max_abs"]=float(err.max().item())
            entry["rms"]=float(err.square().mean().sqrt().item())
            entry["finite"]=bool(torch.isfinite(out).all().item())
            entry["accuracy_pass"]=entry["finite"] and entry["max_abs"]<=0.05 and entry["rms"]<=0.005
            del out,err
            if not entry["accuracy_pass"]: raise AssertionError("Numerical tolerance exceeded")
            entry["median_cuda_ms"],entry["peak_incremental_mib"]=bench(fn)
            entry["status"]="PASS"
    except Exception as e:
        entry["status"]="FAIL"
        entry["error"]=type(e).__name__+": "+str(e)
    report["cases"][name]=entry
    print(name,json.dumps(entry),flush=True)
(OUT/"real_shape_report.json").write_text(json.dumps(report,indent=2)+"\n")
if any(x["status"]!="PASS" for x in report["cases"].values()):
    raise SystemExit("One or more real-shape paths failed; see report")
