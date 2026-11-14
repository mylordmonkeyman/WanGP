"""Research-only CUDA kernel attribution from PyTorch/Kineto Chrome traces.

Use GPU events whose cat is "kernel"/"gpu_memcpy"/"gpu_memset" only.
Avoid double counting CPU "record_function" ranges as GPU kernels.
CUPTI runtime correlation IDs connect actual GPU kernels to CPU launch
events, which are located inside or outside LTX25 Quanto profiling scopes.
Attribution is best-effort; we report unresolved events explicitly.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json

_GPU_CATEGORIES = frozenset(("kernel", "gpu_memcpy", "gpu_memset"))
_CPU_RUNTIME_CATEGORIES = frozenset(("cuda_runtime", "cuda_driver", "hip_runtime"))

def _c(v):
    return str(v) if v is not None else ""

def _category(name,cat):
    n=name.lower()
    if cat=="gpu_memcpy" or "memcpy" in n or "memset" in n:
        return "memory_copy"
    if any(x in n for x in ("fmha","sdpa","attention","flash_attn","flashattention")):
        return "attention"
    if any(x in n for x in ("gemm","gemv","matmul","cublas","cutlass","magma",
                             "wmma","mma_", "mmq","mmv","hgemm","sgemm","bgemm")):
        return "gemm_or_gemv"
    return "other_or_unclassified"

def summarize_chrome_trace(trace_path: str | Path):
    raw=json.loads(Path(trace_path).read_text())
    ev=raw["traceEvents"] if isinstance(raw,dict) else raw
    spans=defaultdict(list)
    launches=defaultdict(list)
    kernels=[]
    meta_counts=Counter()
    for e in ev:
        if e.get("ph")!="X":continue
        cat=str(e.get("cat","")).lower()
        name=str(e.get("name",""))
        duration=float(e.get("dur",0) or 0)
        args=e.get("args",{}) or {}
        tid=(str(e.get("pid","")),str(e.get("tid","")))
        meta_counts[cat]+=1
        if cat in _GPU_CATEGORIES:
            kernels.append((e,name,cat,duration,args))
        elif cat in _CPU_RUNTIME_CATEGORIES or (
             name.startswith(("cudaLaunch","cuLaunch","cudaMemcpy","cudaMemset"))):
            correlation=_c(args.get("correlation",args.get("Correlation ID")))
            if correlation:
                launches[tid].append((float(e.get("ts",0)),correlation,name))
        else:
            if name.startswith(("aten::","quanto::","LTX25::")):
                start=float(e.get("ts",0))
                spans[tid].append((start,start+duration,name,args))
    launch_tags={}
    launch_op={}
    launch_input_types={}
    launch_counts=Counter()
    for tid, events in launches.items():
        begins=sorted(spans.get(tid,()),key=lambda x:x[0])
        events.sort(key=lambda x:x[0])
        active=[]
        cursor=0
        for ts,corr,name in events:
            while cursor<len(begins) and begins[cursor][0]<=ts:
                sp=begins[cursor]
                if sp[1]>ts:
                    active.append(sp)
                cursor+=1
            active=[sp for sp in active if sp[1]>ts]
            tagged=any(sp[2]=="LTX25::quanto_qbytes_mm" for sp in active)
            op=next((sp for sp in reversed(active) if sp[2] in (
                        "aten::mm","aten::addmm","aten::bmm","aten::baddbmm")),None)
            launch_tags[corr]=tagged
            launch_op[corr]=op[2] if op else "unresolved_cpu_op"
            dtype=(op[3].get("Input type",op[3].get("Input Dtype",[])) if op else [])
            launch_input_types[corr]=str(dtype)[:140]
            launch_counts["total"]+=1
            launch_counts["tagged_quanto" if tagged else "outside_quanto"]+=1
    totals=defaultdict(float)
    groups=defaultdict(lambda:{"calls":0,"gpu_ms":0.0})
    top=defaultdict(lambda:{"calls":0,"us":0.0,"tagged_quanto_us":0.0,
                             "other_us":0.0,"unattributed_us":0.0})
    op_rows=defaultdict(lambda:{"calls":0,"gpu_ms":0.0})
    gpu_categories=Counter()
    for event,name,cat,dur,args in kernels:
        correlation=_c(args.get("correlation",args.get("Correlation ID")))
        if not correlation or correlation not in launch_tags:
            attribution="unattributed"
        else:
            attribution="tagged_quanto" if launch_tags[correlation] else "outside_quanto"
        fam=_category(name,cat)
        totals["total_gpu_us"]+=dur
        totals[attribution+"_gpu_us"]+=dur
        totals[fam+"_gpu_us"]+=dur
        gpu_categories[cat]+=1
        key=(fam,attribution)
        g=groups[key]
        g["calls"]+=1;g["gpu_ms"]+=dur/1000
        k=top[(fam,name)]
        k["calls"]+=1
        k["us"]+=dur
        k[attribution+"_us" if attribution=="tagged_quanto" else (
            "other_us" if attribution=="outside_quanto" else "unattributed_us")]+=dur
        if fam=="gemm_or_gemv":
            op=(launch_op[correlation] if attribution!="unattributed" else "unknown")
            o=op_rows[(attribution,op)]
            o["calls"]+=1;o["gpu_ms"]+=dur/1000
    kernel_data=[{
        "family":fam,"kernel":name,
        "calls":row["calls"],
        "gpu_ms":round(row["us"]/1000,5),
        "tagged_quanto_gpu_ms":round(row["tagged_quanto_us"]/1000,5),
        "outside_quanto_gpu_ms":round(row["other_us"]/1000,5),
        "unattributed_gpu_ms":round(row["unattributed_us"]/1000,5),
    } for (fam,name),row in sorted(top.items(),key=lambda kv:-kv[1]["us"])[:150]]
    return {
        "source":"PyTorch export_chrome_trace GPU cat kernel/memcpy/memset, correlated to CPU CUDA launch",
        "actual_gpu_kernel_events":len(kernels),
        "gpu_event_category_counts":dict(gpu_categories),
        "cuda_runtime_launch_events_with_correlation":launch_counts["total"],
        "cuda_runtime_launch_events_inside_tagged_quanto":launch_counts["tagged_quanto"],
        "cuda_runtime_launch_events_outside_tagged_quanto":launch_counts["outside_quanto"],
        "gpu_time_ms_sum":round(totals["total_gpu_us"]/1000,5),
        "attribution_gpu_time_ms":{
            c:round(totals[c+"_gpu_us"]/1000,5)
            for c in ("tagged_quanto","outside_quanto","unattributed")
        },
        "gpu_kernel_family_time_ms":{
            c:round(totals[c+"_gpu_us"]/1000,5)
            for c in ("gemm_or_gemv","attention","memory_copy","other_or_unclassified")
        },
        "gemm_kernel_attribution":[{
            "kind":k,"operator":op,"calls":v["calls"],"gpu_ms":round(v["gpu_ms"],5)}
            for (k,op),v in sorted(op_rows.items(),key=lambda kv:-kv[1]["gpu_ms"])
        ],
        "family_attribution":[{
            "family":fam,"attribution":scope,
            "calls":v["calls"],"gpu_ms":round(v["gpu_ms"],5)}
            for (fam,scope),v in sorted(groups.items(),key=lambda kv:-kv[1]["gpu_ms"])
        ],
        "top_gpu_kernels":kernel_data,
        "scope_caveats":[
            "GPU time is summed kernel durations; parallel CUDA streams may overlap.",
            "Unattributed kernels have no matching CPU runtime correlation and cannot be called missed GEMMs.",
            "The sampled forward includes weight transfers and candidate E custom kernels.",
            "Some kernels (particularly attention) implement matrix multiplication but not named GEMM.",
            "CPU tag is the native Quanto qbytes_mm wrapper, not converted FP16.",
            "No inference that pure kernel time is reducible 1:1 from wall time.",
            "Only one warmed forward per transformer stage is sampled.",
        ],
    }
