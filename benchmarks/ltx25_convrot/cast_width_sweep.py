#!/usr/bin/env python3
"""Synthetic V100 SM70 cast-width tuning. Production vector4 is baseline.

Vector8/16 packed loads + 128/256/512 CTA threads and persistent-grid variants.
Correctness is exact INT8-to-FP16 bit equality with PyTorch for all bytes.
Uses A/B/B/A repetitions and CUDA events. Not a complete-video benchmark.
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

SHAPES=[(2048,2048),(2048,4096),(4096,4096),
        (4096,16384),(16384,4096)]
NAMES=[
    "production_vector4_256",
    "packed8_256", "packed16_256",
    "packed8_128", "packed16_128",
    "packed8_512", "packed16_512",
    "packed8_256_grid640", "packed16_256_grid640",
]
LIB=os.environ.get("LTX25_CAST_SWEEP_LIB","/tmp/libltx25_cast_width_sm70.so")
OUT=Path("/cache/ltx25/results")
SHA=os.environ.get("GITHUB_SHA","local")
TRIALS=32
WARMUP=6

def timer(fn):
    for _ in range(WARMUP):
        rc=fn()
        assert rc==0, f"CUDA warmup failed ({rc})"
    torch.cuda.synchronize()
    events=[]
    for _ in range(TRIALS):
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin.record()
        rc=fn()
        assert rc==0, f"CUDA run failed ({rc})"
        end.record()
        events.append((begin,end))
    torch.cuda.synchronize()
    times=[float(a.elapsed_time(b)) for a,b in events]
    return round(statistics.median(times),6)

def main():
    assert torch.cuda.get_device_capability(0)==(7,0), "Only Tesla V100 SM70"
    dll=ctypes.CDLL(LIB)
    probe=dll.ltx25_cast_width_probe
    ptr=ctypes.c_void_p
    probe.argtypes=[ptr,ptr,ctypes.c_size_t,ctypes.c_int,ptr]
    probe.restype=ctypes.c_int
    stream=ptr(torch.cuda.current_stream().cuda_stream)
    torch.manual_seed(20261113)
    rows=[]
    for n,k in SHAPES:
        q=torch.randint(-128,128,(n,k),dtype=torch.int8,device="cuda")
        expected=q.to(torch.float16)
        actual=torch.empty_like(expected)
        p=ptr(q.data_ptr())
        o=ptr(actual.data_ptr())
        def call(i):
            return probe(p,o,q.numel(),i,stream)
        results=[]
        # Bitwise check for the entire tensor before any timed runs.
        for idx,name in enumerate(NAMES):
            assert call(idx)==0, f"{name}: CUDA error"
            torch.cuda.synchronize()
            if not torch.equal(expected,actual):
                diff=(expected!=actual).sum().item()
                raise AssertionError(f"{name} {n}x{k}: {diff} wrong FP16 values")
        # Pairs in alternating order, so thermal/order drift does not favor
        # one candidate. Separate CUDA events, no cross-device synchronization
        # in any conversion itself.
        base_a=timer(lambda:call(0))
        for idx,name in enumerate(NAMES):
            if idx==0: continue
            t1=timer(lambda idx=idx:call(idx))
            t2=timer(lambda idx=idx:call(idx))
            b2=timer(lambda:call(0))
            baseline=round(statistics.median((base_a,b2)),6)
            variant=round(statistics.median((t1,t2)),6)
            results.append({
                "name":name,"variant":idx,
                "baseline_vector4_ms":baseline,
                "candidate_ms":variant,
                "speedup":round(baseline/variant,5),
                "candidate_rounds_ms":[t1,t2],
                "baseline_rounds_ms":[base_a,b2],
                "byte_traffic_mib":round(3*q.numel()/1048576,2),
                "effective_bandwidth_gib_s":round(3*q.numel()/(variant/1000)/1073741824,3),
                "exact_fp16_output":True,
            })
            base_a=b2
        winner=max(results,key=lambda x:x["speedup"])
        row={"n":n,"k":k,"elements":q.numel(),
             "baseline_vector4_ms":base_a,
             "best_variant":winner["name"],"best_speedup":winner["speedup"],
             "variants":results}
        rows.append(row)
        print("LTX25_CAST_WIDTH_SHAPE "+json.dumps({
            "n":n,"k":k,"best_variant":row["best_variant"],
            "best_speedup":row["best_speedup"],
            "results":[{"name":x["name"],"speedup":x["speedup"],
                        "candidate_ms":x["candidate_ms"]} for x in results]
        },sort_keys=True),flush=True)
        del q,expected,actual
    report={
        "kind":"SM70 isolated exact INT8-to-FP16 packed-width sweep",
        "status":"PASS" if len(rows)==len(SHAPES) else "INCOMPLETE",
        "run_id":os.environ.get("GITHUB_RUN_ID","local"),
        "source_sha":SHA,
        "utc":dt.datetime.now(dt.timezone.utc).isoformat(),
        "device":torch.cuda.get_device_name(0),
        "synthetic_only":True,
        "production_eligible":False,
        "shapes":rows,
    }
    OUT.mkdir(parents=True,exist_ok=True)
    result_file=OUT/f"cast_width_sweep_{SHA[:12]}.json"
    result_file.write_text(json.dumps(report,indent=2)+"\n")
    print("LTX25_CAST_WIDTH_SUMMARY "+json.dumps({
        "status":report["status"],"shapes":len(rows),
        "winning_variants":[[v["n"],v["k"],v["best_variant"],v["best_speedup"]] for v in rows],
    },sort_keys=True),flush=True)

if __name__=="__main__":
    main()
