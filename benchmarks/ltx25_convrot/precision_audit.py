#!/usr/bin/env python3
"""Numerical stress screen of FP32-rotation + bounded cuBLAS on real INT8 layers.

Two distinct checkpoint tensors per shape family, three independent synthetic
seeds and M=16/256. Not a production eligibility validation: no real model
activations or MMGP offload. The original BF16 rotation + Quanto leaf is R0.
R1 rotates original BF16 inputs in FP32 then computes FP32 GEMM and scaling.
"""
from __future__ import annotations

import ctypes
import datetime as dt
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from phase0_probe import github_source_rotation
from phase0_weights import FILE, OUT, FAMILIES, stats, try_actual_qweight
from phase1_cublas_e import load_library, SCRATCH_CAP

SEEDS = (103, 407, 1601)
MS = (16, 256)


def selected(f):
    choices = {shape: [] for shape in FAMILIES}
    for name in f.keys():
        if not name.endswith(".weight"):
            continue
        sl = f.get_slice(name)
        shape = tuple(sl.get_shape())
        if shape not in choices or sl.get_dtype() != "I8":
            continue
        base = name[:-7]
        sname = base + ".weight_scale"
        if sname in f.keys() and f.get_slice(sname).get_dtype() == "F32":
            if len(choices[shape]) < 2:
                choices[shape].append((base, name, sname))
    return choices


def check_one(f, triple, rotate, gemm):
    base, qname, sname = triple
    q = f.get_tensor(qname).to("cuda")
    s = f.get_tensor(sname).to("cuda", dtype=torch.float32).reshape(-1)
    n, k = map(int, q.shape)
    if s.numel() == 1: s = s.expand(n)
    report = {"name": base, "N": n, "K": k, "cases": []}
    if q.numel()*2 > SCRATCH_CAP:
        report.update(status="SKIPPED", error="scratch cap")
        return report
    wrapped, err = try_actual_qweight(q, s[:, None].contiguous())
    if wrapped is None:
        report.update(status="ERROR", error=str(err)[:300])
        return report
    w_scratch = torch.empty((n,k), dtype=torch.float16, device="cuda")
    stream_ptr = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    w_scratch.copy_(q)
    with torch.inference_mode():
        for m in MS:
            x_scratch = torch.empty((m,k), dtype=torch.float16, device="cuda")
            c_scratch = torch.empty((m,n), dtype=torch.float32, device="cuda")
            out = torch.empty((m,n), dtype=torch.bfloat16, device="cuda")
            for seed in SEEDS:
                torch.manual_seed(seed)
                x = torch.randn((m,k), device="cuda", dtype=torch.float32).bfloat16()
                try:
                    xr32 = rotate(x.float(), 256)
                    ref = (xr32 @ q.float().T) * s.reshape(1,-1)
                    r0 = F.linear(rotate(x,256), wrapped)
                    x_scratch.copy_(xr32)
                    status = gemm(ctypes.c_void_p(w_scratch.data_ptr()),
                                  ctypes.c_void_p(x_scratch.data_ptr()),
                                  ctypes.c_void_p(c_scratch.data_ptr()),
                                  n,m,k,stream_ptr)
                    if status != 0: raise RuntimeError(f"cuBLAS status {status}")
                    c_scratch.mul_(s.reshape(1,-1))
                    out.copy_(c_scratch)
                    torch.cuda.synchronize()
                    refstats, candstats = stats(r0, ref), stats(out, ref)
                    precision_gate = (candstats["nonfinite"] == 0
                        and candstats["p99_normalized"] <= refstats["p99_normalized"] + 1e-8
                        and candstats["max_normalized"] <= refstats["max_normalized"] + 1e-8)
                    case = {"M":m, "seed":seed, "status":"PASS",
                            "gate":bool(precision_gate),
                            "R0_p99":refstats["p99_normalized"],
                            "R0_max":refstats["max_normalized"],
                            "candidate_p99":candstats["p99_normalized"],
                            "candidate_max":candstats["max_normalized"],
                            "candidate_nonfinite":candstats["nonfinite"]}
                except Exception as exc:
                    case = {"M":m, "seed":seed, "status":"ERROR",
                            "exception":type(exc).__name__, "error":str(exc)[:250]}
                report["cases"].append(case)
            del x_scratch, c_scratch, out
    report["status"] = "PASS" if all(x["status"]=="PASS" for x in report["cases"]) else "PARTIAL"
    return report


def main():
    sha = os.environ.get("GITHUB_SHA", "local")
    report = {"source_sha":sha, "utc":dt.datetime.now(dt.timezone.utc).isoformat(),
              "kind":"Checkpoint multi-layer synthetic numerical audit",
              "real_model_activations": False, "production_validated": False,
              "MS":MS, "seeds":SEEDS, "results":[]}
    try:
        gemm = load_library()
        rot = github_source_rotation()
        with safe_open(str(FILE), framework="pt", device="cpu") as f:
            choices = selected(f)
            for shape, triples in choices.items():
                for t in triples:
                    item = check_one(f, t, rot, gemm)
                    report["results"].append(item)
                    count=sum(case.get("gate",False) for case in item.get("cases",[]))
                    print("NUMERIC_AUDIT " + json.dumps({"N":item["N"],"K":item["K"],
                          "name":item["name"], "status":item["status"],
                          "passed":count, "total":len(item.get("cases", []))}),flush=True)
        cases=[case for layer in report["results"] for case in layer.get("cases",[])]
        report["total_cases"]=len(cases)
        report["gate_passes"]=sum(bool(x.get("gate")) for x in cases)
        report["gate_failures"]=sum(x.get("status")=="PASS" and not x.get("gate",False) for x in cases)
        report["execution_errors"]=sum(x.get("status")!="PASS" for x in cases)
        report["status"]="PASS" if report["execution_errors"]==0 else "PARTIAL"
    except Exception as exc:
        report.update(status="ERROR",error_type=type(exc).__name__,error=str(exc)[:500])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"precision_audit_{sha[:12]}.json").write_text(json.dumps(report, indent=2) + "\n")
    print("NUMERIC_AUDIT_SUMMARY " + json.dumps({k:v for k,v in report.items() if k!="results"}),flush=True)
    if report["status"]=="ERROR": raise SystemExit(1)

if __name__=="__main__":
    main()
