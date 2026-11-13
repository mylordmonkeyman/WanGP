#!/usr/bin/env python3
"""V100 exact full-ConvRot preflight for shape-specific cuBLAS Algo2.

No model required. Compare candidate tuned GEMM versus validated default
cuBLAS GEMM after the SAME FP32 Hadamard, vector4 cast, range preparation,
FP32 exceptional-row fallback and scaling. Require bit-exact final BF16/
FP16 outputs; do not treat isolated GEMM exactness as enough for promotion.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from shared.kernels import ltx25_cublas_sm70 as e


def main():
    assert e.MODE=="fp32rot"
    assert e.GEMM_TUNE=="algo2_m64"
    assert e.ROT_MODE in ("torch",""),"Must keep validated torch H256"
    assert torch.cuda.get_device_capability()==(7,0)
    torch.manual_seed(20261113)
    n,k=16384,4096
    q=torch.randint(-127,128,(n,k),device="cuda",dtype=torch.int8)
    original=q.clone()
    scale=torch.randn((n,1),device="cuda",dtype=torch.float32)*0.001
    wt=SimpleNamespace(_data=q,_scale=scale)
    cases=[]
    with torch.inference_mode():
        for m in (64,144):
            for dtype in (torch.bfloat16,torch.float16):
                x=torch.randn((m,k),device="cuda").to(dtype)
                # Nonfinite not injected; high-magnitude row triggers
                # the existing GPU-side FP32 exceptional-row path.
                for extreme in (False, True):
                    xx=x.clone()
                    if extreme:
                        # 40000 is finite in both FP16 and BF16, and the
                        # H256 of a uniform row stays close to 40000:
                        # this triggers the >32768 GPU FP32 fallback
                        # without injecting infinities or NaN payloads.
                        xx[0].fill_(40000.0)
                    assert bool(torch.isfinite(xx).all().item()), "Test must use finite inputs"
                    _=e.maybe_linear(xx,wt,None,256)
                    before=e._STATS["gemm_algo2_calls"]
                    with patch.object(torch.Tensor,"item",side_effect=AssertionError("host scalar read")), \
                         patch.object(torch.cuda,"synchronize",side_effect=AssertionError("global CUDA sync")):
                        e.GEMM_TUNE="algo2_m64"
                        fast=e.maybe_linear(xx,wt,None,256)
                        e.GEMM_TUNE="off"
                        base=e.maybe_linear(xx,wt,None,256)
                        e.GEMM_TUNE="algo2_m64"
                    assert fast is not None and base is not None
                    torch.cuda.synchronize()
                    assert torch.equal(fast,base),(
                        f"Algo2 full adapter result differs at m={m},dtype={dtype},extreme={extreme}")
                    delta=e._STATS["gemm_algo2_calls"]-before
                    # The prime/warm call is outside before, and the
                    # diagnostic comparison calls Algo2 only for M64.
                    assert delta==(1 if m==64 else 0),(m,delta)
                    cases.append({"N":n,"K":k,"M":m,"dtype":str(dtype),
                                  "extreme_row":extreme,"bit_exact":True,
                                  "algorithm2_selected":m==64})
        assert torch.equal(q,original),"Original INT8 weights mutated"
        assert len(cases)==8
    print("LTX25_GEMM_ALGO2_PREFLIGHT "+json.dumps({
        "status":"PASS","cases":cases,
        "exactly_one_shape_selected":[n,k,64],
        "original_int8_unchanged":True,
        "host_sync_inside_adapter":False,
        "production_eligible":False,
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
