#!/usr/bin/env python3
"""Opt-in FP32 radix-4 Hadamard numerical gate for SM70 Candidate E.

Compare COMPLETE Candidate E outputs (including weight conversion, FP16
range reduction, cuBLAS, exceptional FP32 rows, scale/bias and output dtype)
to the already-validated torch Hadamard candidate. Rounding order can differ:
record exact-output fraction but require a strict quantitative error bound.

Synthetic checkpoint-shaped INT8 tensors only; does not establish broad
real-video/LoRA/MMGP quality equivalence. Do not enable as default unless the
separate complete-video gate passes.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from shared.kernels import ltx25_cublas_sm70 as e
from shared.qtypes.int8_convrot import _rotate_activation


def main():
    assert e.MODE=="fp32rot", e.MODE
    assert e.ROT_MODE=="radix4", e.ROT_MODE
    assert torch.cuda.get_device_capability()==(7,0)
    torch.manual_seed(20261113)
    results=[]
    shapes=((2048,2048,128),(4096,4096,128),
            (16384,4096,64),(4096,16384,64))
    with torch.inference_mode():
        for n,k,m in shapes:
            q=torch.randint(-64,65,(n,k),device="cuda",dtype=torch.int8)
            orig=q.clone()
            scale=torch.randn((n,1),device="cuda",dtype=torch.float32)*0.002
            wt=SimpleNamespace(_data=q,_scale=scale)
            for dtype in (torch.bfloat16,torch.float16):
                x=torch.randn((m,k),device="cuda").to(dtype)
                # Warmup CUDA extension, skip host synchronization only
                # INSIDE subsequent measured adapter calls.
                _=e.maybe_linear(x,wt,None,256)
                with patch.object(torch.Tensor,"item",side_effect=AssertionError("synchronous scalar read")), \
                     patch.object(torch.cuda,"synchronize",side_effect=AssertionError("global GPU sync")):
                    e.ROT_MODE="radix4"
                    y_fast=e.maybe_linear(x,wt,None,256)
                    e.ROT_MODE="torch"
                    y_base=e.maybe_linear(x,wt,None,256)
                    e.ROT_MODE="radix4"
                assert y_fast is not None and y_base is not None
                assert y_fast.dtype==dtype and y_base.dtype==dtype
                torch.cuda.synchronize()
                delta=(y_fast.float()-y_base.float()).abs()
                rel_rms=float((delta.square().mean().sqrt() /
                        y_base.float().square().mean().sqrt().clamp_min(1e-6)).item())
                normalized=delta/torch.maximum(torch.ones_like(delta),y_base.float().abs())
                max_normalized=float(normalized.max().item())
                p99_normalized=float(torch.quantile(normalized.flatten(),0.99).item())
                exact_frac=float((y_fast==y_base).float().mean().item())
                assert bool(torch.isfinite(y_fast).all().item())
                assert rel_rms<0.003 and max_normalized<0.03, (
                    f"Candidate E changed excessively: {(n,k,m,dtype)}, "
                    f"rms={rel_rms}, max_norm={max_normalized}")
                results.append({
                    "N":n,"K":k,"M":m,"dtype":str(dtype),
                    "full_adapter_relative_rms_error":rel_rms,
                    "full_adapter_max_normalized_error":max_normalized,
                    "full_adapter_p99_normalized_error":p99_normalized,
                    "full_adapter_exact_fraction":exact_frac,
                })
            assert torch.equal(q,orig),"Original checkpoint-style INT8 tensor was modified"
            del q,orig,scale
    assert len(results)==8
    print("LTX25_HADAMARD_PREFLIGHT "+json.dumps({
        "status":"PASS", "cases":results,
        "production_eligible":False,
        "tested_on_real_model":False,
        "native_extension":bool(e._ROT_FN),
        "original_int8_weights_unchanged":True,
        "host_sync_inside_adapter":False,
    },sort_keys=True),flush=True)

if __name__=="__main__":
    main()
