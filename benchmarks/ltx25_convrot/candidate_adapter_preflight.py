#!/usr/bin/env python3
"""Small real CUDA/GEMM path check before loading the expensive 22B video model.

Uses synthetic activations and random INT8 weights of the same shape as the
known passing family. It is a wiring test, not a substitute for video quality.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import torch

from shared.kernels import ltx25_cublas_sm70 as e
from shared.qtypes.int8_convrot import _rotate_activation


def main():
    assert e.MODE == "fp32rot", e.MODE
    torch.manual_seed(20261112)
    n, k, m = 2048, 2048, 128
    q = torch.randint(-64,64,(n,k),dtype=torch.int8,device="cuda")
    scale = torch.full((n,1),0.0025,device="cuda",dtype=torch.float32)
    x = torch.randn((m,k),device="cuda",dtype=torch.float32).to(torch.bfloat16)
    wt = SimpleNamespace(_data=q,_scale=scale)
    with torch.inference_mode():
        y = e.maybe_linear(x,wt,None,256)
        assert y is not None, "Opt-in Candidate E unexpectedly fell back on a passing shape"
        assert y.shape==(m,n) and y.dtype==torch.bfloat16, (y.shape,y.dtype)
        y_ref = (torch.matmul(_rotate_activation(x.float(),256),q.float().T) * scale.T).to(torch.bfloat16)
        torch.cuda.synchronize()
        assert torch.isfinite(y).all().item(), "Candidate E output had nonfinite values"
        max_abs = float((y.float()-y_ref.float()).abs().max().item())
        rms_relative = float((y.float()-y_ref.float()).square().mean().sqrt().item() /
                             max(1e-6,y_ref.float().square().mean().sqrt().item()))
        assert rms_relative < 0.015, f"Candidate E RMS relative difference too large: {rms_relative}"
        disallowed = SimpleNamespace(
            _data=torch.zeros((4096,2048),device="cuda",dtype=torch.int8),
            _scale=torch.ones((4096,1),device="cuda",dtype=torch.float32))
        assert e.maybe_linear(x,disallowed,None,256) is None, "Known failing shape was not excluded"
    print("LTX25_CANDIDATE_ADAPTER_SM70_OK "+json.dumps({
        "M":m,"N":n,"K":k,"max_abs_error":max_abs,
        "rms_relative_error":rms_relative,
        "candidate_calls":e._STATS["used"],
        "known_bad_shape_excluded":True,
        "synthetic_only":True}),flush=True)


if __name__=="__main__":
    main()
