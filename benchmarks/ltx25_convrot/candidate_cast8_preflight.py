#!/usr/bin/env python3
"""V100 opt-in vector8 cast full-adapter exactness check (synthetic inputs).

Run only with WAN2GP_LTX25_CUBLAS_E_MODE=fp32rot,
WAN2GP_LTX25_E_CAST=vector8, and compiled SM70 extension. The production default cast remains vector4; vector8 is opt-in.

Never use host scalar reads/synchronize *inside* either E linear call.
Compare CUDA vector8 E to same-call baseline PyTorch E bit-exactly; also
check the original INT8 checkpoint tensor is unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from shared.kernels import ltx25_cublas_sm70 as e


def main():
    assert e.MODE == "fp32rot" and e.CAST_MODE == "vector8"
    assert torch.cuda.get_device_capability(0) == (7, 0)
    torch.manual_seed(20261113)
    rows = []
    with torch.inference_mode():
        for n, k, m in ((2048, 2048, 128), (4096, 4096, 128),
                        (16384, 4096, 64), (4096, 16384, 64)):
            q = torch.randint(-128, 128, (n, k), device="cuda", dtype=torch.int8)
            original = q.clone()
            s = torch.randn((n, 1), device="cuda", dtype=torch.float32) * 0.001
            wt = SimpleNamespace(_data=q, _scale=s)
            for dtype in (torch.float16, torch.bfloat16):
                x = torch.randn((m, k), device="cuda").to(dtype)
                # Directly verify exact FP16 cast with signed extremes, then
                # verify complete Candidate E outputs on this shape/dtype.
                if dtype == torch.float16:
                    raw_half = torch.empty(q.shape, device="cuda", dtype=torch.float16)
                    import ctypes
                    ptr = ctypes.c_void_p
                    rc = e._cast8_fn()(ptr(q.data_ptr()), ptr(raw_half.data_ptr()),
                                       q.numel(), ptr(torch.cuda.current_stream().cuda_stream))
                    assert rc == 0
                    torch.cuda.synchronize()
                    assert torch.equal(raw_half, q.to(torch.float16)), "Vector8 bit-exact cast failed"
                # Prime cuBLAS and CUDA module loading outside the no-sync guard.
                _ = e.maybe_linear(x, wt, None, 256)
                with patch.object(torch.Tensor, "item", side_effect=AssertionError("CUDA host scalar")), \
                     patch.object(torch.cuda, "synchronize", side_effect=AssertionError("global CUDA synchronization")):
                    e.CAST_MODE = "vector8"
                    y_vec = e.maybe_linear(x, wt, None, 256)
                    e.CAST_MODE = "torch"
                    y_torch = e.maybe_linear(x, wt, None, 256)
                    e.CAST_MODE = "vector8"
                torch.cuda.synchronize()
                assert y_vec is not None and y_torch is not None
                assert torch.equal(y_vec, y_torch), f"Vector8 cast changed Candidate E result for {(n,k,m,dtype)}"
                assert torch.equal(q, original), "Native cast unexpectedly modified original INT8 weight"
                rows.append({"N": n, "K": k, "M": m, "dtype": str(dtype),
                             "full_adapter_exact": True, "original_int8_unchanged": True})
            del q, original, s
    assert len(rows) == 8
    print("LTX25_CAST8_PREFLIGHT " + json.dumps({
        "status": "PASS", "cases": rows, "cuda_host_syncs_inside_adapter": 0,
        "persistent_fp16_weight_copies": 0, "production_eligible": False,
    }), flush=True)


if __name__ == "__main__":
    main()
