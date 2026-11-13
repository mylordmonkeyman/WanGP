#!/usr/bin/env python3
"""CUDA/GEMM wiring and numerical test before loading the expensive 22B model.

Tests BF16 and FP16 activation paths using synthetic INT8 weights. This is
a prerequisite gate, NOT a substitute for video output/visual validation.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

# GitHub Actions invokes this file by path rather than via `python -m`.
# Python otherwise places benchmarks/ltx25_convrot, not the WanGP repository
# root, on sys.path and cannot resolve the top-level `shared` package.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from shared.kernels import ltx25_cublas_sm70 as e
from shared.qtypes.int8_convrot import _rotate_activation


def main():
    assert e.MODE == "fp32rot", e.MODE
    torch.manual_seed(20261112)
    n, k, m = 2048, 2048, 128
    q = torch.randint(-64, 64, (n, k), dtype=torch.int8, device="cuda")
    scale = torch.full((n, 1), 0.0025, device="cuda", dtype=torch.float32)
    wt = SimpleNamespace(_data=q, _scale=scale)
    results = []
    with torch.inference_mode():
        for dtype in (torch.bfloat16, torch.float16):
            x = torch.randn((m, k), device="cuda", dtype=torch.float32).to(dtype)
            y = e.maybe_linear(x, wt, None, 256)
            assert y is not None, f"Candidate E fell back on passing shape {dtype}"
            assert y.shape == (m, n) and y.dtype == dtype, (y.shape, y.dtype)
            x32 = _rotate_activation(x.float(), 256)
            y_ref = (torch.matmul(x32, q.float().T) * scale.T).to(dtype)
            torch.cuda.synchronize()
            assert torch.isfinite(y).all().item(), f"Nonfinite Candidate E output: {dtype}"
            max_abs = float((y.float() - y_ref.float()).abs().max().item())
            rel_rms = float(
                (y.float() - y_ref.float()).square().mean().sqrt().item()
                / max(1e-6, y_ref.float().square().mean().sqrt().item())
            )
            assert rel_rms < 0.015, f"{dtype} RMS error too large: {rel_rms}"
            results.append({"dtype": str(dtype), "max_abs_error": max_abs,
                            "relative_rms_error": rel_rms})
        disallowed = SimpleNamespace(
            _data=torch.zeros((4096, 2048), device="cuda", dtype=torch.int8),
            _scale=torch.ones((4096, 1), device="cuda", dtype=torch.float32))
        assert e.maybe_linear(x, disallowed, None, 256) is None, (
            "Known failing shape was not excluded"
        )
    print("LTX25_CANDIDATE_ADAPTER_SM70_OK " + json.dumps({
        "M": m, "N": n, "K": k, "tests": results,
        "candidate_calls": e._STATS["used"], "known_bad_shape_excluded": True,
        "synthetic_only": True
    }), flush=True)


if __name__ == "__main__":
    main()
