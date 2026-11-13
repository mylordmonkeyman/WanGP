#!/usr/bin/env python3
"""SM70 Candidate E preflight: host-sync-free adapter and exceptional-row math.

Synthetic INT8 weights/activations only. Not a real-model numerical or quality
gate, and not a CUDA-graph or multi-stream test. Run before a 22B model load.
"""
from __future__ import annotations

import ctypes
import json
from pathlib import Path
from unittest.mock import patch
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from shared.kernels import ltx25_cublas_sm70 as e
from shared.qtypes.int8_convrot import _rotate_activation


def adapter_check():
    """Exercise the normal BF16/FP16 model entry without allowing host scalar sync."""
    torch.manual_seed(20261112)
    n, k, m = 2048, 2048, 128
    q = torch.randint(-64, 64, (n, k), dtype=torch.int8, device="cuda")
    scale = torch.full((n, 1), 0.0025, device="cuda", dtype=torch.float32)
    wt = SimpleNamespace(_data=q, _scale=scale)
    results = []
    with torch.inference_mode():
        for dtype in (torch.bfloat16, torch.float16):
            x = torch.randn((m, k), device="cuda", dtype=torch.float32).to(dtype)
            # First invocation also covers shared-library load and counter init.
            # Subsequent calls must not request .item() or explicitly synchronize.
            _ = e.maybe_linear(x, wt, None, 256)
            with patch.object(torch.Tensor, "item", side_effect=AssertionError("host scalar read in adapter")), \
                 patch.object(torch.cuda, "synchronize", side_effect=AssertionError("synchronization in adapter")):
                y = e.maybe_linear(x, wt, None, 256)
            assert y is not None and y.shape == (m, n) and y.dtype == dtype
            x32 = _rotate_activation(x.float(), 256)
            y_ref = (x32 @ q.float().T * scale.T).to(dtype)
            torch.cuda.synchronize()  # Trial boundary, never in the adapter.
            assert torch.isfinite(y).all().item()
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
        assert e.maybe_linear(x, disallowed, None, 256) is None
    return results


def device_range_check():
    """Exercise GPU-only row scaling AND same-call FP32 correction directly."""
    q = torch.randint(-3, 4, (2048, 2048), dtype=torch.int8, device="cuda")
    scale = torch.full((2048,), 0.0005, dtype=torch.float32, device="cuda")
    bias = torch.full((2048,), 0.125, dtype=torch.float32, device="cuda")
    m, k, n = 64, 2048, 2048
    x32 = torch.randn((m, k), dtype=torch.float32, device="cuda") * 0.1
    # Safe row requiring positive rescaling, safe row requiring downscaling,
    # mixed-range row requiring device-side FP32, and all-zero row.
    x32[0] *= 1.0e-10
    x32[1] *= 1.0e+9
    x32[2].fill_(0.0)
    x32[3].fill_(0.0)
    x32[3, 0] = 1.0e+12
    x32[3, 1] = 1.0e-20

    x16 = torch.empty((m, k), dtype=torch.float16, device="cuda")
    c32 = torch.empty((m, n), dtype=torch.float32, device="cuda")
    exp = torch.empty((m,), dtype=torch.int32, device="cuda")
    slow = torch.empty((m,), dtype=torch.int32, device="cuda")
    q16 = q.to(torch.float16)
    counter = torch.zeros((1,), dtype=torch.int32, device="cuda")
    _, gemm, prepare, finish = e._fn()
    ptr = ctypes.c_void_p
    stream = ptr(torch.cuda.current_stream().cuda_stream)
    with torch.inference_mode(), \
         patch.object(torch.Tensor, "item", side_effect=AssertionError("host scalar read in CUDA path")), \
         patch.object(torch.cuda, "synchronize", side_effect=AssertionError("sync in CUDA path")):
        rc = prepare(ptr(x32.data_ptr()), ptr(x16.data_ptr()), ptr(exp.data_ptr()),
                     ptr(slow.data_ptr()), ptr(counter.data_ptr()), m, k, stream)
        assert rc == 0, f"prepare status={rc}"
        rc = gemm(ptr(q16.data_ptr()), ptr(x16.data_ptr()), ptr(c32.data_ptr()),
                  n, m, k, stream)
        assert rc == 0, f"gemm status={rc}"
        rc = finish(ptr(c32.data_ptr()), ptr(x32.data_ptr()), ptr(q.data_ptr()),
                    ptr(scale.data_ptr()), ptr(bias.data_ptr()),
                    ptr(exp.data_ptr()), ptr(slow.data_ptr()), m, n, k, stream)
        assert rc == 0, f"epilogue status={rc}"

    torch.cuda.synchronize()
    assert int(slow[3].item()) == 1, "Expected mixed range row to use FP32"
    assert all(int(slow[i].item()) == 0 for i in (0, 1, 2)), "Expected scaled or zero rows to be safe"
    assert int(counter.item()) >= 1, "Missing GPU exceptional-row count"
    ref = (x32 @ q.float().T) * scale.reshape(1, n) + bias.reshape(1, n)
    normalized = (c32 - ref).abs() / torch.maximum(torch.ones_like(ref), ref.abs())
    max_relative = float(normalized.max().item())
    p99_relative = float(torch.quantile(normalized.flatten(), 0.99).item())
    assert torch.isfinite(c32).all().item()
    assert max_relative < 0.002, f"row prepare/epilogue error too large: {max_relative}"
    return {"fallback_rows": int(counter.item()), "max_normalized_error": max_relative,
            "p99_normalized_error": p99_relative, "scales": exp[:4].cpu().tolist()}


def main():
    assert e.MODE == "fp32rot", e.MODE
    assert torch.cuda.get_device_capability() == (7, 0)
    with torch.inference_mode():
        result = {"adapter": adapter_check(), "device_range": device_range_check()}
    print("LTX25_CANDIDATE_ADAPTER_SM70_OK " + json.dumps({
        **result, "candidate_calls": e._STATS["used"],
        "known_bad_shape_excluded": True, "synthetic_only": True,
        "host_scalar_checks_inside_candidate": 0
    }), flush=True)


if __name__ == "__main__":
    main()
