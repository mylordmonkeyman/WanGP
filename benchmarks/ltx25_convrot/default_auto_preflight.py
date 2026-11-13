#!/usr/bin/env python3
"""SM70 default-mode smoke test with every legacy opt-in variable absent.

Must run with LTX25_CUBLAS_E_LIB unset: proves bundled CUDA compilation/cache
and automatic selection work in a fresh process. Also proves explicit off,
ineligible shape fallback and missing-native auto fallback never break R0.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import json

for name in ("WAN2GP_LTX25_CUBLAS_E_MODE", "WAN2GP_LTX25_E_CAST",
             "WAN2GP_LTX25_E_WORKSPACE", "LTX25_CUBLAS_E_LIB",
             "WAN2GP_LTX25_HADAMARD", "WAN2GP_LTX25_GEMM_TUNE",
             "WAN2GP_LTX25_E_AUTO_BUILD"):
    assert name not in os.environ, f"{name} must be unset for this test"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from shared.kernels import ltx25_cublas_sm70 as e
from shared.kernels import ltx25_workspace_sm70 as ws
from shared.qtypes.int8_convrot import _rotate_activation

def main():
    assert e.MODE == "fp32rot" and e._MODE_IMPLICIT and e.CAST_MODE == "vector4" and e.USE_WORKSPACE
    assert e.ROT_MODE == "radix4" and e._STATS["rotation_mode"] == "radix4"
    assert ws.ENABLED and ws.LIMIT_BYTES == 96 * (1 << 20)
    assert e.LIB == "" and e._STATS["auto_selected"] is True
    assert torch.cuda.get_device_capability(0) == (7, 0)
    assert not e._NATIVE_UNAVAILABLE
    torch.manual_seed(20261113)
    dev = torch.device("cuda:0")
    n, k, m = 2048, 2048, 128
    q = torch.randint(-80, 81, (n, k), device=dev, dtype=torch.int8)
    original = q.clone()
    scale = torch.full((n, 1), 0.003, device=dev, dtype=torch.float32)
    wt = SimpleNamespace(_data=q, _scale=scale)
    samples = []
    with torch.inference_mode():
        for dtype in (torch.bfloat16, torch.float16):
            x = torch.randn((m, k), device=dev, dtype=torch.float32).to(dtype)
            # Initial call resolves CUDA cache and initializes cuBLAS.
            y = e.maybe_linear(x, wt, None, 256)
            assert y is not None and y.dtype == dtype
            with patch.object(torch.Tensor, "item", side_effect=AssertionError("Per-linear CPU scalar read")), \
                 patch.object(torch.cuda, "synchronize", side_effect=AssertionError("Per-linear global sync")):
                again = e.maybe_linear(x, wt, None, 256)
            assert again is not None
            ref = (_rotate_activation(x.float(), 256) @ q.float().T
                   * scale.T).to(dtype)
            torch.cuda.synchronize()
            assert torch.equal(y, again), "Auto selection not reproducible"
            rel = float(((again.float() - ref.float()).square().mean().sqrt()
                         / ref.float().square().mean().sqrt().clamp_min(1e-6)).item())
            assert rel < 0.015, f"{dtype} auto numerical drift: {rel}"
            samples.append({"dtype": str(dtype), "relative_rms_error": rel})
        assert torch.equal(q, original), "Original INT8 weights mutated"
        assert e._FN is not None and e._CAST_FN is not None and e._ROT_FN is not None
        # The previous torch FP32 rotation is still available on demand.
        prev_rotation = e.ROT_MODE
        e.ROT_MODE = "torch"
        try:
            torch_out = e.maybe_linear(x, wt, None, 256)
            assert torch_out is not None and torch_out.dtype == x.dtype
            torch.cuda.synchronize()
            assert torch.isfinite(torch_out).all().item()
        finally:
            e.ROT_MODE = prev_rotation
        assert ws._STATS["acquired"] >= 2
        assert ws._STATS["persistent_weight_copies"] == 0
        assert ws._STATS["max_persistent_bytes"] <= ws.LIMIT_BYTES
        assert e._STATS["used"] >= 4
        assert e._STATS["auto_native_fallback"] == 0

        # Explicit off restores the original WanGP dispatch.
        prev_mode = e.MODE
        e.MODE = "off"
        try:
            assert e.maybe_linear(x, wt, None, 256) is None
        finally:
            e.MODE = prev_mode

        # Previously disallowed shape must never use the custom CUDA kernel.
        bad = SimpleNamespace(
            _data=torch.zeros((4096, 2048), dtype=torch.int8, device=dev),
            _scale=torch.ones((4096, 1), dtype=torch.float32, device=dev))
        assert e.maybe_linear(x, bad, None, 256) is None

        # Simulate unavailable nvcc/shared object in a new auto process.
        # Test fallback only before dispatch (never after queued CUDA work).
        prior_fn, prior_cast, prior_unavailable = e._FN, e._CAST_FN, e._NATIVE_UNAVAILABLE
        prior_count = e._STATS["auto_native_fallback"]
        e._FN = None
        e._CAST_FN = None
        e._NATIVE_UNAVAILABLE = False
        try:
            with patch.object(e, "_native_library", side_effect=FileNotFoundError("intentional missing sm70 extension")):
                assert e.maybe_linear(x, wt, None, 256) is None
            assert e._NATIVE_UNAVAILABLE
            assert e._STATS["auto_native_fallback"] == prior_count + 1
        finally:
            e._FN, e._CAST_FN, e._NATIVE_UNAVAILABLE = prior_fn, prior_cast, prior_unavailable

        # A native extension missing the new radix4 entry point must also
        # fall back before launching any partial accelerated operation.
        prior_count = e._STATS["auto_native_fallback"]
        with patch.object(e, "_rot_fn", side_effect=AttributeError("intentional missing radix4 symbol")):
            assert e.maybe_linear(x, wt, None, 256) is None
        assert e._NATIVE_UNAVAILABLE
        assert e._STATS["auto_native_fallback"] == prior_count + 1
        e._NATIVE_UNAVAILABLE = prior_unavailable

    dll = e._FN[0]
    output = {
        "status": "PASS", "mode": e.MODE, "cast": e.CAST_MODE,
        "workspace_enabled": ws.ENABLED, "real_model_checkpoint": False,
        "extension": str(dll._name), "sample_count": len(samples),
        "samples": samples, "workspace_leases": ws._STATS["acquired"],
        "original_int8_unchanged": True,
        "missing_native_falls_back_before_cuda": True,
        "missing_radix4_symbol_falls_back_before_cuda": True,
        "torch_rotation_rollback_works": True,
        "default_rotation": e.ROT_MODE,
        "explicit_off_works": True,
        "no_host_sync_in_hot_path": True,
    }
    print("LTX25_DEFAULT_AUTO_PREFLIGHT " + json.dumps(output), flush=True)

if __name__ == "__main__":
    main()
