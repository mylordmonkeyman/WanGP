#!/usr/bin/env python3
"""Opt-in bounded scratch-pool correctness and no-host-sync V100 preflight.

Not a performance measurement. Exercises real CUDA stream ordering, pool
growth/reuse, rejection of oversized/nondefault stream cases, and release
on exceptions. No checkpoint or model generation is required.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["WAN2GP_LTX25_E_WORKSPACE"] = "1"

import torch
from shared.kernels import ltx25_cublas_sm70 as e
from shared.kernels import ltx25_workspace_sm70 as workspace
from shared.qtypes.int8_convrot import _rotate_activation


def main():
    assert e.MODE == "fp32rot" and e.USE_WORKSPACE and workspace.ENABLED
    assert torch.cuda.get_device_capability(0) == (7, 0)
    device = torch.device("cuda:0")
    torch.manual_seed(20261113)
    q = torch.randint(-32, 33, (2048, 2048), device=device, dtype=torch.int8)
    scale = torch.full((2048, 1), 0.002, dtype=torch.float32, device=device)
    wt = SimpleNamespace(_data=q, _scale=scale)
    samples = []
    with torch.inference_mode():
        for m, dtype in ((64, torch.bfloat16), (128, torch.bfloat16),
                         (256, torch.float16), (64, torch.bfloat16)):
            x = torch.randn((m, 2048), device=device).to(dtype)
            with patch.object(torch.Tensor, "item", side_effect=AssertionError("synchronous scalar read")), \
                 patch.object(torch.cuda, "synchronize", side_effect=AssertionError("global CUDA synchronization")):
                y = e.maybe_linear(x, wt, None, 256)
            assert y is not None
            ref = (_rotate_activation(x.float(), 256) @ q.float().T
                   * scale.T).to(dtype)
            # Boundary synchronization is okay, NEVER in the adapter.
            torch.cuda.synchronize()
            assert torch.isfinite(y).all().item()
            norm = (y.float() - ref.float()).square().mean().sqrt()
            denom = ref.float().square().mean().sqrt().clamp_min(1e-6)
            error = float((norm / denom).item())
            assert error < 0.015, f"pool changed result: m={m}, dtype={dtype}, rel rms={error}"
            samples.append({"M": m, "dtype": str(dtype), "relative_rms_error": error})

        # Two pending operations use one pool but independent BF16 outputs;
        # checking y1 after y2 catches accidental reuse of returned storage.
        a = torch.randn((64, 2048), device=device).bfloat16()
        b = torch.randn((64, 2048), device=device).bfloat16()
        y1 = e.maybe_linear(a, wt, None, 256)
        y2 = e.maybe_linear(b, wt, None, 256)
        assert y1.data_ptr() != y2.data_ptr(), "Returned outputs unexpectedly alias"
        ref1 = (_rotate_activation(a.float(), 256) @ q.float().T * scale.T).bfloat16()
        ref2 = (_rotate_activation(b.float(), 256) @ q.float().T * scale.T).bfloat16()
        torch.cuda.synchronize()
        for y, ref in ((y1, ref1), (y2, ref2)):
            rel = float(((y.float() - ref.float()).square().mean().sqrt()
                         / ref.float().square().mean().sqrt().clamp_min(1e-6)).item())
            assert rel < 0.015, f"output overwritten by a later reuse, err={rel}"

        before = workspace._STATS["miss_oversize"]
        giant = workspace.try_acquire(8192, 16384, 4096, device,
                                       torch.cuda.current_stream(device))
        assert giant is None, "Pool unexpectedly retained >96MiB scratch"
        assert workspace._STATS["miss_oversize"] > before

        nondefault = torch.cuda.Stream(device=device)
        other = workspace.try_acquire(64, 2048, 2048, device, nondefault)
        assert other is None, "Pool must not cross CUDA streams"

        # A failure inside _fn() occurs after leasing scratch: the adapter
        # must restore pool availability without synchronizing the GPU.
        with patch.object(e, "_fn", side_effect=RuntimeError("intentional test exception")):
            try:
                e.maybe_linear(a, wt, None, 256)
            except RuntimeError as exc:
                assert "intentional test exception" in str(exc)
            else:
                raise AssertionError("Expected _fn failure")
        recovery = e.maybe_linear(a, wt, None, 256)
        assert recovery is not None, "Pool did not recover"
        torch.cuda.synchronize()
        assert torch.isfinite(recovery).all().item()

    assert workspace._STATS["acquired"] >= 7, workspace._STATS
    assert workspace._STATS["max_persistent_bytes"] <= workspace.LIMIT_BYTES
    assert workspace._STATS["persistent_weight_copies"] == 0
    pool = workspace._POOLS[0]
    assert not hasattr(pool, "weight") and not hasattr(pool, "w16")
    report = {"status": "PASS", "kind": "SM70 bounded same-stream workspace preflight",
              "actual_model_activations": False, "production_eligible": False,
              "samples": samples, "workspace": workspace._STATS}
    print("LTX25_WORKSPACE_PREFLIGHT " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
