#!/usr/bin/env python3
"""Smoke the opt-in Quanto-only interception backend on a Tesla V100.

Uses real CUDA Quanto custom op, no checkpoint or persistent weight caches.
Also tests that the Quanto operator namespace is restored after each forward
and that a BF16 sham operation remains bit-identical to native Quanto.
"""
from __future__ import annotations

import json
import torch

# Import and register quanto::qbytes_mm before patching torch.ops.
import optimum.quanto.library.qbytes_mm  # noqa: F401

from shared.kernels import ltx25_selective_fp16_gemm as exp


def main():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (7, 0):
        raise SystemExit("Tesla V100 sm70 required")
    torch.manual_seed(20261114)
    exp.ENABLED = True
    exp.BACKEND = "direct"
    exp.GROUPS = exp.GROUP_NAMES
    exp.OPS = frozenset(("qbytes_mm",))
    exp.GUARD = "always"
    exp.MIN_M = 64
    exp.MIN_DIM = 64
    exp.SHAPE_CENSUS = True
    output = []
    native_op = torch.ops.quanto.qbytes_mm

    for phase in ("stage1", "stage2"):
        exp.set_phase(phase)
        for shape_name, (m, k, n) in (
            ("wide", (128, 512, 1024)),
            ("balanced", (64, 256, 256)),
            ("tall", (128, 1024, 256)),
            ("small", (64, 128, 64)),
        ):
            a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / 64
            b = torch.randint(-16, 16, (n, k), device="cuda", dtype=torch.int8)
            scales = torch.full((n, 1), 1 / 1024, device="cuda", dtype=torch.bfloat16)
            reference = native_op(a, b, scales)
            exp.SHAM = True
            sham = exp.call_transformer(
                lambda x,y,z: torch.ops.quanto.qbytes_mm(x,y,z), a,b,scales)
            assert torch.equal(reference, sham), (phase, shape_name, "sham not exact")
            assert torch.ops.quanto.qbytes_mm is native_op, "Quanto op not restored"
            exp.SHAM = False
            before = exp._COUNTS["converted_qbytes_mm"]
            result = exp.call_transformer(
                lambda x,y,z: torch.ops.quanto.qbytes_mm(x,y,z), a,b,scales)
            assert torch.ops.quanto.qbytes_mm is native_op, "Quanto op not restored"
            assert exp._COUNTS["converted_qbytes_mm"] == before + 1, (phase,shape_name)
            err = result.float() - reference.float()
            deviation = float((err.square().mean().sqrt() /
                              reference.float().square().mean().sqrt().clamp_min(1e-12)).item())
            assert torch.isfinite(result).all().item()
            assert deviation < 0.04, (phase,shape_name,deviation)
            output.append({"phase":phase, "group":shape_name, "m":m,"n":n,"k":k,
                           "normalized_rms_error":deviation})
    # Conversion guard must fall back on huge input, without changing the operator.
    exp.set_phase("stage2")
    huge = torch.full((128, 128), 90000, device="cuda", dtype=torch.bfloat16)
    b = torch.ones((128,128), device="cuda", dtype=torch.int8)
    scale = torch.ones((128,1), device="cuda", dtype=torch.bfloat16)
    before=exp._COUNTS["guard_input_reference"]
    _ = exp.call_transformer(lambda x,y,z: torch.ops.quanto.qbytes_mm(x,y,z),
                              huge,b,scale)
    assert exp._COUNTS["guard_input_reference"] > before
    assert torch.ops.quanto.qbytes_mm is native_op
    print("LTX25_QUANTO_DIRECT_PREFLIGHT_PASS " + json.dumps({
        "tested":output,
        "converted_qbytes_mm":exp._COUNTS["converted_qbytes_mm"],
        "sham_native":exp._COUNTS["sham_native"],
        "guard_input_reference":exp._COUNTS["guard_input_reference"],
        "observed_qbytes_mm":exp._COUNTS["observed_qbytes_mm"],
        "operator_restored":True,
    }, sort_keys=True), flush=True)


if __name__=="__main__":
    main()
