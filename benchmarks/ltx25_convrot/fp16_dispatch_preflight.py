#!/usr/bin/env python3
"""Small mandatory V100 CUDA preflight for selective BF16-to-FP16 GEMM.

Only transient synthetic operands; no model checkpoint, no persistent weights.
Tests all six selectors, both aten.mm and aten.addmm and extreme-value fallback.
Runs once before the full video sweep so a nonfunctional dispatch fails early.
"""
from __future__ import annotations

import json

import torch

from shared.kernels import ltx25_selective_fp16_gemm as exp


def main():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (7, 0):
        raise SystemExit("A Tesla V100 (sm_70) is required for selective GEMM experiments")
    exp.ENABLED = True
    exp.GROUPS = exp.GROUP_NAMES
    exp.OPS = frozenset(("mm", "addmm"))
    exp.GUARD = "always"
    exp.MIN_M, exp.MIN_DIM = 128, 512
    torch.manual_seed(20261114)
    rows = []
    specs = (
        ("wide", 256, 512, 1024),
        ("tall", 256, 1024, 512),
        ("balanced", 256, 512, 512),
    )
    for phase in ("stage1", "stage2"):
        exp.set_phase(phase)
        for group, m, k, n in specs:
            a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.025
            b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16) * 0.025
            bias = torch.randn((n,), device="cuda", dtype=torch.bfloat16) * 0.01
            reference = torch.mm(a, b)
            test = exp.call_transformer(lambda x, w: torch.mm(x, w), a, b)
            torch.cuda.synchronize()
            assert test.dtype == torch.bfloat16 and test.shape == reference.shape
            assert torch.isfinite(test).all().item()
            deviation = (reference.float() - test.float()).abs()
            max_delta = float(deviation.max().item())
            assert max_delta < 0.012, (phase, group, max_delta)
            before = exp._COUNTS["converted_addmm"]
            target = exp.call_transformer(
                lambda x, w, v: torch.addmm(v, x, w), a, b, bias)
            torch.cuda.synchronize()
            assert exp._COUNTS["converted_addmm"] == before + 1, (phase, group)
            assert torch.isfinite(target).all().item()
            assert target.dtype == torch.bfloat16
            rows.append({"phase": phase, "shape": group, "max_abs_delta": round(max_delta, 6)})
    # The overflow gate must restore native BF16 for values unrepresentable in FP16.
    exp.set_phase("stage2")
    huge = torch.full((129, 512), 90000, device="cuda", dtype=torch.bfloat16)
    finite = torch.ones((512, 1024), device="cuda", dtype=torch.bfloat16)
    before = exp._COUNTS["guard_input_reference"]
    _ = exp.call_transformer(lambda x,w: torch.mm(x,w), huge, finite)
    assert exp._COUNTS["guard_input_reference"] > before
    print("LTX25_FP16_DISPATCH_PREFLIGHT_OK " + json.dumps({
        "tested": rows,
        "converted_mm": exp._COUNTS["converted_mm"],
        "converted_addmm": exp._COUNTS["converted_addmm"],
        "guard_input_reference": exp._COUNTS["guard_input_reference"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
