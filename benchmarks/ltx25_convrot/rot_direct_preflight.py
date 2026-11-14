#!/usr/bin/env python3
"""SM70 direct radix4 and fused-output epilogue gates against Candidate E.

Requires GPU V100 and the existing SM70 shared extension. Exact FP32 rotated
values and exact BF16/FP16 full Candidate E output must match in all four combinations. Synthetic only:
a separate full-model visual/audio gate is required before default promotion.
"""
from __future__ import annotations

import ctypes
import json
import statistics
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch

from shared.kernels import ltx25_cublas_sm70 as e


def check_rotations():
    results = []
    ptr = ctypes.c_void_p
    staged = e._rot_fn()
    for dtype in (torch.bfloat16, torch.float16):
        direct = e._rot_direct_fn(dtype)
        for m, k in ((64, 2048), (144, 4096), (1008, 4096)):
            generator = torch.Generator(device="cuda").manual_seed(20261113 + m + k)
            x = (torch.randn((m, k), generator=generator, device="cuda") * 3.0).to(dtype)
            x[0].zero_()
            x[1].fill_(0.03125)
            x[2, :256].fill_(0.0)
            x[2, 0] = 2.0 ** -10
            expected = torch.empty((m, k), device="cuda", dtype=torch.float32)
            actual = torch.empty_like(expected)
            stream = ptr(torch.cuda.current_stream().cuda_stream)
            staged_source = x.float().contiguous()
            with patch.object(torch.Tensor, "item", side_effect=AssertionError("host item in radix4")), \
                 patch.object(torch.cuda, "synchronize", side_effect=AssertionError("host sync in radix4")):
                rc_old = staged(ptr(staged_source.data_ptr()),
                                ptr(expected.data_ptr()), x.numel(), stream)
                rc_new = direct(ptr(x.data_ptr()), ptr(actual.data_ptr()),
                                x.numel(), stream)
                assert rc_old == rc_new == 0, (dtype, m, k, rc_old, rc_new)
            torch.cuda.synchronize()
            assert torch.equal(expected, actual), {
                "dtype": str(dtype), "M": m, "K": k,
                "max_abs_error": float((expected - actual).abs().max().item())}
            results.append({"dtype": str(dtype), "M": m, "K": k, "exact": True})
    return results


def check_adapters():
    torch.manual_seed(20261113)
    results = []
    for dtype in (torch.bfloat16, torch.float16):
        for n, k, m in ((2048, 2048, 1008), (4096, 4096, 144),
                        (16384, 4096, 64)):
            q = torch.randint(-32, 33, (n, k), dtype=torch.int8, device="cuda")
            scale = torch.rand((n, 1), device="cuda") * 0.002 + 0.001
            bias = torch.randn((n,), device="cuda", dtype=torch.float32) * 0.01
            weight = SimpleNamespace(_data=q, _scale=scale)
            x = (torch.randn((m, k), device="cuda") * 0.4).to(dtype)
            x[0].zero_()
            x[1].fill_(0.001953125)
            variants = {}
            for rotation, output in (("staged", "torch"),
                                     ("direct", "torch"),
                                     ("staged", "fused"),
                                     ("direct", "fused")):
                e.ROT_INPUT = rotation
                e.OUTPUT_MODE = output
                with patch.object(torch.Tensor, "item", side_effect=AssertionError("host item in Candidate E")), \
                     patch.object(torch.cuda, "synchronize", side_effect=AssertionError("host sync in Candidate E")):
                    result = e.maybe_linear(x, weight, bias, 256)
                assert result is not None
                variants[(rotation, output)] = result
            torch.cuda.synchronize()
            a = variants[("staged", "torch")]
            for key, b in variants.items():
                assert torch.equal(a, b), {
                    "variant": key, "dtype": str(dtype), "N": n, "K": k, "M": m,
                    "max_abs_error": float((a.float() - b.float()).abs().max().item())}
            results.append({"dtype": str(dtype), "N": n, "K": k, "M": m,
                            "exact": True, "variants": 4})
            del variants, a, b, q, scale, bias, weight, x
    return results


def timed_rotation(dtype=torch.bfloat16, m=1008, k=4096, repeats=20):
    x = torch.randn((m, k), dtype=torch.float32, device="cuda").to(dtype)
    x32 = x.float().contiguous()
    out = torch.empty((m, k), dtype=torch.float32, device="cuda")
    ptr = ctypes.c_void_p
    stream = ptr(torch.cuda.current_stream().cuda_stream)
    f_staged = e._rot_fn()
    f_direct = e._rot_direct_fn(dtype)
    samples = {}
    for mode in ("staged", "direct", "direct", "staged"):
        for _ in range(3):
            if mode == "staged":
                tmp = x.float()
                assert f_staged(ptr(tmp.data_ptr()), ptr(out.data_ptr()), x.numel(), stream) == 0
            else:
                assert f_direct(ptr(x.data_ptr()), ptr(out.data_ptr()), x.numel(), stream) == 0
        torch.cuda.synchronize()
        ms = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            if mode == "staged":
                tmp = x.float()
                assert f_staged(ptr(tmp.data_ptr()), ptr(out.data_ptr()), x.numel(), stream) == 0
            else:
                assert f_direct(ptr(x.data_ptr()), ptr(out.data_ptr()), x.numel(), stream) == 0
            stop.record()
            stop.synchronize()
            ms.append(start.elapsed_time(stop))
        samples.setdefault(mode, []).extend(ms)
    return {"dtype": str(dtype), "M": m, "K": k, "staged_ms": statistics.median(samples["staged"]),
            "direct_ms": statistics.median(samples["direct"]),
            "speedup": statistics.median(samples["staged"]) / statistics.median(samples["direct"])}


def main():
    assert torch.cuda.get_device_capability() == (7, 0)
    assert e.MODE == "fp32rot", e.MODE
    assert e.ROT_MODE == "radix4", e.ROT_MODE
    assert e.CAST_MODE == "vector8", e.CAST_MODE
    with torch.inference_mode():
        exact_rotation = check_rotations()
        exact_adapter = check_adapters()
        timing = [timed_rotation(dtype) for dtype in (torch.bfloat16, torch.float16)]
    print("LTX25_ROT_DIRECT_PREFLIGHT_OK " + json.dumps({
        "rotation": exact_rotation, "adapter": exact_adapter, "microbench": timing,
        "bit_exact": True, "persistent_weight_copies": 0,
        "tested_output_mode": "torch_and_fused",
        "note": "Synthetic preflight; no full-video or speech quality assertion"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
