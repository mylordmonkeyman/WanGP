#!/usr/bin/env python3
"""Read-only, exhaustive BF16 LoRA -> FP16 dynamic-range / cast audit.

Checks every element of every tensor, rather than sampling. Measures:
  * nonfinite source data (NaN, +Inf, -Inf)
  * source finite values outside FP16's maximum magnitude (65504)
  * nonzero source values below FP16's smallest positive subnormal (2**-24)
  * nonzero values in FP16 subnormal range (< 2**-14)
  * actual finite->Inf or nonzero->zero on PyTorch BF16 -> FP16 cast
  * finite nonzero values that round to another finite nonzero value
The final counts distinguish theoretical representability from actual rounding.
No tensor is written, and no GPU is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open

FP16_MAX = float(torch.finfo(torch.float16).max)  # 65504
FP16_MIN_NORMAL = float(torch.finfo(torch.float16).tiny)  # 2**-14
FP16_MIN_SUBNORMAL = 2.0**-24
SOURCE = Path("/cache/ltx25/results/hunyuan15-fp16-downloaded-checkpoints/loras/"
              "hunyuanvideo1.5_t2v_480p_lightx2v_4step_lora_rank_32_bf16.safetensors")
CONVERTED = Path("/cache/ltx25/results/hunyuan15-converted-lora-43951751237-attempt1/"
                 "converted_loras/hunyuan15_t2v_lightx2v4_rank32_wangp_qkv_split_bf16.safetensors")
ISSUES = (
    "nan", "pos_inf", "neg_inf", "above_fp16_max", "nonzero_below_fp16_min_subnormal",
    "actual_cast_to_inf", "actual_cast_to_zero", "finite_nonzero_rounded",
)


def count(mask):
    return int(mask.sum().item())


def inspect(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"LoRA missing from runner cache: {path}")
    counts = Counter()
    affected = []
    dtype_counts = Counter()
    max_finite = 0.0
    min_nonzero = None
    names = []

    # safe_open loads one tensor at a time. Nothing is moved to the GPU.
    with safe_open(str(path), framework="pt", device="cpu") as sf:
        keys = list(sf.keys())
        for index, name in enumerate(keys, 1):
            tensor = sf.get_tensor(name)
            dtype_counts[str(tensor.dtype)] += 1
            src = tensor.float()
            finite = torch.isfinite(src)
            nz_finite = finite & (src != 0)
            magnitude = src.abs()

            casting = tensor.to(torch.float16).float()
            cast_inf = finite & torch.isinf(casting)
            cast_zero = nz_finite & (casting == 0)
            rounded = nz_finite & torch.isfinite(casting) & (casting != 0) & (casting != src)
            item = {
                "nan": count(torch.isnan(src)),
                "pos_inf": count(torch.isposinf(src)),
                "neg_inf": count(torch.isneginf(src)),
                "above_fp16_max": count(finite & (magnitude > FP16_MAX)),
                "nonzero_below_fp16_min_subnormal":
                    count(nz_finite & (magnitude < FP16_MIN_SUBNORMAL)),
                "fp16_subnormal_magnitude":
                    count(nz_finite & (magnitude < FP16_MIN_NORMAL)),
                "actual_cast_to_inf": count(cast_inf),
                "actual_cast_to_zero": count(cast_zero),
                "finite_nonzero_rounded": count(rounded),
                "elements": tensor.numel(),
                "finite": count(finite),
                "nonzero_finite": count(nz_finite),
            }
            counts.update(item)
            finite_values = magnitude[finite]
            if finite_values.numel():
                max_finite = max(max_finite, float(finite_values.max().item()))
            if item["nonzero_finite"]:
                minimum = float(magnitude[nz_finite].min().item())
                min_nonzero = minimum if min_nonzero is None else min(min_nonzero, minimum)
            if any(item[k] for k in ISSUES) or item["fp16_subnormal_magnitude"]:
                affected.append({
                    "name": name, "shape": list(tensor.shape),
                    "source_dtype": str(tensor.dtype),
                    **{k: item[k] for k in ISSUES + ("fp16_subnormal_magnitude",)
                       if item[k]},
                })
            names.append(name)
            del tensor, src, casting, magnitude, finite_values

    print("HUNYUAN15_LORA_FP16_SCAN " + json.dumps({
        "path": str(path), "tensors": len(keys), "elements": counts["elements"],
        "above_fp16_max": counts["above_fp16_max"],
        "below_fp16_min_subnormal": counts["nonzero_below_fp16_min_subnormal"],
        "actual_cast_to_inf": counts["actual_cast_to_inf"],
        "actual_cast_to_zero": counts["actual_cast_to_zero"],
        "nan": counts["nan"], "source_inf": counts["pos_inf"] + counts["neg_inf"],
        "rounded_finite_nonzero": counts["finite_nonzero_rounded"],
        "tensor_count_with_any_findings": len(affected),
        "first_affected": affected[:8],
    }, sort_keys=True), flush=True)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "tensors": len(keys),
        "dtype_tensors": dict(dtype_counts),
        "max_abs_finite": max_finite,
        "min_abs_nonzero_finite": min_nonzero,
        "counts": dict(counts),
        "affected_tensors": affected,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--converted", type=Path, default=CONVERTED)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(min(8, torch.get_num_threads()))
    source = inspect(args.source)
    converted = inspect(args.converted)
    critical = ("nan", "pos_inf", "neg_inf", "above_fp16_max",
                "nonzero_below_fp16_min_subnormal",
                "actual_cast_to_inf", "actual_cast_to_zero")
    summary = {
        "source": source,
        "converted": converted,
        "thresholds": {
            "fp16_max_abs": FP16_MAX,
            "fp16_min_normal_positive": FP16_MIN_NORMAL,
            "fp16_min_subnormal_positive": FP16_MIN_SUBNORMAL,
        },
        "source_zero_critical": all(source["counts"].get(k, 0) == 0 for k in critical),
        "converted_zero_critical": all(converted["counts"].get(k, 0) == 0 for k in critical),
        "note": "BF16-to-FP16 cast examined per element. FP16 subnormals are representable, "
                "but have reduced precision; rounding is counted separately. "
                "This does not test arithmetic overflow during diffusion.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("HUNYUAN15_LORA_FP16_RANGE_AUDIT_COMPLETE " + json.dumps({
        "source_zero_critical": summary["source_zero_critical"],
        "converted_zero_critical": summary["converted_zero_critical"],
        "source": {k: source["counts"].get(k, 0) for k in critical + ("finite_nonzero_rounded", "fp16_subnormal_magnitude")},
        "converted": {k: converted["counts"].get(k, 0) for k in critical + ("finite_nonzero_rounded", "fp16_subnormal_magnitude")},
        "report_path": str(args.output),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
