#!/usr/bin/env python3
"""Inspect safetensors header-only, without loading multi-GB Hunyuan 1.5 weights.

Compare Comfy-Org FP16 or Tencent FP32 against a known WanGP BF16 checkpoint:
  python scripts/hunyuan15_checkpoint_audit.py \
    /path/to/hunyuanvideo1.5_480p_i2v_step_distilled_fp16.safetensors \
    --reference /path/to/hunyuan_video_1.5_i2v_480_step_distilled_bf16.safetensors

The comparison ignores *storage dtype* differences but checks tensor names/shapes.
A key match is necessary, not sufficient, for successful WanGP inference.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import struct
import sys


MAX_HEADER_BYTES = 512 * 1024 * 1024
KNOWN_WRAPPERS = (
    "model.diffusion_model.",
    "diffusion_model.",
    "transformer.",
)


def read_header(path: Path):
    """Read only the safetensors JSON header; no weight bytes are materialized."""
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path}: missing safetensors length prefix")
        header_size = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_size <= MAX_HEADER_BYTES:
            raise ValueError(f"{path}: suspicious header length {header_size}")
        header_bytes = handle.read(header_size)
        if len(header_bytes) != header_size:
            raise ValueError(f"{path}: truncated safetensors header")
        header = json.loads(header_bytes)
    if not isinstance(header, dict):
        raise ValueError(f"{path}: invalid safetensors header")
    weights = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict) or not {"dtype", "shape", "data_offsets"} <= set(entry):
            raise ValueError(f"{path}: malformed tensor metadata for {name}")
        weights[name] = {"dtype": entry["dtype"], "shape": entry["shape"]}
    return weights


def strip_wrapper(name):
    for prefix in KNOWN_WRAPPERS:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def normalize(catalog):
    normalized = {}
    for raw_name, info in catalog.items():
        name = strip_wrapper(raw_name)
        if name in normalized:
            raise ValueError(f"Duplicate name after wrapper normalization: {name}")
        normalized[name] = info
    return normalized


def summarize(path, catalog):
    return {
        "path": str(path),
        "tensors": len(catalog),
        "dtypes": dict(sorted(Counter(x["dtype"] for x in catalog.values()).items())),
        "prefix_examples": sorted(catalog)[:8],
    }


def expand_fused_qkv_metadata(catalog):
    """Expand Comfy QKV header entries into WanGP's separate Q/K/V metadata.

    This is a *shape-only* check. Live tensors must be split on output axis 0
    with Q,K,V ordering, independently validated before numerical inference.
    """
    expanded = {}
    conversions = 0
    for key, value in catalog.items():
        if "_attn_qkv." not in key:
            if key in expanded:
                raise ValueError(f"Duplicate key: {key}")
            expanded[key] = value
            continue
        shape = value["shape"]
        if not shape or shape[0] % 3:
            raise ValueError(f"{key}: QKV first dimension is not a multiple of 3")
        for which in ("q", "k", "v"):
            new_name = key.replace("_attn_qkv.", f"_attn_{which}.")
            if new_name in expanded or new_name in catalog:
                raise ValueError(f"{key}: QKV expansion conflicts with {new_name}")
            expanded[new_name] = {
                "dtype": value["dtype"],
                "shape": [shape[0] // 3, *shape[1:]],
            }
        conversions += 1
    return expanded, conversions


def compare(candidate, reference):
    cand = normalize(candidate)
    ref = normalize(reference)
    candidate_names = set(cand)
    reference_names = set(ref)
    missing = sorted(reference_names - candidate_names)
    unexpected = sorted(candidate_names - reference_names)
    differing_shapes = [
        {"key": key, "candidate": cand[key]["shape"], "reference": ref[key]["shape"]}
        for key in sorted(candidate_names & reference_names)
        if cand[key]["shape"] != ref[key]["shape"]
    ]
    return {
        "matching_keys": len(candidate_names & reference_names),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "shape_mismatches": len(differing_shapes),
        "missing_examples": missing[:30],
        "unexpected_examples": unexpected[:30],
        "shape_examples": differing_shapes[:30],
        "keys_and_shapes_match": not (missing or unexpected or differing_shapes),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, help="Comfy FP16 or Tencent FP32 .safetensors")
    parser.add_argument("--reference", type=Path, help="WanGP BF16 .safetensors of same model variant")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--require-key-match", action="store_true", help="Fail when reference keys/shapes differ")
    args = parser.parse_args()
    if args.require_key_match and not args.reference:
        parser.error("--require-key-match needs --reference")
    candidate = read_header(args.candidate)
    report = {"candidate": summarize(args.candidate, candidate)}
    if args.reference:
        reference = read_header(args.reference)
        report["reference"] = summarize(args.reference, reference)
        report["comparison"] = compare(candidate, reference)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for name in ("candidate", "reference"):
            if name in report:
                item = report[name]
                print(f"{name}: {item['path']} | {item['tensors']} tensors | dtypes={item['dtypes']}")
        if "comparison" in report:
            result = report["comparison"]
            print(
                f"Key+shape compatibility: {result['keys_and_shapes_match']} | "
                f"missing={result['missing_keys']} unexpected={result['unexpected_keys']} "
                f"shape mismatches={result['shape_mismatches']}"
            )
            for category in ("missing_examples", "unexpected_examples", "shape_examples"):
                if result[category]:
                    print(f"{category}: {result[category]}")
    if args.require_key_match and not report["comparison"]["keys_and_shapes_match"]:
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Audit failed: {exc}", file=sys.stderr)
        sys.exit(2)
