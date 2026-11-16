#!/usr/bin/env python3
"""Strict Comfy-Org Hunyuan Video 1.5 LightX2V LoRA -> WanGP layout converter.

Source: Comfy-Org 480p T2V rank-32 adapter with lora_{up,down}, .diff_b,
and .diff tensors. Destination uses real WanGP module names. Comfy's fused
image/text QKV low-rank update is split along its *output* dimension:
  B_q,B_k,B_v = B.chunk(3, dim=0); A is copied for each branch.
Bias differences are split by the same output axis. Refiner self_attn_qkv
remains fused and is renamed only. This preserves the full low-rank update.

The transformer source stays unchanged; this converter never edits base weights
or patches WanGP's loader. Fail-closed against the *actual* FP16 base
checkpoint's key+shape inventory. Conversion is CPU-only, offline.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

from hunyuan15_checkpoint_audit import (
    expand_fused_qkv_metadata, normalize, read_header, strip_wrapper,
)

SUFFIXES = (
    ".lora_down.weight", ".lora_up.weight",
    ".lora_A.weight", ".lora_B.weight",
    ".diff_b", ".diff", ".alpha",
)
SPLIT_SUFFIXES = (".lora_up.weight", ".lora_B.weight", ".diff_b", ".diff")
SOURCE_SHA256 = "8fef6f9b9f2f6aa5259166e78b10a022485391dababc6a8c68585fddd31d5d62"


def decode_lora_key(name):
    for suffix in SUFFIXES:
        if name.endswith(suffix):
            return name[:-len(suffix)], suffix
    raise ValueError(f"Unexpected input adapter tensor: {name}")


def target_modules(module):
    """Return canonical WanGP module names, in Q,K,V order where fused."""
    for stem in ("img_attn", "txt_attn"):
        if module.endswith("." + stem + ".qkv"):
            base = module[: -len(".qkv") - len(stem) - 1]
            return tuple(base + "." + stem + "_" + part for part in "qkv")
    # Comfy exports timestep embedding MLPs as in_layer/out_layer; WanGP
    # TimestepEmbedder uses mlp.0/mlp.2. Vector MLPEmbedder does not.
    for stem in ("time_in", "time_r_in", "guidance_in"):
        if module == stem + ".in_layer":
            return (stem + ".mlp.0",)
        if module == stem + ".out_layer":
            return (stem + ".mlp.2",)
    result = module
    for stem in ("img_attn", "txt_attn", "self_attn"):
        result = result.replace("." + stem + ".qkv", "." + stem + "_qkv")
        result = result.replace("." + stem + ".proj", "." + stem + "_proj")
    for stem in ("img_mlp", "txt_mlp", "mlp"):
        result = result.replace("." + stem + ".0", "." + stem + ".fc1")
        result = result.replace("." + stem + ".2", "." + stem + ".fc2")
    for stem in ("img_mod", "txt_mod"):
        result = result.replace("." + stem + ".lin", "." + stem + ".linear")
    return (result,)


def plan(source, reference):
    """Build a complete checked per-tensor transformation plan, no torch needed."""
    output_shapes = {}
    mapping = {}
    source_modules = defaultdict(set)
    dest_modules = defaultdict(set)
    splits = 0

    for name, item in sorted(source.items()):
        module, suffix = decode_lora_key(name)
        targets = target_modules(module)
        is_split = len(targets) == 3
        if is_split:
            splits += 1
        shape = tuple(item["shape"])
        if is_split and suffix in SPLIT_SUFFIXES:
            if not shape or shape[0] % 3:
                raise ValueError(f"QKV tensor cannot split 3 ways: {name} {shape}")
            output_shape = (shape[0] // 3, *shape[1:])
        else:
            output_shape = shape
        output_keys = []
        for target in targets:
            key = target + suffix
            if key in output_shapes:
                raise ValueError(f"Collision after LoRA mapping: {name} -> {key}")
            weight = reference.get(target + ".weight")
            bias = reference.get(target + ".bias")
            if weight is None:
                raise ValueError(f"Missing target module '{target}' for {name}")
            w_shape = tuple(weight["shape"])
            if suffix in (".lora_down.weight", ".lora_A.weight"):
                if len(output_shape) != 2 or len(w_shape) != 2 or output_shape[1] != w_shape[1]:
                    raise ValueError(f"LoRA down shape mismatch {name}: {output_shape}, model {w_shape}")
            elif suffix in (".lora_up.weight", ".lora_B.weight"):
                if len(output_shape) != 2 or len(w_shape) != 2 or output_shape[0] != w_shape[0]:
                    raise ValueError(f"LoRA up shape mismatch {name}: {output_shape}, model {w_shape}")
            elif suffix == ".diff_b":
                if bias is None or output_shape != tuple(bias["shape"]):
                    raise ValueError(f"Bias delta mismatch {name}: {output_shape}, model {bias}")
            elif suffix == ".diff":
                if output_shape != w_shape:
                    raise ValueError(f"Full weight delta mismatch {name}: {output_shape}, model {w_shape}")
            elif suffix == ".alpha":
                if shape not in ((), (1,)):
                    raise ValueError(f"Invalid LoRA alpha shape {name}: {shape}")
            else:
                raise ValueError(f"Unrecognized LoRA suffix {suffix}")
            output_shapes[key] = output_shape
            output_keys.append(key)
            dest_modules[target].add(suffix)
        source_modules[module].add(suffix)
        mapping[name] = (tuple(output_keys), suffix)

    # Reject incomplete low-rank pairs, and reject a delta-only model with
    # misleading rank-32 names. Some modules legitimately contain only .diff.
    down = {".lora_down.weight", ".lora_A.weight"}
    up = {".lora_up.weight", ".lora_B.weight"}
    paired = 0
    for module, suffixes in dest_modules.items():
        if bool(suffixes & down) != bool(suffixes & up):
            raise ValueError(f"Incomplete LoRA pair on {module}: {sorted(suffixes)}")
        if suffixes & down:
            paired += 1
    if not paired or not splits:
        raise ValueError(f"Expected ranked and fused-QKV tensors, got pairs={paired} splits={splits}")
    return mapping, output_shapes, {
        "input_tensors": len(source),
        "output_tensors": len(output_shapes),
        "input_modules": len(source_modules),
        "output_modules": len(dest_modules),
        "qkv_input_tensors_split": splits,
        "paired_modules": paired,
        "input_dtypes": dict(sorted(Counter(x["dtype"] for x in source.values()).items())),
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as fd:
        for block in iter(lambda: fd.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert(source_path, base_path, destination, report_path=None, *, require_source_sha=True):
    source_path, base_path, destination = map(Path, (source_path, base_path, destination))
    source_header = normalize(read_header(source_path))
    ref, qkv = expand_fused_qkv_metadata(normalize(read_header(base_path)))
    mapping, out_shapes, stats = plan(source_header, ref)
    stats["reference_modules"] = len(ref)
    stats["reference_fused_qkv"] = qkv
    stats["source"] = str(source_path)
    stats["target"] = str(destination)
    digest = sha256_file(source_path)
    stats["source_sha256"] = digest
    if require_source_sha and digest != SOURCE_SHA256:
        raise ValueError(f"Unexpected Comfy-Org source SHA256: {digest}")
    import torch
    from safetensors.torch import load_file, save_file
    source_raw = load_file(str(source_path), device="cpu")
    source = {strip_wrapper(key): tensor for key, tensor in source_raw.items()}
    if len(source) != len(source_raw):
        raise ValueError("Source wrapper stripping collided with another tensor name")
    if set(source) != set(mapping):
        raise ValueError("Source bytes and inspected source header disagree")
    output = {}
    for name, (dest_keys, suffix) in mapping.items():
        data = source[name]
        if len(dest_keys) == 3 and suffix in SPLIT_SUFFIXES:
            tensors = tuple(piece.clone().contiguous() for piece in data.chunk(3, dim=0))
        elif len(dest_keys) == 3:
            tensors = tuple(data.clone().contiguous() for _ in dest_keys)
        else:
            tensors = (data.contiguous(),)
        for key, tensor in zip(dest_keys, tensors):
            if tuple(tensor.shape) != out_shapes[key]:
                raise ValueError(f"Tensor shape mismatch after conversion: {key}")
            output[key] = tensor
    # Safetensors rejects shared underlying tensors; fused branches cloned above.
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(output, str(destination), metadata={
        "format": "WanGP-Hunyuan15-LoRA",
        "source_sha256": digest,
        "conversion": "Comfy modules renamed; image/text QKV B and bias split, A replicated",
    })
    result_header = read_header(destination)
    if set(result_header) != set(out_shapes):
        raise ValueError("Saved LoRA does not have the expected tensor inventory")
    for name, shape in out_shapes.items():
        if tuple(result_header[name]["shape"]) != shape:
            raise ValueError(f"Saved output shape mismatch: {name}")
    stats["destination_bytes"] = destination.stat().st_size
    stats["destination_sha256"] = sha256_file(destination)
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(stats, indent=2) + "\n")
    print("HUNYUAN15_LORA_CONVERSION_PASS " + json.dumps(stats, sort_keys=True), flush=True)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("base", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    convert(args.source, args.base, args.destination, args.report)


if __name__ == "__main__":
    main()
