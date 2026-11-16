#!/usr/bin/env python3
"""Inspect three public Hunyuan 1.5 safetensors headers via HTTP Range.

Only the tiny JSON header is read. No 16-33 GB weight file is downloaded.
This is a one-shot CPU-only research probe, not a video generation test.
"""
from collections import Counter
import json
import struct
from urllib.request import Request, urlopen

from hunyuan15_checkpoint_audit import MAX_HEADER_BYTES, compare, expand_fused_qkv_metadata


URLS = {
    "comfy_fp16": (
        "https://huggingface.co/Comfy-Org/Hunyuan-Video-1.5/resolve/main/"
        "split_files/diffusion_models/hunyuanvideo1.5_480p_i2v_step_distilled_fp16.safetensors"
    ),
    "wangp_bf16": (
        "https://huggingface.co/DeepBeepMeep/HunyuanVideo1.5/resolve/main/"
        "hunyuan_video_1.5_i2v_480_step_distilled_bf16.safetensors"
    ),
    "tencent_original": (
        "https://huggingface.co/tencent/HunyuanVideo-1.5/resolve/main/"
        "transformer/480p_i2v_step_distilled/diffusion_pytorch_model.safetensors"
    ),
}


def fetch_range(url, start, end):
    if end < start:
        raise ValueError("Invalid byte range")
    length = end - start + 1
    request = Request(
        url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity",
                      "User-Agent": "WanGP-Hunyuan15-header-audit/1.0"}
    )
    with urlopen(request, timeout=90) as response:
        status = response.status
        if status not in (200, 206):
            raise OSError(f"Unexpected HTTP status {status}")
        if status == 206:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {start}-"):
                raise OSError(f"Unexpected Content-Range: {content_range}")
        else:
            # Range ignored: stream/discard only up to the requested position.
            # We intentionally never consume the remainder of the weight file.
            remaining = start
            while remaining:
                chunk = response.read(min(remaining, 65536))
                if not chunk:
                    raise OSError("Server closed response while skipping to requested offset")
                remaining -= len(chunk)
        result = response.read(length)
        if len(result) != length:
            raise OSError(f"Expected {length} bytes; got {len(result)}")
        return result


def header_from_url(url):
    prefix = fetch_range(url, 0, 7)
    header_len = struct.unpack("<Q", prefix)[0]
    if not 2 <= header_len <= MAX_HEADER_BYTES:
        raise ValueError(f"Suspicious header size {header_len}")
    header = json.loads(fetch_range(url, 8, 7 + header_len))
    weights = {}
    for key, value in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(value, dict) or not {"dtype", "shape", "data_offsets"} <= set(value):
            raise ValueError(f"Invalid tensor header for {key}")
        weights[key] = {"dtype": value["dtype"], "shape": value["shape"], "data_offsets": value["data_offsets"]}
    return weights, header_len


def fp32_to_fp16_samples(comfy, comfy_header_len, original, original_header_len):
    """Compare a handful of matching *small* tensors by exact FP32->FP16 rounding."""
    groups = (
        "byt5_in.", "double_blocks.0.", "double_blocks.10.",
        "single_blocks.0.", "single_blocks.20.", "final_layer.", "time_in."
    )
    pairs = []
    for group in groups:
        choices = []
        for key in set(comfy) & set(original):
            c, o = comfy[key], original[key]
            c_bytes = c["data_offsets"][1] - c["data_offsets"][0]
            o_bytes = o["data_offsets"][1] - o["data_offsets"][0]
            if (key.startswith(group) and c["dtype"] == "F16" and o["dtype"] == "F32"
                    and c_bytes > 0 and c_bytes <= 16384 and o_bytes == c_bytes * 2
                    and c["shape"] == o["shape"]):
                choices.append((c_bytes, key))
        if choices:
            pairs.append(sorted(choices)[0][1])
    comparisons = []
    total_values = total_matching = 0
    for key in pairs:
        c, o = comfy[key], original[key]
        start_c, end_c = c["data_offsets"]
        start_o, end_o = o["data_offsets"]
        first_c = 8 + comfy_header_len + start_c
        first_o = 8 + original_header_len + start_o
        cdata = fetch_range(URLS["comfy_fp16"], first_c, first_c + end_c - start_c - 1)
        odata = fetch_range(URLS["tencent_original"], first_o, first_o + end_o - start_o - 1)
        matches = 0
        overflow = 0
        values = len(cdata) // 2
        for index in range(values):
            v = struct.unpack_from("<f", odata, index * 4)[0]
            try:
                expected = struct.pack("<e", v)
            except OverflowError:
                expected = b""
                overflow += 1
            if cdata[2*index:2*index+2] == expected:
                matches += 1
        total_values += values
        total_matching += matches
        comparisons.append({
            "key": key, "values": values, "exact_fp32_to_fp16": matches,
            "mismatches": values - matches, "fp16_overflow": overflow
        })
    return {
        "tensors_sampled": len(comparisons),
        "values_sampled": total_values,
        "exact_fp32_to_fp16": total_matching,
        "all_sampled_values_match": total_matching == total_values and bool(comparisons),
        "details": comparisons
    }


def qkv_fusion_order_samples(comfy, comfy_header_len, original, original_header_len):
    """Validate Comfy Q,K,V block order against Tencent separate projections.

    Read tiny slices from start/end of each part, including attention *weight*
    matrices whose full contents are enormous. No full tensor is downloaded.
    """
    keys = (
        "double_blocks.0.img_attn_qkv.weight",
        "double_blocks.0.txt_attn_qkv.weight",
        "double_blocks.10.img_attn_qkv.bias",
        "double_blocks.10.txt_attn_qkv.weight",
    )
    details = []
    matches_total = values_total = 0
    for fused in keys:
        c = comfy.get(fused)
        if c is None or c["dtype"] != "F16":
            raise ValueError(f"Comfy fused QKV missing/not FP16: {fused}")
        size_c = c["data_offsets"][1] - c["data_offsets"][0]
        if size_c % 6:
            raise ValueError(f"QKV bytes not divisible by 3 FP16 chunks: {fused}")
        part_bytes = size_c // 3
        for idx, part in enumerate(("q", "k", "v")):
            name = fused.replace("_attn_qkv.", f"_attn_{part}.")
            o = original.get(name)
            if o is None or o["dtype"] != "F32":
                raise ValueError(f"Tencent split QKV missing/not FP32: {name}")
            if o["shape"] != [c["shape"][0] // 3, *c["shape"][1:]]:
                raise ValueError(f"Inconsistent QKV part shape: {name}")
            if o["data_offsets"][1] - o["data_offsets"][0] != part_bytes * 2:
                raise ValueError(f"Inconsistent QKV part bytes: {name}")
            for place, offset_values in (("head", 0), ("tail", part_bytes // 2 - min(64, part_bytes // 2))):
                nvalues = min(64, part_bytes // 2)
                cpos = 8 + comfy_header_len + c["data_offsets"][0] + idx * part_bytes + offset_values * 2
                opos = 8 + original_header_len + o["data_offsets"][0] + offset_values * 4
                cb = fetch_range(URLS["comfy_fp16"], cpos, cpos + nvalues * 2 - 1)
                ob = fetch_range(URLS["tencent_original"], opos, opos + nvalues * 4 - 1)
                matches = 0
                for j in range(nvalues):
                    v = struct.unpack_from("<f", ob, j * 4)[0]
                    try:
                        rounded = struct.pack("<e", v)
                    except OverflowError:
                        rounded = b""
                    if cb[2*j:2*j+2] == rounded:
                        matches += 1
                details.append({
                    "tensor": fused, "part": part, "location": place,
                    "values": nvalues, "exact_fp32_to_fp16": matches
                })
                values_total += nvalues
                matches_total += matches
    return {
        "fused_tensors": len(keys), "values_sampled": values_total,
        "exact_fp32_to_fp16": matches_total,
        "all_sampled_values_match": matches_total == values_total and values_total > 0,
        "details": details
    }


def main():
    catalogs = {}
    result = {}
    for label, url in URLS.items():
        try:
            weights, header_len = header_from_url(url)
        except Exception as exc:
            result[label] = {"url": url, "error": str(exc)}
            continue
        catalogs[label] = weights
        result[label] = {
            "url": url,
            "header_bytes": header_len,
            "tensor_count": len(weights),
            "dtype_counts": dict(Counter(v["dtype"] for v in weights.values())),
            "example_keys": sorted(weights)[:12],
        }
    if "comfy_fp16" in catalogs and "tencent_original" in catalogs:
        try:
            result["numeric_fp16_source_samples"] = fp32_to_fp16_samples(
                catalogs["comfy_fp16"], result["comfy_fp16"]["header_bytes"],
                catalogs["tencent_original"], result["tencent_original"]["header_bytes"],
            )
        except Exception as exc:
            result["numeric_fp16_source_samples"] = {"error": str(exc)}
    if "comfy_fp16" in catalogs and "tencent_original" in catalogs:
        try:
            result["qkv_fusion_order_samples"] = qkv_fusion_order_samples(
                catalogs["comfy_fp16"], result["comfy_fp16"]["header_bytes"],
                catalogs["tencent_original"], result["tencent_original"]["header_bytes"],
            )
        except Exception as exc:
            result["qkv_fusion_order_samples"] = {"error": str(exc)}
    if "wangp_bf16" in catalogs:
        for label in ("comfy_fp16", "tencent_original"):
            if label in catalogs:
                result[label]["versus_wangp"] = compare(catalogs[label], catalogs["wangp_bf16"])
                if label == "comfy_fp16":
                    expanded, conversion_count = expand_fused_qkv_metadata(catalogs[label])
                    transformed = compare(expanded, catalogs["wangp_bf16"])
                    allowed_extra = {
                        "time_r_in.mlp.0.bias", "time_r_in.mlp.0.weight",
                        "time_r_in.mlp.2.bias", "time_r_in.mlp.2.weight",
                    }
                    result[label]["hypothetical_unfuse"] = {
                        "fused_tensors_split": conversion_count,
                        "comparison": transformed,
                        "only_known_unused_extra_keys": (
                            transformed["missing_keys"] == 0
                            and transformed["shape_mismatches"] == 0
                            and set(transformed["unexpected_examples"]) == allowed_extra
                            and transformed["unexpected_keys"] == len(allowed_extra)
                        ),
                    }
    print(json.dumps(result, indent=2), flush=True)
    return 1 if any("error" in entry for entry in result.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
