#!/usr/bin/env python3
"""Read only the *published* LTX-2.5 INT8 ConvRot safetensors header.

No model payload is downloaded or modified. This is checkpoint *metadata*
inspection, NOT local checkpoint validation or inference shape observation.
Each HTTP response is bounded and closed immediately. A server that ignores
Range will be rejected rather than silently transferring 19.5 GB.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import struct
import sys
import urllib.error
import urllib.request

REVISION = "0e281e1132633c9ff66e1891c045c533e785ce67"
FILE = "ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors"
EXPECTED_LFS_SHA256 = "ae777a7d5ea1830a9d0acdf784a1af278bfe421f109a8a8e2f0e1f2b03af34ef"
EXPECTED_SIZE = 19483116240
URL = f"https://huggingface.co/DeepBeepMeep/LTX-2/resolve/{REVISION}/{FILE}"
OUT = Path("/cache/ltx25/results")
MAX_HEADER = 32 * 1024 * 1024


def bounded_range(start: int, length: int) -> bytes:
    if length <= 0 or length > MAX_HEADER:
        raise RuntimeError(f"Refusing requested range length {length}")
    end = start + length - 1
    req = urllib.request.Request(URL, headers={
        "Range": f"bytes={start}-{end}",
        "User-Agent": "WanGP-Phase0-Metadata-Only/1.0",
        "Accept-Encoding": "identity",
    })
    with urllib.request.urlopen(req, timeout=90) as r:
        if r.status != 206:
            raise RuntimeError(f"HTTP {r.status}: server did not honor Range; refused to read model payload")
        content_range = r.headers.get("Content-Range", "")
        expected = f"bytes {start}-{end}/"
        if not content_range.startswith(expected):
            raise RuntimeError(f"Unexpected Content-Range: {content_range[:150]}")
        data = r.read(length + 1)
        if len(data) != length:
            raise RuntimeError(f"Wrong range size {len(data)}, wanted {length}")
        return data


def inventory():
    header_size = struct.unpack("<Q", bounded_range(0, 8))[0]
    if not 16 <= header_size <= MAX_HEADER:
        raise RuntimeError(f"Invalid/bounds-exceeding header size: {header_size}")
    header = json.loads(bounded_range(8, header_size).decode("utf-8"))
    metadata = header.get("__metadata__") or {}
    keys = set(header)
    weights = []
    bad_scales = 0
    groups = {}
    count = 0
    total_stored_i8 = 0
    for key, info in header.items():
        if key == "__metadata__" or not isinstance(info, dict):
            continue
        count += 1
        shape = info.get("shape", [])
        if key.endswith(".weight") and info.get("dtype") == "I8" and len(shape) == 2:
            n, k = map(int, shape)
            g = next((g for g in (256, 64, 16) if k % g == 0), 0)
            groups[str(g)] = groups.get(str(g), 0) + 1
            scale_key = key[:-7] + ".weight_scale"
            scale = header.get(scale_key, {})
            scale_ok = scale.get("dtype") == "F32"
            bad_scales += int(not scale_ok)
            total_stored_i8 += n * k
            weights.append({
                "name": key[:-7], "N": n, "K": k, "group_candidate": g,
                "scale_dtype": scale.get("dtype"), "scale_shape": scale.get("shape"),
                "scale_is_fp32": scale_ok
            })
    weights.sort(key=lambda x: x["K"] * x["N"], reverse=True)
    return {
        "header_length_bytes": header_size, "total_tensors": count,
        "quantization_format": metadata.get("quantization_format"),
        "convrot_format": metadata.get("convrot_format"),
        "int8_matrix_count": len(weights), "int8_elements": total_stored_i8,
        "int8_missing_fp32_scale": bad_scales,
        "shape_group_candidates": groups,
        "largest_16_weight_matrices": weights[:16],
        "matrix_shape_frequencies": sorted([
            {"N": n, "K": k, "count": sum(1 for x in weights if (x["N"], x["K"]) == (n,k))}
            for n,k in set((x["N"], x["K"]) for x in weights)
        ], key=lambda x: (-x["count"], -(x["N"]*x["K"])))[:50],
        "caveat": "N/K are checkpoint matrix shapes; M is unknown until actual generation. Group candidates can differ after config/fused splits."
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sha = os.environ.get("GITHUB_SHA", "local")
    report = {
        "kind": "remote_header_metadata_only",
        "source_sha": sha,
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hf_repo": "DeepBeepMeep/LTX-2", "hf_revision": REVISION,
        "filename": FILE, "published_size_bytes": EXPECTED_SIZE,
        "published_lfs_sha256": EXPECTED_LFS_SHA256,
        "actual_weight_payload_downloaded": False,
        "local_weights_sha256_verified": False
    }
    try:
        report["inventory"] = inventory()
        report["status"] = "PASS"
    except Exception as e:
        report["status"] = "ERROR"
        report["error_type"] = type(e).__name__
        report["error"] = str(e)[:700]
    path = OUT / f"phase0_remote_header_{sha[:12]}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("REMOTE_HEADER_RESULT " + json.dumps(report, separators=(",", ":"))[:12000], flush=True)
    print(f"REMOTE_HEADER_PATH={path}", flush=True)
    # Diagnostic isn't a required stage of baseline, so preserve status in JSON
    # without classifying network/proxy failures as passing model execution.


if __name__ == "__main__":
    main()
