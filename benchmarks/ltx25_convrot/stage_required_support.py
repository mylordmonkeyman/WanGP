#!/usr/bin/env python3
"""Stage the two omitted LTX-2.5 compulsory support files, never install pip.

WanGP query_model_files() always lists temporal upscaler and diffusion VAE,
even for a no-upscaling text-to-video request. Other 13 core files were
previously checksum verified; HDR LoRA and scene embeddings separately verified.
Downloads are pinned, bounded and checksum-verified, without hashing ~40GiB
of already known-good files at every benchmark iteration.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

from huggingface_hub import HfApi, hf_hub_download
from support_manifest import REPO, REV, BASE, OUT, ASSETS

ADDITIONAL = (
    "ltx-2.5-temporal-upscaler-x2-1.0_bf16.safetensors",
    "ltx-2.5-22b_diffusion_video_vae_bf16.safetensors",
)
REQUIRE_PRESENT = tuple(name for name in ASSETS if name not in ADDITIONAL)
MAX_DOWNLOAD_BYTES = 8 * 1024**3
MIN_RESERVE_BYTES = 5 * 1024**3


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    started = time.monotonic()
    sha = os.environ.get("GITHUB_SHA", "local")
    OUT.mkdir(parents=True, exist_ok=True)
    BASE.mkdir(parents=True, exist_ok=True)
    report = {"sha": sha, "revision": REV, "status": "ERROR",
              "install_attempted": False, "files": [], "downloads": []}
    try:
        names = list(dict.fromkeys(REQUIRE_PRESENT + ADDITIONAL))
        inventory = {s.rfilename: s for s in HfApi().model_info(
            REPO, revision=REV, files_metadata=True).siblings}
        to_download = []
        required_bytes = 0
        for name in names:
            item = inventory.get(name)
            if item is None:
                raise RuntimeError(f"No upstream file at pinned revision: {name}")
            length = item.size or (item.lfs or {}).get("size")
            if not isinstance(length, int) or length <= 0:
                raise RuntimeError(f"Missing upstream length: {name}")
            path = BASE / name
            if path.is_symlink():
                raise RuntimeError(f"Symlink forbidden for models: {path}")
            valid_size = path.is_file() and path.stat().st_size == length
            if name in ADDITIONAL and not valid_size:
                if path.exists():
                    raise RuntimeError(f"Existing file has wrong size; refusing overwrite: {name}")
                if not (item.lfs or {}).get("sha256"):
                    raise RuntimeError(f"Missing LFS hash: {name}")
                to_download.append((name, length, item.lfs["sha256"]))
                required_bytes += length
            elif not valid_size:
                raise RuntimeError(f"Previously staged core file missing/wrong size: {name}")
            report["files"].append({"name": name, "bytes": length,
                                    "already_present": valid_size})
        if required_bytes > MAX_DOWNLOAD_BYTES:
            raise RuntimeError(f"Download exceeds cap: {required_bytes} bytes")
        if shutil.disk_usage(BASE).free < required_bytes + MIN_RESERVE_BYTES:
            raise RuntimeError("Insufficient disk capacity for bounded support download")
        for name, length, expected_hash in to_download:
            print("STAGE_LTX25_REQUIRED", name, length, flush=True)
            path = Path(hf_hub_download(repo_id=REPO, revision=REV,
                                        filename=name, local_dir=str(BASE)))
            if path.is_symlink() or path.resolve() != (BASE / name).resolve():
                raise RuntimeError(f"Unexpected checkpoint destination: {path}")
            if path.stat().st_size != length or sha256_file(path) != expected_hash:
                raise RuntimeError(f"Size/hash verification failed: {name}")
            report["downloads"].append({"name": name, "bytes": length, "sha256_ok": True})
            print("STAGE_LTX25_VERIFIED", name, flush=True)
        for name in ADDITIONAL:
            item = inventory[name]
            path = BASE / name
            if sha256_file(path) != (item.lfs or {}).get("sha256"):
                raise RuntimeError(f"Pinned SHA256 failed: {name}")
        report["status"] = "PASS"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        report["elapsed_sec"] = round(time.monotonic() - started, 2)
        target = OUT / f"required_support_{sha[:12]}.json"
        target.write_text(json.dumps(report, indent=2) + "\n")
        print("STAGE_LTX25_RESULT", json.dumps(report)[:8000], flush=True)
        if report["status"] != "PASS":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
