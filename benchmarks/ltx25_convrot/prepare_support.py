#!/usr/bin/env python3
"""Resumably stage the pinned LTX-2.5 distilled inference support assets.

Stores real files, not symlinks, only in the runner /cache bind mount.
Requires exact HF Hub revision. Verifies LFS SHA256 for every safetensors file.
Rejects missing metadata, unexpected paths, inadequate disk, and excess
aggregate download volume. Will never download another transformer quant.
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

from support_manifest import REPO, REV, BASE, ASSETS, TRANSFORMER, OUT

MAX_NEW_DOWNLOAD = 100 * 1024**3
DISK_RESERVE = 50 * 1024**3


def hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fd:
        for chunk in iter(lambda: fd.read(16 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "source_sha": os.environ.get("GITHUB_SHA", "local"),
        "repo": REPO, "revision": REV,
        "checkpoint_directory": str(BASE),
        "weights_requantized": False,
        "files": [],
    }
    try:
        siblings = HfApi().model_info(REPO, revision=REV, files_metadata=True).siblings
        by_name = {s.rfilename: s for s in siblings}
        expected = []
        missing_bytes = 0
        for name in ASSETS:
            info = by_name.get(name)
            if not info:
                raise RuntimeError(f"Not in pinned revision: {name}")
            size = info.size or (info.lfs or {}).get("size")
            sha256 = (info.lfs or {}).get("sha256")
            if not isinstance(size, int) or size < 1:
                raise RuntimeError(f"Missing size metadata for {name}")
            if name.endswith(".safetensors") and not sha256:
                raise RuntimeError(f"Missing LFS SHA256 for weight file {name}")
            dest = BASE / name
            if dest.is_symlink():
                raise RuntimeError(f"Existing file is a symlink; refusing: {dest}")
            # Conservative estimate: even apparently existing files might
            # need repair. Verify SHA before reusing.
            if not dest.is_file() or dest.stat().st_size != size:
                missing_bytes += size
            expected.append((name, dest, size, sha256))

        if missing_bytes > MAX_NEW_DOWNLOAD:
            raise RuntimeError(f"New downloads exceed {MAX_NEW_DOWNLOAD} byte cap")
        if shutil.disk_usage(BASE).free < missing_bytes + DISK_RESERVE:
            raise RuntimeError("Insufficient available filesystem capacity")

        for index, (name, dest, size, sha256) in enumerate(expected, 1):
            print(f"[{index}/{len(expected)}] {name}, {size:,} bytes", flush=True)
            if dest.is_file() and dest.stat().st_size == size and sha256:
                found = hash_file(dest)
                if found != sha256:
                    raise RuntimeError(f"Local file SHA256 mismatch: {name} (refusing overwrite)")
                report["files"].append({"path": name, "bytes": size, "hash_verified": True,
                                        "source": "already_present"})
                print("  EXISTING SHA256 VERIFIED", flush=True)
                continue
            path = Path(hf_hub_download(
                repo_id=REPO, revision=REV, filename=name,
                local_dir=str(BASE),
            ))
            if path.is_symlink() or path.resolve() != dest.resolve():
                raise RuntimeError(f"Unexpected Hub download destination or symlink: {path}")
            if path.stat().st_size != size:
                raise RuntimeError(f"Size mismatch for {name}")
            if sha256:
                found = hash_file(path)
                if found != sha256:
                    raise RuntimeError(f"Downloaded SHA256 mismatch: {name}")
            report["files"].append({"path": name, "bytes": size,
                                    "hash_verified": bool(sha256),
                                    "source": "downloaded"})
            print(f"  VERIFIED: {name} (LFS SHA256: {bool(sha256)})", flush=True)

        report["status"] = "PASS"
    except Exception as err:
        report["status"] = "ERROR"
        report["error_type"] = type(err).__name__
        report["error"] = str(err)[:700]
    path = OUT / f"support_download_{report['source_sha'][:12]}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("SUPPORT_DOWNLOAD_RESULT " + json.dumps(report)[:16000], flush=True)
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
