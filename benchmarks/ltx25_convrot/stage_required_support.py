#!/usr/bin/env python3
"""Bounded pinned LTX-2.5 support staging; no pip, no core checkpoint changes.

WanGP LTX2_25 handler query_model_files() requires both spatial and temporal
upscalers plus diffusion video VAE, even for the no-upscaling baseline.
Also verify the separately staged HDR LoRA and HDR scene embeddings, and
repair them by SHA256 if a prior download was truncated. Never modify the
previously verified 40 GiB of core transformer, encoder and other weights.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from huggingface_hub import HfApi, hf_hub_download
from support_manifest import REPO, REV, BASE, OUT, ASSETS

REPAIRABLE = (
    "ltx-2.5-temporal-upscaler-x2-1.0_bf16.safetensors",
    "ltx-2.5-22b_diffusion_video_vae_bf16.safetensors",
    "ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors",
    "ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors",
)
MAX_DOWNLOAD = 8 * 1024**3
RESERVE = 5 * 1024**3


def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    started = time.monotonic()
    sha = os.environ.get("GITHUB_SHA", "local")
    BASE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"sha": sha, "revision": REV, "no_pip": True,
              "status": "BLOCKED", "files": [], "downloads": []}
    target = OUT / f"required_support_{sha[:12]}.json"
    try:
        siblings = HfApi().model_info(REPO, revision=REV, files_metadata=True).siblings
        remote = {item.rfilename: item for item in siblings}
        names = tuple(dict.fromkeys(tuple(ASSETS) + REPAIRABLE))
        repair = []
        new_bytes = 0
        for name in names:
            item = remote.get(name)
            if item is None:
                raise RuntimeError(f"Not found at pinned revision {REV}: {name}")
            expected_size = item.size or (item.lfs or {}).get("size")
            if not isinstance(expected_size, int) or expected_size <= 0:
                raise RuntimeError(f"Bad upstream metadata for {name}")
            path = BASE / name
            if path.is_symlink():
                raise RuntimeError(f"Refusing symlink {path}")
            actual_size = path.stat().st_size if path.is_file() else None
            row = {"name": name, "expected_size": expected_size,
                   "actual_size": actual_size, "path": str(path)}
            report["files"].append(row)
            if name not in REPAIRABLE:
                # All original core assets had SHA verified by the previous
                # stage-support workflow; avoid rehashing ~40GiB each run.
                if actual_size != expected_size:
                    raise RuntimeError(
                        f"Original core checkpoint unexpectedly missing/different: "
                        f"{name}, expected {expected_size}, got {actual_size}")
                row["status"] = "PREVIOUSLY_VERIFIED_SIZE_OK"
                continue
            expected_sha = (item.lfs or {}).get("sha256")
            if not expected_sha:
                raise RuntimeError(f"No official LFS SHA256 for {name}")
            if actual_size == expected_size and file_sha256(path) == expected_sha:
                row["status"] = "SHA256_PASS"
                continue
            # For these four only, download into a separate staging path
            # and replace the checkpoint atomically after full verification.
            row["status"] = "REPAIR_NEEDED" if actual_size is not None else "MISSING"
            row["sha256_expected"] = expected_sha
            repair.append((name, path, expected_size, expected_sha, row))
            new_bytes += expected_size
        if new_bytes > MAX_DOWNLOAD:
            raise RuntimeError(f"Too much new support data: {new_bytes} bytes")
        if new_bytes and shutil.disk_usage(BASE).free < new_bytes + RESERVE:
            raise RuntimeError("Insufficient free disk for staged support weights")

        staging = BASE / ".ltx25-pinned-staging"
        for name, dest, expected_size, expected_sha, row in repair:
            print("LTX25_SUPPORT_FETCH", json.dumps({
                "name": name, "expected_size": expected_size,
                "previous_size": row["actual_size"]}), flush=True)
            staging.mkdir(parents=True, exist_ok=True)
            staged = Path(hf_hub_download(
                repo_id=REPO, revision=REV, filename=name, local_dir=str(staging)))
            expected_stage = staging / name
            if staged.is_symlink() or staged.resolve() != expected_stage.resolve():
                raise RuntimeError(f"Unsafe Hub destination: {staged}")
            if staged.stat().st_size != expected_size:
                raise RuntimeError(f"Downloaded size mismatch: {name}")
            if file_sha256(staged) != expected_sha:
                raise RuntimeError(f"Downloaded SHA256 mismatch: {name}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.is_symlink():
                raise RuntimeError(f"Symlink appeared at destination: {dest}")
            os.replace(staged, dest)
            row["status"] = "REPLACED_SHA256_VERIFIED"
            row["sha256_verified"] = True
            report["downloads"].append({"name": name,
                                         "bytes": expected_size,
                                         "previous_size": row["actual_size"],
                                         "sha256_ok": True})
            print("LTX25_SUPPORT_SHA256_OK", name, flush=True)
        report["status"] = "PASS"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        report["elapsed_sec"] = round(time.monotonic() - started, 2)
        target.write_text(json.dumps(report, indent=2) + "\n")
        print("LTX25_SUPPORT_SUMMARY", json.dumps({
            k: v for k, v in report.items() if k != "files"})[:4000], flush=True)
        for row in report["files"]:
            if row.get("status") not in ("PREVIOUSLY_VERIFIED_SIZE_OK", "SHA256_PASS"):
                print("LTX25_SUPPORT_FILE", json.dumps(row)[:1600], flush=True)
        if report["status"] != "PASS":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
