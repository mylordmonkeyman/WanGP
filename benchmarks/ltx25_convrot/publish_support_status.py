#!/usr/bin/env python3
"""Publish sanitized, hash-only LTX-2.5 support download receipt."""
from __future__ import annotations
import json
import os
from pathlib import Path
from publish_phase0_status import put, REPOSITORY

def main():
    sha = os.environ["GITHUB_SHA"]
    run_id = os.environ["GITHUB_RUN_ID"]
    path = Path(f"/cache/ltx25/results/support_download_{sha[:12]}.json")
    raw = json.loads(path.read_text()) if path.exists() else {}
    files = raw.get("files", [])
    summary = {
        "source_sha": sha,
        "run_id": run_id,
        "run_url": f"https://github.com/{REPOSITORY}/actions/runs/{run_id}",
        "job_status": os.environ.get("JOB_STATUS", "unknown"),
        "download_status": raw.get("status", "report_not_found"),
        "error": raw.get("error", "")[:400],
        "checkpoint_directory": "/cache/ltx25/checkpoints",
        "files_verified": sum(x.get("hash_verified") for x in files),
        "verified_bytes": sum(x.get("bytes", 0) for x in files if x.get("hash_verified")),
        "files": files,
        "note": "Model weights staged, not loaded or benchmarked. Non-LFS small text files validated by Hub's downloader, not independent SHA256."
    }
    put("ltx25-support/latest.json", summary)
    put(f"ltx25-support/runs/{run_id}.json", summary)
    print("Published LTX-2.5 support download receipt", flush=True)

if __name__ == "__main__":
    main()
