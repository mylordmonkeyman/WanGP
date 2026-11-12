#!/usr/bin/env python3
"""Publish a compact, nonsecret Phase 0 run summary to a separate results branch.

Uses the ephemeral GitHub Actions token. No API key is persisted to disk.
The results branch does not trigger the benchmark workflow.
"""
from __future__ import annotations

import base64
import datetime as dt
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request

RESULTS_BRANCH = "perf/ltx25-run-results"
REPOSITORY = "mylordmonkeyman/WanGP"


def request(path: str, method: str = "GET", data: dict | None = None):
    token = os.environ["GH_TOKEN"]
    url = f"https://api.github.com/repos/{REPOSITORY}/{path}"
    body = None if data is None else json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, method=method, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
        "User-Agent": "WanGP-LTX25-V100-Phase0",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as err:
        if err.code == 404 and method == "GET":
            return None
        raise RuntimeError(f"GitHub API HTTP {err.code}: {err.read().decode('utf-8')[:400]}") from None


def put(path: str, doc: dict):
    existing = request("contents/" + path + "?ref=" + urllib.parse.quote(RESULTS_BRANCH, safe=""))
    encoded = base64.b64encode((json.dumps(doc, indent=2, sort_keys=True) + "\n").encode()).decode()
    payload = {
        "message": f"ci(ltx25): report Phase 0 run {doc['run_id']} [skip ci]",
        "branch": RESULTS_BRANCH,
        "content": encoded,
    }
    if existing and "sha" in existing:
        payload["sha"] = existing["sha"]
    request("contents/" + path, "PUT", payload)


def main():
    sha = os.environ["GITHUB_SHA"]
    run_id = os.environ["GITHUB_RUN_ID"]
    probe_file = Path(f"/cache/ltx25/results/phase0_probe_{sha[:12]}.json")
    report = json.loads(probe_file.read_text()) if probe_file.exists() else {}
    remote_file = Path(f"/cache/ltx25/results/phase0_remote_header_{sha[:12]}.json")
    remote = json.loads(remote_file.read_text()) if remote_file.exists() else {}
    remote_inv = remote.get("inventory", {})
    rotations = report.get("rotation", [])
    checkpoint = report.get("checkpoint", {})
    summary = {
        "source_sha": sha,
        "run_id": run_id,
        "run_url": f"https://github.com/{REPOSITORY}/actions/runs/{run_id}",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "job_status": os.environ.get("JOB_STATUS", "unknown"),
        "gpu": report.get("gpu", {}).get("name"),
        "capability": report.get("gpu", {}).get("capability"),
        "torch": report.get("torch"),
        "cuda": report.get("torch_cuda"),
        "phase0_probe_present": bool(report),
        "checkpoint_visible": checkpoint.get("present"),
        "remote_header_status": remote.get("status", "not_run"),
        "remote_header_error": remote.get("error", "")[:300],
        "remote_hf_revision": remote.get("hf_revision"),
        "remote_published_lfs_sha256": remote.get("published_lfs_sha256"),
        "remote_int8_matrix_count": remote_inv.get("int8_matrix_count"),
        "remote_group_candidates": remote_inv.get("shape_group_candidates"),
        "remote_int8_missing_fp32_scale": remote_inv.get("int8_missing_fp32_scale"),
        "remote_header_top_shapes": remote_inv.get("matrix_shape_frequencies", [])[:12],
        "checkpoint_metadata_verified": checkpoint.get("checkpoint_metadata_verified", False),
        "checkpoint_layer_count": checkpoint.get("int8_2d_weight_count"),
        "checkpoint_blocker": checkpoint.get("blocker", "")[:300],
        "checkpoint_inventory_error": checkpoint.get("inventory_error", "")[:300],
        "rotation_test_count": len(rotations),
        "rotation_pass_count": sum(x["status"] == "PASS" for x in rotations),
        "rotation_results": [{
            "M": r["M"], "K": r["K"], "group_size": r["group_size"],
            "status": r["status"], "median_ms": r.get("median_ms"),
            "p99_abs": r.get("p99_abs"), "max_abs": r.get("max_abs"),
            "error": r.get("error", "")[:300]
        } for r in rotations],
        "actual_model_shapes_observed": False,
        "full_pipeline_measured": False,
        "note": "Synthetic diagnostic only; not an LTX-2.5 model benchmark or production accuracy gate."
    }
    put("ltx25-phase0/latest.json", summary)
    put(f"ltx25-phase0/runs/{run_id}.json", summary)
    print(f"Published sanitized Phase 0 report for run {run_id} on {RESULTS_BRANCH}", flush=True)


if __name__ == "__main__":
    main()
