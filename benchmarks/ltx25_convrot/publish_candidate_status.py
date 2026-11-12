#!/usr/bin/env python3
"""Publish an explicitly experimental LTX-2.5 Candidate E video run.

Results are isolated from the trusted R0 baseline. Never label the pilot as
production-eligible or as faster without matched full-model measurements.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from publish_phase0_status import put

OUT = Path("/cache/ltx25/results")


def main():
    sha = os.environ["GITHUB_SHA"]
    run_id = os.environ["GITHUB_RUN_ID"]
    path = OUT / f"runtime_candidate_{sha[:12]}.json"
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        result = {}
    gen = result.get("generation", {})
    doc = {
        "run_id": run_id,
        "source_sha": sha,
        "run_url": f"https://github.com/mylordmonkeyman/WanGP/actions/runs/{run_id}",
        "workflow_status": os.environ.get("JOB_STATUS","unknown"),
        "status": result.get("status", "not_run"),
        "variant": "e_fp32rot",
        "model_type": result.get("model_type"),
        "candidate_e_enabled": True,
        "production_eligible": False,
        "synthetic_precision_caveat": (
            "Earlier FP32-rotation accuracy gate passed for a shape whitelist "
            "except (N=4096,K=2048); production activations not yet validated."),
        "seed": result.get("seed"),
        "frames": result.get("frames"),
        "resolution": result.get("resolution"),
        "candidate_counters": result.get("candidate_counters", {}),
        "gpu_preflight": result.get("gpu_preflight", {}),
        "blocker": result.get("blocker", ""),
        "generation": {
            "returncode": gen.get("returncode"),
            "elapsed_seconds": gen.get("elapsed_seconds"),
            "tail": gen.get("log_tail","")[-3500:],
        },
        "videos": result.get("videos", []),
        "note": "Explicit opt-in experimental subset, not an automatically selected backend.",
    }
    put("ltx25-candidate/latest.json", doc)
    put(f"ltx25-candidate/runs/{run_id}.json", doc)
    print("LTX25_CANDIDATE_PUBLISHED " + json.dumps({
        "run_id":run_id,"status":doc["status"],
        "calls":doc["candidate_counters"].get("used"),
        "videos":doc["videos"],
    })[:6000],flush=True)


if __name__ == "__main__":
    main()
