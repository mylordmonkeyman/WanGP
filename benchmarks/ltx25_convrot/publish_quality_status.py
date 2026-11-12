#!/usr/bin/env python3
"""Publish only quality-variant reports, separate from established 576x320 R0."""
from __future__ import annotations

import json
import os
from pathlib import Path

from publish_phase0_status import put

OUT = Path("/cache/ltx25/results")
VARIANTS = {"r0_quality_1024": "quality_r0", "e_quality_1024": "quality_e"}


def main():
    variant = os.environ["LTX25_BENCH_VARIANT"]
    if variant not in VARIANTS:
        raise SystemExit(f"Unexpected quality variant: {variant}")
    run_id = os.environ["GITHUB_RUN_ID"]
    sha = os.environ["GITHUB_SHA"]
    report = OUT / f"runtime_{VARIANTS[variant]}_{sha[:12]}.json"
    try:
        r = json.loads(report.read_text())
    except (OSError, json.JSONDecodeError):
        r = {}
    generation = r.get("generation", {})
    counters = r.get("candidate_counters", {})
    doc = {
        "run_id": run_id,
        "run_url": f"https://github.com/mylordmonkeyman/WanGP/actions/runs/{run_id}",
        "source_sha": sha,
        "variant": variant,
        "status": r.get("status", "not_run"),
        "workflow_status": os.environ.get("JOB_STATUS", "unknown"),
        "runtime_bootstrap_status": r.get("runtime_bootstrap_status"),
        "model_type": r.get("model_type"),
        "resolution": r.get("resolution"),
        "frames": r.get("frames"),
        "seed": r.get("seed"),
        "num_inference_steps": r.get("num_inference_steps"),
        "candidate_e_enabled": r.get("candidate_e_enabled"),
        "candidate_counters": counters,
        "production_eligible": False if variant.startswith("e_") else None,
        "baked_checkpoint_root": "/opt/ltx25/checkpoints",
        "checkpoint_fetched_in_job": False,
        "checkpoint_sha256_in_job": False,
        "generation": {
            "returncode": generation.get("returncode"),
            "elapsed_seconds": generation.get("elapsed_seconds"),
            "log_tail": generation.get("log_tail", "")[-4500:],
        },
        "blocker": r.get("blocker", ""),
        "error": r.get("error", ""),
        "gpu_preflight": r.get("gpu_preflight"),
        "videos": r.get("videos", []),
        "note": (
            "Matched 1024x576, 49-frame, 8-step text-to-video quality comparison. "
            "Visual review required; do not infer end-to-end speedup from synthetic layer tests."
        ),
    }
    root = "ltx25-quality/" + variant
    put(root + "/latest.json", doc)
    put(root + f"/runs/{run_id}.json", doc)
    print("LTX25_QUALITY_RESULT " + json.dumps({
        "variant": variant, "run_id": run_id,
        "status": doc["status"], "elapsed_seconds": doc["generation"]["elapsed_seconds"],
        "candidate_used": counters.get("used", 0),
        "videos": doc["videos"],
    })[:5000], flush=True)


if __name__ == "__main__":
    main()
