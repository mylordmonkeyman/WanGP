#!/usr/bin/env python3
"""Publish only quality-variant reports, separate from established 576x320 R0."""
from __future__ import annotations

import json
import os
from pathlib import Path

from publish_phase0_status import put

OUT = Path("/cache/ltx25/results")
VARIANTS = {"r0_quality_1024": "quality_r0", "e_quality_1024": "quality_e",
            "e_real_activation_audit_1024": "quality_real_audit",
            "e_profile_1024": "quality_stage_profile",
            "e_workspace_1024": "quality_workspace",
            "e_cast4_1024": "quality_cast4"}


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
        "real_activation_audit": r.get("real_activation_audit"),
        "stage_profile": r.get("stage_profile"),
        "workspace_stats": r.get("workspace_stats"),
        "workspace_video_gate": r.get("workspace_video_gate"),
        "cast4_video_gate": r.get("cast4_video_gate"),
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
            "INSTRUMENTED REAL MODEL ACTIVATIONS: R0/R1/Candidate E comparisons "
            "synchronize and add expensive FP32 work; the recorded generation time "
            "MUST NOT be compared to normal runtime performance. "
            "Only sampled activation rows are checked; visual review and broader "
            "per-shape sampling remain required."
            if variant == "e_real_activation_audit_1024" else
            "INSTRUMENTED STAGE PROFILING: sparse CUDA events and host tracing "
            "affect generation timing. Do NOT treat walltime as a performance result. "
            "Compare only reported stage costs, then rerun a matched uninstrumented "
            "full-video benchmark."
            if variant == "e_profile_1024" else
            "Opt-in bounded reusable scratch (no persistent FP16 weight copy). "
            "Compare this clip against the no-sync E reference by decoded frame hash, "
            "and use controlled repeat runs before claiming a timing speedup."
            if variant == "e_workspace_1024" else
            "Vector4 INT8-to-FP16 conversion pilot (transient FP16 weight, "
            "not cached). Require native cast preflight, exact 49-frame decoded "
            "video equality, and matched repeated uninstrumented timing trials."
            if variant == "e_cast4_1024" else
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
        "real_activation_status": (r.get("real_activation_audit") or {}).get("status"),
        "real_activation_samples": (r.get("real_activation_audit") or {}).get("total_samples"),
        "stage_profile_status": (r.get("stage_profile") or {}).get("status"),
        "stage_profile_samples": (r.get("stage_profile") or {}).get("sample_count"),
        "workspace_leases": (r.get("workspace_stats") or {}).get("acquired"),
        "videos": doc["videos"],
    })[:5000], flush=True)


if __name__ == "__main__":
    main()
