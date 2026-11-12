#!/usr/bin/env python3
"""Publish one bounded runtime-validation batch report to the results branch."""
from __future__ import annotations

import json
import os
from pathlib import Path

from publish_phase0_status import put

ROOT = Path("/cache/ltx25/results")

def load(name):
    try:
        return json.loads((ROOT / name).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def main():
    sha=os.environ["GITHUB_SHA"]
    run_id=os.environ["GITHUB_RUN_ID"]
    boot=load(f"runtime_bootstrap_{sha[:12]}.json")
    base=load(f"runtime_baseline_{sha[:12]}.json")
    doc={
        "source_sha":sha,
        "run_id":run_id,
        "run_url":f"https://github.com/mylordmonkeyman/WanGP/actions/runs/{run_id}",
        "job_status_at_publish":os.environ.get("JOB_STATUS", "unknown"),
        "runtime_bootstrap":{
            "status":boot.get("status","not_run"),
            "error":boot.get("error","")[:300],
            "torch_initial":boot.get("initial_torch",{}).get("tail","")[-350:],
            "torch_after":boot.get("torch_after",{}).get("tail","")[-350:],
            "steps":[{"name":s["group"],"returncode":s["returncode"],
                      "seconds":s.get("seconds"),
                      "detail":s.get("tail","")[-700:] if s["returncode"] else ""}
                     for s in boot.get("steps",[])],
            "imports":[{"name":m["name"],"status":m["status"],
                        "detail":m.get("tail","")[-900:]} for m in boot.get("probes",[])],
        },
        "baseline":{
            "status":base.get("status","not_run"),
            "blocker":base.get("blocker","")[:400],
            "model_type":base.get("model_type"),
            "seed":base.get("seed"),
            "frames":base.get("frames"),
            "resolution":base.get("resolution"),
            "checkpoint_visible":base.get("checkpoint_visible"),
            "gemma_checkpoint_visible":base.get("gemma_checkpoint_visible"),
            "candidate_e_enabled":False,
            "gpu_preflight":base.get("gpu_preflight"),
            "configured_dryrun":{
                "returncode":base.get("configured_cli_dryrun",{}).get("returncode"),
                "elapsed_seconds":base.get("configured_cli_dryrun",{}).get("elapsed_seconds"),
                "tail":base.get("configured_cli_dryrun",{}).get("log_tail","")[-1900:],
            },
            "initial_dryrun":{
                "returncode":base.get("initial_cli_dryrun",{}).get("returncode"),
                "tail":base.get("initial_cli_dryrun",{}).get("log_tail","")[-900:],
            },
            "generation":{
                "returncode":base.get("generation",{}).get("returncode"),
                "elapsed_seconds":base.get("generation",{}).get("elapsed_seconds"),
                "tail":base.get("generation",{}).get("log_tail","")[-2900:],
            },
            "video_dir":base.get("video_dir"),
            "videos":base.get("videos",[]),
        },
        "note":"Only the original WanGP INT8 fallback was used for any output; Candidate E remains disabled."
    }
    put("ltx25-runtime/latest.json",doc)
    put(f"ltx25-runtime/runs/{run_id}.json",doc)
    print("LTX25_RUNTIME_PUBLISH " + json.dumps({
        "run_id":run_id,"runtime":doc["runtime_bootstrap"]["status"],
        "baseline":doc["baseline"]["status"],"videos":doc["baseline"]["videos"]
    })[:2000],flush=True)

if __name__=="__main__":
    main()
