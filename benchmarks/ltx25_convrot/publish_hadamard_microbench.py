#!/usr/bin/env python3
"""Publish isolated rotation benchmark; never confuse with video performance."""
from __future__ import annotations

import json
import os
from pathlib import Path
from publish_phase0_status import put

def main():
    sha=os.environ["GITHUB_SHA"]
    rid=os.environ["GITHUB_RUN_ID"]
    result=Path("/cache/ltx25/results") / f"hadamard_microbench_{sha[:12]}.json"
    try:
        data=json.loads(result.read_text())
    except (OSError,ValueError) as ex:
        data={"status":"ERROR","error":f"{type(ex).__name__}: {ex}"}
    data.update({
        "run_id":rid,
        "run_url":f"https://github.com/mylordmonkeyman/WanGP/actions/runs/{rid}",
        "workflow_status":os.environ.get("JOB_STATUS","unknown"),
        "source_sha":sha,
        "warning":"Synthetic H256 rotation benchmark, not full-video speed or quality validation.",
    })
    put("ltx25-hadamard-microbench/latest.json",data)
    put(f"ltx25-hadamard-microbench/runs/{rid}.json",data)
    print("LTX25_HADAMARD_PUBLISHED "+json.dumps({
        "run_id":rid,"status":data.get("status"),
        "cases":len(data.get("shape_results",[])),
    }),flush=True)

if __name__=="__main__":main()
