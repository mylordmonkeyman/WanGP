#!/usr/bin/env python3
"""Keep isolated V100 conversion evidence separate from video results."""
from __future__ import annotations

import json
import os
from pathlib import Path

from publish_phase0_status import put


def main():
    sha = os.environ["GITHUB_SHA"]
    run_id = os.environ["GITHUB_RUN_ID"]
    result = Path("/cache/ltx25/results") / f"cast_microbench_{sha[:12]}.json"
    try:
        data = json.loads(result.read_text())
    except (FileNotFoundError, ValueError) as exc:
        data = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
    data.update({
        "run_id": run_id, "run_url": f"https://github.com/mylordmonkeyman/WanGP/actions/runs/{run_id}",
        "source_sha": sha,
        "workflow_status": os.environ.get("JOB_STATUS", "unknown"),
        "note": "Isolated synthetic matrix conversion, not an LTX generation speedup or a production kernel.",
    })
    put("ltx25-cast-microbench/latest.json", data)
    put(f"ltx25-cast-microbench/runs/{run_id}.json", data)
    print("LTX25_CAST_PUBLISHED " + json.dumps({
        "run_id": run_id, "status": data.get("status"),
        "shapes": len(data.get("shape_results", []))
    }), flush=True)


if __name__ == "__main__":
    main()
