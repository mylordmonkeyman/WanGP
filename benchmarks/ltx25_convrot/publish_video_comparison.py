#!/usr/bin/env python3
"""Publish bounded decoded-video metrics to the dedicated results branch."""
from __future__ import annotations

import json
import os
from pathlib import Path

from publish_phase0_status import put


def main():
    sha = os.environ["GITHUB_SHA"]
    run_id = os.environ["GITHUB_RUN_ID"]
    path = Path("/cache/ltx25/results") / f"video_comparison_{sha[:12]}.json"
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        report = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
    result = {
        **report, "source_sha": sha, "run_id": run_id,
        "run_url": f"https://github.com/mylordmonkeyman/WanGP/actions/runs/{run_id}",
        "workflow_status_at_publication": os.environ.get("JOB_STATUS", "unknown"),
        "note": (
            "Decoded-pixel identity and PSNR/SSIM are objective difference metrics "
            "for fixed-seed videos. They are NOT perceptual-quality judgments. "
            "Human viewing is required before approving quality."
        ),
    }
    put("ltx25-video-compare/latest.json", result)
    put(f"ltx25-video-compare/runs/{run_id}.json", result)
    print("LTX25_VIDEO_COMPARE_PUBLISHED " + json.dumps({
        "status": result.get("status"), "run_id": run_id,
        "pair_count": len(result.get("pairs", [])),
        "contact_sheet": result.get("contact_sheet"),
    }), flush=True)


if __name__ == "__main__":
    main()
