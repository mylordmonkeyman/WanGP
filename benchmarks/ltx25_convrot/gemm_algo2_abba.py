#!/usr/bin/env python3
"""Matched V100 I2V-independent T2V A-B-B-A benchmark: default vs cuBLAS Algo2.

A=promoted radix4/packed-vector8 Candidate E, stock cuBLAS TensorOp algorithm.
B=same exact full adapter except N16384,K4096,M64 uses cuBLAS Algo2.
Both paths share the same config, exact BF16 inputs/checkpoint/settings, and
no persistent FP16 weights. Require all 49 decoded YUV420P frames to match
before considering speed; do not claim production qualification from 2+2 trials.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from workspace_video_gate import hashes

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("/cache/ltx25/results")
PY = os.environ.get("LTX25_PYTHON", "/opt/venvs/ltx25-cu128/bin/python")
SHA = os.environ.get("GITHUB_SHA", "local")[:12]
VARIANTS = (
    ("default", "e_default_1024", "quality_default", "off"),
    ("algo2", "e_algo2_1024", "quality_algo2", "algo2_m64"),
    ("algo2", "e_algo2_1024", "quality_algo2", "algo2_m64"),
    ("default", "e_default_1024", "quality_default", "off"),
)


def record_one(label: str, variant: str, suffix: str, tune: str) -> dict:
    env = os.environ.copy()
    env["LTX25_BENCH_VARIANT"] = variant
    # The default variant removes all overrides; the Algo2 variant sets them
    # explicitly inside wgp_baseline.py. No per-call CUDA syncs are added.
    proc = subprocess.run(
        [PY, "benchmarks/ltx25_convrot/wgp_baseline.py"],
        cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=1200, check=False)
    print(proc.stdout[-7000:], flush=True)
    if proc.returncode:
        raise RuntimeError(f"{variant}: generation failed, exit={proc.returncode}")
    report = json.loads((OUT / f"runtime_{suffix}_{SHA}.json").read_text())
    if report.get("status") != "PASS":
        raise RuntimeError(f"{variant}: {report.get('status')}: {report.get('blocker')}")
    videos = report.get("videos") or []
    if len(videos) != 1:
        raise RuntimeError(f"{variant}: expected one new video, got {len(videos)}")
    clip = Path(videos[0]["path"])
    counts = report.get("candidate_counters") or {}
    checks = {
        "mode": "fp32rot", "cast_mode": "vector8", "rotation_mode": "radix4",
        "gemm_tune": tune, "used": 16368, "auto_native_fallback": 0}
    for field, expected in checks.items():
        if counts.get(field) != expected:
            raise RuntimeError(f"{variant}: {field}={counts.get(field)!r}, expected {expected!r}")
    calls = int(counts.get("gemm_algo2_calls", 0))
    if (tune == "off" and calls != 0) or (tune == "algo2_m64" and calls <= 0):
        raise RuntimeError(f"{variant}: wrong Algo2 dispatch count={calls}")
    ws = report.get("workspace_stats") or {}
    if ws.get("persistent_weight_copies") != 0 or ws.get("acquired", 0) <= 0:
        raise RuntimeError(f"{variant}: workspace/weight-residency regression: {ws}")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    frames = hashes(ffmpeg, clip)
    row = {
        "variant": label, "algo2_calls": calls,
        "generation_seconds": report["generation"]["elapsed_seconds"],
        "used": counts["used"],
        "native_fallback": counts.get("auto_native_fallback", 0),
        "range_fallback_rows": counts.get("range_fallback"),
        "frames": len(frames),
        "video": str(clip),
    }
    return row, frames


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"gemm_algo2_abba_{SHA}.json"
    rows = []
    expected_frames = None
    expected_range = None
    try:
        for label, variant, suffix, tune in VARIANTS:
            row, frames = record_one(label, variant, suffix, tune)
            if expected_frames is None:
                expected_frames = frames
            if frames != expected_frames:
                raise AssertionError(
                    f"Decoded-frame mismatch on {label}; do not promote Algo2")
            if expected_range is None:
                expected_range = row["range_fallback_rows"]
            elif row["range_fallback_rows"] != expected_range:
                raise AssertionError(
                    "GPU exceptional-row counts differ across matched generations")
            rows.append(row)
            print("LTX25_GEMM_ABBA_ROUND " + json.dumps(row, sort_keys=True), flush=True)
        a = statistics.median(r["generation_seconds"] for r in rows if r["variant"] == "default")
        b = statistics.median(r["generation_seconds"] for r in rows if r["variant"] == "algo2")
        report = {
            "status": "PASS", "source_sha": os.environ.get("GITHUB_SHA", "local"),
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "test": "V100 49-frame 1024x576 full Candidate E GEMM Algo2 A-B-B-A",
            "default": "radix4+vector8+default_tensor_op",
            "candidate": "radix4+vector8+algo2_m64",
            "frames_pixel_identical": True, "audio_compared": False,
            "all_settings_matched": True, "production_eligible": False,
            "default_median_seconds": round(a, 3),
            "algo2_median_seconds": round(b, 3),
            "algo2_seconds_saved": round(a-b, 3),
            "algo2_percent_wall_reduction": round(100*(a-b)/a, 4),
            "rows": rows,
        }
    except BaseException as exc:
        report = {
            "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"[:850],
            "source_sha": os.environ.get("GITHUB_SHA", "local"),
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "production_eligible": False, "rows_completed": rows,
        }
        path.write_text(json.dumps(report, indent=2) + "\n")
        print("LTX25_GEMM_ABBA_FAILED " + json.dumps(report)[:3000], flush=True)
        raise
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("LTX25_GEMM_ABBA_SUMMARY " + json.dumps({
        k: v for k, v in report.items() if k != "rows"
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
