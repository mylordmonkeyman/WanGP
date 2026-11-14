#!/usr/bin/env python3
"""Matched V100 Candidate E full-generation A-B-B-A input/output fusion gate.

A: validated Candidate E radix4+vector8, FP32 staging and torch output cast.
B: identical path except direct BF16/FP16 radix4 input and fused output epilogue.
Compare 49 decoded video frames AND decoded PCM audio across all 4 full runs.
No persistent second weight set. Full quality/long-I2V validation remains separate.
"""
from __future__ import annotations

import datetime as dt
import hashlib
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
REPORT = OUT / f"e_fusions_abba_{SHA}.json"
VARIANTS = (
    ("baseline", "staged", "torch"),
    ("fused", "direct", "fused"),
    ("fused", "direct", "fused"),
    ("baseline", "staged", "torch"),
)


def ffmpeg_path():
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def decoded_audio_hash(ffmpeg: str, clip: Path) -> dict:
    """Hash decoded signed PCM bytes, not potentially variable MP4 metadata."""
    result = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(clip), "-map", "0:a:0",
         "-vn", "-f", "s16le", "-acodec", "pcm_s16le", "-"],
        capture_output=True, timeout=180, check=False)
    if result.returncode:
        raise RuntimeError(
            f"No decodable audio in {clip.name}: {result.stderr[-400:].decode(errors='replace')}")
    if not result.stdout:
        raise RuntimeError(f"Decoded audio is empty: {clip.name}")
    return {"sha256": hashlib.sha256(result.stdout).hexdigest(),
            "pcm_bytes": len(result.stdout)}


def one(label: str, rotation: str, output: str, ffmpeg: str) -> tuple[dict, list[str]]:
    env = os.environ.copy()
    env["LTX25_BENCH_VARIANT"] = "e_hadamard_1024"
    env["WAN2GP_LTX25_CUBLAS_E_MODE"] = "fp32rot"
    env["WAN2GP_LTX25_HADAMARD"] = "radix4"
    env["WAN2GP_LTX25_E_CAST"] = "vector8"
    env["WAN2GP_LTX25_GEMM_TUNE"] = "off"
    env["WAN2GP_LTX25_ROT_INPUT"] = rotation
    env["WAN2GP_LTX25_E_OUTPUT"] = output
    env["WAN2GP_LTX25_E_WORKSPACE"] = "1"
    proc = subprocess.run(
        [PY, "benchmarks/ltx25_convrot/wgp_baseline.py"],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=1500, check=False)
    print(proc.stdout[-5000:], flush=True)
    if proc.returncode:
        raise RuntimeError(f"{label}: generation process exit={proc.returncode}")
    r = json.loads((OUT / f"runtime_quality_hadamard_{SHA}.json").read_text())
    if r.get("status") != "PASS":
        raise RuntimeError(f"{label}: {r.get('status')} {r.get('blocker') or r.get('error')}")
    clips = r.get("videos") or []
    if len(clips) != 1:
        raise RuntimeError(f"{label}: expected exactly one fresh video, got {len(clips)}")
    clip = Path(clips[0]["path"])
    counters = r.get("candidate_counters") or {}
    expected = {"mode": "fp32rot", "cast_mode": "vector8", "rotation_mode": "radix4",
                "rotation_input": rotation, "output_mode": output, "gemm_tune": "off",
                "used": 16368, "auto_native_fallback": 0}
    for key, value in expected.items():
        if counters.get(key) != value:
            raise AssertionError(f"{label}: {key}={counters.get(key)!r}, expected {value!r}")
    workspace = r.get("workspace_stats") or {}
    if workspace.get("persistent_weight_copies") != 0 or workspace.get("acquired", 0) <= 0:
        raise AssertionError(f"{label}: workspace regression {workspace}")
    frames = hashes(ffmpeg, clip)
    if len(frames) != 49:
        raise AssertionError(f"{label}: expected 49 decoded frames, got {len(frames)}")
    audio = decoded_audio_hash(ffmpeg, clip)
    row = {
        "label": label, "rotation_input": rotation, "output_mode": output,
        "seconds": float(r["generation"]["elapsed_seconds"]),
        "clip": str(clip), "decoded_frames": len(frames), "audio": audio,
        "candidate_calls": counters["used"],
        "fallback_rows": counters.get("range_fallback"),
        "workspace_leases": workspace.get("acquired"),
    }
    return row, frames


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    expected_frames = expected_audio = expected_fallback = None
    try:
        ffmpeg = ffmpeg_path()
        for label, rotation, output in VARIANTS:
            row, frames = one(label, rotation, output, ffmpeg)
            if expected_frames is None:
                expected_frames = frames
                expected_audio = row["audio"]
                expected_fallback = row["fallback_rows"]
            else:
                if expected_frames != frames:
                    raise AssertionError(f"Decoded video frames differ: {label} (no promotion)")
                if expected_audio != row["audio"]:
                    raise AssertionError(f"Decoded audio differs: {label} (no promotion)")
                if expected_fallback != row["fallback_rows"]:
                    raise AssertionError(f"FP32 exceptional-row counts differ: {label}")
            rows.append(row)
            print("LTX25_E_FUSIONS_ABBA_ROUND "+json.dumps(row, sort_keys=True), flush=True)
        baseline = statistics.median(x["seconds"] for x in rows if x["label"] == "baseline")
        fused = statistics.median(x["seconds"] for x in rows if x["label"] == "fused")
        report = {
            "status": "PASS",
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "source_sha": os.environ.get("GITHUB_SHA", "local"),
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "test": "V100 49-frame LTX-2.5 Candidate E radix4+vector8 fusion ABBA",
            "all_settings_matched": True,
            "video_frames_bit_identical": True,
            "decoded_audio_pcm_bit_identical": True,
            "persistent_weight_copies": 0,
            "baseline_median_seconds": round(baseline, 3),
            "fused_median_seconds": round(fused, 3),
            "seconds_saved": round(baseline-fused, 3),
            "wall_reduction_percent": round(100 * (baseline-fused) / baseline, 3),
            "production_eligible": False,
            "rows": rows,
        }
    except BaseException as exc:
        report = {
            "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"[:900],
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "source_sha": os.environ.get("GITHUB_SHA", "local"),
            "production_eligible": False, "rows_completed": rows,
        }
        REPORT.write_text(json.dumps(report, indent=2) + "\n")
        print("LTX25_E_FUSIONS_ABBA_FAILED "+json.dumps(report)[:3000], flush=True)
        raise
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print("LTX25_E_FUSIONS_ABBA_SUMMARY "+
          json.dumps({k: v for k, v in report.items() if k != "rows"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
