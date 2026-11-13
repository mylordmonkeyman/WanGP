#!/usr/bin/env python3
"""Decoded-frame diagnostic and strict equality gate for LTX-2.5 V100.

All 49 frames are decoded as yuv420p using the same ffmpeg command.
Only historical torch/exact variants require pixel equality. The user-approved
radix4 Hadamard accumulation legitimately changes decoded pixels; for those
variants, record the mismatch count without treating it as a generation error.
A passing non-strict gate means valid/complete decoding, NOT verified quality.
This script does not create video, download models, or use CUDA.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

BASE = Path("/cache/ltx25/results")
REFERENCE_REPORT = BASE / "runtime_quality_e_a566cf56eae8.json"


def video_from(report_path):
    report = json.loads(report_path.read_text())
    if report.get("status") != "PASS" or len(report.get("videos", [])) != 1:
        raise RuntimeError(f"Incomplete quality report or video list: {report_path}")
    video = Path(report["videos"][0]["path"])
    if not video.is_file():
        raise FileNotFoundError(video)
    return video, report


def hashes(ffmpeg, video):
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
           "-i", str(video), "-map", "0:v:0", "-pix_fmt", "yuv420p",
           "-f", "framemd5", "-"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=180)
    frames = [line.strip() for line in result.stdout.splitlines()
              if line.strip() and not line.startswith("#")]
    if len(frames) != 49:
        raise RuntimeError(f"{video}: expected 49 decoded frames, got {len(frames)}")
    return frames


def main():
    sha = os.environ["GITHUB_SHA"][:12]
    variant = os.environ.get("LTX25_BENCH_VARIANT", "e_workspace_1024")
    variants = {
        "e_workspace_1024": ("quality_workspace", "workspace_video_gate"),
        "e_cast4_1024": ("quality_cast4", "cast4_video_gate"),
        "e_cast8_1024": ("quality_cast8", "cast8_video_gate"),
        "e_combo_1024": ("quality_combo", "combo_video_gate"),
        "e_default_1024": ("quality_default", "default_video_gate"),
        "e_hadamard_1024": ("quality_hadamard", "hadamard_video_gate"),
        "e_flat_mm_1024": ("quality_flat_mm", "flat_mm_video_gate"),
        "e_algo2_1024": ("quality_algo2", "algo2_video_gate"),
        "e_debug_fp32rot_samecfg_1024": ("quality_default_explicit", "debug_explicit_video_gate"),
    }
    if variant not in variants:
        raise ValueError(f"Unsupported decoded video gate variant {variant!r}")
    suffix, gate_key = variants[variant]
    strict_equality = variant not in ("e_default_1024", "e_hadamard_1024")
    current = BASE / f"runtime_{suffix}_{sha}.json"
    video, report = video_from(current)
    gate = {"status": "ERROR", "reference_report": str(REFERENCE_REPORT),
            "comparison": "49 decoded yuv420p frames, reference equality diagnostic",
            "strict_equality_required": strict_equality,
            "quality_assessment": "visual review required for nonidentical frames",
            "new_video": str(video)}
    try:
        reference, _ = video_from(REFERENCE_REPORT)
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        a, b = hashes(ffmpeg, reference), hashes(ffmpeg, video)
        mismatches = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        gate.update(status="PASS" if (not strict_equality or not mismatches) else "FAIL",
                    total_frames=49, mismatched_frames=len(mismatches),
                    first_mismatch=mismatches[:10],
                    decoded_pixels_identical=not mismatches)
    except Exception as exc:
        gate.update(status="ERROR", error=f"{type(exc).__name__}: {exc}"[:500])
    report[gate_key] = gate
    if gate["status"] != "PASS":
        report["status"] = "DECODED_VIDEO_MISMATCH"
        report["blocker"] = "Video decode failed or strict decoded equality test did not pass"
    current.write_text(json.dumps(report, indent=2) + "\n")
    print("LTX25_DECODED_VIDEO_GATE " + json.dumps({
        "variant": variant, **gate}, sort_keys=True), flush=True)
    if gate["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
