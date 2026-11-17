#!/usr/bin/env python3
"""Analyze balanced A/B/B/A MMGP 0-versus-1 persistent-process runs.

Four independent sessions, each warm-up + five hot requests.
Measure same-math H.264-decoded frame parity. This is not a substitute for
lossless frame/latent quality fixtures or 97-frame headroom checks.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

from p0_statistics import compare_baseline_candidate, mad

ORDER = [("A1", 0), ("B1", 1), ("B2", 1), ("A2", 0)]


def frame_md5(video: str) -> list[str]:
    p = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
         "-i", video, "-map", "0:v:0", "-an", "-f", "framemd5", "-"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=120, check=False)
    if p.returncode:
        raise RuntimeError(f"ffmpeg framemd5 failed on {video}: {p.stderr[:1000]}")
    return [line.split(",")[-1].strip()
            for line in p.stdout.splitlines() if line and not line.startswith("#")]


def load(root: Path, label: str, profile: int):
    path = root / label / "p1b" / f"profile_{profile}" / "baseline.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text())
    assert data.get("effective_mmgp_profile_requested") == profile
    assert data.get("warmup_generations") == 1
    assert data.get("measured_generations") == 5
    samples = data["request_e2e_samples_seconds"]
    assert len(samples) == 5
    assert len(data["clips"]) == 6
    assert all(c["decode_pass"] and c["decoded_frames"] == 33
               for c in data["clips"])
    snapshots = [v for v in data.get("mmgp_end_snapshots", []) if v]
    return {
        "report_path": str(path),
        "report": data,
        "samples": samples,
        "median_seconds": statistics.median(samples),
        "max_rss_gib": None if not snapshots else None,
        "pin_bytes": [v.get("pinned_total_bytes") for v in snapshots],
        "moves": data.get("mmgp_bulk_move_block_deltas", []),
    }


def analyze(root: Path):
    sessions = {}
    for label, profile in ORDER:
        sessions[label] = load(root, label, profile)
    reports = [s["report"] for s in sessions.values()]
    for field in ("checkpoint_sha256", "converted_lora_sha256",
                  "resolved_attention", "effective_settings"):
        vals = [json.dumps(x[field], sort_keys=True) for x in reports]
        if len(set(vals)) != 1:
            raise ValueError("Fixed-reference mismatch: " + field)
    samples_a = sessions["A1"]["samples"] + sessions["A2"]["samples"]
    samples_b = sessions["B1"]["samples"] + sessions["B2"]["samples"]
    comparison = compare_baseline_candidate(samples_a, samples_b, bootstrap=20000)
    per_session = {}
    for label, _profile in ORDER:
        entry = sessions[label]
        sample_path = entry["report_path"]
        trace = Path(sample_path).parent / "trace" / "requests.jsonl"
        rows = [json.loads(s) for s in trace.read_text().splitlines() if s.strip()]
        if len(rows) != 6:
            raise ValueError("Trace request count mismatch " + label)
        max_rss = max((x.get("process_vmrss_kib", 0) for x in rows[1:]), default=0)
        max_hwm = max((x.get("process_vmhwm_kib", 0) for x in rows[1:]), default=0)
        pin = entry["pin_bytes"]
        per_session[label] = {
            "requested_mmgp_profile": _profile,
            "median_hot_seconds": entry["median_seconds"],
            "hot_samples_seconds": entry["samples"],
            "hot_mad_seconds": mad(entry["samples"]),
            "max_process_rss_gib": max_rss / 2**20,
            "max_process_rss_highwater_gib": max_hwm / 2**20,
            "pinned_total_bytes": pin,
            "bulk_block_move_deltas_not_h2d_bytes": entry["moves"],
            "peak_torch_allocated_gib": entry["report"]["peak_vram_bytes"] / 2**30,
        }
    # Compare all six same-seed clips per process. Reference is A1's
    # first *hot* encoded video; this detects gross visual parity errors and
    # deterministic codec differences but isn't a lossless latent comparison.
    canonical = frame_md5(reports[0]["clips"][1]["clip"])
    if len(canonical) != 33:
        raise ValueError("Reference decoded frame count is not 33")
    mismatches = {}
    for label, _profile in ORDER:
        clips = sessions[label]["report"]["clips"]
        counts = []
        for i in range(1, 6):
            actual = frame_md5(clips[i]["clip"])
            counts.append(None if len(actual) != 33 else sum(a != b for a, b in zip(canonical, actual)))
        mismatches[label] = counts
    exact_parity = all(v == 0 for values in mismatches.values() for v in values)
    return {
        "status": "BALANCED_SCREEN_COMPLETE_DEFAULT_NOT_PROMOTED",
        "run_order": [v[0] for v in ORDER],
        "run_order_profiles": [v[1] for v in ORDER],
        "sessions": per_session,
        "comparison": comparison,
        "decoded_h264_frame_parity": {
            "all_20_hot_videos_match_reference_frames": exact_parity,
            "frame_mismatches_per_hot_clip": mismatches,
            "reference_decoded_frames": len(canonical),
            "limit": "H.264 decoded-frame parity only, not the required lossless frame and latent numerical quality gate",
        },
        "limitations": [
            "H2D bytes and offload transfer waits still require verified measurement",
            "97-frame memory-headroom probe still required",
            "No profile setting promoted as default by this script",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = analyze(args.root)
    path = args.root / "p1b_balanced_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("P1B_BALANCED_REPORT " + json.dumps({
        "medians": {k: v["median_hot_seconds"] for k, v in report["sessions"].items()},
        "combined_change_percent": report["comparison"]["improvement_percent"],
        "statistical_gate": report["comparison"]["noise_gate"],
        "decoded_h264_frame_parity": report["decoded_h264_frame_parity"]["all_20_hot_videos_match_reference_frames"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
