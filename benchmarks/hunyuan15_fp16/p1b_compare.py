#!/usr/bin/env python3
"""Summarize independently executed P1B MMGP profile probes.

An exploratory 0/1/3 order is NOT a qualified A/B/B/A adoption trial.
Unmeasured H2D bytes or kernel timings are always reported as unverified.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from p0_statistics import compare_baseline_candidate, mad


def summarize(root: Path):
    profiles = {}
    for profile in (0, 1, 3):
        path = root / "p1b" / ("profile_" + str(profile)) / "baseline.json"
        if not path.is_file():
            profiles[str(profile)] = {"status": "failed_or_not_run", "path": str(path)}
            continue
        info = json.loads(path.read_text())
        samples = info["request_e2e_samples_seconds"]
        if len(samples) < 5 or info["measured_generations"] != 5:
            raise ValueError("Insufficient hot samples in " + str(path))
        if info.get("effective_mmgp_profile_requested") != profile:
            raise ValueError("MMGP requested profile mismatch: " + str(path))
        phase = info["phase_samples"]
        def med_phase(key):
            return statistics.median(p[key] for p in phase)
        snap = info.get("mmgp_end_snapshots") or []
        blocks = [v for v in info.get("mmgp_bulk_move_block_deltas", []) if v is not None]
        r = {
            "status": "exploratory_measurement_complete",
            "profile": profile,
            "hot_samples_seconds": samples,
            "hot_median_seconds": statistics.median(samples),
            "hot_mad_seconds": mad(samples),
            "phase_medians_seconds": {
                "denoise_inclusive_provisional": med_phase("denoise_inclusive"),
                "decode_and_copy": med_phase("decode_and_copy"),
                "preconditioning": med_phase("preconditioning"),
                "conditioning": med_phase("conditioning"),
                "output": med_phase("output"),
            },
            "peak_torch_allocated_gib": info["peak_vram_bytes"] / 2**30,
            "bulk_block_moves_per_request": blocks,
            "pinned_pool_snapshots": snap,
            "checkpoint_sha256": info["checkpoint_sha256"],
            "converted_lora_sha256": info["converted_lora_sha256"],
            "resolved_attention": info["resolved_attention"],
            "frames_verified": len(info["clips"]),
        }
        profiles[str(profile)] = r
    base = profiles["0"]
    comparisons = {}
    if base["status"] == "exploratory_measurement_complete":
        for profile in (1, 3):
            candidate = profiles[str(profile)]
            if candidate["status"] != "exploratory_measurement_complete":
                continue
            for key in ("checkpoint_sha256", "converted_lora_sha256", "resolved_attention"):
                if candidate[key] != base[key]:
                    raise ValueError("Profile mismatch for fixed reference: " + key)
            comparisons[str(profile)] = compare_baseline_candidate(
                base["hot_samples_seconds"], candidate["hot_samples_seconds"])
    return {
        "status": "EXPLORATORY_P1B_NOT_ELIGIBLE_FOR_DEFAULT_PROMOTION",
        "reason": "Initial three-profile screen is not a balanced A/B/B/A sweep; H2D bytes and model resident bytes still need attribution.",
        "profiles": profiles,
        "comparisons_to_profile_0": comparisons,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    out = args.root / "p1b_comparison.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print("P1B_EXPLORATORY_REPORT " + json.dumps({
        "profiles": {k: v.get("hot_median_seconds") for k, v in result["profiles"].items()},
        "comparisons": {k: v["noise_gate"] for k, v in result["comparisons_to_profile_0"].items()},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
