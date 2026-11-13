#!/usr/bin/env python3
"""Estimate V100 *isolated-GEMM* savings by measured live (N,K,M) call counts.

The model stage profile must contain full-call M frequencies, not sparse
sample counts. Join those with the prior isolated exact-output cublasGemmEx
selector sweep. This is an optimistic prioritization estimate, NOT evidence
of a full-video speedup, error qualification, or new default eligibility.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def estimate(profile: dict, sweep: dict, wall_seconds: float | None) -> dict:
    by = profile.get("by_shape") or {}
    shapes = sweep.get("shapes") or []
    sweep_lookup = {
        (int(x["N"]), int(x["K"]), int(x["M"])): x
        for x in shapes
    }
    exact = []
    missing = []
    weighted_saving_ms = 0.0
    for shape, record in sorted(by.items()):
        try:
            n, k = map(int, shape.split("x"))
        except ValueError as exc:
            raise ValueError(f"Invalid shape key: {shape}") from exc
        counts = record.get("m_call_counts")
        if not isinstance(counts, dict):
            raise ValueError(
                f"{shape}: m_call_counts absent. Re-run the 2026-11-13 "
                "promoted-baseline sparse profiler; do not extrapolate "
                "from its seven M samples.")
        if sum(map(int, counts.values())) != record.get("eligible_calls"):
            raise ValueError(f"{shape}: M-frequency sum does not match eligible calls")
        for m_text, calls in counts.items():
            m, calls = int(m_text), int(calls)
            case = sweep_lookup.get((n, k, m))
            if case is None:
                missing.append({"N": n, "K": k, "M": m, "calls": calls})
                continue
            baseline = float(case["baseline_median_ms"])
            eligible = [
                c for c in case["candidates"]
                if c.get("status") == "PASS"
                and c.get("bit_exact_fp32") is True
                and isinstance(c.get("median_ms"), (int, float))
            ]
            if not eligible:
                missing.append({"N": n, "K": k, "M": m, "calls": calls,
                                "reason": "No bit-exact candidate"})
                continue
            best = min(eligible, key=lambda x: x["median_ms"])
            saved = max(0.0, baseline - float(best["median_ms"])) * calls
            weighted_saving_ms += saved
            exact.append({
                "N": n, "K": k, "M": m, "calls": calls,
                "default_gemm_ms": baseline,
                "best_exact_selector": best["selector"],
                "best_exact_gemm_ms": best["median_ms"],
                "potential_isolated_gemm_seconds_saved": round(saved/1000.0, 5),
            })
    exact.sort(key=lambda row: row["potential_isolated_gemm_seconds_saved"], reverse=True)
    ans = {
        "kind": "experimental exact-selector V100 GEMM headroom estimate",
        "profile_source_sha": profile.get("source_sha"),
        "sweep_source_sha": sweep.get("source_sha"),
        "full_m_counts_used": True,
        "result_isolated_gemm_only": True,
        "not_full_video_walltime_projection": True,
        "no_live_timing_proof": True,
        "production_eligible": False,
        "exact_shapes": exact,
        "uncovered_shapes": missing,
        "total_optimistic_isolated_gemm_seconds_saved": round(weighted_saving_ms/1000.0, 5),
    }
    if wall_seconds is not None:
        ans["reference_video_wall_seconds"] = wall_seconds
        ans["optimistic_pct_of_wall"] = round(
            (weighted_saving_ms / 1000.0) / wall_seconds * 100.0, 5)
    return ans


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("profile_json", type=Path)
    p.add_argument("sweep_json", type=Path, nargs="+", help="One or more complementary GEMM selector sweeps")
    p.add_argument("--wall-seconds", type=float)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.wall_seconds is not None and args.wall_seconds <= 0:
        p.error("--wall-seconds must be positive")
    profile = json.loads(args.profile_json.read_text())
    sweeps = [json.loads(src.read_text()) for src in args.sweep_json]
    sweep = {
        "shapes": [shape for run in sweeps for shape in run["shapes"]],
        "source_sha": ",".join(str(run.get("source_sha", "unknown"))[:12] for run in sweeps),
    }
    identifiers = [(s["N"],s["K"],s["M"]) for s in sweep["shapes"]]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Input sweep files contain duplicate N,K,M families")
    ans = estimate(profile.get("stage_profile", profile), sweep, args.wall_seconds)
    doc = json.dumps(ans, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(doc)
    print(doc, end="")


if __name__ == "__main__":
    main()
