#!/usr/bin/env python3
"""Summarize live Candidate E CUDA-event timing by exact N,K,M and stage.

Only instrumented Candidate E calls are covered. Extrapolated CUDA stream-event
time is NOT end-to-end video time and must not be reported as a speedup.
No GPU, NumPy, torch, or model dependency: can be run on any Python 3.10+ host.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


STAGES = ("rotation", "alloc", "weight_cast", "prepare", "gemm", "epilogue", "output")


def render(profile: dict, output_dir: Path) -> str:
    if profile.get("status") != "PASS":
        raise ValueError(f"Stage profiler reported {profile.get('status')!r}")
    if not profile.get("instrumented_generation") or not profile.get("real_model_activations"):
        raise ValueError("This is not a live instrumented Candidate E profile")
    workloads = profile.get("by_shape_m")
    if not isinstance(workloads, dict) or not workloads:
        raise ValueError("Missing per-(N,K,M,dtype) instrumentation; update the Git branch")
    calls = int(profile.get("total_eligible_calls", 0))
    covered = int(profile.get("covered_calls", 0))
    if calls <= 0 or covered <= 0:
        raise ValueError("No eligible measured Candidate E calls")
    if covered > calls:
        raise ValueError("Profiler coverage exceeds eligible calls")
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / "profile-shape-costs.csv"
    columns = ("N", "K", "M", "dtype", "eligible_calls", "sample_count",
               "warm_sample_count", "cost_coverage", "estimated_gpu_total_ms") + tuple(
                   "median_" + stage + "_ms" for stage in STAGES
               ) + tuple("estimated_" + stage + "_ms" for stage in STAGES)
    ranked = sorted(workloads.values(),
                    key=lambda row: row.get("estimated_gpu_total_ms") or 0, reverse=True)
    with outfile.open("w", encoding="utf-8", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=columns)
        writer.writeheader()
        for row in ranked:
            median = row.get("warm_gpu_median_ms") or {}
            totals = row.get("estimated_gpu_stage_ms") or {}
            record = {k: row.get(k) for k in columns if not k.startswith(("median_", "estimated_"))}
            record["estimated_gpu_total_ms"] = row.get("estimated_gpu_total_ms")
            record.update({"median_" + stage + "_ms": median.get(stage) for stage in STAGES})
            record.update({"estimated_" + stage + "_ms": totals.get(stage) for stage in STAGES})
            writer.writerow(record)
    estimates = profile.get("estimated_gpu_stage_ms") or {}
    total = float(profile.get("estimated_gpu_total_ms", 0))
    result = [
        "LTX-2.5 241-frame I2V / Candidate E stage instrumentation",
        "==========================================================",
        f"Eligible Candidate E calls: {calls:,}",
        f"Covered by warm samples:  {covered:,} ({100 * covered / calls:.2f}%)",
        f"Sampled CUDA-event ranges: {profile.get('sample_count', 0):,}",
        f"Uncovered workloads: {len(profile.get('uncovered_workloads', [])):,}",
        "",
        "Extrapolated CUDA stream-event time ONLY (not model/video wall time):",
    ]
    stages_ranked = sorted(STAGES, key=lambda stage: estimates.get(stage, 0), reverse=True)
    for stage in stages_ranked:
        milliseconds = float(estimates.get(stage, 0))
        result.append(f"  {stage:<14} {milliseconds / 1000:>10.2f} s"
                      f"  {100 * milliseconds / total if total else 0:>5.1f}% of covered Candidate E events")
    result.extend((
        "",
        "Highest estimated cost by exact matrix shape (N, K, M):",
        "     N      K      M       calls  estimated_s  cast_s    gemm_s   coverage",
    ))
    for row in ranked[:20]:
        cost = row.get("estimated_gpu_stage_ms") or {}
        result.append(
            f"{int(row['N']):>6} {int(row['K']):>6} {int(row['M']):>6}"
            f" {int(row['eligible_calls']):>10}"
            f" {(row.get('estimated_gpu_total_ms') or 0) / 1000:>11.2f}"
            f" {cost.get('weight_cast', 0) / 1000:>8.2f}"
            f" {cost.get('gemm', 0) / 1000:>9.2f}"
            f"   {row.get('cost_coverage', 'unknown')}"
        )
    result.extend((
        "",
        "Important: CUDA events measure elapsed stream intervals, not isolated GPU",
        "kernel execution. Gaps from CPU dispatch, concurrent work, and MMGP",
        "can be included. Weighting sparse samples by call count is an estimate.",
        "This report excludes native-fallback operations, attention, VAE, loading,",
        "video encoding and all other non-Candidate-E model costs.",
        "Do NOT compare the profiled generation wall time to an unprofiled run.",
        "",
        f"CSV: {outfile}",
    ))
    if covered < calls:
        result.append("WARNING: Some (N,K,M,dtype) workload costs have no warm samples.")
    msg = "\n".join(result) + "\n"
    (output_dir / "profile-summary.txt").write_text(msg, encoding="utf-8")
    return msg


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: summarize_stage_profile.py stage_profile.json OUTPUT_DIR", file=sys.stderr)
        return 2
    try:
        path = Path(sys.argv[1])
        profile = json.loads(path.read_text(encoding="utf-8"))
        print(render(profile, Path(sys.argv[2])), end="")
    except (ValueError, OSError, TypeError, KeyError, json.JSONDecodeError) as error:
        print(f"PROFILE SUMMARY ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
