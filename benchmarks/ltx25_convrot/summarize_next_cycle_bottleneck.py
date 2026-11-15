#!/usr/bin/env python3
"""Research-only, conservative breakdown of real LTX-2.5 V100 GPU traces.

Profiler samples are NOT wall-clock time. CUDA GPU event durations are GPU work
with possible stream overlap. Repeated-forward projections exclude H2D copies,
which can be largely absent once MMGP has established residency.
"""
from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path
import sys


def category(name):
    n = name.lower()
    if "memcpy htod" in n or "memcpy hto" in n:
        return "h2d_memcpy"
    if "memcpy" in n or "memset" in n:
        return "other_gpu_copy_or_set"
    if "cast_qint8_to_fp16" in n:
        return "candidate_e_weight_cast"
    if "transform_radix4" in n or "hadamard" in n:
        return "candidate_e_rotation"
    if "prepare_rows" in n:
        return "candidate_e_activation_prepare"
    if "finish_rows" in n or "epilogue" in n:
        return "candidate_e_epilogue"
    if any(k in n for k in ("fmha", "attention", "sdpa", "flash_attn")):
        return "attention"
    if any(k in n for k in ("gemm", "gemv", "cutlass", "cublas", "magma", "wmma", "mma_")):
        return "gemm_or_gemv"
    return "other_gpu_kernels"


def summarize(data):
    out = {"source": "CUDA GPU kernel events from sampled transformer forward",
           "status": data.get("status"),
           "instrumented_phases_seconds": [
               {"name": p["phase"], "seconds": p["wall_seconds"]}
               for p in data.get("phases", [])],
           "kernel_samples": [],
           "attribution_errors": data.get("errors", []),
           "methodology": [
               "GPU durations are SUMS of CUDA device kernel events, not elapsed time; overlapping streams double count.",
               "Each stage supplies only the second forward, not an average or a first-forward transfer baseline.",
               "Phase durations have extra boundary synchronizations; never subtract GPU sample totals from normal wall time.",
               "For CPU overhead and MMGP wait attribution, obtain an interval-aware GPU/CPU trace or matched offload experiment.",
           ]}
    for sample in data.get("chrome_trace_kernel_attribution", []):
        t = defaultdict(float)
        count = defaultdict(int)
        for row in sample.get("top_gpu_kernels", []):
            key = category(row["kernel"])
            t[key] += float(row["gpu_ms"])
            count[key] += int(row["calls"])
        total = float(sample["gpu_time_ms_sum"])
        coverage = sum(t.values())
        if total <= 0 or abs(coverage - total) / total > 0.005:
            raise ValueError(f"Kernel top-list covers only {coverage:.1f} / {total:.1f} ms")
        out["kernel_samples"].append({
            "stage": sample["phase"],
            "forward_index": sample.get("forward_index"),
            "gpu_kernel_duration_sum_ms": total,
            "gpu_kernel_event_count": sample.get("actual_gpu_kernel_events"),
            "categories": [
                {"name": k, "gpu_event_ms": round(ms, 3),
                 "percent_of_summed_gpu_events": round(100*ms/total, 2),
                 "kernel_calls": count[k]}
                for k, ms in sorted(t.items(), key=lambda kv: -kv[1])],
            "gpu_launch_attribution": sample.get("attribution_gpu_time_ms"),
            "top_gpu_kernels": sample.get("top_gpu_kernels", [])[:15],
        })
    return out


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: summarize_next_cycle_bottleneck.py PROFILE_ROOT [OUTPUT.json]")
    root = Path(sys.argv[1])
    output = Path(sys.argv[2]) if len(sys.argv) >= 3 else root / "bottleneck-summary.json"
    profile_paths = sorted((root / "results").glob("*-pipeline.json"))
    if not profile_paths:
        raise RuntimeError(f"No detailed profiles: {root}/results/*-pipeline.json")
    data = {}
    for path in profile_paths:
        if not path.stem.endswith("-pipeline"):
            continue
        name = path.stem[:-len("-pipeline")]
        data[name] = summarize(json.loads(path.read_text()))
    wall = {}
    for path in (root / "results").glob("*-normal-summary.json"):
        info = json.loads(path.read_text())
        wall[info["mode"]] = info.get("wall_s_not_comparable_if_instrumented")
    info = {
        "status": "TRACE_ATTRIBUTION_NOT_WALL_RECOVERY",
        "profiles": data,
        "unprofiled_wall_s_same_run": wall,
        "prior_bit_identical_stable_main_wall_s": {"t2v": 112.322, "i2v": 113.989},
        "prior_baseline_actions_run": 43746795037,
        "critical_path_recovery_seconds": "NOT YET MEASURED: requires matched whole-process A/B run",
        "repeated_kernel_work_screening": "Use 8 Stage1 and 3 Stage2 forwards only after verifying same per-forward residency/shapes; do not extrapolate H2D copies",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(info, indent=2) + "\n")
    print("NEXT_BOTTLENECK_SUMMARY", json.dumps({
        key: [{"stage": item["stage"], "gpu_ms_sum": item["gpu_kernel_duration_sum_ms"],
               "largest": item["categories"][:5]}
              for item in value["kernel_samples"]]
        for key, value in data.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
