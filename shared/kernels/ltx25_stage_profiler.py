# Copyright (c) 2026. Experimental LTX-2.5 Candidate E profiling.
"""Sparse same-stream CUDA-event profiling of live LTX-2.5 ConvRot linears.

Samples each distinct (N,K,M,dtype) workload independently; counts all calls.
No GPU-to-CPU synchronization occurs in an optimized linear. CUDA events
are resolved only at process exit, after generation completes.

Event durations measure *elapsed CUDA stream time*, not isolated kernel time:
concurrent GPU work, CPU enqueue gaps and MMGP overlap may contribute.
Stage totals extrapolate sampled durations; they are prioritization estimates,
NOT wall-clock percentages or end-to-end speedup predictions.
"""
from __future__ import annotations

import atexit
from collections import defaultdict
import datetime as dt
import json
import os
from pathlib import Path
import statistics
import time

import torch

ENABLED = os.environ.get("WAN2GP_LTX25_PROFILE_STAGES", "") == "1"
PATH = Path(os.environ.get("LTX25_STAGE_PROFILE_PATH", "/cache/ltx25/results/stage_profile.json"))
# Sample cold+warm call positions for each exact workload, not just each (N,K).
SAMPLE_CALLS = (1, 2, 16, 64, 256, 1024, 4096)
MAX_SAMPLES = int(os.environ.get("LTX25_STAGE_PROFILE_MAX_SAMPLES", "320"))
MARKS = ("start", "rotation", "alloc", "weight_cast", "prepare",
         "gemm", "epilogue", "output")
_COUNTS = defaultdict(int)
_M_COUNTS = defaultdict(lambda: defaultdict(int))
_KEY_COUNTS = defaultdict(int)
_SAMPLES = []


class _Sample:
    __slots__ = ("shape", "call", "m", "n", "k", "dtype", "device", "stream",
                 "events", "host_us", "_last_ns", "current_index")

    def __init__(self, n, k, m, dtype, device, stream, call):
        self.shape = f"{n}x{k}"
        self.call = call
        self.m = int(m)
        self.n = int(n)
        self.k = int(k)
        self.dtype = str(dtype)
        self.device = device
        self.stream = stream
        self.events = []
        self.host_us = []
        self._last_ns = time.perf_counter_ns()
        self.current_index = 0
        self.mark("start")

    def mark(self, label):
        if label != MARKS[self.current_index]:
            raise RuntimeError(f"Stage profiler ordering error: expected {MARKS[self.current_index]}, got {label}")
        event = torch.cuda.Event(enable_timing=True)
        event.record(self.stream)
        self.events.append(event)
        now = time.perf_counter_ns()
        if self.current_index:
            self.host_us.append((now - self._last_ns) / 1000.0)
        self._last_ns = now
        self.current_index += 1


def begin(n, k, m, dtype, device, stream):
    if not ENABLED:
        return None
    key = f"{n}x{k}"
    _COUNTS[key] += 1
    # Count every actual M under instrumentation, not just sparse samples.
    # This is Python-only bookkeeping; no GPU synchronization in inference.
    _M_COUNTS[key][int(m)] += 1
    workload = (int(n), int(k), int(m), str(dtype))
    _KEY_COUNTS[workload] += 1
    call = _KEY_COUNTS[workload]
    if call not in SAMPLE_CALLS or len(_SAMPLES) >= MAX_SAMPLES:
        return None
    sample = _Sample(n, k, m, dtype, device, stream, call)
    _SAMPLES.append(sample)
    return sample


def mark(sample, label):
    if sample is not None:
        sample.mark(label)


def _summary():
    if not ENABLED:
        return
    records = []
    for sample in _SAMPLES:
        row = {
            "shape": sample.shape, "call_index": sample.call,
            "M": sample.m, "N": sample.n, "K": sample.k,
            "dtype": sample.dtype,
        }
        try:
            if len(sample.events) != len(MARKS):
                raise ValueError(f"incomplete stage marks {len(sample.events)}/{len(MARKS)}")
            sample.events[-1].synchronize()  # At interpreter exit ONLY.
            gpu = [sample.events[i - 1].elapsed_time(sample.events[i])
                   for i in range(1, len(MARKS))]
            if not all(0 <= ms < 3600000 for ms in gpu):
                raise ValueError(f"invalid CUDA event stage durations: {gpu}")
            row.update(status="PASS",
                       gpu_stage_ms=dict(zip(MARKS[1:], gpu)),
                       host_enqueue_us=dict(zip(MARKS[1:], sample.host_us)),
                       total_gpu_event_ms=sum(gpu))
        except Exception as exc:
            row.update(status="ERROR", error=f"{type(exc).__name__}: {exc}"[:400])
        records.append(row)
    stage_names = MARKS[1:]
    # Legacy per-NxK overview. Its timing medians pool distinct M and must not
    # be used for weighted cost ranking; the by_shape_m view below is exact-M.
    by_shape = {}
    for shape in sorted(_COUNTS):
        valid = [r for r in records if r["shape"] == shape and r["status"] == "PASS"
                 and r["call_index"] > 1]
        by_shape[shape] = {
            "eligible_calls": _COUNTS[shape],
            "m_call_counts": {str(m): count for m, count in sorted(_M_COUNTS[shape].items())},
            "warm_sample_count": len(valid),
            "warm_gpu_median_ms": {
                stage: round(statistics.median(r["gpu_stage_ms"][stage] for r in valid), 5)
                for stage in stage_names
            } if valid else {},
            "warning": "Pooled across M; use by_shape_m for weighted stage estimates",
        }

    by_shape_m = {}
    overall = {stage: 0.0 for stage in stage_names}
    covered_calls = 0
    missing_warm_calls = 0
    missing_warm_workloads = []
    for (n, k, m, dtype), calls in sorted(_KEY_COUNTS.items()):
        key = f"N={n},K={k},M={m},dtype={dtype}"
        own = [row for row in records if (row["N"], row["K"], row["M"], row["dtype"])
               == (n, k, m, dtype) and row["status"] == "PASS"]
        # Remove cold sample (#1) when subsequent samples exist.
        warm = [row for row in own if row["call_index"] > 1]
        entry = {
            "N": n, "K": k, "M": m, "dtype": dtype,
            "eligible_calls": calls,
            "sample_count": len(own),
            "warm_sample_count": len(warm),
            "warm_gpu_median_ms": {},
            "estimated_gpu_stage_ms": {},
            "estimated_gpu_total_ms": None,
            "cost_coverage": "sampled" if warm else "uncovered",
        }
        if warm:
            covered_calls += calls
            entry["warm_gpu_median_ms"] = {
                stage: round(statistics.median(row["gpu_stage_ms"][stage] for row in warm), 6)
                for stage in stage_names
            }
            entry["estimated_gpu_stage_ms"] = {
                stage: round(entry["warm_gpu_median_ms"][stage] * calls, 3)
                for stage in stage_names
            }
            for stage in stage_names:
                overall[stage] += entry["warm_gpu_median_ms"][stage] * calls
            entry["estimated_gpu_total_ms"] = round(
                sum(entry["estimated_gpu_stage_ms"].values()), 3)
        else:
            missing_warm_calls += calls
            missing_warm_workloads.append(key)
        by_shape_m[key] = entry

    total_calls = sum(_COUNTS.values())
    estimated_stage = {stage: round(overall[stage], 3) for stage in stage_names}
    estimated_total = sum(overall.values())
    rank = sorted(
        ({"workload": key, **entry} for key, entry in by_shape_m.items()
         if entry["estimated_gpu_total_ms"] is not None),
        key=lambda row: row["estimated_gpu_total_ms"], reverse=True,
    )
    stage_rank = sorted(
        ({"stage": stage, "estimated_gpu_ms": round(value, 3),
          "estimated_share_of_covered_cuda_stream_time_pct":
              round(100 * value / estimated_total, 2) if estimated_total else 0}
         for stage, value in overall.items()),
        key=lambda row: row["estimated_gpu_ms"], reverse=True,
    )
    errors = sum(row["status"] != "PASS" for row in records)
    report = {
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_sha": os.environ.get("GITHUB_SHA", "local"),
        "kind": "Live LTX-2.5 sparse same-stream Candidate E per-(N,K,M,dtype) CUDA event profiler",
        "real_model_activations": True,
        "sampled_only": True,
        "instrumented_generation": True,
        "valid_for_uninstrumented_walltime_claims": False,
        "production_eligible": False,
        "extrapolation_warning": "Weighted CUDA-stream-event timings are approximate and exclude non-Candidate-E execution. They may include Python enqueue gaps, overlap and contention; NOT GPU kernel-only time, full video time or measured speedup.",
        "sparse_call_indices_per_workload": list(SAMPLE_CALLS),
        "max_samples": MAX_SAMPLES,
        "eligible_calls": dict(_COUNTS),
        "total_eligible_calls": total_calls,
        "covered_calls": covered_calls,
        "coverage_fraction": round(covered_calls / total_calls, 6) if total_calls else 0,
        "uncovered_calls": missing_warm_calls,
        "uncovered_workloads": missing_warm_workloads,
        "by_shape": by_shape,
        "by_shape_m": by_shape_m,
        "estimated_gpu_stage_ms": estimated_stage,
        "estimated_gpu_total_ms": round(estimated_total, 3),
        "estimated_stage_rank": stage_rank,
        "top_workloads_by_estimated_gpu_ms": rank[:15],
        "samples": records,
        "sample_count": len(records),
        "execution_errors": errors,
    }
    report["status"] = "PASS" if len(records) >= 14 and errors == 0 and covered_calls > 0 else "FAIL"
    try:
        PATH.parent.mkdir(parents=True, exist_ok=True)
        PATH.write_text(json.dumps(report, indent=2) + "\n")
        print("[LTX25_STAGE_PROFILE] " + json.dumps({
            k: v for k, v in report.items() if k != "samples"
        }, sort_keys=True)[:6000], flush=True)
    except Exception as exc:
        print(f"[LTX25_STAGE_PROFILE_ERROR] {type(exc).__name__}: {exc}", flush=True)


if ENABLED:
    atexit.register(_summary)
