# Copyright (c) 2026. Experimental LTX-2.5 Candidate E profiling.
"""Sparse same-stream CUDA-event profiling of live LTX-2.5 ConvRot linears.

Enqueues no GPU-to-CPU synchronizations in the optimized hot path. Only a
handful of calls per matrix-shape family are sampled. The CUDA events are
resolved at process exit, after model generation has completed.

Event durations are *stream elapsed time*, not isolated kernel durations:
other concurrent GPU work, CPU enqueue gaps and MMGP overlap may contribute.
Treat the first sample per shape as cold. The report intentionally separates
cold and warm samples and never labels instrumented generation as an
uninstrumented benchmark.
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
SAMPLE_CALLS = (1, 2, 16, 64, 256, 1024, 4096)
MAX_SAMPLES = 7 * len(SAMPLE_CALLS)
MARKS = ("start", "rotation", "alloc", "weight_cast", "prepare",
         "gemm", "epilogue", "output")
_COUNTS = defaultdict(int)
_M_COUNTS = defaultdict(lambda: defaultdict(int))
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
    call = _COUNTS[key]
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
            row.update(status="PASS",
                       gpu_stage_ms=dict(zip(MARKS[1:], gpu)),
                       host_enqueue_us=dict(zip(MARKS[1:], sample.host_us)),
                       total_gpu_event_ms=sum(gpu))
        except Exception as exc:
            row.update(status="ERROR", error=f"{type(exc).__name__}: {exc}"[:400])
        records.append(row)
    stage_names = MARKS[1:]
    by_shape = {}
    for shape in sorted(_COUNTS):
        valid = [r for r in records if r["shape"] == shape and r["status"] == "PASS"
                 and r["call_index"] > 1]
        by_shape[shape] = {
            "eligible_calls": _COUNTS[shape],
            "m_call_counts": {str(m): count for m, count in sorted(_M_COUNTS[shape].items())},
            "sample_count": len(valid),
            "warm_gpu_median_ms": {
                stage: round(statistics.median(r["gpu_stage_ms"][stage] for r in valid), 4)
                for stage in stage_names
            } if valid else {},
        }
    report = {
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_sha": os.environ.get("GITHUB_SHA", "local"),
        "kind": "Live LTX-2.5 1024x576 sparse same-stream Candidate E CUDA event profiler",
        "real_model_activations": True,
        "sampled_only": True,
        "instrumented_generation": True,
        "valid_for_uninstrumented_walltime_claims": False,
        "production_eligible": False,
        "sparse_call_indices": list(SAMPLE_CALLS),
        "eligible_calls": dict(_COUNTS),
        "by_shape": by_shape,
        "samples": records,
        "sample_count": len(records),
        "execution_errors": sum(r["status"] != "PASS" for r in records),
    }
    report["status"] = "PASS" if report["sample_count"] >= 14 and report["execution_errors"] == 0 else "FAIL"
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
