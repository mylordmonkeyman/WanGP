"""Temporary V100 MMGP profiling companion, never imported by production sources.

CUDA timing events are recorded without synchronizing during inference. Event
intervals on distinct streams may overlap: do not add them into recoverable wall.
The compute-stream event pair around a stream wait measures a *potential*
copy-dependency stall, not a CPU host block or guaranteed speedup.
"""
from __future__ import annotations
import atexit
from collections import defaultdict
import json
import os
from pathlib import Path
import threading
import time

import torch

PATH = Path(os.environ.get("LTX25_MMGP_PROBE_JSON", "/tmp/ltx25-mmgp-probe.json"))
_lock = threading.Lock()
_copies = []
_waits = []
_forwards = []
_plans = []
_block_loads = []
_errors = []
_clock = time.perf_counter_ns
_forward_indices = defaultdict(int)


def _phase():
    try:
        from shared.kernels import ltx25_cublas_sm70
        return getattr(ltx25_cublas_sm70, "_CURRENT_PHASE", "unknown")
    except Exception:
        return "unknown"


def _event(stream):
    event = torch.cuda.Event(enable_timing=True)
    event.record(stream)
    return event


def copy_start(stream):
    if torch.cuda.is_current_stream_capturing():
        return None
    return _event(stream)


def copy_stop(stream, start, entry_name, nbytes):
    if start is not None:
        end = _event(stream)
        with _lock:
            _copies.append((entry_name, int(nbytes), _phase(), start, end))


def consumer_wait(consumer, transfer, entry_name, future_resolver):
    host_before = _clock()
    cuda_event = transfer if isinstance(transfer, torch.cuda.Event) else future_resolver(transfer)
    host_future_ms = (_clock() - host_before) / 1e6
    if torch.cuda.is_current_stream_capturing():
        consumer.wait_event(cuda_event)
        before = after = None
    else:
        before = _event(consumer)
        consumer.wait_event(cuda_event)
        after = _event(consumer)
    with _lock:
        _waits.append((entry_name, _phase(), host_future_ms, before, after))


def block_load_start():
    return _clock()


def block_load_stop(t0, model_id, block_name, preload):
    with _lock:
        _block_loads.append((str(model_id), str(block_name), bool(preload),
                             _phase(), (_clock() - t0) / 1e6))


def record_plan(model_id, count, nbytes, max_block, window, total_blocks):
    with _lock:
        _plans.append({"model_id": str(model_id), "preloaded_blocks": int(count),
                       "planned_preloaded_bytes": int(nbytes),
                       "largest_shuttle_bytes": int(max_block),
                       "planned_prefetch_window": int(window) if window is not None else None,
                       "registered_entries": int(total_blocks)})


def forward_call(fn, *args, **kwargs):
    phase = _phase()
    with _lock:
        _forward_indices[phase] += 1
        nth = _forward_indices[phase]
    start_host = _clock()
    consumer = torch.cuda.current_stream()
    before = _event(consumer) if not torch.cuda.is_current_stream_capturing() else None
    try:
        return fn(*args, **kwargs)
    finally:
        after = _event(consumer) if before is not None else None
        with _lock:
            _forwards.append((phase, nth, (_clock() - start_host)/1e6, before, after))


def _gpu_ms(a, b):
    if a is None or b is None:
        return None
    try:
        return round(a.elapsed_time(b), 6)
    except Exception as err:
        _errors.append(type(err).__name__ + ": " + str(err))
        return None


def _report():
    try:
        if _copies or _waits or _forwards:
            torch.cuda.synchronize()
        buckets = defaultdict(lambda: {
            "copy_count": 0, "copy_bytes": 0, "copy_event_gpu_ms_sum": 0.0,
            "wait_count": 0, "consumer_wait_gpu_ms_sum": 0.0,
            "wait_for_prefetch_future_host_ms_sum": 0.0,
            "block_load_count": 0, "block_load_host_ms_sum": 0.0})
        copy_entries = defaultdict(lambda: {"calls": 0, "bytes": 0, "gpu_ms_sum": 0.0})
        for entry, nbytes, phase, a, b in _copies:
            elapsed = _gpu_ms(a, b)
            x = buckets[phase]
            x["copy_count"] += 1
            x["copy_bytes"] += nbytes
            x["copy_event_gpu_ms_sum"] += elapsed or 0
            y = copy_entries[entry]
            y["calls"] += 1
            y["bytes"] += nbytes
            y["gpu_ms_sum"] += elapsed or 0
        waits = []
        for entry, phase, host_ms, a, b in _waits:
            gpu_ms = _gpu_ms(a, b)
            x = buckets[phase]
            x["wait_count"] += 1
            x["consumer_wait_gpu_ms_sum"] += gpu_ms or 0
            x["wait_for_prefetch_future_host_ms_sum"] += host_ms
            waits.append({"entry":entry,"phase":phase,"future_host_ms":round(host_ms,6),
                          "consumer_wait_gpu_ms":gpu_ms})
        for model_id, block_name, preload, phase, ms in _block_loads:
            x = buckets[phase]
            x["block_load_count"] += 1
            x["block_load_host_ms_sum"] += ms
        forwards = [{"phase":phase, "number":nth, "host_call_ms":round(ms,5),
                     "gpu_span_ms":_gpu_ms(a,b)}
                    for phase,nth,ms,a,b in _forwards]
        result = {
            "status": "PASS" if not _errors else "INCOMPLETE_EVENT_TIMINGS",
            "measurement_scope": "opt-in, temporary mmgp/offload.py and distilled.py checkout overlay",
            "timings_are": "GPU event intervals, plus host enqueue/future waits; no timing synchronizations during inference",
            "caveats": [
                "GPU copy event sums and CPU host durations are NOT recoverable end-to-end wall time.",
                "Copy streams, prefetches and transformer execution may overlap.",
                "Consumer stream wait event interval is a dependency-stall proxy, not necessarily total MMGP overhead.",
                "Event recorder overhead perturbs instrumented full-process timing; compare to separately measured normal baseline.",
                "Instrumentation captures MMGP _move_block fused-buffer transfers; fallback per-tensor copies can be uncounted."
            ],
            "by_phase": dict(buckets),
            "preload_plans": _plans,
            "transformer_forwards": forwards,
            "longest_consumer_gpu_waits": sorted(
                waits,key=lambda x: x["consumer_wait_gpu_ms"] or 0,reverse=True)[:25],
            "largest_transfer_entries": sorted(
                [{"entry":k,**v} for k,v in copy_entries.items()],
                key=lambda x:x["bytes"],reverse=True)[:25],
            "errors": _errors,
        }
        PATH.parent.mkdir(parents=True, exist_ok=True)
        PATH.write_text(json.dumps(result,indent=2) + "\n")
        print("LTX25_MMGP_PROBE_RESULT "+json.dumps({
            "status": result["status"], "by_phase": result["by_phase"],
            "preload_plan_count":len(_plans),
            "transformer_forward_count":len(forwards),
            "errors": _errors},sort_keys=True),flush=True)
    except Exception as err:
        PATH.parent.mkdir(parents=True,exist_ok=True)
        PATH.write_text(json.dumps({"status":"ERROR","error":repr(err)})+"\n")
        print("LTX25_MMGP_PROBE_ERROR",repr(err),flush=True)


atexit.register(_report)
