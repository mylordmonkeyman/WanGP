"""Opt-in P0 telemetry for Hunyuan 1.5 native FP16 on V100.

No profiler, per-block sync, or model changes in the normal generation path.
The diagnostic pass optionally synchronizes at phase boundaries; its phase
wall-times must NOT be used as uninstrumented adoption measurements.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

_ENABLED = bool(os.environ.get("HUNYUAN15_P0_TRACE_DIR"))
_DIAGNOSTIC = os.environ.get("HUNYUAN15_P0_DIAGNOSTIC", "0") == "1"
_active = None
_load_count = 0


def _clock():
    import torch
    torch.cuda.synchronize()
    return time.perf_counter()


def _placement(model):
    transformer = getattr(model, "model", None)
    blocks = getattr(transformer, "double_blocks", ())
    results = []
    for i, block in enumerate(blocks):
        params = list(block.parameters(recurse=True))
        devices = sorted({str(p.device) for p in params})
        results.append({"block": i, "devices": devices, "parameters": len(params)})
    return results


def begin(request_no, model_type, prompt, seed):
    """Called immediately before the production CLI generation worker."""
    global _active
    if not _ENABLED:
        return
    if _active is not None:
        raise RuntimeError("Overlapping P0 generations are not allowed")
    if model_type != "hunyuan15_v100_t2v_480_fp16_experiment":
        raise RuntimeError("P0 instrumented an unexpected model: " + str(model_type))
    start = _clock()
    _active = {
        "request_no": request_no,
        "pid": os.getpid(),
        "model_type": model_type,
        "prompt": prompt,
        "seed": seed,
        "start": start,
        "events": [{"name": "request_begin", "at": start}],
        "diagnostic_syncs": _DIAGNOSTIC,
    }


def mark(name, details=None):
    """Mark a stage transition without sync except in the diagnostic run."""
    global _load_count
    if _active is None:
        return
    if name == "model_load_begin":
        _load_count += 1
    if _DIAGNOSTIC:
        now = _clock()
    else:
        now = time.perf_counter()
    event = {"name": name, "at": now}
    if details:
        event["details"] = details
    _active["events"].append(event)


def end(model, offloadobj, error=None):
    """Capture state after a written output, outside the measured interval."""
    global _active
    if _active is None:
        return
    import torch
    finished = _clock()
    record, _active = _active, None
    record["end"] = finished
    record["hot_request_seconds"] = round(finished - record["start"], 6)
    record["error"] = str(error) if error is not None else None
    record["model_loads_cumulative"] = _load_count
    record["model_object_id"] = id(model) if model is not None else None
    record["pipeline_object_id"] = id(getattr(model, "pipeline", None)) if model is not None else None
    record["transformer_object_id"] = id(getattr(model, "model", None)) if model is not None else None
    record["offload_object_id"] = id(offloadobj) if offloadobj is not None else None
    record["placements"] = _placement(model) if model is not None else []
    record["cuda_allocated_bytes"] = torch.cuda.memory_allocated()
    record["cuda_reserved_bytes"] = torch.cuda.memory_reserved()
    record["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    try:
        from mmgp import offload
        record["resolved_attention"] = offload.shared_state.get("_attention")
    except Exception as exc:
        record["resolved_attention_error"] = repr(exc)
    # Different MMGP versions expose different pin-pool/budget interfaces.
    # Do not invent values or claim pin reuse without an observed stable metric.
    for name in ("budgets", "profile_no", "profile", "pinned_memory", "pinned_pool"):
        value = getattr(offloadobj, name, None)
        if value is not None and isinstance(value, (str, int, float, bool, dict)):
            try:
                json.dumps(value)
                record["mmgp_" + name] = value
            except (TypeError, ValueError):
                pass
    record["pin_pool_reuse_verified"] = False
    record["h2d_bytes_verified"] = False
    events = {e["name"]: e["at"] for e in record["events"]}
    phases = {}
    for name, begin_name, end_name in (
        ("preconditioning", "request_begin", "conditioning_begin"),
        ("conditioning", "conditioning_begin", "denoise_begin"),
        ("denoise_inclusive", "denoise_begin", "denoise_end"),
        ("decode_and_copy", "denoise_end", "decode_end"),
        ("output", "decode_end", "request_end"),
    ):
        if begin_name in events and end_name in events:
            phases[name] = round(events[end_name] - events[begin_name], 6)
    record["events"].append({"name": "request_end", "at": finished})
    # Recalculate after appending the terminal event.
    if "decode_end" in events:
        phases["output"] = round(finished - events["decode_end"], 6)
    record["phase_seconds"] = phases
    record["phase_timings_diagnostic_only"] = _DIAGNOSTIC
    path = Path(os.environ["HUNYUAN15_P0_TRACE_DIR"]) / "requests.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
