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


def record_qk(block, stream, q, k, q_norm, k_norm):
    """Diagnostic-only pre-RMSNorm reductions; transfer scalars after timing."""
    if _active is None or not _DIAGNOSTIC:
        return
    import torch
    storage = _active.setdefault("_qk_gpu", [])
    for label, x, norm in (("q", q, q_norm), ("k", k, k_norm)):
        bound = torch.finfo(x.dtype).max ** 0.5 if x.dtype == torch.float16 else float("inf")
        # Sampling 16 tokens is enough to flag a discrepancy without keeping
        # complete Q/K tensors alive or adding many GiB of diagnostic state.
        sample = x[:, :min(16, x.shape[1])].detach()
        legacy = sample.clone()
        scale = legacy.pow(2).mean(dim=-1, keepdim=True)
        scale.add_(norm.eps)
        scale.rsqrt_()
        legacy.mul_(scale)
        if hasattr(norm, "weight"):
            legacy.mul_(norm.weight)
        ref = sample.float()
        ref = ref * torch.rsqrt(ref.square().mean(dim=-1, keepdim=True) + norm.eps)
        if hasattr(norm, "weight"):
            ref = ref * norm.weight.float()
        storage.append({
            "block_object_id": id(block), "stream": stream, "tensor": label,
            "dtype": str(x.dtype), "shape": list(x.shape), "safe_square_bound": bound,
            "amax": x.detach().abs().amax(),
            "unsafe_square_elements": (x.detach().abs() > bound).sum(),
            "fp32_reference_sample_max_abs_diff": (legacy.float() - ref).abs().amax(),
        })


def record_attention(q, k, q_lens, k_lens, resolved_mode):
    if _active is None or not _DIAGNOSTIC:
        return
    def copy_lens(lengths):
        if lengths is None:
            return None
        return lengths.detach().clone() if hasattr(lengths, "detach") else list(lengths)
    _active.setdefault("_attention_gpu", []).append({
        "q_shape": list(q.shape), "k_shape": list(k.shape),
        "q_lens": copy_lens(q_lens), "k_lens": copy_lens(k_lens),
        "resolved_mode_at_dispatch": resolved_mode,
    })


def record_vae_config(vae, latent):
    if _active is None or not _DIAGNOSTIC:
        return
    _active["vae_configuration"] = {
        "latent_shape": list(latent.shape),
        "tile_sample_min_size": getattr(vae, "tile_sample_min_size", None),
        "tile_sample_min_tsize": getattr(vae, "tile_sample_min_tsize", None),
        "tile_latent_min_size": getattr(vae, "tile_latent_min_size", None),
        "tile_latent_min_tsize": getattr(vae, "tile_latent_min_tsize", None),
        "overlap_factor": getattr(vae, "tile_overlap_factor", None),
        "spatial_tiling": getattr(vae, "use_spatial_tiling", None),
        "temporal_tiling": getattr(vae, "use_temporal_tiling", None),
        "module_dtype": str(getattr(vae, "_model_dtype", None)),
    }


def record_vae_decoder_call(latent):
    if _active is None or not _DIAGNOSTIC:
        return
    _active.setdefault("vae_decoder_input_shapes", []).append(list(latent.shape))


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
    # Deferred GPU-to-host transfer: reductions are collected during the
    # diagnostic and materialized ONLY after the timed request has ended.
    by_object = {id(b): i for i, b in enumerate(
        getattr(getattr(model, "model", None), "double_blocks", ()))}
    norms = record.pop("_qk_gpu", [])
    record["pre_norm_qk"] = [{
        **{k: v for k, v in entry.items() if k not in (
            "amax", "unsafe_square_elements", "fp32_reference_sample_max_abs_diff")},
        "block": by_object.get(entry["block_object_id"]),
        "amax": entry["amax"].item(),
        "unsafe_square_elements": entry["unsafe_square_elements"].item(),
        "fp32_reference_sample_max_abs_diff":
            entry["fp32_reference_sample_max_abs_diff"].item(),
    } for entry in norms]
    attn = record.pop("_attention_gpu", [])
    for entry in attn:
        for key in ("q_lens", "k_lens"):
            value = entry[key]
            if hasattr(value, "cpu"):
                entry[key] = value.cpu().tolist()
    record["attention_calls"] = attn
    record["vae_decoder_calls"] = len(record.get("vae_decoder_input_shapes", []))
    path = Path(os.environ["HUNYUAN15_P0_TRACE_DIR"]) / "requests.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
