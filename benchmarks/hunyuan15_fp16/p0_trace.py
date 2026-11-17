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



def profile_first_hot_attention(fn, *args, **kwargs):
    """Opt-in CUPTI/PyTorch profiler for exactly one hot diagnostic attention call.

    This is profiler-only, never an adoption wall-clock measurement. Preserve
    the baseline attention function unchanged for all unprofiled calls.
    """
    if (_active is None or not _DIAGNOSTIC or
            _active.get("request_no") != 2 or
            _active.get("_attention_profile_claimed")):
        return fn(*args, **kwargs)
    _active["_attention_profile_claimed"] = True
    import torch
    try:
        from torch.profiler import profile, ProfilerActivity
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                       record_shapes=False, profile_memory=False)
        prof.__enter__()
    except Exception as exc:
        _active["attention_kernel_profile_error"] = "profiler_unavailable: " + repr(exc)
        return fn(*args, **kwargs)
    try:
        result = fn(*args, **kwargs)
    finally:
        import sys
        prof.__exit__(*sys.exc_info())
    try:
        # Only a *single* attention call is profiled. No trace of the entire
        # four-step generation is retained in memory.
        events = list(prof.events())
        gpu = [e for e in events if "cuda" in str(
            getattr(e, "device_type", "")).lower()]
        _active["attention_kernel_profile"] = {
            "request_no": _active["request_no"],
            "cuda_kernel_names": list(dict.fromkeys(e.name for e in gpu))[:40],
            "cuda_event_count": len(gpu),
            "attention_ops": [{
                "name": str(getattr(e, "key", getattr(e, "name", "unknown"))), "count": int(e.count),
                "self_cuda_time_us": float(getattr(e, "self_device_time_total", 0)),
            } for e in prof.key_averages()
                if any(k in e.key.lower() for k in (
                    "attention", "flash", "sdpa", "scaled_dot", "fmha"))][:40],
        }
    except Exception as exc:
        _active["attention_kernel_profile_error"] = "profiler_report: " + repr(exc)
    return result


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



def _mmgp_snapshot(manager):
    """Read-only offload counters: never infer raw PCIe bytes from block count."""
    if manager is None:
        return None
    import importlib
    mm = importlib.import_module("mmgp.offload")
    pool = getattr(mm, "_pool", None)
    pool_stats = pool.stats() if pool is not None else {}
    pinned_mod = getattr(mm, "pinned", None)
    preload = getattr(manager, "preloaded_blocks_per_model", {})
    blocks = getattr(manager, "blocks_of_modules_sizes", {})
    return {
        "offload_object_id": id(manager),
        "bulk_move_blocks_cumulative": getattr(manager, "transfers", None),
        "pinned_total_bytes": getattr(pinned_mod, "total_bytes", None),
        "pinned_pool_epoch": getattr(pool, "epoch", None),
        "pinned_pool_version": getattr(pool, "version", None),
        "pinned_pool_stats": {str(k): {"blocks": int(v[0]), "locked_bytes": int(v[1]),
                             "in_place_bytes": int(v[2])}
                              for k, v in pool_stats.items()},
        "preloaded_block_counts": {str(k): len(v) for k, v in preload.items()},
        "preloaded_block_bytes": {str(k): int(sum(
            blocks.get(str(k) + "/" + str(block_name), 0)
            for block_name in v))
            for k, v in preload.items()},
        "prefetch_window": getattr(manager, "prefetch_window", None),
        "inflight_count": len(getattr(manager, "inflight", {})),
        "pinned_models": sorted(str(v) for v in getattr(manager, "pinned_models", ())),
        "pool_models": sorted(str(v) for v in getattr(manager, "pool_models", ())),
        "max_reservable_memory_bytes": getattr(manager, "max_reservable_memory", None),
    }


def record_request_begin_mmgp(manager):
    if _active is None:
        return
    _active["_mmgp_start"] = _mmgp_snapshot(manager)


def end(model, offloadobj, error=None, effective_profile=None):
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
    record["effective_loaded_profile"] = effective_profile
    record["model_loads_cumulative"] = _load_count
    record["model_object_id"] = id(model) if model is not None else None
    record["pipeline_object_id"] = id(getattr(model, "pipeline", None)) if model is not None else None
    record["transformer_object_id"] = id(getattr(model, "model", None)) if model is not None else None
    record["offload_object_id"] = id(offloadobj) if offloadobj is not None else None
    record["placements"] = _placement(model) if model is not None else []
    record["cuda_allocated_bytes"] = torch.cuda.memory_allocated()
    record["cuda_reserved_bytes"] = torch.cuda.memory_reserved()
    record["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    # Process RSS is observed, not a machine-wide RAM allocation estimate.
    try:
        proc_lines = Path("/proc/self/status").read_text().splitlines()
        for line in proc_lines:
            if line.startswith(("VmRSS:", "VmHWM:", "VmPin:")):
                name, value = line.split(":", 1)
                record["process_" + name.lower() + "_kib"] = int(value.strip().split()[0])
    except (OSError, ValueError, IndexError):
        pass
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
    start_snapshot = record.pop("_mmgp_start", None)
    end_snapshot = _mmgp_snapshot(offloadobj)
    record["mmgp_start"] = start_snapshot
    record["mmgp_end"] = end_snapshot
    if start_snapshot and end_snapshot and start_snapshot["offload_object_id"] == end_snapshot["offload_object_id"]:
        before = start_snapshot["bulk_move_blocks_cumulative"]
        after = end_snapshot["bulk_move_blocks_cumulative"]
        if before is not None and after is not None:
            record["mmgp_bulk_move_block_delta"] = after - before
            record["mmgp_bulk_move_block_delta_is_not_h2d_bytes"] = True
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
