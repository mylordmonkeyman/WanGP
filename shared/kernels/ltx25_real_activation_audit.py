# Copyright (c) 2026. Experimental LTX-2.5 SM70 real-activation audit.
"""Opt-in, bounded *actual model activation* comparison: R0 / R1 / Candidate E.

This is intentionally NOT imported on ordinary ConvRot inference. Sampling
runs only with WAN2GP_LTX25_REAL_ACTIVATION_AUDIT=1, during the real WanGP
denoising forward. It materially slows generation and synchronizes at sample
boundaries. Those synchronizations must never be confused with the optimized
Candidate E normal hot path.

R0: actual QLinear's BF16/FP16 rotated activation + Quantized tensor F.linear
leaf (no substitute/surrogate on failure).
R1: same original BF16/FP16 input values, rotate in FP32; multiply original
INT8 checkpoint weights in FP32; apply original FP32 scale and bias; retain
FP32 outputs as the independent reference.
E: *same operation's* already-returned Candidate E output.

Only summary statistics, never raw activations or model weights, are recorded.
No persistent second weight set. The R1 reference is column-tiled and releases
each FP32 slice promptly; sampled rows are bounded.
"""
from __future__ import annotations

import atexit
import datetime as dt
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

_ENABLED = os.environ.get("WAN2GP_LTX25_REAL_ACTIVATION_AUDIT", "") == "1"
_PATH = Path(os.environ.get("LTX25_REAL_AUDIT_PATH", "/cache/ltx25/results/real_activation_audit.json"))
_TARGET_CALLS = (1, 64, 1024)
_MAX_ROWS = 16
_COL_CHUNK = 256
_COUNTERS: dict[str, int] = {}
_SAMPLES: list[dict] = []


def _comparison(y, ref):
    a = y.float()
    r = ref.float()
    delta = (a - r).abs()
    normalized = delta / r.abs().clamp_min(1.0)
    finite = torch.isfinite(a)
    return {
        "p99_normalized": float(torch.quantile(normalized.flatten(), 0.99).item()),
        "max_normalized": float(normalized.max().item()),
        "p99_abs": float(torch.quantile(delta.flatten(), 0.99).item()),
        "max_abs": float(delta.max().item()),
        "nonfinite": int((~finite).sum().item()),
    }


def _capture(x, weight, bias, group, candidate, call_index, name):
    from shared.qtypes.int8_convrot import _rotate_activation

    q, scale = weight._data, weight._scale
    m = x.numel() // x.shape[-1]
    n, k = q.shape
    # Sample across actual token rows instead of only the first tokens.
    indices = torch.linspace(0, m - 1, steps=min(m, _MAX_ROWS),
                             device=x.device).round().long()
    x_rows = x.reshape(m, k).index_select(0, indices)
    cand_rows = candidate.reshape(m, n).index_select(0, indices)
    # Use EXACTLY the original WanGP activation rotation and tensor subclass
    # for the R0 leaf. Never silently replace a failing R0 with a surrogate.
    rotated_r0 = _rotate_activation(x_rows, group)
    actual_r0 = F.linear(rotated_r0, weight, bias)
    # The R1 reference deliberately performs rotation, INT8 dot and scales
    # in FP32; FP16/BF16 inputs retain their original (rounded) values.
    rotated_r1 = _rotate_activation(x_rows.float(), group)
    scale_flat = scale.reshape(n).float()
    b_flat = None if bias is None else bias.float().reshape(n)
    ref_chunks = []
    for start in range(0, n, _COL_CHUNK):
        end = min(n, start + _COL_CHUNK)
        # At most 256*K FP32 temporary weights. This is an offline audit.
        w = q[start:end].float()
        chunk = (rotated_r1 @ w.T) * scale_flat[start:end]
        if b_flat is not None:
            chunk = chunk + b_flat[start:end]
        ref_chunks.append(chunk)
        del w
    r1 = torch.cat(ref_chunks, dim=-1)
    r0_stats = _comparison(actual_r0, r1)
    e_stats = _comparison(cand_rows, r1)
    allowed_p99 = r0_stats["p99_normalized"] + 1e-8
    allowed_max = r0_stats["max_normalized"] + 1e-8
    gate = (r0_stats["nonfinite"] == 0 and e_stats["nonfinite"] == 0
            and e_stats["p99_normalized"] <= allowed_p99
            and e_stats["max_normalized"] <= allowed_max)
    return {
        "status": "PASS" if gate else "FAIL",
        "R0_leaf": "actual F.linear(rotated_input, qweight, bias)",
        "R1": "FP32 rotation x original INT8 q FP32 GEMM x original FP32 scale + FP32 bias",
        "shape": name, "M": int(m), "N": int(n), "K": int(k),
        "input_dtype": str(x.dtype), "output_dtype": str(candidate.dtype),
        "group": int(group), "call_index": int(call_index),
        "sampled_rows": len(indices), "sample_strategy": "uniform_across_token_rows",
        "has_bias": bias is not None,
        "R0": r0_stats, "E": e_stats, "gate": bool(gate),
    }


def maybe_audit(x, weight, bias, group, candidate):
    if not _ENABLED:
        return
    # No host-device transfers, shape allocations, or extra reads happen
    # outside the explicitly opted-in audit. Sampling requires a completed
    # live model forward, not a synthetic checkpoint-only probe.
    n, k = weight._data.shape
    key = f"{n}x{k}"
    count = _COUNTERS.get(key, 0) + 1
    _COUNTERS[key] = count
    if count not in _TARGET_CALLS:
        return
    try:
        with torch.no_grad():
            sample = _capture(x, weight, bias, group, candidate, count, key)
    except Exception as exc:
        sample = {
            "status": "ERROR", "gate": False, "shape": key,
            "M": int(x.numel() // x.shape[-1]), "N": int(n), "K": int(k),
            "call_index": count, "error_type": type(exc).__name__,
            "error": str(exc)[:600],
        }
    _SAMPLES.append(sample)
    print("LTX25_REAL_ACTIVATION_SAMPLE " + json.dumps(sample, sort_keys=True)[:3000], flush=True)


def _summary():
    if not _ENABLED:
        return
    by_shape = {}
    for s in _SAMPLES:
        row = by_shape.setdefault(s["shape"], {"samples": 0, "passes": 0, "fails": 0, "errors": 0})
        row["samples"] += 1
        if s["status"] == "PASS":
            row["passes"] += 1
        elif s["status"] == "FAIL":
            row["fails"] += 1
        else:
            row["errors"] += 1
    summary = {
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_sha": os.environ.get("GITHUB_SHA", "local"),
        "kind": "Real WanGP LTX-2.5 generation activation, sampled-row FP32 accuracy audit",
        "real_model_activations": True,
        "full_output_distribution_tested": False,
        "production_validated": False,
        "candidate_e_enabled": True,
        "targets": list(_TARGET_CALLS), "max_rows_per_sample": _MAX_ROWS,
        "max_fp32_weight_rows_temporary": _COL_CHUNK,
        "calls_observed": _COUNTERS,
        "by_shape": by_shape,
        "samples": _SAMPLES,
        "total_samples": len(_SAMPLES),
        "gate_passes": sum(s["status"] == "PASS" for s in _SAMPLES),
        "gate_failures": sum(s["status"] == "FAIL" for s in _SAMPLES),
        "execution_errors": sum(s["status"] == "ERROR" for s in _SAMPLES),
    }
    summary["status"] = (
        "PASS" if summary["total_samples"] >= 7
        and summary["execution_errors"] == 0 and summary["gate_failures"] == 0
        else "FAIL"
    )
    try:
        _PATH.parent.mkdir(parents=True, exist_ok=True)
        _PATH.write_text(json.dumps(summary, indent=2) + "\n")
        print("LTX25_REAL_ACTIVATION_SUMMARY " + json.dumps(
            {k: v for k, v in summary.items() if k != "samples"}, sort_keys=True)[:6000], flush=True)
    except Exception as exc:
        print(f"LTX25_REAL_ACTIVATION_WRITE_FAILED {type(exc).__name__}: {exc}", flush=True)


if _ENABLED:
    atexit.register(_summary)
