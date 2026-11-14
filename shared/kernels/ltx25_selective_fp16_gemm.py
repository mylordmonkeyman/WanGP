# Copyright (c) 2026. Experimental V100 BF16 GEMM mixed-precision selector.
"""Opt-in temporary FP16 Tensor Core GEMMs for selected dense BF16 operations.

Opt-in only: WAN2GP_LTX25_FP16_GEMM=1 and LTX25_FP16_GEMM_GROUPS=... .
Scope is one DistilledPipeline transformer forward at a time. Implements a
TorchDispatchMode for aten.mm / aten.addmm (not an unsafe global monkeypatch).

Controls: group selectors stage1:wide, stage1:tall, stage1:balanced,
stage2:wide, stage2:tall, stage2:balanced. Shapes are classified by GEMM
output N vs input K: wide=N>=2*K; tall=K>=2*N; balanced=otherwise.
Both operands must be plain CUDA BF16 tensors, M>=128, N,K>=512 and
capability (7,0). Quantized/subclass tensors and small LoRA factors bypass.

All conversions and outputs are ephemeral. No model parameters are mutated,
no FP16 master weights or on-device duplicate persistent caches are created.
Temporary FP16 output is converted back to BF16 at each individual op.
These changes are NOT numerically identical to BF16 GEMM. Input/output FP16
overflow is checked on first call for each (stage,op,M,N,K) by default;
later input magnitude changes could escape the spot check. Use GUARD=always
for strict correctness checks (and invalid performance timing), GUARD=off
to disable checks, or GUARD=first (default) for benchmarking.

Performance must be compared with an uninstrumented no-selector baseline.
Never enable in production without video/audio quality review.
"""
from __future__ import annotations

import atexit
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from typing import Callable

import torch
from torch.utils._python_dispatch import TorchDispatchMode

ENABLED = os.environ.get("WAN2GP_LTX25_FP16_GEMM", "0") == "1"
GROUPS = frozenset(x.strip() for x in os.environ.get("LTX25_FP16_GEMM_GROUPS", "").split(",") if x.strip())
OPS = frozenset(x.strip() for x in os.environ.get("LTX25_FP16_GEMM_OPS", "mm,addmm").split(",") if x.strip())
GUARD = os.environ.get("LTX25_FP16_GEMM_GUARD", "first").strip()
MIN_M = int(os.environ.get("LTX25_FP16_GEMM_MIN_M", "128"))
MIN_DIM = int(os.environ.get("LTX25_FP16_GEMM_MIN_DIM", "512"))
PATH = Path(os.environ.get("LTX25_FP16_GEMM_STATS_PATH", "/tmp/ltx25-fp16-gemm-stats.json"))
GROUP_NAMES = frozenset(f"{phase}:{group}" for phase in ("stage1", "stage2")
                        for group in ("wide", "tall", "balanced"))
if ENABLED:
    unknown = GROUPS - GROUP_NAMES
    if unknown:
        raise ValueError(f"Unsupported LTX25_FP16_GEMM_GROUPS: {sorted(unknown)}")
    if not GROUPS:
        raise ValueError("FP16 GEMM enabled but LTX25_FP16_GEMM_GROUPS is empty")
    if OPS - {"mm", "addmm"} or not OPS:
        raise ValueError(f"Unsupported LTX25_FP16_GEMM_OPS: {sorted(OPS)}")
    if GUARD not in ("off", "first", "always"):
        raise ValueError(f"Unsupported LTX25_FP16_GEMM_GUARD: {GUARD}")
    if MIN_M < 1 or MIN_DIM < 1:
        raise ValueError("FP16 GEMM minimum dimensions must be positive")

_COUNTS = Counter()
_BY_SHAPE: dict[str, Counter] = defaultdict(Counter)
_BLOCKED: set[str] = set()
_CHECKED: set[str] = set()
_REASONS = Counter()
_MAX_OPS = 0


def _shape_class(k: int, n: int) -> str:
    if n >= 2 * k:
        return "wide"
    if k >= 2 * n:
        return "tall"
    return "balanced"


def _plain_bf16(t: object) -> bool:
    # QTensor and other tensor subclasses have different precision semantics:
    # never dispatch them through dense FP16 copies.
    return (type(t) in (torch.Tensor, torch.nn.Parameter)
            and t.device.type == "cuda" and t.dtype == torch.bfloat16
            and t.layout == torch.strided and t.ndim == 2)


def _safe_convert(t: torch.Tensor) -> torch.Tensor:
    return t.to(dtype=torch.float16)


class _SelectiveMode(TorchDispatchMode):
    def __init__(self, phase: str):
        super().__init__()
        self.phase = phase

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        global _MAX_OPS
        kwargs = kwargs or {}
        op = None
        if func is torch.ops.aten.mm.default:
            op = "mm"
            if len(args) < 2:
                return func(*args, **kwargs)
            a, b = args[:2]
            bias = None
            alpha = beta = 1
        elif func is torch.ops.aten.addmm.default:
            op = "addmm"
            if len(args) < 3:
                return func(*args, **kwargs)
            bias, a, b = args[:3]
            beta = kwargs.get("beta", args[3] if len(args) > 3 else 1)
            alpha = kwargs.get("alpha", args[4] if len(args) > 4 else 1)
        else:
            return func(*args, **kwargs)

        _COUNTS["observed_" + op] += 1
        if op not in OPS:
            return func(*args, **kwargs)
        if not _plain_bf16(a) or not _plain_bf16(b):
            _REASONS["not_dense_bf16_cuda"] += 1
            return func(*args, **kwargs)
        m, k = a.shape
        inner, n = b.shape
        if k != inner or m < MIN_M or n < MIN_DIM or k < MIN_DIM:
            _REASONS["small_or_mismatch"] += 1
            return func(*args, **kwargs)
        group = f"{self.phase}:{_shape_class(k, n)}"
        if group not in GROUPS:
            _REASONS["disabled_group"] += 1
            return func(*args, **kwargs)
        if op == "addmm" and (not isinstance(bias, torch.Tensor) or
                             bias.dtype != torch.bfloat16):
            _REASONS["addmm_bias_not_bf16"] += 1
            return func(*args, **kwargs)
        key = f"{self.phase}:{op}:M={m},N={n},K={k}"
        _COUNTS["eligible"] += 1
        _BY_SHAPE[key]["eligible"] += 1
        if key in _BLOCKED:
            _COUNTS["blocked_reference"] += 1
            _BY_SHAPE[key]["blocked_reference"] += 1
            return func(*args, **kwargs)

        # The current dispatch mode is excluded while executing this handler.
        # Conversions, FP16 Tensor Core GEMM and BF16 casts are strictly transient.
        a16, b16 = _safe_convert(a), _safe_convert(b)
        check = GUARD == "always" or (GUARD == "first" and key not in _CHECKED)
        if check:
            _CHECKED.add(key)
            _COUNTS["guard_checks"] += 1
            # This .item() synchronizes the GPU; first-per-shape only by default.
            if not bool(torch.isfinite(a16).all().item() and torch.isfinite(b16).all().item()):
                _BLOCKED.add(key)
                _COUNTS["guard_input_reference"] += 1
                _BY_SHAPE[key]["guard_input_reference"] += 1
                return func(*args, **kwargs)
        result16 = torch.ops.aten.mm.default(a16, b16)
        if check and not bool(torch.isfinite(result16).all().item()):
            _BLOCKED.add(key)
            _COUNTS["guard_output_reference"] += 1
            _BY_SHAPE[key]["guard_output_reference"] += 1
            return func(*args, **kwargs)
        if op == "addmm":
            # Preserve bias and alpha/beta at FP32 precision before BF16 output
            # conversion; pure FP16 addmm would round or overflow the bias.
            result32 = result16.float()
            if alpha != 1:
                result32 = result32 * float(alpha)
            if beta != 0:
                result32 = result32 + bias.float() * float(beta)
            result = result32.to(torch.bfloat16)
        else:
            result = result16.to(torch.bfloat16)
        _COUNTS["converted_" + op] += 1
        _BY_SHAPE[key]["converted"] += 1
        _MAX_OPS += 1
        return result


def call_transformer(module: Callable, *args, **kwargs):
    """Invoke an actual transformer forward without changing model parameters."""
    if not ENABLED:
        return module(*args, **kwargs)
    # Stage is set by the existing DistilledPipeline at each phase boundary.
    with _SelectiveMode(_CURRENT_PHASE):
        return module(*args, **kwargs)


_CURRENT_PHASE = "stage1"


def set_phase(phase: str) -> None:
    global _CURRENT_PHASE
    if ENABLED:
        if phase not in ("stage1", "stage2"):
            raise ValueError(f"Unexpected FP16 GEMM stage: {phase}")
        _CURRENT_PHASE = phase


def _report():
    if not ENABLED:
        return
    data = {
        "kind": "Research-only selective BF16-to-FP16 GEMM (V100)",
        "enabled": ENABLED, "groups": sorted(GROUPS),
        "ops": sorted(OPS), "guard": GUARD,
        "min_m": MIN_M, "min_dimension": MIN_DIM,
        "stats": dict(_COUNTS), "skip_reasons": dict(_REASONS),
        "by_shape": {k: dict(v) for k, v in sorted(_BY_SHAPE.items())},
        "blocked_shapes": sorted(_BLOCKED),
        "persistent_fp16_weight_copies": 0,
        "output_dtype": "torch.bfloat16",
        "quality_equivalent": False,
        "production_eligible": False,
        "warning": "FP16 GEMMs can produce different or degraded video/audio. Default first-per-shape finite checks do not guard later spikes.",
    }
    try:
        PATH.parent.mkdir(parents=True, exist_ok=True)
        PATH.write_text(json.dumps(data, indent=2) + "\n")
        print("[LTX25_SELECTIVE_FP16_GEMM] " + json.dumps({
            k: v for k, v in data.items() if k != "by_shape"
        }, sort_keys=True), flush=True)
    except Exception as error:
        print(f"[LTX25_SELECTIVE_FP16_GEMM_ERROR] {error}", flush=True)


if ENABLED:
    atexit.register(_report)
