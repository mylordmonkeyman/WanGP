# Copyright (c) 2026. Experimental LTX-2.5 SM70 cuBLAS ConvRot research.
"""Explicitly opt-in cuBLAS HGEMM/FP32-output ConvRot pilot on V100.

This is NOT auto-selection or a validated production backend. It uses only the
checkpoint-backed numerical-audit passing shape families and the original
INT8 weight (the FP16 expansion is a bounded *transient* scratch allocation).
No FP16 master weights are saved or cached, so MMGP can continue to offload
the INT8 weights normally. Unsupported shapes, CPU and CUDA graph capture
always use WanGP's native R0 path.

Activation safety is checked each eligible invocation. FP32 rotation is an
explicit change from the BF16 R0 path. Any borderline visual output must be
reviewed by the user before further optimization.
"""
from __future__ import annotations

import atexit
import ctypes
import json
import os
from pathlib import Path

import torch

MODE = os.environ.get("WAN2GP_LTX25_CUBLAS_E_MODE", "").strip().lower()
LIB = os.environ.get("LTX25_CUBLAS_E_LIB", "")
WEIGHT_CAP = 128 * 1024 * 1024
# Synthetic checkpoint-backed FP32-rotation gate passed these shapes.
# The (N=4096,K=2048) shape failed and must remain native.
PASS_SHAPES = {
    (2048, 2048), (4096, 4096), (2048, 4096), (4096, 16384),
    (16384, 4096), (2048, 8192), (8192, 2048),
}
_FN = None
_STATS = {"used": 0, "fallback": 0, "range_fallback": 0,
          "sizes": {}, "mode": MODE, "production_eligible": False}


def _fn():
    global _FN
    if _FN is not None:
        return _FN
    if not LIB or not Path(LIB).is_file():
        raise RuntimeError("Explicit ConvRot pilot requires LTX25_CUBLAS_E_LIB with compiled sm_70 cuBLAS extension")
    dll = ctypes.CDLL(LIB)
    fn = dll.ltx25_cublas_gemm_f16_f32
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    _FN = (dll, fn)  # Retain the library as long as its callable is active.
    print("[LTX25_CUBLAS_E] Explicit fp32rot pilot enabled; no persistent FP16 weight set", flush=True)
    return _FN


def _record(reason, n, k, m):
    key = f"{n}x{k}"
    row = _STATS["sizes"].setdefault(key, {"calls": 0, "eligible_calls": 0})
    row["calls"] += 1
    if reason == "used":
        _STATS["used"] += 1
        row["eligible_calls"] += 1
    elif reason == "range":
        _STATS["range_fallback"] += 1
    else:
        _STATS["fallback"] += 1


def maybe_linear(x, weight, bias, group):
    """Return experimental result or None for the untouched WanGP R0 path.

    Only called by QLinearInt8ConvRot on an explicit environment opt-in.
    Never called for Quanto non-ConvRot layers or CPU-only weights.
    """
    if MODE != "fp32rot":
        return None
    q = getattr(weight, "_data", None)
    scale = getattr(weight, "_scale", None)
    if q is None or scale is None or q.ndim != 2:
        return None
    n, k = q.shape
    m = x.numel() // x.shape[-1]
    if (n, k) not in PASS_SHAPES or group != 256 or m < 64 or m > 8192:
        _record("shape", n, k, m)
        return None
    if (type(x) is not torch.Tensor or x.dtype not in (torch.bfloat16, torch.float16)
            or not x.is_cuda or q.device != x.device or scale.device != x.device
            or (bias is not None and bias.device != x.device)
            or not q.is_contiguous() or q.dtype != torch.int8
            or q.numel() * 2 > WEIGHT_CAP
            or torch.cuda.get_device_capability(x.device) != (7, 0)
            or torch.compiler.is_compiling()
            or torch.cuda.is_current_stream_capturing()):
        _record("shape", n, k, m)
        return None

    from shared.qtypes.int8_convrot import _rotate_activation

    # Precision path: FP32 Hadamard rotation of BF16 or FP16 input,
    # then FP16 activation GEMM with FP32 accumulation and row epilogue.
    # INT8 weights are exactly representable by FP16; a *temporary*, reused
    # only during this call, holds them. No second long-lived weight set.
    x32 = _rotate_activation(x.float(), group).reshape(m, k)
    x16 = x32.to(torch.float16)
    # FP16 lacks BF16's exponent range. Reject overflows and underflows to 0.
    finite = torch.isfinite(x16).all()
    underflow = ((x32 != 0) & (x16 == 0)).any()
    if not bool(finite.item()) or bool(underflow.item()):
        _record("range", n, k, m)
        return None

    w16 = q.to(torch.float16)
    c32 = torch.empty((m, n), dtype=torch.float32, device=x.device)
    stream = torch.cuda.current_stream(x.device)
    dll, gemm = _fn()
    status = gemm(ctypes.c_void_p(w16.data_ptr()),
                  ctypes.c_void_p(x16.data_ptr()),
                  ctypes.c_void_p(c32.data_ptr()),
                  n, m, k, ctypes.c_void_p(stream.cuda_stream))
    if status != 0:
        raise RuntimeError(f"Candidate E cuBLAS GEMM failed, status={status}, N={n}, M={m}, K={k}")
    scale32 = scale.reshape(n).float()
    c32.mul_(scale32.reshape(1, n))
    if bias is not None:
        c32.add_(bias.float().reshape(1, n))
    out = c32.to(x.dtype).reshape(*x.shape[:-1], n)
    _record("used", n, k, m)
    return out


def _summary():
    if MODE == "fp32rot":
        print("[LTX25_CUBLAS_E] " + json.dumps(_STATS, sort_keys=True)[:6000], flush=True)


if MODE == "fp32rot":
    atexit.register(_summary)
