# Copyright (c) 2026. Experimental LTX-2.5 SM70 cuBLAS ConvRot research.
"""Explicitly opt-in cuBLAS HGEMM/FP32-output ConvRot pilot on V100.

This is NOT auto-selection or a validated production backend. It uses only the
checkpoint-backed numerical-audit passing shape families and the original
INT8 weight (the FP16 expansion is a bounded *transient* scratch allocation).
No FP16 master weights are saved or cached, so MMGP can continue to offload
the INT8 weights normally. Unsupported shapes, CPU and CUDA graph capture
always use WanGP's native R0 path.

Activation safety is checked entirely on the GPU: each FP32-rotated input row
receives a reversible power-of-two scale when representable as normal FP16.
Exceptional rows execute the same operation in FP32 on the GPU *before* its
output is used. No per-linear .item(), host-side range flags, or stream sync.
FP32 rotation is an explicit change from BF16 R0; this remains a research pilot
until real-activation numerical and video quality gates pass.
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
# Tiny persistent device-side counters, not weights. Read only at process exit.
# Allocations/updates happen on the current CUDA stream; normal use is a
# single-threaded WanGP inference stream (concurrent-stream tests still needed).
_BAD_ROW_COUNTS = {}
_STATS = {"used": 0, "fallback": 0, "range_fallback": 0,
          "sizes": {}, "mode": MODE, "production_eligible": False,
          "range_fallback_units": "rows (device-side FP32 dot)",
          "row_range_policy": "power_of_two_normal_fp16_else_fp32_dot"}


def _fn():
    global _FN
    if _FN is not None:
        return _FN
    if not LIB or not Path(LIB).is_file():
        raise RuntimeError("Explicit ConvRot pilot requires LTX25_CUBLAS_E_LIB with compiled sm_70 cuBLAS extension")
    dll = ctypes.CDLL(LIB)
    ptr = ctypes.c_void_p
    gemm = dll.ltx25_cublas_gemm_f16_f32
    gemm.argtypes = [ptr, ptr, ptr, ctypes.c_int, ctypes.c_int, ctypes.c_int, ptr]
    gemm.restype = ctypes.c_int
    prepare = dll.ltx25_cublas_prepare_f32_f16
    prepare.argtypes = [ptr, ptr, ptr, ptr, ptr, ctypes.c_int, ctypes.c_int, ptr]
    prepare.restype = ctypes.c_int
    finish = dll.ltx25_cublas_epilogue_f32
    finish.argtypes = [ptr, ptr, ptr, ptr, ptr, ptr, ptr, ctypes.c_int,
                       ctypes.c_int, ctypes.c_int, ptr]
    finish.restype = ctypes.c_int
    _FN = (dll, gemm, prepare, finish)
    print("[LTX25_CUBLAS_E] Explicit fp32rot GPU-range-guard pilot enabled; no persistent FP16 weights", flush=True)
    return _FN


def _bad_row_counter(device):
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    if index not in _BAD_ROW_COUNTS:
        _BAD_ROW_COUNTS[index] = torch.zeros((1,), device=device, dtype=torch.int32)
    return _BAD_ROW_COUNTS[index]

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
    """Return explicitly selected experimental result or None for WanGP R0.

    The check and any exceptional-row fallback are GPU-only. No unsafe output
    escapes: the row correction runs on the same stream after cuBLAS and before
    conversion to the input dtype. CUDA graph capture remains unsupported.
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
            or scale.dtype != torch.float32 or scale.numel() != n
            or (bias is not None and bias.numel() != n)
            or q.numel() * 2 > WEIGHT_CAP
            or torch.cuda.get_device_capability(x.device) != (7, 0)
            or torch.compiler.is_compiling()
            or torch.cuda.is_current_stream_capturing()):
        _record("shape", n, k, m)
        return None

    from shared.qtypes.int8_convrot import _rotate_activation

    # FP32 Hadamard is the explicitly selected research variant (not R0).
    # Own the temporaries until kernels have been enqueued on this stream;
    # PyTorch's caching allocator tracks the current stream for their reuse.
    x32 = _rotate_activation(x.float(), group).reshape(m, k).contiguous()
    x16 = torch.empty((m, k), dtype=torch.float16, device=x.device)
    exponents = torch.empty((m,), dtype=torch.int32, device=x.device)
    slow_rows = torch.empty((m,), dtype=torch.int32, device=x.device)
    c32 = torch.empty((m, n), dtype=torch.float32, device=x.device)
    w16 = q.to(torch.float16)  # Exact INT8 -> FP16, transient and bounded.
    scale32 = scale.reshape(n).contiguous()
    bias32 = None if bias is None else bias.float().reshape(n).contiguous()
    bad_rows = _bad_row_counter(x.device)

    dll, gemm, prepare, finish = _fn()
    ptr = ctypes.c_void_p
    stream = ptr(torch.cuda.current_stream(x.device).cuda_stream)
    status = prepare(ptr(x32.data_ptr()), ptr(x16.data_ptr()),
                     ptr(exponents.data_ptr()), ptr(slow_rows.data_ptr()),
                     ptr(bad_rows.data_ptr()), m, k, stream)
    if status:
        raise RuntimeError(f"Candidate E row prepare failed, status={status}, N={n}, M={m}, K={k}")
    status = gemm(ptr(w16.data_ptr()), ptr(x16.data_ptr()),
                  ptr(c32.data_ptr()), n, m, k, stream)
    if status:
        raise RuntimeError(f"Candidate E cuBLAS GEMM failed, status={status}, N={n}, M={m}, K={k}")
    status = finish(ptr(c32.data_ptr()), ptr(x32.data_ptr()), ptr(q.data_ptr()),
                    ptr(scale32.data_ptr()),
                    ptr(0 if bias32 is None else bias32.data_ptr()),
                    ptr(exponents.data_ptr()), ptr(slow_rows.data_ptr()),
                    m, n, k, stream)
    if status:
        raise RuntimeError(f"Candidate E FP32 epilogue failed, status={status}, N={n}, M={m}, K={k}")
    out = c32.to(x.dtype).reshape(*x.shape[:-1], n)
    _record("used", n, k, m)
    return out

def _summary():
    if MODE != "fp32rot":
        return
    # One synchronization, only AFTER generation has finished at interpreter
    # exit. This is never in the measured per-linear hot path.
    try:
        _STATS["range_fallback"] = sum(int(t.item()) for t in _BAD_ROW_COUNTS.values())
    except Exception as exc:
        _STATS["range_fallback_read_error"] = f"{type(exc).__name__}: {exc}"
    print("[LTX25_CUBLAS_E] " + json.dumps(_STATS, sort_keys=True)[:6000], flush=True)


if MODE == "fp32rot":
    atexit.register(_summary)
