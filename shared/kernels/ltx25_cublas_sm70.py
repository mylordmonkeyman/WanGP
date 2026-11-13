# Copyright (c) 2026. Experimental LTX-2.5 SM70 cuBLAS ConvRot research.
"""SM70-specific auto-selected cuBLAS FP32-output ConvRot on V100.

Default on eligible V100 INT8 ConvRot shape families. It uses only the
checkpoint-backed numerical-audit passing shape families and the original
INT8 weight (the FP16 expansion is a bounded *transient* scratch allocation).
No FP16 master weights are saved or cached, so MMGP can continue to offload
the INT8 weights normally. Unsupported shapes, CPU and CUDA graph capture
always use WanGP's native R0 path.

Activation safety is checked entirely on the GPU: each FP32-rotated input row
receives a reversible power-of-two scale when representable as nonzero FP16.
Exceptional rows execute the same operation in FP32 on the GPU *before* its
output is used. No per-linear .item(), host-side range flags, or stream sync.
FP32 rotation changes arithmetic relative to native BF16 R0. Candidate E
passed the recorded single-seed numerical/video gates, but this is NOT a
claim of universal perceptual equivalence. Set WAN2GP_LTX25_CUBLAS_E_MODE=off
to restore WanGP's original dispatch without rebuilding or changing weights.
If the compiled extension is unavailable, the implicit default safely uses
native R0. An explicitly requested fp32rot mode retains strict test errors.
"""
from __future__ import annotations

import atexit
import ctypes
import json
import os
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch

_MODE_IMPLICIT = (os.environ.get("WAN2GP_LTX25_DEFAULT_ON") == "1"
                  or "WAN2GP_LTX25_CUBLAS_E_MODE" not in os.environ)
MODE = os.environ.get("WAN2GP_LTX25_CUBLAS_E_MODE", "fp32rot").strip().lower()
if MODE == "auto":
    # Never run the unvalidated auto dispatch. Normalize to the exact
    # fp32rot selector which matched all 49 decoded frames on the V100.
    MODE = "fp32rot"
    _MODE_IMPLICIT = True
    os.environ["WAN2GP_LTX25_CUBLAS_E_MODE"] = "fp32rot"
    os.environ["WAN2GP_LTX25_DEFAULT_ON"] = "1"
REAL_ACTIVATION_AUDIT = os.environ.get("WAN2GP_LTX25_REAL_ACTIVATION_AUDIT", "") == "1"
PROFILE_STAGES = os.environ.get("WAN2GP_LTX25_PROFILE_STAGES", "") == "1"
USE_WORKSPACE = os.environ.get("WAN2GP_LTX25_E_WORKSPACE", "1") == "1"
CAST_MODE = os.environ.get("WAN2GP_LTX25_E_CAST", "vector4").strip().lower()
ROT_MODE = os.environ.get("WAN2GP_LTX25_HADAMARD", "torch").strip().lower()
GEMM_TUNE = os.environ.get("WAN2GP_LTX25_GEMM_TUNE", "off").strip().lower()
LIB = os.environ.get("LTX25_CUBLAS_E_LIB", "")
WEIGHT_CAP = 128 * 1024 * 1024
# Synthetic checkpoint-backed FP32-rotation gate passed these shapes.
# The (N=4096,K=2048) shape failed and must remain native.
PASS_SHAPES = {
    (2048, 2048), (4096, 4096), (2048, 4096), (4096, 16384),
    (16384, 4096), (2048, 8192), (8192, 2048),
}
_FN = None
_CAST_FN = None
_ROT_FN = None
_ALGO2_FN = None
_NATIVE_UNAVAILABLE = False
_NATIVE_FAILURE_REPORTED = False
# Tiny persistent device-side counters, not weights. Read only at process exit.
# Allocations/updates happen on the current CUDA stream; normal use is a
# single-threaded WanGP inference stream (concurrent-stream tests still needed).
_BAD_ROW_COUNTS = {}
_STATS = {"used": 0, "fallback": 0, "range_fallback": 0,
          "sizes": {}, "mode": MODE, "cast_mode": CAST_MODE or "torch_to",
          "production_eligible": False,
          "auto_selected": _MODE_IMPLICIT,
          "auto_native_fallback": 0,
          "range_fallback_units": "rows (device-side FP32 dot)",
          "row_range_policy": "power_of_two_nonzero_fp16_else_fp32_dot",
          "rotation_mode": ROT_MODE,
          "gemm_tune": GEMM_TUNE, "gemm_algo2_calls": 0}


def _native_library():
    """Use an explicit built library, or build/cache the bundled SM70 source.

    No network, model download, checkpoint mutation or persistent weight set.
    Compiles once per source/CUDA version under a filesystem lock. Compilation
    happens before enqueuing any CUDA work for the first eligible linear.
    """
    if LIB:
        if not Path(LIB).is_file():
            raise FileNotFoundError(f"LTX25_CUBLAS_E_LIB does not exist: {LIB}")
        return LIB
    if os.environ.get("WAN2GP_LTX25_E_AUTO_BUILD", "1") != "1":
        raise RuntimeError("Automatic SM70 extension compilation disabled")
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError("nvcc not on PATH; install CUDA toolkit or set LTX25_CUBLAS_E_LIB")
    source = Path(__file__).resolve().parents[2] / "benchmarks" / "ltx25_convrot" / "ltx25_cublas_e.cu"
    if not source.is_file():
        raise FileNotFoundError(f"Bundled SM70 source missing: {source}")
    key = hashlib.sha256(source.read_bytes() + str(torch.version.cuda).encode()).hexdigest()[:20]
    # Linux-specific lock is imported lazily, so importing WanGP on Windows
    # never fails. Unsupported compilation hosts simply use the original R0.
    import fcntl
    cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "wan2gp" / "sm70-convrot"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"libltx25_sm70_{key}.so"
    lock_path = cache / f"libltx25_sm70_{key}.lock"
    if target.is_file() and target.stat().st_size > 0:
        return str(target)
    with lock_path.open("a+b") as lockfile:
        fcntl.flock(lockfile.fileno(), fcntl.LOCK_EX)
        try:
            if target.is_file() and target.stat().st_size > 0:
                return str(target)
            with tempfile.NamedTemporaryFile(prefix="ltx25-sm70-", suffix=".so",
                                             dir=cache, delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                cmd = [nvcc, "-O3", "-std=c++17", "-arch=sm_70", "-shared",
                       "-Xcompiler", "-fPIC", str(source), "-lcublas", "-o", str(tmp_path)]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=180,
                                        check=False)
                if result.returncode != 0 or not tmp_path.is_file() or tmp_path.stat().st_size == 0:
                    raise RuntimeError("sm70 extension nvcc failed: " + result.stderr[-800:])
                os.replace(tmp_path, target)
            finally:
                tmp_path.unlink(missing_ok=True)
        finally:
            fcntl.flock(lockfile.fileno(), fcntl.LOCK_UN)
    return str(target)


def _fn():
    global _FN
    if _FN is not None:
        return _FN
    resolved_lib = _native_library()
    dll = ctypes.CDLL(resolved_lib)
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
    print("[LTX25_CUBLAS_E] SM70 ConvRot enabled: GPU range guard, vector4 cast and bounded scratch defaults; no persistent FP16 weights", flush=True)
    return _FN


def _cast4_fn():
    global _CAST_FN
    if _CAST_FN is None:
        dll = _fn()[0]
        fn = dll.ltx25_cublas_cast_int8_to_f16_4
        ptr = ctypes.c_void_p
        fn.argtypes = [ptr, ptr, ctypes.c_size_t, ptr]
        fn.restype = ctypes.c_int
        _CAST_FN = fn
    return _CAST_FN


def _rot_fn():
    """Bind the opt-in same-stream FP32 radix4 Hadamard output kernel."""
    global _ROT_FN
    if _ROT_FN is None:
        fn = _fn()[0].ltx25_cublas_rot256_f32
        ptr = ctypes.c_void_p
        fn.argtypes = [ptr, ptr, ctypes.c_size_t, ptr]
        fn.restype = ctypes.c_int
        _ROT_FN = fn
    return _ROT_FN


def _algo2_fn():
    """Bind the optional bit-exact SM70 cuBLAS Algo2 for one tested shape."""
    global _ALGO2_FN
    if _ALGO2_FN is None:
        fn = _fn()[0].ltx25_cublas_gemm_f16_f32_algo2
        ptr = ctypes.c_void_p
        fn.argtypes = [ptr, ptr, ptr, ctypes.c_int, ctypes.c_int,
                       ctypes.c_int, ptr]
        fn.restype = ctypes.c_int
        _ALGO2_FN = fn
    return _ALGO2_FN


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

    # Resolve/build the native dependency before touching any activation,
    # ensuring auto mode can safely return to WanGP's original R0 path.
    global _NATIVE_UNAVAILABLE, _NATIVE_FAILURE_REPORTED
    if _NATIVE_UNAVAILABLE:
        _STATS["auto_native_fallback"] += 1
        return None
    try:
        _fn()
        if CAST_MODE == "vector4":
            _cast4_fn()
        if ROT_MODE == "radix4":
            _rot_fn()
        if GEMM_TUNE == "algo2_m64" and (n, k, m) == (16384, 4096, 64):
            _algo2_fn()
    except (OSError, RuntimeError, AttributeError, ImportError) as exc:
        if not _MODE_IMPLICIT:
            raise
        _NATIVE_UNAVAILABLE = True
        _STATS["auto_native_fallback"] += 1
        if not _NATIVE_FAILURE_REPORTED:
            print("[LTX25_CUBLAS_E] SM70 native extension unavailable; using original R0: "
                  + f"{type(exc).__name__}: {str(exc)[:450]}", flush=True)
            _NATIVE_FAILURE_REPORTED = True
        return None

    from shared.qtypes.int8_convrot import _rotate_activation

    if PROFILE_STAGES:
        from shared.kernels import ltx25_stage_profiler as profiler
        profile = profiler.begin(n, k, m, x.dtype, x.device,
                                 torch.cuda.current_stream(x.device))
    else:
        profile = None

    if GEMM_TUNE not in ("off", "", "algo2_m64"):
        raise ValueError(f"Unsupported WAN2GP_LTX25_GEMM_TUNE={GEMM_TUNE!r}")
    use_algo2 = GEMM_TUNE == "algo2_m64" and (n, k, m) == (16384, 4096, 64)

    # Default: use the exact already-validated FP32 dense H256 rotation.
    # Experimental radix4: the same H4-Kronecker matrix in a different
    # FP32 accumulation order. Keep it opt-in until complete video gates pass.
    if ROT_MODE == "radix4":
        x_source32 = x.float().reshape(m, k).contiguous()
        x32 = torch.empty((m, k), dtype=torch.float32, device=x.device)
        ptr = ctypes.c_void_p
        stream = ptr(torch.cuda.current_stream(x.device).cuda_stream)
        status = _rot_fn()(ptr(x_source32.data_ptr()), ptr(x32.data_ptr()),
                           x_source32.numel(), stream)
        if status:
            raise RuntimeError(f"SM70 radix4 Hadamard failed: status={status}, M={m}, K={k}")
    elif ROT_MODE == "flat_mm":
        # Numerically bit-identical to the existing H256 matmul in the
        # isolated V100 shape tests. Reuses the *same* H256 constant,
        # FP32 operands, checkpoint weights and row-range policy.
        from shared.qtypes.int8_convrot import _regular_hadamard
        x_source32 = x.float().reshape(-1, 256)
        h256 = _regular_hadamard(256, x.device, torch.float32)
        x32 = torch.mm(x_source32, h256).reshape(m, k)
    elif ROT_MODE in ("torch", ""):
        x32 = _rotate_activation(x.float(), group).reshape(m, k).contiguous()
    else:
        raise ValueError(f"Unknown WAN2GP_LTX25_HADAMARD={ROT_MODE!r}")
    if PROFILE_STAGES:
        profiler.mark(profile, "rotation")
    scale32 = scale.reshape(n).contiguous()
    bias32 = None if bias is None else bias.float().reshape(n).contiguous()
    bad_rows = _bad_row_counter(x.device)
    lease = None
    if USE_WORKSPACE:
        from shared.kernels import ltx25_workspace_sm70
        lease = ltx25_workspace_sm70.try_acquire(
            m, n, k, x.device, torch.cuda.current_stream(x.device))
    if lease is not None:
        x16, exponents, slow_rows, c32 = (
            lease.x16, lease.exponents, lease.slow_rows, lease.c32)
    else:
        x16 = torch.empty((m, k), dtype=torch.float16, device=x.device)
        exponents = torch.empty((m,), dtype=torch.int32, device=x.device)
        slow_rows = torch.empty((m,), dtype=torch.int32, device=x.device)
        c32 = torch.empty((m, n), dtype=torch.float32, device=x.device)
    if PROFILE_STAGES:
        profiler.mark(profile, "alloc")
    try:
        if CAST_MODE == "vector4":
            # Faster exact integer conversion, still a *transient* weight
            # expansion. No duplicate weights survive this linear call.
            w16 = torch.empty(q.shape, dtype=torch.float16, device=q.device)
            ptr = ctypes.c_void_p
            stream_ptr = ptr(torch.cuda.current_stream(x.device).cuda_stream)
            rc = _cast4_fn()(ptr(q.data_ptr()), ptr(w16.data_ptr()),
                             q.numel(), stream_ptr)
            if rc:
                raise RuntimeError(f"Candidate E vector4 INT8 cast failed, status={rc}")
        elif CAST_MODE in ("", "torch"):
            w16 = q.to(torch.float16)
        else:
            raise ValueError(f"Unsupported WAN2GP_LTX25_E_CAST={CAST_MODE!r}")
        if PROFILE_STAGES:
            profiler.mark(profile, "weight_cast")

        dll, gemm, prepare, finish = _fn()
        ptr = ctypes.c_void_p
        stream = ptr(torch.cuda.current_stream(x.device).cuda_stream)
        status = prepare(ptr(x32.data_ptr()), ptr(x16.data_ptr()),
                         ptr(exponents.data_ptr()), ptr(slow_rows.data_ptr()),
                         ptr(bad_rows.data_ptr()), m, k, stream)
        if status:
            raise RuntimeError(f"Candidate E row prepare failed, status={status}, N={n}, M={m}, K={k}")
        if PROFILE_STAGES:
            profiler.mark(profile, "prepare")
        selected_gemm = _algo2_fn() if use_algo2 else gemm
        status = selected_gemm(ptr(w16.data_ptr()), ptr(x16.data_ptr()),
                               ptr(c32.data_ptr()), n, m, k, stream)
        if status:
            raise RuntimeError(f"Candidate E cuBLAS GEMM failed, status={status}, N={n}, M={m}, K={k}")
        if use_algo2:
            _STATS["gemm_algo2_calls"] += 1
        if PROFILE_STAGES:
            profiler.mark(profile, "gemm")
        status = finish(ptr(c32.data_ptr()), ptr(x32.data_ptr()), ptr(q.data_ptr()),
                        ptr(scale32.data_ptr()),
                        ptr(0 if bias32 is None else bias32.data_ptr()),
                        ptr(exponents.data_ptr()), ptr(slow_rows.data_ptr()),
                        m, n, k, stream)
        if status:
            raise RuntimeError(f"Candidate E FP32 epilogue failed, status={status}, N={n}, M={m}, K={k}")
        if PROFILE_STAGES:
            profiler.mark(profile, "epilogue")
        out = c32.to(x.dtype).reshape(*x.shape[:-1], n)
        if PROFILE_STAGES:
            profiler.mark(profile, "output")
    finally:
        if lease is not None:
            # Output conversion is enqueued: same-stream reuse is safe.
            lease.release()
    _record("used", n, k, m)
    if REAL_ACTIVATION_AUDIT:
        # NOT a performance run: shed the large no-longer-needed temporary
        # operands before the next, explicitly synchronizing R0/R1 audit.
        # All device work is already enqueued on the same current stream.
        del x32, x16, exponents, slow_rows, c32, w16, scale32, bias32
        from shared.kernels import ltx25_real_activation_audit
        ltx25_real_activation_audit.maybe_audit(x, weight, bias, group, out)
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
