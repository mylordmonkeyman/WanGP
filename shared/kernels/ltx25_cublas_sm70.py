# Copyright (c) 2026. Experimental LTX-2.5 SM70 cuBLAS ConvRot research.
"""SM70-specific auto-selected cuBLAS FP32-output ConvRot on V100.

Default on eligible V100 INT8 ConvRot shape families, including (4096,2048)
in both denoising stages unless specifically disabled for a stage. The
(4096,2048) family failed the historical strict per-GEMM numerical audit;
its default was explicitly selected after T2V/I2V video comparisons. All
other families passed that gate. The original INT8 checkpoint is preserved
and the FP16 expansion uses only bounded *transient* scratch allocation.
No FP16 master weights are saved or cached, so MMGP can continue to offload
the INT8 weights normally. Unsupported shapes, CPU and CUDA graph capture
always use WanGP's native R0 path.

Activation safety is checked entirely on the GPU: each FP32-rotated input row
receives a reversible power-of-two scale when representable as nonzero FP16.
Exceptional rows execute the same operation in FP32 on the GPU *before* its
output is used. No per-linear .item(), host-side range flags, or stream sync.
FP32 rotation changes arithmetic relative to native BF16 R0. Recorded
single-seed video comparisons are not universal perceptual-quality gates.
Set WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE1=1 and/or
WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE2=1 to keep that shape on fallback
during the specified denoising stage(s). Set WAN2GP_LTX25_CUBLAS_E_MODE=off
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

_requested_mode = os.environ.get("WAN2GP_LTX25_CUBLAS_E_MODE")
_MODE_IMPLICIT = _requested_mode is None or _requested_mode.strip().lower() == "auto"
MODE = (_requested_mode or "fp32rot").strip().lower()
if MODE == "auto":
    # Compatibility with older explicit auto invocations.
    MODE = "fp32rot"
    os.environ["WAN2GP_LTX25_CUBLAS_E_MODE"] = MODE
if MODE not in ("fp32rot", "off"):
    raise ValueError(f"Unsupported WAN2GP_LTX25_CUBLAS_E_MODE={MODE!r}")
USE_WORKSPACE = os.environ.get("WAN2GP_LTX25_E_WORKSPACE", "1") == "1"
CAST_MODE = os.environ.get("WAN2GP_LTX25_E_CAST", "vector8").strip().lower()
ROT_MODE = os.environ.get("WAN2GP_LTX25_HADAMARD", "radix4").strip().lower()
ROT_INPUT = os.environ.get("WAN2GP_LTX25_ROT_INPUT", "direct").strip().lower()
OUTPUT_MODE = os.environ.get("WAN2GP_LTX25_E_OUTPUT", "fused").strip().lower()
# Experiment only: split narrow-M (<=128), wide-N FP32/BF16 epilogue across CTAs.
# Default is unchanged; no second weight copies or host/device synchronization.
EPILOGUE_TILED = os.environ.get("WAN2GP_LTX25_E_EPILOGUE_TILED", "0") == "1"
# Narrow-row gate defaults to validated 73f threshold (128). An explicit,
# experiment-only override permits 241f audio lengths (~251 rows), capped at 512.
EPILOGUE_TILED_MAX_M = int(os.environ.get("WAN2GP_LTX25_E_EPILOGUE_TILED_MAX_M", "128"))
if not 1 <= EPILOGUE_TILED_MAX_M <= 512:
    raise ValueError("WAN2GP_LTX25_E_EPILOGUE_TILED_MAX_M must be in [1, 512]")
GEMM_TUNE = os.environ.get("WAN2GP_LTX25_GEMM_TUNE", "off").strip().lower()
LIB = os.environ.get("LTX25_CUBLAS_E_LIB", "")
WEIGHT_CAP = 128 * 1024 * 1024
# Research controls: route eligible ConvRot GEMMs below/above these M cutoffs
# to the existing native fallback (then Stage 2's selective FP16, if enabled).
# No change without an explicit environment override.
MIN_M = int(os.environ.get("WAN2GP_LTX25_E_MIN_M", "64"))
MAX_M = int(os.environ.get("WAN2GP_LTX25_E_MAX_M", "8192"))
ROW_CENSUS = os.environ.get("WAN2GP_LTX25_E_ROW_CENSUS", "0") == "1"
if not (1 <= MIN_M <= MAX_M <= 65536):
    raise ValueError(f"Invalid Candidate E row thresholds: min={MIN_M}, max={MAX_M}")
# Seven original families passed the strict checkpoint-backed numerical audit.
# The eighth family (N=4096,K=2048) failed that gate, but the user explicitly
# chose to enable it by default based on 73-frame Distilled T2V/I2V testing.
# The opt-out flags below can independently return it to the unchanged native
# WanGP fallback (and Stage2 selective FP16, when configured) in either stage.
PASS_SHAPES = {
    (2048, 2048), (4096, 4096), (2048, 4096), (4096, 16384),
    (16384, 4096), (2048, 8192), (8192, 2048), (4096, 2048),
}

def _bool_flag(name: str) -> bool:
    """Reject misspelled/ambiguous opt-outs instead of silently bypassing them."""
    value = os.environ.get(name, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"

_DISABLE_4096X2048_STAGE1 = _bool_flag("WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE1")
_DISABLE_4096X2048_STAGE2 = _bool_flag("WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE2")

# Old research-only enable controls duplicated the independent opt-outs.
# Reject stale launch settings rather than silently selecting the wrong path.
_OBSOLETE_STAGE_FLAGS = (
    "WAN2GP_LTX25_E_TEST_4096X2048",
    "WAN2GP_LTX25_E_TEST_4096X2048_PHASES",
)
for _obsolete in _OBSOLETE_STAGE_FLAGS:
    if _obsolete in os.environ:
        raise ValueError(
            f"{_obsolete} is obsolete; use the independent "
            "WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE1/2=1 opt-outs"
        )
ENABLED_4096X2048_PHASES = frozenset(
    phase for phase, disabled in (
        ("stage1", _DISABLE_4096X2048_STAGE1),
        ("stage2", _DISABLE_4096X2048_STAGE2),
    ) if not disabled
)
_CURRENT_PHASE = "stage1"

def set_phase(phase: str) -> None:
    """Select the active denoising stage for the (4096,2048) per-stage gate.

    Both Distilled and Dev two-stage pipelines call this before their
    denoising transformer; independent of Stage2 selective FP16 interceptors.
    """
    global _CURRENT_PHASE
    if phase not in ("stage1", "stage2"):
        raise ValueError(f"Unsupported Candidate E phase {phase!r}")
    _CURRENT_PHASE = phase


_FN = None
_CAST_FN = None
_CAST8_FN = None
_ROT_FN = None
_ROT_DIRECT_FNS = {}
_FINISH_DIRECT_FNS = {}
_FINISH_TILED_FNS = {}
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
          "rotation_mode": ROT_MODE, "rotation_input": ROT_INPUT, "output_mode": OUTPUT_MODE,
          "epilogue_tiled_enabled": EPILOGUE_TILED,
          "epilogue_tiled_max_m": EPILOGUE_TILED_MAX_M,
          "epilogue_tiled_calls": 0,
          "gemm_tune": GEMM_TUNE, "gemm_algo2_calls": 0,
          "min_m": MIN_M, "max_m": MAX_M, "row_census_enabled": ROW_CENSUS,
          "active_4096x2048_phases": sorted(ENABLED_4096X2048_PHASES),
          "disabled_4096x2048_stages": [
              stage for stage, disabled in (
                  ("stage1", _DISABLE_4096X2048_STAGE1),
                  ("stage2", _DISABLE_4096X2048_STAGE2),
              ) if disabled
          ],
          "shape_4096x2048_by_phase": {
              "stage1": {"used": 0, "fallback": 0},
              "stage2": {"used": 0, "fallback": 0},
          },
          "row_census": {}}


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
    print(f"[LTX25_CUBLAS_E] SM70 ConvRot enabled: GPU range guard, cast={CAST_MODE}, rotation={ROT_MODE}, rotation_input={ROT_INPUT}, output_mode={OUTPUT_MODE}, workspace={USE_WORKSPACE}, gemm_tune={GEMM_TUNE}; no persistent FP16 weights", flush=True)
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


def _cast8_fn():
    """Bind the V100 exact packed-8 INT8-to-FP16 cast (default; override with vector4)."""
    global _CAST8_FN
    if _CAST8_FN is None:
        dll = _fn()[0]
        fn = dll.ltx25_cublas_cast_int8_to_f16_8
        ptr = ctypes.c_void_p
        fn.argtypes = [ptr, ptr, ctypes.c_size_t, ptr]
        fn.restype = ctypes.c_int
        _CAST8_FN = fn
    return _CAST8_FN


def _rot_fn():
    """Bind the default-on V100 same-stream FP32 radix-4 Hadamard kernel."""
    global _ROT_FN
    if _ROT_FN is None:
        fn = _fn()[0].ltx25_cublas_rot256_f32
        ptr = ctypes.c_void_p
        fn.argtypes = [ptr, ptr, ctypes.c_size_t, ptr]
        fn.restype = ctypes.c_int
        _ROT_FN = fn
    return _ROT_FN


def _rot_direct_fn(dtype):
    """Bind default direct BF16/FP16 radix4, with FP32 internal math."""
    if dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"Unsupported direct radix4 dtype: {dtype}")
    if dtype not in _ROT_DIRECT_FNS:
        name = ("ltx25_cublas_rot256_from_bf16_f32" if dtype == torch.bfloat16
                else "ltx25_cublas_rot256_from_f16_f32")
        fn = getattr(_fn()[0], name)
        ptr = ctypes.c_void_p
        fn.argtypes = [ptr, ptr, ctypes.c_size_t, ptr]
        fn.restype = ctypes.c_int
        _ROT_DIRECT_FNS[dtype] = fn
    return _ROT_DIRECT_FNS[dtype]


def _fused_output_fn(dtype, tiled=False):
    """Bind original or explicit tiled FP32-corrected epilogue to BF16/FP16."""
    if dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"Unsupported fused epilogue dtype: {dtype}")
    functions = _FINISH_TILED_FNS if tiled else _FINISH_DIRECT_FNS
    if dtype not in functions:
        name = ("ltx25_cublas_epilogue_f32_to_bf16" if dtype == torch.bfloat16
                else "ltx25_cublas_epilogue_f32_to_f16")
        if tiled:
            name += "_tiled"
        fn = getattr(_fn()[0], name)
        ptr = ctypes.c_void_p
        fn.argtypes = [ptr] * 8 + [ctypes.c_int] * 3 + [ptr]
        fn.restype = ctypes.c_int
        functions[dtype] = fn
    return functions[dtype]


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
    if (n, k) == (4096, 2048):
        phase_counts = _STATS["shape_4096x2048_by_phase"][_CURRENT_PHASE]
        phase_counts["used" if reason == "used" else "fallback"] += 1
    if ROW_CENSUS:
        signature = f"M={m},N={n},K={k}"
        tracked = _STATS["row_census"].setdefault(signature, {"used": 0, "fallback": 0})
        tracked["used" if reason == "used" else "fallback"] += 1
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
    if ((n, k) not in PASS_SHAPES
            or ((n, k) == (4096, 2048) and _CURRENT_PHASE not in ENABLED_4096X2048_PHASES)
            or group != 256 or m < MIN_M or m > MAX_M):
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
        elif CAST_MODE == "vector8":
            _cast8_fn()
        if ROT_MODE == "radix4":
            if ROT_INPUT == "direct":
                _rot_direct_fn(x.dtype)
            elif ROT_INPUT in ("staged", ""):
                _rot_fn()
            else:
                raise ValueError(f"Unknown WAN2GP_LTX25_ROT_INPUT={ROT_INPUT!r}")
        if OUTPUT_MODE == "fused":
            _fused_output_fn(x.dtype, tiled=EPILOGUE_TILED and m <= EPILOGUE_TILED_MAX_M and n >= 2048)
        elif OUTPUT_MODE not in ("torch", ""):
            raise ValueError(f"Unknown WAN2GP_LTX25_E_OUTPUT={OUTPUT_MODE!r}")
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

    if GEMM_TUNE not in ("off", "", "algo2_m64"):
        raise ValueError(f"Unsupported WAN2GP_LTX25_GEMM_TUNE={GEMM_TUNE!r}")
    use_algo2 = GEMM_TUNE == "algo2_m64" and (n, k, m) == (16384, 4096, 64)

    # Default: fast FP32 radix-4 rotation with the same H4-Kronecker transform.
    # This changes accumulation order relative to the previous torch H256.
    # User-reviewed 241-frame i2v output looked essentially identical,
    # but decoded pixels are not generally bit-identical.
    # HADAMARD=torch restores the historical dense FP32 H256 rotation.
    if ROT_MODE == "radix4":
        x32 = torch.empty((m, k), dtype=torch.float32, device=x.device)
        ptr = ctypes.c_void_p
        stream = ptr(torch.cuda.current_stream(x.device).cuda_stream)
        if ROT_INPUT == "direct":
            # No intermediate FP32 input tensor: the native kernel converts
            # BF16/FP16 elements directly into FP32 shared memory. Its FP32
            # Hadamard arithmetic/order is otherwise unchanged.
            x_source = x.reshape(m, k).contiguous()
            status = _rot_direct_fn(x.dtype)(
                ptr(x_source.data_ptr()), ptr(x32.data_ptr()),
                x_source.numel(), stream)
        elif ROT_INPUT in ("staged", ""):
            x_source32 = x.float().reshape(m, k).contiguous()
            status = _rot_fn()(ptr(x_source32.data_ptr()), ptr(x32.data_ptr()),
                               x_source32.numel(), stream)
        else:
            raise ValueError(f"Unknown WAN2GP_LTX25_ROT_INPUT={ROT_INPUT!r}")
        if status:
            raise RuntimeError(f"SM70 radix4 Hadamard failed: status={status}, M={m}, K={k}, input={ROT_INPUT}")
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
    try:
        if CAST_MODE in ("vector4", "vector8"):
            # Both exact integer conversion kernels use one transient FP16
            # weight tensor. Vector8 is the default; vector4 remains an override;
            # the original INT8 checkpoint is never mutated or duplicated.
            w16 = torch.empty(q.shape, dtype=torch.float16, device=q.device)
            ptr = ctypes.c_void_p
            stream_ptr = ptr(torch.cuda.current_stream(x.device).cuda_stream)
            fn = _cast8_fn() if CAST_MODE == "vector8" else _cast4_fn()
            rc = fn(ptr(q.data_ptr()), ptr(w16.data_ptr()),
                    q.numel(), stream_ptr)
            if rc:
                raise RuntimeError(f"Candidate E {CAST_MODE} INT8 cast failed, status={rc}")
        elif CAST_MODE in ("", "torch"):
            w16 = q.to(torch.float16)
        else:
            raise ValueError(f"Unsupported WAN2GP_LTX25_E_CAST={CAST_MODE!r}")

        dll, gemm, prepare, finish = _fn()
        ptr = ctypes.c_void_p
        stream = ptr(torch.cuda.current_stream(x.device).cuda_stream)
        status = prepare(ptr(x32.data_ptr()), ptr(x16.data_ptr()),
                         ptr(exponents.data_ptr()), ptr(slow_rows.data_ptr()),
                         ptr(bad_rows.data_ptr()), m, k, stream)
        if status:
            raise RuntimeError(f"Candidate E row prepare failed, status={status}, N={n}, M={m}, K={k}")
        selected_gemm = _algo2_fn() if use_algo2 else gemm
        status = selected_gemm(ptr(w16.data_ptr()), ptr(x16.data_ptr()),
                               ptr(c32.data_ptr()), n, m, k, stream)
        if status:
            raise RuntimeError(f"Candidate E cuBLAS GEMM failed, status={status}, N={n}, M={m}, K={k}")
        if use_algo2:
            _STATS["gemm_algo2_calls"] += 1
        if OUTPUT_MODE == "fused":
            # Fuses the final round-to-nearest conversion into the existing
            # FP32 row-scale/exception-correction pass. No permanent weight
            # copy and no changes to cuBLAS or safe-row numerics.
            out = torch.empty((*x.shape[:-1], n), dtype=x.dtype, device=x.device)
            tiled_epilogue = EPILOGUE_TILED and m <= 128 and n >= 2048
            status = _fused_output_fn(x.dtype, tiled=tiled_epilogue)(
                ptr(c32.data_ptr()), ptr(out.data_ptr()),
                ptr(x32.data_ptr()), ptr(q.data_ptr()),
                ptr(scale32.data_ptr()),
                ptr(0 if bias32 is None else bias32.data_ptr()),
                ptr(exponents.data_ptr()), ptr(slow_rows.data_ptr()),
                m, n, k, stream)
            if tiled_epilogue:
                _STATS["epilogue_tiled_calls"] += 1
        else:
            status = finish(ptr(c32.data_ptr()), ptr(x32.data_ptr()), ptr(q.data_ptr()),
                            ptr(scale32.data_ptr()),
                            ptr(0 if bias32 is None else bias32.data_ptr()),
                            ptr(exponents.data_ptr()), ptr(slow_rows.data_ptr()),
                            m, n, k, stream)
        if status:
            raise RuntimeError(f"Candidate E FP32 epilogue failed, status={status}, N={n}, M={m}, K={k}")
        if OUTPUT_MODE != "fused":
            out = c32.to(x.dtype).reshape(*x.shape[:-1], n)
    finally:
        if lease is not None:
            # Output conversion is enqueued: same-stream reuse is safe.
            lease.release()
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
