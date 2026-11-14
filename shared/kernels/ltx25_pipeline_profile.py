# Copyright (c) 2026. Research-only LTX-2.5 phase and operator profiling.
"""Live LTX-2.5 end-to-end profiling, gated by WAN2GP_LTX25_PIPELINE_PROFILE=1.

Phase marks synchronize CUDA at *major* boundaries to measure CPU+GPU critical
path wall time including model offload/preload waits. These barriers deliberately
perturb scheduling: instrumented timings must not be compared to normal runs.
Operator profiler samples ONLY one warmed transformer forward in each diffusion
stage (the 2nd forward in that phase) to reduce trace size and memory pressure.
Operator self-CUDA times are sample attribution, not full-run totals.
"""
from __future__ import annotations

import atexit
from collections import Counter, defaultdict
import datetime as dt
import json
import os
from pathlib import Path
import time

import torch

ENABLED = os.environ.get("WAN2GP_LTX25_PIPELINE_PROFILE", "") == "1"
OP_SAMPLE = os.environ.get("LTX25_PIPELINE_OP_SAMPLE", "1") == "1"
GEMM_CENSUS = os.environ.get("LTX25_PIPELINE_GEMM_CENSUS", "0") == "1"
PATH = Path(os.environ.get("LTX25_PIPELINE_PROFILE_PATH", "/tmp/ltx25-pipeline-profile.json"))
_PHASE = None
_PHASE_START = None
_PHASES = []
_PHASE_ERRORS = []
_FORWARD_PHASE = "stage1"
_FORWARD_CALLS = defaultdict(int)
_FORWARD_SAMPLES = []
_GEMM_CENSUS_SAMPLES = []
_TRANSFORMER = {}
_PEAK_ALLOCATED = 0
_PEAK_RESERVED = 0


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def boundary(next_phase: str) -> None:
    """Close prior phase and start next. Synchronize only when explicitly enabled."""
    global _PHASE, _PHASE_START, _PEAK_ALLOCATED, _PEAK_RESERVED
    if not ENABLED:
        return
    try:
        _cuda_sync()
    except Exception as error:
        _PHASE_ERRORS.append(f"cuda synchronize: {type(error).__name__}: {error}")
        raise
    now = time.perf_counter()
    if _PHASE is not None:
        row = {"phase": _PHASE, "wall_seconds": round(now - _PHASE_START, 6)}
        if torch.cuda.is_available():
            allocated = int(torch.cuda.memory_allocated())
            reserved = int(torch.cuda.memory_reserved())
            _PEAK_ALLOCATED = max(_PEAK_ALLOCATED, allocated)
            _PEAK_RESERVED = max(_PEAK_RESERVED, reserved)
            row["cuda_allocated_mib_at_end"] = round(allocated / 2**20, 2)
            row["cuda_reserved_mib_at_end"] = round(reserved / 2**20, 2)
        _PHASES.append(row)
    _PHASE = next_phase
    _PHASE_START = now


def set_transformer_phase(phase: str) -> None:
    global _FORWARD_PHASE
    if ENABLED:
        _FORWARD_PHASE = phase


def record_transformer(phase: str, ms: float, calls: int) -> None:
    if ENABLED:
        _TRANSFORMER[phase] = {"forward_cuda_event_seconds": round(ms / 1000.0, 5),
                               "forward_calls": int(calls)}



def _capture_gemm_census(context, phase: str, forward_index: int) -> None:
    """Classify sampled CUDA launches and CPU GEMMs without double-counting.

    Native Quanto calls are tagged with LTX25::quanto_qbytes_mm on the CPU.
    Their inner aten::mm kernels are NOT separate missing-GEMM candidates.
    All times are per one sampled transformer forward, NOT full generation.
    """
    from torch.autograd import DeviceType
    from collections import defaultdict

    def family(name: str) -> str:
        n = name.lower()
        if any(x in n for x in ("attention", "flash", "fmha", "sdpa")):
            return "attention"
        if any(x in n for x in ("gemm", "gemv", "matmul", "cublas",
                                 "cutlass", "sgemm", "hgemm", "bgemm",
                                 "igemm", "wmma", "mma_", "mmq", "mmv")):
            return "gemm_or_gemv"
        if "ltx25" in n:
            return "custom_ltx25_other"
        if any(x in n for x in ("copy", "memcpy", "memset")):
            return "memory"
        return "other_unknown"

    def has_quanto_ancestor(event) -> bool:
        cursor = event
        for _ in range(32):
            if cursor is None:
                break
            if "LTX25::quanto_qbytes_mm" in str(getattr(cursor, "name", "")):
                return True
            cursor = getattr(cursor, "cpu_parent", None)
        return False

    kernel_counts = Counter()
    kernel_time_us = defaultdict(float)
    kernel_family = Counter()
    family_time_us = defaultdict(float)
    kernel_quanto_us = defaultdict(float)
    kernel_other_us = defaultdict(float)
    kernel_unattributed_us = defaultdict(float)
    cpu_matrix = Counter()
    cpu_non_quanto_matrix = Counter()
    cpu_quanto_matrix = Counter()
    cpu_matrix_shape = Counter()
    non_quanto_shapes = Counter()
    other_op = Counter()
    cuda_kernel_events = 0
    cuda_gpu_us = 0.0
    cpu_events = 0
    for ev in context.events():
        name = str(getattr(ev, "name", getattr(ev, "key", "")))
        device = getattr(ev, "device_type", None)
        if device == DeviceType.CUDA or str(device).endswith(".CUDA"):
            # CUDA kernels have a real GPU duration; the CPU event's
            # cuda_time_total attributes are *attribution*, not GPU launches.
            us = float(getattr(ev, "device_time_total", 0) or
                       getattr(ev, "self_device_time_total", 0) or 0)
            cuda_kernel_events += 1
            cuda_gpu_us += us
            kernel_counts[name] += 1
            kernel_time_us[name] += us
            category = family(name)
            kernel_family[category] += 1
            family_time_us[category] += us
            parent = getattr(ev, "cpu_parent", None)
            if parent is None:
                kernel_unattributed_us[name] += us
            elif has_quanto_ancestor(parent):
                kernel_quanto_us[name] += us
            else:
                kernel_other_us[name] += us
        elif device == DeviceType.CPU or str(device).endswith(".CPU"):
            cpu_events += 1
            canonical = name.lower().replace("::", "_")
            # Include all dense and batched matmuls. Nested ops are counted
            # as operator *invocations*, not as independent physical kernels.
            if canonical in ("aten_mm", "aten_addmm", "aten_bmm",
                             "aten_baddbmm", "aten_matmul", "aten_linear",
                             "aten_einsum", "quanto_qbytes_mm"):
                cpu_matrix[name] += 1
                tagged = has_quanto_ancestor(ev)
                if tagged:
                    cpu_quanto_matrix[name] += 1
                else:
                    cpu_non_quanto_matrix[name] += 1
                shapes = getattr(ev, "input_shapes", []) or []
                desc = str(shapes)[:350]
                key = (name, desc)
                cpu_matrix_shape[key] += 1
                if not tagged:
                    non_quanto_shapes[key] += 1
            else:
                other_op[name] += 1

    # Do not mistake stacked high-level ops (e.g. linear -> mm) as distinct GEMMs.
    # CUDA kernel aggregation is the best approximation to actual GPU work.
    def top_shapes(data, limit=120):
        return [{"op": name, "input_shapes": shape, "count": n}
                for (name, shape), n in data.most_common(limit)]
    all_kernels = sorted(kernel_time_us, key=kernel_time_us.get, reverse=True)
    kernel_rows = [{
        "kernel": name, "family": family(name), "calls": int(kernel_counts[name]),
        "gpu_ms": round(kernel_time_us[name] / 1000, 6),
        "in_tagged_quanto_gpu_ms": round(kernel_quanto_us[name] / 1000, 6),
        "outside_tagged_quanto_gpu_ms": round(kernel_other_us[name] / 1000, 6),
        "unattributed_gpu_ms": round(kernel_unattributed_us[name] / 1000, 6),
    } for name in all_kernels[:250]]
    report = {
        "phase": phase, "forward_index": forward_index,
        "cuda_kernel_event_count": cuda_kernel_events,
        "cuda_kernel_gpu_ms_sum": round(cuda_gpu_us / 1000, 6),
        "cpu_events": cpu_events,
        "cuda_kernel_family": [
            {"family": fam, "calls": int(n),
             "gpu_ms": round(family_time_us[fam] / 1000, 6)}
            for fam, n in kernel_family.most_common()
        ],
        "top_cuda_kernels": kernel_rows,
        "cpu_matmul_operator_invocations": dict(cpu_matrix),
        "cpu_matmul_in_tagged_quanto": dict(cpu_quanto_matrix),
        "cpu_matmul_outside_tagged_quanto": dict(cpu_non_quanto_matrix),
        "top_cpu_matmul_shapes_all": top_shapes(cpu_matrix_shape),
        "top_cpu_matmul_shapes_outside_quanto": top_shapes(non_quanto_shapes),
        "top_other_cpu_ops": [{"operator": key, "calls": n}
                              for key, n in other_op.most_common(35)],
        "limitations": [
            "Only the second transformer forward in this stage is sampled.",
            "CUDA kernel family is a name heuristic; other_unknown may contain GEMMs.",
            "High-level CPU mm, linear and matmul can be nested and must not be added.",
            "Tagging native Quanto with a CPU profiler range is best-effort.",
            "CUDA kernels with unattributed cpu_parent are not assumed non-Quanto.",
            "GPU kernel durations may overlap on different CUDA streams.",
            "Attention kernels and custom CUDA kernels may implement matrix multiplications.",
        ],
    }
    _GEMM_CENSUS_SAMPLES.append(report)
    print("[LTX25_GEMM_CENSUS] " + json.dumps({
        "phase": phase, "sampled_forward": forward_index,
        "cuda_kernels": cuda_kernel_events,
        "cuda_kernel_gpu_ms_sum": report["cuda_kernel_gpu_ms_sum"],
        "cpu_gemm_ops": report["cpu_matmul_operator_invocations"],
        "cpu_gemm_outside_quanto": report["cpu_matmul_outside_tagged_quanto"],
        "kernel_families": report["cuda_kernel_family"],
    }, sort_keys=True), flush=True)

def maybe_sample_forward(module, *args, **kwargs):
    """Run transformer unchanged, optionally sample one forward with PyTorch profiler.

    Profiling errors in setup/reporting are recorded without re-running a forward.
    Model forward errors propagate normally and are never swallowed.
    """
    if not ENABLED or not OP_SAMPLE:
        return module(*args, **kwargs)
    phase = _FORWARD_PHASE
    _FORWARD_CALLS[phase] += 1
    call = _FORWARD_CALLS[phase]
    if call != 2:
        return module(*args, **kwargs)
    try:
        from torch.profiler import ProfilerActivity, profile
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        context = profile(activities=activities, record_shapes=GEMM_CENSUS,
                          profile_memory=False, with_stack=False)
        context.start()
    except Exception as error:
        _PHASE_ERRORS.append(f"operator sample start {phase}: {type(error).__name__}: {error}")
        return module(*args, **kwargs)

    try:
        result = module(*args, **kwargs)
    finally:
        try:
            # Only for a single sampled transformer call; no hot-path fence otherwise.
            _cuda_sync()
            context.stop()
            operators = []
            for op in context.key_averages():
                name = str(op.key)
                device_us = float(getattr(op, "self_device_time_total",
                                  getattr(op, "self_cuda_time_total", 0)) or 0)
                cpu_us = float(getattr(op, "self_cpu_time_total", 0) or 0)
                operators.append({
                    "operator": name, "calls": int(getattr(op, "count", 0)),
                    "self_device_ms": round(device_us / 1000.0, 6),
                    "self_cpu_ms": round(cpu_us / 1000.0, 6),
                })
            operators.sort(key=lambda row: row["self_device_ms"], reverse=True)
            attention = [op for op in operators
                         if any(token in op["operator"].lower() for token in
                                ("scaled_dot_product", "efficient_attention",
                                 "flash_attention", "attention_forward"))]
            copy_ops = [op for op in operators
                        if any(token in op["operator"].lower() for token in
                               ("copy_", "to_copy", "memcpy", "to.device", "non_blocking"))]
            # These are PyTorch op attributions, not a full NVTX/PCIe transfer trace.
            if GEMM_CENSUS:
                _capture_gemm_census(context, phase, call)
            _FORWARD_SAMPLES.append({
                "phase": phase, "forward_index": call,
                "operator_count": len(operators),
                "top_device_operators": operators[:35],
                "attention_operators": attention[:30],
                "copy_related_operators": copy_ops[:30],
                "attention_self_device_ms": round(sum(x["self_device_ms"] for x in attention), 4),
                "copy_related_self_device_ms": round(sum(x["self_device_ms"] for x in copy_ops), 4),
                "op_self_device_sum_ms": round(sum(x["self_device_ms"] for x in operators), 4),
                "caveat": "One instrumented forward only. PyTorch ops may not expose native ctypes kernels, transfer streams or all CUDA work.",
            })
        except Exception as error:
            _PHASE_ERRORS.append(f"operator sample stop/report {phase}: {type(error).__name__}: {error}")
            try:
                context.stop()
            except Exception:
                pass
    return result


def _summary() -> None:
    if not ENABLED:
        return
    # The pipeline must mark 'finished' explicitly before returning. Do not
    # count process teardown / file encoding as video or audio VAE time.
    total = round(sum(p["wall_seconds"] for p in _PHASES), 6)
    phases = [p for p in _PHASES if p["phase"] not in ("finished",)]
    report = {
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "kind": "LTX-2.5 241-frame I2V whole-pipeline critical-path phase profiler",
        "status": "PASS" if len(phases) >= 7 and not _PHASE_ERRORS else "PARTIAL",
        "profiling_changes_scheduling": True,
        "comparable_to_uninstrumented_walltime": False,
        "phase_times_are_cpu_plus_gpu_critical_path_with_boundary_cuda_synchronizations": True,
        "phase_total_seconds": round(sum(p["wall_seconds"] for p in phases), 6),
        "phases": phases,
        "transformer_by_phase": _TRANSFORMER,
        "forward_counts": dict(_FORWARD_CALLS),
        "sampled_operator_forwards": _FORWARD_SAMPLES,
        "gemm_census_enabled": GEMM_CENSUS,
        "gemm_census_samples": _GEMM_CENSUS_SAMPLES,
        "peak_allocated_mib_at_boundaries": round(_PEAK_ALLOCATED / 2**20, 2),
        "peak_reserved_mib_at_boundaries": round(_PEAK_RESERVED / 2**20, 2),
        "errors": _PHASE_ERRORS,
        "limitations": [
            "Pipeline phases exclude model initial loading, CLI setup, file encoding and output saving.",
            "CUDA stream and memory-transfer overlap may change with boundary fences.",
            "Transformer forward events include transfer waits/enqueue gaps and cannot independently isolate pure compute.",
            "Only the second transformer forward of each phase is sampled by torch.profiler.",
            "Native ctypes Candidate E kernels and MMGP cross-stream transfers may not be fully attributed to PyTorch operations.",
            "Do not subtract accumulated profiler operator times from whole-pipeline wall time.",
        ],
    }
    try:
        PATH.parent.mkdir(parents=True, exist_ok=True)
        PATH.write_text(json.dumps(report, indent=2) + "\n")
        print("[LTX25_PIPELINE_PROFILE] " + json.dumps({
            "status": report["status"], "phase_total_seconds": report["phase_total_seconds"],
            "phases": [{k: v for k, v in p.items() if k in ("phase", "wall_seconds")}
                       for p in phases],
            "transformer": _TRANSFORMER,
            "operator_samples": len(_FORWARD_SAMPLES),
            "gemm_census_samples": len(_GEMM_CENSUS_SAMPLES),
            "errors": _PHASE_ERRORS, "report": str(PATH),
        }, sort_keys=True), flush=True)
    except Exception as error:
        print(f"[LTX25_PIPELINE_PROFILE_ERROR] {type(error).__name__}: {error}", flush=True)


if ENABLED:
    atexit.register(_summary)
