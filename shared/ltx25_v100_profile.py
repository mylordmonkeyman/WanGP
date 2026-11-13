"""Opt-in single-forward SM70 profiler for the EXISTING BF16 WanGP path.

No changes to math or storage. Profiling is always disabled by default.
The selected forward is instrumented and synchronized only at its END.
Profiler output is diagnostic, not an unperturbed timing benchmark.
"""
from __future__ import annotations

from contextlib import nullcontext
import json
import os
from pathlib import Path
import time
import traceback

_TRACE_ACTIVE = False
_CALLS = 0
_EMITTED = False
_ENABLED = os.environ.get("WAN2GP_LTX25_PROFILE", "") == "1"
_CALL_INDEX = int(os.environ.get("WAN2GP_LTX25_PROFILE_CALL", "2")) if _ENABLED else 2


def scope(name: str):
    """Annotate only during the opt-in trace; zero CUDA sync in a hot path."""
    if not _TRACE_ACTIVE:
        return nullcontext()
    from torch.profiler import record_function
    return record_function(name)


def profiled_convrot_forward(original):
    """Decorate ConvRot's full forward only for the isolated diagnostic job."""
    if not _ENABLED:
        return original

    def wrapped(self, *args, **kwargs):
        with scope("ltx25.convrot.linear_full"):
            return original(self, *args, **kwargs)
    return wrapped


def profiled_attention_forward(original):
    """Label full LTX attention modules, including their Q/K/V projection."""
    if not _ENABLED:
        return original

    def wrapped(self, *args, **kwargs):
        with scope("ltx25.attention.full_module"):
            return original(self, *args, **kwargs)
    return wrapped

def _save(p, *, call_index: int, elapsed: float, exception=None):
    import torch
    target = Path(os.environ.get("LTX25_PROFILE_DIR", "/tmp/ltx25-actual-attention-profile"))
    target.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "FAIL" if exception else "PASS",
        "profiled_forward_index": call_index,
        "forward_wall_s_with_profiling": elapsed,
        "not_baseline_timing": True,
        "note": "Kineto trace from ONE actual LTXModel.forward after the previous model call. "
                "Profiler overhead and end-of-forward synchronization are intentional. "
                "Named GPU event times are inclusive and may overlap; do not sum "
                "them into Amdahl percentages without trace correlation.",
        "model_dtype_policy": "bf16 (unchanged)",
        "convrot_candidate_e": os.environ.get("WAN2GP_LTX25_CUBLAS_E_MODE", ""),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "source_sha": os.environ.get("GITHUB_SHA"),
        "exception": exception,
        "named_scopes": [],
        "top_ops_by_cuda_us": [],
        "kernels": [],
        "trace_created": False,
    }
    try:
        events = p.key_averages()
        for e in events:
            record = {
                "name": e.key,
                "count": int(e.count),
                "cuda_total_us": round(float(e.device_time_total), 3),
                "cuda_self_us": round(float(e.self_device_time_total), 3),
                "cpu_total_us": round(float(e.cpu_time_total), 3),
                "cpu_self_us": round(float(e.self_cpu_time_total), 3),
            }
            if e.key.startswith("ltx25."):
                report["named_scopes"].append(record)
            report["top_ops_by_cuda_us"].append(record)
        report["top_ops_by_cuda_us"].sort(key=lambda r: -r["cuda_self_us"])
        report["top_ops_by_cuda_us"] = report["top_ops_by_cuda_us"][:100]
        distinct_kernels = {}
        for e in p.events():
            if "cuda" not in str(e.device_type).lower() and not str(e.name).startswith("Memcpy "):
                continue
            row = distinct_kernels.setdefault(str(e.name), {"name": str(e.name), "count": 0, "device_us": 0.0})
            row["count"] += 1
            row["device_us"] += float(e.device_time_total)
        report["kernels"] = sorted(distinct_kernels.values(), key=lambda r: -r["device_us"])[:100]
        trace = target / "actual-ltx25-one-forward-chrome.json"
        p.export_chrome_trace(str(trace))
        report["trace_created"] = trace.is_file()
        report["trace_bytes"] = trace.stat().st_size if trace.is_file() else 0
        (target / "operator-table.txt").write_text(
            events.table(sort_by="self_device_time_total", row_limit=100))
    except Exception as exc:
        report["status"] = "FAIL"
        report["save_error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()[-3000:]
    (target / "actual-forward-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("LTX25_PROFILE_SUMMARY " + json.dumps({
        "status": report["status"],
        "profiled_forward_index": call_index,
        "named_scopes": report["named_scopes"],
        "trace_bytes": report["trace_bytes"] if "trace_bytes" in report else None,
        "top_gpu_ops": report["top_ops_by_cuda_us"][:8],
        "error": report.get("save_error") or exception,
    })[:12000], flush=True)
    return report


def profile_one_forward(thunk):
    """Called on LTXModel.forward, preserving the original execution thunk."""
    global _CALLS, _TRACE_ACTIVE, _EMITTED
    if not _ENABLED or _EMITTED:
        return thunk()
    _CALLS += 1
    if _CALLS != _CALL_INDEX:
        return thunk()
    if _CALL_INDEX < 2:
        raise ValueError("Require at least one non-profiled warmup LTXModel.forward")
    from torch.profiler import ProfilerActivity, profile, record_function
    import torch
    _EMITTED = True  # at most one profile, including on failure
    result = None
    error = None
    started = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, profile_memory=False, with_stack=False) as p:
        try:
            _TRACE_ACTIVE = True
            with record_function("ltx25.transformer.one_forward"):
                result = thunk()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            _TRACE_ACTIVE = False
            # Trial-boundary synchronization ONLY, never per attention/linear.
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    report = _save(p, call_index=_CALLS, elapsed=round(elapsed, 5), exception=error)
    if error:
        raise RuntimeError(f"Profiling forward failed: {error}")
    if report["status"] != "PASS" or not report["trace_created"]:
        raise RuntimeError(f"Unable to save LTX-2.5 profiler: {report.get('save_error')}")
    return result
