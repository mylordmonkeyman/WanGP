#!/usr/bin/env python3
"""Three-way *full WanGP dispatcher* attention benchmark for sm_70.

This script IS GPU work. It must not be executed until the user authorizes the
manual/signal-gated V100 GitHub Actions workflow. Inputs are synthetic Q/K/V
using LTX-like shapes, not checkpoint-extracted activation tensors.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from shared.attention import pay_attention
from shared.ltx25_v100_attention import configure as configure_pilot

BACKENDS = ("sdpa", "1cat", "zhinianqin")
FORCE = {"sdpa": "sdpa", "1cat": "ltx25_v100_1cat",
         "zhinianqin": "ltx25_v100_zhinianqin"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", type=Path,
                        default=Path(__file__).with_name("three_way_shapes.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--max-input-mib", type=int, default=512)
    args = parser.parse_args()
    if not 0 <= args.warmup <= 50 or not 1 <= args.iterations <= 100 or not 1 <= args.rounds <= 10:
        parser.error("warmup must be 0..50; iterations 1..100; rounds 1..10")
    if args.max_input_mib < 1 or args.max_input_mib > 2048:
        parser.error("max-input-mib must be 1..2048")
    return args


def dispatch(which, q, k, v):
    """Exercise the real WanGP pay_attention route, not the raw CUDA op."""
    qkv = [q, k, v]
    result = pay_attention(qkv, force_attention=FORCE[which])
    if qkv:
        raise RuntimeError(f"{which} did not consume the qkv list")
    if not isinstance(result, torch.Tensor) or result.shape != q.shape:
        raise RuntimeError(f"{which} returned wrong shape/type: {type(result)}")
    if result.dtype != q.dtype:
        raise RuntimeError(f"{which} returned {result.dtype}, expected {q.dtype}")
    return result


def validate_case(raw, max_input_mib):
    required = ("name", "batch", "tokens", "heads", "head_dim")
    if not isinstance(raw, dict) or any(key not in raw for key in required):
        raise ValueError(f"Each case needs {required}: {raw!r}")
    label = raw["name"]
    if not isinstance(label, str) or not label or len(label) > 100:
        raise ValueError(f"Invalid case label: {label!r}")
    for key in ("batch", "tokens", "heads", "head_dim"):
        if type(raw[key]) is not int or raw[key] < 1:
            raise ValueError(f"{label}: {key} must be a positive integer")
    b, n, h, d = (raw[key] for key in required[1:])
    if d not in (64, 128) or n < 1024 or n > 16384 or h > 64 or b > 4:
        raise ValueError(f"{label}: outside gated SM70 FP16 benchmark shape limits")
    # 3 FP16 input tensors + 1 output, conservative preflight before GPU allocation.
    bounded_mib = (4 * b * n * h * d * 2) / (1024 ** 2)
    if bounded_mib > max_input_mib:
        raise ValueError(f"{label}: QKV+output {bounded_mib:.1f} MiB exceeds {max_input_mib} MiB limit")
    return dict(name=label, batch=b, tokens=n, heads=h, head_dim=d,
                estimated_qkv_output_mib=round(bounded_mib, 1))


def accuracy(ref, got):
    # Comparisons deliberately occur outside the timed section.
    diff = (ref.float() - got.float()).abs()
    max_abs = float(diff.max().item())
    rms = float(diff.square().mean().sqrt().item())
    is_finite = bool(torch.isfinite(got).all().item())
    return {"max_abs": max_abs, "rms": rms, "finite": is_finite,
            "pass": is_finite and max_abs <= 0.05 and rms <= 0.005}


def timed_round(which, q, k, v, warmup, iterations):
    for _ in range(warmup):
        result = dispatch(which, q, k, v)
        del result
    torch.cuda.synchronize(q.device)
    gc.collect()
    # Inputs are already resident. Peak memory is *incremental over Q/K/V*;
    # record total peak as well, to expose layout copies and scratch.
    before = torch.cuda.memory_allocated(q.device)
    torch.cuda.reset_peak_memory_stats(q.device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    host_start = time.perf_counter()
    start.record()
    for _ in range(iterations):
        result = dispatch(which, q, k, v)
        del result
    end.record()
    end.synchronize()
    host_ms = (time.perf_counter() - host_start) * 1000 / iterations
    cuda_ms = start.elapsed_time(end) / iterations
    peak = torch.cuda.max_memory_allocated(q.device)
    return {
        "wall_ms": round(host_ms, 5),
        "cuda_event_ms": round(cuda_ms, 5),
        "peak_allocated_mib": round(peak / 1048576, 3),
        "incremental_peak_mib": round(max(0, peak - before) / 1048576, 3),
        "baseline_allocated_mib": round(before / 1048576, 3),
    }


def markdown(report):
    lines = [
        "## LTX-2.5 SM70 three-way attention benchmark",
        "",
        f"Commit: {report['git_sha']} | Torch: {report['torch']} | GPU: {report['gpu']}",
        "",
        "| Shape | SDPA wall (ms) | 1Cat wall (ms) | zhinianqin wall (ms) | 1Cat speedup | zhinianqin speedup |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case in report["cases"]:
        data = case.get("results", {})
        times = {k: data.get(k, {}).get("median_wall_ms") for k in BACKENDS}
        def display(value):
            return "FAIL" if value is None else f"{value:.3f}"
        def speedup(name):
            base, sample = times["sdpa"], times[name]
            return "n/a" if base is None or sample is None or sample <= 0 else f"{base/sample:.2f}x"
        lines.append(
            f"| {case['name']} | {display(times['sdpa'])} | {display(times['1cat'])} | "
            f"{display(times['zhinianqin'])} | {speedup('1cat')} | {speedup('zhinianqin')} |"
        )
    lines += ["", "Only unmasked FP16 self-attention on synthetic LTX-like shapes; "
              "not a video-generation benchmark, and not a verified LTX-2.5 shape trace.",
              "Wall numbers include the WanGP dispatcher and adapter copies. "
              "All accuracy tests compare with SDPA outside timed sections."]
    return "\n".join(lines) + "\n"


def write_results(report, dest):
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "three_way_report.json").write_text(json.dumps(report, indent=2) + "\n")
    summary = markdown(report)
    (dest / "three_way_summary.md").write_text(summary)
    with (dest / "three_way_measurements.csv").open("w", newline="") as outfile:
        fields = ["shape", "backend", "round", "wall_ms", "cuda_event_ms",
                  "peak_allocated_mib", "incremental_peak_mib", "baseline_allocated_mib"]
        writer = csv.DictWriter(outfile, fields)
        writer.writeheader()
        for case in report["cases"]:
            for backend, entry in case.get("results", {}).items():
                for i, measurement in enumerate(entry.get("measurements", [])):
                    writer.writerow({"shape": case["name"], "backend": backend, "round": i + 1,
                                     **measurement})
    print(summary, flush=True)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # No CUDA work is performed until invocation of this manually gated script.
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("Must run on a Tesla V100 (sm_70)")
    if not torch.__version__.startswith("2.10.0+cu128"):
        raise RuntimeError(f"Pre-baked PyTorch 2.10.0+cu128 required, got {torch.__version__}")

    # All package imports must resolve from the two *isolated build directories*.
    import flash_attn_v100
    import vllm_flash_attn
    assert callable(flash_attn_v100.flash_attn_func)
    assert callable(vllm_flash_attn.flash_attn_varlen_func)

    config = json.loads(args.shapes.read_text())
    cases = [validate_case(x, args.max_input_mib) for x in config["cases"]]
    if len(cases) < 2 or len(cases) > 20 or len(set(x["name"] for x in cases)) != len(cases):
        raise ValueError("Shape file must define 2..20 distinct cases")
    configure_pilot("off", 1024)  # Force explicit mode only inside dispatch.
    report = {
        "status": "running",
        "git_sha": os.environ.get("GITHUB_SHA", "local"),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_capability": list(torch.cuda.get_device_capability()),
        "source_note": config.get("description"),
        "backends": list(BACKENDS),
        "params": {"warmup": args.warmup, "iterations": args.iterations,
                   "rounds": args.rounds, "seed": args.seed},
        "cases": [],
        "errors": [],
    }
    errors = []
    try:
        with torch.inference_mode():
            for index, case in enumerate(cases):
                entry = {"name": case["name"], "shape": case, "results": {}}
                report["cases"].append(entry)
                b, n, h, d = (case[key] for key in ("batch", "tokens", "heads", "head_dim"))
                torch.manual_seed(args.seed + index)
                q = torch.randn((b, n, h, d), device="cuda", dtype=torch.float16)
                k = torch.randn_like(q)
                v = torch.randn_like(q)

                # Reference output and all accuracy checks precede measurements.
                ref = dispatch("sdpa", q, k, v)
                for backend in BACKENDS:
                    detail = {"measurements": []}
                    entry["results"][backend] = detail
                    try:
                        got = dispatch(backend, q, k, v)
                        check = accuracy(ref, got)
                        detail["accuracy"] = check
                        del got
                        if not check["pass"]:
                            raise AssertionError(
                                f"{case['name']} {backend} accuracy mismatch: {check}"
                            )
                    except Exception as exc:
                        detail["error"] = f"{type(exc).__name__}: {exc}"
                        errors.append(detail["error"] + f" / {case['name']}")
                del ref
                torch.cuda.synchronize()
                # Rotate call order every round to reduce warmup/clock bias.
                for round_index in range(args.rounds):
                    order = BACKENDS[round_index % 3:] + BACKENDS[:round_index % 3]
                    for backend in order:
                        detail = entry["results"][backend]
                        if "error" in detail:
                            continue
                        try:
                            sample = timed_round(backend, q, k, v, args.warmup, args.iterations)
                            detail["measurements"].append(sample)
                        except Exception as exc:
                            detail["error"] = f"{type(exc).__name__}: {exc}"
                            errors.append(detail["error"] + f" / {case['name']}")
                for backend in BACKENDS:
                    detail = entry["results"][backend]
                    if detail["measurements"] and "error" not in detail:
                        samples = detail["measurements"]
                        detail["median_wall_ms"] = round(
                            statistics.median(x["wall_ms"] for x in samples), 5
                        )
                        detail["median_cuda_event_ms"] = round(
                            statistics.median(x["cuda_event_ms"] for x in samples), 5
                        )
                        detail["max_incremental_peak_mib"] = max(
                            x["incremental_peak_mib"] for x in samples
                        )
                print(f"Completed shape {case['name']} (case {index+1}/{len(cases)})", flush=True)
                del q, k, v
                gc.collect()
                torch.cuda.empty_cache()
                write_results(report, args.output_dir)
    except Exception as exc:
        errors.append(f"fatal: {type(exc).__name__}: {exc}")
        report["traceback"] = traceback.format_exc()
    report["errors"] = errors
    report["status"] = "FAIL" if errors else "PASS"
    write_results(report, args.output_dir)
    if errors:
        raise SystemExit("Benchmark failed: " + "; ".join(errors[:5]))


if __name__ == "__main__":
    main()
