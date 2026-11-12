#!/usr/bin/env python3
"""Phase 0: non-mutating hardware, checkpoint-header and faithful rotation probe.

Designed to run inside the isolated V100 self-hosted runner. Does not download
models, print credentials, or treat synthetic shapes as real model observations.
This is an exploratory diagnostic, NOT an end-to-end ConvRot baseline.
"""
from __future__ import annotations

import ast
import datetime as dt
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("/cache/ltx25/results")


def command(argv: list[str]) -> dict:
    try:
        p = subprocess.run(argv, text=True, capture_output=True, timeout=15, check=False)
        return {"returncode": p.returncode, "stdout": p.stdout.strip()[:3000], "stderr": p.stderr.strip()[:600]}
    except Exception as exc:
        return {"error": str(exc)}


def github_source_rotation():
    """Extract unchanged WanGP rotation definitions without importing its UI/dependencies.

    Compile only these two functions from the checked-out source. The cache
    registration stub is identity, matching eager register_constant behavior.
    This checks the actual implementation, not a copied approximation.
    """
    source = ROOT / "shared/qtypes/int8_convrot.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    names = {"_regular_hadamard", "_rotate_activation"}
    fns = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in names]
    if len(fns) != 2:
        raise RuntimeError(f"Could not find both WanGP functions: {[x.name for x in fns]}")
    module = ast.Module(body=fns, type_ignores=[])
    ast.fix_missing_locations(module)

    class GraphStub:
        @staticmethod
        def register_constant(x):
            return x

    ns = {"math": math, "torch": torch, "_HADAMARD_CACHE": {}, "cuda_graphs": GraphStub()}
    exec(compile(module, str(source), "exec"), ns)
    return ns["_rotate_activation"]


def rotation_probe(rotate):
    results = []
    torch.manual_seed(20261112)
    for g in (16, 64, 256):
        for m in (16, 256):
            k = 4096
            item = {"source": "actual WanGP _rotate_activation", "synthetic": True,
                    "M": m, "K": k, "group_size": g, "input_dtype": "bfloat16"}
            try:
                x = torch.randn((m, k), device="cuda", dtype=torch.float32).bfloat16()
                # Distinct reference computed on the same BF16 values.
                with torch.no_grad():
                    y_ref = rotate(x.float(), g)
                    for _ in range(2):
                        _ = rotate(x, g)
                    torch.cuda.synchronize()
                    trials_ms = []
                    for _ in range(8):
                        start = torch.cuda.Event(enable_timing=True)
                        stop = torch.cuda.Event(enable_timing=True)
                        start.record()
                        y = rotate(x, g)
                        stop.record()
                        stop.synchronize()   # End of trial, not per inner operation.
                        trials_ms.append(float(start.elapsed_time(stop)))
                    d = (y.float() - y_ref).abs()
                    item.update(status="PASS", median_ms=round(sorted(trials_ms)[len(trials_ms)//2], 4),
                                trials_ms=[round(t, 4) for t in trials_ms],
                                p99_abs=float(torch.quantile(d.flatten(), .99).item()),
                                max_abs=float(d.max().item()),
                                nonfinite=int((~torch.isfinite(y)).sum().item()),
                                output_dtype=str(y.dtype))
            except Exception as exc:
                item.update(status="ERROR", error_type=type(exc).__name__, error=str(exc)[:1200])
            results.append(item)
            print("ROTATION_PROBE " + json.dumps(item), flush=True)
    return results


def checkpoint_header_report():
    # Explicit allowlisted roots only. No recursive filesystem search or downloads.
    roots = [Path(os.environ.get("LTX25_CHECKPOINT_ROOT", "/opt/ltx25/checkpoints"))]
    filename = "ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors"
    available = [(str(root / filename), (root / filename).is_file()) for root in roots]
    match = next((Path(p) for p, exists in available if exists), None)
    result = {"expected_filename": filename, "candidate_paths": available,
              "present": bool(match), "checkpoint_metadata_verified": False}
    if not match:
        result["blocker"] = "Checkpoint not visible to runner; the 17 model assets must be baked into /opt/ltx25/checkpoints in the Docker image."
        return result
    result["size_bytes"] = match.stat().st_size
    result["path"] = str(match)
    try:
        from safetensors import safe_open
        with safe_open(str(match), framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
            keys = f.keys()
            weights = [k for k in keys if k.endswith(".weight") and f.get_slice(k).get_dtype() == "I8"]
            sizes = []
            groups = {}
            for k in weights:
                shape = f.get_slice(k).get_shape()
                if len(shape) != 2:
                    continue
                n, inn = shape
                group = next((g for g in (256, 64, 16) if inn % g == 0), 0)
                groups[str(group)] = groups.get(str(group), 0) + 1
                sizes.append({"name": k, "N": n, "K": inn, "group_candidate": group,
                              "scale_present": k[:-7] + ".weight_scale" in keys})
            result.update(checkpoint_metadata_verified=True, num_tensors=len(keys),
                          quantization_format=meta.get("quantization_format"),
                          convrot_format=meta.get("convrot_format"),
                          int8_2d_weight_count=len(sizes), inferred_group_histogram=groups,
                          representative_weights=sizes[:12],
                          caveat="Shape-derived groups are candidates only; runtime config and fused splits require verification.")
    except Exception as exc:
        result.update(inventory_error=str(exc)[:1000], exception_type=type(exc).__name__)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.cuda.get_device_properties(0)
    report = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_sha": os.environ.get("GITHUB_SHA", "local"),
        "category": "Phase 0 exploratory hardware and source rotation probe",
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cuda_arch_list": torch.cuda.get_arch_list(),
        "visible_device_count": torch.cuda.device_count(),
        "gpu": {"name": dev.name, "capability": f"{dev.major}.{dev.minor}",
                "total_memory_bytes": dev.total_memory},
        "nvidia_smi": command(["nvidia-smi", "--query-gpu=name,uuid,driver_version,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu,memory.used,pcie.link.gen.current,pcie.link.width.current", "--format=csv,noheader"]),
        "nvcc": command(["nvcc", "--version"]),
        "checkpoint": checkpoint_header_report()
    }
    assert torch.cuda.device_count() == 1 and (dev.major, dev.minor) == (7, 0)
    report["rotation"] = rotation_probe(github_source_rotation())
    report["rotation_supported"] = all(x["status"] == "PASS" for x in report["rotation"])
    report["full_pipeline_measured"] = False
    report["actual_model_shapes_observed"] = False
    report["next_gate"] = "Actual checkpoint + runtime tensors, then offload/denoising profiler"
    path = OUT / f"phase0_probe_{report['source_sha'][:12]}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("PHASE0_REPORT " + json.dumps(report), flush=True)
    print(f"REPORT_PATH={path}", flush=True)
    print("PHASE0_PROBE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
