#!/usr/bin/env python3
"""Summarize the LTX-2.5 whole-pipeline critical-path run and two op samples.

No torch dependency. Phase wall times include boundary CUDA fences, so no
unprofiled-vs-profiled wall-clock speed claims are possible.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

STAGES = (
    "text_encoding", "stage1_conditioning", "stage1_denoising",
    "stage1_postprocessing", "spatial_upsampling", "stage2_conditioning",
    "stage2_denoising", "stage2_postprocessing",
    "video_vae_decoding", "audio_vae_decoding",
)


def render(payload: dict, output_dir: Path, process_seconds: float) -> str:
    if payload.get("status") != "PASS":
        raise ValueError(f"Pipeline reported {payload.get('status')!r}; errors: {payload.get('errors')}")
    phases = payload.get("phases", [])
    found = [p["phase"] for p in phases]
    if found != list(STAGES):
        raise ValueError(f"Unexpected phase boundaries {found!r}; expected {list(STAGES)!r}")
    if process_seconds <= 0:
        raise ValueError("Invalid measured subprocess time")
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_csv = output_dir / "pipeline-phases.csv"
    with phase_csv.open("w", newline="", encoding="utf-8") as fd:
        w = csv.DictWriter(fd, fieldnames=["phase", "wall_seconds",
                                            "cuda_allocated_mib_at_end",
                                            "cuda_reserved_mib_at_end"])
        w.writeheader()
        for row in phases:
            w.writerow(row)
    op_csv = output_dir / "pipeline-operator-samples.csv"
    with op_csv.open("w", newline="", encoding="utf-8") as fd:
        w = csv.DictWriter(fd, fieldnames=["phase", "forward_index", "operator",
                    "calls", "self_device_ms", "self_cpu_ms"])
        w.writeheader()
        for sample in payload.get("sampled_operator_forwards", []):
            for op in sample.get("top_device_operators", []):
                w.writerow({"phase": sample["phase"],
                            "forward_index": sample["forward_index"], **op})
    known = sum(float(p["wall_seconds"]) for p in phases)
    unknown = process_seconds - known
    lines = [
        "LTX-2.5 / V100 whole-pipeline I2V profiling",
        "===========================================",
        f"Instrumented WanGP process elapsed: {process_seconds:.2f} s",
        f"Measured pipeline phases (sync boundaries): {known:.2f} s",
        f"Outside pipeline (load / startup / save / teardown): {unknown:.2f} s",
        "CAUTION: The instrumented run cannot be compared with an uninstrumented",
        "benchmark. Model transfers/overlap may change when CUDA is synchronized.",
        "",
        "Major phase critical-path wall times:",
    ]
    for row in phases:
        seconds = float(row["wall_seconds"])
        frac = 100 * seconds / process_seconds
        lines.append(f"  {row['phase']:<25} {seconds:>9.2f} s  {frac:>5.1f}% of instrumented process")
    lines.extend(("", "Transformer-forward CUDA event timing (NOT pure CUDA compute):"))
    for phase in ("stage1", "stage2"):
        record = payload.get("transformer_by_phase", {}).get(phase, {})
        lines.append(f"  {phase}: {record.get('forward_calls', 'n/a')} forwards; "
                     f"{record.get('forward_cuda_event_seconds', 'n/a')} s CUDA stream-event sum")
    lines.extend(("", "One sampled PyTorch operator-profiled transformer forward per phase:"))
    sample_by_phase = {x["phase"]: x for x in payload.get("sampled_operator_forwards", [])}
    for phase in ("stage1", "stage2"):
        sample = sample_by_phase.get(phase)
        if not sample:
            lines.append(f"  {phase}: no operator sample (check profiler errors / GPU support)")
            continue
        lines.append(
            f"  {phase}: {sample['operator_count']} operators, "
            f"attention self-device {sample['attention_self_device_ms']:.3f} ms, "
            f"copy-related self-device {sample['copy_related_self_device_ms']:.3f} ms")
        for op in sample.get("top_device_operators", [])[:12]:
            lines.append(f"     {op['self_device_ms']:>9.3f} ms CUDA self  "
                         f"{op['calls']:>6} calls  {op['operator']}")
        if not sample.get("attention_operators"):
            lines.append("     No separately named SDPA operators in this sample; "
                         "they may be fused or unexposed.")
    if payload.get("errors"):
        lines.extend(("", "Profiler warnings / operator capture errors:"))
        lines.extend("  " + error for error in payload["errors"])
    lines.extend((
        "",
        "Limits: PyTorch ops can miss ctypes CUDA kernels and MMGP cross-stream",
        "copies. Reported 'copy-related' ops are NOT a PCIe traffic measurement.",
        "Times from per-forward op samples cannot be extrapolated across the run.",
        "The difference between process and pipeline is NOT idle GPU time;",
        "it includes model initialization, output processing and file saving.",
        "",
        f"Phase CSV: {phase_csv}",
        f"Operator samples CSV: {op_csv}",
        f"Raw JSON: {output_dir/'pipeline_profile.json'}",
    ))
    result = "\n".join(lines) + "\n"
    (output_dir / "pipeline-summary.txt").write_text(result, encoding="utf-8")
    return result


def main() -> int:
    if len(sys.argv) != 4:
        print("Usage: summarize_pipeline_profile.py profile.json OUTPUT_DIR process-seconds.txt",
              file=sys.stderr)
        return 2
    try:
        profile = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        seconds = float(Path(sys.argv[3]).read_text(encoding="utf-8").strip())
        print(render(profile, Path(sys.argv[2]), seconds), end="")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"PIPELINE_SUMMARY_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
