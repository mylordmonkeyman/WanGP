#!/usr/bin/env python3
"""P0 persistent *production CLI queue* benchmark: Hunyuan 1.5 V100 FP16.

One WanGP --process invocation receives a JSON LIST of settings; its
process_tasks_cli loops in one process while retaining wan_model, pipeline,
MMGP state and encoders. The first completed video is warm-up, never a hot
measurement. No subprocess per measured generation. No downloads.

Do not interpret unsynchronized per-phase CPU markers as GPU phase timings.
Use --diagnostic for synchronized phase diagnosis only; acceptance hot E2E
samples come from the non-diagnostic five-post-warm-up run.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import v100_bar_video_smoke as base
from v100_t2v_cached_steady_state import validate_clip

MODE = "t2v"
BRANCH = "optimization/hunyuan15-v100-fp16"
SOURCE_SHA = "7b4c3f5ab3595eff14450eeb2549779931ede405"
CHECKPOINT = base.WRITABLE_CHECKPOINTS / "hunyuanvideo1.5_480p_t2v_fp16.safetensors"
CONVERTED_LORA = (base.CACHE / "results" /
    "hunyuan15-converted-lora-43951751237-attempt1" / "converted_loras" /
    "hunyuan15_t2v_lightx2v4_rank32_wangp_qkv_split_bf16.safetensors")
OUT = base.OUT / "p0"
EXPECTED_MODEL = "hunyuan15_v100_t2v_480_fp16_experiment"


def settings(prompt=None):
    result = base.make_settings(MODE)
    result.update({
        "model_type": EXPECTED_MODEL,
        "prompt": base.PROMPT if prompt is None else prompt,
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
        "flow_shift": 8.0,
        "activated_loras": [CONVERTED_LORA.name],
        "loras_multipliers": "1.0",
    })
    assert result["seed"] == 20261112
    assert result["video_length"] == 33
    assert result["resolution"] == "832x480"
    return result


def prepare(out: Path, measured: int, changed_prompt=False):
    if measured < 1:
        raise ValueError("Need at least one post-warm-up generation")
    # Only the prompt changes, never the sampling configuration.
    values = [settings() for _ in range(measured + 1)]
    if changed_prompt:
        values[-1] = settings(base.PROMPT + " The camera remains stationary.")
    out.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(base.select_template())
    config.update({
        "checkpoints_paths": [
            str(base.WRITABLE_CHECKPOINTS), str(base.MODEL_DIR),
            str(base.EXTRA_CHECKPOINTS),
        ],
        "transformer_quantization": "bf16",  # WanGP unquantized selector
        "text_encoder_quantization": "int8",
        "transformer_types": [EXPECTED_MODEL],
        "profile": 0, "video_profile": 0,
        "attention_mode": "sdpa", "compile": "disabled",
        "vae_config": 0,
        "save_path": str(out / "videos"),
        "image_save_path": str(out / "videos"),
        "audio_save_path": str(out / "videos"),
        "prevent_power_throttling": False,
        "mixed_precision": "0",
    })
    for folder in ("config", "videos", "trace"):
        (out / folder).mkdir(exist_ok=True)
    (out / "config" / "wgp_config.json").write_text(
        json.dumps(config, indent=2) + "\n")
    (out / "settings.json").write_text(json.dumps(values, indent=2) + "\n")
    (out / "lora_paths.json").write_text(
        json.dumps({"hunyuan_1_5": str(CONVERTED_LORA.parent)}, indent=2) + "\n")
    return values


def md_spread(samples):
    center = statistics.median(samples)
    return statistics.median(abs(x - center) for x in samples)


def collect(out: Path, expected: int, diagnostic: bool):
    path = out / "trace" / "requests.jsonl"
    records = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    if len(records) != expected + 1:
        raise RuntimeError(f"Expected {expected + 1} completed tasks, got {len(records)}")
    if any(r["error"] for r in records):
        raise RuntimeError("Generation failed: " + str([r["error"] for r in records]))
    if len({r["pid"] for r in records}) != 1:
        raise RuntimeError("Not a persistent WanGP process")
    for key in ("model_object_id", "pipeline_object_id",
                "transformer_object_id", "offload_object_id"):
        if None in [r[key] for r in records] or len({r[key] for r in records}) != 1:
            raise RuntimeError("State reconstruction detected: " + key)
    if {r["model_loads_cumulative"] for r in records} != {1}:
        raise RuntimeError("Unexpected checkpoint/model reload during hot queue")
    if any(len(r["placements"]) != 54 for r in records):
        raise RuntimeError("Expected 54 Hunyuan double-stream block placements")
    if any(r["resolved_attention"] != "sdpa" for r in records):
        raise RuntimeError("Unexpected resolved attention; expected SDPA")
    clips = sorted((out / "videos").rglob("*.mp4"))
    if len(clips) != len(records):
        raise RuntimeError(
            f"Expected {len(records)} unique MP4s, got {len(clips)}; output overwrite?")
    clips_checked = [validate_clip(x) for x in clips]
    results = records[1:]  # discard the *entire* first generation
    samples = [r["hot_request_seconds"] for r in results]
    baseline = {
        "status": "P0_DIAGNOSTIC_ONLY" if diagnostic else "P0_HOT_E2E_READY",
        "source_sha": SOURCE_SHA,
        "branch_required": BRANCH,
        "actual_sha": os.environ.get("GITHUB_SHA"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "effective_settings": settings(),
        "requested_attention": "sdpa",
        "resolved_attention": records[-1]["resolved_attention"],
        "process_ids": [r["pid"] for r in records],
        "stable_pipeline_and_transformer": True,
        "model_load_count": records[-1]["model_loads_cumulative"],
        "cold_initial_generation_seconds": records[0]["hot_request_seconds"],
        "warmup_generations": 1,
        "measured_generations": len(samples),
        "request_e2e_median_seconds": statistics.median(samples),
        "request_e2e_mad_seconds": md_spread(samples),
        "request_e2e_samples_seconds": samples,
        "phase_times_are_diagnostic_only": diagnostic,
        "phase_samples": [r["phase_seconds"] for r in results],
        "peak_vram_bytes": max(r["cuda_peak_allocated_bytes"] for r in records),
        "blocks_placement_end": [r["placements"] for r in records],
        "clips": clips_checked,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": base.hash_file(CHECKPOINT),
        "converted_lora": str(CONVERTED_LORA),
        "converted_lora_sha256": base.hash_file(CONVERTED_LORA),
        "unverified_p0_metrics": [
            "live per-step H2D bytes/transfer waits",
            "pin pool allocation/reuse identities",
            "Q/K pre-RMSNorm overflow incidence",
            "actual attention kernel and padded/nonpadded Q/K lengths",
            "executed VAE tile and decoder call counts",
        ],
        "note": "P0 gate remains blocked for unverified metrics. A single PID alone is not used to assert residency.",
    }
    (out / ("diagnostic.json" if diagnostic else "baseline.json")).write_text(
        json.dumps(baseline, indent=2) + "\n")
    return baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--changed-prompt", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    measured = 1 if args.diagnostic else 5
    out = OUT / ("diagnostic" if args.diagnostic else "hot")
    values = prepare(out, measured, args.changed_prompt)
    if args.plan_only:
        print("P0_PLAN_PASS " + json.dumps({
            "output": str(out), "tasks": len(values),
            "settings": values[0], "diagnostic": args.diagnostic,
        }, sort_keys=True))
        return
    base.gpu_preflight()
    if not CHECKPOINT.is_file() or CHECKPOINT.stat().st_size < 15_000_000_000:
        raise RuntimeError("Missing cached original Comfy-Org FP16 base: " + str(CHECKPOINT))
    if not CONVERTED_LORA.is_file() or CONVERTED_LORA.stat().st_size < 300_000_000:
        raise RuntimeError("Missing previously converted rank-32 LoRA: " + str(CONVERTED_LORA))
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(base.ROOT),
        "HUNYUAN15_P0_TRACE_DIR": str(out / "trace"),
        "HUNYUAN15_P0_DIAGNOSTIC": "1" if args.diagnostic else "0",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "GRADIO_ANALYTICS_ENABLED": "False",
        "WAN2GP_LTX2_SKIP_BAKED_LORA_MIGRATION": "1",
        "OMP_NUM_THREADS": "8",
    })
    command = [
        base.PYTHON, "-u", "wgp.py", "--offline",
        "--config", str(out / "config"), "--process", str(out / "settings.json"),
        "--output-dir", str(out / "videos"),
        "--profile", "0", "--attention", "sdpa", "--fp16",
        "--lora-config", str(out / "lora_paths.json"),
        "--no-prevent-power-throttling", "--verbose", "1",
    ]
    print("P0_EXECUTION " + json.dumps({
        "command": command, "tasks": len(values),
        "process_count": 1, "diagnostic": args.diagnostic,
    }), flush=True)
    result = base.run_command(command, out / "generation.log", env, timeout=3600)
    if result["returncode"]:
        raise RuntimeError("Persistent production CLI queue failed: " + str(result))
    report = collect(out, measured, args.diagnostic)
    print("P0_PERSISTENT_QUEUE_PASS " + json.dumps({
        "median_hot_e2e": report["request_e2e_median_seconds"],
        "samples": report["request_e2e_samples_seconds"],
        "stable_model": report["stable_pipeline_and_transformer"],
        "diagnostic": args.diagnostic,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
