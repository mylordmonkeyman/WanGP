#!/usr/bin/env python3
"""Paired cold-process T2V benchmark: BF16-stored versus preconverted FP16 LoRA.

Runs original FP16 Hunyuan 1.5 T2V + the same WanGP-formatted LightX2V
rank-32 LoRA and same 4-step bar-scene settings. The reference adapter is
stored BF16 and cast to FP16 by MMGP on load. The experimental adapter is
converted ONCE to FP16 on disk before any timed inference process starts.

ABBA/BA order (BF16, FP16, FP16, BF16, FP16, BF16), new WanGP subprocess
for each trial, shared V100, no network, matched model/prompt/seed/schedule.
Dry runs, one-time conversion, bytewise verification, report aggregation,
and MP4 frame validation are excluded from every reported wall time.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import v100_bar_video_smoke as base
from v100_t2v_cached_steady_state import validate_clip

MODE = "t2v"
MODEL_TYPE = "hunyuan15_v100_t2v_480_fp16_experiment"
BASE_CHECKPOINT = base.WRITABLE_CHECKPOINTS / "hunyuanvideo1.5_480p_t2v_fp16.safetensors"
BF16_NAME = "hunyuan15_t2v_lightx2v4_rank32_wangp_qkv_split_bf16.safetensors"
FP16_NAME = "hunyuan15_t2v_lightx2v4_rank32_wangp_qkv_split_fp16.safetensors"
BF16_DIR = (
    base.CACHE / "results" / "hunyuan15-converted-lora-43951751237-attempt1"
    / "converted_loras"
)
BF16_FILE = BF16_DIR / BF16_NAME
PREPARED_DIR = base.OUT / "prepared_fp16_lora"
FP16_FILE = PREPARED_DIR / FP16_NAME
OUT_ROOT = base.OUT
ORDER = ("bf16", "fp16", "fp16", "bf16", "fp16", "bf16")
PRIOR_BF16_SECONDS = 62.7


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preconvert_and_verify():
    """All work here is *outside* the timed WanGP generation subprocesses."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if not BF16_FILE.is_file() or BF16_FILE.stat().st_size < 300_000_000:
        raise RuntimeError(f"Missing previously converted, BF16-stored adapter: {BF16_FILE}")
    if not BASE_CHECKPOINT.is_file() or BASE_CHECKPOINT.stat().st_size < 15_000_000_000:
        raise RuntimeError(f"Missing original FP16 base checkpoint: {BASE_CHECKPOINT}")

    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    dst = {}
    source_metadata = {}
    with safe_open(str(BF16_FILE), framework="pt", device="cpu") as f:
        source_metadata = f.metadata() or {}
        for name in f.keys():
            tensor = f.get_tensor(name)
            if tensor.dtype != torch.bfloat16:
                raise RuntimeError(f"Input tensor not BF16: {name} {tensor.dtype}")
            cast = tensor.to(dtype=torch.float16)
            dst[name] = cast.contiguous()
    if len(dst) != 2357:
        raise RuntimeError(f"Expected 2357 tensors, found {len(dst)}")
    metadata = {
        **source_metadata,
        "preconverted_dtype": "F16",
        "preconverted_from": BF16_NAME,
        "conversion_type": "BF16->FP16 via torch.Tensor.to(dtype=torch.float16)",
    }
    save_file(dst, str(FP16_FILE), metadata=metadata)
    del dst

    # Check all elements, not just shape, sample, or an aggregate checksum.
    checked_tensors = 0
    checked_elements = 0
    with safe_open(str(BF16_FILE), framework="pt", device="cpu") as src, \
            safe_open(str(FP16_FILE), framework="pt", device="cpu") as prepared:
        if set(src.keys()) != set(prepared.keys()):
            raise RuntimeError("FP16 safetensors keys differ from source")
        for name in src.keys():
            expected = src.get_tensor(name).to(dtype=torch.float16)
            actual = prepared.get_tensor(name)
            if actual.dtype != torch.float16 or not torch.equal(actual, expected):
                raise RuntimeError(f"Preconverted tensor differs from runtime cast: {name}")
            checked_tensors += 1
            checked_elements += actual.numel()
    report = {
        "source_bf16": str(BF16_FILE),
        "source_bf16_sha256": sha256(BF16_FILE),
        "preconverted_fp16": str(FP16_FILE),
        "preconverted_fp16_sha256": sha256(FP16_FILE),
        "preconverted_file_bytes": FP16_FILE.stat().st_size,
        "identical_to_runtime_cast": True,
        "verified_tensors": checked_tensors,
        "verified_elements": checked_elements,
        "one_time_convert_and_verify_seconds_EXCLUDED_FROM_BENCHMARK":
            round(time.monotonic() - started, 2),
    }
    (OUT_ROOT / "preconversion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("HUNYUAN15_FP16_ON_DISK_PARITY_PASS " + json.dumps(report, sort_keys=True), flush=True)
    return report


def main():
    base.gpu_preflight()  # Reject a busy V100, preserving unrelated runner work.
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    report = preconvert_and_verify()
    assert base.MODEL_TYPES[MODE] == MODEL_TYPE, "Unexpected T2V model definition"
    assert base.MODEL_FILES[MODE] == BASE_CHECKPOINT.name, "Unexpected T2V weights"

    # Mutate only this imported benchmark module's per-run settings.
    original_make_settings = base.make_settings
    original_run_command = base.run_command
    active = {"name": None, "config": None}

    def settings_for_test(mode):
        if mode != MODE:
            raise RuntimeError("Only T2V is supported")
        settings = original_make_settings(mode)
        settings.update({
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
            "flow_shift": 8.0,
            "activated_loras": [active["name"]],
            "loras_multipliers": "1.0",
        })
        return settings

    def run_offline(command, log_path, env, timeout):
        command = list(command)
        if "wgp.py" in command:
            command.insert(command.index("wgp.py") + 1, "--offline")
            command.extend(["--lora-config", str(active["config"])])
        env = dict(env)
        env.update({
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        })
        return original_run_command(command, log_path, env, timeout)

    base.make_settings = settings_for_test
    base.run_command = run_offline
    os.environ["HUNYUAN15_ALLOW_READONLY_CACHE"] = "1"

    trials = []
    for index, dtype in enumerate(ORDER, 1):
        source_dir = BF16_DIR if dtype == "bf16" else PREPARED_DIR
        name = BF16_NAME if dtype == "bf16" else FP16_NAME
        file_path = source_dir / name
        if not file_path.is_file():
            raise RuntimeError(f"Missing LoRA for trial {index}: {file_path}")
        # Crucial: each condition has a fresh, separate Python/WanGP process.
        # base.run_mode times only the generation subprocess, not dry-run or
        # preparation; no one-time BF16->FP16 conversion occurs in this loop.
        trial_dir = OUT_ROOT / "trials" / f"{index:02d}_{dtype}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        lora_config = trial_dir / "lora_paths.json"
        lora_config.write_text(json.dumps({"hunyuan_1_5": str(source_dir)}, indent=2) + "\n")
        base.OUT = trial_dir
        active.update({"name": name, "config": lora_config})
        print("HUNYUAN15_TRIAL_START " + json.dumps({
            "index": index, "stored_dtype": dtype,
            "lora": str(file_path),
            "seed": base.SEED, "frames": base.FRAMES,
            "resolution": base.RESOLUTION, "steps": 4,
            "guidance": 1.0, "flow_shift": 8.0,
        }, sort_keys=True), flush=True)
        result = base.run_mode(MODE)
        log = (trial_dir / MODE / "logs" / "generation.log").read_text(errors="replace")
        expected = f"Lora '{file_path}' was loaded in model"
        if expected not in log:
            raise RuntimeError(f"Adapter not confirmed loaded during trial {index}")
        if f"Loading Model '{BASE_CHECKPOINT}'" not in log:
            raise RuntimeError(f"Wrong base model used in trial {index}")
        if "EXPERIMENTAL Hunyuan 1.5 native FP16 enabled" not in log:
            raise RuntimeError(f"FP16 execution path disabled in trial {index}")
        if any(marker in log for marker in (
            "contains unexpected module keys", "Download URL:",
            "Downloading Model", "Downloading Lora", "Unable to Download",
        )):
            raise RuntimeError(f"Unexpected download or LoRA mismatch during trial {index}")
        settings = json.loads((trial_dir / MODE / "settings.json").read_text())
        if settings["model_type"] != MODEL_TYPE or \
                settings["activated_loras"] != [name] or \
                settings["loras_multipliers"] != "1.0" or \
                settings["num_inference_steps"] != 4 or \
                settings["guidance_scale"] != 1.0 or settings["flow_shift"] != 8.0:
            raise RuntimeError(f"Unmatched generation settings in trial {index}: {settings}")
        clips = [validate_clip(Path(video["name"])) for video in result["videos"]]
        if not clips or any(not clip["decode_pass"] or clip["decoded_frames"] != base.FRAMES
                            for clip in clips):
            raise RuntimeError(f"Invalid MP4 output on trial {index}")
        trial = {
            "index": index,
            "stored_dtype": dtype,
            "lora": str(file_path),
            "generation_wall_seconds": result["run"]["seconds"],
            "clips": clips,
            "settings": settings,
            "run": result["run"],
        }
        trials.append(trial)
        print("HUNYUAN15_LORA_AB_TRIAL_PASS " + json.dumps({
            "index": index, "stored_dtype": dtype,
            "seconds": trial["generation_wall_seconds"],
            "decoded_frames": clips[0]["decoded_frames"],
        }, sort_keys=True), flush=True)
        (OUT_ROOT / "partial_trials.json").write_text(json.dumps(trials, indent=2) + "\n")

    medians = {
        dtype: statistics.median(t["generation_wall_seconds"]
                                 for t in trials if t["stored_dtype"] == dtype)
        for dtype in ("bf16", "fp16")
    }
    means = {
        dtype: round(statistics.mean(t["generation_wall_seconds"]
                                     for t in trials if t["stored_dtype"] == dtype), 3)
        for dtype in ("bf16", "fp16")
    }
    diff = medians["bf16"] - medians["fp16"]
    final = {
        "status": "HUNYUAN15_LORA_DISK_FP16_AB_PASS",
        "order": ORDER,
        "repeats_per_method": 3,
        "trial_seconds": [
            {"trial": x["index"], "stored_dtype": x["stored_dtype"],
             "wall_seconds": x["generation_wall_seconds"]}
            for x in trials
        ],
        "median_seconds": medians,
        "mean_seconds": means,
        "preconverted_fp16_savings_seconds": round(diff, 3),
        "preconverted_fp16_speedup_factor": round(medians["bf16"] / medians["fp16"], 4),
        "preconverted_fp16_savings_percent_of_bf16_time":
            round(diff / medians["bf16"] * 100, 3),
        "prior_single_run_bf16_62_7_seconds": PRIOR_BF16_SECONDS,
        "preconversion": report,
        "fixed_parameters": {
            "checkpoint": str(BASE_CHECKPOINT),
            "prompt": base.PROMPT,
            "frames": base.FRAMES, "resolution": base.RESOLUTION,
            "seed": base.SEED, "steps": 4,
            "guidance": 1.0, "flow_shift": 8.0,
            "lora_multiplier": 1.0,
        },
        "timing_scope": "Fresh WanGP Python process per case: load+inference+decode+save. "
                        "One-time on-disk BF16-to-FP16 conversion, conversion verification, "
                        "dry-run, MP4 validation, and report generation are EXCLUDED.",
        "limitations": "One 6-trial ABBA/BA set, not a high-powered statistical study. "
                       "In-process LoRA cast savings may be masked by system or I/O noise.",
    }
    (OUT_ROOT / "comparison.json").write_text(json.dumps(final, indent=2) + "\n")
    print("HUNYUAN15_LORA_DISK_FP16_AB_PASS " + json.dumps(final, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
