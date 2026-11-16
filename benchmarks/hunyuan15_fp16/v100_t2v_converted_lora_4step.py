#!/usr/bin/env python3
"""Offline 4-step bar T2V: native FP16 *original* Hunyuan 1.5 + converted LoRA.

Check exactly the original 30-step model+prompt+resolution+frames+seed.
Perform strict CPU LoRA conversion before the dry-run and timed subprocess.
Only the diffusion schedule (4 steps/CFG 1/shift 8) and LoRA change.
Compare the whole-process wall clock to the cached 30-step baseline and to
the independent 4-step full-distilled checkpoint test. No downloads.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import v100_bar_video_smoke as base
from v100_t2v_cached_steady_state import validate_clip

sys.path.insert(0, str(base.ROOT / "scripts"))
from hunyuan15_lora_convert import convert

MODE = "t2v"
LORA_SOURCE_NAME = "hunyuanvideo1.5_t2v_480p_lightx2v_4step_lora_rank_32_bf16.safetensors"
LORA_NAME = "hunyuan15_t2v_lightx2v4_rank32_wangp_qkv_split_bf16.safetensors"
LORA_SOURCE = base.WRITABLE_CHECKPOINTS / "loras" / LORA_SOURCE_NAME
CHECKPOINT = base.WRITABLE_CHECKPOINTS / "hunyuanvideo1.5_480p_t2v_fp16.safetensors"
ORIGINAL_MODEL_TYPE = "hunyuan15_v100_t2v_480_fp16_experiment"
BASELINE_SUMMARY = (
    base.CACHE / "results" / "hunyuan15-cached-t2v-43933748031-attempt1"
    / "t2v" / "summary.json"
)
FULL_DISTILLED_SUMMARY = (
    base.CACHE / "results" / "hunyuan15-lightx2v-t2v4-43943578757-attempt1"
    / "t2v" / "summary.json"
)


def seconds_from(path):
    if path.is_file():
        return json.loads(path.read_text()).get("run", {}).get("seconds")
    return None


def main():
    base.gpu_preflight()
    if not CHECKPOINT.is_file() or CHECKPOINT.stat().st_size < 15_000_000_000:
        raise RuntimeError(f"Original base FP16 checkpoint not available: {CHECKPOINT}")
    if not LORA_SOURCE.is_file() or LORA_SOURCE.stat().st_size < 300_000_000:
        raise RuntimeError(f"Original Comfy LoRA not available: {LORA_SOURCE}")

    base.OUT.mkdir(parents=True, exist_ok=True)
    converted_dir = base.OUT / "converted_loras"
    converted_file = converted_dir / LORA_NAME
    converter_report_path = base.OUT / "converted_lora_report.json"
    converter_report = convert(LORA_SOURCE, CHECKPOINT, converted_file, converter_report_path)
    if converter_report["input_tensors"] < 1500 or converter_report["qkv_input_tensors_split"] != 324:
        raise RuntimeError("Converted adapter unexpectedly lacks expected 108 fused QKV modules")
    lora_config = base.OUT / "converted_lora_paths.json"
    lora_config.write_text(json.dumps({"hunyuan_1_5": str(converted_dir)}, indent=2) + "\n")

    # Base model type is the *undistilled*, already tested native-FP16
    # Comfy-Org T2V model. Do not select the full LightX2V checkpoint here.
    assert base.MODEL_TYPES[MODE] == ORIGINAL_MODEL_TYPE
    assert base.MODEL_FILES[MODE] == CHECKPOINT.name
    old_settings = base.make_settings
    def lora_settings(mode):
        if mode != MODE:
            raise RuntimeError("Converted LightX2V adapter is only validated for T2V")
        settings = old_settings(mode)
        settings.update({
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
            "flow_shift": 8.0,
            "activated_loras": [LORA_NAME],
            "loras_multipliers": "1.0",
        })
        return settings
    base.make_settings = lora_settings

    original_run = base.run_command
    def offline_run(command, log_path, env, timeout):
        command = list(command)
        if "wgp.py" in command:
            command.insert(command.index("wgp.py") + 1, "--offline")
            command.extend(["--lora-config", str(lora_config)])
        env = dict(env)
        env.update({
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        })
        return original_run(command, log_path, env, timeout)
    base.run_command = offline_run

    os.environ["HUNYUAN15_ALLOW_READONLY_CACHE"] = "1"
    print("HUNYUAN15_CONVERTED_LORA_TEST_START " + json.dumps({
        "checkpoint": str(CHECKPOINT), "converted_lora": str(converted_file),
        "converted_sha256": converter_report["destination_sha256"],
        "prompt": base.PROMPT, "seed": base.SEED, "frames": base.FRAMES,
        "resolution": base.RESOLUTION, "steps": 4, "guidance": 1.0,
        "flow_shift": 8.0, "offline": True,
    }, sort_keys=True), flush=True)
    report = base.run_mode(MODE)
    generation_log = (base.OUT / MODE / "logs" / "generation.log").read_text(errors="replace")
    if "EXPERIMENTAL Hunyuan 1.5 native FP16 enabled" not in generation_log:
        raise RuntimeError("Experimental native FP16 transformer was not loaded")
    if f"Loading Model '{CHECKPOINT}'" not in generation_log:
        raise RuntimeError("Unexpected model loaded: not original FP16 base")
    if "Hunyuan 1.5 experimental loader: unpacked 216 Comfy QKV tensors" not in generation_log:
        raise RuntimeError("Original Comfy FP16 checkpoint conversion was not exercised")
    if any(marker in generation_log for marker in (
        "Unable to Download", "Download URL:", "Downloading Model",
        "Downloading model", "Downloading Lora", "contains unexpected module keys",
    )):
        raise RuntimeError("Unexpected download or LoRA module rejection")

    settings = json.loads((base.OUT / MODE / "settings.json").read_text())
    assert settings["model_type"] == ORIGINAL_MODEL_TYPE
    assert settings["activated_loras"] == [LORA_NAME]
    assert settings["loras_multipliers"] == "1.0"
    assert settings["num_inference_steps"] == 4
    assert settings["guidance_scale"] == 1.0
    assert settings["flow_shift"] == 8.0

    validation = [validate_clip(Path(video["name"])) for video in report["videos"]]
    if not validation:
        raise RuntimeError("No decoded MP4")
    elapsed = report["run"]["seconds"]
    old_time = seconds_from(BASELINE_SUMMARY)
    full_4step_time = seconds_from(FULL_DISTILLED_SUMMARY)
    comparison = {
        "status": "CONVERTED_LIGHTX2V_LORA_T2V_VIDEO_PASS",
        "model_type": report["model_type"],
        "base_checkpoint": str(CHECKPOINT),
        "source_lora": str(LORA_SOURCE),
        "converted_lora": str(converted_file),
        "converted_lora_sha256": converter_report["destination_sha256"],
        "converted_lora_bytes": converter_report["destination_bytes"],
        "conversion_tensors": converter_report["output_tensors"],
        "lora_multiplier": 1.0,
        "steps": 4, "guidance": 1.0, "flow_shift": 8.0,
        "seed": base.SEED, "frames": base.FRAMES,
        "resolution": base.RESOLUTION, "text_prompt": base.PROMPT,
        "timing_scope": "new Python process; cached original FP16 model+converted LoRA; loading+inference+decode+save; conversion and dry-run excluded",
        "generation_wall_seconds": elapsed,
        "prior_30step_original_wall_seconds": old_time,
        "full_distilled_4step_wall_seconds": full_4step_time,
        "speedup_over_original_30step": round(old_time / elapsed, 3)
            if old_time is not None and elapsed else None,
        "difference_vs_full_distilled_4step_seconds": round(elapsed - full_4step_time, 1)
            if full_4step_time is not None else None,
        "video_validation": validation,
        "note": "Full LightX2V checkpoint and rank-32 extracted LoRA may differ in quality; measuring speed alone is not a fidelity comparison.",
    }
    (base.OUT / MODE / "converted_lora_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n"
    )
    print("HUNYUAN15_CONVERTED_LORA_T2V_PASS " + json.dumps(comparison, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
