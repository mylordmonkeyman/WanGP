#!/usr/bin/env python3
"""Cold-process, offline Hunyuan 1.5 LightX2V *full-checkpoint* 4-step T2V.

This replaces the incompatible Comfy-Org LoRA experiment with the published
16.7 GB full LightX2V 4-step distilled transformer. The baseline remains
the same 30-step 33-frame bar scene: same text, resolution, frame count, seed,
V100 backend, and cold-process timing. No LoRA is loaded or downloaded.

Checkpoint layout is audited using safetensors headers before launching WanGP.
The wall clock includes a new Python process, weights, denoising, decode,
and MP4 saving, but excludes the independent dry run and header audit.
"""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import sys

import v100_bar_video_smoke as base
from v100_t2v_cached_steady_state import validate_clip

sys.path.insert(0, str(base.ROOT / "scripts"))
from hunyuan15_checkpoint_audit import (
    read_header, normalize, expand_fused_qkv_metadata, compare,
)

MODE = "t2v"
DISTILLED_NAME = "hy1.5_t2v_480p_lightx2v_4step.safetensors"
MODEL_TYPE = "hunyuan15_v100_t2v_480_lightx2v_4step_fp16_experiment"
CHECKPOINT = base.WRITABLE_CHECKPOINTS / DISTILLED_NAME
ORIGINAL_CHECKPOINT = (
    base.WRITABLE_CHECKPOINTS / "hunyuanvideo1.5_480p_t2v_fp16.safetensors"
)
BASELINE_SUMMARY = (
    base.CACHE / "results" / "hunyuan15-cached-t2v-43933748031-attempt1"
    / "t2v" / "summary.json"
)

# Use the new model definition's URL to select the *distilled* weights.
# Do not alter the working 30-step research model definition or baseline.
base.MODEL_TYPES[MODE] = MODEL_TYPE
base.MODEL_FILES[MODE] = DISTILLED_NAME


def preflight_checkpoint():
    if not CHECKPOINT.is_file() or CHECKPOINT.stat().st_size < 15_000_000_000:
        raise RuntimeError(f"Full distilled 4-step checkpoint not staged: {CHECKPOINT}")
    if not ORIGINAL_CHECKPOINT.is_file():
        raise RuntimeError(f"Expected original T2V checkpoint for header audit: {ORIGINAL_CHECKPOINT}")
    reference_raw = normalize(read_header(ORIGINAL_CHECKPOINT))
    candidate_raw = normalize(read_header(CHECKPOINT))
    reference, ref_fused = expand_fused_qkv_metadata(reference_raw)
    candidate, cand_fused = expand_fused_qkv_metadata(candidate_raw)
    layout = compare(candidate, reference)
    source_dtypes = dict(sorted(Counter(v["dtype"] for v in candidate_raw.values()).items()))
    report = {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_bytes": CHECKPOINT.stat().st_size,
        "checkpoint_source": "lightx2v/Hy1.5-Distill-Models",
        "model_type": MODEL_TYPE,
        "original_reference": str(ORIGINAL_CHECKPOINT),
        "candidate_source_tensors": len(candidate_raw),
        "expanded_candidate_tensors": len(candidate),
        "reference_expected_tensors": len(reference),
        "candidate_fused_qkv_expansions": cand_fused,
        "reference_fused_qkv_expansions": ref_fused,
        "source_dtypes": source_dtypes,
        "layout": layout,
        "downloads_in_timed_run": False,
        "lora_activated": False,
    }
    base.OUT.mkdir(parents=True, exist_ok=True)
    (base.OUT / "distilled_checkpoint_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print("HUNYUAN15_DISTILLED_CHECKPOINT_AUDIT " + json.dumps(report, sort_keys=True), flush=True)
    if not layout["keys_and_shapes_match"]:
        raise RuntimeError("Distilled checkpoint key/shape mismatch: " + json.dumps(layout))
    if not set(source_dtypes) <= {"F16", "BF16", "F32"}:
        raise RuntimeError(f"Unexpected weight quantization: {source_dtypes}")
    return report


def main():
    base.gpu_preflight()
    preflight = preflight_checkpoint()

    original_settings = base.make_settings
    def distilled_settings(mode):
        if mode != MODE:
            raise RuntimeError("LightX2V 480p 4-step distilled checkpoint is T2V only")
        settings = original_settings(mode)
        settings.update({
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
            "flow_shift": 8.0,
            "activated_loras": [],
            "loras_multipliers": "",
        })
        return settings
    base.make_settings = distilled_settings

    original_run = base.run_command
    def offline_run(cmd, log_path, env, timeout):
        cmd = list(cmd)
        if "wgp.py" in cmd:
            cmd.insert(cmd.index("wgp.py") + 1, "--offline")
        env = dict(env)
        env.update({
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        })
        return original_run(cmd, log_path, env, timeout)
    base.run_command = offline_run

    # User-staged checkpoint directory is owned by the host account; the
    # runner only needs read access because both WanGP calls are offline.
    os.environ["HUNYUAN15_ALLOW_READONLY_CACHE"] = "1"
    report = base.run_mode(MODE)
    generation_log = (base.OUT / MODE / "logs" / "generation.log").read_text(errors="replace")
    if "EXPERIMENTAL Hunyuan 1.5 native FP16 enabled" not in generation_log:
        raise RuntimeError("Experimental native FP16 transformer was not activated")
    if f"Loading Model '{CHECKPOINT}'" not in generation_log:
        raise RuntimeError("WanGP did not load the required LightX2V full checkpoint")
    fused = preflight["candidate_fused_qkv_expansions"]
    if fused and f"unpacked {fused} Comfy QKV tensors" not in generation_log:
        raise RuntimeError(f"Expected {fused} fused QKV tensors to be unpacked")
    if any(marker in generation_log for marker in (
        "Unable to Download", "Download URL:", "Downloading Model",
        "Downloading model", "Downloading Lora",
    )):
        raise RuntimeError("Missing dependency or download attempt in timed run")

    settings = json.loads((base.OUT / MODE / "settings.json").read_text())
    assert settings["model_type"] == MODEL_TYPE
    assert settings["activated_loras"] == []
    assert settings["loras_multipliers"] == ""
    assert settings["num_inference_steps"] == 4
    assert settings["guidance_scale"] == 1.0
    assert settings["flow_shift"] == 8.0
    assert report["model_file"] == str(CHECKPOINT)

    videos = [validate_clip(Path(item["name"])) for item in report["videos"]]
    if not videos:
        raise RuntimeError("Missing decoded MP4")
    baseline_seconds = None
    if BASELINE_SUMMARY.is_file():
        baseline_seconds = json.loads(BASELINE_SUMMARY.read_text()).get("run", {}).get("seconds")
    seconds = report["run"]["seconds"]
    comparison = {
        "status": "LIGHTX2V_4STEP_T2V_VIDEO_PASS",
        "model_type": report["model_type"],
        "checkpoint": str(CHECKPOINT),
        "checkpoint_bytes": preflight["checkpoint_bytes"],
        "checkpoint_source_dtypes": preflight["source_dtypes"],
        "lora": None,
        "steps": 4,
        "guidance": 1.0,
        "flow_shift": 8.0,
        "seed": base.SEED,
        "frames": base.FRAMES,
        "resolution": base.RESOLUTION,
        "text_prompt": base.PROMPT,
        "offline": True,
        "timing_scope": "new Python process, cached full distilled checkpoint; load+inference+decode+save; dry-run/header-audit excluded",
        "generation_wall_seconds": seconds,
        "previous_30step_cached_wall_seconds": baseline_seconds,
        "wall_time_saved_seconds": round(baseline_seconds-seconds, 1)
            if baseline_seconds is not None else None,
        "wall_time_speedup_factor": round(baseline_seconds/seconds, 3)
            if baseline_seconds and seconds else None,
        "video_validation": videos,
        "note": "Distilled 4-step versus original 30-step changes the model and sampling, not an equal-quality or same-step kernel benchmark.",
    }
    (base.OUT / MODE / "lightx2v_4step_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n"
    )
    print("HUNYUAN15_LIGHTX2V_4STEP_T2V_PASS " + json.dumps(comparison, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
