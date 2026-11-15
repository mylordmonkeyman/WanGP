#!/usr/bin/env python3
"""Recover validated successful 73f T2V MP4 from Dev+LoRA smoke (no model run)."""
import json
from pathlib import Path
import re
import shutil
import sys

import t2v73_241_original_stock_fourway as four

ROOT = Path("/cache/ltx25/results/dev-lora8-t2v73-43674180726-attempt1")
CLIP = ROOT / "clips" / "ltx25-dev-int8-convrot-lora-plus1-8step-t2v73.mp4"
PREVIOUS_SECONDS = 108.880
DEV = "ltx-2.5-22b-dev_diffusion_model_int8_convrot.safetensors"
LORA = "ltx-2.5-22b-distilled-lora-450_bf16.safetensors"


def require(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def main():
    genlog = ROOT / "logs/t2v73-01-stage2_fp16-generation.log"
    rec = ROOT / "results/t2v73-01-stage2_fp16.json"
    require(genlog.is_file() and rec.is_file(), "Prior completed generation log/result missing")
    log = genlog.read_text(errors="replace")
    row = json.loads(rec.read_text())
    settings = json.loads((ROOT / "input.json").read_text())
    require(row["returncode"] == 0, "WanGP generation did not succeed")
    require(row["frames"] == 73 and settings["video_length"] == 73
            and settings["resolution"] == "1024x576"
            and settings["num_inference_steps"] == 8
            and settings["seed"] == 20261112,
            "Original generation input mismatch")
    require(settings["loras_multipliers"] == "1"
            and settings["activated_loras"] == [LORA]
            and settings["sample_solver"] == "distilled_8_steps_ancestral",
            "Missing eight-step +1.0 Dev LoRA settings")
    require(DEV in log and re.search(r"Loading Model[^\n]*dev", log),
            "Dev transformer not proven loaded")
    require("Lora 'loras/ltx2/" + LORA + "' was loaded in model" in log,
            "LoRA was not proven loaded")
    require("Default system 'distilled-lora' lora and corresponding multiplier will be ignored" in log,
            "Cannot rule out default duplicate distillation LoRA")
    require("Loading NAD Diffusion Decoder" not in log, "NAD decoder mistakenly used")
    require("Task 1 completed" in log and "Queue completed: 1/1 tasks" in log,
            "WanGP did not complete full generation")
    require("[LTX25_CUBLAS_E]" in log and '"rotation_mode": "radix4"' in log,
            "Optimized Candidate E Radix4 not demonstrated")
    vids = [p for p in (ROOT / "videos").rglob("*.mp4") if p.stat().st_size > 10_000]
    require(len(vids) == 1, "Exactly one successful MP4 required, found " + str(len(vids)))
    CLIP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(vids[0], CLIP)
    four.base.FRAMES = 73
    probe = four.probe_video(CLIP, 73)
    difference = row["wall_seconds"] - PREVIOUS_SECONDS
    report = {
        "status": "PASS",
        "validity": "Successful generation; Stage2 FP16 not exercised on Dev two-stage pipeline",
        "model": DEV, "distillation_lora": LORA, "lora_strength": 1.0,
        "sampler": "distilled_8_steps_ancestral", "vae": "standard convolutional",
        "resolution": "1024x576", "frames": 73, "steps": 8, "seed": 20261112,
        "wall_seconds": row["wall_seconds"],
        "prior_original_distilled_wall_seconds": PREVIOUS_SECONDS,
        "wall_increase_seconds": round(difference, 3),
        "wall_increase_percent": round(100 * difference / PREVIOUS_SECONDS, 2),
        "candidate_e_radix4_exercised": True,
        "stage2_fp16_exercised": False,
        "clip": str(CLIP), "clip_bytes": CLIP.stat().st_size,
        "width": probe["width"], "height": probe["height"],
        "video_frames_sha256": probe["decoded_video_sha256"],
        "audio_sha256": probe["decoded_audio_sha256"],
    }
    (ROOT / "recovered-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("LTX25_DEV8_RECOVERED_SUMMARY", json.dumps(report, sort_keys=True), flush=True)
    print("LTX25_DEV8_RECOVERY_PASS", flush=True)


if __name__ == "__main__":
    main()
