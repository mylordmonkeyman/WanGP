#!/usr/bin/env python3
"""One-off 73f negative-distillation-model T2V smoke on the existing V100 runner.

Reuses the original four-way 73-frame harness, but runs ONLY its Stage2 FP16
Candidate E + Radix4 arm with a separate finetune pointing at the hard-linked
negative-0.4 INT8 ConvRot model. No stock checkpoint or benchmark defaults are
changed; all finetune/config files created here are confined to this checkout.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import t2v73_241_original_stock_fourway as four

ROOT = Path(__file__).resolve().parents[2]
MODEL_TYPE = "ltx25_neg04_t2v73_runner"
MODEL = Path("/cache/ltx25/models/ltx-2.5-22b-distilled-neg04_diffusion_model_int8_convrot.safetensors")
RESULT = Path(os.environ.get(
    "LTX25_NEG04_OUT",
    "/cache/ltx25/results/neg04-t2v73-" + os.getenv("GITHUB_RUN_ID", "manual")
))
BASELINE_SECONDS = 108.88
CASE = "t2v73-01-stage2_fp16-neg04"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    require(MODEL.is_file() and MODEL.stat().st_size > 1_000_000_000,
            "New hard-linked model missing/incomplete: " + str(MODEL))
    require(not RESULT.exists(), "Output directory must be new: " + str(RESULT))
    # Isolated runner checkout: a separate finetune, no overwrite of stock defaults.
    model_def = json.loads((ROOT / "defaults/ltx2_25_22B_distilled.json").read_text())
    model_def["model"]["name"] = "LTX-2.5 22B Distilled negative 0.4 - isolated smoke"
    model_def["model"]["URLs"] = [str(MODEL)]
    model_def["model"]["architecture"] = "ltx2_25_22B"
    model_def["model"]["ltx2_pipeline"] = "distilled"
    finetune = ROOT / "finetunes" / (MODEL_TYPE + ".json")
    require(not finetune.exists(), "Finetune already exists: " + str(finetune))
    finetune.write_text(json.dumps(model_def, indent=2) + "\n")

    RESULT.mkdir(parents=True)
    for name in ("results", "logs", "clips", "config", "videos"):
        (RESULT / name).mkdir()

    four.OUT = RESULT
    four.base.OUT = RESULT
    four.base.EXTENSION = RESULT / "libltx25_cublas_e_sm70.so"
    four.base.FRAMES = 73
    four.base.RESOLUTION = four.RESOLUTION
    four.base.STEPS = four.STEPS
    four.base.SEED = four.SEED
    four.base.PROMPT = four.PROMPT

    sample = four.settings(73)
    sample["model_type"] = MODEL_TYPE
    require(sample["seed"] == 20261112 and sample["video_length"] == 73
            and sample["resolution"] == "1024x576"
            and sample["num_inference_steps"] == 8
            and not sample["activated_loras"],
            "73f benchmark settings unexpectedly changed")
    (RESULT / "input.json").write_text(json.dumps(sample, indent=2) + "\n")

    original_config_for = four.base.config_for

    def isolated_config_for(case, template):
        cfg_dir, target = original_config_for(case, template)
        cfg = cfg_dir / "wgp_config.json"
        config = json.loads(cfg.read_text())
        config["transformer_types"] = [MODEL_TYPE]
        config["checkpoints_paths"] = [str(four.base.CHECKPOINTS), str(MODEL.parent)]
        cfg.write_text(json.dumps(config, indent=2) + "\n")
        return cfg_dir, target

    four.base.config_for = isolated_config_for

    four.base.gpu_free()
    build = four.base.command_run(
        ["nvcc", "-O3", "-std=c++17", "-arch=sm_70",
         "-shared", "-Xcompiler", "-fPIC",
         "benchmarks/ltx25_convrot/ltx25_cublas_e.cu",
         "-lcublas", "-o", str(four.base.EXTENSION)],
        timeout=300,
        log=RESULT / "logs" / "compile-candidate-e.log"
    )
    require(four.base.EXTENSION.is_file(), "Candidate E library compile failed")

    template = four.base.initialize_config()
    manifest = {
        "model": MODEL_TYPE,
        "model_checkpoint": str(MODEL),
        "model_bytes": MODEL.stat().st_size,
        "model_origin": "standalone BF16 distilled plus -0.4 distilled LoRA, merged then INT8 ConvRot",
        "baseline_reference_seconds": BASELINE_SECONDS,
        "benchmark_source": "benchmarks/ltx25_convrot/t2v73_241_original_stock_fourway.py",
        "branch_sha": os.getenv("GITHUB_SHA", "manual"),
        "resolution": four.RESOLUTION,
        "frames": 73, "steps": four.STEPS, "seed": four.SEED,
        "prompt": four.PROMPT,
        "audio_cfg_scale": 1.5,
        "NAG_scale": 1.1,
        "extra_lora": False,
        "arm": "stage2_fp16",
        "candidate_e_mode": "fp32rot",
        "hadamard": "radix4",
        "fp16_groups": four.GROUP_STAGE2,
        "timing": "time.monotonic from fresh subprocess launch through completion"
    }
    (RESULT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    row = four.run_case(73, 1, "stage2_fp16", template)
    generation = Path(row["log"]).read_text(errors="replace")
    require(MODEL.name in generation,
            "Model provenance FAIL: generation log did not confirm negative-0.4 model. "
            "Refusing to classify as new-model comparison.")
    require(re.search(r"Loading Model[^\n]*neg04", generation, re.I),
            "Model provenance FAIL: no Loading Model line for negative-0.4 checkpoint.")

    difference = round(row["wall_seconds"] - BASELINE_SECONDS, 3)
    change_pct = round(100 * difference / BASELINE_SECONDS, 2)
    report = {
        "status": "PASS",
        "checkpoint": str(MODEL),
        "clip": row["clip"],
        "frames": 73, "resolution": "1024x576",
        "steps": 8, "seed": 20261112,
        "wall_seconds": row["wall_seconds"],
        "previous_stage2_fp16_seconds": BASELINE_SECONDS,
        "seconds_difference": difference,
        "percent_difference": change_pct,
        "stage2_fp16": row.get("fp16"),
        "candidate_e": row.get("candidate_e"),
        "video_sha256": row["decoded_video_sha256"],
        "audio_sha256": row["decoded_audio_sha256"],
    }
    (RESULT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("LTX25_NEG04_T2V73_SUMMARY", json.dumps(report, sort_keys=True), flush=True)
    print("LTX25_NEG04_T2V73_PASS", flush=True)


if __name__ == "__main__":
    main()
