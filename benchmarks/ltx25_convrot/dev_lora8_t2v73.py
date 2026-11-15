#!/usr/bin/env python3
"""73-frame LTX-2.5 Dev+1.0 distillation LoRA / 8-step quality smoke.

Uses native WanGP Dev TWO-STAGE pipeline and its 8-step ancestral solver, not the
standalone distilled pipeline. Matched cocktail-bar prompt and seed. Conv VAE only.
"""
import json
import os
from pathlib import Path
import re

import t2v73_241_original_stock_fourway as four

ROOT = Path(__file__).resolve().parents[2]
DEV = Path("/cache/ltx25/models/ltx-2.5-22b-dev_diffusion_model_int8_convrot.safetensors")
VAE = Path("/opt/ltx25/checkpoints/ltx-2.5-22b_video_vae_bf16.safetensors")
LORA_NAME = "ltx-2.5-22b-distilled-lora-450_bf16.safetensors"
LORA = ROOT / "loras/ltx2" / LORA_NAME
MODEL_TYPE = "ltx25_dev_lora8_runner"
OUT = Path(os.environ.get("LTX25_DEV_LORA8_OUT", "/cache/ltx25/results/dev-lora8-t2v73-manual"))
PREV_TIMINGS = {"distilled_conv_seconds": 108.880, "neg04_distilled_conv_seconds": 116.502}


def require(pred, msg):
    if not pred:
        raise RuntimeError(msg)


def main():
    require(DEV.is_file() and DEV.stat().st_size > 1_000_000_000, "Dev checkpoint missing")
    require(VAE.is_file(), "Default Conv VAE missing")
    require(LORA.is_file() and LORA.stat().st_size > 100_000_000, "Official distillation LoRA missing")
    require(not OUT.exists(), "Output path already exists: " + str(OUT))
    for part in ("results", "logs", "clips", "config", "videos"):
        (OUT / part).mkdir(parents=True, exist_ok=False)

    # Write isolated finetune definition in the disposable GHA checkout.
    spec = json.loads((ROOT / "defaults/ltx2_25_22B.json").read_text())
    spec["model"]["name"] = "LTX-2.5 Dev + 1.0 distillation LoRA, 8 steps"
    spec["model"]["URLs"] = [str(DEV)]
    spec["model"]["preload_URLs"] = []   # unrelated optional HDR files not required for T2V
    spec["model"]["ltx2_video_vae_file"] = VAE.name
    spec["model"].pop("ltx2_pipeline", None)  # Dev TWO-STAGE pipeline
    spec["num_inference_steps"] = 8
    spec["sample_solver"] = "distilled_8_steps_ancestral"
    new_def = ROOT / "finetunes" / (MODEL_TYPE + ".json")
    require(not new_def.exists(), "Test finetune already exists")
    new_def.write_text(json.dumps(spec, indent=2) + "\n")

    four.OUT = OUT
    four.base.OUT = OUT
    four.base.EXTENSION = OUT / "libltx25_cublas_e_sm70.so"
    four.base.FRAMES = 73
    four.base.RESOLUTION = "1024x576"
    four.base.STEPS = 8
    four.base.SEED = 20261112
    four.base.PROMPT = four.PROMPT

    settings = four.settings(73)
    settings.update({
        "model_type": MODEL_TYPE,
        "activated_loras": [LORA_NAME],
        "loras_multipliers": "1",
        "sample_solver": "distilled_8_steps_ancestral",
        "guidance_phases": 2,
        "guidance_scale": 1.0,
        "audio_guidance_scale": 1.0,
        "audio_cfg_scale": 1.0,
        "alt_guidance_scale": 1.0,
        "alt_scale": 0.0,
        "cfg_star_switch": 0,
        "apg_switch": 0,
        "perturbation_switch": 0,
    })
    require(settings["video_length"] == 73 and settings["seed"] == 20261112
            and settings["resolution"] == "1024x576"
            and settings["num_inference_steps"] == 8
            and settings["activated_loras"] == [LORA_NAME]
            and settings["loras_multipliers"] == "1",
            "T2V matched config changed")
    (OUT / "input.json").write_text(json.dumps(settings, indent=2) + "\n")

    original_config_for = four.base.config_for
    def dev_config_for(name, template):
        cfg_dir, target = original_config_for(name, template)
        p = cfg_dir / "wgp_config.json"
        cfg = json.loads(p.read_text())
        cfg["transformer_types"] = [MODEL_TYPE]
        cfg["checkpoints_paths"] = ["/opt/ltx25/checkpoints", str(DEV.parent)]
        cfg["profile"] = 0
        cfg["video_profile"] = 0
        p.write_text(json.dumps(cfg, indent=2) + "\n")
        return cfg_dir, target

    four.base.config_for = dev_config_for
    four.base.gpu_free()
    four.base.command_run(
        ["nvcc", "-O3", "-std=c++17", "-arch=sm_70",
         "-shared", "-Xcompiler", "-fPIC",
         "benchmarks/ltx25_convrot/ltx25_cublas_e.cu",
         "-lcublas", "-o", str(four.base.EXTENSION)],
        timeout=300, log=OUT / "logs/compile-candidate-e.log"
    )
    require(four.base.EXTENSION.is_file(), "Candidate E build failed")
    template = four.base.initialize_config()

    manifest = {
        "model": MODEL_TYPE, "dev_checkpoint": str(DEV),
        "checkpoint_size": DEV.stat().st_size,
        "lora": str(LORA.resolve()), "lora_size": LORA.stat().st_size,
        "lora_strength": 1.0, "lora_dynamic": True,
        "sampler": "distilled_8_steps_ancestral", "steps": 8,
        "vae": str(VAE), "pipeline": "two_stage",
        "prompt": four.PROMPT, "seed": 20261112, "frames": 73,
        "resolution": "1024x576", "audio_cfg_scale": 1.0,
        "guidance_scale": 1.0,
        "optimization": "Candidate E radix4 Stage2 FP16 (non-sham)",
        "baseline_seconds": PREV_TIMINGS,
        "wall_time_scope": "fresh WanGP subprocess start to exit"
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("DEV_LORA8_T2V73_BEGIN", json.dumps(manifest), flush=True)

    row = four.run_case(73, 1, "stage2_fp16", template)
    text = Path(row["log"]).read_text(errors="replace")
    require(DEV.name in text and re.search(r"Loading Model[^\n]*dev", text, re.I),
            "Provenance failure: expected Dev checkpoint not observed in generation log")
    require(("Lora '" in text and LORA_NAME in text and "loaded in model" in text),
            "Provenance failure: +1.0 distillation LoRA not confirmed loaded")
    require("Loading NAD Diffusion Decoder" not in text,
            "Unexpected NAD decoder. Conv VAE required.")
    require(row["width"] == 1024 and row["height"] == 576 and row["frames"] == 73,
            "Incorrect MP4 dimensions or frames")

    report = {
        "status": "PASS", "case": "Dev INT8 ConvRot + dynamic +1.0 distillation LoRA",
        "wall_seconds": row["wall_seconds"],
        "seconds_vs_original_distilled": round(row["wall_seconds"] - PREV_TIMINGS["distilled_conv_seconds"], 3),
        "percent_vs_original_distilled": round(100 * (row["wall_seconds"] / PREV_TIMINGS["distilled_conv_seconds"] - 1), 2),
        "video": row["clip"],
        "video_sha256": row["decoded_video_sha256"],
        "audio_sha256": row["decoded_audio_sha256"],
        "fp16_stage2_converted": row["fp16"]["converted_per_stage"]["stage2"],
        "fp16_guard_fallbacks": row["fp16"]["guard_fallbacks"],
        "candidate_e_uses": row["candidate_e"]["used"],
        "input": settings,
    }
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("LTX25_DEV_LORA8_T2V73_SUMMARY", json.dumps(report, sort_keys=True), flush=True)
    print("LTX25_DEV_LORA8_T2V73_PASS", flush=True)


if __name__ == "__main__":
    main()
