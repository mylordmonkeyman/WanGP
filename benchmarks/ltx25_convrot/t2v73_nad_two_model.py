#!/usr/bin/env python3
"""Compare original vs negative-0.4 baked LTX-2.5, both with WanGP NAD DiffVAE.

One isolated GPU runner job, two serial complete subprocesses. Reuses the
73-frame Candidate E/radix4 + Stage2 FP16 harness and its exact prompt/seed.
No new weights, no host checkpoint changes, no production finetune changes.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import statistics

import t2v73_241_original_stock_fourway as four

ROOT = Path(__file__).resolve().parents[2]
CKPTS = Path("/opt/ltx25/checkpoints")
SHARED = Path("/cache/ltx25/models")
VAE_FILENAME = "ltx-2.5-22b_diffusion_video_vae_bf16.safetensors"
VAE = CKPTS / VAE_FILENAME
BASE_MODEL = CKPTS / "ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors"
NEG_MODEL = SHARED / "ltx-2.5-22b-distilled-neg04_diffusion_model_int8_convrot.safetensors"
OUT = Path(os.environ.get(
    "LTX25_NAD2_OUT",
    "/cache/ltx25/results/nad2way73-" + os.getenv("GITHUB_RUN_ID", "manual")
))
CASES = (
    ("original", "ltx25_nad73_original", BASE_MODEL),
    ("negative04", "ltx25_nad73_neg04", NEG_MODEL),
)
BASELINE = {"original_conv_vae_seconds": 108.88,
            "negative04_conv_vae_seconds": 116.502}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def make_finetune(name, model):
    target = ROOT / "finetunes" / (name + ".json")
    require(not target.exists(), "Refusing to overwrite a finetune: " + str(target))
    data = json.loads((ROOT / "defaults/ltx2_25_22B_distilled.json").read_text())
    data["model"]["name"] = "LTX-2.5 NAD 73-frame smoke - " + name
    data["model"]["architecture"] = "ltx2_25_22B"
    data["model"]["ltx2_pipeline"] = "distilled"
    data["model"]["URLs"] = [str(model)]
    # WanGP _get_multi_file_names uses this to select the DiffVAE.
    data["model"]["ltx2_video_vae_file"] = VAE_FILENAME
    target.write_text(json.dumps(data, indent=2) + "\n")
    return target


def model_input(model_type):
    settings = four.settings(73)
    require(settings.get("video_length") == 73
            and settings.get("resolution") == "1024x576"
            and settings.get("num_inference_steps") == 8
            and settings.get("seed") == 20261112
            and settings.get("NAG_scale") == 1.1
            and settings.get("audio_cfg_scale") == 1.5
            and settings.get("activated_loras") == []
            and settings.get("loras_multipliers") == ""
            and not settings.get("video_prompt_type"),
            "Original 73-frame benchmark settings have changed")
    settings["model_type"] = model_type
    return settings


def assert_provenance(row, model):
    log = Path(row["log"]).read_text(errors="replace")
    require(model.name in log, "Model selection not proven in generation log for " + str(model))
    require(re.search(r"Loading Model[^\n]*" + re.escape(model.name), log),
            "Missing exact Loading Model line: " + model.name)
    require("Loading NAD Diffusion Decoder" in log,
            "Diffusion VAE was NOT selected; rejecting test (may have used Conv VAE)")
    require(VAE_FILENAME in log,
            "Expected NAD checkpoint name not found in generation log")
    require(row["frames"] == 73 and row["width"] == 1024 and row["height"] == 576,
            "Output frames/dimensions mismatch")
    require(row.get("fp16", {}).get("converted_per_stage", {}).get("stage1") == 0,
            "Stage1 was unexpectedly converted")
    require(row.get("fp16", {}).get("converted_per_stage", {}).get("stage2", 0) > 0,
            "Stage2 FP16 not exercised")
    require(row.get("candidate_e", {}).get("mode") == "fp32rot"
            and row.get("candidate_e", {}).get("rotation_mode") == "radix4"
            and row.get("candidate_e", {}).get("used", 0) > 0,
            "Candidate E/radix4 was not exercised")


def main():
    require(VAE.is_file() and VAE.stat().st_size > 100_000_000,
            "WanGP's NAD Diffusion VAE not accessible at " + str(VAE))
    for name, mtype, model in CASES:
        require(model.is_file() and model.stat().st_size > 1_000_000_000,
                "Missing/incomplete transformer: " + str(model))
    require(not OUT.exists(), "Result path already exists: " + str(OUT))

    for name in ("results", "logs", "clips", "config", "videos", "inputs"):
        (OUT / name).mkdir(parents=True, exist_ok=False)

    four.OUT = OUT
    four.base.OUT = OUT
    four.base.EXTENSION = OUT / "libltx25_cublas_e_sm70.so"
    four.base.FRAMES = 73
    four.base.RESOLUTION = four.RESOLUTION
    four.base.STEPS = four.STEPS
    four.base.SEED = four.SEED
    four.base.PROMPT = four.PROMPT

    finetunes = {name: str(make_finetune(mtype, model))
                 for name, mtype, model in CASES}

    original_config_for = four.base.config_for
    current_model_type = None

    def config_for_nad(case, template):
        cfg_dir, target = original_config_for(case, template)
        cfg_file = cfg_dir / "wgp_config.json"
        cfg = json.loads(cfg_file.read_text())
        cfg["transformer_types"] = [current_model_type]
        cfg["checkpoints_paths"] = [str(CKPTS), str(SHARED)]
        cfg_file.write_text(json.dumps(cfg, indent=2) + "\n")
        return cfg_dir, target

    four.base.config_for = config_for_nad
    four.base.gpu_free()
    four.base.command_run(
        ["nvcc", "-O3", "-std=c++17", "-arch=sm_70",
         "-shared", "-Xcompiler", "-fPIC",
         "benchmarks/ltx25_convrot/ltx25_cublas_e.cu",
         "-lcublas", "-o", str(four.base.EXTENSION)],
        timeout=300, log=OUT / "logs" / "compile-candidate-e.log",
    )
    require(four.base.EXTENSION.is_file(), "Candidate E library compile failed")

    # initialize_config runs a separate dry-run and is not included in timing.
    sample = model_input(CASES[0][1])
    (OUT / "input.json").write_text(json.dumps(sample, indent=2) + "\n")
    template = four.base.initialize_config()
    manifest = {
        "status": "STARTED",
        "run_sha": os.getenv("GITHUB_SHA", "manual"),
        "vae_checkpoint": str(VAE),
        "vae_bytes": VAE.stat().st_size,
        "cases": [{"variant": name, "finetune": mtype, "model": str(model),
                   "model_bytes": model.stat().st_size}
                  for name, mtype, model in CASES],
        "finetunes": finetunes,
        "prompt": four.PROMPT,
        "frames": 73, "resolution": four.RESOLUTION, "steps": four.STEPS,
        "seed": four.SEED, "NAG_scale": 1.1, "audio_cfg_scale": 1.5,
        "additional_loras": [],
        "optimizations": "Candidate E fp32rot + radix4 + Stage2 direct FP16",
        "timing": "fresh WanGP subprocess creation to exit; excludes prep and quality checks",
        "previous_conv_vae_times": BASELINE,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    rows = []
    for index, (name, mtype, model) in enumerate(CASES, 1):
        current_model_type = mtype
        settings = model_input(mtype)
        (OUT / "input.json").write_text(json.dumps(settings, indent=2) + "\n")
        (OUT / "inputs" / (name + ".json")).write_text(json.dumps(settings, indent=2) + "\n")
        four.base.gpu_free()
        print("LTX25_NAD2WAY73_BEGIN", name, str(model), flush=True)
        try:
            row = four.run_case(73, index, "stage2_fp16", template)
            assert_provenance(row, model)
            row["variant"] = name
            row["model_checkpoint"] = str(model)
            row["vae_checkpoint"] = str(VAE)
            rows.append(row)
            (OUT / "results" / (name + "-validated.json")).write_text(
                json.dumps(row, indent=2) + "\n")
            print("LTX25_NAD2WAY73_CASE_PASS", name, row["wall_seconds"], flush=True)
        finally:
            (OUT / "partial-results.json").write_text(json.dumps(rows, indent=2) + "\n")

    require(len(rows) == 2, "Both transformer runs are required")
    require(rows[0]["decoded_video_sha256"] != rows[1]["decoded_video_sha256"],
            "Identical video frame hashes; checkpoint selection may have failed")
    vals = {row["variant"]: row["wall_seconds"] for row in rows}
    report = {
        "status": "PASS",
        "frames": 73, "resolution": "1024x576", "steps": 8, "seed": 20261112,
        "vae": str(VAE), "model_times_seconds": vals,
        "negative_minus_original_seconds": round(vals["negative04"] - vals["original"], 3),
        "negative_vs_original_percent": round(
            100 * (vals["negative04"] / vals["original"] - 1), 2),
        "relative_to_previous_conv_vae": {
            name: {
                "previous_seconds": baseline,
                "nad_seconds": vals[name],
                "difference_seconds": round(vals[name] - baseline, 3),
                "change_percent": round(100 * (vals[name] / baseline - 1), 2),
            }
            for name, baseline in (
                ("original", BASELINE["original_conv_vae_seconds"]),
                ("negative04", BASELINE["negative04_conv_vae_seconds"]),
            )
        },
        "videos": {row["variant"]: row["clip"] for row in rows},
        "video_hashes": {row["variant"]: row["decoded_video_sha256"] for row in rows},
        "audio_hashes": {row["variant"]: row["decoded_audio_sha256"] for row in rows},
        "candidate_e": {row["variant"]: row.get("candidate_e") for row in rows},
        "fp16": {row["variant"]: row.get("fp16") for row in rows},
    }
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("LTX25_NAD2WAY73_SUMMARY", json.dumps(report, sort_keys=True), flush=True)
    print("LTX25_NAD2WAY73_PASS", flush=True)


if __name__ == "__main__":
    main()
