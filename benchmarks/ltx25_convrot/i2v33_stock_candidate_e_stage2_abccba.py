#!/usr/bin/env python3
"""33f 832x480 I2V: real full-process stock R0 vs Candidate E vs Stage2 FP16.

Matched six-case A-B-C-C-B-A, two fresh WGP processes for each arm.
All timings are complete wgp.py wall time including model startup, MMGP, both
denoising stages, VAE/audio decoding, encoding, and process exit. Dry runs,
compilation, and clip comparisons are expressly outside the timer.
The reference is stock WanGP INT8 ConvRot R0 execution on the SAME source tree
and SAME INT8 ConvRot checkpoint; not a different GGUF checkpoint or upstream
code revision. 'stock' explicitly disables the default-on Candidate E and
the experimental Stage2 direct FP16 interceptor.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

import selective_fp16_t2v_smoke as base

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("LTX25_33F_3WAY_OUT", "/cache/ltx25/results/i2v33-3way-" + os.environ.get("GITHUB_RUN_ID", "manual")))
IMAGE = ROOT / "benchmarks/ltx25_convrot/inputs/profile_i2v_1024x576.png.png"
FRAMES = 33
RESOLUTION = "832x480"
STEPS = 4
SEED = 20261112
ARMS = ("stock", "candidate_e", "stage2_fp16", "stage2_fp16", "candidate_e", "stock")
GROUPS = "stage2:wide,stage2:tall,stage2:balanced"
PROMPT = (
    "Photorealistic bustling upscale cocktail bar. Patrons engage in lively "
    "conversations, laughing, gesturing, sipping drinks, and moving naturally "
    "through the crowd. Bartenders shake cocktails, pour drinks, reach for "
    "bottles, and serve customers. Subtle, independent movements throughout "
    "the scene. Warm lighting and realistic reflections. Static camera, "
    "preserve original composition and character appearances.\n"
    "Audio: Continuous lively bar ambience with overlapping indistinct "
    "conversations, laughter, clinking glasses, cocktail shakers, pouring "
    "drinks, and soft background music. Natural, immersive crowd noise "
    "without intelligible dialogue."
)
base.OUT = OUT
base.EXTENSION = OUT / "libltx25_cublas_e_sm70.so"
base.PROMPT = PROMPT
base.SEED = SEED
base.FRAMES = FRAMES
base.STEPS = STEPS
base.RESOLUTION = RESOLUTION


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def settings():
    return {
        "model_type": "ltx2_25_22B_distilled",
        "prompt": PROMPT,
        "negative_prompt": "",
        "seed": SEED, "num_inference_steps": STEPS,
        "video_length": FRAMES, "resolution": RESOLUTION,
        "NAG_scale": 1.1, "audio_cfg_scale": 1.5,
        "activated_loras": [], "loras_multipliers": "",
        "audio_prompt_type": "", "video_prompt_type": "",
        "image_start": str(IMAGE), "image_prompt_type": "S",
        "image_mode": 0, "spatial_upsampling": "", "temporal_upsampling": "",
    }


def env_for(arm, case):
    # Identical base environment, explicit no-profiler switch and opt-in differences.
    env = base.base_env()
    for k in list(env):
        if k.startswith(("WAN2GP_LTX25_PIPELINE", "LTX25_PIPELINE_",
                         "LTX25_GEMM_PROFILE_", "LTX25_FP16_GEMM_")):
            env.pop(k, None)
    env["WAN2GP_LTX2_BENCH_TRANSFORMER"] = "0"
    env["WAN2GP_LTX25_FP16_GEMM"] = "0"
    env["WAN2GP_LTX25_REAL_ACTIVATION_AUDIT"] = "0"
    env["WAN2GP_LTX25_PROFILE_STAGES"] = "0"
    if arm == "stock":
        # This restores native WanGP R0 without a different checkpoint.
        env["WAN2GP_LTX25_CUBLAS_E_MODE"] = "off"
        env["WAN2GP_LTX25_DEFAULT_ON"] = "0"
        env["WAN2GP_LTX25_HADAMARD"] = "torch"
    else:
        env["WAN2GP_LTX25_CUBLAS_E_MODE"] = "fp32rot"
        env["WAN2GP_LTX25_HADAMARD"] = "radix4"
    if arm == "stage2_fp16":
        env.update({
            "WAN2GP_LTX25_FP16_GEMM": "1",
            "LTX25_FP16_GEMM_BACKEND": "direct",
            "LTX25_FP16_GEMM_GROUPS": GROUPS,
            "LTX25_FP16_GEMM_OPS": "qbytes_mm",
            "LTX25_FP16_GEMM_MIN_M": "1",
            "LTX25_FP16_GEMM_MIN_DIM": "64",
            "LTX25_FP16_GEMM_GUARD": "first",
            "LTX25_FP16_GEMM_SHAM": "0",
            "LTX25_FP16_GEMM_AUDIT_PER_SHAPE": "0",
            "LTX25_FP16_GEMM_SHAPE_CENSUS": "0",
            "LTX25_FP16_GEMM_TRACE": "0",
            "LTX25_FP16_GEMM_STATS_PATH": str(OUT/"results"/f"{case}-fp16.json"),
        })
    return env


def read_e_stats(log):
    found = re.search(r"^\[LTX25_CUBLAS_E\] (\{[^\n]+\})", log, re.M)
    return json.loads(found.group(1)) if found else None


def case_run(arm, index, template):
    name = f"{index:02d}-{arm}"
    cfg, target = base.config_for(name, template)
    env = env_for(arm, name)
    command = [base.PY, "wgp.py", "--offline", "--config", str(cfg),
               "--process", str(OUT/"input.json"), "--output-dir", str(target),
               "--profile", "0", "--attention", "sdpa",
               "--no-prevent-power-throttling", "--verbose", "1"]
    base.command_run(command + ["--dry-run"], env=env, timeout=300,
                     log=OUT/"logs"/f"{name}-dryrun.log")
    base.gpu_free()
    print("THREEWAY_CASE_BEGIN", name, flush=True)
    execution = base.command_run(command, env=env, timeout=1800, check=False,
                                 log=OUT/"logs"/f"{name}-generation.log")
    report = {
        "case": name, "arm": arm, "order": index,
        "wall_seconds": execution["wall_seconds"],
        "returncode": execution["returncode"],
        "input_sha256": sha(OUT/"input.json"),
        "source_sha": os.getenv("GITHUB_SHA", "manual"),
        "log": execution["log"],
    }
    # Record failures, including generation time, for partial artifact recovery.
    result_path = OUT/"results"/f"{name}.json"
    result_path.write_text(json.dumps(report, indent=2)+"\n")
    if execution["returncode"]:
        raise RuntimeError(f"{name}: wgp.py exit {execution['returncode']}: {execution['log_tail']}")
    log = Path(execution["log"]).read_text(errors="replace")
    stats = read_e_stats(log)
    if arm == "stock":
        if stats or "SM70 ConvRot enabled:" in log or "[LTX25_SELECTIVE_FP16_GEMM]" in log:
            raise RuntimeError(name + ": stock path unexpectedly optimized")
    else:
        if stats is None or stats.get("mode") != "fp32rot" or stats.get("rotation_mode") != "radix4" or stats.get("used", 0) <= 0:
            raise RuntimeError(name + ": Candidate E/radix4 not exercised")
        report["candidate_e_stats"] = {
            k: stats.get(k) for k in ("mode", "used", "fallback", "rotation_mode",
                                      "cast_mode", "output_mode", "rotation_input")
        }
    if arm == "stage2_fp16":
        p = OUT/"results"/f"{name}-fp16.json"
        if not p.exists():
            raise RuntimeError(name + ": missing FP16 dispatch report")
        data = json.loads(p.read_text())
        c = data.get("stats", {})
        report["fp16"] = {
            "converted_qbytes_mm": int(c.get("converted_qbytes_mm",0)),
            "observed_qbytes_mm": int(c.get("observed_qbytes_mm",0)),
            "stage1_bypassed": int(c.get("bypassed_unselected_stage1_forwards",0)),
            "guard_checks": int(c.get("guard_checks",0)),
            "persistent_weight_copies": data.get("persistent_fp16_weight_copies"),
            "blocked_shapes": data.get("blocked_shapes"),
        }
        if (data.get("backend") != "direct"
                or int(c.get("converted_qbytes_mm",0)) < 1
                or int(c.get("bypassed_unselected_stage1_forwards",0)) < 1
                or data.get("blocked_shapes")
                or data.get("persistent_fp16_weight_copies") != 0):
            raise RuntimeError(name + ": FP16 dispatch did not execute as selected")
    elif "[LTX25_SELECTIVE_FP16_GEMM]" in log:
        raise RuntimeError(name + ": unintended Stage2 FP16")
    clips = [p for p in target.rglob("*.mp4") if p.stat().st_size > 1024]
    if len(clips) != 1:
        raise RuntimeError(f"{name}: expected one MP4, found {len(clips)}")
    dest = OUT/"clips"/f"{name}.mp4"
    shutil.copy2(clips[0], dest)
    report.update({
        "clip": str(dest), "clip_bytes": dest.stat().st_size,
        "video_hash": base.frames_hash(dest, base.ffmpeg_bin()),
        "audio_hash": base.audio_hash(dest, base.ffmpeg_bin()),
    })
    result_path.write_text(json.dumps(report, indent=2)+"\n")
    print("THREEWAY_CASE_PASS",json.dumps({
        "case":name,"arm":arm,"wall_seconds":report["wall_seconds"],
        "candidate_e_used":(report.get("candidate_e_stats") or {}).get("used"),
        "stage2_converted":(report.get("fp16") or {}).get("converted_qbytes_mm")
    }),flush=True)
    return report


def summarize(rows):
    if len(rows) != 6:
        raise RuntimeError("Incomplete three-way run")
    if len({r["input_sha256"] for r in rows}) != 1:
        raise RuntimeError("Input settings mismatch")
    by_arm = {arm:[r for r in rows if r["arm"] == arm]
              for arm in ("stock","candidate_e","stage2_fp16")}
    if any(len(x) != 2 for x in by_arm.values()):
        raise RuntimeError("Missing repeated runs")
    stock = by_arm["stock"][0]
    ref = stock["clip"]
    baseline = statistics.mean(x["wall_seconds"] for x in by_arm["stock"])
    summary = {
        "status": "MEASURED_NOT_QUALITY_APPROVED",
        "order": list(ARMS), "frames": FRAMES,"resolution":RESOLUTION,"steps":STEPS,
        "wall_timer": "fresh full wgp.py subprocess including startup/load, denoise, encode, audio, exit",
        "comparison_source": "same WanGP revision/checkpoint; stock = native ConvRot R0 with opt-in E disabled",
        "input_sha256": stock["input_sha256"],
        "arms": {}
    }
    for arm, group in by_arm.items():
        mean = statistics.mean(x["wall_seconds"] for x in group)
        summary["arms"][arm] = {
            "wall_seconds": [r["wall_seconds"] for r in group],
            "mean_seconds": round(mean,3),
            "saved_vs_stock_seconds": round(baseline-mean,3),
            "reduction_vs_stock_percent": round(100*(baseline-mean)/baseline,3),
            "speedup_vs_stock": round(baseline/mean,4),
            "video_psnr_vs_stock_db": [
                base.psnr(Path(ref),Path(r["clip"]),base.ffmpeg_bin()).get("average_db")
                for r in group
            ],
            "audio_equal_to_stock": [r["audio_hash"]==stock["audio_hash"] for r in group],
            "exact_video_equal_to_stock": [r["video_hash"]==stock["video_hash"] for r in group],
            "reports": [r["case"] for r in group],
        }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    with (OUT/"summary.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["arm","run_1_s","run_2_s","mean_s","saved_vs_stock_s",
                    "reduction_vs_stock_pct","psnr_vs_stock_run1_db","psnr_vs_stock_run2_db"])
        for arm, result in summary["arms"].items():
            w.writerow([arm,*result["wall_seconds"],result["mean_seconds"],
                        result["saved_vs_stock_seconds"],result["reduction_vs_stock_percent"],
                        *result["video_psnr_vs_stock_db"]])
    print("THREEWAY_SUMMARY",json.dumps(summary,sort_keys=True),flush=True)


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("Refusing nonempty output path "+str(OUT))
    for d in ("logs","clips","settings","config","videos","results"):
        (OUT/d).mkdir(parents=True,exist_ok=True)
    base.gpu_free()
    if not IMAGE.is_file():
        raise RuntimeError("Missing 1024x576 conditioning PNG")
    checkpoint = base.CHECKPOINTS/"ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors"
    if not checkpoint.is_file():
        raise RuntimeError("Missing LTX 2.5 INT8 ConvRot checkpoint")
    # Compile before timing; stock never loads Candidate E library.
    base.command_run(["nvcc","-O3","-std=c++17","-arch=sm_70","-shared",
                      "-Xcompiler","-fPIC",
                      "benchmarks/ltx25_convrot/ltx25_cublas_e.cu",
                      "-lcublas","-o",str(base.EXTENSION)],
                     timeout=300,log=OUT/"logs"/"compile-candidate-e.log")
    if not base.EXTENSION.is_file():
        raise RuntimeError("Missing compiled Candidate E library")
    inp = OUT/"input.json"
    inp.write_text(json.dumps(settings(),indent=2)+"\n")
    template = base.initialize_config()
    (OUT/"manifest.json").write_text(json.dumps({
        "source_sha": os.getenv("GITHUB_SHA","manual"),
        "input_sha256":sha(inp), "conditioning_png_sha256":sha(IMAGE),
        "checkpoint":str(checkpoint),
        "checkpoint_file_bytes":checkpoint.stat().st_size,
        "candidate_e_so_sha256":sha(base.EXTENSION),
        "frames":FRAMES, "resolution":RESOLUTION, "steps":STEPS,
        "seed":SEED,"profile":0,"attention":"sdpa",
        "mmgp_profile":0,"no_lora":True,"runs":list(ARMS),
        "timed_process_only":True,
        "model":"ltx2_25_22B_distilled (INT8 ConvRot)",
    },indent=2)+"\n")
    rows = []
    for i,arm in enumerate(ARMS,1):
        rows.append(case_run(arm,i,template))
        (OUT/"partial-results.json").write_text(json.dumps(rows,indent=2)+"\n")
    summarize(rows)
    print("LTX25_33F_3WAY_PASS",str(OUT),flush=True)


if __name__ == "__main__":
    main()
