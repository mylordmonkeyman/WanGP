#!/usr/bin/env python3
"""Release-style full-process V100 A/B: Candidate E radix4 vs Stage 2-only direct FP16.

Uninstrumented ABBA processes are the ONLY timing source. Separate optional
phase-profiling processes synchronize at major boundaries and are NEVER timing
comparators. Same checkpoint, Candidate E, MMGP profile, SDPA, seed, prompt,
resolution, steps and encoding in both arms. Only Stage 2 Quanto FP16 differs.
"""
from __future__ import annotations
import csv
import datetime as dt
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
OUT = Path(os.getenv("LTX25_RELEASE_AB_ROOT", "/cache/ltx25/results/release-ab-" + os.getenv("GITHUB_RUN_ID", "manual")))
IMAGE = ROOT / "benchmarks/ltx25_convrot/inputs/profile_i2v_1024x576.png.png"
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
COHORTS = {
    "t2v49": ("t2v", 49),
    "i2v49": ("i2v", 49),
    "i2v241": ("i2v", 241),
    "t2v241": ("t2v", 241),
}
STAGE2 = "stage2:wide,stage2:tall,stage2:balanced"
ARMS = ("prior", "stage2", "stage2", "prior")
base.OUT = OUT
base.EXTENSION = OUT / "libltx25_cublas_e_sm70.so"
base.PROMPT = PROMPT
base.SEED = 20261112
base.STEPS = 8
base.RESOLUTION = "1024x576"

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()

def settings(mode, frames):
    data = {
        "model_type": "ltx2_25_22B_distilled",
        "prompt": PROMPT, "negative_prompt": "",
        "seed": base.SEED, "num_inference_steps": 8,
        "video_length": frames, "resolution": "1024x576",
        "NAG_scale": 1.1, "audio_cfg_scale": 1.5,
        "activated_loras": [], "loras_multipliers": "",
        "audio_prompt_type": "", "video_prompt_type": "",
        "image_mode": 0, "spatial_upsampling": "", "temporal_upsampling": "",
    }
    if mode == "i2v":
        data.update(image_start=str(IMAGE), image_prompt_type="S")
    return data

def env_for(arm, name, diagnostic=False):
    env = base.base_env()  # Exactly the same Candidate E radix4/vector8/direct/fused baseline.
    # Scrub inherited instrumentation. No profiler, tracing, audit or sham in timing runs.
    for k in list(env):
        if k.startswith(("WAN2GP_LTX25_PIPELINE", "LTX25_PIPELINE_",
                         "LTX25_GEMM_PROFILE_", "LTX25_FP16_GEMM_")):
            env.pop(k, None)
    env["WAN2GP_LTX25_FP16_GEMM"] = "0"
    env["WAN2GP_LTX2_BENCH_TRANSFORMER"] = "0"
    if arm == "stage2":
        env.update({
            "WAN2GP_LTX25_FP16_GEMM": "1",
            "LTX25_FP16_GEMM_BACKEND": "direct",
            "LTX25_FP16_GEMM_GROUPS": STAGE2,
            "LTX25_FP16_GEMM_OPS": "qbytes_mm",
            "LTX25_FP16_GEMM_MIN_M": "1",
            "LTX25_FP16_GEMM_MIN_DIM": "64",
            "LTX25_FP16_GEMM_GUARD": "first",
            "LTX25_FP16_GEMM_SHAM": "0",
            "LTX25_FP16_GEMM_AUDIT_PER_SHAPE": "0",
            "LTX25_FP16_GEMM_SHAPE_CENSUS": "0",
            "LTX25_FP16_GEMM_TRACE": "0",
            "LTX25_FP16_GEMM_STATS_PATH": str(OUT/"results"/(name+"-fp16.json")),
        })
    if diagnostic:
        env.update({
            "WAN2GP_LTX25_PIPELINE_PROFILE": "1",
            "WAN2GP_LTX2_BENCH_TRANSFORMER": "1",
            "LTX25_PIPELINE_OP_SAMPLE": "0",
            "LTX25_PIPELINE_GEMM_CENSUS": "0",
            "LTX25_PIPELINE_TRACE_ATTRIBUTION": "0",
            "LTX25_PIPELINE_PROFILE_PATH": str(OUT/"results"/(name+"-phases.json")),
        })
    return env

def gpu_snapshot():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu",
         "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
    return r.stdout.strip() if r.returncode == 0 else "unavailable: "+r.stderr[-200:]

def run_case(cohort, mode, frames, arm, index, template, diagnostic=False):
    name = f"{cohort}-{arm}-{'diag' if diagnostic else index}"
    case_cfg, target = base.config_for(name, template)
    inp = OUT/"settings"/f"{cohort}.json"
    env = env_for(arm, name, diagnostic)
    cmd = [base.PY, "wgp.py", "--offline", "--config", str(case_cfg),
           "--process", str(inp), "--output-dir", str(target),
           "--profile", "0", "--attention", "sdpa",
           "--no-prevent-power-throttling", "--verbose", "1"]
    base.command_run(cmd + ["--dry-run"], env=env, timeout=300,
                     log=OUT/"logs"/f"{name}-dryrun.log")
    base.gpu_free()
    clock_before = gpu_snapshot()
    print("RELEASE_AB_START", name, dt.datetime.now(dt.timezone.utc).isoformat(),
          "profile" if diagnostic else "UNINSTRUMENTED", flush=True)
    # time.monotonic() around the entire fresh subprocess, including model load,
    # MMGP offload, diffusion, VAE/audio decode, ffmpeg encoding and process exit.
    # Do NOT enable per-forward profiler or per-shape numerical audits here.
    execution = base.command_run(cmd, env=env, timeout=3600,
                 log=OUT/"logs"/f"{name}-generation.log", check=False)
    clock_after = gpu_snapshot()
    report = {
        "name": name, "cohort": cohort, "mode": mode, "frames": frames,
        "arm": arm, "index": index, "diagnostic": diagnostic,
        "wall_seconds": execution["wall_seconds"],
        "returncode": execution["returncode"],
        "settings_sha256": sha(inp), "clock_before": clock_before,
        "clock_after": clock_after, "source_sha": os.getenv("GITHUB_SHA", "local"),
        "log": execution["log"],
    }
    if execution["returncode"]:
        (OUT/"results"/f"{name}.json").write_text(json.dumps(report, indent=2)+"\n")
        raise RuntimeError(f"{name} returned {execution['returncode']}: {execution['log_tail']}")
    logtext = Path(execution["log"]).read_text(errors="replace")
    if "[LTX25_CUBLAS_E] {" not in logtext:
        raise RuntimeError(name + ": Candidate E never reported stats")
    if diagnostic:
        phases = OUT/"results"/f"{name}-phases.json"
        if not phases.is_file():
            raise RuntimeError(name + ": phase profiler report absent")
        pdata = json.loads(phases.read_text())
        report["phases"] = pdata.get("phases", [])
        report["profile_status"] = pdata.get("status", "UNKNOWN")
        report["profile_errors"] = pdata.get("errors", [])
        # A phase profiler synchronizes CUDA at boundaries; NOT a wall-time A/B.
    else:
        clips = [p for p in target.rglob("*.mp4") if p.stat().st_size > 1024]
        if len(clips) != 1:
            raise RuntimeError(f"{name}: expected exactly one MP4, found {len(clips)}")
        dst = OUT/"clips"/f"{name}.mp4"
        shutil.copy2(clips[0], dst)
        report["clip"] = str(dst)
        report["clip_bytes"] = dst.stat().st_size
        report["video_hash"] = base.frames_hash(dst, base.ffmpeg_bin())
        report["audio_hash"] = base.audio_hash(dst, base.ffmpeg_bin())
    if arm == "stage2":
        statsfile = OUT/"results"/f"{name}-fp16.json"
        if not statsfile.exists():
            raise RuntimeError(f"{name}: missing Stage 2 dispatch stats")
        stats = json.loads(statsfile.read_text())
        count = stats.get("stats", {})
        report["quanto_counts"] = {
            k: int(count.get(k, 0)) for k in (
                "observed_qbytes_mm", "converted_qbytes_mm", "sham_native",
                "guard_input_reference", "guard_output_reference", "guard_checks",
                "bypassed_unselected_stage1_forwards")}
        report["stage_forward_counts"] = stats.get("transformer_forward_calls", {})
        report["skip_reasons"] = stats.get("skip_reasons", {})
        report["by_shape"] = stats.get("by_shape", {})
        if not stats.get("backend") == "direct" or stats.get("sham") is not False:
            raise RuntimeError(name + ": incorrect FP16 backend/sham dispatch")
        if stats.get("persistent_fp16_weight_copies") != 0:
            raise RuntimeError(name + ": forbidden persistent FP16 weight copies")
        if (count.get("converted_qbytes_mm", 0) <= 0
                or count.get("bypassed_unselected_stage1_forwards", 0) <= 0
                or count.get("sham_native", 0) != 0
                or stats.get("blocked_shapes")):
            raise RuntimeError(f"{name}: invalid Stage 2-only dispatch counters: {count}")
        if any(k.startswith("stage1:") and v.get("converted", 0)
               for k, v in stats.get("by_shape", {}).items()):
            raise RuntimeError(name + ": Stage 1 was unexpectedly converted")
    else:
        if "LTX25_SELECTIVE_FP16_GEMM" in logtext:
            raise RuntimeError(name + ": prior baseline unexpectedly used FP16 selector")
    (OUT/"results"/f"{name}.json").write_text(json.dumps(report, indent=2)+"\n")
    print("RELEASE_AB_CASE_PASS", json.dumps({
        "name": name, "seconds": report["wall_seconds"],
        "quanto": report.get("quanto_counts"), "profile": diagnostic}), flush=True)
    return report

def video_psnr(a, b):
    result = base.psnr(Path(a), Path(b), base.ffmpeg_bin())
    return result.get("average_db") if result.get("status") == "PASS" else None

def summarize(cohort, rows):
    a = [r for r in rows if r["arm"] == "prior"]
    b = [r for r in rows if r["arm"] == "stage2"]
    if len(a) != 2 or len(b) != 2:
        raise RuntimeError(cohort + ": incomplete ABBA comparison")
    ma = statistics.mean(r["wall_seconds"] for r in a)
    mb = statistics.mean(r["wall_seconds"] for r in b)
    if any(r["settings_sha256"] != a[0]["settings_sha256"] for r in rows):
        raise RuntimeError("Mismatched settings within "+cohort)
    ref = a[0]
    quality = {
        "prior_repeat_video_psnr_db": video_psnr(ref["clip"], a[1]["clip"]),
        "prior_repeat_audio_equal": ref["audio_hash"] == a[1]["audio_hash"],
        "stage2_video_psnr_db": [video_psnr(ref["clip"], r["clip"]) for r in b],
        "stage2_audio_equal": [ref["audio_hash"] == r["audio_hash"] for r in b],
        "stage2_frames_exact": [ref["video_hash"] == r["video_hash"] for r in b],
    }
    result = {
        "cohort": cohort, "status": "MEASURED_NOT_QUALITY_APPROVED",
        "order": [r["arm"] for r in rows],
        "baseline_wall_seconds": [r["wall_seconds"] for r in a],
        "stage2_wall_seconds": [r["wall_seconds"] for r in b],
        "prior_mean_seconds": round(ma, 3), "stage2_mean_seconds": round(mb, 3),
        "saved_seconds": round(ma-mb, 3),
        "wall_reduction_percent": round(100*(ma-mb)/ma, 3),
        "speedup_factor": round(ma/mb, 4), "quality": quality,
        "stage2_dispatch": [r["quanto_counts"] for r in b],
        "settings_sha256": ref["settings_sha256"],
        "timing_method": "fresh entire wgp.py subprocess; no CUPTI, audit or pipeline phase profiler",
    }
    (OUT/"results"/f"{cohort}-summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print("RELEASE_AB_SUMMARY",json.dumps(result,sort_keys=True),flush=True)
    return result

def main():
    names = [x.strip() for x in os.getenv("LTX25_RELEASE_COHORTS",
                                          "t2v49,i2v49,i2v241").split(",") if x.strip()]
    if not names or any(x not in COHORTS for x in names):
        raise RuntimeError("Invalid LTX25_RELEASE_COHORTS: "+repr(names))
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("Refusing nonempty results path "+str(OUT))
    for d in ("logs","clips","settings","config","videos","results"):
        (OUT/d).mkdir(parents=True,exist_ok=True)
    base.gpu_free()
    if not IMAGE.is_file():
        raise RuntimeError("Missing I2V bar PNG")
    if not (base.CHECKPOINTS/"ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors").is_file():
        raise RuntimeError("Missing checkpoint on offline V100 runner")
    base.command_run(
        ["nvcc", "-O3", "-std=c++17", "-arch=sm_70", "-shared",
         "-Xcompiler", "-fPIC", "benchmarks/ltx25_convrot/ltx25_cublas_e.cu",
         "-lcublas", "-o", str(base.EXTENSION)], timeout=300,
        log=OUT/"logs"/"compile-candidate-e.log")
    if not base.EXTENSION.is_file():
        raise RuntimeError("Candidate E shared library not compiled")
    # An installed release would use the same prebuilt binary. Build and
    # preflight happen before any timing, with identical artifact in both arms.
    (OUT/"input.json").write_text(json.dumps(settings("t2v",49),indent=2)+"\n")
    template = base.initialize_config()
    (OUT/"manifest.json").write_text(json.dumps({
        "source_sha": os.getenv("GITHUB_SHA","local"),
        "image_sha256": sha(IMAGE), "candidate_e_shared_object_sha256": sha(base.EXTENSION),
        "checkpoints_dir": str(base.CHECKPOINTS), "selected_cohorts": names,
        "seed": base.SEED, "steps": 8, "resolution": "1024x576",
        "lora": "none (not installed on V100 runner)",
        "profiling": "off in timed runs; separate diagnostic only",
        "device": subprocess.run(["nvidia-smi", "-L"], text=True,
                                 capture_output=True).stdout.strip(),
    }, indent=2)+"\n")
    results = {}
    for cohort in names:
        mode,frames = COHORTS[cohort]
        (OUT/"settings"/f"{cohort}.json").write_text(
            json.dumps(settings(mode,frames),indent=2)+"\n")
        rows = []
        for i,arm in enumerate(ARMS,1):
            row=run_case(cohort,mode,frames,arm,i,template)
            rows.append(row)
        results[cohort] = summarize(cohort, rows)
        (OUT/"summary.json").write_text(json.dumps(results,indent=2)+"\n")
    # Fine-grained stage accounting is collected in separate instrumented
    # processes, never smuggled into reported release-style wall times.
    if os.getenv("LTX25_RELEASE_DIAGNOSTICS","1") == "1":
        for cohort in names:
            if cohort not in {"t2v49","i2v241"}:
                continue
            mode,frames = COHORTS[cohort]
            for arm in ("prior","stage2"):
                run_case(cohort,mode,frames,arm,0,template,diagnostic=True)
    with (OUT/"summary.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["cohort","prior_mean_s","stage2_mean_s","saved_s","reduction_pct"])
        for k,v in results.items():
            w.writerow([k,v["prior_mean_seconds"],v["stage2_mean_seconds"],
                        v["saved_seconds"],v["wall_reduction_percent"]])
    print("LTX25_RELEASE_AB_PASS",str(OUT),flush=True)

if __name__ == "__main__":
    main()
