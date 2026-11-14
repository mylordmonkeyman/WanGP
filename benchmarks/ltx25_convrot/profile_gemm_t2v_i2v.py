#!/usr/bin/env python3
"""Matched T2V/I2V 49-frame profiler on the offline V100 LTX-2.5 runner.

Runs the SAME prompt, seed, base checkpoint, resolution, steps, NAG, audio
guidance, no LoRA and Candidate E settings in each arm. Only image conditioning
differs. Keeps the underlying actual Quanto GEMMs in BF16 (SHAM) to attribute
native execution. One warmed transformer forward per stage gets full CUPTI/
PyTorch profiler sampled GPU kernel and CPU op shape data. Output stage totals
are actual Quanto full-run counts; GPU kernel time covers SAMPLED forwards only.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback

import selective_fp16_t2v_smoke as base

ROOT = Path(__file__).resolve().parents[2]
INPUT_IMAGE = ROOT / "benchmarks/ltx25_convrot/inputs/profile_i2v_1024x576.png.png"
OUTPUT = Path(os.getenv("LTX25_GEMM_PROFILE_ROOT",
                       f"/cache/ltx25/results/t2v-i2v-gemm-profile-{os.getenv('GITHUB_RUN_ID', 'manual')}")).resolve()
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
SEED=20261112
FRAMES=49
STEPS=8
RESOLUTION="1024x576"
GROUPS="stage1:wide,stage1:tall,stage1:balanced,stage2:wide,stage2:tall,stage2:balanced"

# Use all validated V100 smoke infrastructure, never edit the global runner config.
base.OUT = OUTPUT
base.EXTENSION = OUTPUT / "libltx25_cublas_e_sm70.so"
base.PROMPT = PROMPT
base.SEED = SEED
base.FRAMES = FRAMES
base.STEPS = STEPS
base.RESOLUTION = RESOLUTION

def file_sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def check_png():
    # stdlib dimension check does not require Pillow or decode full file.
    import struct
    with INPUT_IMAGE.open("rb") as f:
        h=f.read(24)
    if len(h)!=24 or h[:8]!=b"\x89PNG\r\n\x1a\n" or h[12:16]!=b"IHDR":
        raise RuntimeError(f"Invalid PNG: {INPUT_IMAGE}")
    width,height=struct.unpack(">II",h[16:24])
    if (width,height)!=(1024,576):
        raise RuntimeError(f"Image dimensions {width}x{height}, expected 1024x576")

def make_settings(mode):
    settings={
      "model_type":"ltx2_25_22B_distilled",
      "prompt":PROMPT,
      "negative_prompt":"",
      "seed":SEED, "num_inference_steps":STEPS,
      "video_length":FRAMES, "resolution":RESOLUTION,
      "NAG_scale":1.1,"audio_cfg_scale":1.5,
      "audio_prompt_type":"","video_prompt_type":"",
      "image_mode":0,
      "activated_loras":[], "loras_multipliers":"",
      "spatial_upsampling":"","temporal_upsampling":"",
    }
    if mode=="i2v":
        settings.update(image_start=str(INPUT_IMAGE),image_prompt_type="S")
    return settings

def case_env(case):
    env=base.base_env()
    env.update({
      "WAN2GP_LTX25_FP16_GEMM":"1",
      "LTX25_FP16_GEMM_BACKEND":"direct",
      "LTX25_FP16_GEMM_SHAM":"1",  # Model numerical pathway identical to native BF16.
      "LTX25_FP16_GEMM_GROUPS":GROUPS,
      "LTX25_FP16_GEMM_OPS":"qbytes_mm",
      "LTX25_FP16_GEMM_MIN_M":"1",
      "LTX25_FP16_GEMM_MIN_DIM":"64",
      "LTX25_FP16_GEMM_GUARD":"first",
      "LTX25_FP16_GEMM_AUDIT_PER_SHAPE":"0",
      "LTX25_FP16_GEMM_SHAPE_CENSUS":"1",
      "LTX25_GEMM_PROFILE_QUANTO_SCOPE":"1",
      "LTX25_FP16_GEMM_STATS_PATH":str(OUTPUT/"results"/f"{case}-quanto.json"),
      "WAN2GP_LTX25_PIPELINE_PROFILE":"1",
      "WAN2GP_LTX2_BENCH_TRANSFORMER":"1",
      "LTX25_PIPELINE_OP_SAMPLE":"1",
      "LTX25_PIPELINE_GEMM_CENSUS":"1",
      "LTX25_PIPELINE_PROFILE_PATH":str(OUTPUT/"results"/f"{case}-pipeline.json"),
    })
    return env

def run_case(case,template):
    env=case_env(case)
    settings=make_settings(case)
    settings_file=OUTPUT/f"{case}-input.json"
    settings_file.write_text(json.dumps(settings,indent=2)+"\n")
    cfg,target=base.config_for(case,template)
    cmd=[base.PY,"wgp.py","--offline","--config",str(cfg),
         "--process",str(settings_file),"--output-dir",str(target),
         "--profile","0","--attention","sdpa",
         "--no-prevent-power-throttling","--verbose","1"]
    print(f"PROFILE_BEGIN {case} {dt.datetime.now(dt.timezone.utc).isoformat()}",flush=True)
    dry=base.command_run(cmd+["--dry-run"],env=env,timeout=300,
                         log=OUTPUT/"logs"/f"{case}-dryrun.log",check=True)
    if "Validation complete. 1/1 task(s) valid." not in (
        OUTPUT/"logs"/f"{case}-dryrun.log").read_text():
        raise RuntimeError(f"Expected dry-run validation marker not seen for {case}")
    base.gpu_free()
    result=base.command_run(cmd,env=env,timeout=2400,
                            log=OUTPUT/"logs"/f"{case}-generation.log",check=False)
    print(f"PROFILE_RUN_EXIT {case} rc={result['returncode']} wall_s={result['wall_seconds']}",
          flush=True)
    if result["returncode"]:
        raise RuntimeError(f"{case} video generation failed: {result['log_tail']}")
    pipeline_file=OUTPUT/"results"/f"{case}-pipeline.json"
    quanto_file=OUTPUT/"results"/f"{case}-quanto.json"
    if not pipeline_file.is_file() or not quanto_file.is_file():
        raise RuntimeError(f"{case} profiling files missing: {pipeline_file}, {quanto_file}")
    pipeline=json.loads(pipeline_file.read_text())
    quanto=json.loads(quanto_file.read_text())
    samples=pipeline.get("gemm_census_samples",[])
    if {s["phase"] for s in samples}!={"stage1","stage2"}:
        raise RuntimeError(f"{case} incomplete profiled stages; samples={len(samples)} "
                           f"errors={pipeline.get('errors')} status={pipeline.get('status')}")
    if any(s["cuda_kernel_event_count"]==0 for s in samples):
        raise RuntimeError(f"{case}: CUDA kernel CUPTI activity missing in a sample")
    counts=quanto.get("stats",{})
    observed=int(counts.get("observed_qbytes_mm",0))
    if observed<=0 or counts.get("converted_qbytes_mm",0)!=0:
        raise RuntimeError(f"{case}: no native Quanto GEMMs or unexpected FP16 conversion {counts}")
    clips=[p for p in target.rglob("*.mp4") if p.stat().st_size>1000]
    if len(clips)!=1:
        raise RuntimeError(f"{case}: expected one output MP4, got {len(clips)}")
    clip=OUTPUT/f"{case}.mp4"
    shutil.copy2(clips[0],clip)
    # Actual full-run Quanto counts from stage-qualified shape signatures.
    stage_counts={"stage1":0,"stage2":0}
    for shape in quanto.get("shape_census",[]):
        stage_counts[shape["stage"]]+=shape["calls"]
    row={
        "mode":case,"status":"PASS","video":str(clip),
        "wall_seconds_instrumented":result["wall_seconds"],
        "settings_sha256":file_sha256(settings_file),
        "stage_quanto_calls_full_run":stage_counts,
        "total_quanto_calls_full_run":observed,
        "transformer_calls":pipeline.get("transformer_by_phase",{}),
        "stage_samples":[{
          "stage":s["phase"],"sampled_forward":s["forward_index"],
          "cpu_gemm_ops_all":s["cpu_matmul_operator_invocations"],
          "cpu_gemm_ops_inside_quanto":s["cpu_matmul_in_tagged_quanto"],
          "cpu_gemm_ops_outside_quanto":s["cpu_matmul_outside_tagged_quanto"],
          "cuda_kernel_count":s["cuda_kernel_event_count"],
          "cuda_kernel_ms":s["cuda_kernel_gpu_ms_sum"],
          "cuda_kernel_categories":s["cuda_kernel_family"],
          "top_unattributed_cpu_shapes":s["top_cpu_matmul_shapes_outside_quanto"][:20],
        } for s in samples],
        "pipeline_errors":pipeline.get("errors",[]),
        "quanto_sham":quanto.get("sham"),
        "quanto_backend":quanto.get("backend"),
    }
    (OUTPUT/"results"/f"{case}-summary.json").write_text(json.dumps(row,indent=2)+"\n")
    print("GEMM_PROFILE_CASE "+json.dumps(row,sort_keys=True),flush=True)
    return row

def write_summary(rows):
    result={
        "settings_match_except_image":True,
        "resolution":RESOLUTION, "frames":FRAMES,"steps":STEPS,"seed":SEED,
        "lora":"none", "nag_scale":1.1,"audio_cfg_scale":1.5,
        "image_sha256":file_sha256(INPUT_IMAGE),
        "model":"LTX-2.5 22B distilled Quanto INT8 ConvRot",
        "method":"Direct Quanto BF16 sham with CUPTI stage sampled per forward",
        "full_run_quanto":{r["mode"]:r["stage_quanto_calls_full_run"] for r in rows},
        "cases":rows,
        "limitations":[
            "The two runs differ intentionally only by first-frame image conditioning.",
            "GPU kernel profiling includes overhead; instrumented wall times cannot establish production speed.",
            "Only one warmed transformer forward per stage is CUPTI profiled.",
            "GPU kernel name classifier is heuristic; unknown CUDA kernels may be GEMMs.",
            "CPU operator hierarchies include nested matmul calls; do not sum CPU operator counts.",
            "Tagged Quanto hierarchy used to avoid double-counting native Quanto's aten::mm.",
            "A CUDA kernel lacking a CPU parent is unknown, not counted as definitely missed.",
        ],
    }
    (OUTPUT/"comparison.json").write_text(json.dumps(result,indent=2)+"\n")
    lines=[
        "LTX-2.5 V100 T2V vs I2V controlled GEMM + GPU kernel census",
        f"Resolution {RESOLUTION}; frames {FRAMES}; 8-step preset; no LoRA; same text/audio prompt",
        f"Image SHA256 {result['image_sha256']}",
        "All runs are native BF16 GEMM (Quanto-direct SHAM), under profiler instrumentation.",
        "",
        "Actual whole-video Quanto calls:",
    ]
    for row in rows:
        line=f"  {row['mode']:4s} Stage 1={row['stage_quanto_calls_full_run']['stage1']:6d} "\
             f"Stage 2={row['stage_quanto_calls_full_run']['stage2']:6d} "\
             f"TOTAL={row['total_quanto_calls_full_run']:6d}"
        lines.append(line)
    lines.append("")
    for row in rows:
        lines.append(f"{row['mode']} sampled forwards (CUDA GPU kernel times, not full run):")
        for s in row["stage_samples"]:
            lines.append(f"  {s['stage']:6s} kernels={s['cuda_kernel_count']:6d} "
                         f"sum_gpu_ms={s['cuda_kernel_ms']:10.2f} "
                         f"outside_Quanto_CPU_GEMMs={s['cpu_gemm_ops_outside_quanto']}")
            for cat in s["cuda_kernel_categories"]:
                lines.append(f"    {cat['family']:<24s} {cat['calls']:>7} kernels; {cat['gpu_ms']:>10.2f} GPU-ms")
    lines.extend(["","Details: results/*-pipeline.json and *-quanto.json",
                  "Caution: one sampled forward per stage; kernel names are heuristic."])
    (OUTPUT/"comparison.txt").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)

def main():
    if not Path(base.PY).is_file():
        raise RuntimeError(f"Python not installed: {base.PY}")
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"Refusing occupied output directory {OUTPUT}")
    for d in ("videos","logs","config","results"):
        (OUTPUT/d).mkdir(parents=True,exist_ok=True)
    check_png()
    base.gpu_free()
    assert (base.CHECKPOINTS/"ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors").is_file()
    base.command_run(["nvcc","-O3","-std=c++17","-arch=sm_70","-shared",
                      "-Xcompiler","-fPIC","benchmarks/ltx25_convrot/ltx25_cublas_e.cu",
                      "-lcublas","-o",str(base.EXTENSION)],timeout=300,
                     log=OUTPUT/"logs"/"compile-candidate-e.log")
    # Initialize template once with T2V process to keep all settings shared.
    (OUTPUT/"input.json").write_text(json.dumps(make_settings("t2v"),indent=2)+"\n")
    template=base.initialize_config()
    rows=[]
    try:
        for mode in ("t2v","i2v"):
            rows.append(run_case(mode,template))
    finally:
        if rows: write_summary(rows)
    if len(rows)!=2:
        raise RuntimeError("Both T2V and I2V reports required")
    print("LTX25_T2V_I2V_GEMM_PROFILE_PASS",str(OUTPUT),flush=True)

if __name__=="__main__":
    main()
