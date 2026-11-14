#!/usr/bin/env python3
"""Full-process Tesla V100 T2V comparison of four execution paths.

Models: same baked LTX-2.5 22B distilled INT8 ConvRot checkpoint and
settings on (1) genuine pre-optimization upstream WanGP checkout at c8d47da,
(2) Candidate E/radix4, (3) Candidate E/radix4 + Stage 2 direct FP16
qbytes_mm, (4) Candidate E/radix4 + both Stage 1 and Stage 2 direct FP16.

One independent 73-frame cohort and one independent 241-frame cohort.
Only subprocess creation through exit is timed; dry-runs, compilation,
video hashing, ffprobe, quality metrics, and V100 checks are NOT timed.
"""
from __future__ import annotations

import argparse
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
import time

import selective_fp16_t2v_smoke as base

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "stock-wangp-original"
ORIGINAL_SHA = "c8d47da27023476777a0874300bbc9a4126acca6"
FRAME_CHOICES = (73, 241)
RESOLUTION = "1024x576"
STEPS = 8
SEED = 20261112
ARMS = ("stock_upstream", "candidate_e", "stage2_fp16", "both_fp16")
GROUP_STAGE2 = "stage2:wide,stage2:tall,stage2:balanced"
GROUP_BOTH = "stage1:wide,stage1:tall,stage1:balanced," + GROUP_STAGE2
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
OUT = None


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def settings(frames):
    # T2V: never pass a conditioning image or image_prompt_type here.
    return {
        "model_type":"ltx2_25_22B_distilled",
        "prompt":PROMPT, "negative_prompt":"",
        "seed":SEED, "num_inference_steps":STEPS,
        "video_length":frames, "resolution":RESOLUTION,
        "NAG_scale":1.1, "audio_cfg_scale":1.5,
        "activated_loras":[], "loras_multipliers":"",
        "audio_prompt_type":"", "video_prompt_type":"",
        "image_mode":0, "spatial_upsampling":"", "temporal_upsampling":"",
    }


def verify_source():
    r = subprocess.run(["git","rev-parse","HEAD"], cwd=UPSTREAM,
                       capture_output=True, text=True, check=True)
    if r.stdout.strip() != ORIGINAL_SHA:
        raise RuntimeError(f"Wrong original upstream commit: {r.stdout.strip()}")
    stock = (UPSTREAM/"shared/qtypes/int8_convrot.py").read_text()
    if "WAN2GP_LTX25_CUBLAS_E_MODE" in stock or "ltx25_cublas_sm70" in stock:
        raise RuntimeError("Historical stock ConvRot source unexpectedly modified")
    for p in ("shared/kernels/ltx25_cublas_sm70.py",
              "shared/kernels/ltx25_selective_fp16_gemm.py"):
        if (UPSTREAM/p).exists():
            raise RuntimeError("Historical upstream contains custom optimization: "+p)
    return {
        "original_commit":ORIGINAL_SHA,
        "upstream_repository":"deepbeepmeep/WanGP",
        "stock_convrot_sha256":sha(UPSTREAM/"shared/qtypes/int8_convrot.py"),
        "stock_wgp_sha256":sha(UPSTREAM/"wgp.py"),
    }


def optimized_env(arm, name):
    env=base.base_env()
    for k in list(env):
        if k.startswith(("LTX25_FP16_GEMM_", "LTX25_PIPELINE_", "LTX25_GEMM_PROFILE_",
                         "WAN2GP_LTX25_PIPELINE", "WAN2GP_LTX2_BENCH_TRANSFORMER")):
            env.pop(k, None)
    env.update({
        "WAN2GP_LTX25_CUBLAS_E_MODE":"fp32rot",
        "WAN2GP_LTX25_HADAMARD":"radix4",
        "WAN2GP_LTX25_FP16_GEMM":"0",
        "WAN2GP_LTX25_REAL_ACTIVATION_AUDIT":"0",
        "WAN2GP_LTX25_PROFILE_STAGES":"0",
        "WAN2GP_LTX2_BENCH_TRANSFORMER":"0",
    })
    if arm in ("stage2_fp16", "both_fp16"):
        env.update({
            "WAN2GP_LTX25_FP16_GEMM":"1",
            "LTX25_FP16_GEMM_BACKEND":"direct",
            "LTX25_FP16_GEMM_GROUPS":GROUP_STAGE2 if arm=="stage2_fp16" else GROUP_BOTH,
            "LTX25_FP16_GEMM_OPS":"qbytes_mm",
            "LTX25_FP16_GEMM_MIN_M":"1",
            "LTX25_FP16_GEMM_MIN_DIM":"64",
            "LTX25_FP16_GEMM_GUARD":"first",
            "LTX25_FP16_GEMM_SHAM":"0",
            "LTX25_FP16_GEMM_AUDIT_PER_SHAPE":"0",
            "LTX25_FP16_GEMM_SHAPE_CENSUS":"0",
            "LTX25_FP16_GEMM_TRACE":"0",
            "LTX25_FP16_GEMM_STATS_PATH":str(OUT/"results"/(name+"-fp16.json")),
        })
    return env


def historical_env():
    env=dict(os.environ)
    for k in list(env):
        if k.startswith(("WAN2GP_LTX25_", "LTX25_FP16_GEMM_", "LTX25_CUBLAS_",
                         "LTX25_GEMM_", "LTX25_PIPELINE_", "WAN2GP_LTX2_BENCH_TRANSFORMER")):
            env.pop(k,None)
    env.update({
        "PYTHONPATH":str(UPSTREAM),
        "GRADIO_ANALYTICS_ENABLED":"False",
        "OMP_NUM_THREADS":"8",
        "HF_HUB_OFFLINE":"1",
        "TRANSFORMERS_OFFLINE":"1",
    })
    return env


def execute(argv, cwd, env, log_path, timeout=3600):
    started=time.monotonic()
    with log_path.open("w",encoding="utf8") as file:
        try:
            code=subprocess.run(argv,cwd=cwd,env=env,stdout=file,
                                stderr=subprocess.STDOUT,timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            file.write(f"\nTIMEOUT {timeout}s\n")
            code=124
    return {"returncode":code, "wall_seconds":round(time.monotonic()-started,3),
            "log":str(log_path)}


def probe_video(path, expected_frames):
    ffmpeg=base.ffmpeg_bin()
    # The source stream line reports the encoded size. Reject any silent
    # WanGP resolution realignment so reported 1024x576 is truly measured.
    p=subprocess.run([ffmpeg,"-hide_banner","-i",str(path)],
                     capture_output=True,text=True,timeout=30)
    video_line=next((line for line in p.stderr.splitlines() if "Video:" in line), "")
    dims=re.search(r"(?<![0-9])(\d{3,5})x(\d{3,5})(?![0-9])",video_line)
    if dims is None:
        raise RuntimeError("Cannot identify encoded video dimensions: "+p.stderr[-800:])
    width,height=map(int,dims.groups())
    if (width,height)!=(1024,576):
        raise RuntimeError(f"Encoded {width}x{height}, expected 1024x576")
    base.FRAMES=expected_frames
    return {
        "width":width,"height":height,"frames":expected_frames,
        "decoded_video_sha256":base.frames_hash(path,ffmpeg),
        "decoded_audio_sha256":base.audio_hash(path,ffmpeg),
    }


def read_candidate_stats(text):
    m=re.search(r"^\[LTX25_CUBLAS_E\] (\{[^\n]+\})",text,re.M)
    return json.loads(m.group(1)) if m else None


def verify_fp16(path, arm):
    report=json.loads(path.read_text())
    stats=report.get("stats",{})
    if report.get("backend")!="direct" or report.get("sham") is not False:
        raise RuntimeError(arm+": FP16 is not direct non-sham")
    if report.get("persistent_fp16_weight_copies")!=0:
        raise RuntimeError(arm+": persistent FP16 weight copies forbidden")
    if int(stats.get("converted_qbytes_mm",0))<1:
        raise RuntimeError(arm+": did not convert any real Quanto GEMMs")
    if int(stats.get("guard_output_reference",0))+int(stats.get("guard_input_reference",0)):
        # Do not silently pass for an arm whose FP16 range is unsafe.
        print("T2V_FP16_NUMERICAL_GUARD_FALLBACK",arm,stats,flush=True)
    forwards=report.get("transformer_forward_calls",{})
    by_shape=report.get("by_shape",{})
    converted_per_stage={
        phase:sum(int(v.get("converted",0))
                  for key,v in by_shape.items() if key.startswith(phase+":"))
        for phase in ("stage1","stage2")
    }
    if converted_per_stage["stage2"]<1:
        raise RuntimeError(arm+": Stage 2 conversion not exercised")
    if arm=="stage2_fp16":
        if converted_per_stage["stage1"]!=0:
            raise RuntimeError("Stage 1 was unexpectedly converted in Stage 2-only arm")
        if int(stats.get("bypassed_unselected_stage1_forwards",0))<1:
            raise RuntimeError("Stage 1 was not bypassed in Stage 2-only arm")
    if arm=="both_fp16":
        if converted_per_stage["stage1"]<1 or int(forwards.get("stage1",0))<1:
            raise RuntimeError("Stage 1 was not converted in both-stage FP16 arm")
    return {
        "converted_per_stage":converted_per_stage,
        "observed_qbytes_mm":int(stats.get("observed_qbytes_mm",0)),
        "converted_qbytes_mm":int(stats.get("converted_qbytes_mm",0)),
        "guard_checks":int(stats.get("guard_checks",0)),
        "guard_fallbacks":int(stats.get("guard_input_reference",0))+int(stats.get("guard_output_reference",0)),
        "blocked_shapes":report.get("blocked_shapes",[]),
        "forward_counts":forwards,
        "skip_reasons":report.get("skip_reasons",{}),
        "persistent_fp16_weight_copies":report.get("persistent_fp16_weight_copies"),
    }


def run_case(frames,index,arm,template):
    name=f"t2v{frames}-{index:02d}-{arm}"
    cfg,target=base.config_for(name,template)
    is_stock=arm=="stock_upstream"
    cwd=UPSTREAM if is_stock else ROOT
    env=historical_env() if is_stock else optimized_env(arm,name)
    cmd=[base.PY,"wgp.py","--offline","--config",str(cfg),
         "--process",str(OUT/"input.json"),"--output-dir",str(target),
         "--profile","0","--attention","sdpa","--no-prevent-power-throttling",
         "--verbose","1"]
    dry=execute(cmd+["--dry-run"],cwd,env,OUT/"logs"/f"{name}-dryrun.log",timeout=300)
    if dry["returncode"]:
        raise RuntimeError(name+": dry run failed: "+
                           Path(dry["log"]).read_text(errors="replace")[-2500:])
    base.gpu_free()
    print("T2V_4WAY_BEGIN",name,flush=True)
    timed=execute(cmd,cwd,env,OUT/"logs"/f"{name}-generation.log",timeout=3600)
    row={
        "name":name,"arm":arm,"frames":frames,"index":index,
        "resolution_requested":RESOLUTION,"steps":STEPS,"seed":SEED,
        "wall_seconds":timed["wall_seconds"],
        "returncode":timed["returncode"],"log":timed["log"],
        "settings_sha256":sha(OUT/"input.json"),
        "code_sha":ORIGINAL_SHA if is_stock else os.getenv("GITHUB_SHA","manual"),
        "source":"historical-upstream" if is_stock else "optimized",
    }
    record=OUT/"results"/(name+".json")
    record.write_text(json.dumps(row,indent=2)+"\n")
    if timed["returncode"]:
        raise RuntimeError(name+f": generation exited {timed['returncode']}: "+
                           Path(timed["log"]).read_text(errors="replace")[-3500:])
    text=Path(timed["log"]).read_text(errors="replace")
    if is_stock:
        if "LTX25_CUBLAS_E" in text or "LTX25_SELECTIVE_FP16_GEMM" in text:
            raise RuntimeError(name+": historical source unexpectedly ran optimization")
    else:
        e=read_candidate_stats(text)
        if not e or e.get("mode")!="fp32rot" or e.get("rotation_mode")!="radix4" or e.get("used",0)<=0:
            raise RuntimeError(name+": Candidate E radix4 did not execute")
        row["candidate_e"]={k:e.get(k) for k in (
            "used","fallback","range_fallback","rotation_mode","mode","cast_mode",
            "rotation_input","output_mode")}
    if arm in ("stage2_fp16","both_fp16"):
        fp16=OUT/"results"/(name+"-fp16.json")
        if not fp16.is_file():
            raise RuntimeError(name+": selector failed to produce FP16 stats")
        row["fp16"]=verify_fp16(fp16,arm)
    elif "LTX25_SELECTIVE_FP16_GEMM" in text:
        raise RuntimeError(name+": disabled FP16 selector unexpectedly ran")
    clips=[p for p in target.rglob("*.mp4") if p.stat().st_size>1024]
    if len(clips)!=1:
        raise RuntimeError(name+f": expected 1 MP4, found {len(clips)}")
    dst=OUT/"clips"/(name+".mp4")
    shutil.copy2(clips[0],dst)
    row.update(probe_video(dst,frames))
    row["clip"]=str(dst)
    row["clip_bytes"]=dst.stat().st_size
    record.write_text(json.dumps(row,indent=2)+"\n")
    print("T2V_4WAY_PASS",json.dumps({
        "arm":arm,"frames":frames,"wall_seconds":row["wall_seconds"],
        "encoded":[row["width"],row["height"]],
        "candidate_e_used":row.get("candidate_e",{}).get("used"),
        "fp16":row.get("fp16",{}).get("converted_per_stage")
    }),flush=True)
    return row


def finalize(frames,rows):
    if len(rows)!=4 or [x["arm"] for x in rows]!=list(ARMS):
        raise RuntimeError("Missing/reordered execution arms")
    if len({x["settings_sha256"] for x in rows})!=1:
        raise RuntimeError("Comparison input settings changed")
    stock=rows[0]
    summary={"status":"MEASURED_NOT_QUALITY_APPROVED",
             "frames":frames,"resolution_requested":RESOLUTION,
             "resolution_actual":f"{stock['width']}x{stock['height']}",
             "steps":STEPS,"seed":SEED,
             "source_revision_stock":ORIGINAL_SHA,
             "timing":"time.monotonic() spanning fresh complete wgp.py subprocess only",
             "prompt":PROMPT,
             "results":[]}
    for r in rows:
        ps=base.psnr(Path(stock["clip"]),Path(r["clip"]),base.ffmpeg_bin())
        comparison={
            "arm":r["arm"],"wall_seconds":r["wall_seconds"],
            "saved_vs_stock_seconds":round(stock["wall_seconds"]-r["wall_seconds"],3),
            "wall_reduction_vs_stock_percent":round(100*(stock["wall_seconds"]-r["wall_seconds"])/stock["wall_seconds"],3),
            "speedup_vs_stock":round(stock["wall_seconds"]/r["wall_seconds"],4),
            "video_psnr_db":ps.get("average_db"),
            "psnr_status":ps.get("status"),
            "audio_pcm_equal":r["decoded_audio_sha256"]==stock["decoded_audio_sha256"],
            "video_frames_exact":r["decoded_video_sha256"]==stock["decoded_video_sha256"],
            "clip_name":Path(r["clip"]).name,
            "fp16_converted":r.get("fp16",{}).get("converted_per_stage",{}),
            "fp16_guard_fallbacks":r.get("fp16",{}).get("guard_fallbacks",0),
        }
        summary["results"].append(comparison)
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    with (OUT/"summary.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=[
            "arm","wall_seconds","saved_vs_stock_seconds",
            "wall_reduction_vs_stock_percent","speedup_vs_stock",
            "video_psnr_db","psnr_status","audio_pcm_equal",
            "video_frames_exact","clip_name","fp16_converted","fp16_guard_fallbacks"
        ],extrasaction="ignore")
        writer.writeheader()
        for x in summary["results"]: writer.writerow(x)
    print("T2V_4WAY_SUMMARY",json.dumps(summary,sort_keys=True),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--frames",required=True,type=int,choices=FRAME_CHOICES)
    args=parser.parse_args()
    frames=args.frames
    global OUT
    OUT=Path(os.getenv("LTX25_T2V_4WAY_OUT",
                       f"/cache/ltx25/results/t2v4way-{frames}-{os.getenv('GITHUB_RUN_ID','manual')}"))
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("Output directory already contains files: "+str(OUT))
    for d in ("results","logs","clips","config","videos"):
        (OUT/d).mkdir(parents=True,exist_ok=True)
    base.OUT=OUT
    base.EXTENSION=OUT/"libltx25_cublas_e_sm70.so"
    base.FRAMES=frames
    base.RESOLUTION=RESOLUTION
    base.STEPS=STEPS
    base.SEED=SEED
    base.PROMPT=PROMPT
    base.gpu_free()
    provenance=verify_source()
    checkpoint=base.CHECKPOINTS/"ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors"
    if not checkpoint.is_file():
        raise RuntimeError("Baked model missing: "+str(checkpoint))
    build=base.command_run(
        ["nvcc","-O3","-std=c++17","-arch=sm_70","-shared","-Xcompiler","-fPIC",
         "benchmarks/ltx25_convrot/ltx25_cublas_e.cu",
         "-lcublas","-o",str(base.EXTENSION)],
        timeout=300,log=OUT/"logs"/"compile-candidate-e.log")
    if not base.EXTENSION.is_file():
        raise RuntimeError("Candidate E library compilation produced no output")
    inp=OUT/"input.json"
    inp.write_text(json.dumps(settings(frames),indent=2)+"\n")
    template=base.initialize_config()
    manifest={
        **provenance,"optimized_source":os.getenv("GITHUB_SHA","manual"),
        "checkpoint":str(checkpoint),"checkpoint_bytes":checkpoint.stat().st_size,
        "candidate_e_lib_sha256":sha(base.EXTENSION),
        "settings_sha256":sha(inp),
        "requested_resolution":RESOLUTION,
        "frames":frames,"steps":STEPS,"seed":SEED,
        "prompt":PROMPT, "model":"ltx2_25_22B_distilled",
        "video_type":"T2V — no image conditioning",
        "attention":"sdpa","mmgp_profile":0,"use_lora":False,
        "order":list(ARMS),
        "candidate_e_workspace":True,
        "timed_window":"wgp.py process creation to termination; no prep/quality/postprocess",
        "cuda_device":subprocess.run(["nvidia-smi","-L"],capture_output=True,text=True).stdout.strip(),
    }
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    rows=[]
    for index,arm in enumerate(ARMS,1):
        rows.append(run_case(frames,index,arm,template))
        (OUT/"partial-results.json").write_text(json.dumps(rows,indent=2)+"\n")
    finalize(frames,rows)
    print("LTX25_T2V_4WAY_ALL_PASS",frames,str(OUT),flush=True)


if __name__=="__main__":
    main()
