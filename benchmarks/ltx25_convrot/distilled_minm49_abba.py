#!/usr/bin/env python3
"""Distilled ConvRot minM 64 vs 16: paired 49f A-B-B-A full-process videos.

Original distilled checkpoint, 1024x576, seed 20261112, 8 steps, stock
WanGP Conv VAE, current Stage2 selective FP16, Stage1 native. Only the
Candidate E minimum M threshold differs across the A/B arms.
"""
from __future__ import annotations
import json,os,statistics
from pathlib import Path

import t2v73_241_original_stock_fourway as four

OUT=Path(os.environ.get("LTX25_DISTILLED_MINM_OUT",
                        "/cache/ltx25/results/distilled-minm-abba-manual"))
FRAMES=49
CASES=(("A1",64),("B1",16),("B2",16),("A2",64))
CHECKPOINT=Path("/opt/ltx25/checkpoints/ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors")
def require(cond,msg):
    if not cond:raise RuntimeError(msg)


def main():
    require(CHECKPOINT.is_file() and CHECKPOINT.stat().st_size>1_000_000_000,"Missing checkpoint")
    require(not OUT.exists(),"Output must not already exist")
    for part in ("config","videos","logs","results","clips"):
        (OUT/part).mkdir(parents=True,exist_ok=False)
    four.OUT=OUT;four.base.OUT=OUT
    four.base.EXTENSION=OUT/"libltx25_cublas_e_sm70.so"
    four.base.FRAMES=FRAMES
    four.base.RESOLUTION="1024x576"
    four.base.STEPS=8
    four.base.SEED=20261112
    four.base.PROMPT=four.PROMPT
    settings=four.settings(FRAMES)
    (OUT/"input.json").write_text(json.dumps(settings,indent=2)+"\n")
    four.base.gpu_free()
    four.base.command_run(["nvcc","-O3","-std=c++17","-arch=sm_70","-shared",
        "-Xcompiler","-fPIC","benchmarks/ltx25_convrot/ltx25_cublas_e.cu",
        "-lcublas","-o",str(four.base.EXTENSION)],timeout=300,
        log=OUT/"logs/compile-candidate-e.log")
    require(four.base.EXTENSION.is_file(),"Candidate E compilation missing")
    original_env=four.optimized_env
    current_min=64

    def env_for(arm,name):
        env=original_env(arm,name)
        env.update({
            "WAN2GP_LTX25_E_MIN_M":str(current_min),
            "WAN2GP_LTX25_E_MAX_M":"8192",
            "WAN2GP_LTX25_E_ROW_CENSUS":"1",
            "WAN2GP_LTX25_E_TEST_4096X2048":"0",
        })
        return env

    four.optimized_env=env_for
    template=four.base.initialize_config()
    (OUT/"manifest.json").write_text(json.dumps({
        "model":str(CHECKPOINT),"lora":"none","vae":"standard Conv VAE",
        "prompt":four.PROMPT,"seed":20261112,"frames":FRAMES,"steps":8,
        "resolution":"1024x576","stage2_fp16":True,"stage1_fp16":False,
        "order":list(CASES),"max_m":8192,"tested_min_m":[64,16],
        "unapproved_shape_4096x2048":False,
        "Candidate_E":"FP32-rot radix4, vector8, transient FP16 cuBLAS",
        "timing":"fresh wgp.py process including loading and saving",
        "source_sha":os.environ.get("GITHUB_SHA","manual"),
    },indent=2)+"\n")

    rows=[]
    for i,(label,min_m) in enumerate(CASES,1):
        current_min=min_m
        four.base.gpu_free()
        print("DISTILLED_MINM_BEGIN",label,min_m,flush=True)
        row=four.run_case(FRAMES,i,"stage2_fp16",template)
        log=Path(row["log"]).read_text(errors="replace")
        require(CHECKPOINT.name in log,"Wrong checkpoint")
        require("Loading NAD Diffusion Decoder" not in log,"Wrong VAE")
        stats=four.read_candidate_stats(log)
        require(stats and stats.get("min_m")==min_m and stats.get("max_m")==8192,
                "Incorrect effective E thresholds")
        require(not stats.get("experimental_failed_shape_enabled"),
                "Unsafe shape enabled")
        census=stats.get("row_census",{})
        require(stats.get("row_census_enabled") and any(k.startswith("M=51,") for k in census),
                "Real 49f M51 workload not observed")
        fp=row["fp16"]["converted_per_stage"]
        require(fp["stage1"]==0 and fp["stage2"]>0,"Stage2-only selection failed")
        row.update(label=label,minimum_m=min_m,candidate_e_row_census=census)
        rows.append(row)
        (OUT/"partial-results.json").write_text(json.dumps(rows,indent=2)+"\n")
        print("DISTILLED_MINM_CASE_PASS",json.dumps({
            "label":label,"min_m":min_m,"seconds":row["wall_seconds"],
            "E_used":row["candidate_e"]["used"],"FP16":fp,
            "M51":{k:v for k,v in census.items() if k.startswith("M=51,")}
        },sort_keys=True),flush=True)
    a=[r["wall_seconds"] for r in rows if r["minimum_m"]==64]
    b=[r["wall_seconds"] for r in rows if r["minimum_m"]==16]
    require(len(a)==2 and len(b)==2,"Invalid A/B pairing")
    median_a,median_b=statistics.median(a),statistics.median(b)
    ffmpeg=four.base.ffmpeg_bin()
    baseline=Path(rows[0]["clip"])
    comparisons=[]
    for r in rows:
        psnr=four.base.psnr(baseline,Path(r["clip"]),ffmpeg)
        comparisons.append({
            "label":r["label"],"minimum_m":r["minimum_m"],
            "wall_seconds":r["wall_seconds"],
            "e_calls":r["candidate_e"]["used"],
            "stage2_converted":r["fp16"]["converted_per_stage"]["stage2"],
            "vs_A1_video_psnr_db":psnr.get("average_db"),
            "vs_A1_psnr_status":psnr.get("status"),
            "video_same_as_A1":r["decoded_video_sha256"]==rows[0]["decoded_video_sha256"],
            "audio_same_as_A1":r["decoded_audio_sha256"]==rows[0]["decoded_audio_sha256"],
            "clip":r["clip"]
        })
    report={
        "status":"MEASURED_NOT_QUALITY_APPROVED",
        "median_min64_s":round(median_a,3),
        "median_min16_s":round(median_b,3),
        "delta_min16_minus_min64_s":round(median_b-median_a,3),
        "speedup_percent":round(100*(median_a-median_b)/median_a,2),
        "rows":comparisons,
        "limitations":"One ABBA, visual review required; output arithmetic intentionally differs",
    }
    (OUT/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print("LTX25_DISTILLED_MINM_ABBA_PASS",json.dumps(report,sort_keys=True),flush=True)


if __name__=="__main__":main()
