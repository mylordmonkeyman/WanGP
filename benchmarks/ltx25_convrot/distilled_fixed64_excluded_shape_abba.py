#!/usr/bin/env python3
"""Isolated Distilled A-B-B-A for opt-in N=4096,K=2048 at fixed MIN_M=64.

Four independent 49-frame 1024x576/8-step/seed=20261112 runs. Only
WAN2GP_LTX25_E_TEST_4096X2048 is varied. The experimental family was
previously rejected by a strict numerical gate, so the script measures
stock-relative PSNR, SSIM and decoded PCM audio, not quality approval.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import numpy as np
import t2v73_241_original_stock_fourway as four
import distilled_stock49_reference_psnr as stockmetrics

OUT = Path(os.environ.get("LTX25_DISTILLED_FIXED64_OUT",
                          "/cache/ltx25/results/distilled-fixed64-shape-manual"))
STOCK_DIR=Path("/cache/ltx25/results/distilled-stock49-43681488566-attempt1")
STOCK_CLIP=STOCK_DIR/"clips/t2v49-01-stock_upstream.mp4"
STOCK_INPUT=STOCK_DIR/"input.json"
CHECKPOINT=Path("/opt/ltx25/checkpoints/ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors")
CASES=(("A1_off",False),("B1_on",True),("B2_on",True),("A2_off",False))
FRAMES=49


def require(cond,msg):
    if not cond:
        raise RuntimeError(msg)


def excluded_census(census):
    rows={}
    for signature,counts in census.items():
        if signature.endswith(",N=4096,K=2048"):
            rows[signature]={"used":int(counts.get("used",0)),
                             "fallback":int(counts.get("fallback",0))}
    return rows


def main():
    require(CHECKPOINT.is_file() and CHECKPOINT.stat().st_size>1_000_000_000,
            "Distilled checkpoint absent")
    require(STOCK_CLIP.is_file() and STOCK_CLIP.stat().st_size>100_000,
            "Pinned true stock WanGP reference not found")
    require(STOCK_INPUT.is_file(),"Matched stock input JSON unavailable")
    require(not OUT.exists(),"Unique result directory already exists")
    for d in ("config","videos","logs","results","clips"):
        (OUT/d).mkdir(parents=True,exist_ok=False)

    four.OUT=OUT
    four.base.OUT=OUT
    four.base.FRAMES=FRAMES
    four.base.RESOLUTION="1024x576"
    four.base.STEPS=8
    four.base.SEED=20261112
    four.base.PROMPT=four.PROMPT
    four.base.EXTENSION=OUT/"libltx25_cublas_e_sm70.so"
    input_data=four.settings(FRAMES)
    # Verify the entire input structure against the unmodified, stock run:
    # prompt, solver, CFG, seed, model identifier, LoRAs and all other keys.
    stock_data=json.loads(STOCK_INPUT.read_text())
    require(input_data==stock_data,
            f"Test settings are not identical to true stock input: {sorted(set(input_data)^set(stock_data))}")
    (OUT/"input.json").write_text(json.dumps(input_data,indent=2)+"\n")
    ffmpeg=four.base.ffmpeg_bin()
    stock_probe=four.probe_video(STOCK_CLIP,FRAMES)
    stock_pcm=stockmetrics.audio_pcm(STOCK_CLIP,ffmpeg)
    stock_float=np.frombuffer(stock_pcm,dtype=np.int16).astype(np.float64)/32768
    require(len(stock_float)>0,"No stock audio PCM")

    four.base.gpu_free()
    four.base.command_run([
        "nvcc","-O3","-std=c++17","-arch=sm_70","-shared","-Xcompiler","-fPIC",
        "benchmarks/ltx25_convrot/ltx25_cublas_e.cu","-lcublas",
        "-o",str(four.base.EXTENSION)
    ],timeout=300,log=OUT/"logs/compile-candidate-e.log")
    require(four.base.EXTENSION.is_file(),"Cannot compile Candidate E library")
    old_env=four.optimized_env
    active_enable=False
    def env_for(arm,name):
        env=old_env(arm,name)
        env.update({
            "WAN2GP_LTX25_E_MIN_M":"64",
            "WAN2GP_LTX25_E_MAX_M":"8192",
            "WAN2GP_LTX25_E_ROW_CENSUS":"1",
            "WAN2GP_LTX25_E_TEST_4096X2048":"1" if active_enable else "0",
        })
        return env
    four.optimized_env=env_for
    template=four.base.initialize_config()
    manifest={
        "kind":"Distilled + Candidate E isolated experimental weight family A/B/B/A",
        "model_checkpoint":str(CHECKPOINT),
        "stock_source":"deepbeepmeep/WanGP c8d47da27023476777a0874300bbc9a4126acca6",
        "stock_video":str(STOCK_CLIP),
        "stock_decoded_video_sha256":stock_probe["decoded_video_sha256"],
        "stock_decoded_audio_sha256":stock_probe["decoded_audio_sha256"],
        "settings_sha256":hashlib.sha256((OUT/"input.json").read_bytes()).hexdigest(),
        "seed":20261112,"steps":8,"frames":FRAMES,"resolution":"1024x576",
        "min_m":64,"max_m":8192,"candidate_e_radix":"radix4",
        "stage1_fp16":False,"stage2_fp16":True,
        "only_changed_variable":"WAN2GP_LTX25_E_TEST_4096X2048",
        "arms":list(CASES),"prompt":four.PROMPT,
        "source_sha":os.getenv("GITHUB_SHA","manual"),
        "warning":"Closer to stock is not a ground truth assessment of visual quality."
    }
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    report=[]
    for index,(label,enabled) in enumerate(CASES,1):
        active_enable=enabled
        four.base.gpu_free()
        print("FIXED64_SHAPE_BEGIN",label,int(enabled),flush=True)
        row=four.run_case(FRAMES,index,"stage2_fp16",template)
        log=Path(row["log"]).read_text(errors="replace")
        require(CHECKPOINT.name in log and "Loading NAD Diffusion Decoder" not in log,
                "Wrong model or decoder")
        require(row["settings_sha256"]==manifest["settings_sha256"],
                "Run settings differ from reference")
        e=four.read_candidate_stats(log)
        require(e is not None and e.get("experimental_failed_shape_enabled")==enabled,
                f"Experimental family flag mismatch: {label}")
        require(e.get("min_m")==64 and e.get("max_m")==8192 and e.get("row_census_enabled") is True,
                f"Row thresholds or census wrong: {label}")
        family=excluded_census(e["row_census"])
        require(family,"Model contains no observed excluded family GEMMs")
        family_used=sum(x["used"] for x in family.values())
        family_fallback=sum(x["fallback"] for x in family.values())
        # M<64 must always fall back even when the family is allowed;
        # the >=64 eligible rows must switch from fallback to Candidate E.
        if enabled:
            require(family_used>0 and family_fallback>0,
                    f"B: large shapes not switched or small-M override leaked: {family}")
        else:
            require(family_used==0 and family_fallback>0,
                    f"A: excluded shape was used despite opt-out: {family}")
        require(row["fp16"]["converted_per_stage"]["stage1"]==0
                and row["fp16"]["converted_per_stage"]["stage2"]>0,
                f"Stage2 FP16 isolation lost on {label}")
        require(row["fp16"]["persistent_fp16_weight_copies"]==0,
                "Persistent FP16 weight copies forbidden")
        require(row["fp16"]["guard_fallbacks"]==0,
                "FP16 safety guard unexpectedly fell back")

        clip=Path(row["clip"])
        video_psnr=stockmetrics.metric(ffmpeg,STOCK_CLIP,clip,"psnr")
        video_ssim=stockmetrics.metric(ffmpeg,STOCK_CLIP,clip,"ssim")
        pcm=stockmetrics.audio_pcm(clip,ffmpeg)
        same_length=len(pcm)==len(stock_pcm)
        if same_length:
            x=np.frombuffer(pcm,dtype=np.int16).astype(np.float64)/32768
            mse=float(np.mean((x-stock_float)**2))
            sig=float(np.mean(stock_float**2))
            audio_snr=math.inf if mse==0 else round(10*math.log10(max(sig,1e-20)/mse),6)
        else:
            audio_snr=None
        item={
            "case":label,"enabled_4096x2048":enabled,
            "min_m":64,"max_m":8192,
            "wall_seconds":row["wall_seconds"],
            "candidate_e_total_calls":row["candidate_e"]["used"],
            "candidate_e_excluded_family_uses":family_used,
            "excluded_family_fallbacks":family_fallback,
            "excluded_family_by_m":family,
            "stage1_fp16_converted":row["fp16"]["converted_per_stage"]["stage1"],
            "stage2_fp16_converted":row["fp16"]["converted_per_stage"]["stage2"],
            "psnr_db_vs_true_stock":video_psnr,
            "ssim_vs_true_stock":video_ssim,
            "audio_snr_db_vs_true_stock":audio_snr,
            "audio_same_as_stock":pcm==stock_pcm,
            "decoded_audio_sample_count_match":same_length,
            "video_exact_vs_stock":row["decoded_video_sha256"]==stock_probe["decoded_video_sha256"],
            "video_decoded_sha256":row["decoded_video_sha256"],
            "audio_decoded_sha256":row["decoded_audio_sha256"],
            "clip":str(clip),
        }
        report.append(item)
        (OUT/"partial-results.json").write_text(json.dumps(report,indent=2)+"\n")
        print("FIXED64_SHAPE_CASE_PASS",json.dumps({
            k:item[k] for k in (
                "case","enabled_4096x2048","wall_seconds","candidate_e_total_calls",
                "candidate_e_excluded_family_uses","excluded_family_fallbacks",
                "stage2_fp16_converted","psnr_db_vs_true_stock",
                "ssim_vs_true_stock","audio_snr_db_vs_true_stock")
        },sort_keys=True),flush=True)

    for enabled in (False,True):
        pair=[x for x in report if x["enabled_4096x2048"]==enabled]
        require(len(pair)==2,"Missing duplicate arm")
        require(len({x["video_decoded_sha256"] for x in pair})==1,
                "Identical configuration did not reproduce decoded video")
        require(len({x["audio_decoded_sha256"] for x in pair})==1,
                "Identical configuration did not reproduce decoded audio")
    baseline=[x for x in report if not x["enabled_4096x2048"]]
    trial=[x for x in report if x["enabled_4096x2048"]]
    before=statistics.median(x["wall_seconds"] for x in baseline)
    after=statistics.median(x["wall_seconds"] for x in trial)
    summary={
        "status":"MEASURED_NOT_QUALITY_APPROVED",
        "on_off_only_change":"WAN2GP_LTX25_E_TEST_4096X2048",
        "min_m":64,"max_m":8192,
        "off_median_seconds":round(before,3),
        "on_median_seconds":round(after,3),
        "on_minus_off_seconds":round(after-before,3),
        "on_speedup_percent":round((before-after)*100/before,3),
        "off_vs_stock_psnr_db":baseline[0]["psnr_db_vs_true_stock"],
        "on_vs_stock_psnr_db":trial[0]["psnr_db_vs_true_stock"],
        "psnr_delta_on_minus_off_db":round(
            trial[0]["psnr_db_vs_true_stock"]-baseline[0]["psnr_db_vs_true_stock"],6),
        "off_vs_stock_ssim":baseline[0]["ssim_vs_true_stock"],
        "on_vs_stock_ssim":trial[0]["ssim_vs_true_stock"],
        "ssim_delta_on_minus_off":round(
            trial[0]["ssim_vs_true_stock"]-baseline[0]["ssim_vs_true_stock"],6),
        "off_audio_snr_db":baseline[0]["audio_snr_db_vs_true_stock"],
        "on_audio_snr_db":trial[0]["audio_snr_db_vs_true_stock"],
        "runs":report,
        "warning":"Historical per-layer p99/max precision gate failed 15/15; higher PSNR is only closer to stock, not perceptual quality."
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    with (OUT/"summary.csv").open("w",newline="") as f:
        fields=[k for k in report[0] if k!="excluded_family_by_m"]
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader();writer.writerows(report)
    print("LTX25_DISTILLED_FIXED64_SHAPE_ABBA_PASS",json.dumps({
        k:v for k,v in summary.items() if k!="runs"
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
