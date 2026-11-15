#!/usr/bin/env python3
"""73f Distilled T2V and I2V: stock vs three experimental Candidate E shapes.

Arms for the historically REJECTED INT8 weight shape (N=4096,K=2048):
  stock: genuine unmodified upstream WanGP (no Candidate E)
  off: approved 7-family Candidate E only (excluded family OFF both stages)
  stage2: excluded family OFF stage1, ON stage2
  both: excluded family ON stage1 and stage2
All three optimized arms retain the same M=64..8192 cutoff, Candidate E
+ Radix-4 and Stage2 *selective* FP16 on / Stage1 selective FP16 off.
No extra persistent FP16 weight set. Identical image (for I2V), prompt,
seed, eight inference steps, MMGP profile 0, decoder and audio settings.
Numbers are measured relative to a true original stock run *per mode*;
higher PSNR or SSIM does not imply better aesthetic quality.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics

import numpy as np
import t2v73_241_original_stock_fourway as four
import distilled_stock49_reference_psnr as stockmetrics
from profile_gemm_t2v_i2v import INPUT_IMAGE, PROMPT, check_png

ROOT=Path(__file__).resolve().parents[2]
FRAMES=73
STEPS=8
SEED=20261112
RESOLUTION="1024x576"
CHECKPOINT=Path("/opt/ltx25/checkpoints/ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors")
CASES=(("stock","stock_upstream",None),
       ("off","stage2_fp16","none"),
       ("stage2","stage2_fp16","stage2"),
       ("both","stage2_fp16","stage1,stage2"))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def hash_file(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def audit_stage2(path, arm):
    # The novel test arms can correctly leave zero Quanto GEMMs in Stage 2:
    # Candidate E has consumed every target operation. Do NOT conflate
    # that result with "selective FP16 wasn't enabled"; verify the reports.
    data=json.loads(path.read_text())
    raw=data.get("stats",{})
    require(data.get("backend")=="direct" and data.get("sham") is False,
            "Wrong Stage 2 FP16 interception backend")
    require(data.get("persistent_fp16_weight_copies")==0,
            "Second persistent FP16 weight set unexpectedly created")
    require(int(raw.get("guard_input_reference",0))==0 and
            int(raw.get("guard_output_reference",0))==0,
            "Stage 2 FP16 had numerical guard fallbacks")
    by_shape=data.get("by_shape",{})
    per_phase={
        phase:sum(int(v.get("converted",0)) for key,v in by_shape.items()
                  if key.startswith(phase+":"))
        for phase in ("stage1","stage2")
    }
    require(per_phase["stage1"]==0,"Accidentally enabled Stage 1 selective FP16")
    forwards=data.get("transformer_forward_calls",{})
    require(int(forwards.get("stage1",0))>0 and int(forwards.get("stage2",0))>0,
            "Stage-boundary forward calls were not captured")
    require(int(raw.get("bypassed_unselected_stage1_forwards",0))>0,
            "Stage 1 must skip the Stage 2-only selective FP16 interception")
    return {
        "converted_per_stage":per_phase,
        "converted_qbytes_mm":int(raw.get("converted_qbytes_mm",0)),
        "observed_qbytes_mm":int(raw.get("observed_qbytes_mm",0)),
        "forward_counts":forwards,
        "guard_fallbacks":0,
        "persistent_fp16_weight_copies":0,
    }


def audio_stats(path, reference_pcm, reference_float, ffmpeg):
    pcm=stockmetrics.audio_pcm(path,ffmpeg)
    if len(pcm)!=len(reference_pcm):
        return {"samples_match":False,"same_as_stock":False,"audio_snr_db":None}
    ref=np.frombuffer(pcm,dtype=np.int16).astype(np.float64)/32768
    mse=float(np.mean((reference_float-ref)**2))
    power=float(np.mean(reference_float**2))
    snr=math.inf if mse==0 else round(10*math.log10(max(power,1e-20)/mse),6)
    return {"samples_match":True,"same_as_stock":pcm==reference_pcm,
            "audio_snr_db":snr}


def run_mode(mode):
    require(mode in ("i2v","t2v"),"Unsupported mode")
    out=Path(os.environ.get("LTX25_BAR73_OUT",
        f"/cache/ltx25/results/bar73-stage-shape-{mode}-{os.environ.get('GITHUB_RUN_ID','manual')}"
    )).resolve()
    require(not out.exists(),f"Result directory already in use: {out}")
    require(CHECKPOINT.is_file() and CHECKPOINT.stat().st_size>1_000_000_000,
            "Distilled INT8 ConvRot checkpoint missing")
    require(INPUT_IMAGE.is_file(),"Original uploaded bar image missing")
    check_png()
    for dirname in ("results","logs","clips","config","videos"):
        (out/dirname).mkdir(parents=True,exist_ok=False)

    four.OUT=out
    four.base.OUT=out
    four.base.EXTENSION=out/"libltx25_cublas_e_sm70.so"
    four.base.FRAMES=FRAMES
    four.base.RESOLUTION=RESOLUTION
    four.base.STEPS=STEPS
    four.base.SEED=SEED
    four.base.PROMPT=PROMPT
    # Both modes share identical original user bar text/audio prompt.
    input_data=four.settings(FRAMES)
    if mode=="i2v":
        input_data["image_start"]=str(INPUT_IMAGE)
        input_data["image_prompt_type"]="S"
    else:
        require("image_start" not in input_data
                and "image_prompt_type" not in input_data,
                "T2V must not have image conditioning")
    (out/"input.json").write_text(json.dumps(input_data,indent=2)+"\n")
    setting_hash=hash_file(out/"input.json")
    provenance=four.verify_source()
    ffmpeg=four.base.ffmpeg_bin()
    four.base.gpu_free()
    four.base.command_run([
        "nvcc","-O3","-std=c++17","-arch=sm_70","-shared",
        "-Xcompiler","-fPIC","benchmarks/ltx25_convrot/ltx25_cublas_e.cu",
        "-lcublas","-o",str(four.base.EXTENSION)
    ],timeout=300,log=out/"logs/candidate-e-compile.log")
    require(four.base.EXTENSION.is_file(),"Candidate E source extension not compiled")

    old_env=four.optimized_env
    old_check=four.verify_fp16
    phase_mode="none"
    def shaped_env(arm,name):
        env=old_env(arm,name)
        env.update({
            "WAN2GP_LTX25_E_MIN_M":"64",
            "WAN2GP_LTX25_E_MAX_M":"8192",
            "WAN2GP_LTX25_E_ROW_CENSUS":"1",
            "WAN2GP_LTX25_E_TEST_4096X2048":"0",
            "WAN2GP_LTX25_E_TEST_4096X2048_PHASES":phase_mode,
        })
        return env
    def checked_fp16(path,arm):
        if phase_mode=="none":
            return old_check(path,arm)
        return audit_stage2(path,arm)
    four.optimized_env=shaped_env
    four.verify_fp16=checked_fp16
    template=four.base.initialize_config()

    manifest={
        "status":"RUNNING","mode":mode,
        "question":"Candidate E N4096,K2048 stage1+stage2 vs stage2-only vs off, against actual upstream stock",
        "model":"LTX-2.5 22B Distilled INT8 ConvRot",
        "checkpoint":str(CHECKPOINT),"stock_upstream":provenance,
        "prompt":PROMPT,"seed":SEED,"frames":FRAMES,"steps":STEPS,
        "resolution":RESOLUTION,"audio_guidance":1.5,"NAG_scale":1.1,
        "input_image":str(INPUT_IMAGE) if mode=="i2v" else None,
        "input_image_sha256":hash_file(INPUT_IMAGE),
        "image_prompt_type":"S" if mode=="i2v" else None,
        "settings_sha256":setting_hash,
        "min_m":64,"max_m":8192,"Stage1_selective_FP16":False,
        "Stage2_selective_FP16":True,
        "Candidate_E":"fp32rot/radix4, vector8, fused",
        "order":list(CASES),
        "quality_warning":"Stock-reference similarity is not subjective/perceptual quality",
        "source_sha":os.getenv("GITHUB_SHA","manual"),
        "no_weight_conversion_during_run":True,
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    rows=[]
    reference=None
    audio_reference=None
    audio_float=None
    for index,(label,arm,selected) in enumerate(CASES,1):
        phase_mode=selected or "none"
        four.base.gpu_free()
        print("BAR73_SHAPE_BEGIN",json.dumps({"mode":mode,"arm":label,"phases":phase_mode}),flush=True)
        row=four.run_case(FRAMES,index,arm,template)
        require(row["settings_sha256"]==setting_hash,
                "Input settings changed between arms")
        log=Path(row["log"]).read_text(errors="replace")
        require(CHECKPOINT.name in log and "Loading NAD Diffusion Decoder" not in log,
                "Incorrect checkpoint or VAE")
        require(row["frames"]==FRAMES and row["width"]==1024 and row["height"]==576,
                "Unexpected video size or length")
        if label=="stock":
            require(row["source"]=="historical-upstream","Stock was not genuine historical upstream")
            reference=row
            audio_reference=stockmetrics.audio_pcm(Path(row["clip"]),ffmpeg)
            audio_float=np.frombuffer(audio_reference,dtype=np.int16).astype(np.float64)/32768
            metadata=None
            stage_usage=None
        else:
            require(row["source"]=="optimized","Optimized run did not use custom fork")
            metadata=four.read_candidate_stats(log)
            require(metadata is not None,"Candidate E report missing")
            require(metadata.get("experimental_failed_shape_phases")==
                    ([] if label=="off" else ["stage2"] if label=="stage2" else ["stage1","stage2"]),
                    f"Wrong Candidate E staged opt-in: {metadata.get('experimental_failed_shape_phases')}")
            require(metadata.get("min_m")==64 and metadata.get("max_m")==8192,
                    "Candidate E threshold unexpectedly changed")
            require(metadata.get("mode")=="fp32rot"
                    and metadata.get("rotation_mode")=="radix4"
                    and metadata.get("row_census_enabled") is True,
                    "Candidate E kernel configuration or row census incorrect")
            stage_usage=metadata.get("experimental_shape_by_phase")
            require(isinstance(stage_usage,dict) and set(stage_usage)=={"stage1","stage2"},
                    "Missing independent Stage1/Stage2 target shape counters")
            stage1=stage_usage["stage1"]
            stage2=stage_usage["stage2"]
            require(int(stage1["used"])+int(stage1["fallback"])>0
                    and int(stage2["used"])+int(stage2["fallback"])>0,
                    "Target family absent in either stage — experiment invalid")
            if label=="off":
                require(int(stage1["used"])==0 and int(stage2["used"])==0,
                        "Off arm still routed target shape to Candidate E")
            elif label=="stage2":
                require(int(stage1["used"])==0 and int(stage2["used"])>0,
                        "Stage2-only arm incorrectly changed Stage1 or failed to select Stage2")
            elif label=="both":
                require(int(stage1["used"])>0 and int(stage2["used"])>0,
                        "Both-stage arm failed to exercise both stages")
            require(row["fp16"]["converted_per_stage"]["stage1"]==0,
                    "Stage1 selective FP16 unintentionally enabled")
            require(row["fp16"]["guard_fallbacks"]==0
                    and row["fp16"]["persistent_fp16_weight_copies"]==0,
                    "FP16 fallback or persistent FP16 weight copy found")
        clip=Path(row["clip"])
        if label=="stock":
            psnr=math.inf
            ssim=1.0
            audio={"samples_match":True,"same_as_stock":True,"audio_snr_db":math.inf}
        else:
            psnr=stockmetrics.metric(ffmpeg,Path(reference["clip"]),clip,"psnr")
            ssim=stockmetrics.metric(ffmpeg,Path(reference["clip"]),clip,"ssim")
            audio=audio_stats(clip,audio_reference,audio_float,ffmpeg)
            require(audio["samples_match"],"Reference and candidate audio lengths differ")
        item={
            "mode":mode,"arm":label,"phases":phase_mode,
            "wall_seconds":row["wall_seconds"],
            "E_total_used":metadata["used"] if metadata is not None else None,
            "excluded_shape_by_phase":stage_usage,
            "stage2_selective_fp16_converted":(
                row.get("fp16",{}).get("converted_per_stage",{}).get("stage2")),
            "PSNR_db_vs_stock":psnr,
            "SSIM_vs_stock":ssim,
            "audio_SNR_db_vs_stock":audio["audio_snr_db"],
            "audio_pcm_identical_to_stock":audio["same_as_stock"],
            "video_decoded_sha256":row["decoded_video_sha256"],
            "audio_decoded_sha256":row["decoded_audio_sha256"],
            "mp4":str(clip),
        }
        rows.append(item)
        (out/"partial-results.json").write_text(json.dumps(rows,indent=2,allow_nan=True)+"\n")
        print("BAR73_SHAPE_CASE_PASS",json.dumps({
            k:item[k] for k in (
                "mode","arm","phases","wall_seconds","E_total_used",
                "excluded_shape_by_phase","stage2_selective_fp16_converted",
                "PSNR_db_vs_stock","SSIM_vs_stock","audio_SNR_db_vs_stock")
        },sort_keys=True),flush=True)
    require(len(rows)==4,"Incomplete video comparison")
    off=rows[1]
    results={
        "status":"MEASURED_NOT_PERCEPTUALLY_APPROVED",
        "mode":mode,"frames":FRAMES,"settings_sha256":setting_hash,
        "reference_source":"deepbeepmeep/WanGP c8d47da27023476777a0874300bbc9a4126acca6",
        "stock_wall_seconds":rows[0]["wall_seconds"],
        "optimized_arms":[
            {"arm":r["arm"],"seconds":r["wall_seconds"],
             "seconds_vs_off":round(r["wall_seconds"]-off["wall_seconds"],3),
             "speedup_pct_vs_off":round(100*(off["wall_seconds"]-r["wall_seconds"])/off["wall_seconds"],3),
             "psnr_db_vs_stock":r["PSNR_db_vs_stock"],
             "psnr_delta_vs_off_db":round(r["PSNR_db_vs_stock"]-off["PSNR_db_vs_stock"],6),
             "ssim_vs_stock":r["SSIM_vs_stock"],
             "ssim_delta_vs_off":round(r["SSIM_vs_stock"]-off["SSIM_vs_stock"],6),
             "audio_snr_db_vs_stock":r["audio_SNR_db_vs_stock"],
             "excluded_shape_by_phase":r["excluded_shape_by_phase"]}
            for r in rows[1:]
        ],
        "arms":rows,
        "limitations":[
            "One video per configuration; wall-time speed changes can be noise.",
            "Stock similarity does not establish perceptual quality.",
            "Candidate E N4096 K2048 failed earlier checkpoint precision p99/max gate.",
        ],
    }
    (out/"summary.json").write_text(json.dumps(results,indent=2,allow_nan=True)+"\n")
    fields=[key for key in rows[0] if key!="excluded_shape_by_phase"]
    with (out/"summary.csv").open("w",newline="") as csv_file:
        writer=csv.DictWriter(csv_file,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print("LTX25_BAR73_STAGE_SHAPE_PASS",json.dumps({
        "mode":mode,"summary":str(out/"summary.json"),
        "stock_seconds":rows[0]["wall_seconds"],
        "arms":results["optimized_arms"],
    },sort_keys=True),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",required=True,choices=("t2v","i2v"))
    args=parser.parse_args()
    run_mode(args.mode)
