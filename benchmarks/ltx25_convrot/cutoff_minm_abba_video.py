#!/usr/bin/env python3
"""Controlled full 17-frame Dev+LoRA video A/B/B/A for Candidate E minimum M.

Tests M>=64 (current) against M>=16 (candidate) while leaving Stage2 FP16
on for the remaining Quanto GEMMs. No model data or production defaults change.
Each run is a new timed WanGP subprocess. Encoded video hashes are recorded
for visual comparison; correctness/quality are not assumed from close timing.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import statistics
import shutil

import t2v73_241_original_stock_fourway as four

ROOT=Path(__file__).resolve().parents[2]
DEV=Path("/cache/ltx25/models/ltx-2.5-22b-dev_diffusion_model_int8_convrot.safetensors")
LORA=Path("/cache/ltx25/results/dev-lora-cache/ltx-2.5-22b-distilled-lora-450_bf16.safetensors")
LORA_NAME=LORA.name
VAE=Path("/opt/ltx25/checkpoints/ltx-2.5-22b_video_vae_bf16.safetensors")
TYPE="ltx25_dev_minm_abba"
OUT=Path(os.environ.get("LTX25_MINM_VIDEO_OUT","/cache/ltx25/results/minm-video-manual"))
CASES=(("A1",64),("B1",16),("B2",16),("A2",64))


def require(cond,msg):
    if not cond:raise RuntimeError(msg)


def main():
    for p in (DEV,LORA,VAE):require(p.is_file() and p.stat().st_size>100_000_000,"Missing "+str(p))
    require(not OUT.exists(),"Result path must not exist: "+str(OUT))
    for d in ("results","logs","clips","config","videos","inputs"):
        (OUT/d).mkdir(parents=True,exist_ok=False)
    d=json.loads((ROOT/"defaults/ltx2_25_22B.json").read_text())
    d["model"]["name"]="LTX-2.5 Dev + dynamic LoRA Candidate E minM ABBA test"
    d["model"]["URLs"]=[str(DEV)]
    d["model"]["preload_URLs"]=[]
    d["model"]["ltx2_video_vae_file"]=VAE.name
    d["model"].pop("ltx2_pipeline",None)
    d["num_inference_steps"]=8
    d["sample_solver"]="distilled_8_steps_ancestral"
    f=ROOT/"finetunes"/(TYPE+".json")
    require(not f.exists(),"Finetune already exists")
    f.write_text(json.dumps(d,indent=2)+"\n")
    ld=ROOT/"loras/ltx2"/LORA_NAME
    ld.parent.mkdir(parents=True,exist_ok=True)
    require(not ld.exists(),"LoRA already present at "+str(ld))
    ld.symlink_to(LORA)

    four.OUT=OUT
    four.base.OUT=OUT
    four.base.EXTENSION=OUT/"libltx25_cublas_e_sm70.so"
    four.base.FRAMES=17
    four.base.RESOLUTION="1024x576"
    four.base.STEPS=8
    four.base.SEED=20261112
    four.base.PROMPT=four.PROMPT
    inp=four.settings(17)
    inp.update({
        "model_type":TYPE,"activated_loras":[LORA_NAME],"loras_multipliers":"1",
        "sample_solver":"distilled_8_steps_ancestral",
        "guidance_phases":2,"guidance_scale":1.0,
        "audio_guidance_scale":1.0,"audio_cfg_scale":1.0,
        "alt_guidance_scale":1.0,"alt_scale":0.0,
        "cfg_star_switch":0,"apg_switch":0,"perturbation_switch":0,
    })
    (OUT/"input.json").write_text(json.dumps(inp,indent=2)+"\n")
    four.base.gpu_free()
    four.base.command_run(["nvcc","-O3","-std=c++17","-arch=sm_70","-shared",
        "-Xcompiler","-fPIC","benchmarks/ltx25_convrot/ltx25_cublas_e.cu",
        "-lcublas","-o",str(four.base.EXTENSION)],timeout=300,
        log=OUT/"logs/compile-candidate-e.log")
    require(four.base.EXTENSION.is_file(),"Missing native extension")
    original_cfg=four.base.config_for
    current_min=64
    original_env=four.optimized_env

    def config_for(name, template):
        cfg_dir,target=original_cfg(name,template)
        p=cfg_dir/"wgp_config.json"
        cfg=json.loads(p.read_text())
        cfg["transformer_types"]=[TYPE]
        cfg["checkpoints_paths"]=["/opt/ltx25/checkpoints",str(DEV.parent)]
        cfg["profile"]=0;cfg["video_profile"]=0
        p.write_text(json.dumps(cfg,indent=2)+"\n")
        return cfg_dir,target

    def env_for(arm,name):
        env=original_env(arm,name)
        env["WAN2GP_LTX25_E_MIN_M"]=str(current_min)
        env["WAN2GP_LTX25_E_MAX_M"]="8192"
        env["WAN2GP_LTX25_E_ROW_CENSUS"]="1"
        return env

    four.base.config_for=config_for
    four.optimized_env=env_for
    template=four.base.initialize_config()
    manifest={
        "kind":"Candidate E vs Stage2 FP16 lower row cutoff ABBA",
        "model":str(DEV),"distillation_lora":str(LORA),"lora_strength":1.0,
        "vae":str(VAE),"seed":20261112,
        "frames":17,"resolution":"1024x576","steps":8,
        "prompt":four.PROMPT,"order":list(CASES),
        "candidate_e_max_m":8192,"stage1_fp16":False,"stage2_fp16":True,
        "source_sha":os.getenv("GITHUB_SHA","manual"),
        "timing":"complete standalone wgp.py launch through exit, with load and MP4 save",
    }
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    rows=[]
    for i,(name,minimum) in enumerate(CASES,1):
        current_min=minimum
        four.base.gpu_free()
        print("MINM_VIDEO_BEGIN",name,minimum,flush=True)
        row=four.run_case(17,i,"stage2_fp16",template)
        log=Path(row["log"]).read_text(errors="replace")
        require(DEV.name in log and LORA_NAME in log and "loaded in model" in log,
                name+": Dev or LoRA provenance missing")
        require("Loading NAD Diffusion Decoder" not in log,"Wrong VAE")
        raw=four.read_candidate_stats(log)
        require(raw is not None and raw.get("min_m")==minimum
                and raw.get("max_m")==8192 and raw.get("row_census_enabled"),
                name+": Candidate E cutoff or census not confirmed")
        counts=raw["row_census"]
        require(any(k.startswith("M=18,") for k in counts),
                name+": representative M18 row missing")
        require(row["fp16"]["converted_per_stage"]["stage1"]==0
                and row["fp16"]["converted_per_stage"]["stage2"]>0,
                name+": FP16 stage isolation failed")
        row["label"]=name
        row["candidate_e_min_m"]=minimum
        row["row_census"]=counts
        rows.append(row)
        (OUT/"partial-results.json").write_text(json.dumps(rows,indent=2)+"\n")
        print("MINM_VIDEO_CASE_PASS",json.dumps({
            "label":name,"min_m":minimum,"seconds":row["wall_seconds"],
            "candidate_e_uses":row["candidate_e"]["used"],
            "stage2_fp16":row["fp16"]["converted_per_stage"]["stage2"],
            "M18":{k:v for k,v in counts.items() if k.startswith("M=18,")},
            "clip":row["clip"]
        },sort_keys=True),flush=True)

    A=[r["wall_seconds"] for r in rows if r["candidate_e_min_m"]==64]
    B=[r["wall_seconds"] for r in rows if r["candidate_e_min_m"]==16]
    require(len(A)==2 and len(B)==2,"Incomplete ABBA")
    a,b=statistics.median(A),statistics.median(B)
    summary={
        "status":"PASS","baseline_min64_median_seconds":round(a,3),
        "candidate_min16_median_seconds":round(b,3),
        "candidate_minus_baseline_seconds":round(b-a,3),
        "candidate_vs_baseline_percent":round(100*(b/a-1),2),
        "per_run":[{"case":r["label"],"min_m":r["candidate_e_min_m"],
                    "wall_seconds":r["wall_seconds"],
                    "stage2_fp16_converted":r["fp16"]["converted_per_stage"]["stage2"],
                    "candidate_e_used":r["candidate_e"]["used"],
                    "video":r["clip"],"video_hash":r["decoded_video_sha256"],
                    "audio_hash":r["decoded_audio_sha256"],
                    "row_census":r["row_census"]}
                   for r in rows],
        "notes":"One ABBA cohort; video quality requires visual review, numeric differences expected."
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("LTX25_MINM_VIDEO_ABBA_PASS",json.dumps({
        k:v for k,v in summary.items() if k!="per_run"
    },sort_keys=True),flush=True)


if __name__=="__main__":main()
