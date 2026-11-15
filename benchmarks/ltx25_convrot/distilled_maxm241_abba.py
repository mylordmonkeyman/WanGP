#!/usr/bin/env python3
"""Long T2V Distilled Candidate E upper cutoff 8192 vs 18432 ABBA.

Actual 241-frame token geometry contains M=17856 and 1,296 Stage2 Quanto
calls with existing allowlisted (N,K) weights. Keep minimum 64 and the
previously rejected N4096,K2048 shape excluded in every trial.
"""
from __future__ import annotations
import json,os,statistics
from pathlib import Path

import t2v73_241_original_stock_fourway as four

OUT=Path(os.getenv("LTX25_DISTILLED_MAXM_OUT","/cache/ltx25/results/distilled-maxm-abba-manual"))
CHECKPOINT=Path("/opt/ltx25/checkpoints/ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors")
CASES=(("A1",8192),("B1",18432),("B2",18432),("A2",8192))
FRAMES=241
M_TARGET=17856
WEIGHTS={(2048,4096),(4096,4096)}


def require(cond,msg):
    if not cond:raise RuntimeError(msg)


def count_m_target(census,key):
    return sum(int(stats.get(key,0)) for signature,stats in census.items()
               if signature.startswith(f"M={M_TARGET},")
               and tuple(int(z.split("=")[1]) for z in signature.split(",")[1:]) in WEIGHTS)


def main():
    require(CHECKPOINT.is_file() and CHECKPOINT.stat().st_size>1_000_000_000,"Missing original Distilled checkpoint")
    require(not OUT.exists(),"Existing output path not allowed")
    for dirname in ("config","videos","logs","results","clips"):
        (OUT/dirname).mkdir(parents=True,exist_ok=False)
    four.OUT=OUT;four.base.OUT=OUT
    four.base.EXTENSION=OUT/"libltx25_cublas_e_sm70.so"
    four.base.FRAMES=FRAMES
    four.base.RESOLUTION="1024x576";four.base.STEPS=8;four.base.SEED=20261112
    four.base.PROMPT=four.PROMPT
    (OUT/"input.json").write_text(json.dumps(four.settings(FRAMES),indent=2)+"\n")
    four.base.gpu_free()
    four.base.command_run(["nvcc","-O3","-std=c++17","-arch=sm_70","-shared",
        "-Xcompiler","-fPIC","benchmarks/ltx25_convrot/ltx25_cublas_e.cu",
        "-lcublas","-o",str(four.base.EXTENSION)],timeout=300,
        log=OUT/"logs/compile-candidate-e.log")
    require(four.base.EXTENSION.is_file(),"Missing extension")
    old_env=four.optimized_env
    current_max=8192
    def candidate_env(arm,name):
        env=old_env(arm,name)
        env.update({
            "WAN2GP_LTX25_E_MIN_M":"64",
            "WAN2GP_LTX25_E_MAX_M":str(current_max),
            "WAN2GP_LTX25_E_ROW_CENSUS":"1",
            "WAN2GP_LTX25_E_TEST_4096X2048":"0",
        })
        return env
    four.optimized_env=candidate_env
    template=four.base.initialize_config()
    (OUT/"manifest.json").write_text(json.dumps({
        "kind":"241-frame Distilled upper-bound Candidate E A/B/B/A",
        "seed":20261112,"frames":FRAMES,"steps":8,"resolution":"1024x576",
        "model":str(CHECKPOINT),"lora":None,"vae":"stock Conv VAE",
        "min_m":64,"max_m_tested":[8192,18432],
        "excluded_weight_shape_flag":"off",
        "stage1_fp16":False,"stage2_fp16":True,
        "targeted_model_workload":{"M":M_TARGET,"approved_NK":sorted(map(list,WEIGHTS))},
        "order":list(CASES),"prompt":four.PROMPT,
        "source_sha":os.environ.get("GITHUB_SHA","manual")
    },indent=2)+"\n")

    reports=[]
    for idx,(label,max_m) in enumerate(CASES,1):
        current_max=max_m
        four.base.gpu_free()
        print("DISTILLED_MAXM_BEGIN",label,max_m,flush=True)
        row=four.run_case(FRAMES,idx,"stage2_fp16",template)
        log=Path(row["log"]).read_text(errors="replace")
        require(CHECKPOINT.name in log and "Loading NAD Diffusion Decoder" not in log,
                "Wrong model or decoder")
        candidate=four.read_candidate_stats(log)
        require(candidate and candidate.get("min_m")==64 and candidate.get("max_m")==max_m,
                "Wrong effective cutoff")
        require(candidate.get("row_census_enabled") and not candidate.get("experimental_failed_shape_enabled"),
                "Unapproved weight family or missing census")
        census=candidate.get("row_census",{})
        used=count_m_target(census,"used")
        fallback=count_m_target(census,"fallback")
        require(used==1296 and fallback==0 if max_m==18432 else used==0 and fallback==1296,
                f"Unexpected live M17856 routing: used={used}, fallback={fallback}")
        require(row["fp16"]["converted_per_stage"]["stage1"]==0
                and row["fp16"]["converted_per_stage"]["stage2"]>0,
                "Stage2 FP16 default mismatch")
        row.update(label=label,max_m=max_m,
                   highM_candidate_e_calls=used,highM_fallback_calls=fallback)
        reports.append(row)
        (OUT/"partial-results.json").write_text(json.dumps(reports,indent=2)+"\n")
        print("DISTILLED_MAXM_CASE_PASS",json.dumps({
            "case":label,"max_m":max_m,"seconds":row["wall_seconds"],
            "E_used":row["candidate_e"]["used"],
            "HighM_used":used,"HighM_fallback":fallback,
            "stage2_fp16":row["fp16"]["converted_per_stage"]["stage2"]
        },sort_keys=True),flush=True)
    a=[r["wall_seconds"] for r in reports if r["max_m"]==8192]
    b=[r["wall_seconds"] for r in reports if r["max_m"]==18432]
    require(len(a)==2 and len(b)==2,"Incomplete ABBA")
    base=statistics.median(a);candidate=statistics.median(b)
    ffmpeg=four.base.ffmpeg_bin()
    comparisons=[]
    for row in reports:
        video=four.base.psnr(Path(reports[0]["clip"]),Path(row["clip"]),ffmpeg)
        comparisons.append({
            "case":row["label"],"max_m":row["max_m"],"wall_seconds":row["wall_seconds"],
            "E_calls":row["candidate_e"]["used"],
            "M17856_E_calls":row["highM_candidate_e_calls"],
            "stage2_fp16_converted":row["fp16"]["converted_per_stage"]["stage2"],
            "vs_baseline_video_psnr":video.get("average_db"),
            "psnr_status":video.get("status"),
            "audio_equal":row["decoded_audio_sha256"]==reports[0]["decoded_audio_sha256"],
            "video_exact":row["decoded_video_sha256"]==reports[0]["decoded_video_sha256"],
            "clip":row["clip"],
        })
    summary={
        "status":"MEASURED_NOT_QUALITY_APPROVED",
        "baseline_max8192_median_s":round(base,3),
        "trial_max18432_median_s":round(candidate,3),
        "trial_minus_baseline_s":round(candidate-base,3),
        "trial_speedup_percent":round(100*(base-candidate)/base,3),
        "M17856_selected_calls_trial":1296,
        "comparison":comparisons,
        "limits":"One ABBA cohort; changed FP32 rotation arithmetic requires visual review before any default promotion"
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("LTX25_DISTILLED_MAXM_ABBA_PASS",json.dumps(summary,sort_keys=True),flush=True)


if __name__=="__main__":main()
