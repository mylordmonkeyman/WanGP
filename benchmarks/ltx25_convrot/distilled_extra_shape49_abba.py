#!/usr/bin/env python3
"""Distilled 49f ABBA: research-only N4096,K2048 Candidate E vs fallback.

One INT8 ConvRot checkpoint, M>=16 for all arms, same Radix4, Stage2 FP16
default. Only the previously rejected shape is opt-in for B arms.
This produces quality-review videos but NEVER auto-promotes rejected shape.
"""
import json,os,statistics
from pathlib import Path
import t2v73_241_original_stock_fourway as four

OUT=Path(os.getenv("LTX25_DISTILLED_EXTRA_OUT","/cache/ltx25/results/distilled-extra-shape-manual"))
FRAMES=49
CASES=(("A1",False),("B1",True),("B2",True),("A2",False))
CHECKPOINT=Path("/opt/ltx25/checkpoints/ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors")


def require(condition,message):
    if not condition:raise RuntimeError(message)


def main():
    require(CHECKPOINT.is_file() and CHECKPOINT.stat().st_size>1_000_000_000,"Missing checkpoint")
    require(not OUT.exists(),"Output directory must not exist")
    for subdir in ("config","videos","logs","results","clips"):
        (OUT/subdir).mkdir(parents=True,exist_ok=False)
    four.OUT=OUT;four.base.OUT=OUT
    four.base.EXTENSION=OUT/"libltx25_cublas_e_sm70.so"
    four.base.FRAMES=FRAMES
    four.base.RESOLUTION="1024x576";four.base.STEPS=8;four.base.SEED=20261112
    four.base.PROMPT=four.PROMPT
    (OUT/"input.json").write_text(json.dumps(four.settings(FRAMES),indent=2)+"\n")
    four.base.gpu_free()
    four.base.command_run(["nvcc","-O3","-std=c++17","-arch=sm_70","-shared","-Xcompiler","-fPIC",
                           "benchmarks/ltx25_convrot/ltx25_cublas_e.cu","-lcublas",
                           "-o",str(four.base.EXTENSION)],timeout=300,
                           log=OUT/"logs/compile-candidate-e.log")
    require(four.base.EXTENSION.is_file(),"Native Candidate E library missing")
    original_env=four.optimized_env
    enable=False

    def case_env(arm,name):
        env=original_env(arm,name)
        env.update({
            "WAN2GP_LTX25_E_MIN_M":"16",
            "WAN2GP_LTX25_E_MAX_M":"8192",
            "WAN2GP_LTX25_E_ROW_CENSUS":"1",
            "WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE1":"0" if enable else "1",
            "WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE2":"0" if enable else "1",
        })
        return env

    four.optimized_env=case_env
    # When the opt-in family is enabled, Candidate E consumes every eligible
    # Stage2 Quanto GEMM at this 49-frame token geometry. Zero downstream
    # selective-FP16 conversions is a *correct routing result*, not a
    # missing-activation failure. Preserve all safety/provenance checks.
    original_verify_fp16 = four.verify_fp16
    def verify_effective_fp16(path, arm):
        if not enable:
            return original_verify_fp16(path, arm)
        report = json.loads(path.read_text())
        stats = report.get("stats", {})
        assert arm == "stage2_fp16"
        require(report.get("backend") == "direct" and report.get("sham") is False,
                "Experimental B: wrong FP16 backend or sham enabled")
        require(report.get("persistent_fp16_weight_copies") == 0,
                "Experimental B: unexpected persistent FP16 weight copies")
        require(int(stats.get("observed_qbytes_mm", 0)) == 0
                and int(stats.get("converted_qbytes_mm", 0)) == 0,
                "Experimental B: unexpectedly routed to Quanto (requires separate audit)")
        require(int(stats.get("guard_input_reference", 0)) == 0
                and int(stats.get("guard_output_reference", 0)) == 0,
                "Experimental B: FP16 numerical guard fallback")
        return {
            "converted_per_stage": {"stage1": 0, "stage2": 0},
            "observed_qbytes_mm": 0,
            "converted_qbytes_mm": 0,
            "guard_fallbacks": 0,
            "persistent_fp16_weight_copies": 0,
            "candidate_e_preempted_stage2_quanto": True,
        }
    four.verify_fp16 = verify_effective_fp16
    template=four.base.initialize_config()
    (OUT/"manifest.json").write_text(json.dumps({
        "purpose":"Research-only previously rejected N4096,K2048 shape trial",
        "model":str(CHECKPOINT),"LoRA":False,"VAE":"standard Conv VAE",
        "seed":20261112,"frames":FRAMES,"steps":8,"resolution":"1024x576",
        "min_m":16,"max_m":8192,"stage2_fp16":True,"stage1_fp16":False,
        "baseline":"candidate E seven approved N,K shapes",
        "trial":"adds rejected N4096,K2048 only; unchanged minM for all runs",
        "order":list(CASES),"prompt":four.PROMPT,
        "source_sha":os.getenv("GITHUB_SHA","manual"),
    },indent=2)+"\n")
    rows=[]
    for idx,(label,flag) in enumerate(CASES,1):
        enable=flag
        four.base.gpu_free()
        print("DISTILLED_EXTRA_SHAPE_BEGIN",label,flag,flush=True)
        row=four.run_case(FRAMES,idx,"stage2_fp16",template)
        logs=Path(row["log"]).read_text(errors="replace")
        require(CHECKPOINT.name in logs and "Loading NAD Diffusion Decoder" not in logs,
                "Incorrect model or VAE")
        audit=four.read_candidate_stats(logs)
        require(audit and audit.get("active_4096x2048_phases")==(["stage1","stage2"] if flag else []),
                "Experimental shape selection wrong")
        require(audit.get("min_m")==16 and audit.get("max_m")==8192,
                "Unexpected Candidate E cutoffs")
        census=audit.get("row_census",{})
        matches=[v for k,v in census.items() if ",N=4096,K=2048" in k]
        require(matches,"Excluded shape wasn't observed")
        used=sum(int(x.get("used",0)) for x in matches)
        fallback=sum(int(x.get("fallback",0)) for x in matches)
        require((used>0 and fallback==0) if flag else (used==0 and fallback>0),
                "Excluded shape dispatch didn't switch as requested")
        require(row["fp16"]["converted_per_stage"]["stage1"]==0,
                "Stage1 FP16 accidentally changed")
        row.update(label=label,experimental_shape_enabled=flag,
                   candidate_e_excluded_shape_uses=used,
                   excluded_shape_fallbacks=fallback)
        rows.append(row)
        (OUT/"partial-results.json").write_text(json.dumps(rows,indent=2)+"\n")
        print("DISTILLED_EXTRA_SHAPE_CASE_PASS",json.dumps({
            "label":label,"enabled":flag,"seconds":row["wall_seconds"],
            "candidate_e_total":row["candidate_e"]["used"],
            "excluded_uses":used,"excluded_fallbacks":fallback,
            "stage2_fp16":row["fp16"]["converted_per_stage"]["stage2"]
        },sort_keys=True),flush=True)
    baseline=statistics.median(r["wall_seconds"] for r in rows if not r["experimental_shape_enabled"])
    trial=statistics.median(r["wall_seconds"] for r in rows if r["experimental_shape_enabled"])
    ffmpeg=four.base.ffmpeg_bin()
    comparisons=[]
    for r in rows:
        psnr=four.base.psnr(Path(rows[0]["clip"]),Path(r["clip"]),ffmpeg)
        comparisons.append({
            "case":r["label"],"shape_enabled":r["experimental_shape_enabled"],
            "wall_s":r["wall_seconds"],
            "candidate_e_calls":r["candidate_e"]["used"],
            "excluded_e_calls":r["candidate_e_excluded_shape_uses"],
            "stage2_fp16":r["fp16"]["converted_per_stage"]["stage2"],
            "vs_baseline_video_psnr":psnr.get("average_db"),
            "vs_baseline_video_psnr_status":psnr.get("status"),
            "audio_equal":r["decoded_audio_sha256"]==rows[0]["decoded_audio_sha256"],
            "video_exact":r["decoded_video_sha256"]==rows[0]["decoded_video_sha256"],
            "video":r["clip"]
        })
    summary={
        "status":"MEASURED_NOT_QUALITY_APPROVED",
        "baseline_median_seconds":round(baseline,3),
        "trial_median_seconds":round(trial,3),
        "trial_minus_baseline_s":round(trial-baseline,3),
        "trial_speedup_percent":round(100*(baseline-trial)/baseline,3),
        "comparisons":comparisons,
        "safety":"Previously failed historical precision gate; do not promote without additional evidence and visual review",
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("LTX25_DISTILLED_EXTRA_SHAPE_ABBA_PASS",json.dumps(summary,sort_keys=True),flush=True)


if __name__=="__main__":main()
