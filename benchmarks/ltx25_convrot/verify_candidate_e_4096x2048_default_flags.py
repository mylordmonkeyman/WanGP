#!/usr/bin/env python3
"""SM70 49-frame real-video default/disable-stage1/disable-stage2/disable-both.

Validates ALL four stage-flag combinations in a fresh WanGP subprocess using
real LTX-2.5 Distilled INT8 ConvRot and the same bar T2V prompt/seed.
Flags are omitted, not set to zero, for the default-on arm.
"""
import csv
import json
import os
from pathlib import Path
import statistics

import t2v73_241_original_stock_fourway as four
import distilled_bar73_stage_shape_fourway as prior
import distilled_stock49_reference_psnr as metrics

OUT=Path(os.environ.get("LTX25_E_DEFAULT_FLAGS_OUT",
    "/cache/ltx25/results/candidate-e-default-flags-manual"))
CASES=(
    ("default_both",False,False),
    ("disable_stage1",True,False),
    ("disable_stage2",False,True),
    ("disable_both",True,True),
)
FRAMES=49
CHECKPOINT=Path("/opt/ltx25/checkpoints/ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors")
STOCK=Path("/cache/ltx25/results/distilled-stock49-43681488566-attempt1/clips/t2v49-01-stock_upstream.mp4")


def require(x,msg):
    if not x:raise RuntimeError(msg)


def main():
    require(CHECKPOINT.is_file() and CHECKPOINT.stat().st_size>1_000_000_000,
            "Missing checkpoint")
    require(STOCK.is_file(),"Missing original 49f stock reference")
    require(not OUT.exists(),"Output exists, needs fresh unique run")
    for directory in ("results","logs","clips","config","videos"):
        (OUT/directory).mkdir(parents=True,exist_ok=False)

    four.OUT=OUT; four.base.OUT=OUT
    four.base.FRAMES=FRAMES; four.base.STEPS=8
    four.base.SEED=20261112; four.base.RESOLUTION="1024x576"
    four.base.PROMPT=four.PROMPT
    four.base.EXTENSION=OUT/"libltx25_cublas_e_sm70.so"
    (OUT/"input.json").write_text(json.dumps(four.settings(FRAMES),indent=2)+"\n")
    ffmpeg=four.base.ffmpeg_bin()
    four.base.gpu_free()
    four.base.command_run([
        "nvcc","-O3","-std=c++17","-arch=sm_70",
        "-shared","-Xcompiler","-fPIC",
        "benchmarks/ltx25_convrot/ltx25_cublas_e.cu","-lcublas",
        "-o",str(four.base.EXTENSION),
    ],timeout=300,log=OUT/"logs/compile-e.log")
    require(four.base.EXTENSION.is_file(),"Candidate E extension missing")

    old_env=four.optimized_env
    case_stage1=False
    case_stage2=False
    def config_env(arm,name):
        env=old_env(arm,name)
        for k in ("WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE1",
                  "WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE2"):
            env.pop(k,None)
        # The default case does not specify either flag. The other cases
        # pass ONLY the requested explicit opt-out(s).
        if case_stage1:
            env["WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE1"]="1"
        if case_stage2:
            env["WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE2"]="1"
        env.update({
            "WAN2GP_LTX25_E_MIN_M":"64",
            "WAN2GP_LTX25_E_MAX_M":"8192",
            "WAN2GP_LTX25_E_ROW_CENSUS":"1",
        })
        return env
    four.optimized_env=config_env
    four.verify_fp16=prior.audit_stage2
    template=four.base.initialize_config()
    stock_audio=metrics.audio_pcm(STOCK,ffmpeg)
    rows=[]
    for index,(label,disable_stage1,disable_stage2) in enumerate(CASES,1):
        case_stage1,case_stage2=disable_stage1,disable_stage2
        four.base.gpu_free()
        print("DEFAULT_FLAGS49_BEGIN",label,flush=True)
        row=four.run_case(FRAMES,index,"stage2_fp16",template)
        logfile=Path(row["log"]).read_text(errors="replace")
        require(CHECKPOINT.name in logfile
                and "Loading NAD Diffusion Decoder" not in logfile,
                "Wrong checkpoint or decoder")
        e=four.read_candidate_stats(logfile)
        require(e and e.get("mode")=="fp32rot" and e.get("rotation_mode")=="radix4",
                "Candidate E wrong kernel mode")
        require(e.get("min_m")==64 and e.get("max_m")==8192
                and e.get("row_census_enabled") is True,
                "Unexpected Candidate E row eligibility")
        expected=["stage1","stage2"]
        if disable_stage1:expected.remove("stage1")
        if disable_stage2:expected.remove("stage2")
        require(e.get("active_4096x2048_phases")==expected,
                "Effective default/opt-out phases differ")
        expected_disabled=[a for a,v in (("stage1",disable_stage1),
                                        ("stage2",disable_stage2)) if v]
        require(e.get("disabled_4096x2048_stages")==expected_disabled,
                "Opt-out stage list differs from actual env flags")
        target=e.get("shape_4096x2048_by_phase",{})
        require(set(target)=={"stage1","stage2"},"Stage counters absent")
        for stage,total,off in (("stage1",384,disable_stage1),
                                ("stage2",144,disable_stage2)):
            observed=(int(target[stage]["used"]),int(target[stage]["fallback"]))
            want=(0,total) if off else (total,0)
            require(observed==want,
                    f"{label} {stage}: expected dispatch {want}, got {observed}")
        require(row["fp16"]["converted_per_stage"]["stage1"]==0,
                "Stage1 selective FP16 unexpectedly enabled")
        # M_MIN=64 leaves other, non-eligible fallback GEMMs routed
        # through Stage2 transient FP16: 1728 regardless of Stage1 flag.
        # Disabling the target family in Stage2 adds exactly 144 Quanto
        # GEMMs to the 1728 baseline, for 1872 total.
        expected_stage2_converted=1872 if disable_stage2 else 1728
        require(row["fp16"]["converted_per_stage"]["stage2"]==expected_stage2_converted,
                f"Stage2 selector expected {expected_stage2_converted}, got "
                f"{row['fp16']['converted_per_stage']['stage2']}")
        require(row["fp16"]["persistent_fp16_weight_copies"]==0
                and row["fp16"]["guard_fallbacks"]==0,
                "Unsafe FP16 fallback or persistent weight copy")
        clip=Path(row["clip"])
        record={
            "case":label,
            "disable_stage1":disable_stage1,
            "disable_stage2":disable_stage2,
            "effective_phases":expected,
            "wall_s":row["wall_seconds"],
            "target_stage1_used":int(target["stage1"]["used"]),
            "target_stage2_used":int(target["stage2"]["used"]),
            "candidate_e_used_total":int(e["used"]),
            "stage2_fp16_converted":row["fp16"]["converted_per_stage"]["stage2"],
            "psnr_vs_stock_db":metrics.metric(ffmpeg,STOCK,clip,"psnr"),
            "ssim_vs_stock":metrics.metric(ffmpeg,STOCK,clip,"ssim"),
            "audio_pcm_equal_to_stock":metrics.audio_pcm(clip,ffmpeg)==stock_audio,
            "video_sha256":row["decoded_video_sha256"],
            "audio_sha256":row["decoded_audio_sha256"],
            "clip":str(clip),
        }
        rows.append(record)
        (OUT/"partial-results.json").write_text(json.dumps(rows,indent=2)+"\n")
        print("DEFAULT_FLAGS49_CASE_PASS",json.dumps(record,sort_keys=True),flush=True)
    require(len(rows)==4,"Not all default+optout arms finished")
    summary={
        "status":"PASS_NOT_PERCEPTUALLY_APPROVED",
        "model":"LTX-2.5 Distilled INT8 ConvRot",
        "frames":49,"resolution":"1024x576",
        "default_shape_4096x2048_both_stages_enabled":True,
        "stage_1_and_2_disable_flags_operational":True,
        "only_new_flags_changed_in_each_optimized_run":True,
        "control":"Actual unchanged WanGP stock video reference run 43681488566",
        "rows":rows,
        "warning":"Shape failed historical strict per-GEMM error gate; user explicitly requested default despite this."
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    with (OUT/"summary.csv").open("w",newline="") as f:
        fields=[k for k in rows[0] if k not in ("effective_phases","video_sha256","audio_sha256","clip")]
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader();writer.writerows(rows)
    print("LTX25_E_DEFAULT_FLAGS49_PASS",json.dumps({
        "cases":len(rows),
        "summary":str(OUT/"summary.json"),
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
