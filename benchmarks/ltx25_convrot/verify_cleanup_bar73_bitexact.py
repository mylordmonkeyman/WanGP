#!/usr/bin/env python3
"""Prove retiring profiling did not change LTX2.5 73f T2V/I2V output.

Both stages of Candidate E N4096,K2048 are enabled with NO opt-out flags.
Hashes are pinned to the successful original A-B-B-A bar-prompt experiment
43686816635. Compares decoded frames/PCM, not MP4 container bytes.
"""
import argparse
import json
import os
from pathlib import Path

import t2v73_241_original_stock_fourway as four
import distilled_bar73_stage_shape_fourway as prior
from profile_gemm_t2v_i2v import INPUT_IMAGE,PROMPT,check_png

CHECKPOINT=Path("/opt/ltx25/checkpoints/ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors")
KNOWN={
    "t2v":{
        "video":"e94615c258b08ccc8ed6d8c3f011ee30f185f8b26eec6fbee0a291c88c48ccce",
        "audio":"242da9d646db239c35f16614176fee243d138e8bba5f6ce993f3e60ffa29da2e",
    },
    "i2v":{
        "video":"bf5829284d684b4a8d07ce993df21854fb8af3e179209b0c7f99096520acef2a",
        "audio":"65a7dc1bb11e5d203367bdcc7cd150d5cb6c1d35f800ef063b17e30a90c0cae8",
    },
}


def require(cond,msg):
    if not cond:
        raise RuntimeError(msg)


def main(mode):
    require(mode in KNOWN,"Unsupported mode")
    require(CHECKPOINT.is_file() and CHECKPOINT.stat().st_size>1_000_000_000,
            "Original Distilled INT8 ConvRot checkpoint absent")
    check_png()
    root=Path(os.environ.get("LTX25_CLEAN_REGRESSION_OUT",
        f"/cache/ltx25/results/cleanup-e73-{mode}-{os.environ.get('GITHUB_RUN_ID','manual')}"
    )).resolve()
    require(not root.exists(),f"Output directory already exists: {root}")
    for d in ("results","logs","clips","config","videos"):
        (root/d).mkdir(parents=True,exist_ok=False)
    four.OUT=root
    four.base.OUT=root
    four.base.FRAMES=73
    four.base.RESOLUTION="1024x576"
    four.base.STEPS=8
    four.base.SEED=20261112
    four.base.PROMPT=PROMPT
    four.base.EXTENSION=root/"libltx25_cublas_e_sm70.so"

    settings=four.settings(73)
    if mode=="i2v":
        settings.update(image_start=str(INPUT_IMAGE),image_prompt_type="S")
    require(settings.get("model_type")=="ltx2_25_22B_distilled"
            and settings.get("num_inference_steps")==8
            and settings.get("resolution")=="1024x576"
            and settings.get("video_length")==73,
            "Changed inference settings")
    (root/"input.json").write_text(json.dumps(settings,indent=2)+"\n")
    previous=Path(f"/cache/ltx25/results/bar73-stage-shape-{mode}-43683731705-attempt1/input.json")
    require(previous.exists() and json.loads(previous.read_text())==settings,
            "Settings differ from pre-cleanup source runs")

    four.base.gpu_free()
    four.base.command_run([
        "nvcc","-O3","-std=c++17","-arch=sm_70",
        "-shared","-Xcompiler","-fPIC",
        "benchmarks/ltx25_convrot/ltx25_cublas_e.cu","-lcublas",
        "-o",str(four.base.EXTENSION),
    ],timeout=300,log=root/"logs/compile.log")
    require(four.base.EXTENSION.is_file(),"Missing SM70 native extension")
    orig_env=four.optimized_env
    def env_for(arm,name):
        env=orig_env(arm,name)
        for old in ("WAN2GP_LTX25_E_TEST_4096X2048",
                    "WAN2GP_LTX25_E_TEST_4096X2048_PHASES",
                    "WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE1",
                    "WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE2",
                    "WAN2GP_LTX25_DEFAULT_ON"):
            env.pop(old,None)
        env.update({
            "WAN2GP_LTX25_E_MIN_M":"64",
            "WAN2GP_LTX25_E_MAX_M":"8192",
            "WAN2GP_LTX25_E_ROW_CENSUS":"1",
        })
        return env
    four.optimized_env=env_for
    four.verify_fp16=prior.audit_stage2
    template=four.base.initialize_config()
    four.base.gpu_free()
    print("CLEANUP_BAR73_BEGIN",mode,flush=True)
    row=four.run_case(73,1,"stage2_fp16",template)
    log=Path(row["log"]).read_text(errors="replace")
    require(CHECKPOINT.name in log
            and "Loading NAD Diffusion Decoder" not in log,
            "Wrong checkpoint or decoder")
    e=four.read_candidate_stats(log)
    require(e is not None and e.get("mode")=="fp32rot"
            and e.get("rotation_mode")=="radix4"
            and e.get("active_4096x2048_phases")==["stage1","stage2"],
            "Default Candidate E both-stage opt-in not active")
    stage=e.get("shape_4096x2048_by_phase",{})
    require(stage.get("stage1")=={"used":384,"fallback":0}
            and stage.get("stage2")=={"used":144,"fallback":0},
            "Target-family per-stage routing changed")
    require(e.get("used")==27360,"Total Candidate E call count drifted")
    require(row["fp16"]["guard_fallbacks"]==0 and
            row["fp16"]["persistent_fp16_weight_copies"]==0,
            "Unsafe selective FP16 guard fallback or persistent weight copy")
    require(row["decoded_video_sha256"]==KNOWN[mode]["video"],
            "VIDEO NUMERICAL REGRESSION after cleanup")
    require(row["decoded_audio_sha256"]==KNOWN[mode]["audio"],
            "AUDIO NUMERICAL REGRESSION after cleanup")
    report={
        "status":"PASS","mode":mode,
        "reference_run":43686816635,
        "resolution":"1024x576","frames":73,"steps":8,
        "seconds":row["wall_seconds"],
        "candidate_e_used":e["used"],
        "per_stage_target_calls":stage,
        "video_frames_sha256":row["decoded_video_sha256"],
        "pcm_audio_sha256":row["decoded_audio_sha256"],
        "matches_precleanup_video":True,
        "matches_precleanup_audio":True,
        "original_bar_input":str(INPUT_IMAGE) if mode=="i2v" else None,
    }
    (root/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print("LTX25_CLEANUP_BAR73_BITEXACT_PASS",json.dumps(report,sort_keys=True),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("t2v","i2v"),required=True)
    args=parser.parse_args()
    main(args.mode)
