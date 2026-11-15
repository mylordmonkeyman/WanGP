#!/usr/bin/env python3
"""Real V100 validation: default Stage2 FP16 Dev/Distilled, plus Dev both-stage.

Keeps real MMGP/offload LoRA handling, uses no persistent FP16 weight copies,
and executes each case in a fresh wgp.py subprocess with a distinct file set.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import t2v73_241_original_stock_fourway as four

ROOT = Path(__file__).resolve().parents[2]
DEV_MODEL = Path("/cache/ltx25/models/ltx-2.5-22b-dev_diffusion_model_int8_convrot.safetensors")
DISTILLED = Path("/opt/ltx25/checkpoints/ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors")
CONV_VAE = Path("/opt/ltx25/checkpoints/ltx-2.5-22b_video_vae_bf16.safetensors")
LORA_NAME = "ltx-2.5-22b-distilled-lora-450_bf16.safetensors"
LORA_FILE = Path("/cache/ltx25/results/dev-lora-cache") / LORA_NAME
MODEL_TYPE = "ltx25_dev_fp16_stage_runner"
OUT = Path(os.environ.get("LTX25_PHASE_OUT", "/cache/ltx25/results/fp16-phase-default-manual"))
CASES = (
    ("dev_stage2_default", MODEL_TYPE, 73, "stage2_fp16", "default"),
    ("dev_both_explicit", MODEL_TYPE, 17, "both_fp16", "both"),
    ("distilled_stage2_default", "ltx2_25_22B_distilled", 17, "stage2_fp16", "default"),
)
EXPLICIT_BOTH = four.GROUP_BOTH
DEFAULT_STAGE2 = four.GROUP_STAGE2


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def flags_preflight():
    code = (
        "import json; from shared.kernels import ltx25_selective_fp16_gemm as f;"
        "print(json.dumps({'enabled':f.ENABLED,'groups':sorted(f.GROUPS),"
        "'ops':sorted(f.OPS),'backend':f.BACKEND,'min_m':f.MIN_M,"
        "'min_dim':f.MIN_DIM}))"
    )
    cases = (
        ("default", {}, True, DEFAULT_STAGE2.split(",")),
        ("off", {"WAN2GP_LTX25_FP16_GEMM": "0"}, False, DEFAULT_STAGE2.split(",")),
        ("both", {"WAN2GP_LTX25_FP16_GEMM": "1",
                  "LTX25_FP16_GEMM_GROUPS": EXPLICIT_BOTH}, True, EXPLICIT_BOTH.split(",")),
        ("stage1", {"WAN2GP_LTX25_FP16_GEMM": "1",
                    "LTX25_FP16_GEMM_GROUPS": "stage1:wide,stage1:tall,stage1:balanced"},
         True, ["stage1:wide","stage1:tall","stage1:balanced"]),
    )
    for name, overrides, expected_on, groups in cases:
        env = dict(os.environ)
        for k in list(env):
            if k == "WAN2GP_LTX25_FP16_GEMM" or k.startswith("LTX25_FP16_GEMM_"):
                env.pop(k, None)
        env.update(overrides)
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT, env=env,
            capture_output=True, text=True, timeout=40, check=True
        )
        data = json.loads(result.stdout.strip().splitlines()[-1])
        require(data["enabled"] == expected_on, f"{name}: enabled={data['enabled']}")
        require(data["groups"] == sorted(groups), f"{name}: groups {data['groups']}")
        if name == "default":
            require(data["ops"] == ["qbytes_mm"]
                    and data["backend"] == "direct"
                    and data["min_m"] == 1 and data["min_dim"] == 64,
                    f"Unexpected default selector options: {data}")
        print("FP16_FLAG_PREFLIGHT_PASS", name, json.dumps(data,sort_keys=True),flush=True)
    print("FP16_FLAG_PREFLIGHT_ALL_PASS",flush=True)


def main():
    flags_preflight()
    for p in (DEV_MODEL, DISTILLED, CONV_VAE, LORA_FILE):
        require(p.is_file() and p.stat().st_size > 100_000_000, "Missing: " + str(p))
    require(not OUT.exists(), "Output path already exists: " + str(OUT))
    for d in ("results","logs","clips","config","videos","inputs"):
        (OUT/d).mkdir(parents=True,exist_ok=False)

    # Model definitions and links live only in this disposable Actions checkout.
    spec = json.loads((ROOT/"defaults/ltx2_25_22B.json").read_text())
    spec["model"]["name"] = "LTX-2.5 Dev INT8 ConvRot FP16 stage regression"
    spec["model"]["URLs"] = [str(DEV_MODEL)]
    spec["model"]["preload_URLs"] = []
    spec["model"]["ltx2_video_vae_file"] = CONV_VAE.name
    spec["model"].pop("ltx2_pipeline",None)  # Distilled LoRA on Dev's native two-stage path
    spec["num_inference_steps"] = 8
    spec["sample_solver"] = "distilled_8_steps_ancestral"
    finetune = ROOT/"finetunes"/(MODEL_TYPE+".json")
    require(not finetune.exists(), "Refusing to overwrite finetune")
    finetune.write_text(json.dumps(spec,indent=2)+"\n")
    lora_dest = ROOT/"loras/ltx2"/LORA_NAME
    lora_dest.parent.mkdir(parents=True,exist_ok=True)
    if lora_dest.is_symlink():
        lora_dest.unlink()
    require(not lora_dest.exists(), "Unexpected existing LoRA path: "+str(lora_dest))
    lora_dest.symlink_to(LORA_FILE)

    four.OUT = OUT
    four.base.OUT = OUT
    four.base.EXTENSION = OUT/"libltx25_cublas_e_sm70.so"
    four.base.RESOLUTION = "1024x576"
    four.base.STEPS = 8
    four.base.SEED = 20261112
    four.base.PROMPT = four.PROMPT
    four.base.gpu_free()
    four.base.command_run(
        ["nvcc","-O3","-std=c++17","-arch=sm_70","-shared",
         "-Xcompiler","-fPIC","benchmarks/ltx25_convrot/ltx25_cublas_e.cu",
         "-lcublas","-o",str(four.base.EXTENSION)],
        timeout=300, log=OUT/"logs/compile-candidate-e.log",
    )
    require(four.base.EXTENSION.is_file(),"Candidate E compilation failed")

    original_config_for = four.base.config_for
    current_model_type = None
    current_mode = None
    original_optimized_env = four.optimized_env

    def custom_config_for(case, template):
        cfg_dir, target = original_config_for(case, template)
        path = cfg_dir/"wgp_config.json"
        cfg = json.loads(path.read_text())
        cfg["transformer_types"] = [current_model_type]
        cfg["checkpoints_paths"] = ["/opt/ltx25/checkpoints",str(DEV_MODEL.parent)]
        cfg["profile"] = 0
        cfg["video_profile"] = 0
        path.write_text(json.dumps(cfg,indent=2)+"\n")
        return cfg_dir, target

    def custom_optimized_env(arm, name):
        env = original_optimized_env("candidate_e", name)
        # Demonstrate actual source defaults: no implicit env selectors carried
        # in from benchmark configurations. Stats output path is harmless.
        for k in list(env):
            if k == "WAN2GP_LTX25_FP16_GEMM" or k.startswith("LTX25_FP16_GEMM_"):
                env.pop(k,None)
        env["LTX25_FP16_GEMM_STATS_PATH"] = str(OUT/"results"/(name+"-fp16.json"))
        if current_mode == "both":
            env["WAN2GP_LTX25_FP16_GEMM"]="1"
            env["LTX25_FP16_GEMM_GROUPS"] = EXPLICIT_BOTH
        return env

    four.base.config_for = custom_config_for
    four.optimized_env = custom_optimized_env
    # initialize_config only prepares the JSON configuration template; it is
    # not timed, and does not execute a generation.
    template = four.base.initialize_config()

    manifest = {
        "source_sha":os.getenv("GITHUB_SHA","manual"),
        "prompt": four.PROMPT, "resolution":"1024x576",
        "seed":20261112, "step_count":8,
        "models":{"dev":str(DEV_MODEL),"distilled":str(DISTILLED)},
        "distillation_lora":str(LORA_FILE),"lora_strength":1.0,
        "vae":str(CONV_VAE),"cases":[x[0] for x in CASES],
        "controls":{
            "implicit": "V100 Stage2 FP16 only, direct Quanto M>=1, N/K>=64",
            "both": "WAN2GP_LTX25_FP16_GEMM=1 and LTX25_FP16_GEMM_GROUPS="+EXPLICIT_BOTH,
            "off": "WAN2GP_LTX25_FP16_GEMM=0"
        },
        "timing": "fresh wgp.py process, including model load, generation, decoding, file save",
    }
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")

    reports=[]
    for i,(label,model_type,frames,arm,mode) in enumerate(CASES,1):
        current_model_type=model_type
        current_mode=mode
        four.base.FRAMES=frames
        settings=four.settings(frames)
        if model_type == MODEL_TYPE:
            settings.update({
                "model_type":model_type,"activated_loras":[LORA_NAME],"loras_multipliers":"1",
                "sample_solver":"distilled_8_steps_ancestral",
                "guidance_phases":2,"guidance_scale":1.0,
                "audio_guidance_scale":1.0,"audio_cfg_scale":1.0,
                "alt_guidance_scale":1.0,"alt_scale":0.0,
                "cfg_star_switch":0,"apg_switch":0,"perturbation_switch":0,
            })
        require(settings["seed"]==20261112 and settings["num_inference_steps"]==8
                and settings["resolution"]=="1024x576",
                "Unexpected input mutation")
        (OUT/"input.json").write_text(json.dumps(settings,indent=2)+"\n")
        (OUT/"inputs"/(label+".json")).write_text(json.dumps(settings,indent=2)+"\n")
        four.base.gpu_free()
        print("LTX25_PHASE_BEGIN",label,frames,model_type,mode,flush=True)
        row=four.run_case(frames,i,arm,template)
        log=Path(row["log"]).read_text(errors="replace")
        model_name=DEV_MODEL.name if model_type==MODEL_TYPE else DISTILLED.name
        require(model_name in log,"Model provenance failure for "+label)
        if model_type==MODEL_TYPE:
            require("Lora '" in log and LORA_NAME in log and "loaded in model" in log,
                    "LoRA activation not shown for "+label)
        require("Loading NAD Diffusion Decoder" not in log,
                "Unexpected NAD decoder in "+label)
        fp16=row["fp16"]
        require(fp16["converted_per_stage"]["stage2"]>0,
                "Stage2 FP16 not exercised in "+label)
        require(fp16["converted_per_stage"]["stage1"] > 0 if mode=="both" else fp16["converted_per_stage"]["stage1"] == 0,
                "Stage1 FP16 flag not honored in "+label)
        require(fp16["persistent_fp16_weight_copies"]==0,
                "Persistent duplicate FP16 weights in "+label)
        require(fp16["guard_fallbacks"]==0,"FP16 numerical guard fallback in "+label)
        row["variant"]=label
        row["model_checkpoint"]=str(DEV_MODEL if model_type==MODEL_TYPE else DISTILLED)
        row["lora_strength"] = 1.0 if model_type == MODEL_TYPE else 0.0
        reports.append(row)
        (OUT/"partial-results.json").write_text(json.dumps(reports,indent=2)+"\n")
        print("LTX25_PHASE_CASE_PASS",label,json.dumps({
            "wall_seconds":row["wall_seconds"],"fp16":fp16["converted_per_stage"],
            "candidate_e_uses":row["candidate_e"]["used"],"video":row["clip"]
        }),flush=True)

    require(len(reports)==len(CASES),"Expected three complete generation records")
    summary={
        "status":"PASS","case_count":len(reports),"tests":[
            {"variant":r["variant"],"frames":r["frames"],
             "wall_seconds":r["wall_seconds"],
             "fp16":r["fp16"]["converted_per_stage"],
             "candidate_e_uses":r["candidate_e"]["used"],
             "video":r["clip"],
             "video_hash":r["decoded_video_sha256"],
             "audio_hash":r["decoded_audio_sha256"]}
             for r in reports
        ]}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("LTX25_PHASE_DEFAULT_SUMMARY",json.dumps(summary,sort_keys=True),flush=True)
    print("LTX25_PHASE_DEFAULT_PASS",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--flags-only",action="store_true")
    args=parser.parse_args()
    if args.flags_only:
        flags_preflight()
    else:
        main()
