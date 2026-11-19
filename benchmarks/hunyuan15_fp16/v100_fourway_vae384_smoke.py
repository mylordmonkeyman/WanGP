#!/usr/bin/env python3
"""Hunyuan 1.5 V100 native-FP16 480/720 T2V/I2V opt-out smoke matrix.

Each of four real models generates a 33-frame MP4 with --no-hunyuan15-vae-384,
then another MP4 with the default enabled (no enable flag). Every run
requires the same rank32 converted LightX2V 480p T2V 4-step LoRA even
where training mismatch may reduce quality. Never silently drop the LoRA.
No model conversion, downloads, MMGP sweep, or production config mutation.
"""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

import v100_bar_video_smoke as base

ROOT = base.ROOT
CACHE = base.CACHE
PYTHON = base.PYTHON
OUT = Path(os.environ.get("HUNYUAN15_FOURWAY_OUT",
    str(CACHE / "results" / ("hunyuan15-fourway-smoke-" +
    os.environ.get("GITHUB_RUN_ID", "manual") + "-attempt" + os.environ.get("GITHUB_RUN_ATTEMPT", "1")))))
STAGED = CACHE / "results" / "hunyuan15-fourway-fp16-checkpoints"
CACHED = base.WRITABLE_CHECKPOINTS
LORA = (CACHE / "results" / "hunyuan15-converted-lora-43951751237-attempt1" /
        "converted_loras" / "hunyuan15_t2v_lightx2v4_rank32_wangp_qkv_split_bf16.safetensors")
CASES = (
    ("t2v", "480p", "hunyuan15_v100_t2v_480_fp16_experiment",
     "hunyuanvideo1.5_480p_t2v_fp16.safetensors", "832x480"),
    ("i2v", "480p", "hunyuan15_v100_i2v_480_step_fp16_experiment",
     "hunyuanvideo1.5_480p_i2v_step_distilled_fp16.safetensors", "832x480"),
    ("t2v", "720p", "hunyuan15_v100_t2v_720_fp16_experiment",
     "hunyuanvideo1.5_720p_t2v_fp16.safetensors", "1280x720"),
    ("i2v", "720p", "hunyuan15_v100_i2v_720_fp16_experiment",
     "hunyuanvideo1.5_720p_i2v_fp16.safetensors", "1280x720"),
)
CHECKPOINT_ROOTS = (STAGED, CACHED, base.MODEL_DIR, base.EXTRA_CHECKPOINTS)


def find_checkpoint(name):
    for folder in CHECKPOINT_ROOTS:
        p = folder / name
        if p.is_file() and p.stat().st_size > 15_000_000_000:
            return p
    raise FileNotFoundError("Full native FP16 model missing: " + name)


def preflight():
    base.gpu_preflight()
    if not LORA.is_file() or LORA.stat().st_size < 300_000_000:
        raise RuntimeError("Required converted LightX2V adapter is absent: " + str(LORA))
    for mode, res, model_type, name, output in CASES:
        p = find_checkpoint(name)
        trial = ROOT / "finetunes" / (model_type + ".json")
        if not trial.is_file():
            raise RuntimeError("Native-FP16 model definition missing: " + str(trial))
        d = json.loads(trial.read_text())
        if (not d["model"].get("hunyuan15_native_fp16")
                or d["model"].get("dtype") != "fp16"
                or d["model"]["architecture"] != "hunyuan_1_5_" + mode
                or d["resolution"] != output):
            raise RuntimeError("Unexpected model or finetune definition: " + str(trial))
        print("HUNYUAN15_FOURWAY_MODEL_READY " + json.dumps(
            {"case":mode+"_"+res,"checkpoint":str(p),"bytes":p.stat().st_size,
             "model_type":model_type}),flush=True)
    if shutil.which("ffprobe") is None or shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg/ffprobe required to validate all rendered MP4s")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(exist_ok=True)
    return base.hash_file(LORA)


def make_settings(mode, model_type, resolution):
    v = base.make_settings(mode)
    v.update({
        "model_type": model_type,
        "resolution": resolution,
        "video_length": 33,
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
        "flow_shift": 8.0,
        "activated_loras": [LORA.name],
        "loras_multipliers": "1.0",
        "skip_steps_cache_type": "",
    })
    if mode == "i2v":
        v["image_prompt_type"] = "S"
        v["image_start"] = str(base.IMAGE)
    else:
        v["image_prompt_type"] = ""
        v.pop("image_start", None)
    return v


def prepare(run_dir, mode, model_type, resolution):
    for folder in ("config", "videos"):
        (run_dir / folder).mkdir(parents=True,exist_ok=True)
    config=copy.deepcopy(base.select_template())
    config.update({
        "checkpoints_paths": [str(p) for p in CHECKPOINT_ROOTS],
        "transformer_quantization": "bf16",  # WanGP's 16-bit unquantized selection token
        "text_encoder_quantization": "int8",
        "transformer_types": [model_type],
        "profile": 1, "video_profile": 1,
        "attention_mode": "sdpa", "compile": "disabled",
        "vae_config": 0, "preload_model_policy": [],
        "mixed_precision": "0",
        "save_path": str(run_dir/"videos"),
        "image_save_path": str(run_dir/"videos"),
        "audio_save_path": str(run_dir/"videos"),
        "prevent_power_throttling": False,
    })
    (run_dir/"config"/"wgp_config.json").write_text(json.dumps(config,indent=2)+"\n")
    (run_dir/"settings.json").write_text(json.dumps(make_settings(mode,model_type,resolution),indent=2)+"\n")
    (run_dir/"lora_paths.json").write_text(json.dumps({"hunyuan_1_5":str(LORA.parent)},indent=2)+"\n")


def validate_video(video, resolution):
    p = subprocess.run(["ffprobe","-v","error","-select_streams","v:0",
        "-show_entries","stream=width,height,nb_frames,codec_name",
        "-of","json",str(video)],capture_output=True,text=True,timeout=90)
    if p.returncode:
        raise RuntimeError("ffprobe failed: "+p.stderr[-2000:])
    streams=json.loads(p.stdout).get("streams",[])
    if len(streams)!=1:
        raise RuntimeError("No single video stream in "+str(video))
    stream=streams[0]
    width,height=[int(v) for v in resolution.split("x")]
    if (stream.get("width"),stream.get("height"))!=(width,height):
        raise RuntimeError("Unexpected output frame size "+str(stream))
    p = subprocess.run(["ffmpeg","-hide_banner","-nostdin","-v","error",
        "-i",str(video),"-map","0:v:0","-f","framemd5","-"],
        capture_output=True,text=True,timeout=180)
    if p.returncode:
        raise RuntimeError("ffmpeg decode failed: "+p.stderr[-2000:])
    hashes=[line for line in p.stdout.splitlines() if line and not line.startswith("#")]
    if len(hashes)!=33:
        raise RuntimeError(f"Expected 33 decoded frames; found {len(hashes)}")
    return {"frames":33,"width":width,"height":height,"codec":stream.get("codec_name"),
            "bytes":video.stat().st_size,"sha256":base.hash_file(video)}


def video_quality(reference, candidate):
    metrics={}
    for name, pattern in (("psnr_db",r"average:([0-9.]+|inf)"),
                          ("ssim",r"All:([0-9.]+|inf)")):
        filt=name.split("_",1)[0] if name=="psnr_db" else "ssim"
        p=subprocess.run(["ffmpeg","-hide_banner","-nostdin","-loglevel","info",
            "-i",str(reference),"-i",str(candidate),"-filter_complex",
            "[0:v]setpts=PTS-STARTPTS[a];[1:v]setpts=PTS-STARTPTS[b];[a][b]"+filt,
            "-an","-f","null","-"],capture_output=True,text=True,timeout=180)
        if p.returncode:
            raise RuntimeError(f"FFmpeg {filt} failed: "+p.stderr[-1500:])
        found=re.findall(pattern,p.stderr)
        if not found:
            raise RuntimeError(f"FFmpeg {filt} summary missing: "+p.stderr[-1500:])
        metrics[name]=float(found[-1])
    return metrics


def generate(mode,res,model_type,checkpoint,resolution,variant):
    # One entirely fresh process per case: explicitly exercise the ordinary
    # Hunyuan VAE default path, not the P3C experimental benchmark hook.
    rundir=OUT/(mode+"_"+res+"_"+variant)
    prepare(rundir,mode,model_type,resolution)
    env=dict(os.environ)
    for key in list(env):
        if key.startswith(("HUNYUAN15_P3C_","HUNYUAN15_P4_","WAN2GP_LTX25_",
                           "LTX25_FP16_GEMM_","LTX25_CUBLAS_")):
            env.pop(key,None)
    env.update({
        "PYTHONPATH":str(ROOT),
        "HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1",
        "HF_DATASETS_OFFLINE":"1","HF_HUB_DISABLE_TELEMETRY":"1",
        "GRADIO_ANALYTICS_ENABLED":"False","OMP_NUM_THREADS":"8",
        "WAN2GP_LTX2_SKIP_BAKED_LORA_MIGRATION":"1",
    })
    argv=[PYTHON,"-u","wgp.py","--offline","--config",str(rundir/"config"),
          "--process",str(rundir/"settings.json"),
          "--output-dir",str(rundir/"videos"),"--attention","sdpa",
          "--fp16","--lora-config",str(rundir/"lora_paths.json"),
          "--no-prevent-power-throttling","--verbose","1"]
    if variant=="stock256":
        argv.append("--no-hunyuan15-vae-384")
    started=time.time()
    print("HUNYUAN15_FOURWAY_CASE_BEGIN "+json.dumps({
        "mode":mode,"resolution":res,"vae":variant,"checkpoint":checkpoint,
        "model_type":model_type,"lora":str(LORA),
        "steps":4,"guidance":1,"shift":8,"command":argv}),flush=True)
    result=base.run_command(argv,rundir/"generation.log",env,timeout=2400)
    log=(rundir/"generation.log").read_text(errors="replace")
    if result["returncode"]!=0:
        raise RuntimeError(f"WanGP failed: {mode}/{res}/{variant} rc={result['returncode']}")
    if "EXPERIMENTAL Hunyuan 1.5 native FP16 enabled" not in log:
        raise RuntimeError("Did not exercise native FP16 model path")
    if "Hunyuan 1.5 experimental loader: unpacked 216 Comfy QKV tensors" not in log:
        raise RuntimeError("Hunyuan Comfy QKV adapter not exercised")
    if ("HUNYUAN15_VAE384_DEFAULT" in log) != (variant=="default384"):
        raise RuntimeError("VAE 384 default/opt-out did not match requested variant")
    if (f"Loading Model '{checkpoint}'" not in log
        and str(find_checkpoint(checkpoint)) not in log):
        raise RuntimeError("Expected checkpoint not found in loader logs")
    if any(marker in log for marker in ("Downloading Model","Downloading Lora",
                                        "contains unexpected module keys")):
        raise RuntimeError("Unexpected download or incompatible adapter")
    videos=[p for p in (rundir/"videos").rglob("*.mp4") if p.stat().st_size>1024]
    if len(videos)!=1:
        raise RuntimeError(f"Expected one MP4, found {len(videos)}")
    dest=OUT/"videos"/f"hunyuan15-{mode}-{res}-{variant}-fp16-lightx2v.mp4"
    shutil.copy2(videos[0],dest)
    inspected=validate_video(dest,resolution)
    report={"mode":mode,"resolution":res,"vae":variant,"model_type":model_type,
            "checkpoint":checkpoint,"video":str(dest),"duration_wall_seconds":result["seconds"],
            "run_seconds_from_process":result["seconds"],
            "LoRA":LORA.name,"settings":make_settings(mode,model_type,resolution),
            "validation":inspected,"status":"success"}
    (rundir/"result.json").write_text(json.dumps(report,indent=2)+"\n")
    print("HUNYUAN15_FOURWAY_CASE_PASS "+json.dumps({
        "mode":mode,"resolution":res,"vae":variant,"video":dest.name,
        "seconds":result["seconds"],**inspected}),flush=True)
    return report


def main():
    lora_sha=preflight()
    report={"status":"RUNNING","branch":"optimization/hunyuan15-v100-fp16",
        "lora_sha256":lora_sha,"lora":str(LORA),"cases":{},
        "comparisons":{},"failures":{},"test_settings":{"frames":33,
            "steps":4,"guidance":1,"flow_shift":8,"mmgp":1,"vae_config":0,
            "attention":"sdpa"}}
    for mode,res,model_type,filename,resolution in CASES:
        for variant in ("stock256","default384"):
            key=mode+"_"+res+"_"+variant
            try:
                result=generate(mode,res,model_type,filename,resolution,variant)
                report["cases"][key]=result
            except Exception as exc:
                report["failures"][key]=str(exc)
                traceback.print_exc()
                print("HUNYUAN15_FOURWAY_CASE_FAILED "+
                    json.dumps({"case":key,"error":str(exc)}),flush=True)
            finally:
                (OUT/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
        a=report["cases"].get(mode+"_"+res+"_stock256")
        b=report["cases"].get(mode+"_"+res+"_default384")
        if a and b:
            quality=video_quality(Path(a["video"]),Path(b["video"]))
            comparison={
                "stock256_wall_seconds":a["duration_wall_seconds"],
                "default384_wall_seconds":b["duration_wall_seconds"],
                "delta_wall_seconds":a["duration_wall_seconds"]-b["duration_wall_seconds"],
                **quality,
                "h264_diagnostic_only":True,
                "note":"End-to-end independent cold generations, not a hot repeated A/B test"}
            report["comparisons"][mode+"_"+res]=comparison
            print("HUNYUAN15_FOURWAY_COMPARISON "+json.dumps(
                {"case":mode+"_"+res,**comparison}),flush=True)
            (OUT/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    report["status"]="COMPLETE" if not report["failures"] else "PARTIAL_FAILURE"
    (OUT/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print("HUNYUAN15_FOURWAY_FINAL "+json.dumps({"status":report["status"],
        "successes":len(report["cases"]),"failed":report["failures"],
        "comparisons":report["comparisons"]}),flush=True)
    if report["failures"]:
        raise RuntimeError("One or more smoke cases failed; MP4s for successful cases retained")


if __name__=="__main__":
    main()
