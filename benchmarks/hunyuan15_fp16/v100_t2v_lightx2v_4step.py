#!/usr/bin/env python3
"""Cold-process, cached-checkpoint Hunyuan 1.5 LightX2V 4-step T2V smoke.

The user previously staged the 341 MB Comfy-Org 480p T2V LoRA. Reuse the
EXACT 30-step full-FP16 checkpoint, text, seed, dimensions and frame count.
Change only LoRA activation, steps 30->4, CFG 6->1, flow shift 5->8, per
WanGP's own built-in Hunyuan 1.5 LightX2V 4-step preset.

Preflight/LoRA hash/offline dry run are EXCLUDED from the wall-clock figure.
Generation timing includes a new Python process, model load, inference, VAE,
and MP4 saving. Do not report it as hot-model diffusion-only speed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import v100_bar_video_smoke as base
from v100_t2v_cached_steady_state import validate_clip

MODE = "t2v"
LORA_NAME = "hunyuanvideo1.5_t2v_480p_lightx2v_4step_lora_rank_32_bf16.safetensors"
LORA_DIR = base.WRITABLE_CHECKPOINTS / "loras"
LORA_FILE = LORA_DIR / LORA_NAME
# SHA256 for the published 341 MB safetensors, Comfy-Org HF Xet pointer.
LORA_SHA256 = "8fef6f9b9f2f6aa5259166e78b10a022485391dababc6a8c68585fddd31d5d62"
CHECKPOINT = base.WRITABLE_CHECKPOINTS / base.MODEL_FILES[MODE]
BASELINE_SUMMARY = (base.CACHE/"results"
    /"hunyuan15-cached-t2v-43933748031-attempt1"/"t2v"/"summary.json")


def main():
    base.gpu_preflight()
    if not CHECKPOINT.is_file() or CHECKPOINT.stat().st_size < 15_000_000_000:
        raise RuntimeError(f"Cached unquantized T2V checkpoint missing: {CHECKPOINT}")
    if not LORA_FILE.is_file() or LORA_FILE.stat().st_size < 300_000_000:
        raise RuntimeError(f"Comfy-Org LoRA not staged in Docker at {LORA_FILE}")
    actual_digest = base.hash_file(LORA_FILE)
    if actual_digest != LORA_SHA256:
        raise RuntimeError(f"LoRA SHA256 mismatch: {actual_digest} != {LORA_SHA256}")
    print("HUNYUAN15_LIGHTX2V_PREFLIGHT_PASS " + json.dumps({
        "checkpoint":str(CHECKPOINT),"checkpoint_bytes":CHECKPOINT.stat().st_size,
        "lora":str(LORA_FILE),"lora_bytes":LORA_FILE.stat().st_size,
        "lora_sha256":actual_digest,"downloads_in_timed_run":False
    },sort_keys=True),flush=True)

    base.OUT.mkdir(parents=True,exist_ok=True)
    config_path = base.OUT / "lightx2v_lora_paths.json"
    config_path.write_text(json.dumps({"hunyuan_1_5":str(LORA_DIR)},indent=2)+"\n")
    assert LORA_FILE.is_file()

    original_settings=base.make_settings
    def lora_settings(mode):
        if mode != MODE:
            raise RuntimeError("LightX2V LoRA is T2V ONLY: I2V unvalidated")
        s = original_settings(mode)
        s.update({
            "num_inference_steps":4,
            "guidance_scale":1.0,
            "flow_shift":8.0,
            "activated_loras":[LORA_NAME],
            "loras_multipliers":"1.0",
        })
        return s
    base.make_settings = lora_settings

    original_run=base.run_command
    def offline_run(cmd,log_path,env,timeout):
        cmd=list(cmd)
        if "wgp.py" in cmd:
            cmd.insert(cmd.index("wgp.py")+1,"--offline")
            cmd.extend(["--lora-config",str(config_path)])
        env=dict(env)
        env.update({
            "HF_HUB_OFFLINE":"1",
            "TRANSFORMERS_OFFLINE":"1",
            "HF_DATASETS_OFFLINE":"1",
        })
        return original_run(cmd,log_path,env,timeout)
    base.run_command=offline_run

    report=base.run_mode(MODE)
    log=(base.OUT/MODE/"logs"/"generation.log").read_text(errors="replace")
    if ("EXPERIMENTAL Hunyuan 1.5 native FP16 enabled" not in log
        or "Hunyuan 1.5 experimental loader: unpacked 216 Comfy QKV tensors" not in log):
        raise RuntimeError("Experimental FP16 transformer not loaded")
    if any(x in log for x in ("Unable to Download","Downloading Model","Downloading model","Downloading Lora")):
        raise RuntimeError("Missing dependency / download during timed measurement")
    settings=json.loads((base.OUT/MODE/"settings.json").read_text())
    assert settings["activated_loras"] == [LORA_NAME], settings["activated_loras"]
    assert settings["loras_multipliers"] == "1.0"
    assert settings["num_inference_steps"] == 4
    assert settings["guidance_scale"] == 1.0
    assert settings["flow_shift"] == 8.0

    videos=[validate_clip(Path(i["name"])) for i in report["videos"]]
    if not videos:
        raise RuntimeError("Missing output clip")

    old_time=None
    if BASELINE_SUMMARY.is_file():
        old_time=json.loads(BASELINE_SUMMARY.read_text()).get("run",{}).get("seconds")
    elapsed=report["run"]["seconds"]
    comparison={
        "status":"LIGHTX2V_4STEP_T2V_VIDEO_PASS",
        "model_type":report["model_type"],
        "checkpoint":str(CHECKPOINT),"checkpoint_bytes":CHECKPOINT.stat().st_size,
        "lora":str(LORA_FILE),"lora_bytes":LORA_FILE.stat().st_size,
        "lora_sha256":actual_digest,"lora_multiplier":1.0,
        "steps":4,"guidance":1.0,"flow_shift":8.0,
        "seed":base.SEED,"frames":base.FRAMES,"resolution":base.RESOLUTION,
        "text_prompt":base.PROMPT,"offline":True,
        "timing_scope":"new Python process, cached model/LoRA; loading+inference+decode+save; dry-run excluded",
        "generation_wall_seconds":elapsed,
        "previous_30step_cached_wall_seconds":old_time,
        "wall_time_saved_seconds":round(old_time-elapsed,1) if old_time is not None else None,
        "wall_time_speedup_factor":round(old_time/elapsed,3) if old_time and elapsed else None,
        "video_validation":videos,
        "note":"4-step versus 30-step is a quality/sampling change; speedup is not a same-quality or same-step kernel benchmark",
    }
    (base.OUT/MODE/"lightx2v_4step_comparison.json").write_text(
        json.dumps(comparison,indent=2)+"\n")
    print("HUNYUAN15_LIGHTX2V_4STEP_T2V_PASS "+json.dumps(comparison,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
