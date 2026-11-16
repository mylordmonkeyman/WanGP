#!/usr/bin/env python3
"""Offline, already-cached T2V 480p / 33f V100 wall-time measurement.

Reuses the SAME WanGP settings and subprocess implementation as the successful
Hunyuan 1.5 smoke. Only the model-download state changes: the ~16 GB T2V
checkpoint and ancillary files must exist before the clock starts.

This reports full WanGP process wall time (model load, denoise, decode, save),
NOT hot-model throughput; the model still reloads in each new process.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import v100_bar_video_smoke as base


MODE = "t2v"
WARM_CHECKPOINT = base.WRITABLE_CHECKPOINTS / base.MODEL_FILES[MODE]
BASELINE_SUMMARY = (base.CACHE/"results"
    /"hunyuan15-fp16-smoke-t2v-43929714398-attempt1"/"t2v"/"summary.json")


def validate_clip(clip: Path) -> dict:
    """Actually decode every frame rather than relying on file size or exit code."""
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg not available to validate MP4")
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-nostdin", "-v", "error",
         "-i", str(clip), "-map", "0:v:0", "-f", "framemd5", "-"],
        capture_output=True, text=True, timeout=240)
    if result.returncode:
        raise RuntimeError("MP4 frame decode failed: " + result.stderr[-2500:])
    decoded = sum(bool(row.strip()) and not row.startswith("#")
                  for row in result.stdout.splitlines())
    if decoded != base.FRAMES:
        raise RuntimeError(f"Decoded {decoded} frames, expected {base.FRAMES}")
    return {"clip": str(clip), "decoded_frames": decoded, "decode_pass": True,
            "size_bytes": clip.stat().st_size}


def main():
    base.gpu_preflight()
    if not WARM_CHECKPOINT.is_file() or WARM_CHECKPOINT.stat().st_size < 15_000_000_000:
        raise RuntimeError(f"Prior-run cached full T2V FP16 checkpoint missing: {WARM_CHECKPOINT}")
    checkpoint_bytes = WARM_CHECKPOINT.stat().st_size
    print("HUNYUAN15_CACHE_HIT", json.dumps({
        "checkpoint": str(WARM_CHECKPOINT), "bytes": checkpoint_bytes,
        "method": "fully-offline-independent-process"}), flush=True)

    # Guarantee the original test's local model code never downloads anything.
    # Offline mode is explicitly understood by WanGP's built-in downloader.
    underlying = base.run_command
    def offline_command(args, log_path, env, timeout):
        args = list(args)
        if "wgp.py" in args:
            args.insert(args.index("wgp.py")+1, "--offline")
        env = dict(env)
        env.update({"HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1",
                    "HF_DATASETS_OFFLINE":"1"})
        return underlying(args, log_path, env, timeout)
    base.run_command = offline_command

    report = base.run_mode(MODE)
    current_log = (base.OUT/MODE/"logs"/"generation.log").read_text(errors="replace")
    if not ("EXPERIMENTAL Hunyuan 1.5 native FP16 enabled" in current_log and
            "Hunyuan 1.5 experimental loader: unpacked 216 Comfy QKV tensors" in current_log):
        raise RuntimeError("Cached T2V did not exercise experimental native FP16 loader")
    if any(marker in current_log for marker in (
        "Downloading ", "Download URL:", "Unable to Download", "Downloading model", "Downloading Model"
    )):
        raise RuntimeError("Offline cache miss detected in generated video log")
    if WARM_CHECKPOINT.stat().st_size != checkpoint_bytes:
        raise RuntimeError("Checkpoint unexpectedly modified during offline run")

    clips = [Path(x["name"]) for x in report["videos"]]
    validation = [validate_clip(clip) for clip in clips]
    baseline = None
    if BASELINE_SUMMARY.is_file():
        previous = json.loads(BASELINE_SUMMARY.read_text())
        baseline = previous.get("run", {}).get("seconds")
    comparison = {
        "status": "OFFLINE_VIDEO_DECODE_PASS",
        "new_generation_wall_seconds": report["run"]["seconds"],
        "first_t2v_run_including_download_seconds": baseline,
        "saved_seconds_vs_download_run": round(baseline-report["run"]["seconds"],1)
            if baseline is not None else None,
        "model": report["model_type"], "precision": "native-fp16-unquantized",
        "steps": 30, "frames": base.FRAMES, "resolution": base.RESOLUTION,
        "seed": base.SEED, "checkpoint": str(WARM_CHECKPOINT),
        "checkpoint_bytes": checkpoint_bytes, "download_prohibited": True,
        "first_run_source": "https://github.com/mylordmonkeyman/WanGP/actions/runs/43929714398",
        "timing_scope": "new WanGP Python process start to process exit; includes load + denoise + decode + MP4 save, excludes dry-run",
        "not_hot_model_benchmark": True,
        "not_directly_comparable_to_8_step_i2v": True,
        "videos": validation,
    }
    result_file = base.OUT/MODE/"steady_state_comparison.json"
    result_file.write_text(json.dumps(comparison,indent=2)+"\n")
    print("HUNYUAN15_T2V_CACHED_STEADY_STATE_PASS "+json.dumps(comparison,sort_keys=True),flush=True)


if __name__ == "__main__":
    main()
