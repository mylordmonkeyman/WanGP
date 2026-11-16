#!/usr/bin/env python3
"""End-to-end Hunyuan 1.5 V100 FP16 bar-scene experiment.

Uses isolated per-run WanGP configuration; never edits the production UI config.
All logs, input settings, metadata and (when successful) MP4s are retained under
/cache/ltx25/results/hunyuan15-fp16-smoke-<runid>[-attemptN].
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path("/cache/ltx25")
MODEL_DIR = CACHE / "models"
EXTRA_CHECKPOINTS = Path("/opt/ltx25/checkpoints")
# Runner user cannot write /cache/ltx25/models (host-owned). WanGP writes
# dependency downloads to its FIRST checkpoints_paths entry: use shared
# writable results storage while preserving the existing user checkpoint.
WRITABLE_CHECKPOINTS = CACHE / "results" / "hunyuan15-fp16-downloaded-checkpoints"
IMAGE = ROOT / "benchmarks/ltx25_convrot/inputs/profile_i2v_1024x576.png.png"
PYTHON = os.environ.get("HUNYUAN15_PYTHON", "/opt/venvs/ltx25-cu128/bin/python")
OUT = Path(os.environ.get("HUNYUAN15_TEST_OUT", str(CACHE / "results" /
    ("hunyuan15-fp16-smoke-" + os.environ.get("GITHUB_RUN_ID", "manual")
     + "-attempt" + os.environ.get("GITHUB_RUN_ATTEMPT", "1")))))
MODEL_FILES = {
    "i2v": "hunyuanvideo1.5_480p_i2v_step_distilled_fp16.safetensors",
    "t2v": "hunyuanvideo1.5_480p_t2v_fp16.safetensors",
}
MODEL_TYPES = {
    "i2v": "hunyuan15_v100_i2v_480_step_fp16_experiment",
    "t2v": "hunyuan15_v100_t2v_480_fp16_experiment",
}
PROMPT = (
    "Photorealistic bustling upscale cocktail bar. Patrons engage in lively "
    "conversations, laughing, gesturing, sipping drinks, and moving naturally "
    "through the crowd. Bartenders shake cocktails, pour drinks, reach for "
    "bottles, and serve customers. Subtle, independent movements throughout "
    "the scene. Warm lighting and realistic reflections. Static camera, "
    "preserve original composition and character appearances.\n"
    "Audio: Continuous lively bar ambience with overlapping indistinct "
    "conversations, laughter, clinking glasses, cocktail shakers, pouring "
    "drinks, and soft background music. Natural, immersive crowd noise "
    "without intelligible dialogue."
)
SEED = 20261112
FRAMES = 33
RESOLUTION = "832x480"


def hash_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1048576), b""):
            digest.update(part)
    return digest.hexdigest()


def gpu_preflight():
    import struct
    if not IMAGE.is_file():
        raise RuntimeError(f"Prior LTX bar image absent: {IMAGE}")
    header = IMAGE.read_bytes()[:24]
    if header[:8] != b"\x89PNG\r\n\x1a\n" or struct.unpack(">II", header[16:24]) != (1024, 576):
        raise RuntimeError("Wrong bar input image (expected 1024x576 PNG)")
    p = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
        "--format=csv,noheader"], text=True, capture_output=True, check=True)
    print("HUNYUAN15_V100:", p.stdout.strip(), flush=True)
    if "V100" not in p.stdout:
        raise RuntimeError("Not running on V100 runner")
    p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader"], text=True, capture_output=True, check=True)
    apps = [x for x in p.stdout.splitlines() if re.search(r"^\s*\d+\s*,", x)]
    if apps:
        raise RuntimeError("V100 occupied, preserve ongoing work: " + repr(apps))
    if not Path(PYTHON).is_file():
        raise RuntimeError(f"Prebaked CUDA Python missing: {PYTHON}")
    print("BAR_IMAGE_SHA256:", hash_file(IMAGE), flush=True)


def select_template():
    for candidate in (
        CACHE/"config-e_hadamard_1024/wgp_config.json",
        CACHE/"config-e_default_1024/wgp_config.json",
        CACHE/"config-e_quality_1024/wgp_config.json",
    ):
        if candidate.is_file():
            print("CONFIG_TEMPLATE:", candidate, flush=True)
            return json.loads(candidate.read_text())
    print("CONFIG_TEMPLATE: none; using minimal isolated config", flush=True)
    return {}


def make_settings(mode):
    settings = {
        "model_type": MODEL_TYPES[mode],
        "prompt": PROMPT,
        "negative_prompt": "",
        "seed": SEED,
        "resolution": RESOLUTION,
        "video_length": FRAMES,
        "num_inference_steps": 8 if mode == "i2v" else 30,
        "guidance_scale": 1.0 if mode == "i2v" else 6.0,
        "flow_shift": 5.0,
        "image_mode": 0,
        "activated_loras": [],
        "loras_multipliers": "",
        "skip_steps_cache_type": "",
        "image_prompt_type": "S" if mode == "i2v" else "",
        "video_prompt_type": "",
        "spatial_upsampling": "",
        "temporal_upsampling": "",
    }
    if mode == "i2v":
        settings["image_start"] = str(IMAGE)
    return settings


def run_command(cmd, log_path: Path, env: dict, timeout: int):
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fd:
        process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   bufsize=1, errors="replace", start_new_session=True)
        import selectors, signal
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            while True:
                if time.monotonic() - started >= timeout:
                    os.killpg(process.pid, signal.SIGTERM)
                    try: process.wait(timeout=15)
                    except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL)
                    print(f"TIMEOUT after {timeout}s", file=fd, flush=True)
                    return {"returncode": 124, "seconds": round(time.monotonic()-started, 1)}
                ready = selector.select(timeout=2)
                for key, _ in ready:
                    line = key.fileobj.readline()
                    if line:
                        fd.write(line)
                        fd.flush()
                        print(line, end="", flush=True)
                if process.poll() is not None:
                    for line in process.stdout:
                        fd.write(line); print(line, end="", flush=True)
                    break
            return {"returncode": process.wait(), "seconds": round(time.monotonic()-started, 1)}
        finally:
            selector.close()


def run_mode(mode):
    mode_dir = OUT/mode
    for name in ("config", "videos", "logs", "clips"):
        (mode_dir/name).mkdir(parents=True, exist_ok=True)
    WRITABLE_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    if not os.access(WRITABLE_CHECKPOINTS, os.W_OK):
        # A staged, read-only source cache is valid for explicitly offline tests.
        # Other benchmarks still require a writable cache for dependencies.
        if not (os.environ.get("HUNYUAN15_ALLOW_READONLY_CACHE") == "1"
                and os.environ.get("HF_HUB_OFFLINE") == "1"):
            raise RuntimeError(f"No write access to download cache: {WRITABLE_CHECKPOINTS}")
        print(f"HUNYUAN15_READONLY_OFFLINE_CACHE {WRITABLE_CHECKPOINTS}", flush=True)
    template = copy.deepcopy(select_template())
    template.update({
        "checkpoints_paths": [str(WRITABLE_CHECKPOINTS), str(MODEL_DIR), str(EXTRA_CHECKPOINTS)],
        "transformer_quantization": "bf16",  # WanGP selection token for unquantized
        "text_encoder_quantization": "int8",
        "transformer_types": [MODEL_TYPES[mode]],
        "profile": 0, "video_profile": 0,
        "attention_mode": "sdpa",
        "compile": "disabled",
        "save_path": str(mode_dir/"videos"),
        "image_save_path": str(mode_dir/"videos"),
        "audio_save_path": str(mode_dir/"videos"),
        "prevent_power_throttling": False,
        "mixed_precision": "0",
    })
    config_file = mode_dir/"config"/"wgp_config.json"
    config_file.write_text(json.dumps(template, indent=2) + "\n")
    settings_file = mode_dir/"settings.json"
    settings_file.write_text(json.dumps(make_settings(mode), indent=2) + "\n")
    path = MODEL_DIR / MODEL_FILES[mode]
    if not path.is_file() and (WRITABLE_CHECKPOINTS / MODEL_FILES[mode]).is_file():
        path = WRITABLE_CHECKPOINTS / MODEL_FILES[mode]
    print(f"HUNYUAN15_CHECKPOINT mode={mode} path={path} exists={path.is_file()}",flush=True)
    if mode == "i2v" and not path.is_file():
        raise RuntimeError(f"User-provided I2V FP16 file not visible: {path}")
    env = dict(os.environ)
    for name in list(env):
        if name.startswith(("WAN2GP_LTX25_", "LTX25_FP16_GEMM_", "LTX25_CUBLAS_", "LTX25_GEMM_", "LTX25_PIPELINE_")):
            env.pop(name, None)
    env.update({
        "PYTHONPATH": str(ROOT),
        "GRADIO_ANALYTICS_ENABLED": "False",
        "OMP_NUM_THREADS": "8",
        "WAN2GP_LTX2_SKIP_BAKED_LORA_MIGRATION": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    })
    # No --offline: allow model ancillary files to download to the private
    # per-test checkpoint root if the runner has outbound internet.
    args = [PYTHON, "-u", "wgp.py",
        "--config", str(mode_dir/"config"), "--process", str(settings_file),
        "--output-dir", str(mode_dir/"videos"),
        "--profile", "0", "--attention", "sdpa", "--fp16",
        "--no-prevent-power-throttling", "--verbose", "1"]
    print("HUNYUAN15_DRYRUN_BEGIN mode="+mode, flush=True)
    dry = run_command(args+["--dry-run"], mode_dir/"logs"/"dryrun.log", env, 600)
    print("HUNYUAN15_DRYRUN_RESULT", json.dumps(dry), flush=True)
    if dry["returncode"] != 0:
        raise RuntimeError(f"{mode} dryrun failed rc={dry['returncode']}")
    print("HUNYUAN15_VIDEO_BEGIN mode="+mode,flush=True)
    run = run_command(args, mode_dir/"logs"/"generation.log", env, 7200)
    print("HUNYUAN15_VIDEO_RESULT", json.dumps(run), flush=True)
    clips = [p for p in (mode_dir/"videos").rglob("*.mp4") if p.stat().st_size > 1024]
    report = {"mode":mode, "status":"generated" if run["returncode"]==0 and clips else "failed",
              "source_sha":os.environ.get("GITHUB_SHA"), "run":run,
              "model_file":str(path), "model_bytes":path.stat().st_size if path.is_file() else None,
              "model_type":MODEL_TYPES[mode],"frames_requested":FRAMES,
              "resolution_requested":RESOLUTION,"seed":SEED,"prompt":PROMPT,
              "input_image":str(IMAGE) if mode=="i2v" else None,
              "image_sha256":hash_file(IMAGE) if mode=="i2v" else None,
              "videos":[]}
    for i,video in enumerate(clips,1):
        destination = mode_dir/"clips"/f"hunyuan15-{mode}-fp16-{i}.mp4"
        shutil.copy2(video,destination)
        report["videos"].append({"name":str(destination),"size":destination.stat().st_size,
                                 "sha256":hash_file(destination)})
    (mode_dir/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print("HUNYUAN15_CASE_SUMMARY "+json.dumps(report,sort_keys=True),flush=True)
    if report["status"]!="generated":
        raise RuntimeError(f"No {mode.upper()} MP4 was produced; see generation log")
    return report


def main():
    mode = os.environ.get("HUNYUAN15_TEST_MODE", "i2v")
    if mode not in ("i2v", "t2v"):
        raise RuntimeError("HUNYUAN15_TEST_MODE must be i2v or t2v")
    gpu_preflight()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        run_mode(mode)
    except Exception:
        traceback.print_exc()
        raise


if __name__=="__main__":
    main()
