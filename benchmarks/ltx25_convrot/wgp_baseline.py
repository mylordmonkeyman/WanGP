#!/usr/bin/env python3
"""Consolidated first LTX-2.5 WanGP CLI dry-run + baseline attempt.

This script ONLY runs on the isolated self-hosted runner. All configuration,
model checkpoints and outputs live under /cache/ltx25. The original WanGP
install and model directories on the host are never modified.

Model startup may fail for package compatibility or actual model resources;
these are reported, not hidden. No candidate acceleration is enabled.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("/cache/ltx25/results")
BASE = Path("/cache/ltx25")
PY = os.environ.get("LTX25_PYTHON", "/opt/venvs/ltx25-cu128/bin/python")
CFG = BASE / "config-baseline"
VIDEOS = BASE / "videos" / "r0_baseline"
PARAMS = BASE / "baseline_49frame_settings.json"
SRC = BASE / "checkpoints"
MODEL_FILE = SRC / "ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors"
TEXT_FILE = SRC / "gemma4-12b-ltx-v1/gemma4-12b-ltx-v1_int8_convrot.safetensors"

# One baseline clip only, fixed prompt/seed for a later controlled comparison.
SETTINGS = {
    "model_type": "ltx2_25_22B_distilled",
    "prompt": "A sunlit coastal boardwalk in the early morning. A small red sailboat slowly crosses the blue harbor, gentle waves and distant palm trees. Smooth steady cinematic camera, natural detail, no text.",
    "seed": 20261112,
    "num_inference_steps": 8,
    "video_length": 49,
    "resolution": "576x320",
    "audio_prompt_type": "",
    "video_prompt_type": "",
    "image_mode": 0,
    "spatial_upsampling": "",
    "temporal_upsampling": "",
}


def run(cmd, timeout, log_path):
    began = time.monotonic()
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["WAN2GP_INT8_CONVROT_DTYPE_DEBUG"] = "1"
    env["GRADIO_ANALYTICS_ENABLED"] = "False"
    env["OMP_NUM_THREADS"] = "8"
    try:
        with log_path.open("w", encoding="utf-8") as fd:
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=fd,
                                  stderr=subprocess.STDOUT, timeout=timeout)
        code = proc.returncode
    except subprocess.TimeoutExpired:
        code = 124
        with log_path.open("a") as fd:
            fd.write(f"\n[PHASE0] Command exceeded {timeout} seconds\n")
    except Exception as err:
        code = -1
        with log_path.open("a") as fd:
            fd.write(f"\n[PHASE0] Invocation failed: {type(err).__name__}: {err}\n")
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
    return {"returncode": code, "elapsed_seconds": round(time.monotonic()-began, 1),
            "log_path": str(log_path), "log_tail": tail}


def gpu_busy():
    # GPU sharing invalidates any performance claim. Skip generation if another
    # host process owns GPU memory. nvidia-smi is advisory, not an ownership lock.
    proc = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
         "--format=csv,noheader"], capture_output=True, text=True, timeout=20
    )
    if proc.returncode:
        return {"busy": True, "reason":"nvidia-smi query failed", "detail": proc.stderr[-300:]}
    owners = [x.strip() for x in proc.stdout.splitlines() if x.strip()
              and not x.strip().startswith("No running processes")]
    return {"busy": bool(owners), "reason":"other GPU compute processes" if owners else "",
            "entries": owners[:6]}


def cmd(dry_run):
    args = [
        PY, "wgp.py", "--offline", "--config", str(CFG),
        "--process", str(PARAMS), "--output-dir", str(VIDEOS),
        "--profile", "0", "--attention", "sdpa",
        "--no-prevent-power-throttling", "--verbose", "1",
    ]
    if dry_run:
        args.append("--dry-run")
    return args


def main():
    sha = os.environ.get("GITHUB_SHA", "local")
    OUT.mkdir(parents=True, exist_ok=True)
    CFG.mkdir(parents=True, exist_ok=True)
    VIDEOS.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(json.dumps(SETTINGS, indent=2) + "\n")
    bootstrap_file = OUT / f"runtime_bootstrap_{sha[:12]}.json"
    bootstrap = json.loads(bootstrap_file.read_text()) if bootstrap_file.exists() else {}
    report = {
        "sha": sha, "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "kind": "Unmodified WanGP end-to-end R0 CLI baseline",
        "model_type": SETTINGS["model_type"],
        "seed": SETTINGS["seed"], "frames": SETTINGS["video_length"],
        "resolution": SETTINGS["resolution"], "num_inference_steps": SETTINGS["num_inference_steps"],
        "checkpoint_visible": MODEL_FILE.is_file(),
        "gemma_checkpoint_visible": TEXT_FILE.is_file(),
        "candidate_e_enabled": False,
        "video_dir": str(VIDEOS), "config_dir": str(CFG),
        "runtime_bootstrap_status": bootstrap.get("status", "not_run"),
    }
    result_path = OUT / f"runtime_baseline_{sha[:12]}.json"
    try:
        if bootstrap.get("status") != "PASS":
            report.update(status="BLOCKED", blocker="Full LTX-2.5 import prerequisites not yet satisfied")
            return
        if not MODEL_FILE.is_file() or not TEXT_FILE.is_file():
            report.update(status="BLOCKED", blocker="Transformer or text encoder checkpoint absent")
            return

        # A fresh installation needs WanGP to generate its default config.
        # Subsequent test runs reuse the isolated config and need only one
        # configured dry-run, avoiding an unnecessary second CLI startup.
        cfg = CFG / "wgp_config.json"
        if not cfg.is_file():
            first = run(cmd(True), timeout=300, log_path=OUT / f"cli_first_{sha[:12]}.log")
            report["initial_cli_dryrun"] = first
        if not cfg.is_file():
            report.update(status="BLOCKED", blocker="WanGP did not initialize isolated config")
            return
        conf = json.loads(cfg.read_text())
        conf.update({
            "checkpoints_paths": [str(SRC)],
            "transformer_quantization": "int8",
            "text_encoder_quantization": "int8",
            "int8_kernels": "disabled",
            "fp8_kernels": "pytorch",
            "transformer_types": ["ltx2_25_22B_distilled"],
            "profile": 0, "video_profile": 0,
            "attention_mode": "sdpa",
            "compile": "disabled",
            "save_path": str(VIDEOS),
            "image_save_path": str(VIDEOS),
            "audio_save_path": str(VIDEOS),
            "prevent_power_throttling": False,
        })
        cfg.write_text(json.dumps(conf, indent=2) + "\n")
        second = run(cmd(True), timeout=300, log_path=OUT / f"cli_dryrun_{sha[:12]}.log")
        report["configured_cli_dryrun"] = second
        if second["returncode"]:
            report.update(status="DRYRUN_BLOCKED", blocker="WanGP settings validation returned nonzero")
            return

        availability = gpu_busy()
        report["gpu_preflight"] = availability
        if availability["busy"]:
            report.update(status="GPU_BUSY", blocker="Other GPU workload detected; baseline generation skipped")
            return

        # Permit one 49-frame reference video; never loop or repeat on failure.
        # Require a *new* output from this invocation, not a stale clip from a
        # previous attempt in the same persistent results directory.
        before = {p.resolve() for p in VIDEOS.rglob("*.mp4")}
        result = run(cmd(False), timeout=3600, log_path=OUT / f"baseline_generation_{sha[:12]}.log")
        report["generation"] = result
        files = [p for p in VIDEOS.rglob("*.mp4") if p.resolve() not in before]
        report["videos"] = [
            {"path": str(p), "filename": p.name, "size_bytes": p.stat().st_size}
            for p in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:6]
        ]
        report["status"] = "PASS" if result["returncode"] == 0 and report["videos"] else "GENERATION_BLOCKED"
    except Exception as err:
        report.update(status="ERROR", error_type=type(err).__name__, error=str(err)[:800])
    finally:
        result_path.write_text(json.dumps(report, indent=2) + "\n")
        print("LTX25_BASELINE_STATUS " + report.get("status", "UNKNOWN"), flush=True)
        print("LTX25_BASELINE_SUMMARY " + json.dumps({
            k:v for k,v in report.items() if k not in ("initial_cli_dryrun", "configured_cli_dryrun","generation")
        })[:8000], flush=True)
        for key in ("initial_cli_dryrun", "configured_cli_dryrun", "generation"):
            if key in report:
                print("LTX25_BASELINE_" + key.upper() + " " + json.dumps(report[key])[-6000:], flush=True)


if __name__=="__main__":
    main()
