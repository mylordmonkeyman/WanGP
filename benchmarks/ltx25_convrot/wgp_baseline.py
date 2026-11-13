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
# The pilot uses independent config and output paths but the *identical*
# model, prompt, seed, frame count, resolution, and R0 fallback settings.
VARIANT = os.environ.get("LTX25_BENCH_VARIANT", "r0_baseline")
VARIANTS = {
    "r0_baseline": ("576x320", False),
    "e_fp32rot": ("576x320", True),
    "r0_quality_1024": ("1024x576", False),
    "e_quality_1024": ("1024x576", True),
    "e_real_activation_audit_1024": ("1024x576", True),
    "e_profile_1024": ("1024x576", True),
    "e_workspace_1024": ("1024x576", True),
    "e_cast4_1024": ("1024x576", True),
    "e_combo_1024": ("1024x576", True),
    "e_default_1024": ("1024x576", True),
    "e_debug_fp32rot_samecfg_1024": ("1024x576", True),
}
if VARIANT not in VARIANTS:
    raise ValueError(f"Unsupported LTX25_BENCH_VARIANT: {VARIANT!r}")
RESOLUTION, IS_CANDIDATE = VARIANTS[VARIANT]
IS_REAL_AUDIT = VARIANT == "e_real_activation_audit_1024"
IS_STAGE_PROFILE = VARIANT == "e_profile_1024"
IS_WORKSPACE = VARIANT in ("e_workspace_1024", "e_combo_1024")
IS_CAST4 = VARIANT in ("e_cast4_1024", "e_combo_1024")
IS_DEFAULT = VARIANT == "e_default_1024"
IS_DEBUG_EXPLICIT = VARIANT == "e_debug_fp32rot_samecfg_1024"
if not IS_DEFAULT and IS_CANDIDATE != (os.environ.get("WAN2GP_LTX25_CUBLAS_E_MODE", "").lower() == "fp32rot"):
    raise ValueError("Candidate mode and selected benchmark variant disagree")
CFG = BASE / ("config-e_default_1024" if IS_DEBUG_EXPLICIT else f"config-{VARIANT}")
VIDEOS = BASE / "videos" / VARIANT
PARAMS = BASE / f"{VARIANT}_49frame_settings.json"
SRC = Path(os.environ.get("LTX25_CHECKPOINT_ROOT", "/opt/ltx25/checkpoints"))
IMAGE_MANIFEST = Path("/opt/ltx25/model-image-manifest.json")
MODEL_FILE = SRC / "ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors"
TEXT_FILE = SRC / "gemma4-12b-ltx-v1/gemma4-12b-ltx-v1_int8_convrot.safetensors"

# One baseline clip only, fixed prompt/seed for a later controlled comparison.
SETTINGS = {
    "model_type": "ltx2_25_22B_distilled",
    "prompt": "A sunlit coastal boardwalk in the early morning. A small red sailboat slowly crosses the blue harbor, gentle waves and distant palm trees. Smooth steady cinematic camera, natural detail, no text.",
    "seed": 20261112,
    "num_inference_steps": 8,
    "video_length": 49,
    "resolution": RESOLUTION,
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
    if IS_DEBUG_EXPLICIT:
        # Diagnostic: same e_default config and default cast/workspace, but
        # explicit fp32rot mode to isolate dispatch selection from settings.
        env["WAN2GP_LTX25_CUBLAS_E_MODE"] = "fp32rot"
        for name in ("WAN2GP_LTX25_E_CAST", "WAN2GP_LTX25_E_WORKSPACE",
                     "LTX25_CUBLAS_E_LIB"):
            env.pop(name, None)
    if IS_DEFAULT:
        # The default-selection gate MUST exercise *no environment opt-ins*,
        # including no explicit library path: force auto native build/cache.
        for name in ("WAN2GP_LTX25_CUBLAS_E_MODE", "WAN2GP_LTX25_E_CAST",
                     "WAN2GP_LTX25_E_WORKSPACE", "WAN2GP_LTX25_E_AUTO_BUILD",
                     "LTX25_CUBLAS_E_LIB"):
            env.pop(name, None)
    if IS_CAST4:
        env["WAN2GP_LTX25_E_CAST"] = "vector4"
    if IS_WORKSPACE:
        env["WAN2GP_LTX25_E_WORKSPACE"] = "1"
        env["LTX25_E_WORKSPACE_MAX_MIB"] = "96"
    if IS_STAGE_PROFILE:
        # Sparse CUDA-event instrumentation, always an explicitly separated run.
        env["WAN2GP_LTX25_PROFILE_STAGES"] = "1"
        env["LTX25_STAGE_PROFILE_PATH"] = str(
            OUT / f"stage_profile_{os.environ.get('GITHUB_SHA', 'local')[:12]}.json")
    if IS_REAL_AUDIT:
        # Numerically instrumented run, never compare its wall time with
        # an uninstrumented baseline. Keep the summary in isolated cache.
        env["WAN2GP_LTX25_REAL_ACTIVATION_AUDIT"] = "1"
        env["LTX25_REAL_AUDIT_PATH"] = str(
            OUT / f"real_activation_{os.environ.get('GITHUB_SHA', 'local')[:12]}.json")
    # This fixed T2V benchmark activates no LoRAs. Keep all 17 assets
    # immutable inside /opt/ltx25/checkpoints; don't try to move a baked
    # HDR LoRA into the checkout's loras/ directory during CLI startup.
    env["WAN2GP_LTX2_SKIP_BAKED_LORA_MIGRATION"] = "1"
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
    entries = []
    for owner in owners[:6]:
        pid = owner.split(",", 1)[0].strip()
        # Only executable name, never process args: avoid leaking tokens or
        # private strings embedded in another GPU user's command line.
        name = ""
        if pid.isdecimal():
            try:
                p = subprocess.run(["ps", "-p", pid, "-o", "comm="],
                                   capture_output=True, text=True, timeout=3)
                if p.returncode == 0:
                    name = p.stdout.strip()[:80]
            except Exception:
                pass
        entries.append({"gpu_entry": owner, "process_name": name or "unknown"})
    return {"busy": bool(owners), "reason":"other GPU compute processes" if owners else "",
            "entries": entries}


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
        "kind": "Experimental Candidate E fp32rot" if IS_CANDIDATE else "Unmodified WanGP R0",
        "variant": VARIANT,
        "model_type": SETTINGS["model_type"],
        "seed": SETTINGS["seed"], "frames": SETTINGS["video_length"],
        "resolution": SETTINGS["resolution"], "num_inference_steps": SETTINGS["num_inference_steps"],
        "checkpoint_visible": MODEL_FILE.is_file(),
        "gemma_checkpoint_visible": TEXT_FILE.is_file(),
        "candidate_e_enabled": IS_CANDIDATE,
        "video_dir": str(VIDEOS), "config_dir": str(CFG),
        "runtime_bootstrap_status": bootstrap.get("status", "not_run"),
    }
    result_kind = {"r0_baseline": "baseline", "e_fp32rot": "candidate",
                   "r0_quality_1024": "quality_r0", "e_quality_1024": "quality_e",
                   "e_real_activation_audit_1024": "quality_real_audit",
                   "e_profile_1024": "quality_stage_profile",
                   "e_workspace_1024": "quality_workspace",
                   "e_cast4_1024": "quality_cast4",
                   "e_combo_1024": "quality_combo",
                   "e_default_1024": "quality_default",
                   "e_debug_fp32rot_samecfg_1024": "quality_default_explicit"}[VARIANT]
    result_path = OUT / f"runtime_{result_kind}_{sha[:12]}.json"
    try:
        if bootstrap.get("status") != "PASS":
            report.update(status="BLOCKED", blocker="Full LTX-2.5 import prerequisites not yet satisfied")
            return
        if not IMAGE_MANIFEST.is_file():
            report.update(status="BLOCKED", blocker="Model assets are not baked in the runner Docker image")
            return
        if not MODEL_FILE.is_file() or not TEXT_FILE.is_file():
            report.update(status="BLOCKED", blocker="Image-baked transformer or text encoder checkpoint absent")
            return
        if VARIANT in ("e_quality_1024", "e_real_activation_audit_1024", "e_profile_1024", "e_workspace_1024", "e_cast4_1024", "e_combo_1024", "e_default_1024", "e_debug_fp32rot_samecfg_1024"):
            refdir = BASE / "videos" / "r0_quality_1024"
            if not refdir.is_dir() or not any(refdir.glob("*.mp4")):
                report.update(
                    status="BLOCKED",
                    blocker="Create the matched 1024x576 R0 video before testing Candidate E"
                )
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

        # Permit one 49-frame reference clip at the variant's exact resolution;
        # never loop or repeat on failure.
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
        if IS_CANDIDATE:
            # Adapter logs at interpreter exit. Never label a clip as
            # accelerated when eligible calls did not actually execute.
            import re
            matches = re.findall(r"\[LTX25_CUBLAS_E\] (\{[^\n]+\})",
                                 result["log_tail"])
            if matches:
                try:
                    report["candidate_counters"] = json.loads(matches[-1])
                except json.JSONDecodeError:
                    pass
            if report["status"] == "PASS" and report.get("candidate_counters", {}).get("used", 0) <= 0:
                report["status"] = "NO_CANDIDATE_CALLS"
                report["blocker"] = "A clip was generated but no ConvRot Candidate E calls were recorded"
        if IS_DEFAULT and report["status"] == "PASS":
            counters = report.get("candidate_counters", {})
            if (counters.get("mode") != "fp32rot" or
                    counters.get("auto_selected") is not True or
                    counters.get("cast_mode") != "vector4" or
                    counters.get("used", 0) <= 0 or
                    counters.get("auto_native_fallback", 0) != 0):
                report["status"] = "DEFAULT_AUTO_NOT_SELECTED"
                report["blocker"] = "Default automatic SM70 vector4 mode did not execute successfully"
        if IS_CAST4 and report["status"] == "PASS":
            cast_mode = report.get("candidate_counters", {}).get("cast_mode")
            if cast_mode != "vector4":
                report["status"] = "CAST4_NOT_SELECTED"
                report["blocker"] = f"Expected vector4 cast counters, got {cast_mode!r}"
        if IS_WORKSPACE or IS_DEFAULT or IS_DEBUG_EXPLICIT:
            import re
            # The workspace prints a bounded JSON summary at interpreter exit.
            ws_matches = re.findall(r"\[LTX25_WORKSPACE\] (\{[^\n]+\})",
                                     result["log_tail"])
            if ws_matches:
                report["workspace_stats"] = json.loads(ws_matches[-1])
            if report["status"] == "PASS" and report.get("workspace_stats", {}).get("acquired", 0) <= 0:
                report["status"] = "NO_WORKSPACE_REUSE"
                report["blocker"] = "Workspace opt-in requested but no scratch leases were used"
        if IS_STAGE_PROFILE:
            stage_path = OUT / f"stage_profile_{sha[:12]}.json"
            if stage_path.is_file():
                report["stage_profile"] = json.loads(stage_path.read_text())
                if report["status"] == "PASS" and report["stage_profile"].get("status") != "PASS":
                    report["status"] = "STAGE_PROFILE_FAILED"
                    report["blocker"] = "Sparse live-model CUDA-event stage profile incomplete"
            elif report["status"] == "PASS":
                report["status"] = "NO_STAGE_PROFILE"
                report["blocker"] = "CUDA-event stage profiler failed to write its summary"
        if IS_REAL_AUDIT:
            audit_path = OUT / f"real_activation_{sha[:12]}.json"
            if audit_path.is_file():
                report["real_activation_audit"] = json.loads(audit_path.read_text())
                if report["status"] == "PASS" and report["real_activation_audit"].get("status") != "PASS":
                    report["status"] = "NUMERICAL_GATE_FAILED"
                    report["blocker"] = "Real activation R0/R1/Candidate E accuracy gate failed; see published audit"
            elif report["status"] == "PASS":
                report["status"] = "NO_REAL_ACTIVATION_AUDIT"
                report["blocker"] = "No real activation audit summary was written by generation process"
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
        # A blocked CLI dry-run is not a successful workflow step. Publish
        # status in the finally block first, then propagate a nonzero exit.
        if report.get("status") != "PASS":
            raise SystemExit(1)


if __name__=="__main__":
    main()
