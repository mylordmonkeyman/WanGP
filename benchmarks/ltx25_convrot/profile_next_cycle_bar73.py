#!/usr/bin/env python3
"""One-off baseline bottleneck study for validated main-equivalent 73f bar T2V/I2V.

Normal runs give production wall time and decoded hashes. Separate profiled
runs enable preexisting phase/CUPTI attribution; their timings are NOT speed
comparisons. This does not change a production hot path or cache weights.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import traceback

import selective_fp16_t2v_smoke as base
import t2v73_241_original_stock_fourway as four
from profile_gemm_t2v_i2v import PROMPT, INPUT_IMAGE, check_png
from verify_cleanup_bar73_bitexact import KNOWN

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("LTX25_NEXT_PROFILE_ROOT",
    "/cache/ltx25/results/next-profile-" + os.environ.get("GITHUB_RUN_ID", "manual"))).resolve()
SHA = "523cb6f8cf3aeca2daa38a61a3d7e81c01c61c43"
MODES = ("t2v", "i2v")
CHECKPOINT = base.CHECKPOINTS / "ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fd:
        for chunk in iter(lambda: fd.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def phase_summary(report: dict) -> dict:
    stage_forward = report.get("transformer_by_phase", {})
    return {
        "status": report.get("status"),
        "total_sync_instrumented_phase_s": report.get("phase_total_seconds"),
        "phases_sync_instrumented": [
            {"phase": item["phase"], "wall_s": item["wall_seconds"]}
            for item in report.get("phases", [])
        ],
        "transformer_cuda_events": stage_forward,
        "sampled_forward_kernel_attribution": [{
            "phase": item.get("phase"),
            "kernel_gpu_time_ms_sum": item.get("gpu_time_ms_sum"),
            "kernel_families_gpu_ms": item.get("gpu_kernel_family_time_ms"),
            "gpu_memcpy_or_memset_events": item.get("gpu_event_category_counts"),
            "scope_gpu_ms": item.get("attribution_gpu_time_ms"),
            "top_gpu_kernels": item.get("top_gpu_kernels", [])[:35],
        } for item in report.get("chrome_trace_kernel_attribution", [])],
        "operator_samples": [{
            "phase": item.get("phase"),
            "attention_self_device_ms": item.get("attention_self_device_ms"),
            "copy_related_self_device_ms": item.get("copy_related_self_device_ms"),
            "op_self_device_sum_ms": item.get("op_self_device_sum_ms"),
            "top_ops": item.get("top_device_operators", [])[:20],
        } for item in report.get("sampled_operator_forwards", [])],
        "errors": report.get("errors", []),
        "caveat": "Profiled phase durations contain deliberate CUDA boundaries; GPU kernel sums may overlap and cannot be subtracted from wall-clock time.",
    }


def run(mode: str, instrumented: bool, template: dict):
    name = f"{mode}-{'sampled' if instrumented else 'normal'}"
    settings = four.settings(73)
    if mode == "i2v":
        settings.update(image_start=str(INPUT_IMAGE), image_prompt_type="S")
    settings_path = OUT / "settings" / f"{name}.json"
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    reference_settings = Path(
        f"/cache/ltx25/results/bar73-stage-shape-{mode}-43683731705-attempt1/input.json")
    if reference_settings.exists():
        require(json.loads(reference_settings.read_text()) == settings,
                f"{name}: mismatched historical bar workload")
    cfg, target = base.config_for(name, template)
    env = four.optimized_env("stage2_fp16", name)
    for key in ("WAN2GP_LTX25_E_TEST_4096X2048",
                "WAN2GP_LTX25_E_TEST_4096X2048_PHASES",
                "WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE1",
                "WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE2",
                "WAN2GP_LTX25_DEFAULT_ON",
                "WAN2GP_LTX25_PIPELINE_PROFILE",
                "WAN2GP_LTX25_PROFILE_STAGES",
                "LTX25_PIPELINE_TRACE_ATTRIBUTION"):
        env.pop(key, None)
    if instrumented:
        env.update({
            "WAN2GP_LTX25_PIPELINE_PROFILE": "1",
            "LTX25_PIPELINE_OP_SAMPLE": "1",
            "LTX25_PIPELINE_GEMM_CENSUS": "1",
            "LTX25_PIPELINE_TRACE_ATTRIBUTION": "1",
            "LTX25_PIPELINE_TRACE_RETAIN": "0",
            "LTX25_PIPELINE_PROFILE_PATH": str(OUT / "results" / (name + "-pipeline.json")),
        })
    cmd = [base.PY, "wgp.py", "--offline", "--config", str(cfg),
           "--process", str(settings_path), "--output-dir", str(target),
           "--profile", "0", "--attention", "sdpa",
           "--no-prevent-power-throttling", "--verbose", "1"]
    dry = base.command_run(cmd + ["--dry-run"], env=env, timeout=300,
                           log=OUT / "logs" / (name + "-dryrun.log"))
    require(dry["returncode"] == 0, f"{name} dry run failed")
    base.gpu_free()
    print("NEXT_PROFILE_BEGIN", name, flush=True)
    timed = base.command_run(cmd, env=env, timeout=3600,
                             log=OUT / "logs" / (name + "-generation.log"),
                             check=False)
    require(timed["returncode"] == 0,
            f"{name}: generation failed: {timed['log_tail']}")
    log = Path(timed["log"]).read_text(errors="replace")
    candidate = four.read_candidate_stats(log)
    require(candidate and candidate.get("mode") == "fp32rot"
            and candidate.get("rotation_mode") == "radix4"
            and candidate.get("used") == 27360
            and candidate.get("active_4096x2048_phases") == ["stage1", "stage2"],
            f"{name}: Candidate E coverage changed: {candidate}")
    fp16_file = OUT / "results" / (name + "-fp16.json")
    fp16 = four.verify_fp16(fp16_file, "stage2_fp16")
    require(fp16.get("persistent_fp16_weight_copies") == 0, "Persistent FP16 weights")
    require(fp16.get("guard_fallbacks") == 0, f"{name}: FP16 guard fallback")
    clips = [path for path in target.rglob("*.mp4") if path.stat().st_size > 1024]
    require(len(clips) == 1, f"{name}: wanted exactly one MP4, found {len(clips)}")
    hashes = four.probe_video(clips[0], 73)
    matches = {
        "decoded_video_matches_main": hashes["decoded_video_sha256"] == KNOWN[mode]["video"],
        "pcm_audio_matches_main": hashes["decoded_audio_sha256"] == KNOWN[mode]["audio"],
    }
    if not instrumented:
        require(all(matches.values()), f"{name}: changed main decoded outputs: {matches}")
    record = {
        "name": name, "mode": mode, "instrumented": instrumented,
        "wall_s_not_comparable_if_instrumented": timed["wall_seconds"],
        "settings_sha256": digest(settings_path),
        "decoded": hashes, **matches,
        "candidate_e_calls": candidate["used"],
        "candidate_e_per_stage_4096x2048": candidate.get("shape_4096x2048_by_phase"),
        "fp16": fp16,
        "video_path": str(clips[0]),
    }
    if instrumented:
        pipeline_path = OUT / "results" / (name + "-pipeline.json")
        require(pipeline_path.is_file(), f"{name}: missing pipeline report")
        pipeline = json.loads(pipeline_path.read_text())
        record["phase_profile"] = phase_summary(pipeline)
        require(len(pipeline.get("chrome_trace_kernel_attribution", [])) == 2,
                f"{name}: sampled CUDA kernel attribution unavailable: {pipeline.get('errors')}")
        require(pipeline.get("status") == "PASS",
                f"{name}: pipeline profile partial: {pipeline.get('errors')}")
    (OUT / "results" / (name + "-summary.json")).write_text(
        json.dumps(record, indent=2) + "\n")
    print("NEXT_PROFILE_CASE_PASS", json.dumps({
        "name": name, "wall_seconds": timed["wall_seconds"],
        "instrumented": instrumented, "hashes": matches,
        "candidate_e_calls": candidate["used"],
        "phase_status": record.get("phase_profile", {}).get("status"),
    }, sort_keys=True), flush=True)
    return record


def main():
    require(not OUT.exists(), f"Output root already exists: {OUT}")
    for directory in ("settings", "config", "videos", "logs", "results"):
        (OUT / directory).mkdir(parents=True, exist_ok=False)
    check_png()
    require(CHECKPOINT.is_file() and CHECKPOINT.stat().st_size > 1_000_000_000,
            "Distilled ConvRot checkpoint missing")
    base.gpu_free()
    base.OUT = OUT
    base.EXTENSION = OUT / "libltx25_cublas_e_sm70.so"
    base.FRAMES, base.STEPS, base.SEED = 73, 8, 20261112
    base.RESOLUTION, base.PROMPT = "1024x576", PROMPT
    four.OUT = OUT
    (OUT / "manifest.json").write_text(json.dumps({
        "baseline_main_sha": SHA, "branch_sha": os.getenv("GITHUB_SHA", "manual"),
        "model": str(CHECKPOINT), "model_size_bytes": CHECKPOINT.stat().st_size,
        "original_bar_image": str(INPUT_IMAGE), "image_sha256": digest(INPUT_IMAGE),
        "seed": 20261112, "steps": 8, "resolution": "1024x576", "frames": 73,
        "mmgp_profile": 0, "attention": "sdpa", "stage2_fp16": True,
        "candidate_e_4096x2048_both_stages": True,
        "timing_contract": "normal subprocess wall only; instrumented runs for attribution only",
    }, indent=2) + "\n")
    base.command_run(["nvcc", "-O3", "-std=c++17", "-arch=sm_70",
                      "-shared", "-Xcompiler", "-fPIC",
                      "benchmarks/ltx25_convrot/ltx25_cublas_e.cu",
                      "-lcublas", "-o", str(base.EXTENSION)],
                     timeout=300, log=OUT / "logs" / "compile.log")
    require(base.EXTENSION.is_file(), "Native Candidate E extension not built")
    (OUT / "input.json").write_text(json.dumps(four.settings(73), indent=2) + "\n")
    template = base.initialize_config()
    results = []
    # Production measurements first; trace runs separately to avoid contamination.
    for instrumented in (False, True):
        for mode in MODES:
            row = run(mode, instrumented, template)
            results.append(row)
            (OUT / "partial.json").write_text(json.dumps(results, indent=2) + "\n")
    (OUT / "summary.json").write_text(json.dumps({
        "status": "PROFILED", "baseline_sha": SHA,
        "unprofiled_wall_times": {r["mode"]: r["wall_s_not_comparable_if_instrumented"]
                                 for r in results if not r["instrumented"]},
        "results": results,
        "limitations": [
            "One unprofiled run per mode is a sanity baseline, not a speed distribution",
            "Profiling stages synchronizes phase boundaries and one forward in each stage",
            "CUDA kernel durations are summed within sampled forwards, not full-run exclusive time",
            "Transfer/kernel overlap and recovery must be validated before numeric speed predictions",
        ],
    }, indent=2) + "\n")
    print("LTX25_NEXT_PROFILE_PASS", str(OUT), flush=True)


if __name__ == "__main__":
    main()
