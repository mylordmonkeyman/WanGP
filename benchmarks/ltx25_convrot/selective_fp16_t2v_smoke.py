#!/usr/bin/env python3
"""Real V100 49-frame T2V smoke, numerical audit and fair matched speed ABBA.

Uses offline baked checkpoint and exactly one image-free LTX-2.5 T2V settings
JSON. Candidate E direct/fused/radix4/vector8 remains identical in all arms.

The A/B/A/B timing control is:
  A: BF16 sham dispatch, same TorchDispatchMode/shape filters as B.
  B: FP16 selective stage2/all, audit disabled.
Order A B B A to reduce drift; additionally run native unwrapped BF16.
All output review videos are retained and packaged as artifacts by workflow.
Audited comparisons are NEVER included in fair timing figures.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PY = os.getenv("LTX25_PYTHON", "/opt/venvs/ltx25-cu128/bin/python")
CACHE = Path("/cache/ltx25")
OUT = Path(os.getenv("LTX25_FP16_SMOKE_ROOT", str(CACHE / "results" / ("selective-fp16-smoke-" + os.getenv("GITHUB_RUN_ID", "manual")))))
CHECKPOINTS = Path("/opt/ltx25/checkpoints")
EXTENSION = OUT / "libltx25_cublas_e_sm70.so"
PROMPT = "A sunlit coastal boardwalk in the early morning. A small red sailboat slowly crosses the blue harbor, gentle waves and distant palm trees. Smooth steady cinematic camera, natural detail, no text."
STEPS = 8
FRAMES = 49
RESOLUTION = "1024x576"
SEED = 20261112
GROUPS = ("wide", "tall", "balanced")
STAGE2 = ",".join("stage2:" + g for g in GROUPS)
BOTH = ",".join(f"{stage}:{g}" for stage in ("stage1", "stage2") for g in GROUPS)
CASES = (
    ("native-a", "native", "", 0, "native control"),
    ("sham-a1", "sham", STAGE2, 0, "A1 fair BF16 dispatch"),
    ("fp16-b1", "fp16", STAGE2, 0, "B1 FP16 Stage2 all, timing"),
    ("fp16-b2", "fp16", STAGE2, 0, "B2 FP16 Stage2 all, timing"),
    ("sham-a2", "sham", STAGE2, 0, "A2 fair BF16 dispatch"),
    ("s2-wide", "fp16", "stage2:wide", 2, "Stage2 wide numerical/quality"),
    ("s2-tall", "fp16", "stage2:tall", 2, "Stage2 tall numerical/quality"),
    ("s2-balanced", "fp16", "stage2:balanced", 2, "Stage2 balanced numerical/quality"),
    ("s1-wide", "fp16", "stage1:wide", 2, "Stage1 wide numerical/quality"),
    ("both-all", "fp16", BOTH, 2, "Stage1+Stage2 combined numerical/quality"),
    ("s2-all-audit", "fp16", STAGE2, 2, "Stage2 all numerical/quality"),
    ("native-z", "native", "", 0, "native drift control"),
)


def ffmpeg_bin():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        result = shutil.which("ffmpeg")
        if not result:
            raise RuntimeError("ffmpeg and imageio_ffmpeg unavailable")
        return result


def command_run(cmd, env=None, timeout=1200, log=None, check=True):
    start = time.monotonic()
    if log:
        with log.open("w", encoding="utf8") as fd:
            try:
                p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fd,
                                   stderr=subprocess.STDOUT, timeout=timeout)
                code = p.returncode
            except subprocess.TimeoutExpired:
                code = 124
                fd.write(f"\nERROR timeout {timeout}s\n")
    else:
        p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
        code = p.returncode
    result = {"returncode": code, "wall_seconds": round(time.monotonic() - start, 3)}
    if log:
        result["log"] = str(log)
        result["log_tail"] = log.read_text(errors="replace")[-2500:]
    else:
        result["stdout"] = p.stdout[-2000:]
        result["stderr"] = p.stderr[-1000:]
    if check and code:
        raise RuntimeError(f"command exit={code}: {cmd[:3]}: {result}")
    return result


def gpu_free():
    p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                        "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
    if p.returncode:
        raise RuntimeError("nvidia-smi failed: " + p.stderr)
    active = [s for s in p.stdout.splitlines() if re.match(r"^\s*\d+\s*,", s)]
    if active:
        raise RuntimeError("V100 occupied: " + "; ".join(active))


def initialize_config():
    preferred = (CACHE / "config-e_hadamard_1024" / "wgp_config.json",
                 CACHE / "config-e_default_1024" / "wgp_config.json",
                 CACHE / "config-e_quality_1024" / "wgp_config.json")
    for candidate in preferred:
        if candidate.is_file():
            return json.loads(candidate.read_text())
    # No global changes: WanGP --dry-run initializes a fresh independent config.
    cfg_dir = OUT / "init-config"
    cfg_dir.mkdir(exist_ok=True)
    cmd = [PY, "wgp.py", "--offline", "--config", str(cfg_dir),
           "--process", str(OUT / "input.json"), "--output-dir", str(OUT / "videos"),
           "--profile", "0", "--attention", "sdpa", "--dry-run"]
    command_run(cmd, env=base_env(), timeout=300,
                log=OUT/"init-dryrun.log", check=False)
    path = cfg_dir / "wgp_config.json"
    if not path.is_file():
        raise RuntimeError(f"No WanGP config generated at {path}")
    return json.loads(path.read_text())


def base_env():
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("WAN2GP_LTX25_", "LTX25_FP16_GEMM_", "LTX25_CUBLAS_")):
            env.pop(key, None)
    env.update({
        "WAN2GP_LTX25_CUBLAS_E_MODE": "fp32rot",
        "WAN2GP_LTX25_HADAMARD": "radix4",
        "WAN2GP_LTX25_E_CAST": "vector8",
        "WAN2GP_LTX25_GEMM_TUNE": "off",
        "WAN2GP_LTX25_ROT_INPUT": "direct",
        "WAN2GP_LTX25_E_OUTPUT": "fused",
        "WAN2GP_LTX25_E_WORKSPACE": "1",
        "LTX25_E_WORKSPACE_MAX_MIB": "96",
        "LTX25_CUBLAS_E_LIB": str(EXTENSION),
        "WAN2GP_LTX2_SKIP_BAKED_LORA_MIGRATION": "1",
        "GRADIO_ANALYTICS_ENABLED": "False",
        "OMP_NUM_THREADS": "8",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    })
    return env


def config_for(case, template):
    base = dict(template)
    case_cfg = OUT / "config" / case
    case_cfg.mkdir(exist_ok=True, parents=True)
    target = OUT / "videos" / case
    target.mkdir(exist_ok=True, parents=True)
    base.update({
        "checkpoints_paths": [str(CHECKPOINTS)],
        "transformer_quantization": "int8",
        "text_encoder_quantization": "int8",
        "int8_kernels": "disabled",
        "fp8_kernels": "pytorch",
        "transformer_types": ["ltx2_25_22B_distilled"],
        "profile": 0, "video_profile": 0,
        "attention_mode": "sdpa",
        "compile": "disabled",
        "save_path": str(target),
        "image_save_path": str(target),
        "audio_save_path": str(target),
        "prevent_power_throttling": False,
    })
    (case_cfg/"wgp_config.json").write_text(json.dumps(base, indent=2) + "\n")
    return case_cfg, target


def environment(case, kind, groups, audit):
    env = base_env()
    if kind != "native":
        env.update({
            "WAN2GP_LTX25_FP16_GEMM": "1",
            "LTX25_FP16_GEMM_GROUPS": groups,
            "LTX25_FP16_GEMM_SHAM": "1" if kind == "sham" else "0",
            "LTX25_FP16_GEMM_GUARD": "first",
            "LTX25_FP16_GEMM_AUDIT_PER_SHAPE": str(audit),
            "LTX25_FP16_GEMM_STATS_PATH": str(OUT / "results" / f"{case}-fp16-stats.json"),
        })
    return env


def frames_hash(clip, ffmpeg):
    cmd = [ffmpeg, "-nostdin", "-v", "error", "-i", str(clip),
           "-map", "0:v:0", "-f", "framemd5", "-"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=140)
    if p.returncode:
        raise RuntimeError(f"Decoding video failed: {p.stderr[-400:]}")
    frames = [l.rsplit(",", 1)[-1].strip() for l in p.stdout.splitlines()
              if l.strip() and not l.startswith("#")]
    if len(frames) != FRAMES:
        raise RuntimeError(f"Decoded frames {len(frames)}, expected {FRAMES}: {clip}")
    return hashlib.sha256("|".join(frames).encode()).hexdigest()


def audio_hash(clip, ffmpeg):
    cmd = [ffmpeg, "-nostdin", "-v", "error", "-i", str(clip),
           "-map", "0:a:0", "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = p.stdout.read(1024*1024)
        if not chunk: break
        digest.update(chunk)
        size += len(chunk)
    errors = p.stderr.read()
    if p.wait(timeout=120) or size == 0:
        raise RuntimeError(f"No decoded audio: {clip}: {errors[-500:]}")
    return digest.hexdigest()


def psnr(clip, ref, ffmpeg):
    cmd = [ffmpeg, "-nostdin", "-v", "info",
           "-i", str(clip), "-i", str(ref), "-filter_complex",
           "[0:v][1:v]psnr", "-an", "-f", "null", "-"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    if p.returncode:
        return {"status": "ERROR", "detail": p.stderr[-650:]}
    m = re.search(r"average:([0-9.]+|inf)", p.stderr)
    return {"status": "PASS", "average_db": float(m.group(1)) if m else None}


def run_case(case, kind, groups, audit, template, ffmpeg, reference=None):
    env = environment(case, kind, groups, audit)
    case_cfg, target = config_for(case, template)
    cmd = [PY, "wgp.py", "--offline", "--config", str(case_cfg),
           "--process", str(OUT/"input.json"), "--output-dir", str(target),
           "--profile", "0", "--attention", "sdpa",
           "--no-prevent-power-throttling", "--verbose", "1"]
    report = {"variant": case, "kind": kind, "groups": groups, "audit_per_shape": audit,
              "settings_sha256": hashlib.sha256((OUT/"input.json").read_bytes()).hexdigest(),
              "started": dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        dry = command_run(cmd + ["--dry-run"], env=env, timeout=300,
                          log=OUT/"logs"/f"{case}-dryrun.log")
        report["dryrun"] = {"returncode": dry["returncode"], "wall_seconds": dry["wall_seconds"]}
        # No external user process may own the V100 between runs.
        gpu_free()
        generated = command_run(cmd, env=env, timeout=900,
                                log=OUT/"logs"/f"{case}-generation.log", check=False)
        report.update(generation=generated, wall_seconds=generated["wall_seconds"])
        if generated["returncode"]:
            raise RuntimeError("Generation exited nonzero: " + generated["log_tail"][-1800:])
        clips = [p for p in target.rglob("*.mp4") if p.stat().st_size > 0]
        if len(clips) != 1:
            raise RuntimeError(f"Expected one MP4, found {len(clips)}: {clips}")
        clip = clips[0]
        final = OUT/f"{case}.mp4"
        shutil.copy2(clip, final)
        report.update(clip=str(final), mp4_bytes=final.stat().st_size,
                      video_sha256=frames_hash(final, ffmpeg),
                      audio_sha256=audio_hash(final, ffmpeg))
        if reference:
            report.update(video_exact=report["video_sha256"] == reference["video_sha256"],
                          audio_exact=report["audio_sha256"] == reference["audio_sha256"],
                          video_psnr=psnr(final, Path(reference["clip"]), ffmpeg))
        if kind != "native":
            stats_path = OUT/"results"/f"{case}-fp16-stats.json"
            if not stats_path.is_file():
                raise RuntimeError("Missing on-exit selective GEMM counters")
            stats = json.loads(stats_path.read_text())
            report["converted_mm"] = int(stats.get("stats", {}).get("converted_mm", 0))
            report["converted_addmm"] = int(stats.get("stats", {}).get("converted_addmm", 0))
            report["sham_native"] = int(stats.get("stats", {}).get("sham_native", 0))
            report["numerical_sample_count"] = int(stats.get("numerical_sample_count", 0))
            samples = stats.get("numerical_samples", [])
            report["worst_normalized_rms"] = max(
                (float(x["normalized_rms"]) for x in samples), default=None)
            report["worst_abs_gemm_error"] = max(
                (float(x["max_abs"]) for x in samples), default=None)
            report["nonfinite_audit_elements"] = sum(
                int(x["fp16_nonfinite"]) for x in samples)
            if kind == "fp16" and not report["converted_mm"] and not report["converted_addmm"]:
                raise RuntimeError("No selected BF16 GEMMs converted; invalid experimental arm")
            if kind == "sham" and not report["sham_native"]:
                raise RuntimeError("Sham control did not intercept eligible BF16 GEMMs")
            if audit and report["numerical_sample_count"] == 0:
                raise RuntimeError("Audit requested but no live-model GEMM differences measured")
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"[:2300]
    (OUT/"results"/f"{case}.json").write_text(json.dumps(report, indent=2) + "\n")
    print("SMOKE_CASE " + json.dumps({k:v for k,v in report.items() if k not in ("generation","dryrun")},
                                      sort_keys=True), flush=True)
    return report


def report_summary(rows):
    ref = next((r for r in rows if r["variant"]=="native-a" and r["status"]=="PASS"), None)
    if ref:
        for row in rows:
            if row["status"]=="PASS" and row["variant"]!="native-a":
                if "video_psnr" not in row:
                    row["video_psnr"] = psnr(Path(row["clip"]), Path(ref["clip"]), ffmpeg_bin())
                row["video_exact"] = row.get("video_sha256")==ref["video_sha256"]
                row["audio_exact"] = row.get("audio_sha256")==ref["audio_sha256"]
    keys = ["variant","kind","groups","audit_per_shape","status","wall_seconds",
            "converted_mm","converted_addmm","sham_native","numerical_sample_count",
            "worst_normalized_rms","worst_abs_gemm_error","nonfinite_audit_elements",
            "video_exact","audio_exact","psnr_db","error"]
    with (OUT/"smoke-results.csv").open("w",newline="") as fd:
        w = csv.DictWriter(fd,fieldnames=keys,extrasaction="ignore");w.writeheader()
        for row in rows:
            item = dict(row)
            item["psnr_db"] = (row.get("video_psnr") or {}).get("average_db")
            w.writerow(item)
    timings = {r["variant"]: r["wall_seconds"] for r in rows if r["status"]=="PASS"}
    abba = {}
    controls = ("sham-a1","fp16-b1","fp16-b2","sham-a2")
    if all(c in timings for c in controls):
        reference_s = (timings["sham-a1"] + timings["sham-a2"])/2
        fp16_s = (timings["fp16-b1"]+timings["fp16-b2"])/2
        abba = {"reference_mean_seconds": round(reference_s,3),
                "fp16_mean_seconds": round(fp16_s,3),
                "seconds_saved": round(reference_s-fp16_s,3),
                "time_reduction_percent": round(100*(reference_s-fp16_s)/reference_s,3),
                "balanced_order":"BF16-sham / FP16 / FP16 / BF16-sham",
                "audits_disabled_all_four": True}
    report = {"kind":"V100 short real-model T2V BF16/FP16 numerical + quality smoke",
              "utc":dt.datetime.now(dt.timezone.utc).isoformat(),
              "source_sha":os.getenv("GITHUB_SHA","local"),
              "resolution":RESOLUTION,"frames":FRAMES,"steps":STEPS,"seed":SEED,
              "cases":rows,"matched_sham_vs_fp16_abba":abba,
              "status":"PASS" if all(r["status"]=="PASS" for r in rows) and abba else "PARTIAL"}
    (OUT/"smoke-summary.json").write_text(json.dumps(report,indent=2)+"\n")
    lines = ["V100 short LTX-2.5 T2V selective FP16 GEMM smoke", "="*52,
             f"Result: {report['status']}; {len(rows)} attempted videos; {sum(x['status']=='PASS' for x in rows)} passed",
             f"Resolution={RESOLUTION} frames={FRAMES} steps={STEPS}; identical prompt/settings/seed",
             "Full generation times below include model loading and file encoding.",
             f"{'Variant':<20} {'Status':<8} {'seconds':>9} {'FP16 ops':>10} {'PSNR dB':>9} {'max norm RMS':>13}"]
    for r in rows:
        fp16=int(r.get("converted_mm",0))+int(r.get("converted_addmm",0))
        db = (r.get("video_psnr") or {}).get("average_db")
        lines.append(f"{r['variant']:<20} {r['status']:<8} {r.get('wall_seconds',0):>9.2f}"
                     f" {fp16:>10} {str(round(db,2)) if db is not None else 'n/a':>9}"
                     f" {str(round(r['worst_normalized_rms'],6)) if r.get('worst_normalized_rms') is not None else 'n/a':>13}")
    lines.append("")
    if abba:
        lines.append("FAIR ABBA sham control vs FP16 Stage2 all: " + json.dumps(abba))
    else:
        lines.append("ABBA incomplete; NO fair timing result.")
    lines.extend(("Numerical audits use the same live BF16 operands for native vs FP16,",
                  "but are enabled only on quality/audit cases, never ABBA timing arms.",
                  "PSNR uses decoded video against native-a; audio is hashed from PCM.",
                  "Operator audit max/normalized-RMS is not a guarantee of visual fidelity.",
                  "Experiment does not enable FP16 for production.",
                  ""))
    (OUT/"smoke-summary.txt").write_text("\n".join(lines))
    print("\n".join(lines),flush=True)
    return report


def main():
    if not Path(PY).is_file():
        raise RuntimeError(f"Missing Python venv {PY}")
    if not (CHECKPOINTS/"ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors").is_file():
        raise RuntimeError("Expected baked model missing (offline runner required)")
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Refusing nonempty output directory: {OUT}")
    for d in ("videos","logs","config","results"):
        (OUT/d).mkdir(parents=True,exist_ok=True)
    gpu_free()
    print("T2V_SMOKE_ROOT",OUT,flush=True)
    command_run(["nvcc", "-O3", "-std=c++17", "-arch=sm_70",
                 "-shared", "-Xcompiler", "-fPIC",
                 "benchmarks/ltx25_convrot/ltx25_cublas_e.cu",
                 "-lcublas", "-o", str(EXTENSION)], timeout=240,
                log=OUT/"logs"/"compile-candidate-e.log")
    if not EXTENSION.is_file() or EXTENSION.stat().st_size == 0:
        raise RuntimeError("Candidate E extension compilation produced no shared library")
    settings = {"model_type":"ltx2_25_22B_distilled","prompt":PROMPT,
                "seed":SEED,"num_inference_steps":STEPS,"video_length":FRAMES,
                "resolution":RESOLUTION,"audio_prompt_type":"","video_prompt_type":"",
                "image_mode":0,"spatial_upsampling":"","temporal_upsampling":""}
    (OUT/"input.json").write_text(json.dumps(settings,indent=2)+"\n")
    template = initialize_config()
    ffmpeg = ffmpeg_bin()
    rows=[]
    for case,kind,groups,audit,description in CASES:
        print("START_SMOKE_CASE",case,description,flush=True)
        previous = next((r for r in rows if r["variant"]=="native-a" and r["status"]=="PASS"),None)
        rows.append(run_case(case,kind,groups,audit,template,ffmpeg,previous))
        report_summary(rows)
    result=report_summary(rows)
    if result["status"]!="PASS":
        sys.exit("Partial T2V smoke: see smoke-summary.txt for failures")


if __name__=="__main__":
    main()
