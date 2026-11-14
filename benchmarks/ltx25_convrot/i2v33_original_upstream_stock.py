#!/usr/bin/env python3
"""Independent stock WanGP main-at-fork I2V versus previously measured flag-stock.

Runs c8d47da... source tree with NO modified ConvRot or selective FP16 code.
Exact same cached checkpoint, 33f 832x480, 4 steps, prompt, image and config as
the previous six-arm experiment. Two whole-process stock replays.
"""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time

import i2v33_stock_candidate_e_stage2_abccba as previous
import selective_fp16_t2v_smoke as base

ROOT=Path(__file__).resolve().parents[2]
UPSTREAM=ROOT/"stock-wangp-original"
FORK_SHA="c8d47da27023476777a0874300bbc9a4126acca6"
PREVIOUS=Path("/cache/ltx25/results/i2v33-3way-43636917878")
OUT=Path(os.environ.get("LTX25_ORIGINAL_STOCK_OUT",f"/cache/ltx25/results/original-stock-i2v33-{os.environ.get('GITHUB_RUN_ID','manual')}"))
base.OUT=OUT
base.FRAMES=33
base.RESOLUTION="832x480"
base.STEPS=4
base.PROMPT=previous.PROMPT
base.SEED=previous.SEED


def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()


def original_env():
    env=dict(os.environ)
    for k in list(env):
        if k.startswith(("WAN2GP_LTX25_", "LTX25_FP16_GEMM_", "LTX25_CUBLAS_",
                         "LTX25_GEMM_", "LTX25_PIPELINE_", "WAN2GP_LTX2_BENCH_TRANSFORMER")):
            env.pop(k,None)
    env.update({
        # Strict import isolation: no performance branch modules on Python path.
        "PYTHONPATH":str(UPSTREAM),
        "WAN2GP_LTX2_SKIP_BAKED_LORA_MIGRATION":"1",
        "GRADIO_ANALYTICS_ENABLED":"False",
        "OMP_NUM_THREADS":"8",
        "HF_HUB_OFFLINE":"1",
        "TRANSFORMERS_OFFLINE":"1",
    })
    return env


def execute(argv, log, *, timeout=1800, env=None, check=True):
    t=time.monotonic()
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("w") as f:
        try:
            p=subprocess.run(argv,cwd=UPSTREAM,env=env or original_env(),
                             stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
            rc=p.returncode
        except subprocess.TimeoutExpired:
            f.write(f"\nERROR: timeout {timeout}s\n")
            rc=124
    secs=round(time.monotonic()-t,3)
    if check and rc:
        raise RuntimeError(f"Original source command exited {rc}: {log.read_text(errors='replace')[-3000:]}")
    return {"wall_seconds":secs,"returncode":rc,"log":str(log)}


def verified_upstream():
    result=subprocess.run(["git","rev-parse","HEAD"],cwd=UPSTREAM,text=True,capture_output=True,check=True)
    if result.stdout.strip()!=FORK_SHA:
        raise RuntimeError("Original source ref changed: "+result.stdout.strip())
    expected=UPSTREAM/"shared/qtypes/int8_convrot.py"
    src=expected.read_text()
    if "ltx25_cublas_sm70" in src or "WAN2GP_LTX25_CUBLAS_E_MODE" in src:
        raise RuntimeError("Original source contains Candidate E modification")
    if (UPSTREAM/"shared/kernels/ltx25_selective_fp16_gemm.py").exists():
        raise RuntimeError("Original source unexpectedly contains selective FP16 patch")
    if (UPSTREAM/"shared/kernels/ltx25_cublas_sm70.py").exists():
        raise RuntimeError("Original source unexpectedly includes custom SM70 candidate")
    return {"source_sha":FORK_SHA,"native_int8_convrot_file_sha256":sha(expected),
            "original_wgp_sha256":sha(UPSTREAM/"wgp.py"),
            "source_repo":"deepbeepmeep/WanGP",
            "no_custom_sm70_module":True,"no_custom_stage2_fp16_module":True}


def run_case(i,template,source):
    name=f"original-stock-{i}"
    cfg,target=base.config_for(name,template)
    env=original_env()
    argv=[base.PY,"wgp.py","--offline","--config",str(cfg),
          "--process",str(OUT/"input.json"),"--output-dir",str(target),
          "--profile","0","--attention","sdpa",
          "--no-prevent-power-throttling","--verbose","1"]
    execute(argv+["--dry-run"], OUT/"logs"/f"{name}-dryrun.log",
            env=env, timeout=300)
    base.gpu_free()
    print("ORIGINAL_STOCK_BEGIN",name,flush=True)
    result=execute(argv,OUT/"logs"/f"{name}-generation.log",env=env,check=False)
    report={"case":name,"arm":"original_upstream_main","run":i,
            "wall_seconds":result["wall_seconds"],"returncode":result["returncode"],
            "input_sha256":sha(OUT/"input.json"),"source_sha":FORK_SHA,
            "log":result["log"]}
    record=OUT/"results"/f"{name}.json"
    record.write_text(json.dumps(report,indent=2)+"\n")
    if result["returncode"]:
        raise RuntimeError(name+" failed: "+Path(result["log"]).read_text(errors="replace")[-3500:])
    log=Path(result["log"]).read_text(errors="replace")
    if "LTX25_CUBLAS_E" in log or "LTX25_SELECTIVE_FP16_GEMM" in log:
        raise RuntimeError(name+": custom SM70 code executed in purported original source")
    clips=[p for p in target.rglob("*.mp4") if p.stat().st_size>1024]
    if len(clips)!=1:
        raise RuntimeError(name+f": expected exactly one MP4, found {len(clips)}")
    dest=OUT/"clips"/f"{name}.mp4"
    shutil.copy2(clips[0],dest)
    report.update({"clip":str(dest),"clip_bytes":dest.stat().st_size,
                   "frames_hash":base.frames_hash(dest,base.ffmpeg_bin()),
                   "pcm_audio_sha256":base.audio_hash(dest,base.ffmpeg_bin())})
    record.write_text(json.dumps(report,indent=2)+"\n")
    print("ORIGINAL_STOCK_CASE_PASS",json.dumps({
        "name":name,"wall_seconds":report["wall_seconds"],
        "video_hash":report["frames_hash"],"audio_hash":report["pcm_audio_sha256"],
    }),flush=True)
    return report


def summarize(rows):
    prior_files=[PREVIOUS/"results"/f"{name}.json" for name in ("01-stock","06-stock")]
    prior=[json.loads(p.read_text()) for p in prior_files]
    old_setting=PREVIOUS/"input.json"
    if not old_setting.exists() or sha(OUT/"input.json")!=sha(old_setting):
        raise RuntimeError("Prior stock input settings differ from original upstream test")
    if any(r["input_sha256"]!=rows[0]["input_sha256"] for r in rows+prior):
        raise RuntimeError("Original and prior settings don't match")
    control=Path(prior[0]["clip"])
    if not control.is_file():
        raise RuntimeError("Previous native-R0 stock clip is missing on V100 runner")
    original_mean=statistics.mean(r["wall_seconds"] for r in rows)
    previous_mean=statistics.mean(r["wall_seconds"] for r in prior)
    summary={
        "status":"MEASURED_NOT_QUALITY_APPROVED","model":"LTX-2.5 22B distilled INT8 ConvRot",
        "source_repo":"deepbeepmeep/WanGP","source_sha":FORK_SHA,
        "settings_identical":True,"settings_sha256":rows[0]["input_sha256"],
        "frames":33,"resolution_requested":"832x480","diffusion_steps":4,
        "timing_method":"fresh wgp.py subprocess startup to process exit, dry-run excluded",
        "original_stock":{"wall_seconds":[r["wall_seconds"] for r in rows],
                          "mean_seconds":round(original_mean,3)},
        "flag_stock_previous":{"wall_seconds":[r["wall_seconds"] for r in prior],
                               "mean_seconds":round(previous_mean,3)},
        "difference_original_minus_flagstock_seconds":round(original_mean-previous_mean,3),
        "difference_original_minus_flagstock_percent":round(100*(original_mean-previous_mean)/previous_mean,3),
        "comparison_to_previous_native_stock":{
            "video_psnr_db":[base.psnr(Path(r["clip"]),control,base.ffmpeg_bin()).get("average_db") for r in rows],
            "audio_equal":[r["pcm_audio_sha256"]==prior[0]["audio_hash"] for r in rows],
            "decoded_frames_exact":[r["frames_hash"]==prior[0]["video_hash"] for r in rows],
            "original_repeat_frames_equal":rows[0]["frames_hash"]==rows[1]["frames_hash"],
            "original_repeat_audio_equal":rows[0]["pcm_audio_sha256"]==rows[1]["pcm_audio_sha256"],
            "old_stock_clip":str(control)
        },
        "original_source_hashes":verified_upstream()
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    with (OUT/"summary.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["arm","run1_s","run2_s","mean_s"])
        for key in ("original_stock","flag_stock_previous"):
            row=summary[key];w.writerow([key,*row["wall_seconds"],row["mean_seconds"]])
    print("ORIGINAL_STOCK_SUMMARY",json.dumps(summary,sort_keys=True),flush=True)


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("Refusing to overwrite nonempty result path: "+str(OUT))
    for d in ("logs","clips","results","config","videos"):
        (OUT/d).mkdir(parents=True,exist_ok=True)
    base.gpu_free()
    source=verified_upstream()
    if not (PREVIOUS/"input.json").is_file():
        raise RuntimeError("Previous run's exact input settings unavailable. Do not silently substitute.")
    # Copy exact JSON used for the prior head-to-head, including original image path.
    shutil.copy2(PREVIOUS/"input.json", OUT/"input.json")
    previous_input=json.loads((OUT/"input.json").read_text())
    for key,expect in (("video_length",33),("resolution","832x480"),("num_inference_steps",4),
                       ("seed",20261112),("model_type","ltx2_25_22B_distilled")):
        if previous_input.get(key)!=expect:raise RuntimeError(f"Unexpected {key}")
    if not Path(previous_input["image_start"]).is_file():
        raise RuntimeError("Previous conditioning image no longer accessible")
    checkpoint=base.CHECKPOINTS/"ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors"
    if not checkpoint.exists():raise RuntimeError("Original checkpoint unavailable")
    # Keep same config template and MMGP/attention settings as previous test.
    template=base.initialize_config()
    (OUT/"manifest.json").write_text(json.dumps({
        **source,"settings_sha256":sha(OUT/"input.json"),
        "checkpoint":str(checkpoint),"checkpoint_bytes":checkpoint.stat().st_size,
        "image":previous_input["image_start"],"image_sha256":sha(Path(previous_input["image_start"])),
        "prior_run":43636917878,"same_stable_checkpoint":True,
        "same_venv":base.PY,"config_template":"same cached template selection as prior test",
        "profile":0,"attention":"sdpa","mmgp_profile":0,
        "quality_reference":"cached prior 01-stock.mp4",
        "runs":2,
    },indent=2)+"\n")
    rows=[]
    for i in (1,2):
        rows.append(run_case(i,template,source))
        (OUT/"partial-results.json").write_text(json.dumps(rows,indent=2)+"\n")
    summarize(rows)
    print("LTX25_ORIGINAL_STOCK_PASS",str(OUT),flush=True)


if __name__=="__main__":
    main()
