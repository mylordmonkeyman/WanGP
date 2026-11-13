#!/usr/bin/env python3
"""CPU-only diagnostic: decoded video PSNR/SSIM for experimental H256.

This is NOT an acceptance gate. Pixel differences do not necessarily imply
perceptual quality degradation; high PSNR/SSIM does not guarantee temporal
fidelity, faces, prompt adherence or LoRA reliability.

No CUDA allocations, model loads, network, weights or external uploads.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess

ROOT=Path("/cache/ltx25/results")
ORIGINAL=ROOT/"runtime_quality_e_a566cf56eae8.json"
EXPERIMENT=ROOT/"runtime_quality_hadamard_585dc54bfc7d.json"
OUT=ROOT/"hadamard_video_metrics.json"


def movie(path):
    d=json.loads(path.read_text())
    clips=d.get("videos",[])
    assert len(clips)==1,(path,"wrong video count",len(clips))
    file=Path(clips[0]["path"])
    assert file.is_file(),file
    return file


def run_filter(ffmpeg, left, right, filter_name, name):
    stats=ROOT/f"hadamard_{name}_frame_stats.txt"
    # Both videos contain the same 49 frames, same codec and dimensions.
    expr=f"[0:v:0][1:v:0]{filter_name}=stats_file={stats}[v]"
    args=[ffmpeg,"-hide_banner","-nostdin","-loglevel","error",
          "-i",str(left),"-i",str(right),
          "-filter_complex",expr,"-map","[v]","-an","-f","null","-"]
    proc=subprocess.run(args,timeout=220,capture_output=True,text=True)
    if proc.returncode!=0:
        raise RuntimeError(f"ffmpeg {filter_name} failed: {proc.stderr[-700:]}")
    text=stats.read_text(errors="replace")
    key="psnr_avg" if name=="psnr" else "All"
    vals=[]
    for line in text.splitlines():
        matches=re.findall(r"(?:^|\s)"+re.escape(key)+r":([0-9.eE+-]+|inf)",line)
        if matches:
            vals.append(float(matches[-1]))
    if len(vals)!=49:
        raise RuntimeError(f"{name}: expected 49 frame scores, got {len(vals)}")
    return {
        "frames":len(vals),
        "mean":statistics.mean(vals),
        "median":statistics.median(vals),
        "min":min(vals),"max":max(vals),
        "per_frame":vals,
    }


def main():
    ffmpeg=shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg executable missing")
    ref=movie(ORIGINAL)
    experimental=movie(EXPERIMENT)
    psnr=run_filter(ffmpeg,ref,experimental,"psnr","psnr")
    ssim=run_filter(ffmpeg,ref,experimental,"ssim","ssim")
    report={
        "status":"PASS",
        "kind":"CPU-only H256 decoded-video change magnitude, not perceptual acceptance",
        "source_sha":os.environ.get("GITHUB_SHA","local"),
        "run_id":os.environ.get("GITHUB_RUN_ID"),
        "utc":dt.datetime.now(dt.timezone.utc).isoformat(),
        "reference":"validated torch Hadamard FP32 Candidate E fixed seed 20261112",
        "candidate":"radix4 Hadamard H256 fixed seed 20261112",
        "res":"1024x576",
        "frame_count":49,
        "model_quality_validated":False,
        "has_visual_human_review":False,
        "psnr_db":psnr,
        "ssim":ssim,
    }
    OUT.write_text(json.dumps(report,indent=2)+"\n")
    print("LTX25_H256_VIDEO_METRICS "+json.dumps({
        "status":"PASS", "psnr_mean_db":psnr["mean"],"psnr_min_db":psnr["min"],
        "ssim_mean":ssim["mean"],"ssim_min":ssim["min"],
        "frame_count":49,"quality_accepted":False,
    }),flush=True)


if __name__=="__main__":main()
