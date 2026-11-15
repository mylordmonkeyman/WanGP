#!/usr/bin/env python3
"""Generate actual unmodified WanGP 49-frame Distilled stock and compare variants.

The stock reference is pinned deepbeepmeep/WanGP commit c8d47da. Compare
same exact model, prompt, seed, 1024x576, 49 frames, 8 denoise steps, Conv
VAE, NAG=1.1, audio_cfg=1.5. Decode video to PSNR/SSIM and PCM audio.
This is reference similarity, NOT an objective perceptual-quality score.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import t2v73_241_original_stock_fourway as four

OUT=Path(os.environ.get("LTX25_STOCK49_OUT","/cache/ltx25/results/distilled-stock49-reference-manual"))
EXISTING=(
    ("min64_A1","/cache/ltx25/results/distilled-minm49-43678572555-attempt1/clips/t2v49-01-stage2_fp16.mp4"),
    ("min16_B1","/cache/ltx25/results/distilled-minm49-43678572555-attempt1/clips/t2v49-02-stage2_fp16.mp4"),
    ("min16_B2","/cache/ltx25/results/distilled-minm49-43678572555-attempt1/clips/t2v49-03-stage2_fp16.mp4"),
    ("min64_A2","/cache/ltx25/results/distilled-minm49-43678572555-attempt1/clips/t2v49-04-stage2_fp16.mp4"),
    ("experimental_excluded_A1","/cache/ltx25/results/distilled-extra-shape49-43681081038-attempt1/clips/t2v49-01-stage2_fp16.mp4"),
    ("experimental_excluded_B1","/cache/ltx25/results/distilled-extra-shape49-43681081038-attempt1/clips/t2v49-02-stage2_fp16.mp4"),
    ("experimental_excluded_B2","/cache/ltx25/results/distilled-extra-shape49-43681081038-attempt1/clips/t2v49-03-stage2_fp16.mp4"),
    ("experimental_excluded_A2","/cache/ltx25/results/distilled-extra-shape49-43681081038-attempt1/clips/t2v49-04-stage2_fp16.mp4"),
)


def require(cond,msg):
    if not cond:raise RuntimeError(msg)


def audio_pcm(path, ffmpeg):
    cmd=[ffmpeg,"-nostdin","-v","error","-i",str(path),"-map","0:a:0",
         "-c:a","pcm_s16le","-f","s16le","pipe:1"]
    result=subprocess.run(cmd,capture_output=True,timeout=90)
    require(result.returncode==0 and len(result.stdout)>0,
            "Cannot decode audio: "+str(path)+" "+result.stderr[-400:].decode(errors="replace"))
    return result.stdout


def metric(ffmpeg,stock,variant,filter_name):
    # Compare synchronized decoded video, not encoded file bytes.
    cmd=[ffmpeg,"-nostdin","-v","info","-i",str(stock),"-i",str(variant),
         "-filter_complex",f"[0:v][1:v]{filter_name}","-an","-f","null","-"]
    out=subprocess.run(cmd,capture_output=True,text=True,timeout=240)
    require(out.returncode==0,"Failed "+filter_name+" for "+str(variant)+": "+out.stderr[-1200:])
    if filter_name=="psnr":
        m=re.search(r"average:([0-9.]+|inf)",out.stderr)
    else:
        m=re.search(r"All:([0-9.]+)",out.stderr)
    require(m is not None,filter_name+" statistics missing for "+str(variant))
    return float(m.group(1))


def main():
    require(not OUT.exists(),"Output path already used: "+str(OUT))
    for d in ("results","logs","clips","config","videos"):
        (OUT/d).mkdir(parents=True,exist_ok=False)
    four.OUT=OUT
    four.base.OUT=OUT
    four.base.FRAMES=49
    four.base.RESOLUTION="1024x576"
    four.base.STEPS=8
    four.base.SEED=20261112
    four.base.PROMPT=four.PROMPT
    four.base.EXTENSION=OUT/"stock-does-not-use-candidate-e.so"
    provenance=four.verify_source()
    (OUT/"input.json").write_text(json.dumps(four.settings(49),indent=2)+"\n")
    template=four.base.initialize_config()
    four.base.gpu_free()
    stock=four.run_case(49,1,"stock_upstream",template)
    require(stock["frames"]==49 and stock["width"]==1024 and stock["height"]==576,
            "Stock video dimensions changed")
    require(stock["arm"]=="stock_upstream","Not stock generation")
    stock_clip=Path(stock["clip"])
    ffmpeg=four.base.ffmpeg_bin()
    pcm0=audio_pcm(stock_clip,ffmpeg)
    ref=np.frombuffer(pcm0,dtype=np.int16).astype(np.float64)/32768
    report={
        "kind":"Original WanGP 49f vs current Candidate E/FP16 cutoff alternatives",
        "reference":"deepbeepmeep/WanGP stock c8d47da27023476777a0874300bbc9a4126acca6",
        "provenance":provenance,
        "stock":stock,
        "source_sha":os.environ.get("GITHUB_SHA","manual"),
        "model":"original LTX-2.5 22B Distilled INT8 ConvRot checkpoint",
        "frames":49,"steps":8,"resolution":"1024x576","seed":20261112,
        "prompt":four.PROMPT,"NAG_scale":1.1,"audio_cfg_scale":1.5,
        "warning":"Higher PSNR/SSIM means closer to stock, not necessarily higher perceptual quality",
        "variants":[],
    }
    for name,path in EXISTING:
        clip=Path(path)
        if not clip.is_file():
            print("STOCK49_VARIANT_MISSING",name,path,flush=True)
            continue
        # Verify frame count and dimensions by decoding; do not assume.
        video=four.probe_video(clip,49)
        psnr=metric(ffmpeg,stock_clip,clip,"psnr")
        ssim=metric(ffmpeg,stock_clip,clip,"ssim")
        pcm=audio_pcm(clip,ffmpeg)
        pcm_equal=pcm==pcm0
        candidate=np.frombuffer(pcm,dtype=np.int16).astype(np.float64)/32768
        if len(candidate)==len(ref):
            mse=float(np.mean((ref-candidate)**2))
            power=float(np.mean(ref**2))
            audio_snr=math.inf if mse==0 else round(10*math.log10(max(power,1e-20)/mse),6)
        else:
            audio_snr=None
        row={
            "variant":name,"mp4":str(clip),"video_psnr_vs_stock":psnr,
            "video_ssim_vs_stock":ssim,
            "decoded_audio_pcm_equal_to_stock":pcm_equal,
            "audio_snr_vs_stock_db":audio_snr,
            "audio_pcm_samples_match":len(candidate)==len(ref),
            "video_exact_to_stock":video["decoded_video_sha256"]==stock["decoded_video_sha256"],
            "decoded_audio_sha256":hashlib.sha256(pcm).hexdigest(),
        }
        report["variants"].append(row)
        (OUT/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
        print("STOCK49_COMPARISON",json.dumps(row,sort_keys=True),flush=True)
    require(sum(x["variant"].startswith("min") for x in report["variants"])==4,
            "Missing any of four minimum cutoff comparison videos")
    report["status"]="PASS"
    (OUT/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    with (OUT/"summary.csv").open("w",newline="") as f:
        columns=list(report["variants"][0])
        w=csv.DictWriter(f,fieldnames=columns)
        w.writeheader()
        w.writerows(report["variants"])
    print("LTX25_STOCK49_REFERENCE_PASS",json.dumps({
        "stock_seconds":stock["wall_seconds"],
        "variant_count":len(report["variants"]),
        "summary":str(OUT/"summary.json"),
    },sort_keys=True),flush=True)


if __name__=="__main__":main()
