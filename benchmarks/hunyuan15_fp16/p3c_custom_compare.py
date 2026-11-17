#!/usr/bin/env python3
"""P3C exploratory 256/384/512 Hunyuan 1.5 VAE tile trial summary.

The MMGP profile is held at the selected native-FP16 default; this is not
a memory-profile experiment. Geometry changes can change decoded pixels,
so exact parity is recorded but not misrepresented as a guarantee.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import statistics
import subprocess

from p0_statistics import compare_baseline_candidate, mad
from p1b_balanced_compare import frame_md5

ARMS=(256,384,512)
MARK="HUNYUAN15_P3C_RUNTIME_DECODER_CALLS "
SELECTED="HUNYUAN15_P3C_TILE_SELECTED "


def _lines(path:Path, token:str):
    return [json.loads(line.split(token,1)[1])
            for line in path.read_text(errors="replace").splitlines()
            if token in line]


def _ffmpeg_quality(baseline:str, candidate:str, metric:str):
    assert metric in ("psnr", "ssim")
    # Decode both clips with identical aligned PTS and report aggregate
    # pixel differences. These are H264-decoded, not pre-encode lossless,
    # and must be treated as provisional visual-quality diagnostics.
    cmd=["ffmpeg","-hide_banner","-nostdin","-loglevel","info",
         "-i",baseline,"-i",candidate,
         "-filter_complex",
         "[0:v]setpts=PTS-STARTPTS[a];[1:v]setpts=PTS-STARTPTS[b];[a][b]"+metric,
         "-an","-f","null","-"]
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=100)
    if p.returncode:
        raise RuntimeError(f"ffmpeg {metric} failed: {p.stderr[-1000:]}")
    pat=(r"average:([0-9.]+|inf)" if metric=="psnr"
         else r"All:([0-9.]+|inf)")
    vals=re.findall(pat,p.stderr)
    if not vals:
        raise RuntimeError(f"ffmpeg {metric} output missing summary: {p.stderr[-1000:]}")
    return vals[-1] if vals[-1]=="inf" else float(vals[-1])


def load(root:Path, spatial:int):
    base=root/f"arm_{spatial}"/"p3c"/f"spatial_{spatial}"
    report=base/"baseline.json"
    if not report.exists():
        return {"status":"not_run_or_failed","spatial":spatial}
    data=json.loads(report.read_text())
    if data.get("p3c_spatial_tile_requested")!=spatial:
        raise ValueError("Wrong tile arm report")
    if data.get("effective_mmgp_profile_requested")!=1:
        raise ValueError("P3C does not use selected native-FP16 default")
    if data.get("effective_vae_config_requested")!=0 or data.get("resolved_attention")!="sdpa":
        raise ValueError("Changed VAE preset or attention backend")
    if data.get("warmup_generations")!=1 or data.get("measured_generations")!=5 or len(data["clips"])!=6:
        raise ValueError("Not warmup plus five hot generations")
    trace=base/"generation.log"
    runtime=_lines(trace,MARK)
    config=_lines(trace,SELECTED)
    expected=15 if spatial==256 else 6
    if len(runtime)!=6 or len(config)!=6:
        raise ValueError(f"Missing P3C instrumented decoder records: {spatial}, {len(runtime)}/{len(config)}")
    calls=[v["spatial_calls"] for v in runtime]
    if calls!=[expected]*6:
        raise ValueError(f"Runtime decoder count differs from source geometry: {spatial}, calls={calls}")
    if any(x["tile_sample_min_size"]!=spatial or x["tile_sample_min_tsize"]!=64
           or x["parallel_tiling"] or not x["spatial_tiling"] for x in runtime):
        raise ValueError("VAE tiling state or geometry changed unexpectedly")
    if any(x["tile"]["tile_sample_min_size"]!=spatial or
           x["overridden"]!=(spatial!=256) for x in config):
        raise ValueError("P3C selected tile geometry was not applied")
    durations=data["request_e2e_samples_seconds"]
    return {
        "status":"complete","spatial":spatial,
        "model_sha256":data["checkpoint_sha256"],
        "lora_sha256":data["converted_lora_sha256"],
        "effective_settings":data["effective_settings"],
        "med_hot_seconds":statistics.median(durations),
        "mad_hot_seconds":mad(durations),
        "hot_samples_seconds":durations,
        "med_decode_and_copy_seconds":statistics.median(x["decode_and_copy"] for x in data["phase_samples"]),
        "peak_torch_allocated_gib":data["peak_vram_bytes"]/2**30,
        "actual_decoder_calls":calls,
        "clips":data["clips"],
    }


def summarize(root:Path):
    arms={str(n):load(root,n) for n in ARMS}
    base=arms["256"]
    if base["status"]!="complete":
        raise RuntimeError("P3C fixed 256px native-FP16 baseline failed")
    baseline_clips = list(base["clips"])  # Preserve source clips before stripping report payload
    baseline=frame_md5(baseline_clips[1]["clip"])
    if len(baseline)!=33:raise RuntimeError("Baseline is not 33 decoded frames")
    for arm in arms.values():
        if arm["status"]=="complete":
            for key in ("model_sha256","lora_sha256","effective_settings"):
                if arm[key]!=base[key]:
                    raise RuntimeError(f"P3C changed {key}")
            pairwise=[]
            for index,clip in enumerate(arm["clips"][1:],1):
                hashes=frame_md5(clip["clip"])
                if len(hashes)!=33:raise RuntimeError(f"Wrong frame count for {clip['clip']}")
                item={"mismatched_decoded_h264_frames":sum(a!=b for a,b in zip(baseline,hashes))}
                if arm is not base:
                    item["psnr_h264_db"]=_ffmpeg_quality(baseline_clips[index]["clip"],clip["clip"],"psnr")
                    item["ssim_h264"]=_ffmpeg_quality(baseline_clips[index]["clip"],clip["clip"],"ssim")
                pairwise.append(item)
            arm["frame_comparisons"]=pairwise
            del arm["clips"]
    comp={}
    for spatial in (384,512):
        arm=arms[str(spatial)]
        if arm["status"]=="complete":
            comp[str(spatial)]=compare_baseline_candidate(
                base["hot_samples_seconds"],arm["hot_samples_seconds"])
    return {
        "status":"P3C_CUSTOM_VAE_TILES_EXPLORATORY_NOT_PROMOTED",
        "arms":arms,
        "versus_256":comp,
        "limitations":[
            "MMGP fixed at default 1 throughout; no memory-profile comparisons",
            "H264 PSNR/SSIM is diagnostic, not pre-encode latent/lossless parity",
            "candidate VAE geometry may change output even with fixed latent",
            "single sequential arm order, not balanced ABBA; repeat acceptance needs independent sessions",
            "PyTorch allocated VRAM peak excludes all non-PyTorch memory and headroom",
        ],
    }


def main():
    ap=argparse.ArgumentParser();ap.add_argument("root",type=Path)
    root=ap.parse_args().root
    result=summarize(root)
    (root/"p3c_custom_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print("P3C_CUSTOM_VAE_REPORT "+json.dumps({
        "hot_medians":{k:v.get("med_hot_seconds") for k,v in result["arms"].items()},
        "decode_medians":{k:v.get("med_decode_and_copy_seconds") for k,v in result["arms"].items()},
        "decoder_calls":{k:v.get("actual_decoder_calls") for k,v in result["arms"].items()},
        "max_vram_gib":{k:v.get("peak_torch_allocated_gib") for k,v in result["arms"].items()},
        "quality":{k:v.get("frame_comparisons") for k,v in result["arms"].items()},
    },sort_keys=True),flush=True)


if __name__=="__main__":main()
