#!/usr/bin/env python3
"""Balanced A/B/B/A 256px vs 384px VAE spatial tiling report.

Native FP16 MMGP profile 1 is *fixed*, not benchmarked here. Sessions are
separate persistent WanGP processes with one warmup and five hot 33-frame
requests. Keep results even if quality thresholds fail. No default promotion.
"""
from __future__ import annotations
import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

from p0_statistics import compare_baseline_candidate, mad
from p1b_balanced_compare import frame_md5

SESSIONS=(("a1",256),("b1",384),("b2",384),("a2",256))
TRACE_MARK="HUNYUAN15_P3C_RUNTIME_DECODER_CALLS "
TILE_MARK="HUNYUAN15_P3C_TILE_SELECTED "


def json_messages(file:Path, mark:str):
    return [json.loads(line.split(mark,1)[1]) for line in file.read_text(
        errors="replace").splitlines() if mark in line]


def load(root:Path, session:str, tile:int):
    folder=root/f"{session}_{tile}"/"p3c"/f"spatial_{tile}"
    path=folder/"baseline.json"
    report=json.loads(path.read_text())
    if (report.get("p3c_spatial_tile_requested")!=tile or
            report.get("effective_mmgp_profile_requested")!=1 or
            report.get("effective_vae_config_requested")!=0 or
            report.get("resolved_attention")!="sdpa"):
        raise ValueError(f"Invalid fixed-profile native-FP16 P3D arm {session}")
    if report.get("warmup_generations")!=1 or report.get("measured_generations")!=5:
        raise ValueError("P3D requires 1 warmup + 5 measured hot")
    if len(report["clips"])!=6 or len(report["request_e2e_samples_seconds"])!=5:
        raise ValueError("P3D missing clips or measurements")
    for clip in report["clips"]:
        if clip.get("decoded_frames") != 33:
            raise ValueError("P3D clip is not 33 frames: "+str(clip))
    runtime=json_messages(folder/"generation.log",TRACE_MARK)
    selected=json_messages(folder/"generation.log",TILE_MARK)
    expected=15 if tile==256 else 6
    if len(runtime)!=6 or len(selected)!=6:
        raise ValueError(f"P3D missing VAE runtime records {session}")
    if any(t.get("spatial_calls")!=expected or t.get("tile_sample_min_size")!=tile
           or t.get("tile_sample_min_tsize")!=64 for t in runtime):
        raise ValueError(f"P3D unexpected VAE tile/decoder count {session}")
    if any(t.get("tile",{}).get("tile_sample_min_size")!=tile
           or t.get("overridden")!=(tile==384) for t in selected):
        raise ValueError(f"P3D incorrect tile selection {session}")
    samples=report["request_e2e_samples_seconds"]
    peak_used,lowest_free=read_memory(root/f"nvidia_{session}_{tile}.csv")
    return {
        "session":session,"tile_px":tile,
        "hot_samples_seconds":samples,
        "hot_median_seconds":statistics.median(samples),
        "hot_mad_seconds":mad(samples),
        "decode_copy_median_seconds":statistics.median(
            x["decode_and_copy"] for x in report["phase_samples"]),
        "peak_torch_allocated_gib":report["peak_vram_bytes"]/2**30,
        "device_peak_used_mib":peak_used,
        "device_lowest_free_mib":lowest_free,
        "model_sha256":report["checkpoint_sha256"],
        "lora_sha256":report["converted_lora_sha256"],
        "settings":report["effective_settings"],
        "decoder_calls":expected,
        "clips":[x["clip"] for x in report["clips"]],
    }


def read_memory(path:Path):
    # nvidia-smi --format=csv,noheader,nounits:  timestamp, used MiB, free MiB
    used,free=[],[]
    if not path.is_file():raise ValueError("Missing device-memory monitor: "+str(path))
    for line in path.read_text(errors="replace").splitlines():
        fields=[x.strip() for x in line.split(",")]
        if len(fields)<3:continue
        try:
            u=int(fields[-2]); f=int(fields[-1])
        except ValueError:continue
        used.append(u); free.append(f)
    if len(used)<5:raise ValueError("Too few nvidia-smi samples: "+str(path))
    return max(used),min(free)


def ffmpeg_quality(ref:str,candidate:str,metric:str):
    if metric not in ("psnr","ssim"):raise ValueError(metric)
    args=["ffmpeg","-nostdin","-hide_banner","-loglevel","info",
          "-i",ref,"-i",candidate,"-filter_complex",
          "[0:v]setpts=PTS-STARTPTS[a];[1:v]setpts=PTS-STARTPTS[b];[a][b]"+metric,
          "-an","-f","null","-"]
    p=subprocess.run(args,capture_output=True,text=True,timeout=120)
    if p.returncode:raise RuntimeError("FFmpeg quality failed: "+p.stderr[-1200:])
    vals=re.findall((r"average:([0-9.]+|inf)" if metric=="psnr"
                      else r"All:([0-9.]+|inf)"),p.stderr)
    if not vals:raise ValueError("No "+metric+" result: "+p.stderr[-1200:])
    return float(vals[-1])


def decoded_frame_hashes(path:str):
    h=frame_md5(path)
    if len(h)!=33:raise ValueError("Expected 33 decoded frames: "+path)
    return h


def evaluate(root:Path):
    arms={name:load(root,name,tile) for name,tile in SESSIONS}
    for name,arm in arms.items():
        for k in ("model_sha256","lora_sha256","settings"):
            if arm[k]!=arms["a1"][k]:
                raise ValueError(f"P3D changed reference {k} at {name}")
    # Evaluate all repeats within each arm and between paired session arms.
    frame_cache={}
    def frames(path):
        if path not in frame_cache:frame_cache[path]=decoded_frame_hashes(path)
        return frame_cache[path]
    same_arm={}
    for name,arm in arms.items():
        ref=frames(arm["clips"][1])
        counts=[sum(a!=b for a,b in zip(ref,frames(path)))
                for path in arm["clips"][1:]]
        same_arm[name]=counts
    cross_session={}
    for left,right in (("a1","a2"),("b1","b2")):
        counts=[sum(a!=b for a,b in zip(
            frames(arms[left]["clips"][i]),frames(arms[right]["clips"][i])))
            for i in range(1,6)]
        cross_session[f"{left}_{right}"]=counts
    quality=[]
    for candidate,baseline in (("b1","a1"),("b2","a2")):
        for i in range(1,6):
            a=arms[baseline]["clips"][i];b=arms[candidate]["clips"][i]
            diff=sum(x!=y for x,y in zip(frames(a),frames(b)))
            quality.append({
                "baseline_session":baseline,"candidate_session":candidate,
                "hot_index":i,"changed_h264_frames":diff,
                "psnr_db":ffmpeg_quality(a,b,"psnr"),
                "ssim":ffmpeg_quality(a,b,"ssim"),
            })
    btimes=arms["a1"]["hot_samples_seconds"]+arms["a2"]["hot_samples_seconds"]
    ctimes=arms["b1"]["hot_samples_seconds"]+arms["b2"]["hot_samples_seconds"]
    stats=compare_baseline_candidate(btimes,ctimes)
    for arm in arms.values():del arm["clips"];del arm["settings"]
    report={
        "status":"P3D_BALANCED_256_384_MEASURED_NO_AUTOMATIC_PROMOTION",
        "order":[a for a,_ in SESSIONS],
        "results":arms,
        "pooled_performance":stats,
        "pooled_decode_copy_medians_seconds":{
            "256":statistics.median([arms["a1"]["decode_copy_median_seconds"],arms["a2"]["decode_copy_median_seconds"]]),
            "384":statistics.median([arms["b1"]["decode_copy_median_seconds"],arms["b2"]["decode_copy_median_seconds"]]),
        },
        "same_arm_frame_mismatches":same_arm,
        "repeat_session_frame_mismatches":cross_session,
        "matched_clip_quality":quality,
        "minimum_psnr_db":min(q["psnr_db"] for q in quality),
        "minimum_ssim":min(q["ssim"] for q in quality),
        "fidelity_screen_pass":all(q["psnr_db"]>=45 and q["ssim"]>=0.99 for q in quality),
        "minimum_recorded_free_gpu_mib":min(x["device_lowest_free_mib"] for x in arms.values()),
        "limitations":[
            "FFmpeg H264 quality is not pre-encode VAE output or latent parity",
            "VAE seam inspection and 97-/241-frame T2V and I2V remain outstanding",
            "nvidia-smi sampled once per second; may miss instantaneous VRAM peaks",
            "One A/B/B/A sequence, 5 hot clips per arm, no independent-day replication",
            "No VAE default change without separate explicit approval and quality review",
            "MMGP profile1 fixed across all arms; no profile comparisons",
        ],
    }
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument("root",type=Path)
    root=p.parse_args().root
    result=evaluate(root)
    (root/"p3d_balanced_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    perf=result["pooled_performance"]
    print("P3D_BALANCED_REPORT "+json.dumps({
        "hot_medians_seconds":{
            "256":perf["baseline_median_s"],"384":perf["candidate_median_s"]},
        "speedup_pct":perf["improvement_percent"],
        "noise_gate":perf["noise_gate"],
        "95ci_delta_s":perf["bootstrap_95ci_delta_s"],
        "min_psnr_db":result["minimum_psnr_db"],
        "min_ssim":result["minimum_ssim"],
        "lowest_free_gpu_mib":result["minimum_recorded_free_gpu_mib"],
        "fidelity_screen_pass":result["fidelity_screen_pass"],
        "within_arm_frame_mismatches":result["same_arm_frame_mismatches"],
        "across_session_frame_mismatches":result["repeat_session_frame_mismatches"],
    },sort_keys=True),flush=True)


if __name__=="__main__":main()
