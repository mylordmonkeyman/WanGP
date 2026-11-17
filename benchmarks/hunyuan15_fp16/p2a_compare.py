#!/usr/bin/env python3
"""Compare baseline and optional Hunyuan 1.5 CPU sequence-length path.

One independent persistent six-generation process per arm, first generation
discarded; this is a screening trial, NOT the required balanced A/B/B/A gate.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics

from p0_statistics import compare_baseline_candidate
from p1b_balanced_compare import frame_md5


def load(folder: Path, enabled: bool):
    p=folder/"p0/hot/baseline.json"
    data=json.loads(p.read_text())
    if data.get("p2a_cpu_seqlens_requested") is not enabled:
        raise ValueError("P2A flag mismatch in " + str(p))
    if data.get("effective_mmgp_profile_requested") != 0:
        raise ValueError("P2A trial changed MMGP profile")
    if data.get("effective_vae_config_requested") != 0:
        raise ValueError("P2A trial changed VAE config")
    if data.get("resolved_attention") != "sdpa":
        raise ValueError("P2A experiment must use SDPA")
    if data.get("measured_generations") != 5 or data.get("warmup_generations") != 1:
        raise ValueError("Insufficient samples")
    clips=data["clips"]
    if len(clips) != 6 or not all(c["decode_pass"] and c["decoded_frames"]==33 for c in clips):
        raise ValueError("Invalid clip parity fixtures")
    return data


def compare(root: Path):
    a,b=load(root/"A",False),load(root/"B",True)
    for field in ("checkpoint_sha256","converted_lora_sha256",
                  "effective_settings","resolved_attention"):
        if a[field] != b[field]:
            raise ValueError("Reference mismatch in " + field)
    stat=compare_baseline_candidate(a["request_e2e_samples_seconds"],
                                    b["request_e2e_samples_seconds"])
    ref=frame_md5(a["clips"][1]["clip"])
    if len(ref)!=33:
        raise ValueError("Reference video did not decode as 33 frames")
    diffs={}
    for key,arm in (("A",a),("B",b)):
        diffs[key]=[]
        for i in range(1,6):
            actual=frame_md5(arm["clips"][i]["clip"])
            diffs[key].append(None if len(actual)!=33 else
                              sum(x!=y for x,y in zip(ref,actual)))
    exact=all(v==0 for arm in diffs.values() for v in arm)
    return {
        "status":"P2A_EXPLORATORY_NOT_DEFAULT_PROMOTION",
        "baseline_samples_seconds":a["request_e2e_samples_seconds"],
        "candidate_samples_seconds":b["request_e2e_samples_seconds"],
        "comparison":stat,
        "all_10_hot_videos_identical_decoded_h264_frames":exact,
        "frame_mismatch_count_by_arm_and_hot_video":diffs,
        "peak_torch_allocated_gib":{"baseline":a["peak_vram_bytes"]/2**30,
                                   "candidate":b["peak_vram_bytes"]/2**30},
        "limitation":"A/B screen only; run balanced A/B/B/A with lossless frame and latent quality checks before any adoption",
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("root",type=Path)
    args=parser.parse_args()
    data=compare(args.root)
    (args.root/"p2a_comparison.json").write_text(json.dumps(data,indent=2)+"\n")
    print("P2A_EXPLORATORY_REPORT " + json.dumps({
        "median_baseline_s":data["comparison"]["baseline_median_s"],
        "median_candidate_s":data["comparison"]["candidate_median_s"],
        "change_percent":data["comparison"]["improvement_percent"],
        "noise_gate":data["comparison"]["noise_gate"],
        "h264_frame_parity":data["all_10_hot_videos_identical_decoded_h264_frames"],
    },sort_keys=True),flush=True)


if __name__ == "__main__":
    main()
