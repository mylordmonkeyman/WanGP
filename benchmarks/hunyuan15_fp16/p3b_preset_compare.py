#!/usr/bin/env python3
"""Exploratory stock Hunyuan1.5 VAE presets; no tiling-code changes.

Decoded H264 frames are diagnostic. Adoption requires lossless decode
comparison and balanced hot E2E tests, especially when seams change.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics

from p0_statistics import compare_baseline_candidate, mad
from p1b_balanced_compare import frame_md5


def load(root: Path, n: int):
    p = root / f"preset_{n}" / "p3b" / f"vae_config_{n}" / "baseline.json"
    if not p.exists():
        return {"status":"failed_or_not_run","path":str(p)}
    d=json.loads(p.read_text())
    if (d.get("effective_vae_config_requested") != n or
        d.get("effective_mmgp_profile_requested") != 0 or
        d.get("resolved_attention") != "sdpa"):
        raise ValueError("VAE experiment accidentally changed fixed reference: "+str(p))
    samples=d["request_e2e_samples_seconds"]
    if len(samples)!=5 or d.get("measured_generations")!=5 or len(d["clips"])!=6:
        raise ValueError("VAE experiment lacks warmup and five hot clips: "+str(p))
    return {
        "status":"complete", "preset":n,"hot_samples_seconds":samples,
        "hot_median_seconds":statistics.median(samples),
        "mad_seconds":mad(samples),
        "decode_and_copy_median_seconds":statistics.median(
            p["decode_and_copy"] for p in d["phase_samples"]),
        "provisional_denoise_median_seconds":statistics.median(
            p["denoise_inclusive"] for p in d["phase_samples"]),
        "peak_torch_allocated_gib":d["peak_vram_bytes"]/2**30,
        "model_sha256":d["checkpoint_sha256"],
        "lora_sha256":d["converted_lora_sha256"],
        "settings":d["effective_settings"],
        "clips":d["clips"],
    }


def evaluate(root:Path):
    arms={str(n):load(root,n) for n in range(4)}
    if arms["0"]["status"]!="complete":
        raise ValueError("Unmodified preset 0 baseline is missing")
    base=arms["0"]
    for arm in arms.values():
        if arm["status"] != "complete":continue
        for key in ("model_sha256","lora_sha256","settings"):
            if arm[key]!=base[key]:raise ValueError("Mismatched model/sample settings: "+key)
    comp={}
    for k in ("1","2","3"):
        if arms[k]["status"]=="complete":
            comp[k]=compare_baseline_candidate(base["hot_samples_seconds"],
                                              arms[k]["hot_samples_seconds"])
    reference=frame_md5(base["clips"][1]["clip"])
    if len(reference)!=33: raise ValueError("Baseline not 33 frames")
    parity={}
    for n,arm in arms.items():
        if arm["status"]!="complete":
            parity[n]=None
            continue
        diffs=[]
        for clip in arm["clips"][1:]:
            hashes=frame_md5(clip["clip"])
            diffs.append(None if len(hashes)!=33 else
                         sum(a!=b for a,b in zip(reference,hashes)))
        parity[n]=diffs
        del arm["clips"]
        del arm["settings"]
    return {
        "status":"P3B_STOCK_PRESET_SCREEN_NO_DEFAULT_PROMOTION",
        "presets":arms,
        "versus_preset0":comp,
        "decoded_h264_frame_mismatch_counts_by_preset":parity,
        "limitations":[
            "VAE decoder call counts per preset require separate synchronized diagnostic",
            "Decoded H264 parity is not lossless frame or latent fidelity gate",
            "No same-math guarantee when VAE tiling geometry changes",
            "Further custom VAE tile and untiled decoding experiments must be gated by memory and fidelity",
        ],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    root=p.parse_args().root
    data=evaluate(root)
    (root/"p3b_preset_summary.json").write_text(json.dumps(data,indent=2)+"\n")
    print("P3B_PRESET_REPORT "+json.dumps({
        "medians":{k:v.get("hot_median_seconds") for k,v in data["presets"].items()},
        "decode_medians":{k:v.get("decode_and_copy_median_seconds") for k,v in data["presets"].items()},
        "parity":data["decoded_h264_frame_mismatch_counts_by_preset"],
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
