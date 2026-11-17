#!/usr/bin/env python3
"""P2B image MLP chunk-count (4,2,1) native FP16 exploratory report.

All arms keep the same selected native FP16 MMGP default=1. Never
compare, retest or vary MMGP as part of this benchmark.
"""
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path
from p0_statistics import compare_baseline_candidate,mad
from p1b_balanced_compare import frame_md5


def read(root:Path,divide:int):
    p=root/f"divide_{divide}"/"p2b"/f"divide_{divide}"/"baseline.json"
    if not p.is_file():return {"status":"failed_or_not_run","divide":divide}
    d=json.loads(p.read_text())
    if d.get("p2b_mlp_divide_requested")!=divide:
        raise ValueError(f"Wrong P2B arm: {p}")
    if d["effective_mmgp_profile_requested"]!=1 or d["effective_vae_config_requested"]!=0 or d["resolved_attention"]!="sdpa":
        raise ValueError("P2B changed fixed model/engine/vae settings")
    if d["measured_generations"]!=5 or d["warmup_generations"]!=1 or len(d["clips"])!=6:
        raise ValueError("P2B must be five hot generations and one warmup")
    return {
        "status":"complete","divide":divide,
        "hot_samples_seconds":d["request_e2e_samples_seconds"],
        "hot_median_seconds":statistics.median(d["request_e2e_samples_seconds"]),
        "hot_mad_seconds":mad(d["request_e2e_samples_seconds"]),
        "denoise_inclusive_seconds_provisional":statistics.median(x["denoise_inclusive"] for x in d["phase_samples"]),
        "peak_torch_allocated_gib":d["peak_vram_bytes"]/2**30,
        "checkpoint_sha256":d["checkpoint_sha256"],
        "converted_lora_sha256":d["converted_lora_sha256"],
        "settings":d["effective_settings"],"clips":d["clips"],
    }


def summarize(root:Path):
    arms={str(n):read(root,n) for n in (4,2,1)}
    a=arms["4"]
    if a["status"]!="complete":raise RuntimeError("Stock divide=4 baseline is missing")
    reference_settings={key:a[key] for key in ("checkpoint_sha256","converted_lora_sha256","settings")}
    reference=frame_md5(a["clips"][1]["clip"])
    if len(reference)!=33:raise RuntimeError("Baseline not 33 decoded frames")
    for arm in arms.values():
        if arm["status"]!="complete":continue
        for key in ("checkpoint_sha256","converted_lora_sha256","settings"):
            if arm[key]!=reference_settings[key]:
                raise RuntimeError(f"P2B changed model/reference setting {key}")
        diffs=[]
        for clip in arm["clips"][1:]:
            checks=frame_md5(clip["clip"])
            if len(checks)!=33:raise RuntimeError("Wrong decoded frame count")
            diffs.append(sum(x!=y for x,y in zip(reference,checks)))
        arm["decoded_frame_mismatch_counts"]=diffs
        del arm["clips"]
        del arm["settings"]
    comp={}
    for divide in (2,1):
        b=arms[str(divide)]
        if b["status"]=="complete":
            comp[str(divide)]=compare_baseline_candidate(a["hot_samples_seconds"],b["hot_samples_seconds"])
    return {"status":"P2B_EXPLORATORY_NO_DEFAULT_PROMOTION",
            "arms":arms,"vs_divide4":comp,
            "limitations":["fixed MMGP profile 1 throughout; not a memory profile test",
                           "decoded H264 frame parity only; latent/lossless numeric tolerance not yet tested",
                           "single sequential arm order; balanced sessions needed for promotion",
                           "peak PyTorch-allocated VRAM is not total V100 memory use"]}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("root",type=Path)
    root=ap.parse_args().root
    report=summarize(root)
    (root/"p2b_mlp_summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print("P2B_MLP_REPORT "+json.dumps({
        "hot_median_seconds":{k:v.get("hot_median_seconds") for k,v in report["arms"].items()},
        "denoise_median_provisional":{k:v.get("denoise_inclusive_seconds_provisional") for k,v in report["arms"].items()},
        "vram_gib":{k:v.get("peak_torch_allocated_gib") for k,v in report["arms"].items()},
        "decoded_frame_mismatches":{k:v.get("decoded_frame_mismatch_counts") for k,v in report["arms"].items()},
    },sort_keys=True),flush=True)


if __name__=="__main__":main()
