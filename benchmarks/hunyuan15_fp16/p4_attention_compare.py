#!/usr/bin/env python3
"""P4 actual V100 SDPA CUDA-event attribution: fixed-profile1 control/events.

CUDA intervals are measured with events on the live PyTorch stream
around every pay_attention and scaled_dot_product_attention call.
They are not a kernel-name capture, nor is the instrumented request
a substitute for uninstrumented end-to-end performance measurements.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from p1b_balanced_compare import frame_md5
from p0_statistics import mad


def load(root: Path, arm: str, enabled: bool):
    p = root / arm / "p4" / f"events_{int(enabled)}" / "baseline.json"
    if not p.is_file():
        raise RuntimeError(f"Missing P4 baseline report: {p}")
    d = json.loads(p.read_text())
    if (d.get("p4_cuda_events_enabled") is not enabled or
            d.get("effective_mmgp_profile_requested") != 1 or
            d.get("resolved_attention") != "sdpa" or
            d.get("effective_vae_config_requested") != 0 or
            d.get("warmup_generations") != 1 or
            d.get("measured_generations") != 5 or
            len(d.get("clips", [])) != 6):
        raise RuntimeError(f"Incorrect P4 arm/reference configuration: {p}")
    if len(d["request_e2e_samples_seconds"]) != 5:
        raise RuntimeError("P4 requires 5 hot requests")
    if enabled:
        records = d.get("p4_attention_hot_samples", [])
        if len(records) != 5:
            raise RuntimeError("Missing five P4 CUDA-event samples")
        for r in records:
            if r["status"] != "P4_ATTENTION_EVENTS_RECORDED":
                raise RuntimeError("P4 CUDA-event recording failed")
            for k in ("pay_attention","sdpa_core"):
                if r["intervals"][k]["count"] != 216:
                    raise RuntimeError("P4 expected 216 " + k + " calls")
        d["p4_attention_metrics"] = {
            "median_sdpa_gpu_stream_seconds": statistics.median(
                r["intervals"]["sdpa_core"]["gpu_stream_total_ms"] for r in records) / 1000,
            "median_outer_gpu_stream_seconds": statistics.median(
                r["intervals"]["pay_attention"]["gpu_stream_total_ms"] for r in records) / 1000,
            "median_host_sdpa_call_seconds": statistics.median(
                r["intervals"]["sdpa_core"]["host_call_total_ms"] for r in records) / 1000,
            "median_host_outer_call_seconds": statistics.median(
                r["intervals"]["pay_attention"]["host_call_total_ms"] for r in records) / 1000,
            "median_sdpa_fraction_of_e2e": statistics.median(
                r["sdpa_stream_fraction_of_request"] for r in records),
            "hot_per_video_samples": records,
        }
    else:
        if any(d.get("p4_attention_hot_samples", [])):
            raise RuntimeError("P4 control arm unexpectedly contains CUDA instrumentation")
    return d


def evaluate(root: Path):
    control=load(root,"control",False)
    events=load(root,"events",True)
    for key in ("checkpoint_sha256","converted_lora_sha256","effective_settings"):
        if control[key] != events[key]:
            raise RuntimeError("P4 model/LoRA/settings mismatch: "+key)
    ref=frame_md5(control["clips"][1]["clip"])
    if len(ref) != 33:
        raise RuntimeError("P4 video baseline not 33 decoded frames")
    parity={}
    for name,arm in (("control",control),("events",events)):
        diffs=[]
        for clip in arm["clips"][1:]:
            hashes=frame_md5(clip["clip"])
            if len(hashes) != len(ref):
                raise RuntimeError("P4 clip changed frame count")
            diffs.append(sum(a!=b for a,b in zip(ref,hashes)))
        parity[name]=diffs
    if any(x for row in parity.values() for x in row):
        raise RuntimeError("P4 event-only instrumentation changed decoded H264 output")
    metrics=events["p4_attention_metrics"]
    # The event arm necessarily has additional event-recording overhead.
    # The control is for observing that cost, not a competing attention kernel.
    return {
        "status":"P4_SDPA_CUDA_EVENT_ATTRIBUTION_NO_KERNEL_SWITCH",
        "mmgp_profile":1,
        "control_hot_e2e_seconds":control["request_e2e_samples_seconds"],
        "control_hot_e2e_median_seconds":statistics.median(control["request_e2e_samples_seconds"]),
        "control_hot_e2e_mad_seconds":mad(control["request_e2e_samples_seconds"]),
        "instrumented_hot_e2e_seconds":events["request_e2e_samples_seconds"],
        "instrumented_hot_e2e_median_seconds":statistics.median(events["request_e2e_samples_seconds"]),
        "instrumentation_overhead_median_delta_seconds":
            statistics.median(events["request_e2e_samples_seconds"]) -
            statistics.median(control["request_e2e_samples_seconds"]),
        "attention":metrics,
        "decoded_h264_frame_mismatch_counts":parity,
        "limitations":[
            "Events measure GPU stream elapsed intervals, not specific named CUDA kernels",
            "No per-call device synchronization; each entire request is synchronized once at end",
            "Host dispatch overhead and GPU stream time overlap; do not add their totals",
            "Instrumented request duration includes CUDA event recording overhead",
            "Control and instrumented arms are in separate sequential sessions; overhead difference may include drift",
            "No attention replacement was tested; candidate kernels require separate compatibility, numeric and hot-E2E checks",
        ],
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("root",type=Path)
    root=ap.parse_args().root
    d=evaluate(root)
    (root/"p4_attention_report.json").write_text(json.dumps(d,indent=2)+"\n")
    a=d["attention"]
    print("P4_SDPA_EVENT_REPORT "+json.dumps({
        "control_e2e_median_s":d["control_hot_e2e_median_seconds"],
        "instrumented_e2e_median_s":d["instrumented_hot_e2e_median_seconds"],
        "cuda_sdpa_total_s_median":a["median_sdpa_gpu_stream_seconds"],
        "cuda_pay_attention_total_s_median":a["median_outer_gpu_stream_seconds"],
        "host_sdpa_total_s_median":a["median_host_sdpa_call_seconds"],
        "sdpa_fraction_e2e_median":a["median_sdpa_fraction_of_e2e"],
        "all_33_frames_identical":True,
    },sort_keys=True),flush=True)


if __name__ == "__main__":
    main()
