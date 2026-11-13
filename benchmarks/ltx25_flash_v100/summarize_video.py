#!/usr/bin/env python3
"""Summarize matched 49-frame LTX-2.5 video tests without GPU interaction."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys

VARIANTS = ("attn_sdpa_576", "attn_1cat_576", "attn_zhinianqin_576")
KEYS = {"attn_sdpa_576": "sdpa", "attn_1cat_576": "1cat", "attn_zhinianqin_576": "zhinianqin"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sha", default=os.environ.get("GITHUB_SHA", "local"))
    parser.add_argument("--source", type=Path, default=Path("/cache/ltx25/results"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = {}
    errors = []
    for variant in VARIANTS:
        key = KEYS[variant]
        filename = args.source / f"runtime_attention_{variant}_{args.sha[:12]}.json"
        if not filename.is_file():
            records[key] = {"status": "MISSING", "expected_file": str(filename)}
            errors.append(f"{key}: missing result file")
            continue
        try:
            rec = json.loads(filename.read_text())
        except Exception as exc:
            records[key] = {"status": "INVALID_JSON", "error": str(exc)}
            errors.append(f"{key}: cannot parse results")
            continue
        if rec.get("variant") != variant or rec.get("sha") != args.sha:
            errors.append(f"{key}: mismatched generation variant/commit metadata")
        if rec.get("status") != "PASS":
            errors.append(f"{key}: {rec.get('status')}: {rec.get('blocker', '')}")
        if key != "sdpa" and rec.get("candidate_calls", 0) < 1:
            errors.append(f"{key}: no verified CUDA attention calls")
        generation = rec.get("generation", {})
        queue_seconds = rec.get("queue_completed_seconds")
        videos = rec.get("videos", [])
        records[key] = {
            "status": rec.get("status", "UNKNOWN"),
            "backend": rec.get("attention_backend"),
            "elapsed_seconds": generation.get("elapsed_seconds"),
            "queue_completed_seconds": queue_seconds,
            "generation_returncode": generation.get("returncode"),
            "candidate_calls": rec.get("candidate_calls", 0),
            "candidate_banner_found": rec.get("candidate_banner_found", False),
            "attention_shape_stats": rec.get("attention_shape_stats", {}),
            "video": videos[0] if videos else None,
            "result_file": str(filename),
            "generation_log": generation.get("log_path"),
        }
    baseline = records.get("sdpa", {}).get("elapsed_seconds")
    baseline_queue = records.get("sdpa", {}).get("queue_completed_seconds")
    for key, rec in records.items():
        elapsed = rec.get("elapsed_seconds")
        queue_elapsed = rec.get("queue_completed_seconds")
        rec["speedup_vs_sdpa"] = round(baseline / elapsed, 4) if (
            baseline is not None and elapsed and elapsed > 0
        ) else None
        rec["queue_speedup_vs_sdpa"] = round(baseline_queue / queue_elapsed, 4) if (
            baseline_queue is not None and queue_elapsed and queue_elapsed > 0
        ) else None

    final = {
        "status": "PASS" if not errors else "FAIL",
        "sha": args.sha,
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "comparison": "matched-seed 576x320 49-frame 8-step LTX-2.5 T2V; "
                      "FP16, R0 ConvRot, profile 0, SDPA global default",
        "note": "The sampled prompt/settings/seed match. Video quality requires "
                "visual review. CLI elapsed includes interpreter and model startup; "
                "queue time (when logged) better isolates generation.",
        "records": records, "errors": errors,
    }
    (args.output_dir / "video_report.json").write_text(json.dumps(final, indent=2) + "\n")
    lines = [
        "# LTX-2.5 real video three-way attention",
        "",
        f"Status: **{final['status']}** | Commit: {args.sha}",
        "",
        "| Backend | CLI wall (s) | Queue (s) | Queue vs SDPA | Calls | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for key in ("sdpa", "1cat", "zhinianqin"):
        rec = records.get(key, {})
        sec = rec.get("elapsed_seconds")
        speed = rec.get("speedup_vs_sdpa")
        qsec = rec.get("queue_completed_seconds")
        qspeed = rec.get("queue_speedup_vs_sdpa")
        lines.append(f"| {key} | {sec if sec is not None else 'n/a'} | "
                     f"{qsec if qsec is not None else 'n/a'} | "
                     f"{f'{qspeed:.2f}x' if qspeed is not None else 'n/a'} | "
                     f"{rec.get('candidate_calls', 0)} | {rec.get('status', 'MISSING')} |")
    lines += ["", "**Scope:** identical T2V settings, seed and resolution; two external "
              "extensions selected at launch time. Observed actual attention shapes and "
              "logs are in the JSON report. Output quality is not proved by timing alone.",
              ""]
    if errors:
        lines += ["Errors:"] + [f"- {e}" for e in errors]
    summary = "\n".join(lines) + "\n"
    (args.output_dir / "video_summary.md").write_text(summary)
    print(summary, flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
