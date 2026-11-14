#!/usr/bin/env python3
"""Aggregate selective FP16 GEMM I2V benchmark variants and build local video gallery.

This tool is offline and uses ffmpeg/ffprobe for optional decoded equality
verification. It never asserts FP16 variants are perceptually equivalent.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys


def _decoded_video_hash(clip: Path, frames: int) -> str:
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(clip),
           "-map", "0:v:0", "-an", "-c:v", "rawvideo", "-pix_fmt", "yuv420p",
           "-f", "framemd5", "-"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
    frame_md5 = [x.split(",")[-1].strip() for x in out.stdout.splitlines()
                 if x.strip() and not x.startswith("#")]
    if len(frame_md5) != frames:
        raise RuntimeError(f"Video has {len(frame_md5)} decoded frames; expected {frames}: {clip}")
    return hashlib.sha256("|".join(frame_md5).encode()).hexdigest()


def _decoded_audio_hash(clip: Path) -> str:
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(clip),
           "-map", "0:a:0", "-vn", "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    digest = hashlib.sha256()
    total = 0
    try:
        while True:
            chunk = proc.stdout.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        err = proc.stderr.read()
        code = proc.wait(timeout=90)
        if code:
            raise RuntimeError(f"Audio decoding failed: {clip}: {err[-300:]}")
        if not total:
            raise RuntimeError(f"No audio decoded: {clip}")
        return digest.hexdigest()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def _video_preview(clip: Path, thumbnail: Path) -> None:
    # Best effort only; MP4 remains the authoritative artifact.
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-ss", "2",
                    "-i", str(clip), "-frames:v", "1", "-q:v", "4",
                    "-vf", "scale=512:-2", str(thumbnail)],
                   capture_output=True, timeout=120)


def render(root: Path, cases: list[str]) -> str:
    results = root / "results"
    metadata = root / "settings" / "input.json"
    config = json.loads(metadata.read_text())
    frames = int(config["video_length"])
    rows = []
    previews = root / "previews"
    previews.mkdir(exist_ok=True)
    for case in cases:
        row_file = results / f"{case}-1.json"
        if row_file.is_file():
            row = json.loads(row_file.read_text())
            clip = root / f"{case}.mp4"
            row["status"] = "PASS" if clip.is_file() and clip.stat().st_size else "MISSING_VIDEO"
            if row["status"] == "PASS":
                try:
                    row["decoded_video_sha256"] = _decoded_video_hash(clip, frames)
                    row["decoded_audio_sha256"] = _decoded_audio_hash(clip)
                except Exception as exc:
                    row["status"] = "DECODE_ERROR"
                    row["decode_error"] = f"{type(exc).__name__}: {exc}"[:350]
                _video_preview(clip, previews / f"{case}.jpg")
        else:
            row = {"variant": case, "index": 1, "status": "FAILED", "clip": ""}
        rows.append(row)
    reference = rows[0]
    for row in rows:
        if row["status"] == "PASS" and reference["status"] == "PASS":
            row["video_exact_to_reference"] = (
                row["decoded_video_sha256"] == reference["decoded_video_sha256"])
            row["audio_exact_to_reference"] = (
                row["decoded_audio_sha256"] == reference["decoded_audio_sha256"])
            if "seconds" in row and "seconds" in reference:
                row["seconds_saved_vs_reference"] = round(
                    reference["seconds"] - row["seconds"], 3)
                row["wall_reduction_percent"] = round(
                    100 * (reference["seconds"] - row["seconds"]) / reference["seconds"], 3)
    columns = ["variant", "status", "seconds", "seconds_saved_vs_reference",
               "wall_reduction_percent", "selective_fp16_gemm_calls",
               "selective_fp16_guard_fallback", "candidate_calls",
               "workspace_leases", "persistent_weight_copies",
               "video_exact_to_reference", "audio_exact_to_reference",
               "decoded_video_sha256", "decoded_audio_sha256", "clip", "decode_error"]
    with (results / "sweep-benchmark.csv").open("w", newline="", encoding="utf-8") as fd:
        writer = csv.DictWriter(fd, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = [
        "LTX-2.5 / V100 selective BF16-to-FP16 GEMM sweep",
        "===============================================",
        f"Frames: {frames}; resolution: {config['resolution']}; steps: {config['num_inference_steps']}",
        "Each generation uses the same image, prompt, LoRA and seed.",
        "Per-run wall times include process startup/model load and are sequential.",
        "GPU/CPU warming, power, throttling, and run order can bias results.",
        "Only output review establishes subjective quality; exact hashes are supplementary.",
        "",
        f"{'Case':<21}{'Status':<15}{'Wall s':>10}{'Save s':>10}{'FP16 GEMMs':>14}  Video/audio exact?",
    ]
    for row in rows:
        exact = "N/A" if row["status"] != "PASS" else (
            str(row.get("video_exact_to_reference", "N/A")) + "/" +
            str(row.get("audio_exact_to_reference", "N/A")))
        summary.append(f"{row['variant']:<21}{row['status']:<15}"
                       f"{row.get('seconds', 0):>10.2f}"
                       f"{row.get('seconds_saved_vs_reference', 0):>10.2f}"
                       f"{row.get('selective_fp16_gemm_calls', 0):>14}  {exact}")
    summary.extend(("", "Videos: " + str(root / "*.mp4"),
                    "Local review page: " + str(root / "review-index.html"),
                    "CSV: " + str(results / "sweep-benchmark.csv"),
                    "Per-case FP16 stats: results/<case>-fp16-stats.json",
                    "FAILED/DECODE_ERROR cases must not be promoted.",
                    "A single A/B sweep is exploratory; confirm winning cases with interleaved repeats.",
                    ""))
    (root / "sweep-summary.txt").write_text("\n".join(summary), encoding="utf-8")

    cards = []
    for row in rows:
        case = row["variant"]
        clip = f"{case}.mp4"
        thumb = f"previews/{case}.jpg"
        banner = f"{row.get('status','unknown')} | {row.get('seconds', '?')} s | FP16 GEMMs: {row.get('selective_fp16_gemm_calls', 0)}"
        cards.append(
            '<article><h2>' + html.escape(case) + '</h2>'
            '<p>' + html.escape(banner) + '</p>' +
            (f'<video controls preload="none" playsinline poster="{html.escape(thumb, quote=True)}" '
             f'src="{html.escape(clip, quote=True)}"></video>'
             f'<p><a href="{html.escape(clip, quote=True)}">Open MP4</a></p>'
             if row["status"] == "PASS" else '<p>No successful video generated. Check logs.</p>')
            + '</article>'
        )
    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LTX-2.5 Selective FP16 GEMM Video Review</title>
<style>body{font:15px system-ui,sans-serif;margin:2rem;background:#111827;color:#edf2f7}
h1{font-size:1.6rem}p{color:#cbd5e1}a{color:#93c5fd}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px}
article{background:#1f2937;border-radius:12px;padding:18px}h2{font-size:1.05rem}video{width:100%;border-radius:8px;background:#000}</style>
</head><body><h1>LTX-2.5 BF16 Reference and FP16 GEMM Variants</h1>
<p>Review visual consistency, faces, movement, and soundtrack. Same seed and input do not guarantee the same output after dtype changes.</p>
<main>""" + "\n".join(cards) + "</main></body></html>\n"
    (root / "review-index.html").write_text(page, encoding="utf-8")
    return "\n".join(summary) + "\n"


def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: summarize_selective_fp16_sweep.py ROOT CASE1,CASE2,...")
    root = Path(sys.argv[1]).resolve()
    cases = sys.argv[2].split(",")
    if not cases or cases[0] != "bf16-reference":
        sys.exit("First case must be bf16-reference")
    print(render(root, cases), end="")


if __name__ == "__main__":
    main()
