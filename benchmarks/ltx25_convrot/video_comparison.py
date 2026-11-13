#!/usr/bin/env python3
"""CPU-only, decoded-video reproducibility and difference audit of cached clips.

Reads persistent outputs from four SPECIFIC completed matched-seed runs.
Never starts a model, downloads a checkpoint, or modifies the original clips.
Metrics measure pixel differences, NOT aesthetic or perceptual quality.

The contact sheet is for human visual inspection; PSNR/SSIM alone cannot
establish that one plausible model output is better than another.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

BASE = Path("/cache/ltx25/results")
SHA = os.environ.get("GITHUB_SHA", "local")
OUT = BASE / "video_comparison_current"
REPORT = BASE / f"video_comparison_{SHA[:12]}.json"
CASES = {
    "R0": ("quality_r0", "9d80ef2bccde", 157.9),
    "Archived-E": ("quality_e", "4adc57c0d17f", 112.8),
    "No-sync-E": ("quality_e", "a566cf56eae", 98.3),
    "Audit-E": ("quality_real_audit", "e8e1398a09c6", 98.5),
}
PAIRS = [
    ("R0", "Archived-E"),
    ("R0", "No-sync-E"),
    ("Archived-E", "No-sync-E"),
    ("No-sync-E", "Audit-E"),
]


def ffmpeg_path():
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, OSError) as exc:
        raise RuntimeError(f"No ffmpeg executable in runner or imageio_ffmpeg: {exc}") from exc


def run(args, timeout=180):
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True, timeout=timeout, check=False)
    if p.returncode:
        raise RuntimeError(
            f"{Path(args[0]).name} exited {p.returncode}: {p.stderr[-1400:]}"
        )
    return p.stdout, p.stderr


def video_for(label):
    suffix, source, seconds = CASES[label]
    path = BASE / f"runtime_{suffix}_{source}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing matched-run report: {path}")
    report = json.loads(path.read_text())
    if report.get("status") != "PASS" or len(report.get("videos", [])) != 1:
        raise ValueError(f"{label}: unexpected run state or video count: {path}")
    videopath = Path(report["videos"][0]["path"])
    if not videopath.is_file():
        raise FileNotFoundError(f"Missing persistent runner video: {videopath}")
    if videopath.stat().st_size != report["videos"][0]["size_bytes"]:
        raise ValueError(f"{label}: original video size changed since the matched run")
    return videopath, {"path": str(videopath), "source_sha_prefix": source,
                       "generation_seconds": seconds, "mp4_bytes": videopath.stat().st_size}


def decoded_hash(ffmpeg, video):
    stdout, _ = run([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
                     "-i", str(video), "-map", "0:v:0", "-pix_fmt", "yuv420p",
                     "-f", "framemd5", "-"], timeout=180)
    frames = [line for line in stdout.splitlines() if line.strip() and not line.startswith("#")]
    if len(frames) != 49:
        raise ValueError(f"Expected 49 decoded frames, found {len(frames)}: {video}")
    # ffmpeg framemd5 is deterministic for the decoded YUV420p pixel stream.
    normalized = "\n".join(frames) + "\n"
    return {"frame_count": len(frames),
            "decoded_framemd5_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
            "all_frames": frames}


def metric(ffmpeg, a, b, kind):
    if kind not in ("ssim", "psnr"):
        raise ValueError(kind)
    _, stderr = run([
        ffmpeg, "-hide_banner", "-nostdin", "-i", str(a), "-i", str(b),
        "-filter_complex", f"[0:v][1:v]{kind}",
        "-an", "-f", "null", "-"
    ], timeout=180)
    if kind == "ssim":
        hits = re.findall(r"All:\s*([0-9.]+)", stderr)
    else:
        hits = re.findall(r"average:\s*([0-9.]+|inf)", stderr)
    if not hits:
        raise ValueError(f"No {kind} score in ffmpeg output: {stderr[-1400:]}")
    return float(hits[-1]) if hits[-1] != "inf" else "infinite"


def contact_sheet(ffmpeg, videos):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return {"status": "NOT_AVAILABLE", "reason": "Pillow not installed"}
    width, height, label_height = 320, 180, 25
    names = list(videos)
    canvas = Image.new("RGB", (width * 5, (height + label_height) * len(names)),
                       (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    for row, name in enumerate(names):
        temp_dir = OUT / name.replace("-", "_").lower()
        temp_dir.mkdir(parents=True, exist_ok=True)
        # Five frame indices uniformly spanning the 49-frame generated clip.
        flt = "select='eq(n,0)+eq(n,12)+eq(n,24)+eq(n,36)+eq(n,48)',scale=320:180"
        run([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
             "-i", str(videos[name]), "-vf", flt, "-fps_mode", "vfr",
             "-frames:v", "5", str(temp_dir / "frame-%02d.png")], timeout=180)
        frames = sorted(temp_dir.glob("frame-*.png"))
        if len(frames) != 5:
            raise RuntimeError(f"Expected five images from {name}, got {len(frames)}")
        for col, file in enumerate(frames):
            canvas.paste(Image.open(file).convert("RGB"), (col * width, row * (height + label_height) + label_height))
        draw.text((8, row * (height + label_height) + 5), name + " | frames 0 / 12 / 24 / 36 / 48", fill=(20, 20, 20))
    dst = OUT / "contact-sheet.jpg"
    canvas.save(dst, quality=90, optimize=True)
    return {"status": "PASS", "path": str(dst), "bytes": dst.stat().st_size}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_sha": SHA,
        "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
        "kind": "Independent cached LTX-2.5 decoded-video reproducibility and pixel difference check",
        "video_source": "four already-generated fixed-seed 1024x576 49-frame 8-step runs",
        "new_generation": False,
        "downloaded_checkpoints": False,
        "perceptual_quality_validated": False,
        "status": "ERROR", "videos": {}, "pairs": [],
    }
    try:
        ffmpeg = ffmpeg_path()
        videos = {}
        hashes = {}
        for name in CASES:
            path, meta = video_for(name)
            h = decoded_hash(ffmpeg, path)
            hashes[name] = h.pop("all_frames")
            meta.update(h)
            videos[name] = path
            report["videos"][name] = meta
        for first, second in PAIRS:
            exact = hashes[first] == hashes[second]
            row = {"reference": first, "comparison": second,
                   "decoded_pixels_identical": exact}
            if exact:
                row["ssim"] = 1.0
                row["psnr_db"] = "infinite"
            else:
                row["ssim"] = metric(ffmpeg, videos[first], videos[second], "ssim")
                row["psnr_db"] = metric(ffmpeg, videos[first], videos[second], "psnr")
            report["pairs"].append(row)
            print("LTX25_VIDEO_PAIR " + json.dumps(row, sort_keys=True), flush=True)
        report["contact_sheet"] = contact_sheet(ffmpeg, videos)
        report["status"] = "PASS"
    except Exception as exc:
        report["status"] = "ERROR"
        report["error"] = f"{type(exc).__name__}: {exc}"[:1600]
    finally:
        REPORT.write_text(json.dumps(report, indent=2) + "\n")
        (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("LTX25_VIDEO_COMPARISON_RESULT " + json.dumps(
            {k: v for k, v in report.items() if k not in ("videos", "pairs")})[:2000], flush=True)
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
