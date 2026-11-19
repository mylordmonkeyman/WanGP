#!/usr/bin/env python3
"""One-shot monitored recovery of final missing Hunyuan 1.5 I2V 720p VAE384 MP4.

The previous four-model V100 run generated 7 of 8 requested videos and was
interrupted by an abrupt host reboot during the last case's VAE decoding.
Only retry I2V/720p/default384. Persist temperatures/VRAM/power sampling
outside GitHub workspace so it survives another host reboot. Once successful,
package the seven prior verified videos plus the eighth into one archive.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

import v100_fourway_vae384_smoke as matrix

OUT = matrix.OUT
PRIOR_ROOT = Path("/cache/ltx25/results/hunyuan15-fourway-smoke-44425831072-attempt1")
CASE = ("i2v", "720p", "hunyuan15_v100_i2v_720_fp16_experiment",
        "hunyuanvideo1.5_720p_i2v_fp16.safetensors", "1280x720")
MONITOR = OUT / "host-monitor.csv"


def host_status():
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,"
         "memory.used,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=8
    )
    if proc.returncode:
        raise RuntimeError("nvidia-smi failed: " + proc.stderr[-200:])
    meminfo = Path("/proc/meminfo").read_text()
    mem_available = next((row.split(":")[1].strip().split()[0]
        for row in meminfo.splitlines() if row.startswith("MemAvailable:")), "unknown")
    uptime = Path("/proc/uptime").read_text().split()[0]
    return (proc.stdout.strip().replace(", ", ",").replace("\n",";"),
            mem_available, uptime)


def monitor_run(stop):
    OUT.mkdir(parents=True, exist_ok=True)
    with MONITOR.open("a", encoding="utf8") as stream:
        stream.write("time_utc,temp_c,power_w,used_mib,free_mib,util_pct,mem_available_kib,uptime_sec\n")
        stream.flush()
        os.fsync(stream.fileno())
        while not stop.is_set():
            try:
                gpu, memory, uptime = host_status()
                stream.write(time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()) +
                             "," + gpu + "," + memory + "," + uptime + "\n")
            except Exception as exc:
                stream.write(time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()) +
                             ",SAMPLING_ERROR," + repr(exc).replace(",", ";") + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            stop.wait(2.0)


def inspect_video(path, mode, resolution):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_frames",
         "-show_entries", "stream=width,height,nb_read_frames,codec_name",
         "-of", "json", str(path)],
        capture_output=True, text=True, timeout=90
    )
    if probe.returncode:
        raise RuntimeError("Could not probe "+str(path)+": "+probe.stderr[-500:])
    v = json.loads(probe.stdout)["streams"][0]
    expected = (848, 480) if (mode,resolution)==("i2v","480p") else (
        (832,480) if resolution=="480p" else (1280,720)
    )
    if (int(v["width"]),int(v["height"]),int(v["nb_read_frames"])) != (*expected,33):
        raise RuntimeError("Unexpected dimensions or frame count for "+str(path)+": "+str(v))
    return {"bytes":path.stat().st_size,"sha256":matrix.base.hash_file(path),
            "video":path.name,"width":int(v["width"]),"height":int(v["height"]),
            "frames":33}


def assemble_all_eight(new_video):
    dest=OUT/"all-eight-videos"
    dest.mkdir(parents=True,exist_ok=True)
    result={}
    for mode,res,model_type,name,resolution in matrix.CASES:
        for variant in ("stock256","default384"):
            filename=f"hunyuan15-{mode}-{res}-{variant}-fp16-lightx2v.mp4"
            src = new_video if (mode,res,variant)==("i2v","720p","default384") else PRIOR_ROOT/"videos"/filename
            if not src.is_file() or src.stat().st_size<=1024:
                raise RuntimeError("Previously generated comparison MP4 missing: "+str(src))
            info=inspect_video(src,mode,res)
            shutil.copy2(src,dest/filename)
            result[mode+"_"+res+"_"+variant]=info
    return result


def main():
    print("HUNYUAN15_I2V720_RETRY_START "+json.dumps({
        "case":"i2v_720p_default384", "source_run":44425831072,
        "host_uptime":Path("/proc/uptime").read_text().strip(),
        "note":"one-shot; previous host rebooted in decode, not a model/LoRA failure",
        "checkpoint_dir":str(matrix.STAGED),
        "lora":str(matrix.LORA)}),flush=True)
    matrix.preflight()  # validates ALL checkpoint/LoRA paths and idle GPU
    boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    result={"source_run":44425831072,"case":"i2v_720p_default384",
            "boot_id":boot_id, "monitor":str(MONITOR),
            "generation":None,"all_eight":None}
    stop=threading.Event()
    worker=threading.Thread(target=monitor_run,args=(stop,),daemon=True)
    worker.start()
    try:
        run=matrix.generate(*CASE,"default384")
        result["generation"]=run
        actual=Path(run["video"])
        if not actual.is_file():
            raise RuntimeError("Reported MP4 was not saved: "+str(actual))
        original=PRIOR_ROOT/"videos"/"hunyuan15-i2v-720p-stock256-fp16-lightx2v.mp4"
        quality=matrix.video_quality(original,actual)
        result["quality_vs_stock256"]=quality
        print("HUNYUAN15_I2V720_QUALITY "+json.dumps(quality,sort_keys=True),flush=True)
        result["all_eight"]=assemble_all_eight(actual)
        result["status"]="EIGHT_MP4S_VERIFIED"
        print("HUNYUAN15_EIGHT_MP4S_VERIFIED "+json.dumps({
            "videos":sorted(result["all_eight"]),
            "quality_vs_stock256":quality,
            "new_video":str(actual),
            "boot_id":boot_id},sort_keys=True),flush=True)
    except BaseException as exc:
        result["status"]="FAILED"
        result["error"]=repr(exc)
        traceback.print_exc()
        raise
    finally:
        stop.set()
        worker.join(timeout=5)
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/"recovery-final-summary.json").write_text(
            json.dumps(result,indent=2)+"\n",encoding="utf8")


if __name__=="__main__":
    main()
