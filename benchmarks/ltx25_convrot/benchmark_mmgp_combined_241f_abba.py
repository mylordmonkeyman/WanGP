#!/usr/bin/env python3
"""Uninstrumented 241f T2V/I2V ABBA: main-equivalent original vs P1+P2.

A: original MMGP packed=0/adaptive8=0, otherwise stable-main-equivalent.
B: bounded host packed preloads=1 and adaptive 1/4/8 workers=1.
Use identical real video input and exact decoded A1 vs B1/B2/A2 hashes.
Wall times measure whole generation subprocess; no GPU profiling.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import statistics

import selective_fp16_t2v_smoke as base
import t2v73_241_original_stock_fourway as four
import distilled_bar73_stage_shape_fourway as stage_shape
from profile_gemm_t2v_i2v import PROMPT, INPUT_IMAGE, check_png

OUT = Path(os.environ.get("LTX25_MMGP_COMBINED_241F_ROOT",
         "/cache/ltx25/results/mmgp-combined-241f-" + os.environ.get("GITHUB_RUN_ID","manual")))
ORDER = ("A1", "B1", "B2", "A2")
MODES = ("t2v", "i2v")
CHECKPOINT = base.CHECKPOINTS / "ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors"
BASELINES = {}  # 241f baseline decoded hashes established by A1, verified again by A2
BASE_STATS = {}


def require(cond,msg):
    if not cond: raise RuntimeError(msg)


def case(mode,arm,config):
    name=f"{mode}-{arm}"
    pack_preload=arm.startswith("B")
    settings=four.settings(241)
    if mode=="i2v": settings.update(image_start=str(INPUT_IMAGE), image_prompt_type="S")
    settings_path=OUT/"settings"/f"{name}.json"
    settings_path.write_text(json.dumps(settings,indent=2)+"\n")
    cfg,target=base.config_for(name,config)
    env=four.optimized_env("stage2_fp16",name)
    for key in ("WAN2GP_LTX25_E_TEST_4096X2048",
                "WAN2GP_LTX25_E_TEST_4096X2048_PHASES",
                "WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE1",
                "WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE2",
                "WAN2GP_LTX25_PIPELINE_PROFILE",
                "WAN2GP_LTX25_PROFILE_STAGES",
                "LTX25_PIPELINE_TRACE_ATTRIBUTION"):
        env.pop(key,None)
    env["WAN2GP_LTX25_E_EPILOGUE_TILED_MAX_M"]="128"
    env["WAN2GP_LTX25_E_EPILOGUE_TILED"]="0"
    env["WAN2GP_LTX25_STAGE2_KEEP_CUDA_CACHE"]="0"
    env["WAN2GP_LTX25_MMGP_PACKED_PRELOAD"]="1" if pack_preload else "0"
    env["WAN2GP_LTX25_MMGP_PACKED_ADAPTIVE8"]="1" if pack_preload else "0"
    env["WAN2GP_LTX25_MMGP_PACKED_PROFILE"]="0"
    env["LTX25_MMGP_PROBE_JSON"]=""
    env.pop("WAN2GP_LTX25_MMGP_P3_PROFILE",None)
    cmd=[base.PY,"wgp.py","--offline","--config",str(cfg),"--process",str(settings_path),
         "--output-dir",str(target),"--profile","0","--attention","sdpa",
         "--no-prevent-power-throttling","--verbose","1"]
    dry=base.command_run(cmd+["--dry-run"],env=env,timeout=300,
                          log=OUT/"logs"/f"{name}-dryrun.log")
    require(dry["returncode"]==0,f"{name}: dry run failed")
    base.gpu_free()
    print("MMGP_COMBINED_241F_ABBA_BEGIN",name,flush=True)
    run=base.command_run(cmd,env=env,timeout=3600,
                         log=OUT/"logs"/f"{name}-generation.log",check=False)
    require(run["returncode"]==0,f"{name}: failed: {run['log_tail']}")
    log=Path(run["log"]).read_text(errors="replace")
    stats=four.read_candidate_stats(log)
    require(stats and stats.get("used",0)>0 and stats.get("mode")=="fp32rot"
            and stats.get("rotation_mode")=="radix4"
            and stats.get("active_4096x2048_phases")==["stage1","stage2"],
            f"{name}: candidate/phase coverage changed: {stats}")
    require(stats.get("epilogue_tiled_enabled") is False and
            int(stats.get("epilogue_tiled_calls",-1))==0,
            f"{name}: tiled epilogue unexpectedly active")
    marker="LTX25_MMGP_PACKED_PRELOAD "
    packed_stats=[json.loads(x.split(marker,1)[1].strip())
                  for x in log.splitlines() if marker in x]
    if pack_preload:
        require(packed_stats and any(x.get("packed_blocks",0)>0 for x in packed_stats),
                f"{name}: packed preload did not execute {packed_stats}")
    else:
        require(not packed_stats,f"{name}: disabled packed preload executed")
    fp16=stage_shape.audit_stage2(OUT/"results"/f"{name}-fp16.json", "stage2_fp16")
    require(fp16["persistent_fp16_weight_copies"]==0,
            f"{name}: found persistent FP16 weights")
    clips=[p for p in target.rglob("*.mp4") if p.stat().st_size>1024]
    require(len(clips)==1,f"{name}: output mismatch {len(clips)}")
    decoded=four.probe_video(clips[0],241)
    if arm=="A1":
        BASELINES[mode]={"video":decoded["decoded_video_sha256"],
                         "audio":decoded["decoded_audio_sha256"],
                         "clip":str(clips[0])}
        BASE_STATS[mode]={"candidate_calls":stats["used"],
                          "native_fallback_calls":stats.get("fallback"),
                          "exceptional_rows":stats.get("range_fallback")}
    require(mode in BASELINES,f"{name}: missing A1 decoded baseline")
    ref=BASELINES[mode]
    matches={"video":decoded["decoded_video_sha256"]==ref["video"],
             "audio":decoded["decoded_audio_sha256"]==ref["audio"]}
    require(stats["used"]==BASE_STATS[mode]["candidate_calls"],
            f"{name}: Candidate E call count changed vs A1")
    require(stats.get("fallback")==BASE_STATS[mode]["native_fallback_calls"],
            f"{name}: native fallbacks changed vs A1")
    require(stats.get("range_fallback")==BASE_STATS[mode]["exceptional_rows"],
            f"{name}: exceptional-row correction count changed vs A1")
    result={"mode":mode,"arm":arm,"pack_preload":pack_preload,
            "adaptive8":pack_preload,
            "frames":241,
            "baseline_source":"A1 main-equivalent MMGP packed and adaptive flags disabled",
            "elapsed_wall_s":run["wall_seconds"],"candidate_e_calls":stats["used"],
            "tiled_epilogue_calls":0,"packed_preload_stats":packed_stats,
            "exceptional_fp32_rows":stats.get("range_fallback"),
            "native_fallback_calls":stats.get("fallback"),
            "video_and_audio_bit_identical":all(matches.values()),
            "matches_A1":matches,"decoded_hashes":decoded,
            "stats":fp16,"output_video":str(clips[0])}
    (OUT/"results"/f"{name}-summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print("MMGP_COMBINED_241F_ABBA_CASE",json.dumps({k:v for k,v in result.items()
          if k not in ("decoded_hashes","stats","output_video")},sort_keys=True),flush=True)
    if not all(matches.values()):
        # Emit diagnostics without silently accepting a numerical regression;
        # other arms are still run so the complete paired comparison survives.
        result["fidelity_status"]="MISMATCH_REQUIRES_QUALITY_ANALYSIS"
        if not matches["video"]:
            result["video_psnr"]=base.psnr(Path(ref["clip"]),clips[0],base.ffmpeg_bin())
        (OUT/"results"/f"{name}-summary.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def main():
    require(not OUT.exists(),f"Output already exists: {OUT}")
    for folder in ("settings","config","videos","results","logs"):
        (OUT/folder).mkdir(parents=True,exist_ok=False)
    check_png()
    require(CHECKPOINT.exists(),"INT8 ConvRot checkpoint missing")
    base.OUT=OUT
    base.EXTENSION=OUT/"libltx25_cublas_e_sm70.so"
    base.FRAMES,base.STEPS,base.SEED=241,8,20261112
    base.RESOLUTION,base.PROMPT="1024x576",PROMPT
    four.OUT=OUT
    base.command_run(["nvcc","-O3","-std=c++17","-arch=sm_70",
                      "-shared","-Xcompiler","-fPIC",
                      "benchmarks/ltx25_convrot/ltx25_cublas_e.cu",
                      "-lcublas","-o",str(base.EXTENSION)],
                     timeout=300,log=OUT/"logs"/"compile.log")
    require(base.EXTENSION.exists(),"CUDA extension unavailable")
    cfg=base.initialize_config()
    (OUT/"manifest.json").write_text(json.dumps({
       "baseline_main":"523cb6f8cf3aeca2daa38a61a3d7e81c01c61c43",
       "development_sha":os.environ.get("GITHUB_SHA"),
       "experiment":"P1+P2 combined versus main-equivalent baseline",
       "flag_default":"off","bounded_CPU_pinned_scratch_buffers":2,
       "arm_A":"packed=0 adaptive8=0","arm_B":"packed=1 adaptive8=1",
       "no_gpu_profiler":True,
       "seed":20261112,"frames":241,"resolution":"1024x576","steps":8,
       "model":str(CHECKPOINT),"profile":0,"attention":"sdpa",
       "case_order":list(ORDER),"modes":list(MODES),
       "note":"A1/A2 baseline main-equivalent; B enables both P1+P2; matched decoded-video and audio hashes; whole-process elapsed seconds."
    },indent=2)+"\n")
    out=[]
    for mode in MODES:
        for arm in ORDER:
            out.append(case(mode,arm,cfg))
            (OUT/"partial.json").write_text(json.dumps(out,indent=2)+"\n")
    summary={}
    for mode in MODES:
        a=[x["elapsed_wall_s"] for x in out if x["mode"]==mode and not x["pack_preload"]]
        b=[x["elapsed_wall_s"] for x in out if x["mode"]==mode and x["pack_preload"]]
        summary[mode]={"baseline_a_wall_s":a,"combined_p1p2_b_wall_s":b,
                       "mean_baseline_wall_s":statistics.mean(a),
                       "mean_combined_p1p2_wall_s":statistics.mean(b),
                       "speedup_percent":100*(statistics.mean(a)-statistics.mean(b))/statistics.mean(a),
                       "all_decoded_outputs_bit_identical":all(x["video_and_audio_bit_identical"]
                                                               for x in out if x["mode"]==mode)}
    (OUT/"summary.json").write_text(json.dumps({"status":"PASS","modes":summary,
        "cases":out,"interpretation":"Matched 241f uninstrumented full-process ABBA, combined P1+P2 against original main-equivalent MMGP; no second GPU weight set"},
        indent=2)+"\n")
    print("LTX25_MMGP_COMBINED_241F_ABBA_SUMMARY",json.dumps(summary,sort_keys=True),flush=True)
    require(all(x["video_and_audio_bit_identical"] for x in out),
            "241f packed-preload video/audio mismatched baseline; see per-case quality diagnostics")
    print("LTX25_MMGP_COMBINED_241F_ABBA_PASS",json.dumps(summary,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
