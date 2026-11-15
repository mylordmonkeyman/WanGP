#!/usr/bin/env python3
"""Matched 241f T2V/I2V A-B-B-A: Candidate E legacy vs opt-in tiled epilogue.

A: stable main-equivalent kernel, same exact settings and checkpoint.
B: ONLY WAN2GP_LTX25_E_EPILOGUE_TILED=1, narrow-M (<=128) wide-N.
No permanent weight sets, no profiling instrumentation, no other GPU changes.
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

OUT = Path(os.environ.get("LTX25_EPILOGUE_ABBA_ROOT",
         "/cache/ltx25/results/tiled-epilogue-abba-" + os.environ.get("GITHUB_RUN_ID","manual")))
ORDER = ("A1", "B1", "B2", "A2")
MODES = ("t2v", "i2v")
CHECKPOINT = base.CHECKPOINTS / "ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors"
BASELINES = {}  # 241f baseline decoded hashes established by A1, verified again by A2
BASE_STATS = {}


def require(cond,msg):
    if not cond: raise RuntimeError(msg)


def case(mode,arm,config):
    name=f"{mode}-{arm}"
    is_tiled=arm.startswith("B")
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
    # Hold max-M constant across A and B; ONLY tiling enable differs.
    # Real 241f audio activations are ~251 rows; the default 128 gate would
    # produce zero tiled calls and would not test the new kernel.
    env["WAN2GP_LTX25_E_EPILOGUE_TILED_MAX_M"]="256"
    env["WAN2GP_LTX25_E_EPILOGUE_TILED"]="1" if is_tiled else "0"
    cmd=[base.PY,"wgp.py","--offline","--config",str(cfg),"--process",str(settings_path),
         "--output-dir",str(target),"--profile","0","--attention","sdpa",
         "--no-prevent-power-throttling","--verbose","1"]
    dry=base.command_run(cmd+["--dry-run"],env=env,timeout=300,
                          log=OUT/"logs"/f"{name}-dryrun.log")
    require(dry["returncode"]==0,f"{name}: dry run failed")
    base.gpu_free()
    print("EPILOGUE_ABBA_BEGIN",name,flush=True)
    run=base.command_run(cmd,env=env,timeout=3600,
                         log=OUT/"logs"/f"{name}-generation.log",check=False)
    require(run["returncode"]==0,f"{name}: failed: {run['log_tail']}")
    log=Path(run["log"]).read_text(errors="replace")
    stats=four.read_candidate_stats(log)
    require(stats and stats.get("used",0)>0 and stats.get("mode")=="fp32rot"
            and stats.get("rotation_mode")=="radix4"
            and stats.get("active_4096x2048_phases")==["stage1","stage2"],
            f"{name}: candidate/phase coverage changed: {stats}")
    require(int(stats.get("epilogue_tiled_max_m",-1))==256,
            f"{name}: experimental 241f M gate was not applied")
    require(bool(stats.get("epilogue_tiled_enabled"))==is_tiled,
            f"{name}: tiled flag incorrect {stats}")
    tiled_calls=int(stats.get("epilogue_tiled_calls",0))
    require((tiled_calls>0)==is_tiled,
            f"{name}: did not execute expected tiled epilogue: {tiled_calls}")
    if is_tiled:
        require(tiled_calls<stats["used"],
                f"{name}: tiled path should only be used for narrow-M family")
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
    result={"mode":mode,"arm":arm,"tiled":is_tiled,
            "frames":241,"tiled_max_m":256,
            "baseline_source":"A1 main-equivalent Candidate E at 241f",
            "elapsed_wall_s":run["wall_seconds"],"candidate_e_calls":stats["used"],
            "tiled_epilogue_calls":tiled_calls,
            "exceptional_fp32_rows":stats.get("range_fallback"),
            "native_fallback_calls":stats.get("fallback"),
            "video_and_audio_bit_identical":all(matches.values()),
            "matches_A1":matches,"decoded_hashes":decoded,
            "stats":fp16,"output_video":str(clips[0])}
    (OUT/"results"/f"{name}-summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print("EPILOGUE_ABBA_CASE",json.dumps({k:v for k,v in result.items()
          if k not in ("decoded_hashes","stats","output_video")},sort_keys=True),flush=True)
    if not all(matches.values()):
        # Emit diagnostics without silently accepting a numerical regression;
        # other arms are still run so the complete paired comparison survives.
        result["fidelity_status"]="MISMATCH_REQUIRES_QUALITY_ANALYSIS"
        if not matches["video"]:
            result["video_psnr"]=base.psnr(Path(ref["clip"]),clips[0],base.ffmpeg_bin())
        (OUT/"results"/f"{name}-summary.json").write_text(json.dumps(result,indent=2)+"\\n")
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
       "experiment":"WAN2GP_LTX25_E_EPILOGUE_TILED=1",
       "flag_default":"off","shape_gate":"experimental 241f: M<=256, N>=2048; default stays M<=128",
       "seed":20261112,"frames":241,"resolution":"1024x576","steps":8,
       "model":str(CHECKPOINT),"profile":0,"attention":"sdpa",
       "case_order":list(ORDER),"modes":list(MODES),
       "note":"A1 and A2 are main-equivalent; 241f hashes not previously pinned. B changes only tiling. Baseline comparisons are decoded frames and audio."
    },indent=2)+"\n")
    out=[]
    for mode in MODES:
        for arm in ORDER:
            out.append(case(mode,arm,cfg))
            (OUT/"partial.json").write_text(json.dumps(out,indent=2)+"\n")
    summary={}
    for mode in MODES:
        a=[x["elapsed_wall_s"] for x in out if x["mode"]==mode and not x["tiled"]]
        b=[x["elapsed_wall_s"] for x in out if x["mode"]==mode and x["tiled"]]
        summary[mode]={"baseline_a_wall_s":a,"tiled_b_wall_s":b,
                       "mean_baseline_wall_s":statistics.mean(a),
                       "mean_tiled_wall_s":statistics.mean(b),
                       "speedup_percent":100*(statistics.mean(a)-statistics.mean(b))/statistics.mean(a),
                       "all_decoded_outputs_bit_identical":all(x["video_and_audio_bit_identical"]
                                                               for x in out if x["mode"]==mode)}
    (OUT/"summary.json").write_text(json.dumps({"status":"PASS","modes":summary,
        "cases":out,"interpretation":"ABBA is matched 241f full-process subprocess wall time; no torch profiler enabled; A1 and A2 establish 241f decoded baseline"},
        indent=2)+"\n")
    print("EPILOGUE_TILED_241F_ABBA_SUMMARY",json.dumps(summary,sort_keys=True),flush=True)
    require(all(x["video_and_audio_bit_identical"] for x in out),
            "241f tiled video/audio mismatched baseline; see per-case quality diagnostics")
    print("EPILOGUE_TILED_241F_ABBA_PASS",json.dumps(summary,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
