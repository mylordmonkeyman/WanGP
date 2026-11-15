#!/usr/bin/env python3
"""241-frame T2V: census 8192-row ceiling, run 16384 trial only if relevant.

Single immutable INT8 ConvRot transformer checkpoint, standard Conv VAE, same
prompt/seed. First run with unchanged 64/8192. Only run an expensive second
241f generation if *supported* (N,K) families actually hit 8193..16384 rows.
The exclusion of the known-bad (N=4096,K=2048) family is immutable.
"""
import json
import os
import re
from pathlib import Path

import t2v73_241_original_stock_fourway as four
from shared.kernels import ltx25_cublas_sm70 as candidate  # allowed: no CUDA init

OUT=Path(os.environ.get("LTX25_MAXM_VIDEO_OUT","/cache/ltx25/results/maxm-video-manual"))
MODEL=Path("/opt/ltx25/checkpoints/ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors")


def require(cond,msg):
    if not cond:raise RuntimeError(msg)


def allowed_above_cutoff(census):
    rows=[]
    for signature, counts in census.items():
        match=re.fullmatch(r"M=(\d+),N=(\d+),K=(\d+)",signature)
        if not match:continue
        m,n,k=map(int,match.groups())
        if 8192<m<=16384 and (n,k) in candidate.PASS_SHAPES:
            rows.append({"M":m,"N":n,"K":k,
                         "old_fallback_calls":int(counts.get("fallback",0)),
                         "bytes_candidate_c32":m*n*4})
    return rows


def main():
    require(MODEL.is_file() and MODEL.stat().st_size>1_000_000_000,"Missing original distilled")
    require(not OUT.exists(),"Output directory already exists: "+str(OUT))
    for d in ("results","logs","clips","config","videos"):
        (OUT/d).mkdir(parents=True,exist_ok=False)
    four.OUT=OUT;four.base.OUT=OUT
    four.base.EXTENSION=OUT/"libltx25_cublas_e_sm70.so"
    four.base.FRAMES=241
    four.base.RESOLUTION="1024x576";four.base.STEPS=8
    four.base.SEED=20261112;four.base.PROMPT=four.PROMPT
    (OUT/"input.json").write_text(json.dumps(four.settings(241),indent=2)+"\n")
    four.base.gpu_free()
    four.base.command_run([
        "nvcc","-O3","-std=c++17","-arch=sm_70","-shared","-Xcompiler","-fPIC",
        "benchmarks/ltx25_convrot/ltx25_cublas_e.cu",
        "-lcublas","-o",str(four.base.EXTENSION)
    ],timeout=300,log=OUT/"logs/compile-candidate-e.log")
    require(four.base.EXTENSION.is_file(),"Native library missing")
    original_env=four.optimized_env
    current_max=8192

    def use_env(arm,name):
        env=original_env(arm,name)
        env["WAN2GP_LTX25_E_MIN_M"]="64"
        env["WAN2GP_LTX25_E_MAX_M"]=str(current_max)
        env["WAN2GP_LTX25_E_ROW_CENSUS"]="1"
        return env

    four.optimized_env=use_env
    template=four.base.initialize_config()
    results=[]
    for i,(label,limit) in enumerate((("baseline_8192",8192),("trial_16384",16384)),1):
        if i==2 and not eligible:
            print("MAXM_NO_ELIGIBLE_LIVE_ROWS_OVER_8192 skipping trial full video",flush=True)
            break
        current_max=limit
        print("MAXM_VIDEO_BEGIN",label,limit,flush=True)
        row=four.run_case(241,i,"stage2_fp16",template)
        log=Path(row["log"]).read_text(errors="replace")
        require(MODEL.name in log,"Checkpoint provenance mismatch")
        st=four.read_candidate_stats(log)
        require(st and st.get("max_m")==limit and st.get("min_m")==64
                and st.get("row_census_enabled"),"Census and threshold not active")
        eligible=allowed_above_cutoff(st["row_census"])
        if i==2:
            require(st["used"]>results[0]["candidate_e"]["used"],
                    "High-M experimental threshold did not exercise more Candidate E calls")
        info={"label":label,"max_m":limit,
              "wall_seconds":row["wall_seconds"],
              "candidate_e":row["candidate_e"],
              "stage2_fp16":row["fp16"]["converted_per_stage"]["stage2"],
              "candidate_e_row_census":st["row_census"],
              "eligible_above_8192":eligible,
              "clip":row["clip"],"video_hash":row["decoded_video_sha256"],
              "audio_hash":row["decoded_audio_sha256"]}
        results.append(info)
        (OUT/"partial-results.json").write_text(json.dumps(results,indent=2)+"\n")
        print("MAXM_VIDEO_CASE_PASS",json.dumps({
            "label":label,"seconds":row["wall_seconds"],
            "candidate_e_used":st["used"],"stage2_fp16":info["stage2_fp16"],
            "eligible_above_8192":eligible
        },sort_keys=True),flush=True)
    if len(results)==1:
        decision="NO_SUPPORTED_LIVE_SHAPES_IN_8193_TO_16384"
    else:
        decision="TRIAL_COMPLETED_REQUIRES_VISUAL_QUALITY_REVIEW"
    report={
        "status":"PASS","decision":decision,
        "baseline_max_m":8192,"candidate_max_m":16384,
        "frames":241,"steps":8,"seed":20261112,"resolution":"1024x576",
        "model":str(MODEL),"prompt":four.PROMPT,
        "runs":results,
        "trial_minus_baseline_seconds":round(results[1]["wall_seconds"]-results[0]["wall_seconds"],3) if len(results)==2 else None,
        "caveat":"A/B is single-run exploratory timing, not proof of a reproducible advantage; unrelated shapes remain excluded."
    }
    (OUT/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print("LTX25_MAXM_241_CENSUS_PASS",json.dumps({
        "decision":decision,"times":[r["wall_seconds"] for r in results],
        "eligible_above_8192":results[0]["eligible_above_8192"]
    },sort_keys=True),flush=True)


if __name__=="__main__":main()
