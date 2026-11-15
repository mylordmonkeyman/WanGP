#!/usr/bin/env python3
"""Two independently repeated 73f runs of Stage2-only vs both-stage Candidate E.

A1 / B1 / B2 / A2, where A permits the historically rejected (N4096,K2048)
weight family only during Stage2, and B permits it in both Stage1 and Stage2.
Everything else remains unchanged. Full fresh WanGP subprocess timings.
Compare each video to the SAME original unmodified stock WanGP reference
from the successful 73f four-way T2V/I2V experiment (run 43683731705).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import numpy as np
import distilled_bar73_stage_shape_fourway as prior
import distilled_stock49_reference_psnr as metrics
import t2v73_241_original_stock_fourway as four
from profile_gemm_t2v_i2v import INPUT_IMAGE, PROMPT, check_png

FRAMES=73
SEED=20261112
STEPS=8
RESOLUTION="1024x576"
CHECKPOINT=Path("/opt/ltx25/checkpoints/ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors")
PRIOR_RUN_ID=43683731705
CASES=(("A1_stage2_only","stage2"),("B1_both","stage1,stage2"),
       ("B2_both","stage1,stage2"),("A2_stage2_only","stage2"))


def require(condition,msg):
    if not condition:
        raise RuntimeError(msg)


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()


def stock_from_previous(mode):
    root=Path(f"/cache/ltx25/results/bar73-stage-shape-{mode}-{PRIOR_RUN_ID}-attempt1")
    manifest=root/"manifest.json"
    input_path=root/"input.json"
    stock=root/"clips/t2v73-01-stock_upstream.mp4"
    summary=root/"summary.json"
    for path in (manifest,input_path,stock,summary):
        require(path.is_file() and path.stat().st_size>0,f"Missing previous 73f {mode} reference: {path}")
    data=json.loads(manifest.read_text())
    result=json.loads(summary.read_text())
    require(data.get("mode")==mode and data.get("frames")==FRAMES,
            "Reference manifest mode/frame mismatch")
    require(data.get("stock_upstream",{}).get("original_commit")==four.ORIGINAL_SHA,
            "Reference is not pinned unmodified stock WanGP")
    require(result.get("status")=="MEASURED_NOT_PERCEPTUALLY_APPROVED"
            and result.get("mode")==mode
            and result.get("arms",[{}])[0].get("arm")=="stock",
            "Prior stock run did not pass original 73f tests")
    assert isinstance(data.get("settings_sha256"),str)
    require(sha(input_path)==data["settings_sha256"],"Stock input JSON has changed")
    actual_hash=prior.hash_file(stock)
    require(actual_hash and len(actual_hash)==64,"Stock MP4 cannot be hashed")
    return stock,input_path,data,result


def audio_snr(ffmpeg,stock_pcm,stock_float,generated):
    raw=metrics.audio_pcm(generated,ffmpeg)
    require(len(raw)==len(stock_pcm),"Reference and generated audio lengths differ")
    current=np.frombuffer(raw,dtype=np.int16).astype(np.float64)/32768
    mse=float(np.mean((stock_float-current)**2))
    power=float(np.mean(stock_float**2))
    snr=math.inf if mse==0 else round(10*math.log10(max(power,1e-20)/mse),6)
    return {"snr_db":snr,"exact":raw==stock_pcm}


def benchmark(mode):
    require(mode in ("i2v","t2v"),"Invalid mode")
    out=Path(os.environ.get("LTX25_BAR73_REPEAT_OUT",
        f"/cache/ltx25/results/bar73-shape-repeat-{mode}-{os.environ.get('GITHUB_RUN_ID','manual')}"
    )).resolve()
    require(not out.exists(),"Output already exists: "+str(out))
    require(CHECKPOINT.is_file() and CHECKPOINT.stat().st_size>1_000_000_000,
            "Original Distilled INT8 ConvRot checkpoint missing")
    require(INPUT_IMAGE.is_file(),"Original bar PNG missing")
    check_png()

    stock_clip,stock_input,stock_manifest,stock_summary=stock_from_previous(mode)
    four.base.FRAMES=FRAMES
    four.base.STEPS=STEPS
    four.base.SEED=SEED
    four.base.RESOLUTION=RESOLUTION
    four.base.PROMPT=PROMPT
    fresh_settings=four.settings(FRAMES)
    if mode=="i2v":
        fresh_settings.update(image_start=str(INPUT_IMAGE),image_prompt_type="S")
    else:
        require("image_start" not in fresh_settings and "image_prompt_type" not in fresh_settings,
                "T2V unexpectedly includes image conditioning")
    require(fresh_settings==json.loads(stock_input.read_text()),
            "Reference and current run do not have identical full settings")
    if mode=="i2v":
        require(prior.hash_file(INPUT_IMAGE)==stock_manifest["input_image_sha256"],
                "Original bar image bytes have changed")

    for d in ("results","logs","clips","config","videos"):
        (out/d).mkdir(parents=True,exist_ok=False)
    four.OUT=out
    four.base.OUT=out
    four.base.EXTENSION=out/"libltx25_cublas_e_sm70.so"
    (out/"input.json").write_text(json.dumps(fresh_settings,indent=2)+"\n")
    require(sha(out/"input.json")==stock_manifest["settings_sha256"],
            "New input JSON not byte-for-byte identical to stock reference")

    provenance=four.verify_source()
    require(provenance["original_commit"]==four.ORIGINAL_SHA,"Wrong stock provenance")
    ffmpeg=four.base.ffmpeg_bin()
    ref_video=four.probe_video(stock_clip,FRAMES)
    stock_pcm=metrics.audio_pcm(stock_clip,ffmpeg)
    stock_float=np.frombuffer(stock_pcm,dtype=np.int16).astype(np.float64)/32768

    four.base.gpu_free()
    four.base.command_run(["nvcc","-O3","-std=c++17","-arch=sm_70",
        "-shared","-Xcompiler","-fPIC",
        "benchmarks/ltx25_convrot/ltx25_cublas_e.cu","-lcublas",
        "-o",str(four.base.EXTENSION)
    ],timeout=300,log=out/"logs/candidate-e-compile.log")
    require(four.base.EXTENSION.is_file(),"Native Candidate E extension missing")

    original_env=four.optimized_env
    original_verify=four.verify_fp16
    phase_setting="stage2"

    def env_for(arm,case):
        env=original_env(arm,case)
        env.update({
            "WAN2GP_LTX25_E_MIN_M":"64",
            "WAN2GP_LTX25_E_MAX_M":"8192",
            "WAN2GP_LTX25_E_ROW_CENSUS":"1",
            "WAN2GP_LTX25_E_TEST_4096X2048":"0",
            "WAN2GP_LTX25_E_TEST_4096X2048_PHASES":phase_setting,
        })
        return env

    def validate_fp16(stats_path,arm):
        require(arm=="stage2_fp16","Wrong selective FP16 arm")
        return prior.audit_stage2(stats_path,arm)

    four.optimized_env=env_for
    four.verify_fp16=validate_fp16
    template=four.base.initialize_config()
    (out/"manifest.json").write_text(json.dumps({
        "experiment":"N=4096 K=2048 stage2-only vs both-stage repeated ABBA",
        "mode":mode,"frames":FRAMES,"steps":STEPS,"resolution":RESOLUTION,"seed":SEED,
        "model":str(CHECKPOINT),"image_file":str(INPUT_IMAGE) if mode=="i2v" else None,
        "bar_image_sha256":prior.hash_file(INPUT_IMAGE),
        "prompt":PROMPT,"NAG_scale":1.1,"audio_cfg_scale":1.5,"lora":"none",
        "input_sha256":sha(out/"input.json"),
        "stock_original_sha":four.ORIGINAL_SHA,
        "stock_ref_source_run":PRIOR_RUN_ID,
        "stock_ref_video":str(stock_clip),
        "stock_ref_video_decoded_sha256":ref_video["decoded_video_sha256"],
        "stock_ref_video_file_sha256":prior.hash_file(stock_clip),
        "candidate_e":"fp32rot radix4 vector8 fused, validated seven families always",
        "min_m":64,"max_m":8192,
        "stage1_selective_fp16":False,"stage2_selective_fp16":True,
        "only_change":"WAN2GP_LTX25_E_TEST_4096X2048_PHASES",
        "order":list(CASES),
        "timing":"fresh process startup through output-save/exit; no dry-run or metrics in timer",
        "run_sha":os.environ.get("GITHUB_SHA","manual"),
        "quality_note":"PSNR and SSIM to stock measure similarity, not perceptual quality"
    },indent=2)+"\n")

    rows=[]
    for idx,(case,phases) in enumerate(CASES,1):
        phase_setting=phases
        four.base.gpu_free()
        print("REPEAT73_BEGIN",json.dumps({"mode":mode,"arm":case,"phases":phases}),flush=True)
        run=four.run_case(FRAMES,idx,"stage2_fp16",template)
        require(run["settings_sha256"]==sha(out/"input.json"),
                "Input JSON hash changed during ABBA")
        log=Path(run["log"]).read_text(errors="replace")
        require(CHECKPOINT.name in log
                and "Loading NAD Diffusion Decoder" not in log,
                "Wrong checkpoint or VAE")
        require(run["frames"]==FRAMES and
                (run["width"],run["height"])==(1024,576),
                "Wrong output frames/resolution")

        candidate=four.read_candidate_stats(log)
        require(candidate is not None and candidate.get("mode")=="fp32rot"
                and candidate.get("rotation_mode")=="radix4"
                and candidate.get("min_m")==64
                and candidate.get("max_m")==8192
                and candidate.get("row_census_enabled") is True,
                "Candidate E dispatch/source mismatched")
        expected_phases=["stage2"] if phases=="stage2" else ["stage1","stage2"]
        require(candidate.get("experimental_failed_shape_phases")==expected_phases,
                "Failed to set experimental Candidate E stage allowlist")
        per_stage=candidate.get("experimental_shape_by_phase",{})
        require(set(per_stage)=={"stage1","stage2"},
                "Missing per-stage routing counts")
        for stage in ("stage1","stage2"):
            expected_uses=144 if stage=="stage2" else (384 if phases!="stage2" else 0)
            expected_fallback=0 if expected_uses else 384
            actual_uses=int(per_stage[stage].get("used",-1))
            actual_fallback=int(per_stage[stage].get("fallback",-1))
            require((actual_uses,actual_fallback)==(expected_uses,expected_fallback),
                    f"{mode}/{case} {stage} routing mismatch: {(actual_uses,actual_fallback)}")
        require(run["candidate_e"]["used"]==(26976 if phases=="stage2" else 27360),
                "Unexpected total Candidate E GEMM count")
        fp=run["fp16"]
        require(fp["converted_per_stage"]["stage1"]==0
                and fp["converted_per_stage"]["stage2"]==0,
                "The opted-in family was not consumed by Candidate E on Stage2")
        require(fp["persistent_fp16_weight_copies"]==0
                and fp["guard_fallbacks"]==0,
                "Unexpected selective FP16 unsafe fallback")

        clip=Path(run["clip"])
        stock_psnr=metrics.metric(ffmpeg,stock_clip,clip,"psnr")
        stock_ssim=metrics.metric(ffmpeg,stock_clip,clip,"ssim")
        audio=audio_snr(ffmpeg,stock_pcm,stock_float,clip)
        record={
            "case":case,"mode":mode,
            "experimental_4096x2048_phases":phases,
            "wall_seconds":run["wall_seconds"],
            "total_candidate_e_gemms":run["candidate_e"]["used"],
            "target_stage1_used":int(per_stage["stage1"]["used"]),
            "target_stage1_fallback":int(per_stage["stage1"]["fallback"]),
            "target_stage2_used":int(per_stage["stage2"]["used"]),
            "target_stage2_fallback":int(per_stage["stage2"]["fallback"]),
            "stage1_selective_fp16_count":fp["converted_per_stage"]["stage1"],
            "stage2_selective_fp16_count":fp["converted_per_stage"]["stage2"],
            "psnr_db_to_stock":stock_psnr,
            "ssim_to_stock":stock_ssim,
            "pcm_audio_snr_db_to_stock":audio["snr_db"],
            "audio_identical_to_stock":audio["exact"],
            "video_frame_sha256":run["decoded_video_sha256"],
            "decoded_audio_sha256":run["decoded_audio_sha256"],
            "clip":str(clip),
        }
        rows.append(record)
        (out/"partial-results.json").write_text(json.dumps(rows,indent=2)+"\n")
        print("REPEAT73_CASE_PASS",json.dumps(record,sort_keys=True),flush=True)

    by_stage2=[r for r in rows if r["experimental_4096x2048_phases"]=="stage2"]
    by_both=[r for r in rows if r["experimental_4096x2048_phases"]=="stage1,stage2"]
    require(len(by_stage2)==len(by_both)==2,"Missing ABBA repetitions")
    for name,group in (("stage2",by_stage2),("both",by_both)):
        require(len({r["video_frame_sha256"] for r in group})==1,
                name+" same-seed videos differ: non-reproducible")
        require(len({r["decoded_audio_sha256"] for r in group})==1,
                name+" same-seed audio differs: non-reproducible")
    t_a=statistics.median(x["wall_seconds"] for x in by_stage2)
    t_b=statistics.median(x["wall_seconds"] for x in by_both)
    summary={
        "status":"MEASURED_NOT_PERCEPTUALLY_APPROVED",
        "mode":mode,
        "frames":FRAMES,
        "reference_stock_run":PRIOR_RUN_ID,
        "stage2_only_median_seconds":round(t_a,3),
        "both_stages_median_seconds":round(t_b,3),
        "speedup_both_vs_stage2_percent":round((t_a-t_b)/t_a*100,3),
        "seconds_saved_both_vs_stage2":round(t_a-t_b,3),
        "A1_B1_ordered_diff_seconds":round(rows[0]["wall_seconds"]-rows[1]["wall_seconds"],3),
        "A2_B2_ordered_diff_seconds":round(rows[3]["wall_seconds"]-rows[2]["wall_seconds"],3),
        "stage2_psnr_db_to_stock":by_stage2[0]["psnr_db_to_stock"],
        "both_psnr_db_to_stock":by_both[0]["psnr_db_to_stock"],
        "psnr_delta_both_minus_stage2_db":round(
            by_both[0]["psnr_db_to_stock"]-by_stage2[0]["psnr_db_to_stock"],6),
        "stage2_ssim_to_stock":by_stage2[0]["ssim_to_stock"],
        "both_ssim_to_stock":by_both[0]["ssim_to_stock"],
        "ssim_delta_both_minus_stage2":round(
            by_both[0]["ssim_to_stock"]-by_stage2[0]["ssim_to_stock"],6),
        "stage2_audio_snr_db_to_stock":by_stage2[0]["pcm_audio_snr_db_to_stock"],
        "both_audio_snr_db_to_stock":by_both[0]["pcm_audio_snr_db_to_stock"],
        "audio_snr_delta_both_minus_stage2_db":round(
            by_both[0]["pcm_audio_snr_db_to_stock"]-by_stage2[0]["pcm_audio_snr_db_to_stock"],6),
        "stage2_vs_both_video_psnr_db":metrics.metric(ffmpeg,Path(by_stage2[0]["clip"]),Path(by_both[0]["clip"]),"psnr"),
        "stage2_vs_both_video_ssim":metrics.metric(ffmpeg,Path(by_stage2[0]["clip"]),Path(by_both[0]["clip"]),"ssim"),
        "video_reproducible_each_arm":True,
        "audio_reproducible_each_arm":True,
        "raw":rows,
        "limitations":[
            "The rejected weight family previously failed strict per-GEMM numerical audit.",
            "A matched single-seed 73f bar prompt does not establish quality on other prompts/frames.",
            "PSNR/SSIM against stock measure similarity to stock, not ground-truth visual quality.",
            "ABBA two repetitions per configuration improves timing confidence, not a population-wide speed estimate."
        ],
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    fields=[key for key in rows[0] if key not in ("video_frame_sha256","decoded_audio_sha256","clip")]
    with (out/"summary.csv").open("w",newline="") as fd:
        writer=csv.DictWriter(fd,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print("LTX25_BAR73_REPEAT_ABBA_PASS",json.dumps({
        k:v for k,v in summary.items() if k!="raw"
    },sort_keys=True),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",required=True,choices=("t2v","i2v"))
    args=parser.parse_args()
    benchmark(args.mode)
