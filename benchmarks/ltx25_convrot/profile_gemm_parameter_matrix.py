#!/usr/bin/env python3
"""V100 offline LTX-2.5 Quanto GEMM census: isolate frames, resolution and LoRA.

Baseline matched 49-frame 1024x576 T2V/I2V already counted 624 Quanto
qbytes_mm per transformer forward (Stage1 4992, Stage2 1872).
This script runs controlled I2V variations in one independent runner.
No FP16 conversions are performed: direct Quanto sham records native calls.
"""
import json, hashlib, os, subprocess, shutil, sys, traceback, re
from pathlib import Path

import selective_fp16_t2v_smoke as base
import profile_gemm_t2v_i2v as matched

ROOT=matched.ROOT
IMAGE=matched.INPUT_IMAGE
OUT=Path(os.getenv("LTX25_GEMM_MATRIX_ROOT",
            "/cache/ltx25/results/ltx25-gemm-parameters-"+os.getenv("GITHUB_RUN_ID","manual"))).resolve()
base.OUT=OUT
base.EXTENSION=OUT/"libltx25_cublas_e_sm70.so"
PROMPT=matched.PROMPT
LORA_BASENAME="LTX_10Eros_LoRA_Wan2GPfix_fro99-avgrank77.safetensors"
LO_RA_PATHS=(
 Path("/var/lib/llama/models/ltx-2.5/lora")/LORA_BASENAME,
 Path("/opt/ltx25/checkpoints/lora")/LORA_BASENAME,
 Path("/opt/ltx25/checkpoints")/LORA_BASENAME,
 Path("/cache/ltx25/lora")/LORA_BASENAME,
)

def file_hash(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
 return h.hexdigest()

def image_for_resolution(resolution):
 # Preserve the uploaded image pixels; WanGP manages scaling/cropping.
 return str(IMAGE)

# Original no-LoRA 49 1024x576 baseline is already established.
CASES=[
 {"name":"i2v-49-square-no-lora","frames":49,"resolution":"1024x1024","lora":False},
 {"name":"i2v-241-wide-no-lora","frames":241,"resolution":"1024x576","lora":False},
 {"name":"i2v-241-square-no-lora","frames":241,"resolution":"1024x1024","lora":False},
 {"name":"i2v-49-wide-with-lora","frames":49,"resolution":"1024x576","lora":True},
 {"name":"i2v-241-square-with-lora","frames":241,"resolution":"1024x1024","lora":True},
 {"name":"t2v-49-wide-with-lora","frames":49,"resolution":"1024x576","lora":True},
]

def env_for(name):
 env=base.base_env()
 env.update({
   "WAN2GP_LTX25_FP16_GEMM":"1",
   "LTX25_FP16_GEMM_BACKEND":"direct",
   "LTX25_FP16_GEMM_OPS":"qbytes_mm",
   "LTX25_FP16_GEMM_GROUPS":matched.GROUPS,
   "LTX25_FP16_GEMM_MIN_M":"1",
   "LTX25_FP16_GEMM_MIN_DIM":"64",
   "LTX25_FP16_GEMM_SHAM":"1",
   "LTX25_FP16_GEMM_AUDIT_PER_SHAPE":"0",
   "LTX25_FP16_GEMM_GUARD":"first",
   "LTX25_FP16_GEMM_SHAPE_CENSUS":"1",
   "LTX25_FP16_GEMM_STATS_PATH":str(OUT/"results"/(name+"-quanto.json")),
   "WAN2GP_LTX25_PIPELINE_PROFILE":"0",
   "WAN2GP_LTX2_BENCH_TRANSFORMER":"0",
   "LTX25_GEMM_PROFILE_QUANTO_SCOPE":"0",
 })
 return env

def run_case(case,template,lora_path):
 name=case["name"]
 has_lora=bool(case["lora"])
 if has_lora and not lora_path:
  row={**case,"status":"SKIPPED_LORA_MISSING","reason":"Eros LoRA absent from runner accessible paths"}
  (OUT/"results"/(name+"-summary.json")).write_text(json.dumps(row,indent=2)+"\n")
  print("MATRIX_CASE "+json.dumps(row),flush=True)
  return row
 if name.startswith("t2v"):
  image=None
 else: image=image_for_resolution(case["resolution"])
 settings={
  "model_type":"ltx2_25_22B_distilled",
  "prompt":PROMPT, "negative_prompt":"",
  "seed":20261112, "num_inference_steps":8,
  "video_length":case["frames"], "resolution":case["resolution"],
  "NAG_scale":1.1,"audio_cfg_scale":1.5,
  "audio_prompt_type":"","video_prompt_type":"","image_mode":0,
  "activated_loras":[str(lora_path)] if has_lora else [],
  "loras_multipliers":"1" if has_lora else "",
  "spatial_upsampling":"","temporal_upsampling":"",
 }
 if image: settings.update(image_start=image,image_prompt_type="S")
 settings_file=OUT/(name+"-input.json")
 settings_file.write_text(json.dumps(settings,indent=2)+"\n")
 cfg,target=base.config_for(name,template)
 cmd=[base.PY,"wgp.py","--offline","--config",str(cfg),
      "--process",str(settings_file),"--output-dir",str(target),
      "--profile","0","--attention","sdpa",
      "--no-prevent-power-throttling","--verbose","1"]
 env=env_for(name)
 print("MATRIX_START",name,flush=True)
 pre=base.command_run(cmd+["--dry-run"],env=env,timeout=300,
     log=OUT/"logs"/(name+"-dryrun.log"),check=False)
 if pre["returncode"] or "Validation complete. 1/1 task(s) valid." not in (
     OUT/"logs"/(name+"-dryrun.log")).read_text():
  raise RuntimeError("Dryrun failed "+name+" "+pre.get("log_tail","")[-800:])
 base.gpu_free()
 run=base.command_run(cmd,env=env,timeout=2500,
   log=OUT/"logs"/(name+"-generation.log"),check=False)
 if run["returncode"]:
  raise RuntimeError("Generation failed "+name+" "+run.get("log_tail","")[-1200:])
 p=OUT/"results"/(name+"-quanto.json")
 if not p.is_file():
  raise RuntimeError("Missing Quanto counts "+name)
 q=json.loads(p.read_text())
 stats=q["stats"]
 observed=int(stats.get("observed_qbytes_mm",0))
 counts={"stage1":0,"stage2":0}
 shape_rows=[]
 for item in q.get("shape_census",[]):
  counts[item["stage"]]+=item["calls"]
  shape_rows.append({k:item[k] for k in ("stage","op","M","N","K","calls","group","flops_per_call","total_flops")})
 if sum(counts.values())!=observed or observed==0:
  raise RuntimeError(f"Census incomplete: {name} {counts} {observed}")
 # Get actual number of observed ops per stage even with nondefault schedules.
 row={**case,"status":"PASS","wall_seconds_instrumented":run["wall_seconds"],
   "stage_quanto_calls_full_run":counts,
   "total_quanto":observed,
   "stage1_per_step":round(counts["stage1"]/8,3),
   "stage2_per_step":round(counts["stage2"]/3,3),
   "unique_shapes":len(shape_rows),
   "shape_census":shape_rows,
   "settings_sha256":file_hash(settings_file),
   "lora_path":str(lora_path) if has_lora else None,
   "lora_sha256":file_hash(lora_path) if has_lora else None,
   "selected_checkpoint":"ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors",
   "selector_backend":q["backend"],"selector_sham":q["sham"],
   "converted":stats.get("converted_qbytes_mm",0),"numerical_audit_samples":q.get("numerical_sample_count",0),
   "clip_count":len([x for x in target.rglob("*.mp4") if x.stat().st_size>1000]),
 }
 if row["converted"] or not row["selector_sham"]:raise RuntimeError("Not native BF16 sham")
 (OUT/"results"/(name+"-summary.json")).write_text(json.dumps(row,indent=2)+"\n")
 print("MATRIX_CASE "+json.dumps({k:v for k,v in row.items() if k!="shape_census"},sort_keys=True),flush=True)
 return row

def main():
 if OUT.exists() and any(OUT.iterdir()):raise RuntimeError("Nonempty output "+str(OUT))
 for s in ("videos","logs","results","config"): (OUT/s).mkdir(parents=True,exist_ok=True)
 assert IMAGE.is_file()
 base.gpu_free()
 pre=["nvcc","-O3","-std=c++17","-arch=sm_70","-shared","-Xcompiler","-fPIC",
     "benchmarks/ltx25_convrot/ltx25_cublas_e.cu","-lcublas","-o",str(base.EXTENSION)]
 base.command_run(pre,timeout=300,log=OUT/"logs"/"compile.log")
 (OUT/"input.json").write_text(json.dumps(matched.make_settings("t2v"),indent=2)+"\n")
 template=base.initialize_config()
 lora=next((p for p in LO_RA_PATHS if p.is_file() and p.stat().st_size>0),None)
 print("MATRIX_LORA_AVAILABLE",bool(lora),str(lora) if lora else "",flush=True)
 rows=[]
 try:
  for case in CASES:
   try:rows.append(run_case(case,template,lora))
   except Exception as exc:
    row={**case,"status":"FAIL","error":repr(exc),"traceback":traceback.format_exc()[-1500:]}
    print("MATRIX_CASE "+json.dumps(row),flush=True)
    rows.append(row)
 finally:
  summary={"baseline_49_wide_no_lora":{"stage1":4992,"stage2":1872},
           "baseline_previous_241_square_lora":{"stage1":384,"stage2":1440},
           "lora_available":bool(lora),"lora_path":str(lora) if lora else None,"cases":rows}
  (OUT/"matrix-comparison.json").write_text(json.dumps(summary,indent=2)+"\n")
  lines=["Controlled V100 LTX-2.5 Quanto interception matrix",
         "Only I2V/T2V, frame count, resolution and Eros LoRA vary (all else matched)",
         f"LORA_AVAILABLE={bool(lora)}",
         "No FP16 conversions. All numbers are native INT8 Quanto invocation counts.",
         f"{'Case':36s} {'stage1':>8s} {'stage2':>8s} {'s1/step':>9s} {'s2/step':>9s} {'status':>20s}"]
  for x in rows:
   c=x.get("stage_quanto_calls_full_run",{})
   lines.append(f"{x['name']:36s} {c.get('stage1',0):8d} {c.get('stage2',0):8d} "
    f"{x.get('stage1_per_step',0):9.2f} {x.get('stage2_per_step',0):9.2f} {x['status']:>20s}")
  (OUT/"matrix-comparison.txt").write_text("\n".join(lines)+"\n")
  print("\n".join(lines),flush=True)
 if not any(x.get("status")=="PASS" for x in rows):raise RuntimeError("No successful matrix cases")
 print("LTX25_GEMM_PARAMETER_MATRIX_COMPLETE",OUT,flush=True)

if __name__=="__main__":main()
