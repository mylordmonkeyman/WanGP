#!/usr/bin/env bash
# LTX-2.5 22B INT8 ConvRot 241-frame I2V: BF16 sham vs Stage2 FP16 vs both-stage FP16.
# Same input, LoRA, settings and Candidate E optimizations in every arm.
# Research only. Never changes WanGP production defaults or model weights.
set -Eeuo pipefail

usage() {
cat <<'HELP'
Usage:
  bash run_ltx25_fp16_i2v_241_3way.sh IMAGE 'POSITIVE PROMPT' 'NEGATIVE PROMPT'

Three configurations (direct Quanto-only interception in ALL arms):
  sham      Native BF16 GEMMs, but the same interception overhead
  s2-all    FP16 for eligible Quanto GEMMs in Stage 2; Stage 1 native BF16
  both-all  FP16 for eligible Quanto GEMMs in both transformer stages

Defaults match existing 241-frame Candidate E I2V harness:
  FRAMES=241 RESOLUTION=1024x1024 STEPS=8 SEED=20261112
  LORA=/var/lib/llama/models/ltx-2.5/lora/LTX_10Eros_LoRA_Wan2GPfix_fro99-avgrank77.safetensors
  LORA_STRENGTH=1 NAG_SCALE=1.1 AUDIO_CFG_SCALE=1.5
  MMGP profile 0, SDPA, Candidate E fp32rot/radix4/vector8/direct/fused

Default REPEATS=2: sham, s2-all, both-all, both-all, s2-all, sham
  REPEATS=1          only 3 videos, quick exploratory timing
  LORA=none          disable LoRA, otherwise identical in all arms
  RUN_ROOT=/path     unique/empty results directory
  PY=/path           CUDA V100 Python virtual environment
  CONFIG_TEMPLATE=/path/wgp_config.json
  CHECK_GPU_IDLE=0   explicitly allow running with GPU work (invalid speed)
  SKIP_PREFLIGHT=1   skip non-timed V100 numerical preflight

Outputs in RUN_ROOT:
  sham-1.mp4, s2-all-1.mp4, both-all-1.mp4 (and -2 for repeats)
  sham.mp4, s2-all.mp4, both-all.mp4 (aliases of first-run clips)
  results/*-seconds.txt, results/*-fp16-stats.json, results/*.json
  results/quality-*.json, results/*-frame-psnr.txt, results/worst-frames/*.png
  benchmark.csv, quality-summary.json, summary.txt, run-manifest.txt

Timing is FULL video generation process wall time (including model startup);
numerical GEMM audit is disabled for fair timing. Quality uses losslessly
decoded MP4 pixels, PSNR, SSIM, audio PCM hashes, and worst-frame PNGs.
The sham arm is the timed BF16 reference, not a separately uninstrumented
no-selector baseline. No changes to persistent FP16 weights or defaults.
HELP
}
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi
if (( $# != 3 )); then usage >&2; exit 2; fi
IMAGE="$1"; PROMPT="$2"; NEGATIVE_PROMPT="$3"
[[ -n "$PROMPT" && -n "$NEGATIVE_PROMPT" ]] || { echo "Prompts cannot be empty" >&2; exit 2; }

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)}"
REPO_ROOT="$(realpath -e "$REPO_ROOT")"
for f in wgp.py shared/kernels/ltx25_selective_fp16_gemm.py \
  shared/kernels/ltx25_cublas_sm70.py benchmarks/ltx25_convrot/ltx25_cublas_e.cu \
  benchmarks/ltx25_convrot/fp16_quanto_direct_preflight.py; do
  [[ -s "$REPO_ROOT/$f" ]] || { echo "Missing $f in $REPO_ROOT; git pull feature branch." >&2; exit 1; }
done
grep -Fq 'def _direct_quanto_scope' "$REPO_ROOT/shared/kernels/ltx25_selective_fp16_gemm.py" || {
  echo "Direct Quanto interception code is missing. Update feature branch." >&2; exit 1;
}
IMAGE="$(realpath -e "$IMAGE")"
[[ -s "$IMAGE" ]] || { echo "Image absent: $IMAGE" >&2; exit 1; }

FRAMES="${FRAMES:-241}"; STEPS="${STEPS:-8}"; SEED="${SEED:-20261112}"
RESOLUTION="${RESOLUTION:-1024x1024}"
LORA="${LORA:-/var/lib/llama/models/ltx-2.5/lora/LTX_10Eros_LoRA_Wan2GPfix_fro99-avgrank77.safetensors}"
LORA_STRENGTH="${LORA_STRENGTH:-1}"
NAG_SCALE="${NAG_SCALE:-1.1}"; AUDIO_CFG_SCALE="${AUDIO_CFG_SCALE:-1.5}"
REPEATS="${REPEATS:-2}"
[[ "$REPEATS" == 1 || "$REPEATS" == 2 ]] || { echo "REPEATS must be 1 or 2" >&2; exit 2; }
[[ "$FRAMES" =~ ^[1-9][0-9]*$ && "$STEPS" =~ ^[1-9][0-9]*$ && "$SEED" =~ ^[0-9]+$ ]] || exit 2
[[ "$RESOLUTION" =~ ^[1-9][0-9]*x[1-9][0-9]*$ ]] || exit 2
for val in "$LORA_STRENGTH" "$NAG_SCALE" "$AUDIO_CFG_SCALE"; do
  [[ "$val" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "Not a valid numeric parameter: $val" >&2; exit 2; }
done
if [[ "$LORA" != none && -n "$LORA" ]]; then
  LORA="$(realpath -e "$LORA")"
  [[ -s "$LORA" ]] || { echo "LoRA unavailable: $LORA" >&2; exit 1; }
fi

# Venv symlink must remain intact; do not realpath Python executable.
if [[ -z "${PY:-}" ]]; then
  for p in /opt/ai/wan2gp/.venv/bin/python "$REPO_ROOT/.venv/bin/python" /opt/venvs/ltx25-cu128/bin/python; do
    if [[ -x "$p" ]] && "$p" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.get_device_capability(0)==(7,0)' >/dev/null 2>&1; then
      PY="$p"; break
    fi
  done
fi
[[ -n "${PY:-}" && -x "$PY" ]] || { echo "Set PY=/path/to/V100/CUDA/Quanto/python" >&2; exit 1; }
"$PY" -c 'import torch, optimum.quanto.library.qbytes_mm; assert torch.cuda.get_device_capability(0)==(7,0)'
if [[ -z "${CONFIG_TEMPLATE:-}" ]]; then
  for p in "$REPO_ROOT/wgp_config.json" /opt/ai/wan2gp/wgp_config.json /cache/ltx25/config-r0_quality_1024/wgp_config.json; do
    if [[ -s "$p" ]]; then CONFIG_TEMPLATE="$p"; break; fi
  done
fi
[[ -n "${CONFIG_TEMPLATE:-}" && -s "$CONFIG_TEMPLATE" ]] || {
  echo "Set CONFIG_TEMPLATE=/path/to/existing/wgp_config.json" >&2; exit 1;
}
CONFIG_TEMPLATE="$(realpath -e "$CONFIG_TEMPLATE")"
for cmd in nvcc ffmpeg ffprobe nvidia-smi; do
  command -v "$cmd" >/dev/null || { echo "Missing $cmd" >&2; exit 1; }
done

RUN_ROOT="${RUN_ROOT:-$HOME/ai/ltx25-fp16-i2v-3way-results/$(date -u +%Y%m%d-%H%M%S)-$$}"
[[ ! -e "$RUN_ROOT" || -z "$(ls -A "$RUN_ROOT" 2>/dev/null)" ]] || {
  echo "Output directory must be new/empty: $RUN_ROOT" >&2; exit 1;
}
mkdir -p "$RUN_ROOT"/{settings,config,videos,results}
RUN_ROOT="$(realpath -e "$RUN_ROOT")"
cd "$REPO_ROOT"

check_gpu_idle() {
  [[ "${CHECK_GPU_IDLE:-1}" == 1 ]] || return 0
  local busy
  busy="$(nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader | grep -E '^[[:space:]]*[0-9]+[[:space:]]*,' || true)"
  [[ -z "$busy" ]] || { echo "GPU busy; refusing invalid benchmark: $busy" >&2; return 1; }
}
check_gpu_idle

EXTENSION="$RUN_ROOT/libltx25_cublas_e_sm70.so"
echo "=== Compile Candidate E (excluded from timings) ==="
nvcc -O3 -std=c++17 -arch=sm_70 -shared -Xcompiler -fPIC \
  benchmarks/ltx25_convrot/ltx25_cublas_e.cu -lcublas -o "$EXTENSION"
[[ -s "$EXTENSION" ]] || exit 1

SETTINGS="$RUN_ROOT/settings/input.json"
"$PY" - "$SETTINGS" "$IMAGE" "$PROMPT" "$NEGATIVE_PROMPT" "$LORA" \
  "$LORA_STRENGTH" "$SEED" "$STEPS" "$FRAMES" "$RESOLUTION" \
  "$NAG_SCALE" "$AUDIO_CFG_SCALE" <<'PYSET'
import json,sys
from pathlib import Path
out,image,prompt,negative,lora,strength,seed,steps,frames,resolution,nag,audio=sys.argv[1:]
data={
    "model_type":"ltx2_25_22B_distilled",
    "prompt":prompt,"negative_prompt":negative,
    "NAG_scale":float(nag),"audio_cfg_scale":float(audio),
    "image_start":image,"image_prompt_type":"S","image_mode":0,
    "activated_loras":[] if lora in ("","none") else [lora],
    "loras_multipliers":"" if lora in ("","none") else strength,
    "seed":int(seed),"num_inference_steps":int(steps),
    "video_length":int(frames),"resolution":resolution,
    "audio_prompt_type":"","video_prompt_type":"",
    "spatial_upsampling":"","temporal_upsampling":"",
}
Path(out).write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n")
PYSET
{
  echo "repo=$REPO_ROOT"
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse HEAD)"
  echo "input_image=$IMAGE"
  echo "input_image_sha256=$(sha256sum "$IMAGE" | cut -d' ' -f1)"
  echo "settings_sha256=$(sha256sum "$SETTINGS" | cut -d' ' -f1)"
  echo "lora=$LORA; lora_strength=$LORA_STRENGTH"
  echo "frames=$FRAMES; resolution=$RESOLUTION; steps=$STEPS; seed=$SEED"
  echo "repeats=$REPEATS"
  echo "backend=direct; quanto_sham_control=1; fp16_min_m=1; min_dim=64"
  echo "ops=qbytes_mm; guard=first; audit_per_shape=0"
  echo "candidate_e=fp32rot/radix4/vector8/direct/fused/workspace96"
  echo "python=$PY"
  echo "started_utc=$(date -u --iso-8601=seconds)"
} > "$RUN_ROOT/run-manifest.txt"

# Same Candidate E path for all cases. FP16 flags are the ONLY arm difference.
run_wangp() (
  local case_name="$1" index="$2"; shift 2
  unset WAN2GP_LTX25_DEFAULT_ON WAN2GP_LTX25_REAL_ACTIVATION_AUDIT WAN2GP_LTX25_PROFILE_STAGES
  unset WAN2GP_LTX25_HADAMARD WAN2GP_LTX25_E_CAST WAN2GP_LTX25_GEMM_TUNE
  unset WAN2GP_LTX25_ROT_INPUT WAN2GP_LTX25_E_OUTPUT WAN2GP_LTX25_E_WORKSPACE
  unset WAN2GP_LTX25_CUBLAS_E_MODE LTX25_E_WORKSPACE_MAX_MIB LTX25_CUBLAS_E_LIB
  unset WAN2GP_LTX25_PIPELINE_PROFILE WAN2GP_LTX2_BENCH_TRANSFORMER
  unset WAN2GP_LTX25_FP16_GEMM LTX25_FP16_GEMM_GROUPS LTX25_FP16_GEMM_OPS
  unset LTX25_FP16_GEMM_SHAM LTX25_FP16_GEMM_BACKEND LTX25_FP16_GEMM_GUARD
  unset LTX25_FP16_GEMM_AUDIT_PER_SHAPE LTX25_FP16_GEMM_MIN_M LTX25_FP16_GEMM_MIN_DIM
  unset LTX25_FP16_GEMM_ALLOW_SHAPES LTX25_FP16_GEMM_SHAPE_CENSUS
  export WAN2GP_LTX25_CUBLAS_E_MODE=fp32rot WAN2GP_LTX25_HADAMARD=radix4
  export WAN2GP_LTX25_E_CAST=vector8 WAN2GP_LTX25_GEMM_TUNE=off
  export WAN2GP_LTX25_E_WORKSPACE=1 LTX25_E_WORKSPACE_MAX_MIB=96
  export WAN2GP_LTX25_ROT_INPUT=direct WAN2GP_LTX25_E_OUTPUT=fused
  export LTX25_CUBLAS_E_LIB="$EXTENSION"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WAN2GP_LTX2_SKIP_BAKED_LORA_MIGRATION=1
  export GRADIO_ANALYTICS_ENABLED=False OMP_NUM_THREADS=8
  export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  export WAN2GP_LTX25_FP16_GEMM=1 LTX25_FP16_GEMM_BACKEND=direct
  export LTX25_FP16_GEMM_OPS=qbytes_mm
  export LTX25_FP16_GEMM_MIN_M=1 LTX25_FP16_GEMM_MIN_DIM=64
  export LTX25_FP16_GEMM_GUARD=first LTX25_FP16_GEMM_AUDIT_PER_SHAPE=0
  export LTX25_FP16_GEMM_STATS_PATH="$RUN_ROOT/results/$case_name-$index-fp16-stats.json"
  case "$case_name" in
    sham)
      export LTX25_FP16_GEMM_SHAM=1
      export LTX25_FP16_GEMM_GROUPS=stage1:wide,stage1:tall,stage1:balanced,stage2:wide,stage2:tall,stage2:balanced
      ;;
    s2-all)
      export LTX25_FP16_GEMM_SHAM=0
      export LTX25_FP16_GEMM_GROUPS=stage2:wide,stage2:tall,stage2:balanced
      ;;
    both-all)
      export LTX25_FP16_GEMM_SHAM=0
      export LTX25_FP16_GEMM_GROUPS=stage1:wide,stage1:tall,stage1:balanced,stage2:wide,stage2:tall,stage2:balanced
      ;;
    *) echo "Unexpected arm: $case_name" >&2; exit 2 ;;
  esac
  exec "$PY" "$@"
)

if [[ "${SKIP_PREFLIGHT:-0}" != 1 ]]; then
  echo "=== Direct Quanto V100 numerical preflight (not timed) ==="
  PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" \
    benchmarks/ltx25_convrot/fp16_quanto_direct_preflight.py \
    2>&1 | tee "$RUN_ROOT/results/direct-quanto-preflight.log"
  grep -Fq 'LTX25_QUANTO_DIRECT_PREFLIGHT_PASS ' "$RUN_ROOT/results/direct-quanto-preflight.log" || {
    echo "Preflight failed; refusing real I2V runs" >&2; exit 1;
  }
fi

declare -A COUNTS=( [sham]=0 [s2-all]=0 [both-all]=0 )
ORDER=(sham s2-all both-all)
if [[ "$REPEATS" == 2 ]]; then ORDER+=(both-all s2-all sham); fi

run_one() {
  local case_name="$1" index="$2"
  local tag="$case_name-$index"
  local cfg="$RUN_ROOT/config/$tag" out="$RUN_ROOT/videos/$tag"
  local log="$RUN_ROOT/results/$tag-generation.log"
  mkdir -p "$cfg" "$out"
  cp "$CONFIG_TEMPLATE" "$cfg/wgp_config.json"
  local -a args=(wgp.py --offline --config "$cfg" --process "$SETTINGS" \
    --output-dir "$out" --profile 0 --attention sdpa \
    --no-prevent-power-throttling --verbose 1)
  echo "=== $tag dry-run validation ==="
  if ! run_wangp "$case_name" "$index" "${args[@]}" --dry-run \
       > "$RUN_ROOT/results/$tag-dryrun.log" 2>&1; then
    tail -75 "$RUN_ROOT/results/$tag-dryrun.log" >&2; return 1
  fi
  grep -Fq 'Validation complete. 1/1 task(s) valid.' \
    "$RUN_ROOT/results/$tag-dryrun.log" || {
      echo "Validation marker missing: $tag" >&2; return 1;
    }
  check_gpu_idle
  echo "=== $tag: GENERATE $FRAMES frames at $RESOLUTION ==="
  local begin end
  begin="$(date +%s.%N)"
  if run_wangp "$case_name" "$index" "${args[@]}" 2>&1 | tee "$log"; then
    :
  else
    echo "Generation failed: $tag; see $log" >&2
    return 1
  fi
  end="$(date +%s.%N)"
  grep -Fq 'Task 1 completed' "$log" || {
    echo "Completion marker missing for $tag" >&2; return 1;
  }
  awk -v a="$begin" -v b="$end" 'BEGIN {printf "%.3f\n",b-a}' \
    > "$RUN_ROOT/results/$tag-seconds.txt"
  local -a clips=()
  mapfile -d '' -t clips < <(find "$out" -type f -name '*.mp4' -size +0c -print0)
  [[ "${#clips[@]}" == 1 ]] || {
    echo "Expected one MP4 for $tag; found ${#clips[@]}" >&2; return 1;
  }
  cp "${clips[0]}" "$RUN_ROOT/$tag.mp4"
  if [[ "$index" == 1 ]]; then
    ln "$RUN_ROOT/$tag.mp4" "$RUN_ROOT/$case_name.mp4" 2>/dev/null ||
      cp "$RUN_ROOT/$tag.mp4" "$RUN_ROOT/$case_name.mp4"
  fi
  "$PY" - "$RUN_ROOT/results/$tag-fp16-stats.json" "$RUN_ROOT/results/$tag-seconds.txt" \
    "$RUN_ROOT/results/$tag.json" "$case_name" "$index" "$RUN_ROOT/$tag.mp4" \
    "$SETTINGS" "$log" <<'PYRUN'
import hashlib,json,sys
from pathlib import Path
stats_file,seconds_file,out,case,idx,clip,settings,log=map(Path,sys.argv[1:4]) if False else sys.argv[1:]
stats=json.loads(Path(stats_file).read_text())
assert stats.get("backend")=="direct",stats
assert stats.get("persistent_fp16_weight_copies")==0,stats
c=stats["stats"]
observed=int(c.get("observed_qbytes_mm",0))
converted=int(c.get("converted_qbytes_mm",0))
sham_native=int(c.get("sham_native",0))
assert observed>0, ("No Quanto ops observed",case,c)
if case=="sham":
    assert converted==0 and sham_native>0, (case,c)
else:
    assert converted>0 and sham_native==0, (case,c)
assert not stats.get("blocked_shapes"), (case,stats.get("blocked_shapes"))
text=Path(log).read_text(errors="replace")
assert "[LTX25_CUBLAS_E] {" in text, ("Candidate E stats missing",case)
result={
    "variant":case,"index":int(idx),
    "wall_seconds":float(Path(seconds_file).read_text()),
    "clip":str(clip),
    "settings_sha256":hashlib.sha256(Path(settings).read_bytes()).hexdigest(),
    "observed_qbytes_mm":observed,
    "converted_qbytes_mm":converted,"sham_native_qbytes_mm":sham_native,
    "coverage_percent":round(100*converted/observed,3),
    "guard_input_fallback":int(c.get("guard_input_reference",0)),
    "guard_output_fallback":int(c.get("guard_output_reference",0)),
    "finite_guard_checks":int(c.get("guard_checks",0)),
    "candidate_e_enabled":True,
    "audit_per_shape":stats["audit_per_shape"],
}
assert result["audit_per_shape"]==0, result
Path(out).write_text(json.dumps(result,indent=2)+"\n")
print("I2V_CASE_OK",json.dumps(result,sort_keys=True),flush=True)
PYRUN
}

for case_name in "${ORDER[@]}"; do
  n=$(( COUNTS[$case_name] + 1 ))
  COUNTS[$case_name]="$n"
  run_one "$case_name" "$n"
done

# Aggregate timings and assess quality at decoded-video/audio level. This part
# is outside timing, including all ffmpeg decoding and side-by-side PNG export.
"$PY" - "$RUN_ROOT" <<'PYREPORT'
import csv,hashlib,json,math,re,statistics,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]); res=root/"results"
settings=json.loads((root/"settings/input.json").read_text())
frames=int(settings["video_length"])
cases=("sham","s2-all","both-all")
rows=[]
for case in cases:
    for path in sorted(res.glob(f"{case}-[0-9].json")):
        rows.append(json.loads(path.read_text()))
assert len(rows)>=3

def ffmpeg(*args):
    p=subprocess.run(("ffmpeg","-hide_banner","-nostdin","-y",*map(str,args)),
                      capture_output=True,text=True,timeout=1800)
    if p.returncode:
        raise RuntimeError(f"ffmpeg failed: {p.stderr[-1200:]}")
    return p.stderr

def pcm_hash(clip):
    cmd=["ffmpeg","-v","error","-nostdin","-i",str(clip),
         "-map","0:a:0","-vn","-c:a","pcm_s16le","-f","s16le","pipe:1"]
    proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    h=hashlib.sha256(); n=0
    try:
        while True:
            chunk=proc.stdout.read(1024*1024)
            if not chunk: break
            h.update(chunk); n+=len(chunk)
        stderr=proc.stderr.read()
        if proc.wait(timeout=120) or not n:
            raise RuntimeError(f"No decoded audio: {clip}: {stderr[-300:]!r}")
        return h.hexdigest()
    finally:
        if proc.poll() is None: proc.kill();proc.wait()

def frame_count(clip):
    p=subprocess.run(["ffprobe","-v","error","-count_frames","-select_streams","v:0",
        "-show_entries","stream=nb_read_frames","-of","default=nw=1:nk=1",str(clip)],
        capture_output=True,text=True,check=True,timeout=300)
    return int(p.stdout.strip())

# Verify each clip is complete and compute reproducibility hashes.
for row in rows:
    clip=Path(row["clip"])
    assert clip.is_file() and clip.stat().st_size>10000,clip
    count=frame_count(clip)
    assert count==frames,(clip,count,frames)
    row["decoded_frames"]=count
    row["pcm_audio_sha256"]=pcm_hash(clip)

ref=root/"sham-1.mp4"
ref_audio=next(r["pcm_audio_sha256"] for r in rows if r["variant"]=="sham" and r["index"]==1)
reports={}
out_png=res/"worst-frames";out_png.mkdir(exist_ok=True)
for case in ("s2-all","both-all"):
    cand=root/f"{case}-1.mp4"
    psnr_file=res/f"{case}-frame-psnr.txt"
    stderr=ffmpeg("-v","info","-i",ref,"-i",cand,
        "-filter_complex",f"[0:v:0]format=yuv420p[r];[1:v:0]format=yuv420p[c];[r][c]psnr=stats_file={psnr_file}",
        "-f","null","-")
    # Final PSNR is printed as average:XX in FFmpeg diagnostics.
    m=re.search(r"PSNR y:.*?average:([\d.+-]+|inf)",stderr,re.I)
    if not m: raise RuntimeError(f"Unable to parse PSNR for {case}: {stderr[-1300:]}")
    psnr=float(m.group(1))
    ssim_log=res/f"{case}-frame-ssim.txt"
    stderr=ffmpeg("-v","info","-i",ref,"-i",cand,
        "-filter_complex",f"[0:v:0]format=yuv420p[r];[1:v:0]format=yuv420p[c];[r][c]ssim=stats_file={ssim_log}",
        "-f","null","-")
    m=re.search(r"SSIM Y:.*?All:([\d.]+)",stderr)
    if not m: raise RuntimeError(f"Unable to parse SSIM for {case}: {stderr[-1300:]}")
    ssim=float(m.group(1))
    perframe=[]
    for line in psnr_file.read_text().splitlines():
        z=re.search(r"\bn:\s*(\d+).*?\bmse_avg:\s*([\d.]+).*?\bpsnr_avg:\s*([\d.]+|inf)",line)
        if z: perframe.append({"frame_1based":int(z.group(1)),
                               "mse":float(z.group(2)),"psnr":float(z.group(3))})
    assert len(perframe)==frames,(case,len(perframe),frames)
    worst=sorted(perframe,key=lambda v:v["mse"],reverse=True)[:3]
    for d in worst:
        # 0-based frame selector, and native reference always on the LEFT.
        idx=d["frame_1based"]-1
        png=out_png/f"{case}-frame-{d['frame_1based']:03d}-side-by-side.png"
        vf=f"[0:v]select='eq(n,{idx})',setpts=PTS-STARTPTS[a];[1:v]select='eq(n,{idx})',setpts=PTS-STARTPTS[b];[a][b]hstack=inputs=2"
        ffmpeg("-v","error","-i",ref,"-i",cand,"-filter_complex",vf,
               "-frames:v","1",png)
        assert png.is_file() and png.stat().st_size>1000,png
    reports[case]={"reference":"sham-1.mp4","candidate":f"{case}-1.mp4",
        "psnr_db":round(psnr,6),"ssim":round(ssim,6),
        "worst_three_frames":worst,
        "audio_exact":next(x["pcm_audio_sha256"] for x in rows if x["variant"]==case and x["index"]==1)==ref_audio,
        "per_frame_psnr_file":str(psnr_file)}
    (res/f"quality-{case}.json").write_text(json.dumps(reports[case],indent=2)+"\n")

for row in rows:
    row["audio_exact_to_sham"]=(row["pcm_audio_sha256"]==ref_audio)

fields=["variant","index","wall_seconds","observed_qbytes_mm",
        "converted_qbytes_mm","coverage_percent","guard_input_fallback",
        "guard_output_fallback","decoded_frames","audio_exact_to_sham","clip"]
with (root/"benchmark.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
    w.writeheader();w.writerows(rows)
bycase={case:[r["wall_seconds"] for r in rows if r["variant"]==case] for case in cases}
ref_mean=statistics.mean(bycase["sham"])
summary={"frames":frames,"resolution":settings["resolution"],"steps":settings["num_inference_steps"],
         "settings_sha256":rows[0]["settings_sha256"],
         "repeats":len(bycase["sham"]),
         "measurements":rows,"quality":reports,"timing":{}}
for case in cases:
    vals=bycase[case]
    summary["timing"][case]={
        "runs_seconds":vals,"mean_seconds":round(statistics.mean(vals),3),
        "median_seconds":round(statistics.median(vals),3),
        "sham_mean_seconds":round(ref_mean,3),
        "mean_reduction_percent_vs_sham":round(100*(ref_mean-statistics.mean(vals))/ref_mean,3),
        "mean_seconds_saved_vs_sham":round(ref_mean-statistics.mean(vals),3)}
(root/"quality-summary.json").write_text(json.dumps(summary,indent=2)+"\n")
lines=["LTX-2.5 INT8 ConvRot I2V 241-frame direct Quanto 3-arm comparison",
       "Reference: BF16 sham with same Quanto-only interception overhead",
       f"Frames={frames} resolution={settings['resolution']} steps={settings['num_inference_steps']}",
       "All timings include model startup, model loading and video encoding.",
       "No numerical audit during timed runs. PSNR and SSIM computed offline.",
       "",
       f"{'Arm':<12} {'mean secs':>11} {'vs sham':>10} {'FP16 calls':>12} {'coverage':>10} {'PSNR dB':>10} {'SSIM':>9}"]
for case in cases:
    t=summary["timing"][case]
    row=next(r for r in rows if r["variant"]==case and r["index"]==1)
    q=reports.get(case,{})
    lines.append(f"{case:<12} {t['mean_seconds']:>11.2f} {t['mean_reduction_percent_vs_sham']:>9.2f}% "
                 f"{row['converted_qbytes_mm']:>12} {row['coverage_percent']:>9.1f}% "
                 f"{q.get('psnr_db',float('nan')):>10.2f} {q.get('ssim',float('nan')):>9.5f}")
lines.extend(["","Review original videos in the RUN_ROOT directory.",
              "Full per-run data: benchmark.csv and quality-summary.json.",
              "Worst-frame lossless comparisons: results/worst-frames/*.png (left sham/right FP16).",
              "GEMM coverage denominator = observed Quanto qbytes_mm GEMMs, NOT all operations.",
              "PSNR/SSIM measure similarity, not subjective image quality.",
              "The sham arm is instrumented BF16; no uninstrumented native BF16 arm.",
              "Results remain experimental and are not promoted to production."])
(root/"summary.txt").write_text("\n".join(lines)+"\n")
print("\n".join(lines),flush=True)
PYREPORT
echo "=== COMPLETE: $RUN_ROOT ==="
echo "Videos: $RUN_ROOT/{sham,s2-all,both-all}.mp4"
echo "Timing and FP16 coverage: $RUN_ROOT/benchmark.csv"
echo "Decoded video/audio quality and sample worst frames: $RUN_ROOT/quality-summary.json"
