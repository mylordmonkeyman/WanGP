#!/usr/bin/env bash
# Matched 241-frame LTX-2.5 I2V + LoRA Candidate E fusion test on V100.
# Baseline: radix4 + vector8; Fused: same with direct radix4 input and fused output.
# Both arms use one identical settings JSON. No stock WanGP arm in this test.
set -euo pipefail

usage() {
cat <<'HELP'
Usage:
  ./run_candidate_e_fusions_i2v_241.sh IMAGE 'POSITIVE PROMPT' 'NEGATIVE PROMPT'

Defaults: 241 frames, 1024x1024, 8 steps, seed 20261112, one baseline
and one fused run, the LTX_10Eros LoRA at strength 1.0.
Optional: ABBA=1 runs baseline-fused-fused-baseline (four generations).
Optional: NAG_SCALE=1.1 AUDIO_CFG_SCALE=1.5 LORA=none SEED=...
          FRAMES=241 RESOLUTION=1024x1024 STEPS=8
          PY=/path/to/venv/bin/python CONFIG_TEMPLATE=/path/to/wgp_config.json
          RUN_ROOT=/path/to/new/directory CHECK_GPU_IDLE=1
Produces baseline.mp4, fused.mp4, run-manifest.txt, results/*.csv,
and video-audio-equality.json. Exact decoded frame/audio equality is REQUIRED.
HELP
}

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi
if (( $# != 3 )); then usage >&2; exit 2; fi
IMAGE="$1"; PROMPT="$2"; NEGATIVE_PROMPT="$3"
[[ -n "$PROMPT" && -n "$NEGATIVE_PROMPT" ]] || { echo "Prompts must be nonempty" >&2; exit 2; }
REPO_ROOT="${REPO_ROOT:-$HOME/ai/wangp-hadamard}"
REPO_ROOT="$(realpath -e "$REPO_ROOT")"
[[ -s "$REPO_ROOT/wgp.py" ]] || { echo "Not a WanGP checkout: $REPO_ROOT" >&2; exit 1; }
for f in shared/kernels/ltx25_cublas_sm70.py benchmarks/ltx25_convrot/ltx25_cublas_e.cu benchmarks/ltx25_convrot/rot_direct_preflight.py; do
  [[ -s "$REPO_ROOT/$f" ]] || { echo "Missing $f. Pull perf/ltx25-convrot-sm70 first." >&2; exit 1; }
done
grep -Fq 'WAN2GP_LTX25_ROT_INPUT' "$REPO_ROOT/shared/kernels/ltx25_cublas_sm70.py" || { echo "Direct input code missing" >&2; exit 1; }
grep -Fq 'WAN2GP_LTX25_E_OUTPUT' "$REPO_ROOT/shared/kernels/ltx25_cublas_sm70.py" || { echo "Fused epilogue code missing" >&2; exit 1; }

IMAGE="$(realpath -e "$IMAGE")"
[[ -s "$IMAGE" ]] || { echo "Image missing: $IMAGE" >&2; exit 1; }
LORA="${LORA:-/var/lib/llama/models/ltx-2.5/lora/LTX_10Eros_LoRA_Wan2GPfix_fro99-avgrank77.safetensors}"
LORA_STRENGTH="${LORA_STRENGTH:-1}"
if [[ "$LORA" != none && -n "$LORA" ]]; then
  LORA="$(realpath -e "$LORA")"
  [[ -s "$LORA" ]] || { echo "LoRA missing: $LORA" >&2; exit 1; }
fi
SEED="${SEED:-20261112}"
FRAMES="${FRAMES:-241}"
STEPS="${STEPS:-8}"
RESOLUTION="${RESOLUTION:-1024x1024}"
NAG_SCALE="${NAG_SCALE:-1.1}"
AUDIO_CFG_SCALE="${AUDIO_CFG_SCALE:-1.5}"
ABBA="${ABBA:-0}"
[[ "$SEED" =~ ^[0-9]+$ && "$FRAMES" =~ ^[1-9][0-9]*$ && "$STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid SEED, FRAMES or STEPS" >&2; exit 2; }
[[ "$RESOLUTION" =~ ^[1-9][0-9]*x[1-9][0-9]*$ ]] || { echo "Invalid RESOLUTION" >&2; exit 2; }
[[ "$ABBA" == 0 || "$ABBA" == 1 ]] || { echo "ABBA must be 0 or 1" >&2; exit 2; }
for val in "$NAG_SCALE" "$AUDIO_CFG_SCALE" "$LORA_STRENGTH"; do
  [[ "$val" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "Invalid numeric setting: $val" >&2; exit 2; }
done

# DO NOT realpath a Python venv executable: its symlink retains pyvenv.cfg.
if [[ -z "${PY:-}" ]]; then
  for p in /opt/ai/wan2gp/.venv/bin/python "$REPO_ROOT/.venv/bin/python" /opt/venvs/ltx25-cu128/bin/python; do
    if [[ -x "$p" ]] && "$p" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.get_device_capability(0)==(7,0)' >/dev/null 2>&1; then
      PY="$p"; break
    fi
  done
fi
[[ -n "${PY:-}" && -x "$PY" ]] || { echo "Set PY=/path/to/working/venv/bin/python" >&2; exit 1; }
PY="$(cd "$(dirname "$PY")" && pwd -P)/$(basename "$PY")"
"$PY" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.get_device_capability(0)==(7,0)' || exit 1
if [[ -z "${CONFIG_TEMPLATE:-}" ]]; then
  for p in "$REPO_ROOT/wgp_config.json" /opt/ai/wan2gp/wgp_config.json /cache/ltx25/config-r0_quality_1024/wgp_config.json; do
    if [[ -s "$p" ]]; then CONFIG_TEMPLATE="$p"; break; fi
  done
fi
[[ -n "${CONFIG_TEMPLATE:-}" && -s "$CONFIG_TEMPLATE" ]] || { echo "Set CONFIG_TEMPLATE to an existing wgp_config.json" >&2; exit 1; }
CONFIG_TEMPLATE="$(realpath -e "$CONFIG_TEMPLATE")"
command -v nvcc >/dev/null || { echo "nvcc unavailable" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi unavailable" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg unavailable (needed for decoded frame/audio checks)" >&2; exit 1; }
command -v ffprobe >/dev/null || { echo "ffprobe unavailable" >&2; exit 1; }

RUN_ROOT="${RUN_ROOT:-$HOME/ai/candidate-e-fusions-i2v-results/$(date -u +%Y%m%d-%H%M%S)-$$}"
if [[ -e "$RUN_ROOT" && -n "$(ls -A "$RUN_ROOT" 2>/dev/null)" ]]; then
  echo "RUN_ROOT must be new/empty: $RUN_ROOT" >&2; exit 1
fi
mkdir -p "$RUN_ROOT"/{settings,config,videos,results}
RUN_ROOT="$(realpath -e "$RUN_ROOT")"
cd "$REPO_ROOT"

check_gpu_idle() {
  [[ "${CHECK_GPU_IDLE:-1}" == 1 ]] || return 0
  local apps
  apps="$(nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader)"
  if grep -Eq '^[[:space:]]*[0-9]+[[:space:]]*,' <<<"$apps"; then
    echo "V100 is occupied; refusing benchmark:" >&2
    echo "$apps" >&2
    return 1
  fi
}
check_gpu_idle

EXTENSION="$RUN_ROOT/libltx25_cublas_e_sm70.so"
echo "=== Compiling Candidate E CUDA extension (not timed) ==="
nvcc -O3 -std=c++17 -arch=sm_70 -shared -Xcompiler -fPIC benchmarks/ltx25_convrot/ltx25_cublas_e.cu -lcublas -o "$EXTENSION"
[[ -s "$EXTENSION" ]] || { echo "CUDA extension build failed" >&2; exit 1; }

SETTINGS="$RUN_ROOT/settings/input.json"
"$PY" - "$SETTINGS" "$IMAGE" "$PROMPT" "$NEGATIVE_PROMPT" "$LORA" "$LORA_STRENGTH" "$SEED" "$STEPS" "$FRAMES" "$RESOLUTION" "$NAG_SCALE" "$AUDIO_CFG_SCALE" <<'PYSET'
import json,sys
from pathlib import Path
out,image,prompt,negative,lora,strength,seed,steps,frames,resolution,nag,audio=sys.argv[1:]
data={
    "model_type":"ltx2_25_22B_distilled", "prompt":prompt, "negative_prompt":negative,
    "NAG_scale":float(nag), "audio_cfg_scale":float(audio),
    "image_start":image, "image_prompt_type":"S", "image_mode":0,
    "activated_loras":[] if lora in ("", "none") else [lora],
    "loras_multipliers":"" if lora in ("","none") else strength,
    "seed":int(seed), "num_inference_steps":int(steps),
    "video_length":int(frames), "resolution":resolution,
    "audio_prompt_type":"", "video_prompt_type":"",
    "spatial_upsampling":"", "temporal_upsampling":""
}
Path(out).write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n")
PYSET

{
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse HEAD)"
  echo "input_image=$IMAGE"
  echo "image_sha256=$(sha256sum "$IMAGE" | cut -d' ' -f1)"
  echo "lora=$LORA"
  echo "lora_strength=$LORA_STRENGTH"
  echo "settings_sha256=$(sha256sum "$SETTINGS" | cut -d' ' -f1)"
  echo "cuda_binary_sha256=$(sha256sum "$EXTENSION" | cut -d' ' -f1)"
  echo "frames=$FRAMES; resolution=$RESOLUTION; steps=$STEPS; seed=$SEED"
  echo "nag_scale=$NAG_SCALE; audio_cfg_scale=$AUDIO_CFG_SCALE"
  if [[ "${FP16_GEMM_SWEEP_ONLY:-0}" == "1" ]]; then
    echo "arms=BF16-reference(direct,fused) plus modular experimental FP16 GEMM selectors"
    echo "sweep_cases=${SWEEP_CASES:-bf16-reference,fp16-s1-wide,fp16-s1-tall,fp16-s1-balanced,fp16-s1-all,fp16-s2-wide,fp16-s2-tall,fp16-s2-balanced,fp16-s2-all,fp16-both-all}"
    echo "sweep_guard=${SWEEP_GUARD:-first}; sweep_ops=${SWEEP_OPS:-mm,addmm}"
  elif [[ "${PROFILE_PIPELINE_ONLY:-0}" == "1" ]]; then
    echo "arms=pipeline(direct,fused,whole_pipeline_phases+two_operator_samples); ABBA=not_applicable"
  elif [[ "${PROFILE_ONLY:-0}" == "1" ]]; then
    echo "arms=profile(direct,fused,live_cuda_events); ABBA=not_applicable"
  else
    echo "arms=baseline(staged,torch) fused(direct,fused); ABBA=$ABBA"
  fi
  echo "python=$PY"
  echo "started_utc=$(date -u --iso-8601=seconds)"
} > "$RUN_ROOT/results/run-manifest.txt"

# Explicitly discard inherited flags; modes differ ONLY in the two new switches.
run_wangp() (
  local variant="$1"; shift
  unset WAN2GP_LTX25_DEFAULT_ON WAN2GP_LTX25_REAL_ACTIVATION_AUDIT WAN2GP_LTX25_PROFILE_STAGES
  unset WAN2GP_LTX25_HADAMARD WAN2GP_LTX25_E_CAST WAN2GP_LTX25_GEMM_TUNE
  unset WAN2GP_LTX25_ROT_INPUT WAN2GP_LTX25_E_OUTPUT WAN2GP_LTX25_E_WORKSPACE
  unset WAN2GP_LTX25_CUBLAS_E_MODE LTX25_E_WORKSPACE_MAX_MIB LTX25_CUBLAS_E_LIB
  unset WAN2GP_LTX25_PIPELINE_PROFILE WAN2GP_LTX2_BENCH_TRANSFORMER LTX25_PIPELINE_PROFILE_PATH
  unset WAN2GP_LTX25_FP16_GEMM LTX25_FP16_GEMM_GROUPS LTX25_FP16_GEMM_STATS_PATH
  unset LTX25_FP16_GEMM_GUARD LTX25_FP16_GEMM_OPS
  export WAN2GP_LTX25_CUBLAS_E_MODE=fp32rot WAN2GP_LTX25_HADAMARD=radix4
  export WAN2GP_LTX25_E_CAST=vector8 WAN2GP_LTX25_GEMM_TUNE=off
  export WAN2GP_LTX25_E_WORKSPACE=1 LTX25_E_WORKSPACE_MAX_MIB=96
  export LTX25_CUBLAS_E_LIB="$EXTENSION"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WAN2GP_LTX2_SKIP_BAKED_LORA_MIGRATION=1
  export GRADIO_ANALYTICS_ENABLED=False OMP_NUM_THREADS=8
  # Direct execution of benchmarks/ltx25_convrot/*.py otherwise omits the repository root from sys.path.
  export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  if [[ "$variant" == baseline ]]; then
    export WAN2GP_LTX25_ROT_INPUT=staged WAN2GP_LTX25_E_OUTPUT=torch
  else
    export WAN2GP_LTX25_ROT_INPUT=direct WAN2GP_LTX25_E_OUTPUT=fused
  fi
  if [[ "$variant" == profile ]]; then
    # The profiler synchronizes CUDA events only at interpreter exit.
    export WAN2GP_LTX25_PROFILE_STAGES=1
    export LTX25_STAGE_PROFILE_PATH="$RUN_ROOT/results/stage_profile.json"
  fi
  if [[ "$variant" == fp16-* ]]; then
    local selectors=""
    case "$variant" in
      fp16-s1-wide) selectors="stage1:wide" ;;
      fp16-s1-tall) selectors="stage1:tall" ;;
      fp16-s1-balanced) selectors="stage1:balanced" ;;
      fp16-s1-all) selectors="stage1:wide,stage1:tall,stage1:balanced" ;;
      fp16-s2-wide) selectors="stage2:wide" ;;
      fp16-s2-tall) selectors="stage2:tall" ;;
      fp16-s2-balanced) selectors="stage2:balanced" ;;
      fp16-s2-all) selectors="stage2:wide,stage2:tall,stage2:balanced" ;;
      fp16-both-all) selectors="stage1:wide,stage1:tall,stage1:balanced,stage2:wide,stage2:tall,stage2:balanced" ;;
      *) echo "Unknown modular FP16 GEMM case: $variant" >&2; return 2 ;;
    esac
    export WAN2GP_LTX25_FP16_GEMM=1
    export LTX25_FP16_GEMM_GROUPS="$selectors"
    export LTX25_FP16_GEMM_STATS_PATH="$RUN_ROOT/results/$variant-fp16-stats.json"
    export LTX25_FP16_GEMM_GUARD="${SWEEP_GUARD:-first}"
    export LTX25_FP16_GEMM_OPS="${SWEEP_OPS:-mm,addmm}"
    export LTX25_FP16_GEMM_MIN_M="${SWEEP_MIN_M:-128}"
    export LTX25_FP16_GEMM_MIN_DIM="${SWEEP_MIN_DIM:-512}"
  fi
  if [[ "$variant" == pipeline ]]; then
    # Instrument only the actual generation, never the dry-run validation.
    local arg is_dryrun=0
    for arg in "$@"; do
      if [[ "$arg" == --dry-run ]]; then is_dryrun=1; fi
    done
    if [[ "$is_dryrun" == 0 ]]; then
      export WAN2GP_LTX25_PIPELINE_PROFILE=1
      export WAN2GP_LTX2_BENCH_TRANSFORMER=1
      export LTX25_PIPELINE_PROFILE_PATH="$RUN_ROOT/results/pipeline_profile.json"
      export LTX25_PIPELINE_OP_SAMPLE="${LTX25_PIPELINE_OP_SAMPLE:-1}"
    fi
  fi
  exec "$PY" "$@"
)

if [[ "${SKIP_PREFLIGHT:-0}" != 1 ]]; then
  echo "=== Synthetic exact-output preflight ==="
  run_wangp fused benchmarks/ltx25_convrot/rot_direct_preflight.py 2>&1 | tee "$RUN_ROOT/results/preflight.log"
  grep -Fq 'LTX25_ROT_DIRECT_PREFLIGHT_OK ' "$RUN_ROOT/results/preflight.log" || {
    echo "Exactness preflight failed" >&2; exit 1
  }
fi

run_one() {
  local variant="$1" index="$2"
  local label="$variant-$index" cfg="$RUN_ROOT/config/$variant-$index"
  local out="$RUN_ROOT/videos/$variant-$index"
  local log="$RUN_ROOT/results/$variant-$index-generation.log"
  mkdir -p "$cfg" "$out"
  cp "$CONFIG_TEMPLATE" "$cfg/wgp_config.json"
  local -a args=(wgp.py --offline --config "$cfg" --process "$SETTINGS" --output-dir "$out" --profile 0 --attention sdpa --no-prevent-power-throttling --verbose 1)
  echo "=== $label validating settings ==="
  run_wangp "$variant" "${args[@]}" --dry-run > "$RUN_ROOT/results/$label-dryrun.log" 2>&1 || {
    tail -70 "$RUN_ROOT/results/$label-dryrun.log" >&2; return 1
  }
  grep -Fq 'Validation complete. 1/1 task(s) valid.' "$RUN_ROOT/results/$label-dryrun.log" || {
    echo "Dry-run validation marker missing for $label" >&2; return 1
  }
  check_gpu_idle
  echo "=== $label GENERATING ==="
  local begin end
  begin="$(date +%s.%N)"
  if run_wangp "$variant" "${args[@]}" 2>&1 | tee "$log"; then
    :
  else
    echo "Failed to generate $label; see $log" >&2
    return 1
  fi
  end="$(date +%s.%N)"
  grep -Fq 'Task 1 completed' "$log" || { echo "Completion marker missing $label" >&2; return 1; }
  awk -v a="$begin" -v b="$end" 'BEGIN {printf "%.3f\n", b-a}' > "$RUN_ROOT/results/$label-seconds.txt"
  local -a clips=()
  mapfile -d '' -t clips < <(find "$out" -type f -name '*.mp4' -size +0c -print0)
  [[ "${#clips[@]}" == 1 ]] || { echo "Expected one clip for $label, found ${#clips[@]}" >&2; return 1; }
  local clip="$RUN_ROOT/$label.mp4"
  cp "${clips[0]}" "$clip"
  if [[ "$index" == 1 ]]; then
    ln "$clip" "$RUN_ROOT/$variant.mp4" 2>/dev/null || cp "$clip" "$RUN_ROOT/$variant.mp4"
  fi
  "$PY" - "$variant" "$index" "$clip" "$log" "$RUN_ROOT/results/$label-seconds.txt" "$SETTINGS" "$RUN_ROOT/results/$label.json" <<'PYRUN'
import hashlib,json,sys
from pathlib import Path
variant,index,clip,log,time_file,settings,out=sys.argv[1:]
content=Path(log).read_text(errors="replace")
rows=[]
for line in content.splitlines():
    if "[LTX25_CUBLAS_E] {" in line:
        try: rows.append(json.loads(line.split("[LTX25_CUBLAS_E] ",1)[1]))
        except ValueError: pass
assert rows, "Missing Candidate E counters"
c=rows[-1]
expect={
    "mode":"fp32rot", "rotation_mode":"radix4", "cast_mode":"vector8",
    "gemm_tune":"off", "rotation_input":"staged" if variant=="baseline" else "direct",
    "output_mode":"torch" if variant=="baseline" else "fused"
}
assert all(c.get(k)==v for k,v in expect.items()), (expect,c)
assert c.get("used",0)>0 and c.get("auto_native_fallback",0)==0, c
assert c.get("gemm_algo2_calls",0)==0, c
ws=[]
for line in content.splitlines():
    if "[LTX25_WORKSPACE] {" in line:
        try: ws.append(json.loads(line.split("[LTX25_WORKSPACE] ",1)[1]))
        except ValueError: pass
assert ws, "Missing workspace counters"
w=ws[-1]
assert w.get("persistent_weight_copies")==0 and w.get("acquired",0)>0,w
fp16_stats_path=Path(log).parent/f"{variant}-fp16-stats.json"
fp16_stats=None
if variant.startswith("fp16-"):
    assert fp16_stats_path.is_file(),f"Missing FP16 dispatch counters: {fp16_stats_path}"
    fp16_stats=json.loads(fp16_stats_path.read_text())
    assert fp16_stats.get("persistent_fp16_weight_copies")==0,fp16_stats
    converted=sum(fp16_stats.get("stats",{}).get("converted_"+op,0) for op in ("mm","addmm"))
    if converted<=0:
        raise RuntimeError(f"Selected case {variant} did not actually convert any BF16 GEMMs; review selectors and shapes")
else:
    converted=0
row={
  "variant":variant,"index":int(index),"seconds":float(Path(time_file).read_text()),
  "clip":clip,"settings_sha256":hashlib.sha256(Path(settings).read_bytes()).hexdigest(),
  "candidate_calls":c["used"],"fallback_rows":c.get("range_fallback"),
  "workspace_leases":w["acquired"],"persistent_weight_copies":w["persistent_weight_copies"],
  "selective_fp16_gemm_calls":converted,
  "selective_fp16_guard_fallback":0 if fp16_stats is None else (
    fp16_stats.get("stats",{}).get("guard_input_reference",0)+
    fp16_stats.get("stats",{}).get("guard_output_reference",0)),
}
Path(out).write_text(json.dumps(row,indent=2)+"\n")
print("VERIFIED",variant,"wall_seconds",row["seconds"],"optimized_calls",c["used"],
      "range_fallback_rows",row["fallback_rows"])
PYRUN
}

if [[ "${FP16_GEMM_SWEEP_ONLY:-0}" == "1" ]]; then
  echo "=== Selective BF16-to-FP16 GEMM per-case video sweep ==="
  declare -a cases=()
  IFS=',' read -ra cases <<< "${SWEEP_CASES:-bf16-reference,fp16-s1-wide,fp16-s1-tall,fp16-s1-balanced,fp16-s1-all,fp16-s2-wide,fp16-s2-tall,fp16-s2-balanced,fp16-s2-all,fp16-both-all}"
  [[ "${#cases[@]}" -ge 2 && "${cases[0]}" == bf16-reference ]] || {
    echo "SWEEP_CASES must start with bf16-reference and include >=1 FP16 case" >&2; exit 2;
  }
  declare -A seen_cases=()
  failed=0
  for case_name in "${cases[@]}"; do
    if [[ ! "$case_name" =~ ^(bf16-reference|fp16-s[12]-(wide|tall|balanced|all)|fp16-both-all)$ ]] || [[ -v "seen_cases[$case_name]" ]]; then
      echo "Invalid or duplicate SWEEP_CASES entry: $case_name" >&2; exit 2
    fi
    seen_cases["$case_name"]=1
  done
  for case_name in "${cases[@]}"; do
    if run_one "$case_name" 1; then
      echo "=== CASE COMPLETE: $case_name ==="
    else
      echo "=== CASE FAILED: $case_name (continuing other variants) ===" >&2
      printf '%s\n' "$case_name" >> "$RUN_ROOT/results/failed-cases.txt"
      failed=$((failed + 1))
    fi
  done
  "$PY" benchmarks/ltx25_convrot/summarize_selective_fp16_sweep.py "$RUN_ROOT" "${SWEEP_CASES:-bf16-reference,fp16-s1-wide,fp16-s1-tall,fp16-s1-balanced,fp16-s1-all,fp16-s2-wide,fp16-s2-tall,fp16-s2-balanced,fp16-s2-all,fp16-both-all}" \
    | tee "$RUN_ROOT/sweep-summary.txt"
  echo "=== Selective FP16 GEMM sweep finished (failed cases: $failed). Results: $RUN_ROOT ==="
  echo "Review videos: $RUN_ROOT/review-index.html"
  echo "Benchmark CSV: $RUN_ROOT/results/sweep-benchmark.csv"
  if (( failed > 0 )); then exit 1; fi
  exit 0
fi

if [[ "${PROFILE_PIPELINE_ONLY:-0}" == "1" ]]; then
  echo "=== ONE whole-pipeline instrumented Candidate E I2V run ==="
  run_one pipeline 1
  PIPELINE_JSON="$RUN_ROOT/results/pipeline_profile.json"
  [[ -s "$PIPELINE_JSON" ]] || { echo "Pipeline profile JSON missing: $PIPELINE_JSON" >&2; exit 1; }
  "$PY" benchmarks/ltx25_convrot/summarize_pipeline_profile.py     "$PIPELINE_JSON" "$RUN_ROOT/results" "$RUN_ROOT/results/pipeline-1-seconds.txt"     | tee "$RUN_ROOT/pipeline-summary.txt"
  echo "=== Whole-pipeline profile complete: $RUN_ROOT ==="
  echo "Video: $RUN_ROOT/pipeline.mp4"
  echo "Summary: $RUN_ROOT/pipeline-summary.txt"
  echo "Raw JSON: $RUN_ROOT/results/pipeline_profile.json"
  echo "Phase CSV: $RUN_ROOT/results/pipeline-phases.csv"
  echo "Op CSV: $RUN_ROOT/results/pipeline-operator-samples.csv"
  exit 0
fi

if [[ "${PROFILE_ONLY:-0}" == "1" ]]; then
  echo "=== ONE instrumented Candidate E (direct+fused) 241-frame I2V run ==="
  run_one profile 1
  PROFILE_JSON="$RUN_ROOT/results/stage_profile.json"
  [[ -s "$PROFILE_JSON" ]] || { echo "Profiler JSON missing: $PROFILE_JSON" >&2; exit 1; }
  "$PY" benchmarks/ltx25_convrot/summarize_stage_profile.py     "$PROFILE_JSON" "$RUN_ROOT/results" | tee "$RUN_ROOT/profile-summary.txt"
  echo "=== Profile complete: $RUN_ROOT ==="
  echo "Video: $RUN_ROOT/profile.mp4"
  echo "Summary: $RUN_ROOT/profile-summary.txt"
  echo "Per-shape CSV: $RUN_ROOT/results/profile-shape-costs.csv"
  exit 0
fi

run_one baseline 1
run_one fused 1
if [[ "$ABBA" == 1 ]]; then
  run_one fused 2
  run_one baseline 2
fi

"$PY" - "$RUN_ROOT" "$FRAMES" <<'PYCOMP'
import csv,hashlib,json,statistics,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]); frames_required=int(sys.argv[2])
results=root/"results"
rows=[json.loads(p.read_text()) for p in results.glob("baseline-*.json")]
rows += [json.loads(p.read_text()) for p in results.glob("fused-*.json")]
rows.sort(key=lambda r: (r["index"],0 if r["variant"]=="baseline" else 1))
report={"status":"FAIL","decoded_video_identical":False,"decoded_audio_identical":False,"runs":rows}
path=results/"video-audio-equality.json"
try:
    fingerprints=[]
    for row in rows:
        clip=Path(row["clip"])
        # FrameMD5 hashes decoded raw YUV420P pixels per frame, not MP4 metadata.
        cmd=["ffmpeg","-v","error","-nostdin","-i",str(clip),"-map","0:v:0",
             "-an","-c:v","rawvideo","-pix_fmt","yuv420p","-f","framemd5","-"]
        p=subprocess.run(cmd,capture_output=True,text=True,check=True,timeout=300)
        video=tuple(line.split(",")[-1].strip() for line in p.stdout.splitlines()
                    if line.strip() and not line.startswith("#"))
        assert len(video)==frames_required,(clip,len(video),frames_required)
        # Digest PCM rather than container bytes; streaming avoids buffering.
        cmd=["ffmpeg","-v","error","-nostdin","-i",str(clip),"-map","0:a:0",
             "-vn","-c:a","pcm_s16le","-f","s16le","pipe:1"]
        proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        digest=hashlib.sha256(); pcm_bytes=0
        while True:
            chunk=proc.stdout.read(1024*1024)
            if not chunk: break
            digest.update(chunk); pcm_bytes+=len(chunk)
        err=proc.stderr.read()
        rc=proc.wait(timeout=120)
        assert rc==0 and pcm_bytes>0,(clip,rc,err[-500:])
        row["video_frame_md5_sha256"]=hashlib.sha256("|".join(video).encode()).hexdigest()
        row["decoded_audio_sha256"]=digest.hexdigest()
        row["decoded_pcm_bytes"]=pcm_bytes
        fingerprints.append((video,(digest.hexdigest(),pcm_bytes)))
    first_video,first_audio=fingerprints[0]
    report["decoded_video_identical"]=all(v==first_video for v,a in fingerprints[1:])
    report["decoded_audio_identical"]=all(a==first_audio for v,a in fingerprints[1:])
    report["settings_identical"]=len({r["settings_sha256"] for r in rows})==1
    report["dispatch_counts_identical"]=len({(r["candidate_calls"],r["fallback_rows"]) for r in rows})==1
    baselines=[r["seconds"] for r in rows if r["variant"]=="baseline"]
    fused=[r["seconds"] for r in rows if r["variant"]=="fused"]
    a=statistics.median(baselines);b=statistics.median(fused)
    report.update(baseline_median_seconds=round(a,3),fused_median_seconds=round(b,3),
                  seconds_saved=round(a-b,3),time_reduction_percent=round(100*(a-b)/a,3),
                  faster=a>b)
    report["status"]="PASS" if all(report[k] for k in
        ("decoded_video_identical","decoded_audio_identical","settings_identical",
         "dispatch_counts_identical")) else "FAIL"
except Exception as exc:
    report["error"]=f"{type(exc).__name__}: {exc}"
finally:
    path.write_text(json.dumps(report,indent=2)+"\n")
    keys=["variant","index","seconds","clip","candidate_calls","fallback_rows",
          "workspace_leases","settings_sha256","video_frame_md5_sha256",
          "decoded_audio_sha256","decoded_pcm_bytes"]
    with (results/"benchmark-summary.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction="ignore")
        writer.writeheader();writer.writerows(rows)
    print("RESULT",json.dumps({k:v for k,v in report.items() if k!="runs"},indent=2))
    print("Videos:",root/"baseline.mp4",root/"fused.mp4")
    print("Details:",path)
if report["status"]!="PASS":
    sys.exit("Decoded video/audio or matched-settings comparison failed; keep baseline default.")
PYCOMP

echo "=== PASS: exact decoded audio/video; results in $RUN_ROOT ==="
