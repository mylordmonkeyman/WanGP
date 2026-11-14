#!/usr/bin/env bash
# Profile one real 241-frame LTX-2.5 I2V pipeline: phases + sampled operators.
# Runs Candidate E with exactly the same generation settings as earlier tests.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'HELP'
Usage:
  bash ./run_ltx25_full_pipeline_profile_i2v_241.sh IMAGE 'POSITIVE PROMPT' 'NEGATIVE PROMPT'

Defaults: 241 frames, 1024x1024, 8 steps, LoRA LTX_10Eros, seed 20261112,
NAG 1.1, audio CFG 1.5, SDPA, MMGP profile 0, Candidate E direct+fused.
Only one video generated. Instrumentation includes:
 - synchronized critical-path phase wall time: text, two-stage conditioning,
   Stage 1 and 2 diffusion, spatial upscale, video VAE, audio VAE
 - transformer-forward CUDA event totals for each stage
 - one PyTorch CPU/CUDA operator sample from the 2nd transformer forward
   in each stage, showing SDPA/copy-related and other visible ops

Optional: LTX25_PIPELINE_OP_SAMPLE=0 to disable the torch.profiler samples
          (major phase and transformer timings stay enabled).
          PY=/path/to/venv/bin/python CONFIG_TEMPLATE=/path/to/wgp_config.json
          LORA=none or LORA=/path/to/file.safetensors
          SEED=... FRAMES=241 RESOLUTION=1024x1024 STEPS=8
          NAG_SCALE=1.1 AUDIO_CFG_SCALE=1.5
          RUN_ROOT=/path/to/empty/folder CHECK_GPU_IDLE=1 SKIP_PREFLIGHT=1

Outputs under RUN_ROOT:
  pipeline-summary.txt, pipeline.mp4
  results/pipeline_profile.json, results/pipeline-phases.csv
  results/pipeline-operator-samples.csv, results/pipeline-1-generation.log

Profiling adds boundary CUDA synchronization and selected profiler overhead.
NEVER compare this run's elapsed time directly with prior uninstrumented runs.
HELP
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
[[ -s "$SCRIPT_DIR/benchmarks/ltx25_convrot/summarize_pipeline_profile.py" ]] || {
  echo "Pull latest perf/ltx25-convrot-sm70 first" >&2; exit 1;
}
export REPO_ROOT="$SCRIPT_DIR"
export PROFILE_PIPELINE_ONLY=1
export PROFILE_ONLY=0
export SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-1}"
export RUN_ROOT="${RUN_ROOT:-$HOME/ai/candidate-e-whole-pipeline-i2v-results/$(date -u +%Y%m%d-%H%M%S)-$$}"
exec bash "$SCRIPT_DIR/run_candidate_e_fusions_i2v_241.sh" "$@"
