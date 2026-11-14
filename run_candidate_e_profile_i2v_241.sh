#!/usr/bin/env bash
# One 241-frame real-model LTX-2.5 I2V run with sparse Candidate E CUDA events.
# Uses the same settings, LoRA, MMGP profile and SDPA as prior 241-frame A/B.
# This is NOT an uninstrumented wall-clock benchmark.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'HELP'
Usage:
  bash ./run_candidate_e_profile_i2v_241.sh IMAGE 'POSITIVE PROMPT' 'NEGATIVE PROMPT'

Default: 241 frames, 1024x1024, 8 steps, seed 20261112,
LTX_10Eros LoRA at strength 1, NAG 1.1, Audio CFG 1.5,
Candidate E direct BF16/FP16 radix4 + vector8 + fused output.
SDPA, MMGP profile 0; one video, one real-model profiler trace.

Optional environment:
  PY=/path/to/venv/bin/python
  CONFIG_TEMPLATE=/path/to/wgp_config.json
  LORA=/path/to/lora.safetensors    (or LORA=none)
  RESOLUTION=1280x704             FRAMES=241
  SEED=20261112                   STEPS=8
  NAG_SCALE=1.1                    AUDIO_CFG_SCALE=1.5
  RUN_ROOT=/path/to/new/directory  CHECK_GPU_IDLE=1
  SKIP_PREFLIGHT=1                 LTX25_STAGE_PROFILE_MAX_SAMPLES=320

Outputs: profile.mp4, profile-summary.txt, results/stage_profile.json,
results/profile-shape-costs.csv and generation logs.
This profiler counts all eligible calls, sparsely samples CUDA event timing
by exact (N,K,M,dtype), and ranks estimated stage costs. It does NOT
measure isolated kernel time or predict total-video speedup.
HELP
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
[[ -s "$SCRIPT_DIR/run_candidate_e_fusions_i2v_241.sh" ]] || {
  echo "Missing companion 241-frame benchmark script; pull the feature branch" >&2
  exit 1
}
[[ -s "$SCRIPT_DIR/benchmarks/ltx25_convrot/summarize_stage_profile.py" ]] || {
  echo "Missing profile summary utility; pull the feature branch" >&2
  exit 1
}
export REPO_ROOT="$SCRIPT_DIR"
export PROFILE_ONLY=1
# Preflight was already validated in CI; avoiding it makes this a one-GPU-run task.
export SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-1}"
export RUN_ROOT="${RUN_ROOT:-$HOME/ai/candidate-e-stage-profile-i2v-results/$(date -u +%Y%m%d-%H%M%S)-$$}"
exec bash "$SCRIPT_DIR/run_candidate_e_fusions_i2v_241.sh" "$@"
