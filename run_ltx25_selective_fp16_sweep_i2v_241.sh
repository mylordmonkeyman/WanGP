#!/usr/bin/env bash
# One command: generate BF16 reference and independent FP16 GEMM shape/phase variants.
# All videos use the same I2V image, prompt, negative prompt, LoRA, seed and settings.
set -euo pipefail

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  cat <<'HELP'
Usage:
  bash ./run_ltx25_selective_fp16_sweep_i2v_241.sh IMAGE 'POSITIVE PROMPT' 'NEGATIVE PROMPT'

Default: 241 frames, 1024x1024, 8 steps, seed 20261112,
         LTX_10Eros LoRA, NAG 1.1, audio CFG 1.5, MMGP 0, SDPA.
         Candidate E direct radix4, Vector8 and fused output in all arms.

SWEEP_CASES is a comma-separated sequence (always starts bf16-reference):
  bf16-reference    Existing Candidate E, native BF16 GEMMs
  fp16-s1-wide      Only Stage 1 GEMMs with N >= 2*K
  fp16-s1-tall      Only Stage 1 GEMMs with K >= 2*N
  fp16-s1-balanced  Only Stage 1 GEMMs with N ~ K
  fp16-s1-all       All three Stage 1 groups
  fp16-s2-wide      Only Stage 2 wide GEMMs
  fp16-s2-tall      Only Stage 2 tall GEMMs
  fp16-s2-balanced  Only Stage 2 balanced GEMMs
  fp16-s2-all       All three Stage 2 groups
  fp16-both-all     All three groups in both stages

DEFAULT: all ten cases, one video per case (long-running).
To select just the most important groups:
  SWEEP_CASES=bf16-reference,fp16-s2-wide,fp16-s2-tall,fp16-s2-balanced,fp16-s2-all,fp16-both-all bash ...
To do a short exploratory sweep:
  FRAMES=49 SWEEP_CASES=bf16-reference,fp16-s2-all,fp16-both-all bash ...

Optional:
  SWEEP_OPS=mm,addmm        aten.mm and aten.addmm (both by default)
  SWEEP_GUARD=first         first-per-shape input/output FP16 finite checks
  SWEEP_GUARD=always        full finite checks (INVALID for speed comparisons)
  SWEEP_GUARD=off           fastest, no finite-value guard (higher quality risk)
  SWEEP_MIN_M=128 SWEEP_MIN_DIM=512
  PY=/path/to/python CONFIG_TEMPLATE=/path/to/wgp_config.json
  LORA=/path/to/lora.safetensors (or LORA=none)
  SEED=... FRAMES=241 RESOLUTION=1024x1024 STEPS=8
  RUN_ROOT=/path/to/empty/output
  CHECK_GPU_IDLE=1 SKIP_PREFLIGHT=1

Outputs: sweep-summary.txt, results/sweep-benchmark.csv,
         review-index.html, each variant's named MP4 and generation log,
         results/fp16-<variant>-fp16-stats.json.
The generated HTML page is portable with the whole output folder.
FP16 results are NOT assumed identical or visually acceptable.
No persistent FP16 master weights are created.
HELP
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
for p in run_candidate_e_fusions_i2v_241.sh shared/kernels/ltx25_selective_fp16_gemm.py benchmarks/ltx25_convrot/summarize_selective_fp16_sweep.py; do
  [[ -s "$SCRIPT_DIR/$p" ]] || { echo "Missing $p; git pull the feature branch" >&2; exit 1; }
done
export REPO_ROOT="$SCRIPT_DIR"
export FP16_GEMM_SWEEP_ONLY=1
export PROFILE_ONLY=0 PROFILE_PIPELINE_ONLY=0
export SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-1}"
export RUN_ROOT="${RUN_ROOT:-$HOME/ai/candidate-e-selective-fp16-sweep-results/$(date -u +%Y%m%d-%H%M%S)-$$}"
exec bash "$SCRIPT_DIR/run_candidate_e_fusions_i2v_241.sh" "$@"
