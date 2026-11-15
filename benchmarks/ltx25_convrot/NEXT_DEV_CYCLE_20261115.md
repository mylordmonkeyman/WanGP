# LTX-2.5 / V100: next development cycle (2026-11-15)

**Branch:** `perf/ltx25-v100-next-optimizations-20261115`  
**Source baseline:** `main` at `523cb6f8cf3aeca2daa38a61a3d7e81c01c61c43`.  
**Baseline status:** 49-frame four-way opt-out regression and 73-frame
T2V/I2V bar-prompt decoded-video + PCM-audio bit-exact regression passed.

## Goal

Improve real end-to-end LTX-2.5 Distilled/Dev video generation speed on a
single Tesla V100 32 GB without silently changing the validated release.
Do **not** turn on an unvalidated feature by default. Avoid duplicate
persistent FP16 weight sets, extra GPU synchronizations, and sub-4-bit
checkpoint alternatives.

## First investigation: remaining end-to-end bottlenecks

1. Reproduce **73-frame, 1024×576, 8-step** bar-prompt Distilled T2V and
   image-conditioned I2V against the `main` code baseline (same seed,
   checkpoint, image, and no additional LoRAs).
2. Attribute time separately to Candidate E activation rotation, transient
   INT8→FP16 weight expansion, cuBLAS GEMM, FP32 row prepare and epilogue,
   Stage 2 selective-FP16 interception, and non-GEMM steps (offload,
   attention, VAE). Use **standalone one-off benchmark instrumentation**,
   not permanent hot-path profiling or a second resident weight cache.
3. Rank opportunities by actual whole-video latency contribution, not
   isolated microbenchmark ratios. Document GPU synchronization and memory
   overhead separately.
4. Prototype **only the highest-yield bottleneck** behind a new, reversible
   research flag on this development branch. Preserve default-on Candidate
   E eight-family routing and the Stage 1/2 opt-outs until full validation.

Potential candidates include reducing transient weight-cast allocations,
improving the fallback/Stage 2 selective FP16 path without widening Stage 1,
and eliminating redundant per-linear kernel launches. A proposed
rotation/prepare fusion must preserve the exact GPU exceptional-row
correction and output numerics; do not assume fusion is feasible or faster.

## Promotion gates

- SM70 native/CUDA correctness and guards under the actual checkpoint.
- Performance measurements with matched A–B–B–A **whole-process** runs
  (T2V and I2V) plus relevant standalone GEMM microbenchmarks.
- Stock/main-relative PSNR, SSIM, audio SNR, and worst-frame heat maps where
  the new kernel is not bit-identical; reject or revert any clear artifacts.
- Prefer byte-identical decoded video/audio to `main` for a pure
  performance refactor.
- No merge to `main` until the user explicitly promotes new defaults.

## Known limitations of the current baseline

- Candidate E `(N=4096,K=2048)` is enabled in both stages by user choice
  despite earlier strict per-GEMM p99/max precision-audit failures. Each
  stage has an independent disable flag.
- Stage 2 FP16 Quanto interception is a scoped process-wide operation and
  should not run concurrently in multiple model-inference threads until
  hardened; a first-per-shape finite guard is not a per-call guarantee.
- Previous cuBLASLt heuristic and GEMM algorithm sweep had too little
  whole-pipeline upside to justify adopting more dispatch variants.
