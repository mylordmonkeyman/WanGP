# LTX-2.5 V100 — 73f T2V/I2V bottleneck attribution (November 15, 2026)

**Status:** measured sampled CUDA kernel attribution; experimental optimization selection. This is **not** a demonstrated end-to-end optimization.

- Stable baseline: `main` at `523cb6f8cf3aeca2daa38a61a3d7e81c01c61c43`.
- Sampled profiler run: [GitHub Actions 43748002222](https://github.com/mylordmonkeyman/WanGP/actions/runs/43748002222), **PASS**, both T2V and I2V, 1024×576, 73 frames, eight diffusion steps, original bar prompt/image, seed 20261112, MMGP profile 0, SDPA, no LoRA.
- Matched uninstrumented 73f baseline: [run 43746795037](https://github.com/mylordmonkeyman/WanGP/actions/runs/43746795037): **T2V 112.322 s, I2V 113.989 s**, decoded video and PCM audio bit-identical to pinned main. Independent earlier cleaned main check: T2V 112.736 s, I2V 113.777 s.
- The run uses main-equivalent Candidate E, Radix-4, both-stage (4096,2048), selective Stage2 FP16 intercept enabled. Candidate E used **27,360 calls** in each clip; both-stage 4096×2048 coverage 384+144 with no fallbacks. **0 persistent expanded weights**.
- Normal and profiled timings are intentionally separate: trace runs take ~205–209 s and must **not** be used as performance baselines.

## True device-kernel durations in a single sampled transformer forward

Numbers are **milliseconds of accumulated GPU kernel execution events** in the second forward of each stage. These are NOT elapsed stage times. Independent H2D/compute streams can overlap and event sums cannot be subtracted from wall time.

| GPU kernel category | T2V stage1 | T2V stage2 | I2V stage1 | I2V stage2 |
| --- | ---: | ---: | ---: | ---: |
| GEMM/GEMV (all backends) | 139.6 | 1092.4 | 1369.6 | 2393.2 |
| H2D DMA | 194.1 | 698.4 | 1957.9 | 1811.3 |
| Candidate E final epilogue | 30.2 | 97.4 | 421.8 | 1249.3 |
| Candidate E INT8→FP16 cast | 36.7 | 265.2 | 357.4 | 571.0 |
| Attention | 16.6 | 506.3 | 162.4 | 1106.4 |
| Rotation (Radix-4) | 8.6 | 109.8 | 84.1 | 239.9 |
| Activation preparation | 10.0 | 114.5 | 97.6 | 254.9 |
| Other kernels/metadata/copies | 37.5 | 385.2 | 366.1 | 851.5 |
| **Sum of GPU device event durations** | **473.4** | **3269.2** | **4816.9** | **8477.6** |

**Measured perturbation/variance:** 73-frame T2V and I2V have *the same* observed primary `aten::addmm` input families (stage1 video tokens 1440; stage2 video tokens 5760), yet sampled GEMM/H2D/kernel times vary very substantially. Stage1 GPU event sums vary ~10× even though normal full-generation times vary ~1.5%. Do **not** extrapolate the four samples as interchangeable per-step timings or assign speedups from them. The sampling machinery introduces extra synchronization and profiler overhead. MMGP chooses 64/64 transformer blocks resident after its activation census, but initial phase changes and H2D are still present and must be studied separately.

### Whole process stage attribution

Only the separately instrumented pipeline supplies phase boundaries. Observed approximate *instrumented* stage1 denoising: T2V 62.650 s, I2V 60.231 s; stage2 denoising: T2V 67.431 s, I2V 72.504 s; text encoding ~11.2–11.4 s; spatial upsampling ~5.0–5.8 s; final video VAE decoding ~15.2 s. These include profiling overhead and cannot be added to the ~113 s uninstrumented wall baseline.

Offload / H2D GPU events occurred in both sampled stages. Exact transfer bytes, true copy/compute overlap, CPU dispatch time and MMGP wait time remain unmeasured. These are **open attribution gaps**, not zero-time categories.

## Amdahl-aware experimental ranking

The *screening calculations* below multiply second-forward GPU device event time by 8 Stage1 and 3 Stage2 calls, assuming representativeness and **excluding H2D**. They are hypothetical GPU-work exposure, **not** recovered elapsed time. They need matched whole-video timing to establish actual benefit.

| Candidate | Measured repeated-kernel exposure proxy | Illustrative 25% kernel work reduction | Complexity | Numerical risk | Decision |
| --- | --- | --- | --- | --- | --- |
| **1. Tiled Candidate E FP32-corrected epilogue** | T2V ~0.53 s, I2V ~7.12 s | ~0.13–1.78 s GPU work; **wall benefit unknown** | Low–medium | Low: identical per-element FP32 FMA order, same FP32 scales and BF16 narrowing | **Prototype first**, reversible opt-in, strict decoded hash and R0/R1 checks |
| **2. Faster transient INT8→FP16 cast / conversion scheduling** | T2V ~1.09 s, I2V ~4.57 s | ~0.27–1.14 s GPU work; **wall benefit unknown** | Medium | Low–medium: cast itself is exact, but lifetime/stream ownership matters | Next if tiled epilogue fails or wins weakly; no persistent duplicate weight set |
| **3. MMGP transfer/offload wait reduction** | H2D GPU duration per sampled pair T2V ~0.89 s, I2V ~3.77 s, **not safely scalable** | Unknown until first/cold/warm/overlap traced | Medium–high | Moderate synchronization/memory pressure risk | Keep high research priority; measure transfer bytes, blocks and critical path before modifying MMGP |

GEMM remains a large GPU-work category, but prior cuBLAS/cublasLt heuristic tuning was unsuccessful. Do not repeat without new evidence. Attention is substantial in stage2 (~0.51–1.11 s/sample), but alternative arithmetic risks visual fidelity and should not displace lower-risk measured opportunities first. Radix-4 itself is not the primary current hotspot.

**Promotion rule:** the next experiment is not eligible for default or `main` until targeted CUDA microbench accuracy tests, whole-video A–B–B–A, hashes/PSNR/SSIM/audio metrics, repeatability and VRAM/regression gates pass. Any performance improvement inferred from this report alone is a hypothesis.
