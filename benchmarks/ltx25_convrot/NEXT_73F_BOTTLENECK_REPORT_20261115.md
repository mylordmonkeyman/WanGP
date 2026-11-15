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


## First reversible implementation: N-tiled epilogue

Prototype: `finish_rows_low_tiled` in `benchmarks/ltx25_convrot/ltx25_cublas_e.cu`, exposed only by `WAN2GP_LTX25_E_EPILOGUE_TILED=1` in `shared/kernels/ltx25_cublas_sm70.py`. With the default `0`, the original kernel is used. The experimental dispatch is deliberately restricted to `M<=128, N>=2048`, after the SM70 microbench found regressions for `M=512` and `M=1440`. No persistent expanded weights, per-GEMM host sync, new precision policy or second FP16 storage is introduced.

[Microbenchmark 43749614759](https://github.com/mylordmonkeyman/WanGP/actions/runs/43749614759) **PASS**: real observed video/audio shape families with synthetic representative inputs. BF16 output is **bit-identical** to original epilogue with and without bias, including two exceptional-row fallback scenarios. Median GPU-event latency ratios (original / tiled):

| Shape M×N×K | Row case | Measured ratio |
| --- | --- | ---: |
| 76×2048×2048 | all safe | 1.17× |
| 76×4096×2048 | all safe | 1.30× |
| 76×16384×4096 | all safe | 2.95× |
| 76×16384×4096 | one FP32 exception | 27.16× |
| 76×16384×4096 | two FP32 exceptions | 18.18× |
| 1440×4096×2048 | all safe | 0.96× (regression; **not selected**) |
| 512×8192×4096 | all safe | 0.97× (regression; **not selected**) |

This is *not yet* an end-to-end improvement or independent R0/R1 numerical validation; the exact baseline epilogue is preserved and compared directly. Full 73f T2V/I2V A–B–B–A is [Actions run 43749888611](https://github.com/mylordmonkeyman/WanGP/actions/runs/43749888611); no claim is made from that run until complete.

## Amdahl-aware experimental ranking

The *screening calculations* below multiply second-forward GPU device event time by 8 Stage1 and 3 Stage2 calls, assuming representativeness and **excluding H2D**. They are hypothetical GPU-work exposure, **not** recovered elapsed time. They need matched whole-video timing to establish actual benefit.

| Candidate | Measured repeated-kernel exposure proxy | Illustrative 25% kernel work reduction | Complexity | Numerical risk | Decision |
| --- | --- | --- | --- | --- | --- |
| **1. Tiled Candidate E FP32-corrected epilogue** | T2V ~0.53 s, I2V ~7.12 s | ~0.13–1.78 s GPU work; **wall benefit unknown** | Low–medium | Low: identical per-element FP32 FMA order, same FP32 scales and BF16 narrowing | **Prototype first**, reversible opt-in, strict decoded hash and R0/R1 checks |
| **2. Faster transient INT8→FP16 cast / conversion scheduling** | T2V ~1.09 s, I2V ~4.57 s | ~0.27–1.14 s GPU work; **wall benefit unknown** | Medium | Low–medium: cast itself is exact, but lifetime/stream ownership matters | Next if tiled epilogue fails or wins weakly; no persistent duplicate weight set |
| **3. MMGP transfer/offload wait reduction** | H2D GPU duration per sampled pair T2V ~0.89 s, I2V ~3.77 s, **not safely scalable** | Unknown until first/cold/warm/overlap traced | Medium–high | Moderate synchronization/memory pressure risk | Keep high research priority; measure transfer bytes, blocks and critical path before modifying MMGP |

GEMM remains a large GPU-work category, but prior cuBLAS/cublasLt heuristic tuning was unsuccessful. Do not repeat without new evidence. Attention is substantial in stage2 (~0.51–1.11 s/sample), but alternative arithmetic risks visual fidelity and should not displace lower-risk measured opportunities first. Radix-4 itself is not the primary current hotspot.

**Promotion rule:** the next experiment is not eligible for default or `main` until targeted CUDA microbench accuracy tests, whole-video A–B–B–A, hashes/PSNR/SSIM/audio metrics, repeatability and VRAM/regression gates pass. Any performance improvement inferred from this report alone is a hypothesis.


## Experimental candidate 1 result — tiled Candidate E epilogue

**Result:** rejected for promotion on these workloads. See [V100 full-process 73f T2V/I2V A–B–B–A run 43749888611](https://github.com/mylordmonkeyman/WanGP/actions/runs/43749888611) and [V100 bit-parity microbenchmark run 43749614759](https://github.com/mylordmonkeyman/WanGP/actions/runs/43749614759). Development implementation is guarded by `WAN2GP_LTX25_E_EPILOGUE_TILED=1`; default remains off. The tiled kernel only dispatches for `M<=128, N>=2048`; existing kernel is the fallback.

| Mode | Baseline A1 | Tiled B1 | Tiled B2 | Baseline A2 | A mean | B mean | End-to-end B improvement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T2V | 111.913 s | 111.384 s | 111.928 s | 112.075 s | 111.994 s | 111.656 s | +0.302% |
| I2V | 113.577 s | 113.537 s | 113.618 s | 113.326 s | 113.452 s | 113.578 s | -0.111% |

All **eight videos and their decoded audio** were bit-identical to the pinned stable baseline. Candidate E executed 27,360 calls per run, of which exactly 9,216 were tiled in B runs; no extra persistent weight copies. The same exception count per mode was observed across all arms (8,898 T2V and 7,103 I2V FP32 exceptional rows). GPU-only BF16 epilogue microbenchmarks passed strict `torch.equal`, including exceptional-row cases, and showed up to 2.95× faster typical narrow-row wide-N execution and 18–27× on contrived exceptional-row inputs. These did **not** translate into a reliable full-process benefit. Do not merge, enable by default, or claim recoverable wall time.

**Updated priorities:** (1) quantify memory transfer and model residency critical-path wait vs GPU execution; (2) investigate exact transient INT8→FP16 conversion scheduling, without another persistent weight set; (3) revisit fused Candidate E pipeline only if a controlled full-video result shows a larger recoverable fraction. The tiled epilogue can remain as a reversible diagnostic option.

## Follow-up experiment — 241-frame tiled epilogue

**Outcome: no meaningful end-to-end gain; reject promotion.** [Successful 241f V100 T2V/I2V A–B–B–A run 43756157528](https://github.com/mylordmonkeyman/WanGP/actions/runs/43756157528). Matching 1024×576 bar T2V/I2V, seed 20261112, eight steps, same checkpoint and SDPA/MMGP configuration. Each mode was run A1–B1–B2–A2 with full fresh-process wall-clock timing; no profiling instrumented the timed video executions. Baseline hashes for 241f came from the A1 control and were checked again against A2; **all eight decoded video and audio outputs match exactly**. Every run had 45,216 Candidate E uses, 1,824 original native fallback calls, and the same exceptional-row correction count within mode (5,851 T2V; 5,578 I2V).

| Mode | A1 original | B1 tiled | B2 tiled | A2 original | A mean | B mean | B wall improvement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T2V 241f | 222.928 s | 222.761 s | 222.623 s | 223.613 s | 223.271 s | 222.692 s | +0.259% |
| I2V 241f | 225.606 s | 225.564 s | 225.418 s | 225.206 s | 225.406 s | 225.491 s | -0.038% |

The 241f **experiment-only** override `WAN2GP_LTX25_E_EPILOGUE_TILED_MAX_M=256` widens the tile selection to include relevant M≈251 audio-length rows; default remains **128**, and tiling itself remains **off** by default. Only `WAN2GP_LTX25_E_EPILOGUE_TILED` changes between A and B in this cohort. B executed 30,144 tiled epilogues; A executed zero. The first run [43755051067](https://github.com/mylordmonkeyman/WanGP/actions/runs/43755051067) stopped at B1 when a second hardcoded 128-row gate bypassed tiling; this bug was corrected in the development branch, and the passing rerun confirms real tiled dispatch. Stable `main` is unchanged.

The associated M=251 GPU-only bit-parity microbenchmark confirmed BF16 bit identity for all five shapes/modes and median original/tiled ratios of ~1.00× (251×2048×2048, safe), 1.12× (251×4096×2048, safe), 1.35× (251×16384×4096, safe), 26.86× (one FP32-exception row) and 18.79× (multiple exceptions). The dramatic exceptional-row microbenchmark ratios reflect increased CTA parallelism in very long per-output FP32 dot-product fallback work; they are **not** representative full-process speedups. With negligible whole-video gain at both 73f and 241f, leave tiled opt-in only, do not promote, and proceed to measured transfer/offload critical-path and transient conversion work.
