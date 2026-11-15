# Cold-forward MMGP optimization evidence — 2026-11-15

Work branch: `perf/ltx25-v100-mmgp-transfer-20261115`. Archived tiled work: `archive/ltx25-v100-tiled-epilogue-20261115` at `8788b047`. Stable `main` remains `523cb6f8` and the tiled epilogue is still default OFF.

## First vs second transformer forward

[Verified 73f T2V/I2V run 43763757911](https://github.com/mylordmonkeyman/WanGP/actions/runs/43763757911): 1024×576, 8 steps, same seed/checkpoint/attention/MMGP profile and default Candidate E, original I2V bar image. Both decoded video/audio match pinned stable hashes exactly. Temporary runner instrumentation restored production source byte-for-byte.

| T2V measurement | Stage 1 first | Stage 1 second | Stage 2 first | Stage 2 second |
| --- | ---: | ---: | ---: | ---: |
| Transformer CUDA span (s) | 4.002 | 2.516 | 12.216 | 6.316 |
| MMGP block-loader CPU sum (s) | 1.295 | 0.455 | **5.697** | **0.383** |
| GPU H2D event-duration sum (s) | 1.717 | 1.580 | 1.711 | 1.576 |
| H2D bytes (GiB) | 18.00 | 16.56 | 18.00 | 16.56 |
| GPU consumer-wait event sum (ms) | 27.7 | 0.65 | 22.9 | 0.64 |

I2V independently reproduced Stage 2 CPU loader 5.663 s first / 0.380 s second, similar H2D GPU durations, and ~26 ms first GPU consumer waits.

**Interpretation:** The cold Stage 2 discrepancy appears primarily related to CPU-side MMGP block loading (allocation, parameter installation, cache management or other host work), not increased GPU copy duration alone. These CPU/GPU intervals overlap and profiling perturbs behavior: none is a verified recoverable wall saving.

## Gated next steps

- First-vs-warm CUDA traces: 1st and 2nd forward of each stage, both T2V and I2V, classify actual GEMM, attention, cast, H2D and other kernel work independently.
- Decompose MMGP block-loader host time into move/allocate/transfer enqueue, parameter installation, prefetch futures, unload and cache checks. Keep outputs bit-identical.
- Prototype only an evidenced bottleneck behind a new flag, then whole-process A–B–B–A with output fidelity and VRAM gates. No second persistent expanded weight set or changes to main/defaults.

## First/second CUDA kernel evidence

[Paired 73f T2V/I2V profiler run 43763934932](https://github.com/mylordmonkeyman/WanGP/actions/runs/43763934932) **PASS**, four sampled forwards per mode, decoded video/audio bit-identical. The *I2V* Stage 2 pair is the cleaner comparable kernel attribution:

| I2V Stage 2 GPU event category | First forward | Second forward |
| --- | ---: | ---: |
| GEMM/GEMV kernel durations (sum) | 2.311 s | 2.300 s |
| Attention kernel durations (sum) | 1.059 s | 1.051 s |
| H2D and other copy durations (sum) | **3.669 s** | **1.606 s** |
| H2D memcpy event count | **8,174** | **46** |
| Total GPU kernel and memory event-duration sum | 9.725 s | 7.742 s |

The first forward has essentially the same main math kernel duration as the second; the large change is transfer fragmentation and accompanying CPU-side loading. CUDA event sums include concurrent streams and must **not** be subtracted from wall time.

**Trace defect:** The T2V Stage 1 second-forward exported trace contains one absurd multi-billion-millisecond cast event. This sample is invalid and excluded from conclusions. The T2V Stage 2 warm trace also differs sharply in captured kernel counts, so do not treat it as reliable for direct first/second computational cost subtraction. Use the clean I2V Stage 2 pair and independently measured MMGP per-forward CUDA events.

## Reversible first implementation under test

`WAN2GP_LTX25_STAGE2_KEEP_CUDA_CACHE=1` in `models/ltx2/ltx_pipelines/distilled.py` skips **only** the one CUDA caching-allocator purge just after spatial upscaling; the existing `gc.collect()` and `torch.cuda.synchronize()` remain. Flag defaults to off. Model weights and numerical kernels are untouched, and it does not create a second persistent FP16 weight set. The hypothesis is that retaining free allocator blocks reduces the large cold Stage2 CPU block-load cost, but it may not: MMGP auto-preload can also release model blocks to make room for the upscaler.

[73f T2V/I2V A–B–B–A end-to-end run 43765110798](https://github.com/mylordmonkeyman/WanGP/actions/runs/43765110798) tests eight fresh processes and checks every decoded video/audio hash against the stable main. Do not promote unless output parity, memory safety and a reproducible whole-video saving are demonstrated.
