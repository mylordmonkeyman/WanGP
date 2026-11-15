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
