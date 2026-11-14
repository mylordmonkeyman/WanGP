# LTX-2.5 SM70 T2V/I2V GEMM coverage: controlled investigation (2026-11-14)

Status: measured on self-hosted V100, research only. Branch: \`perf/ltx25-convrot-sm70\`.
No production dtype/kernel defaults or model weights were modified.

## Controlled comparison

Same LTX-2.5 22B INT8 ConvRot checkpoint, uploaded 1024x576 bar PNG only for I2V, same exact cocktail-bar/audio text, seed 20261112, NAG 1.1, audio CFG 1.5, no user LoRA, Candidate E, SDPA, and an opt-in *native-BF16 sham* Quanto scope.

| Test | Stage 1 observed native Quanto GEMMs | Stage 2 observed native Quanto GEMMs |
| --- | ---: | ---: |
| 49-frame 1024x576 T2V | 4992 | 1872 |
| 49-frame 1024x576 I2V | 4992 | 1872 |
| 49-frame 1024x1024 I2V | 4992 | 1872 |
| 241-frame 1024x576 I2V | 384 | 1440 |
| 241-frame 1024x1024 I2V | 384 | 1440 |

A test of the Eros user LoRA was skipped because its weight file was absent on the runner. It is not required to explain the observed count reversal.

Source runs:
- Controlled matched 49-frame T2V vs I2V CUPTI/Quanto: [43621331504](https://github.com/mylordmonkeyman/WanGP/actions/runs/43621331504)
- 49/241, wide/square, LoRA-isolation matrix: [43623031774](https://github.com/mylordmonkeyman/WanGP/actions/runs/43623031774)
- Matched dense GEMM microbench and real GPU CUPTI event attribution: [43623786206](https://github.com/mylordmonkeyman/WanGP/actions/runs/43623786206)

### Root cause

In \`shared/qtypes/int8_convrot.py\`, \`QLinear.forward()\` first attempts \`ltx25_cublas_sm70.maybe_linear()\`; if ineligible, it falls back to native quantized linear, which invokes \`torch.ops.quanto.qbytes_mm\`. The Candidate E guard in \`shared/kernels/ltx25_cublas_sm70.py\` is:

\`\`\`python
if (n, k) not in PASS_SHAPES or group != 256 or m < 64 or m > 8192:
    return None
\`\`\`

Thus the short-video audio GEMMs with \`M=51\` fall below the allowed minimum: they are observable as native Quanto. With 241 frames, those audio projections are larger, pass Candidate E eligibility, and **disappear from Quanto's count without disappearing from GPU work**. Conversely 241-frame Stage 2 video projections have \`M=17856\` at 1024x576 and \`M=31248\` at 1024x1024, exceeding Candidate E's 8192-row maximum. They fall back to native Quanto.

Candidate E counters reflect this exact migration:

| Weight \`N x K\` | Candidate E eligible, 49-frame square | 241-frame square |
| --- | ---: | ---: |
| 2048 x 2048 | 1120 | 6400 |
| 2048 x 8192 | 16 | 544 |
| 8192 x 2048 | 16 | 544 |
| 4096 x 4096 | 4288 | 3424 |
| 2048 x 4096 | 1584 | 1152 |

These are candidate counts, not total GEMM operations. The 4096x2048 weight shape remains outside the Candidate E tested allowlist for independent numerical reasons. The size gate explains **the count reversal**; the model mode itself does not change the matched count pattern.

### Additional GEMM operators beyond Quanto

In a warmed 49-frame transformer forward, sampled Torch profiler CPU ops included 624 \`aten::mm\` calls plus 316 \`aten::addmm\` calls **outside** the tagged native Quanto scope, per stage. Higher-level \`aten::matmul\` and \`aten::linear\` wrap these underlying ops and must not be added again.

The trace attribution uses \`LTX25::quanto_qbytes_mm\` CPU scope markers, native CUDA launch correlation IDs, and only GPU events with \`cat\` of \`kernel\`, \`gpu_memcpy\`, or \`gpu_memset\`. CUDA kernel duration sums are **per one sampled warmed transformer forward**, not wall-clock video times. Kernel name categories are heuristic and GPU duration sums may overlap across streams.

| Sampled 49-frame forward | Stage 1 | Stage 2 |
| --- | ---: | ---: |
| All GPU kernel duration sum | 3665 ms | 6706 ms |
| Duration inside tagged Quanto scope (all GPU kernels) | 402 ms | 634 ms |
| Tagged native Quanto \`aten::mm\` GEMM kernels | 348 ms | 581 ms |
| Un-tagged \`aten::mm\` GEMM kernels | 37 ms | 52 ms |
| Un-tagged \`aten::addmm\` GEMM kernels | 156 ms | 156 ms |
| Other GEMM-named kernels (often already accelerated Candidate E) | 632 ms | 1506 ms |

A 10-shape synthetic V100 microbenchmark with transient FP16 casts and BF16 output shows **~92.5ms GPU savings per Stage1 forward and ~76.1ms per Stage2 forward** (net after slower shapes); extrapolating to 49 frames, \`8*92.5 + 3*76.1 = 968 ms\` **GPU-only theoretical estimate**, not validated end-to-end video speed and not proof of quality equivalence. Small M=408, N=256, K=256 \`mm\` is slower with temporary FP16 casts; the small N=32 \`addmm\` shapes are promising microbench candidates.

## Conclusions and next engineering target

1. The 49-vs-241-frame Quanto-count reversal is a **conditional dispatch change**, not evidence that many GEMMs vanished, and not an inherent T2V/I2V difference.
2. "100% Quanto coverage" means 100% of qbytes calls reaching the direct wrapper, not all transformer linear algebra.
3. Remaining dense GEMMs exist, but their transient-FP16 benefit is modest in the synthetic test and may be negative once integrated into full video generation.
4. High-value potential work: **evaluate** the excluded \`M>8192\` Stage2 video shapes through *bounded streaming tiling* in the existing Candidate E path instead of applying FP16 casting to every tiny non-Quanto \`mm\`. This is only a hypothesis; enforce the pre-existing numerical gates, bounded transient workspace, and no persistent duplicated model weights. Do not change defaults without repeated wall-to-wall A/B and quality (including face and audio) checks.
5. Sampled kernel time is *not* directly subtractable from wall time. Synchronization, host launch, PCIe transfers, stream overlap and CPU+GPU critical-path effects need a live-video comparison.
