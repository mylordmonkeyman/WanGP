# Experimental SM70 Candidate E — device-side FP16 range guard

**Scope:** LTX-2.5 22B Distilled INT8 ConvRot, NVIDIA V100 (SM70).
**Status:** Research-only, opt-in through `WAN2GP_LTX25_CUBLAS_E_MODE=fp32rot`.
**Production eligible:** NO. This is not an `auto` backend.
**Reversible snapshot:** `archive/ltx25-convrot-sm70-pre-sync-20261112`, commit
`4adc57c0d17f960d91415564db68fa6ab9360537`.

## What changed

The archived Python adapter inspected the converted FP16 activations with
`torch.isfinite(x16).all().item()` and an underflow `any().item()` for
**each optimized linear**. Both host scalar reads force CUDA stream
synchronization and can interrupt overlapped inference/offload.

The new adapter uses three asynchronous native functions on the *same*
current CUDA stream:

1. `ltx25_cublas_prepare_f32_f16`: reduce each FP32-rotated activation
   row on the device. Choose an exact power-of-two scale when all nonzero
   values remain representable in FP16. Allow *subnormal* FP16 operands down
   to `2^-24`, which are tested on the actual V100 by the preflight.
   Preserve a conservative FP32 fallback for rows with maxima above 32768,
   mixed ranges that cannot avoid underflow, or nonfinite inputs.
   Unsafe rows are zeroed before cuBLAS so they cannot contaminate its
   FP32 output.
2. `ltx25_cublas_gemm_f16_f32`: unscaled checkpoint INT8 weights are
   temporarily cast exactly to FP16 and multiplied by the prepared FP16
   activations. cuBLAS writes **FP32** C. The checkpoint and stored FP32
   scales are unchanged.
3. `ltx25_cublas_epilogue_f32`: normal rows unscale in FP32; unsafe
   rows compute a full direct FP32 dot with the *original* INT8 weights
   and the current FP32-rotated activation. Apply the stored FP32 row
   scales and optional bias. This fallback completes **in the same
   operation**, before the BF16/FP16 output is consumed.

An exceptionally ill-conditioned row has a slower FP32 path but does not
cause CPU polling. A small persistent device counter is read only at
process exit, never in the measured layer hot path.

**Do not replace this with delayed error flags or removal of the safety
gate.** The experiment has not established production-grade numerical
equivalence for the actual model's intermediate activations.

## Isolated smoke test on the V100

From the feature-branch checkout, with the baked runner's CUDA/PyTorch
environment and offline model cache already prepared:

```bash
nvcc -O3 -std=c++17 -arch=sm_70 -shared -Xcompiler -fPIC \
  benchmarks/ltx25_convrot/ltx25_cublas_e.cu \
  -lcublas -o /tmp/libltx25_cublas_e_quality.so

WAN2GP_LTX25_CUBLAS_E_MODE=fp32rot \
LTX25_CUBLAS_E_LIB=/tmp/libltx25_cublas_e_quality.so \
/opt/venvs/ltx25-cu128/bin/python \
  benchmarks/ltx25_convrot/candidate_adapter_preflight.py
```

The test covers BF16 and FP16 inputs, explicit rejection of a disallowed
weight shape, a GPU FP32 exception, safe power-of-two scaling, direct SM70
cuBLAS preservation of an FP16 *subnormal* operand, and Python-level
assertions that neither `Tensor.item()` nor `torch.cuda.synchronize()`
occurs during the candidate call. This is **synthetic** testing; passing
it does not establish the spec's real-activation R0/R1 precision gate.

The self-hosted V100 workflow
`.github/workflows/ltx25-quality-1024.yml` performs this preflight
before launching a 1024×576, 49-frame, eight-step, fixed-seed video
with MMGP profile 0 and SDPA, and publishes raw run status to
`perf/ltx25-run-results`. Ordinary source pushes do not launch jobs;
the E signal file must be changed deliberately.

## Single-run performance record (not confidence intervals)

| Version | Full generation | Device FP32 fallback | Status |
|---|---:|---:|---|
| R0 unmodified WanGP | 157.9 s | N/A | Completed |
| Archived opt-in FP32-rotation E | 112.8 s | 1,800 whole-call range fallbacks | Completed |
| First GPU-only guard, FP16 *normal* constraint | 156.9 s | 306,419 *rows* | Completed but too slow |
| Revised GPU-only guard, permits validated FP16 subnormals | 98.5 s; repeated **98.3 s** | 4,434 *rows* in both runs | Completed twice |

All use 1024×576 / 49 frames / 8 steps / seed 20261112,
matching prompt and generation settings. Both revised-guard timings are independent identical-setting runs (98.5 s
and 98.3 s). The R0 and archived E figures are single runs, and there is no
interleaved R0/E median. CPU/GPU clocks, overlap, peak memory, and
qualitative video differences remain to be compared.

Evidence:
- R0: https://github.com/mylordmonkeyman/WanGP/actions/runs/43336958612
- Archived E: https://github.com/mylordmonkeyman/WanGP/actions/runs/43338937862
- Overconservative guard: https://github.com/mylordmonkeyman/WanGP/actions/runs/43366246858
- Revised guard: https://github.com/mylordmonkeyman/WanGP/actions/runs/43366794156
- Revised guard repetition (identical model settings): https://github.com/mylordmonkeyman/WanGP/actions/runs/43367142966

Counter caution: `range_fallback` in the archived adapter counted **calls**
whose *entire linear* was rejected; the revised counter counts individual
**rows** corrected on GPU. These counts are not directly comparable.

## Remaining gates

- Repeat R0/E trials under controlled GPU clocks, PCIe/thermal state,
  and offload profile, and report complete generation and denoising timing.
- **Completed bounded actual-activation screen:** 17/17 R0/R1/E sampled
  comparisons passed over seven eligible shapes (up to 16 rows per call).
  Broader token-row coverage, different prompts/seeds, real LoRA workloads,
  and full-output distribution tests are still required for production.
  Synthetic preflight alone remains insufficient.
- Compare at least three matched rendered clips with blind review and
  appropriate quantitative quality metrics; the present clip is not a
  claim of visual equivalence.
- Test LoRA, bias, MMGP offload/lifetime, changing inputs, and concurrency.
  CUDA Graph capture/torch.compile intentionally remain **ineligible**.
- Profile the cost of GPU row reduction, exceptional-row FP32 arithmetic,
  cuBLAS, allocation, and output conversion. The transient FP16 weight
  conversion is still allocated per call and is a separate optimization.
- Do not change other GPU backends, global dtype, checkpoint quantization,
  or the default selection while this work remains experimental.

## Real LTX-2.5 generation activation gate (completed)

The opt-in-only audit samples actual WanGP tensors during the fixed-seed
1024×576 generation. It computes the original native BF16 rotated
`F.linear` quantized-weight leaf (R0), an independent FP32 rotation/GEMM/
scale/bias reference (R1), and the candidate's already-generated BF16 output
(E) for the same live token rows. R1 is column-tiled, with at most 256 INT8
weight rows converted transiently to FP32 per chunk. The production fast path
does not import or execute this audit.

**Run:** https://github.com/mylordmonkeyman/WanGP/actions/runs/43368460982

- **17/17 samples passed**, zero numerical-gate failures and zero execution
  errors; all 7 passing checkpoint matrix-shape families were represented.
- Sampled calls 1, 64, and 1024 when reached by the family; only call 1
  was available for two low-frequency shapes.
- Each sample examined 16 token rows, selected across the actual M tokens;
  the test is not a full M×N numerical distribution audit.
- The FP32 reference observed output error for E was no worse than R0 at
  both p99 and maximum normalized error in every sample.
- The video was generated successfully, with 16,368 optimized linear calls
  and 4,434 device-side FP32 exceptional rows, consistent with previous
  no-sync tests. The audit run time (98.5 seconds) is not a speed benchmark
  because it performs extra reference work.

| Shape (N×K) | Samples | Max E p99 normalized | Max R0 p99 normalized |
|---|---:|---:|---:|
| 4096×4096 | 3 | 0.00339 | 0.00789 |
| 16384×4096 | 3 | 0.00353 | 0.00722 |
| 4096×16384 | 3 | 0.00341 | 0.00559 |
| 2048×2048 | 3 | 0.00340 | 0.01790 |
| 8192×2048 | 1 | 0.00346 | 0.00728 |
| 2048×8192 | 1 | 0.00238 | 0.00434 |
| 2048×4096 | 3 | 0.00379 | 0.02852 |

Maxima in different columns may come from different samples; consult the
per-sample JSON rather than treating this table as a matched numerator/
denominator ratio.

## Decoded-video reproducibility and pixel-difference audit (completed)

The isolated **no-generation** analysis uses cached R0, archived E, no-sync
E and audit E MP4s from the same prompt/seed/settings. It decodes and
verifies 49 frames per clip using FFmpeg framemd5, calculates SSIM and PSNR
on the decoded pixel stream, and uploads a five-frame-per-clip contact sheet.

**Run and contact sheet artifact:**
https://github.com/mylordmonkeyman/WanGP/actions/runs/43369208361

| Comparison | Decoded pixels identical? | SSIM | PSNR (dB) |
|---|---|---:|---:|
| R0 vs archived E | No | 0.888115 | 26.3523 |
| R0 vs no-sync E | No | 0.891763 | 26.5518 |
| Archived E vs no-sync E | No | 0.939273 | 30.0955 |
| No-sync E vs live-audit E | **Yes, all 49 frames** | 1.000000 | Infinite |

Inter-version differences are expected when the arithmetic is changed;
neither SSIM nor PSNR is a perceptual-quality metric or a pass/fail visual
gate. These outputs visibly vary in sail color, boat position and water
texture. Blind human assessment across several prompts/seeds is still
required before `auto` promotion. Reproducibility of no-sync E with
and without the audit strengthens the runtime consistency evidence only.
