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

## Priority 2: stage profiling and reusable scratch — staged, not GPU validated

**Last V100-validated no-sync reference snapshot:**
`archive/ltx25-convrot-sm70-post-sync-validated-20261113` at
`32bc42e6389e79ab6f742da9896364aa5bd74b3c`.

The active branch now contains opt-in-only instrumentation and a scratch
experiment. The original `fp32rot` default behavior remains unchanged;
no automatic selection or production promotion has occurred.

**Live-model stage profiler:** `WAN2GP_LTX25_PROFILE_STAGES=1` enables
`shared/kernels/ltx25_stage_profiler.py`. It records a sparse set of
same-stream CUDA events around rotation, scratch setup, INT8->FP16 weight
conversion, range preparation, cuBLAS, FP32 epilogue and output conversion.
Events are resolved **only at interpreter exit**. Event-to-event time
includes host enqueue gaps and potential concurrent GPU traffic; compare
warm calls separately from first-use initialization. The dedicated
`e_profile_1024` run publishes raw stage samples under
`ltx25-quality/e_profile_1024/` on `perf/ltx25-run-results`.

**Bounded scratch pilot:** `WAN2GP_LTX25_E_WORKSPACE=1` enables
`shared/kernels/ltx25_workspace_sm70.py`. One default CUDA stream per
GPU may reuse FP16 activation, FP32 accumulator, and two int32 row-state
buffers (default cumulative capacity <=96 MiB). Oversized operations,
nondefault streams, and concurrent/reentrant calls bypass the pool and
retain their original ephemeral allocations. Leases release after output
conversion has been queued on the same stream; no host synchronization.
The original INT8 checkpoint and FP32 row scales remain MMGP-owned.
**No persistent FP16 weight copies are cached**; `q.to(torch.float16)`
still happens for each call. This first optimization targets allocation
churn only; weight conversion must be measured before trying an alternate
kernel or approach.

**Workspace quality gate:** `e_workspace_1024` runs
`candidate_workspace_preflight.py`, then the same fixed-seed 1024x576
49-frame generation, then `workspace_video_gate.py`, which demands exact
decoded-yuv420p equality across **all 49 frames** against the previously
validated no-sync E clip. A mismatch is recorded as a failed gate.
Do not interpret equality of this one clip as general visual validation.

### Current test state

- Hosted **CPU-only source checks PASSED** (Python compilation and
  strict opt-in / transient weight contract):
  https://github.com/mylordmonkeyman/WanGP/actions/runs/43454931381
  and https://github.com/mylordmonkeyman/WanGP/actions/runs/43455283021.
- First stage-profile run:
  https://github.com/mylordmonkeyman/WanGP/actions/runs/43453044580
  **SKIPPED GPU GENERATION (GPU_BUSY)**.
- Retried stage-profile run:
  https://github.com/mylordmonkeyman/WanGP/actions/runs/43454440544
  **SKIPPED GPU GENERATION (GPU_BUSY)**. Another GPU compute owner
  was reported as PID `124951` using `342 MiB`; PID's executable name
  is not visible in the runner's process namespace.
- The quality workflow now checks GPU ownership **before any CUDA
  preflight** so no further profiling/benchmark work will consume
  V100 resources while another process owns the device.
- The GPU scratch path has **not** yet passed V100 preflight, full video
  pixel-equivalence or a matched uninstrumented timing comparison.
  There is **no measured Priority 2 speedup yet**.

### Next exact sequence, when the GPU is idle

1. Deliberately update `benchmarks/ltx25_convrot/stage_profile_dispatch.signal`
   to trigger `e_profile_1024`, evaluate warm GPU stage times per shape.
2. Deliberately update `benchmarks/ltx25_convrot/workspace_dispatch.signal`
   to trigger `e_workspace_1024`, requiring V100 preflight, actual leases,
   full video and strict 49-frame equality. Do not run concurrently.
3. If both pass, run matched **uninstrumented** E vs workspace E
   repetitions with idle GPU and comparable clocks/thermal conditions.
4. Retain the scratch pilot only if it measurably speeds full generation
   without destabilizing MMGP or increasing peak VRAM unacceptably.
   Otherwise revert to the validated no-sync snapshot and prioritize
   the dominant operation identified by the profiler.

## Priority 2 V100 run evidence (Nov 13, 2026)

User confirmed that the competing VRAM owner was terminated before these
tests were dispatched. The existing V100 ownership guard verified GPU idle
for each completed generation.

### Sparse real-model CUDA-event stage profiling

**PASS:** https://github.com/mylordmonkeyman/WanGP/actions/runs/43455709423

- Fixed 1024x576, 49 frames, 8 steps, same seed and MMGP profile 0.
- 37 actual eligible-call samples across all seven families; no profiler
  errors; 16,368 optimized calls and 4,434 device-side FP32 fallback rows.
- Instrumented generation: **101.4 s**, NOT a comparable uninstrumented
  wall-time baseline.
- Median warm-stage CUDA event observations (milliseconds):

| N×K | Scratch setup | INT8→FP16 | GEMM | Rotation |
|---|---:|---:|---:|---:|
| 16384×4096 | 0.0020 | 0.4767 | 0.5596 | 0.0461 |
| 4096×16384 | 0.0020 | 0.4767 | 0.5852 | 0.1341 |
| 4096×4096 | 0.0143 | 0.1321 | 0.4649 | 0.2494 |
| 2048×2048 | 0.0072 | 0.0368 | 0.1331 | 0.1249 |

Interpretation: for the two frequent 64M-weight families, the FP16
cast is comparable to cuBLAS GEMM time, while scratch setup is tiny.
Samples are sparse and do not isolate GPU time from host enqueue gaps
or concurrent memory activity. This makes weight conversion a credible
next performance experiment; it is not evidence that scratch reuse
alone will yield a large speedup.

### Bounded reusable scratch pilot

**PASS:** https://github.com/mylordmonkeyman/WanGP/actions/runs/43456268178

- Completed full matched generation at **97.2 s** (single uninstrumented
  workspace run; not an established speedup over 98.3/98.5 s reference).
- Acquired the scratch lease for 16,360/16,368 eligible calls; eight
  oversize calls correctly used ephemeral allocations.
- Maximum persistent scratch = 100,171,264 bytes (95.5 MiB; under 96 MiB cap).
- No nondefault stream or reentrant reuse; zero persistent FP16 weight sets.
- Opt-in V100 workspace preflight and stream/lifetime safety tests passed.
- The resulting MP4's **all 49 decoded frames are exactly identical**
  to established no-sync Candidate E (`a566cf56eae8`).
- Production default remains unchanged, as does checkpoint quantization.

### Isolated conversion experiment

New standalone SM70 kernels and a separate workflow compare PyTorch
`q.to(float16)` and preallocated PyTorch `copy_` against CUDA scalar and
vector4 INT8→FP16 conversions using exact synthetic int8 tensors with
checkpoint-realistic shapes. This test does not load the model, retain
weights, or change Candidate E. Source:
`benchmarks/ltx25_convrot/ltx25_cast_bench.cu` and
`benchmarks/ltx25_convrot/cast_microbench.py`.

**No conversion speedup is established until the isolated kernel completes
and matches PyTorch bit-exactly across the tested shapes.** If promising,
a separate opt-in full-generation path will be required with V100
preflight, real-activation R0/R1 gate and exact decoded-frame regression.

## Priority 2: vector4 cast and combined V100 gates (Nov 13, 2026)

The isolated, explicitly signaled SM70 conversion microbenchmark passed
with exact FP16 output bits for **all six tested matrix shapes**.
Evidence: https://github.com/mylordmonkeyman/WanGP/actions/runs/43456883623

| Weight shape N×K | PyTorch q.to(FP16) median (ms) | Vector4 median (ms) | Native speedup |
|---|---:|---:|---:|
| 2048×2048 | 0.0553 | 0.0343 | 1.61× |
| 4096×4096 | 0.1310 | 0.1024 | 1.28× |
| 2048×4096 | 0.0686 | 0.0533 | 1.29× |
| 16384×4096 | 0.5028 | 0.3963 | 1.27× |
| 4096×16384 | 0.4813 | 0.3973 | 1.21× |
| 8192×2048 | 0.1244 | 0.1024 | 1.22× |

These are isolated CUDA-event medians of 50 conversions using synthetic
but checkpoint-realistic INT8 shapes. Preallocated PyTorch copy_ and native
scalar-kernel controls were also measured. The results do NOT directly
predict complete generation time.

`WAN2GP_LTX25_E_CAST=vector4` enables an opt-in exact vectorized CUDA
conversion inside the existing `ltx25_cublas_e.cu` extension. It creates
the same **transient FP16 weight buffer** as the PyTorch cast and uses the
same original INT8 tensor, FP32 checkpoint row scales and FP32 fallback.
The kernel issues no device-to-host synchronization; default Candidate E
still uses `q.to(torch.float16)`.

### Full real-model gate — vector4 only

**PASS:** https://github.com/mylordmonkeyman/WanGP/actions/runs/43458031578

- Full 1024×576 49-frame 8-step generation: **95.8 seconds**, one run.
- CUDA preflight: vector4 and PyTorch-cast complete Candidate E outputs
  matched **exactly** for FP16 and BF16 inputs and four eligible layer shapes.
- Original checkpoint INT8 values unchanged; no per-linear CPU sync.
- All 49 decoded frames **identical** to validated original Candidate E.
- 16,368 optimized linear calls; 4,434 FP32 exceptional input rows.
- This suggests about 2.6% wall-time improvement versus the previous
  98.3–98.5 s uninstrumented E results, **pending controlled repeats**.

### Full real-model gate — vector4 + scratch workspace

**PASS:** https://github.com/mylordmonkeyman/WanGP/actions/runs/43458837188

- Full generation: **95.6 seconds**, one run.
- Both independent opt-in CUDA preflights pass.
- 16,360/16,368 calls reused workspace; eight oversized calls used the
  ephemeral fallback. Scratch max 100,171,264 bytes (~95.5 MiB).
- Zero persistent FP16 weight copies; vector4 kernel still converts the
  weight into a transient buffer every call.
- **All 49 decoded video frames identical** to known-good Candidate E.
- The 0.2 s difference between vector4-only and combined variants
  is too small to claim separately without more repeats.

Reversible exact tested-source snapshot:
`archive/ltx25-convrot-sm70-vector4-workspace-validated-20261113` at
`0d76d647c79c7af0d5c10d09c2269a7b5ecb2fb5`.

**Remaining acceptance:** fresh same-session uninstrumented baseline
and combined repeats; representative peak VRAM/thermal control; broader
real activations, prompts/seeds and LoRA/MMGP compatibility. Neither
experimental variant is production-eligible or auto-selected.

### Fresh same-session B–A–B–A generation timing screen

All four runs used the same 1024×576, 49-frame, eight-step, fixed-seed
generation, single V100, MMGP profile 0, SDPA, and the same experiment
branch code apart from result notes and signal triggers. They used the
early GPU-ownership guard. The A variant uses original PyTorch FP16
cast with no reusable scratch; B enables both transient vector4 cast and
bounded reusable activation/accumulator scratch. Each generation succeeded.

| Run order | Variant | Time (s) | Run |
|---|---|---:|---|
| B1 | Vector4 + workspace | 95.6 | [43458837188](https://github.com/mylordmonkeyman/WanGP/actions/runs/43458837188) |
| A1 | Default E: torch cast/no reusable pool | 96.6 | [43459483961](https://github.com/mylordmonkeyman/WanGP/actions/runs/43459483961) |
| B2 | Vector4 + workspace | 96.3 | [43460002568](https://github.com/mylordmonkeyman/WanGP/actions/runs/43460002568) |
| A2 | Default E: torch cast/no reusable pool | 97.5 | [43460664510](https://github.com/mylordmonkeyman/WanGP/actions/runs/43460664510) |

The actual chronological order was **B1 → A1 → B2 → A2**.
A is the unmodified Candidate E PyTorch-cast control, and B is the
combined experimental variant.

Two-run arithmetic mean: A = **97.05 s**, B = **95.95 s**;
mean observed difference **1.10 s (~1.13%)**. The B run durations
were 95.6 and 96.3 s, versus A runs of 96.6 and 97.5 s. Sample sizes
are too small for a confidence interval and clocks/thermal state were
not captured; treat the advantage as **preliminary**. Workspace alone
had previously measured 97.2 s (one run).

Both B runs passed the **strict 49-decoded-frame equality test** against
the validated original E video, and the original GPU-side 4,434 FP32
exceptional row count remained unchanged. No persistent FP16 weight
copies were introduced. Broader prompts, LoRA/MMGP lifetime tests
and real-model production numerical/visual gates remain pending.
