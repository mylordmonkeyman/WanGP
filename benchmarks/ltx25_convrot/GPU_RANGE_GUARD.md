# Experimental SM70 Candidate E — device-side FP16 range guard

**Scope:** LTX-2.5 22B Distilled INT8 ConvRot, NVIDIA V100 (SM70).
**Status:** Research-only, opt-in through \`WAN2GP_LTX25_CUBLAS_E_MODE=fp32rot\`.
**Production eligible:** NO. This is not an \`auto\` backend.
**Reversible snapshot:** \`archive/ltx25-convrot-sm70-pre-sync-20261112\`, commit
\`4adc57c0d17f960d91415564db68fa6ab9360537\`.

## What changed

The archived Python adapter inspected the converted FP16 activations with
\`torch.isfinite(x16).all().item()\` and an underflow \`any().item()\` for
**each optimized linear**. Both host scalar reads force CUDA stream
synchronization and can interrupt overlapped inference/offload.

The new adapter uses three asynchronous native functions on the *same*
current CUDA stream:

1. \`ltx25_cublas_prepare_f32_f16\`: reduce each FP32-rotated activation
   row on the device. Choose an exact power-of-two scale when all nonzero
   values remain representable in FP16. Allow *subnormal* FP16 operands down
   to \`2^-24\`, which are tested on the actual V100 by the preflight.
   Preserve a conservative FP32 fallback for rows with maxima above 32768,
   mixed ranges that cannot avoid underflow, or nonfinite inputs.
   Unsafe rows are zeroed before cuBLAS so they cannot contaminate its
   FP32 output.
2. \`ltx25_cublas_gemm_f16_f32\`: unscaled checkpoint INT8 weights are
   temporarily cast exactly to FP16 and multiplied by the prepared FP16
   activations. cuBLAS writes **FP32** C. The checkpoint and stored FP32
   scales are unchanged.
3. \`ltx25_cublas_epilogue_f32\`: normal rows unscale in FP32; unsafe
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

\`\`\`bash
nvcc -O3 -std=c++17 -arch=sm_70 -shared -Xcompiler -fPIC \
  benchmarks/ltx25_convrot/ltx25_cublas_e.cu \
  -lcublas -o /tmp/libltx25_cublas_e_quality.so

WAN2GP_LTX25_CUBLAS_E_MODE=fp32rot \
LTX25_CUBLAS_E_LIB=/tmp/libltx25_cublas_e_quality.so \
/opt/venvs/ltx25-cu128/bin/python \
  benchmarks/ltx25_convrot/candidate_adapter_preflight.py
\`\`\`

The test covers BF16 and FP16 inputs, explicit rejection of a disallowed
weight shape, a GPU FP32 exception, safe power-of-two scaling, direct SM70
cuBLAS preservation of an FP16 *subnormal* operand, and Python-level
assertions that neither \`Tensor.item()\` nor \`torch.cuda.synchronize()\`
occurs during the candidate call. This is **synthetic** testing; passing
it does not establish the spec's real-activation R0/R1 precision gate.

The self-hosted V100 workflow
\`.github/workflows/ltx25-quality-1024.yml\` performs this preflight
before launching a 1024×576, 49-frame, eight-step, fixed-seed video
with MMGP profile 0 and SDPA, and publishes raw run status to
\`perf/ltx25-run-results\`. Ordinary source pushes do not launch jobs;
the E signal file must be changed deliberately.

## Single-run performance record (not confidence intervals)

| Version | Full generation | Device FP32 fallback | Status |
|---|---:|---:|---|
| R0 unmodified WanGP | 157.9 s | N/A | Completed |
| Archived opt-in FP32-rotation E | 112.8 s | 1,800 whole-call range fallbacks | Completed |
| First GPU-only guard, FP16 *normal* constraint | 156.9 s | 306,419 *rows* | Completed but too slow |
| Revised GPU-only guard, permits validated FP16 subnormals | 98.5 s | 4,434 *rows* | Completed |

All use 1024×576 / 49 frames / 8 steps / seed 20261112,
matching prompt and generation settings. Each timing is an independent
single run, not a repeated interleaved median; CPU/GPU clocks, overlap,
peak memory, and qualitative video differences remain to be compared.

Evidence:
- R0: https://github.com/mylordmonkeyman/WanGP/actions/runs/43336958612
- Archived E: https://github.com/mylordmonkeyman/WanGP/actions/runs/43338937862
- Overconservative guard: https://github.com/mylordmonkeyman/WanGP/actions/runs/43366246858
- Revised guard: https://github.com/mylordmonkeyman/WanGP/actions/runs/43366794156

Counter caution: \`range_fallback\` in the archived adapter counted **calls**
whose *entire linear* was rejected; the revised counter counts individual
**rows** corrected on GPU. These counts are not directly comparable.

## Remaining gates

- Repeat R0/E trials under controlled GPU clocks, PCIe/thermal state,
  and offload profile, and report complete generation and denoising timing.
- Capture representative *actual* LTX-2.5 activations, validate candidate
  **versus R1** and R0 versus R1 using the spec's per-shape p99/max
  normalized error acceptance rule. Synthetic preflight is insufficient.
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
