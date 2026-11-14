# Candidate E follow-on optimizations (November 13, 2026)

**Base backup:** `backup/ltx25-candidate-e-radix4-vector8-2026-11-13`, SHA `ae0499f4d768ff378392e78aa0d2421fff17c568`.
**Development:** `perf/ltx25-convrot-sm70`.
**Status:** Direct radix-4 input and fused BF16/FP16 output promoted as V100 Candidate E defaults on 2026-11-13, following exact preflight, 49-frame ABBA, and 241-frame I2V checks.

## Stable baseline

- Candidate E = SM70 FP32 radix-4 rotation + CUDA-safe per-row FP16 range guard / FP32 exceptional-row correction + transient exact vector8 INT8-to-FP16 weights + cuBLAS FP16 operands and FP32 accumulator/output + final BF16/FP16 conversion.
- Current defaults: `WAN2GP_LTX25_ROT_INPUT=direct` and `WAN2GP_LTX25_E_OUTPUT=fused`. Explicitly set `staged`/`torch` for previous Candidate E behavior.
- Roll back to original WanGP with `WAN2GP_LTX25_CUBLAS_E_MODE=off`.
- Neither experimental optimization changes the checkpoint or caches a second persistent weight set.

## Experiment A: direct BF16/FP16 radix-4 input

`WAN2GP_LTX25_ROT_INPUT=direct`

Previously: `x.float().reshape(...).contiguous()` allocates a full FP32 input, then CUDA radix4 loads the FP32 tensor.

Promoted: one CUDA radix4 kernel loads original BF16/FP16 input and converts each value exactly to FP32 in its shared-memory staging buffer. Same four stages, input order, output FP32 values, and reduction order. One PyTorch conversion kernel and one transient FP32 activation allocation eliminated per eligible call.

Both modes retain `WAN2GP_LTX25_HADAMARD=radix4` and `WAN2GP_LTX25_E_CAST=vector8`. Unsupported layers continue to WanGP's existing R0 fallback.

## Experiment B: fused FP32 epilogue to BF16/FP16

`WAN2GP_LTX25_E_OUTPUT=fused`

Previously: FP32 row-scale / optional bias / exceptional-row correction write FP32 `c32`, then PyTorch `c32.to(x.dtype)` launches a separate conversion and allocates output.

Promoted: a dtype-specialized copy of the existing epilogue computes the same FP32 values, writes final BF16/FP16 directly with explicit RN conversion, and does not need to store the corrected FP32 value back to `c32`. It removes one conversion kernel launch per eligible linear. The existing FP32-output epilogue remains intact for rollback via `WAN2GP_LTX25_E_OUTPUT=torch`. No changes to the range guard, cuBLAS math, FP32 exceptional-row correction, weight expansion, or video/audio conditioning.

## Tests

Run the self-hosted V100 workflow:
`.github/workflows/ltx25-rot-direct-v100.yml`

It:
1. Compiles the native SM70 CUDA source (does not require a free GPU).
2. Refuses GPU tests while another process owns the V100.
3. Compares staged and direct FP32 Hadamard tensors **bit-for-bit** for BF16 and FP16 with several token shapes.
4. Compares full Candidate E outputs **bit-for-bit** for each combination of staged/direct rotation and torch/fused output, with bias and FP32 row guard.
5. Runs alternating (A-B-B-A) GPU-event timing for input rotation and fused epilogue vs. torch cast.
6. Does not itself assert full-video/audio equivalence; those checks were subsequently performed in the matched 49-frame ABBA and 241-frame I2V tests.

### Full-model promotion evidence and limitations

- Native CUDA compilation and all BF16/FP16 preflights successful on V100.
- Completed 49-frame two-per-arm ABBA run (matched inputs; decoded video/audio identical) and a separate 241-frame I2V A/B run (matched input image, prompt, seed, LoRA; decoded video/audio identical). The 241-frame result has only one run per arm.
- Verified **identical decoded video frames and generated audio** between current Candidate E and the direct+fused variant for both tested workloads.
- Check transient/persistent VRAM use; no persistent alternate weight set, no unguarded host sync, no overflow/underflow regressions.
- Measured 49-frame ABBA improvement: 4.95% (baseline median ~96.0s, direct+fused median ~91.25s). Measured 241-frame A/B improvement: 0.96% (572.336s vs 566.844s). The latter is small enough to include run-to-run noise; promotion was user-authorized with rollback preserved.

Note: earlier Candidate E output had a user-observed unwanted-speech issue relative to stock. These two optimizations are intended to preserve Candidate E results, not fix that speech difference. Compare audio separately before declaring a new default.
