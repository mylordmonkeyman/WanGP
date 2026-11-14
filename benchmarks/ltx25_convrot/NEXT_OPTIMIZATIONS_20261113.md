# Candidate E follow-on optimizations (November 13, 2026)

**Base backup:** `backup/ltx25-candidate-e-radix4-vector8-2026-11-13`, SHA `ae0499f4d768ff378392e78aa0d2421fff17c568`.
**Development:** `perf/ltx25-convrot-sm70`.
**Status:** Experimental, opt-in; no promotion until V100 preflight and controlled full-video quality and timings pass.

## Stable baseline

- Candidate E = SM70 FP32 radix-4 rotation + CUDA-safe per-row FP16 range guard / FP32 exceptional-row correction + transient exact vector8 INT8-to-FP16 weights + cuBLAS FP16 operands and FP32 accumulator/output + final BF16/FP16 conversion.
- The stable default remains `WAN2GP_LTX25_ROT_INPUT=staged` and `WAN2GP_LTX25_E_OUTPUT=torch`.
- Roll back to original WanGP with `WAN2GP_LTX25_CUBLAS_E_MODE=off`.
- Neither experimental optimization changes the checkpoint or caches a second persistent weight set.

## Experiment A: direct BF16/FP16 radix-4 input

`WAN2GP_LTX25_ROT_INPUT=direct`

Previously: `x.float().reshape(...).contiguous()` allocates a full FP32 input, then CUDA radix4 loads the FP32 tensor.

Experimental: one CUDA radix4 kernel loads original BF16/FP16 input and converts each value exactly to FP32 in its shared-memory staging buffer. Same four stages, input order, output FP32 values, and reduction order. One PyTorch conversion kernel and one transient FP32 activation allocation eliminated per eligible call.

Both modes retain `WAN2GP_LTX25_HADAMARD=radix4` and `WAN2GP_LTX25_E_CAST=vector8`. Unsupported layers continue to WanGP's existing R0 fallback.

## Experiment B: fused FP32 epilogue to BF16/FP16

`WAN2GP_LTX25_E_OUTPUT=fused`

Previously: FP32 row-scale / optional bias / exceptional-row correction write FP32 `c32`, then PyTorch `c32.to(x.dtype)` launches a separate conversion and allocates output.

Experimental: a dtype-specialized copy of the existing epilogue computes the same FP32 values, writes final BF16/FP16 directly with explicit RN conversion, and does not need to store the corrected FP32 value back to `c32`. It removes one conversion kernel launch per eligible linear. The existing FP32-output epilogue remains intact and is the default. No changes to the range guard, cuBLAS math, FP32 exceptional-row correction, weight expansion, or video/audio conditioning.

## Tests

Run the self-hosted V100 workflow:
`.github/workflows/ltx25-rot-direct-v100.yml`

It:
1. Compiles the native SM70 CUDA source (does not require a free GPU).
2. Refuses GPU tests while another process owns the V100.
3. Compares staged and direct FP32 Hadamard tensors **bit-for-bit** for BF16 and FP16 with several token shapes.
4. Compares full Candidate E outputs **bit-for-bit** for each combination of staged/direct rotation and torch/fused output, with bias and FP32 row guard.
5. Runs alternating (A-B-B-A) GPU-event timing for input rotation and fused epilogue vs. torch cast.
6. Does not assert full video or spoken-audio equivalence: that needs the user's 241-frame I2V visual+audio gate.

### Full-model promotion checklist

- Native CUDA compilation and all BF16/FP16 preflights successful on V100.
- At least two interleaved matched video-generation runs per contender with same input image, prompt, negative prompt, seed, 8-step distilled LoRA, 241 frames, SDPA, MMGP profile0.
- Verify **identical decoded video frames and identical generated audio** against the current Candidate E baseline, or understand any differences before promotion.
- Check transient/persistent VRAM use; no persistent alternate weight set, no unguarded host sync, no overflow/underflow regressions.
- Promote only if end-to-end advantage exceeds noise. Otherwise retain the validated baseline defaults.

Note: the earlier Candidate E output has a user-observed unwanted-speech issue relative to stock. These two optimizations are intended to preserve Candidate E results, not fix that speech difference. Compare audio separately before declaring a new default.
