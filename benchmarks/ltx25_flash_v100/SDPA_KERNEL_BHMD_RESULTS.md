# Verified V100 SDPA CUDA kernel and 1Cat lower-level BHMD results

Tested on Tesla V100 SM70, PyTorch 2.10.0+cu128, CUDA 12.8, 2026-11-13.

GitHub Actions run:
https://github.com/mylordmonkeyman/WanGP/actions/runs/43451369156

Full JSON, Chrome traces and measurements:
https://github.com/mylordmonkeyman/WanGP/actions/runs/43451369156/artifacts/14050057224

## Exact SDPA runtime selection (observed)

- BF16 QKV, B=1, T=315, H=32, D=128: PyTorch operators
  aten::scaled_dot_product_attention -> aten::_scaled_dot_product_attention_math.
  CUDA events include volta_sgemm_64x64_tn, volta_sgemm_64x64_nn,
  softmax_warp_forward<float,float,float,...>, and elementwise/copy kernels.
  Forced backend probes on this BF16 shape: MATH succeeds;
  FLASH_ATTENTION, EFFICIENT_ATTENTION and CUDNN_ATTENTION all reject it.
  This is NOT a fused BF16 attention kernel.
- FP16 QKV, B=1, T=4096, H=8, D=128: PyTorch operators
  aten::scaled_dot_product_attention -> aten::_scaled_dot_product_efficient_attention
  -> aten::_efficient_attention_forward.
  Actual GPU kernel: fmha_cutlassF_f16_aligned_32x128_rf_sm70
  (PyTorchMemEffAttention AttentionKernel on cutlass::arch::Sm70).
  This is PyTorch's fused CUTLASS memory-efficient attention on Volta.

## Controlled same-run comparisons

All numbers are median CUDA event milliseconds per invocation, profiled
separately from timing. All cases passed numerical accuracy comparisons.

### Real-video shape: 315 tokens, 32 heads, D128, BF16 input

| Path | ms | Peak extra GPU memory |
|---|---:|---:|
| PyTorch native BF16 SDPA (math) | 0.43076 | 46.952 MiB |
| BF16 -> FP16 SDPA -> BF16 (including conversion) | **0.15215** | 9.844 MiB |
| BF16 -> FP16 1Cat public -> BF16 | 0.27511 | 19.726 MiB |
| BF16 -> FP16 1Cat direct BHMD views -> BF16 | 0.25993 | 17.266 MiB |

In this run, converted FP16 SDPA was 2.83x faster than native BF16
math SDPA. The 1Cat direct BHMD path was 1.71x slower than converted
SDPA, even including both dtypes' conversion costs.

### FP16 already in WanGP BMHD layout

| Shape | PyTorch SDPA (BMHD) | 1Cat public (BMHD) | 1Cat direct BHMD views |
|---|---:|---:|---:|
| T315 H32 D128 | **0.09770** | 0.22627 | 0.20475 |
| T1024 H8 D128 | **0.21432** | 0.45471 | 0.44083 |
| T4096 H8 D128 | **2.29492** | 4.60490 | 4.48161 |
| T8192 H32 D128 | **32.63615** | 71.56033 | 71.31398 |

The public 1Cat wrapper explicitly converts to contiguous BHMD
for Q/K/V and converts the output back to BMHD. Direct BHMD avoids
some wrapper conversions, but transposed BMHD views still cause
internal contiguous input copies. Improvements from API change
alone are small, especially for long sequences.

### Both APIs given preformatted contiguous BHMD tensors

Cost to materialize BHMD QKV is excluded equally for both operators.
These results more closely compare CUDA execution rather than total
WanGP pathway integration costs.

| Shape | PyTorch SDPA prepacked | 1Cat direct BHMD prepacked |
|---|---:|---:|
| T315 H32 D128 | **0.09323** | 0.16080 |
| T1024 H8 D128 | **0.21008** | 0.40237 |
| T4096 H8 D128 | **2.04949** | 4.38144 |
| T8192 H32 D128 | **31.95148** | 70.68993 |

At T8192, even prepacked PyTorch is 2.21x as fast as prepacked 1Cat.
The underlying 1Cat CUDA execution, not merely public wrapper copies,
is the primary source of its long-sequence slowdown.

## Recommendation

- BF16 native SDPA uses PyTorch's math path on SM70; FP16 uses
  fmha_cutlassF_f16_aligned_32x128_rf_sm70.
- For real BF16 video attention, investigate using PyTorch FP16 SDPA
  with bounded BF16-to-FP16 / FP16-to-BF16 conversions. Numerically
  passing random synthetic QKV is necessary, but not full LTX-2.5
  output-quality validation.
- Do not replace SDPA with the current 1Cat kernel; its direct
  contiguous BHMD path is slower on all tested shapes.
- These microbenchmarks are not whole-video speed measurements.
