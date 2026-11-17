# P0 V100 FP16 persistent hot baseline — first measured result

**Workflow:** https://github.com/mylordmonkeyman/WanGP/actions/runs/44013580271  
**Run / source checkout:** `44013580271` / `e52afbdef1be8a2450591bc8ae043b6d84c21161`  
**Environment:** Tesla V100 32GB, offline original Comfy-Org HunyuanVideo 1.5 FP16 and preconverted LightX2V rank-32 LoRA.  
**Settings:** 832×480, 33 frames, four denoise steps, guidance 1, flow_shift 8, seed 20261112, fixed cocktail-bar prompt, requested/resolved SDPA, MMGP 0.  
**Model SHA256:** `6889f79cf053812e22afa8095cec77c4ad913b43f112dd10a425845f44e3234b`  
**Converted LoRA SHA256:** `9257f364a89792d17087919c37b73891d32f8b560ee59054a8f04abbe0c6a1c5`

## Measurements (not optimization gains)

| Observation | Measured seconds |
|---|---:|
| Whole cold process, one video | 68.9 |
| Cold request within that process | 49.891 |
| First generation in the persistent queue, discarded warm-up | 44.428 |
| Five subsequent hot E2E samples | 34.609, 34.818, 34.684, 34.631, 34.651 |
| Hot E2E median (MAD) | **34.651 (0.034)** |
| Provisional denoise phase median | **19.657** |
| VAE decode/GPU→CPU copy phase median | **11.706** |
| Separate output phase median | **1.609** |
| Preconditioning phase median | **1.693** |
| Conditioning phase median, repeated prompt | **0.002** |

The provisional phases reconcile approximately to hot E2E, but may misattribute
early MMGP activation and transfers before denoise_begin. They are **not yet a
verified inclusive-denoise measurement**, and no optimized implementation
has been benchmarked against this reference.

**State checks:** one PID, one model/pipeline/transformer/offload identity, one
checkpoint/model load, expected 54 double-stream blocks, resolved SDPA, and
six separately produced, decodable 33-frame MP4s. The five hot measurements
exclude the initial generation. The changed-prompt run also passed, with
hot timings `34.770, 34.720, 34.578, 34.653, 36.156` seconds; the last
changed prompt incurred ~1.415 seconds in the conditioning phase.

## Separate synchronized diagnostic (not adoption timing)

- Actual attention Q/K tensor shape `[1, 14159, 16, 128]`, for 14,040 video
  tokens and 119 effective text tokens. The Hunyuan conditioning text tensor
  has padding `[1, 511, 3584]` and text_mask `[1, 511]`.
- Exactly **216 attention calls** (54 blocks × 4 steps) in one diagnostic request.
- Across **864 Q/K pre-normalization tensor observations** (image/text Q/K,
  54 blocks × 4 denoising steps), largest absolute pre-RMSNorm value was
  **44.5**, with **zero** elements above the FP16 square-overflow threshold
  of approximately 255.94 in the measured sample. This is **not** a proof
  of safety across seeds or other workloads.
- The maximum absolute difference between legacy norm and the sampled FP32
  norm reference was **0.03578** (reference evaluates 16 tokens per tensor,
  not a full-tensor numerical parity gate).
- VAE module runtime dtype was **`torch.float16`**, not FP32 merely because
  its checkpoint filename mentions FP32. Tiling enabled spatially and
  temporally, configured sample tile 256 pixels, 64 frames, overlap 0.25.
  Latent `[1, 32, 9, 30, 52]`. Exactly **15 executed decoder calls**.
- At completion, all 54 block parameter placements reported CPU; this
  snapshot does not show actual residency DURING denoise or prove that no
  weights are reused in pinned host storage.
- Peak `torch.cuda.max_memory_allocated()` was **5.64 GiB**; this is a
  PyTorch allocator peak, not a total device utilization/VRAM headroom claim.

## Attention dispatch and synchronization follow-up

Profiler-only runs:
- Initial report parsing failed on `FunctionEventAvg.name` (GitHub Actions
  `44016941390`); corrected to `.key` and reran as
  https://github.com/mylordmonkeyman/WanGP/actions/runs/44019009553 .
- Resolved PyTorch dispatch operators: `aten::scaled_dot_product_attention`
  → `aten::_scaled_dot_product_efficient_attention`
  → `aten::_efficient_attention_forward`.
- **Exact CUDA kernel function name remains unknown.** The CUDA profiler
  reported a CUPTI callback-thread error; its visible events were two copy
  categories, not the expected attention kernel name. Do not falsely claim
  a particular CUDA kernel from the PyTorch operator label.
- P0 traces show **216 SDPA calls per 4-step clip**, each carrying
  `q_lens=k_lens=[14159]`. Existing batch-1 SDPA code performs two
  scalar `.item()` reads per call (code-derived **432 synchronous reads
  per clip**). A separately flagged P2A candidate converts that length
  once per transformer forward (four forward calls at the reference geometry,
  expected four GPU→CPU length reads); the performance/quality consequence
  remains under test, and no default path was changed.

## P0 gate outstanding

Offload-inclusive denoise preloads, per-request H2D/D2H bytes/waits,
pinned-host-pool reuse and retained allocation identity, actual selected
SDPA CUDA kernel, RoPE/GEMM/LoRA profiler attribution, checkpoint and encoder
startup subphases, VAE decode autocast/accumulation dtypes and full-frame
lossless correctness fixtures are not yet established. Explicitly **do not
promote** an attention, precision, memory or VAE optimization based solely
on this first run.

A follow-up branch-only exploratory **P1B MMGP 0/1/3** experiment uses
additional read-only MMGP state telemetry. Its three-arm sequence is
hypothesis screening, not the balanced A/B/B/A adoption trial required by
the implementation specification.

## Artifacts

GitHub Actions run `44013580271` has `hunyuan15-p0-hot-results`
(JSONL traces, logs and reports) and `hunyuan15-p0-hot-clips` (MP4s).
