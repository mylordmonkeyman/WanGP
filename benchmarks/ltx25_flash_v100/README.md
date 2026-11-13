# LTX-2.5 Tesla V100 FlashAttention pilot (draft; no GPU tests run)

Branch: perf/ltx25-v100-flash-attn-dual  
Snapshot parent: 4adc57c0d17f960d91415564db68fa6ab9360537

## Scope / safety

This is an **opt-in** Python adapter for two independently installed, external
CUDA extensions. No CUDA code is vendored. Neither dependency is downloaded,
built, installed, or imported during WanGP startup. The defaults and existing
non-LTX attention paths are unchanged.

Only LTX's normal **self-attention** path can select the experimental backend.
The initial gate is deliberately strict:

- NVIDIA sm_70, FP16 Q/K/V, inference (no autograd);
- batch-major Q/K/V of identical shape, head dimension 64 or 128;
- noncausal, unmasked, no dropout, no window, no variable-length padding;
- query sequence at least 1024 tokens by default (configurable);
- **not** inside CUDA graph capture.

Everything else uses WanGP's existing SDPA fallback, including masked video or
audio attention and all LTX text/audio/video cross-attention. For an eligible
operation, a missing selected package or a CUDA kernel exception is an **error**,
not a silently hidden fallback. The first eligible dispatch prints a backend
banner so benchmark logs distinguish it from SDPA.

The normal LTX attention_kit projection/head-group path is preserved, including
its current memory allocation behavior. Beware that the backends may require
new Q/K/V layout copies inside their public Python wrappers.

## Runtime selection

WanGP launch flags (not a hot-swappable UI option):

    python wgp.py --offline --fp16 --ltx25-v100-attention=off
    python wgp.py --offline --fp16 --ltx25-v100-attention=1cat
    python wgp.py --offline --fp16 --ltx25-v100-attention=zhinianqin

Optional threshold (for any selection):

    --ltx25-v100-min-tokens=1024

Environment-only equivalent, for wrappers:

    WANGP_LTX25_V100_ATTENTION=zhinianqin
    WANGP_LTX25_V100_MIN_TOKENS=1024

CLI overrides environment when supplied. Defaults: off and 1024.
These are **process-launch flags**; restart WanGP to switch backends.

The global existing --attention=... mode still controls the standard WanGP
paths. Explicit non-default LTX AttentionFunction selections take priority over
the pilot flag. When the pilot is off, original behavior is preserved.

## External implementations (not installed by this branch)

### 1cat

Source: https://github.com/1CatAI/1Cat-vLLM/tree/main/flash-attention-v100

Use the independent flash-attention-v100 CUDA package, **not** the vLLM server.
Python import: flash_attn_v100; API: flash_attn_func(q, k, v, ...), BMHD.
Its public wrapper internally transposes and makes contiguous copies.
The wrapper's copy/allocation costs must be included in measurements.
Verify compilation for sm_70 and your exact torch/CUDA ABI before using.

### zhinianqin

Source: https://github.com/zhinianqin/flash-attention-v100

Python import: vllm_flash_attn; API: flash_attn_varlen_func(...), THD.
The vLLM wrapper uses a **noncanonical argument order**; the adapter calls
it by keyword, reshaping BMHD to THD and passing cached GPU int32 cu_seqlens.
The repository currently targets PyTorch 2.10 / CUDA 12.8. Its build.sh
modifies vLLM site-packages and deletes a flash_attn directory; DO NOT
run build.sh unchanged in your existing WanGP environment. Build and validate
in an isolated environment before deciding on any installation.

Both extensions are external and their version/ABI compatibility remain
UNVERIFIED against the user's installed WanGP container.

## Files changed

- shared/ltx25_v100_attention.py — lazy load, shape/dtype gate, both adapters.
- shared/attention.py — early per-call experimental route, fallback to SDPA.
- models/ltx2/ltx_core/model/transformer/attention.py — LTX self-attention opts in;
  preserve existing attention_kit fast path and explicit attention overrides.
- shared/cli_args.py — launch flags.
- wgp.py — propagate launch flags.

No GitHub Actions workflow, dispatch signal, Docker environment, or quantization
kernel was changed. **Do not trigger any GPU workflow without authorization.**

## Proposed validation, only after explicit authorization

1. Install each CUDA extension in a throwaway environment with the **same**
   torch and CUDA ABI as the target WanGP environment. Validate imports and
   a standalone unmasked FP16 D64/D128 kernel parity test.
2. Capture **real** per-layer LTX Q/K/V shapes and check they meet the gate.
   Compare full adapter runtime (including copies, allocations and cu_seqlens)
   against SDPA. Use CUDA events, warmup, matched shapes and peak VRAM.
3. Test one reproducible WanGP I2V generation with identical model checkpoint,
   settings, prompt, seed and resolution for off, 1cat and zhinianqin.
4. Reject any candidate that regresses full generation time, numerical
   stability, output quality, or peak VRAM, regardless of kernel-only TFLOPS.
5. Only after matching results, consider expanding eligibility to unmasked
   cross-attention, GQA, masking or CUDA graph capture separately.

This branch is a draft integration, **not** evidence that either external
extension works or improves LTX-2.5 speed on the target V100.
