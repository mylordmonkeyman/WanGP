# LTX-2.5 Tesla V100 FlashAttention pilot and three-way benchmark

Branch: perf/ltx25-v100-flash-attn-dual  
Snapshot parent: 4adc57c0d17f960d91415564db68fa6ab9360537

## Scope / safety

This is an **opt-in** Python adapter for two independently installed, external
CUDA extensions. No CUDA code is vendored. Neither dependency is downloaded,
built, or installed by this branch. The experimental adapter lazily imports the
selected package only on its first eligible call. The defaults and existing
non-LTX attention paths are unchanged.

Only genuine LTX-2.5 transformer instances, tagged at model initialization,
can select an experimental backend through LTX's normal **self-attention** path.
Other LTX versions remain on their existing attention dispatch.
The initial gate is deliberately strict:

- NVIDIA sm_70, FP16 Q/K/V, inference (no autograd);
- batch-major Q/K/V of identical shape, head dimension 64 or 128;
- noncausal, unmasked, no dropout, no window, no variable-length padding;
- query sequence at least 1024 tokens by default (configurable);
- **not** inside CUDA graph capture.

Ineligible attention is rejected as early as possible, before LTX's projection
and attention_kit staging, so normal global attention behavior is preserved.
Remaining ineligible calls fall back to SDPA. This includes masked video/audio
attention and all LTX text/audio/video cross-attention. For an eligible
operation, a missing selected package or a CUDA kernel exception is an **error**,
not a silently hidden fallback. The first eligible dispatch prints a backend
banner so benchmark logs distinguish it from SDPA.

The normal LTX attention_kit projection/head-group path is preserved. Its
original staged Sage/SDPA behavior is retained for calls screened out before
Q/K/V projection; eligible V100 calls may have a different VRAM profile. Beware that the backends may require
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

Both extensions **compiled and passed** 256-token FP16 D64/D128 inference
smokes on the pre-baked V100 Actions runner (run #43371330815, PyTorch
2.10.0+cu128). Compatibility against the user's separate WanGP container and
longer LTX-2.5 shapes is still unverified.

## Files changed

- shared/ltx25_v100_attention.py — lazy load, shape/dtype gate, both adapters.
- shared/attention.py — early per-call experimental route, fallback to SDPA.
- models/ltx2/ltx2.py — mark only actual LTX-2.5 transformer attention modules.
- models/ltx2/ltx_core/model/transformer/attention.py — marked LTX self-attention opts in;
  preserve existing attention_kit fast path and explicit attention overrides.
- shared/cli_args.py — launch flags.
- wgp.py — propagate launch flags.

The previously authorized compile probe and its signal live on this branch.
A **separate** benchmark workflow is now prepared with no benchmark dispatch signal.
Neither quantization kernels, Docker environment, nor existing production
WanGP installation are modified by this pilot.

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

Initial SM70 compilation and 256-token correctness are established. No
long-sequence benchmarks or LTX-2.5 video comparisons have been run;
**no speedup has yet been established.**


## Three-way benchmark workflow (prepared, NOT DISPATCHED)

Workflow: [.github/workflows/ltx25-v100-attention-threeway.yml](../../.github/workflows/ltx25-v100-attention-threeway.yml)

The workflow runs **only** after an explicit GitHub workflow_dispatch on
perf/ltx25-v100-flash-attn-dual, or after a new user-authorized commit
changing benchmarks/ltx25_flash_v100/benchmark_dispatch.signal.
**There is no benchmark dispatch-signal file committed.** Other commits do
not trigger this workflow, and no GPU-runner dispatch is permitted without
the user's explicit instruction.

One isolated self-hosted V100 job does the following:

1. Checks out the exact approved commit. Checks Python/Bash syntax.
2. Uses the **proven** compile_on_runner.sh to build pinned 1Cat and
   zhinianqin CUDA 12.8 extensions in RUNNER_TEMP against the baked
   /opt/venvs/ltx25-cu128 interpreter. No global pip installs,
   checkpoint downloads, model loads, or vLLM server installation.
3. Installs missing non-PyTorch Python tools (if any) with pip --target
   under the same temporary work directory, verifies key imports, and
   checks each extension was loaded from its correct source directory.
   Checks sm_70, PyTorch 2.10.0+cu128, and Python development headers.
4. Runs mocked adapter-contract tests and the pinned extension FP16 smoke.
5. Benchmarks **SDPA, 1Cat, zhinianqin** through WanGP's pay_attention
   dispatcher, with matching random FP16 Q/K/V, warmup, three rotated-order
   rounds, and eight iterations per backend/round by default.
6. Checks FP16 accuracy against SDPA for every shape; reports median
   **dispatch wall-time** and CUDA-event time, incremental and absolute peak
   allocated VRAM, and per-case SDPA-relative speedups. Failed selected
   kernels are never silently reported as a successful run.
7. Uploads compiler/benchmark logs, JSON, CSV, Markdown summary, and
   reusable .so/Python wrappers with an ABI manifest.

### Shape source and limits

three_way_shapes.json contains seven **synthetic LTX-like** self-attention
shapes, including D64/D128, H8/H32 and up to 8,192 tokens, plus one B2 case.
They are **not** actual per-layer traces from the user's checkpoint.
The benchmark enforces a maximum Q/K/V+output size before allocation, does
not call a text encoder, and does not generate video.

This is a **microbenchmark** for selecting the candidate before a later
matched-seed full-video test. It is not evidence of an end-to-end WanGP
speedup; memory head splitting and GPU-to-CPU synchronization elsewhere can
still dominate performance.

Workflow links:
- https://github.com/mylordmonkeyman/WanGP/actions/workflows/ltx25-v100-attention-threeway.yml
- https://github.com/mylordmonkeyman/WanGP/actions/runs/43371330815 (earlier successful compile-only test)
