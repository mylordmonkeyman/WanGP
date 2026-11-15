# Candidate E rejected weight family at fixed 64-row minimum — Distilled T2V49

GitHub Actions: [successful four-generation ABBA run](https://github.com/mylordmonkeyman/WanGP/actions/runs/43682531028), benchmark script:
`benchmarks/ltx25_convrot/distilled_fixed64_excluded_shape_abba.py`.

## Question

Does allowing Candidate E for the previously excluded ConvRot weight
family `(N=4096,K=2048)` improve speed and similarity to **actual stock WanGP**
when **MIN_M stays at 64**?

## Experimental design

- LTX-2.5 original **Distilled INT8 ConvRot** transformer; **no LoRA**
- V100 (SM70); 49 T2V frames, 1024x576, eight steps, fixed seed 20261112
- Identical cocktail-bar prompt, CFG/NAG/audio guidance, conventional Conv VAE
- All arms use Candidate E + Radix-4, Stage2 transient FP16 direct Quanto,
  Stage1 selective FP16 off, `WAN2GP_LTX25_E_MIN_M=64`,
  `WAN2GP_LTX25_E_MAX_M=8192`
- **Only toggled:** `WAN2GP_LTX25_E_TEST_4096X2048` (research-only)
- Matched actual stock reference: [unmodified upstream WanGP
  c8d47da](https://github.com/deepbeepmeep/WanGP/commit/c8d47da27023476777a0874300bbc9a4126acca6),
  run 43681488566, decoded 49-frame stock MP4
- The harness checks entire input JSON equality to the stock run,
  validates model and VAE provenance, routes the exact target weight family,
  rejects FP16 guard fallbacks and persistent weight duplicates, and
  checks repeated-video and audio-bit reproducibility
- All four whole-process timings include loading, generation and file save

## Result (A–B–B–A)

| Arm | Experimental family | Wall seconds | Target family Candidate E calls | Candidate E total | Stage2 selective FP16 |
|---|---|---:|---:|---:|---:|
| A1 | Off | 101.479 | 0 | 16464 | 1872 |
| B1 | On | 101.507 | 528 | 16992 | 1728 |
| B2 | On | 101.251 | 528 | 16992 | 1728 |
| A2 | Off | 102.153 | 0 | 16464 | 1872 |
| **Median off/on** | | **101.816 / 101.379** | | | |

**Observed on-arm advantage:** 0.437 seconds, or **0.429%**.
The run-pair variability is of comparable scale. This is not an
established practically significant speed improvement.

Exact target-family census: `M=1008,N=4096,K=2048`: 384 calls;
`M=4032,N=4096,K=2048`: 144 calls. Both are above minimum 64;
there are no sub-64 target-family calls in this 49-frame workload.

## Stock-reference fidelity

All numbers compare each generated output against the *same* actual
unmodified stock MP4, not against the other optimized arm.

| Metric | Family off | Family on | On minus off |
|---|---:|---:|---:|
| Video PSNR | 23.396925 dB | 23.107871 dB | **−0.289054 dB** |
| Video SSIM | 0.712951 | 0.707326 | **−0.005625** |
| Decoded audio PCM SNR | 3.809711 dB | 3.625239 dB | **−0.184472 dB** |

Both off runs decoded identically to one another; likewise both on runs.
A separate RGB per-frame analysis of the first off/on pair found that
**43/49 frames** had lower PSNR relative to stock with the family enabled,
and six had higher PSNR.

**Caveats:**
- Stock similarity is not perceptual or semantic video quality. Neither
  these metrics nor sample precision establish which video looks better.
- The excluded shape **failed the strict per-GEMM p99/max error gate in
  15/15 checkpoint-backed cases** in earlier run 43678722472; that
  remains a safety reason not to promote.
- The previous *minimum 16* test improved stock similarity when enabling
  this family, but it changes a different selection of GEMMs: outcomes
  cannot be extrapolated to a minimum of 64.
- Global family enable affects Stage1 (384 calls) and Stage2 (144 calls);
  stage-specific experiments would be required to attribute the output
  change to a particular stage.

## Decision

**Keep original minimum 64, upper maximum 8192 and the seven-shape
Candidate E allowlist as defaults.** Do not enable `(4096,2048)`
by default on Distilled or Dev. The research flag remains opt-in only.
No changes have been deployed to the installed `/opt/ai/wan2gp` service.

Artifacts: [videos](https://github.com/mylordmonkeyman/WanGP/actions/runs/43682531028/artifacts/14149381721);
[complete statistics, settings and logs](https://github.com/mylordmonkeyman/WanGP/actions/runs/43682531028/artifacts/14149386744).
