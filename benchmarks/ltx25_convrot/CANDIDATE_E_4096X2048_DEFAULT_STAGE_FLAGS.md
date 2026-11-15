# Candidate E (N=4096, K=2048): default ON, independent stage opt-outs

**Status:** User-requested default routing in the `perf/ltx25-distilled-shape-stage-i2v-t2v73-20261114` branch, not deployed to the host or merged into `main`.

## Active SM70/Volta behavior

On the V100, when Candidate E is active (`WAN2GP_LTX25_CUBLAS_E_MODE=fp32rot`), the `(N=4096,K=2048)` family **defaults ON in both Stage 1 and Stage 2**. Candidate E's existing seven permitted shape families are not changed. Normal row bounds remain `64 <= M <= 8192`. Failed/unsupported individual calls still take the existing WanGP fallback path.

Set **either** opt-out flag independently; they take effect at Python module import and therefore require a fresh WanGP inference process after changing the environment:

| Flags | Stage 1: N4096,K2048 | Stage 2: N4096,K2048 |
| --- | --- | --- |
| Neither set (default) | Candidate E | Candidate E |
| `WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE1=1` | Fallback | Candidate E |
| `WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE2=1` | Candidate E | Fallback |
| Both flags `=1` | Fallback | Fallback |

When Stage 2 falls back, it may pass through the existing, separately configured transient Stage 2 selective-FP16 GEMM path; a disabled Candidate E family does **not** globally disable Stage 2 FP16. When Stage 1 falls back, native WanGP execution is unchanged. The original INT8 checkpoint remains untouched; no persistent duplicate FP16 weights are created.

Examples:

```bash
# Both stages ON (default): remove obsolete experiment flags and opt-outs.
unset WAN2GP_LTX25_E_TEST_4096X2048
unset WAN2GP_LTX25_E_TEST_4096X2048_PHASES
unset WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE1
unset WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE2

# Disable only Stage 1 for one command:
WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE1=1 python wgp.py ...

# Disable only Stage 2:
WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE2=1 python wgp.py ...

# Disable both:
WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE1=1 \
WAN2GP_LTX25_E_DISABLE_4096X2048_STAGE2=1 python wgp.py ...

# Disable all Candidate E shape families, independently from the flags above:
WAN2GP_LTX25_CUBLAS_E_MODE=off python wgp.py ...
```

Both new opt-out flags only accept the values `0` and `1`. Invalid values raise `ValueError` immediately rather than silently ignoring the request.

## Retired experiment flags

The duplicate research controls `WAN2GP_LTX25_E_TEST_4096X2048` and
`WAN2GP_LTX25_E_TEST_4096X2048_PHASES` are **no longer supported**.
If an older launch script exports either one, the kernel raises a
`ValueError` explaining the two canonical per-stage opt-outs, instead of
silently selecting the wrong path. The `WAN2GP_LTX25_DEFAULT_ON` environment
flag is also unnecessary: `WAN2GP_LTX25_CUBLAS_E_MODE` alone controls
Candidate E. Use the settings in the table above.

Retired per-GEMM stage profiling and the Distilled pipeline event profiler
have been detached from the normal inference path. Historical benchmark
scripts/workflows are retained for provenance; the currently maintained
regression scripts have been migrated to the canonical controls.

## Existing evidence and caution

The user explicitly selected default enablement after 73-frame bar-prompt Distilled T2V/I2V tests, including a repeated Stage2-only vs both-stage A–B–B–A experiment:

- In T2V, both-stage expansion saved **0.965%** median end-to-end time over Stage2-only and increased stock-reference video PSNR by **0.073715 dB** and SSIM by **0.008679**.
- In I2V, both-stage expansion saved **0.468%** median time, but **reduced** stock-reference PSNR by **0.258766 dB** and SSIM by **0.006931**. I2V audio SNR relative to stock improved by **2.161072 dB**.
- Measurements: [successful repeated Stage2 vs both T2V/I2V run](https://github.com/mylordmonkeyman/WanGP/actions/runs/43686816635).
- **Safety caveat:** The `(4096,2048)` family **failed 15/15 strict checkpoint-backed per-GEMM p99/max numerical gates** in [audit run 43678722472](https://github.com/mylordmonkeyman/WanGP/actions/runs/43678722472). The default change is an explicit experimental tradeoff choice, not retroactive precision certification.
- Stock-relative PSNR and SSIM indicate similarity to the original model output, not perceptual correctness or visual quality on all inputs. Monitor videos for artifacts before broad rollout.

The change is limited to this GitHub development branch. No installed `/opt/ai/wan2gp` code, container or service has been changed by the GitHub edit.

## Verification

- Import-time env contract: `benchmarks/ltx25_convrot/test_candidate_e_4096x2048_stage_flags.py`.
- End-to-end 49-frame Distilled T2V: `benchmarks/ltx25_convrot/verify_candidate_e_4096x2048_default_flags.py`.
- Workflow: `.github/workflows/ltx25-candidate-e-default-4096x2048-flags.yml`.

## Completed final V100 validation (2026-11-15)

[GitHub Actions successful run 43739803642](https://github.com/mylordmonkeyman/WanGP/actions/runs/43739803642) passed 12 import-time valid flag combinations, 5 invalid-flag rejection cases, and all four matched actual **49-frame** 1024×576 Distilled T2V generations.

| 4096x2048 selection | Used in Stage1 | Used in Stage2 | Stage2 transient FP16 GEMMs | PSNR to original stock |
|---|---:|---:|---:|---:|
| Default both ON | 384 | 144 | 1728 | 23.107871 dB |
| Disable Stage1 | 0 | 144 | 1728 | 23.394218 dB |
| Disable Stage2 | 384 | 0 | 1872 | 23.106580 dB |
| Disable both | 0 | 0 | 1872 | 23.396925 dB |

All dispatch counts match the expected flags. Each was a single-seed run, so small timing differences should not be interpreted as statistically established acceleration. The strict per-GEMM gate caution above remains in force.

[Results and full logs](https://github.com/mylordmonkeyman/WanGP/actions/runs/43739803642/artifacts/14172312004) |
[Four comparison MP4s](https://github.com/mylordmonkeyman/WanGP/actions/runs/43739803642/artifacts/14171794228)

## Post-cleanup full-video numerical regression

After retiring the Candidate E and Distilled whole-pipeline profiling hooks
and removing the duplicate shape-enablement environment flags:

- [49-frame four-way stage opt-out regression](https://github.com/mylordmonkeyman/WanGP/actions/runs/43741570613):
  **PASS**, all four routes exercised correctly; decoded video hashes and
  stock-reference PSNR/SSIM unchanged from the pre-cleanup outputs.
  Import-time flag contract: 5 valid configurations and 9 invalid/rejected
  configurations, including the two retired legacy research switches.
- [73-frame T2V/I2V bit-exact regression](https://github.com/mylordmonkeyman/WanGP/actions/runs/43742195972):
  **PASS** for both modes using the original bar prompt and image.
  Each cleaned output's decoded frame SHA-256 and PCM-audio SHA-256 matched
  the corresponding pre-cleanup video from run 43686816635 exactly.
  Both modes retained 27,360 Candidate E calls, including precisely 384
  Stage 1 and 144 Stage 2 calls for the (4096,2048) family.
  T2V took 112.736s and I2V took 113.777s (individual whole-process runs).

These checks establish absence of numerical output regressions on the tested
seed and settings; they do not override the previously failed strict per-GEMM
accuracy gate or establish universal perceptual equivalence.
