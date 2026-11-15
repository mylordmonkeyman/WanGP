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
# Both stages ON (default): unset old research flags and new opt-outs.
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

## Backward compatibility

Old benchmark controls retain their prior explicit behavior:

- `WAN2GP_LTX25_E_TEST_4096X2048=0` disables the family in both stages.
- `WAN2GP_LTX25_E_TEST_4096X2048=1` enables both stages.
- `WAN2GP_LTX25_E_TEST_4096X2048_PHASES=none` disables both; `stage1`, `stage2`, or `stage1,stage2` selectively enables those stages.
- **The new opt-out flags always win** over both old overrides.

**Important:** Old scripts exporting the old experimental flag with `=0` will NOT get the new default until that flag is removed.

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
