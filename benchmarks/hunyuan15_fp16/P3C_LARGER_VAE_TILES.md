# P3C — larger Hunyuan 1.5 native-FP16 VAE spatial tile comparison

**Branch:** `optimization/hunyuan15-v100-fp16` (no changes to `main`).
**Experiment run:** https://github.com/mylordmonkeyman/WanGP/actions/runs/44027986619

## Prior measured P3B and P2C evidence

P3B V100 run [44022640415](https://github.com/mylordmonkeyman/WanGP/actions/runs/44022640415) completed with 5 hot requests after warmup per VAE preset:

| Existing `vae_config` | Pixel tile | Median hot E2E (s) | Median decode and copy (s) | Decoded H264 frame mismatches vs auto |
|---|---:|---:|---:|---:|
| 0 (auto, resolved to preset1) | 256 | 34.456433 | 11.624608 | 0 of 33 × 5 |
| 1 | 256 | 34.586495 | 11.632271 | 0 of 33 × 5 |
| 2 | 192 | 34.874509 | 11.909888 | 33 of 33 × 5 |
| 3 | 128 | 38.474094 | 15.539814 | 33 of 33 × 5 |

The 256px setting is an unchanged baseline, **not** an optimization. Smaller tiles are not attractive speed candidates on the tested reference; changed H264 frame hashes do not alone establish a quality regression.

P2C run [44024732809](https://github.com/mylordmonkeyman/WanGP/actions/runs/44024732809) passed explicit debug-latent skip/save and 33-frame decoded parity. This is I/O hygiene, and hot speedup is not established; its opt-in remains independent of P3C.

## User-imposed fixed MMGP setting

**MMGP profile 1 is the selected native-FP16 default, not a variable.**
The user explicitly forbids further MMGP tests, validation, or performance comparisons. All P3C arms keep the same native-FP16 finetune and profile-1 default; no `--profile` argument is passed by the new harness. Do not compare profiles 0/1/3 or re-run the cancelled profile verification.

## Candidate and code path

The existing `wgp.py` `VAE_tile_size` dictionary is altered only for explicitly tagged native-FP16 Hunyuan 1.5 **when** `HUNYUAN15_P3C_TRACE=1` and an explicit `HUNYUAN15_P3C_SPATIAL_TILE=384` or `512` is supplied. The 256px arm does not set the override and retains the original object. Normal production behavior, BF16, and other models never enter this policy.

`shared/hunyuan15_p3c_tile_policy.py` validates the original 256/64/0.25 auto geometry, 16× spatial and 4× temporal factors, 832×480×33 dimensions, and the required integer-overlap assertions. Invalid requests fail closed. The generated spatial tile/latent geometry is:

| Tile | Latent tile | Spatial latent stride | Expected row × column positions |
|---:|---:|---:|---:|
| 256 | 16 | 12 | 3 × 5 = 15 |
| 384 | 24 | 18 | 2 × 3 = 6 |
| 512 | 32 | 24 | 2 × 3 = 6 |

These are **source-derived counts**, not results. Behind the same experimental flag, `AutoencoderKLConv3D.decode` counts actual calls to the spatial decoder, and the comparison script requires actual 15/6/6 calls with all 33 output frames. It also checks that spatial and temporal tiling remain selected and that the intended geometry took effect.

## Measurement and acceptance boundaries

Use original Comfy-Org FP16 transformer, the same converted rank-32 LightX2V LoRA, 4 steps, 832×480, 33 frames, original bar prompt, seed 20261112, SDPA and disabled compilation. One warmup + five hot generations per arm, fresh persistent WanGP process for each arm and no downloads. Sequence is 256 → 384 → 512, serialized under `v100-serialized-optimization-experiments`; 512 is blocked if 384 PyTorch peak allocated exceeds 16 GiB. No second GPU job runs in parallel.

Record median and MAD hot request time, preliminary decode/copy duration, PyTorch allocated peak, 33-frame count, executed spatial decoder calls, image/frame hashes, FFmpeg PSNR and SSIM on H264-decoded corresponding clips. These are **not lossless pre-encode parity metrics**, so visual seam inspection, lossless frame/latent comparisons, peak reserve/overall VRAM headroom, and a balanced follow-up are required before adopting any custom tile as default.

**Promotion state:** not promoted; measured results below do not satisfy all quality and repeatability gates.

## Completed V100 trial — 2026-11-17

[Run 44027986619](https://github.com/mylordmonkeyman/WanGP/actions/runs/44027986619) completed all three video-generation arms successfully, but the post-generation report step failed with `KeyError: 'clips'` (report mutates/removes the reference clips before reusing them). The report-script bug was corrected in commit `202fada0`; **no GPU rerun is needed for these measurements**. The uploaded reports and 18 clips were independently analyzed offline; source: `p3c_local_analysis.json` from archived artifacts, with FFmpeg decoded H264 comparisons. All arms: one warmup plus five hot runs, native-only selected MMGP1 fixed, same model/LoRA/seed, 832×480×33, 4 steps.

| Spatial tile | Actual decoder calls per generation | Median hot E2E (s) | Median decode/copy (s) | Median denoise (s) | Peak PyTorch allocated (GiB) | PSNR vs 256 (dB) | SSIM vs 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 256 | 15 | 34.080267 | 11.568600 | 19.111365 | 5.640 | reference | reference |
| 384 | 6 | 33.086775 | 10.624149 | 19.128679 | 9.534 | 48.01802 | 0.992565 |
| 512 | 6 | 33.538176 | 10.981524 | 19.127194 | 14.249 | 48.04734 | 0.992586 |

The 384 tile improves hot E2E by 0.993492 s (2.91%) and decode/copy by 0.944451 s (8.16%). The 512 tile improves hot E2E by 0.542091 s (1.59%) and decode/copy by 0.587076 s (5.08%). The actual spatial decoder-call reductions are 15→6, confirmed in all six generations per arm. All 33 decoded H264 frames differ by hash for both larger tiles, but PSNR/SSIM are high and consistent across all five hot clips per arm. The unchanged 256 baseline is bit-identical to itself across hot clips.

The 32 GiB V100 has *nominal* 32-minus-PyTorch-allocated differences of approximately 26.36 / 22.47 / 17.75 GiB respectively, **not actual GPU headroom**: PyTorch allocated peaks omit cached/reserved, CUDA context, model offload and other allocations. No claim about measured free device memory is supported by these figures. Tile-seam inspection and lossless pre-encode comparison remain outstanding; H264 PSNR/SSIM alone cannot establish seam absence. The single sequential 256→384→512 sweep is not balanced or replicated. **Do not promote either tile** before visual seam review, memory-reserve/device-headroom measurement and a balanced follow-up with explicit quality thresholds. 384 is the provisional speed candidate, not a production default.

[Report artifact](https://github.com/mylordmonkeyman/WanGP/actions/runs/44027986619/artifacts/14304716305) · [Video artifact](https://github.com/mylordmonkeyman/WanGP/actions/runs/44027986619/artifacts/14305040483)
