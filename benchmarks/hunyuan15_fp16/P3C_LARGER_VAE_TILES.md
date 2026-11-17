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

**Promotion state:** not promoted; await measured results from the linked run. A mere reduction in source-derived decoder positions is not a speedup claim.
