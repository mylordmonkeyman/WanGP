# HunyuanVideo 1.5 FP16 / V100 — P1B initial MMGP profile screen

**Source workflow:** https://github.com/mylordmonkeyman/WanGP/actions/runs/44016638220

This is the original fixed 832×480 / 33-frame / 4-step T2V + converted
rank-32 LightX2V LoRA and explicit SDPA. The unmodified Comfy-Org transformer
SHA256 begins `6889f79cf0`; the adapter SHA256 begins `9257f364a8`.
Each profile ran in its own process with one entire warm-up generation
discarded, followed by five real hot generations.

| Request metric | Profile 0 AutoRAM/AutoVRAM | Profile 1 HighRAM/HighVRAM | Profile 3 LowRAM/HighVRAM |
|---|---:|---:|---:|
| Hot median (s) | **34.629** | **33.915** | **34.213** |
| MAD (s) | 0.0329 | 0.1135 | 0.0770 |
| Improvement vs profile 0 | — | **0.714 s / 2.06%** | **0.416 s / 1.20%** |
| Provisional denoise median (s) | 19.661 | 19.117 | 19.187 |
| VAE decode+copy median (s) | 11.677 | 11.493 | 11.751 |
| End-of-request MMGP bulk block moves, each hot request | 272 | 272 | 271 |
| End-of-request transformer preloaded blocks | 0 | 0 | 0 |
| MMGP pinned host bytes | 32 MiB | ~25.02 GiB | ~16.19 GiB |
| Approx. process RSS after hot requests | 27.6 GiB | 51.5 GiB | 43.7 GiB |
| PyTorch allocated peak VRAM | 5.64 GiB | 5.64 GiB | 5.64 GiB |

**The retained MMGP pinning is not GPU residency:** profile 1 allocates
materially more locked host RAM. Its unchanged ~272 bulk block moves per
request suggest reduced transfer/staging overhead rather than eliminating
streaming. A bulk-move count is not an H2D-byte count. The end-of-request
offload snapshots cannot establish the live residency/transfer waits DURING
denoise.

The first-screen robust noise gate (MAD-based + bootstrap CI) classified both
profile 1 and profile 3 as improvements. However, the sequential order 0,1,3
could confound temporal run drift; it does **not** meet the balanced A/B/B/A
adoption criterion.

## Output verification

- All 18 MP4s passed the original per-clip 33-frame decode test.
- Independently decoded **all 18 MP4s using ffmpeg framemd5** from GitHub
  Actions artifact `hunyuan15-p1b-profile-clips`: **all 18 had identical
  decoded frame checksum sequences**, across every profile and repeat.
- MP4 **container byte hashes differed** (expected with timestamp/metadata);
  do not confuse this with frame differences.
- H.264 decoded frame equality supports this fixed case, but is not lossless
  latent/parity testing or a proof of longer-sequence stability.

## User-directed default decision — engineering exception

On November 16, 2026, the user explicitly directed that **MMGP profile 1
become the default for native-FP16 Hunyuan 1.5** despite the initially
inconclusive full A/B/B/A adoption noise gate. This is a user-authorized
engineering exception, **not a claim that the performance/statistical
qualification criteria were satisfied**.

Only the three finetunes with `hunyuan15_native_fp16=true` on
`optimization/hunyuan15-v100-fp16` carry `default_mmgp_profile: 1`.
`wgp.py::compute_profile` applies this model-scoped default to both the
**load** and **loaded-profile reuse/cache** decision, avoiding inadvertent
reconstruction between successive generations. Explicit per-request memory
profile overrides and `--profile 0` (including the historical P0/P1
benchmark configurations) retain priority. Original BF16 Hunyuan,
unrelated WanGP models, `main` and research branches are unchanged.

**Cost:** profile 1 uses about **25 GiB pinned host RAM**, compared with
about **32 MiB** for profile 0. Benchmark gains were **2.06% in the initial
0/1/3 screen** and **1.53% in the balanced 0/1/1/0 repeat**; the latter
did *not* pass the prescribed statistical noise gate. That caveat
persists even though the default is now selected.

**Verification:** Model-scoped CPU policy tests are passing. Runtime
two-video GPU verification without any explicit CLI profile selection is
queued at https://github.com/mylordmonkeyman/WanGP/actions/runs/44023325587
behind the serialized P3B VAE preset experiment. Until that run succeeds,
classify production-path default activation as *configured, runtime
confirmation pending*.

## Next gates

Balanced independent sessions 0,1,1,0 are being tested via
https://github.com/mylordmonkeyman/WanGP/actions/runs/44019180939
(five hot generations per session, first full warmup discarded).
Further gating: verify memory under 97 frames, confirm H2D bytes/waits,
monitor system RAM pressure with other models present, and save lossless
frame/latent probes before considering a default profile change.

**No MMGP profile has been promoted. No changes to `main` or research branch.**
