# P2B native-FP16 image-MLP chunk-size exploratory sweep

Run: https://github.com/mylordmonkeyman/WanGP/actions/runs/44030011405 (17 November 2026; source c0c5df98). All arms used the selected fixed native-only MMGP profile 1, unchanged FP16 checkpoint and converted LoRA, SDPA and VAE preset 0. One warmup and five hot requests per arm; sequential 4, 2, 1 ordering. GPU numeric microgate passed; all three video-generation arms exited 0 and produced 18 MP4 clips. The CPU summarizer alone failed with `KeyError: 'settings'` because it deleted baseline settings before comparing later arms; fixed in commit b9c695e.

| Image-MLP divide | Five hot E2E seconds | Median E2E (s) | Peak Torch allocated (GiB) |
|---|---|---:|---:|
| 4 | 34.189200, 33.913743, 34.087887, 34.116222, 34.165040 | 34.116222 | 5.640 |
| 2 | 34.471880, 34.321732, 33.999812, 34.044919, 34.107109 | 34.107109 | 5.640 |
| 1 | 33.666534, 33.713227, 33.920457, 33.892602, 33.846786 | 33.846786 | 5.640 |

Relative to divide 4, divide 2 improves the median by 0.009113 seconds (0.027%); divide 1 improves by 0.269436 seconds (0.790%). These small differences are **not** sufficient for promotion: sequential-arm drift is uncontrolled and the saved video quality report did not finish. The numeric microgate pass is not a substitute for decoded H264 parity, latent comparisons or framewise PSNR/SSIM. Torch allocated memory is not device-wide peak memory. The job artifacts retain all 18 MP4s and raw reports: [reports](https://github.com/mylordmonkeyman/WanGP/actions/runs/44030011405/artifacts/14304459722), [clips](https://github.com/mylordmonkeyman/WanGP/actions/runs/44030011405/artifacts/14304464652).

**Decision:** no default change; preserve original assets. Next, recover the CPU-only summarizer from the retained baseline JSON and MP4s, then assess whether balanced repeat and lossless parity are justified. Do not rerun the GPU sweep solely to recover the report.

## CPU-only recovery of retained MP4 artifacts (17 November 2026)

Downloaded original run 44030011405 report and clip ZIPs; decoded all 18 MP4s locally with OpenCV/FFmpeg-backed H264 decoding (33 frames, 832×480 each). Compared every hot clip to the first hot divide-4 baseline using exact decoded BGR pixel equality, full-video 8-bit RGB-channel MSE PSNR, and mean per-frame SSIM (scikit-image). This is a post-hoc decoded-video check, not a lossless latent comparison.

| Divide | Hot clips | Decoded frames differing per clip | PSNR vs divide 4 | Mean frame SSIM vs divide 4 |
|---|---:|---:|---:|---:|
| 4 | 5 | 0/33 | identical | 1.000000 |
| 2 | 5 | 33/33 | 27.94556 dB | 0.912163 |
| 1 | 5 | 33/33 | 27.94556 dB | 0.912163 |

All five hot clips within each arm are pixel-identical. Divide 2 and divide 1 are also exactly pixel-identical to each other across all 33 frames. All arms have matching checkpoint and converted-LoRA SHA256 values and identical recorded effective generation settings; the chunk-size setting is the intended arm difference. The 27.95 dB / 0.912 SSIM change from divide 4 is materially larger than the 0.79% timing advantage; **do not promote divide 1 or divide 2**. Root cause of the divide-4 versus divide-2/1 image difference remains unverified (possible changed numeric execution path); a numeric microgate pass did not establish full-generation parity. No V100 job was rerun for recovery. This recovery uses decoded H264 and does not establish latent or pre-encode equality, and CPU OpenCV's decoder differs from the workflow's FFmpeg framehash path.
