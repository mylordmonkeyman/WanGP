# Unattended V100 FlashAttention follow-through

This is the state handoff for automated follow-up. The user has explicitly
authorized commits, workflow changes, and V100 runner tests as needed. Keep
work in perf/ltx25-v100-flash-attn-dual; preserve the original ConvRot and
production installations. Do not make unverified speedup claims.

## Work already verified

- Original two-extension build and small D64/D128 parity PASS:
  https://github.com/mylordmonkeyman/WanGP/actions/runs/43371330815
- Three-way full WanGP pay_attention benchmark run dispatched:
  https://github.com/mylordmonkeyman/WanGP/actions/runs/43384835108
- Hosted CPU static preflight PASS:
  https://github.com/mylordmonkeyman/WanGP/actions/runs/43386188825
- Expanded hosted CPU preflight PASS:
  https://github.com/mylordmonkeyman/WanGP/actions/runs/43386605842

## Follow-through gates

1. Wait for **43384835108** completion and inspect GitHub job steps, detailed
   build and test logs, the *three_way_report.json* file, and any errors.
   Time/VRAM comparisons are synthetic LTX-like shapes, not an actual model trace.
2. On failure, implement a minimal fix, commit it in the experimental branch,
   and re-authorize ONLY the three-way workflow (alter its existing signal
   file or use workflow_dispatch). Track the new run ID. Never silently
   reclassify an error as a successful SDPA result.
3. If the microbenchmark succeeded and both compiled packages passed, verify
   it uploaded a named artifact:
   ltx25-v100-sm70-cu128-extensions-43384835108
   (or change the pinned artifact run-ID in the integration workflow to
   the later successful run).
4. Run the real model integration workflow using a new user-approved signal
   in benchmarks/ltx25_flash_v100/video_dispatch.signal (not committed yet):
   https://github.com/mylordmonkeyman/WanGP/actions/workflows/ltx25-v100-attention-video.yml
   This must use the artifact from the successful build rather than compile
   again. It creates the same 49-frame 576x320 seed20261112 T2V clip with
   SDPA, 1Cat, and zhinianqin, uses isolated configs, avoids ConvRot Candidate
   E, and records actual candidate calls and shapes.
5. The real-video quality/performance gate is:
   - 3 newly generated MP4 outputs, valid ffprobe metadata;
   - all required model assets from the baked runner image;
   - selected CUDA attention calls >0 for both candidate modes;
   - no new exceptions/nonfinite outputs/obvious fidelity regressions;
   - WanGP queue duration compared separately from CLI/model load time;
   - generated clips require visual review for subjective quality.

The image-baked ConvRot prior R0 test used some BF16 internal activations even
on SM70. This pilot initially accepts only FP16 QKV for both new adapters.
The video commands explicitly use --fp16. If no candidate calls appear, first
inspect the real QKV dtypes, head-splitting sequence sizes, model attention
override, or masks. Do NOT claim acceleration if silent fallbacks remain.
If BF16-to-FP16 conversion is needed, benchmark conversion/copy costs and
SDPA parity before enabling it, and cast output back to original dtype.

## Source files

- shared/ltx25_v100_attention.py: candidate adapters and optional
  WANGP_LTX25_V100_TRACE=1 (CPU-only counter/shape metadata, no GPU sync).
- benchmarks/ltx25_flash_v100/three_way_benchmark.py and
  three_way_shapes.json: microbenchmark.
- benchmarks/ltx25_flash_v100/wgp_three_way_video.py and summarize_video.py:
  actual video comparison/report.
- .github/workflows/ltx25-v100-attention-threeway.yml:
  opt-in performance run.
- .github/workflows/ltx25-v100-attention-video.yml:
  opt-in actual video run; downloads compiled extensions from run 43384835108.
- .github/workflows/ltx25-v100-cpu-preflight.yml:
  normal hosted CPU-only preflight.

No more GPU tests are pending via workflow automation other than the
already-running three-way benchmark, until its completion is inspected.
