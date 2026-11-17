# HunyuanVideo 1.5 native-FP16 V100 optimization — P0 status

Source: `research/hunyuan15-v100-native-fp16-20261116` at
`7b4c3f5ab3595eff14450eeb2549779931ede405`.

Development branch: `optimization/hunyuan15-v100-fp16`.
No work is authorized on `main` or the source research branch.

## Implementation status

| Requirement | State | Evidence / limitation |
|---|---|---|
| Branch ancestry | Complete | Branch created from pinned research SHA; no source changes |
| Fixed Comfy-Org T2V FP16 + converted rank-32 LoRA | Harness configured | Offline cached checkpoint and preconverted adapter required |
| Single persistent process | **Measured; validated** | One process over six tasks, stable PID/model/pipeline/offload IDs, single checkpoint/model load; run `44013580271` |
| One warm-up + five hot requests | **Measured** | Hot E2E 34.609 / 34.818 / 34.684 / 34.631 / 34.651 s; median 34.651 s, MAD 0.034 s |
| Model and MMGP identity | **Measured, MMGP residency incomplete** | 54 blocks on CPU after requests; stable offload identity; read-only pool/transfer snapshots added for P1B |
| Hot offload-inclusive denoise | **Provisional 19.657 s median** | Early transformer preloads and H2D still require attribution; do not promote as verified inclusive-denoise time |
| Q/K pre-RMSNorm safety and FP32 reference | **Measured one seed; more seeds pending** | 864 tensor observations, maximum abs(Q/K) 44.5, zero FP16-square threshold violations, max sampled FP32 reference difference 0.03578 |
| Attention dispatch shapes and lengths | **Measured; kernel profiling queued** | 216 SDPA calls, Q/K [1,14159,16,128], 14,040 video + 119 text tokens; kernel identity pending |
| VAE tile geometry and executed decoder calls | **Measured** | Runtime module FP16; 256px/64-frame preset 1, overlap .25, 15 decoder calls; arithmetic dtype finer probe pending |
| Separate cold one-video reference | **Measured** | Whole-process 68.9 s, request time 49.891 s |
| Changed prompt sequence | **Validated** | Five hot durations 34.770–36.156 s, last changed prompt incurred ~1.415 s conditioning |
| CPU-only fixture and noise gate | **Passed** | All eight tests, syntax and ancestor guards passed on Github-hosted runner |
| CPU-only branch push workflow | **Passing** | `.github/workflows/hunyuan15-p0-cpu.yml`, branch-only, no GPU jobs |
| V100 workflow | **Completed successfully** | P0 `44013580271`; P1B `44016638220` launched; SDPA kernel profiler `44016941390` queued |

**P0 has a credible first hot baseline, but its full gate has NOT passed.**
MMGP H2D and pin-pool lifecycle, actual SDPA CUDA kernel, LoRA/kernel
attribution, and startup subphases remain open. Run results and direct links:
[`P0_RESULTS.md`](P0_RESULTS.md). No performance default, model math,
scheduler, VAE precision, memory profile, or attention kernel has been changed.

## Reproduce on the existing runner when the GPU is idle

All commands operate in a checked-out copy of the optimization branch in the
existing WanGP runner environment. Reuse the checkpoint caches; do not download
fresh model copies. Invoke the three modes **serially** (never concurrently).

```bash
export HUNYUAN15_PYTHON=/opt/venvs/ltx25-cu128/bin/python
export HUNYUAN15_TEST_OUT=/cache/ltx25/results/hunyuan15-p0-manual
export PYTHONPATH=.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
python3 -m unittest discover -v -s benchmarks/hunyuan15_fp16 -p test_p0_hot.py
"$HUNYUAN15_PYTHON" -u benchmarks/hunyuan15_fp16/p0_hot_queue.py --cold-only
"$HUNYUAN15_PYTHON" -u benchmarks/hunyuan15_fp16/p0_hot_queue.py
"$HUNYUAN15_PYTHON" -u benchmarks/hunyuan15_fp16/p0_hot_queue.py --diagnostic
"$HUNYUAN15_PYTHON" -u benchmarks/hunyuan15_fp16/p0_hot_queue.py --changed-prompt
```

Outputs are under `$HUNYUAN15_TEST_OUT/p0/{cold,hot,diagnostic,changed_prompt}/`.
The harness must exit with an error on GPU occupancy, cache mismatch/missing
assets, model reconstruction, unexpected attention mode, missing outputs,
incorrect 33-frame decode, or other validation failure.

Workflow file:
`.github/workflows/hunyuan15-v100-fp16-p0-hot.yml`

### GitHub runner launch

The user explicitly authorized a branch-local push trigger for autonomous workflow
starts. The GitHub connection exposes file commits and workflow run monitoring but
not a direct `workflow_dispatch` action. Prior WanGP branches used push triggers.

GPU tests remain manual-in-the-sense-of-intent: the P0 V100 workflow is triggered
by **only** a push to `optimization/hunyuan15-v100-fp16` changing
`.github/workflow-triggers/hunyuan15-p0-run.txt`, or by manual
`workflow_dispatch`. Ordinary code commits, PRs, and schedules cannot launch
this V100 job. All V100 work is serialized, first checks for an idle GPU, and
uses offline caches. Do not use a broad push trigger or touch `main`.

## Interpretation of current telemetry

`baseline.json` will contain the five *per-request* hot E2E measurements.
`diagnostic.json` contains separate phase boundary measurements; those
include deliberate CUDA synchronization and cannot be used for adoption.
`cold.json` contains one whole-process one-video wall time. Phase markers
from the pipeline may not include early transformer H2D transfers, so the
phase labeled `denoise_inclusive` is **provisional until that attribution is
verified**. GPU pin-pool reuse and H2D metrics are currently explicitly
unverified rather than invented.

Do not start P1 performance adoption comparisons or custom kernels until
the remaining P0 gate items are resolved with actual runner results.
