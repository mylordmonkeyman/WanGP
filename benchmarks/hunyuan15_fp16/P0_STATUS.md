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
| Single persistent process | Implemented, unexecuted | One `wgp.py --process settings.json` containing six tasks; production `process_tasks_cli` reuses `wan_model` |
| One warm-up + five hot requests | Implemented, unexecuted | First full generation discarded; request start/end CUDA sync |
| Model and MMGP identity | Instrumented, unexecuted | Same PID plus object IDs, 54 block end placements, model load count |
| Hot offload-inclusive denoise | Diagnostic incomplete | Pipeline timing boundaries instrumented; early MMGP preloads/H2D remain to be traced |
| Q/K pre-RMSNorm safety and FP32 reference | Diagnostic instrumented, unexecuted | Per-block image/text Q/K maxima, overflow counts; sample FP32-reference norm error |
| Attention dispatch shapes and lengths | Diagnostic instrumented, unexecuted | Actual Q/K shapes and available q_lens/k_lens values; kernel name profiling still required |
| VAE tile geometry and executed decoder calls | Diagnostic instrumented, unexecuted | VAE state, per-decoder-call input shapes; VAE precision probe still incomplete |
| Separate cold one-video reference | Harness configured, unexecuted | Entire CLI process wall time, with one task only |
| Changed prompt sequence | Harness configured, unexecuted | Same persistent queue, last request changes prompt only |
| CPU-only fixture and noise gate | Committed, execution not verified | MAD, bootstrap CI, fixed config tests |
| CPU-only branch push workflow | Committed, execution not verified | `.github/workflows/hunyuan15-p0-cpu.yml`; no GPU jobs |
| V100 workflow | Committed, not executed | Manual dispatch only; idle GPU guard; single-job V100 serialization |

**P0 has NOT passed.** All numerical and speed conclusions are pending actual
runner tests. No performance default, model math, scheduler, VAE precision,
memory profile, or attention kernel has been changed.

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

### GitHub dispatch limitation

GitHub requires a `workflow_dispatch` workflow to exist on the repository
default branch before it can be triggered by Actions/CLI/API. This workflow is
intentionally **branch-only**, under the specification's prohibition on editing
`main`. Therefore it is staged for review/reuse but cannot be dispatched
through GitHub Actions yet without an existing default-branch dispatch entry
point or a separate authorized change to that restriction. Do not add an
automatic `push` or `schedule` trigger to work around the restriction.

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
