# LTX-2.5 SM70 runner: baked dependency image

## Current: complete model image required before further GPU tests

The Python-only image `local/gha-v100-ltx25:cu128-ltx25-baked-v1`
was verified, and the original R0 baseline at **576x320, 49 frames,
8+3 denoising steps** succeeded in GitHub run **43311535571**
(113.3 seconds wall time). The next matched visual-quality comparison
is **1024x576, 49 frames**: first the unmodified R0 path, then the
explicitly opted-in SM70 cuBLAS Candidate E path. No higher-resolution
run has been performed yet.

**Model files are no longer fetched or SHA256-hashed by Actions
benchmark jobs.** A new host-only image build stage takes the 17 existing
files from the verified checkpoint cache and bakes them at
`/opt/ltx25/checkpoints`, together with an image-build-time checksum
inventory. The baked image is
`local/gha-v100-ltx25:cu128-ltx25-complete-v1`; no test should start
until that image is deployed.

Run on `mdsserver` (not from WSL):

```bash
git -C /tmp/ltx25-bake-src.dmHY1b/WanGP pull --ff-only
B=/tmp/ltx25-bake-src.dmHY1b/WanGP/benchmarks/ltx25_convrot/docker

# One-time image creation, NOT an Actions test. This does not
# download any models and it fails before building if disk is too full.
bash "$B/bake_model_image.sh"

# Verify Compose service/mounts and expected change without disruption.
bash "$B/deploy_complete_runner.sh"

# Recreate only this runner service, preserving its Compose settings,
# GPU access and persistent mounts.
LTX25_APPLY=1 bash "$B/deploy_complete_runner.sh"
```

The builder checks available Docker data-root storage and requires
roughly **2x the total checkpoint bytes plus 10 GiB** of free space
before it starts. If that check fails, do **not** force the build
on the 100 GiB system LV: move/expand Docker's data-root onto a larger
filesystem deliberately. Do not prune images/volumes used by other
LLM services without reviewing them.

The deployment script is **dry-run by default**. It requires Docker
Compose labels on the current runner and will fail rather than guess
if the runner was started with plain `docker run`. Applying it checks
for active GitHub Actions jobs, preserves the original Compose files
and creates a separate image override for this runner service. It does
not modify or remove the old image.

Once the complete-image runner is online, use the
`ltx25-quality-1024.yml` workflow, **R0 first**. When workflow dispatch
isn't registered on GitHub's default branch, a change to
`benchmarks/ltx25_convrot/quality_r0_dispatch.signal` triggers one R0
run from this feature branch. A later separate change to
`quality_e_dispatch.signal` triggers Candidate E. Ordinary code
commits trigger neither. Do not dispatch both concurrently.

High-resolution outputs are separate:

- `/cache/ltx25/videos/r0_quality_1024/`
- `/cache/ltx25/videos/e_quality_1024/`
- Existing `/cache/ltx25/videos/r0_baseline/` remains unchanged.

Results publish to `perf/ltx25-run-results` under
`ltx25-quality/{variant}/latest.json`. The quality runs are offline
and use stat-only checkpoint checks; the image build performs SHA256
once. Candidate E still requires visual inspection and actual usage
counters before any performance claim.

---

This directory replaces *per-job pip installation* in the three LTX-2.5
workflows. Use it only for the self-hosted V100 runner
`gha-v100-ltx25`, while all Actions jobs on the branch are completed.

## Historical Python-only build (already completed; do not rerun)

```bash
git clone --depth 1 --single-branch \
  -b perf/ltx25-convrot-sm70 \
  https://github.com/mylordmonkeyman/WanGP.git \
  /opt/ai/WanGP-ltx25-build

bash /opt/ai/WanGP-ltx25-build/benchmarks/ltx25_convrot/docker/bake_runner_image.sh

# HDR weights were already staged and verified; no longer run
# stage_hdr_weights.sh in any benchmark workflow.
```

The bake script uses an already tested Python 3.12/CUDA 12.8 virtual
environment and copies **only installed Python packages** into the build
context. It never uses `docker commit` on the live runner, which may contain
GitHub Actions credentials. The clean image is built FROM
`local/gha-v100-ltx25:cuda12.8` and tagged
`local/gha-v100-ltx25:cu128-ltx25-baked-v1`. It adds system Python 3.12
development headers, compiler tools, and media runtime libraries.

The script does NOT recreate or interrupt the running runner. It installs
the matching system development headers in the existing container and maps
`/opt/venvs/ltx25-cu128` to the already-tested venv under the persistent
`/cache` volume. All future GPU tests use that stable path without pip.

When you intentionally recreate the runner later, use the new baked image
tag in your **existing Docker Compose/service definition** so the original
volumes, registration, runner labels, and GPU selection are preserved.
The new image has the complete Python environment as real files under
`/opt/venvs/ltx25-cu128`, not as a /cache symlink.

The original Python-only bake used a persistent checkpoint cache and
was sufficient for the first reference video. **For all subsequent GPU tests,
use the complete model image instead.** Its 17 required model assets are
copied and SHA256-inventoried during the host-only image build, never fetched
or hashed during a test.

## Recover an interrupted or failed build without copying 9.3GB again

The Dockerfile must **not** call `torch.cuda.get_arch_list()` at build
time: Docker BuildKit normally exposes no GPU, so that API can return
an empty list even for a valid CUDA wheel. The Dockerfile checks CUDA
build support only, and the bake script checks SM70 in a separate
`docker run --gpus all` runtime test.

If your first attempt reached the Dockerfile's final validation step,
reuse the **existing** build context:

```bash
git -C /tmp/ltx25-bake-src.dmHY1b/WanGP pull --ff-only

LTX25_BUILD_CONTEXT=/home/mymonkeyman/ltx25-bake.TGoLuSzn \
  bash /tmp/ltx25-bake-src.dmHY1b/WanGP/benchmarks/ltx25_convrot/docker/bake_runner_image.sh
```

The script refreshes only the Dockerfile in that directory, reuses the
prior package snapshot, takes advantage of BuildKit's cached successful
layers, builds and tests the image with access to the GPU, and configures
the existing runner **without replacing the runner container**. Only
run while all LTX-2.5 workflows are inactive.

## After a successful image build but failed live-runner apt setup

If `docker build` completed and printed `BAKED_SM70_OK` but the script
then failed with `setgroups ... Operation not permitted`, do **not**
rebuild. The currently running GitHub runner has a restricted security
context that prevents apt from dropping privileges to its `_apt` user.

Run the independent activation script:

```bash
git -C /tmp/ltx25-bake-src.dmHY1b/WanGP pull --ff-only
bash /tmp/ltx25-bake-src.dmHY1b/WanGP/benchmarks/ltx25_convrot/docker/activate_baked_runner.sh
```

It copies only the already-built Python 3.12 development headers from the
clean image into the current runner and aliases the tested persistent venv
to `/opt/venvs/ltx25-cu128`. It verifies CUDA SM70, Python imports, and a
native C compile. There are **no** apt calls, pip installs, image builds, or
runner restarts. The image remains available for a planned future
Compose/service recreation.

## Quick verification / next run

```bash
docker exec gha-v100-ltx25 /opt/venvs/ltx25-cu128/bin/python \
  -c 'import torch,matplotlib,onnxruntime;print(torch.__version__)'

# Historical R0 was already completed. Use ltx25-quality-1024.yml
# only after the complete model image has been built and deployed.
```

Start one workflow at a time; do not edit a dispatch signal file or trigger
a replacement while any test is pending or in progress.

The whole Python environment comes from a frozen *package snapshot*,
not a resolved/PyPI install at image-build time. `freeze.txt` is retained
inside the image for provenance. This method prioritizes exact compatibility
with the working V100 runtime; when deliberate dependency upgrades are
needed, test them first and rebuild a new versioned image from the updated
stable venv.
