# LTX-2.5 SM70 runner: baked dependency image

This directory replaces *per-job pip installation* in the three LTX-2.5
workflows. Use it only for the self-hosted V100 runner
`gha-v100-ltx25`, while all Actions jobs on the branch are completed.

## One-time build from the host

```bash
git clone --depth 1 --single-branch \
  -b perf/ltx25-convrot-sm70 \
  https://github.com/mylordmonkeyman/WanGP.git \
  /opt/ai/WanGP-ltx25-build

bash /opt/ai/WanGP-ltx25-build/benchmarks/ltx25_convrot/docker/bake_runner_image.sh

bash /opt/ai/WanGP-ltx25-build/benchmarks/ltx25_convrot/docker/stage_hdr_weights.sh
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

Keep model checkpoints out of the image. They are stored in
`/cache/ltx25/checkpoints` and verified independently. The second script
downloads the pinned HDR scene embeddings and LoRA discovered when the
first real CLI generation attempt got through the dry-run.

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

## Quick verification / next run

```bash
docker exec gha-v100-ltx25 /opt/venvs/ltx25-cu128/bin/python \
  -c 'import torch,matplotlib,onnxruntime;print(torch.__version__)'

gh workflow run ltx25-runtime-batch.yml \
  --repo mylordmonkeyman/WanGP \
  --ref perf/ltx25-convrot-sm70
```

Start one workflow at a time; do not edit a dispatch signal file or trigger
a replacement while any test is pending or in progress.

The whole Python environment comes from a frozen *package snapshot*,
not a resolved/PyPI install at image-build time. `freeze.txt` is retained
inside the image for provenance. This method prioritizes exact compatibility
with the working V100 runtime; when deliberate dependency upgrades are
needed, test them first and rebuild a new versioned image from the updated
stable venv.
