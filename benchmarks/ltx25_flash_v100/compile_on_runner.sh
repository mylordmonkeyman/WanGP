#!/usr/bin/env bash
# Compile-only V100 CI pilot. Runs inside an explicitly requested workflow.
# Keeps all source trees and build outputs under RUNNER_TEMP.
set -euo pipefail

: "${RUNNER_TEMP:?This script must run inside a GitHub Actions runner}"
REPO_ROOT="$(pwd)"
WORK="$RUNNER_TEMP/ltx25-flash-v100-compile"
LOGS="$WORK/logs"
mkdir -p "$LOGS" "$WORK/ccache" "$WORK/build-tools"

PYTHON=/opt/venvs/ltx25-cu128/bin/python
CAT_REV=c65714de843b2e743b0e234727eb1c1ccdb29f20
ZHIN_REV=c2eda5e6115b98c3ba4bfd181570668742eece22
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$WORK/build-tools/bin:$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$WORK/build-tools${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST=7.0
export MAX_JOBS=8
export NVCC_THREADS=1
export CCACHE_DIR="$WORK/ccache"
export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

echo "=== Environment ===" | tee "$LOGS/environment.txt"
{
    date -u
    uname -a
    command -v nvcc
    nvcc --version
    nvidia-smi -L
    "$PYTHON" - <<'PY'
import torch, sys
print("Python", sys.version)
print("torch", torch.__version__)
print("CUDA", torch.version.cuda)
print("SM", torch.cuda.get_device_capability())
assert torch.__version__.startswith("2.10.0+cu128"), "Unexpected baked PyTorch; stop rather than mutate it"
assert torch.cuda.get_device_capability() == (7, 0), "Runner is not SM70"
PY
} 2>&1 | tee -a "$LOGS/environment.txt"

# Zhinianqin requires CMake >=3.26; install build utilities only in RUNNER_TEMP
# when absent/too old. Never pip-install into /opt/venvs or the host system.
if ! "$PYTHON" - <<'PY'
import shutil, subprocess
from importlib.metadata import version
from packaging.version import Version
assert shutil.which("cmake"), "cmake missing"
v = subprocess.check_output(["cmake", "--version"], text=True).splitlines()[0].split()[-1]
assert Version(v) >= Version("3.26.1"), "cmake too old"
assert shutil.which("ninja"), "ninja missing"
import jinja2, wheel
print("Build tools present:", v, "jinja2", version("jinja2"), "wheel", version("wheel"))
PY
then
    "$PYTHON" -m pip install --no-deps --target "$WORK/build-tools" \
        "cmake>=3.26.1,<4" "ninja>=1.11" "jinja2>=3.1" "wheel>=0.45" \
        "packaging>=24.2" "setuptools>=77.0.3,<81.0.0" \
        2>&1 | tee "$LOGS/build-tools-install.txt"
fi
cmake --version | head -1 | tee "$LOGS/cmake.txt"

# Clone fixed upstream revisions. For 1Cat, sparse checkout prevents
# downloading the full vLLM source tree.
echo "=== Fetch pinned 1Cat source ==="
git clone --filter=blob:none --depth=1 --no-checkout \
    https://github.com/1CatAI/1Cat-vLLM.git "$WORK/cat"
git -C "$WORK/cat" sparse-checkout init --cone
git -C "$WORK/cat" sparse-checkout set flash-attention-v100
git -C "$WORK/cat" fetch --depth=1 origin "$CAT_REV"
git -C "$WORK/cat" checkout --detach "$CAT_REV"
test "$(git -C "$WORK/cat" rev-parse HEAD)" = "$CAT_REV"

echo "=== Fetch pinned zhinianqin source ==="
git clone --filter=blob:none --depth=1 --no-checkout \
    https://github.com/zhinianqin/flash-attention-v100.git "$WORK/zhin"
git -C "$WORK/zhin" fetch --depth=1 origin "$ZHIN_REV"
git -C "$WORK/zhin" checkout --detach "$ZHIN_REV"
git -C "$WORK/zhin" submodule update --init --depth=1 csrc/cutlass
test "$(git -C "$WORK/zhin" rev-parse HEAD)" = "$ZHIN_REV"

run_build_and_smoke() {
    local name="$1" path="$2" build_rc smoke_rc
    echo "=== Building $name ==="
    set +e
    (
        cd "$path"
        timeout 2700 "$PYTHON" setup.py build_ext --inplace
    ) 2>&1 | tee "$LOGS/$name-build.log"
    build_rc="${PIPESTATUS[0]}"
    set -e
    echo "$build_rc" > "$LOGS/$name-build.exit"
    echo "Build status $name: $build_rc"

    smoke_rc=99
    if [[ "$build_rc" -eq 0 ]]; then
        echo "=== Import + FP16 D64/D128 noncausal smoke: $name ==="
        set +e
        "$PYTHON" "$REPO_ROOT/benchmarks/ltx25_flash_v100/compile_smoke.py" \
            --backend "$name" --source "$path" \
            2>&1 | tee "$LOGS/$name-smoke.log"
        smoke_rc="${PIPESTATUS[0]}"
        set -e
    else
        echo "Skipped smoke because compilation failed" > "$LOGS/$name-smoke.log"
    fi
    echo "$smoke_rc" > "$LOGS/$name-smoke.exit"
    echo "Smoke status $name: $smoke_rc"
}

run_build_and_smoke 1cat "$WORK/cat/flash-attention-v100"
run_build_and_smoke zhinianqin "$WORK/zhin"

{
  echo "### SM70 LTX-2.5 experimental attention compile"
  echo
  echo "Source branch/commit: $GITHUB_REF_NAME / $GITHUB_SHA"
  echo "CUDA: 12.8; PyTorch: pre-baked 2.10.0+cu128"
  echo
  echo "| Backend | Build (0=pass) | Smoke (0=pass, 99=skipped) |"
  echo "|---|---:|---:|"
  for name in 1cat zhinianqin; do
    echo "| $name | $(cat "$LOGS/$name-build.exit") | $(cat "$LOGS/$name-smoke.exit") |"
  done
  echo
  echo "Full compiler logs are included in the workflow artifact."
} | tee "$LOGS/summary.md" >> "${GITHUB_STEP_SUMMARY:-/dev/null}"

for name in 1cat zhinianqin; do
    if [[ "$(cat "$LOGS/$name-build.exit")" != 0 ]] ||
       [[ "$(cat "$LOGS/$name-smoke.exit")" != 0 ]]; then
        echo "FAIL: $name (see logs)"
        exit 1
    fi
done
echo "PASS: both CUDA extensions built, imported, and passed D64/D128 smoke tests"
