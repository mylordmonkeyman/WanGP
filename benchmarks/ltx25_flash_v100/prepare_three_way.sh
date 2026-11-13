#!/usr/bin/env bash
# One-shot V100 GPU build+dependency preparation. NEVER called automatically
# on a code push. Called only by the opt-in benchmark GitHub Actions workflow.
set -euo pipefail

if [[ "${GITHUB_ACTIONS:-}" != "true" ]]; then
    echo "Refusing execution outside an explicitly triggered GitHub Actions job." >&2
    exit 2
fi
: "${RUNNER_TEMP:?RUNNER_TEMP must exist}"
: "${GITHUB_SHA:?GITHUB_SHA must exist}"
if [[ "$(basename "$(pwd)")" != "WanGP" ]]; then
    echo "Expected to run from the WanGP checkout, got $(pwd)" >&2
    exit 2
fi

PYTHON=/opt/venvs/ltx25-cu128/bin/python
WORK="$RUNNER_TEMP/ltx25-flash-v100-compile"
RESULTS="$WORK/benchmark-results"
mkdir -p "$RESULTS" "$WORK/logs"

# Python syntax tests do not allocate V100 memory. This workflow as a whole
# still requires the user's explicit dispatch permission.
"$PYTHON" -m py_compile \
    shared/ltx25_v100_attention.py \
    shared/attention.py \
    models/ltx2/ltx2.py \
    models/ltx2/ltx_core/model/transformer/attention.py \
    benchmarks/ltx25_flash_v100/three_way_benchmark.py \
    benchmarks/ltx25_flash_v100/compile_smoke.py
bash -n benchmarks/ltx25_flash_v100/compile_on_runner.sh

# Previously verified on this exact runner image by run #43371330815.
# Installs missing compiler tooling with pip --target under RUNNER_TEMP,
# NEVER into /opt/venvs, site-packages, or host /usr/local.
bash benchmarks/ltx25_flash_v100/compile_on_runner.sh

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$WORK/build-tools/bin:$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$WORK/build-tools:$WORK/cat/flash-attention-v100:$WORK/zhin:$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST=7.0

# Fill any missing *Python-only* build/runtime dependencies in RUNNER_TEMP.
# The version-locked PyTorch CUDA installation and MMGP are never changed.
mapfile -t EXTRA_PYTHON_DEPS < <("$PYTHON" - <<'PY'
import importlib.util
optional = {
    "packaging": "packaging>=24.2",
    "setuptools": "setuptools>=77.0.3,<81",
    "wheel": "wheel>=0.45",
    "jinja2": "jinja2>=3.1",
    "einops": "einops>=0.8,<1",
    "numpy": "numpy>=1.26,<3",
    "tqdm": "tqdm>=4.66",
    "psutil": "psutil>=6",
}
for module, package in optional.items():
    if importlib.util.find_spec(module) is None:
        print(package)
PY
)
if (( ${#EXTRA_PYTHON_DEPS[@]} )); then
    echo "Installing missing pure-Python/runtime wheels into temporary directory: ${EXTRA_PYTHON_DEPS[*]}"
    "$PYTHON" -m pip install --no-deps --target "$WORK/build-tools" \
        "${EXTRA_PYTHON_DEPS[@]}" \
        2>&1 | tee "$WORK/logs/benchmark-missing-deps-install.log"
fi
"$PYTHON" -m unittest benchmarks.ltx25_flash_v100.test_routes \
    2>&1 | tee "$WORK/logs/adapter-contract-tests.log"

echo "=== Verify Python imports and provenance ==="
"$PYTHON" - <<'PY' | tee "$WORK/logs/benchmark-dependencies.log"
import importlib, json, pathlib, sys, sysconfig
import torch
modules = ["torch", "packaging", "setuptools", "wheel", "jinja2",
           "numpy", "einops", "tqdm", "psutil", "mmgp",
           "flash_attn_v100", "vllm_flash_attn"]
details = {}
missing = []
for name in modules:
    try:
        obj = importlib.import_module(name)
        details[name] = {"file": str(getattr(obj, "__file__", "(built-in)")),
                         "version": str(getattr(obj, "__version__", "unknown"))}
    except Exception as exc:
        missing.append(f"{name}: {type(exc).__name__}: {exc}")
if missing:
    raise SystemExit("Required benchmark dependencies missing: " + "; ".join(missing))
from shared.attention import pay_attention
assert callable(pay_attention)
assert torch.__version__.startswith("2.10.0+cu128"), torch.__version__
assert torch.cuda.get_device_capability() == (7, 0)
assert pathlib.Path(sysconfig.get_path("include"), "Python.h").is_file()
paths = {
    "flash_attn_v100": pathlib.Path(details["flash_attn_v100"]["file"]).resolve(),
    "vllm_flash_attn": pathlib.Path(details["vllm_flash_attn"]["file"]).resolve(),
}
# Never accidentally load an unrelated extension installed in the base venv.
assert "ltx25-flash-v100-compile/cat/flash-attention-v100" in str(paths["flash_attn_v100"]), paths
assert "ltx25-flash-v100-compile/zhin" in str(paths["vllm_flash_attn"]), paths
print(json.dumps({"dependencies": details,
                  "torch_cuda": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name(),
                  "python": sys.version.split()[0]}, indent=2))
PY

# Reusable ABI-locked compiled Python wrappers + .so for future experiments.
# We never upload baked model weights or the runner's Python environment.
echo "=== Package built extensions for reuse ==="
BUNDLE="$WORK/compiled-runtime"
mkdir -p "$BUNDLE/1cat" "$BUNDLE/zhinianqin"
cp -a "$WORK/cat/flash-attention-v100/flash_attn_v100" "$BUNDLE/1cat/"
find "$WORK/cat/flash-attention-v100" -maxdepth 1 -name 'flash_attn_v100_cuda*.so' \
    -exec cp -v {} "$BUNDLE/1cat/" \;
cp -a "$WORK/zhin/vllm_flash_attn" "$BUNDLE/zhinianqin/"
cat > "$BUNDLE/ABI.txt" <<'EOF'
Environment: Linux x86_64, CPython 3.12, PyTorch 2.10.0+cu128, CUDA 12.8, NVIDIA sm70.
1CatAI/1Cat-vLLM: c65714de843b2e743b0e234727eb1c1ccdb29f20
zhinianqin/flash-attention-v100: c2eda5e6115b98c3ba4bfd181570668742eece22
Only use with a matching PyTorch/CUDA ABI. No model weights included.
EOF
if ! find "$BUNDLE/1cat" -name 'flash_attn_v100_cuda*.so' | grep -q .; then
    echo "Missing 1Cat compiled .so" >&2
    exit 1
fi
if ! find "$BUNDLE/zhinianqin" -name '*_vllm_fa2_C*.so' | grep -q .; then
    echo "Missing zhinianqin compiled .so" >&2
    exit 1
fi
tar -C "$BUNDLE" -czf "$WORK/compiled-v100-attention-sm70-cu128.tar.gz" .
sha256sum "$WORK/compiled-v100-attention-sm70-cu128.tar.gz" \
    | tee "$WORK/logs/compiled-bundle-sha256.txt"
echo "Finished dependency validation and reusable extension packaging."
