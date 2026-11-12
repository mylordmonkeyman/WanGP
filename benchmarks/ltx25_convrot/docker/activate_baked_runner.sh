#!/usr/bin/env bash
# Activate the already-built LTX-2.5 Python environment on the existing
# GitHub Actions runner WITHOUT apt, image rebuild, or runner restart.
#
# Docker exec -u root can write headers, but apt attempts setuid/setgid to
# the _apt user and fails when the runner lacks those Linux capabilities.
# Copy the Python headers from our credential-free baked image instead.
set -Eeuo pipefail

REPO="mylordmonkeyman/WanGP"
CONTAINER="${LTX25_RUNNER_CONTAINER:-gha-v100-ltx25}"
IMAGE="${LTX25_BAKED_IMAGE:-local/gha-v100-ltx25:cu128-ltx25-baked-v1}"
SOURCE=/cache/venvs/ltx25-torch210-cu128
ALIAS=/opt/venvs/ltx25-cu128

die() { echo "ERROR: $*" >&2; exit 1; }
command -v docker >/dev/null || die "docker is required"
command -v gh >/dev/null || die "gh is required to check that jobs are idle"
docker image inspect "$IMAGE" >/dev/null || die "Baked image missing: $IMAGE"
[[ "$(docker inspect --format '{{.State.Running}}' "$CONTAINER")" == true ]] \
  || die "Runner container is not running"

# Fail closed if the dedicated runner's Actions branch has an outstanding job.
ACTIVE="$(gh api "repos/$REPO/actions/runs?branch=perf%2Fltx25-convrot-sm70&per_page=100" \
 --jq '[.workflow_runs[] | select(.status == "in_progress" or .status == "queued" or .status == "waiting" or .status == "pending" or .status == "requested")] | length')" \
 || die "Could not verify GitHub Actions idle state"
[[ "$ACTIVE" == 0 ]] || die "$ACTIVE workflows are active/queued. Not modifying the running runner."

docker exec "$CONTAINER" "$SOURCE/bin/python" -c \
  'import torch;assert torch.__version__.startswith("2.10.0+cu128")' \
  || die "The existing persistent Python environment is unavailable"

echo "=== Copy only matching Ubuntu Python 3.12 development headers from baked image ==="
TMP="$(mktemp -d)"
BAKED=""
cleanup() {
  if [[ -n "$BAKED" ]]; then docker rm -f "$BAKED" >/dev/null 2>&1 || true; fi
  rm -rf "$TMP"
}
trap cleanup EXIT

BAKED="$(docker create --entrypoint /bin/true "$IMAGE")"
mkdir -p "$TMP/include/python3.12" "$TMP/include/x86_64-linux-gnu/python3.12"
docker cp "$BAKED:/usr/include/python3.12/." "$TMP/include/python3.12/"
test -s "$TMP/include/python3.12/Python.h" || die "Baked image lacks Python.h"
if docker cp "$BAKED:/usr/include/x86_64-linux-gnu/python3.12/." \
  "$TMP/include/x86_64-linux-gnu/python3.12/" 2>/dev/null; then
  echo "Copied multiarch Python headers from baked image"
else
  echo "No separate multiarch header directory present"
fi

docker exec -u 0 "$CONTAINER" sh -ec '
  mkdir -p /usr/include/python3.12 \
    /usr/include/x86_64-linux-gnu/python3.12 /opt/venvs
'
docker cp "$TMP/include/python3.12/." "$CONTAINER:/usr/include/python3.12/"
if [[ -n "$(find "$TMP/include/x86_64-linux-gnu/python3.12" -mindepth 1 -print -quit)" ]]; then
  docker cp "$TMP/include/x86_64-linux-gnu/python3.12/." \
    "$CONTAINER:/usr/include/x86_64-linux-gnu/python3.12/"
fi

echo "=== Reuse tested /cache venv at stable workflow path ==="
docker exec -u 0 "$CONTAINER" sh -ec '
  if [ -e /opt/venvs/ltx25-cu128 ] && [ ! -L /opt/venvs/ltx25-cu128 ]; then
    echo "Refusing to overwrite existing non-symlink venv" >&2
    exit 1
  fi
  ln -sfn /cache/venvs/ltx25-torch210-cu128 /opt/venvs/ltx25-cu128
'

echo "=== Validate Python.h, CUDA/SM70, and native C compiler ==="
docker exec -i "$CONTAINER" "$ALIAS/bin/python" - <<'PY'
import importlib, os, pathlib, subprocess, sys, sysconfig
import torch, torchaudio, torchvision
h = pathlib.Path(sysconfig.get_path("include")) / "Python.h"
assert h.is_file(), f"Python.h missing: {h}"
assert torch.__version__.startswith("2.10.0+cu128"), torch.__version__
assert torchaudio.__version__.startswith("2.10.0+cu128"), torchaudio.__version__
assert torchvision.__version__.startswith("0.25.0+cu128"), torchvision.__version__
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (7, 0)
assert "sm_70" in torch.cuda.get_arch_list()
for mod in ("matplotlib", "onnxruntime", "diffusers", "transformers", "optimum.quanto"):
    importlib.import_module(mod)
compiler = os.environ.get("CC","cc")
r = subprocess.run(
    [compiler, "-fsyntax-only", "-x", "c", "-I", str(h.parent), "-"],
    input="#include <Python.h>\nint main(void){return PY_MAJOR_VERSION == 3 ? 0 : 1;}\n",
    capture_output=True,text=True)
assert r.returncode == 0, f"C compiler/Python headers failed: {r.stderr[-1800:]}"
print("ACTIVE_RUNTIME_OK",sys.executable,torch.__version__,f"Python.h={h}")
print("PYTHON_C_HEADERS_COMPILE_OK")
PY

echo "=== Success: live runner unchanged, no apt and no pip ==="
docker ps --filter "name=$CONTAINER" --format 'table {{.Names}}\t{{.Status}}'
echo "The installed image is still $IMAGE; switch the service to this tag on a planned runner recreation."
