#!/usr/bin/env bash
# Build a credential-free image from the existing TESTED runner site-packages.
# No docker commit: the live GitHub runner can contain registration secrets.
set -Eeuo pipefail
REPO=mylordmonkeyman/WanGP
CONTAINER="${LTX25_RUNNER_CONTAINER:-gha-v100-ltx25}"
BASE="${LTX25_BASE_IMAGE:-local/gha-v100-ltx25:cuda12.8}"
TAG="${LTX25_BAKED_IMAGE:-local/gha-v100-ltx25:cu128-ltx25-baked-v1}"
SRC_VENV=/cache/venvs/ltx25-torch210-cu128
DEST_VENV=/opt/venvs/ltx25-cu128
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
die() { echo "ERROR: $*" >&2; exit 1; }
command -v docker >/dev/null || die "docker missing"
command -v gh >/dev/null || die "GitHub CLI missing (required to verify no active GPU jobs)"
docker info >/dev/null || die "Cannot reach Docker"
[[ "$(docker inspect --format '{{.State.Running}}' "$CONTAINER")" == true ]] || die "Runner is not Up"
docker image inspect "$BASE" >/dev/null || die "Base image missing: $BASE"

echo "=== Check runner idle; fail closed ==="
ACTIVE="$(gh api "repos/$REPO/actions/runs?branch=perf%2Fltx25-convrot-sm70&per_page=100" \
  --jq '[.workflow_runs[] | select(.status == "in_progress" or .status == "queued" or .status == "waiting" or .status == "pending" or .status == "requested")] | length')" \
  || die "Cannot inspect GitHub Actions status"
[[ "$ACTIVE" == 0 ]] || die "$ACTIVE GitHub workflow(s) still active/queued"
docker exec -i "$CONTAINER" "$SRC_VENV/bin/python" - <<'PY'
import sys, torch, torchaudio, torchvision, matplotlib, onnxruntime
assert sys.version_info[:2] == (3,12)
assert torch.__version__.startswith("2.10.0+cu128")
assert torchvision.__version__.startswith("0.25.0+cu128")
assert torchaudio.__version__.startswith("2.10.0+cu128")
print("Current tested runtime:",torch.__version__)
PY

echo "=== Snapshot only installed Python wheels (not runner credentials/models) ==="
CONTEXT="$(mktemp -d "${LTX25_BUILD_ROOT:-$HOME}/ltx25-bake.XXXXXXXX")"
echo "Build context: $CONTEXT"
SITE="$(docker exec "$CONTAINER" "$SRC_VENV/bin/python" -c \
  'import site; print(next(p for p in site.getsitepackages() if p.endswith("site-packages")))')"
[[ "$SITE" == "$SRC_VENV/lib/python3.12/site-packages" ]] || die "Unexpected venv package path: $SITE"
mkdir -p "$CONTEXT/site-packages"
docker cp "$CONTAINER:$SITE/." "$CONTEXT/site-packages/"
docker exec "$CONTAINER" "$SRC_VENV/bin/python" -m pip freeze --all > "$CONTEXT/freeze.txt"
cp "$SCRIPT_DIR/Dockerfile.snapshot" "$CONTEXT/Dockerfile"

echo "=== Build versioned runner image $TAG ==="
DOCKER_BUILDKIT=1 docker build --progress=plain \
  --build-arg "BASE_IMAGE=$BASE" -t "$TAG" \
  -f "$CONTEXT/Dockerfile" "$CONTEXT"
docker run --rm --entrypoint "$DEST_VENV/bin/python" "$TAG" \
  -c 'import torch,matplotlib,onnxruntime; print("BAKED",torch.__version__)'

echo "=== Prepare existing runner, without restart or changing its mounts ==="
docker exec -u root "$CONTAINER" bash -lc '
  set -euo pipefail
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    python3.12-dev python3-dev build-essential pkg-config
  test -f /usr/include/python3.12/Python.h
  mkdir -p /opt/venvs
  if [ -e /opt/venvs/ltx25-cu128 ] && [ ! -L /opt/venvs/ltx25-cu128 ]; then
    echo "Refusing to replace an existing real venv" >&2
    exit 1
  fi
  ln -sfn /cache/venvs/ltx25-torch210-cu128 /opt/venvs/ltx25-cu128
  /opt/venvs/ltx25-cu128/bin/python -c \
    "import os,sysconfig; assert os.path.isfile(os.path.join(sysconfig.get_path(\"include\"),\"Python.h\"))"
'
echo "=== SUCCESS: no runner restart ==="
docker image ls "$TAG"
docker exec "$CONTAINER" "$DEST_VENV/bin/python" -c \
  'import torch;print("Active:",torch.__version__)'
echo "Baked image for future container recreation: $TAG"
echo "Existing runner: active and now supports the zero-pip workflow."
echo "Build context retained at $CONTEXT (can be removed after testing)."
