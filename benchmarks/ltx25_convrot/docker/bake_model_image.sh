#!/usr/bin/env bash
# Build an immutable model+Python V100 GitHub Actions runner image on mdsserver.
# This is a HOST-ONLY one-time image BUILD step, never a test-workflow step.
# It DOES NOT download checkpoints, mutate model files or restart the runner.
# The source checkpoint cache is used read-only as a BuildKit named context.
set -Eeuo pipefail

REPO=mylordmonkeyman/WanGP
CONTAINER="${LTX25_RUNNER_CONTAINER:-gha-v100-ltx25}"
BASE="${LTX25_BAKED_IMAGE:-local/gha-v100-ltx25:cu128-ltx25-baked-v1}"
IMAGE="${LTX25_COMPLETE_IMAGE:-local/gha-v100-ltx25:cu128-ltx25-complete-v1}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
die() { echo "ERROR: $*" >&2; exit 1; }

for tool in docker gh python3; do command -v "$tool" >/dev/null || die "$tool missing"; done
[[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" == true ]] || die "Runner not running"
docker image inspect "$BASE" >/dev/null || die "Base Python-baked image missing: $BASE"
ACTIVE="$(gh api "repos/$REPO/actions/runs?branch=perf%2Fltx25-convrot-sm70&per_page=100" \
  --jq '[.workflow_runs[] | select(.status=="in_progress" or .status=="queued" or .status=="waiting" or .status=="pending" or .status=="requested")] | length')" \
  || die "Cannot verify idle Actions state"
[[ "$ACTIVE" == 0 ]] || die "$ACTIVE pending/active workflows: no model bake while runner is working"

# Resolve the SOURCE on the HOST from existing Docker mounts. Importantly:
# /cache is a writable bind mount and would hide checkpoint files baked there.
# The output image instead stores physical files under /opt/ltx25/checkpoints.
RAW="$(docker inspect -f '{{json .Mounts}}' "$CONTAINER")"
SOURCE="$(printf '%s' "$RAW" | python3 -c '
import json,os,sys
mounts=json.load(sys.stdin)
path="/cache/ltx25/checkpoints"
matches=[m for m in mounts if path==m["Destination"] or path.startswith(m["Destination"].rstrip("/")+"/")]
if not matches:
    raise SystemExit("No persistent /cache mount found; do not guess model source path")
m=max(matches,key=lambda m:len(m["Destination"]))
relative=path[len(m["Destination"]):].lstrip("/")
print(os.path.join(m["Source"],relative))
')" || die "Could not identify host-side checkpoint directory"
echo "Existing host checkpoint source: $SOURCE"
[[ -d "$SOURCE" && -r "$SOURCE" ]] || die "Host cannot read source directory: $SOURCE"
[[ ! -L "$SOURCE" ]] || die "Source must be a real checkpoint directory, not a symlink"

# Reject symlinks and missing/mismatched files before starting a multi-GiB
# build. This check reads ONLY file names and inode sizes, NOT safetensors
# content or SHA256. SHA256 inventory is generated ONCE in the Dockerfile.
mapfile -t FILES < "$SCRIPT_DIR/required_models.txt"
[[ "${#FILES[@]}" == 17 ]] || die "Expected exactly 17 model assets"
TOTAL=0
for file in "${FILES[@]}"; do
  [[ "$file" != /* && "$file" != *..* && "$file" != *//* ]] || die "Unsafe model path: $file"
  target="$SOURCE/$file"
  [[ -f "$target" && -r "$target" && ! -L "$target" && -s "$target" ]] || die "Missing/unreadable/symlinked required file: $target"
  size="$(stat -c %s "$target")"
  TOTAL=$((TOTAL+size))
  printf '%12s  %s\n' "$size" "$file"
done
echo "Model asset total: $TOTAL bytes"

# Export/unpack may require additional working space. Check before large
# build. Fail safely instead of exhausting Docker's storage or the root LV.
ROOT="$(docker info --format '{{.DockerRootDir}}')"
FREE="$(df -PB1 "$ROOT" | awk 'END {print $4}')"
REQUIRED=$((TOTAL*2 + 10*1024*1024*1024))
echo "Docker storage: $ROOT | free: $FREE bytes | minimum target: $REQUIRED bytes"
if (( FREE < REQUIRED )); then
  echo "Not enough Docker storage for a ~45GB checkpoint layer plus export temp space." >&2
  echo "Move/expand Docker data-root onto the larger filesystem first; do NOT force a build on the 100GB root LV." >&2
  exit 1
fi

CONTEXT="$(mktemp -d "${LTX25_BUILD_ROOT:-$HOME}/ltx25-model-context.XXXXXXXX")"
trap 'rm -rf "$CONTEXT"' EXIT
cp "$SCRIPT_DIR/Dockerfile.models" "$CONTEXT/Dockerfile"
cp "$SCRIPT_DIR/required_models.txt" "$CONTEXT/required_models.txt"

echo "=== Build immutable model image; only here is checkpoint SHA256 computed ==="
DOCKER_BUILDKIT=1 docker build --progress=plain \
  --build-context "ltx25_weights=$SOURCE" \
  --build-arg "BASE_IMAGE=$BASE" \
  -f "$CONTEXT/Dockerfile" -t "$IMAGE" "$CONTEXT"

echo "=== Verify baked files and package without remounting /cache ==="
docker run --rm --entrypoint /opt/venvs/ltx25-cu128/bin/python "$IMAGE" -c '
from pathlib import Path
import json,torch,mutagen
p=Path("/opt/ltx25/model-image-manifest.json")
data=json.loads(p.read_text())
assert len(data["files"])==17
assert all((Path("/opt/ltx25/checkpoints")/f["path"]).stat().st_size==f["bytes"] for f in data["files"])
assert torch.__version__.startswith("2.10.0+cu128")
print("LTX25_COMPLETE_IMAGE_OK",len(data["files"]),"pinned assets",torch.__version__)
'
echo "=== DONE, image built: $IMAGE ==="
echo "No runner container was recreated. Switch its existing Compose/deploy image to $IMAGE when ready."
