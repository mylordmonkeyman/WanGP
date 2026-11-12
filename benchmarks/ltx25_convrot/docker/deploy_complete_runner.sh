#!/usr/bin/env bash
# Deploy the already-built immutable LTX-2.5 checkpoint image into an existing
# Docker Compose GitHub Actions runner, preserving service config and mounts.
# Dry-run by default. Set LTX25_APPLY=1 for explicit opt-in recreation.
# Does not download, pip install, hash safetensors or build an image.
set -Eeuo pipefail
REPO=mylordmonkeyman/WanGP
C="${LTX25_RUNNER_CONTAINER:-gha-v100-ltx25}"
IMAGE="${LTX25_COMPLETE_IMAGE:-local/gha-v100-ltx25:cu128-ltx25-complete-v1}"
die(){ echo "ERROR: $*" >&2; exit 1; }
for bin in docker gh; do command -v "$bin" >/dev/null || die "Missing $bin"; done
docker image inspect "$IMAGE" >/dev/null || die "Complete image absent. First run bake_model_image.sh"
[[ "$(docker inspect -f '{{.State.Running}}' "$C")" == true ]] \
  || die "Existing runner $C is not running"
ACTIVE="$(gh api "repos/$REPO/actions/runs?branch=perf%2Fltx25-convrot-sm70&per_page=100" \
  --jq '[.workflow_runs[] | select(.status=="in_progress" or .status=="queued" or .status=="waiting" or .status=="pending" or .status=="requested")] | length')" \
  || die "Could not verify Actions idle state"
[[ "$ACTIVE" == 0 ]] || die "Runner busy with $ACTIVE queued/active workflows; no recreate"
project="$(docker inspect -f '{{index .Config.Labels "com.docker.compose.project"}}' "$C")"
service="$(docker inspect -f '{{index .Config.Labels "com.docker.compose.service"}}' "$C")"
configfiles="$(docker inspect -f '{{index .Config.Labels "com.docker.compose.project.config_files"}}' "$C")"
[[ -n "$project" && "$project" != "<no value>" ]] || die "Runner is not Compose-managed (no project label); do not guess recreation options"
[[ "$service" =~ ^[A-Za-z0-9_.-]+$ ]] || die "Invalid Compose service label: $service"
[[ -n "$configfiles" && "$configfiles" != "<no value>" ]] || die "Compose configuration file labels absent"
IFS=',' read -ra sourcefiles <<< "$configfiles"
(( "${#sourcefiles[@]}" > 0 )) || die "No source Compose files"
args=()
for file in "${sourcefiles[@]}"; do
  [[ -f "$file" ]] || die "Missing original Compose file: $file"
  args+=(-f "$file")
done
where="$(dirname -- "${sourcefiles[0]}")"
override="$where/ltx25-complete-image.override.yml"
[[ ! -e "$override" ]] || die "Existing override file $override; inspect manually, will not overwrite"
original_image="$(docker inspect -f '{{.Config.Image}}' "$C")"
current_id="$(docker inspect -f '{{.Image}}' "$C")"
target_id="$(docker image inspect --format '{{.Id}}' "$IMAGE")"
[[ "$current_id" != "$target_id" ]] || die "Complete model image already active; nothing to do"

echo "Project: $project | service: $service"
echo "Old: $original_image"
echo "New: $IMAGE"
echo "Compose files: $configfiles"
echo "Override to create: $override"
echo "Service definition will be reused with its original volumes, credentials, labels and GPUs."
docker compose -p "$project" "${args[@]}" config --services | grep -Fx "$service" >/dev/null \
  || die "Cannot find original Compose service $service"

if [[ "${LTX25_APPLY:-0}" != 1 ]]; then
  echo "DRY_RUN_COMPLETE; no container changed. Rerun with LTX25_APPLY=1 to recreate just this service."
  exit 0
fi

temp="$(mktemp "$where/.ltx25-override.XXXXXX")"
trap 'rm -f "$temp"' EXIT
printf 'services:\n  %s:\n    image: %s\n    pull_policy: never\n' "$service" "$IMAGE" > "$temp"
mv -n "$temp" "$override"
echo "Recreating only the idle GitHub runner service..."
docker compose -p "$project" "${args[@]}" -f "$override" \
  up -d --no-deps --force-recreate "$service"
new_id="$(docker inspect -f '{{.Image}}' "$C")"
[[ "$new_id" == "$target_id" ]] || die "Recreated container is not running the new image"
docker exec -i "$C" /opt/venvs/ltx25-cu128/bin/python - <<'PY'
from pathlib import Path
import json, mutagen, torch
root = Path("/opt/ltx25/checkpoints")
manifest = json.loads(Path("/opt/ltx25/model-image-manifest.json").read_text())
assert len(manifest["files"]) == 17
assert all((root/f["path"]).stat().st_size == f["bytes"] for f in manifest["files"])
assert torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (7,0)
print("COMPLETE_RUNNER_READY", torch.__version__, len(manifest["files"]), "baked assets")
PY
echo "Deployment complete. Ensure the self-hosted GitHub runner reconnects before dispatching the quality workflow."
echo "Rollback, if needed: put original image '$original_image' in '$override', then docker compose up -d --no-deps --force-recreate $service."
