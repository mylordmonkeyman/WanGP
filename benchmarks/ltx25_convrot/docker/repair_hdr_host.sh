#!/usr/bin/env bash
# Manual HOST-ONLY repair. Never called by a GitHub Actions workflow.
set -Eeuo pipefail
D="${LTX25_HOST_CHECKPOINT_DIR:-/var/lib/llama/gha-runners/ltx25/cache/ltx25/checkpoints}"
C="${LTX25_RUNNER_CONTAINER:-gha-v100-ltx25}"
F=ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors
SHA=0b39eac09672c18d0d10f401d5e71a68a73225f1169df826b092b7e5ef489675
SIZE=1308787472
REV=0e281e1132633c9ff66e1891c045c533e785ce67
URL="https://huggingface.co/DeepBeepMeep/LTX-2/resolve/$REV/$F"
[[ -d "$D" && -w "$D" ]] || { echo "Not writable: $D" >&2; exit 1; }
[[ ! -L "$D" && ! -L "$D/$F" ]] || { echo "Refusing symlink" >&2; exit 1; }
check() { [[ -f "$1" && ! -L "$1" ]] && [[ "$(stat -c %s "$1")" == "$SIZE" ]] && printf "%s  %s\n" "$SHA" "$1" | sha256sum -c --status; }
if [[ -e "$D/$F" ]]; then
  check "$D/$F" && { echo "HDR_HOST_VERIFIED $D/$F"; exit 0; }
  echo "Existing host file is unreadable or invalid; refusing overwrite: $D/$F" >&2; exit 1
fi
T="$(mktemp "$D/.ltx25-hdr.XXXXXXXX.part")"
trap 'rm -f "$T"' EXIT
if docker exec "$C" test -f "/cache/ltx25/checkpoints/$F" 2>/dev/null; then
  echo "Copying previously verified file from runner; no download"
  docker cp "$C:/cache/ltx25/checkpoints/$F" "$T"
else
  echo "File absent from container too; downloading once on HOST (outside tests): $URL"
  command -v curl >/dev/null || { echo "curl missing" >&2; exit 1; }
  AVAIL="$(df -PB1 "$D" | awk 'END {print $4}')"
  (( AVAIL > SIZE + 2*1024*1024*1024 )) || { echo "Disk full" >&2; exit 1; }
  curl --fail --location --retry 3 --connect-timeout 30 --output "$T" "$URL"
fi
check "$T" || { echo "Pinned size or SHA256 mismatch: $T" >&2; exit 1; }
chmod 0644 "$T"
[[ ! -e "$D/$F" ]] || { echo "Destination appeared unexpectedly" >&2; exit 1; }
mv -T "$T" "$D/$F"
echo "HDR_HOST_VERIFIED $D/$F ($SIZE bytes); safe to retry image build"
