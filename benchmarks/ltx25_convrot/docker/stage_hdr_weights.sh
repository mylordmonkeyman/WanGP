#!/usr/bin/env bash
# Stage the additional files needed by WanGP's offline LTX-2.5 baseline.
# No image rebuild, pip installs, workflow dispatch, or GPU tests.
set -Eeuo pipefail
CONTAINER="${LTX25_RUNNER_CONTAINER:-gha-v100-ltx25}"
PY=/opt/venvs/ltx25-cu128/bin/python
docker exec -i "$CONTAINER" "$PY" - <<'PY'
import hashlib
from pathlib import Path
import shutil
from huggingface_hub import HfApi, hf_hub_download

REPO="DeepBeepMeep/LTX-2"
REV="0e281e1132633c9ff66e1891c045c533e785ce67"
NAMES=(
    "ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors",
    "ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors",
)
DEST=Path("/cache/ltx25/checkpoints")
DEST.mkdir(parents=True,exist_ok=True)
index={s.rfilename:s for s in HfApi().model_info(REPO,revision=REV,files_metadata=True).siblings}
total=0
for name in NAMES:
    item=index.get(name)
    if item is None: raise RuntimeError(f"Absent in pinned revision: {name}")
    if not item.lfs or not item.lfs.get("sha256"):
        raise RuntimeError(f"No official LFS SHA256 for {name}")
    total+=item.size or item.lfs["size"]
if total>8*1024**3: raise RuntimeError("8 GiB unexpected download size; stopping")
if shutil.disk_usage(DEST).free < total+2*1024**3:
    raise RuntimeError("Not enough disk for expected HDR files")
for name in NAMES:
    item=index[name]
    size=item.size or item.lfs["size"]
    expected_sha=item.lfs["sha256"]
    path=DEST/name
    if path.is_symlink(): raise RuntimeError(f"Refusing symlink: {path}")
    if not path.is_file() or path.stat().st_size!=size:
        print(f"Downloading {name} ({size:,} bytes)",flush=True)
        result=Path(hf_hub_download(repo_id=REPO,filename=name,revision=REV,local_dir=str(DEST)))
        if result!=path: raise RuntimeError(f"Wrong destination: {result}")
    if path.stat().st_size!=size: raise RuntimeError(f"Wrong file size: {name}")
    hasher=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(16*1024*1024),b""):hasher.update(chunk)
    if hasher.hexdigest()!=expected_sha: raise RuntimeError(f"SHA256 mismatch: {name}")
    print("LTX25_HDR_VERIFIED",name,flush=True)
PY
echo "Both pinned HDR support files verified in persistent checkpoint storage."
