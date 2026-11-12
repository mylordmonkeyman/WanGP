#!/usr/bin/env python3
"""Plan supporting LTX-2.5 model assets with Hugging Face metadata only.

Does not download, convert, or execute weights. Uses the same frozen Hub
revision as the verified INT8 transformer; records missing files/sizes before
initiating an expensive independent download job.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import shutil

from huggingface_hub import HfApi

REPO = "DeepBeepMeep/LTX-2"
REV = "0e281e1132633c9ff66e1891c045c533e785ce67"
BASE = Path("/cache/ltx25/checkpoints")
OUT = Path("/cache/ltx25/results")
TRANSFORMER = "ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors"
ASSETS = [
    TRANSFORMER,
    "ltx-2.5-22b_video_vae_bf16.safetensors",
    "ltx-2.5-22b_audio_vae_bf16.safetensors",
    "ltx-2.5-22b_vocoder_bf16.safetensors",
    "ltx-2.5-22b_text_embedding_projection_bf16.safetensors",
    "ltx-2.5-22b_video_embeddings_connector_int8_convrot.safetensors",
    "ltx-2.5-22b_audio_embeddings_connector_int8_convrot.safetensors",
    "ltx-2.5-spatial-upscaler-x2-1.0_bf16.safetensors",
    # WanGP query_model_files() requires both upscalers and the diffusion VAE.
    "ltx-2.5-temporal-upscaler-x2-1.0_bf16.safetensors",
    "ltx-2.5-22b_diffusion_video_vae_bf16.safetensors",
    "ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors",
    "ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors",
    "gemma4-12b-ltx-v1/gemma4-12b-ltx-v1_int8_convrot.safetensors",
    "gemma4-12b-ltx-v1/config.json",
    "gemma4-12b-ltx-v1/chat_template.jinja",
    "gemma4-12b-ltx-v1/tokenizer.json",
    "gemma4-12b-ltx-v1/tokenizer_config.json",
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"source_sha": os.environ.get("GITHUB_SHA", "local"),
              "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "repo": REPO, "revision": REV, "downloads_started": False,
              "goal": "LTX-2.5 22B distilled INT8 ConvRot with default BF16 VAE and INT8 Gemma4"}
    try:
        info = HfApi().model_info(REPO, revision=REV, files_metadata=True)
        by_name = {x.rfilename: x for x in info.siblings}
        inventory = []
        total_missing = 0
        missing = []
        for name in ASSETS:
            s = by_name.get(name)
            if s is None:
                missing.append(name)
                inventory.append({"path": name, "exists_upstream": False})
                continue
            size = s.size or (s.lfs or {}).get("size")
            expected = (s.lfs or {}).get("sha256")
            dest = BASE / name
            installed = dest.is_file() and (size is None or dest.stat().st_size == size)
            inventory.append({"path": name, "exists_upstream": True, "size_bytes": size,
                              "sha256": expected, "local_size_matches": installed})
            if not installed and isinstance(size, int):
                total_missing += size
        report.update(status="PASS" if not missing else "MISSING_UPSTREAM",
                      files=inventory, missing_upstream=missing,
                      total_remaining_bytes=total_missing,
                      free_disk_bytes=shutil.disk_usage(BASE).free,
                      note="Local size match is NOT SHA256 verification. Downloader must hash every completed file.")
    except Exception as e:
        report.update(status="ERROR", error_type=type(e).__name__, error=str(e)[:800])
    path = OUT / f"support_manifest_{report['source_sha'][:12]}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("SUPPORT_MANIFEST " + json.dumps(report)[:20000], flush=True)
    print(f"SUPPORT_MANIFEST_PATH={path}", flush=True)


if __name__ == "__main__":
    main()
