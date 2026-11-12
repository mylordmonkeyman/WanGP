#!/usr/bin/env python3
"""Validate pre-baked V100 WanGP runtime, without pip or downloads.

Run once before the actual CLI dry-run. Never triggers a model generation,
touches checkpoint weights, or falls back to runtime dependency installation.
"""
from __future__ import annotations

import datetime as dt
import importlib
import json
import os
from pathlib import Path
import sys
import sysconfig
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("/cache/ltx25/results")
PROBES = [
    "mmgp.offload",
    "models.ltx2.ltx_core.model.transformer",
    "models.ltx2.ltx_pipelines.distilled",
    "models.ltx2.ltx2",
]

def main():
    sha = os.environ.get("GITHUB_SHA", "local")
    report = {
        "sha": sha,
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "profile": "prebaked-v100-sm70",
        "no_pip": True,
        "no_runtime_downloads": True,
        "model_weights_loaded": False,
        "steps": [],
        "probes": [],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    try:
        import torch, torchaudio, torchvision
        assert sys.version_info[:2] == (3,12), sys.version
        assert torch.__version__.startswith("2.10.0+cu128"), torch.__version__
        assert torchaudio.__version__.startswith("2.10.0+cu128"), torchaudio.__version__
        assert torchvision.__version__.startswith("0.25.0+cu128"), torchvision.__version__
        assert torch.cuda.is_available(), "No GPU visible"
        assert torch.cuda.get_device_capability() == (7,0), torch.cuda.get_device_capability()
        assert "sm_70" in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
        header = Path(sysconfig.get_path("include")) / "Python.h"
        assert header.is_file(), f"Missing development headers: {header}"
        report["torch_after"]={"returncode":0,"tail":f"SM70_TORCH_OK {torch.__version__} Python.h={header}"}
        for module in PROBES:
            item = {"name":module}
            try:
                importlib.import_module(module)
                item["status"]="PASS"
            except Exception as exc:
                item.update(status="ERROR", detail=f"{type(exc).__name__}: {exc}",
                            traceback_tail=traceback.format_exc()[-1000:])
            report["probes"].append(item)
            print("BAKED_IMPORT " + json.dumps(item)[:1600], flush=True)
        report["status"] = "PASS" if all(p["status"]=="PASS" for p in report["probes"]) else "BLOCKED"
    except Exception as exc:
        report["status"]="BLOCKED"
        report["error"]=f"{type(exc).__name__}: {exc}"
        report["torch_after"]={"returncode":1,"tail":report["error"]}
    finally:
        target=OUT / f"runtime_bootstrap_{sha[:12]}.json"
        target.write_text(json.dumps(report,indent=2)+"\n")
        print("BAKED_RUNTIME_STATUS", report["status"],flush=True)
        if report["status"]!="PASS":
            raise SystemExit(1)

if __name__=="__main__":
    main()
