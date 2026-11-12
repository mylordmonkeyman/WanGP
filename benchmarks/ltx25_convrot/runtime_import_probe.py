#!/usr/bin/env python3
"""Lightweight LTX-2.5 runtime importability probe, with no inference/model load.

Separately tests the actual model constructor, pipeline and offload modules.
Records missing prerequisites for incremental isolated-vEnv installation.
Never substitutes "import succeeded" for a successful full-model execution.
"""
import datetime as dt
import importlib
import json
import os
from pathlib import Path
import traceback
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path("/cache/ltx25/results")
MODULES = [
    "mmgp.offload",
    "models.ltx2.ltx_core.model.transformer",
    "models.ltx2.ltx_pipelines.distilled",
    "models.ltx2.ltx2",
]

def main():
    report = {"source_sha": os.environ.get("GITHUB_SHA", "local"),
              "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "source": "WanGP import-only diagnostics", "weights_loaded": False,
              "full_pipeline_measured": False, "modules": []}
    for name in MODULES:
        item = {"name": name}
        try:
            importlib.import_module(name)
            item["status"] = "PASS"
        except Exception as err:
            item.update(status="ERROR", exception=type(err).__name__,
                        error=str(err)[:500],
                        traceback_tail=traceback.format_exc().strip().splitlines()[-8:])
        report["modules"].append(item)
        print("RUNTIME_IMPORT " + json.dumps(item), flush=True)
    report["status"] = "PASS" if all(x["status"] == "PASS" for x in report["modules"]) else "BLOCKED"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"runtime_import_{report['source_sha'][:12]}.json").write_text(json.dumps(report, indent=2) + "\n")
    print("RUNTIME_IMPORT_PROBE " + report["status"], flush=True)

if __name__ == "__main__":
    main()
