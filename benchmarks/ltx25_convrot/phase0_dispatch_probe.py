#!/usr/bin/env python3
"""Isolated QLinearInt8ConvRot.forward dispatch check (not an MMGP pipeline run).

Use real saved INT8 data, FP32 scales, synthetic BF16 inputs and an explicit
minimal container for the forward() attributes. Checks that this specific
SM70 process takes the same numerical route as the existing R0 leaf. Does NOT
test real module hooks/offload, graph capture, or input shape distributions.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors import safe_open

from phase0_probe import github_source_rotation
from phase0_weights import FILE, OUT, tested_shapes, try_actual_qweight

def main():
    sha = os.environ.get("GITHUB_SHA", "local")
    report = {"source_sha": sha,
              "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "kind": "reconstructed ConvRot forward dispatcher on real checkpoint weights",
              "input": "synthetic BF16 M=16",
              "actual_model_hooks": False, "full_pipeline": False,
              "layers": []}
    try:
        from shared.qtypes.int8_convrot import QLinearInt8ConvRot, _QINT8_CONVROT_QTYPE
        from shared.kernels import int8_backend
        report["initial_int8_backend_name"] = str(getattr(int8_backend, "_backend", "unknown"))
        rotate = github_source_rotation()
        with safe_open(str(FILE), framework="pt", device="cpu") as f:
            for base, qkey, skey in tested_shapes(f):
                q = f.get_tensor(qkey).to("cuda")
                s = f.get_tensor(skey).to("cuda", dtype=torch.float32).reshape(-1)
                if s.numel() == 1:
                    s = s.expand(q.shape[0])
                w, err = try_actual_qweight(q, s[:, None].contiguous())
                if w is None:
                    report["layers"].append({"layer": base, "status": "ERROR", "reason": str(err)[:300]})
                    continue
                torch.manual_seed(777)
                x = torch.randn(16, q.shape[1], device="cuda", dtype=torch.float32).bfloat16()
                container = SimpleNamespace(qweight=w, weight_qtype=_QINT8_CONVROT_QTYPE,
                                            _convrot_group_size=256, bias=None)
                try:
                    actual = QLinearInt8ConvRot.forward(container, x)
                    original_leaf = F.linear(rotate(x, 256), w)
                    torch.cuda.synchronize()
                    diff = (actual.float() - original_leaf.float()).abs()
                    item = {
                        "layer": base, "N": q.shape[0], "K": q.shape[1],
                        "status": "PASS" if torch.equal(actual, original_leaf) else "MISMATCH",
                        "dispatcher_vs_leaf_max_abs": float(diff.max().item()),
                        "output_dtype": str(actual.dtype),
                        "shape": list(actual.shape),
                    }
                except Exception as e:
                    item = {"layer": base, "status": "ERROR",
                            "exception": type(e).__name__, "error": str(e)[:400]}
                report["layers"].append(item)
                print("CONVROT_DISPATCH " + json.dumps(item), flush=True)
                del q, s, w, x
        report["status"] = "PASS" if all(x["status"] == "PASS" for x in report["layers"]) else "PARTIAL"
    except Exception as e:
        report.update(status="ERROR", error_type=type(e).__name__, error=str(e)[:700])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"phase0_dispatch_{sha[:12]}.json").write_text(json.dumps(report, indent=2) + "\n")
    print("DISPATCH_STATUS " + report["status"], flush=True)
    if report["status"] == "ERROR":
        raise SystemExit(1)

if __name__ == "__main__":
    main()
