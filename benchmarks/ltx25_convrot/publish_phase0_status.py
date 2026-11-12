#!/usr/bin/env python3
"""Publish a compact, nonsecret Phase 0 run summary to a separate results branch.

Uses the ephemeral GitHub Actions token. No API key is persisted to disk.
The results branch does not trigger the benchmark workflow.
"""
from __future__ import annotations

import base64
import datetime as dt
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request

RESULTS_BRANCH = "perf/ltx25-run-results"
REPOSITORY = "mylordmonkeyman/WanGP"


def request(path: str, method: str = "GET", data: dict | None = None):
    token = os.environ["GH_TOKEN"]
    url = f"https://api.github.com/repos/{REPOSITORY}/{path}"
    body = None if data is None else json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, method=method, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
        "User-Agent": "WanGP-LTX25-V100-Phase0",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as err:
        if err.code == 404 and method == "GET":
            return None
        raise RuntimeError(f"GitHub API HTTP {err.code}: {err.read().decode('utf-8')[:400]}") from None


def put(path: str, doc: dict):
    existing = request("contents/" + path + "?ref=" + urllib.parse.quote(RESULTS_BRANCH, safe=""))
    encoded = base64.b64encode((json.dumps(doc, indent=2, sort_keys=True) + "\n").encode()).decode()
    payload = {
        "message": f"ci(ltx25): report Phase 0 run {doc['run_id']} [skip ci]",
        "branch": RESULTS_BRANCH,
        "content": encoded,
    }
    if existing and "sha" in existing:
        payload["sha"] = existing["sha"]
    request("contents/" + path, "PUT", payload)


def main():
    sha = os.environ["GITHUB_SHA"]
    run_id = os.environ["GITHUB_RUN_ID"]
    probe_file = Path(f"/cache/ltx25/results/phase0_probe_{sha[:12]}.json")
    report = json.loads(probe_file.read_text()) if probe_file.exists() else {}
    remote_file = Path(f"/cache/ltx25/results/phase0_remote_header_{sha[:12]}.json")
    remote = json.loads(remote_file.read_text()) if remote_file.exists() else {}
    remote_inv = remote.get("inventory", {})
    import_path = Path(f"/cache/ltx25/results/runtime_import_{sha[:12]}.json")
    import_report = json.loads(import_path.read_text()) if import_path.exists() else {}
    manifest_file = Path(f"/cache/ltx25/results/support_manifest_{sha[:12]}.json")
    manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}
    e_path = Path(f"/cache/ltx25/results/phase1_cublas_e_{sha[:12]}.json")
    e_report = json.loads(e_path.read_text()) if e_path.exists() else {}
    e_results = e_report.get("results", [])
    weight_path = Path(f"/cache/ltx25/results/phase0_weights_{sha[:12]}.json")
    weights = json.loads(weight_path.read_text()) if weight_path.exists() else {}
    weights_layers = weights.get("layers", [])
    rotations = report.get("rotation", [])
    checkpoint = report.get("checkpoint", {})
    summary = {
        "source_sha": sha,
        "run_id": run_id,
        "run_url": f"https://github.com/{REPOSITORY}/actions/runs/{run_id}",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "job_status": os.environ.get("JOB_STATUS", "unknown"),
        "gpu": report.get("gpu", {}).get("name"),
        "capability": report.get("gpu", {}).get("capability"),
        "torch": report.get("torch"),
        "cuda": report.get("torch_cuda"),
        "phase0_probe_present": bool(report),
        "checkpoint_visible": checkpoint.get("present"),
        "runtime_import": {
            "status": import_report.get("status", "not_run"),
            "weights_loaded": False,
            "modules": [{"name": x["name"], "status": x["status"],
                         "error": x.get("error", "")[:300],
                         "traceback_tail": x.get("traceback_tail", [])}
                        for x in import_report.get("modules", [])]
        },
        "support_manifest": {
            "status": manifest.get("status", "not_run"),
            "error": manifest.get("error", "")[:400],
            "remaining_bytes": manifest.get("total_remaining_bytes"),
            "free_disk_bytes": manifest.get("free_disk_bytes"),
            "missing_upstream": manifest.get("missing_upstream", []),
            "files": [{
                "path": f["path"], "exists_upstream": f.get("exists_upstream"),
                "size_bytes": f.get("size_bytes"), "local_size_matches": f.get("local_size_matches"),
                "sha256": f.get("sha256"),
            } for f in manifest.get("files", [])]
        },
        "cublas_e": {
            "status": e_report.get("status", "not_run"),
            "error": e_report.get("error", "")[:400],
            "FP32_output": e_report.get("FP32_output", False),
            "synthetic_activations_only": True,
            "production_eligible": False,
            "shape_families": len(e_results),
            "results": [{
                "N": layer.get("N"), "K": layer.get("K"),
                "status": layer.get("status"),
                "error": layer.get("error", "")[:300],
                "tests": [{
                    "M": t["M"], "gate": t["precision_gate_pass"],
                    "R0_p99": t["R0"]["p99_normalized"],
                    "R0_max": t["R0"]["max_normalized"],
                    "E_p99": t["E"]["p99_normalized"],
                    "E_max": t["E"]["max_normalized"],
                    "fp32rot_gate": t.get("fp32rot_precision_gate_pass"),
                    "fp32rot_p99": t.get("E_fp32rot", {}).get("p99_normalized"),
                    "fp32rot_max": t.get("E_fp32rot", {}).get("max_normalized"),
                    "fp32rot_median_ms": t.get("E_fp32rot_full_layer_median_ms"),
                    "fp32rot_speedup": t.get("speedup_ratio_R0_over_fp32rot"),
                    "R0_median_ms": t["R0_full_layer_median_ms"],
                    "E_median_ms": t["E_full_layer_median_ms"],
                    "speedup": t["speedup_ratio_R0_over_E"],
                    "fp16_subnormal_count": t["range"]["fp16_subnormal_count"]
                } for t in layer.get("tests", [])]
            } for layer in e_results]
        },
        "weight_diagnostic": {
            "status": weights.get("status", "not_run"),
            "error": weights.get("error", "")[:400],
            "real_checkpoint_weights": weights.get("real_checkpoint_weights", False),
            "real_model_activations": False,
            "actual_fallback_dispatch_confirmed": False,
            "selected_matrix_families": weights.get("selected_matrix_families"),
            "layers_passed": sum(x.get("status") == "PASS" for x in weights_layers),
            "layers": [{
                "N": layer["N"], "K": layer["K"], "name": layer["layer"],
                "status": layer["status"],
                "real_wrapper_init": layer.get("real_wrapper_init"),
                "real_wrapper_error": layer.get("real_wrapper_error"),
                "tests": [{
                    "M": t["M"],
                    "R0_leaf_status": t.get("R0_leaf", {}).get("status", "not_run"),
                    "R0_leaf_error": t.get("R0_leaf", {}).get("message", "")[:300],
                    "R0_p99_normalized": t.get("R0_leaf", {}).get("vs_R1_fp32", {}).get("p99_normalized"),
                    "R0_max_normalized": t.get("R0_leaf", {}).get("vs_R1_fp32", {}).get("max_normalized"),
                    "R0_median_ms": t.get("R0_leaf", {}).get("timing", {}).get("median_ms"),
                    "surrogate_median_ms": t.get("bf16_dense_surrogate", {}).get("timing", {}).get("median_ms"),
                } for t in layer.get("tests", [])]
            } for layer in weights_layers]
        },
        "remote_header_status": remote.get("status", "not_run"),
        "remote_header_error": remote.get("error", "")[:300],
        "remote_hf_revision": remote.get("hf_revision"),
        "remote_published_lfs_sha256": remote.get("published_lfs_sha256"),
        "remote_int8_matrix_count": remote_inv.get("int8_matrix_count"),
        "remote_group_candidates": remote_inv.get("shape_group_candidates"),
        "remote_int8_missing_fp32_scale": remote_inv.get("int8_missing_fp32_scale"),
        "remote_header_top_shapes": remote_inv.get("matrix_shape_frequencies", [])[:12],
        "checkpoint_metadata_verified": checkpoint.get("checkpoint_metadata_verified", False),
        "checkpoint_layer_count": checkpoint.get("int8_2d_weight_count"),
        "checkpoint_blocker": checkpoint.get("blocker", "")[:300],
        "checkpoint_inventory_error": checkpoint.get("inventory_error", "")[:300],
        "rotation_test_count": len(rotations),
        "rotation_pass_count": sum(x["status"] == "PASS" for x in rotations),
        "rotation_results": [{
            "M": r["M"], "K": r["K"], "group_size": r["group_size"],
            "status": r["status"], "median_ms": r.get("median_ms"),
            "p99_abs": r.get("p99_abs"), "max_abs": r.get("max_abs"),
            "error": r.get("error", "")[:300]
        } for r in rotations],
        "actual_model_shapes_observed": False,
        "full_pipeline_measured": False,
        "note": "Synthetic diagnostic only; not an LTX-2.5 model benchmark or production accuracy gate."
    }
    put("ltx25-phase0/latest.json", summary)
    put(f"ltx25-phase0/runs/{run_id}.json", summary)
    print(f"Published sanitized Phase 0 report for run {run_id} on {RESULTS_BRANCH}", flush=True)


if __name__ == "__main__":
    main()
