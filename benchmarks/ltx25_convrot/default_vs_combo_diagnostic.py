#!/usr/bin/env python3
"""CPU-only compare the two isolated V100 benchmark configurations.

Only emits key names, SHA256 digests, and values from a small public-safe
allowlist. Never publishes arbitrary config values, secrets, raw prompts,
tokenizer contents, weights, process args, or paths under a private home.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re

BASE = Path("/cache/ltx25")
OUT = BASE / "results"
RUNS = {
    "known_good_combo": "0d76d647c79c",
    "default_auto": "d305bfd5885f",
}
CONF = {
    "known_good_combo": BASE / "config-e_combo_1024" / "wgp_config.json",
    "default_auto": BASE / "config-e_default_1024" / "wgp_config.json",
}
SAFE_KEYS = {
    "model_type", "last_model_type", "transformer_types",
    "transformer_quantization", "text_encoder_quantization",
    "int8_kernels", "fp8_kernels", "profile", "video_profile",
    "attention_mode", "compile", "seed", "num_inference_steps",
    "video_length", "resolution", "enable_lora", "lora_enabled",
    "no_lora", "attention", "fps", "vae_profile", "offload_profile",
}
PATTERNS = (
    "denoising first phase", "denoising second phase",
    "lora", "adapters", "model type", "steps", "guidance",
    "quantization", "profile", "offload",
    "loading model", "task 1 completed",
)

def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()[:16]

def load(path):
    if not path.exists():
        return {"__file_missing__": str(path)}
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        return {"__not_dict__": type(obj).__name__}
    return obj

def compare(a,b):
    keys = sorted(set(a)|set(b))
    changes = []
    for k in keys:
        if k in a and k in b and digest(a[k]) == digest(b[k]):
            continue
        entry = {"key": str(k)[:120], "combo_hash": digest(a.get(k,"<MISSING>")),
                 "default_hash": digest(b.get(k,"<MISSING>")),
                 "combo_missing": k not in a, "default_missing": k not in b}
        if k in SAFE_KEYS:
            for label,data in (("combo_value",a),("default_value",b)):
                v=data.get(k)
                if v is None or isinstance(v, (str,int,float,bool)):
                    entry[label] = str(v)[:110]
                elif isinstance(v,list) and len(v)<8 and all(isinstance(x,(str,int,float,bool)) for x in v):
                    entry[label] = v
        changes.append(entry)
    return {"combo_key_count":len(a),"default_key_count":len(b),
            "different_key_count":len(changes),"changed_keys":changes}

def signal_lines(label,sha):
    file=OUT/f"baseline_generation_{sha}.log"
    if not file.is_file():
        return {"file_exists":False}
    lines=file.read_text(errors="replace").splitlines()
    # Only write public-safe pattern counts, never raw private log lines.
    filtered = {}
    for pattern in PATTERNS:
        filtered[pattern] = sum(pattern in line.lower() for line in lines)
    return {"file_exists":True,"log_lines":len(lines),"patterns":filtered,
            "log_sha256":hashlib.sha256(file.read_bytes()).hexdigest()[:16]}

def main():
    ca=load(CONF["known_good_combo"]);cb=load(CONF["default_auto"])
    sa=load(BASE/"e_combo_1024_49frame_settings.json")
    sb=load(BASE/"e_default_1024_49frame_settings.json")
    report={
        "kind":"CPU-only V100 benchmark config difference (hashed values only)",
        "run_id":os.environ.get("GITHUB_RUN_ID"),
        "source_sha":os.environ.get("GITHUB_SHA"),
        "status":"PASS",
        "config":compare(ca,cb),
        "settings":compare(sa,sb),
        "log_patterns":{name:signal_lines(name,sha) for name,sha in RUNS.items()},
    }
    file=OUT/f"default_vs_combo_{os.environ.get('GITHUB_SHA','local')[:12]}.json"
    file.write_text(json.dumps(report,indent=2)+"\n")
    print("LTX25_CONFIG_DIAGNOSTIC "+json.dumps(report)[:11000],flush=True)

if __name__=="__main__":main()
