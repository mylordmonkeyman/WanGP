#!/usr/bin/env python3
"""Header-only inventory of INT8 ConvRot weight shapes in on-runner LTX-2.5.

Never loads weights into RAM/VRAM. Contrasts actual 22B Dev and Distilled
checkpoint (N,K) families with the eight historical checkpoint-backed tests
and the seven-shape Candidate E allowlist.
"""
from collections import Counter
import json
import os
from pathlib import Path

from safetensors import safe_open

FILES = {
    "dev": Path("/cache/ltx25/models/ltx-2.5-22b-dev_diffusion_model_int8_convrot.safetensors"),
    "distilled": Path("/opt/ltx25/checkpoints/ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors"),
}
ALLOWED={(2048,2048),(4096,4096),(2048,4096),(4096,16384),
         (16384,4096),(2048,8192),(8192,2048)}
PREVIOUSLY_TESTED=ALLOWED | {(4096,2048)}
WEIGHT_BYTES_CAP=128*1024*1024
OUT=Path(os.getenv("LTX25_SHAPE_INVENTORY_OUT","/cache/ltx25/results/candidate-e-untested-weight-shapes.json"))


def collect(label,path):
    assert path.is_file(), f"checkpoint missing: {path}"
    tally=Counter()
    instances={}
    irrelevant={"not_i8":0, "missing_fp32_scale":0,
                "weight_not_2d":0, "not_rotation_group_256":0}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys=set(f.keys())
        metadata=f.metadata() or {}
        for key in keys:
            if not key.endswith(".weight"):
                continue
            tensor=f.get_slice(key)
            shape=tensor.get_shape()
            if len(shape)!=2:
                irrelevant["weight_not_2d"]+=1;continue
            if tensor.get_dtype()!="I8":
                irrelevant["not_i8"]+=1;continue
            n,k=map(int,shape)
            scale_key=key[:-7]+".weight_scale"
            if scale_key not in keys or f.get_slice(scale_key).get_dtype()!="F32":
                irrelevant["missing_fp32_scale"]+=1;continue
            if k%256:
                irrelevant["not_rotation_group_256"]+=1;continue
            tally[(n,k)]+=1
            instances.setdefault((n,k),[])
            if len(instances[(n,k)])<4:instances[(n,k)].append(key)
    families=[]
    for (n,k),count in sorted(tally.items(),key=lambda pair:(-pair[1],pair[0])):
        eligible_by_basic_shape=(n*k*2 <= WEIGHT_BYTES_CAP)
        category=("allowlisted" if (n,k) in ALLOWED else
                  "previously_tested_failed" if (n,k) in PREVIOUSLY_TESTED else
                  "untested")
        families.append({
            "N":n,"K":k,"count_layers":count,
            "classification":category,
            "group256":True,"under_128mib_transient_weight_cap":eligible_by_basic_shape,
            "potential_new_candidate_e_family":category=="untested" and eligible_by_basic_shape,
            "sample_layers":instances[(n,k)]
        })
    return {
        "checkpoint":str(path),"file_bytes":path.stat().st_size,
        "metadata":{x:metadata.get(x) for x in ("quantization_format","convrot_format")},
        "total_int8_group256_fp32scale_layers":sum(tally.values()),
        "unique_int8_group256_fp32scale_shapes":len(tally),
        "previously_tested_shapes_present":len(set(tally)&PREVIOUSLY_TESTED),
        "already_allowlisted":len(set(tally)&ALLOWED),
        "new_unchecked_families":sum(x["potential_new_candidate_e_family"] for x in families),
        "new_unchecked_layers":sum(x["count_layers"] for x in families if x["potential_new_candidate_e_family"]),
        "excluded_or_unknown":irrelevant,
        "families":families,
    }


def main():
    results={label:collect(label,path) for label,path in FILES.items()}
    data={
        "purpose":"actual Dev and Distilled INT8 ConvRot weight-family census",
        "tested_historical_nk":sorted(map(list,PREVIOUSLY_TESTED)),
        "allowed_nk":sorted(map(list,ALLOWED)),
        "weight_cap_bytes":WEIGHT_BYTES_CAP,
        "row_count_note":"Checkpoint weight dimensions alone do not establish if a shape is hit at a given M/step/stage.",
        "candidate_limitations":[
            "Uses matching FP32 scale and group-size inferred from K%256, not live runtime module inspection.",
            "Classifies new families as potential only; numerical, VRAM, kernel and video gates still required.",
            "No checkpoint weights loaded into VRAM or copied; safetensors headers only.",
        ],
        "models":results,
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(data,indent=2)+"\n")
    for name,res in results.items():
        print("CANDIDATE_E_WEIGHT_CENSUS_MODEL",name,json.dumps({
            key:res[key] for key in (
                "total_int8_group256_fp32scale_layers",
                "unique_int8_group256_fp32scale_shapes",
                "previously_tested_shapes_present",
                "new_unchecked_families",
                "new_unchecked_layers",
                "excluded_or_unknown"
            )},sort_keys=True),flush=True)
        for row in res["families"]:
            print("CANDIDATE_E_WEIGHT_FAMILY",name,json.dumps({
                key:row[key] for key in (
                    "N","K","count_layers","classification",
                    "under_128mib_transient_weight_cap",
                    "potential_new_candidate_e_family"
                )
            },sort_keys=True),flush=True)
    print("CANDIDATE_E_WEIGHT_SHAPE_INVENTORY_PASS",OUT,flush=True)


if __name__=="__main__":
    main()
