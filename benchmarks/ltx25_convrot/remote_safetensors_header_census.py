#!/usr/bin/env python3
"""Range-read safetensors headers remotely; never download multi-GB tensors."""
import json,os,struct,urllib.request,collections
from pathlib import Path
FNS={
 "dev":"ltx-2.5-22b-dev_diffusion_model_int8_convrot.safetensors",
 "distilled":"ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors",
}
PASS={(2048,2048),(4096,4096),(2048,4096),(4096,16384),
      (16384,4096),(2048,8192),(8192,2048)}
TESTED=PASS | {(4096,2048)}

def get_header(fn):
    u="https://huggingface.co/DeepBeepMeep/LTX-2/resolve/main/"+fn
    request=urllib.request.Request(u,headers={"Range":"bytes=0-8388607","User-Agent":"WanGP-checkpoint-header-audit/1"})
    with urllib.request.urlopen(request, timeout=180) as f:
        head=f.read(8)
        if len(head)!=8:raise RuntimeError("Cannot read file header")
        size=struct.unpack("<Q",head)[0]
        if size>8*1024*1024:raise RuntimeError(f"Large safetensors header length {size}")
        data=f.read(size)
        if len(data)!=size:raise RuntimeError(f"Short safetensors header: {len(data)}/{size}")
        return json.loads(data),size

def inventory(fn):
    data,nbytes=get_header(fn)
    categories=collections.Counter()
    shapes=collections.defaultdict(lambda:{"count":0,"samples":[]})
    for name,t in data.items():
        if not name.endswith(".weight") or not isinstance(t,dict):continue
        shape=t["shape"]
        if len(shape)!=2 or t["dtype"]!="I8":
            categories["not_2d_int8"]+=1;continue
        base=name[:-7]
        scale=data.get(base+".weight_scale")
        if not isinstance(scale,dict) or scale["dtype"]!="F32":
            categories["missing_fp32_scale"]+=1;continue
        N,K=shape
        if K%256:
            categories["other_rotation_group"]+=1;continue
        row=shapes[(N,K)]
        row["count"]+=1
        if len(row["samples"])<2:row["samples"].append(name)
    results=[]
    for (N,K),v in sorted(shapes.items(),key=lambda kv:(-kv[1]["count"],kv[0])):
        verdict="allowed" if (N,K) in PASS else "rejected_earlier" if (N,K) in TESTED else "not_tested"
        row={"N":N,"K":K,"layers":v["count"],"verdict":verdict,
             "candidate_weight_cap":N*K*2<=128*2**20,"samples":v["samples"]}
        results.append(row)
        print("HEADER_WEIGHT_FAMILY",fn,json.dumps({k:row[k] for k in ("N","K","layers","verdict","candidate_weight_cap")}),flush=True)
    return {"filename":fn,"header_bytes":nbytes,"metadata":data.get("__metadata__",{}),
            "families":results,"not_candidates":dict(categories)}

def main():
    output={label:inventory(fn) for label,fn in FNS.items()}
    p=Path("checkpoint_shape_header_inventory.json")
    p.write_text(json.dumps(output,indent=2)+"\n")
    print("HEADER_WEIGHT_SHAPE_PASS",json.dumps({k:{"families":len(v["families"]),
          "untested":sum(r["verdict"]=="not_tested" for r in v["families"]),
          "newly_eligible":sum(r["verdict"]=="not_tested" and r["candidate_weight_cap"] for r in v["families"])}
          for k,v in output.items()}),flush=True)
if __name__=="__main__":main()
