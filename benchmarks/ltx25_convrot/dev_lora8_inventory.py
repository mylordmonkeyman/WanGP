#!/usr/bin/env python3
import json, os, pathlib, shutil
from safetensors import safe_open
roots = [pathlib.Path("/opt/ltx25/checkpoints"), pathlib.Path("/cache/ltx25")]
patterns = [
    "*dev*diffusion_model*bf16*.safetensors",
    "*dev*diffusion_model*int8_convrot*.safetensors",
    "*dev*transformer*bf16*.safetensors",
    "*dev*transformer*convrot*.safetensors",
    "*distilled*lora*450*.safetensors",
    "*distilled*diffusion_model*int8_convrot*.safetensors",
]
found={}
for pat in patterns:
    hits=[]
    for root in roots:
        if not root.is_dir(): continue
        for path in root.rglob(pat):
            if not path.is_file(): continue
            stat=path.stat()
            hits.append({"path":str(path),"GiB":round(stat.st_size/(1024**3),2)})
    found[pat]=hits
for root in roots:
    if root.is_dir():
        usage=shutil.disk_usage(root)
        print("STORAGE",str(root),{"free_GiB":round(usage.free/(1024**3),2),"total_GiB":round(usage.total/(1024**3),2)},flush=True)
print("DEV_LORA8_INVENTORY",json.dumps(found),flush=True)
for pat in patterns:
    for hit in found[pat]:
        if hit["GiB"]>1.0:
            path=hit["path"]
            try:
                with safe_open(path,framework="pt",device="cpu") as r:
                    metadata=r.metadata() or {}
                    print("CHECKPOINT_INFO",json.dumps({
                        "file":path,
                        "tensor_count":len(r.keys()),
                        "metadata_keys":sorted(metadata.keys()),
                        "quantization_format":metadata.get("quantization_format"),
                        "convrot_format":metadata.get("convrot_format"),
                        "example_tensor_keys":r.keys()[:6],
                    }),flush=True)
            except Exception as ex:
                print("CHECKPOINT_ERROR",path,type(ex).__name__,str(ex)[:250],flush=True)
