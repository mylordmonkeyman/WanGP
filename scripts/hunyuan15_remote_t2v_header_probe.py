#!/usr/bin/env python3
"""Compare Hunyuan 1.5 480p T2V Comfy FP16 and WanGP BF16 headers without weight downloads."""
import json
from collections import Counter
import hunyuan15_remote_header_probe as probe
from hunyuan15_checkpoint_audit import compare, expand_fused_qkv_metadata

URLS = {
    "comfy_480p_t2v_fp16":
        "https://huggingface.co/Comfy-Org/Hunyuan-Video-1.5/resolve/main/"
        "split_files/diffusion_models/hunyuanvideo1.5_480p_t2v_fp16.safetensors",
    "wangp_480p_t2v_bf16":
        "https://huggingface.co/DeepBeepMeep/HunyuanVideo1.5/resolve/main/"
        "hunyuan_video_1.5_480_bf16.safetensors",
    "tencent_480p_t2v_fp32":
        "https://huggingface.co/tencent/HunyuanVideo-1.5/resolve/main/"
        "transformer/480p_t2v/diffusion_pytorch_model.safetensors",
}

def main():
    result = {}
    tensors = {}
    for label, url in URLS.items():
        try:
            items, header_len = probe.header_from_url(url)
            tensors[label] = items
            result[label] = {
                "header_bytes": header_len,
                "tensor_count": len(items),
                "dtype_counts": dict(Counter(t["dtype"] for t in items.values())),
            }
        except Exception as exc:
            result[label] = {"error": str(exc)}
    if "wangp_480p_t2v_bf16" in tensors:
        reference = tensors["wangp_480p_t2v_bf16"]
        for label in ("comfy_480p_t2v_fp16", "tencent_480p_t2v_fp32"):
            if label in tensors:
                catalog = tensors[label]
                if label.startswith("comfy"):
                    catalog, unfused = expand_fused_qkv_metadata(catalog)
                    result[label]["fused_qkv_unpacked"] = unfused
                result[label]["versus_wangp"] = compare(catalog, reference)
    print(json.dumps(result, indent=2))
    return int(any("error" in x for x in result.values()))

if __name__ == "__main__":
    raise SystemExit(main())
