#!/usr/bin/env python3
"""Small *GPU* smoke test for one compiled external FlashAttention-V100 package.

Called exclusively from the explicitly authorized compilation workflow.
No checkpoint downloads or video generation.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["1cat", "zhinianqin"], required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()

    src = args.source.resolve()
    if not src.is_dir():
        raise RuntimeError(f"Extension source directory is missing: {src}")
    sys.path.insert(0, str(src))

    assert torch.cuda.is_available(), "CUDA is not available on the selected runner"
    assert torch.cuda.get_device_capability() == (7, 0), "Runner must be sm_70"
    assert torch.version.cuda is not None, "Requires CUDA-enabled PyTorch"

    if args.backend == "1cat":
        from flash_attn_v100 import flash_attn_func as op
    else:
        from vllm_flash_attn import flash_attn_varlen_func as op

    results = []
    with torch.inference_mode():
        for dim in (64, 128):
            torch.manual_seed(725 + dim)
            # Test representative small noncausal FP16 shapes, no padded masks.
            b, tokens, heads = 1, 256, 8
            q = torch.randn(b, tokens, heads, dim, device="cuda", dtype=torch.float16)
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            if args.backend == "1cat":
                got = op(q, k, v, dropout_p=0.0, causal=False)
            else:
                cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
                got = op(
                    q=q.reshape(-1, heads, dim),
                    k=k.reshape(-1, heads, dim),
                    v=v.reshape(-1, heads, dim),
                    cu_seqlens_q=cu,
                    cu_seqlens_k=cu,
                    max_seqlen_q=tokens,
                    max_seqlen_k=tokens,
                    dropout_p=0.0,
                    causal=False,
                ).reshape(b, tokens, heads, dim)
            torch.cuda.synchronize()
            ref = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                is_causal=False,
            ).transpose(1, 2).contiguous()
            if got.shape != ref.shape:
                raise AssertionError(f"unexpected output shape {got.shape} vs {ref.shape}")
            # Criterion is deliberately tolerant for a first compilation smoke,
            # not a production precision gate.
            err = (got.float() - ref.float()).abs()
            max_error = err.max().item()
            if not torch.isfinite(got).all() or max_error > 0.05:
                raise AssertionError(f"{args.backend} D{dim}: non-finite or max abs error={max_error}")
            results.append({"backend": args.backend, "head_dim": dim, "max_abs_error": max_error})

    print(json.dumps({
        "result": "PASS",
        "backend": args.backend,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "tests": results,
    }, indent=2))


if __name__ == "__main__":
    main()
