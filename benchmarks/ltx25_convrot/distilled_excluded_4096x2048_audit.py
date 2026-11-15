#!/usr/bin/env python3
"""Checkpoint-backed (4096,2048) ConvRot numerical audit, V100 only.

Comparison uses 3 *different real checkpoint layers* and five M shapes.
Native WanGP R0 BF16, candidate-E FP32rot/radix4, and Stage2 direct FP16
are measured independently against an FP32 rotated INT8+FP32-scale reference.
No full FP16 weight set is retained between operations. This is a research
gate, not automatic approval of a previously excluded model layer.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors import safe_open
from shared.kernels import ltx25_cublas_sm70 as e
from shared.kernels import ltx25_selective_fp16_gemm as selector
from shared.qtypes.int8_convrot import _rotate_activation
from phase0_weights import try_actual_qweight

CKPT=Path("/opt/ltx25/checkpoints/ltx-2.5-22b-distilled_diffusion_model_int8_convrot.safetensors")
OUT=Path(os.environ.get("LTX25_EXCLUDED_SHAPE_OUT","/cache/ltx25/results/excluded-shape-audit.json"))
SHAPE=(4096,2048)
MS=(18,51,1008,4032,5760)


def require(p,msg):
    if not p:raise RuntimeError(msg)


def stats(output,ref):
    got=output.float()
    delta=got-ref.float()
    abs_diff=delta.abs()
    denom=ref.float().abs().clamp_min(1.0)
    normalized=(abs_diff/denom).reshape(-1)
    # PyTorch quantile has an upper input element limit on large M x N.
    # Use identical deterministic strided samples for all three methods,
    # retaining the exact global maximum over every element.
    stride=max(1, (normalized.numel()+2_097_151)//2_097_152)
    return {
        "normalized_rms":float((delta.square().mean().sqrt()/ref.float().square().mean().sqrt().clamp_min(1e-12)).item()),
        "p99_normalized":float(torch.quantile(normalized[::stride],.99).item()),
        "p99_sample_stride":stride,
        "max_normalized":float(normalized.max().item()),
        "max_abs":float(abs_diff.max().item()),
        "nonfinite":int((~torch.isfinite(got)).sum().item()),
    }


def ms_time(fn):
    for _ in range(2):fn()
    torch.cuda.synchronize()
    results=[]
    for _ in range(3):
        a=torch.cuda.Event(enable_timing=True)
        b=torch.cuda.Event(enable_timing=True)
        a.record(); out=fn(); b.record(); b.synchronize()
        results.append(float(a.elapsed_time(b)))
        del out
    return statistics.median(results)


def main():
    require(torch.cuda.is_available() and torch.cuda.get_device_capability(0)==(7,0),"V100 required")
    require(CKPT.is_file(),"checkpoint unavailable")
    require(e.MODE=="fp32rot" and e.ROT_MODE=="radix4" and e.CAST_MODE=="vector8",
            "unexpected Candidate E implementation")
    require(e.MIN_M==64 and e.MAX_M==8192,"source defaults not unmodified")
    require(os.getenv("WAN2GP_LTX25_E_TEST_4096X2048")=="1"
            and SHAPE in e.PASS_SHAPES,"excluded shape research opt-in not active")
    require(selector.ENABLED and selector.BACKEND=="direct"
            and "qbytes_mm" in selector.OPS,"FP16 competitor not configured")
    # Load tensor *slices*, not the entire 19GB model. Three independent
    # checkpoint layers from first/middle/last of the excluded family.
    with safe_open(str(CKPT),framework="pt",device="cpu") as f:
        allkeys=set(f.keys())
        available=sorted(key for key in allkeys
                         if key.endswith(".weight")
                         and f.get_slice(key).get_shape()==list(SHAPE)
                         and f.get_slice(key).get_dtype()=="I8"
                         and key[:-7]+".weight_scale" in allkeys)
        require(len(available)>=3,"Expected multiple checkpoint-backed layers")
        keys=[available[0],available[len(available)//2],available[-1]]
        result={"status":"RUNNING","shape":{"N":4096,"K":2048},
                "checkpoint":str(CKPT),"num_shape_layers":len(available),
                "sampled_checkpoint_layers":keys,"examples":[],
                "baseline":"WanGP BF16 ConvRot R0","test":"Current Candidate E radix4 + direct Stage2 FP16",
                "historical_gate":"Candidate p99/max normalized error <= native BF16 R0 against FP32 reference",
                "remarks":"Synthetic activations, real checkpoint weights/scales; no live activation distribution measured"}
        for layer_idx,key in enumerate(keys):
            q=f.get_tensor(key).to(device="cuda",dtype=torch.int8).contiguous()
            sc=f.get_tensor(key[:-7]+".weight_scale").to(device="cuda",dtype=torch.float32).reshape(-1,1).contiguous()
            require(sc.numel()==SHAPE[0],"bad scale")
            qweight,problem=try_actual_qweight(q,sc)
            require(qweight is not None,f"Actual Quanto weight unsupported: {problem}")
            weight=SimpleNamespace(_data=q,_scale=sc)
            scale16=sc.to(dtype=torch.bfloat16)
            rows=[]
            for m in MS:
                torch.manual_seed(20261115+layer_idx*199+m)
                x=(torch.randn((m,2048),device="cuda",dtype=torch.float32)*.4).to(torch.bfloat16)
                # A tiny number of range-relevant values; model-quality stats
                # must include both ordinary and high-variance activations.
                x[0,:].zero_()
                x[min(m-1,1),:].fill_(1/32)
                def native():
                    return F.linear(_rotate_activation(x,256),qweight)
                def fp16():
                    rotated=_rotate_activation(x,256)
                    return selector._SelectiveMode("stage2").__torch_dispatch__(
                        torch.ops.quanto.qbytes_mm.default,(),(rotated,q,scale16),{})
                def candidate():
                    return e.maybe_linear(x,weight,None,256)
                with torch.inference_mode():
                    y_r0=native().clone()
                    y_fp16=fp16().clone()
                    # bypass row-count cutoff for numerical assessment only
                    e.MIN_M=1
                    try:
                        y_e=candidate().clone()
                    finally:
                        e.MIN_M=64
                    require(y_e is not None,"Candidate E returned None for "+str(m))
                    torch.cuda.synchronize()
                    ref=(_rotate_activation(x.float(),256) @ q.float().T)*sc.T
                    err_r0=stats(y_r0,ref)
                    err_e=stats(y_e,ref)
                    err_fp16=stats(y_fp16,ref)
                    versus_r0=stats(y_e,y_r0.float())
                    versus_fp16=stats(y_e,y_fp16.float())
                    gate=(err_e["nonfinite"]==0
                          and err_e["p99_normalized"]<=err_r0["p99_normalized"]+1e-8
                          and err_e["max_normalized"]<=err_r0["max_normalized"]+1e-8)
                    row={"checkpoint_layer":key,"M":m,
                         "native_r0_vs_fp32":err_r0,
                         "candidate_e_vs_fp32":err_e,
                         "stage2_fp16_vs_fp32":err_fp16,
                         "candidate_e_vs_native_r0":versus_r0,
                         "candidate_e_vs_stage2_fp16":versus_fp16,
                         "passes_historical_error_gate":gate,
                         "fp16_candidate_safety":{"output_nonfinite":err_e["nonfinite"]},
                        }
                    if layer_idx==0 and m in (18,51,4032,5760):
                        e.MIN_M=1
                        try:
                            # interleaved ABBA pair for two *complete*
                            # ConvRot operations with the same q and scale.
                            a1=ms_time(candidate)
                            b1=ms_time(fp16)
                            b2=ms_time(fp16)
                            a2=ms_time(candidate)
                        finally:
                            e.MIN_M=64
                        row["candidate_e_gpu_ms"]=(a1+a2)/2
                        row["stage2_fp16_gpu_ms"]=(b1+b2)/2
                        row["speedup_candidate_vs_fp16"]=((b1+b2)/(a1+a2))
                    rows.append(row)
                    print("EXCLUDED_SHAPE_SAMPLE",json.dumps({
                        "index":layer_idx,"M":m,"historical_precision_gate":gate,
                        "r0_nrmse":round(err_r0["normalized_rms"],6),
                        "e_nrmse":round(err_e["normalized_rms"],6),
                        "fp16_nrmse":round(err_fp16["normalized_rms"],6),
                        "e_vs_r0_nrmse":round(versus_r0["normalized_rms"],6),
                        "e_vs_fp16_nrmse":round(versus_fp16["normalized_rms"],6),
                        "speedup_ratio":row.get("speedup_candidate_vs_fp16"),
                        "candidate_nonfinite":err_e["nonfinite"]
                    }),flush=True)
                    del x,y_r0,y_fp16,y_e,ref
            result["examples"].extend(rows)
            del q,sc,qweight,weight,scale16
            torch.cuda.empty_cache()
            OUT.parent.mkdir(parents=True,exist_ok=True)
            OUT.write_text(json.dumps(result,indent=2)+"\n")

    total=len(result["examples"])
    failed=[(x["checkpoint_layer"],x["M"]) for x in result["examples"]
            if not x["passes_historical_error_gate"]]
    result["historical_gate_results"]={"passed":total-len(failed),"total":total,
                                       "failed_cases":failed}
    result["all_candidate_outputs_finite"]=all(
        x["candidate_e_vs_fp32"]["nonfinite"]==0 for x in result["examples"])
    result["status"]="PASS" if result["all_candidate_outputs_finite"] else "UNSAFE_NONFINITE"
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    print("LTX25_EXCLUDED_CHECKPOINT_AUDIT_FINISHED",json.dumps({
        "status":result["status"],"total":total,
        "historical_precision_gates_passed":total-len(failed),
        "failed":failed,"file":str(OUT)
    }),flush=True)
    require(result["all_candidate_outputs_finite"],"non-finite Candidate E output")


if __name__=="__main__":
    main()
