#!/usr/bin/env python3
"""One-time V100 matrix pilot for Candidate E large-M row tiling.

The 241-frame 1024x576 Stage 2 native qbytes_mm calls have M=17856,
N/K=(4096,4096) or (2048,4096). Candidate E currently rejects M>8192.
Use unchanged existing Candidate E kernel in bounded 4096/8192 row tiles.
Measure native Quanto vs transient FP16 vs Candidate E row tiling on
synthetic INT8 ConvRot weights and BF16 inputs, with RMS/finite error gates.

This does not modify WanGP model code, weights, selectors or defaults.
No persistent duplicate FP16 weight cache. Only bounded transient scratch.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import torch
import optimum.quanto.library.qbytes_mm  # register Quanto custom torch op

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.getenv("LTX25_LARGE_M_TILE_ROOT",
                "/cache/ltx25/results/large-m-candidate-e-"+os.getenv("GITHUB_RUN_ID","manual")))
PY=Path(sys.executable)
EXTENSION=OUT/"libltx25_cublas_e_sm70.so"

def compile_ext():
    cmd=["nvcc","-O3","-std=c++17","-arch=sm_70",
         "-shared","-Xcompiler","-fPIC",
         "benchmarks/ltx25_convrot/ltx25_cublas_e.cu",
         "-lcublas","-o",str(EXTENSION)]
    p=subprocess.run(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                     encoding="utf-8",timeout=300)
    (OUT/"compile.log").write_text(p.stdout)
    if p.returncode or not EXTENSION.is_file():
        raise RuntimeError(f"Candidate E build failed: {p.stdout[-2500:]}")
    print("LARGE_M_CANDIDATEE_COMPILED",EXTENSION,flush=True)

def validate_env():
    assert torch.cuda.is_available() and torch.cuda.get_device_capability(0)==(7,0)
    assert os.environ.get("WAN2GP_LTX25_CUBLAS_E_MODE")=="fp32rot"
    assert os.environ.get("WAN2GP_LTX25_HADAMARD")=="radix4"
    assert os.environ.get("WAN2GP_LTX25_E_CAST")=="vector8"
    assert os.environ.get("WAN2GP_LTX25_ROT_INPUT")=="direct"
    assert os.environ.get("WAN2GP_LTX25_E_OUTPUT")=="fused"
    assert os.environ.get("LTX25_CUBLAS_E_LIB")==str(EXTENSION)

def event_time(fn,reps):
    torch.cuda.synchronize()
    start=torch.cuda.Event(enable_timing=True)
    stop=torch.cuda.Event(enable_timing=True)
    st=time.perf_counter()
    start.record()
    for _ in range(reps):
        out=fn()
        if not torch.is_tensor(out):
            raise RuntimeError(f"Non Tensor output {type(out)}")
    stop.record()
    stop.synchronize()
    elapsed_gpu_ms=start.elapsed_time(stop)/reps
    elapsed_wall_s=(time.perf_counter()-st)/reps
    return elapsed_gpu_ms,elapsed_wall_s

def check_fidelity(ref,result):
    a=ref.float()
    b=result.float()
    diff=a-b
    rmse=float(diff.square().mean().sqrt().item())
    denom=float(a.square().mean().sqrt().item())
    nrmse=rmse/max(denom,1e-10)
    max_abs=float(diff.abs().max().item())
    nonfinite=int((~torch.isfinite(result)).sum().item())
    return {"normalized_rms":round(nrmse,8),
            "max_abs_error":round(max_abs,6),
            "nonfinite":nonfinite,
            "bf16_output":result.dtype==torch.bfloat16}

def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("Refusing existing output files in "+str(OUT))
    OUT.mkdir(parents=True,exist_ok=True)
    validate_env()
    compile_ext()
    from shared.kernels import ltx25_cublas_sm70 as candidate
    from shared.qtypes.int8_convrot import _rotate_activation

    torch.manual_seed(20261112)
    torch.backends.cuda.matmul.allow_tf32=False
    rows=[]
    cases=[(17856,4096,4096),(17856,2048,4096)]
    for M,N,K in cases:
        torch.cuda.empty_cache()
        print("LARGE_M_SHAPE",M,N,K,flush=True)
        # Synthetic, bounded scale ensures no BF16/FP16 overflow.
        w=torch.randint(-16,17,(N,K),device="cuda",dtype=torch.int8)
        scale=(torch.rand((N,1),device="cuda",dtype=torch.float32)*.008+.002).contiguous()
        bias=(torch.randn((N,),device="cuda",dtype=torch.float32)*.01).bfloat16()
        x=(torch.randn((M,K),device="cuda",dtype=torch.float32)*.08).bfloat16()
        scale_bf16=scale.bfloat16()
        weight=SimpleNamespace(_data=w,_scale=scale)
        print("LARGE_M_MEMORY_GIB",round(torch.cuda.memory_allocated()/2**30,3),flush=True)

        def native():
            rot=_rotate_activation(x,256)
            return (torch.ops.quanto.qbytes_mm(rot,w,scale_bf16)+bias).bfloat16()

        def direct_fp16():
            rot=_rotate_activation(x,256)
            dequant=scale_bf16*w
            out=torch.mm(rot.half(),dequant.half().T)
            return (out.bfloat16()+bias).bfloat16()

        def tiled_candidate(tile_rows):
            chunks=[]
            for offset in range(0,M,tile_rows):
                result=candidate.maybe_linear(x[offset:offset+tile_rows],weight,bias,256)
                if result is None:
                    raise RuntimeError(f"Candidate E tile rejected: {(M,N,K,tile_rows,offset)}")
                chunks.append(result)
            return torch.cat(chunks,dim=0)

        funcs={"native_bf16_quanto":native,
               "direct_fp16_quanto_path":direct_fp16,
               "candidate_e_tile4096":lambda:tiled_candidate(4096),
               "candidate_e_tile8192":lambda:tiled_candidate(8192)}
        # Warm kernels/BLAS caches, then benchmark multiple samples in interleaved
        # and reversed order. We report GPU elapsed AND synced wall per generation.
        for name,fn in funcs.items():
            fn()
            torch.cuda.synchronize()
        reference=native()
        torch.cuda.synchronize()
        for name,fn in funcs.items():
            result=fn()
            torch.cuda.synchronize()
            fidelity=check_fidelity(reference,result)
            if fidelity["nonfinite"] or not fidelity["bf16_output"]:
                raise RuntimeError(f"Invalid output {name} {fidelity}")
            samples=[]
            for _ in range(3):
                gpu_ms,wall_s=event_time(fn,1)
                samples.append({"gpu_ms":round(gpu_ms,4),
                                "wall_seconds":round(wall_s,6)})
            row={
                "M":M,"N":N,"K":K,"variant":name,
                "median_gpu_ms":round(sorted(x["gpu_ms"] for x in samples)[1],4),
                "median_wall_seconds":round(sorted(x["wall_seconds"] for x in samples)[1],6),
                "samples":samples,
                **fidelity,
            }
            rows.append(row)
            print("LARGE_M_TILE_RESULT "+json.dumps(row,sort_keys=True),flush=True)
        del w,scale,scale_bf16,bias,x,weight,reference
        torch.cuda.empty_cache()

    base_by_shape={(r["M"],r["N"],r["K"]):r for r in rows
                   if r["variant"]=="native_bf16_quanto"}
    for row in rows:
        base=base_by_shape[(row["M"],row["N"],row["K"])]
        row["speedup_gpu_over_native"]=round(base["median_gpu_ms"]/max(row["median_gpu_ms"],1e-9),4)
        row["speedup_wall_over_native"]=round(base["median_wall_seconds"]/max(row["median_wall_seconds"],1e-9),4)
    result={
        "status":"PASS","torch_version":torch.__version__,
        "gpu":torch.cuda.get_device_name(0),
        "source":"Synthetic input and quantized weights; shapes are real 241-frame I2V Stage2 native Quanto qbytes_mm observations",
        "candidate_e_tile_rows":[4096,8192],
        "candidate_e_dtype":"BF16 input/output; int8 weights; transient FP16 Tensor Core GEMM and FP32 accum/epilogue",
        "numerical_safety":"per-shape full output comparison with native BF16 for synthetic tensors; no guarantee for live model",
        "limitations":[
            "Synthetic weights/activations only, NOT live model quality",
            "No model-side wiring or persistent weight copies",
            "Includes CUDA kernel completion and synchronized per-call wall time, but excludes full model/MMGP offload",
            "On full model, Candidate E FP32 rotation changes arithmetic from native BF16",
            "Repeat with real I2V on matched seed/input and evaluate faces/audio before enabling.",
        ],
        "rows":rows,
    }
    (OUT/"tile-pilot.json").write_text(json.dumps(result,indent=2)+"\n")
    with (OUT/"tile-pilot.csv").open("w",newline="") as f:
        fields=["M","N","K","variant","median_gpu_ms","median_wall_seconds",
                "speedup_gpu_over_native","speedup_wall_over_native",
                "normalized_rms","max_abs_error","nonfinite","bf16_output"]
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader();writer.writerows(rows)
    print("LTX25_LARGE_M_TILE_PILOT_PASS "+json.dumps({
        "rows":len(rows),
        "best_speedup":max(r["speedup_wall_over_native"] for r in rows),
        "max_nonfinite":max(r["nonfinite"] for r in rows)}),flush=True)

if __name__=="__main__":
    main()
