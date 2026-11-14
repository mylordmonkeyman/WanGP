#!/usr/bin/env python3
"""Long 241-frame I2V vs existing 49-frame matched-profile GEMM attribution.

Profiles one warmed forward in each stage using native BF16 Quanto sham.
One control output at 241 frames, 1024x576, no LoRA, same bar prompt as 49.
Exposes CPU GEMM shape census and actual CUDA kernel correlation to Quanto.
No production model changes; profiler timings are not wall-performance timings.
"""
import json,os
from pathlib import Path
import selective_fp16_t2v_smoke as base
import profile_gemm_t2v_i2v as matched

OUT=Path(os.getenv("LTX25_LONG_GEMM_PROFILE_ROOT",
                  "/cache/ltx25/results/long-i2v-gemms-"+os.getenv("GITHUB_RUN_ID","manual"))).resolve()
base.OUT=OUT
base.EXTENSION=OUT/"libltx25_cublas_e_sm70.so"
base.FRAMES=241
matched.OUTPUT=OUT
matched.FRAMES=241

original=matched.case_env
def traced_env(case):
    env=original(case)
    env.update({"LTX25_PIPELINE_TRACE_ATTRIBUTION":"1",
                "LTX25_PIPELINE_TRACE_RETAIN":"0"})
    return env
matched.case_env=traced_env

def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Output root must be empty: {OUT}")
    for name in ("videos","config","logs","results"):
        (OUT/name).mkdir(parents=True,exist_ok=True)
    matched.check_png()
    base.gpu_free()
    base.command_run(["nvcc","-O3","-std=c++17","-arch=sm_70",
                      "-shared","-Xcompiler","-fPIC",
                      "benchmarks/ltx25_convrot/ltx25_cublas_e.cu",
                      "-lcublas","-o",str(base.EXTENSION)],
                     timeout=300,log=OUT/"logs"/"compile.log")
    (OUT/"input.json").write_text(json.dumps(matched.make_settings("t2v"),indent=2)+"\n")
    template=base.initialize_config()
    row=matched.run_case("i2v",template)
    q=json.loads((OUT/"results/i2v-quanto.json").read_text())
    p=json.loads((OUT/"results/i2v-pipeline.json").read_text())
    traces=p.get("chrome_trace_kernel_attribution",[])
    cpu=p.get("gemm_census_samples",[])
    if len(traces)!=2 or len(cpu)!=2:
        raise RuntimeError(f"Missing stage samples; cpu={len(cpu)} gpu={len(traces)} errors={p.get('errors')}")
    counts=row["stage_quanto_calls_full_run"]
    if counts!={"stage1":384,"stage2":1440}:
        raise RuntimeError(f"Frame-count reversal not reproduced in 241-frame sample: {counts}")
    all_shapes={}
    for s in cpu:
        all_shapes[s["phase"]]=s["top_cpu_matmul_shapes_outside_quanto"]
    output={
        "status":"PASS","mode":"i2v","frames":241,"resolution":"1024x576",
        "stage_quanto_full_run_counts":counts,
        "cpu_non_quanto_matmul_shapes_one_forward":all_shapes,
        "stage_cuda_attribution":traces,
        "pipeline_errors":p.get("errors",[]),
        "note":"Profiler timing is sampled GPU kernel duration, not end-to-end speed.",
    }
    (OUT/"long-vs-short-gemm-profile.json").write_text(json.dumps(output,indent=2)+"\n")
    print("LTX25_LONG_I2V_CUDA_GEMM_PROFILE_PASS "+json.dumps({
        "stage_quanto":counts,
        "cpu_outside_quanto":[
            {"stage":s["phase"],
             "outside_quanto":s["cpu_matmul_outside_tagged_quanto"],
             "top_shapes":s["top_cpu_matmul_shapes_outside_quanto"][:6]}
            for s in cpu],
        "cuda_gemm_attribution":[
            {"phase":t["phase"],"gemm_kernel_attribution":t["gemm_kernel_attribution"],
             "gpu_kernel_ms_sum":t["gpu_time_ms_sum"]}
            for t in traces]
    },sort_keys=True),flush=True)

if __name__=="__main__":main()
