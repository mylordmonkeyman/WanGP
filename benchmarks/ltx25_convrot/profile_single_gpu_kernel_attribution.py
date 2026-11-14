#!/usr/bin/env python3
"""One controlled 49-frame T2V run with CUPTI GPU launches correlated to Quanto.

A single sampled transformer forward in Stage 1 and Stage 2 is exported
to Chrome trace and attributed by CPU runtime-launch correlation IDs.
Each stage report separates actual GPU kernels from CPU profiler ranges.
Does not report instrumented video wall time as production performance.
"""
from pathlib import Path
import json,os

import selective_fp16_t2v_smoke as base
import profile_gemm_t2v_i2v as matched

ROOT=matched.ROOT
OUT=Path(os.getenv("LTX25_CHROME_ATTRIBUTION_ROOT",
                  "/cache/ltx25/results/chrome-cuda-attribution-"+os.getenv("GITHUB_RUN_ID","manual"))).resolve()
base.OUT=OUT
base.EXTENSION=OUT/"libltx25_cublas_e_sm70.so"
matched.OUTPUT=OUT

original=matched.case_env
def attribute_env(case):
    env=original(case)
    env.update({"LTX25_PIPELINE_TRACE_ATTRIBUTION":"1",
                "LTX25_PIPELINE_TRACE_RETAIN":"0"})
    return env
matched.case_env=attribute_env

def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Output directory not empty: {OUT}")
    for name in ("videos","config","logs","results"):
        (OUT/name).mkdir(parents=True,exist_ok=True)
    base.gpu_free()
    base.command_run(["nvcc","-O3","-std=c++17","-arch=sm_70",
                      "-shared","-Xcompiler","-fPIC",
                      "benchmarks/ltx25_convrot/ltx25_cublas_e.cu",
                      "-lcublas","-o",str(base.EXTENSION)],
                     timeout=300, log=OUT/"logs"/"compile.log")
    (OUT/"input.json").write_text(json.dumps(matched.make_settings("t2v"),indent=2)+"\n")
    template=base.initialize_config()
    case=matched.run_case("t2v",template)
    p=OUT/"results"/"t2v-pipeline.json"
    report=json.loads(p.read_text())
    samples=report.get("chrome_trace_kernel_attribution",[])
    if len(samples)!=2 or {s["phase"] for s in samples}!={"stage1","stage2"}:
        raise RuntimeError(f"Missing both attributed sampled stages: {len(samples)} errors={report.get('errors')}")
    output={"status":"PASS","mode":"t2v","resolution":"1024x576","frames":49,
            "sampled_attributed_cuda_stages":samples,
            "pipeline_errors":report["errors"],
            "stage_quanto_full_run_calls":case["stage_quanto_calls_full_run"],
            "reference_caveat":"Single profiled forward; CUDA kernel durations are sums and not whole-video wall times"}
    (OUT/"cuda-kernel-attribution.json").write_text(json.dumps(output,indent=2)+"\n")
    print("LTX25_CUDA_REAL_KERNEL_ATTRIBUTION_PASS",
          json.dumps([{"stage":s["phase"],
                      "cuda_kernels":s["actual_gpu_kernel_events"],
                      "sum_gpu_ms":s["gpu_time_ms_sum"],
                      "scopes_gpu_ms":s["attribution_gpu_time_ms"],
                      "gemm_attribution":s["gemm_kernel_attribution"]}
                     for s in samples],sort_keys=True),flush=True)

if __name__=="__main__":main()
