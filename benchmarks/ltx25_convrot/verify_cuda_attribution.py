#!/usr/bin/env python3
"""CPU-side deterministic unit test for Chrome trace CUDA-kernel attribution."""
import json,tempfile
from pathlib import Path
from shared.kernels.ltx25_cuda_trace_attribution import summarize_chrome_trace

def main():
    ev=[
      {"name":"LTX25::quanto_qbytes_mm","cat":"user_annotation","ph":"X","pid":1,"tid":10,"ts":100,"dur":35},
      {"name":"aten::mm","cat":"cpu_op","ph":"X","pid":1,"tid":10,"ts":105,"dur":20,
       "args":{"Input type":["BFloat16","BFloat16"]}},
      {"name":"cudaLaunchKernel","cat":"cuda_runtime","ph":"X","pid":1,"tid":10,"ts":110,"dur":4,"args":{"correlation":1}},
      {"name":"aten::addmm","cat":"cpu_op","ph":"X","pid":1,"tid":10,"ts":200,"dur":30},
      {"name":"cudaLaunchKernel","cat":"cuda_runtime","ph":"X","pid":1,"tid":10,"ts":210,"dur":4,"args":{"correlation":2}},
      {"name":"magma_sgemm_cuda","cat":"kernel","ph":"X","pid":0,"tid":7,"ts":400,"dur":700,
       "args":{"correlation":1}},
      {"name":"cublasLt_HGEMM","cat":"kernel","ph":"X","pid":0,"tid":7,"ts":1200,"dur":900,
       "args":{"correlation":2}},
      {"name":"mystery_kernel","cat":"kernel","ph":"X","pid":0,"tid":7,"ts":2400,"dur":100,
       "args":{"correlation":777}},
      # Included CPU ranges must NOT be mistaken for actual CUDA kernels.
      {"name":"LTX25::quanto_qbytes_mm","cat":"cpu_op","ph":"X","pid":1,"tid":10,"ts":100,"dur":35,
       "args":{"External id":1}}
    ]
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/"trace.json"
        path.write_text(json.dumps({"traceEvents":ev}))
        report=summarize_chrome_trace(path)
    assert report["actual_gpu_kernel_events"]==3,report
    assert report["cuda_runtime_launch_events_inside_tagged_quanto"]==1,report
    assert report["cuda_runtime_launch_events_outside_tagged_quanto"]==1,report
    times=report["attribution_gpu_time_ms"]
    assert times=={"tagged_quanto":0.7,"outside_quanto":0.9,"unattributed":0.1},times
    ops={(x["kind"],x["operator"]):x["gpu_ms"] for x in report["gemm_kernel_attribution"]}
    assert ops[("tagged_quanto","aten::mm")]==0.7,ops
    assert ops[("outside_quanto","aten::addmm")]==0.9,ops
    print("LTX25_CUDA_CORRELATION_UNIT_PASS")

if __name__=="__main__":main()
