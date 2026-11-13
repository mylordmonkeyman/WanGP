// Isolated SM70 cuBLAS legacy algorithm benchmark for existing Candidate E.
// No production integration; no weights retained or modified, no host sync.
// Exactly mirrors ltx25_cublas_gemm_f16_f32's transpose/shape/accumulation.
// selector=-1: existing CUBLAS_GEMM_DEFAULT_TENSOR_OP reference.
// selector=-2: CUBLAS_GEMM_DEFAULT control.
// selector 0..23: CUBLAS_GEMM_ALGO0_TENSOR_OP + selector.
// Unsupported selectors return the cuBLAS status; never silently fallback.
#include <cuda_runtime.h>
#include <cublas_v2.h>

extern "C" int ltx25_cublas_gemm_algo_probe(
    const void* q_rowmajor, const void* x_rowmajor, void* c_rowmajor,
    int n, int m, int k, int selector, void* stream_ptr) {
  if (!q_rowmajor || !x_rowmajor || !c_rowmajor ||
      n <= 0 || m <= 0 || k <= 0) return -1;
  if (selector < -2 || selector > 23) return -2;
  static thread_local cublasHandle_t handle = nullptr;
  if (!handle) {
    const auto rc=cublasCreate(&handle);
    if (rc!=CUBLAS_STATUS_SUCCESS) return 1000+int(rc);
  }
  auto stream=reinterpret_cast<cudaStream_t>(stream_ptr);
  auto st=cublasSetStream(handle,stream);
  if(st!=CUBLAS_STATUS_SUCCESS) return 1100+int(st);
  cublasGemmAlgo_t algorithm=selector==-1
    ? CUBLAS_GEMM_DEFAULT_TENSOR_OP
    : selector==-2
      ? CUBLAS_GEMM_DEFAULT
      : static_cast<cublasGemmAlgo_t>(int(CUBLAS_GEMM_ALGO0_TENSOR_OP)+selector);
  const float alpha=1.0f,beta=0.0f;
  st=cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,
      n,m,k,&alpha,
      q_rowmajor,CUDA_R_16F,k,
      x_rowmajor,CUDA_R_16F,k,
      &beta,c_rowmajor,CUDA_R_32F,n,
      CUBLAS_COMPUTE_32F,algorithm);
  return st==CUBLAS_STATUS_SUCCESS ? 0 : 1200+int(st);
}
