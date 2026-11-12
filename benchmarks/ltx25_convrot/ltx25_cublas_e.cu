// LTX-2.5 Phase 1 exploratory Candidate E: row-major FP16xFP16 -> FP32 cuBLAS.
// All weight casting and scratch allocation are caller-owned and bounded.
// Input Q[N,K] is *unscaled* FP16 converted exactly from stored INT8.
// Input X[M,K] is FP16 (rotated BF16 converted only on safe test inputs).
// Output C[M,N] is FP32. Row scales/bias are applied by Python in FP32.
// Not yet a WanGP runtime backend, graph-safe path, or production kernel.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

extern "C" int ltx25_cublas_gemm_f16_f32(
    const void* q_rowmajor, const void* x_rowmajor, void* c_rowmajor,
    int n, int m, int k, void* stream_ptr) {
  if (!q_rowmajor || !x_rowmajor || !c_rowmajor || n <= 0 || m <= 0 || k <= 0)
    return -1;
  // Each Python benchmark uses one thread and one active stream. Keep the
  // handle across calls to avoid timing handle creation with layer latency.
  static thread_local cublasHandle_t handle = nullptr;
  if (!handle) {
    cublasStatus_t create = cublasCreate(&handle);
    if (create != CUBLAS_STATUS_SUCCESS) return 1000 + int(create);
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  cublasStatus_t set = cublasSetStream(handle, stream);
  if (set != CUBLAS_STATUS_SUCCESS) return 1100 + int(set);

  const float alpha = 1.0f, beta = 0.0f;
  // Col-major C_col[N,M] = op(Q_col[K,N])^T[N,K] * X_col[K,M].
  // This writes the exact same bytes as row-major C[M,N].
  cublasStatus_t status = cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N,
      n, m, k,
      &alpha,
      q_rowmajor, CUDA_R_16F, k,
      x_rowmajor, CUDA_R_16F, k,
      &beta,
      c_rowmajor, CUDA_R_32F, n,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return status == CUBLAS_STATUS_SUCCESS ? 0 : 1200 + int(status);
}
