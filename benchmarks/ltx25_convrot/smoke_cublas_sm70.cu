// LTX-2.5 ConvRot Phase 0: verify SM70 compilation and FP32-output cuBLAS GEMM.
// This is a functionality smoke test, NOT a performance benchmark.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    return 1; \
  } \
} while (0)
#define CHECK_CUBLAS(call) do { \
  cublasStatus_t s = (call); \
  if (s != CUBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "cuBLAS error %s:%d: %d\\n", __FILE__, __LINE__, int(s)); \
    return 1; \
  } \
} while (0)

int main() {
  int count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&count));
  if (count != 1) {
    std::fprintf(stderr, "Expected exactly one visible GPU, got %d\\n", count);
    return 1;
  }
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s, SM %d%d\\n", prop.name, prop.major, prop.minor);
  if (prop.major != 7 || prop.minor != 0) return 1;

  constexpr int N = 16;
  __half A[N*N], B[N*N];
  for (int col = 0; col < N; ++col) {
    for (int row = 0; row < N; ++row) {
      A[row + col*N] = __float2half(row == col ? 1.0f : 0.0f);
      B[row + col*N] = __float2half(float(row + 2*col) / 16.0f);
    }
  }
  __half *devA = nullptr, *devB = nullptr;
  float *devC = nullptr;
  CHECK_CUDA(cudaMalloc((void**)&devA, sizeof(A)));
  CHECK_CUDA(cudaMalloc((void**)&devB, sizeof(B)));
  CHECK_CUDA(cudaMalloc((void**)&devC, sizeof(float)*N*N));
  CHECK_CUDA(cudaMemcpy(devA, A, sizeof(A), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(devB, B, sizeof(B), cudaMemcpyHostToDevice));
  cublasHandle_t handle{};
  CHECK_CUBLAS(cublasCreate(&handle));
  const float alpha = 1.0f, beta = 0.0f;
  CHECK_CUBLAS(cublasGemmEx(
    handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
    &alpha, devA, CUDA_R_16F, N, devB, CUDA_R_16F, N,
    &beta, devC, CUDA_R_32F, N,
    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  CHECK_CUDA(cudaDeviceSynchronize());
  float C[N*N]{};
  CHECK_CUDA(cudaMemcpy(C, devC, sizeof(C), cudaMemcpyDeviceToHost));
  for (int i = 0; i < N*N; ++i) {
    float expected = __half2float(B[i]); // Identity * B
    if (!std::isfinite(C[i]) || std::fabs(C[i] - expected) > 1e-6f) {
      std::fprintf(stderr, "Mismatch at %d: got %.9g, expected %.9g\\n", i, C[i], expected);
      return 1;
    }
  }
  CHECK_CUBLAS(cublasDestroy(handle));
  CHECK_CUDA(cudaFree(devA));
  CHECK_CUDA(cudaFree(devB));
  CHECK_CUDA(cudaFree(devC));
  std::puts("PASS: SM70 cuBLAS FP16 operands, FP32 compute and FP32 C");
  return 0;
}
