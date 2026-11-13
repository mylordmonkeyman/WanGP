// LTX-2.5 Candidate E, SM70: FP16xFP16 -> FP32 cuBLAS with GPU-only
// activation range protection and an FP32 same-operation fallback per row.
// No host/device synchronization, dynamic allocation or global state in these
// entry points. The caller owns live scratch until all queued kernels finish.
// Kept separate from the other WanGP INT8 backends.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cfloat>
#include <cmath>

namespace {

constexpr int THREADS = 256;

// A block owns one input row. A power-of-two rescale preserves magnitude
// exactly in FP32 and ensures *all nonzero* FP16 operands are normal, when
// possible. An ill-conditioned/nonfinite row is sent through FP32 instead.
// This kernel never round-trips a device flag or an activation to the host.
__global__ void prepare_rows(const float* __restrict__ input,
                             __half* __restrict__ output,
                             int* __restrict__ exponents,
                             int* __restrict__ slow_rows,
                             unsigned int* __restrict__ slow_count,
                             int m, int k) {
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  if (row >= m) return;
  __shared__ float largest[THREADS], smallest[THREADS];
  __shared__ int invalid[THREADS], exponent_shared, slow_shared;
  float hi = 0.0f, lo = FLT_MAX;
  int bad = 0;
  const size_t start = static_cast<size_t>(row) * k;
  for (int i = lane; i < k; i += THREADS) {
    const float v = input[start + i];
    if (!isfinite(v)) {
      bad = 1;
    } else if (v != 0.0f) {
      const float a = fabsf(v);
      hi = fmaxf(hi, a);
      lo = fminf(lo, a);
    }
  }
  largest[lane] = hi;
  smallest[lane] = lo;
  invalid[lane] = bad;
  __syncthreads();
  for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
    if (lane < stride) {
      largest[lane] = fmaxf(largest[lane], largest[lane + stride]);
      smallest[lane] = fminf(smallest[lane], smallest[lane + stride]);
      invalid[lane] |= invalid[lane + stride];
    }
    __syncthreads();
  }
  if (lane == 0) {
    int exp2 = 0;
    int slow = invalid[0];
    if (!slow && largest[0] > 32768.0f) {
      // Large-magnitude FP16 operands can magnify cancellation error on
      // unscale. Preserve FP32 arithmetic for this entire row instead.
      slow = 1;
    }
    if (!slow && largest[0] > 0.0f) {
      // max*2^exp2 < 2^15 and min*2^exp2 >= 2^-14.
      // Keep the cast away from FP16 overflow AND subnormals/FTZ.
      const int upper = 14 - ilogbf(largest[0]);
      const int lower = -14 - ilogbf(smallest[0]);
      if (lower > upper) {
        slow = 1;
      } else {
        exp2 = lower > 0 ? lower : (upper < 0 ? upper : 0);
      }
    }
    exponent_shared = exp2;
    slow_shared = slow;
    exponents[row] = exp2;
    slow_rows[row] = slow;
    if (slow && slow_count) atomicAdd(slow_count, 1U);
  }
  __syncthreads();
  for (int i = lane; i < k; i += THREADS) {
    output[start + i] = slow_shared ? __float2half_rn(0.0f)
                                   : __float2half_rn(ldexpf(input[start + i], exponent_shared));
  }
}

// For normal rows: undo the exact power-of-two input scale before applying
// stored FP32 output-row weight scales and optional FP32 bias.
// For exceptional rows: compute the SAME dot product from the FP32 rotated
// input and original INT8 weights. This is deliberately slow, but executes
// only for device-marked exceptional rows. It is never a late-step fallback;
// the result of the current operation is corrected before use.
// One block per row makes the common all-safe case inexpensive.
__global__ void finish_rows(float* __restrict__ c32,
                            const float* __restrict__ x32,
                            const signed char* __restrict__ q,
                            const float* __restrict__ scale,
                            const float* __restrict__ bias,
                            const int* __restrict__ exponents,
                            const int* __restrict__ slow_rows,
                            int m, int n, int k) {
  const int row = blockIdx.x;
  if (row >= m) return;
  const int slow = slow_rows[row];
  const int exponent = exponents[row];
  const size_t xstart = static_cast<size_t>(row) * k;
  const size_t outstart = static_cast<size_t>(row) * n;
  for (int col = threadIdx.x; col < n; col += blockDim.x) {
    float value;
    if (slow) {
      const size_t qstart = static_cast<size_t>(col) * k;
      float sum = 0.0f;
      for (int i = 0; i < k; ++i) {
        sum = fmaf(x32[xstart + i], static_cast<float>(q[qstart + i]), sum);
      }
      value = sum;
    } else {
      value = ldexpf(c32[outstart + col], -exponent);
    }
    value = __fmul_rn(value, scale[col]);
    if (bias) value = __fadd_rn(value, bias[col]);
    c32[outstart + col] = value;
  }
}

}  // namespace

extern "C" int ltx25_cublas_prepare_f32_f16(
    const void* x32, void* x16, void* exponents, void* slow_rows,
    void* slow_count, int m, int k, void* stream_ptr) {
  if (!x32 || !x16 || !exponents || !slow_rows || m <= 0 || k <= 0) return -1;
  const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  prepare_rows<<<m, THREADS, 0, stream>>>(
      static_cast<const float*>(x32), static_cast<__half*>(x16),
      static_cast<int*>(exponents), static_cast<int*>(slow_rows),
      static_cast<unsigned int*>(slow_count), m, k);
  const auto err = cudaPeekAtLastError();
  return err == cudaSuccess ? 0 : 2000 + int(err);
}

// Q is the original unscaled INT8 cast exactly into temporary FP16.
// Column-major cuBLAS operands produce a row-major FP32 [M,N] output.
// The host CUDA API dispatches operations asynchronously on caller's stream.
extern "C" int ltx25_cublas_gemm_f16_f32(
    const void* q_rowmajor, const void* x_rowmajor, void* c_rowmajor,
    int n, int m, int k, void* stream_ptr) {
  if (!q_rowmajor || !x_rowmajor || !c_rowmajor || n <= 0 || m <= 0 || k <= 0) return -1;
  static thread_local cublasHandle_t handle = nullptr;
  if (!handle) {
    const cublasStatus_t created = cublasCreate(&handle);
    if (created != CUBLAS_STATUS_SUCCESS) return 1000 + int(created);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const cublasStatus_t selected = cublasSetStream(handle, stream);
  if (selected != CUBLAS_STATUS_SUCCESS) return 1100 + int(selected);

  const float alpha = 1.0f, beta = 0.0f;
  const cublasStatus_t status = cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
      &alpha, q_rowmajor, CUDA_R_16F, k,
      x_rowmajor, CUDA_R_16F, k,
      &beta, c_rowmajor, CUDA_R_32F, n,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return status == CUBLAS_STATUS_SUCCESS ? 0 : 1200 + int(status);
}

extern "C" int ltx25_cublas_epilogue_f32(
    void* c32, const void* x32, const void* q, const void* scale,
    const void* bias, const void* exponents, const void* slow_rows,
    int m, int n, int k, void* stream_ptr) {
  if (!c32 || !x32 || !q || !scale || !exponents || !slow_rows ||
      m <= 0 || n <= 0 || k <= 0) return -1;
  const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  finish_rows<<<m, THREADS, 0, stream>>>(
      static_cast<float*>(c32), static_cast<const float*>(x32),
      static_cast<const signed char*>(q), static_cast<const float*>(scale),
      static_cast<const float*>(bias), static_cast<const int*>(exponents),
      static_cast<const int*>(slow_rows), m, n, k);
  const auto err = cudaPeekAtLastError();
  return err == cudaSuccess ? 0 : 3000 + int(err);
}
