// LTX-2.5 Candidate E, SM70: FP16xFP16 -> FP32 cuBLAS with GPU-only
// activation range protection and an FP32 same-operation fallback per row.
// No host/device synchronization, dynamic allocation or global state in these
// entry points. The caller owns live scratch until all queued kernels finish.
// Kept separate from the other WanGP INT8 backends.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <climits>

namespace {

constexpr int THREADS = 256;

// A block owns one input row. Power-of-two rescaling is exact in FP32;
// every nonzero input must survive FP16 conversion without underflow to zero.
// SM70 supports FP16 subnormals, and the V100 preflight separately tests the
// cuBLAS subnormal contribution. Unsupported ranges use FP32 same-row math.
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
      // max*2^exp2 < 2^15 and min*2^exp2 >= 2^-24.
      // Permit tested FP16 subnormals, but never drop a nonzero input.
      const int upper = 14 - ilogbf(largest[0]);
      const int lower = -24 - ilogbf(smallest[0]);
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

// Exact final conversion fused into the FP32 epilogue (Candidate E default). CUDA's RN
// conversion is the same nearest-even rounding used by torch Tensor.to.
// Unlike the original finish_rows, the output is written directly as BF16 or
// FP16; the FP32 c32 scratch is read but need not be written back.
template <typename OutT>
__device__ __forceinline__ OutT narrow_output(float value);

template <>
__device__ __forceinline__ __half narrow_output<__half>(float value) {
  return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 narrow_output<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

template <typename OutT>
__global__ void finish_rows_low(float* __restrict__ c32,
                                OutT* __restrict__ output,
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
    output[outstart + col] = narrow_output<OutT>(value);
  }
}

// Experimental 256-point Kronecker Hadamard, exact H4 sign/order.
// FP32 rounding differs slightly from torch.matmul; only opt-in.
// One block per 256-element activation group. No persistent state.
constexpr int ROT_GROUP=256;
constexpr int ROT_THREADS=256;
template <typename InputT>
__global__ void transform_radix4_256_typed(const InputT* __restrict__ input,
                                          float* __restrict__ output,
                                          size_t group_count) {
  size_t group=static_cast<size_t>(blockIdx.x);
  if(group>=group_count) return;
  const int tid=threadIdx.x;
  // Double buffering in shared memory prevents read/write hazards.
  __shared__ float buf_a[ROT_GROUP];
  __shared__ float buf_b[ROT_GROUP];
  const size_t start=group*ROT_GROUP;
  // BF16/FP16 -> FP32 is exact; identical arithmetic to staging x.float().
  buf_a[tid]=static_cast<float>(input[start+tid]);
  __syncthreads();
  float* src=buf_a;
  float* dst=buf_b;
  #pragma unroll
  for (int stage=0;stage<4;++stage) {
    const int stride=1<<(2*stage);  // 1,4,16,64
    const int digit=(tid/stride)&3;
    const int base=tid-digit*stride;
    const float a=src[base];
    const float b=src[base+stride];
    const float c=src[base+2*stride];
    const float d=src[base+3*stride];
    float y=0.0f;
    if(digit==0)      y=((a+b)+c)-d;
    else if(digit==1) y=((a+b)-c)+d;
    else if(digit==2) y=((a-b)+c)+d;
    else              y=((-a+b)+c)+d;
    dst[tid]=y;
    __syncthreads();
    float* old=src; src=dst; dst=old;
  }
  // The exact normalization used by _regular_hadamard is 1/sqrt(256).
  output[start+tid]=src[tid]*0.0625f;
}

// Optional experimental faster INT8 -> FP16 weight expansion.
// Exact conversion: every signed INT8 value is precisely representable in FP16.
// This fills a bounded *transient* FP16 tensor only; no weight cache or
// second persistent weight set is retained.
__global__ void cast_qint8_to_fp16_four(const uint32_t* __restrict__ q,
                                       __half2* __restrict__ dst,
                                       size_t groups) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= groups) return;
  const uint32_t packed = q[i];
  const int x0 = static_cast<int8_t>(packed & 0xffU);
  const int x1 = static_cast<int8_t>((packed >> 8) & 0xffU);
  const int x2 = static_cast<int8_t>((packed >> 16) & 0xffU);
  const int x3 = static_cast<int8_t>((packed >> 24) & 0xffU);
  dst[2 * i] = __halves2half2(__float2half_rn(static_cast<float>(x0)),
                              __float2half_rn(static_cast<float>(x1)));
  dst[2 * i + 1] = __halves2half2(__float2half_rn(static_cast<float>(x2)),
                                  __float2half_rn(static_cast<float>(x3)));
}

// Experimental exact INT8 -> FP16 vector8 using one coalesced uint2 load
// and one aligned uint4 (16-byte) store per 8 weights. Benchmarked on V100:
// 1.29x-1.51x faster than vector4 across five actual matrix shapes.
// Still transient output: does not cache any FP16 model weights.
__device__ __forceinline__ uint32_t cast_eight_pack2(uint32_t a, uint32_t b) {
  union { __half2 halves; uint32_t raw; } bits;
  bits.halves=__halves2half2(
      __float2half_rn(static_cast<float>(static_cast<int8_t>(a & 0xffU))),
      __float2half_rn(static_cast<float>(static_cast<int8_t>(b & 0xffU))));
  return bits.raw;
}
__global__ void cast_qint8_to_fp16_eight(const uint2* __restrict__ q,
                                        uint4* __restrict__ dst,
                                        size_t groups) {
  const size_t i=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if (i >= groups) return;
  const uint2 packed=q[i];
  dst[i]=make_uint4(
      cast_eight_pack2(packed.x,packed.x>>8),
      cast_eight_pack2(packed.x>>16,packed.x>>24),
      cast_eight_pack2(packed.y,packed.y>>8),
      cast_eight_pack2(packed.y>>16,packed.y>>24));
}

}  // namespace

// Opt-in FP32 rotation replacement, same stream, output in caller-owned
// ephemeral memory. Never part of the default candidate unless selected.
extern "C" int ltx25_cublas_rot256_f32(const void* input, void* output,
                                        size_t count, void* stream_ptr) {
  if (!input || !output || !count || (count % 256) || input == output)
    return -1;
  const size_t groups=count/256;
  if (groups>static_cast<size_t>(INT_MAX)) return -2;
  const auto stream=reinterpret_cast<cudaStream_t>(stream_ptr);
  transform_radix4_256_typed<float><<<static_cast<unsigned int>(groups),256,0,stream>>>(
      static_cast<const float*>(input),static_cast<float*>(output),groups);
  const auto err=cudaPeekAtLastError();
  return err==cudaSuccess ? 0 : 5000+int(err);
}

// Default direct low-precision input to FP32 radix-4 Hadamard.
// The 256-element shared-memory buffer is FP32 and all arithmetic/order is
// identical to the float-input kernel. Eliminates x.float() staging, one
// device allocation and one PyTorch conversion launch per eligible call.
// Passed SM70 preflight and matched full-generation checks; staging remains selectable.
template <typename InputT>
static int launch_radix4_direct(const void* input, void* output,
                                size_t count, void* stream_ptr) {
  if (!input || !output || !count || (count % 256) || input == output)
    return -1;
  const size_t groups = count / 256;
  if (groups > static_cast<size_t>(INT_MAX)) return -2;
  const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  transform_radix4_256_typed<InputT><<<static_cast<unsigned int>(groups), 256, 0, stream>>>(
      static_cast<const InputT*>(input), static_cast<float*>(output), groups);
  const auto err = cudaPeekAtLastError();
  return err == cudaSuccess ? 0 : 7000 + int(err);
}

extern "C" int ltx25_cublas_rot256_from_bf16_f32(
    const void* input, void* output, size_t count, void* stream_ptr) {
  return launch_radix4_direct<__nv_bfloat16>(input, output, count, stream_ptr);
}

extern "C" int ltx25_cublas_rot256_from_f16_f32(
    const void* input, void* output, size_t count, void* stream_ptr) {
  return launch_radix4_direct<__half>(input, output, count, stream_ptr);
}

extern "C" int ltx25_cublas_cast_int8_to_f16_4(
    const void* q, void* dst, size_t count, void* stream_ptr) {
  if (!q || !dst || count == 0 || (count & 3U) ||
      (reinterpret_cast<uintptr_t>(q) & 3U) ||
      (reinterpret_cast<uintptr_t>(dst) & 3U)) return -1;
  const size_t groups = count / 4;
  const size_t blocks = (groups + THREADS - 1) / THREADS;
  if (blocks > static_cast<size_t>(UINT32_MAX)) return -2;
  const auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  cast_qint8_to_fp16_four<<<static_cast<unsigned int>(blocks), THREADS, 0, stream>>>(
      static_cast<const uint32_t*>(q), reinterpret_cast<__half2*>(dst), groups);
  const auto err = cudaPeekAtLastError();
  return err == cudaSuccess ? 0 : 4000 + int(err);
}

// Optional vector8 weight cast: exact FP16 conversion with aligned
// 8-byte source reads and 16-byte destination writes, same CUDA stream.
// No scratch/state, no host synchronization, no persistent FP16 copies.
extern "C" int ltx25_cublas_cast_int8_to_f16_8(
    const void* q, void* dst, size_t count, void* stream_ptr) {
  if (!q || !dst || !count || (count & 7U) ||
      (reinterpret_cast<uintptr_t>(q) & 7U) ||
      (reinterpret_cast<uintptr_t>(dst) & 15U)) return -1;
  const size_t groups=count/8;
  const size_t blocks=(groups + THREADS - 1)/THREADS;
  if (blocks > static_cast<size_t>(UINT32_MAX)) return -2;
  const auto stream=reinterpret_cast<cudaStream_t>(stream_ptr);
  cast_qint8_to_fp16_eight<<<static_cast<unsigned int>(blocks),THREADS,0,stream>>>(
      static_cast<const uint2*>(q),static_cast<uint4*>(dst),groups);
  const auto err=cudaPeekAtLastError();
  return err == cudaSuccess ? 0 : 6000 + int(err);
}

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

// Opt-in, shape-specific SM70 cuBLAS Tensor Op algorithm 2.
// Isolated V100 sweep showed bit-exact FP32 outputs for N16384 K4096 M64.
// All other shapes MUST use the existing CUBLAS_GEMM_DEFAULT_TENSOR_OP.
extern "C" int ltx25_cublas_gemm_f16_f32_algo2(
    const void* q_rowmajor, const void* x_rowmajor, void* c_rowmajor,
    int n, int m, int k, void* stream_ptr) {
  if (!q_rowmajor || !x_rowmajor || !c_rowmajor) return -1;
  if (n != 16384 || k != 4096 || m != 64) return -2;
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
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_ALGO2_TENSOR_OP);
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

// Default fused-output epilogue entry points. The original
// ltx25_cublas_epilogue_f32 remains available via WAN2GP_LTX25_E_OUTPUT=torch.
template <typename OutT>
static int launch_epilogue_low(
    void* c32, void* output, const void* x32, const void* q,
    const void* scale, const void* bias, const void* exponents,
    const void* slow_rows, int m, int n, int k, void* stream_ptr) {
  if (!c32 || !output || !x32 || !q || !scale || !exponents ||
      !slow_rows || m <= 0 || n <= 0 || k <= 0)
    return -1;
  const auto stream=reinterpret_cast<cudaStream_t>(stream_ptr);
  finish_rows_low<OutT><<<m, THREADS, 0, stream>>>(
      static_cast<float*>(c32), static_cast<OutT*>(output),
      static_cast<const float*>(x32), static_cast<const signed char*>(q),
      static_cast<const float*>(scale), static_cast<const float*>(bias),
      static_cast<const int*>(exponents), static_cast<const int*>(slow_rows),
      m, n, k);
  const auto err=cudaPeekAtLastError();
  return err==cudaSuccess ? 0 : 8000+int(err);
}

extern "C" int ltx25_cublas_epilogue_f32_to_bf16(
    void* c32, void* output, const void* x32, const void* q,
    const void* scale, const void* bias, const void* exponents,
    const void* slow_rows, int m, int n, int k, void* stream_ptr) {
  return launch_epilogue_low<__nv_bfloat16>(
      c32, output, x32, q, scale, bias, exponents, slow_rows,
      m, n, k, stream_ptr);
}

extern "C" int ltx25_cublas_epilogue_f32_to_f16(
    void* c32, void* output, const void* x32, const void* q,
    const void* scale, const void* bias, const void* exponents,
    const void* slow_rows, int m, int n, int k, void* stream_ptr) {
  return launch_epilogue_low<__half>(
      c32, output, x32, q, scale, bias, exponents, slow_rows,
      m, n, k, stream_ptr);
}
