// Isolated INT8 -> FP16 conversion microbenchmark for SM70 V100.
// These CUDA kernels are NOT wired into production or Candidate E.
// No host/device synchronization; the caller owns storage and stream.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {

__global__ void cast_scalar(const int8_t* __restrict__ q,
                            __half* __restrict__ dst, size_t count) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = __float2half_rn(static_cast<float>(q[i]));
}

// Each CUDA thread loads four int8 values in one aligned 32-bit load and
// writes four FP16 values in two aligned 32-bit half2 stores.
__global__ void cast_vector4(const uint32_t* __restrict__ q,
                             __half2* __restrict__ dst,
                             size_t groups) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= groups) return;
    uint32_t packed = q[i];
    const int x0 = static_cast<int8_t>(packed & 0xFFu);
    const int x1 = static_cast<int8_t>((packed >> 8) & 0xFFu);
    const int x2 = static_cast<int8_t>((packed >> 16) & 0xFFu);
    const int x3 = static_cast<int8_t>((packed >> 24) & 0xFFu);
    dst[2*i] = __halves2half2(
        __float2half_rn(static_cast<float>(x0)),
        __float2half_rn(static_cast<float>(x1)));
    dst[2*i+1] = __halves2half2(
        __float2half_rn(static_cast<float>(x2)),
        __float2half_rn(static_cast<float>(x3)));
}

} // namespace

extern "C" int ltx25_cast_int8_half(
    const void* q, void* dst, size_t count, int variant, void* stream_ptr) {
    if (!q || !dst || count == 0) return -1;
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    constexpr int block = 256;
    if (variant == 0) {
        const size_t blocks = (count + block - 1) / block;
        if (blocks > static_cast<size_t>(UINT32_MAX)) return -3;
        cast_scalar<<<static_cast<unsigned int>(blocks), block, 0, stream>>>(
            static_cast<const int8_t*>(q), static_cast<__half*>(dst), count);
    } else if (variant == 4) {
        if ((count % 4) ||
            (reinterpret_cast<uintptr_t>(q) % 4) ||
            (reinterpret_cast<uintptr_t>(dst) % 4)) return -2;
        const size_t groups = count / 4;
        const size_t blocks = (groups + block - 1) / block;
        if (blocks > static_cast<size_t>(UINT32_MAX)) return -3;
        cast_vector4<<<static_cast<unsigned int>(blocks), block, 0, stream>>>(
            static_cast<const uint32_t*>(q),
            reinterpret_cast<__half2*>(dst), groups);
    } else {
        return -4;
    }
    const auto err = cudaPeekAtLastError();
    return err == cudaSuccess ? 0 : 1000 + int(err);
}
