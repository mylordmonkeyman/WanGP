// Isolated radix-4 Kronecker Hadamard on SM70.
// Research microbenchmark only; NOT used by default WanGP ConvRot.
// Exactly matches the SIGN / PERMUTATION convention of
// shared.qtypes.int8_convrot._regular_hadamard:
// h4=[[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]
// and H256 = (h4 kron h4 kron h4 kron h4)/16.
// FP32 additions have a different rounding order from torch.matmul, so
// outputs are compared numerically and must pass full-video quality gates.
// No CUDA synchronization, global kernel state, or persistent weights.
#include <cuda_runtime.h>
#include <cstdint>
#include <climits>

namespace {
constexpr int GROUP=256;
constexpr int THREADS=256;

__global__ void transform_radix4_256(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     size_t group_count) {
  size_t group=static_cast<size_t>(blockIdx.x);
  if(group>=group_count) return;
  const int tid=threadIdx.x;
  // Double buffering in shared memory prevents read/write hazards.
  __shared__ float buf_a[GROUP];
  __shared__ float buf_b[GROUP];
  const size_t start=group*GROUP;
  buf_a[tid]=input[start+tid];
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
} // namespace

extern "C" int ltx25_hadamard256_f32(const void* input, void* output,
                                      size_t element_count, void* stream_ptr) {
  if (!input || !output || !element_count || element_count%GROUP) return -1;
  if (input == output) return -2;
  const auto groups=element_count/GROUP;
  if(groups>static_cast<size_t>(INT_MAX)) return -3;
  cudaStream_t stream=reinterpret_cast<cudaStream_t>(stream_ptr);
  transform_radix4_256<<<static_cast<unsigned int>(groups), THREADS, 0, stream>>>(
       static_cast<const float*>(input), static_cast<float*>(output), groups);
  const auto err=cudaPeekAtLastError();
  return err==cudaSuccess ? 0 : 1000+int(err);
}
