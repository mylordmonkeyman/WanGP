// Isolated Tesla V100 (sm_70) INT8->FP16 weight-expansion experiment.
// Compare production-equivalent vector4 conversion with packed vector8/16
// loads+stores and launch configuration variants. All variants convert exactly.
// No model weights, persistent FP16 copies, or production dispatch changes.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <climits>

namespace {
__device__ __forceinline__ uint32_t pack2(uint32_t a, uint32_t b) {
  union { __half2 v; uint32_t u; } bits;
  bits.v = __halves2half2(
    __float2half_rn(static_cast<float>(static_cast<int8_t>(a & 255U))),
    __float2half_rn(static_cast<float>(static_cast<int8_t>(b & 255U))));
  return bits.u;
}

// Copied from the existing validated Candidate E vector4 kernel.
__global__ void production_vector4(const uint32_t* __restrict__ q,
                                    __half2* __restrict__ dst, size_t groups) {
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

template<int WIDTH, bool GRID_STRIDE>
__global__ void vector_expand(const void* __restrict__ qv,
                              void* __restrict__ outv, size_t groups) {
  const size_t first=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  const size_t stride=size_t(gridDim.x)*blockDim.x;
  for(size_t i=first;i<groups;i+=stride) {
    if constexpr(WIDTH==8) {
      const uint2 bits=static_cast<const uint2*>(qv)[i];
      const uint4 expanded=make_uint4(
        pack2(bits.x, bits.x>>8), pack2(bits.x>>16,bits.x>>24),
        pack2(bits.y, bits.y>>8), pack2(bits.y>>16,bits.y>>24));
      static_cast<uint4*>(outv)[i]=expanded;
    } else {
      const uint4 bits=static_cast<const uint4*>(qv)[i];
      const uint4 lo=make_uint4(
        pack2(bits.x, bits.x>>8), pack2(bits.x>>16,bits.x>>24),
        pack2(bits.y, bits.y>>8), pack2(bits.y>>16,bits.y>>24));
      const uint4 hi=make_uint4(
        pack2(bits.z, bits.z>>8), pack2(bits.z>>16,bits.z>>24),
        pack2(bits.w, bits.w>>8), pack2(bits.w>>16,bits.w>>24));
      static_cast<uint4*>(outv)[2*i]=lo;
      static_cast<uint4*>(outv)[2*i+1]=hi;
    }
    if constexpr(!GRID_STRIDE) break;
  }
}
} // namespace

// Variant 0 is production-equivalent vector4 (256 threads).
// 1: v8/256, 2: v16/256, 3: v8/128, 4: v16/128,
// 5: v8/512, 6: v16/512, 7: v8/256 grid-stride 640 blocks,
// 8: v16/256 grid-stride 640 blocks.
extern "C" int ltx25_cast_width_probe(const void* q, void* out,
                                     size_t count, int variant, void* streamv) {
  if (!q||!out||!count||count%16||reinterpret_cast<uintptr_t>(q)%16||
      reinterpret_cast<uintptr_t>(out)%16) return -1;
  if(variant<0||variant>8) return -2;
  int width=variant==0?4:(variant==2||variant==4||variant==6||variant==8?16:8);
  int threads=(variant==3||variant==4)?128:(variant==5||variant==6)?512:256;
  const size_t groups=count/width;
  const size_t blocks=(groups+threads-1)/threads;
  if(blocks>UINT_MAX) return -3;
  const cudaStream_t stream=reinterpret_cast<cudaStream_t>(streamv);
  if(variant==0)
    production_vector4<<<static_cast<unsigned>(blocks),256,0,stream>>>(
      static_cast<const uint32_t*>(q),static_cast<__half2*>(out),groups);
  else if(variant==1)
    vector_expand<8,false><<<static_cast<unsigned>(blocks),256,0,stream>>>(q,out,groups);
  else if(variant==2)
    vector_expand<16,false><<<static_cast<unsigned>(blocks),256,0,stream>>>(q,out,groups);
  else if(variant==3)
    vector_expand<8,false><<<static_cast<unsigned>(blocks),128,0,stream>>>(q,out,groups);
  else if(variant==4)
    vector_expand<16,false><<<static_cast<unsigned>(blocks),128,0,stream>>>(q,out,groups);
  else if(variant==5)
    vector_expand<8,false><<<static_cast<unsigned>(blocks),512,0,stream>>>(q,out,groups);
  else if(variant==6)
    vector_expand<16,false><<<static_cast<unsigned>(blocks),512,0,stream>>>(q,out,groups);
  else if(variant==7)
    vector_expand<8,true><<<640,256,0,stream>>>(q,out,groups);
  else
    vector_expand<16,true><<<640,256,0,stream>>>(q,out,groups);
  cudaError_t err=cudaPeekAtLastError();
  return err==cudaSuccess?0:1000+int(err);
}
