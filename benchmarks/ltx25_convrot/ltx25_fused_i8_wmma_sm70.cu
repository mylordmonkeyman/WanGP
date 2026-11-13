// Experimental SM70 fused INT8 weight expansion + WMMA GEMM.
// Input q is original signed INT8 Q[N,K] row-major; X is FP16 X[M,K].
// Output FP32 C[M,N]. No full FP16 weight copy, quantization or allocations.
// Experiment ONLY: not wired into WanGP production dispatch.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cstdint>

using namespace nvcuda;

namespace {
constexpr int BM=32, BN=64, BK=32, THREADS=256;
__global__ void fused_i8_wmma(const int8_t* __restrict__ q,
                              const __half* __restrict__ x,
                              float* __restrict__ out,
                              int n,int m,int k) {
  const int nbase=blockIdx.x * BN;
  const int mbase=blockIdx.y * BM;
  const int warp=threadIdx.x/32;
  const int wm=warp/4;
  const int wn=warp%4;
  // Logical B is [K,N] column major. Each original q weight row is
  // an adjacent K sequence, so shared memory B[col*BK+k] is col-major.
  __shared__ __align__(16) __half btile[BN*BK];
  wmma::fragment<wmma::accumulator,16,16,16,float> acc;
  wmma::fill_fragment(acc,0.0f);
  for (int kb=0;kb<k;kb+=BK) {
    // All 256 threads cooperatively expand only a 64x32 INT8 weight tile.
    // No original weight is cached or transformed beyond this CTA iteration.
    #pragma unroll 1
    for(int i=threadIdx.x;i<BN*BK;i+=THREADS) {
      const int col=i/BK;
      const int kk=i%BK;
      btile[i]=__float2half_rn(float(q[(size_t(nbase+col)*k)+kb+kk]));
    }
    __syncthreads();
    #pragma unroll
    for(int kk=0;kk<BK;kk+=16) {
      wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> a;
      wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> b;
      wmma::load_matrix_sync(a,x+size_t(mbase+wm*16)*k+kb+kk,k);
      wmma::load_matrix_sync(b,btile+wn*16*BK+kk,BK);
      wmma::mma_sync(acc,a,b,acc);
    }
    __syncthreads();
  }
  wmma::store_matrix_sync(out+size_t(mbase+wm*16)*n+nbase+wn*16,
                          acc,n,wmma::mem_row_major);
}

// Identical to production vector4 INT8->FP16 cast, used for the baseline.
__global__ void expand_int8_four(const uint32_t* __restrict__ q,
                                 __half2* __restrict__ w, size_t groups) {
  const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=groups) return;
  uint32_t a=q[i];
  const int8_t q0=int8_t(a&255),q1=int8_t((a>>8)&255);
  const int8_t q2=int8_t((a>>16)&255),q3=int8_t((a>>24)&255);
  w[2*i]=__halves2half2(__float2half_rn(float(q0)),__float2half_rn(float(q1)));
  w[2*i+1]=__halves2half2(__float2half_rn(float(q2)),__float2half_rn(float(q3)));
}
} // namespace

extern "C" int ltx25_fused_i8_wmma(const void* q, const void* x,
                                  void* c, int n,int m,int k,void* ptr) {
  if(!q||!x||!c||n<=0||m<=0||k<=0) return -1;
  if(n%BN||m%BM||k%BK) return -2;
  const cudaStream_t stream=reinterpret_cast<cudaStream_t>(ptr);
  fused_i8_wmma<<<dim3(n/BN,m/BM),THREADS,0,stream>>>(
      static_cast<const int8_t*>(q), static_cast<const __half*>(x),
      static_cast<float*>(c),n,m,k);
  cudaError_t err=cudaPeekAtLastError();
  return err==cudaSuccess?0:1000+int(err);
}

extern "C" int ltx25_baseline_cast4(const void* q,void* w,
                                    size_t count,void* ptr) {
  if(!q||!w||!count||(count&3U))return -1;
  cudaStream_t stream=reinterpret_cast<cudaStream_t>(ptr);
  const size_t groups=count/4;
  expand_int8_four<<<(groups+255)/256,256,0,stream>>>(
      static_cast<const uint32_t*>(q),reinterpret_cast<__half2*>(w),groups);
  cudaError_t err=cudaPeekAtLastError();
  return err==cudaSuccess?0:1000+int(err);
}

extern "C" int ltx25_baseline_gemm(const void* w,const void* x,void* c,
                                    int n,int m,int k,void* ptr) {
  if(!w||!x||!c||n<=0||m<=0||k<=0)return -1;
  static thread_local cublasHandle_t handle=nullptr;
  if(!handle) {
    cublasStatus_t s=cublasCreate(&handle);
    if(s!=CUBLAS_STATUS_SUCCESS)return 1100+int(s);
  }
  auto stream=reinterpret_cast<cudaStream_t>(ptr);
  cublasStatus_t s=cublasSetStream(handle,stream);
  if(s!=CUBLAS_STATUS_SUCCESS)return 1200+int(s);
  float alpha=1.f,beta=0.f;
  s=cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,
                 &alpha,w,CUDA_R_16F,k,x,CUDA_R_16F,k,
                 &beta,c,CUDA_R_32F,n,CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  return s==CUBLAS_STATUS_SUCCESS?0:1300+int(s);
}
