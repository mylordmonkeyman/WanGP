// Isolated CUDA 12.x cuBLASLt heuristic probe on SM70, never production dispatch.
// FP16 A/B operands and FP32 C/D, exactly the Candidate E GEMM contract.
// Column-major layouts match ltx25_cublas_e.cu's Q^T @ X math and bit layout.
// Context/heuristic selection is done outside CUDA-event timing. Native run()
// only selects a pre-resolved algorithm and queues matmul on caller stream.
// No model weights are cached, no checkpoint conversion, no CPU/GPU sync.
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <new>
#include <vector>

namespace {
struct ProbeContext {
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr, d = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  std::vector<cublasLtMatmulHeuristicResult_t> algos;
};

void release(ProbeContext* p) {
  if (!p) return;
  if (p->preference) cublasLtMatmulPreferenceDestroy(p->preference);
  if (p->a) cublasLtMatrixLayoutDestroy(p->a);
  if (p->b) cublasLtMatrixLayoutDestroy(p->b);
  if (p->c) cublasLtMatrixLayoutDestroy(p->c);
  if (p->d) cublasLtMatrixLayoutDestroy(p->d);
  if (p->desc) cublasLtMatmulDescDestroy(p->desc);
  if (p->handle) cublasLtDestroy(p->handle);
  delete p;
}
} // namespace

extern "C" int ltx25_lt_probe_create(
    void** opaque, int n, int m, int k,
    size_t max_workspace, int max_algos) {
  if (!opaque || n <= 0 || m <= 0 || k <= 0 ||
      max_algos <= 0 || max_algos > 64) return -1;
  *opaque = nullptr;
  ProbeContext* p = new(std::nothrow) ProbeContext();
  if (!p) return -2;
  cublasStatus_t s;
#define CHECK(call) do { s=(call); if (s != CUBLAS_STATUS_SUCCESS) { release(p); return 1000+int(s); } } while (0)
  CHECK(cublasLtCreate(&p->handle));
  CHECK(cublasLtMatmulDescCreate(&p->desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  CHECK(cublasLtMatmulDescSetAttribute(
      p->desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  CHECK(cublasLtMatmulDescSetAttribute(
      p->desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
  // q_rowmajor[N,K] viewed as column-major [K,N] then transposed [N,K].
  // x_rowmajor[M,K] viewed as column-major [K,M] without transpose.
  // c_rowmajor[M,N] viewed as column-major [N,M] FP32 output.
  CHECK(cublasLtMatrixLayoutCreate(&p->a, CUDA_R_16F, k, n, k));
  CHECK(cublasLtMatrixLayoutCreate(&p->b, CUDA_R_16F, k, m, k));
  CHECK(cublasLtMatrixLayoutCreate(&p->c, CUDA_R_32F, n, m, n));
  CHECK(cublasLtMatrixLayoutCreate(&p->d, CUDA_R_32F, n, m, n));
  CHECK(cublasLtMatmulPreferenceCreate(&p->preference));
  CHECK(cublasLtMatmulPreferenceSetAttribute(
      p->preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &max_workspace, sizeof(max_workspace)));
  p->algos.resize(max_algos);
  int returned = 0;
  CHECK(cublasLtMatmulAlgoGetHeuristic(
      p->handle,p->desc,p->a,p->b,p->c,p->d,p->preference,
      max_algos,p->algos.data(),&returned));
  if (returned < 0 || returned > max_algos) {
    release(p); return -3;
  }
  p->algos.resize(returned);
  *opaque = static_cast<void*>(p);
#undef CHECK
  return 0;
}

extern "C" int ltx25_lt_probe_count(void* opaque) {
  auto* p=static_cast<ProbeContext*>(opaque);
  return p ? static_cast<int>(p->algos.size()) : -1;
}

extern "C" int ltx25_lt_probe_workspace(void* opaque, int algo, size_t* bytes) {
  auto* p=static_cast<ProbeContext*>(opaque);
  if (!p || !bytes || algo<0 || size_t(algo)>=p->algos.size()) return -1;
  *bytes=p->algos[algo].workspaceSize;
  return 0;
}

extern "C" int ltx25_lt_probe_run(
    void* opaque, int algo, const void* q, const void* x,
    void* output, void* workspace, size_t workspace_bytes, void* stream) {
  auto* p=static_cast<ProbeContext*>(opaque);
  if (!p || !q || !x || !output ||
      algo<0 || size_t(algo)>=p->algos.size()) return -1;
  auto& item=p->algos[algo];
  if (item.workspaceSize > workspace_bytes ||
      (item.workspaceSize && !workspace)) return -2;
  const float alpha=1.0f,beta=0.0f;
  const auto st=cublasLtMatmul(
      p->handle,p->desc,&alpha,
      q,p->a,x,p->b,&beta,
      output,p->c,output,p->d,
      &item.algo,workspace,workspace_bytes,
      reinterpret_cast<cudaStream_t>(stream));
  return st==CUBLAS_STATUS_SUCCESS ? 0 : 2000+int(st);
}

extern "C" void ltx25_lt_probe_destroy(void* opaque) {
  release(static_cast<ProbeContext*>(opaque));
}
