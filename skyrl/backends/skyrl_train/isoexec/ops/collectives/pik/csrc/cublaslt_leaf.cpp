// cuBLASLt leaf GEMM: the fastest legal kernel inside a pik leaf.
//
// The reduction plan constrains only how K is cut and recombined. INSIDE a leaf,
// any kernel is legal provided its K-order depends on nothing but the data. So the
// leaf kernel does not have to be ours -- and it shouldn't be, because Triton on
// Blackwell tops out around 1000 TFLOP/s where cuBLAS does 1600.
//
// Two things torch's matmul cannot give us, both of which we need:
//
//   1. bf16 x bf16 -> FP32 output. The combine tree is fp32 at TP=1 (where there is
//      no all-reduce at all), so leaf partials must be fp32 at every TP size. Going
//      through bf16 would round each leaf and make the result depend on where the
//      rank boundary fell -- exactly the bug we are killing.
//
//   2. A PINNED algorithm with CUBLASLT_REDUCTION_SCHEME_NONE. cuBLAS picks kernels
//      by shape and reaches for split-K when K is large, which reorders the K
//      accumulation -- that is precisely why torch.matmul is not batch-invariant.
//      We ask the heuristic for non-split-K algos only, then cache the winner keyed
//      on (K, N, dtype) and NOT on M. Same kernel, same reduction scheme, every
//      batch size. Determinism becomes a constraint we impose rather than a cuBLAS
//      heuristic we hope holds.
//
// beta=1 is exposed so the caller can fuse the bottom level of the combine tree:
// running leaf 2i with beta=0 and leaf 2i+1 with beta=1 into the same buffer yields
// exactly (l_2i + l_2i+1) in fp32 -- a real tree node, for free. That halves the
// split-K workspace.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>

#define CHECK_CUBLAS(x)                                                     \
  do {                                                                      \
    cublasStatus_t s_ = (x);                                                \
    if (s_ != CUBLAS_STATUS_SUCCESS)                                        \
      throw std::runtime_error("cuBLASLt error " + std::to_string(s_) +     \
                               " at " __FILE__ ":" + std::to_string(__LINE__)); \
  } while (0)

namespace {

cublasLtHandle_t handle() {
  static cublasLtHandle_t h = nullptr;
  static std::once_flag once;
  std::call_once(once, [] { CHECK_CUBLAS(cublasLtCreate(&h)); });
  return h;
}

// Workspace for cuBLASLt. Not used for split-K (we forbid that); cuBLASLt still
// wants scratch for some non-split-K kernels.
void* workspace(size_t bytes) {
  static void* ptr = nullptr;
  static size_t cap = 0;
  if (cap < bytes) {
    if (ptr) cudaFree(ptr);
    if (cudaMalloc(&ptr, bytes) != cudaSuccess)
      throw std::runtime_error("pik: cudaMalloc for cuBLASLt workspace failed");
    cap = bytes;
  }
  return ptr;
}

constexpr size_t kWorkspaceBytes = 32u << 20;

// Key deliberately EXCLUDES M: one algo for every batch size is what makes the
// K-order batch-invariant. It DOES include the leading dims, because a leaf is a
// strided view (x[:, j*LK:(j+1)*LK]) whose lda differs from its K.
using AlgoKey = std::tuple<int64_t, int64_t, int64_t, int64_t, int>;  // K,N,lda,ldb,dtype
std::map<AlgoKey, cublasLtMatmulAlgo_t> g_algos;
std::mutex g_algo_mu;

}  // namespace

// x: [M, K] bf16/fp16, row-major (may be a K-slice view: stride(0) >= K)
// w: [N, K] bf16/fp16, row-major (may be a K-slice view)  -- native nn.Linear layout
// out: [M, N] fp32, row-major.   out = x @ w^T + beta * out
//
// Leaf slices are passed as strided VIEWS, never copies: a leaf is x[:, j*LK:(j+1)*LK],
// which is contiguous along K and strided along M. cuBLASLt takes that natively via ld.
void cublaslt_leaf_gemm(torch::Tensor x, torch::Tensor w, torch::Tensor out, double beta) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda(), "tensors must be CUDA");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && out.dim() == 2, "expected 2-D");
  TORCH_CHECK(x.stride(1) == 1 && w.stride(1) == 1 && out.stride(1) == 1,
              "x/w/out must be contiguous along their last dim");
  TORCH_CHECK(x.scalar_type() == w.scalar_type(), "x and w dtype must match");

  // fp32 out for row-parallel leaves (the combine tree is fp32). bf16 out for
  // column-parallel layers, where K is never sharded: there is no tree, nothing is
  // ever recombined, and forcing fp32 there would just double the store traffic for
  // a value we immediately round anyway. Compute is fp32 in both cases.
  const bool out_f32 = out.scalar_type() == torch::kFloat32;
  TORCH_CHECK(out_f32 || out.scalar_type() == torch::kBFloat16,
              "out must be fp32 (row-parallel: the tree is fp32) or bf16 (column-parallel)");
  const cudaDataType_t d_type = out_f32 ? CUDA_R_32F : CUDA_R_16BF;

  const int64_t M = x.size(0), K = x.size(1);
  const int64_t N = w.size(0);
  const int64_t lda = w.stride(0);   // w  : col-major [K,N], ld = w.stride(0)
  const int64_t ldb = x.stride(0);   // x  : col-major [K,M], ld = x.stride(0)
  const int64_t ldc = out.stride(0); // out: col-major [N,M], ld = out.stride(0)
  TORCH_CHECK(w.size(1) == K, "K mismatch");
  TORCH_CHECK(out.size(0) == M && out.size(1) == N, "out shape mismatch");

  const bool is_bf16 = x.scalar_type() == torch::kBFloat16;
  TORCH_CHECK(is_bf16 || x.scalar_type() == torch::kHalf, "x must be bf16 or fp16");
  const cudaDataType_t ab_type = is_bf16 ? CUDA_R_16BF : CUDA_R_16F;

  // Row-major [M,K] is column-major [K,M]; row-major [N,K] is column-major [K,N];
  // row-major out [M,N] is column-major [N,M]. So compute, in column-major:
  //     out_cm[N,M] = op_T(w_cm[K,N]) * op_N(x_cm[K,M])
  // i.e. an ordinary TN GEMM with m=N, n=M, k=K.
  cublasLtMatmulDesc_t op = nullptr;
  CHECK_CUBLAS(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));

  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&la, ab_type, K, N, lda));     // w, col-major [K,N]
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&lb, ab_type, K, M, ldb));     // x, col-major [K,M]
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&lc, d_type, N, M, ldc));    // out, col-major [N,M]

  const AlgoKey key{K, N, lda, ldb,
                    static_cast<int>(x.scalar_type()) * 100 + static_cast<int>(out.scalar_type())};
  cublasLtMatmulAlgo_t algo;
  bool have_algo = false;
  {
    std::lock_guard<std::mutex> g(g_algo_mu);
    auto it = g_algos.find(key);
    if (it != g_algos.end()) {
      algo = it->second;
      have_algo = true;
    }
  }

  if (!have_algo) {
    // Ask only for algos that do NOT split K. If cuBLAS has no such kernel for this
    // shape we must fail loudly rather than silently accept a reordering reduction.
    cublasLtMatmulPreference_t pref = nullptr;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
    size_t ws = kWorkspaceBytes;
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
    uint32_t mask = CUBLASLT_REDUCTION_SCHEME_NONE;
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &mask, sizeof(mask)));

    // Probe the heuristic at a large M so we pick a kernel tuned for the compute-
    // bound regime; the same algo is then reused for every M, which is the point.
    cublasLtMatrixLayout_t lb_probe = nullptr, lc_probe = nullptr;
    const int64_t M_probe = 8192;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&lb_probe, ab_type, K, M_probe, ldb));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&lc_probe, d_type, N, M_probe, ldc));

    cublasLtMatmulHeuristicResult_t res[8];
    int found = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(handle(), op, la, lb_probe, lc_probe,
                                                lc_probe, pref, 8, res, &found));
    cublasLtMatrixLayoutDestroy(lb_probe);
    cublasLtMatrixLayoutDestroy(lc_probe);
    cublasLtMatmulPreferenceDestroy(pref);

    if (found == 0) {
      cublasLtMatrixLayoutDestroy(la);
      cublasLtMatrixLayoutDestroy(lb);
      cublasLtMatrixLayoutDestroy(lc);
      cublasLtMatmulDescDestroy(op);
      throw std::runtime_error(
          "pik: cuBLASLt has no non-split-K algo for K=" + std::to_string(K) +
          " N=" + std::to_string(N) +
          ". Refusing to fall back to a split-K kernel, which would break "
          "TP-invariance. Use the Triton backend for this shape.");
    }
    algo = res[0].algo;
    std::lock_guard<std::mutex> g(g_algo_mu);
    g_algos[key] = algo;
  }

  const float alpha = 1.0f;
  const float beta_f = static_cast<float>(beta);
  auto stream = at::cuda::getCurrentCUDAStream();

  CHECK_CUBLAS(cublasLtMatmul(handle(), op, &alpha, w.data_ptr(), la, x.data_ptr(), lb,
                              &beta_f, out.data_ptr(), lc, out.data_ptr(), lc, &algo,
                              workspace(kWorkspaceBytes), kWorkspaceBytes, stream));

  cublasLtMatrixLayoutDestroy(la);
  cublasLtMatrixLayoutDestroy(lb);
  cublasLtMatrixLayoutDestroy(lc);
  cublasLtMatmulDescDestroy(op);
}

// --------------------------------------------------------------------------------------------
// Strided-batched leaf GEMM: ALL m leaf partials in ONE cuBLASLt call (batch dim = leaf index).
//
// At small M the sequential leaf loop launches m thin GEMMs that cannot fill the machine and
// the beta=1 pair trick serializes each pair; batching over leaves multiplies the grid by m in
// one launch. Measured at the GLM-4.7-Flash decode shapes this is the fastest legal small-M
// path (2.4-7x over the sequential loop, ahead of the hand-written fused mma kernel).
//
// The SAME determinism discipline as leaf_gemm, one difference deliberately called out:
//   * REDUCTION_SCHEME_NONE only -- split-K is refused, loudly.
//   * ONE pinned algo per (K, N, lda, ldb, batch) -- the key EXCLUDES M, so a single kernel
//     serves every batch size and the K-order is batch-invariant by construction.
//   * the heuristic is probed at M_probe = 128 (decode scale, the regime this path is gated
//     to by SKYRL_ISOEXEC_PIK_BATCHED_LEAVES_MAX_M) rather than leaf_gemm's 8192 -- probing a
//     small-M path at trainer scale would pin a compute-bound kernel it never runs at.
// Whether the pinned batched algo's per-leaf K-order matches the sequential path's is NOT
// assumed: pik/gemm.py admission bit-compares both paths on live operands per shape, and the
// nightly battery sweeps M 1..8192. (Measured on H100: bit-identical everywhere -- the
// pik/arch.py premise, K-order fixed by the hardware k-step, holding across kernels.)
//
// p is the [m, M, N] leaf workspace (fp32, or bf16 under a bf16-leaf plan -- rounding a LEAF
// is the one TP-independent rounding point); the caller folds it with the shipped
// tree_reduce_kernel, the same combine tree as every other schedule.

namespace {
using BatchKey = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int>;  // K,N,lda,ldb,batch,dtypes
std::map<BatchKey, cublasLtMatmulAlgo_t> g_batch_algos;
constexpr int64_t kBatchedMProbe = 128;
}  // namespace

void cublaslt_leaf_gemm_batched(torch::Tensor x, torch::Tensor w, torch::Tensor p,
                                int64_t leaf_k) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && p.is_cuda(), "tensors must be CUDA");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && p.dim() == 3, "x/w 2-D, p [m, M, N]");
  TORCH_CHECK(x.stride(1) == 1 && w.stride(1) == 1, "x/w must be contiguous along K");
  TORCH_CHECK(p.is_contiguous(), "leaf workspace must be contiguous");
  TORCH_CHECK(x.scalar_type() == w.scalar_type(), "x and w dtype must match");

  const int64_t M = x.size(0), K = x.size(1);
  const int64_t N = w.size(0);
  TORCH_CHECK(M > 0, "M=0 must be handled by the caller (cuBLASLt rejects it)");
  TORCH_CHECK(leaf_k > 0 && K % leaf_k == 0, "leaf_k must divide K_local");
  const int64_t batch = K / leaf_k;
  TORCH_CHECK(w.size(1) == K, "K mismatch");
  TORCH_CHECK(p.size(0) == batch && p.size(1) == M && p.size(2) == N, "p shape mismatch");

  const bool out_f32 = p.scalar_type() == torch::kFloat32;
  TORCH_CHECK(out_f32 || p.scalar_type() == torch::kBFloat16,
              "leaf partials must be fp32 (exact tree) or bf16 (bf16-leaf plan)");
  const cudaDataType_t d_type = out_f32 ? CUDA_R_32F : CUDA_R_16BF;
  const bool is_bf16 = x.scalar_type() == torch::kBFloat16;
  TORCH_CHECK(is_bf16 || x.scalar_type() == torch::kHalf, "x must be bf16 or fp16");
  const cudaDataType_t ab_type = is_bf16 ? CUDA_R_16BF : CUDA_R_16F;

  const int64_t lda = w.stride(0), ldb = x.stride(0), ldc = N;
  // leaf j of x/w starts j*leaf_k elements into the row; that IS the batch stride.
  const int64_t strideA = leaf_k, strideB = leaf_k, strideC = M * N;
  const int32_t bc = static_cast<int32_t>(batch);

  cublasLtMatmulDesc_t op = nullptr;
  CHECK_CUBLAS(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));

  auto make_layouts = [&](int64_t rows_M, cublasLtMatrixLayout_t& la, cublasLtMatrixLayout_t& lb,
                          cublasLtMatrixLayout_t& lc) {
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&la, ab_type, leaf_k, N, lda));      // w leaf, cm [K,N]
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&lb, ab_type, leaf_k, rows_M, ldb)); // x leaf, cm [K,M]
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&lc, d_type, N, rows_M, ldc));       // p[j], cm [N,M]
    const int64_t sC = rows_M * N;
    for (auto pr : {std::make_pair(la, strideA), std::make_pair(lb, strideB),
                    std::make_pair(lc, sC)}) {
      CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
          pr.first, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)));
      CHECK_CUBLAS(cublasLtMatrixLayoutSetAttribute(
          pr.first, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &pr.second, sizeof(pr.second)));
    }
  };

  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  make_layouts(M, la, lb, lc);

  const BatchKey key{leaf_k, N, lda, ldb, batch,
                     static_cast<int>(x.scalar_type()) * 100 + static_cast<int>(p.scalar_type())};
  cublasLtMatmulAlgo_t algo;
  bool have_algo = false;
  {
    std::lock_guard<std::mutex> g(g_algo_mu);
    auto it = g_batch_algos.find(key);
    if (it != g_batch_algos.end()) {
      algo = it->second;
      have_algo = true;
    }
  }

  if (!have_algo) {
    cublasLtMatmulPreference_t pref = nullptr;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
    size_t ws = kWorkspaceBytes;
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
    uint32_t mask = CUBLASLT_REDUCTION_SCHEME_NONE;
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &mask, sizeof(mask)));

    cublasLtMatrixLayout_t la_p = nullptr, lb_p = nullptr, lc_p = nullptr;
    make_layouts(kBatchedMProbe, la_p, lb_p, lc_p);
    cublasLtMatmulHeuristicResult_t res[8];
    int found = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(handle(), op, la_p, lb_p, lc_p, lc_p, pref, 8,
                                                res, &found));
    cublasLtMatrixLayoutDestroy(la_p);
    cublasLtMatrixLayoutDestroy(lb_p);
    cublasLtMatrixLayoutDestroy(lc_p);
    cublasLtMatmulPreferenceDestroy(pref);

    if (found == 0) {
      cublasLtMatrixLayoutDestroy(la);
      cublasLtMatrixLayoutDestroy(lb);
      cublasLtMatrixLayoutDestroy(lc);
      cublasLtMatmulDescDestroy(op);
      throw std::runtime_error(
          "pik: cuBLASLt has no non-split-K STRIDED-BATCH algo for leaf_k=" +
          std::to_string(leaf_k) + " N=" + std::to_string(N) + " batch=" + std::to_string(batch) +
          ". Refusing split-K; the caller falls back to the sequential leaf loop.");
    }
    algo = res[0].algo;
    std::lock_guard<std::mutex> g(g_algo_mu);
    g_batch_algos[key] = algo;
  }

  const float alpha = 1.0f, beta = 0.0f;
  auto stream = at::cuda::getCurrentCUDAStream();
  CHECK_CUBLAS(cublasLtMatmul(handle(), op, &alpha, w.data_ptr(), la, x.data_ptr(), lb, &beta,
                              p.data_ptr(), lc, p.data_ptr(), lc, &algo,
                              workspace(kWorkspaceBytes), kWorkspaceBytes, stream));

  cublasLtMatrixLayoutDestroy(la);
  cublasLtMatrixLayoutDestroy(lb);
  cublasLtMatrixLayoutDestroy(lc);
  cublasLtMatmulDescDestroy(op);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("leaf_gemm", &cublaslt_leaf_gemm,
        "bf16/fp16 x -> fp32 out, pinned non-split-K cuBLASLt algo",
        py::arg("x"), py::arg("w"), py::arg("out"), py::arg("beta") = 0.0);
  m.def("leaf_gemm_batched", &cublaslt_leaf_gemm_batched,
        "all m leaf partials in one strided-batch call, pinned non-split-K algo (M-free key)",
        py::arg("x"), py::arg("w"), py::arg("p"), py::arg("leaf_k"));
}
