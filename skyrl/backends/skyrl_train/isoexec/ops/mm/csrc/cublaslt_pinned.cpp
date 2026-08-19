// Production pinned-cuBLASLt dense-GEMM provider for the IsoExec decode/trainer forward.
//
// WHAT THIS IS
// ------------
// A drop-in numerical replacement for vLLM's batch-invariant Triton `matmul_persistent` on the
// handful of bf16 dense GEMMs where a PINNED non-split-K cuBLASLt kernel is both faster AND legal
// for IsoExec. "Legal" here has one meaning only: the result of row m must not depend on the batch
// M it was computed in, and it must be run-to-run deterministic. cuBLAS gives neither by default --
// it reaches for split-K when K is large (reordering the K reduction) and picks a different kernel
// per M. We defeat both by pinning ONE algorithm per (K, N, dtype, m_bucket) with the reduction
// scheme forced to NONE (no split-K), and we PROVE the pin obeys that with a hard introspection
// assert at pin time rather than trusting the heuristic mask.
//
// This is NOT bit-equal to the Triton kernel it replaces -- landing it MOVES the frozen gate
// signature (a Tier-2, signature-moving change, accepted; the orchestrator re-freezes after
// landing). Its internal contract, which this file enforces, is: M-invariant, deterministic,
// graph-safe, both-runtimes-identical, and fail-loud (any doubt at pin time throws, and the Python
// wrapper routes the throw to the Triton fallthrough).
//
// LAYOUT (matches nn.Linear / the pik leaf GEMM, so the validated column-major TN math is reused
// verbatim):
//   x   : [M, K] bf16, row-major (last dim contiguous)     -- the activations
//   w   : [N, K] bf16, row-major (last dim contiguous)      -- the weight (NOT transposed)
//   out : [M, N] bf16, row-major (last dim contiguous)      -- out = x @ w^T
// Row-major [M,K] is column-major [K,M]; row-major [N,K] is column-major [K,N]; row-major out
// [M,N] is column-major [N,M]. So in column-major we compute out_cm[N,M] = op_T(w_cm) * op_N(x_cm),
// an ordinary TN GEMM with m=N, n=M, k=K.
//
// THE TWO M-BUCKETS (both proven bit-identical -- cross-pin, wave-4 Track B):
//   bucket 0 "decode"  : algo probed at M_probe=512   -- selected when M < 1024
//   bucket 1 "trainer" : algo probed at M_probe=8192  -- selected when M >= 1024
// pin@512 and pin@8192 produce BIT-IDENTICAL outputs on every production shape (different output
// tiles, identical K-reduction order), so the bucket split is a PURE PERF choice: the invariant
// does not depend on which bucket ran. That is what makes a decode-vs-trainer M split legal.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#define CK(x)                                                                         \
  do {                                                                                \
    cublasStatus_t s_ = (x);                                                          \
    if (s_ != CUBLAS_STATUS_SUCCESS)                                                  \
      throw std::runtime_error("cuBLASLt error " + std::to_string(s_) +               \
                               " at " __FILE__ ":" + std::to_string(__LINE__));       \
  } while (0)

namespace {

constexpr size_t kWorkspaceBytes = 32u << 20;  // 32 MiB, fixed, allocated at init.
constexpr int64_t kMDecodeMax = 1024;          // M < this -> decode bucket (probe@512).
constexpr int64_t kMProbeDecode = 512;
constexpr int64_t kMProbeTrainer = 8192;

// -- Global cuBLASLt handle (created once at init, never in the hot path) -----------------------
cublasLtHandle_t g_handle = nullptr;
std::once_flag g_handle_once;

cublasLtHandle_t handle() {
  std::call_once(g_handle_once, [] { CK(cublasLtCreate(&g_handle)); });
  return g_handle;
}

// -- Fixed workspaces, ALLOCATED AT INIT --------------------------------------------------------
// cudaMalloc during CUDA-graph capture is illegal, so we never allocate in mm(). init() allocates
// up front; mm() asserts the pointer is live and reuses it. A cached algo + fixed workspace makes
// cublasLtMatmul a pure kernel launch -> graph-capturable (proven: bench_graph.py).
//
// ONE WORKSPACE PER CONCURRENCY SLOT (2026-08-10). The workspace is per-CALL scratch, so two
// cublasLtMatmul launches that OVERLAP IN TIME must not share one. That never arose while every
// call went out on the current stream; it arises the moment the MoE grouped path dispatches its
// per-expert GEMMs across several streams (mirroring TE's own multi_stream_cublas_gemm, which
// round-robins over 4 compute streams). Callers that stay single-stream pass slot 0 and are
// byte-for-byte unaffected -- same allocation, same pointer, same everything.
//
// The workspace CANNOT move a result bit: it is scratch for a fixed algo, not a partial-sum
// buffer whose layout could reorder a reduction -- and split-K (the only scheme that would put a
// reduction in there) is refused at pin time a few lines below. (Verified on device: slot 0 and
// slot 3 produce bit-identical output on the same operands.)
//
// THE ONE INVARIANT OF THIS FILE A CALLER CAN VIOLATE BY ACCIDENT: any caller that issues mm() on
// more than one stream concurrently MUST call init(n) with n >= the number of concurrent streams
// AND pass a DISTINCT slot per stream. Getting it wrong is fail-closed if the slot was never
// allocated (workspace_for returns null -> TORCH_CHECK), but is a SILENT DATA RACE if two streams
// pass the same allocated slot. There is no way to detect the latter from inside mm(), because a
// slot legitimately serves many *sequential* calls. See moe_expert_cublaslt._grouped_forward for
// the reference use, and ISOEXEC_GROUPED_GEMM_ANALYSIS.md 7.1 trigger T6.
constexpr int kMaxSlots = 16;
std::vector<void*> g_workspaces;  // index = concurrency slot
std::mutex g_ws_mu;

void alloc_workspaces(int slots) {
  if (slots < 1) slots = 1;
  if (slots > kMaxSlots)
    throw std::runtime_error("cublaslt_pinned: slots must be <= " + std::to_string(kMaxSlots));
  std::lock_guard<std::mutex> g(g_ws_mu);
  while ((int)g_workspaces.size() < slots) {
    void* p = nullptr;
    if (cudaMalloc(&p, kWorkspaceBytes) != cudaSuccess)
      throw std::runtime_error("cublaslt_pinned: cudaMalloc for a 32MiB workspace failed (slot " +
                               std::to_string(g_workspaces.size()) + ")");
    g_workspaces.push_back(p);
  }
}

void* workspace_for(int slot) {
  std::lock_guard<std::mutex> g(g_ws_mu);
  if (slot < 0 || slot >= (int)g_workspaces.size()) return nullptr;
  return g_workspaces[slot];
}

// -- Pin cache: ONE algo per (K, N, lda, ldb, m_bucket). Deliberately EXCLUDES M so a row's
//    K-reduction order is identical at every batch size within a bucket; buckets themselves are
//    proven bit-identical, so the cache is invariance-safe by construction + certification. -------
using AlgoKey = std::tuple<int64_t, int64_t, int64_t, int64_t, int>;  // K, N, lda, ldb, bucket
std::map<AlgoKey, cublasLtMatmulAlgo_t> g_algos;
std::mutex g_algo_mu;

// Read (config_id, tile_id, stages_id, splitk_num, reduction_scheme) off a selected algo.
struct AlgoAttrs {
  int cfg = -1, tile = -1, stages = -1, splitk = -1;
  uint32_t redsc = 0xffffffffu;
};
AlgoAttrs introspect_algo(const cublasLtMatmulAlgo_t& algo) {
  AlgoAttrs a;
  cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID, &a.cfg, sizeof(a.cfg), nullptr);
  cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &a.tile, sizeof(a.tile), nullptr);
  cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &a.stages, sizeof(a.stages), nullptr);
  cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &a.splitk, sizeof(a.splitk), nullptr);
  cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &a.redsc, sizeof(a.redsc), nullptr);
  return a;
}

// bf16-only guard. This provider is bf16 x bf16 -> bf16. fp32 (router, lm_head) and the N=1 gate
// scalar are EXCLUDED at the Python layer (legal cuBLASLt is slower there); we also refuse them
// here so a mis-routed call throws instead of silently running a slower/illegal kernel.
void require_bf16(const torch::Tensor& t, const char* what) {
  TORCH_CHECK(t.scalar_type() == torch::kBFloat16,
              "cublaslt_pinned: ", what, " must be bf16 (this provider is bf16-only; fp32/fp16 "
              "shapes stay on the Triton path)");
}

}  // namespace

// Create the handle + allocate `slots` fixed workspaces. Call at install time, before any
// CUDA-graph capture and before the hot path. Idempotent and MONOTONIC: calling it again with a
// larger `slots` allocates only the missing ones, so an already-installed single-stream caller is
// never disturbed by a later multi-stream one asking for more.
void init(int64_t slots) {
  handle();
  alloc_workspaces((int)slots);
}

// out = x @ w^T.   x:[M,K] bf16, w:[N,K] bf16, out:[M,N] bf16 -- all row-major (last dim contig).
// Pins (and hard-asserts non-split-K/reduction-NONE) on first use of each (K,N,layout,bucket).
void mm(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t slot) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda(), "cublaslt_pinned: tensors must be CUDA");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && out.dim() == 2, "cublaslt_pinned: expected 2-D tensors");
  require_bf16(x, "x");
  require_bf16(w, "w");
  require_bf16(out, "out");
  TORCH_CHECK(x.stride(1) == 1 && w.stride(1) == 1 && out.stride(1) == 1,
              "cublaslt_pinned: x/w/out must be contiguous along the last dim (K-contiguous "
              "W.t() layout only)");
  void* const ws = workspace_for((int)slot);
  TORCH_CHECK(ws != nullptr,
              "cublaslt_pinned: no workspace for slot ", slot,
              " -- init(slots) must be called with enough slots before mm() (never allocate in "
              "the hot path / during graph capture). Concurrent streams MUST use distinct slots: "
              "the workspace is per-call scratch, not shared state.");

  const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(w.size(1) == K, "cublaslt_pinned: K mismatch (w.size(1) != x.size(1))");
  TORCH_CHECK(out.size(0) == M && out.size(1) == N, "cublaslt_pinned: out shape mismatch");

  const int64_t lda = w.stride(0);    // w   col-major [K,N], ld = w.stride(0)
  const int64_t ldb = x.stride(0);    // x   col-major [K,M], ld = x.stride(0)
  const int64_t ldc = out.stride(0);  // out col-major [N,M], ld = out.stride(0)

  const int bucket = (M < kMDecodeMax) ? 0 : 1;
  const int64_t M_probe = (bucket == 0) ? kMProbeDecode : kMProbeTrainer;

  cublasLtMatmulDesc_t op = nullptr;
  CK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));

  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  CK(cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, K, N, lda));  // w   col-major [K,N]
  CK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, K, M, ldb));  // x   col-major [K,M]
  CK(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, N, M, ldc));  // out col-major [N,M]

  const AlgoKey key{K, N, lda, ldb, bucket};
  cublasLtMatmulAlgo_t algo;
  bool have = false;
  {
    std::lock_guard<std::mutex> g(g_algo_mu);
    auto it = g_algos.find(key);
    if (it != g_algos.end()) { algo = it->second; have = true; }
  }

  if (!have) {
    // Ask the heuristic for NON-split-K algos only, probed at the bucket's representative M.
    cublasLtMatmulPreference_t pref = nullptr;
    CK(cublasLtMatmulPreferenceCreate(&pref));
    size_t ws = kWorkspaceBytes;
    CK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
    uint32_t mask = CUBLASLT_REDUCTION_SCHEME_NONE;
    CK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &mask, sizeof(mask)));

    cublasLtMatrixLayout_t lbp = nullptr, lcp = nullptr;
    CK(cublasLtMatrixLayoutCreate(&lbp, CUDA_R_16BF, K, M_probe, ldb));
    CK(cublasLtMatrixLayoutCreate(&lcp, CUDA_R_16BF, N, M_probe, ldc));
    cublasLtMatmulHeuristicResult_t res[8];
    int found = 0;
    CK(cublasLtMatmulAlgoGetHeuristic(handle(), op, la, lbp, lcp, lcp, pref, 8, res, &found));
    cublasLtMatrixLayoutDestroy(lbp);
    cublasLtMatrixLayoutDestroy(lcp);
    cublasLtMatmulPreferenceDestroy(pref);

    if (found == 0) {
      cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc);
      cublasLtMatmulDescDestroy(op);
      throw std::runtime_error(
          "cublaslt_pinned: no non-split-K algo for K=" + std::to_string(K) + " N=" +
          std::to_string(N) + " bucket=" + std::to_string(bucket) +
          ". Refusing to fall back to a split-K kernel (would break M-invariance). Use Triton.");
    }

    // HARD INTROSPECTION ASSERT at pin time. The reduction-scheme MASK above already restricts the
    // search, but we do not trust it: we read the chosen algo back and REFUSE it unless it is
    // provably non-split-K with reduction scheme NONE. A pinned algo that splits K would reorder
    // the K accumulation and silently destroy IsoExec.
    const AlgoAttrs a = introspect_algo(res[0].algo);
    if (a.splitk != 1 || a.redsc != CUBLASLT_REDUCTION_SCHEME_NONE) {
      cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc);
      cublasLtMatmulDescDestroy(op);
      throw std::runtime_error(
          "cublaslt_pinned: heuristic returned a split-K / reducing algo for K=" +
          std::to_string(K) + " N=" + std::to_string(N) + " bucket=" + std::to_string(bucket) +
          " (splitK=" + std::to_string(a.splitk) + " reductionScheme=" + std::to_string(a.redsc) +
          "). Refusing to pin it. Use Triton for this shape.");
    }

    algo = res[0].algo;
    std::lock_guard<std::mutex> g(g_algo_mu);
    g_algos[key] = algo;
  }

  const float alpha = 1.0f, beta = 0.0f;
  auto stream = at::cuda::getCurrentCUDAStream();
  CK(cublasLtMatmul(handle(), op, &alpha, w.data_ptr(), la, x.data_ptr(), lb, &beta,
                    out.data_ptr(), lc, out.data_ptr(), lc, &algo, ws, kWorkspaceBytes, stream));

  cublasLtMatrixLayoutDestroy(la);
  cublasLtMatrixLayoutDestroy(lb);
  cublasLtMatrixLayoutDestroy(lc);
  cublasLtMatmulDescDestroy(op);
}

// Preview the algo the heuristic would pin for (K, N, bucket) at that bucket's probe-M, with the
// non-split-K mask applied -- returns {cfg, tile, stages, splitk, redsc, found, waves(x1000)}.
// -1 fields on found==0. Used by the certification battery to independently confirm splitk==1 and
// reductionScheme==NONE for every table shape/bucket before trusting the provider.
std::vector<int64_t> probe(int64_t K, int64_t N, int bucket) {
  const int64_t M_probe = (bucket == 0) ? kMProbeDecode : kMProbeTrainer;
  const int64_t lda = K, ldb = K, ldc = N;  // contiguous production layout

  cublasLtMatmulDesc_t op = nullptr;
  CK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));

  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
  CK(cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, K, N, lda));
  CK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, K, M_probe, ldb));
  CK(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, N, M_probe, ldc));
  cublasLtMatmulPreference_t pref = nullptr;
  CK(cublasLtMatmulPreferenceCreate(&pref));
  size_t ws = kWorkspaceBytes;
  CK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
  uint32_t mask = CUBLASLT_REDUCTION_SCHEME_NONE;
  CK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &mask, sizeof(mask)));

  cublasLtMatmulHeuristicResult_t res[8];
  int found = 0;
  CK(cublasLtMatmulAlgoGetHeuristic(handle(), op, la, lb, lc, lc, pref, 8, res, &found));
  std::vector<int64_t> outv(7, -1);
  outv[5] = found;
  if (found > 0) {
    const AlgoAttrs a = introspect_algo(res[0].algo);
    outv[0] = a.cfg; outv[1] = a.tile; outv[2] = a.stages; outv[3] = a.splitk;
    outv[4] = (int64_t)a.redsc; outv[6] = (int64_t)(res[0].wavesCount * 1000.0f);
  }
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(la);
  cublasLtMatrixLayoutDestroy(lb);
  cublasLtMatrixLayoutDestroy(lc);
  cublasLtMatmulDescDestroy(op);
  return outv;
}

// Number of distinct (K,N,lda,ldb,bucket) algos currently pinned (battery/self-check bookkeeping).
int64_t pinned_count() {
  std::lock_guard<std::mutex> g(g_algo_mu);
  return (int64_t)g_algos.size();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &init, "create cuBLASLt handle + allocate `slots` fixed 32MiB workspaces",
        py::arg("slots") = 1);
  m.def("mm", &mm, "out = x @ w^T, pinned non-split-K bf16 cuBLASLt (M-bucketed); `slot` selects "
                   "the per-call workspace and MUST be distinct across concurrent streams",
        py::arg("x"), py::arg("w"), py::arg("out"), py::arg("slot") = 0);
  m.def("probe", &probe, "preview pinned algo attrs for (K,N,bucket)",
        py::arg("K"), py::arg("N"), py::arg("bucket"));
  m.def("pinned_count", &pinned_count, "number of distinct algos pinned so far");
}
