// Fused leaf-tree GEMM: the whole m-leaf reduction plan in ONE kernel launch.
//
// out[M,N] = tree( leaf_0, ..., leaf_{m-1} ),   leaf_j = x[:, j*LK:(j+1)*LK] @ w[:, j*LK:(j+1)*LK]^T
//
// The shipped small-M path (pik/gemm.py _cublas_tree, m >= 2) is a SEQUENTIAL loop of cuBLASLt
// leaf GEMMs -- each too thin to fill an H100 at decode M, and the beta=1 pair trick forbids
// overlap. This kernel computes every leaf partial for an output tile in fp32 registers, walks
// K strictly sequentially in hardware k16 MMA steps, folds the leaves with the fixed balanced
// binary tree (pik/plan.py combine_order: left operand = lower leaf indices) in-register, and
// stores the fp32 root ONCE. No leaf workspace, no separate fold kernel, no launch sequence.
//
// WHY THE BITS MATCH THE SHIPPED PATH (measured, not argued -- and re-checked at first use by
// gemm._try_fused_leafgemm's admission bit-compare):
//   * pik's premise (pik/plan.py, pik/arch.py, verified by `python -m pik.arch --verify`): inside
//     a non-split-K GEMM the fp32 accumulation order along K is sequential over MMA k-steps
//     (k = 16 on Hopper) and nothing else moves it. Probed on this fleet's H100: a chained
//     mma.sync.m16n8k16 f32.bf16 accumulator is bit-identical to BOTH cuBLASLt's pinned
//     non-split-K algo and Triton's tl.dot, across M in {1..8192} (incl. 127/128/129),
//     leaf_k in {192, 640, 1280}, adversarial magnitudes spanning ~1e15.
//   * the fold is the same fp32 IEEE adds in the same combine_order the tree_reduce_kernel /
//     beta=1 pair trick realize (a cuBLASLt beta=1 accumulate is an fp32 add of the same two
//     leaf values -> same bits).
//   * bf16-leaf plans round each leaf ONCE (round-to-nearest-even, __float2bfloat16), exactly
//     where cuBLASLt's bf16-D store rounds; internal tree nodes stay fp32.
//
// FIXED SCHEDULE (the auditable part):
//   * grid = (ceil(N/BN), ceil(M/16)); one CTA per output tile; NO split-K, NO atomics.
//   * per CTA: leaves walked j = 0..m-1; within a leaf, K walked in ascending 64-element
//     chunks (cp.async double-buffered), each chunk in ascending k16 mma.sync steps.
//   * leaf_k % 64 == 0 is REQUIRED (all shipped row-parallel shapes satisfy it). This is not
//     laziness: a zero-padded k-tail would append `acc + 0.0` adds the shipped path does not
//     perform, and -0.0 + 0.0 = +0.0 could flip a sign bit. No padding, no extra adds.
//   * M/N tile tails are zero-FILLED rows in smem: every output element depends only on its own
//     x row and w row, so pad rows perturb nothing and their stores are masked off.
//   * tile shape cannot move bits (same premise; the admission check and the nightly battery
//     re-verify): BN=64 (4 warps x 2 n8-tiles) default, BN=32 for tiny M to double the CTA count.
//
// Deliberately NOT here: wgmma/TMA. mma.sync k16 chains are the proven-bit-identical unit; a
// wgmma schedule would have to re-prove the premise for warpgroup-wide accumulators. At decode
// M this kernel is bandwidth-bound on w and already saturates; revisit only with the probe.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cstdint>

#define DEVI __device__ __forceinline__

namespace {

constexpr int BM = 16;      // output rows per CTA (one m16 mma row-block)
constexpr int BK = 64;      // K elements staged per chunk
constexpr int PAD = 8;      // smem row pad (elements) -- keeps ldmatrix banks spread
constexpr int MAXD = 3;     // tree depth for m <= 8 leaves

DEVI uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

DEVI void cp16(void* dst_smem, const void* src_gmem, int src_bytes) {
  // 16B cp.async with zero-fill of the bytes past src_bytes.
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_u32(dst_smem)),
               "l"(src_gmem), "r"(src_bytes));
}

DEVI void zfill16(void* dst_smem) {
  *reinterpret_cast<uint4*>(dst_smem) = make_uint4(0u, 0u, 0u, 0u);
}

DEVI void commit() { asm volatile("cp.async.commit_group;\n"); }
DEVI void wait1() { asm volatile("cp.async.wait_group 1;\n"); }
DEVI void wait0() { asm volatile("cp.async.wait_group 0;\n"); }

DEVI void ldmatrix_x4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_u32(p)));
}

DEVI void ldmatrix_x2(uint32_t (&r)[2], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(smem_u32(p)));
}

DEVI void mma_16x8x16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Stage one (leaf, 64-wide K chunk) of x and w into smem, 16B segments, zero-filling rows past
// M/N. k chunk is always fully inside a leaf (leaf_k % BK == 0), so there is never a K tail.
template <int BN>
DEVI void stage_chunk(__nv_bfloat16* As, __nv_bfloat16* Ws, const __nv_bfloat16* x, int64_t ldx,
                      const __nv_bfloat16* w, int64_t ldw, int M, int N, int row0, int col0,
                      int64_t k0) {
  constexpr int SEGS_A = BM * (BK / 8);        // 128 segments of 16B
  constexpr int SEGS_W = BN * (BK / 8);
  constexpr int PITCH = BK + PAD;
  const int tid = threadIdx.x;
  const int nthr = blockDim.x;
  for (int s = tid; s < SEGS_A; s += nthr) {
    const int r = s >> 3, seg = s & 7;
    void* dst = As + r * PITCH + seg * 8;
    const int gr = row0 + r;
    if (gr < M) {
      cp16(dst, x + (int64_t)gr * ldx + k0 + seg * 8, 16);
    } else {
      zfill16(dst);
    }
  }
  for (int s = tid; s < SEGS_W; s += nthr) {
    const int r = s >> 3, seg = s & 7;
    void* dst = Ws + r * PITCH + seg * 8;
    const int gc = col0 + r;
    if (gc < N) {
      cp16(dst, w + (int64_t)gc * ldw + k0 + seg * 8, 16);
    } else {
      zfill16(dst);
    }
  }
}

// NT = n8-tiles per warp (4 warps): NT=2 -> BN=64, NT=1 -> BN=32.
template <int NT, bool BF16_LEAF>
__global__ void __launch_bounds__(128) fused_leaftree_kernel(
    const __nv_bfloat16* __restrict__ x, int64_t ldx, const __nv_bfloat16* __restrict__ w,
    int64_t ldw, float* __restrict__ out, int64_t ldo, int M, int N, int leaf_k, int m_leaves,
    int log2_m) {
  constexpr int BN = 4 * 8 * NT;
  constexpr int PITCH = BK + PAD;

  __shared__ __nv_bfloat16 As[2][BM * PITCH];
  __shared__ __nv_bfloat16 Ws[2][BN * PITCH];

  const int row0 = blockIdx.y * BM;
  const int col0 = blockIdx.x * BN;
  if (row0 >= M) return;

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int gid = lane >> 2;   // fragment row group
  const int tid4 = lane & 3;   // fragment col group
  const int nw0 = warp * 8 * NT;  // this warp's n offset inside the CTA tile

  float acc[NT][4];             // current leaf's partial
  float stack[MAXD + 1][NT][4]; // binary-counter carry stack (combine_order fold)
#pragma unroll
  for (int t = 0; t < NT; ++t)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[t][i] = 0.f;

  const int chunks_per_leaf = leaf_k / BK;
  const int total_chunks = m_leaves * chunks_per_leaf;

  // chunk c covers leaf (c / chunks_per_leaf), k offset (c % chunks_per_leaf) * BK
  auto k_of = [&](int c) {
    const int leaf = c / chunks_per_leaf;
    return (int64_t)leaf * leaf_k + (int64_t)(c % chunks_per_leaf) * BK;
  };

  stage_chunk<BN>(As[0], Ws[0], x, ldx, w, ldw, M, N, row0, col0, k_of(0));
  commit();

  for (int c = 0; c < total_chunks; ++c) {
    const int cur = c & 1;
    const bool has_next = (c + 1) < total_chunks;
    if (has_next) {
      stage_chunk<BN>(As[cur ^ 1], Ws[cur ^ 1], x, ldx, w, ldw, M, N, row0, col0, k_of(c + 1));
      commit();
      wait1();
    } else {
      wait0();
    }
    __syncthreads();

    // ---- compute: BK/16 = 4 mma k16 steps, strictly ascending k -------------------------
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      uint32_t a[4];
      // A fragment: matrices [rows 0-8 @kk, rows 8-16 @kk, rows 0-8 @kk+8, rows 8-16 @kk+8]
      {
        const int g = lane >> 3, r = lane & 7;
        const __nv_bfloat16* p = As[cur] + ((g & 1) * 8 + r) * PITCH + kk + (g >> 1) * 8;
        ldmatrix_x4(a, p);
      }
      if (NT == 2) {
        uint32_t b[4];
        // B matrices: [n0-8 @kk, n8-16 @kk, n0-8 @kk+8, n8-16 @kk+8]
        const int g = lane >> 3, r = lane & 7;
        const __nv_bfloat16* p = Ws[cur] + (nw0 + (g & 1) * 8 + r) * PITCH + kk + (g >> 1) * 8;
        ldmatrix_x4(b, p);
        mma_16x8x16(acc[0], a, b[0], b[2]);
        mma_16x8x16(acc[1], a, b[1], b[3]);
      } else {
        uint32_t b[2];
        // B matrices: [n0-8 @kk, n0-8 @kk+8] (lanes 0-15 supply addresses)
        const int g = (lane >> 3) & 1, r = lane & 7;
        const __nv_bfloat16* p = Ws[cur] + (nw0 + r) * PITCH + kk + g * 8;
        ldmatrix_x2(b, p);
        mma_16x8x16(acc[0], a, b[0], b[1]);
      }
    }
    __syncthreads();  // compute done before the next stage overwrites this buffer

    // ---- leaf boundary: fold into the fixed balanced tree (combine_order) --------------
    if ((c + 1) % chunks_per_leaf == 0) {
      const int j = c / chunks_per_leaf;  // leaf just finished
      if (BF16_LEAF) {
        // round the LEAF only -- the one TP-independent rounding point (RN, same as the
        // cuBLASLt bf16-D store the sequential path uses)
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
          for (int i = 0; i < 4; ++i)
            acc[t][i] = __bfloat162float(__float2bfloat16(acc[t][i]));
      }
      int lvl = 0;
      int tz = j;
      while (tz & 1) {  // binary-counter carries == combine_order, left = lower leaf indices
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
          for (int i = 0; i < 4; ++i) acc[t][i] = stack[lvl][t][i] + acc[t][i];
        tz >>= 1;
        ++lvl;
      }
#pragma unroll
      for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          stack[lvl][t][i] = acc[t][i];
          acc[t][i] = 0.f;
        }
    }
  }

  // root of the tree over all m leaves
  float(&res)[NT][4] = stack[log2_m];

  const int r0 = row0 + gid, r1 = row0 + gid + 8;
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    const int cbase = col0 + nw0 + t * 8 + tid4 * 2;
    if (r0 < M) {
      if (cbase < N) out[(int64_t)r0 * ldo + cbase] = res[t][0];
      if (cbase + 1 < N) out[(int64_t)r0 * ldo + cbase + 1] = res[t][1];
    }
    if (r1 < M) {
      if (cbase < N) out[(int64_t)r1 * ldo + cbase] = res[t][2];
      if (cbase + 1 < N) out[(int64_t)r1 * ldo + cbase + 1] = res[t][3];
    }
  }
}

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

}  // namespace

// x: [M, m*leaf_k] bf16 (stride(1)==1, stride(0) arbitrary but 16B-aligned)
// w: [N, m*leaf_k] bf16 (same)
// out: [M, N] fp32 -- receives tree(leaf partials), written exactly once
// bn32: use the BN=32 tile (more CTAs; for tiny M). Tile choice cannot move bits.
void fused_leaftree(torch::Tensor x, torch::Tensor w, torch::Tensor out, int64_t leaf_k,
                    bool bf16_leaf, bool bn32) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda(), "tensors must be CUDA");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && out.dim() == 2, "expected 2-D");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && w.scalar_type() == torch::kBFloat16,
              "x/w must be bf16");
  TORCH_CHECK(out.scalar_type() == torch::kFloat32,
              "out must be fp32: an m>=2 partial is an internal tree node");
  TORCH_CHECK(x.stride(1) == 1 && w.stride(1) == 1 && out.stride(1) == 1,
              "x/w/out must be contiguous along the last dim");

  const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
  TORCH_CHECK(w.size(1) == K, "K mismatch");
  TORCH_CHECK(out.size(0) == M && out.size(1) == N, "out shape mismatch");
  TORCH_CHECK(M > 0, "M=0 must be handled by the caller (cuBLASLt-rule parity)");
  TORCH_CHECK(leaf_k > 0 && K % leaf_k == 0, "leaf_k must divide K_local");
  const int64_t m = K / leaf_k;
  TORCH_CHECK(m >= 2 && m <= 8 && (m & (m - 1)) == 0,
              "fused leaf tree supports m in {2,4,8}; m=1 is a plain GEMM");
  TORCH_CHECK(leaf_k % BK == 0,
              "leaf_k must be a multiple of 64: a zero-padded K tail would add "
              "acc+0.0 steps the sequential path does not perform");
  TORCH_CHECK(aligned16(x.data_ptr()) && aligned16(w.data_ptr()), "x/w base must be 16B-aligned");
  TORCH_CHECK(x.stride(0) % 8 == 0 && w.stride(0) % 8 == 0,
              "x/w row stride must be a multiple of 8 elements (16B cp.async segments)");

  const int log2_m = (m == 2) ? 1 : (m == 4) ? 2 : 3;
  auto stream = at::cuda::getCurrentCUDAStream();
  const int BN = bn32 ? 32 : 64;
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 block(128);

#define LAUNCH(NT, BL)                                                                   \
  fused_leaftree_kernel<NT, BL><<<grid, block, 0, stream>>>(                             \
      (const __nv_bfloat16*)x.data_ptr(), x.stride(0), (const __nv_bfloat16*)w.data_ptr(), \
      w.stride(0), out.data_ptr<float>(), out.stride(0), (int)M, (int)N, (int)leaf_k,    \
      (int)m, log2_m)

  if (bn32) {
    if (bf16_leaf) LAUNCH(1, true); else LAUNCH(1, false);
  } else {
    if (bf16_leaf) LAUNCH(2, true); else LAUNCH(2, false);
  }
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_leaftree", &fused_leaftree,
        "one-launch m-leaf tree GEMM: bf16 x/w -> fp32 tree root, fixed mma.sync k16 schedule",
        py::arg("x"), py::arg("w"), py::arg("out"), py::arg("leaf_k"), py::arg("bf16_leaf") = false,
        py::arg("bn32") = false);
}
