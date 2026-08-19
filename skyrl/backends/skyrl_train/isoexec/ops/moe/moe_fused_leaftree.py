"""The fused MoE GEMM with the pik leaf tree INSIDE the kernel: one launch, one fp32 store.

The pik-fc2 leaf tree makes the expert fc2 ETP-invariant by cutting the reduction into ``N_LEAVES`` fixed
leaves, rounding each leaf to the compute dtype, and folding the fp32-promoted leaves with a balanced
binary tree. Doing that at the Python level costs ``n_leaves`` extra full ``[T, H]`` memory sweeps, so this
module vendors vLLM's ``fused_moe_kernel`` (stripped to the dense-bf16 / top_k=1 / no-bias path we run) and
does the whole fold in registers.

Each leaf's inner loop is the native kernel's K-loop at ``K = K_leaf`` from a zero accumulator, so a leaf
value equals the separate-call-per-leaf value; ``acc.to(compute_type)`` is the same round the native kernel
applies at its C store and the promotion back to fp32 is exact. The streaming fold merges leaf ``i`` into
the pending stack by its trailing-one bits, which reproduces ``_tree_sum``'s balanced tree with the lower
leaf index on the left. The C store stays fp32 because it is this rank's leaf-subtree PARTIAL, which the
cross-rank pik tree consumes -- rounding it to bf16 here would change the cross-rank sum.

``N_LEAVES`` must be a power of two <= 8, and ``K`` must divide evenly into that many leaves (asserted
host-side). ``BLOCK_SIZE_K`` sets the K-tiling and therefore the accumulation order, so it cannot be
retuned. The launch grid, block map inputs and pid->tile mapping are unchanged from the native call, so the
path stays shape-static and sync-free and CUDA-graph capture is unaffected. Three variants (runtime-K,
constexpr-K, stream) are dispatched purely from static shape and leaf-count data and are bitwise identical:
same fp expression, different addressing and scheduling.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def fused_moe_leaftree_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    N_LEAVES: tl.constexpr,
    compute_type: tl.constexpr,
    WIRE_STORE: tl.constexpr,
):
    # pid -> (pid_m, pid_n): verbatim from vLLM's fused_moe_kernel (grouped for L2 reuse). The
    # mapping has no numerical effect (each program computes an independent C tile).
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + offs
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    # our block map (moe_fused_experts._block_map) never emits -1 experts, so no zero-fill branch.
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    K_LEAF = K // N_LEAVES  # host asserts divisibility

    # Streaming balanced fold. Leaves arrive in index order; merging by trailing-one bits of the
    # leaf index reproduces _tree_sum's balanced tree with the earlier partial always on the LEFT:
    #   ((l0+l1)+(l2+l3)) + ((l4+l5)+(l6+l7)).
    # s0/s1/s2 hold the pending partial of tree levels 0/1/2. All branches below are on constexpr
    # ints, so tracing prunes them: N_LEAVES=1 keeps nothing, =2 keeps s0, =8 keeps all three.
    for leaf in tl.static_range(N_LEAVES):
        a_ptrs = a_ptr + (offs_token[:, None] * stride_am + (leaf * K_LEAF + offs_k[None, :]) * stride_ak)
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + ((leaf * K_LEAF + offs_k[:, None]) * stride_bk + offs_bn[None, :] * stride_bn)
        )
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for kb in range(tl.cdiv(K_LEAF, BLOCK_SIZE_K)):
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K_LEAF - kb * BLOCK_SIZE_K),
                other=0.0,
            )
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K_LEAF - kb * BLOCK_SIZE_K, other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        # the leaf value: rounded to the compute dtype exactly where the native kernel rounds its
        # C store, then promoted fp32 (exact) for the tree.
        v = acc.to(compute_type).to(tl.float32)

        if (leaf & 1) == 1:
            v = s0 + v  # noqa: F821  (assigned on the previous unrolled iteration)
        if (leaf & 3) == 3:
            v = s1 + v  # noqa: F821
        if (leaf & 7) == 7:
            v = s2 + v  # noqa: F821
        if leaf == N_LEAVES - 1:
            result = v
        elif (leaf & 1) == 0:
            s0 = v  # noqa: F841  (consumed on a later unrolled iteration)
        elif (leaf & 3) == 1:
            s1 = v  # noqa: F841
        else:
            s2 = v  # noqa: F841

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    if WIRE_STORE:
        # Owner-combine wire stage (n_leaves == 1 only; the wrapper enforces it): C is the bf16 symmetric
        # staging buffer and the store IS the wire round. At one leaf ``result == extf(truncf(acc))``, so
        # ``result.to(compute_type) == truncf(acc)`` -- the RNE round-trip through the wider type is the
        # identity, i.e. the exact bytes the exchange's stage ``copy_`` produced from the fp32 store.
        tl.store(c_ptrs, result.to(compute_type), mask=c_mask)
    else:
        tl.store(c_ptrs, result, mask=c_mask)  # C is fp32; no conversion


@triton.jit
def fused_moe_leaftree_kernel_unrolled(
    a_ptr,
    b_ptr,
    c_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    N_LEAVES: tl.constexpr,
    compute_type: tl.constexpr,
):
    """The same kernel with K constexpr: the leaf/K loop-nest fully unrolls (see module docstring).
    Same fp expression, different instruction scheduling -- bitwise-identical to the runtime-K
    variant. Used for N_LEAVES >= 4, where the tiny per-leaf loops otherwise defeat pipelining."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token = tl.load(sorted_token_ids_ptr + pid_m * BLOCK_SIZE_M + offs).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    K_LEAF: tl.constexpr = K // N_LEAVES
    KB_PER_LEAF: tl.constexpr = (K_LEAF + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K

    for leaf in tl.static_range(N_LEAVES):
        a_ptrs = a_ptr + (offs_token[:, None] * stride_am + (leaf * K_LEAF + offs_k[None, :]) * stride_ak)
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + ((leaf * K_LEAF + offs_k[:, None]) * stride_bk + offs_bn[None, :] * stride_bn)
        )
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for kb in tl.static_range(KB_PER_LEAF):
            if (kb + 1) * BLOCK_SIZE_K <= K_LEAF:  # constexpr: full block, the k-mask prunes away
                a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
                b = tl.load(b_ptrs)
            else:
                a = tl.load(
                    a_ptrs,
                    mask=token_mask[:, None] & (offs_k[None, :] < K_LEAF - kb * BLOCK_SIZE_K),
                    other=0.0,
                )
                b = tl.load(b_ptrs, mask=offs_k[:, None] < K_LEAF - kb * BLOCK_SIZE_K, other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        v = acc.to(compute_type).to(tl.float32)
        if (leaf & 1) == 1:
            v = s0 + v  # noqa: F821
        if (leaf & 3) == 3:
            v = s1 + v  # noqa: F821
        if (leaf & 7) == 7:
            v = s2 + v  # noqa: F821
        if leaf == N_LEAVES - 1:
            result = v
        elif (leaf & 1) == 0:
            s0 = v  # noqa: F841  (consumed on a later unrolled iteration)
        elif (leaf & 3) == 1:
            s1 = v  # noqa: F841
        else:
            s2 = v  # noqa: F841

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, result, mask=c_mask)


@triton.jit
def fused_moe_leaftree_kernel_stream(
    a_ptr,
    b_ptr,
    c_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    N_LEAVES: tl.constexpr,
    compute_type: tl.constexpr,
):
    """Register-lean constexpr-K variant. Same fp expression as the two kernels above; the wins are purely
    in what the compiler has to keep live. Requires ``K_LEAF % BLOCK_SIZE_K == 0``; the wrapper checks and
    falls back to :func:`fused_moe_leaftree_kernel_unrolled` otherwise.

    Two changes, both address-level or type-level, neither arithmetic:

    1. ONE CONTINUOUSLY-ADVANCING POINTER PAIR. The leaves are contiguous K-ranges stepped by the same
       BLOCK_K, so leaf ``i+1``'s first k-block starts exactly where leaf ``i``'s last one ended: a single
       pointer pair hoisted out of the leaf loop visits the identical addresses in the identical order,
       instead of rebuilding an int64 pointer tile per leaf. With ``K_LEAF`` a multiple of ``BLOCK_SIZE_K``
       the per-leaf tail masks are vacuous and prune away.

    2. THE LEVEL-0 PENDING PARTIAL IS HELD IN ``compute_type``. ``s0`` is always a BARE leaf, i.e. exactly
       ``acc.to(compute_type).to(tl.float32)``, so narrowing it again is the identity and halves that
       tile's footprint. ``s1``/``s2`` are sums of leaves and stay fp32.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token = tl.load(sorted_token_ids_ptr + pid_m * BLOCK_SIZE_M + offs).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    K_LEAF: tl.constexpr = K // N_LEAVES
    KB_PER_LEAF: tl.constexpr = K_LEAF // BLOCK_SIZE_K  # exact; the wrapper asserts

    # the single pointer pair -- set up once, advanced straight through every leaf's K-range.
    a_ptrs = a_ptr + (offs_token[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    for leaf in tl.static_range(N_LEAVES):
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for kb in tl.static_range(KB_PER_LEAF):
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        vc = acc.to(compute_type)  # the leaf's bf16 round, exactly where the native kernel rounds
        v = vc.to(tl.float32)  # exact
        if (leaf & 1) == 1:
            v = s0.to(tl.float32) + v  # noqa: F821  (exact: s0 is a bare leaf, compute_type-representable)
        if (leaf & 3) == 3:
            v = s1 + v  # noqa: F821
        if (leaf & 7) == 7:
            v = s2 + v  # noqa: F821
        if leaf == N_LEAVES - 1:
            result = v
        elif (leaf & 1) == 0:
            s0 = vc  # noqa: F841  (bare leaf -- held narrow, lossless)
        elif (leaf & 3) == 1:
            s1 = v  # noqa: F841
        else:
            s2 = v  # noqa: F841

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, result, mask=c_mask)


# Launch parameters per leaf count. FIXED CONSTANTS keyed only on n_leaves (a topology constant, not the
# token count), so batch-invariance is untouched. num_warps/num_stages are bitwise-neutral.
_STREAM_BLOCK_N = 128  # amortizes the per-leaf epilogue over 2x the output tile
_STREAM_NUM_WARPS = 4
_STREAM_NUM_STAGES = 3


def invoke_fused_moe_leaftree_kernel(
    A: torch.Tensor,  # [T, K] expert-grouped rows (bf16), K = this rank's moe_intermediate shard
    B: torch.Tensor,  # [E, N, K] stacked fc2 weights (bf16)
    C: torch.Tensor,  # [T, 1, N] fp32 out: this rank's leaf-subtree partial
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    n_leaves: int,
    config: dict,
    compute_type: tl.dtype,
    wire_store: bool = False,
) -> None:
    """Launch wrapper, mirroring vLLM's ``invoke_fused_moe_triton_kernel`` for our top_k=1 usage
    (including its small-batch EM shrink). Strides are passed through, so A/B may be strided views.

    ``wire_store``: C is the bf16 symmetric wire buffer and the kernel stores the bf16 leaf value
    directly. ``n_leaves`` must be 1 -- a multi-leaf result is an fp32 internal tree node and rounding it
    on the wire would move bits. Base-kernel path only."""
    K = B.size(2)
    if K % n_leaves != 0:
        raise RuntimeError(f"[isoexec-moe] leaf-tree kernel: K={K} not divisible by n_leaves={n_leaves}")
    if n_leaves not in (1, 2, 4, 8):
        raise RuntimeError(f"[isoexec-moe] leaf-tree kernel: n_leaves must be a power of two <= 8, got {n_leaves}")
    if wire_store:
        if n_leaves != 1:
            raise RuntimeError(
                f"[isoexec-moe] leaf-tree wire store: n_leaves={n_leaves} != 1 -- a multi-leaf result "
                "is an fp32 internal tree node and rounding it on the wire would move bits"
            )
        assert C.dtype == torch.bfloat16 and sorted_token_ids.stride(0) == 1
    else:
        assert C.dtype == torch.float32 and sorted_token_ids.stride(0) == 1

    N = B.size(1)
    BLOCK_M, BLOCK_K = config["BLOCK_SIZE_M"], config["BLOCK_SIZE_K"]
    EM = sorted_token_ids.size(0)
    if A.size(0) < BLOCK_M:
        EM = min(EM, A.size(0) * BLOCK_M)

    args = (
        A,
        B,
        C,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
    )
    strides = (
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
    )
    # num_valid_tokens = A.size(0): top_k = 1, each row is one (token, expert) pair.

    # BLOCK_SIZE_N IS FREE TO RETUNE, unlike BLOCK_SIZE_M/K. BLOCK_SIZE_K sets the K-tiling and hence the
    # fp32 accumulation order -- the thing batch-invariance pins -- and BLOCK_SIZE_M additionally feeds
    # the host-side block map. BLOCK_SIZE_N only splits the OUTPUT columns across programs: every output
    # element still sees the same K-loop from the same zero accumulator, so widening it is bitwise. It
    # stays a fixed constant keyed on n_leaves, never on the token count.
    if n_leaves >= 4 and (K // n_leaves) % BLOCK_K == 0 and N % _STREAM_BLOCK_N == 0:
        BLOCK_N = _STREAM_BLOCK_N
        grid = (triton.cdiv(EM, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        fused_moe_leaftree_kernel_stream[grid](
            *args,
            EM,
            A.size(0),
            *strides,
            K=K,
            BLOCK_SIZE_M=BLOCK_M,
            BLOCK_SIZE_N=BLOCK_N,
            BLOCK_SIZE_K=BLOCK_K,
            GROUP_SIZE_M=config["GROUP_SIZE_M"],
            N_LEAVES=n_leaves,
            compute_type=compute_type,
            num_warps=_STREAM_NUM_WARPS,
            num_stages=_STREAM_NUM_STAGES,
        )
        return

    grid = lambda META: (triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)  # noqa: E731
    common = dict(
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=BLOCK_K,
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        N_LEAVES=n_leaves,
        compute_type=compute_type,
    )
    if n_leaves >= 4:
        # constexpr-K variant: fully unrolled loop nest.
        fused_moe_leaftree_kernel_unrolled[grid](*args, EM, A.size(0), *strides, K=K, **common)
    else:
        fused_moe_leaftree_kernel[grid](*args, K, EM, A.size(0), *strides, WIRE_STORE=wire_store, **common)
