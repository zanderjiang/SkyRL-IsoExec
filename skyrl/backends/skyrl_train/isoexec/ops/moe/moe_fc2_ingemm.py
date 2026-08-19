"""fc2 leaf tree INSIDE the indexed expert GEMM (``SKYRL_ISOEXEC_MOE_FC2_INGEMM``, default OFF).

Combines ``moe_indexed_bmm``'s gather-free B-operand indexing with ``moe_leafcombine_kernel``'s fused leaf
fold: one launch, no staging copies, no per-leaf output buffers and one fp32 store, replacing the buffer
path's per-leaf ``.contiguous()`` slices, weight gather, ``G`` bmm launches and separate fold.

WHERE THE ROUNDING LIVES is the whole difficulty. The buffer path's leaf ``i`` is an ``aten::bmm`` whose
result dtype is the operand dtype, so vLLM's ``bmm_kernel`` accumulates that leaf's K-range in fp32 and
ROUNDS ONCE to bf16 at its store; only afterwards does the fold promote back to fp32 and add. An in-GEMM
tree that folded the raw fp32 accumulators would be MORE accurate and WRONG: it would not match, and the
ETP composition with the engine's tree would break. So the kernel materializes each leaf's rounding in
register -- ``acc.to(operand dtype)``, then back to fp32, which is exact -- and folds that, the same two
steps ``moe_fused_leaftree`` performs on the engine side.

The rest follows the two predecessors: the per-leaf K-loop is ``bmm_kernel``'s K-loop at ``K = L`` from a
zero accumulator, with the same ``BLOCK_SIZE_K`` stepping, the same ``tl.dot(a, b, acc)`` form and the same
tail mask (``L`` is NOT assumed to divide ``BLOCK_SIZE_K``, and reproducing that masked tail exactly is
load-bearing); the B batch index is ``idx[pid_b]``, so the weight stack is never gathered; and the fold is
the trailing-one streaming merge that reproduces ``_tree_sum``'s balanced tree, with ``s0`` held in the
narrow dtype because a level-0 partial is always a bare leaf.

ADDRESSING IS OWNED EXPLICITLY. ``bmm``'s ``A_LARGE``/``B_LARGE``/``C_LARGE`` int32->int64 switches key off
``tensor.numel()``, which bounds addresses only for a CONTIGUOUS operand -- a strided leaf view can address
past 2^31 elements while its ``numel()`` says otherwise. This kernel derives the switches from the
operand's actual addressable reach, ``sum_d (shape[d] - 1) * |stride[d]|``. Widening is value-neutral, so
erring wide is free while erring narrow is an illegal memory access; that is what licenses passing the full
``inter`` and the full weight stack straight to the kernel.

Gradients carry no bitwise contract (IsoExec constrains the forward only). The combine is a linear sum, so
its VJP broadcasts the incoming fp32 grad to every leaf rounded to the leaf dtype, and because the leaves
partition ``K`` exactly the per-leaf bmm VJPs concatenate into the full-K dgrad/wgrad -- so the backward is
``moe_indexed_bmm.indexed_bmm_backward`` applied to that cast gradient, reused verbatim. First use runs a
fail-closed bitwise self-check against the exact production expression; any mismatch or raise disables the
provider permanently and ``_leaftree_fc2`` keeps the buffer path.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False


_ENV_GATE = "SKYRL_ISOEXEC_MOE_FC2_INGEMM"

# Fail-closed provider state (mm_cublaslt discipline): first use runs the bitwise self-check; any
# failure or raise disables the provider PERMANENTLY and callers keep the buffer leaf tree.
_STATE = {"checked": False, "ok": False}

_MAX_LEAVES = 8

#: EXECUTION census, not an admission one. ``fc2_ingemm_ready`` is a PREDICATE: once the self-check has
#: passed it answers the same thing forever, so differencing it across two A/B arms measures zero whichever
#: arm ran. These counters move once per kernel launch, so their difference is the number of times the
#: lever's own code actually executed.
_COUNTS = {"served": 0, "programs": 0, "rows": 0, "leaves": 0, "reported": 0}


def _report_fc2_ingemm() -> None:
    """Emit execution evidence at power-of-two launch counts."""
    served = _COUNTS["served"]
    if served < 1 or (served & (served - 1)) != 0 or served == _COUNTS["reported"]:
        return
    _COUNTS["reported"] = served
    print(
        f"[ISOEXEC-MOE-FC2-INGEMM] pid={os.getpid()} served={served} "
        f"programs={_COUNTS['programs']} rows={_COUNTS['rows']} leaves={_COUNTS['leaves']}",
        flush=True,
    )


def fc2_ingemm_enabled() -> bool:
    """Default OFF until live-gated. Read per call so an in-process A/B can flip it."""
    return os.environ.get(_ENV_GATE, "0") == "1"


if HAVE_TRITON:

    @triton.jit
    def bmm_kernel_indexed_leaftree(
        a_ptr,  # (*, ) A, (B, M, K) -- the FULL moe_intermediate, never a per-leaf slice
        b_ptr,  # (*, ) B, (E, K, N) -- the expert-weight STACK, never a gathered copy
        idx_ptr,  # (*, ) int64 [B] tile -> expert map (ignored when HAS_IDX is False)
        c_ptr,  # (*, ) C, (B, M, N) fp32 -- this rank's leaf-subtree root
        B,
        M,
        N,
        stride_ab,
        stride_am,
        stride_ak,
        stride_bb,
        stride_bk,
        stride_bn,
        stride_cb,
        stride_cm,
        stride_cn,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        A_LARGE: tl.constexpr,
        B_LARGE: tl.constexpr,
        C_LARGE: tl.constexpr,
        N_LEAVES: tl.constexpr,
        K_LEAF: tl.constexpr,
        HAS_IDX: tl.constexpr,
    ):
        """``C[t] = tree_sum_i bf16( A[t, :, iL:(i+1)L] @ B[idx[t], iL:(i+1)L, :] )`` in one pass.

        Structurally ``bmm_kernel_indexed`` (itself vLLM's ``bmm_kernel`` with the one B-pointer line
        changed) wrapped in an unrolled leaf loop with the streaming balanced fold. Everything that can
        reach a VALUE is unchanged from that kernel: the tile decomposition, the row/col masks, the
        ``BLOCK_SIZE_K`` stepping, the zero-initialized fp32 accumulator, the ``tl.dot(a, b, accumulator)``
        form, the tail mask ``k_valid``, and the single RNE round of the accumulator to the operand dtype.
        What is added is that the accumulator is reset per leaf and its rounded value folded in register
        instead of stored, and that the K offsets carry the leaf's base ``leaf * K_LEAF``.

        ``K_LEAF`` is ``tl.constexpr`` so the whole leaf/K nest is ``tl.static_range`` and flattens into
        one straight-line instruction stream, letting the loads prefetch ACROSS leaf boundaries. It costs
        one JIT specialization per distinct moe_intermediate shard width.
        """
        pid_b = tl.program_id(0)
        pid = tl.program_id(1)

        if pid_b >= B:
            return

        # number of tiles along M / N
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

        if pid_m >= num_pid_m or pid_n >= num_pid_n:
            return

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_m = offs_m < M
        mask_n = offs_n < N

        if A_LARGE or B_LARGE or C_LARGE:
            offs_m = offs_m.to(tl.int64)
            offs_n = offs_n.to(tl.int64)

        offs_m = tl.where(mask_m, offs_m, 0)
        offs_n = tl.where(mask_n, offs_n, 0)

        offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N)

        a_batch_ptr = a_ptr + pid_b.to(tl.int64) * stride_ab
        if HAS_IDX:
            # the gather-free indirection: the expert weight stack indexed per PROGRAM.
            b_batch_ptr = b_ptr + tl.load(idx_ptr + pid_b) * stride_bb
        else:
            # tile id == expert id (the one-tile-per-expert grid); caller passes the matching slice.
            b_batch_ptr = b_ptr + pid_b.to(tl.int64) * stride_bb
        c_batch_ptr = c_ptr + pid_b.to(tl.int64) * stride_cb

        offs_k_mask = tl.arange(0, BLOCK_SIZE_K)
        # k-blocks per leaf, ceil: K_LEAF need NOT be a multiple of BLOCK_SIZE_K (at the 35B shard
        # width it is 96 against 64, so the second block of every leaf is a 32-lane masked tail --
        # exactly what the per-leaf bmm at K=96 does, and reproducing it is load-bearing).
        KB_PER_LEAF: tl.constexpr = (K_LEAF + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K

        for leaf in tl.static_range(N_LEAVES):
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            for ki in tl.static_range(KB_PER_LEAF):
                if A_LARGE or B_LARGE:
                    offs_k = leaf * K_LEAF + ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
                else:
                    offs_k = leaf * K_LEAF + ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

                a_ptrs = a_batch_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
                b_ptrs = b_batch_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

                # valid K lanes WITHIN THIS LEAF -- the tail mask of a K=K_LEAF bmm, so the lanes
                # that would spill into leaf+1 read as 0.0 and never enter this leaf's dot.
                k_valid = offs_k_mask < (K_LEAF - ki * BLOCK_SIZE_K)
                a_mask = mask_m[:, None] & k_valid[None, :]
                b_mask = k_valid[:, None] & mask_n[None, :]

                a = tl.load(a_ptrs, mask=a_mask, other=0.0)
                b = tl.load(b_ptrs, mask=b_mask, other=0.0)
                accumulator = tl.dot(a, b, accumulator)

            # THE ROUNDING THAT MAKES THIS BIT-EXACT. The buffer path's leaf is a bf16 TENSOR: the
            # bmm rounds its fp32 accumulator once at the store, and the fold sums those rounded
            # values. Fold the raw accumulator instead and the result is more accurate and wrong.
            vc = accumulator.to(a_ptr.dtype.element_ty)  # RNE, the same conversion bmm_kernel emits
            v = vc.to(tl.float32)  # exact, as _tree_sum's .float() is exact

            # streaming balanced fold: merge leaf i into the pending stack by the trailing-one bits
            # of i, earlier partial on the LEFT -- ((l0+l1)+(l2+l3)) + ((l4+l5)+(l6+l7)). Every
            # branch is on a Python int (tl.static_range), so tracing prunes them to straight line.
            if (leaf & 1) == 1:
                v = s0.to(tl.float32) + v  # noqa: F821  (exact: s0 is a BARE leaf, already narrow)
            if (leaf & 3) == 3:
                v = s1 + v  # noqa: F821
            if (leaf & 7) == 7:
                v = s2 + v  # noqa: F821
            if leaf == N_LEAVES - 1:
                result = v
            elif (leaf & 1) == 0:
                s0 = vc  # noqa: F841  (held in the narrow dtype -- lossless, halves the tile)
            elif (leaf & 3) == 1:
                s1 = v  # noqa: F841
            else:
                s2 = v  # noqa: F841

        c_m = offs_m
        c_n = offs_n
        if C_LARGE:
            c_m = c_m.to(tl.int64)
            c_n = c_n.to(tl.int64)

        c_ptrs = c_batch_ptr + stride_cm * c_m[:, None] + stride_cn * c_n[None, :]
        c_mask = mask_m[:, None] & mask_n[None, :]
        # fp32 store: C is this rank's leaf-subtree PARTIAL, which pik's cross-rank tree consumes.
        tl.store(c_ptrs, result, mask=c_mask)


def _reach(t: torch.Tensor) -> int:
    """Largest linear ELEMENT offset any program can form for ``t`` -- ``sum_d (shape-1)*|stride|``.

    This is the bound ``numel()`` is standing in for in vLLM's ``*_LARGE`` switches, and the two
    agree exactly for a contiguous tensor. They do NOT agree for a strided view (a K-slice of a
    [B, M, K] tensor has a small numel and the full tensor's reach), which is precisely why the
    buffer path had to ``.contiguous()`` every leaf before handing it to the bmm. Owning this lets
    the kernel take the un-staged operands.
    """
    return sum((s - 1) * abs(st) for s, st in zip(t.shape, t.stride()) if s > 0)


def _bmm_config(dtype):
    """vLLM's batch-invariant bmm block sizes for ``dtype`` -- shared with ``moe_indexed_bmm`` so
    the two kernels cannot drift apart (identical BLOCK_SIZE_K is a precondition of bit-identity)."""
    from .moe_indexed_bmm import bmm_launch_config

    return bmm_launch_config(dtype)


# TILE GEOMETRY: a SCHEDULE knob on a FIXED arithmetic.
#
# This kernel inherits its launch geometry from ``bmm_launch_config``, which vLLM tuned for a kernel that
# holds ONE fp32 accumulator. The leaf tree holds FOUR live ``[BLOCK_M, BLOCK_N]`` tiles (``accumulator``
# plus the three pending partials; peak liveness log2(N_LEAVES)+1 is intrinsic to a balanced tree over
# in-order leaves), so at the inherited ``[128, 128]`` the tiles alone exhaust the 255-register budget and
# the kernel spills heavily. Narrowing BLOCK_N is what removes the spill.
#
# WHY BLOCK_M/BLOCK_N ARE FREE AND BLOCK_K IS NOT: a block along a REDUCED axis is ARITHMETIC (it moves the
# accumulation order), a block along a non-reduced axis is SCHEDULE. The reduction runs over K only; M and
# N index independent output elements, each of which sees the identical leaf loop from its own zero
# accumulator. BLOCK_SIZE_K is deliberately NOT exposed below.
#
# DEFAULT IS TODAY'S GEOMETRY: every knob is empty by default, so an unset environment resolves to
# ``bmm_launch_config`` byte for byte and this block is inert. Overrides still pass through the same
# fail-closed first-use self-check against the buffer path, so an override can never make a run wrong --
# only faster or slower.
_ENV_BLOCK_M = "SKYRL_ISOEXEC_MOE_FC2_INGEMM_BLOCK_M"
_ENV_BLOCK_N = "SKYRL_ISOEXEC_MOE_FC2_INGEMM_BLOCK_N"
_ENV_WARPS = "SKYRL_ISOEXEC_MOE_FC2_INGEMM_WARPS"
_ENV_STAGES = "SKYRL_ISOEXEC_MOE_FC2_INGEMM_STAGES"

#: config key -> env var, for the banner and the tests. NOTE that the reads themselves are spelled
#: out one per knob below rather than driven from this dict: the registry-faithfulness scanner
#: (core/tests/test_flags_faithful._scan_file) resolves an env NAME only from a literal or a
#: module-level constant, so a loop over this mapping would make four registered flags look unread.
_GEOM_ENV = {
    "BLOCK_SIZE_M": _ENV_BLOCK_M,
    "BLOCK_SIZE_N": _ENV_BLOCK_N,
    "num_warps": _ENV_WARPS,
    "num_stages": _ENV_STAGES,
}


def _apply_geom_override(cfg: dict, key: str, env: str, raw: str) -> None:
    raw = raw.strip()
    if not raw or raw == _AUTO:
        return
    try:
        val = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"[isoexec-moe] {env}={raw!r} is not an integer") from exc
    if val <= 0 or (val & (val - 1)) != 0:
        raise RuntimeError(f"[isoexec-moe] {env}={val} must be a positive power of two")
    cfg[key] = val


# DERIVING the tile instead of pinning it.
#
# A geometry that is right only for one model keeps its flag forever, so the tile comes from a rule. The
# kernel's register pressure is dominated by the LIVE fp32 accumulator tiles: a balanced fold over in-order
# leaves holds ``log2(N_LEAVES) + 1`` of them at peak -- a property of the fold, not of the operands -- and
# each tile is ``BLOCK_M x BLOCK_N`` fp32 spread over ``num_warps * 32`` threads, i.e.
#
#     accumulator registers/thread = BLOCK_M * BLOCK_N * (log2(N_LEAVES) + 1) / (num_warps * 32)
#
# There is a single threshold on that number below which ptxas stops spilling. Measured offline over the
# whole (N_LEAVES x BLOCK_N) grid, every geometry at or under 64 accumulator registers per thread is
# spill-free and every geometry above it spills, so the budget is 64 -- a quarter of ptxas's 255, the rest
# being the kernel's non-accumulator pressure, which grows with the tile and is why the frontier is stated
# on the accumulators rather than on a fixed overhead.
#
# NOTE WHAT THE INPUTS ARE. BLOCK_M, num_warps and dtype come from ``bmm_launch_config``; N_LEAVES is the
# pik tree width. Hidden size, FFN width, expert count and tile capacity appear NOWHERE, so the rule
# transfers to any model this stack serves. What does not transfer is the 64, which is a property of this
# kernel on this architecture and this Triton, so it is verified rather than trusted: an offline test
# compiles the derived geometry at every supported leaf count and asserts ptxas reports zero spill.
# once, for every model at once.
_AUTO = "auto"
_PTXAS_REG_BUDGET = 255
_ACCUM_REG_BUDGET = 64  # measured spill frontier; see the derivation above
_MIN_BLOCK_N = 16  # below this the wgmma N-extent stops paying for the extra programs


def live_accumulator_tiles(n_leaves: int) -> int:
    """Peak live ``[BLOCK_M, BLOCK_N]`` fp32 tiles for a balanced fold over ``n_leaves`` leaves."""
    return max(1, int(n_leaves).bit_length() - 1) + 1


def accumulator_regs(block_m: int, block_n: int, num_warps: int, n_leaves: int) -> float:
    """Registers/thread consumed by the live accumulator tiles alone."""
    return block_m * block_n * live_accumulator_tiles(n_leaves) / (num_warps * 32)


def derived_block_n(block_m: int, block_n: int, num_warps: int, n_leaves: int) -> int:
    """Widest power-of-two ``BLOCK_SIZE_N`` (<= the inherited one) whose accumulators fit ptxas.

    Pure, model-free, and total: it never widens past the geometry vLLM tuned, and it never goes
    below ``_MIN_BLOCK_N`` even if the budget says it should -- a shape that cannot fit at 16 is a
    shape whose spill needs re-deriving, not one to keep halving.
    """
    while block_n > _MIN_BLOCK_N and accumulator_regs(block_m, block_n, num_warps, n_leaves) > _ACCUM_REG_BUDGET:
        block_n //= 2
    return block_n


def leaftree_launch_config(dtype, n_leaves: int | None = None) -> dict:
    """``bmm_launch_config(dtype)`` with the SCHEDULE knobs optionally overridden or DERIVED.

    ``BLOCK_SIZE_K`` has no env path on purpose: it is the ARITHMETIC knob (it sets the K-tiling and
    therefore the fp32 accumulation order), and moving it is a different claim with a different
    proof. Everything else here changes which output elements share a program and how the program is
    scheduled, never what any element computes.

    ``SKYRL_ISOEXEC_MOE_FC2_INGEMM_BLOCK_N=auto`` selects the derived tile above instead of a pinned
    integer. The DEFAULT is still the inherited geometry: the derivation removes a spill, and a
    spill-free kernel is not automatically a faster one, so the switch waits on the production-shape
    A/B rather than being assumed. ``n_leaves=None`` (the banner, which has no call in hand) derives
    against the worst supported liveness, which can only choose a narrower tile, never a spilling one.
    """
    cfg = dict(_bmm_config(dtype))
    block_n_raw = os.environ.get(_ENV_BLOCK_N, "").strip()
    _apply_geom_override(cfg, "BLOCK_SIZE_M", _ENV_BLOCK_M, os.environ.get(_ENV_BLOCK_M, ""))
    _apply_geom_override(cfg, "BLOCK_SIZE_N", _ENV_BLOCK_N, block_n_raw)
    _apply_geom_override(cfg, "num_warps", _ENV_WARPS, os.environ.get(_ENV_WARPS, ""))
    _apply_geom_override(cfg, "num_stages", _ENV_STAGES, os.environ.get(_ENV_STAGES, ""))
    if block_n_raw == _AUTO:
        cfg["BLOCK_SIZE_N"] = derived_block_n(
            cfg["BLOCK_SIZE_M"],
            cfg["BLOCK_SIZE_N"],
            cfg["num_warps"],
            _MAX_LEAVES if n_leaves is None else n_leaves,
        )
    return cfg


def leaftree_geometry(dtype, n_leaves: int | None = None) -> str:
    """One-line geometry description for the banner -- an operator must never have to infer which
    tile actually ran from which environment variables happened to be exported."""
    c = leaftree_launch_config(dtype, n_leaves)
    deviates = any(os.environ.get(env, "").strip() for env in _GEOM_ENV.values())
    derived = os.environ.get(_ENV_BLOCK_N, "").strip() == _AUTO
    origin = " (derived)" if derived else (" (OVERRIDDEN)" if deviates else " (inherited)")
    return (
        f"[{c['BLOCK_SIZE_M']},{c['BLOCK_SIZE_N']},{c['BLOCK_SIZE_K']}] w{c['num_warps']} s{c['num_stages']}" + origin
    )


def leaftree_shape_supported(a: torch.Tensor, w: torch.Tensor, n_leaves: int) -> bool:
    """Preconditions for engaging the in-GEMM tree. Anything else -> the buffer path, unchanged.

    ``K % n_leaves == 0`` is required rather than tolerated: the buffer path computes
    ``lf = f // n_leaves`` and its leaves cover only ``[0, lf*n_leaves)``, SILENTLY DROPPING a
    ragged tail. Rather than reproduce that, refuse the shape -- the trainer never hits it (shard
    widths are powers of two times the leaf count) and refusing is fail-closed.
    """
    if not HAVE_TRITON:
        return False
    if n_leaves < 2 or n_leaves > _MAX_LEAVES or (n_leaves & (n_leaves - 1)) != 0:
        return False
    if a.ndim != 3 or w.ndim != 3 or not a.is_cuda:
        return False
    if a.dtype != w.dtype or a.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    K = a.shape[2]
    return K == w.shape[1] and K % n_leaves == 0


def bmm_indexed_leaftree(a: torch.Tensor, w: torch.Tensor, idx, n_leaves: int, *, config=None) -> torch.Tensor:
    """(B, M, K) x (E, K, N)[idx] -> fp32 (B, M, N), leaf tree folded in register.

    ``idx=None`` means tile id == expert id, i.e. ``w`` is already this chunk's slice (the
    one-tile-per-expert grid); the kernel then uses ``pid_b`` directly and no map is read.

    ``config`` overrides the resolved launch geometry for THIS call only. It exists for the A/B
    bench, which must not re-read the environment inside a timed region (host work between two CUDA
    events can drain the launch queue and be charged to the kernel). Production passes None.
    """
    assert HAVE_TRITON, "moe_fc2_ingemm requires triton"
    B, M, K = a.shape
    N = w.shape[2]
    if idx is not None:
        if idx.dtype != torch.long:
            idx = idx.long()
        idx = idx.contiguous()

    c = torch.empty((B, M, N), device=a.device, dtype=torch.float32)
    if B == 0 or M == 0 or N == 0:
        return c

    cfg = dict(config) if config is not None else leaftree_launch_config(a.dtype, n_leaves)
    grid = (
        B,
        triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"]),
    )

    bmm_kernel_indexed_leaftree[grid](
        a,
        w,
        idx,
        c,
        B,
        M,
        N,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        # REACH, not numel: see _reach. Widening is value-neutral, narrowing is an OOB read.
        A_LARGE=_reach(a) > 2**31,
        B_LARGE=_reach(w) > 2**31,
        C_LARGE=_reach(c) > 2**31,
        N_LEAVES=n_leaves,
        K_LEAF=K // n_leaves,
        HAS_IDX=idx is not None,
        **cfg,
    )
    _COUNTS["served"] += 1
    _COUNTS["programs"] += grid[0] * grid[1]
    _COUNTS["rows"] += B * M
    _COUNTS["leaves"] += n_leaves
    _report_fc2_ingemm()
    return c


# autograd: the leaf tree's VJP composed with the indexed bmm's, which is just the indexed bmm's
class _IndexedLeafTreeBmm(torch.autograd.Function):
    """Forward = the fused in-GEMM tree. Backward = ``indexed_bmm_backward`` on ``g.to(A.dtype)``.

    The cast IS the tree's VJP: the fold is a plain sum of the fp32 promotions of the leaves, so every leaf
    receives the incoming gradient unchanged, rounded back to the leaf dtype -- exactly what ``add``'s
    identity backward composed with ``.float()``'s cast-back produces on the buffer path. And because the
    leaves PARTITION K, the per-leaf bmm VJPs concatenate into the single full-K dgrad/wgrad pair the
    indexed bmm already implements, so there is no leaf loop in the backward at all.

    The saved set is ``(a, w, idx)`` -- the moe_intermediate and the weight STACK -- where the buffer path
    saved a contiguous A-slice and B-slice per leaf off a gathered ``[chunk, f, h]``.
    """

    @staticmethod
    def forward(ctx, a, w, idx, n_leaves):
        ctx.save_for_backward(a, w, idx)
        ctx.leaf_dtype = a.dtype
        return bmm_indexed_leaftree(a, w, idx, n_leaves)

    @staticmethod
    def backward(ctx, g):
        from .moe_indexed_bmm import indexed_bmm_backward

        a, w, idx = ctx.saved_tensors
        # the tree's VJP: broadcast the fp32 root grad to every leaf, rounded to the leaf dtype.
        gl = g.to(ctx.leaf_dtype).contiguous()
        da, dw = indexed_bmm_backward(a, w, idx, gl, need_da=ctx.needs_input_grad[0], need_dw=ctx.needs_input_grad[1])
        return da, dw, None, None


def ingemm_leaftree_fc2(inter: torch.Tensor, w2: torch.Tensor, idx, n_leaves: int) -> torch.Tensor:
    """Differentiable fp32 leaf-tree fc2 with the expert stack indexed in-kernel."""
    return _IndexedLeafTreeBmm.apply(inter, w2, idx, n_leaves)


# fail-closed first-use self-check
def _bitcmp(x: torch.Tensor, y: torch.Tensor) -> int:
    """Bit-pattern mismatch count (``torch.equal`` is blind to signed zero)."""
    return int((x.view(torch.int32) != y.view(torch.int32)).sum().item())


def _buffer_path_ref(inter, w2_slice, n_leaves):
    """``moe_batched_experts._leaftree_fc2``'s buffer arm, copied so the check compares against the
    expression it must replace rather than a paraphrase of it: per-leaf ``.contiguous()`` slices,
    per-leaf ``aten::bmm`` (the batch-invariant kernel, bf16 out), fp32 promote, balanced tree."""
    f = w2_slice.shape[1]
    lf = f // n_leaves
    ic = inter.contiguous()
    leaves = [
        torch.ops.aten.bmm(
            ic[:, :, i * lf : (i + 1) * lf].contiguous(), w2_slice[:, i * lf : (i + 1) * lf, :].contiguous()
        )
        for i in range(n_leaves)
    ]
    nodes = [leaf.float() for leaf in leaves]
    while len(nodes) > 1:
        nodes = [nodes[i] + nodes[i + 1] if i + 1 < len(nodes) else nodes[i] for i in range(0, len(nodes), 2)]
    return nodes[0]


@torch.no_grad()
def _self_check(device) -> bool:
    """Bitwise: the fused kernel == per-leaf bmm on gathered/staged slices, folded by ``_tree_sum``.

    The cases are chosen so a wrong ROUNDING SITE and a wrong TREE ORDER are both observable:
      * ``K_LEAF`` deliberately NOT a multiple of ``BLOCK_SIZE_K`` (96 vs 64 -- the production 35B
        shard width) so every leaf carries a masked tail; a kernel that streamed K straight through
        the leaf boundaries would pass on aligned widths and fail here.
      * leaf magnitudes are spread across exponents by construction (the K-range of leaf i is scaled
        by 10^(i-4) through ``w2``), so folding in the wrong order, or folding raw fp32 accumulators
        instead of their bf16 rounds, perturbs the low bits and is CAUGHT. Equal-magnitude randoms
        would hide both.
      * repeated and absent experts in ``idx``; a strided (transposed-stack) B, as production; the
        ``idx=None`` arm; leaf counts G/ETP for ETP in {1, 2, 4}; and the fp32 operand arm.
    """
    torch.manual_seed(0)
    cases = [
        # (E, B, M, K, N, n_leaves, dtype) -- K/n_leaves = 96 is the production ragged leaf width
        (8, 19, 128, 768, 2048, 8, torch.bfloat16),  # production-shaped: ragged L, strided B
        (16, 16, 128, 512, 384, 8, torch.bfloat16),  # L=64 == BLOCK_SIZE_K exactly (no tail)
        (8, 3, 64, 768, 112, 8, torch.bfloat16),  # small M, ragged N, n_tiles < E
        (8, 19, 128, 768, 256, 4, torch.bfloat16),  # ETP=2 rank
        (8, 5, 128, 768, 256, 2, torch.bfloat16),  # ETP=4 rank
        (8, 19, 128, 768, 112, 8, torch.float32),  # fp32 operand arm (BLOCK_SIZE_K=32)
    ]
    for E, B, M, K, N, n_leaves, dtype in cases:
        # param layout [E, N, K] stacked contiguous then transposed -> strides (N*K, 1, K), as
        # production builds it; the leaf slices off THIS are what the buffer path has to stage.
        stack = torch.randn(E, N, K, device=device, dtype=dtype) * 0.05
        lf = K // n_leaves
        for i in range(n_leaves):
            stack[:, :, i * lf : (i + 1) * lf] *= 10.0 ** (i - 4)  # spread leaf exponents
        w = stack.transpose(1, 2)  # [E, K, N]
        a = torch.randn(B, M, K, device=device, dtype=dtype) * 0.05
        # The `cap`-padded rows the stager leaves behind are EXACTLY zero, and a zero row can drive
        # an accumulator to -0.0 (a masked or zero lane contributes -0.0 whenever its weight is
        # negative). -0.0 is the one value for which `acc + 0.0 != acc`, so any candidate geometry
        # that changed how many zero lanes enter a dot would show up here and nowhere else. Half of
        # the last tile is zeroed to keep that case in every arm of the check.
        a[-1, M // 2 :, :] = 0.0
        idx = torch.randint(0, E, (B,), device=device, dtype=torch.long).sort().values  # expert-major

        ref = _buffer_path_ref(a, w[idx], n_leaves)
        got = bmm_indexed_leaftree(a, w, idx, n_leaves)
        bad = _bitcmp(got, ref)
        if bad == 0 and B <= E:
            # the idx=None arm: tile id == expert id, w pre-sliced by the caller
            ref2 = _buffer_path_ref(a, w[:B], n_leaves)
            got2 = bmm_indexed_leaftree(a, w[:B], None, n_leaves)
            bad = _bitcmp(got2, ref2)
        if bad:
            print(
                f"[ISOEXEC-MOE-FC2-INGEMM] SELF-CHECK FAIL (E={E},B={B},M={M},K={K},N={N},"
                f"leaves={n_leaves},{dtype}): {bad} mismatched bit patterns vs the buffer leaf "
                "tree. Disabling the in-GEMM fc2 tree (buffer-path fallthrough).",
                flush=True,
            )
            return False
    return True


def fc2_ingemm_ready(device) -> bool:
    """True iff the flag is on AND the one-time bitwise self-check passed on this stack.

    Fail-closed: a failed or raising self-check disables the provider permanently and the caller
    keeps the buffer leaf tree, so the flag can never make a run WRONG, only faster.
    """
    if not fc2_ingemm_enabled() or not HAVE_TRITON:
        return False
    if not _STATE["checked"]:
        _STATE["checked"] = True
        try:
            _STATE["ok"] = _self_check(device)
        except Exception as e:  # noqa: BLE001
            print(
                f"[ISOEXEC-MOE-FC2-INGEMM] self-check raised ({type(e).__name__}: {e}) -- "
                "buffer leaf-tree fallthrough.",
                flush=True,
            )
            _STATE["ok"] = False
        if _STATE["ok"]:
            # Name the OWNER of the fc2 fold explicitly. The two flags cannot compose -- with the
            # tree inside the GEMM there are no leaf buffers left for a combine kernel to read --
            # so say so rather than leaving an operator to infer it from two independent banners.
            combine = os.environ.get("SKYRL_ISOEXEC_MOE_FUSED_LEAFCOMBINE", "0") == "1"
            geom = leaftree_geometry(torch.bfloat16)
            print(
                "[ISOEXEC-MOE-FC2-INGEMM] in-GEMM pik-fc2 leaf tree ENABLED (self-check bit-exact vs "
                "the buffer tree): per fc2 call, 8 bmm launches + 16 staging copies + 1 weight "
                f"gather + 8 leaf buffers + 1 combine -> ONE launch, one fp32 store. Geometry {geom} "
                "-- the self-check above ran AT this geometry, so an override is bit-proven, not "
                "assumed. This provider "
                "now OWNS the fc2 leaf fold on every site (forward and both recompute paths)"
                + (
                    "; SKYRL_ISOEXEC_MOE_FUSED_LEAFCOMBINE=1 is INERT there (no leaf buffers exist "
                    "to fold) and still owns the fold only if this provider falls back."
                    if combine
                    else "."
                ),
                flush=True,
            )
    return _STATE["ok"]
