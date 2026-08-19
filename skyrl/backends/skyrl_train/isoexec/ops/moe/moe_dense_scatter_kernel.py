"""MoE routing MECHANICS around a pinned sigmoid+bias (``noaux_tc``) score chain.

The score chain itself -- sigmoid, bias-steered top-k, gather, normalisation, scaling -- is neither
replaced nor re-derived: the fused path calls megatron's own ``topk_routing_with_score_function`` with
``dense_output=True`` and takes its early return, so ``top_indices`` (membership AND column order, ties
included) comes from exactly the bytes that produce it today. What is replaced is the integer and ordering
work around it -- the zeros+scatter dense build and the transpose + radix-argsort permute -- which performs
no floating-point arithmetic at all and is therefore bit-free by construction.

Re-deriving the top-k here would NOT be safe. The selection key is ``scores + expert_bias`` while the
denominator sums ``gather(scores, top_indices)``, so two experts can tie exactly in the key while their
raw scores differ; permuting them permutes unequal entries of ATen's fixed positional row-sum tree and
changes every prob in the row. For the same reason ``SKYRL_ISOEXEC_MOE_DETERMINISTIC``'s ``sorted=True``
forcing is a correctness requirement on this router rather than a determinism nicety -- megatron asks for
``sorted=torch.is_grad_enabled()``, so without it the engine and trainer forwards see different column
orders.

Two flags, both default OFF and both engine-only on instances marked by
:func:`mark_engine_routing_mechanics`: ``SKYRL_ISOEXEC_MOE_DENSE_SCATTER`` (:func:`fused_dense_from_topk`,
one kernel for the dense probs/map build) and ``SKYRL_ISOEXEC_MOE_PERMUTE_SORT`` (reuses
``moe_router_o2_kernel``'s ``fused_permute_index`` counting sort, which requires ``ep_size == 1``; see
:func:`permute_sort_active`). Neither kernel has an ``autograd.Function``, so on a colocated trainer they
would sever the MoE backward -- the instance mark is what scopes the class patch to the engine.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False


_ENV_DENSE = "SKYRL_ISOEXEC_MOE_DENSE_SCATTER"
_ENV_SORT = "SKYRL_ISOEXEC_MOE_PERMUTE_SORT"


def dense_scatter_enabled() -> bool:
    """Default OFF. Re-read per call so an in-process A/B can flip it between forwards."""
    return os.environ.get(_ENV_DENSE, "0") == "1"


def permute_sort_enabled() -> bool:
    """Default OFF. Re-read per call so an in-process A/B can flip it between forwards."""
    return os.environ.get(_ENV_SORT, "0") == "1"


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


# Pinned launch shape. This kernel has no reduction and no arithmetic, so `num_warps` cannot move a
# rounding tree; it is pinned anyway so the configuration the bitwise proof was taken at is the one that
# runs.
_DENSE_WARPS = 1

# `K` drives `tl.static_range`, so the store loop is fully unrolled and compile time is O(k). Past this
# bound a compile is indistinguishable from a hang at engine init.
_MAX_K = 32
# A program keeps a whole dense row in registers; beyond this it spills.
_MAX_BLOCK_E = 8192
# Triton row offsets are int32 by default; past 2^31 elements that arithmetic wraps and the stores hit
# arbitrary memory.
_MAX_ELEMS = 2**31


if HAVE_TRITON:

    @triton.jit
    def _dense_from_topk_kernel(
        IDX,  # [T, K] int64   top_indices, straight from torch.topk -- NOT re-derived
        PROBS,  # [T, K] fp32  the finished routing weights -- NOT recomputed
        RPROBS,  # [T, E] fp32  out
        RMAP,  # [T, E] int8   out (viewed as bool by the caller)
        T,
        E,
        K: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        """``zeros_like(logits).scatter(1, idx, probs)`` and its boolean twin, in one pass.

        NO ARITHMETIC: ``tl.where`` is a select, so the fp32 payload is multiplexed, never added,
        multiplied or converted, and NaN/subnormal payloads pass through as the exact bit patterns the
        score chain produced. The zero background is a true ``+0.0``, matching ``torch.zeros_like``.
        """
        t = tl.program_id(0)
        cols = tl.arange(0, BLOCK_E)
        cmask = cols < E

        dprob = tl.zeros([BLOCK_E], tl.float32)
        dmap = tl.zeros([BLOCK_E], tl.int8)
        # K is constexpr -> fully unrolled, everything stays in registers. torch.topk returns K
        # DISTINCT columns, so no two iterations can target the same lane and the "last write wins"
        # question that makes torch.scatter order-dependent on duplicate indices never arises.
        for j in tl.static_range(K):
            ij = tl.load(IDX + t * K + j)
            pj = tl.load(PROBS + t * K + j)
            hit = cols == ij
            dprob = tl.where(hit, pj, dprob)
            dmap = tl.where(hit, 1, dmap).to(tl.int8)

        tl.store(RPROBS + t * E + cols, dprob, mask=cmask)
        tl.store(RMAP + t * E + cols, dmap, mask=cmask)


def dense_can_handle(top_indices: torch.Tensor, probs: torch.Tensor, num_experts: int) -> bool:
    """Host-side, shape-only envelope check. No device sync, CUDA-graph safe.

    Refuses rather than raises: a bare ``assert`` inside a vLLM worker kills the Ray actor, and the
    fallback is the eager sequence, which is bitwise-identical by definition.
    """
    if not HAVE_TRITON:
        return False
    if top_indices.dim() != 2 or probs.dim() != 2:
        return False
    if not top_indices.is_cuda or not probs.is_cuda:
        return False
    if top_indices.shape != probs.shape:
        return False
    if top_indices.dtype is not torch.int64:
        return False
    if probs.dtype is not torch.float32:
        return False  # the router is fp32 end to end; a bf16 payload is a different contract
    T, K = top_indices.shape
    E = int(num_experts)
    if T == 0 or E == 0 or K == 0 or K > E:
        return False
    if K > _MAX_K:
        return False
    if _next_pow2(E) > _MAX_BLOCK_E:
        return False
    if T * E >= _MAX_ELEMS:
        return False
    return True


@torch.no_grad()
def fused_dense_from_topk(
    top_indices: torch.Tensor,
    probs: torch.Tensor,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(routing_probs [T,E] fp32, routing_map [T,E] bool)`` from the compact top-k form.

    Bitwise-equal to :func:`ref_dense_from_topk` by construction -- it moves the same fp32 words to
    the same addresses. Falls back to that reference outside :func:`dense_can_handle`'s envelope.
    """
    if not dense_can_handle(top_indices, probs, num_experts):
        return ref_dense_from_topk(top_indices, probs, num_experts)
    T, K = top_indices.shape
    E = int(num_experts)
    idx = top_indices.contiguous()
    prb = probs.contiguous()
    rprobs = torch.empty(T, E, dtype=torch.float32, device=idx.device)
    rmap8 = torch.empty(T, E, dtype=torch.int8, device=idx.device)
    _dense_from_topk_kernel[(T,)](
        idx,
        prb,
        rprobs,
        rmap8,
        T,
        E,
        K=K,
        BLOCK_E=_next_pow2(E),
        num_warps=_DENSE_WARPS,
        # Structurally vacuous -- this kernel has no `acc += b*c` for FFMA contraction to reach -- but
        # set anyway so the launch configuration matches every other kernel here.
        enable_fp_fusion=False,
    )
    return rprobs, rmap8.view(torch.bool)


def ref_dense_from_topk(top_indices, probs, num_experts):
    """megatron's dense build, transcribed. Gate target AND fallback."""
    T = top_indices.shape[0]
    zeros = torch.zeros(T, int(num_experts), dtype=probs.dtype, device=probs.device)
    routing_probs = zeros.scatter(1, top_indices, probs)
    routing_map = zeros.int().scatter(1, top_indices, 1).bool()
    return routing_probs, routing_map


_installed = False
_orig_routing = None
_orig_topk_routing = None
_marked = 0

# Set by the patched ``TopKRouter.routing`` for the duration of ONE synchronous routing call and read by
# the wrapper around ``topk_routing_with_score_function``. That module-level function has no ``self``, so
# this is how the engine instance mark reaches it; single-threaded, the same assumption
# ``moe_batch_invariant._make_sorted_topk_routing`` relies on when it swaps ``torch.topk`` globally.
_dense_scope = False


def _dense_active(router) -> bool:
    """This routing call takes the fused dense build iff flag + engine mark + validated envelope.

    Written for the sigmoid/``noaux_tc`` family. Refuses token dropping (which rewrites the dense tensors
    afterwards), group top-k, sinkhorn, a router replay and TE's fused router -- configurations whose
    routing chain is not the one gated here.
    """
    if not dense_scatter_enabled():
        return False
    if not getattr(router, "_isoexec_routing_mechanics", False):
        return False
    cfg = router.config
    return (
        getattr(router, "score_function", None) in ("sigmoid", "sqrtsoftplus", "softmax")
        and getattr(cfg, "moe_router_group_topk", None) in (None, 0)
        and getattr(cfg, "moe_expert_capacity_factor", None) is None
        and not getattr(cfg, "moe_router_fusion", False)
        and getattr(router, "routing_type", "topk") != "sinkhorn"
        and getattr(router, "router_replay", None) is None
    )


def install_dense_scatter() -> bool:
    """Patch ``TopKRouter.routing`` + ``topk_routing_with_score_function``. Idempotent, both sides.

    Two patches because the mark lives on the instance and the work lives in a module-level function:
    ``routing`` publishes "this call is a marked engine instance", and the wrapper around
    ``topk_routing_with_score_function`` takes megatron's ``dense_output=True`` early return and finishes
    the build in Triton. Every unmarked instance -- i.e. every trainer router -- reaches neither.

    The wrapper is installed OVER whatever is currently bound, so
    ``moe_batch_invariant._make_sorted_topk_routing``'s ``sorted=True`` forcing (which must already be in
    place) still applies to the top-k inside.
    """
    global _installed, _orig_routing, _orig_topk_routing
    if _installed:
        return True
    try:
        from megatron.core.transformer.moe import router as _router_mod
        from megatron.core.transformer.moe.router import TopKRouter
    except Exception:  # pragma: no cover
        return False

    _orig_routing = TopKRouter.routing
    _orig_topk_routing = _router_mod.topk_routing_with_score_function

    def _routing(self, logits, padding_mask=None, **kw):
        global _dense_scope
        want = padding_mask is None and not (self.training and torch.is_grad_enabled()) and _dense_active(self)
        prev, _dense_scope = _dense_scope, want
        try:
            return _orig_routing(self, logits, padding_mask=padding_mask, **kw)
        finally:
            _dense_scope = prev

    def _topk_routing(*args, **kwargs):
        if not _dense_scope or kwargs.get("dense_output", False):
            return _orig_topk_routing(*args, **kwargs)
        # megatron's OWN score chain, megatron's OWN arguments, stopped one step early.
        probs, top_indices = _orig_topk_routing(*args, dense_output=True, **kwargs)
        num_experts = args[0].shape[1]
        if not dense_can_handle(top_indices, probs, num_experts):
            # Outside the envelope: finish it the eager way. Same tensors, same bits.
            return ref_dense_from_topk(top_indices, probs, num_experts)
        return fused_dense_from_topk(top_indices, probs, num_experts)

    TopKRouter.routing = _routing
    _router_mod.topk_routing_with_score_function = _topk_routing
    _installed = True
    return True


def permute_sort_active(disp) -> bool:
    """The dispatcher takes the counting-sort index build iff flag + engine mark + EP=1.

    THE EP GUARD IS LOAD-BEARING: megatron's ``argsort(descending=True)[:T*topk]`` only equals "the True
    positions in ascending flat index" when this rank owns every expert. At EP>1 the slice spills into the
    False region and the counting sort would leave that tail uninitialised -- a silent garbage read, not a
    rounding difference.
    """
    return (
        permute_sort_enabled()
        and getattr(disp, "_isoexec_routing_mechanics", False)
        and getattr(disp, "ep_size", 1) == 1
    )


def mark_engine_routing_mechanics(model) -> int:
    """Mark every MoE router AND token dispatcher in the ENGINE model. Never called on the trainer.

    With ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` the class patches reach both runtimes; the instance mark is
    what scopes them to one.
    """
    global _marked
    n_r = n_d = 0
    for m in model.modules():
        r = getattr(m, "router", None)
        if r is not None and hasattr(r, "topk"):
            r._isoexec_routing_mechanics = True
            n_r += 1
        d = getattr(m, "token_dispatcher", None)
        if d is not None:
            d._isoexec_routing_mechanics = True
            n_d += 1
    _marked = n_r
    print(
        f"[ISOEXEC-MOE] ENGINE routing mechanics marked on {n_r} routers / {n_d} dispatchers; "
        f"{_ENV_DENSE}={'ON (1 kernel for the dense build)' if dense_scatter_enabled() else 'OFF (eager zeros+scatter x2)'}; "
        f"{_ENV_SORT}={'ON (counting sort)' if permute_sort_enabled() else 'OFF (eager transpose+radix argsort)'}",
        flush=True,
    )
    return n_r
