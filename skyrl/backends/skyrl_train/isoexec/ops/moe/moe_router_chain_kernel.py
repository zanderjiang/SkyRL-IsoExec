"""The two ARITHMETIC ends of the pre-transform MoE router chain, and the index build behind them.

``moe_dense_scatter_kernel`` already replaces the routing MECHANICS without touching a floating-point
operation. This module composes with it and fuses the two arithmetic ends -- the score function plus the
bias add, and the normalisation tail (eps, divide, scaling factor, dense scatter) -- plus the index build
behind them. Every operation fused here is either a copy or a single correctly-rounded IEEE-754 primitive,
and a correctly-rounded primitive has exactly one answer, so there is no association order to disagree
about.

Three things are deliberately NOT fused. ``torch.topk`` is inherited verbatim, ties included: when a bias
steers SELECTION but not the WEIGHTS (DeepSeek ``noaux_tc``), two experts can tie exactly in
``scores + bias`` while their raw scores differ, so permuting a tied group permutes a length-k vector of
UNEQUAL entries through ATen's fixed positional sum tree and the denominator moves in the last ulp. The
denominator itself is that ATen row-sum, whose reduction tree matches no plausible transcription, so this
module calls ``.sum`` and consumes its output -- which is also why the gathered ``[T, k]`` tensor must stay
materialised in its own launch. The gating GEMM is owned elsewhere.

The primitives are spelled with care: ``core.triton_nonftz``'s ``sigmoid`` and ``div_rn`` rather than
Triton's ``/`` (an approximate reciprocal) or ``libdevice.div_rn`` (which flushes subnormal quotients), and
``sqrt.rn.f32`` rather than ``libdevice.sqrt``. ``enable_fp_fusion=False`` on every launch is NOT vacuous
here the way it is in the pure-copy kernels: ``p * scale`` sits next to ``den + eps``, where a contracting
compiler could reach for an FFMA. And none of it is argued into production -- every shape is admitted by
running megatron's own ``topk_routing_with_score_function`` on the live operands and comparing the dense
outputs as BITS, with a positive control that must fail, before the fused path is taken even once.

Nothing here names a model. The wrapper serves the pre-transform score-function family: score functions
that are an ELEMENTWISE map applied to the logits BEFORE selection (``sigmoid``, ``sqrtsoftplus``);
``softmax`` is a row reduction and belongs to ``moe_router_o2_kernel``. ``HAS_DEN`` / ``HAS_EPS`` /
``HAS_SCALE`` are constexpr, derived from megatron's own call arguments, and with all three False
:func:`fused_router_tail` is exactly ``moe_dense_scatter_kernel.fused_dense_from_topk``. ``BLOCK_E`` and
``K`` are derived from ``E`` and ``k``; because no reduction tree here depends on the block shape, ``k``
does NOT have to be a power of two, unlike in ``moe_router_o2_kernel``.

INSTALLATION IS ENGINE-ONLY BY INSTANCE MARK, reusing the ``_isoexec_routing_mechanics`` mark that
``mark_engine_routing_mechanics`` already sets: the class binding is shared with a colocated trainer, these
kernels have no ``autograd.Function``, and a forward-only replacement there would sever the MoE backward
with the gate green throughout. ORDERING TRAP: ``moe_batch_invariant._make_sorted_topk_routing`` forces
``sorted=True`` by swapping ``torch.topk`` AROUND the call it wraps, and this wrapper installs OUTSIDE that
one, so a ``torch.topk`` issued from here is NOT forced -- the transcription passes ``sorted=True``
explicitly, and the per-shape admission (which compares through the forcing wrapper) would fail if that
were ever wrong.

Flags, all default OFF and re-read per call: ``SKYRL_ISOEXEC_MOE_ROUTER_SCORE`` (score fn + bias add),
``SKYRL_ISOEXEC_MOE_ROUTER_TAIL`` (eps + div + scale + dense) and ``SKYRL_ISOEXEC_MOE_PERMUTE_INDEX_1K``
(counts + permute index).
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    from ...core import triton_nonftz as _nonftz

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False


_ENV_SCORE = "SKYRL_ISOEXEC_MOE_ROUTER_SCORE"
_ENV_TAIL = "SKYRL_ISOEXEC_MOE_ROUTER_TAIL"
_ENV_INDEX1K = "SKYRL_ISOEXEC_MOE_PERMUTE_INDEX_1K"

# The PROPERTY this module serves: a score function that is an elementwise map applied to the logits
# BEFORE selection. Named as a set, so adding a member is a one-line, testable change and so the
# refusal below can never be read as "not GLM".
_PRE_TRANSFORM = ("sigmoid", "sqrtsoftplus")

# megatron's own normalisation epsilon, read from here rather than passed so the transcription and the
# reference cannot drift apart silently; the admission gate is what proves it.
_MEGATRON_NORM_EPS = 1e-20

# `K` and the score-fn selector drive `tl.static_range` / constexpr branches, so the loops are fully
# unrolled and compile time is O(k). Past this bound a Triton compile is indistinguishable from a hang at
# engine init.
_MAX_K = 32
# A program keeps a whole dense [E] row in registers; beyond this it spills.
_MAX_BLOCK_E = 8192
# Triton row offsets are int32 by default; past 2**31 elements that arithmetic wraps and the loads read
# arbitrary memory.
_MAX_ELEMS = 2**31
# `fused_permute_index_1k` has every program recompute the per-expert counts, so it reads the [T,E]
# map E times instead of once. That is free at decode (T=128, E=64: 8 KiB per program) and a real
# cost at prefill, so the single-kernel form is admitted only below this many map elements and the
# two-kernel `fused_permute_index` serves everything above it. A SIZE property, not a phase name.
_MAX_INLINE_COUNT_ELEMS = 1 << 18

# PINNED LAUNCH SHAPE. One token per program on one warp -- the configuration every router kernel in
# this tree measured as best, and the configuration these kernels' bitwise checks were taken under.
# There is no cross-lane float reduction anywhere in this module, so `num_warps` cannot move a
# rounding tree; it is pinned anyway, because a kernel whose bitwise proof was taken at one launch
# configuration must not silently run at another.
_SCORE_WARPS = 1
_TAIL_WARPS = 1
_INDEX_WARPS = 4
_INDEX_BLOCK_T = 512


def score_fuse_enabled() -> bool:
    """Default OFF. Re-read per call so an in-process A/B can flip it between forwards."""
    return os.environ.get(_ENV_SCORE, "0") == "1"


def router_tail_enabled() -> bool:
    """Default OFF. Re-read per call so an in-process A/B can flip it between forwards."""
    return os.environ.get(_ENV_TAIL, "0") == "1"


def permute_index_1k_enabled() -> bool:
    """Default OFF. Re-read per call so an in-process A/B can flip it between forwards."""
    return os.environ.get(_ENV_INDEX1K, "0") == "1"


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


# Constexpr selector for the score function. Integers rather than strings because Triton constexpr
# string comparison is not a thing; the mapping is one place and the envelope check keys on it.
_FN_SIGMOID = 0
_FN_SQRTSOFTPLUS = 1
_FN_CODE = {"sigmoid": _FN_SIGMOID, "sqrtsoftplus": _FN_SQRTSOFTPLUS}

# torch.nn.functional.softplus's linear branch: beta=1, threshold=20 -> `where(x > 20, x, log1p(e^x))`.
# Not a tuning constant: it is torch's own default and reproducing it is the point.
_SOFTPLUS_THRESHOLD = 20.0


if HAVE_TRITON:
    # Triton refuses a plain global inside a @jit'ed function (it can only close over constexprs),
    # so the threshold crosses the boundary as one. It is DERIVED from the module-level value above
    # rather than re-typed, so torch's own default has exactly one home.
    _SOFTPLUS_THRESHOLD_TL = tl.constexpr(_SOFTPLUS_THRESHOLD)

    @triton.jit
    def _score_bias_kernel(
        LOGITS,  # [T, E] fp32
        BIAS,  # [E] fp32 (unused when HAS_BIAS is False)
        SCORES,  # [T, E] fp32 out -- the WEIGHT scores, bias-free
        SEL,  # [T, E] fp32 out -- the SELECTION key (unused when HAS_BIAS is False)
        T,
        E,
        stride_l,
        FN: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        """``scores = f(logits)`` and ``sel = scores + bias``, in one pass.

        Reproduces, bit for bit::

            scores = torch.sigmoid(logits.float())                  # or softplus(...).sqrt()
            scores_for_routing = scores + expert_bias.float()

        BOTH outputs are stored because BOTH are consumed downstream by DIFFERENT consumers: the top-k
        reads ``SEL``, the prob gather reads ``SCORES``. That asymmetry is the whole of ``noaux_tc``, and
        it is why the tie argument that licenses ``fused_o2`` does not hold here.

        Every operation is a single correctly-rounded fp32 primitive on one element, taken from
        ``core.triton_nonftz`` wherever the naive Triton spelling is not correctly rounded (``tl.exp``
        lowers to ``ex2.approx``, ``libdevice.sqrt`` is wrong on some normal inputs, ``libdevice.log1p``
        flushes). There is no reduction, so there is no association order for this kernel to get wrong.
        """
        t = tl.program_id(0)
        cols = tl.arange(0, BLOCK_E)
        cmask = cols < E
        x = tl.load(LOGITS + t * stride_l + cols, mask=cmask, other=0.0)

        if FN == 0:  # sigmoid
            s = _nonftz.sigmoid(x)
        else:  # sqrtsoftplus: torch's softplus(beta=1, threshold=20) then sqrt
            sp = tl.where(x > _SOFTPLUS_THRESHOLD_TL, x, _nonftz.log1p(libdevice.exp(x)))
            s = _nonftz.sqrt(sp)

        tl.store(SCORES + t * E + cols, s, mask=cmask)
        if HAS_BIAS:
            b = tl.load(BIAS + cols, mask=cmask, other=0.0)
            # A plain IEEE fp32 add: one rounding, one answer. `enable_fp_fusion=False` on the
            # launch keeps a compiler from contracting it with anything.
            tl.store(SEL + t * E + cols, s + b, mask=cmask)

    @triton.jit
    def _router_tail_kernel(
        IDX,  # [T, K] int64  top_indices, straight from torch.topk -- NOT re-derived
        VALS,  # [T, K] fp32  gather(scores, top_indices) -- NOT recomputed
        DEN,  # [T, 1] fp32  the ATen row-sum -- NOT recomputed
        RPROBS,  # [T, E] fp32 out
        RMAP,  # [T, E] int8  out (viewed as bool by the caller)
        T,
        E,
        EPS,
        SCALE,
        K: tl.constexpr,
        BLOCK_E: tl.constexpr,
        HAS_DEN: tl.constexpr,
        HAS_EPS: tl.constexpr,
        HAS_SCALE: tl.constexpr,
    ):
        """The normalisation tail and the dense build, in one pass.

        Reproduces, bit for bit, megatron's deterministic-algorithms branch::

            probs = scores / (scores.sum(-1, keepdim=True) + 1e-20)   # sum comes in as DEN
            probs = probs * scaling_factor
            routing_probs = zeros_like(logits); routing_probs.index_put_((rows, idx), probs)
            routing_map  = zeros_like(...);     routing_map.index_put_((rows, idx), ones); .bool()

        THE DENOMINATOR IS AN INPUT. ``DEN`` is ATen's own reduction output; this kernel adds the epsilon
        to it and divides by it, and never re-associates it.

        Three roundings, each with one correct answer: ``den + eps`` (IEEE add), ``v / d``
        (``div.rn.f32`` via inline PTX -- Triton's ``/`` is an approximate reciprocal and
        ``libdevice.div_rn`` flushes subnormal quotients, which a router row with a massive-activation
        outlier produces routinely), and ``p * scale`` (IEEE mul). The scatter is a ``tl.where``
        multiplex, not arithmetic.

        ``torch.topk`` returns K DISTINCT columns, so no two unrolled iterations target the same lane and
        the "last write wins" question that makes ``torch.scatter`` order-dependent on duplicate indices
        never arises.
        """
        t = tl.program_id(0)
        cols = tl.arange(0, BLOCK_E)
        cmask = cols < E

        if HAS_DEN:
            d = tl.load(DEN + t)
            if HAS_EPS:
                d = d + EPS
        else:
            d = 1.0

        dprob = tl.zeros([BLOCK_E], tl.float32)
        dmap = tl.zeros([BLOCK_E], tl.int8)
        for j in tl.static_range(K):
            ij = tl.load(IDX + t * K + j)
            v = tl.load(VALS + t * K + j)
            if HAS_DEN:
                v = _nonftz.div_rn(v, d)
            if HAS_SCALE:
                v = v * SCALE
            hit = cols == ij
            dprob = tl.where(hit, v, dprob)
            dmap = tl.where(hit, 1, dmap).to(tl.int8)

        tl.store(RPROBS + t * E + cols, dprob, mask=cmask)
        tl.store(RMAP + t * E + cols, dmap, mask=cmask)

    @triton.jit
    def _permute_index_1k_kernel(
        RMAP,  # [T, E] int8
        RPROBS,  # [T, E] fp32
        SORTED_IDX,  # [T*k] int64 out
        PERM_PROBS,  # [T*k] fp32  out
        COUNTS,  # [E] int64      out
        T,
        E,
        BLOCK_T: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        """``fused_permute_index``'s two kernels as one: counts are recomputed, not read.

        ``moe_router_o2_kernel`` splits this because ``_permute_index_kernel`` needs the EXCLUSIVE PREFIX
        of the per-expert counts, which is a cross-program quantity. Here every program recomputes the
        whole ``[E]`` count vector from the map in registers, so the prefix is a local masked sum and the
        launch count is 1. The cost is an E-fold re-read of the map, which is why the caller admits this
        form only below ``_MAX_INLINE_COUNT_ELEMS`` and runs the two-kernel form above it.

        Pure integer indexing plus one fp32 GATHER: no arithmetic on the probs, their bit patterns are
        loaded and stored. The permutation reproduced is
        ``argsort(map.T.reshape(-1), descending=True, stable=True)[:T*k]``, a stable partition of a
        BOOLEAN key, i.e. a counting sort by definition.
        """
        e = tl.program_id(0)
        ecols = tl.arange(0, BLOCK_E)
        emask = ecols < E

        # pass 1: the whole [E] count vector, in registers
        cnt = tl.zeros([BLOCK_E], tl.int64)
        for t0 in tl.range(0, T, BLOCK_T):
            ts = t0 + tl.arange(0, BLOCK_T)
            blk = tl.load(
                RMAP + ts[:, None] * E + ecols[None, :],
                mask=(ts < T)[:, None] & emask[None, :],
                other=0,
            ).to(tl.int64)
            cnt += tl.sum(blk, axis=0)
        base = tl.sum(tl.where(ecols < e, cnt, 0), axis=0)
        own = tl.sum(tl.where(ecols == e, cnt, 0), axis=0)
        tl.store(COUNTS + e, own)

        # pass 2: scatter this expert's tokens into [base, base+own)
        run = tl.zeros([], tl.int64) + 0
        for t0 in tl.range(0, T, BLOCK_T):
            ts = t0 + tl.arange(0, BLOCK_T)
            tmask = ts < T
            m = tl.load(RMAP + ts * E + e, mask=tmask, other=0).to(tl.int64)
            rank = tl.cumsum(m, axis=0) - m  # exclusive rank within this tile
            pos = base + run + rank
            hit = tmask & (m == 1)
            tl.store(SORTED_IDX + pos, ts.to(tl.int64), mask=hit)
            pr = tl.load(RPROBS + ts * E + e, mask=hit, other=0.0)
            tl.store(PERM_PROBS + pos, pr, mask=hit)
            run += tl.sum(m, axis=0)


# envelopes -- HOST-SIDE and shape-only, so no device sync and no CUDA-graph break. Every clause is a
# property of the operands. A bare `assert` in a vLLM worker kills the Ray actor, and the fallback is
# megatron's own code, so these refuse rather than raise.
def score_can_handle(logits: torch.Tensor, bias: Optional[torch.Tensor], score_function: str) -> bool:
    if not HAVE_TRITON:
        return False
    if score_function not in _FN_CODE:
        return False
    if logits.dim() != 2 or not logits.is_cuda or logits.dtype is not torch.float32:
        return False
    T, E = logits.shape
    if T == 0 or E == 0:
        return False
    if bias is not None and (bias.dim() != 1 or bias.shape[0] != E or bias.dtype is not torch.float32):
        # A non-fp32 bias means megatron's `expert_bias.float()` is a real cast whose output this
        # kernel would have to reproduce from a different input dtype. Refuse the PROPERTY.
        return False
    if _next_pow2(E) > _MAX_BLOCK_E or T * E >= _MAX_ELEMS:
        return False
    return True


def tail_can_handle(
    top_indices: torch.Tensor, vals: torch.Tensor, den: Optional[torch.Tensor], num_experts: int
) -> bool:
    if not HAVE_TRITON:
        return False
    if top_indices.dim() != 2 or vals.dim() != 2 or top_indices.shape != vals.shape:
        return False
    if not top_indices.is_cuda or not vals.is_cuda:
        return False
    if top_indices.dtype is not torch.int64 or vals.dtype is not torch.float32:
        return False
    if den is not None and (not den.is_cuda or den.dtype is not torch.float32 or den.numel() != top_indices.shape[0]):
        return False
    T, K = top_indices.shape
    E = int(num_experts)
    if T == 0 or E == 0 or K == 0 or K > E or K > _MAX_K:
        return False
    if _next_pow2(E) > _MAX_BLOCK_E or T * E >= _MAX_ELEMS:
        return False
    return True


def permute_index_1k_can_handle(routing_map: torch.Tensor, topk: int) -> bool:
    if not HAVE_TRITON:
        return False
    if routing_map.dim() != 2 or not routing_map.is_cuda or routing_map.dtype is not torch.bool:
        return False
    T, E = routing_map.shape
    if T == 0 or E == 0 or topk <= 0:
        return False
    if _next_pow2(E) > _MAX_BLOCK_E or T * E >= _MAX_ELEMS:
        return False
    # The E-fold map re-read is the whole trade this form makes; above this it loses to the split.
    if T * E > _MAX_INLINE_COUNT_ELEMS:
        return False
    return True


# the fused forms
@torch.no_grad()
def fused_score_bias(
    logits: torch.Tensor,
    bias: Optional[torch.Tensor],
    score_function: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(scores [T,E], sel [T,E])`` -- the weight scores and the selection key, one kernel.

    ``sel is scores`` when there is no bias (megatron selects on the scores themselves), so callers
    may compare identity to learn whether a bias was applied.
    """
    if not score_can_handle(logits, bias, score_function):
        return ref_score_bias(logits, bias, score_function)
    T, E = logits.shape
    logits = logits.contiguous()
    scores = torch.empty(T, E, dtype=torch.float32, device=logits.device)
    sel = torch.empty(T, E, dtype=torch.float32, device=logits.device) if bias is not None else scores
    _score_bias_kernel[(T,)](
        logits,
        bias if bias is not None else logits,
        scores,
        sel,
        T,
        E,
        logits.stride(0),
        FN=_FN_CODE[score_function],
        HAS_BIAS=bias is not None,
        BLOCK_E=_next_pow2(E),
        num_warps=_SCORE_WARPS,
        enable_fp_fusion=False,
    )
    return scores, sel


@torch.no_grad()
def fused_router_tail(
    top_indices: torch.Tensor,
    vals: torch.Tensor,
    den: Optional[torch.Tensor],
    num_experts: int,
    *,
    eps: Optional[float] = None,
    scaling_factor: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(routing_probs [T,E] fp32, routing_map [T,E] bool)`` from the compact top-k form.

    ``den`` is ATen's own ``[T,1]`` row-sum -- never recomputed here. Pass ``den=None`` (and no eps,
    no scale) and this is exactly ``moe_dense_scatter_kernel.fused_dense_from_topk``; see
    :func:`tail_subsumes_dense_scatter`.
    """
    if not tail_can_handle(top_indices, vals, den, num_experts):
        return ref_router_tail(top_indices, vals, den, num_experts, eps=eps, scaling_factor=scaling_factor)
    T, K = top_indices.shape
    E = int(num_experts)
    rprobs = torch.empty(T, E, dtype=torch.float32, device=top_indices.device)
    rmap8 = torch.empty(T, E, dtype=torch.int8, device=top_indices.device)
    _router_tail_kernel[(T,)](
        top_indices.contiguous(),
        vals.contiguous(),
        den.contiguous() if den is not None else top_indices,
        rprobs,
        rmap8,
        T,
        E,
        float(eps) if eps else 0.0,
        float(scaling_factor) if scaling_factor else 1.0,
        K=K,
        BLOCK_E=_next_pow2(E),
        HAS_DEN=den is not None,
        HAS_EPS=bool(eps),
        HAS_SCALE=bool(scaling_factor),
        num_warps=_TAIL_WARPS,
        enable_fp_fusion=False,  # NOT vacuous here: `p * scale` sits beside `den + eps`
    )
    return rprobs, rmap8.view(torch.bool)


@torch.no_grad()
def fused_permute_index_1k(
    routing_map: torch.Tensor,
    routing_probs: torch.Tensor,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(sorted_indices [T*k] int64, permuted_probs [T*k], tokens_per_expert [E] int64)``.

    Same contract and same outputs as ``moe_router_o2_kernel.fused_permute_index``, in ONE launch.
    Falls back to that two-kernel form outside the envelope -- not to the eager argsort, because the
    two-kernel form is itself already admitted and strictly better than eager.
    """
    if not permute_index_1k_can_handle(routing_map, topk):
        from .moe_router_o2_kernel import fused_permute_index

        return fused_permute_index(routing_map, routing_probs, topk)
    T, E = routing_map.shape
    dev = routing_map.device
    rmap8 = routing_map.view(torch.int8).contiguous()
    sorted_idx = torch.empty(T * topk, dtype=torch.int64, device=dev)
    perm_probs = torch.empty(T * topk, dtype=routing_probs.dtype, device=dev)
    counts = torch.empty(E, dtype=torch.int64, device=dev)
    _permute_index_1k_kernel[(E,)](
        rmap8,
        routing_probs.contiguous(),
        sorted_idx,
        perm_probs,
        counts,
        T,
        E,
        BLOCK_T=_INDEX_BLOCK_T,
        BLOCK_E=_next_pow2(E),
        num_warps=_INDEX_WARPS,
        enable_fp_fusion=False,
    )
    return sorted_idx, perm_probs, counts


# references -- megatron's expressions, transcribed. The gate compares against THESE and against
# megatron's own function; these are also the documented fallbacks.
def ref_score_bias(logits, bias, score_function):
    """megatron's score-function and bias expression, transcribed."""
    if score_function == "sigmoid":
        scores = torch.sigmoid(logits.float())
    elif score_function == "sqrtsoftplus":
        scores = torch.nn.functional.softplus(logits.float()).sqrt()
    else:
        raise ValueError(f"not a pre-transform score function: {score_function}")
    sel = scores + bias.float() if bias is not None else scores
    return scores, sel


def ref_router_tail(top_indices, vals, den, num_experts, *, eps=None, scaling_factor=None):
    """megatron's normalisation tail and dense build (deterministic branch), transcribed."""
    probs = vals
    if den is not None:
        d = den.reshape(-1, 1)
        probs = probs / (d + eps if eps else d)
    if scaling_factor:
        probs = probs * scaling_factor
    T = top_indices.shape[0]
    E = int(num_experts)
    routing_probs = torch.zeros(T, E, dtype=probs.dtype, device=probs.device)
    rows = torch.arange(T, device=probs.device).unsqueeze(1)
    routing_probs.index_put_((rows, top_indices), probs, accumulate=False)
    routing_map = torch.zeros(T, E, dtype=probs.dtype, device=probs.device)
    routing_map.index_put_((rows, top_indices), torch.ones_like(probs), accumulate=False)
    return routing_probs, routing_map.bool()


def ref_prenorm_routing(logits, topk, *, score_function, expert_bias, scaling_factor):
    """The whole pre-transform chain, eager, exactly as megatron writes it. The gate's target."""
    scores, sel = ref_score_bias(logits, expert_bias, score_function)
    if expert_bias is not None:
        _, top_indices = torch.topk(sel, k=topk, dim=1, sorted=True)
        vals = torch.gather(scores, dim=1, index=top_indices)
    else:
        vals, top_indices = torch.topk(scores, k=topk, dim=1, sorted=True)
    den = vals.sum(dim=-1, keepdim=True) if topk > 1 else None
    return ref_router_tail(
        top_indices,
        vals,
        den,
        logits.shape[1],
        eps=_MEGATRON_NORM_EPS if topk > 1 else None,
        scaling_factor=scaling_factor,
    )


def fused_prenorm_routing(logits, topk, *, score_function, expert_bias, scaling_factor):
    """The chain with its two fusable ends replaced. THE MIDDLE IS ATen AND STAYS ATen.

    Reading top to bottom, this is the whole legality argument in executable form: the two ends are
    Triton, the top-k and the row-sum -- the two operations the stack has proven unbridgeable -- are
    torch calls on the same tensors in the same order.
    """
    if score_fuse_enabled() and score_can_handle(logits, expert_bias, score_function):
        scores, sel = fused_score_bias(logits, expert_bias, score_function)
    else:
        scores, sel = ref_score_bias(logits, expert_bias, score_function)

    # CLOSED (tie order): ATen's top-k, `sorted=True` explicit because this call site sits OUTSIDE
    # `_make_sorted_topk_routing`'s swap. See the module docstring's ORDERING TRAP.
    if expert_bias is not None:
        _, top_indices = torch.topk(sel, k=topk, dim=1, sorted=True)
        vals = torch.gather(scores, dim=1, index=top_indices)
    else:
        vals, top_indices = torch.topk(scores, k=topk, dim=1, sorted=True)

    # CLOSED: ATen's reduction, consumed as an operand and never re-associated.
    den = vals.sum(dim=-1, keepdim=True) if topk > 1 else None

    eps = _MEGATRON_NORM_EPS if topk > 1 else None
    if router_tail_enabled() and tail_can_handle(top_indices, vals, den, logits.shape[1]):
        return fused_router_tail(top_indices, vals, den, logits.shape[1], eps=eps, scaling_factor=scaling_factor)
    return ref_router_tail(top_indices, vals, den, logits.shape[1], eps=eps, scaling_factor=scaling_factor)


def tail_subsumes_dense_scatter(top_indices, probs, num_experts):
    """Is :func:`fused_router_tail` at ``(den=None, eps=None, scale=None)`` the dense scatter?

    Asserted rather than claimed, because "X is a special case of Y" is the kind of generality
    statement that is true when written and false two commits later. Returns (ok, why).
    """
    from .moe_dense_scatter_kernel import fused_dense_from_topk

    a_p, a_m = fused_router_tail(top_indices, probs, None, num_experts)
    b_p, b_m = fused_dense_from_topk(top_indices, probs, num_experts)
    if _bitcmp(a_p, b_p) != 0:
        return False, "routing_probs differ from moe_dense_scatter_kernel's"
    if not torch.equal(a_m, b_m):
        return False, "routing_map differs from moe_dense_scatter_kernel's"
    return True, ""


# admission -- per shape, on LIVE operands, against megatron's own function. Fail-closed, loud, cached
# permanently. This is the arbiter: every "correctly rounded" claim above is a hypothesis until this
# passes, and a toolchain that breaks one of them fails here instead of shipping a bit.
_STATE: dict[tuple, object] = {}


def _bitcmp(a: torch.Tensor, b: torch.Tensor) -> int:
    """Differing element count, compared as BITS (so -0.0 != 0.0 and NaN payloads count)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return -1
    if a.numel() == 0:
        return 0
    view = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
    return int((a.detach().contiguous().view(view) != b.detach().contiguous().view(view)).sum().item())


def _admit(logits, topk, score_function, expert_bias, scaling_factor, orig_fn) -> tuple[bool, str]:
    """Six gates. A failure of any one runs megatron's chain and no bits move."""
    with torch.no_grad():
        x = logits.detach()
        E = x.shape[1]

        # (i) THE REFERENCE IS MEGATRON'S OWN FUNCTION, not our transcription. This is what catches
        #     a megatron upgrade that changes the chain under us -- the failure mode a transcribed
        #     reference is structurally blind to.
        ref_p, ref_m = orig_fn(
            x,
            topk,
            use_pre_softmax=False,
            num_groups=None,
            group_topk=None,
            scaling_factor=scaling_factor,
            score_function=score_function,
            expert_bias=expert_bias,
            fused=False,
        )
        got_p, got_m = fused_prenorm_routing(
            x, topk, score_function=score_function, expert_bias=expert_bias, scaling_factor=scaling_factor
        )
        bad = _bitcmp(ref_p, got_p)
        if bad != 0:
            return False, f"routing_probs differ from megatron's chain ({bad} of {ref_p.numel()} elements)"
        if not torch.equal(ref_m, got_m):
            return False, "routing_map differs from megatron's chain (a DIFFERENT expert set)"

        # (ii) TRANSCRIPTION PARITY. Our eager reference must equal megatron's too -- otherwise (i)
        #      passing would only mean two wrongs cancelling inside the fused path.
        t_p, t_m = ref_prenorm_routing(
            x, topk, score_function=score_function, expert_bias=expert_bias, scaling_factor=scaling_factor
        )
        if _bitcmp(ref_p, t_p) != 0 or not torch.equal(ref_m, t_m):
            return False, "the transcribed reference does not reproduce megatron -- the chain moved"

        # (iii) ROW PROVENANCE, against an index marker rather than a payload. A wrong-but-symmetric
        #       permutation of the selected columns passes a value compare on a random payload; it
        #       cannot pass this, because the marker makes every routed slot distinguishable.
        #       Expressed on the dense map: the set of selected experts per row, in order of column.
        _, sel_ref = ref_score_bias(x, expert_bias, score_function)
        _, idx_ref = torch.topk(sel_ref, k=topk, dim=1, sorted=True)
        marker = torch.zeros(x.shape[0], E, dtype=torch.int64, device=x.device)
        marker.scatter_(1, idx_ref, torch.arange(1, topk + 1, device=x.device).expand(x.shape[0], topk))
        got_scores, got_sel = (
            fused_score_bias(x, expert_bias, score_function)
            if (score_fuse_enabled() and score_can_handle(x, expert_bias, score_function))
            else ref_score_bias(x, expert_bias, score_function)
        )
        _, idx_got = torch.topk(got_sel, k=topk, dim=1, sorted=True)
        marker_got = torch.zeros_like(marker)
        marker_got.scatter_(1, idx_got, torch.arange(1, topk + 1, device=x.device).expand(x.shape[0], topk))
        if not torch.equal(marker, marker_got):
            return False, "row provenance differs: the selected experts or their COLUMN ORDER moved"
        if _bitcmp(got_scores, ref_score_bias(x, expert_bias, score_function)[0]) != 0:
            return False, "the fused score function does not reproduce ATen bitwise"

        # (iv) THE POSITIVE CONTROL, WHICH MUST FAIL. Perturb the last selected column of every row
        #      by one expert index and require the dense output to MOVE. A gate that cannot fail is
        #      not a gate -- and on a router the realistic vacuity is a compare that is blind to the
        #      routing decision itself, which is exactly what this probes.
        idx_bad = idx_ref.clone()
        idx_bad[:, -1] = (idx_bad[:, -1] + 1) % E
        vals_bad = torch.gather(got_scores, 1, idx_bad)
        den_bad = vals_bad.sum(dim=-1, keepdim=True) if topk > 1 else None
        bad_p, bad_m = ref_router_tail(
            idx_bad,
            vals_bad,
            den_bad,
            E,
            eps=_MEGATRON_NORM_EPS if topk > 1 else None,
            scaling_factor=scaling_factor,
        )
        if _bitcmp(bad_p, ref_p) == 0 and torch.equal(bad_m, ref_m):
            return False, "positive control did not move the output -- the comparison is vacuous"

        # (v) DISTINCT top-k columns. The dense build's "no two lanes collide" argument rests on it.
        srt = idx_ref.sort(dim=1).values
        if topk > 1 and bool((srt[:, 1:] == srt[:, :-1]).any().item()):
            return False, "torch.topk returned a duplicate expert index in some row"

        # (vi) DETERMINISM. Two identical calls must agree bit for bit.
        again_p, again_m = fused_prenorm_routing(
            x, topk, score_function=score_function, expert_bias=expert_bias, scaling_factor=scaling_factor
        )
        if _bitcmp(again_p, got_p) != 0 or not torch.equal(again_m, got_m):
            return False, "two identical fused calls disagree"

    return True, ""


def routing_chain_ready(logits, topk, score_function, expert_bias, scaling_factor, orig_fn) -> bool:
    """Is the fused chain admitted for this shape? Fail-closed, loud, cached permanently."""
    if not (score_fuse_enabled() or router_tail_enabled()):
        return False
    if logits is None or logits.dim() != 2 or logits.numel() == 0:
        return False
    if getattr(logits.device, "type", None) != "cuda":
        return False
    key = (
        int(logits.shape[0]),
        int(logits.shape[1]),
        int(topk),
        score_function,
        expert_bias is not None,
        float(scaling_factor or 0.0),
        score_fuse_enabled(),
        router_tail_enabled(),
    )
    state = _STATE.get(key)
    if state is True:
        return True
    if isinstance(state, str):
        return False
    if torch.cuda.is_current_stream_capturing():
        return False  # never let a captured step be the thing that decides a shape
    try:
        ok, why = _admit(logits, topk, score_function, expert_bias, scaling_factor, orig_fn)
    except Exception as e:  # noqa: BLE001
        ok, why = False, f"admission raised ({type(e).__name__}: {e})"
    if ok:
        _STATE[key] = True
        print(
            f"[ISOEXEC-MOE-ROUTER-CHAIN] ADMITTED T={key[0]} E={key[1]} k={key[2]} fn={score_function} "
            f"bias={key[4]} scale={key[5]}: the fused ends reproduce megatron's chain BITWISE "
            f"(probs and map), provenance-exact, non-vacuous and deterministic; the top-k and the "
            f"row-sum still run in ATen.",
            flush=True,
        )
    else:
        _STATE[key] = why
        print(
            f"[ISOEXEC-MOE-ROUTER-CHAIN] REFUSED T={key[0]} E={key[1]} k={key[2]} fn={score_function}: "
            f"{why}. Falling back to megatron's chain for this shape, permanently.",
            flush=True,
        )
    return ok


# installation -- ENGINE-ONLY BY INSTANCE MARK. See the module docstring.
_installed = False
_orig_routing = None
_orig_topk_routing = None
_chain_scope: Optional[object] = None


def chain_active(router) -> bool:
    """This routing call takes the fused chain iff a flag + the engine mark + a property envelope.

    Every clause names the PROPERTY it needs. In particular the score-function clause is a membership
    test in ``_PRE_TRANSFORM`` -- the family whose score function is an elementwise pre-selection map
    -- and not an enumeration of models, and the ``group_topk`` clause refuses because
    ``group_limited_topk`` is a DIFFERENT selection, not because any particular model uses it.
    """
    if not (score_fuse_enabled() or router_tail_enabled()):
        return False
    if not getattr(router, "_isoexec_routing_mechanics", False):
        return False
    cfg = router.config
    return (
        getattr(router, "score_function", None) in _PRE_TRANSFORM
        and getattr(cfg, "moe_router_group_topk", None) in (None, 0)
        and getattr(cfg, "moe_router_num_groups", None) in (None, 0)
        and getattr(cfg, "moe_expert_capacity_factor", None) is None
        and not getattr(cfg, "moe_router_fusion", False)
        and getattr(router, "routing_type", "topk") != "sinkhorn"
        and getattr(router, "router_replay", None) is None
    )


def install_router_chain() -> bool:
    """Patch ``TopKRouter.routing`` + ``topk_routing_with_score_function``. Idempotent, both sides.

    Installed OVER whatever currently holds the binding, so ``moe_dense_scatter_kernel``'s wrapper
    and ``_make_sorted_topk_routing`` both remain in the chain and serve every call this one
    declines. When this one fires it produces the finished dense pair itself and returns, so the
    inner wrappers are not entered twice.
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
        global _chain_scope
        want = padding_mask is None and not (self.training and torch.is_grad_enabled()) and chain_active(self)
        prev, _chain_scope = _chain_scope, (self if want else None)
        try:
            return _orig_routing(self, logits, padding_mask=padding_mask, **kw)
        finally:
            _chain_scope = prev

    def _topk_routing(*args, **kwargs):
        r = _chain_scope
        if r is None or kwargs.get("dense_output", False):
            return _orig_topk_routing(*args, **kwargs)
        logits = args[0]
        topk = args[1] if len(args) > 1 else kwargs["topk"]
        sf = kwargs.get("score_function", "softmax")
        bias = kwargs.get("expert_bias", None)
        scale = kwargs.get("scaling_factor", None)
        if sf not in _PRE_TRANSFORM or kwargs.get("group_topk") or kwargs.get("router_replay") is not None:
            return _orig_topk_routing(*args, **kwargs)
        if not routing_chain_ready(logits, topk, sf, bias, scale, _orig_topk_routing):
            return _orig_topk_routing(*args, **kwargs)
        return fused_prenorm_routing(logits, topk, score_function=sf, expert_bias=bias, scaling_factor=scale)

    TopKRouter.routing = _routing
    _router_mod.topk_routing_with_score_function = _topk_routing
    _installed = True
    return True


def router_chain_banner() -> str:
    """The line a live arm must be grepped for. A provider that prints no banner cannot be cited."""
    return (
        f"[ISOEXEC-MOE] router chain: {_ENV_SCORE}="
        f"{'ON (1 kernel for score+bias)' if score_fuse_enabled() else 'OFF (eager sigmoid + add)'}; "
        f"{_ENV_TAIL}={'ON (1 kernel for eps+div+scale+dense)' if router_tail_enabled() else 'OFF (4 eager kernels)'}; "
        f"{_ENV_INDEX1K}={'ON (1 kernel counting sort)' if permute_index_1k_enabled() else 'OFF (2 kernels)'}"
    )
