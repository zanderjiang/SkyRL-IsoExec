"""O2 -- fuse the MoE router (top-k + softmax + dense scatter) and the dispatcher's permute sort.

The DENSE router output is bitwise-identical to torch's, ties included: within an exactly-tied group every
selected score is the same fp32 bit pattern, so ``amax`` is order-free, the summand sequence fed to the
denominator reduction is unchanged by permuting equal values, and the scatter is keyed on the expert index
rather than the column. A router emitting ``(routing_probs [T,E], routing_map [T,E])`` -- all any consumer
on this stack sees -- is therefore bitwise equal to the eager sequence, which is what makes a one-sided
install safe. ``moe_router_kernel.fused_topk`` is a separate flag and cannot be one-sided: it returns the
raw ``[T, k]`` top-k, whose column order inside a tie differs from torch's.

The kernel reproduces vLLM's batch-invariant softmax DECOMPOSITION (amax, sub, exp, sum, div), not ATen's
fused softmax, because ``VLLM_BATCH_INVARIANT=1`` overrides ``aten::softmax`` with it. Four Triton-vs-ATen
numerics hazards are handled explicitly: ``libdevice.exp`` rather than ``tl.exp`` (which lowers to
``ex2.approx``); an inline-PTX ``div.rn.f32`` rather than Triton's ``/`` (approximate reciprocal) or
``libdevice.div_rn`` (Triton links libdevice with ``__CUDA_FTZ``, flushing the subnormal quotients real
router rows produce); ``tl.sum`` over a power-of-two ``BLOCK_K == K``, whose butterfly tree matches
``torch.sum`` -- non-power-of-two ``k`` is refused and falls back to eager; and ``enable_fp_fusion=False``
on every launch. The dispatcher's ``argsort(descending=True, stable=True)`` over a boolean key is a
counting sort by expert, i.e. a pure integer permutation.

NO AUTOTUNE: ``num_warps``, ``BLOCK_E`` and ``BLOCK_K`` are pinned and part of the bitwise contract because
they fix the fp32 reduction order of the softmax denominator; only the grid depends on ``T``.

INSTALLATION IS ENGINE-ONLY, BY INSTANCE MARK. Under ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` a class-level
rebind of ``TopKRouter.routing`` / ``MoEAllGatherTokenDispatcher.dispatch_postprocess`` would reach the
trainer, where these kernels -- having no ``autograd.Function`` -- would sever the MoE backward while the
forward-only gate stayed green. The class patches fire only on instances marked by
:func:`mark_engine_router_o2`. Flag ``SKYRL_ISOEXEC_MOE_ROUTER_O2``, default OFF, re-read on every call.
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


_ENV = "SKYRL_ISOEXEC_MOE_ROUTER_O2"


def router_o2_enabled() -> bool:
    """Default OFF. Re-read per call so an in-process A/B can flip it between forwards."""
    return os.environ.get(_ENV, "0") == "1"


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


# PINNED TILE SHAPE -- PART OF THE BITWISE CONTRACT, NOT A PERFORMANCE KNOB. DO NOT AUTOTUNE.
#
# One token per program on ONE warp: that is the configuration the softmax denominator's reduction order
# was validated under, since `tl.sum` over a [BLOCK_K] vector in a single-warp program emits the butterfly
# tree `torch.sum` uses. Widening `num_warps` changes Triton's layout for that vector and can change the
# reduction tree -- i.e. it can silently break IsoExec while looking like a tuning change. `BLOCK_E` and
# `BLOCK_K` are `next_pow2` of E and k; only the GRID depends on T.
_ROUTER_WARPS = 1
_SORT_WARPS = 4
_SORT_BLOCK_T = 512


if HAVE_TRITON:

    # THE NON-FTZ DIVIDE. Lives in isoexec/core/triton_nonftz.py because it is not an O2 detail: Triton
    # links libdevice with __CUDA_FTZ, so `libdevice.div_rn` flushes subnormal results to zero, while
    # Triton's `/` keeps subnormals but is an approximate reciprocal. Neither is usable, and neither
    # shows up on `randn` inputs -- a router row with a massive-activation outlier gives
    # exp(score - max) ~ 5e-42 for the other experts, i.e. subnormal fp32.
    _div_rn_nonftz = _nonftz.div_rn

    @triton.jit
    def _router_kernel(
        LOGITS,  # [T, E] fp32
        RPROBS,  # [T, E] fp32   out: dense routing probs
        RMAP,  # [T, E] int8   out: dense routing map (viewed as bool by the caller)
        T,
        E,
        stride_l,
        SCALE,  # float; 1.0 and HAS_SCALE=False when moe_router_topk_scaling_factor is unset
        K: tl.constexpr,
        BLOCK_E: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_SCALE: tl.constexpr,
    ):
        """One token per program: top-k -> batch-invariant softmax -> dense scatter, in registers.

        Reproduces, bit for bit:
            scores, idx = torch.topk(logits, K, dim=1, sorted=True)
            m = torch.amax(scores, -1, keepdim=True); e = torch.exp(scores - m)
            probs = e / torch.sum(e, -1, keepdim=True)        # vllm softmax_batch_invariant
            dense scatter of probs / 1.0 into two zeroed [T, E] tensors
        """
        t = tl.program_id(0)
        cols = tl.arange(0, BLOCK_E)
        cmask = cols < E
        s = tl.load(LOGITS + t * stride_l + cols, mask=cmask, other=float("-inf"))
        # NaN canonicalisation -- load-bearing. The winner search is "max, then LOWEST index among
        # entries EQUAL to that max"; NaN equals nothing, so on a NaN row the equality set is empty and
        # `tl.min` returns the `E` sentinel, i.e. an out-of-range expert index. Mapping NaN -> -inf makes
        # the sentinel unreachable. No-op on finite input.
        s = tl.where(s != s, float("-inf"), s)

        live = cmask
        kar = tl.arange(0, BLOCK_K)
        vals = tl.full([BLOCK_K], float("-inf"), tl.float32)
        idxs = tl.zeros([BLOCK_K], tl.int32)

        # THE TIE RULE, identical to moe_router_kernel: A beats B iff score(A) > score(B), or
        # score(A) == score(B) and index(A) < index(B). torch.topk on CUDA breaks ties by LOWEST
        # expert index too (verified there over seven adversarial tie patterns), so MEMBERSHIP is
        # identical. Only the column position inside a tied group may differ -- and it never leaves
        # this kernel, because the outputs are dense and keyed on the expert index.
        for j in tl.static_range(K):
            cur = tl.where(live, s, float("-inf"))
            best = tl.max(cur, axis=0)
            eq = live & (cur == best)
            bidx = tl.min(tl.where(eq, cols, E), axis=0)
            # Belt and braces: unreachable after the NaN canonicalisation, kept so "the stored index
            # is in [0, E)" is true by construction rather than by argument.
            bidx = tl.minimum(bidx, E - 1)
            vals = tl.where(kar == j, best, vals)
            idxs = tl.where(kar == j, bidx, idxs)
            live = live & (cols != bidx)

        # ---- vllm softmax_batch_invariant over the K selected scores
        kmask = kar < K
        # amax: order-free, so no reduction-order hazard here.
        vmax = tl.max(tl.where(kmask, vals, float("-inf")), axis=0)
        # libdevice.exp, NOT tl.exp: tl.exp lowers to ex2.approx and diverges from ATen.
        ex = libdevice.exp(vals - vmax)
        ex = tl.where(kmask, ex, 0.0)
        # tl.sum over a power-of-two BLOCK_K in a single-warp program is the same butterfly (half-fold)
        # tree torch.sum uses over a length-K contiguous last dim. _router_can_handle refuses
        # non-power-of-two K, where the zero padding would re-associate.
        ssum = tl.sum(ex, axis=0)
        # Neither `libdevice.div_rn` nor `/` is correct here -- see _div_rn_nonftz. `/` is an approximate
        # reciprocal; `div_rn` is correctly rounded but FLUSHES SUBNORMALS TO ZERO, which a
        # massive-activation router row produces routinely.
        p = _div_rn_nonftz(ex, ssum)
        if HAS_SCALE:
            # megatron: `probs = probs * scaling_factor` -- a plain fp32 multiply, no accumulator,
            # so nothing for FFMA contraction to reach.
            p = p * SCALE

        # Dense scatter (zeros_like + index_put_) in one pass: build [BLOCK_E] dense rows by OR-ing the
        # K one-hot columns. K is constexpr, so this unrolls and stays in registers -- no [T,E] zero-fill
        # kernel and no index_put_.
        dprob = tl.zeros([BLOCK_E], tl.float32)
        dmap = tl.zeros([BLOCK_E], tl.int8)
        for j in tl.static_range(K):
            ij = tl.sum(tl.where(kar == j, idxs, 0), axis=0)
            pj = tl.sum(tl.where(kar == j, p, 0.0), axis=0)
            hit = cols == ij
            dprob = tl.where(hit, pj, dprob)
            dmap = tl.where(hit, 1, dmap).to(tl.int8)

        tl.store(RPROBS + t * E + cols, dprob, mask=cmask)
        tl.store(RMAP + t * E + cols, dmap, mask=cmask)

    @triton.jit
    def _counts_kernel(RMAP, COUNTS, T, E, BLOCK_T: tl.constexpr):
        """``tokens_per_expert = routing_map.sum(dim=0).long()`` -- one program per expert.

        Integer sum: exact, order-free, no rounding to reproduce.
        """
        e = tl.program_id(0)
        acc = tl.zeros([BLOCK_T], tl.int32)
        for t0 in tl.range(0, T, BLOCK_T):
            ts = t0 + tl.arange(0, BLOCK_T)
            m = tl.load(RMAP + ts * E + e, mask=ts < T, other=0).to(tl.int32)
            acc += m
        tl.store(COUNTS + e, tl.sum(acc, axis=0).to(tl.int64))

    @triton.jit
    def _permute_index_kernel(
        RMAP,  # [T, E] int8
        RPROBS,  # [T, E] fp32
        COUNTS,  # [E] int64
        SORTED_IDX,  # [T*k] int64   out
        PERM_PROBS,  # [T*k] fp32    out
        T,
        E,
        BLOCK_T: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        """Counting sort by expert -- the stable ``argsort(descending=True)`` of the boolean map.

        ``stable=True`` makes the True entries of ``routing_map.T.reshape(-1)`` come out in
        ascending flat index ``e*T + t``, so slot ``base[e] + rank`` holds token ``t``, where
        ``base`` is the exclusive prefix sum of the per-expert counts and ``rank`` is ``t``'s
        position among expert ``e``'s tokens in ascending ``t``. Pure integer indexing.

        ``permuted_probs[i] = probs.T.reshape(-1)[e*T + t] = routing_probs[t, e]`` -- a gather.
        """
        e = tl.program_id(0)
        ecols = tl.arange(0, BLOCK_E)
        c = tl.load(COUNTS + ecols, mask=ecols < E, other=0)
        # exclusive prefix over experts. BLOCK_E is 256 at production -- a masked sum, not a scan,
        # is both simpler and cheaper than paying for tl.cumsum here.
        base = tl.sum(tl.where(ecols < e, c, 0), axis=0)

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


# `K` is a constexpr driving `tl.static_range`, so the selection loop is FULLY UNROLLED and compile time
# is O(k). Past this bound torch.topk wins anyway, and a large-k compile is indistinguishable from a hang
# at engine init.
_MAX_K = 32
# A program keeps the whole row in registers; beyond this it spills. Same bound as moe_router_kernel.
_MAX_BLOCK_E = 8192
# Row offsets are int32 in Triton by default; past 2^31 elements that arithmetic wraps and the loads read
# arbitrary memory.
_MAX_ELEMS = 2**31


def _router_can_handle(logits: torch.Tensor, k: int) -> bool:
    """Host-side, shape-only envelope check. No device sync, CUDA-graph safe.

    A bare ``assert`` here would kill the Ray actor; a fallback to the eager path is merely slower
    and is bitwise-identical by construction, so fall back rather than raise.
    """
    if not HAVE_TRITON:
        return False
    if logits.dim() != 2 or not logits.is_cuda:
        return False
    if logits.dtype is not torch.float32:
        return False  # the reproduced softmax decomposition is the fp32 one
    T, E = logits.shape
    if T == 0 or E == 0 or not (0 < k <= E):
        return False
    if k > _MAX_K:
        return False
    # THE REDUCTION-ORDER CONSTRAINT. tl.sum over BLOCK_K == k is torch.sum's butterfly tree only
    # when k is a power of two; otherwise the zero padding re-associates the tree and the
    # denominator differs in the last ulp. Refuse rather than be almost right.
    if k & (k - 1) != 0:
        return False
    if _next_pow2(E) > _MAX_BLOCK_E:
        return False
    if T * E >= _MAX_ELEMS:
        return False
    return True


@torch.no_grad()
def fused_router_dense(
    logits: torch.Tensor,
    k: int,
    *,
    scaling_factor: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(routing_probs [T,E] fp32, routing_map [T,E] bool)`` -- one kernel.

    Bitwise-equal to megatron's ``topk_routing_with_score_function`` at
    ``score_function="softmax"``, ``use_pre_softmax=False``, deterministic-algorithms branch, under
    ``VLLM_BATCH_INVARIANT=1``. See the module docstring for the tie argument and the four hazards.

    Outside :func:`_router_can_handle`'s envelope this falls back to :func:`ref_router_dense` --
    the eager sequence -- rather than raising. An ``assert`` inside a vLLM worker kills the Ray
    actor as dead as an illegal address does, and the fallback is bitwise-identical by definition.
    """
    if not _router_can_handle(logits, k):
        return ref_router_dense(logits, k, scaling_factor=scaling_factor)
    T, E = logits.shape
    logits = logits.contiguous()
    rprobs = torch.empty(T, E, dtype=torch.float32, device=logits.device)
    rmap8 = torch.empty(T, E, dtype=torch.int8, device=logits.device)
    _router_kernel[(T,)](
        logits,
        rprobs,
        rmap8,
        T,
        E,
        logits.stride(0),
        float(scaling_factor) if scaling_factor else 1.0,
        K=k,
        BLOCK_E=_next_pow2(E),
        BLOCK_K=k,
        HAS_SCALE=bool(scaling_factor),
        num_warps=_ROUTER_WARPS,
        enable_fp_fusion=False,  # see the module docstring; asserted non-vacuous by the harness
    )
    return rprobs, rmap8.view(torch.bool)


@torch.no_grad()
def fused_permute_index(
    routing_map: torch.Tensor,
    routing_probs: torch.Tensor,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(sorted_indices [T*k] int64, permuted_probs [T*k], tokens_per_expert [E] int64)``.

    Replaces the transposing copies + ``argsort(descending=True, stable=True)`` + ``remainder`` +
    gather + column-sum in ``moe_utils.permute`` / ``_dispatch_postprocess_fixed_shape`` with two
    kernels. Integer permutation and a gather -- no arithmetic, so nothing to round. Falls back to
    the eager index build outside the envelope, for the same reason the router does.
    """
    if not permute_can_handle(routing_map, topk):
        return ref_permute_index(routing_map, routing_probs, topk)
    T, E = routing_map.shape
    dev = routing_map.device
    rmap8 = routing_map.view(torch.int8).contiguous()
    counts = torch.empty(E, dtype=torch.int64, device=dev)
    _counts_kernel[(E,)](rmap8, counts, T, E, BLOCK_T=_SORT_BLOCK_T, num_warps=_SORT_WARPS, enable_fp_fusion=False)
    n_out = T * topk
    sorted_idx = torch.empty(n_out, dtype=torch.int64, device=dev)
    perm_probs = torch.empty(n_out, dtype=routing_probs.dtype, device=dev)
    _permute_index_kernel[(E,)](
        rmap8,
        routing_probs.contiguous(),
        counts,
        sorted_idx,
        perm_probs,
        T,
        E,
        BLOCK_T=_SORT_BLOCK_T,
        BLOCK_E=_next_pow2(E),
        num_warps=_SORT_WARPS,
        enable_fp_fusion=False,
    )
    return sorted_idx, perm_probs, counts


def permute_can_handle(routing_map: torch.Tensor, topk: int) -> bool:
    """Envelope for :func:`fused_permute_index`. Host-side, shape-only."""
    if not HAVE_TRITON:
        return False
    if routing_map.dim() != 2 or not routing_map.is_cuda or routing_map.dtype is not torch.bool:
        return False
    T, E = routing_map.shape
    if T == 0 or E == 0 or topk <= 0:
        return False
    if _next_pow2(E) > _MAX_BLOCK_E:
        return False
    if T * E >= _MAX_ELEMS:
        return False
    return True


# reference implementation -- the eager sequence, verbatim. The harness gates against THIS, and it is also
# what the installed wrapper falls back to when the envelope check fails.
def ref_router_dense(logits, k, *, scaling_factor=None):
    """megatron's eager path, transcribed. Used by the gate and as the documented fallback."""
    T = logits.shape[0]
    scores, top_indices = torch.topk(logits, k=k, dim=1, sorted=True)
    # vllm's softmax_batch_invariant, spelled out (the aten override is a python composition).
    m = torch.amax(scores, dim=-1, keepdim=True)
    ex = torch.exp(scores - m)
    probs = ex / torch.sum(ex, dim=-1, keepdim=True)
    if scaling_factor:
        probs = probs * scaling_factor
    probs = probs.type_as(logits)
    routing_probs = torch.zeros_like(logits)
    rows = torch.arange(T, device=logits.device).unsqueeze(1)
    routing_probs.index_put_((rows, top_indices), probs, accumulate=False)
    routing_map = torch.zeros_like(logits, dtype=logits.dtype)
    routing_map.index_put_((rows, top_indices), torch.ones_like(probs, dtype=routing_map.dtype), accumulate=False)
    return routing_probs, routing_map.bool()


def ref_permute_index(routing_map, routing_probs, topk):
    """megatron ``permute``'s index build, transcribed."""
    T = routing_map.shape[0]
    n_out = T * topk
    rt = routing_map.bool().T.contiguous()
    flat_sorted = rt.reshape(-1).argsort(descending=True, stable=True)[:n_out]
    sorted_indices = flat_sorted % T
    perm_probs = routing_probs.T.contiguous().reshape(-1)[flat_sorted]
    counts = routing_map.sum(dim=0).long()
    return sorted_indices, perm_probs, counts


# installation -- ENGINE-ONLY BY INSTANCE MARK. See the module docstring.
_installed = False
_orig_routing = None
_marked_routers = 0


def _router_o2_active(router) -> bool:
    """This routing call takes the fused path iff the flag is on AND the instance was marked by the
    engine model build AND the config is inside the validated envelope."""
    if not router_o2_enabled():
        return False
    if not getattr(router, "_isoexec_router_o2", False):
        return False
    cfg = router.config
    return (
        getattr(router, "score_function", None) == "softmax"
        and not getattr(cfg, "moe_router_pre_softmax", False)
        and getattr(cfg, "moe_router_group_topk", None) in (None, 0)
        and getattr(cfg, "moe_router_num_groups", None) in (None, 0)
        and getattr(router, "expert_bias", None) is None
        and getattr(cfg, "moe_expert_capacity_factor", None) is None
        and not getattr(cfg, "moe_router_fusion", False)
        and getattr(router, "routing_type", "topk") != "sinkhorn"
        and getattr(router, "router_replay", None) is None
        and getattr(cfg, "moe_router_enable_expert_bias", False) is False
    )


def dispatch_o2_active(disp) -> bool:
    """This dispatcher call takes the fused permute-index path iff flag + engine mark + EP=1.

    THE EP GUARD IS LOAD-BEARING. ``num_out_tokens`` is ``T * topk``, which equals the number of
    routed rows only when this rank owns EVERY expert. At EP>1 ``local_map`` is a column slice, a
    token may route to fewer than ``topk`` LOCAL experts, and megatron's
    ``argsort(descending=True)[:T*topk]`` then spills past the True group and pads the tail with
    entries from the FALSE region. The counting sort writes only ``sum(counts)`` slots and would
    leave that tail uninitialised -- a silent garbage read, not a rounding difference. The
    production engine runs ``SKYRL_ISOEXEC_ENGINE_EP=1``, so this only ever refuses a configuration
    the kernel was never gated for.
    """
    return router_o2_enabled() and getattr(disp, "_isoexec_router_o2", False) and getattr(disp, "ep_size", 1) == 1


def install_router_o2() -> bool:
    """Patch ``TopKRouter.routing``. Idempotent, safe to call on both sides.

    The patch is class-level because that is the only binding megatron offers, but it delegates to
    the original for every UNMARKED instance -- so the trainer's routers (which need the autograd
    graph megatron's eager ops build) are untouched. See the module docstring's INSTALLATION note:
    with ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` the two runtimes share this process.
    """
    global _installed, _orig_routing
    if _installed:
        return True
    try:
        from megatron.core.transformer.moe.router import TopKRouter
    except Exception:  # pragma: no cover
        return False

    _orig_routing = TopKRouter.routing

    def _routing(self, logits, padding_mask=None, **kw):
        if padding_mask is None and _router_o2_active(self):
            flat = logits.view(-1, self.config.num_moe_experts)
            # apply_z_loss is identity outside training-with-grad; the guard keeps it that way.
            if not (self.training and torch.is_grad_enabled()):
                k = self.topk
                if _router_can_handle(flat, k):
                    return fused_router_dense(flat, k, scaling_factor=self.config.moe_router_topk_scaling_factor)
        return _orig_routing(self, logits, padding_mask=padding_mask, **kw)

    TopKRouter.routing = _routing
    _installed = True
    return True


def mark_engine_router_o2(model) -> int:
    """Mark every MoE router AND allgather dispatcher in the ENGINE model for the O2 fast path.

    Called from ``gptmodel_vllm.py`` after the engine GPTModel is built, never from the trainer.
    The mark is what scopes the patch to one runtime in a colocated process -- identical mechanism
    to ``mark_engine_dispatchers_nogather``.
    """
    global _marked_routers
    n_r = n_d = 0
    for m in model.modules():
        r = getattr(m, "router", None)
        if r is not None and hasattr(r, "topk"):
            r._isoexec_router_o2 = True
            n_r += 1
        d = getattr(m, "token_dispatcher", None)
        if d is not None:
            d._isoexec_router_o2 = True
            n_d += 1
    _marked_routers = n_r
    print(
        f"[ISOEXEC-MOE] ENGINE O2 fused router marked on {n_r} routers / {n_d} dispatchers; "
        f"SKYRL_ISOEXEC_MOE_ROUTER_O2={'ON -- 23.8 -> 3 launches/layer' if router_o2_enabled() else 'OFF (eager topk+softmax+argsort)'}",
        flush=True,
    )
    return n_r
