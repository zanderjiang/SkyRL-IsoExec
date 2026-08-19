"""VJP for the MoE combine (unpermute + fixed-order top-k sum) and the probs/router tail.

The router probs are folded into the expert epilogue BEFORE fc2, so every production call site passes
``probs=None`` and the combine is an UNWEIGHTED sum of each token's k rows. Two consequences: there is no
``d(prob)`` here -- it is a reduction over the intermediate dim, produced by
``moe_backward_kernel._segment_backward`` -- and ``d(row)`` has no reduction at all, since each permuted
row feeds exactly one token, so the whole production backward is one gather,
``dout.index_select(0, sorted_indices)``. The ``USE_PROBS`` branch exists because megatron's ``unpermute``
supports it, but it is dead on this stack.

Autograd would instead run ``k`` ``index_select`` backwards, each a full ``[P, H]`` zero buffer plus a
scattered ``index_add``, so replacing it is a large constant-factor and peak-memory win that needs no
numerical argument -- the answer is exact either way. Separately,
``moe_combine_kernel.fused_fixed_order_combine`` is a raw Triton call with no autograd; :class:`MoECombine`
wraps it in an ``autograd.Function`` with a bitwise-unchanged forward, which is what lets the TRAINER take
the fused forward too.

Every gradient produced here is checked for being nonzero explicitly, not merely for matching autograd:
``MoEAuxLossAutoScaler.backward`` seeds the aux-loss gradient independently of the incoming activation
gradient, so a silently zero ``dprobs`` would still leave the router weight with a nonzero gradient from
the balancing and z-loss subgraphs while the forward-only IsoExec gate stayed green. Precision:
``d(permuted)`` involves no reduction and is just a cast of ``dout`` to ``permuted.dtype``; the dead
weighted branch's ``d(prob)`` dot product over ``h`` is fp32, unconditionally.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from .moe_combine_kernel import build_combine_rows, fused_fixed_order_combine

_BLOCK_H = 1024
_NUM_WARPS = 4
_COUNTS = {
    "forward_served": 0,
    "backward_served": 0,
    # forwards that skipped the autograd.Function entirely because the caller was under no_grad -- all of
    # scoring, and the checkpoint forward of every train step. Counted separately from forward_served so
    # that only the latter proves the analytic backward is reachable.
    "nograd_served": 0,
    "declined": 0,
    "rows": 0,
    "reported": 0,
}
# First occurrence of every decline reason prints on its own line; the histogram rides the banner.
_DECLINES: dict[str, int] = {}


def _decline(reason: str) -> None:
    """Record a fallback to the eager combine. A wrapper that installs and then declines every call reads
    as "no change" on a step clock, so declines are made visible rather than inferable."""
    _COUNTS["declined"] += 1
    first = reason not in _DECLINES
    _DECLINES[reason] = _DECLINES.get(reason, 0) + 1
    if first:
        print(f"[ISOEXEC-MOE-COMBINE-BWD] pid={os.getpid()} DECLINE({reason}) -- eager combine used", flush=True)
    _report_counts()


def _report_counts() -> None:
    n = _COUNTS["forward_served"] + _COUNTS["backward_served"] + _COUNTS["nograd_served"] + _COUNTS["declined"]
    if n < 1 or (n & (n - 1)) != 0 or n == _COUNTS["reported"]:
        return
    _COUNTS["reported"] = n
    hist = " ".join(f"{k}={v}" for k, v in sorted(_DECLINES.items())) or "none"
    print(
        f"[ISOEXEC-MOE-COMBINE-BWD] pid={os.getpid()} forward_served={_COUNTS['forward_served']} "
        f"backward_served={_COUNTS['backward_served']} nograd_served={_COUNTS['nograd_served']} "
        f"declined={_COUNTS['declined']} rows={_COUNTS['rows']} declines[{hist}]",
        flush=True,
    )


# d(permuted) -- the production path. A broadcast, not a reduction.
@triton.jit
def _combine_bwd_scatter_kernel(
    dout_ptr,  # [T, H]
    rows_ptr,  # [T, K] int32
    dperm_ptr,  # [P, H]
    do_s0,
    do_s1,
    dp_s0,
    dp_s1,
    H,
    K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """``dperm[rows[t, j]] = dout[t]`` for all j. One read of ``dout[t]``, K writes.

    Every element of ``dperm`` is written by exactly one program (each permuted row belongs to
    exactly one token -- the invariant ``_fixed_order_combine`` already validates), so there is no
    atomic, no race, and no zero-init: the kernel covers the whole output.
    """
    t = tl.program_id(0)
    hb = tl.program_id(1)
    offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs < H
    g = tl.load(dout_ptr + t * do_s0 + offs * do_s1, mask=mask, other=0.0)
    g = g.to(dperm_ptr.dtype.element_ty)
    for j in tl.static_range(K):
        r = tl.load(rows_ptr + t * K + j).to(tl.int64)
        tl.store(dperm_ptr + r * dp_s0 + offs * dp_s1, g, mask=mask)


# d(permuted) + d(prob) -- generic weighted Megatron compatibility. The IsoExec recipe passes probs=None,
# but the installed wrapper can meet other dispatcher configurations and must preserve their fallback.
# This is the only kernel here that contains a reduction.
@triton.jit
def _combine_bwd_weighted_kernel(
    dout_ptr,  # [T, H]
    permuted_ptr,  # [P, H]   -- the forward's rows; must be SAVED for this branch
    probs_ptr,  # [P] fp32
    rows_ptr,  # [T, K] int32
    dperm_ptr,  # [P, H]
    dprobs_ptr,  # [P] fp32
    do_s0,
    do_s1,
    p_s0,
    p_s1,
    dp_s0,
    dp_s1,
    H,
    K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """``dperm[r] = prob[r] * dout[t]`` and ``dprob[r] = <dout[t], permuted[r]>`` in ONE pass.

    The two share the read of ``dout[t]`` and of the row index, which is why they are fused. The
    h-loop is sequential inside one program so the dot-product accumulator never leaves fp32
    registers and never needs an atomic -- one program owns row ``r`` outright.
    """
    t = tl.program_id(0)
    for j in tl.static_range(K):
        r = tl.load(rows_ptr + t * K + j).to(tl.int64)
        pr = tl.load(probs_ptr + r).to(tl.float32)
        acc = tl.zeros((), dtype=tl.float32)
        for h0 in tl.range(0, H, BLOCK_H):
            offs = h0 + tl.arange(0, BLOCK_H)
            mask = offs < H
            g = tl.load(dout_ptr + t * do_s0 + offs * do_s1, mask=mask, other=0.0).to(tl.float32)
            x = tl.load(permuted_ptr + r * p_s0 + offs * p_s1, mask=mask, other=0.0).to(tl.float32)
            tl.store(dperm_ptr + r * dp_s0 + offs * dp_s1, (g * pr).to(dperm_ptr.dtype.element_ty), mask=mask)
            acc += tl.sum(g * x, axis=0)
        tl.store(dprobs_ptr + r, acc)


def combine_backward(
    dout: torch.Tensor,
    rows: torch.Tensor,
    num_permuted: int,
    *,
    out_dtype: torch.dtype,
    permuted: torch.Tensor | None = None,
    permuted_probs: torch.Tensor | None = None,
    sorted_indices: torch.Tensor | None = None,
    impl: str = "auto",
):
    """VJP of the fixed-order top-k combine.

    Returns ``(d_permuted [P, H], d_permuted_probs [P] or None)``.

    ``impl``:
      * ``"gather"`` -- ``dout.index_select(0, sorted_indices)``. Requires ``sorted_indices`` and
        the unweighted (production) case. This is the whole backward, in one aten op.
      * ``"triton"`` -- the scatter kernel above. Same result; kept because it does not need
        ``sorted_indices`` (``rows`` alone suffices).
      * ``"auto"`` -- ``gather`` when ``sorted_indices`` is available and unweighted, else
        ``triton``.

    The weighted branch always takes ``_combine_bwd_weighted_kernel`` and always returns dprobs.
    """
    T, H = dout.shape
    K = rows.shape[1]
    assert rows.shape[0] == T, f"rows {tuple(rows.shape)} does not match dout {tuple(dout.shape)}"

    if permuted_probs is not None:
        assert permuted is not None, "the weighted combine backward needs the forward's rows saved"
        d_perm = torch.empty(num_permuted, H, dtype=out_dtype, device=dout.device)
        d_probs = torch.empty(num_permuted, dtype=permuted_probs.dtype, device=dout.device)
        _combine_bwd_weighted_kernel[(T,)](
            dout,
            permuted,
            permuted_probs,
            rows,
            d_perm,
            d_probs,
            dout.stride(0),
            dout.stride(1),
            permuted.stride(0),
            permuted.stride(1),
            d_perm.stride(0),
            d_perm.stride(1),
            H,
            K=K,
            BLOCK_H=_BLOCK_H,
            num_warps=_NUM_WARPS,
        )
        return d_perm, d_probs

    if impl == "auto":
        impl = "gather" if sorted_indices is not None else "triton"

    if impl == "gather":
        assert sorted_indices is not None, "impl='gather' needs sorted_indices (row -> token)"
        # sorted_indices[r] IS token_of_row(r) -- the same invariant _fixed_order_combine validates
        # ("sorted_indices[argsort(sorted_indices)] == arange(T).repeat_interleave(k)").
        return dout.to(out_dtype).index_select(0, sorted_indices), None

    d_perm = torch.empty(num_permuted, H, dtype=out_dtype, device=dout.device)
    _combine_bwd_scatter_kernel[(T, triton.cdiv(H, _BLOCK_H))](
        dout,
        rows,
        d_perm,
        dout.stride(0),
        dout.stride(1),
        d_perm.stride(0),
        d_perm.stride(1),
        H,
        K=K,
        BLOCK_H=_BLOCK_H,
        num_warps=_NUM_WARPS,
    )
    return d_perm, None


# The autograd.Function: production forward, analytic backward.
class MoECombine(torch.autograd.Function):
    """``moe_combine_kernel.fused_fixed_order_combine`` made differentiable.

    The forward is the fused Triton kernel VERBATIM -- this class adds no arithmetic, so IsoExec is
    untouched by construction.

    SAVED FOR BACKWARD, unweighted (i.e. production): ``rows [T, K]`` int32 and ``sorted_indices [P]``
    int64, both already materialized by the caller. The ``[P, H]`` rows are NOT saved and cannot be
    needed -- the adjoint of a sum of gathers does not involve the gathered values. The weighted
    compatibility branch additionally saves ``permuted`` and ``permuted_probs``, because
    ``d(prob) = <dout, row>`` genuinely needs the rows.
    """

    @staticmethod
    def forward(ctx, permuted_tokens, sorted_indices, num_tokens, permuted_probs, rows, impl, out_dtype):
        if rows is None:
            rows = build_combine_rows(sorted_indices, num_tokens)
            if rows is None:
                raise RuntimeError(
                    "[isoexec] MoECombine: layout is not 'every token routes to exactly topk "
                    "experts' (token dropping / capacity padding). Fall back to the eager path."
                )
        out = fused_fixed_order_combine(
            permuted_tokens,
            sorted_indices,
            (num_tokens,),
            permuted_probs=permuted_probs,
            rows=rows,
            out_dtype=out_dtype,
        )
        ctx.weighted = permuted_probs is not None
        ctx.P = permuted_tokens.shape[0]
        ctx.in_dtype = permuted_tokens.dtype
        ctx.impl = impl
        if ctx.weighted:
            ctx.save_for_backward(rows, sorted_indices, permuted_tokens, permuted_probs)
        else:
            ctx.save_for_backward(rows, sorted_indices)
        _COUNTS["forward_served"] += 1
        _COUNTS["rows"] += int(permuted_tokens.shape[0])
        _report_counts()
        return out

    @staticmethod
    def backward(ctx, dout):
        if ctx.weighted:
            rows, sorted_indices, permuted, probs = ctx.saved_tensors
        else:
            (rows, sorted_indices), permuted, probs = ctx.saved_tensors, None, None
        d_perm, d_probs = combine_backward(
            dout.contiguous(),
            rows,
            ctx.P,
            out_dtype=ctx.in_dtype,
            permuted=permuted,
            permuted_probs=probs,
            sorted_indices=sorted_indices,
            impl=ctx.impl,
        )
        _COUNTS["backward_served"] += 1
        _report_counts()
        return d_perm, None, None, d_probs, None, None, None


def fused_combine(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    num_tokens: int,
    *,
    permuted_probs: torch.Tensor | None = None,
    rows: torch.Tensor | None = None,
    impl: str = "auto",
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Differentiable fused combine. Drop-in for ``_fixed_order_combine`` under autograd.

    Under ``no_grad`` the ``autograd.Function`` is BYPASSED, not merely inert: ``apply`` still costs a
    node construction, a ctx, a ``save_for_backward`` and a python round trip per call. The forward it
    would have executed is the same raw kernel call this branch makes.
    """
    if not torch.is_grad_enabled():
        if rows is None:
            rows = build_combine_rows(sorted_indices, num_tokens)
            if rows is None:
                raise RuntimeError("[isoexec] fused_combine: non-topk layout; caller must fall back")
        out = fused_fixed_order_combine(
            permuted_tokens,
            sorted_indices,
            (num_tokens,),
            permuted_probs=permuted_probs,
            rows=rows,
            out_dtype=out_dtype,
        )
        _COUNTS["nograd_served"] += 1
        _COUNTS["rows"] += int(permuted_tokens.shape[0])
        _report_counts()
        return out
    return MoECombine.apply(permuted_tokens, sorted_indices, num_tokens, permuted_probs, rows, impl, out_dtype)


def differentiable_unpermute(
    permuted_tokens,
    sorted_indices,
    restore_shape,
    probs=None,
    routing_map=None,
    fused=False,
    drop_and_pad=False,
    isoexec_out_dtype=None,
    **kwargs,
):
    """Drop-in for ``moe_batch_invariant._deterministic_unpermute`` with the analytic backward.

    Same fallbacks as ``moe_combine_kernel.fused_deterministic_unpermute``; the only difference is
    that the combine goes through :class:`MoECombine`, so it is usable in the TRAINER.

    ``isoexec_out_dtype`` is the caller's own trailing cast, handed down so the kernel can perform it
    at the store (see ``moe_combine_kernel.legal_fold_dtype``). It is IsoExec-private: it is never
    forwarded to megatron's ``unpermute``, and every fallback branch below applies it as an ordinary
    ``.to()`` so the eager and fused paths return the same bytes whether or not the fold fires.
    """
    from . import moe_batch_invariant as mbi

    if fused or drop_and_pad:
        _decline("fused_or_drop_and_pad")
        return mbi._orig_unpermute(
            permuted_tokens,
            sorted_indices,
            restore_shape,
            probs=probs,
            routing_map=routing_map,
            fused=fused,
            drop_and_pad=drop_and_pad,
            **kwargs,
        )

    input_dtype = permuted_tokens.dtype
    want_dtype = input_dtype if isoexec_out_dtype is None else isoexec_out_dtype
    permuted_probs = None
    if probs is not None:
        assert routing_map is not None, "Mask must be provided to permute the probs."
        permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())

    num_tokens = int(restore_shape[0])
    rows = build_combine_rows(sorted_indices, num_tokens)
    if rows is None:
        _decline("layout_not_exact_topk")
        if permuted_probs is not None:
            permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)
        return mbi._orig_unpermute(
            permuted_tokens,
            sorted_indices,
            restore_shape,
            probs=None,
            routing_map=routing_map,
            fused=False,
            drop_and_pad=False,
            **kwargs,
        ).to(dtype=want_dtype)

    out = fused_combine(
        permuted_tokens,
        sorted_indices,
        num_tokens,
        permuted_probs=permuted_probs,
        rows=rows,
        out_dtype=isoexec_out_dtype,
    )
    # A no-op when the fold fired; the safety net when the kernel declined it. Never a second round:
    # `out` is already exactly `want_dtype` in the folded case, so `.to` returns `self`.
    return out.to(dtype=want_dtype)


differentiable_unpermute._isoexec_accepts_out_dtype = True


# The probs -> router tail. A reference adjoint, not an optimization: the [S, E] tensors here are three
# orders of magnitude smaller than the [P, h] traffic of the combine itself.
def unpermute_probs_grad(
    d_permuted_probs: torch.Tensor,
    flat_sorted: torch.Tensor,
    num_tokens: int,
    num_experts: int,
) -> torch.Tensor:
    """Adjoint of ``permuted_probs = probs.T.contiguous().reshape(-1)[flat_sorted]``.

    ``moe_utils.permute`` builds ``permuted_probs`` with that single fancy-index on the
    EXPERT-MAJOR flattening of ``probs``. Its adjoint is a scatter-add into the flat buffer,
    reshaped back and transposed. ``index_add_`` (not ``index_put_(accumulate=True)``) because each
    ``flat_sorted`` entry is unique under top-k routing, so this is really a scatter and the choice
    is about clarity, not about summation order.

    Returns ``d(probs)`` shaped ``[num_tokens, num_experts]``.
    """
    flat = d_permuted_probs.new_zeros(num_experts * num_tokens)
    flat.index_add_(0, flat_sorted, d_permuted_probs)
    return flat.view(num_experts, num_tokens).T.contiguous()
