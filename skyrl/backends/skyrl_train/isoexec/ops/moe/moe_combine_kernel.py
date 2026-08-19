"""Fused MoE combine: unpermute + fixed-order top-k sum in ONE Triton kernel.

``_fixed_order_combine`` seeds ``out[t]`` with a copy of the token's lowest-expert permuted row and adds
the remaining ``k-1`` rows in ascending permuted-row order, accumulating in ``permuted.dtype``. This kernel
walks the same rows in the same order in registers and writes ``[T, H]`` once; nothing arithmetic changes.
With pik-fc2 on, the accumulation is fp32 with a single bf16 round afterwards; with it off the accumulation
is bf16 and rounds after every add, which ``ROUND_BF16`` reproduces, so the kernel is a drop-in for both.
``probs`` is None in production (the router probs are folded into the expert epilogue); the weighted branch
exists only for a generic Megatron dispatcher.

INSTALLATION IS ENGINE-ONLY BY INSTANCE MARK. ``unpermute`` is a module-level function, so the only binding
megatron offers is a process-global rebind, and under ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` the engine and
the trainer share the process -- handing the trainer this bare ``no_grad`` kernel would sever the MoE
backward while the forward-only IsoExec gate stayed green. Since a free function has no ``self`` to mark,
the mark goes on the DISPATCHER and a scope wrapper on its combine entry points publishes it for the
duration of the call. Everything unmarked keeps ``_deterministic_unpermute``, or with
``SKYRL_ISOEXEC_MOE_FUSED_COMBINE_TRAINER=1`` the autograd-wrapped variant.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

# Element block along hidden. H is a reduction-free axis here (each hidden element is summed independently
# over the k rows), so this is a pure occupancy knob and ANY value gives bitwise the same result. Fixed
# rather than autotuned so the engine and the trainer cannot pick different configs.
_BLOCK_H = int(os.environ.get("SKYRL_ISOEXEC_MOE_COMBINE_BLOCK_H", "512"))
_NUM_WARPS = int(os.environ.get("SKYRL_ISOEXEC_MOE_COMBINE_WARPS", "4"))


@triton.jit
def _combine_topk_kernel(
    permuted_ptr,  # [P, H]
    rows_ptr,  # [T, K] int32, rows[t, j] = permuted-row index of token t's j-th expert
    probs_ptr,  # [P] fp32 or dummy
    out_ptr,  # [T, H]
    p_stride0,
    p_stride1,
    o_stride0,
    o_stride1,
    H,
    K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    USE_PROBS: tl.constexpr,
    ROUND_BF16: tl.constexpr,
):
    t = tl.program_id(0)
    hb = tl.program_id(1)
    offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs < H

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    # static_range so `j` is a python int: the j==0 "copy, don't add" special case (the
    # index_select that seeds `out` in _fixed_order_combine) is resolved at compile time, and the
    # k adds are emitted in ascending-j order with no reassociation.
    for j in tl.static_range(K):
        r = tl.load(rows_ptr + t * K + j).to(tl.int64)
        x = tl.load(permuted_ptr + r * p_stride0 + offs * p_stride1, mask=mask, other=0.0).to(tl.float32)
        if USE_PROBS:
            # torch promotes bf16_rows * fp32_probs to fp32 BEFORE the sum; fp32 rows stay fp32.
            x = x * tl.load(probs_ptr + r).to(tl.float32)
        if j == 0:
            acc = x  # seed = a pure copy of the lowest-expert row (no add, so no rounding)
        else:
            acc = acc + x
            if ROUND_BF16:
                # bf16 accumulation: torch's bf16 add computes in fp32 and rounds ONCE per add.
                # Reproduce that round here so an 8-term bf16 chain matches term for term.
                acc = acc.to(tl.bfloat16).to(tl.float32)

    tl.store(out_ptr + t * o_stride0 + offs * o_stride1, acc.to(out_ptr.dtype.element_ty), mask=mask)


def legal_fold_dtype(acc_out_dtype: torch.dtype, out_dtype: torch.dtype | None) -> torch.dtype:
    """Resolve the kernel's STORE dtype, refusing anything that would move a rounding boundary.

    The accumulator dtype (and therefore ``ROUND_BF16``) is decided by ``acc_out_dtype`` and is NEVER
    keyed on this. Only two cases are admissible: ``out_dtype`` absent or equal to ``acc_out_dtype``, so
    the store is unchanged; or an fp32 accumulator with a bf16 ``out_dtype``, which moves the caller's own
    trailing ``.to(torch.bfloat16)`` into the store -- still exactly one round, now in a register. Anything
    else would change the number or order of roundings, so it raises.
    """
    if out_dtype is None or out_dtype == acc_out_dtype:
        return acc_out_dtype
    if acc_out_dtype == torch.float32 and out_dtype == torch.bfloat16:
        return torch.bfloat16
    raise RuntimeError(
        "[isoexec] fused MoE combine: refusing out_dtype="
        f"{out_dtype} against accumulate dtype {acc_out_dtype}. The only fold that preserves the "
        "rounding schedule is fp32-accumulate -> bf16 store; anything else changes the number of "
        "roundings and must be classified as a COMPOSITION EVENT, not a fold."
    )


def build_combine_rows(sorted_indices: torch.Tensor, num_tokens: int) -> torch.Tensor | None:
    """``argsort(sorted_indices, stable=True).view(T, k)`` as int32, or None if the layout is not
    "every token routes to exactly k experts" (token dropping / capacity padding).

    Identical to what ``_fixed_order_combine`` computes; kept as its own function so a caller can
    cache it. int32 is a pure index-width change (P <= T*topk is far below 2^31 at any shape this
    stack runs) and does not touch the arithmetic.
    """
    n = int(sorted_indices.numel())
    if num_tokens == 0 or n == 0 or n % num_tokens != 0:
        return None
    k = n // num_tokens
    from .moe_fused_permute import get_preemitted_combine_rows

    rows = get_preemitted_combine_rows(sorted_indices, num_tokens)
    if rows is not None:
        return rows
    # Counting sort instead of the small-P radix sort. Same permutation, emitted directly as [T, k]
    # int32 so the fused path skips the view/cast too.
    from .moe_combine_rows_kernel import stable_combine_rows

    rows = stable_combine_rows(sorted_indices, num_tokens, dtype=torch.int32)
    if rows is not None:
        return rows
    return torch.argsort(sorted_indices, stable=True).view(num_tokens, k).to(torch.int32).contiguous()


# THE ROUND FOLD (SKYRL_ISOEXEC_MOE_COMBINE_FOLD_ROUND, default OFF). The pik alltoall combine writes the
# fp32 top-k sum to a [T, H] fp32 tensor and immediately reads it back to produce bf16
# (`combine_postprocess`). Asking the kernel to store bf16 deletes that round trip and cannot change a
# bit: it is the SAME single round-to-nearest-even of the SAME fp32 accumulator. The request travels as a
# keyword argument through a module-level rebind that several other levers also wrap, so `folded` is
# incremented inside the only place that can honour it; the caller keeps its unconditional
# `.to(torch.bfloat16)`, so a binding that drops the kwarg loses the saving but never correctness.
_FOLD = {"folded": 0, "requested": 0, "reported": 0}


def fold_round_enabled() -> bool:
    return os.environ.get("SKYRL_ISOEXEC_MOE_COMBINE_FOLD_ROUND", "0") == "1"


def fold_stats() -> dict:
    """Copy of the fold counters; live engagement requires ``folded > 0``, not ``requested > 0``."""
    return dict(_FOLD)


def note_fold_request() -> None:
    """Called by the combine call site when it ASKS for the fold. See ``fold_stats``."""
    _FOLD["requested"] += 1
    n = _FOLD["requested"]
    if (n & (n - 1)) == 0 and n != _FOLD["reported"]:
        _FOLD["reported"] = n
        print(
            f"[ISOEXEC-MOE-COMBINE-FOLD] pid={os.getpid()} requested={n} folded={_FOLD['folded']}"
            + ("" if _FOLD["folded"] else "  <-- INERT: the kwarg is not reaching the kernel"),
            flush=True,
        )


def fused_fixed_order_combine(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape,
    *,
    permuted_probs: torch.Tensor | None = None,
    rows: torch.Tensor | None = None,
    validate: bool = False,
    out_dtype: torch.dtype | None = None,
):
    """Drop-in for ``moe_batch_invariant._fixed_order_combine``, fused into one Triton kernel.

    Returns ``[T, H]`` in ``permuted_tokens.dtype`` (or fp32 when ``permuted_probs`` is given and
    promotes), or ``None`` when the layout is unsupported -- so the caller can fall back exactly
    as the production code does.

    ``permuted_probs`` is the ALREADY-PERMUTED ``[P]`` prob vector (what
    ``probs.T.masked_select(routing_map.T)`` produces); production passes None.
    """
    num_tokens = int(restore_shape[0])
    if rows is None:
        rows = build_combine_rows(sorted_indices, num_tokens)
        if rows is None:
            return None
    k = rows.shape[1]

    if validate:
        expected = torch.arange(num_tokens, device=sorted_indices.device).repeat_interleave(k)
        got = sorted_indices[rows.view(-1).long()].to(expected.dtype)
        if not torch.equal(got, expected):
            raise RuntimeError(
                "[isoexec] fused MoE combine: tokens do not each route to exactly topk experts "
                f"(num_tokens={num_tokens}, permuted_rows={int(sorted_indices.numel())})."
            )

    H = permuted_tokens.shape[-1]
    in_dtype = permuted_tokens.dtype
    use_probs = permuted_probs is not None
    # Match torch's promotion rule for the generic weighted compatibility branch.
    acc_out_dtype = torch.promote_types(in_dtype, permuted_probs.dtype) if use_probs else in_dtype
    # bf16 accumulation rounds after every add; fp32 accumulation does not round at all.
    # KEYED ON acc_out_dtype AND NOT ON THE STORE DTYPE. Folding a trailing bf16 cast into the store
    # must not turn an fp32 sum into a per-term-rounded bf16 sum -- those are different functions.
    round_bf16 = acc_out_dtype == torch.bfloat16
    store_dtype = legal_fold_dtype(acc_out_dtype, out_dtype)
    if store_dtype != acc_out_dtype:
        _FOLD["folded"] += 1

    out = torch.empty(num_tokens, H, dtype=store_dtype, device=permuted_tokens.device)
    grid = (num_tokens, triton.cdiv(H, _BLOCK_H))
    _combine_topk_kernel[grid](
        permuted_tokens,
        rows,
        permuted_probs if use_probs else permuted_tokens,  # dummy ptr when unused
        out,
        permuted_tokens.stride(0),
        permuted_tokens.stride(1),
        out.stride(0),
        out.stride(1),
        H,
        K=k,
        BLOCK_H=_BLOCK_H,
        USE_PROBS=use_probs,
        ROUND_BF16=round_bf16,
        num_warps=_NUM_WARPS,
        # NON-NEGOTIABLE. Without it LLVM contracts `acc + x * prob` into a single FMA -- one rounding
        # step instead of two -- and the weighted combine drifts from torch's. The unweighted production
        # path has no multiply, but the flag stays on so no future edit can reintroduce a contraction.
        enable_fp_fusion=False,
    )
    return out


def fused_deterministic_unpermute(
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
    """Drop-in for ``moe_batch_invariant._deterministic_unpermute`` using the fused kernel.

    Same fallbacks: ``fused``/``drop_and_pad`` and any non-topk layout go to megatron's original.

    ``isoexec_out_dtype`` is the caller's trailing cast, moved into the kernel's store. It is a
    ISOEXEC-PRIVATE kwarg and is never forwarded to megatron's own ``unpermute``.
    """
    from . import moe_batch_invariant as mbi

    if fused or drop_and_pad:
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

    # Counter ticks HERE, one level above the fused call: the dispatcher wraps the
    # module-global ``fused_fixed_order_combine`` symbol, so a tick inside it would miss every
    # region-served call. This caller is the site's single entry.
    out = fused_fixed_order_combine(
        permuted_tokens,
        sorted_indices,
        restore_shape,
        permuted_probs=permuted_probs,
        out_dtype=isoexec_out_dtype,
    )
    if out is None:
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
    out = out.to(dtype=want_dtype)
    return out


fused_deterministic_unpermute._isoexec_accepts_out_dtype = True


_fused_combine_installed = False
# Depth-counted so a nested combine (none today, but the flex dispatcher composes) cannot leave the
# scope stuck on. Single-threaded forward on both runtimes, same assumption
# `_make_sorted_topk_routing` makes.
_ACTIVE = 0


def fused_combine_enabled() -> bool:
    return os.environ.get("SKYRL_ISOEXEC_MOE_FUSED_COMBINE", "0") == "1"


def _engine_combine_active() -> bool:
    return _ACTIVE > 0


def _scoped(orig, name):
    """Wrap a dispatcher combine entry point so the fused binding is live only for MARKED instances."""

    def wrapper(self, *a, **kw):
        global _ACTIVE
        marked = getattr(self, "_isoexec_engine_combine", False)
        if not marked:
            return orig(self, *a, **kw)
        _ACTIVE += 1
        try:
            return orig(self, *a, **kw)
        finally:
            _ACTIVE -= 1

    wrapper.__name__ = name
    wrapper._isoexec_combine_scope = True
    return wrapper


def install_fused_combine(side: str = "ENGINE") -> bool:
    """Install the fused combine as an ENGINE-SCOPED binding. Requires ``enable_moe_deterministic_ops()``
    first (it captures megatron's original for the fallback path).

    SIDE MATTERS. :func:`fused_deterministic_unpermute` calls ``fused_fixed_order_combine``, a RAW Triton
    call whose output has no ``grad_fn``, so on a runtime that runs backward it severs the graph while the
    forward-only IsoExec gate stays green. The trainer therefore gets
    ``moe_combine_backward.differentiable_unpermute`` -- the same kernel wrapped in an
    ``autograd.Function``, forward bitwise-identical, backward analytic. The engine keeps the bare call,
    which runs under ``no_grad``.

    Because both runtimes share this process, "the trainer gets the wrapped variant" cannot be expressed
    by WHICH side calls this function -- the last rebind would win for both. It is expressed by the
    dispatcher instance mark.
    """
    global _fused_combine_installed
    if _fused_combine_installed:
        return True
    from megatron.core.transformer.moe import moe_utils, token_dispatcher
    from megatron.core.transformer.moe.token_dispatcher import (
        MoEAllGatherTokenDispatcher,
        MoEAlltoAllTokenDispatcher,
    )

    from . import moe_batch_invariant as mbi

    if mbi._orig_unpermute is None:
        raise RuntimeError("[isoexec] install_fused_combine: call enable_moe_deterministic_ops() first")

    def _unpermute(permuted_tokens, sorted_indices, restore_shape, **kw):
        if _engine_combine_active() and fused_combine_enabled():
            return fused_deterministic_unpermute(permuted_tokens, sorted_indices, restore_shape, **kw)
        if os.environ.get("SKYRL_ISOEXEC_MOE_FUSED_COMBINE_TRAINER", "0") == "1":
            from .moe_combine_backward import differentiable_unpermute

            return differentiable_unpermute(permuted_tokens, sorted_indices, restore_shape, **kw)
        return mbi._deterministic_unpermute(permuted_tokens, sorted_indices, restore_shape, **kw)

    # Every branch of `_unpermute` forwards **kw to a binding that accepts the round-fold request,
    # so the wrapper itself must publish the marker or the fold would be inert exactly when the
    # fused combine is installed -- the one configuration it exists for.
    _unpermute._isoexec_accepts_out_dtype = True
    moe_utils.unpermute = _unpermute
    token_dispatcher.unpermute = _unpermute
    # Installed LAST in prepare_isoexec_moe, so these wrappers sit OUTSIDE the pik-fc2 combine
    # patches and the scope is live for the unpermute they reach.
    for cls, meth in (
        (MoEAllGatherTokenDispatcher, "combine_preprocess"),
        (MoEAlltoAllTokenDispatcher, "combine_postprocess"),
    ):
        cur = getattr(cls, meth)
        if not getattr(cur, "_isoexec_combine_scope", False):
            setattr(cls, meth, _scoped(cur, meth))
    _fused_combine_installed = True
    print(
        f"[ISOEXEC-MOE] {side}: MoE combine binding installed (engine-marked instances -> fused Triton "
        "unpermute+top-k kernel; everything else -> eager fixed-order combine)",
        flush=True,
    )
    return True


_marked_combine = 0


def mark_engine_fused_combine(model) -> int:
    """Mark every token dispatcher in the ENGINE model for the fused combine.

    Called after the engine GPTModel is built, never from the trainer. Marking is UNCONDITIONAL so the
    line below logs the flag's value AT ENGINE BUILD: if it prints OFF on a run that set it ON, the env
    never reached the actor and any A/B against that run is void.
    """
    global _marked_combine
    n = 0
    for m in model.modules():
        d = getattr(m, "token_dispatcher", None)
        if d is not None:
            d._isoexec_engine_combine = True
            n += 1
    _marked_combine = n
    state = "ON -- 25.8 -> 1 launch/layer" if fused_combine_enabled() else "OFF (8x gather+add per layer)"
    print(
        f"[ISOEXEC-MOE] ENGINE fused MoE combine marked on {n} dispatchers; SKYRL_ISOEXEC_MOE_FUSED_COMBINE={state}",
        flush=True,
    )
    return n
