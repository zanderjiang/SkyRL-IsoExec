"""Shared batch-invariant GDN ops -- one implementation used by both the engine and the trainer.

The two runtimes must execute the same code for every op, and that code must be batch- and
context-invariant. ``gdn_causal_conv`` expresses the width-4 causal depthwise conv as a sum of shifted
scaled copies -- all elementwise, so token t depends on tokens t-3..t and nothing else -- accumulating
in fp32 and rounding once, rather than picking one of vLLM's two fused kernels (prefill vs decode do
not agree bitwise). ``gdn_l2norm`` delegates to vLLM's row-local ``l2norm_fwd``, and ``gdn_chunk`` is
the chunked delta-rule kernel with autotune configs pinned (see ``gdn_batch_invariant``).
"""

from __future__ import annotations

import os

import torch

from ...autofuse.bwd_compile import call_region


# Which delta-rule kernel the whole stack runs on. The requirement is only that the trainer, engine
# prefill and engine decode all evaluate the core with the SAME kernel; the two ways to satisfy that
# trade cost in opposite directions:
#
#   "chunk"      everything runs the chunked-parallel kernel that training wants. Decode then re-runs
#                the chunk kernel over the whole open chunk every step (~C/2 token-rows of work per
#                decoded token) and keeps an open-chunk buffer per live request per layer.
#   "recurrent"  everything runs the recurrent kernel that decode wants. Decode is native (one
#                token-row, no buffers), while training and prefill pay for a scan that is sequential
#                in T and launches one tiny program per (sequence, head, v-block).
#
# Both are bitwise; which is faster end to end depends on the rollout/train mix, so the kernel is a
# switch rather than a decision baked into the call sites.
#
# The chunk kernel needs its autotuner pinned to be invariant. The recurrent kernel is invariant by
# construction: its grid is (1, NV, N*HV), so no reduction crosses a sequence; each program carries the
# state in fp32 registers and walks tokens in a plain `for` loop, which is prefix invariance; and it
# does not autotune, with `do_not_specialize=["N", "T"]` so decode (T=1) and prefill (T=P) run the same
# compiled kernel. Chaining prefill -> decode is then bitwise as long as the state round-trips exactly,
# which it does because the ssm state is stored in fp32.
def gdn_kernel_mode() -> str:
    """The delta-rule kernel this process runs. Read at call time; every GDN site must agree.

    Vocabulary, default and parsing live in ``core/gdn_kernel_env`` -- the same answer the
    DECLARATION site (models/qwen3_5) hashes into the contract. They used to be parsed separately
    here, which is how ``SKYRL_ISOEXEC_GDN_KERNEL=CPR`` made both runtimes derive the same
    WRONG contract and the handshake MATCH.
    """
    from ...core.gdn_kernel_env import gdn_kernel_mode as _mode

    return _mode()


def recurrent_mode() -> bool:
    return gdn_kernel_mode() == "recurrent"


def cpr_mode() -> bool:
    return gdn_kernel_mode() == "cpr"


# Minimal vLLM mamba pages under cpr (engine memory only; moves no bits). cpr never
# reads vLLM's native GDN state pages -- CprGDN keeps its own private pools and uses the vLLM
# pages only as a slot-id source -- yet those full-size pages dominate the KV pool, because vLLM pads
# the mamba page up to one attention page and bumps the attention block size until it covers the real
# state. Every live request then pins memory nothing reads, starving attention KV into preemption.
#
# With this flag both state-shape sources (config-time page sizing and the runtime MambaSpec, which
# must agree or vLLM's page unification asserts) report GDN_CPR_MIN_STATE_SHAPES, so the mamba page
# collapses to one minimal attention page. Everything cpr needs from the pages survives,
# because it never depended on their size: slot ids stay unique per live request per kv-cache group,
# stable for the request's lifetime, positive for live requests, and a freed id is only re-issued to a
# request that then prefills. Scoped to cpr with private pools: recurrent mode and
# SKYRL_ISOEXEC_GDN_NATIVE_STATE=1 read the native pages for real state and keep full shapes.
_CPR_MIN_PAGES_ENV = "SKYRL_ISOEXEC_GDN_CPR_MIN_PAGES"

# (conv-like, ssm-like) minimal shapes. Two entries so the (conv, ssm) kv_cache tuple structure
# survives (gdn_engine_patch/gdn_gptmodel read kv_cache[1].shape[0] as a capacity fallback).
# Sized 16 bytes each so the runner's strided carve-out keeps the second (fp32) tensor aligned
# for ANY state-dtype combination (gpu_model_runner asserts storage_offset % dtype_size == 0).
GDN_CPR_MIN_STATE_SHAPES = ((1, 8), (1, 1, 4))


def gdn_cpr_min_pages() -> bool:
    """True iff vLLM's native GDN state pages should be MINIMAL (slot-id source only).

    Read at call time like every mode flag. Only true in cpr-with-private-pools mode:
    recurrent/native-state compositions store REAL state in those pages.
    """
    if os.environ.get(_CPR_MIN_PAGES_ENV, "0").lower() in ("", "0", "false", "no"):
        return False
    if not cpr_mode():
        return False
    from .gdn_recurrent_state import (
        native_state_enabled,  # lazy: that module imports gdn_ops
    )

    return not native_state_enabled()


# Swap the conv (only) to the native vLLM pair -- causal_conv1d_fn on the trainer/prefill,
# causal_conv1d_update at decode -- while l2norm/gating/core stay the isoexec composition. The native
# pair is one varlen launch instead of the eager per-sequence loop. The flag must flip both runtimes in
# one run: the two convs round differently (bias in the fp32 accumulator vs added after the taps), so a
# one-sided flip moves the forward on one side only.
_NATIVE_CONV_ENV = "SKYRL_ISOEXEC_GDN_NATIVE_CONV"


def gdn_native_conv_enabled() -> bool:
    """True iff cpr runs the native vLLM conv pair on both runtimes."""
    return os.environ.get(_NATIVE_CONV_ENV, "0").lower() not in ("", "0", "false", "no")


def gdn_causal_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    return_final_state: bool = False,
):
    """Width-`W` causal depthwise conv over a token sequence. Batch/prefix invariant by construction.

    Args:
        x: ``[T, D]`` input tokens (a single sequence, or one chunk of one).
        weight: ``[D, W]`` depthwise taps; ``weight[:, -1]`` multiplies the current token.
        bias: ``[D]`` or None.
        initial_state: ``[D, W-1]`` the previous ``W-1`` inputs (oldest first), or None for a fresh
            sequence (equivalent to zeros).
        activation: ``"silu"`` / ``"swish"`` or None.
        return_final_state: also return the ``[D, W-1]`` state after consuming ``x``.

    Returns:
        ``y [T, D]``, and ``final_state [D, W-1]`` when requested.

    Every op below is elementwise over the token axis, so ``y[t]`` is a pure function of
    ``x[t-W+1 .. t]``; slicing a prefix, changing the batch, or splitting the sequence into chunks
    cannot change it.
    """
    if x.ndim != 2:
        raise ValueError(f"gdn_causal_conv expects x=[T, D], got {tuple(x.shape)}")
    T, D = x.shape
    W = weight.shape[-1]
    if weight.shape[0] != D:
        raise ValueError(f"weight {tuple(weight.shape)} incompatible with x dim {D}")

    if initial_state is None:
        pad = x.new_zeros(W - 1, D)
    else:
        if initial_state.shape != (D, W - 1):
            raise ValueError(f"initial_state must be [D, W-1]={(D, W - 1)}, got {tuple(initial_state.shape)}")
        pad = initial_state.transpose(0, 1).to(x.dtype)  # [W-1, D], oldest first
    xp = torch.cat([pad, x], dim=0)  # [T + W - 1, D]

    # y[t] = sum_i w[:, i] * xp[t + i]   (i = 0..W-1; i = W-1 is the current token)
    acc = torch.zeros(T, D, dtype=torch.float32, device=x.device)
    wf = weight.float()
    for i in range(W):
        acc = acc + wf[:, i].unsqueeze(0) * xp[i : i + T].float()
    if bias is not None:
        acc = acc + bias.float().unsqueeze(0)

    if activation in ("silu", "swish"):
        acc = acc * torch.sigmoid(acc)
    elif activation is not None:
        raise ValueError(f"unsupported activation {activation!r}")

    y = acc.to(x.dtype)
    if not return_final_state:
        return y
    final_state = xp[T:].transpose(0, 1).contiguous()  # last W-1 inputs, [D, W-1]
    return y, final_state


def gdn_causal_conv_batched(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
) -> torch.Tensor:
    """Batched :func:`gdn_causal_conv`: ``x [N, T, D]``, ``initial_state [N, D, W-1]`` -> ``y [N, T, D]``.

    The same elementwise shifted-sum expression with a leading batch dim, so ``y[i]`` is
    bitwise-identical to ``gdn_causal_conv(x[i], ..., initial_state[i])``: every op is elementwise over
    (batch, token) and fp32-accumulated in the same fixed order. This lets chunk-consistent decode run
    one conv over all open chunks instead of a per-slot python loop.
    """
    if x.ndim != 3:
        raise ValueError(f"gdn_causal_conv_batched expects x=[N, T, D], got {tuple(x.shape)}")
    N, T, D = x.shape
    W = weight.shape[-1]
    if weight.shape[0] != D:
        raise ValueError(f"weight {tuple(weight.shape)} incompatible with x dim {D}")

    if initial_state is None:
        pad = x.new_zeros(N, W - 1, D)
    else:
        if initial_state.shape != (N, D, W - 1):
            raise ValueError(f"initial_state must be [N, D, W-1]={(N, D, W - 1)}, got {tuple(initial_state.shape)}")
        pad = initial_state.transpose(1, 2).to(x.dtype)  # [N, W-1, D], oldest first
    xp = torch.cat([pad, x], dim=1)  # [N, T + W - 1, D]

    acc = torch.zeros(N, T, D, dtype=torch.float32, device=x.device)
    wf = weight.float()
    for i in range(W):
        acc = acc + wf[:, i].unsqueeze(0).unsqueeze(0) * xp[:, i : i + T].float()
    if bias is not None:
        acc = acc + bias.float().unsqueeze(0).unsqueeze(0)

    if activation in ("silu", "swish"):
        acc = acc * torch.sigmoid(acc)
    elif activation is not None:
        raise ValueError(f"unsupported activation {activation!r}")
    return acc.to(x.dtype)


L2NORM_EPS = 1e-6


class _GdnL2NormAutograd(torch.autograd.Function):
    """vLLM's ``l2norm_fwd`` in the forward, autograd of the same expression in the backward.

    ``l2norm_fwd`` is a bare Triton launch writing into ``torch.empty_like(x)``, so its result carries
    no autograd history and backprop through it would silently deliver zero gradient to q and k while
    the loss still fell. Forward keeps the kernel (bitwise with the engine); backward differentiates
    ``x * rsqrt(sum(x^2) + eps)``, the expression the kernel evaluates.
    """

    @staticmethod
    def forward(ctx, x):
        from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd

        with torch.no_grad():
            y = l2norm_fwd(x)
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        with torch.enable_grad():
            xd = x.detach().float().requires_grad_(True)
            y = xd * torch.rsqrt(xd.pow(2).sum(-1, keepdim=True) + L2NORM_EPS)
            (dx,) = torch.autograd.grad(y, xd, dy.float())
        return dx.to(x.dtype)


def gdn_l2norm(x: torch.Tensor) -> torch.Tensor:
    """Row-local L2 normalisation, via the same kernel the engine and trainer both import.

    Meta/FakeTensor execution is the official compiler-adapter surface for this opaque manual op.
    It describes only the output tensor contract; it neither decomposes nor reimplements the
    arithmetic, and therefore cannot become an AUTOFUSE candidate itself.
    """
    if x.device.type == "meta":
        return torch.empty_strided(tuple(x.shape), tuple(x.stride()), dtype=x.dtype, device="meta")
    from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd

    if torch.is_grad_enabled() and x.requires_grad:
        return _GdnL2NormAutograd.apply(x)
    return l2norm_fwd(x)


def fla_chunk_size() -> int:
    from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE

    return FLA_CHUNK_SIZE


def _gdn_chunk_fwd(
    q, k, v, g, beta, initial_state, output_final_state, cu_seqlens, chunk_indices=None, chunk_offsets=None
):
    """The bitwise forward: vLLM's vendored FLA chunk kernel with pinned autotune configs.

    ``chunk_indices``/``chunk_offsets`` may be supplied by the caller. They are a pure function of
    ``cu_seqlens``, so chunk-consistent decode, where every GDN layer in a step is handed the same
    cu_seqlens, computes them once per step rather than once per layer.
    """
    from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule
    from vllm.model_executor.layers.fla.ops.index import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
    )
    from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE

    from .gdn_batch_invariant import pin_fla_autotune_configs

    pin_fla_autotune_configs()  # idempotent; must be in effect before the first launch

    if cu_seqlens is not None and chunk_indices is None:
        # `prepare_chunk_indices` is `@tensor_cache`d on tensor identity and vLLM recycles its
        # metadata buffers, so feeding it the caller's tensor can hand back a chunk map built for a
        # previous batch's cu_seqlens. The recompute is expensive (a `.tolist()` device sync plus one
        # CPU arange per sequence), so a fresh clone per call is wrong in the other direction.
        # `fla_stable_clone` keeps the trap shut -- a recycled buffer has a different id or a bumped
        # _version -- while giving every layer of a forward the same object, so FLA's cache hits.
        # Callers that already know the chunk map should pass it in and skip all of this.
        from .packed_meta_cache import fla_stable_clone

        cu_fresh = fla_stable_clone(cu_seqlens)
        chunk_indices = prepare_chunk_indices(cu_fresh, FLA_CHUNK_SIZE)
        chunk_offsets = prepare_chunk_offsets(cu_fresh, FLA_CHUNK_SIZE)

    return chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        use_qk_l2norm_in_kernel=False,  # done outside, identically on both sides
    )


def _torch_chunk_gdr_one(q, k, v, g, beta, initial_state, chunk_size):
    """Differentiable fp32 chunked delta rule for a batch of equal-length sequences.

    ``q..beta``: ``[B, T, H, D]``. B>1 is the micro_*_batch_size_per_gpu>1 trainer micro-forward;
    every sequence is advanced independently (the batch dim never enters a reduction), so row
    values match B=1 exactly.

    This is the megatron ``torch_chunk_gated_delta_rule`` reference, unchanged except that it takes
    ``initial_state`` in the kernel's ``[N, H, V, K]`` layout. It exists only to supply a
    vector-Jacobian product (see :class:`_GdnChunkAutograd`); it never runs in the forward.
    """
    q, k, v, beta, g = (x.transpose(1, 2).contiguous().float() for x in (q, k, v, beta, g))
    _, num_heads, T, k_dim = k.shape
    v_dim = v.shape[-1]

    pad = (chunk_size - T % chunk_size) % chunk_size
    q, k, v = (torch.nn.functional.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    beta, g = (torch.nn.functional.pad(x, (0, pad)) for x in (beta, g))
    Tp = T + pad
    q = q * (k_dim**-0.5)

    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)
    q, k, v, k_beta, v_beta = (
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (q, k, v, k_beta, v_beta)
    )
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    eye_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    g = g.cumsum(dim=-1)
    decay_mask = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().tril()
    # A is strictly lower triangular, and the inverse (I - A)^-1 is only ever used for the two matmuls
    # below -- which is a triangular solve. One batched `solve_triangular` (unitriangular, since I - A
    # has a unit diagonal by construction) replaces the reference's forward-substitution python loop.
    # This function is the VJP reference and never runs in the forward, so agreement to fp32 rounding
    # is enough; the backward only has to be the gradient of the forward to floating-point accuracy.
    A = -((k_beta @ k.transpose(-1, -2)) * decay_mask).masked_fill(eye_mask, 0)
    eye = torch.eye(chunk_size, dtype=A.dtype, device=A.device)
    rhs = torch.cat([v_beta, k_beta * g.exp().unsqueeze(-1)], dim=-1)
    sol = torch.linalg.solve_triangular(eye - A, rhs, upper=False, unitriangular=True)
    u, k_cumdecay = sol[..., :v_dim], sol[..., v_dim:]
    if initial_state is None:
        state = q.new_zeros(q.shape[0], num_heads, k_dim, v_dim)
    else:
        state = initial_state.transpose(-1, -2).float()  # [N,H,V,K] -> [N,H,K,V]

    # The chunk loop below is sequential (each step carries `state`) but launch-bound rather than
    # compute-bound, so everything that does not depend on `state` is hoisted and computed batched
    # across all chunks, leaving three matmuls inside the loop.
    strict_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)
    attn_all = (q @ k.transpose(-1, -2) * decay_mask).masked_fill(strict_mask, 0)  # [.., NC, C, C]
    qg_all = q * g.exp().unsqueeze(-1)  # [.., NC, C, K]
    g_last = g[..., -1:]  # [.., NC, 1]
    kg_all = (k * (g_last - g).exp().unsqueeze(-1)).transpose(-1, -2)  # [.., NC, K, C]
    decay_last = g_last.unsqueeze(-1).exp()  # [.., NC, 1, 1]

    outs = []
    for i in range(Tp // chunk_size):
        v_new = u[:, :, i] - k_cumdecay[:, :, i] @ state
        outs.append(qg_all[:, :, i] @ state + attn_all[:, :, i] @ v_new)
        state = state * decay_last[:, :, i] + kg_all[:, :, i] @ v_new

    o = torch.stack(outs, dim=2).reshape(q.shape[0], num_heads, Tp, v_dim)[:, :, :T]
    return o.transpose(1, 2)  # [B, T, H, V]


def _torch_chunk_gdr(q, k, v, g, beta, initial_state, cu_seqlens, chunk_size):
    if cu_seqlens is None:
        return _torch_chunk_gdr_one(q, k, v, g, beta, initial_state, chunk_size)
    from .packed_meta_cache import cu_list

    bounds = cu_list(cu_seqlens)  # memoized host read; the backward pays it once, not once/layer
    outs = []
    for n, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])):
        s0 = None if initial_state is None else initial_state[n : n + 1]
        outs.append(_torch_chunk_gdr_one(q[:, s:e], k[:, s:e], v[:, s:e], g[:, s:e], beta[:, s:e], s0, chunk_size))
    return torch.cat(outs, dim=1)


class _GdnChunkAutograd(torch.autograd.Function):
    """Bitwise kernel forward + reference VJP backward.

    vLLM vendors FLA's chunk kernel for inference only: it defines a ``forward`` and no ``backward``,
    so autograd raises as soon as the trainer backprops through it. The forward must stay that exact
    kernel -- it is why decode and training agree bitwise -- so the backward instead differentiates
    :func:`_torch_chunk_gdr`, the fp32 torch reference for the same function, at the same inputs. That
    gradient need only be the gradient of the forward to floating-point accuracy.
    """

    @staticmethod
    def forward(ctx, q, k, v, g, beta, initial_state, cu_seqlens, chunk_size):
        with torch.no_grad():
            o, _ = _gdn_chunk_fwd(q, k, v, g, beta, initial_state, False, cu_seqlens)
        ctx.save_for_backward(q, k, v, g, beta, initial_state)
        ctx.cu_seqlens = cu_seqlens
        ctx.chunk_size = chunk_size
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, g, beta, initial_state = ctx.saved_tensors
        from .gdn_fla_backward import fla_backward_enabled, fla_chunk_vjp

        if fla_backward_enabled():
            # FLA's fused Triton backward, correct to floating-point accuracy; only the forward, which
            # is unchanged, needs to be bitwise. The reference VJP below is the fallback.
            grads = fla_chunk_vjp(q, k, v, g, beta, do, initial_state, ctx.cu_seqlens, ctx.chunk_size)
            return (*grads, None, None, None)
        with torch.enable_grad():
            leaves = [t.detach().requires_grad_(True) for t in (q, k, v, g, beta)]
            o = _torch_chunk_gdr(*leaves, initial_state, ctx.cu_seqlens, ctx.chunk_size)
            grads = torch.autograd.grad(o, leaves, do.float())
        grads = [gr.to(t.dtype) for gr, t in zip(grads, (q, k, v, g, beta))]
        return (*grads, None, None, None)


def gdn_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
):
    """`chunk_gated_delta_rule` with pinned configs. q/k must already be L2-normalised.

    Returns ``(o, final_state)``. ``final_state`` is meaningful only when the trailing chunk of each
    sequence is FULL -- for a partial chunk it is the state after that partial chunk, which is not a
    point on the chunk grid.

    ``chunk_indices``/``chunk_offsets`` are an optional precomputed chunk map for ``cu_seqlens``
    (see :func:`_gdn_chunk_fwd`); omit them and they are derived, at the cost of a GPU sync.

    Under ``torch.no_grad`` (both rollout paths and the trainer's scoring forward) this is exactly
    the vLLM kernel. When a gradient is required it routes through :class:`_GdnChunkAutograd`, whose
    forward is that same kernel -- so the training forward stays bitwise equal to the rollout.
    """
    needs_grad = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in (q, k, v, g, beta, initial_state)
    )
    if not needs_grad:
        return _gdn_chunk_fwd(
            q, k, v, g, beta, initial_state, output_final_state, cu_seqlens, chunk_indices, chunk_offsets
        )

    if output_final_state:
        raise NotImplementedError(
            "isoexec GDN: output_final_state is not differentiable (training never asks for it; "
            "only chunk-consistent decode does, under no_grad)."
        )
    o = _GdnChunkAutograd.apply(q, k, v, g, beta, initial_state, cu_seqlens, fla_chunk_size())
    return o, None


def gdn_recurrent_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """The raw fused recurrent delta-rule scan. Advances ``ssm_state`` IN PLACE. Returns ``o``.

    This is the single forward every runtime shares in ``recurrent`` mode: the trainer, the engine's
    prefill and the engine's decode all land here.

    Args:
        q, k: ``[B, T, H, K]``, already L2-normalised and GQA-expanded to ``H == HV``.
        v: ``[B, T, HV, V]``; g, beta: ``[B, T, HV]``.
        ssm_state: the fp32 state POOL, ``[S, HV, V, K]``. Rows are addressed by ``state_indices``,
            never by position -- see below. Written in place.
        state_indices: which row of the pool each (sequence, token) reads/writes.
            ``[N]`` for a one-token-per-sequence decode; ``[N, Tmax]`` otherwise.
        cu_seqlens: ``[N+1]`` int32 for a packed batch (then ``B == 1``); None for ``[B, T, ...]``.

    ``state_indices`` is never None. The kernel's other path addresses the state by the sequence's
    token offset, which would make ``initial_state`` a per-token ``[T, HV, V, K]`` array -- hundreds of
    KB per token per layer. The continuous-batching path indexes by row instead, so it is used
    unconditionally, including in training where the pool is a throwaway ``[N+1]`` scratch.

    The kernel stores the running state after every token (there is no final-state-only flag) but skips
    the store when the index is <= 0. A caller that wants only the final state therefore zeroes every
    column of ``state_indices`` except each sequence's last real token, and the state is written once
    per sequence. The skip cannot perturb ``o``: the state lives in registers and the store is
    write-only.
    """
    from vllm.model_executor.layers.fla.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule,
    )

    o, _ = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=None,  # -> K**-0.5, the same scale chunk_gated_delta_rule and the fp32 reference use
        initial_state=ssm_state,
        inplace_final_state=True,  # final state is written back into the pool, at state_indices
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=False,  # done outside, identically on both sides
    )
    return o


def _recurrent_scratch_state(q, v, cu_seqlens):
    """A throwaway state pool + index map for a forward that wants no state in and none out.

    That is every training call: sequences start from a zero state and nothing downstream reads the
    final one. Row 0 of the pool is unused because the kernel treats index <= 0 as "skip this lane", so
    sequence n lives at row n+1. Only column 0 of the index map is set: it is what the initial-state
    load reads (it must be a valid row, or the kernel returns immediately), and it makes the kernel
    perform exactly one throwaway state store per sequence instead of one per token.
    """
    if cu_seqlens is not None:
        N = cu_seqlens.numel() - 1
        lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()  # host-side; training already syncs here
        Tmax = max(lens) if lens else 1
    else:
        N, Tmax = q.shape[0], q.shape[1]

    HV, V, K = v.shape[-2], v.shape[-1], q.shape[-1]
    ssm_state = q.new_zeros(N + 1, HV, V, K, dtype=torch.float32)
    # Stride-stable index grid. The kernels take the grid's row stride as a tl.constexpr, so every
    # distinct stride is a separate Triton compile; an [N, Tmax] allocation would recompile for every
    # distinct max sequence length. Allocating a power-of-2-bucketed row width and handing out a
    # [:, :Tmax] view keeps the values and semantics identical while the stride takes ~log2 values.
    pad = max(64, 1 << (Tmax - 1).bit_length())
    idx = torch.zeros(N, pad, dtype=torch.int32, device=q.device)
    idx[:, 0] = torch.arange(1, N + 1, dtype=torch.int32, device=q.device)
    return ssm_state, idx[:, :Tmax]


class _GdnRecurrentAutograd(torch.autograd.Function):
    """Recurrent kernel forward + the SAME fp32 chunked reference VJP the chunk path uses.

    The chunked and the recurrent delta rule are two evaluation strategies for one mathematical
    function, so a VJP of that function is a VJP of either. Differentiating the recurrent scan directly
    would also mean a T-long sequential python loop in the backward, whereas ``_torch_chunk_gdr`` stays
    chunked and batched, so only the forward differs between the two modes.
    """

    @staticmethod
    def forward(ctx, q, k, v, g, beta, cu_seqlens, chunk_size):
        with torch.no_grad():
            ssm_state, idx = _recurrent_scratch_state(q, v, cu_seqlens)
            o = gdn_recurrent_kernel(q, k, v, g, beta, ssm_state=ssm_state, state_indices=idx, cu_seqlens=cu_seqlens)
        ctx.save_for_backward(q, k, v, g, beta)
        ctx.cu_seqlens = cu_seqlens
        ctx.chunk_size = chunk_size
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, g, beta = ctx.saved_tensors
        from .gdn_fla_backward import fla_backward_enabled, fla_chunk_vjp

        if fla_backward_enabled():
            # The chunk VJP is a valid gradient of the recurrent forward: same function, two
            # evaluations. See the class docstring.
            grads = fla_chunk_vjp(q, k, v, g, beta, do, None, ctx.cu_seqlens, ctx.chunk_size)
            return (*grads, None, None)
        with torch.enable_grad():
            leaves = [t.detach().requires_grad_(True) for t in (q, k, v, g, beta)]
            o = _torch_chunk_gdr(*leaves, None, ctx.cu_seqlens, ctx.chunk_size)
            grads = torch.autograd.grad(o, leaves, do.float())
        grads = [gr.to(t.dtype) for gr, t in zip(grads, (q, k, v, g, beta))]
        return (*grads, None, None)


# The native composition: vLLM's own GDN kernels, one composition for trainer and engine.
_NATIVE_KERNELS_ENV = "SKYRL_ISOEXEC_GDN_NATIVE_KERNELS"


def gdn_native_kernels_enabled() -> bool:
    """True iff the GDN core runs vLLM's native fused kernels on BOTH the trainer and the engine.

    The native composition is ``causal_conv1d_fn``/``causal_conv1d_update`` for the conv and
    ``fused_sigmoid_gating_delta_rule_update`` for the core, with l2norm, GQA mapping and fp32
    ``exp(A_log)`` sigmoid gating all in-kernel. It is not bitwise-equal to the eager isoexec
    composition (in-kernel rsqrt-multiply l2norm vs ``l2norm_fwd``, bf16- vs fp32-exp gating,
    bias-in-accumulator conv), so the flag must flip both runtimes in the same run; each reads it at
    call time. Requires recurrent mode and native state.
    """
    return os.environ.get(_NATIVE_KERNELS_ENV, "0").lower() not in ("", "0", "false", "no")


def gdn_native_core_kernel(q, k, v, a, b, A_log, dt_bias, *, ssm_state, state_indices, cu_seqlens):
    """The raw native GDN core: vLLM's ``fused_sigmoid_gating_delta_rule_update``. Returns ``o``.

    ``q, k``: raw (un-normalised) ``[B, T, H, K]``. ``H`` may be the GQA-compressed head count (the
    kernel maps ``i_h = i_hv // (HV // H)``) or already expanded to ``HV``; the mapped values are the
    same bits either way, so the engine's compressed heads and Megatron's expanded heads agree.
    ``a``/``b`` are the raw gating inputs ``[T, HV]``, from which the kernel computes
    ``g = -exp(fp32(A_log)) * softplus(a + dt_bias)`` and ``beta = sigmoid(b)`` in fp32 in-kernel.
    ``state_indices``: ``[N]`` for one-token decode, else ``[N, Tmax]`` with column 0 the
    initial-state row and the last real token's column the final-state row (0 skips). Advances
    ``ssm_state`` in place.

    One varlen call equals T sequential decode calls bitwise, including the fp32 state round-trip, and
    a chunk split at any token boundary is exact -- which is what makes chunked prefill and align-mode
    prefix caching exact under this kernel.
    """
    from .gdn_native_core_bv64 import maybe_native_core_bv64

    tiled = maybe_native_core_bv64(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        ssm_state=ssm_state,
        state_indices=state_indices,
        cu_seqlens=cu_seqlens,
    )
    if tiled is not None:
        return tiled

    from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update,
    )

    o, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        initial_state=ssm_state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        use_qk_l2norm_in_kernel=True,
    )
    return o


def _eager_native_composition(q, k, v, a, b, A_log, dt_bias, cu_seqlens, chunk_size):
    """Differentiable fp32 eager equivalent of the native composition, for the VJP only.

    Mirrors what the fused kernel computes -- rsqrt-multiply l2norm, fp32 ``exp(A_log)`` gating --
    then runs the fp32 chunked reference for the scan. It never runs in the forward, so it does not
    have to be bitwise, only a faithful function whose gradient is the gradient of the forward.
    """
    qn = q.float() * torch.rsqrt((q.float() ** 2).sum(-1, keepdim=True) + 1e-6)
    kn = k.float() * torch.rsqrt((k.float() ** 2).sum(-1, keepdim=True) + 1e-6)
    HV = v.shape[-2]
    H = q.shape[-2]
    if H != HV:  # GQA: expand compressed heads the way the kernel maps them
        rep = HV // H
        qn = qn.repeat_interleave(rep, dim=-2)
        kn = kn.repeat_interleave(rep, dim=-2)
    g = -torch.exp(A_log.float()) * torch.nn.functional.softplus(a.float() + dt_bias.float())
    beta = b.float().sigmoid()
    # a/b may arrive [T, HV] (engine convention) or [1, T, HV] (Megatron packed); the reference
    # wants them shaped like v's leading dims.
    g = g.reshape(v.shape[0], v.shape[1], HV)
    beta = beta.reshape(v.shape[0], v.shape[1], HV)
    return _torch_chunk_gdr(qn, kn, v, g, beta, None, cu_seqlens, chunk_size)


class _GdnNativeCoreAutograd(torch.autograd.Function):
    """Native fused-kernel forward + eager fp32 reference VJP.

    Same trade as :class:`_GdnChunkAutograd` / :class:`_GdnRecurrentAutograd`: the forward is the exact
    engine kernel, the backward differentiates a faithful eager fp32 equivalent. Because gating and
    l2norm are in-kernel here, ``a``/``b``/``A_log``/``dt_bias`` are forward inputs and get their grads
    from the same eager recompute.
    """

    @staticmethod
    def forward(ctx, q, k, v, a, b, A_log, dt_bias, cu_seqlens, chunk_size):
        with torch.no_grad():
            ssm_state, idx = _recurrent_scratch_state(q, v, cu_seqlens)
            o = gdn_native_core_kernel(
                q, k, v, a, b, A_log, dt_bias, ssm_state=ssm_state, state_indices=idx, cu_seqlens=cu_seqlens
            )
        ctx.save_for_backward(q, k, v, a, b, A_log, dt_bias)
        ctx.cu_seqlens = cu_seqlens
        ctx.chunk_size = chunk_size
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, a, b, A_log, dt_bias = ctx.saved_tensors
        from .gdn_fla_backward import fla_backward_enabled, fla_chunk_vjp

        if fla_backward_enabled():
            # FLA's fused Triton backward for the scan, chained through the elementwise prep grads:
            # rebuild the kernel's in-kernel prep (l2norm, GQA expand, fp32 gating) as a differentiable
            # eager graph, take the scan's grads at the detached intermediates via fla_chunk_vjp, then
            # backprop the prep graph to reach the raw leaves -- including A_log/dt_bias, whose grads
            # flow only through here now that gating is in-kernel.
            HV = v.shape[-2]
            with torch.enable_grad():
                leaves = [t.detach().requires_grad_(True) for t in (q, k, a, b, A_log, dt_bias)]
                ql, kl, al, bl, Al, dl = leaves
                qn = (ql.float() * torch.rsqrt((ql.float() ** 2).sum(-1, keepdim=True) + 1e-6)).to(q.dtype)
                kn = (kl.float() * torch.rsqrt((kl.float() ** 2).sum(-1, keepdim=True) + 1e-6)).to(k.dtype)
                if q.shape[-2] != HV:
                    rep = HV // q.shape[-2]
                    qn = qn.repeat_interleave(rep, dim=-2)
                    kn = kn.repeat_interleave(rep, dim=-2)
                g = -torch.exp(Al.float()) * torch.nn.functional.softplus(al.float() + dl.float())
                beta = bl.sigmoid()
                g = g.reshape(v.shape[0], v.shape[1], HV)
                beta = beta.reshape(v.shape[0], v.shape[1], HV)
                dqn, dkn, dv, dg, dbeta = fla_chunk_vjp(
                    qn.detach(), kn.detach(), v, g.detach(), beta.detach(), do, None, ctx.cu_seqlens, ctx.chunk_size
                )
                prep_grads = torch.autograd.grad(
                    [qn, kn, g, beta],
                    leaves,
                    [dqn.to(qn.dtype), dkn.to(kn.dtype), dg.to(g.dtype), dbeta.to(beta.dtype)],
                )
            out = [
                prep_grads[0].to(q.dtype),
                prep_grads[1].to(k.dtype),
                dv.to(v.dtype),
                prep_grads[2].to(a.dtype),
                prep_grads[3].to(b.dtype),
                prep_grads[4].to(A_log.dtype),
                prep_grads[5].to(dt_bias.dtype),
            ]
            return (*out, None, None)

        with torch.enable_grad():
            leaves = [t.detach().requires_grad_(True) for t in (q, k, v, a, b, A_log, dt_bias)]
            o = _eager_native_composition(*leaves, ctx.cu_seqlens, ctx.chunk_size)
            grads = torch.autograd.grad(o, leaves, do.float())
        grads = [gr.to(t.dtype) for gr, t in zip(grads, (q, k, v, a, b, A_log, dt_bias))]
        return (*grads, None, None)


class _GdnNativeCprAutograd(_GdnNativeCoreAutograd):
    """Trainer door: native CPR forward with the same native-composition VJP.

    The backward is inherited verbatim from :class:`_GdnNativeCoreAutograd`, because native
    CPR is another evaluation strategy of the same delta-rule function (fused_sigmoid within
    chunks, chunk-pass boundary states on the eager matched prep).
    """

    @staticmethod
    def forward(ctx, q, k, v, a, b, A_log, dt_bias, cu_seqlens, chunk_size):
        from .gdn_cpr import gdn_native_cpr_fwd

        with torch.no_grad():
            o = gdn_native_cpr_fwd(q, k, v, a, b, A_log, dt_bias, cu_seqlens=cu_seqlens, chunk_size=chunk_size)
        ctx.save_for_backward(q, k, v, a, b, A_log, dt_bias)
        ctx.cu_seqlens = cu_seqlens
        ctx.chunk_size = chunk_size
        return o


def gdn_native_cpr(q, k, v, a, b, A_log, dt_bias, *, cu_seqlens=None):
    """Training/scoring entry for the NATIVE CPR composition (raw q/k + raw a/b).

    Accepts both the packed layout (B==1 plus cu_seqlens) and the padded ``[B, T, ...]`` layout the
    scoring path sends (no cu_seqlens); a padded batch is flattened to B independent packed sequences of
    length T, which is what the recurrent/native [B, T] kernels do to each batch row. The reshapes are
    graph ops, so grads flow through them.
    """
    unbatch = None
    if cu_seqlens is None:
        B, T = q.shape[0], q.shape[1]
        if B != 1:
            q = q.reshape(1, B * T, *q.shape[2:])
            k = k.reshape(1, B * T, *k.shape[2:])
            v = v.reshape(1, B * T, *v.shape[2:])
            a = a.reshape(B * T, a.shape[-1])
            b = b.reshape(B * T, b.shape[-1])
            unbatch = (B, T)
        cu_seqlens = torch.arange(0, (max(B, 1) + 1) * T, T, dtype=torch.int32, device=q.device)
    needs_grad = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in (q, k, v, a, b, A_log, dt_bias)
    )
    if not needs_grad:
        from .gdn_cpr import gdn_native_cpr_fwd

        with torch.no_grad():
            o = gdn_native_cpr_fwd(q, k, v, a, b, A_log, dt_bias, cu_seqlens=cu_seqlens, chunk_size=fla_chunk_size())
    else:
        o = _GdnNativeCprAutograd.apply(q, k, v, a, b, A_log, dt_bias, cu_seqlens, fla_chunk_size())
    if unbatch is not None:
        B, T = unbatch
        o = o.reshape(B, T, *o.shape[2:])
    return o


def gdn_native_core(q, k, v, a, b, A_log, dt_bias, *, cu_seqlens=None):
    """Training/scoring entry for the native composition: zero initial state, no state out.

    The trainer-facing twin of :func:`gdn_recurrent` for ``SKYRL_ISOEXEC_GDN_NATIVE_KERNELS=1``.
    Under ``no_grad`` (scoring, rollout probes) this is exactly the engine kernel; with a gradient
    it routes through :class:`_GdnNativeCoreAutograd`, whose forward is that same kernel.
    """
    needs_grad = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in (q, k, v, a, b, A_log, dt_bias)
    )
    if not needs_grad:
        with torch.no_grad():
            ssm_state, idx = _recurrent_scratch_state(q, v, cu_seqlens)
            return gdn_native_core_kernel(
                q, k, v, a, b, A_log, dt_bias, ssm_state=ssm_state, state_indices=idx, cu_seqlens=cu_seqlens
            )
    return _GdnNativeCoreAutograd.apply(q, k, v, a, b, A_log, dt_bias, cu_seqlens, fla_chunk_size())


_NATIVE_CONV_BWD_COUNTS = {
    "served": 0,
    "tokens": 0,
    "taps": 0,
    "row_chunks": 0,
    "reported": 0,
}
_NATIVE_CONV_ANALYTIC_BWD_ENABLED = os.environ.get("SKYRL_ISOEXEC_GDN_ANALYTIC_CONV_BWD", "0") == "1"


def _report_native_conv_backward() -> None:
    served = _NATIVE_CONV_BWD_COUNTS["served"]
    if served < 1 or (served & (served - 1)) != 0 or served == _NATIVE_CONV_BWD_COUNTS["reported"]:
        return
    _NATIVE_CONV_BWD_COUNTS["reported"] = served
    print(
        "[ISOEXEC-GDN-CONV-BWD] "
        f"pid={os.getpid()} served={served} tokens={_NATIVE_CONV_BWD_COUNTS['tokens']} "
        f"taps={_NATIVE_CONV_BWD_COUNTS['taps']} row_chunks={_NATIVE_CONV_BWD_COUNTS['row_chunks']}",
        flush=True,
    )


def _native_conv_backward_rows() -> int:
    rows = int(os.environ.get("SKYRL_ISOEXEC_GDN_CONV_BWD_ROWS", "8192"))
    if rows <= 0:
        raise ValueError("SKYRL_ISOEXEC_GDN_CONV_BWD_ROWS must be positive")
    return rows


def _conv_vjp_dz_chunk(xw, wf, pos, seq_starts, taps):
    """One row-chunk of the conv preactivation, in explicit ascending tap order.

    Backward-only: reachable only from :func:`_native_conv_vjp`. ``xw`` is the window ``x[lo:stop]``
    with ``lo`` the earliest row any tap of this chunk can read, and ``pos``/``seq_starts`` are already
    rebased by ``lo``. Windowing keeps the compiled artifact's shape key stable across bins (it depends
    on ``chunk_rows`` and ``taps``, not the bin's token count), avoiding a per-bin recompile. The
    rebase preserves values: ``source_local < 0`` can only occur when ``lo == 0``, where local and
    global indices coincide.
    """
    acc = torch.zeros((pos.shape[0], xw.shape[1]), dtype=torch.float32, device=xw.device)
    for tap in range(taps):
        lag = taps - 1 - tap
        source = pos - lag
        valid = source >= seq_starts
        source = source.clamp(min=0)
        shifted = xw.index_select(0, source)
        acc = acc + shifted.float() * valid.unsqueeze(1) * wf[:, tap]
    return acc


def _conv_vjp_act_chunk(z, dyc, silu):
    """One row-chunk of ``dL/d(preactivation)``. BACKWARD-ONLY (see :func:`_conv_vjp_dz_chunk`)."""
    if silu:
        sigmoid = torch.sigmoid(z)
        derivative = sigmoid * (1.0 + z * (1.0 - sigmoid))
        return dyc.float() * derivative
    return dyc.float()


def _conv_vjp_dxdw_chunk(xw, dzw, wf, pos_x, seq_starts_x, pos_z, seq_ends_z, taps):
    """One row-chunk of ``dx`` plus this chunk's ``dw`` partial. BACKWARD-ONLY.

    Two windows, both rebased so the shape key does not carry the bin's token count: ``xw`` is
    ``x[lo:stop]`` (taps reach backwards) and ``dzw`` is ``dz[start:hi]`` (they reach forwards). The
    forward clamp to ``dzw.shape[0] - 1`` is the local spelling of the global ``clamp(max=total-1)``
    and can only fire on the final chunk.

    Returns ``(dxf, dw_partial)`` shaped like ``wf``; the caller folds it with one ``dwf.add_`` per
    chunk, so every ``dw[:, tap]`` accumulates its chunks in ascending row order.
    """
    rows, channels = pos_x.shape[0], xw.shape[1]
    dxf = torch.zeros((rows, channels), dtype=torch.float32, device=xw.device)
    dw_cols = []
    last = dzw.shape[0] - 1
    for tap in range(taps):
        lag = taps - 1 - tap
        source = pos_x - lag
        source_valid = source >= seq_starts_x
        source = source.clamp(min=0)
        shifted_x = xw.index_select(0, source).float() * source_valid.unsqueeze(1)
        dw_cols.append((dzw[:rows] * shifted_x).sum(dim=0))

        output = pos_z + lag
        output_valid = output < seq_ends_z
        output = output.clamp(max=max(0, last))
        shifted_dz = dzw.index_select(0, output) * output_valid.unsqueeze(1)
        dxf = dxf + shifted_dz * wf[:, tap]
    return dxf, torch.stack(dw_cols, dim=1)


def _native_conv_vjp(dy, x, weight, bias, cu_seqlens, activation):
    """Atomics-free packed depthwise-conv VJP, generic in tap count and sequence lengths.

    One output element has one owner. ``dx[token, channel]`` gathers its at-most-W future
    cotangents in a fixed tap loop; ``dw[channel, tap]`` uses deterministic chunk reductions.
    Sequence boundaries come from device-side integer comparisons, so there is no D2H read or
    Python loop over packed sequences. The recomputed preactivation/dz is the only additional
    full-size fp32 buffer; dx accumulation, shifted gathers, and activation temporaries are
    row-chunked before writing the required return dtype.
    """
    if activation not in (None, "silu", "swish"):
        raise ValueError(f"unsupported activation {activation!r}")
    if x.ndim != 2 or weight.ndim != 2 or weight.shape[0] != x.shape[1]:
        raise ValueError(f"native conv VJP expects x=[T,D], weight=[D,W], got {tuple(x.shape)}, {tuple(weight.shape)}")

    total, channels = x.shape
    taps = weight.shape[1]
    chunk_rows = _native_conv_backward_rows()
    wf = weight.float()
    positions = torch.arange(total, dtype=torch.long, device=x.device)
    bounds = cu_seqlens.to(device=x.device, dtype=torch.long)
    sequence = torch.searchsorted(bounds[1:], positions, right=True)
    starts = bounds.index_select(0, sequence)
    ends = bounds.index_select(0, sequence + 1)

    # Recompute the preactivation in explicit tap order. Backward-only, so it need not reproduce the
    # fused forward's rounding, but it is the same fp32 expression.
    #
    # The three per-chunk bodies are module-level functions so they can be routed through
    # ``call_region`` (autofuse/bwd_compile). Each `dz` element still accumulates its taps in ascending
    # tap order onto a zero start, and each `dw[:, tap]` still accumulates its chunks in ascending row
    # order; ``call_region`` runs the eager body verbatim when backward compilation is off.
    dz = torch.empty((total, channels), dtype=torch.float32, device=x.device)
    for start in range(0, total, chunk_rows):
        stop = min(start + chunk_rows, total)
        lo = max(0, start - (taps - 1))  # the earliest row any tap of this chunk can read
        dz[start:stop] = call_region(
            "gdn.conv_vjp.dz_chunk",
            _conv_vjp_dz_chunk,
            x[lo:stop],
            wf,
            positions[start:stop] - lo,
            starts[start:stop] - lo,
            taps,
        )
    if bias is not None:
        dz.add_(bias.float())

    # Recycle preactivation storage as dz, bounding sigmoid/derivative temporaries by one row chunk.
    dbf = torch.zeros(channels, dtype=torch.float32, device=x.device) if bias is not None else None
    silu = activation in ("silu", "swish")
    for start in range(0, total, chunk_rows):
        stop = min(start + chunk_rows, total)
        adjusted = call_region("gdn.conv_vjp.act_chunk", _conv_vjp_act_chunk, dz[start:stop], dy[start:stop], silu)
        dz[start:stop].copy_(adjusted)
        if dbf is not None:
            dbf.add_(adjusted.sum(dim=0))

    # Gather, never scatter: each dx element has exactly one owner. Each row chunk finishes in fp32 and
    # is written out in the return dtype, avoiding a second full-size fp32 tensor. Tap order per dx
    # element is unchanged, and dw[:, tap] chunks are still added in ascending row order.
    dx = torch.empty_like(x)
    dwf = torch.zeros_like(wf)
    for start in range(0, total, chunk_rows):
        stop = min(start + chunk_rows, total)
        lo = max(0, start - (taps - 1))  # earliest x row read by any tap of this chunk
        hi = min(total, stop + taps - 1)  # latest dz row read by any tap of this chunk
        dxf, dw_partial = call_region(
            "gdn.conv_vjp.dxdw_chunk",
            _conv_vjp_dxdw_chunk,
            x[lo:stop],
            dz[start:hi],
            wf,
            positions[start:stop] - lo,
            starts[start:stop] - lo,
            positions[start:stop] - start,
            ends[start:stop] - start,
            taps,
        )
        dwf.add_(dw_partial)
        dx[start:stop].copy_(dxf.to(x.dtype))

    dw = dwf.to(weight.dtype)
    db = dbf.to(bias.dtype) if bias is not None else None
    return dx, dw, db


def _native_conv_channel_last_input(x: torch.Tensor) -> torch.Tensor:
    """Return vLLM's ``[D, T]`` channel-last causal-conv view of ``x [T, D]``.

    ``causal_conv1d_fn`` selects its fast prefill kernel only when the channel dimension has stride one.
    A row-major ``[T, D]`` tensor already has that storage, so the transpose is a zero-copy ``[D, T]``
    view with strides ``(1, D)``; compacting after the transpose gives strides ``(T, 1)`` and selects
    the much slower channel-first kernel. A non-row-major input is normalized before the transpose.
    """
    if x.ndim != 2:
        raise ValueError(f"native GDN conv expects x=[T, D], got {tuple(x.shape)}")
    if x.shape[1] <= 1:
        raise ValueError(f"native GDN conv requires D > 1, got x shape {tuple(x.shape)}")
    if x.stride(1) != 1:
        x = x.contiguous()
    x_dt = x.transpose(0, 1)
    if x_dt.stride(0) != 1 or x_dt.stride(1) <= 1:
        raise RuntimeError(
            "native GDN conv input must be channel-last [D, T] with strides (1, >1), "
            f"got shape={tuple(x_dt.shape)} strides={tuple(x_dt.stride())}"
        )
    return x_dt


class _GdnNativeConvAutograd(torch.autograd.Function):
    """vLLM ``causal_conv1d_fn`` forward + default-OFF atomics-free analytic conv VJP.

    The kernel and the eager conv compute the same mathematical function, differing only in rounding
    (fp32 accumulator preloaded with bias vs bias added after the taps), so the analytic gradient is the
    gradient of the kernel forward to floating-point accuracy.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, cu_seqlens, activation):
        # x [T, D] packed; one varlen kernel launch for the whole batch, where the eager path loops
        # sequences in python.
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn

        from .packed_meta_cache import causal_conv1d_metadata

        with torch.no_grad():
            N = cu_seqlens.numel() - 1
            W = weight.shape[-1]
            metadata = causal_conv1d_metadata(cu_seqlens)
            if metadata is None:
                # Exact cache-off fallback: retain vLLM's original per-call construction so the
                # packed-meta flag remains a clean causal ablation.
                scratch = x.new_zeros(N + 1, x.shape[-1], W - 1)
                idx = torch.arange(1, N + 1, dtype=torch.int32, device=x.device)
                has0 = torch.zeros(N, dtype=torch.bool, device=x.device)
            else:
                # Every has-initial-state bit is false, so the kernel never reads conv_states for
                # the output: it synthesizes the width-1 zero prefix in registers and only WRITES
                # final states that this stateless trainer call discards. Empty storage therefore
                # removes an unobservable memset without changing output or persistent state.
                scratch = x.new_empty(N + 1, x.shape[-1], W - 1)
                idx = metadata.cache_indices
                has0 = metadata.has_initial_state
            y = causal_conv1d_fn(
                _native_conv_channel_last_input(x),
                weight,
                bias,
                conv_states=scratch,
                query_start_loc=cu_seqlens,
                cache_indices=idx,
                has_initial_state=has0,
                activation=activation,
                metadata=metadata,
            ).transpose(0, 1)
        ctx.save_for_backward(x, weight, bias, cu_seqlens)
        ctx.activation = activation
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, bias, cu_seqlens = ctx.saved_tensors
        if not _NATIVE_CONV_ANALYTIC_BWD_ENABLED:
            # Default VJP, kept inline so the default-off path adds no helper-call overhead.
            bounds = cu_seqlens.tolist()
            with torch.enable_grad():
                xl = x.detach().requires_grad_(True)
                wl = weight.detach().requires_grad_(True)
                bl = bias.detach().requires_grad_(True) if bias is not None else None
                ys = []
                for start, stop in zip(bounds[:-1], bounds[1:]):
                    y, _ = gdn_causal_conv(
                        xl[start:stop],
                        wl,
                        bl,
                        initial_state=None,
                        activation=ctx.activation,
                        return_final_state=True,
                    )
                    ys.append(y)
                y = torch.cat(ys, dim=0)
                leaves = [xl, wl] + ([bl] if bl is not None else [])
                grads = torch.autograd.grad(y, leaves, dy)
            dx, dw = grads[0].to(x.dtype), grads[1].to(weight.dtype)
            db = grads[2].to(bias.dtype) if bias is not None else None
            return dx, dw, db, None, None

        dx, dw, db = _native_conv_vjp(dy, x, weight, bias, cu_seqlens, ctx.activation)
        total, taps = x.shape[0], weight.shape[1]
        chunks = -(-total // _native_conv_backward_rows())
        _NATIVE_CONV_BWD_COUNTS["served"] += 1
        _NATIVE_CONV_BWD_COUNTS["tokens"] += total
        _NATIVE_CONV_BWD_COUNTS["taps"] += taps
        _NATIVE_CONV_BWD_COUNTS["row_chunks"] += (2 * taps + 1) * chunks
        _report_native_conv_backward()
        return dx, dw, db, None, None


def gdn_native_conv(x, weight, bias, *, cu_seqlens, activation="silu"):
    """Packed varlen conv through vLLM's ``causal_conv1d_fn``: ``x [T, D] -> y [T, D]``."""
    needs_grad = torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (x, weight, bias))
    if not needs_grad:
        return _GdnNativeConvAutograd.forward(_NoCtx(), x, weight, bias, cu_seqlens, activation)
    return _GdnNativeConvAutograd.apply(x, weight, bias, cu_seqlens, activation)


class _NoCtx:
    """Context stand-in for calling an autograd.Function's forward outside autograd (no_grad path)."""

    def save_for_backward(self, *_):
        pass


def gdn_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
):
    """Training/prefill-shaped recurrent delta rule: sequences start at zero, no state comes back.

    The trainer-facing twin of :func:`gdn_chunk`: same signature and ``(o, final_state)`` return, so the
    ``fla`` shim can route Megatron at either one. Under ``no_grad`` this is exactly the vLLM kernel;
    with a gradient it routes through :class:`_GdnRecurrentAutograd`, whose forward is that same kernel.
    The engine does not come through here -- it owns a real state pool and calls
    :func:`gdn_recurrent_kernel` directly.
    """
    if initial_state is not None or output_final_state:
        raise NotImplementedError(
            "isoexec GDN: gdn_recurrent is the stateless (training/prefill-from-zero) entry point. "
            "The engine carries state through gdn_recurrent_kernel + a real state pool."
        )

    needs_grad = torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (q, k, v, g, beta))
    if not needs_grad:
        with torch.no_grad():
            ssm_state, idx = _recurrent_scratch_state(q, v, cu_seqlens)
            o = gdn_recurrent_kernel(q, k, v, g, beta, ssm_state=ssm_state, state_indices=idx, cu_seqlens=cu_seqlens)
        return o, None

    o = _GdnRecurrentAutograd.apply(q, k, v, g, beta, cu_seqlens, fla_chunk_size())
    return o, None


class _GdnCprAutograd(torch.autograd.Function):
    """Chunk-synced forward plus the same fp32 chunked reference VJP the other modes use.

    Chunk-synced is another evaluation strategy of the same delta-rule function (boundary states by the
    chunk state pass, within-chunk outputs by the recurrent scan), so a VJP of the function is a VJP of
    this evaluation and FLA's fused chunk backward applies as-is.
    """

    @staticmethod
    def forward(ctx, q, k, v, g, beta, cu_seqlens, chunk_size):
        from .gdn_cpr import gdn_cpr_fwd

        with torch.no_grad():
            o = gdn_cpr_fwd(q, k, v, g, beta, cu_seqlens=cu_seqlens, chunk_size=chunk_size)
        ctx.save_for_backward(q, k, v, g, beta)
        ctx.cu_seqlens = cu_seqlens
        ctx.chunk_size = chunk_size
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, g, beta = ctx.saved_tensors
        from .gdn_fla_backward import fla_backward_enabled, fla_chunk_vjp

        if fla_backward_enabled():
            grads = fla_chunk_vjp(q, k, v, g, beta, do, None, ctx.cu_seqlens, ctx.chunk_size)
            return (*grads, None, None)
        with torch.enable_grad():
            leaves = [t.detach().requires_grad_(True) for t in (q, k, v, g, beta)]
            o = _torch_chunk_gdr(*leaves, None, ctx.cu_seqlens, ctx.chunk_size)
            grads = torch.autograd.grad(o, leaves, do.float())
        grads = [gr.to(t.dtype) for gr, t in zip(grads, (q, k, v, g, beta))]
        return (*grads, None, None)


def gdn_cpr(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
):
    """Training/scoring-shaped CPR forward: same ``(o, final_state)`` contract as its twins.

    The engine does not come through here -- it owns a real state pool and runs
    ``CprGDN.prefill``/``.decode``, which evaluate the same canonical function incrementally.
    """
    if initial_state is not None or output_final_state:
        raise NotImplementedError(
            "isoexec GDN: gdn_cpr is the stateless (training/prefill-from-zero) entry "
            "point. The engine carries state through CprGDN."
        )
    if cu_seqlens is None:
        if q.shape[0] != 1:
            # Padded [B, T, ...] layout (the scoring path): flatten to B independent packed sequences,
            # the same treatment the sibling [B, T] kernels give batch rows. Reshapes are graph ops, so
            # grads flow.
            B, T = q.shape[0], q.shape[1]
            q = q.reshape(1, B * T, *q.shape[2:])
            k = k.reshape(1, B * T, *k.shape[2:])
            v = v.reshape(1, B * T, *v.shape[2:])
            g = g.reshape(1, B * T, *g.shape[2:])
            beta = beta.reshape(1, B * T, *beta.shape[2:])
            cu_seqlens = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=q.device)

            needs_grad = torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (q, k, v, g, beta))
            if not needs_grad:
                from .gdn_cpr import gdn_cpr_fwd

                with torch.no_grad():
                    o = gdn_cpr_fwd(q, k, v, g, beta, cu_seqlens=cu_seqlens, chunk_size=fla_chunk_size())
            else:
                o = _GdnCprAutograd.apply(q, k, v, g, beta, cu_seqlens, fla_chunk_size())
            return o.reshape(B, T, *o.shape[2:]), None
        cu_seqlens = torch.tensor([0, q.shape[1]], dtype=torch.int32, device=q.device)

    needs_grad = torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (q, k, v, g, beta))
    if not needs_grad:
        from .gdn_cpr import gdn_cpr_fwd

        with torch.no_grad():
            o = gdn_cpr_fwd(q, k, v, g, beta, cu_seqlens=cu_seqlens, chunk_size=fla_chunk_size())
        return o, None

    o = _GdnCprAutograd.apply(q, k, v, g, beta, cu_seqlens, fla_chunk_size())
    return o, None


def gdn_core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
):
    """:func:`gdn_chunk` or :func:`gdn_recurrent`, per :func:`gdn_kernel_mode`. The trainer's door.

    ``SKYRL_ISOEXEC_GDN_TRAINER_KERNEL`` (unset by default) overrides the kernel for this door only; the
    engine's decode/prefill reach RecurrentGDN / CprGDN directly and keep
    ``SKYRL_ISOEXEC_GDN_KERNEL``. Setting it to ``chunk`` while the engine runs ``recurrent`` deliberately
    relaxes the GDN core so the rollout-vs-train gap measures the chunk-vs-recurrent mismatch. That is
    not a IsoExec configuration."""
    if any(t.device.type == "meta" for t in (q, k, v, g, beta)):
        if not all(t.device.type == "meta" for t in (q, k, v, g, beta)):
            raise ValueError("gdn_core abstract contract requires all operands on meta")
        if q.shape[:-1] != k.shape[:-1] or q.shape[:-1] != v.shape[:-1]:
            raise ValueError("gdn_core q/k/v leading dimensions differ")
        if g.shape != q.shape[:-1] or beta.shape != q.shape[:-1]:
            raise ValueError("gdn_core g/beta dimensions differ from q/k/v")
        output = torch.empty_strided(tuple(v.shape), tuple(v.stride()), dtype=v.dtype, device="meta")
        if not output_final_state:
            return output, None
        if initial_state is not None:
            final_state = torch.empty_strided(
                tuple(initial_state.shape),
                tuple(initial_state.stride()),
                dtype=initial_state.dtype,
                device="meta",
            )
        else:
            # FLA's recurrent state contract is [batch, heads, key_dim, value_dim].
            final_state = torch.empty(
                (q.shape[0], q.shape[2], q.shape[3], v.shape[3]),
                dtype=torch.float32,
                device="meta",
            )
        return output, final_state

    from ...core.gdn_kernel_env import gdn_trainer_kernel_override

    override = gdn_trainer_kernel_override()
    if override == "cpr" or (not override and cpr_mode()):
        return gdn_cpr(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
    if override == "recurrent" or (not override and recurrent_mode()):
        return gdn_recurrent(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
    return gdn_chunk(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )


def gdn_gate_and_beta(
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """g = -exp(A_log) * softplus(a + dt_bias) in fp32; beta = sigmoid(b). Elementwise -> invariant.

    Mirrors megatron ``GatedDeltaNet._compute_g_and_beta`` exactly (fp32 g, beta in the input dtype).
    ``A_log.exp()`` is taken in the parameter's own dtype rather than upcast first, because megatron
    stores A_log in ``params_dtype`` (bf16) and exponentiates before the fp32 multiply, and
    ``exp(bf16(x)).float()`` is not ``exp(float(x))``.
    """
    g = -A_log.exp() * torch.nn.functional.softplus(a.float() + dt_bias.float())
    beta = b.sigmoid()
    return g, beta
