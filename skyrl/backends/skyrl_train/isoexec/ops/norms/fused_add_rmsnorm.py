"""Fold the residual add into the adjacent zero-centred RMSNorm.

At every Megatron TransformerLayer boundary a bf16 residual add materialises a full ``[T, H]``
tensor that the following RMSNorm reads straight back. This kernel does both in one pass: it writes
the added residual stream (the next layer needs it) and, reading the add result from registers,
writes the norm output -- saving one full read pass per site.

BIT-EQUAL BY COMPOSITION. The add is reproduced exactly as ATen's ``CUDAFunctor_add`` (promote both
bf16 operands to fp32, add, round once to bf16), and the norm then re-promotes that bf16 value to
fp32, which is lossless, and runs the same reduction tree as the standalone fused RMSNorm. No
intermediate round is skipped: the add's bf16 round happens in-kernel exactly where eager performs
it, so reading the result from registers rather than DRAM cannot move a bit.

THE REDUCTION TILE IS THE BITWISE CONTRACT, inherited from ``fused_outnorm``. ``F.rms_norm`` reaches
torch's vendored quack kernel whose ``threads_per_row`` is chosen from N alone, fixing the fp32
reduction tree; ``_tile_for`` gives each row exactly that many threads, and depends on width only,
never on row count. ``enable_fp_fusion=False`` and the ``tl.rsqrt`` rstd come from the same contract.

Installed on both sides via an autograd.Function: fused Triton forward, reference-composition
backward (the eager add+norm recomputed under ``enable_grad``), so trainer grads match eager by
construction. Configurations this kernel does not own -- fp32 residual connections, nonzero hidden
dropout, a bias, or N above the quack ladder's maximum -- fall back to eager at the caller.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# quack's ladder -- the fp32 reduction order, hence the bitwise contract. Duplicated from
# fused_outnorm.py so this file stands alone.
_THREADS_PER_ROW_LADDER = ((64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128))
_MAX_N = 16384


def _threads_per_row(N: int) -> int:
    for limit, t in _THREADS_PER_ROW_LADDER:
        if N <= limit:
            return t
    return 256


def _tile_for(N: int) -> tuple[int, int]:
    """``(rows_per_program, num_warps)`` giving each row exactly quack's ``threads_per_row``."""
    tpr = _threads_per_row(N)
    if tpr <= 32:
        return 32 // tpr, 1
    return 1, tpr // 32


# =================================================================================================
# Fused kernel: added = bf16(residual + x); normed = rms(added) rounded bf16, then * (1 + w) bf16.
# Two stores (residual stream + norm output); ONE read of x + ONE read of residual.
# =================================================================================================
@triton.jit
def _gamma(w_ptr, c, N: tl.constexpr):
    """``1.0 + weight`` in registers from the LIVE parameter (K2a). Rounds to bf16 -- eager forms
    ``1.0 + self.weight`` as a bf16 tensor (fp32 opmath, one round) before multiplying."""
    w = tl.load(w_ptr + c, mask=c < N, other=0.0).to(tl.float32)
    return (1.0 + w).to(w_ptr.dtype.element_ty).to(tl.float32)


@triton.jit
def _add_rms_gamma_kernel(
    x_ptr,  # attn_out / mlp_out  (bda's x_with_bias[0])
    res_ptr,  # residual
    gam_ptr,  # the norm's raw zero-centred weight
    added_ptr,  # OUT: residual stream (= bf16(residual + x)) -- next layer needs it
    o_ptr,  # OUT: normed = rms(added) * (1 + w)
    M,
    eps,
    N: tl.constexpr,
    BN: tl.constexpr,
    MB: tl.constexpr,
):
    # int64 row index -- `rows * N` byte-offset multiply wraps at 2^31 in int32 and reads the wrong
    # rows without faulting (isoexec-bmm-int32-overflow). Free insurance at our shapes.
    rows = tl.program_id(0).to(tl.int64) * MB + tl.arange(0, MB)[:, None]
    c = tl.arange(0, BN)[None, :]
    rmask = rows < M
    mask = rmask & (c < N)

    x = tl.load(x_ptr + rows * N + c, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(res_ptr + rows * N + c, mask=mask, other=0.0).to(tl.float32)
    # Down-cast 1 -- the residual add, exactly ATen's CUDAFunctor_add: fp32 opmath, one round.
    added = (x + r).to(added_ptr.dtype.element_ty)
    tl.store(added_ptr + rows * N + c, added, mask=mask)

    # Re-promote the stored bf16 to fp32 (exact) so the reduction sees the identical bytes the norm
    # would have read back from DRAM.
    h = added.to(tl.float32)
    s = tl.sum(h * h, 1)[:, None]
    # tl.rsqrt is rsqrt.approx.f32, matching quack fastmath. Its argument is at least eps, far above
    # the subnormal boundary.
    rstd = tl.rsqrt(s / N + eps)
    nb = (h * rstd).to(o_ptr.dtype.element_ty)  # down-cast 2: F.rms_norm output is bf16 pre-gamma
    gam = _gamma(gam_ptr, c, N)
    # Down-cast 3: bf16 elementwise multiply, fp32 opmath, one round at the store.
    tl.store(o_ptr + rows * N + c, (nb.to(tl.float32) * gam).to(o_ptr.dtype.element_ty), mask=mask)


def fused_add_rms_norm_gamma(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bit-equal to ``h = residual + x; (h, F.rms_norm(h,(N,),None,eps) * (1.0 + weight))``.

    ``weight`` is the raw zero-centred parameter; the ``1.0 +`` happens in registers, so there is no
    cached gamma to go stale across a weight sync.
    """
    assert residual.dtype == torch.bfloat16, "specialised to a bf16 residual stream"
    assert x.dtype == torch.bfloat16, "shipped bda path is all-bf16 (bias=None, dropout=0)"
    N = weight.shape[-1]
    assert N == x.shape[-1] == residual.shape[-1] and N <= _MAX_N
    xc = x.contiguous()
    rc = residual.contiguous()
    wc = weight.contiguous()
    x2 = xc.reshape(-1, N)
    r2 = rc.reshape(-1, N)
    M = x2.shape[0]
    added = torch.empty_like(x2)
    out = torch.empty_like(x2)
    mb, nw = _tile_for(N)
    _add_rms_gamma_kernel[(triton.cdiv(M, mb),)](
        x2,
        r2,
        wc,
        added,
        out,
        M,
        eps,
        N=N,
        BN=triton.next_power_of_2(N),
        MB=mb,
        num_warps=nw,
        enable_fp_fusion=False,  # FFMA contraction gives addcmul, not eager
    )
    return added.view_as(x), out.view_as(x)


# Fused Triton forward, reference-composition backward: the backward carries no bitwise constraint,
# so it recomputes the eager add+norm under enable_grad and differentiates that.
class _FusedAddRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, weight, eps):
        added, normed = fused_add_rms_norm_gamma(x, residual, weight, eps)
        ctx.save_for_backward(x, residual, weight)
        ctx.eps = eps
        return added, normed

    @staticmethod
    def backward(ctx, grad_added, grad_normed):
        x, residual, weight = ctx.saved_tensors
        N = weight.shape[-1]
        with torch.enable_grad():
            xd = x.detach().requires_grad_(x.requires_grad)
            rd = residual.detach().requires_grad_(residual.requires_grad)
            wd = weight.detach().requires_grad_(weight.requires_grad)
            added = rd + xd  # matches the training-mode out-of-place bias-dropout-add
            normed = torch.nn.functional.rms_norm(added, (N,), None, ctx.eps) * (1.0 + wd)
        inputs = [t for t in (xd, rd, wd)]
        needs = [xd.requires_grad, rd.requires_grad, wd.requires_grad]
        wanted = [t for t, n in zip(inputs, needs) if n]
        grads = (
            torch.autograd.grad(
                (added, normed),
                wanted,
                grad_outputs=(grad_added, grad_normed),
                retain_graph=False,
                allow_unused=True,
            )
            if wanted
            else ()
        )
        it = iter(grads)
        gx = next(it) if needs[0] else None
        gr = next(it) if needs[1] else None
        gw = next(it) if needs[2] else None
        return gx, gr, gw, None


def fused_add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Autograd-tracked drop-in. Returns ``(added_residual_stream, normed)``."""
    return _FusedAddRMSNorm.apply(x, residual, weight, eps)


# Widths where the norm half is proven bit-equal to eager. The kernel always matches the standalone
# fused RMSNorm, but that ladder mispredicts quack's config at a few widths (3072/5120/6144), where a
# handful of elements per ten million diverge from eager -- enough to move trainer bits. Any width
# not on this list falls back; extend it only with a per-width bit gate.
_BIT_EQUAL_PROVEN_WIDTHS = frozenset({2048})


def can_fuse(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    hidden_dropout: float,
    fp32_residual_connection: bool,
) -> bool:
    return (
        bias is None
        and hidden_dropout == 0.0
        and not fp32_residual_connection
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and residual.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and weight.shape[-1] == x.shape[-1] == residual.shape[-1]
        and weight.shape[-1] in _BIT_EQUAL_PROVEN_WIDTHS
    )


# Both-sides install, at the within-layer seam only: self_attn_bda -> pre_mlp_layernorm. The add
# result is consumed twice -- as the norm's input and as the next bda's residual -- and this returns
# both, so the seam must sit where one TransformerLayer.forward holds both. The self_attn_bda is
# deferred into a `_DeferredAdd` package that lives only between those two sub-calls and is
# materialised before forward returns, so pipeline p2p, activation checkpointing and graph capture
# (all of which move tensors BETWEEN layers) only ever see an ordinary tensor.
#
# A class-level rebind is correct here because the fusion goes through the autograd.Function, which
# preserves the trainer backward by construction; any config the kernel does not own falls back to
# the original sub-method.
class _DeferredAdd:
    """Carries an un-done residual add ``(x, residual)`` between _forward_attention and _forward_mlp.

    Transient: created by the deferred self_attn_bda, consumed by the patched pre-mlp fuse, and
    never stored or returned from forward()."""

    __slots__ = ("x", "residual")

    def __init__(self, x, residual):
        self.x = x
        self.residual = residual


_ZC_RMSNORM = None  # resolved lazily to avoid importing megatron at module import


def _is_zc_rmsnorm(m) -> bool:
    global _ZC_RMSNORM
    if _ZC_RMSNORM is None:
        try:
            from .zero_centered_norm import ZeroCenteredTorchRMSNorm

            _ZC_RMSNORM = ZeroCenteredTorchRMSNorm
        except Exception:
            return False
    return isinstance(m, _ZC_RMSNORM)


def _decoder_only_fusable(self) -> bool:
    """Guards for the decoder-only path this seam owns; anything else falls back to eager."""
    from megatron.core.transformer.identity_op import IdentityOp

    cfg = self.config
    return (
        not getattr(cfg, "fp32_residual_connection", False)
        and float(getattr(self, "hidden_dropout", 0.0)) == 0.0
        and not getattr(self, "recompute_input_layernorm", False)
        and not getattr(self, "recompute_pre_mlp_layernorm", False)
        and not getattr(self, "offload_attn_norm", False)
        and not getattr(self, "offload_mlp_norm", False)
        # Decoder-only: the cross-attn block between self_attn_bda and pre_mlp_layernorm is a pure
        # passthrough, so deferring the add past it is exact.
        and isinstance(self.pre_cross_attn_layernorm, IdentityOp)
        and isinstance(self.cross_attention, IdentityOp)
        and _is_zc_rmsnorm(self.pre_mlp_layernorm)
    )


_ORIG_FA = None
_ORIG_FM = None
_INSTALLED = False


def _ix_forward_attention(self, *args, **kwargs):
    # Only take the fast path for the decoder-only config this owns.
    if not _decoder_only_fusable(self):
        return _ORIG_FA(self, *args, **kwargs)
    # Bind the leading positional/keyword args the same way Megatron does.
    hidden_states = args[0] if args else kwargs["hidden_states"]
    attention_mask = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)

    input_layernorm_output = self.input_layernorm(hidden_states)
    residual = hidden_states
    attention_output_with_bias = self.self_attention(
        input_layernorm_output,
        attention_mask=attention_mask,
        inference_context=kwargs.get("inference_context", None),
        rotary_pos_emb=kwargs.get("rotary_pos_emb", None),
        rotary_pos_cos=kwargs.get("rotary_pos_cos", None),
        rotary_pos_sin=kwargs.get("rotary_pos_sin", None),
        rotary_pos_cos_sin=kwargs.get("rotary_pos_cos_sin", None),
        attention_bias=kwargs.get("attention_bias", None),
        packed_seq_params=kwargs.get("packed_seq_params", None),
        sequence_len_offset=kwargs.get("sequence_len_offset", None),
    )
    attn_out, attn_bias = attention_output_with_bias
    if attn_bias is not None:
        # A projection bias is present, which this seam does not own: fall back entirely.
        return _ORIG_FA(self, *args, **kwargs)
    # Defer self_attn_bda. The decoder-only cross-attn block is a passthrough, so skipping it is
    # exact, and the package is consumed by _ix_forward_mlp within this same forward().
    return _DeferredAdd(attn_out, residual), kwargs.get("context", None)


def _ix_forward_mlp(self, hidden_states, inference_context=None, padding_mask=None):
    if not isinstance(hidden_states, _DeferredAdd):
        return _ORIG_FM(self, hidden_states, inference_context, padding_mask)
    pkg = hidden_states
    norm = self.pre_mlp_layernorm
    ok = _is_zc_rmsnorm(norm) and can_fuse(
        pkg.x, pkg.residual, norm.weight, None, 0.0, getattr(self.config, "fp32_residual_connection", False)
    )
    if not ok:
        # Materialise the deferred add and run the original mlp path on a real tensor.
        return _ORIG_FM(self, pkg.residual + pkg.x, inference_context, padding_mask)
    added, normed = fused_add_rmsnorm(pkg.x, pkg.residual, norm.weight, norm.eps)
    # ``added`` is the residual stream for mlp_bda and ``normed`` is pre_mlp_layernorm's output.
    # The original _forward_mlp cannot be handed a precomputed norm, so the remaining tail is
    # inlined: decoder-only, no recompute or offload, no mlp chunking.
    mlp_output_with_bias = self.mlp(normed, padding_mask=padding_mask)
    with self.bias_dropout_add_exec_handler():
        out = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
            mlp_output_with_bias, added, self.hidden_dropout
        )
    from megatron.core.utils import make_viewless_tensor

    return make_viewless_tensor(inp=out, requires_grad=out.requires_grad, keep_graph=True)


def install(*, enabled: bool = True) -> bool:
    """Route the self_attn_bda -> pre_mlp_layernorm seam through fused_add_rmsnorm on both runtimes.

    Idempotent class-level patch of ``TransformerLayer._forward_attention`` / ``_forward_mlp``,
    correct on both sides because the fusion goes through the autograd.Function. Any config the
    kernel does not own falls back to the original sub-methods.
    """
    global _ORIG_FA, _ORIG_FM, _INSTALLED
    if _INSTALLED or not enabled:
        return _INSTALLED
    try:
        from megatron.core.transformer.transformer_layer import TransformerLayer
    except Exception:
        return False
    _ORIG_FA = TransformerLayer._forward_attention
    _ORIG_FM = TransformerLayer._forward_mlp
    TransformerLayer._forward_attention = _ix_forward_attention
    TransformerLayer._forward_mlp = _ix_forward_mlp
    print(
        "[ISOEXEC-NORMS] fused add+RMSNorm installed on TransformerLayer (within-layer seam, "
        "both runtimes; bit-equal at width 2048)",
        flush=True,
    )
    _INSTALLED = True
    return True
