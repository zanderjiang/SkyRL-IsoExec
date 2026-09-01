"""Byte-exact launch fusion for the elementwise half of ``native_matched_prep``.

The canonical boundary preparation stays defined by ``gdn_cpr.native_matched_prep``; this
module only collapses its eager ``a/b -> g/beta`` expression into one launch. The key normalisation
is deliberately left alone, because PyTorch's fp32 reduction order differs from vLLM's row-norm
kernel for some 128-wide rows and substituting it would change bits.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

FLAG = "SKYRL_ISOEXEC_GDN_MATCHED_PREP_FUSED_GATE"
_L2_MIN_ELEMENTS = 8 * 1024 * 1024

_COUNTS = {"gate_served": 0, "gate_declined": 0, "l2_served": 0, "l2_declined": 0}


def matched_prep_fused_gate_enabled() -> bool:
    """Default on; ``0`` restores the canonical eager elementwise expression."""
    return os.environ.get(FLAG, "1").strip().lower() not in ("", "0", "false", "no")


@triton.jit
def _matched_prep_square_kernel(x_ptr, square_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    x = tl.load(x_ptr + i, mask=mask, other=0.0).to(tl.float32)
    tl.store(square_ptr + i, x * x, mask=mask)


@triton.jit
def _matched_prep_finish_l2_kernel(
    x_ptr,
    square_sum_ptr,
    out_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    x = tl.load(x_ptr + i, mask=mask, other=0.0).to(tl.float32)
    square_sum = tl.load(square_sum_ptr + i // K, mask=mask, other=0.0)
    # Canonical post-reduction sequence: add eps, rsqrt, multiply, wire cast.
    y = x * tl.rsqrt(square_sum + 1e-6)
    tl.store(out_ptr + i, y, mask=mask)


def maybe_matched_prep_fused_l2(k_raw: torch.Tensor, *, force: bool = False) -> torch.Tensor | None:
    """Exact key normalisation with PyTorch's reduction association preserved.

    The square and post-reduction elementwise chains are fused, but the middle ``torch.sum`` is
    retained: its fp32 association is the canonical pin and differs from a one-kernel Triton row
    reduction on 128-wide inputs.
    """
    if not matched_prep_fused_gate_enabled():
        _COUNTS["l2_declined"] += 1
        return None
    if (
        k_raw.device.type != "cuda"
        or k_raw.dtype not in (torch.bfloat16, torch.float32)
        or k_raw.ndim < 2
        or k_raw.shape[-1] <= 0
        or not k_raw.is_contiguous()
        or k_raw.numel() == 0
        # The three-launch form only pays off on large bf16 inputs on SM90; elsewhere the eager
        # chain is exact and faster.
        or (not force and (k_raw.dtype != torch.bfloat16 or k_raw.numel() < _L2_MIN_ELEMENTS))
        or (not force and torch.cuda.get_device_capability(k_raw.device) != (9, 0))
    ):
        _COUNTS["l2_declined"] += 1
        return None
    square = torch.empty_like(k_raw, dtype=torch.float32)
    out = torch.empty_like(k_raw)
    n = k_raw.numel()
    block = 256
    _matched_prep_square_kernel[(triton.cdiv(n, block),)](
        k_raw,
        square,
        N=n,
        BLOCK=block,
        num_warps=4,
        enable_fp_fusion=False,
    )
    # Do not replace this with a Triton reduction: the fp32 association is observable.
    square_sum = square.sum(dim=-1, keepdim=True)
    _matched_prep_finish_l2_kernel[(triton.cdiv(n, block),)](
        k_raw,
        square_sum,
        out,
        N=n,
        K=k_raw.shape[-1],
        BLOCK=block,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _COUNTS["l2_served"] += 1
    return out


@triton.jit
def _div_rn_nonftz(a, b):
    # Triton's `/` is reciprocal-approximate and libdevice div flushes subnormal quotients; ATen
    # sigmoid does neither. Same contract as gdn_fused_prep._sigmoid, signed zeros included.
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        "=r,r,r",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _neg_preserve_zero(a):
    """Flip the fp32 sign bit without letting LLVM reassociate ``-(a*b)``.

    Harmless except when ``exp(A_log)`` underflows: eager materialises ``-0`` before multiplying,
    while the reassociated form yields ``+0`` for a positive softplus.
    """
    return tl.inline_asm_elementwise(
        "xor.b32 $0, $1, 0x80000000;",
        "=r,r",
        [a],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _matched_prep_gate_kernel(
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    g_ptr,
    beta_ptr,
    T,
    H: tl.constexpr,
    A_STRIDE_T: tl.constexpr,
    B_STRIDE_T: tl.constexpr,
    A_LOG_STRIDE_T: tl.constexpr,
    DT_STRIDE_T: tl.constexpr,
    A_TOKENWISE: tl.constexpr,
    DT_TOKENWISE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)[:, None]
    h = tl.arange(0, BLOCK_H)[None, :]
    mask = (t < T) & (h < H)
    i = t * H + h

    av = tl.load(a_ptr + t * A_STRIDE_T + h, mask=mask, other=0.0).to(tl.float32)
    bv = tl.load(b_ptr + t * B_STRIDE_T + h, mask=mask, other=0.0).to(tl.float32)
    param_mask = mask if A_TOKENWISE else h < H
    al = tl.load(
        A_log_ptr + (t * A_LOG_STRIDE_T + h if A_TOKENWISE else h),
        mask=param_mask,
        other=0.0,
    ).to(tl.float32)
    param_mask = mask if DT_TOKENWISE else h < H
    db = tl.load(
        dt_bias_ptr + (t * DT_STRIDE_T + h if DT_TOKENWISE else h),
        mask=param_mask,
        other=0.0,
    ).to(tl.float32)

    # Character-for-character native_matched_prep arithmetic: log(1 + exp(x)), not softplus/log1p,
    # with a <= 20 threshold. `enable_fp_fusion=False` prevents contraction across these fp32 ops.
    x = av + db
    ex = libdevice.exp(x)
    softplus = tl.where(x <= 20.0, libdevice.log(1.0 + ex), x)
    g = _neg_preserve_zero(libdevice.exp(al)) * softplus
    beta = _div_rn_nonftz(tl.full(x.shape, 1.0, tl.float32), 1.0 + libdevice.exp(-bv))

    tl.store(g_ptr + i, g, mask=mask)
    tl.store(beta_ptr + i, beta, mask=mask)


def maybe_matched_prep_fused_gate(
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return exact ``(g fp32, beta b.dtype)``, or ``None`` for the eager fallback.

    Admission is the kernel's complete indexing contract and depends only on runtime shape, dtype,
    layout and device.
    """
    if not matched_prep_fused_gate_enabled():
        _COUNTS["gate_declined"] += 1
        return None

    # Trainer scoring carries the packed batch singleton as [1,T,H] and the engine uses [T,H];
    # requiring viewability keeps that squeeze free and rejects a materialising reshape.
    if a.ndim not in (2, 3) or (a.ndim == 3 and a.shape[0] != 1) or b.shape != a.shape:
        _COUNTS["gate_declined"] += 1
        return None
    try:
        a_rows = a.view(-1, a.shape[-1])
        b_rows = b.view(-1, b.shape[-1])
    except (RuntimeError, ValueError):
        _COUNTS["gate_declined"] += 1
        return None

    def _dense_rows(t: torch.Tensor) -> bool:
        return t.ndim == 2 and t.stride(1) == 1 and t.stride(0) >= t.shape[1]

    def _parameter_rows(t: torch.Tensor) -> torch.Tensor | None:
        if t.numel() == a_rows.shape[1] and t.shape[-1:] == (a_rows.shape[1],):
            try:
                broadcast = t.view(a_rows.shape[1])
            except (RuntimeError, ValueError):
                return None
            return broadcast if broadcast.stride(0) == 1 else None
        if t.numel() != a_rows.numel() or t.shape[-1:] != (a_rows.shape[1],):
            return None
        try:
            rows = t.view_as(a_rows)
        except (RuntimeError, ValueError):
            return None
        return rows if _dense_rows(rows) else None

    A_log_rows = _parameter_rows(A_log)
    dt_bias_rows = _parameter_rows(dt_bias)
    if (
        a.device.type != "cuda"
        or b.shape != a.shape
        or b.dtype != a.dtype
        or a.dtype not in (torch.bfloat16, torch.float32)
        or A_log_rows is None
        or dt_bias_rows is None
        or A_log.dtype not in (torch.bfloat16, torch.float32)
        or dt_bias.dtype not in (torch.bfloat16, torch.float32)
        or any(t.device != a.device for t in (b, A_log, dt_bias))
        # Projection splits are dense along heads but row-strided by the full packed width;
        # arbitrary inner strides or overlapping rows still decline.
        or not _dense_rows(a_rows)
        or not _dense_rows(b_rows)
        or a.numel() == 0
    ):
        _COUNTS["gate_declined"] += 1
        return None

    # Eager pointwise outputs are compact even when their projection inputs are views.
    g = torch.empty(a_rows.shape, dtype=torch.float32, device=a.device)
    beta = torch.empty(a_rows.shape, dtype=b.dtype, device=b.device)
    block_t = 64
    _matched_prep_gate_kernel[(triton.cdiv(a_rows.shape[0], block_t),)](
        a_rows,
        b_rows,
        A_log_rows,
        dt_bias_rows,
        g,
        beta,
        T=a_rows.shape[0],
        H=a_rows.shape[1],
        A_STRIDE_T=a_rows.stride(0),
        B_STRIDE_T=b_rows.stride(0),
        A_LOG_STRIDE_T=A_log_rows.stride(0) if A_log_rows.ndim == 2 else 0,
        DT_STRIDE_T=dt_bias_rows.stride(0) if dt_bias_rows.ndim == 2 else 0,
        A_TOKENWISE=A_log_rows.ndim == 2,
        DT_TOKENWISE=dt_bias_rows.ndim == 2,
        BLOCK_T=block_t,
        BLOCK_H=triton.next_power_of_2(a_rows.shape[1]),
        num_warps=4,
        enable_fp_fusion=False,
    )
    _COUNTS["gate_served"] += 1
    return g.view(a.shape), beta.view(b.shape)
