"""Fused zero-centred RMSNorm, and the GatedDeltaNet gated output norm.

Two eager expressions become one kernel each: ``F.rms_norm(x, (N,), None, eps) * (1.0 + weight)``
(three launches, at every norm site) and the GDN gated out-norm (seven launches per GDN layer).
Both are bitwise-equal to the expressions they replace, so they are installed engine-only and the
trainer keeps the eager path -- no autograd.Function and no backward.

THE REDUCTION TILE IS THE BITWISE CONTRACT, NOT A TUNING KNOB. ``F.rms_norm`` dispatches to torch's
vendored quack CuTe-DSL kernel, whose launch config is chosen from N alone -- which is why the op is
batch-invariant -- and that config fixes ``threads_per_row``, which fixes the reduction tree.
:func:`_tile_for` inverts the same ladder into a Triton ``(rows_per_program, num_warps)`` pair. Do
not autotune, do not widen the block for occupancy, and never let the row count reach the tile: the
shape may affect the grid only. A wrong tile costs a handful of elements in ten million, which no
``allclose`` will show and which the live gate shows as a floor that moved.

The other rounding pins, each measured rather than argued:

  * ``rstd`` is ``tl.rsqrt`` (rsqrt.approx.f32), matching quack's fastmath;
    ``libdevice.rsqrt_rn`` and ``1.0 / sqrt(x)`` do not match.
  * ``enable_fp_fusion=False``: Triton would contract ``acc + b*c`` into one FFMA, which is
    bitwise-equal to ``torch.addcmul`` and not to eager.
  * The down-cast chain, in order: fp32 reduce -> round to bf16 -> multiply gamma in bf16 -> promote
    to fp32 -> multiply by the fp32 SiLU -> round to bf16. The plain RMSNorm truncates it after the
    second round. A naive fused kernel stays in fp32 from reduction to store and gets all three wrong.
  * SiLU is ``x / (1 + exp(-x))``, ATen's fp32 form -- the opposite choice from the GDN conv site,
    which writes ``x * sigmoid(x)``. The form is a property of the call site.
  * ``exp`` is ``libdevice.exp``, since ``tl.exp`` is ``ex2.approx``; the divide is Triton's non-FTZ
    ``div_rn``, since Triton links libdevice with ``__CUDA_FTZ`` and that flushes a subnormal
    quotient -- which the SiLU produces for gate values below about -88.

``1.0 + weight`` IS FORMED IN REGISTERS, not cached at weight sync. Caching it would need an
invalidation at a boundary a CUDA-graph replay cannot skip; miss one and the engine serves last
step's norm weights with no error and a forward-only gate that cannot see it. Folding the add into
the kernel gives the identical launch saving with no second copy of the weights to go stale.

Installed per instance on the engine's model, never on the class: the trainer's norm must keep the
``1.0 +`` inside ``forward`` or ``weight`` leaves the autograd graph, and the GDN hook likewise runs
only in the engine's inference forward -- a grad_fn-less Triton call on the trainer would silently
sever the backward.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ...core import triton_nonftz as _nonftz

FLAG = "SKYRL_ISOEXEC_GDN_FUSED_OUTNORM"

# quack's ladder: this decides the fp32 reduction order, so it is the bitwise contract.
_THREADS_PER_ROW_LADDER = ((64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128))
# Above this width quack switches to a clustered launch, splitting one row across CTAs and reducing
# in an order a single-CTA Triton program cannot reproduce. Refuse rather than guess.
_MAX_N = 16384

_LOGGED = False
_FALLBACK_LOGGED = False


def fused_outnorm_enabled() -> bool:
    """``SKYRL_ISOEXEC_GDN_FUSED_OUTNORM=1``, default off.

    Logged once per process at first read, so the value the engine actor saw appears in its own log
    rather than only the value the launcher exported.
    """
    global _LOGGED
    on = os.environ.get(FLAG, "0") == "1"
    if not _LOGGED:
        _LOGGED = True
        print(
            f"[ISOEXEC-GDN] {FLAG}={os.environ.get(FLAG, '<unset>')} -> fused out-norm {'ON' if on else 'OFF'}",
            flush=True,
        )
    return on


def _threads_per_row(N: int) -> int:
    for limit, t in _THREADS_PER_ROW_LADDER:
        if N <= limit:
            return t
    return 256


def _tile_for(N: int) -> tuple[int, int]:
    """``(rows_per_program, num_warps)`` giving each row exactly quack's ``threads_per_row``.

    A Triton warp is 32 lanes, so ``rows_per_program = 32 // tpr`` when the row fits inside one warp
    and ``num_warps = tpr // 32`` when it does not. Both are pinned constants for a given N, and
    neither depends on the number of rows.
    """
    tpr = _threads_per_row(N)
    if tpr <= 32:
        return 32 // tpr, 1
    return 1, tpr // 32


def _supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dtype in (torch.bfloat16, torch.float16)
        and weight.dtype == x.dtype
        and x.shape[-1] <= _MAX_N
        and x.is_contiguous()
        and weight.is_contiguous()
    )


# Zero-centred RMSNorm: rms_norm(x) rounded to bf16, then * (1 + weight) in bf16.
@triton.jit
def _gamma(w_ptr, c, N: tl.constexpr):
    """``1.0 + weight``, formed in registers from the live parameter.

    Down-cast 1 of the chain: eager evaluates ``1.0 + self.weight`` as a bf16 tensor and only then
    multiplies, so keeping the sum in fp32 here would skip a round and change the function.
    """
    w = tl.load(w_ptr + c, mask=c < N, other=0.0).to(tl.float32)
    return (1.0 + w).to(w_ptr.dtype.element_ty).to(tl.float32)


@triton.jit
def _rms_gamma_kernel(
    x_ptr,
    gam_ptr,
    o_ptr,
    M,
    eps,
    N: tl.constexpr,
    BN: tl.constexpr,
    MB: tl.constexpr,
):
    # int64 row index: `rows * N` is an offset multiply, and past 2^31 elements an int32 index
    # silently wraps and reads the wrong rows rather than faulting.
    rows = tl.program_id(0).to(tl.int64) * MB + tl.arange(0, MB)[:, None]
    c = tl.arange(0, BN)[None, :]
    rmask = rows < M
    mask = rmask & (c < N)

    x = tl.load(x_ptr + rows * N + c, mask=mask, other=0.0).to(tl.float32)
    # The fp32 tree here is fixed by MB/num_warps at the launch, not by M. tl.rsqrt is
    # rsqrt.approx.f32, which is what quack emits with fastmath=True.
    #
    # The FTZ'd rsqrt flushes a subnormal argument and returns inf, which cannot happen here: the
    # argument is `sum/N + eps` and so is at least eps, far above the subnormal boundary. A caller
    # passing eps=0 would break that, which is why eps is never defaulted in this file.
    s = tl.sum(x * x, 1)[:, None]
    rstd = tl.rsqrt(s / N + eps)
    # Down-cast 2: F.rms_norm's output is bf16, before gamma is applied.
    nb = (x * rstd).to(o_ptr.dtype.element_ty)
    gam = _gamma(gam_ptr, c, N)
    # Down-cast 3: a bf16 elementwise multiply, which ATen computes in fp32 and rounds once.
    tl.store(o_ptr + rows * N + c, (nb.to(tl.float32) * gam).to(o_ptr.dtype.element_ty), mask=mask)


def fused_rms_norm_gamma(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Bitwise-equal to ``F.rms_norm(x, (N,), None, eps) * (1.0 + weight)``.

    ``weight`` is the raw zero-centred parameter; the ``1.0 +`` happens in registers.
    """
    N = x.shape[-1]
    if not x.is_contiguous() or not weight.is_contiguous():
        # Rows are addressed as `base + rows*N + c` with implicit unit strides, so a strided input
        # would silently normalise the wrong elements rather than fault.
        raise RuntimeError(f"x/weight must be contiguous, got {x.stride()} / {weight.stride()}")
    x2 = x.reshape(-1, N)
    M = x2.shape[0]
    mb, nw = _tile_for(N)
    out = torch.empty_like(x2)
    _rms_gamma_kernel[(triton.cdiv(M, mb),)](
        x2,
        weight,
        out,
        M,
        eps,
        N=N,
        BN=triton.next_power_of_2(N),
        MB=mb,
        num_warps=nw,
        enable_fp_fusion=False,  # FFMA contraction would give addcmul, not eager
    )
    return out.view(x.shape)


# GDN gated output norm: the RMSNorm chain, then * SiLU(gate) in fp32, then one down-cast.
@triton.jit
def _gated_out_norm_kernel(
    x_ptr,
    gate_ptr,
    gam_ptr,
    o_ptr,
    M,
    stride_gate_t,
    eps,
    HV: tl.constexpr,
    N: tl.constexpr,
    BN: tl.constexpr,
    MB: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * MB + tl.arange(0, MB)[:, None]  # int64: see _rms_gamma_kernel
    c = tl.arange(0, BN)[None, :]
    rmask = rows < M
    mask = rmask & (c < N)

    x = tl.load(x_ptr + rows * N + c, mask=mask, other=0.0).to(tl.float32)
    s = tl.sum(x * x, 1)[:, None]
    rstd = tl.rsqrt(s / N + eps)
    nb = (x * rstd).to(x_ptr.dtype.element_ty)  # down-cast 2
    gam = _gamma(gam_ptr, c, N)
    y = (nb.to(tl.float32) * gam).to(x_ptr.dtype.element_ty)  # down-cast 3

    # `gate` is a last-dim slice of the in_proj output, so flattening it across the token axis
    # cannot be a view and eager pays a copy. Reading it strided removes that: output row r is
    # (token r // HV, v-head r % HV).
    t = rows // HV
    h = rows % HV
    g = tl.load(gate_ptr + t * stride_gate_t + h * N + c, mask=mask, other=0.0).to(tl.float32)
    # ATen's fp32 SiLU is x / (1 + exp(-x)), not x * sigmoid(x).
    #
    # The divide must be the non-FTZ one. Triton links libdevice with __CUDA_FTZ, so
    # `libdevice.div_rn` flushes a subnormal quotient to zero. Both operands here are always normal,
    # but for g below about -88 the QUOTIENT is fp32 subnormal, and ATen keeps it. `_nonftz.div_rn`
    # is inline PTX `div.rn.f32`: correctly rounded and subnormal-preserving.
    #
    # `libdevice.exp` is audited safe on subnormals and stays; `tl.exp` is ex2.approx and must not
    # be used here.
    act = _nonftz.div_rn(g, 1.0 + libdevice.exp(-g))
    # Down-cast 4, the single round after the fp32 gate multiply.
    tl.store(o_ptr + rows * N + c, (y.to(tl.float32) * act).to(o_ptr.dtype.element_ty), mask=mask)


def fused_gated_out_norm(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Bitwise-equal to ``_eager_apply_gated_norm`` with a zero-centred ``out_norm``.

    ``x`` is ``[T, HV, N]`` contiguous (the GDN core output); ``gate`` is ``[T, HV, N]`` viewing a
    ``[T, in_proj]`` slice, so only its token stride is free. Returns ``[T*HV, N]``.
    """
    T, HV, N = x.shape
    if gate.shape != x.shape:
        raise ValueError(f"gate shape {tuple(gate.shape)} != x shape {tuple(x.shape)}")
    if gate.stride(1) != N or gate.stride(2) != 1:
        # Only the token stride may differ from contiguous; the head and feature axes use implicit
        # strides, so another layout would silently read the wrong gate rather than fault.
        raise RuntimeError(f"gate must be unit-stride within a token, got strides {gate.stride()}")
    M = T * HV
    mb, nw = _tile_for(N)
    out = torch.empty(M, N, dtype=x.dtype, device=x.device)
    _gated_out_norm_kernel[(triton.cdiv(M, mb),)](
        x,
        gate,
        weight,
        out,
        M,
        gate.stride(0),
        eps,
        HV=HV,
        N=N,
        BN=triton.next_power_of_2(N),
        MB=mb,
        num_warps=nw,
        enable_fp_fusion=False,
    )
    return out


# Install: engine only, per-instance rebinds on the engine's model, never on the class.
def _fused_norm_forward(self, x: torch.Tensor) -> torch.Tensor:
    """``rms_norm(x) * (1.0 + weight)`` in one kernel, reading ``self.weight`` live.

    There is no cached gamma to go stale: a weight sync needs no notification and a replayed CUDA
    graph reads the parameter's own storage.
    """
    import torch.nn.functional as F

    w = self.weight
    if not _supported(x, w):
        # Correct but unfused. Said once, because a run that falls back everywhere looks exactly
        # like a run where the flag never arrived.
        global _FALLBACK_LOGGED
        if not _FALLBACK_LOGGED:
            _FALLBACK_LOGGED = True
            print(
                f"[ISOEXEC-GDN] fused norm FELL BACK to eager: shape={tuple(x.shape)} dtype={x.dtype} "
                f"contiguous={x.is_contiguous()} weight_dtype={w.dtype}",
                flush=True,
            )
        return F.rms_norm(x, self.hidden_size, None, self.eps) * (1.0 + self.weight)
    return fused_rms_norm_gamma(x, w, self.eps)


def install_engine_fused_norms(gpt_modules) -> int:
    """Rebind ``forward`` on every ``ZeroCenteredTorchRMSNorm`` instance of the engine's GPTModel.

    Returns the number of instances swapped, 0 if the flag is off. Instance-level, so the trainer
    process -- which constructs the very same class -- is untouched by construction.
    """
    if not fused_outnorm_enabled():
        return 0
    from .zero_centered_norm import ZeroCenteredTorchRMSNorm

    n = 0
    widths: dict[int, int] = {}
    for m in gpt_modules.modules():
        if not isinstance(m, ZeroCenteredTorchRMSNorm):
            continue
        # The marker the GDN forward reads to decide whether it may use the fused out-norm. An
        # attribute rather than a buffer, so there is nothing to keep in sync.
        m._ix_fused_norm = True
        m.forward = _fused_norm_forward.__get__(m, type(m))
        n += 1
        w = m.hidden_size[0]
        widths[w] = widths.get(w, 0) + 1
    if n:
        tiles = {w: _tile_for(w) for w in widths}
        print(
            f"[ISOEXEC-GDN] fused zero-centred RMSNorm on {n} engine norm(s) "
            f"(widths {dict(sorted(widths.items()))}, tiles rows/warps {tiles}) -- K2 + K2a. "
            "EVERY width here must appear in gdn_fused_outnorm_test.py: the reduction tile is chosen "
            "per width, so an ungated width is an ungated bitwise contract.",
            flush=True,
        )
    return n
