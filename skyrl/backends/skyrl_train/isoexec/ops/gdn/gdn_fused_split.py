"""Materialize native-core q/k/v and optional alpha/beta outputs in one copy kernel.

The vendor recurrent core requires contiguous inputs. This kernel combines the independent slice
copies without arithmetic or byte changes. It serves layouts with a unit last stride and declines
others to the ATen ``contiguous`` chain; ``SKYRL_ISOEXEC_GDN_FUSED_SPLIT=0`` disables it entirely.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

BANNER = "[ISOEXEC-GDN-SPLIT]"
_ENV = "SKYRL_ISOEXEC_GDN_FUSED_SPLIT"

# Launch shape only, nothing here rounds: 128 bf16 columns is a 256-byte contiguous run per row,
# and 16 rows keeps the narrow a/b segments from needing their own grid.
_BLOCK_T = 16
_BLOCK_C = 128

_served = 0
_declined = 0
_reported = 0
_decline_reason = ""


def fused_split_enabled() -> bool:
    """Default on. Pure byte movement, so the flag is a bisection kill switch, not a numerics gate."""
    return os.environ.get(_ENV, "1").lower() not in ("", "0", "false", "no")


@triton.jit
def _copy_tile(src, dst, srow, coff, W, T, rb, cb, BT: tl.constexpr, BC: tl.constexpr):
    """``dst[r, c] = src[r, coff + c]`` for one tile. Load and store, no arithmetic on the value."""
    r = (rb * BT + tl.arange(0, BT)).to(tl.int64)
    c = (cb * BC + tl.arange(0, BC)).to(tl.int64)
    m = (r[:, None] < T) & (c[None, :] < W)
    x = tl.load(src + r[:, None] * srow + coff + c[None, :], mask=m)
    tl.store(dst + r[:, None] * W + c[None, :], x, mask=m)


@triton.jit
def _split_qkvab_kernel(
    y_ptr,
    a_ptr,
    b_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    ao_ptr,
    bo_ptr,
    T,
    sy,
    sa,
    sb,
    KD,
    VD,
    HA,
    BT: tl.constexpr,
    BC: tl.constexpr,
):
    """Five contiguous outputs from two strided sources, in one launch.

    ``tl.program_id(2)`` selects the segment and is uniform within a program, so the branch is a
    grid-level dispatch, not divergence. The grid's third dimension is 3 when there is no alpha/beta
    to materialise, and segments 3 and 4 are then never scheduled.
    """
    rb = tl.program_id(0)
    cb = tl.program_id(1)
    seg = tl.program_id(2)
    if seg == 0:
        _copy_tile(y_ptr, q_ptr, sy, 0, KD, T, rb, cb, BT, BC)
    elif seg == 1:
        _copy_tile(y_ptr, k_ptr, sy, KD, KD, T, rb, cb, BT, BC)
    elif seg == 2:
        _copy_tile(y_ptr, v_ptr, sy, 2 * KD, VD, T, rb, cb, BT, BC)
    elif seg == 3:
        _copy_tile(a_ptr, ao_ptr, sa, 0, HA, T, rb, cb, BT, BC)
    else:
        _copy_tile(b_ptr, bo_ptr, sb, 0, HA, T, rb, cb, BT, BC)


def fused_split_qkvab(
    y: torch.Tensor,
    kd: int,
    vd: int,
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
):
    """``(q, k, v, a_c, b_c)``: the five contiguous tensors, in one launch.

    ``y`` is the post-conv ``[T, 2*kd + vd]`` row and q/k/v are its three column slices made
    contiguous. ``a``/``b``, when given, are ``[T, HA]`` views compacted in the same launch; when
    omitted the last two entries are ``None`` and the grid carries three segments. Returns ``None``
    for anything that is not the expected shape, so the caller falls back to the ATen chain.
    Host-free and shape-static, so it captures into a CUDA graph.
    """
    global _served, _declined, _reported, _decline_reason

    def _decline(why: str):
        global _declined
        _declined += 1
        globals()["_decline_reason"] = why
        _maybe_report()
        return None

    if y.dim() != 2 or y.stride(-1) != 1:
        return _decline(f"y.dim={y.dim()} y.stride(-1)={y.stride(-1)} (expected 2-D, unit last stride)")
    T, D = y.shape
    if 2 * kd + vd != D:
        return _decline(f"2*kd+vd={2 * kd + vd} != y.shape[-1]={D}")
    if T == 0:
        return _decline("T=0")
    ha = 0
    if a is not None or b is not None:
        if a is None or b is None:
            return _decline("alpha/beta must be given together")
        if a.dim() != 2 or b.dim() != 2 or a.stride(-1) != 1 or b.stride(-1) != 1:
            return _decline(f"a.dim={a.dim()} b.dim={b.dim()} a.stride(-1)={a.stride(-1)} b.stride(-1)={b.stride(-1)}")
        if a.shape != b.shape or a.shape[0] != T or a.dtype != b.dtype:
            return _decline(f"a.shape={tuple(a.shape)} b.shape={tuple(b.shape)} T={T}")
        ha = a.shape[1]

    q = torch.empty(T, kd, dtype=y.dtype, device=y.device)
    k = torch.empty(T, kd, dtype=y.dtype, device=y.device)
    v = torch.empty(T, vd, dtype=y.dtype, device=y.device)
    ac = torch.empty(T, ha, dtype=a.dtype, device=a.device) if ha else None
    bc = torch.empty(T, ha, dtype=b.dtype, device=b.device) if ha else None

    nseg = 5 if ha else 3
    widest = max(kd, vd, ha)
    grid = (triton.cdiv(T, _BLOCK_T), triton.cdiv(widest, _BLOCK_C), nseg)
    _split_qkvab_kernel[grid](
        y,
        a if ha else y,  # unused when nseg==3, but a pointer arg still has to be a tensor
        b if ha else y,
        q,
        k,
        v,
        ac if ha else q,
        bc if ha else q,
        T,
        y.stride(0),
        a.stride(0) if ha else 0,
        b.stride(0) if ha else 0,
        kd,
        vd,
        ha,
        BT=_BLOCK_T,
        BC=_BLOCK_C,
        num_warps=4,
    )
    _served += 1
    _maybe_report()
    return q, k, v, ac, bc


def contiguous_ab(a: torch.Tensor, b: torch.Tensor):
    """``a``/``b`` compacted at the call site: the flag-off and non-native-core path.

    ``.contiguous()`` on an already-contiguous tensor returns ``self`` and launches nothing.
    """
    return a.contiguous(), b.contiguous()


def defer_ab(state) -> bool:
    """Whether ``state``'s core runs the fused split, i.e. whether alpha/beta may reach it strided.

    Only the native-core composition does; every other path has alpha/beta compacted by the caller.
    """
    return fused_split_enabled() and bool(getattr(state, "_native_core", False))


def _maybe_report() -> None:
    """Log once at the first call and then at power-of-two call counts."""
    global _reported
    total = _served + _declined
    if total < 1 or (total & (total - 1)) != 0 or total == _reported:
        return
    _reported = total
    print(
        f"{BANNER} pid={os.getpid()} served={_served} declined={_declined} "
        f"(python calls, NOT graph replays)" + (f" last_decline={_decline_reason}" if _declined else ""),
        flush=True,
    )
