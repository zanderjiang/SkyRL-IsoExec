"""One-launch open-chunk buffer scatter for CprGDN.decode.

Pure data movement in the arrival dtypes; pad lanes resolve to row 0 (the null row).
Host-free and shape-static so it captures into a CUDA graph.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

_ENV = "SKYRL_ISOEXEC_GDN_CPR_FUSED_SCATTER"


def fused_scatter_enabled() -> bool:
    """Default on; the env var is a kill switch for bisection."""
    return os.environ.get(_ENV, "1").lower() not in ("", "0", "false", "no")


@triton.jit
def _cpr_scatter_kernel(
    k_ptr,
    v_ptr,
    g_ptr,
    b_ptr,
    kbuf_ptr,
    vbuf_ptr,
    gbuf_ptr,
    bbuf_ptr,
    rows_ptr,
    pos_ptr,
    lastused_ptr,
    clock_ptr,
    k_row_stride,
    v_row_stride,
    g_row_stride,
    b_row_stride,
    C: tl.constexpr,
    HVK: tl.constexpr,  # HV * K elements per lane in k
    HVV: tl.constexpr,  # HV * V elements per lane in v
    HV: tl.constexpr,
    HAS_LU: tl.constexpr,
    BLK: tl.constexpr,
):
    n = tl.program_id(0)
    row = tl.load(rows_ptr + n).to(tl.int64)
    pos = tl.load(pos_ptr + row)
    col = pos % C

    offs = tl.arange(0, BLK)
    # k [n, HV, K] -> k_buf[row, col]
    src = k_ptr + n * k_row_stride
    dst = kbuf_ptr + (row * C + col) * HVK
    for i in range(0, HVK, BLK):
        m = (i + offs) < HVK
        tl.store(dst + i + offs, tl.load(src + i + offs, mask=m), mask=m)
    # v [n, HV, V] -> v_buf[row, col]
    src = v_ptr + n * v_row_stride
    dst = vbuf_ptr + (row * C + col) * HVV
    for i in range(0, HVV, BLK):
        m = (i + offs) < HVV
        tl.store(dst + i + offs, tl.load(src + i + offs, mask=m), mask=m)
    # g fp32 / beta [n, HV]
    hofs = tl.arange(0, HV)
    tl.store(gbuf_ptr + (row * C + col) * HV + hofs, tl.load(g_ptr + n * g_row_stride + hofs))
    tl.store(bbuf_ptr + (row * C + col) * HV + hofs, tl.load(b_ptr + n * b_row_stride + hofs))

    # Live lanes advance; the null row is re-stored as 0 (racing pad lanes all store the same 0).
    tl.store(pos_ptr + row, pos + tl.where(row > 0, 1, 0))

    # The clock is bumped by the caller, so racing null rows all store an identical stamp.
    if HAS_LU:
        tl.store(lastused_ptr + row, tl.load(clock_ptr))


def cpr_buffer_scatter(
    k, v, g, beta, rows, pos, k_buf, v_buf, g_buf, b_buf, chunk_size: int, last_used=None, clock=None
) -> None:
    """Scatter this step's post-prep values into the open-chunk buffers and advance ``pos``.

    Inner dims must be contiguous (row strides are free); rows may repeat only on the null row.
    ``last_used`` (int64 LRU array) and ``clock`` (0-d int64, pre-bumped) are passed together or not at all.
    """
    N, Hk, K = k.shape  # Hk may be the GQA-compressed head count (the native path buffers raw k)
    V = v.shape[-1]
    HV = g.shape[-1]
    HVpow = triton.next_power_of_2(HV)
    if HVpow != HV:
        raise ValueError(f"[isoexec-gdn] cpr scatter: HV={HV} must be a power of 2 (pad the head dim path)")
    has_lu = last_used is not None
    if has_lu != (clock is not None):
        raise ValueError("[isoexec-gdn] cpr scatter: last_used and clock are passed together or not at all")
    if has_lu and (last_used.dtype != torch.int64 or clock.dtype != torch.int64 or clock.numel() != 1):
        raise TypeError("[isoexec-gdn] cpr scatter: last_used must be int64[rows] and clock a 0-d int64")
    if k.dtype != k_buf.dtype or v.dtype != v_buf.dtype or g.dtype != g_buf.dtype or beta.dtype != b_buf.dtype:
        raise TypeError("[isoexec-gdn] cpr scatter: caller converts dtypes up front (pure copies only)")
    _cpr_scatter_kernel[(N,)](
        k,
        v,
        g,
        beta,
        k_buf,
        v_buf,
        g_buf,
        b_buf,
        rows,
        pos,
        last_used if has_lu else rows,
        clock if has_lu else rows,
        k.stride(0),
        v.stride(0),
        g.stride(0),
        beta.stride(0),
        C=chunk_size,
        HVK=Hk * K,
        HVV=HV * V,
        HV=HV,
        HAS_LU=has_lu,
        BLK=256,
        num_warps=2,
    )
