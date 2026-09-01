"""One-launch slot->row resolution for the captured decode graph.

Host-free and shape-static so it captures into a CUDA graph.
``SKYRL_ISOEXEC_GDN_CPR_FUSED_ROWS=0`` (or an unexpected shape/dtype) falls back to the ATen chain.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

BANNER = "[ISOEXEC-GDN-ROWS]"
_ENV = "SKYRL_ISOEXEC_GDN_CPR_FUSED_ROWS"

_served = 0
_declined = 0
_reported = 0
_decline_reason = ""


def fused_rows_enabled() -> bool:
    """Default on; the env var is a kill switch for bisection."""
    return os.environ.get(_ENV, "1").lower() not in ("", "0", "false", "no")


@triton.jit
def _cpr_rows_kernel(
    slots_ptr,
    map_ptr,
    rows_ptr,
    rows32_ptr,
    n,
    map_n,
    BLK: tl.constexpr,
):
    offs = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = offs < n
    s = tl.load(slots_ptr + offs, mask=m, other=0).to(tl.int64)
    # Matches ``slots.long().clamp_(0, map_n - 1)``.
    s = tl.minimum(tl.maximum(s, 0), map_n - 1)
    r = tl.load(map_ptr + s, mask=m, other=0)
    tl.store(rows_ptr + offs, r, mask=m)
    tl.store(rows32_ptr + offs, r.to(tl.int32), mask=m)


def cpr_resolve_rows(slots: torch.Tensor, slot2row: torch.Tensor):
    """``(rows_int64, rows_int32)`` for a decode batch's engine slot ids; ``None`` to request fallback."""
    global _served, _declined, _reported, _decline_reason
    if slots.dim() != 1 or slot2row.dim() != 1 or slot2row.dtype != torch.int64:
        _declined += 1
        _decline_reason = (
            f"slots.dim={slots.dim()} map.dim={slot2row.dim()} map.dtype={slot2row.dtype} "
            "(expected 1-D slots, 1-D int64 map)"
        )
        _maybe_report()
        return None
    n = slots.numel()
    if n == 0:
        # Avoid handing Triton a zero-sized grid at capture time.
        z = torch.empty(0, dtype=torch.int64, device=slot2row.device)
        return z, z.to(torch.int32)
    rows = torch.empty(n, dtype=torch.int64, device=slot2row.device)
    rows32 = torch.empty(n, dtype=torch.int32, device=slot2row.device)
    blk = min(1024, max(32, triton.next_power_of_2(max(n, 1))))
    _cpr_rows_kernel[(triton.cdiv(n, blk),)](
        slots,
        slot2row,
        rows,
        rows32,
        n,
        slot2row.numel(),
        BLK=blk,
        num_warps=4,
    )
    _served += 1
    _maybe_report()
    return rows, rows32


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
