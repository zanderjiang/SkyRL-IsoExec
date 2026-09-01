"""Triton kernels for the debug-mode tensor digest (see ``thash.py`` for the scheme).

Split out of ``thash`` so that module imports on a machine with no triton: everything here is
reached only through :func:`available`, and every function has a bit-identical eager twin in
``thash`` that is used when this module cannot load.

One pass over the tensor does the whole digest: the int view is loaded once, the position weight
is computed in registers, and every mantissa rung of the ladder is accumulated from that same
load. Per-block partial sums land in an int64 buffer that is reduced with one ``sum``; integer
addition mod 2**64 is associative, so the block decomposition does not change the result.
"""

from __future__ import annotations

import threading
from typing import List, Optional, Sequence

import torch
import triton
import triton.language as tl

# Mirrors of the thash constants, in the signed-int64 form triton arithmetic uses.
_MIX1 = -4658895280553007687  # 0xBF58476D1CE4E5B9
_MIX2 = -7723592293110705685  # 0x94D049BB133111EB
_GOLDEN = -7046029254386353131  # 0x9E3779B97F4A7C15
_LANE = -7046029254386353131  # per-lane decorrelation multiplier (odd)
_GROUP_SHIFT = 3  # one splitmix64 per 2**3 elements; see thash's scheme note

_MAX_MASKS = 8  # deepest ladder is fp64: full + 7 rungs
BLOCK = 4096
NUM_WARPS = 4


@triton.jit
def _lsr(z, n: tl.constexpr):
    """Logical right shift. ``>>`` on int64 is arithmetic, and triton mis-widens the 37-bit mask
    literal that would undo the sign fill, so the shift is done in uint64 (verified equal to
    ``thash._lsr`` bit for bit)."""
    return (z.to(tl.uint64) >> n).to(tl.int64)


@triton.jit
def _mix64(z):
    z = z ^ _lsr(z, 30)
    z = z * -4658895280553007687
    z = z ^ _lsr(z, 27)
    z = z * -7723592293110705685
    return z ^ _lsr(z, 31)


@triton.jit
def _weights(offs, salt):
    """Position weight stream: one splitmix64 per group of 8, decorrelated per lane, forced odd."""
    base = _mix64((offs >> 3) ^ salt)
    return (base ^ ((offs & 7) * -7046029254386353131)) | 1


@triton.jit
def _digest_kernel(
    iv_ptr,
    out_ptr,
    n,
    salt,
    m0,
    m1,
    m2,
    m3,
    m4,
    m5,
    m6,
    m7,
    NMASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Accumulate sum_i (x_i & mask_j) * w_i into out[j], one atomic per block per mask.

    Atomics rather than per-block partials + a torch ``sum``: integer addition is associative, so
    the atomic order is irrelevant, the reduction costs no second launch (measured the same
    kernel time and 0.012ms less end to end at 67MB), and the result is already the ONE value the
    caller needs -- which is what makes the fixed floor a single launch plus a single D2H.
    """
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    ok = offs < n
    raw = tl.load(iv_ptr + offs, mask=ok, other=0).to(tl.int64)
    w = _weights(offs, salt)
    tl.atomic_add(out_ptr, tl.sum((raw & m0) * w, axis=0))
    if NMASK > 1:
        tl.atomic_add(out_ptr + 1, tl.sum((raw & m1) * w, axis=0))
    if NMASK > 2:
        tl.atomic_add(out_ptr + 2, tl.sum((raw & m2) * w, axis=0))
    if NMASK > 3:
        tl.atomic_add(out_ptr + 3, tl.sum((raw & m3) * w, axis=0))
    if NMASK > 4:
        tl.atomic_add(out_ptr + 4, tl.sum((raw & m4) * w, axis=0))
    if NMASK > 5:
        tl.atomic_add(out_ptr + 5, tl.sum((raw & m5) * w, axis=0))
    if NMASK > 6:
        tl.atomic_add(out_ptr + 6, tl.sum((raw & m6) * w, axis=0))
    if NMASK > 7:
        tl.atomic_add(out_ptr + 7, tl.sum((raw & m7) * w, axis=0))


@triton.jit
def _segment_kernel(iv_ptr, out_ptr, n, seg_numel, seed, and_mask, BLOCK: tl.constexpr):
    """One program per (segment, block); every segment digest in a single launch."""
    seg = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1).to(tl.int64)
    base = seg * seg_numel
    offs = blk * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    ok = (offs < seg_numel) & (base + offs < n)
    raw = tl.load(iv_ptr + base + offs, mask=ok, other=0).to(tl.int64) & and_mask
    salt = _mix64(seed ^ _mix64(base + 1)) ^ -7046029254386353131
    tl.atomic_add(out_ptr + seg, tl.sum(raw * _weights(offs, salt), axis=0))


# -- zeroed output arena ----------------------------------------------------------------------
#
# The kernels accumulate, so their output must start at zero, and a `torch.zeros` per digest is
# another launch on the critical path. Instead: one zeroed buffer per device, handed out with a
# bump cursor and re-zeroed in one shot when it wraps. A slot is always read (the caller's D2H)
# before the cursor can wrap back onto it, so a wrap can never clear a live slot.

_ARENA_SLOTS = 1 << 13
_arenas: dict = {}
_arena_lock = threading.Lock()


def _slots(device: torch.device, count: int) -> torch.Tensor:
    key = (device.type, device.index)
    with _arena_lock:
        ent = _arenas.get(key)
        if ent is None or ent[0].numel() < count:
            ent = [torch.zeros(max(_ARENA_SLOTS, count), dtype=torch.int64, device=device), 0]
            _arenas[key] = ent
        buf, cursor = ent
        if cursor + count > buf.numel():
            buf.zero_()
            cursor = 0
        ent[1] = cursor + count
        return buf[cursor : cursor + count]


def _reset_arenas() -> None:
    with _arena_lock:
        _arenas.clear()


def available(device: torch.device) -> bool:
    return device.type == "cuda"


def weighted_sums(iv: torch.Tensor, masks: Sequence[int], salt: int) -> List[int]:
    """``[sum_i (x_i & mask_j) * w_i mod 2**64 for each mask]`` -- one launch, one sync."""
    nm = len(masks)
    if not 1 <= nm <= _MAX_MASKS:
        raise ValueError(f"weighted_sums: {nm} masks (need 1..{_MAX_MASKS})")
    n = iv.numel()
    out = _slots(iv.device, nm)
    args = list(masks) + [0] * (_MAX_MASKS - nm)
    grid = (max(1, triton.cdiv(n, BLOCK)),)
    _digest_kernel[grid](iv, out, n, salt, *args, NMASK=nm, BLOCK=BLOCK, num_warps=NUM_WARPS)
    return out.tolist()


def segment_sums(iv: torch.Tensor, nseg: int, seg_numel: int, mask: int, seed: int) -> List[int]:
    """Every segment's weighted sum in one launch; segment i is keyed by salt(seed, i*seg_numel)."""
    out = _slots(iv.device, nseg)
    grid = (nseg, max(1, triton.cdiv(seg_numel, BLOCK)))
    _segment_kernel[grid](iv, out, iv.numel(), seg_numel, seed, mask, BLOCK=BLOCK, num_warps=NUM_WARPS)
    return out.tolist()


def preload(device: Optional[torch.device] = None) -> None:
    """Compile both kernels now, so the first traced region does not pay JIT time mid-forward."""
    dev = device or torch.device("cuda")
    probe = torch.zeros(BLOCK + 1, dtype=torch.int16, device=dev)
    for nm in range(1, _MAX_MASKS + 1):
        weighted_sums(probe, [-1] * nm, 0)
    segment_sums(probe, 2, BLOCK, -1, 0)
