"""Counting sort replacing ``torch.argsort(sorted_indices, stable=True)`` in the MoE combine.

``sorted_indices[p]`` is the token index of permuted row ``p``, and every token routes to exactly ``topk``
experts on this stack, so bucket offsets are just ``t * K`` and all that is left is the within-bucket rank
-- a running cumsum over ascending ``p``, which is what makes the result stable. The output is an integer
permutation identical to ``torch.argsort``, and above ``SKYRL_ISOEXEC_MOE_COMBINE_SORT_MAX_WORK`` the caller
falls back to ``torch.argsort`` anyway. Gated on ``SKYRL_ISOEXEC_MOE_COMBINE_SORT``, default OFF.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

# Pinned, never autotuned: a fixed config keeps the engine and the trainer from picking different tiles.
# They are genuinely free parameters -- the output is an integer permutation whose definition ("ascending
# p within a token") does not reference the tile, and the cross-chunk running `base` makes the result
# independent of how P is chunked.
_BLOCK_P = int(os.environ.get("SKYRL_ISOEXEC_MOE_COMBINE_SORT_BLOCK_P", "1024"))
_NUM_WARPS = int(os.environ.get("SKYRL_ISOEXEC_MOE_COMBINE_SORT_WARPS", "4"))
# T * padded_P above which we hand back to torch.argsort (see SHAPE GUARD above).
_MAX_WORK = int(os.environ.get("SKYRL_ISOEXEC_MOE_COMBINE_SORT_MAX_WORK", str(1 << 25)))


def combine_sort_enabled() -> bool:
    return os.environ.get("SKYRL_ISOEXEC_MOE_COMBINE_SORT", "0") == "1"


@triton.jit
def _stable_bucket_rows_kernel(
    sidx_ptr,  # [P] int, sidx[p] = token index of permuted row p
    out_ptr,  # [T, K] -- out[t, j] = the j-th smallest p with sidx[p] == t
    P,
    K: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    t = tl.program_id(0)
    base = 0
    # Runtime loop over P in fixed chunks. `base` carries the count of matches already emitted, so the
    # rank is ascending in p across the whole array -- that is the stability guarantee.
    for p0 in range(0, P, BLOCK_P):
        offs = p0 + tl.arange(0, BLOCK_P)
        inb = offs < P
        s = tl.load(sidx_ptr + offs, mask=inb, other=-1)
        hit = (s == t) & inb
        hi = hit.to(tl.int32)
        # exclusive prefix sum -> position of this match within the token's bucket
        rank = tl.cumsum(hi, axis=0) - hi + base
        tl.store(
            out_ptr + t.to(tl.int64) * K + rank.to(tl.int64),
            offs,
            mask=hit & (rank < K),
        )
        base += tl.sum(hi, axis=0)


def _can_fuse(num_tokens: int, n: int) -> bool:
    if num_tokens <= 0 or n <= 0 or n % num_tokens != 0:
        return False
    padded_p = triton.cdiv(n, _BLOCK_P) * _BLOCK_P
    return num_tokens * padded_p <= _MAX_WORK


def stable_combine_rows(sorted_indices: torch.Tensor, num_tokens: int, dtype=torch.int64):
    """``argsort(sorted_indices, stable=True).view(T, k)`` by counting sort, or ``None`` when the shape
    guard declines and the caller should fall back to ``torch.argsort`` (identical values).

    ``sorted_indices`` must be 1-D on CUDA. The returned tensor is ``[T, k]`` and contiguous.
    """
    n = int(sorted_indices.numel())
    if not (combine_sort_enabled() and sorted_indices.is_cuda and sorted_indices.dim() == 1):
        return None
    if not _can_fuse(num_tokens, n):
        return None
    k = n // num_tokens
    sidx = sorted_indices.contiguous()
    out = torch.empty(num_tokens, k, dtype=dtype, device=sidx.device)
    _stable_bucket_rows_kernel[(num_tokens,)](
        sidx,
        out,
        n,
        K=k,
        BLOCK_P=_BLOCK_P,
        num_warps=_NUM_WARPS,
    )
    return out


def stable_argsort_order(sorted_indices: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Drop-in for ``torch.argsort(sorted_indices, stable=True)`` on the combine's mapping.

    Falls back to ``torch.argsort`` whenever the fused path declines, so the caller never has to
    branch. Output dtype is int64, matching ``torch.argsort``.
    """
    rows = stable_combine_rows(sorted_indices, num_tokens, dtype=torch.int64)
    if rows is None:
        return torch.argsort(sorted_indices, stable=True)
    order = rows.view(-1)
    return order
