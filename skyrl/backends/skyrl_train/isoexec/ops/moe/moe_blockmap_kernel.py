"""Build the MoE expert block map in one Triton launch instead of a dozen small torch ops.

Produces the ``sorted_token_ids`` / ``expert_ids`` / ``num_tokens_post_padded`` metadata both expert GEMMs
consume. Every program recomputes the ``E``-wide prefix sums in registers, so all programs are independent
and the launch count is 1. The output is pure integer metadata; the grid is sized from the host-known bound
``ceil(T / BLOCK_M) + E`` and nothing is read back to the host, so CUDA-graph capture is unaffected. Gated
on ``SKYRL_ISOEXEC_MOE_FUSED_BLOCKMAP``, default OFF.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

_ENV = "SKYRL_ISOEXEC_MOE_FUSED_BLOCKMAP"


def fused_blockmap_enabled() -> bool:
    """Read at call time so the flag can be flipped between forwards. Default OFF."""
    return os.environ.get(_ENV, "0") == "1"


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


@triton.jit
def _block_map_kernel(
    counts_ptr,  # [E] int64, rows per expert
    sti_ptr,  # [max_blocks * BLOCK_M] int32 out
    eids_ptr,  # [max_blocks] int32 out
    ntpp_ptr,  # [1] int32 out
    T,
    E,
    max_blocks,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """One program per block of the map, recomputing the E-wide prefix sums in registers.

    ``cu[e]`` is the first ROW of expert e (exclusive cumsum of counts) and ``bcu[e]`` its first BLOCK
    (exclusive cumsum of ``ceil(counts / BLOCK_M)``); block -> expert is "the last expert whose first
    block is <= b", which is what the torch version's ``searchsorted(bcu, blk, right=True) - 1`` gives.
    """
    pid = tl.program_id(0)

    e = tl.arange(0, BLOCK_E)
    live = e < E
    counts = tl.load(counts_ptr + e, mask=live, other=0).to(tl.int32)

    nb = (counts + BLOCK_M - 1) // BLOCK_M  # blocks per expert
    bcu = tl.cumsum(nb, axis=0) - nb  # first block of each expert
    cu = tl.cumsum(counts, axis=0) - counts  # first row of each expert

    if pid == 0:
        # the padded row count the GEMM early-returns against. Device scalar: no host readback.
        tl.store(ntpp_ptr, tl.sum(nb, axis=0) * BLOCK_M)

    if pid >= max_blocks:
        return

    # which expert owns this block: the last e with bcu[e] <= pid, among live experts. Dead lanes are
    # excluded rather than clamped, so an empty tail cannot claim the block.
    owns = live & (bcu <= pid)
    my_e = tl.max(tl.where(owns, e, 0), axis=0)
    # ...and clamp into range exactly as the torch version's `.clamp_(0, E - 1)` does, for blocks past the
    # last expert's last block (pure padding, masked out below anyway).
    my_e = tl.minimum(my_e, E - 1)

    my_bcu = tl.sum(tl.where(e == my_e, bcu, 0), axis=0)
    my_cu = tl.sum(tl.where(e == my_e, cu, 0), axis=0)
    my_cnt = tl.sum(tl.where(e == my_e, counts, 0), axis=0)

    tl.store(eids_ptr + pid, my_e)

    j = pid - my_bcu  # this block's index within its expert
    slot = tl.arange(0, BLOCK_M)
    rows = my_cu + j * BLOCK_M + slot
    # padding slots get T, which is >= num_valid_tokens, so the GEMM masks them off.
    valid = rows < (my_cu + my_cnt)
    tl.store(sti_ptr + pid * BLOCK_M + slot, tl.where(valid, rows, T))


@torch.no_grad()
def fused_block_map(counts: torch.Tensor, T: int, E: int, block_m: int):
    """Drop-in for ``moe_fused_experts._block_map``, returning the identical four values.

    ``(sorted_token_ids [max_blocks*block_m] int32, expert_ids [max_blocks] int32,
       num_tokens_post_padded [1] int32, max_blocks)``
    """
    assert HAVE_TRITON, "moe_blockmap_kernel requires triton"
    dev = counts.device
    max_blocks = (T + block_m - 1) // block_m + E

    sti = torch.empty(max_blocks * block_m, dtype=torch.int32, device=dev)
    eids = torch.empty(max_blocks, dtype=torch.int32, device=dev)
    ntpp = torch.empty(1, dtype=torch.int32, device=dev)

    _block_map_kernel[(max_blocks,)](
        counts,
        sti,
        eids,
        ntpp,
        T,
        E,
        max_blocks,
        BLOCK_M=block_m,
        BLOCK_E=_next_pow2(E),
        num_warps=4,
    )
    return sti, eids, ntpp, max_blocks
