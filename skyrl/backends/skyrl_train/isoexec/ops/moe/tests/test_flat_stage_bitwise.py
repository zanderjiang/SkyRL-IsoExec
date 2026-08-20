"""``SKYRL_ISOEXEC_MOE_FLAT_STAGE`` must be a change of ADDRESS ARITHMETIC and nothing else.

The flag swaps the staged-tile buffer's two-tensor advanced indexing

    xp[tile_idx, row_idx] = x            out[tile_idx, row_idx]

for the equivalent 1-D form on the flattened first two axes

    xp_flat.index_copy_(0, tile_idx*cap + row_idx, x)     out_flat.index_select(0, ...)

because ``tile_idx * cap + row_idx`` is a bijection onto ``[n_tiles*cap]``. No floating-point
operation appears on either side, so the obligation is ``torch.equal`` -- on the FORWARD bytes and
on the BACKWARD bytes, since the two forms have different VJPs (advanced-index gather/scatter vs
``index_select`` / ``index_add_``) even though the values they carry are the same.

Runs on CPU (both kernels exist there) so it is part of the CPU suite; the perf claim it protects
was measured on GPU and lives in the private repo's trainer-halo analysis report.
"""

import pytest
import torch

from skyrl.backends.skyrl_train.isoexec.ops.moe.moe_batched_experts import (
    _flat_tile_offsets,
)


def _grid(rows: int, cap: int, e_local: int, device):
    """A tile grid shaped exactly like ``_batched_experts_forward``'s prefill/trainer branch."""
    counts = torch.full((e_local,), rows // e_local, dtype=torch.long, device=device)
    counts[-1] += rows - int(counts.sum())
    cu = torch.zeros(e_local + 1, dtype=torch.long, device=device)
    cu[1:] = counts.cumsum(0)
    tile_cu = torch.zeros(e_local + 1, dtype=torch.long, device=device)
    tile_cu[1:] = ((counts + cap - 1) // cap).cumsum(0)
    n_tiles = int(tile_cu[-1])
    idx = torch.arange(rows, device=device)
    tok_expert = torch.searchsorted(cu[1:], idx, right=True).clamp_(max=e_local - 1)
    off = idx - cu[tok_expert]
    return tile_cu[tok_expert] + off // cap, off % cap, n_tiles


@pytest.mark.parametrize("rows,cap,e_local,h", [(1024, 128, 8, 64), (777, 128, 8, 32), (37, 16, 4, 8)])
def test_flat_offsets_are_a_bijection(rows, cap, e_local, h):
    tile_idx, row_idx, n_tiles = _grid(rows, cap, e_local, "cpu")
    flat = _flat_tile_offsets(tile_idx, row_idx, cap)
    assert int(flat.max()) < n_tiles * cap
    assert int(flat.min()) >= 0
    # distinct destinations: a duplicate would make index_copy_ order-dependent where index_put_
    # (accumulate=False) is not, which is the one way this swap could stop being bitwise.
    assert len(torch.unique(flat)) == rows


@pytest.mark.parametrize("rows,cap,e_local,h", [(1024, 128, 8, 64), (777, 128, 8, 32), (37, 16, 4, 8)])
def test_stage_and_unstage_are_bitwise_forward_and_backward(rows, cap, e_local, h):
    torch.manual_seed(0)
    dev = "cpu"
    tile_idx, row_idx, n_tiles = _grid(rows, cap, e_local, dev)
    flat = _flat_tile_offsets(tile_idx, row_idx, cap)

    x0 = torch.randn(rows, h, dtype=torch.float32)
    probs0 = torch.rand(rows, dtype=torch.float32)
    # a per-slot weight so the backward is not a constant-ones gradient (which would hide an
    # ordering bug in the VJP scatter)
    wsel = torch.randn(rows, h)

    def run(flat_path: bool):
        x = x0.clone().requires_grad_(True)
        probs = probs0.clone().requires_grad_(True)
        if flat_path:
            xp_flat = x.new_zeros(n_tiles * cap, h)
            xp_flat.index_copy_(0, flat, x)
            xp = xp_flat.view(n_tiles, cap, h)
            pp_flat = probs.new_zeros(n_tiles * cap)
            pp_flat.index_copy_(0, flat, probs)
            pp = pp_flat.view(n_tiles, cap)
        else:
            xp = x.new_zeros(n_tiles, cap, h)
            xp[tile_idx, row_idx] = x
            pp = probs.new_zeros(n_tiles, cap)
            pp[tile_idx, row_idx] = probs
        # stand-in for the expert GEMM + epilogue: anything that keeps every staged slot live
        out = xp * pp.unsqueeze(-1)
        if flat_path:
            local = out.reshape(n_tiles * cap, h).index_select(0, flat)
        else:
            local = out[tile_idx, row_idx]
        (local * wsel).sum().backward()
        return xp.detach().clone(), pp.detach().clone(), local.detach().clone(), x.grad.clone(), probs.grad.clone()

    ref = run(False)
    got = run(True)
    for name, a, b in zip(("xp", "pp", "output_local", "grad_x", "grad_probs"), ref, got):
        assert torch.equal(a, b), f"{name} is not bitwise-equal between the two addressing forms"


def test_padding_slots_stay_zero():
    """Slots no token maps to must hold zeros on BOTH paths -- the expert GEMM reads them."""
    tile_idx, row_idx, n_tiles = _grid(37, 16, 4, "cpu")
    flat = _flat_tile_offsets(tile_idx, row_idx, 16)
    x = torch.randn(37, 8)
    a = x.new_zeros(n_tiles, 16, 8)
    a[tile_idx, row_idx] = x
    b = x.new_zeros(n_tiles * 16, 8)
    b.index_copy_(0, flat, x)
    assert torch.equal(a, b.view(n_tiles, 16, 8))
    assert n_tiles * 16 > 37, "fixture must actually contain padding slots"
