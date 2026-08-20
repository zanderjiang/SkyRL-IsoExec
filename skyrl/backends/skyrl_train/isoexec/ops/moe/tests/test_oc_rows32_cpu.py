"""CPU proofs for WAVE-11 rows32: the owner-combine ``my_rows`` as an int32 slice view.

The whole change is integer plumbing -- the same permutation the counting-sort kernel always
computed, minus an int64 materialization and a pad that is vacuous whenever T divides by world.
There is no float anywhere; the obligation is ORDER (the permutation must be element-for-element
what ``torch.argsort(stable=True)`` returns) and EQUALITY of the two construction paths.
"""

from __future__ import annotations

import pytest
import torch

from skyrl.backends.skyrl_train.isoexec.ops.moe import moe_pik_combine_owner as OC


def _old_my_rows(rows: torch.Tensor, T: int, k: int, world: int, rank: int) -> torch.Tensor:
    """The landed construction, verbatim from _owner_combine's general path."""
    s = -(-T // world)
    Tp = s * world
    rows_p = rows.new_zeros(Tp, k)
    rows_p[:T] = rows
    return rows_p[rank * s : (rank + 1) * s].contiguous().to(torch.int32)


def _new_my_rows(rows32: torch.Tensor, T: int, world: int, rank: int) -> torch.Tensor:
    """The rows32 fast path: a contiguous slice VIEW, valid only at T % world == 0."""
    s = T // world
    return rows32[rank * s : (rank + 1) * s]


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("T", [8, 64, 512, 320])
@pytest.mark.parametrize("k", [1, 2, 8])
def test_slice_view_equals_the_landed_pad_slice_cast(world, T, k):
    if T % world != 0:
        pytest.skip("fast path is gated on divisibility; the general path handles the rest")
    g = torch.Generator().manual_seed(world * 1000 + T * 10 + k)
    order = torch.randperm(T * k, generator=g)
    rows64 = order.view(T, k)
    rows32 = rows64.to(torch.int32)
    for rank in range(world):
        old = _old_my_rows(rows64, T, k, world, rank)
        new = _new_my_rows(rows32, T, world, rank)
        assert new.dtype == torch.int32 and new.is_contiguous()
        assert torch.equal(old, new), (world, T, k, rank)


@pytest.mark.parametrize("world", [8])
@pytest.mark.parametrize("T", [7, 321, 511])
def test_indivisible_T_takes_the_landed_path(world, T):
    """The fast path must NOT be offered at indivisible T -- _owner_combine's guard is
    ``rows.dtype == int32 and T % world == 0``; here we pin the general path's own output so the
    guard's fallback is known-good for every dtype it can receive."""
    k = 2
    g = torch.Generator().manual_seed(T)
    order = torch.randperm(T * k, generator=g)
    for dtype in (torch.int64, torch.int32):
        rows = order.view(T, k).to(dtype)
        for rank in range(world):
            got = _old_my_rows(rows, T, k, world, rank)
            s = -(-T // world)
            assert got.shape == (s, k) and got.dtype == torch.int32
            # padded tokens must read row 0 (they are sliced off after the exchange)
            flat_start = rank * s
            for i in range(s):
                t = flat_start + i
                expect = rows[t].to(torch.int32) if t < T else torch.zeros(k, dtype=torch.int32)
                assert torch.equal(got[i], expect), (T, rank, i)


def test_build_rows_falls_back_off_gpu(monkeypatch):
    """stable_combine_rows declines CPU tensors, so _build_rows must return the landed int64 form
    whatever the flag says -- the fast path can never be wrongly taken."""
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_OC_ROWS32", "1")
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_COMBINE_SORT", "1")
    T, k = 12, 2
    g = torch.Generator().manual_seed(3)
    sidx = torch.repeat_interleave(torch.arange(T), k)[torch.randperm(T * k, generator=g)]
    rows = OC._build_rows(sidx, T, k)
    assert rows.shape == (T, k) and rows.dtype == torch.int64
    assert torch.equal(rows.view(-1), torch.argsort(sidx, stable=True))


def test_default_is_on_and_zero_is_the_escape_hatch(monkeypatch):
    # Default flipped to ON 2026-08-14 (battery green at world 2/4/8, live A/B -0.17 ms/step);
    # =0 remains the bisection escape hatch back to the landed path.
    monkeypatch.delenv("SKYRL_ISOEXEC_MOE_OC_ROWS32", raising=False)
    assert OC._rows32_enabled()
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_OC_ROWS32", "0")
    assert not OC._rows32_enabled()
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_OC_ROWS32", "1")
    assert OC._rows32_enabled()
