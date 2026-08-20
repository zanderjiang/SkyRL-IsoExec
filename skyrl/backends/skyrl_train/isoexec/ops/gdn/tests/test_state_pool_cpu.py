"""CPU guards for the GDN chunk-synced state pool (`_state_pool`) and the op-count instrument.

WHAT IS BEING GUARDED. The pool build used to be three full-size passes --

    h_flat = h.reshape(...).to(torch.float32)      # a second full-size fp32 buffer
    pool   = torch.zeros(n_chunks + 1, ...)        # a memset of rows that are all overwritten
    pool[1 : n_chunks + 1] = h_flat[:n_chunks]

-- and is now one ``empty`` + a one-row ``zero_`` + one ``copy_`` that performs the widening cast
itself. The claim is BITWISE-NEUTRAL BY CONSTRUCTION on a gate-critical forward, so the tests do
not check a tolerance: they check ``torch.equal`` against the old expression, on bf16 payloads
chosen to include the values a rounding bug would move (signed zeros, denormals, inf, NaN).

They also check the SAVING, in the unit in which it is real. This is not an op-count win -- at the
dispatcher it is +1 op -- it is a device-traffic and peak-memory win, so the test measures elements
written, not calls. A claim that is not measured in its own unit is a banner, and this campaign has
shipped five of those already.
"""

from __future__ import annotations

import torch

from skyrl.backends.skyrl_train.isoexec.ops.gdn.gdn_chunk_synced import _state_pool
from _op_census import count_ops  # vendored copy of the private repo's tools/op_census.py


def _reference(h: torch.Tensor, n_chunks: int, device) -> torch.Tensor:
    """The expression `_state_pool` replaced, transcribed verbatim."""
    h_flat = h.reshape(-1, *h.shape[-3:]).to(torch.float32)
    pool = torch.zeros(n_chunks + 1, *h_flat.shape[-3:], dtype=torch.float32, device=device)
    pool[1 : n_chunks + 1] = h_flat[:n_chunks]
    return pool


def _h(n_rows: int = 7, hv: int = 2, v: int = 4, k: int = 3, dtype=torch.bfloat16) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(1, n_rows, hv, v, k, dtype=torch.float32).to(dtype)


def test_pool_is_bit_identical_to_the_old_expression():
    h = _h()
    for n_chunks in (1, 3, 7):
        got = _state_pool(h, n_chunks, "cpu")
        want = _reference(h, n_chunks, "cpu")
        assert got.shape == want.shape and got.dtype == want.dtype
        assert torch.equal(got, want), f"pool differs at n_chunks={n_chunks}"


def test_pool_is_bit_identical_on_the_payloads_a_cast_bug_would_move():
    """Signed zeros, denormals, inf and NaN survive a widening bf16->fp32 cast unchanged."""
    vals = torch.tensor(
        [0.0, -0.0, float("inf"), float("-inf"), float("nan"), 1.0, -1.0, 2.0**-133, 3.389e38],
        dtype=torch.float32,
    ).to(torch.bfloat16)
    h = vals.reshape(1, 3, 1, 1, 3).contiguous()
    got = _state_pool(h, 3, "cpu")
    want = _reference(h, 3, "cpu")
    # compare as BITS so -0.0 != 0.0 and NaN payloads count
    assert torch.equal(got.view(torch.int32), want.view(torch.int32))


def test_row_zero_is_the_null_row_and_every_other_row_is_written():
    h = _h(n_rows=5)
    pool = _state_pool(h, 5, "cpu")
    assert torch.equal(pool[0], torch.zeros_like(pool[0]))
    flat = h.reshape(-1, *h.shape[-3:])
    for r in range(5):
        assert torch.equal(pool[r + 1], flat[r].to(torch.float32))


def test_fp32_input_is_handled_the_same_way():
    h = _h(dtype=torch.float32)
    assert torch.equal(_state_pool(h, 4, "cpu"), _reference(h, 4, "cpu"))


def test_partial_pool_takes_only_the_first_n_chunks_rows():
    """`h` carries NT_pad rows; only the first n_chunks are boundary states."""
    h = _h(n_rows=9)
    assert torch.equal(_state_pool(h, 4, "cpu"), _reference(h, 4, "cpu"))


def test_the_two_full_size_passes_are_gone_and_the_memset_is_one_row():
    """The claim, measured rather than asserted -- and stated in the unit that is actually true.

    At the DISPATCHER this change is op-neutral (+1: a ``select`` + ``zero_`` for row 0 replaces the
    single ``zeros``). What it removes is DEVICE TRAFFIC and peak memory: the full-size ``_to_copy``
    that materialised a second fp32 copy of the whole pool, and the memset of rows that are then
    all overwritten. So the assertions are on elements touched, not on a count.
    """
    h = _h(n_rows=6)
    with count_ops() as before:
        _reference(h, 6, "cpu")
    with count_ops() as after:
        _state_pool(h, 6, "cpu")

    row = h[0, 0].numel()
    pool_elems = 7 * row
    # the separate full-size widening cast is gone entirely
    assert any("_to_copy" in k for k in before.by_op()), before.report()
    assert not any("_to_copy" in k for k in after.by_op()), after.report()
    # the memset went from the whole pool to exactly one row
    before_zero = sum(v for k, v in before.elems.items() if "zero" in k)
    after_zero = sum(v for k, v in after.elems.items() if "zero" in k)
    assert before_zero >= pool_elems, (before_zero, pool_elems)
    assert after_zero == row, (after_zero, row)

    # elements actually WRITTEN (views and empties move nothing) fall by the two removed passes
    def written(c):
        return sum(v for k, v in c.elems.items() if any(w in k for w in ("copy_", "_to_copy", "zeros", "zero_")))

    assert written(before) - written(after) >= 2 * row * 6, (written(before), written(after))


def test_op_census_counts_and_attributes():
    x = torch.ones(4)
    with count_ops() as c:
        (x + x).mul(2.0)
    assert c.total() >= 2
    assert any("add" in k for k in c.by_op())
    assert any("test_state_pool_cpu.py:" in s for s in c.by_site())
    assert "ATen dispatches" in c.report()


def test_op_census_does_not_change_results():
    torch.manual_seed(1)
    x = torch.randn(16, 8)
    plain = (x @ x.t()).sin()
    with count_ops():
        traced = (x @ x.t()).sin()
    assert torch.equal(plain, traced)
