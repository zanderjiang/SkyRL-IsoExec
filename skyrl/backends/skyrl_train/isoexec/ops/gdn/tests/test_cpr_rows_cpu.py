"""CPU tests for the fused slot->row resolution (``gdn_cpr_rows`` + ``RecurrentGDN._rows_pair``).

The fused kernel must be a launch-structure change only, so ``_rows_legacy`` (a verbatim copy of
the pre-change body) is the oracle. Triton is GPU-only: CPU covers the seam, the decline predicates
and the transcribed arithmetic; the bitwise kernel comparison is skipped without a device.
"""

import pytest

torch = pytest.importorskip("torch")

from skyrl.backends.skyrl_train.isoexec.ops.gdn.gdn_recurrent_state import (  # noqa: E402
    RecurrentGDN,
)

CUDA = torch.cuda.is_available()


# the oracle and a pool carrying only the slot map
def _rows_legacy(slot2row: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """Verbatim copy of ``RecurrentGDN._rows``'s pre-change non-native body."""
    safe = slots.long().clamp_(0, slot2row.numel() - 1)
    return slot2row[safe]


def make_pool(map_size: int = 4096, rows: int = 64, device: str = "cpu") -> RecurrentGDN:
    """A ``RecurrentGDN`` carrying only the fields ``_rows_pair`` touches (``__init__`` would need
    a live parallel state)."""
    rg = RecurrentGDN.__new__(RecurrentGDN)
    rg._native = False
    g = torch.Generator().manual_seed(1234)
    rg.slot2row = torch.randint(0, rows, (map_size,), dtype=torch.int64, generator=g).to(device)
    return rg


def adversarial_slots(map_size: int, device: str = "cpu") -> list[torch.Tensor]:
    """Every slot vector a captured replay can actually present."""
    return [
        torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device),
        torch.tensor([-1, -1, -1, -1], dtype=torch.int32, device=device),  # NULL_BLOCK_ID padding
        torch.tensor([map_size - 1, map_size, map_size + 7, 1 << 20], dtype=torch.int32, device=device),
        torch.tensor([5, 5, 5, 5], dtype=torch.int32, device=device),  # duplicates -> one row
        torch.tensor([-3, 0, map_size, 17, -1, 9], dtype=torch.int32, device=device),  # mixed
        torch.arange(512, dtype=torch.int32, device=device),  # the production decode width
        torch.tensor([], dtype=torch.int32, device=device),  # an empty decode half
        torch.tensor([0, 1], dtype=torch.int64, device=device),  # already int64 (native-state form)
    ]


# the seam: _rows_pair must equal (legacy rows, its int32 cast) whatever path it takes
def test_rows_pair_matches_legacy_on_cpu():
    """No CUDA -> the fused path is not even attempted, and the answer is the legacy one."""
    rg = make_pool()
    for slots in adversarial_slots(rg.slot2row.numel()):
        rows, rows32 = rg._rows_pair(slots)
        want = _rows_legacy(rg.slot2row, slots)
        assert rows.dtype is torch.int64 and rows32.dtype is torch.int32
        assert torch.equal(rows, want)
        assert torch.equal(rows32, want.to(torch.int32))


def test_rows_pair_honours_the_kill_switch(monkeypatch):
    """``SKYRL_ISOEXEC_GDN_CPR_FUSED_ROWS=0`` takes the ATen chain, on any device."""
    from skyrl.backends.skyrl_train.isoexec.ops.gdn import gdn_cpr_rows

    def _rows_stats():
        # inlined: the accessor was stripped; the counters survive
        return gdn_cpr_rows._served, gdn_cpr_rows._declined, gdn_cpr_rows._decline_reason

    monkeypatch.setenv("SKYRL_ISOEXEC_GDN_CPR_FUSED_ROWS", "0")
    assert not gdn_cpr_rows.fused_rows_enabled()
    before = _rows_stats()[0]
    rg = make_pool(device="cuda" if CUDA else "cpu")
    slots = torch.arange(64, dtype=torch.int32, device=rg.slot2row.device)
    rows, rows32 = rg._rows_pair(slots)
    assert torch.equal(rows, _rows_legacy(rg.slot2row, slots))
    assert _rows_stats()[0] == before, "kill switch still served a fused call"


def test_native_state_keeps_the_aten_form():
    """The ``_native`` clamp is a different function, so the fused kernel must not be reached."""
    rg = make_pool()
    rg._native = True
    rg.ssm_state = torch.zeros(16, 1)
    slots = torch.tensor([-1, 0, 15, 16, 999], dtype=torch.int32)
    rows, rows32 = rg._rows_pair(slots)
    s = slots.long()
    want = torch.where(s < 16, s, torch.zeros_like(s)).clamp_(min=0)
    assert torch.equal(rows, want)
    assert torch.equal(rows32, want.to(torch.int32))


# the decline predicates: anything that is not the production shape falls back
@pytest.mark.parametrize(
    "slots, smap, why",
    [
        (torch.zeros(4, 2, dtype=torch.int32), torch.zeros(8, dtype=torch.int64), "2-D slots"),
        (torch.zeros(4, dtype=torch.int32), torch.zeros(8, 2, dtype=torch.int64), "2-D map"),
        (torch.zeros(4, dtype=torch.int32), torch.zeros(8, dtype=torch.int32), "int32 map"),
    ],
)
def test_declines_rather_than_guesses(slots, smap, why):
    from skyrl.backends.skyrl_train.isoexec.ops.gdn.gdn_cpr_rows import cpr_resolve_rows

    assert cpr_resolve_rows(slots, smap) is None, why


def test_decline_is_counted_and_reported(capsys):
    from skyrl.backends.skyrl_train.isoexec.ops.gdn import gdn_cpr_rows

    def _rows_stats():
        # inlined: the accessor was stripped; the counters survive
        return gdn_cpr_rows._served, gdn_cpr_rows._declined, gdn_cpr_rows._decline_reason

    before = _rows_stats()[1]
    gdn_cpr_rows.cpr_resolve_rows(torch.zeros(4, dtype=torch.int32), torch.zeros(8, dtype=torch.int32))
    assert _rows_stats()[1] == before + 1
    assert "expected 1-D slots" in _rows_stats()[2]


# the arithmetic the kernel transcribes, checked without a device
def test_transcribed_arithmetic_equals_the_aten_chain():
    """The Triton body's ``min(max(s, 0), map_n - 1)`` equals ATen's ``clamp_(0, map_n - 1)``."""
    smap = torch.arange(97, dtype=torch.int64) * 3
    for slots in adversarial_slots(smap.numel()):
        s = slots.long()
        kernel_form = smap[torch.minimum(torch.maximum(s, torch.zeros_like(s)), torch.full_like(s, smap.numel() - 1))]
        assert torch.equal(kernel_form, _rows_legacy(smap, slots))


# the real kernel, when there is a device to run it on
@pytest.mark.skipif(not CUDA, reason="the fused rows kernel is Triton; needs a CUDA device")
def test_fused_kernel_is_bitwise_the_aten_chain():
    from skyrl.backends.skyrl_train.isoexec.ops.gdn.gdn_cpr_rows import cpr_resolve_rows

    rg = make_pool(device="cuda")
    for slots in adversarial_slots(rg.slot2row.numel(), device="cuda"):
        pair = cpr_resolve_rows(slots, rg.slot2row)
        assert pair is not None
        rows, rows32 = pair
        want = _rows_legacy(rg.slot2row, slots)
        assert torch.equal(rows, want)
        assert torch.equal(rows32, want.to(torch.int32))


@pytest.mark.skipif(not CUDA, reason="cuda graph capture needs a device")
def test_fused_kernel_captures_into_a_cuda_graph():
    """The whole point is that it lives inside the captured decode graph: no host work, no sync."""
    from skyrl.backends.skyrl_train.isoexec.ops.gdn.gdn_cpr_rows import cpr_resolve_rows

    rg = make_pool(device="cuda")
    slots = torch.arange(512, dtype=torch.int32, device="cuda")
    cpr_resolve_rows(slots, rg.slot2row)  # warm the Triton compile outside capture
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    holder = {}
    with torch.cuda.graph(g):
        holder["out"] = cpr_resolve_rows(slots, rg.slot2row)
    slots.copy_(torch.randint(-4, rg.slot2row.numel() + 4, (512,), device="cuda").to(torch.int32))
    g.replay()
    torch.cuda.synchronize()
    rows, rows32 = holder["out"]
    want = _rows_legacy(rg.slot2row, slots)
    assert torch.equal(rows, want)
    assert torch.equal(rows32, want.to(torch.int32))
