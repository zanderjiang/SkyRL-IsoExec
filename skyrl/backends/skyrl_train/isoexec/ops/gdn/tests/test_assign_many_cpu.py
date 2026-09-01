"""CPU tests for the batched prefill row claim (``RecurrentGDN._assign_many``).

The batching must be a throughput change only, so every test is an equivalence against
``_assign_legacy`` (a verbatim copy of the pre-change body). The trap it guards: deferring the
clock bump lets a naive batched argmin hand the same row to K slots -- silent state sharing.
"""

import random
from collections import OrderedDict

import pytest

torch = pytest.importorskip("torch")

from skyrl.backends.skyrl_train.isoexec.ops.gdn import (  # noqa: E402
    gdn_recurrent_state as _grs,  # noqa: E402
)
from skyrl.backends.skyrl_train.isoexec.ops.gdn.gdn_cpr_state import (  # noqa: E402
    CprGDN,
)


def host_bookkeeping_stats():
    # inlined: the public module keeps the counters but not this test accessor
    return dict(_grs._HOST_BOOKKEEPING_STATS)


from skyrl.backends.skyrl_train.isoexec.ops.gdn.gdn_recurrent_state import (  # noqa: E402
    RecurrentGDN,
)


# a pool with only the slot bookkeeping in it
def make_pool(capacity: int = 8, map_size: int = 4096) -> RecurrentGDN:
    """A ``RecurrentGDN`` carrying only the fields the row claim touches (``__init__`` would
    allocate the state arena and resolve kernel flags the bookkeeping never reads)."""
    p = object.__new__(RecurrentGDN)
    p._native = False
    p.capacity = capacity
    rows = capacity + 1  # row 0 is the null row, as in __init__
    p.slot2row = torch.zeros(map_size, dtype=torch.long)
    p.last_used = torch.zeros(rows, dtype=torch.long)
    p._clock = torch.zeros((), dtype=torch.long)
    p._slot2row = OrderedDict()
    p._free = list(range(1, rows))
    p._row2slot = {}
    p._free_set = set(p._free)
    p._slot_pending = {}
    return p


def _assign_legacy(p: RecurrentGDN, slot: int) -> int:
    """The pre-change ``_assign`` body, verbatim: the oracle every test compares against."""
    row = p._slot2row.pop(slot, None)
    if row is None:
        if p._free:
            row = p._free.pop()
        else:
            row = int(p.last_used[1:].argmin()) + 1  # row 0 is not evictable
            for _s, _r in list(p._slot2row.items()):
                if _r == row:
                    del p._slot2row[_s]
                    break
    p._slot2row[slot] = row
    if slot >= p.slot2row.numel():
        raise RuntimeError(f"[isoexec-gdn] state slot {slot} exceeds the slot map ({p.slot2row.numel()}).")
    p._set_slot(slot, row)
    p.flush_slot_map()
    p._clock += 1
    p.last_used[row] = p._clock
    return row


def snapshot(p: RecurrentGDN) -> dict:
    """Everything a caller or a later step can observe about the claim."""
    return {
        "slot2row_host": dict(p._slot2row),
        "slot2row_order": list(p._slot2row.items()),
        "free": list(p._free),
        "last_used": p.last_used.tolist(),
        "clock": int(p._clock),
        "device_map": p.slot2row.tolist(),
        "pending": dict(p._slot_pending),
    }


def run_both(batches, capacity=8, map_size=4096):
    """Drive the same batch sequence through the legacy per-slot path and the batched one."""
    old, new = make_pool(capacity, map_size), make_pool(capacity, map_size)
    rows_old, rows_new = [], []
    for batch in batches:
        rows_old.append([_assign_legacy(old, int(s)) for s in batch])
        rows_new.append(new._assign_many(batch))
    return (old, rows_old), (new, rows_new)


# equivalence with the per-slot code it replaces
def test_free_list_phase_is_identical():
    """Before anything is evicted the claim is pure free-list order, and it must not drift."""
    (old, ro), (new, rn) = run_both([[10, 11, 12], [13, 14]])
    assert ro == rn
    assert snapshot(old) == snapshot(new)


def test_repeated_slot_gets_its_row_back():
    """A chunked-prefill continuation re-presents its slot; ``_assign`` hands the SAME row back."""
    (old, ro), (new, rn) = run_both([[10, 11], [11, 10, 12]])
    assert ro == rn
    assert rn[1][0] == rn[0][1] and rn[1][1] == rn[0][0]
    assert snapshot(old) == snapshot(new)


def test_eviction_phase_is_identical():
    """Past the free list every claim evicts, and the victim choice must match slot for slot."""
    batches = [[100 + i] for i in range(8)] + [[200, 201, 202], [203], [204, 205]]
    (old, ro), (new, rn) = run_both(batches)
    assert ro == rn
    assert snapshot(old) == snapshot(new)


@pytest.mark.parametrize("seed", range(12))
def test_randomised_workloads_are_identical(seed):
    """Mixed fresh / re-presented / evicting batches of ragged width, against the oracle."""
    rng = random.Random(seed)
    live: list[int] = []
    batches = []
    next_slot = 1000
    for _ in range(14):
        n = rng.randint(1, 6)
        batch = []
        for _ in range(n):
            if live and rng.random() < 0.35:
                s = rng.choice(live)
                if s in batch:  # a slot may not appear twice in one scheduler batch
                    continue
            else:
                s, next_slot = next_slot, next_slot + 1
                live.append(s)
            batch.append(s)
        if batch:
            batches.append(batch)
    (old, ro), (new, rn) = run_both(batches, capacity=6)
    assert ro == rn
    assert snapshot(old) == snapshot(new)


# the k-smallest trap
def _naive_batched_rows(p: RecurrentGDN, slots):
    """The wrong batched eviction -- argmin re-run K times over one stale ``last_used`` snapshot --
    kept executable so the correct test beside it is demonstrably non-vacuous."""
    snap = p.last_used.clone()
    rows = []
    for s in slots:
        row = p._slot2row.get(int(s))
        if row is None:
            row = int(snap[1:].argmin()) + 1  # never bumped -> the same answer every time
        rows.append(row)
    return rows


def _fill(p: RecurrentGDN, n: int):
    """Claim ``n`` rows so the free list is empty and ``last_used`` is strictly ordered."""
    for i in range(n):
        p._assign_many([500 + i])


def test_naive_repeated_argmin_is_the_bug():
    """The stale-snapshot form hands the same row to every slot in the batch."""
    p = make_pool(capacity=4)
    _fill(p, 4)
    naive = _naive_batched_rows(p, [900, 901, 902])
    assert len(set(naive)) == 1, f"the naive form is supposed to collide, got {naive}"


def test_two_evictions_in_one_batch_take_the_k_smallest():
    """K evictions in one batch take the K least-recently-used rows, all distinct."""
    p = make_pool(capacity=4)
    _fill(p, 4)  # rows claimed in order -> last_used is strictly increasing over rows 1..4
    lru_order = sorted(range(1, 5), key=lambda r: (p.last_used[r].item(), r))
    rows = p._assign_many([900, 901, 902])
    assert len(set(rows)) == 3, f"a batch of three claims must occupy three rows, got {rows}"
    assert rows == lru_order[:3]


def test_a_four_eviction_batch_matches_the_oracle():
    """A batch that evicts four times over still matches the oracle."""
    (old, ro), (new, rn) = run_both([[500 + i] for i in range(6)] + [[900, 901, 902, 903]], capacity=6)
    assert ro == rn
    assert len(set(rn[-1])) == 4, f"four claims must occupy four rows, got {rn[-1]}"
    assert snapshot(old) == snapshot(new)


def test_ties_break_on_the_lowest_row_like_argmin():
    """A fresh pool's ``last_used`` is all zeros: ``argmin`` takes the FIRST minimum, so must we."""
    p = make_pool(capacity=5)
    p._free.clear()  # no free rows, every last_used still 0 -> a pure tie
    p._free_set.clear()
    rows = p._assign_many([700, 701, 702])
    assert rows == [1, 2, 3]


def test_a_row_claimed_from_the_free_list_is_not_evicted_later_in_the_same_batch():
    """A row taken from the free list has stale ``last_used``, so it must still be excluded from
    the same batch's eviction candidates."""
    p = make_pool(capacity=4)
    _fill(p, 3)  # rows 4, 3, 2 claimed (free list pops from the end); row 1 still free
    assert p._free == [1]
    rows = p._assign_many([900, 901, 902])
    assert len(set(rows)) == 3, f"rows collided: {rows}"
    assert rows[0] == 1  # the free row, then two evictions


def test_a_re_presented_slot_is_not_evicted_later_in_the_same_batch():
    """Same exclusion, entered through the map-hit branch rather than the free list."""
    p = make_pool(capacity=3)
    _fill(p, 3)
    oldest = min(p._slot2row.values(), key=lambda r: p.last_used[r].item())
    keeper = next(s for s, r in p._slot2row.items() if r == oldest)
    rows = p._assign_many([keeper, 900, 901])
    assert rows[0] == oldest
    assert len(set(rows)) == 3, f"the re-presented row was evicted inside its own batch: {rows}"


def test_oversubscribed_batch_raises_instead_of_double_claiming():
    """More prompts than rows is unserviceable; the old code would have silently doubled up."""
    p = make_pool(capacity=3)
    with pytest.raises(RuntimeError, match="evictable state rows"):
        p._assign_many([900, 901, 902, 903])


# what the hoist was for: the device traffic
def test_one_flush_and_one_clock_scatter_per_batch(monkeypatch):
    """Device work is per batch, not per slot."""
    p = make_pool(capacity=16)
    flushes = []
    real = RecurrentGDN.flush_slot_map
    monkeypatch.setattr(
        RecurrentGDN,
        "flush_slot_map",
        lambda self: (flushes.append(len(self._slot_pending)), real(self))[1],
    )
    rows = p._assign_many(list(range(300, 312)))
    assert len(rows) == 12
    assert flushes == [12], f"expected ONE flush carrying all 12 edits, got {flushes}"


def test_flush_moves_keys_and_values_in_one_transfer():
    """``flush_slot_map`` stages one buffer, not two, and the map still reads back exactly."""
    p = make_pool(capacity=8)
    rows = p._assign_many([31, 17, 4])
    assert p._slot_pending == {}
    for s, r in zip([31, 17, 4], rows):
        assert int(p.slot2row[s]) == r


def test_clock_advances_one_tick_per_slot_in_order():
    """A later eviction ranks this batch's rows by WHEN in the batch they were claimed."""
    p = make_pool(capacity=8)
    rows = p._assign_many([21, 22, 23])
    assert int(p._clock) == 3
    assert [int(p.last_used[r]) for r in rows] == [1, 2, 3]


def test_duplicate_slot_inside_one_batch_keeps_the_last_tick():
    """A duplicate ``index_put_`` is nondeterministic in torch, so the dedup is pinned."""
    p = make_pool(capacity=8)
    rows = p._assign_many([21, 21])
    assert rows[0] == rows[1]
    assert int(p.last_used[rows[0]]) == 2
    assert int(p._clock) == 2


def test_slot_past_the_map_still_raises_loudly():
    """The slot map cannot grow under a captured graph, so an out-of-range slot must raise."""
    p = make_pool(capacity=8, map_size=64)
    with pytest.raises(RuntimeError, match="SLOT_MAP_SIZE"):
        p._assign_many([10, 64])


def test_empty_batch_touches_nothing():
    p = make_pool(capacity=8)
    before = snapshot(p)
    assert p._assign_many([]) == []
    assert snapshot(p) == before


def test_assign_is_assign_many_of_one():
    """``cpr_apc_adopt`` still calls the single-slot door; it must stay the write-through path."""
    old, new = make_pool(capacity=6), make_pool(capacity=6)
    for s in (40, 41, 42):
        assert _assign_legacy(old, s) == new._assign(s)
    assert snapshot(old) == snapshot(new)
    assert new._slot_pending == {}, "_assign must still write through, not defer"


# driver-managed ownership and exact direct/offline fallback
def _device_decode_touch(p: RecurrentGDN, slots) -> None:
    """The two device bookkeeping ops decode enqueues, expressed on CPU for the oracle."""

    rows = torch.tensor([p._slot2row.get(int(s), 0) for s in slots], dtype=torch.long)
    p._clock += 1
    p.last_used[rows] = p._clock


def test_offline_full_pool_reads_device_truth_once_per_batch():
    p = make_pool(capacity=3)
    p._assign_many([10, 11, 12])
    # A direct/offline decode has no scheduler lifecycle oracle, so make row 2 newest on device
    # only: the batch-level fallback must import device truth, not guess from host touches.
    _device_decode_touch(p, [next(s for s, r in p._slot2row.items() if r == 2)])
    expected = min(range(1, 4), key=lambda r: (int(p.last_used[r]), r))
    before = host_bookkeeping_stats()["lru_device_fallback"]
    assert p._assign_many([90]) == [expected]
    assert host_bookkeeping_stats()["lru_device_fallback"] == before + 1


def test_driver_managed_full_pool_refuses_without_partial_mutation():
    p = make_pool(capacity=3)
    p._assign_many([10, 11])  # one free row cannot serve two new slots
    p._driver_managed = True
    before = snapshot(p)
    with pytest.raises(RuntimeError, match="driver-managed state pool exhausted"):
        p._assign_many([90, 91])
    assert snapshot(p) == before


def test_driver_managed_full_pool_never_evicts_a_mapped_row():
    p = make_pool(capacity=3)
    p._assign_many([10, 11, 12])
    p._driver_managed = True
    before = snapshot(p)
    with pytest.raises(RuntimeError, match="driver-managed state pool exhausted"):
        p._assign_many([90])
    assert snapshot(p) == before


def test_free_list_may_not_contain_a_mapped_row():
    p = make_pool(capacity=3)
    row = p._assign_many([10])[0]
    p._free.append(row)  # inject the lifecycle corruption the preflight must catch
    before = snapshot(p)
    with pytest.raises(RuntimeError, match="free-list rows"):
        p._assign_many([90])
    assert snapshot(p) == before


def test_release_refuses_to_free_a_row_with_an_alias_mapping():
    p = make_pool(capacity=3)
    row = p._assign_many([10])[0]
    p._slot2row[11] = row  # inject an impossible double owner
    before_free = list(p._free)
    with pytest.raises(RuntimeError, match="also mapped"):
        p.release_slot(10)
    assert p._slot2row[10] == row and p._slot2row[11] == row
    assert p._free == before_free


def test_inverse_ownership_stays_exact_across_rebind_release_and_reuse():
    p = make_pool(capacity=3)
    rows = p._assign_many([10, 11])
    assert p._row2slot == {rows[0]: 10, rows[1]: 11}
    assert p._free_set == set(p._free)

    p.rebind_slot(10, 20)
    assert 10 not in p._slot2row and p._slot2row[20] == rows[0]
    assert p._row2slot[rows[0]] == 20

    assert p.release_slot(11) == rows[1]
    assert rows[1] in p._free_set and rows[1] in p._free
    assert rows[1] not in p._row2slot

    assert p._assign_many([30]) == [rows[1]]
    assert p._row2slot[rows[1]] == 30
    assert rows[1] not in p._free_set


def test_inverse_ownership_is_lazily_reconstructed_for_legacy_pool():
    p = make_pool(capacity=3)
    del p._row2slot
    del p._free_set
    row = p._assign_many([10])[0]
    assert p._row2slot == {row: 10}
    assert p._free_set == set(p._free)


def test_position_mirror_and_missing_row_fallback_are_value_identical():
    ly = object.__new__(CprGDN)
    ly.pos = torch.tensor([0, 17, 64, 129], dtype=torch.long)
    ly._row_pos = {1: 17, 2: 64}

    before = host_bookkeeping_stats()
    assert ly._prefill_pos0([2, 1], torch.tensor([2, 1])) == [64, 17]
    mid = host_bookkeeping_stats()
    assert mid["pos_mirror"] == before["pos_mirror"] + 1
    assert mid["pos_device_fallback"] == before["pos_device_fallback"]

    assert ly._prefill_pos0([3], torch.tensor([3])) == [129]
    after = host_bookkeeping_stats()
    assert after["pos_device_fallback"] == mid["pos_device_fallback"] + 1
    assert ly._row_pos[3] == 129


def test_unknown_driver_split_clears_stale_decode_positions():
    ly = object.__new__(CprGDN)
    ly._driver_managed = False
    ly._row_pos = {1: 64}
    ly._slot2row = OrderedDict([(10, 1)])
    ly.note_slot_positions(None)
    assert ly._driver_managed
    assert ly._row_pos == {}
