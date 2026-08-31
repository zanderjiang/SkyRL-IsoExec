"""CPU tests for the per-forward GDN sequence-metadata memo (``sequence_metadata``).

The memo must be a host-work change only: a hit returns exactly what a rebuild would have, and the
key (``id(cu) + cu._version``, pinned by a strong reference) invalidates on any in-place write.
"""

import pytest

torch = pytest.importorskip("torch")

from skyrl.backends.skyrl_train.isoexec.ops.gdn import (  # noqa: E402
    gdn_cpr as gcs,  # noqa: E402
)

CHUNK = 64


@pytest.fixture(autouse=True)
def _clean_cache(monkeypatch):
    gcs._SEQ_CACHE.clear()
    for k in gcs._SEQ_STATS:
        gcs._SEQ_STATS[k] = 0
    monkeypatch.delenv("SKYRL_ISOEXEC_GDN_SEQ_META_CACHE", raising=False)
    yield
    gcs._SEQ_CACHE.clear()


def _cu(lens):
    out = [0]
    for n in lens:
        out.append(out[-1] + n)
    return torch.tensor(out, dtype=torch.int32)


def _rebuild(cu, chunk_size=CHUNK):
    """Exactly what a layer did before the memo existed."""
    lens = gcs.packed_lens(cu)
    cu_chunked, state_indices, n_chunks = gcs.chunked_cu_seqlens_and_indices(cu, chunk_size, cu.device, lens=lens)
    return lens, cu.clone(), cu_chunked, state_indices, n_chunks


def _assert_same(a, b):
    assert a[0] == b[0]
    assert torch.equal(a[1], b[1])
    assert torch.equal(a[2], b[2])
    assert torch.equal(a[3], b[3])
    assert a[4] == b[4]


# 1. exactness against the pre-memo rebuild
@pytest.mark.parametrize(
    "lens",
    [
        [64],  # one exact chunk
        [63],  # one short chunk
        [65],  # a straddling second chunk
        [128, 64],
        [7648],  # the re-trace's microbatch shape
        [5473, 2175],  # the measured mean sequence, packed in pairs
        [1, 1, 1, 1],  # degenerate short sequences
        [8617, 5304, 4096, 2560],  # the measured max + typical bin
        [],  # empty batch
    ],
)
def test_memo_equals_rebuild(lens):
    cu = _cu(lens)
    want = _rebuild(cu)
    got = gcs.sequence_metadata(cu, CHUNK, cu.device)
    _assert_same(got, want)
    # a second call (a later GDN layer of the same forward) must agree too
    _assert_same(gcs.sequence_metadata(cu, CHUNK, cu.device), want)


def test_hit_returns_the_same_objects():
    """A hit hands back the identical objects, so no device op or host loop re-runs."""
    cu = _cu([4096, 2048])
    first = gcs.sequence_metadata(cu, CHUNK, cu.device)
    second = gcs.sequence_metadata(cu, CHUNK, cu.device)
    assert first is second
    for x, y in zip(first, second):
        if torch.is_tensor(x):
            assert x is y


def test_thirty_layers_build_once():
    """One build per forward, not one per GDN layer."""
    cu = _cu([5473, 2175])
    for _ in range(30):
        gcs.sequence_metadata(cu, CHUNK, cu.device)
    census = gcs.seq_meta_census()
    assert census["built"] == 1
    assert census["served"] == 29
    assert census["declined"] == 0


# 2. key soundness
def test_in_place_write_invalidates():
    cu = _cu([128, 128])
    first = gcs.sequence_metadata(cu, CHUNK, cu.device)
    assert first[0] == [128, 128]
    cu[1] = 64  # an in-place write bumps the version counter
    second = gcs.sequence_metadata(cu, CHUNK, cu.device)
    assert second is not first
    _assert_same(second, _rebuild(cu))
    assert second[0] == [64, 192]


def test_write_through_an_alias_invalidates():
    """Version counters are shared across views, so a write through the BASE must invalidate."""
    base = torch.zeros(3, dtype=torch.int32)
    base[1] = 100
    base[2] = 300
    view = base.view(-1)
    first = gcs.sequence_metadata(view, CHUNK, view.device)
    assert first[0] == [100, 200]
    base[2] = 200
    second = gcs.sequence_metadata(view, CHUNK, view.device)
    assert second is not first
    assert second[0] == [100, 100]


def test_distinct_tensor_same_values_is_never_served_a_stale_entry():
    """Two distinct tensors with equal values get independent entries, not each other's."""
    a = _cu([256, 256])
    b = _cu([256, 256])
    ga = gcs.sequence_metadata(a, CHUNK, a.device)
    gb = gcs.sequence_metadata(b, CHUNK, b.device)
    _assert_same(gb, _rebuild(b))
    assert ga[0] == gb[0]


def test_different_chunk_size_is_a_different_entry():
    cu = _cu([256])
    g64 = gcs.sequence_metadata(cu, 64, cu.device)
    g32 = gcs.sequence_metadata(cu, 32, cu.device)
    assert g64[4] == 4 and g32[4] == 8
    _assert_same(g32, _rebuild(cu, 32))


def test_oversized_cu_is_declined_but_still_correct():
    cu = torch.arange(0, 9000, dtype=torch.int32)  # numel > 8192 -> not cacheable
    got = gcs.sequence_metadata(cu, CHUNK, cu.device)
    _assert_same(got, _rebuild(cu))
    assert gcs.seq_meta_census()["declined"] == 1
    assert gcs.seq_meta_census()["served"] == 0


def test_cache_is_bounded():
    keep = [_cu([64 * (i + 1)]) for i in range(gcs._SEQ_CACHE_MAX + 4)]
    for cu in keep:
        gcs.sequence_metadata(cu, CHUNK, cu.device)
    assert len(gcs._SEQ_CACHE) <= gcs._SEQ_CACHE_MAX


def test_entry_pins_its_key_tensor():
    """The entry holds a strong reference, so the id() half of the key cannot be reused."""
    cu = _cu([512])
    gcs.sequence_metadata(cu, CHUNK, cu.device)
    (key, (pinned, _entry)) = next(iter(gcs._SEQ_CACHE.items()))
    assert pinned is cu
    assert key[0] == id(cu)


# 3. the OFF path
def test_flag_off_rebuilds_every_call(monkeypatch):
    monkeypatch.setenv("SKYRL_ISOEXEC_GDN_SEQ_META_CACHE", "0")
    assert not gcs.seq_meta_cache_enabled()
    cu = _cu([1024, 1024])
    a = gcs.sequence_metadata(cu, CHUNK, cu.device)
    b = gcs.sequence_metadata(cu, CHUNK, cu.device)
    assert a is not b
    _assert_same(a, _rebuild(cu))
    _assert_same(b, _rebuild(cu))
    census = gcs.seq_meta_census()
    assert census["served"] == 0 and census["built"] == 0 and census["declined"] == 2


@pytest.mark.parametrize("val,expect", [("0", False), ("false", False), ("no", False), ("", False), ("1", True)])
def test_flag_parsing(monkeypatch, val, expect):
    monkeypatch.setenv("SKYRL_ISOEXEC_GDN_SEQ_META_CACHE", val)
    assert gcs.seq_meta_cache_enabled() is expect


def test_on_and_off_agree_exactly(monkeypatch):
    """The flag is neutral: same tensors with the memo on or off."""
    cu = _cu([3000, 1500, 700])
    monkeypatch.setenv("SKYRL_ISOEXEC_GDN_SEQ_META_CACHE", "0")
    off = gcs.sequence_metadata(cu, CHUNK, cu.device)
    monkeypatch.setenv("SKYRL_ISOEXEC_GDN_SEQ_META_CACHE", "1")
    on = gcs.sequence_metadata(cu, CHUNK, cu.device)
    _assert_same(on, off)
