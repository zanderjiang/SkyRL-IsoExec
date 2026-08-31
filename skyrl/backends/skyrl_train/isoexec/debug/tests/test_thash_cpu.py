"""CPU tests for the debug-mode tensor digest (``debug/thash.py``).

Covers chunking invariance, bit/permutation sensitivity, shape and dtype separation, the k-ladder,
segment localization, and a pure-Python reference. GPU equivalence is env-gated.
"""

from __future__ import annotations

import os
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[6]))

from skyrl.backends.skyrl_train.isoexec.debug import thash  # noqa: E402

torch.manual_seed(0)


def test_chunking_invariance():
    t = torch.randn(517, 129, dtype=torch.float32)
    digests = {thash.tensor_digest(t, chunk_numel=c) for c in (1 << 22, 4096, 977, 64, t.numel())}
    assert len(digests) == 1, f"digest depends on chunking: {digests}"
    b = t.to(torch.bfloat16)
    assert thash.tensor_digest(b, chunk_numel=101) == thash.tensor_digest(b, chunk_numel=1 << 20)


def test_equality_and_clone():
    t = torch.randn(64, 33, dtype=torch.bfloat16)
    assert thash.tensor_digest(t) == thash.tensor_digest(t.clone())


def test_non_contiguous():
    t = torch.randn(32, 64, dtype=torch.bfloat16)
    strided = t[:, ::2]
    assert not strided.is_contiguous()
    assert thash.tensor_digest(strided) == thash.tensor_digest(strided.contiguous())
    tt = t.t()
    assert thash.tensor_digest(tt) == thash.tensor_digest(tt.contiguous())
    # a transposed view has different logical content than its base
    sq = torch.randn(16, 16)
    assert thash.tensor_digest(sq) != thash.tensor_digest(sq.t())


def test_single_bit_flip_changes_digest():
    for dtype, ivt in ((torch.bfloat16, torch.int16), (torch.float32, torch.int32)):
        t = torch.randn(37, 19, dtype=dtype)
        for flat_idx in (0, 100, t.numel() - 1):
            u = t.clone()
            iv = u.view(-1).view(ivt)
            iv[flat_idx] = iv[flat_idx] ^ 1
            assert thash.tensor_digest(u) != thash.tensor_digest(t), (dtype, flat_idx)


def test_permutation_sensitivity():
    t = torch.arange(1, 4097, dtype=torch.float32)
    perm = torch.randperm(t.numel())
    assert not torch.equal(t, t[perm])
    assert thash.tensor_digest(t) != thash.tensor_digest(t[perm])
    # an unweighted sum would collide here
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([2.0, 1.0])
    assert thash.tensor_digest(a) != thash.tensor_digest(b)


def test_shape_and_dtype_separation():
    t = torch.randn(6, 8)
    assert thash.tensor_digest(t) != thash.tensor_digest(t.reshape(8, 6))
    assert thash.tensor_digest(t) != thash.tensor_digest(t.reshape(48))
    b = torch.randn(11, 13, dtype=torch.bfloat16)
    assert thash.tensor_digest(b) != thash.tensor_digest(b.view(torch.int16))
    assert thash.tensor_digest(b) != thash.tensor_digest(b, k=3)  # truncation level is in the header


def test_dtype_coverage():
    seen = set()
    for dtype in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
    ):
        if dtype.is_floating_point:
            t = torch.randn(9, 7).to(dtype)
        elif dtype == torch.bool:
            t = torch.rand(9, 7) > 0.5
        else:
            t = torch.randint(0, 100, (9, 7), dtype=dtype)
        d = thash.tensor_digest(t)
        assert isinstance(d, int) and 0 <= d < 1 << 64
        assert d == thash.tensor_digest(t.clone(), chunk_numel=13)
        seen.add(d)
    assert len(seen) == 10


def test_ladder_monotone_threshold():
    ks = (6, 4, 2, 0)
    for bit in range(7):  # bf16: 7 explicit mantissa bits, flip mantissa bit `bit`
        a = torch.rand(50, dtype=torch.bfloat16) + 0.5
        u = a.clone()
        iv = u.view(torch.int16)
        iv[7] = iv[7] ^ (1 << bit)
        la, lb = thash.digest_ladder(a, ks), thash.digest_ladder(u, ks)
        assert la["full"] != lb["full"]
        matched = [k for k in ks if la[f"k{k}"] == lb[f"k{k}"]]
        diverged = [k for k in ks if la[f"k{k}"] != lb[f"k{k}"]]
        # rung k keeps the top k mantissa bits: it sees the flip iff bit >= 7-k
        expect_matched = [k for k in ks if bit < 7 - k]
        assert matched == expect_matched, (bit, matched, diverged)
        # monotone: every matched rung is coarser than every diverged rung
        assert not diverged or all(m < min(diverged) for m in matched)


def test_ladder_equal_tensors_match_everywhere():
    t = torch.randn(31, 5, dtype=torch.bfloat16)
    la, lb = thash.digest_ladder(t), thash.digest_ladder(t.clone())
    assert la == lb
    assert set(la) == {"full", "k6", "k4", "k2", "k0"}
    i = torch.randint(0, 9, (4, 4))
    assert set(thash.digest_ladder(i)) == {"full"}  # no ladder for int dtypes


def test_ladder_magnitude_tracks_perturbation():
    # a_i = 1 + i * 2**-10, exact in fp32, mantissa bits below bit 10 all zero -> no carry noise.
    a = 1.0 + torch.arange(512, dtype=torch.float32) * 2.0**-10
    ks = (6, 4, 2, 0)
    la = thash.digest_ladder(a, ks)
    # tiny perturbation (~1e-6 relative): diverges at full precision, matches the whole ladder
    tiny = (a.double() * (1.0 + 2.0**-20)).float()
    lt = thash.digest_ladder(tiny, ks)
    assert la["full"] != lt["full"]
    assert all(la[f"k{k}"] == lt[f"k{k}"] for k in ks)
    # coarse perturbation (~5% relative, exponent preserved): fine rungs diverge, k0 matches
    coarse = (a * 1.05).float()
    lc = thash.digest_ladder(coarse, ks)
    assert la["full"] != lc["full"] and la["k6"] != lc["k6"]
    assert la["k0"] == lc["k0"]


def test_segment_localization():
    a = torch.randn(16, 32, dtype=torch.float32)
    b = a.clone()
    b[5, 7] += 1.0
    sa = thash.segment_digests(a, rows_per_segment=4)
    sb = thash.segment_digests(b, rows_per_segment=4)
    assert len(sa) == len(sb) == 4
    assert [i for i in range(4) if sa[i] != sb[i]] == [1]  # row 5 lives in segment 1
    assert thash.segment_digests(a, rows_per_segment=4) == sa  # deterministic


def test_edge_shapes():
    assert thash.tensor_digest(torch.tensor(3.5)) == thash.tensor_digest(torch.tensor(3.5))
    e = torch.empty(0, 8)
    assert thash.tensor_digest(e) == thash.tensor_digest(e.clone())
    assert thash.tensor_digest(e) != thash.tensor_digest(torch.empty(0, 4))


def test_iter_tensor_outputs():
    t1, t2 = torch.zeros(2), torch.ones(3)
    assert [p for p, _ in thash.iter_tensor_outputs(t1)] == ["0"]
    got = list(thash.iter_tensor_outputs((t1, None, t2)))
    assert [p for p, _ in got] == ["0", "2"]
    assert list(thash.iter_tensor_outputs("nope")) == []


def test_iter_tensor_outputs_descends_nested_and_dict():
    """Nested tuples and dict outputs are descended, not dropped."""
    t = torch.zeros(2)
    assert [p for p, _ in thash.iter_tensor_outputs((t, (t, t)))] == ["0", "1.0", "1.1"]
    assert [p for p, _ in thash.iter_tensor_outputs({"h": t, "aux": [t]})] == ["h", "aux.0"]
    assert list(thash.iter_tensor_outputs(None)) == []
    # too deep to descend -> a (path, None) marker, never a silent drop
    deep = (((t,),),)
    assert list(thash.iter_tensor_outputs(deep, max_depth=2)) == [("0.0", None)]


def test_extended_dtype_coverage():
    """fp8 / unsigned widths / complex digest through their integer view."""
    for name in ("float8_e4m3fn", "float8_e5m2", "uint16", "uint32", "complex64", "complex128"):
        dtype = getattr(torch, name, None)
        if dtype is None:
            continue
        t = torch.zeros(6, 4).to(dtype)
        u = t.clone()
        u.view(torch.int8)[3] = 1  # one bit somewhere in the buffer
        assert thash.tensor_digest(t) == thash.tensor_digest(t.clone(), chunk_numel=7), name
        assert thash.tensor_digest(t) != thash.tensor_digest(u), name
        assert len(thash.segment_digests(t, rows_per_segment=2)) == 3, name


def test_unsupported_dtype_raises():
    """The digest refuses loudly, so trace.py can record 'unrecordable: <reason>'."""
    exotic = next(
        (
            getattr(torch, n)
            for n in ("float8_e8m0fnu", "bits8", "float4_e2m1fn_x2")
            if getattr(torch, n, None) is not None and getattr(torch, n) not in thash._DTYPE_TABLE
        ),
        None,
    )
    if exotic is None:
        return
    try:
        thash.tensor_digest(torch.zeros(4).to(exotic))
    except TypeError as e:
        assert "unsupported dtype" in str(e)
    else:
        raise AssertionError("expected TypeError for an unsupported dtype")


def test_ladder_depth_is_per_dtype():
    """Ladder depth follows the dtype, so a 1-ULP fp32 difference stays resolvable."""
    assert thash.ladder_for(torch.bfloat16) == (6, 4, 2, 0)  # k=6 is bf16's finest expressible
    assert max(thash.ladder_for(torch.float32)) == 22
    assert max(thash.ladder_for(torch.float64)) == 48
    assert thash.ladder_for(torch.int32) == ()
    x = torch.randn(64, 32, dtype=torch.float32)
    y = x.clone()
    y.view(-1).view(torch.int32)[100] ^= 1  # 1 ULP == 2**-23 relative
    la, lb = thash.digest_ladder(x), thash.digest_ladder(y)
    assert la["full"] != lb["full"]
    assert la["k22"] == lb["k22"]  # bounded below 2**-22


def test_gpu_equivalence_skip_gated():
    if os.environ.get("SKYRL_ISOEXEC_DEBUG_TEST_GPU") != "1":
        print("  (skipped: set SKYRL_ISOEXEC_DEBUG_TEST_GPU=1 on an idle GPU node)")
        return
    if not torch.cuda.is_available():
        print("  (skipped: no CUDA)")
        return
    t = torch.randn(1024, 512, dtype=torch.bfloat16)
    assert thash.tensor_digest(t) == thash.tensor_digest(t.cuda())
    assert thash.digest_ladder(t) == thash.digest_ladder(t.cuda())


# -- an independent reference, so "eager == triton" cannot both be wrong the same way ---------

_U64 = (1 << 64) - 1


def _ref_mix(z: int) -> int:
    z &= _U64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _U64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _U64
    return z ^ (z >> 31)


def _ref_body(t, k, seed, off=0, count=None, salt_seed=None):
    """The weighted modular sum, written from the docstring in pure Python ints."""
    view, umask, mbits = thash._DTYPE_TABLE[t.dtype]
    vals = t.detach().cpu().contiguous().reshape(-1).view(view).to(torch.int64).tolist()
    if count is not None:
        vals = vals[off : off + count]
    mask, kept = _U64 if umask is None else umask, mbits
    if k is not None and mbits is not None and k < mbits:
        mask &= _U64 ^ ((1 << (mbits - k)) - 1)
        kept = k
    salt = _ref_mix(seed if salt_seed is None else salt_seed) ^ 0x9E3779B97F4A7C15
    acc = 0
    for i, v in enumerate(vals):
        w = ((_ref_mix((i >> 3) ^ salt) ^ ((i & 7) * 0x9E3779B97F4A7C15)) & _U64) | 1
        acc = (acc + ((v & _U64) & mask) * w) & _U64
    return acc, (kept if mbits is not None else -1)


def ref_digest(t, k=None, seed=0) -> int:
    acc, kept = _ref_body(t, k, seed)
    return thash._mix64_i(acc ^ thash._header(t, kept, seed))


def ref_segments(t, rows_per_segment, k=None, seed=0):
    view, _um, _mb = thash._DTYPE_TABLE[t.dtype]
    n = t.detach().cpu().contiguous().reshape(-1).view(view).numel()
    rows = t.shape[thash.segment_axis(t)]
    seg_numel = rows_per_segment * (n // rows)
    nseg = (rows + rows_per_segment - 1) // rows_per_segment
    head = thash._header(t, _ref_body(t, k, seed, 0, 0)[1], seed)
    out = []
    for si in range(nseg):
        acc, _ = _ref_body(
            t,
            k,
            seed,
            si * seg_numel,
            min(seg_numel, n - si * seg_numel),
            salt_seed=seed ^ _ref_mix(si * seg_numel + 1),
        )
        out.append(f"{thash._mix64_i(acc ^ thash._mix64_i(head ^ (si + 1))):016x}")
    return out


_REF_SHAPES = ((1,), (7,), (8,), (9,), (1, 5, 3), (13, 11), (3, 1, 1))


def test_matches_pure_python_reference():
    for dtype in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.int8,
        torch.uint8,
        torch.int32,
        torch.int64,
        torch.bool,
    ):
        for shape in _REF_SHAPES:
            t = torch.randn(shape).to(dtype) if dtype.is_floating_point else torch.randint(0, 90, shape).to(dtype)
            assert thash.tensor_digest(t) == ref_digest(t), (dtype, shape)
            for k in thash.ladder_for(dtype):
                assert thash.tensor_digest(t, k=k) == ref_digest(t, k=k), (dtype, shape, k)
            for rps in (1, 2, 64):
                assert thash.segment_digests(t, rows_per_segment=rps) == ref_segments(t, rps), (dtype, shape, rps)


def test_ladder_is_one_pass_but_equals_per_rung_digests():
    """The fused multi-mask pass must produce exactly the per-rung tensor_digest values."""
    for dtype in (torch.bfloat16, torch.float32, torch.float64):
        t = (torch.randn(37, 9) * 5).to(dtype)
        lad = thash.digest_ladder(t)
        assert lad["full"] == f"{thash.tensor_digest(t):016x}"
        assert set(lad) == {"full"} | {f"k{k}" for k in thash.ladder_for(dtype)}
        for k in thash.ladder_for(dtype):
            assert lad[f"k{k}"] == f"{thash.tensor_digest(t, k=k):016x}"


def test_segment_axis_is_first_non_unit_dim():
    """Segmentation uses the first non-unit dim, so a [1,T,H,D] tensor is not one segment."""
    assert thash.segment_axis(torch.zeros(1, 64, 4, 8)) == 1
    assert thash.segment_axis(torch.zeros(64, 1, 2048)) == 0
    assert thash.segment_axis(torch.zeros(1, 1, 1)) == 0
    assert thash.segment_axis(torch.zeros(7)) == 0
    t = torch.randn(1, 64, 4, 8, dtype=torch.float32)
    segs = thash.segment_digests(t, rows_per_segment=16)
    assert len(segs) == 4
    b = t.clone()
    b[0, 40, 1, 3] += 1.0
    assert [i for i, (x, y) in enumerate(zip(segs, thash.segment_digests(b, rows_per_segment=16))) if x != y] == [2]


def test_segment_index_means_the_same_slab_on_both_sides():
    """Segment i covers the same token slab on both side layouts (about which rows, not equal
    hex -- the digests still fold the shape)."""
    core = torch.randn(1, 32, 2, 4, dtype=torch.float32)  # trainer gdn.core door layout
    flat = core.reshape(32, 8)  # same tokens, engine-side layout
    a, b = core.clone(), flat.clone()
    a[0, 20, 1, 2] += 1.0
    b[20, 6] += 1.0
    da = [
        i
        for i, (x, y) in enumerate(
            zip(thash.segment_digests(core, rows_per_segment=8), thash.segment_digests(a, rows_per_segment=8))
        )
        if x != y
    ]
    db = [
        i
        for i, (x, y) in enumerate(
            zip(thash.segment_digests(flat, rows_per_segment=8), thash.segment_digests(b, rows_per_segment=8))
        )
        if x != y
    ]
    assert da == db == [2]


def test_eager_backend_override_matches_default():
    t = (torch.randn(129, 7) * 4).to(torch.bfloat16)
    want = thash.tensor_digest(t)
    os.environ[thash.ENV_BACKEND] = "eager"
    thash._reset_backend_for_tests()
    try:
        assert thash.digest_backend(t.device) == "eager"
        assert thash.tensor_digest(t) == want
    finally:
        os.environ.pop(thash.ENV_BACKEND, None)
        thash._reset_backend_for_tests()


def _run():
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f()
            print("PASS", n)
    print(f"\n{sum(1 for n in globals() if n.startswith('test_'))} passed")


if __name__ == "__main__":
    _run()
