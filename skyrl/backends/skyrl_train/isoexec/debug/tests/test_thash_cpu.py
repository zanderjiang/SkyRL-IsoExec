"""CPU guarantees for the debug-mode tensor digest (``debug/thash.py``).

Covers: determinism across reduction chunkings, single-bit and permutation sensitivity, shape
and dtype separation, non-contiguous inputs, the mantissa-truncation k-ladder (monotone rung
threshold), segment localization, and edge shapes. GPU equivalence is skip-gated behind
SKYRL_ISOEXEC_DEBUG_TEST_GPU=1 and must not run on a busy node.

Run (CPU only):
    python skyrl/backends/skyrl_train/isoexec/debug/tests/test_thash_cpu.py
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
    # the weakness the weighted scheme fixes: an unweighted sum would collide here
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
    assert [i for i, _ in thash.iter_tensor_outputs(t1)] == [0]
    got = list(thash.iter_tensor_outputs((t1, None, t2)))
    assert [i for i, _ in got] == [0, 2]
    assert list(thash.iter_tensor_outputs("nope")) == []


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


def _run():
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f()
            print("PASS", n)
    print(f"\n{sum(1 for n in globals() if n.startswith('test_'))} passed")


if __name__ == "__main__":
    _run()
