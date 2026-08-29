"""Triton digest backend == eager digest backend == pure-Python reference, bit for bit.

The trace is only meaningful if the two sides digest identically, and the two sides may not run
the same backend (a CPU-only comparison harness, a torch build without triton, one side pinned
to ``SKYRL_ISOEXEC_DEBUG_DIGEST=eager``). So this is an equality suite, not a numerics suite:
every dtype in the table, edge and ragged shapes, every ladder rung, every segmentation, both
devices, several chunkings -- all against the independent reference in ``test_thash_cpu``.

Skipped without CUDA. Run:
    python -m pytest skyrl/backends/skyrl_train/isoexec/debug/tests/test_thash_gpu.py
"""

from __future__ import annotations

import os
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[6]))

from skyrl.backends.skyrl_train.isoexec.debug import thash  # noqa: E402
from skyrl.backends.skyrl_train.isoexec.debug.tests.test_thash_cpu import ref_digest, ref_segments  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

_DTYPES = [
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
]
for _n in ("float8_e4m3fn", "float8_e5m2", "uint16", "uint32"):
    _dt = getattr(torch, _n, None)
    if _dt is not None:
        _DTYPES.append(_dt)

# Sizes that straddle the kernel's BLOCK (4096) and the weight group (8), plus a ragged tail.
_SHAPES = [(1,), (7,), (8,), (9,), (1, 63, 3, 5), (33, 17), (1, 4096), (4097,), (3, 1, 1), (129, 65)]


def _sample(dtype, shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, generator=g).to(torch.bool)
    if dtype.is_floating_point:
        return (torch.randn(shape, generator=g) * 7).to(dtype)
    return torch.randint(0, 100, shape, generator=g).to(dtype)


def _eager(fn, *a, **kw):
    os.environ[thash.ENV_BACKEND] = "eager"
    thash._reset_backend_for_tests()
    try:
        return fn(*a, **kw)
    finally:
        os.environ.pop(thash.ENV_BACKEND, None)
        thash._reset_backend_for_tests()


@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: str(d).replace("torch.", ""))
def test_triton_equals_eager_equals_reference(dtype):
    for shape in _SHAPES:
        cpu = _sample(dtype, shape)
        gpu = cpu.cuda()
        want = ref_digest(cpu)
        assert thash.digest_backend(gpu.device) == "triton"
        got = {
            "ref": want,
            "cpu_eager": thash.tensor_digest(cpu),
            "gpu_triton": thash.tensor_digest(gpu),
            "gpu_triton_chunked": thash.tensor_digest(gpu, chunk_numel=997),
            "gpu_eager": _eager(thash.tensor_digest, gpu),
            "gpu_eager_chunk13": _eager(thash.tensor_digest, gpu, chunk_numel=13),
            "cpu_eager_chunk1": thash.tensor_digest(cpu, chunk_numel=1),
        }
        assert len(set(got.values())) == 1, (dtype, shape, {k: hex(v) for k, v in got.items()})


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
def test_ladder_triton_equals_eager_equals_reference(dtype):
    for shape in ((9,), (1, 63, 3, 5), (4097,)):
        cpu = _sample(dtype, shape, seed=3)
        gpu = cpu.cuda()
        want = {"full": f"{ref_digest(cpu):016x}"}
        want.update({f"k{k}": f"{ref_digest(cpu, k=k):016x}" for k in thash.ladder_for(dtype)})
        assert thash.digest_ladder(gpu) == want, (dtype, shape)
        assert thash.digest_ladder(cpu) == want
        assert _eager(thash.digest_ladder, gpu) == want


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.int32])
def test_segments_triton_equals_eager_equals_reference(dtype):
    for shape in ((9,), (1, 63, 3, 5), (33, 17), (4097,)):
        cpu = _sample(dtype, shape, seed=5)
        gpu = cpu.cuda()
        for rps in (1, 3, 16, 4096):
            want = ref_segments(cpu, rps)
            assert thash.segment_digests(gpu, rows_per_segment=rps) == want, (dtype, shape, rps)
            assert thash.segment_digests(gpu, rows_per_segment=rps, chunk_numel=101) == want
            assert thash.segment_digests(cpu, rows_per_segment=rps) == want
            assert _eager(thash.segment_digests, gpu, rows_per_segment=rps) == want


def test_large_tensor_matches_across_backends():
    """Past one kernel block and one eager chunk, where the block/chunk decomposition differs."""
    t = (torch.randn(4096, 512, generator=torch.Generator().manual_seed(11)) * 3).to(torch.bfloat16).cuda()
    want = thash.tensor_digest(t)
    assert _eager(thash.tensor_digest, t) == want
    assert _eager(thash.tensor_digest, t, chunk_numel=1 << 16) == want
    assert thash.tensor_digest(t.cpu()) == want
    assert thash.digest_ladder(t) == _eager(thash.digest_ladder, t)


def test_sensitivity_on_gpu():
    t = (torch.randn(64, 64) * 3).to(torch.bfloat16).cuda()
    base = thash.tensor_digest(t)
    flipped = t.clone()
    flipped.reshape(-1).view(torch.int16)[1234] ^= 1  # one mantissa bit
    assert thash.tensor_digest(flipped) != base
    swapped = t.clone()
    swapped[0, 0], swapped[0, 1] = t[0, 1].clone(), t[0, 0].clone()
    assert thash.tensor_digest(swapped) != base  # in-group permutation
    assert thash.tensor_digest(t.flip(0).contiguous()) != base
    assert thash.tensor_digest(t.t().contiguous().t()) == base  # logical order, not layout


def test_empty_and_scalar_on_gpu():
    """Degenerate tensors take the same path on both devices; segmenting them is one digest."""
    for t in (torch.empty(0, 8, device="cuda"), torch.tensor(3.5, device="cuda")):
        assert thash.tensor_digest(t) == ref_digest(t.cpu())
        segs = thash.segment_digests(t, rows_per_segment=4)
        assert segs == [f"{thash.tensor_digest(t):016x}"] == thash.segment_digests(t.cpu(), rows_per_segment=4)


def test_preload_compiles_both_kernels():
    assert thash.preload(torch.device("cuda")) is True
