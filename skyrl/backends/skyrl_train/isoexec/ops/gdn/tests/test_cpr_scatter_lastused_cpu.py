"""The LRU stamp folded into the fused buffer scatter (``cpr_buffer_scatter(last_used=, clock=)``).

The clock bump must happen above the scatter so every program reads a value nobody writes.
Triton is GPU-only, so the CPU half pins the argument contract and value equivalence needs a device.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from skyrl.backends.skyrl_train.isoexec.ops.gdn.gdn_cpr_scatter import (  # noqa: E402
    cpr_buffer_scatter,
)

CUDA = torch.cuda.is_available()


def _bufs(rows=8, C=64, HV=4, K=128, V=128, N=5, device="cpu"):
    g = torch.Generator().manual_seed(7)
    k = torch.randn(N, HV, K, generator=g).to(torch.bfloat16).to(device)
    v = torch.randn(N, HV, V, generator=g).to(torch.bfloat16).to(device)
    gt = torch.randn(N, HV, generator=g).to(device)
    b = torch.randn(N, HV, generator=g).to(torch.bfloat16).to(device)
    return dict(
        k=k,
        v=v,
        g=gt,
        beta=b,
        rows=torch.tensor([1, 2, 3, 0, 0], dtype=torch.int64, device=device)[:N],
        pos=torch.zeros(rows, dtype=torch.int64, device=device),
        k_buf=torch.zeros(rows, C, HV, K, dtype=torch.bfloat16, device=device),
        v_buf=torch.zeros(rows, C, HV, V, dtype=torch.bfloat16, device=device),
        g_buf=torch.zeros(rows, C, HV, device=device),
        b_buf=torch.zeros(rows, C, HV, dtype=torch.bfloat16, device=device),
        chunk_size=C,
    )


def _call(d, **kw):
    return cpr_buffer_scatter(
        d["k"], d["v"], d["g"], d["beta"], d["rows"], d["pos"],
        d["k_buf"], d["v_buf"], d["g_buf"], d["b_buf"], d["chunk_size"], **kw,
    )  # fmt: skip


def test_last_used_and_clock_are_all_or_nothing():
    d = _bufs()
    lu = torch.zeros(8, dtype=torch.int64)
    clk = torch.zeros((), dtype=torch.int64)
    with pytest.raises(ValueError, match="passed together"):
        _call(d, last_used=lu)
    with pytest.raises(ValueError, match="passed together"):
        _call(d, clock=clk)


def test_stamp_dtypes_are_checked_before_the_launch():
    d = _bufs()
    with pytest.raises(TypeError, match="int64"):
        _call(d, last_used=torch.zeros(8, dtype=torch.int32), clock=torch.zeros((), dtype=torch.int64))
    with pytest.raises(TypeError, match="int64"):
        _call(d, last_used=torch.zeros(8, dtype=torch.int64), clock=torch.zeros(3, dtype=torch.int64))


@pytest.mark.skipif(not CUDA, reason="the scatter is Triton; needs a CUDA device")
def test_stamp_equals_the_index_put_it_replaces():
    """Same values as ``last_used[rows] = clock``, including the duplicated null row."""
    d = _bufs(device="cuda")
    clk = torch.full((), 41, dtype=torch.int64, device="cuda")
    lu = torch.zeros(8, dtype=torch.int64, device="cuda")
    ref = torch.zeros(8, dtype=torch.int64, device="cuda")
    ref[d["rows"]] = clk
    _call(d, last_used=lu, clock=clk)
    torch.cuda.synchronize()
    assert torch.equal(lu, ref)


@pytest.mark.skipif(not CUDA, reason="the scatter is Triton; needs a CUDA device")
def test_omitting_the_stamp_leaves_last_used_untouched():
    d = _bufs(device="cuda")
    lu = torch.arange(8, dtype=torch.int64, device="cuda")
    before = lu.clone()
    _call(d)
    torch.cuda.synchronize()
    assert torch.equal(lu, before)
