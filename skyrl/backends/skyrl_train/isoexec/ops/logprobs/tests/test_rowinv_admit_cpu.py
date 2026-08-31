"""Admission-layer contract split for ``ops/logprobs/rowinv.py``: dtype is per-call, not structure.

One trainer process alternates bf16 (scoring) and fp32 (training) logits, so a payload dtype flip
must not trip the drift gate, while genuine structural drift still must. CPU-only: the gate's
cache and contract compare run identically whether the call serves or declines.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from skyrl.backends.skyrl_train.isoexec.ops.logprobs import rowinv  # noqa: E402

V = 64


@pytest.fixture(autouse=True)
def _armed_and_fresh(monkeypatch):
    """Arm the flags and clear the per-group admission cache around every test."""
    monkeypatch.setenv("SKYRL_ISOEXEC", "1")
    monkeypatch.setenv(rowinv.LEAVES_ENV, "8")
    rowinv._reset_for_test()
    yield
    rowinv._reset_for_test()


def _call(x: torch.Tensor, t: torch.Tensor, *, end: int, src: torch.dtype):
    return rowinv.rowinv_sampled_logprobs(
        x,
        t,
        vocab_start_index=0,
        vocab_end_index=end,
        group=None,
        src_dtype=src,
        reference=lambda: (_ for _ in ()).throw(AssertionError("reference must not run on a decline")),
    )


def test_dtype_change_is_per_call_eligibility_not_structural_drift():
    """The crash shape: bf16 admission state, then an fp32 call on the same group.

    On CPU every call declines (the op needs CUDA), but the group cache and its immutable contract
    are still created on the first call, so the dtype flip must decline politely, never raise.
    """
    t64 = torch.zeros(2, dtype=torch.int64)
    x16 = torch.randn(2, V).to(torch.bfloat16)
    x32 = x16.float()  # the exact widen: same underlying values, the training forward's dtype

    assert _call(x16, t64, end=V, src=torch.bfloat16) is None  # scoring-shaped call seeds the cache
    assert _call(x32, t64, end=V, src=torch.float32) is None  # the crash site: must NOT raise
    assert _call(x16, t64, end=V, src=torch.bfloat16) is None  # alternation, not a one-way switch
    # target dtype is the same class of per-call fact: int32 after int64 must not raise.
    assert _call(x16, torch.zeros(2, dtype=torch.int32), end=V, src=torch.bfloat16) is None


def test_fp32_source_dtype_is_eligible_not_refused():
    """fp32 must pass the source-dtype eligibility term.

    Refusing it would hand the training forward to the incumbent while scoring stayed on rowinv.
    On CPU the decline must name the CUDA requirement (an earlier term), never the source dtype.
    """
    x32 = torch.randn(2, V)
    _call(x32, torch.zeros(2, dtype=torch.int64), end=V, src=torch.float32)
    reason = rowinv.stats()["decline_reason"]
    assert "source dtype" not in reason, reason
    assert "CUDA" in reason, reason


def test_changed_vocab_partition_still_raises_structural_drift():
    """Genuine structural drift keeps its teeth: a changed shard width raises, dtype fix or not."""
    t = torch.zeros(2, dtype=torch.int64)
    assert _call(torch.randn(2, V).to(torch.bfloat16), t, end=V, src=torch.bfloat16) is None
    with pytest.raises(RuntimeError, match="STRUCTURAL DRIFT"):
        _call(torch.randn(2, V // 2).to(torch.bfloat16), t, end=V // 2, src=torch.bfloat16)


def test_env_flip_after_admission_still_raises_structural_drift():
    """The env term stays immutable: flipping the master switch after the cache exists raises."""
    t = torch.zeros(2, dtype=torch.int64)
    x = torch.randn(2, V).to(torch.bfloat16)
    assert _call(x, t, end=V, src=torch.bfloat16) is None
    os.environ["SKYRL_ISOEXEC"] = "0"
    try:
        with pytest.raises(RuntimeError, match="STRUCTURAL DRIFT"):
            _call(x, t, end=V, src=torch.bfloat16)
    finally:
        os.environ["SKYRL_ISOEXEC"] = "1"


def test_dtypes_remain_in_the_hot_key():
    """Per-call does not mean unkeyed: distinct dtypes must latch distinct hot entries."""
    t64 = torch.zeros(2, dtype=torch.int64)
    x16 = torch.randn(2, V).to(torch.bfloat16)
    x32 = x16.float()
    k16 = rowinv._hot_key(x16, t64, 0, V, torch.bfloat16)
    k32 = rowinv._hot_key(x32, t64, 0, V, torch.float32)
    assert k16 != k32
    k_t32 = rowinv._hot_key(x16, t64.to(torch.int32), 0, V, torch.bfloat16)
    assert k_t32 != k16
