"""Admission-layer contract split for ``ops/logprobs/rowinv.py``: dtype is per-call, not structure.

The failure this pins: one trainer process scores with bf16 logits (the scoring forward under
``SKYRL_ISOEXEC_SCORING_LOGITS_BF16``) and trains with fp32 logits (``Float16Module`` upcasts the
training forward). Treating the payload dtype as an IMMUTABLE per-process structural fact makes
the dtype flip after admission raise ``STRUCTURAL DRIFT`` mid-run. The rule mirrors
``ops/collectives/logprob_gather_wire.py``: facts
that select the collective sequence stay immutable and still raise; payload dtypes are per-call
eligibility fields of the voted signature.

These tests exercise the ``_admit`` gate itself, CPU-only -- the gate's group cache and contract
compare run identically whether the call ultimately serves or declines, so no CUDA device, Triton
kernel, or collective is needed to reproduce the crash shape:

  * a dtype change between calls on one group NEVER trips the drift gate (under the pre-fix
    contract these exact calls raised ``STRUCTURAL DRIFT``);
  * genuine structural drift -- a changed vocabulary partition / shard width -- STILL raises,
    so the fix cannot silently disable the safety gate;
  * the dtypes remain in the hot-path key, so distinct dtypes latch distinct hot entries rather
    than colliding.

The end-to-end half of the regression -- both dtypes actually SERVE and agree bitwise -- needs a
GPU and lives in ``rowinv_gpu.py`` (section 7), and at TP>1 in ``rowinv_tp_dist.py`` (the bf16
phase deliberately reuses the fp32-admitted subgroup).

Run (CPU only):
    uv run --extra dev pytest skyrl/backends/skyrl_train/isoexec/ops/logprobs/tests/test_rowinv_admit_cpu.py -q
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

    On CPU every call declines (the op needs CUDA), but the group cache and its immutable
    structural contract are created on the first call regardless -- exactly the state a live run
    is in when the training forward's dtype moves. With logits/src/
    target dtypes inside the immutable tuple) the second call below raised STRUCTURAL DRIFT; now
    the dtype is a per-call signature field and the calls must decline politely, never raise.
    """
    t64 = torch.zeros(2, dtype=torch.int64)
    x16 = torch.randn(2, V).to(torch.bfloat16)
    x32 = x16.float()  # the exact widen: same underlying values, the training forward's dtype

    assert _call(x16, t64, end=V, src=torch.bfloat16) is None  # scoring-shaped call seeds the cache
    assert _call(x32, t64, end=V, src=torch.float32) is None  # the crash site: must NOT raise
    assert _call(x16, t64, end=V, src=torch.bfloat16) is None  # alternation, not a one-way switch
    # target dtype is the same class of per-call payload fact: int32 after int64 must not raise.
    assert _call(x16, torch.zeros(2, dtype=torch.int32), end=V, src=torch.bfloat16) is None


def test_fp32_source_dtype_is_eligible_not_refused():
    """The fp32 training forward declares src_dtype=fp32 (the dispatch passes the payload dtype).

    Refusing it would silently hand the training forward to the incumbent ATen tree while scoring
    stayed on rowinv -- two functions, PPO ratio off 1 by construction -- so fp32 must pass the
    source-dtype eligibility term. On CPU the decline must therefore name the CUDA requirement
    (an earlier term), never the source dtype.
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
    """The env term stays immutable: flipping the master switch after the cache exists raises.

    Rowinv has no flag of its own any more -- it is the composed default -- so the env fact that
    can still change under a live group is ``SKYRL_ISOEXEC`` itself, and it selects the collective
    sequence exactly as the old flag did.
    """
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
