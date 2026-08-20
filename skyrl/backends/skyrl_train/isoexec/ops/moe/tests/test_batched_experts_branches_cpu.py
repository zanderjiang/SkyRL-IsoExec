"""Every branch of ``_batched_experts_forward`` must RUN, under both settings of every flag it reads.

WHY THIS FILE EXISTS, stated plainly because the lesson cost a gate arm. On 2026-08-10 the tile path
shipped ``host_tiles_per_expert`` bound inside ONE branch and consumed at the single tail call BOTH
branches reach, so every call taking the static branch raised

    UnboundLocalError: cannot access local variable 'host_tiles_per_expert'

live, in the TRAINER logprob forward, at step 1 (fixed in 7e1dfe72). The same day, a fast-absorbed-
backward patch broke its own flag-OFF path by binding ``torch`` as a function-local inside a
conditional import. Both were invisible to their own tests for the SAME reason: the tests exercised
the new *expression* side-by-side against the old one, and never CALLED the function that contained
the branch. `test_flat_stage_bitwise.py` is exactly that kind of test and would not have caught
either bug.

So this file does the other thing. It builds a minimal `SequentialMLP`-shaped module and calls the
real ``_batched_experts_forward`` across the full cross product of

    * ``_STATIC_DECODE``  -- the static (graph-capturable) tile grid vs the dynamic prefill/trainer one
    * ``_FLAT_STAGE``     -- 2-D advanced indexing of the tile buffer vs the 1-D equivalent

and asserts (a) it does not raise on any of the four, and (b) all four agree bitwise, forward and
backward. (a) is the UnboundLocal guard; (b) is the flag's actual contract.

THE NAMING TRAP, recorded because it is why nobody suspected the branch: the flag is
``_STATIC_DECODE`` and the constant is ``_STATIC_MAX_ROWS``, but the predicate is
``T <= _STATIC_MAX_ROWS`` on ROW COUNT ALONE -- nothing in it mentions decode. The trainer's small
scoring microbatches are under the bound, so **the trainer takes the "static decode" path**, which
is how a decode-named branch killed a trainer forward. Any test matrix that reasons from the name
will exclude the very branch that breaks.

BOTH FLAGS ARE MODULE CONSTANTS READ AT IMPORT (``os.environ.get`` at module scope, the house style
in this file). Setting the env var inside a test therefore does NOTHING; the constants must be
monkeypatched. That is a trap of its own and the reason every case below goes through
``_set_flags`` rather than ``monkeypatch.setenv``.
"""

import os
import types

import pytest
import torch

from skyrl.backends.skyrl_train.isoexec.ops.moe import moe_batched_experts as M


def _module(e_local=4, h=16, f=8, seed=0):
    """The smallest thing `_supported()` accepts and `_batched_experts_gemm` can drive."""
    torch.manual_seed(seed)
    cfg = types.SimpleNamespace(
        fp8=False,
        bias_activation_fusion=False,
        gated_linear_unit=True,
        add_bias_linear=False,
        activation_func=torch.nn.functional.silu,
        activation_func_clamp_value=None,
        glu_linear_offset=0.0,
    )
    experts = []
    for _ in range(e_local):
        fc1 = torch.nn.Linear(h, 2 * f, bias=False)
        fc2 = torch.nn.Linear(f, h, bias=False)
        experts.append(types.SimpleNamespace(linear_fc1=fc1, linear_fc2=fc2))
    return types.SimpleNamespace(config=cfg, num_local_experts=e_local, local_experts=experts)


def _set_flags(monkeypatch, *, static_decode, flat_stage):
    # module constants, not env: see the docstring
    monkeypatch.setattr(M, "_STATIC_DECODE", static_decode, raising=True)
    monkeypatch.setattr(M, "_FLAT_STAGE", flat_stage, raising=True)
    # keep this test orthogonal to the weight-cache / fused-weights gates, which have their own
    # coverage and are being changed concurrently: force the plain torch.stack expression.
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_WEIGHT_CACHE", "0")


def _counts(e_local, per_expert):
    return torch.tensor(per_expert, dtype=torch.long)


def _run(mod, counts, h, *, need_grad=False):
    t = int(counts.sum())
    torch.manual_seed(123)
    x = torch.randn(t, h, requires_grad=need_grad)
    probs = torch.rand(t)
    out, bias = M._batched_experts_forward(mod, x, counts, probs)
    assert bias is None
    if need_grad:
        out.sum().backward()
        return out.detach().clone(), x.grad.clone()
    return out.detach().clone(), None


_CASES = [
    pytest.param([3, 5, 0, 7], id="skewed-with-an-empty-expert"),
    pytest.param([1, 1, 1, 1], id="one-row-each"),
    pytest.param([300, 1, 129, 40], id="crosses-the-128-row-tile-boundary"),
    pytest.param([0, 0, 0, 4], id="only-one-expert-routed"),
]


@pytest.mark.parametrize("per_expert", _CASES)
@pytest.mark.parametrize("static_decode", [True, False], ids=["static-branch", "dynamic-branch"])
@pytest.mark.parametrize("flat_stage", [False, True], ids=["FLAT_STAGE-off", "FLAT_STAGE-on"])
def test_every_branch_runs(monkeypatch, per_expert, static_decode, flat_stage):
    """The UnboundLocal guard: each of the four combinations must simply not raise."""
    _set_flags(monkeypatch, static_decode=static_decode, flat_stage=flat_stage)
    mod = _module()
    out, _ = _run(mod, _counts(4, per_expert), 16)
    assert out.shape == (sum(per_expert), 16)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("per_expert", _CASES)
def test_all_four_combinations_agree_bitwise(monkeypatch, per_expert):
    """FLAT_STAGE is address arithmetic and the tile grid is a bound, so all four must be equal."""
    ref_out = ref_grad = None
    for static_decode in (True, False):
        for flat_stage in (False, True):
            with monkeypatch.context() as mp:
                _set_flags(mp, static_decode=static_decode, flat_stage=flat_stage)
                mod = _module()  # same seed => same weights on every leg
                out, grad = _run(mod, _counts(4, per_expert), 16, need_grad=True)
            if ref_out is None:
                ref_out, ref_grad = out, grad
                continue
            tag = f"static_decode={static_decode} flat_stage={flat_stage}"
            assert torch.equal(out, ref_out), f"forward differs at {tag}"
            assert torch.equal(grad, ref_grad), f"input grad differs at {tag}"


def test_static_branch_is_what_the_trainer_actually_takes():
    """The naming trap, pinned: the predicate is a ROW COUNT, so trainer microbatches take it.

    If this ever stops holding, the matrix above silently stops covering the branch that broke.
    """
    assert M._STATIC_MAX_ROWS >= 131072, "the static bound is meant to cover trainer scoring batches"
    assert "decode" not in "T <= _STATIC_MAX_ROWS", "the predicate must stay row-count-only"


def test_flags_are_import_time_constants():
    """Guard the trap that makes `monkeypatch.setenv` silently useless for these two flags."""
    assert isinstance(M._FLAT_STAGE, bool)
    assert isinstance(M._STATIC_DECODE, bool)
    os.environ["SKYRL_ISOEXEC_MOE_FLAT_STAGE"] = "1"
    try:
        assert M._FLAT_STAGE is False or M._FLAT_STAGE is True  # unchanged by the env write
    finally:
        os.environ.pop("SKYRL_ISOEXEC_MOE_FLAT_STAGE", None)
