"""CPU regression test for the routing-mechanics INSTALL PLUMBING (no GPU, no Triton).

The bitwise gate for the kernels themselves is
the private repo's nightly ``moe_routing_mechanics_test.py`` and it needs a GPU. What is tested HERE is
the part most likely to rot silently and least likely to be noticed when it does: the two-level
patch that carries the engine INSTANCE MARK into a MODULE-LEVEL function.

``TopKRouter.routing`` has a ``self`` and therefore a mark; ``topk_routing_with_score_function``
does the work and has neither. ``install_dense_scatter`` bridges them with a scope flag published
for the duration of one synchronous routing call. If that bridge breaks in either direction the
failure is silent and asymmetric:

  * scope never set   -> the fused path is dead, the flag is inert, an A/B measures a baseline
                         against itself (four flags in this program already did exactly that);
  * scope never reset -> a TRAINER router in the same process (colocated under
                         ``VLLM_ENABLE_V1_MULTIPROCESSING=0``) takes a path with no
                         ``autograd.Function`` and its MoE backward is severed with the
                         forward-only gate green throughout -- the failure that froze 32B of 35B
                         params for five live steps.

Neither shows up in a numerics gate. Both show up here, and on CPU, because
``dense_can_handle`` refuses a non-CUDA tensor and the wrapper falls back to
``ref_dense_from_topk`` -- so the PLUMBING runs end to end while the kernel never launches.
"""

from __future__ import annotations

import pytest
import torch

torch_moe = pytest.importorskip("megatron.core.transformer.moe.router")

from skyrl.backends.skyrl_train.isoexec.ops.moe.moe_dense_scatter_kernel import (  # noqa: E402
    dense_can_handle,
    install_dense_scatter,
    permute_sort_active,
    ref_dense_from_topk,
)

E, K, T = 32, 4, 64


def _reference(logits, bias):
    """The eager reference, reached through ``moe_utils`` -- which the fixture has already wrapped
    in production's ``sorted=True`` forcing, and which ``install_dense_scatter`` does NOT patch (it
    patches the ``router`` module's binding). So this is the arm the fused path must match."""
    from megatron.core.transformer.moe import moe_utils

    return moe_utils.topk_routing_with_score_function(
        logits,
        K,
        use_pre_softmax=False,
        scaling_factor=1.8,
        score_function="sigmoid",
        expert_bias=bias,
    )


class _Cfg:
    num_moe_experts = E
    moe_router_pre_softmax = False
    moe_router_group_topk = None
    moe_router_num_groups = None
    moe_router_topk_scaling_factor = 1.8
    moe_expert_capacity_factor = None
    moe_router_fusion = False
    moe_router_enable_expert_bias = True


class _Router:
    """The minimum surface ``TopKRouter.routing`` touches on the no-grad inference path."""

    def __init__(self, bias):
        self.config = _Cfg()
        self.topk = K
        self.score_function = "sigmoid"
        self.routing_type = "topk"
        self.router_replay = None
        self.expert_bias = bias
        self.training = False
        self.enable_expert_bias = True

    def apply_z_loss(self, x, padding_mask=None):
        return x

    def sinkhorn_load_balancing(self, x):  # pragma: no cover - must never be reached
        raise AssertionError("sinkhorn is out of envelope and must not be selected")

    def _apply_expert_bias(self, routing_map, padding_mask=None):
        pass

    def is_aux_loss_enabled(self):
        return False


_sorted_forcing_installed = False


def _install_production_order():
    """Reproduce ``prepare_isoexec_moe``'s install ORDER, because it is load-bearing here.

    ``enable_moe_deterministic_ops`` runs FIRST and wraps ``topk_routing_with_score_function`` so
    the router's ``torch.topk(..., sorted=torch.is_grad_enabled())`` is forced to ``sorted=True``
    in both grad modes; ``install_dense_scatter`` then wraps THAT. Skipping the first step does not
    merely change a launch config -- see ``test_sorted_forcing_is_load_bearing_on_this_router``,
    which measures 23/2048 probs moving without it.

    Only the sorted-forcing half is installed here: the rest of ``enable_moe_deterministic_ops``
    registers CUDA aten overrides that a CPU test has no business touching.
    """
    global _sorted_forcing_installed
    from megatron.core.transformer.moe import moe_utils
    from megatron.core.transformer.moe import router as router_mod

    from skyrl.backends.skyrl_train.isoexec.ops.moe.moe_batch_invariant import (
        _make_sorted_topk_routing,
    )

    if not _sorted_forcing_installed:
        wrapped = _make_sorted_topk_routing(moe_utils.topk_routing_with_score_function)
        moe_utils.topk_routing_with_score_function = wrapped
        router_mod.topk_routing_with_score_function = wrapped
        _sorted_forcing_installed = True
    assert install_dense_scatter()


@pytest.fixture()
def fixture():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(T, E, dtype=torch.float32, generator=g)
    bias = torch.randn(E, dtype=torch.float32, generator=g) * 0.05
    _install_production_order()
    return logits, bias


def test_sorted_forcing_is_load_bearing_on_this_router():
    """POSITIVE CONTROL, and the sharpest evidence in this file for why a fused top-k is refused.

    megatron calls ``torch.topk(scores, k, dim=1, sorted=torch.is_grad_enabled())``. On an ENGINE
    forward -- ``torch.inference_mode()`` -- that is ``sorted=False``, so the top-k comes back in a
    DIFFERENT COLUMN ORDER than it does with grad on. For a softmax router that would be harmless
    (the denominator sums the selected scores themselves, and permuting them permutes equal-length
    sequences of the same values). For THIS router it is not harmless, because the denominator sums
    ``gather(scores, top_indices)`` while selection ran on ``scores + expert_bias``.

    Measured below on raw megatron, no zero-KL kernels involved at all:

        membership (routing_map) : IDENTICAL
        probs                    : 23 / 2048 elements differ

    That is the same mechanism the tie counterexample exercises, arriving here through a plain
    launch-config difference rather than an exact tie -- which is why
    ``SKYRL_ISOEXEC_MOE_DETERMINISTIC``'s ``sorted=True`` forcing is a correctness requirement on
    this model and not a determinism nicety, and why any Triton top-k would have to reproduce
    ATen's column order exactly rather than merely its membership.

    Written against ``torch.topk`` directly rather than against megatron's function, so it cannot
    be perturbed by whether some other test in the session has already installed the forcing
    wrapper -- a global-state dependency would make this control quietly vacuous, which is the one
    thing a control must never be.
    """
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(T, E, dtype=torch.float32, generator=g)
    bias = torch.randn(E, dtype=torch.float32, generator=g) * 0.05

    # megatron's sigmoid + expert_bias branch, spelled out (moe_utils.py:800-818).
    scores = torch.sigmoid(logits.float())
    key = scores + bias.float()

    def finish(sorted_flag):
        _, idx = torch.topk(key, k=K, dim=1, sorted=sorted_flag)
        sel = torch.gather(scores, dim=1, index=idx)
        return (sel / (sel.sum(dim=-1, keepdim=True) + 1e-20)) * 1.8, idx

    p_sorted, i_sorted = finish(True)  # what a grad-enabled forward gets
    p_unsorted, i_unsorted = finish(False)  # what an inference_mode forward gets

    assert torch.equal(
        i_sorted.sort(dim=1).values, i_unsorted.sort(dim=1).values
    ), "membership must not move; only the column order does"
    n_diff = int((p_sorted != p_unsorted).sum())
    assert n_diff > 0, (
        "POSITIVE CONTROL IS VACUOUS: column order no longer changes this router's probs. "
        "If this ever fires, re-examine the tie argument in moe_dense_scatter_kernel's docstring -- "
        "the whole reason torch.topk is left in place is that its column order is observable here."
    )


def test_envelope_refuses_cpu_so_this_file_exercises_plumbing_not_kernels():
    idx = torch.zeros(4, K, dtype=torch.int64)
    prb = torch.zeros(4, K, dtype=torch.float32)
    assert not dense_can_handle(idx, prb, E), "a CPU tensor must fall back, never launch Triton"


def test_marked_router_is_bitwise_equal_to_megatron(monkeypatch, fixture):
    logits, bias = fixture
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_DENSE_SCATTER", "1")
    want_p, want_m = _reference(logits, bias)

    r = _Router(bias)
    r._isoexec_routing_mechanics = True
    got_p, got_m = torch_moe.TopKRouter.routing(r, logits.view(T, 1, E))

    assert torch.equal(got_p, want_p)
    assert torch.equal(got_m, want_m)
    assert got_m.dtype is torch.bool


def test_guards_hold_under_inference_mode_not_just_no_grad(monkeypatch, fixture):
    """Run the guards where vLLM actually runs them: inside ``torch.inference_mode()``.

    THIS IS NOT A FORMALITY. Two ops on this stack shipped eligibility guards that walked
    ``tensor._base`` to recognise a marked tensor, and both were silently dead in the engine for
    their entire shipped history while printing green install banners -- because
    ``inference_mode`` makes tensors INFERENCE TENSORS, for which torch skips view tracking
    entirely (``_is_view()`` False, ``_base`` None). The bug does not reproduce under
    ``no_grad``, which is what those guards were tested under.

    This module's guards are structurally immune: they read plain Python attributes on the
    ROUTER and DISPATCHER modules (``_isoexec_routing_mechanics``), an env var, and
    ``torch.is_grad_enabled()`` -- no tensor identity, no view chain, no autograd metadata. But
    "structurally immune" is exactly the argument that made the other two look fine, so it is
    asserted here instead of believed.
    """
    from skyrl.backends.skyrl_train.isoexec.ops.moe import moe_dense_scatter_kernel as M

    logits, bias = fixture
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_DENSE_SCATTER", "1")
    want_p, want_m = _reference(logits, bias)

    with torch.inference_mode():
        r = _Router(bias)
        r._isoexec_routing_mechanics = True
        assert M._dense_active(r), "the guard must still fire on an inference-mode forward"
        got_p, got_m = torch_moe.TopKRouter.routing(r, logits.view(T, 1, E))
        assert torch.equal(got_p, want_p)
        assert torch.equal(got_m, want_m)
        assert M._dense_scope is False

        # ...and the refusals must still refuse, i.e. the guard is not merely always-True here.
        u = _Router(bias)
        assert not M._dense_active(u), "an unmarked router must still be refused under inference_mode"

    # the dispatcher predicate too
    class _Disp:
        ep_size = 1

    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_PERMUTE_SORT", "1")
    with torch.inference_mode():
        d = _Disp()
        assert not M.permute_sort_active(d)
        d._isoexec_routing_mechanics = True
        assert M.permute_sort_active(d)
        d.ep_size = 2
        assert not M.permute_sort_active(d)


def test_unmarked_router_takes_megatrons_path(monkeypatch, fixture):
    """The trainer's routers are never marked, and the flag must not reach them."""
    logits, bias = fixture
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_DENSE_SCATTER", "1")
    want_p, want_m = _reference(logits, bias)

    r = _Router(bias)  # deliberately NOT marked
    got_p, got_m = torch_moe.TopKRouter.routing(r, logits.view(T, 1, E))
    assert torch.equal(got_p, want_p) and torch.equal(got_m, want_m)


def test_flag_off_is_inert(monkeypatch, fixture):
    logits, bias = fixture
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_DENSE_SCATTER", "0")
    want_p, want_m = _reference(logits, bias)
    r = _Router(bias)
    r._isoexec_routing_mechanics = True
    got_p, got_m = torch_moe.TopKRouter.routing(r, logits.view(T, 1, E))
    assert torch.equal(got_p, want_p) and torch.equal(got_m, want_m)


def test_scope_is_reset_after_every_call(monkeypatch, fixture):
    """The half of the bridge whose failure severs a trainer backward, not the half that is slow."""
    from skyrl.backends.skyrl_train.isoexec.ops.moe import moe_dense_scatter_kernel as M

    logits, bias = fixture
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_DENSE_SCATTER", "1")
    assert M._dense_scope is False
    r = _Router(bias)
    r._isoexec_routing_mechanics = True
    torch_moe.TopKRouter.routing(r, logits.view(T, 1, E))
    assert M._dense_scope is False, "a leaked scope makes the NEXT router -- possibly the trainer's -- fuse"

    # ...including when the routing call raises part-way through.
    class _Boom(_Router):
        def apply_z_loss(self, x, padding_mask=None):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        b = _Boom(bias)
        b._isoexec_routing_mechanics = True
        torch_moe.TopKRouter.routing(b, logits.view(T, 1, E))
    assert M._dense_scope is False


@pytest.mark.parametrize(
    "attr,value",
    [
        ("moe_expert_capacity_factor", 1.5),  # token dropping rewrites the dense tensors afterwards
        ("moe_router_group_topk", 4),  # a different routing chain entirely
        ("moe_router_fusion", True),  # TE's fused router; not this function
    ],
)
def test_out_of_envelope_configs_decline(monkeypatch, fixture, attr, value):
    from skyrl.backends.skyrl_train.isoexec.ops.moe import moe_dense_scatter_kernel as M

    logits, bias = fixture
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_DENSE_SCATTER", "1")
    r = _Router(bias)
    r._isoexec_routing_mechanics = True
    assert M._dense_active(r)
    setattr(r.config, attr, value)
    assert not M._dense_active(r)
    setattr(r.config, attr, getattr(_Cfg, attr))


def test_permute_sort_predicate(monkeypatch):
    """The flag split: independent of ROUTER_O2, and the EP=1 guard survives it."""

    class _Disp:
        ep_size = 1

    d = _Disp()
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_ROUTER_O2", "0")
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_PERMUTE_SORT", "1")
    assert not permute_sort_active(d), "an unmarked (trainer) dispatcher must never fuse"
    d._isoexec_routing_mechanics = True
    assert permute_sort_active(d)

    from skyrl.backends.skyrl_train.isoexec.ops.moe.moe_router_o2_kernel import (
        dispatch_o2_active,
        router_o2_enabled,
    )

    assert not router_o2_enabled(), "PERMUTE_SORT must not switch on the O2 ROUTER"
    assert not dispatch_o2_active(d)

    d.ep_size = 2
    assert not permute_sort_active(d), "EP>1: megatron's argsort slice spills past the True group"

    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_PERMUTE_SORT", "0")
    d.ep_size = 1
    assert not permute_sort_active(d)


def test_ref_dense_matches_megatron_in_both_determinism_branches(fixture):
    """The reference the fused kernel is gated against IS megatron's dense build, both branches."""
    from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function

    logits, bias = fixture
    probs, top_indices = topk_routing_with_score_function(
        logits,
        K,
        use_pre_softmax=False,
        scaling_factor=1.8,
        score_function="sigmoid",
        expert_bias=bias,
        dense_output=True,
    )
    xp, xm = ref_dense_from_topk(top_indices, probs, E)
    for det in (False, True):
        prev = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(det, warn_only=True)
        try:
            rp, rm = _reference(logits, bias)
        finally:
            torch.use_deterministic_algorithms(prev, warn_only=True)
        assert torch.equal(xp, rp), f"deterministic={det}"
        assert torch.equal(xm, rm), f"deterministic={det}"

    # the precondition the dense build rests on: torch.topk returns K DISTINCT columns
    srt = top_indices.sort(dim=1).values
    assert int((srt[:, 1:] == srt[:, :-1]).sum()) == 0
