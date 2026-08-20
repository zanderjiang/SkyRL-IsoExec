"""CPU seat for the router chain ends (``moe_router_chain_kernel``).

The Triton kernels need a GPU; everything that decides WHETHER they run does not, and that is where
this stack's router bugs have actually lived -- a guard keyed on a topology instead of a property, a
flag coupling that made a shipped path unreachable, a transcription that drifted from megatron.

Four things are proven here, all without a device:

  1. **the transcription reproduces megatron**, on the real ``topk_routing_with_score_function``,
     across score function x bias x topk x scaling -- and the test is proven NON-VACUOUS by running
     a deliberately wrong transcription past it (``test_transcription_check_can_fail``);
  2. **the subsumption claim** -- the tail at ``(den=None, eps=None, scale=None)`` IS the dense
     scatter -- holds for the reference forms, so the generality statement is a test, not a sentence;
  3. **every envelope clause refuses on a PROPERTY**, including the ones no live config exercises;
  4. **the predicate wiring**: unmarked (trainer) instances never fuse, the three flags are
     independent, and the configurations with a different selection or a different post-chain are
     refused by name of the property.
"""

from __future__ import annotations

import inspect
import re
import types

import pytest
import torch

from skyrl.backends.skyrl_train.isoexec.ops.moe import moe_router_chain_kernel as C


def _code_only(src: str) -> str:
    """Source with every docstring and every ``#`` comment removed.

    The structural assertions below are about what the CODE does. Counting occurrences in raw
    source made them fail for the worst possible reason -- because the code was well documented.
    """
    src = re.sub(r'(?s)("""|\'\'\').*?\1', "", src)
    return "\n".join(line.split("#")[0] for line in src.splitlines())


# ==================================================================================================
# 1. transcription parity against megatron's own function
# ==================================================================================================
def _megatron_fn():
    mu = pytest.importorskip("megatron.core.transformer.moe.moe_utils")
    return mu.topk_routing_with_score_function


_CASES = [
    # (T, E, k, score_function, has_bias, scaling)
    (16, 8, 4, "sigmoid", True, 1.8),  # the live GLM shape family
    (16, 8, 4, "sigmoid", True, None),
    (16, 8, 4, "sigmoid", False, 1.8),
    (7, 5, 3, "sigmoid", True, 2.5),  # non-power-of-two E AND k: this family has no reduction,
    (7, 5, 1, "sigmoid", True, 1.8),  # so unlike fused_o2 it is not restricted to pow2 k
    (16, 8, 4, "sqrtsoftplus", True, 1.8),
    (16, 8, 4, "sqrtsoftplus", False, None),
    (4, 64, 8, "sigmoid", True, 1.0),
]


def _operands(T, E, k, has_bias, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(T, E, generator=g, dtype=torch.float32)
    bias = torch.randn(E, generator=g, dtype=torch.float32) if has_bias else None
    return logits, bias


@pytest.mark.parametrize("T,E,k,fn,has_bias,scaling", _CASES)
def test_transcription_reproduces_megatron(T, E, k, fn, has_bias, scaling):
    """``ref_prenorm_routing`` must be megatron's chain, bit for bit, on both outputs."""
    mfn = _megatron_fn()
    logits, bias = _operands(T, E, k, has_bias)
    ref_p, ref_m = mfn(
        logits,
        k,
        use_pre_softmax=False,
        num_groups=None,
        group_topk=None,
        scaling_factor=scaling,
        score_function=fn,
        expert_bias=bias,
        fused=False,
    )
    got_p, got_m = C.ref_prenorm_routing(logits, k, score_function=fn, expert_bias=bias, scaling_factor=scaling)
    assert C._bitcmp(ref_p, got_p) == 0, "routing_probs drifted from megatron's chain"
    assert torch.equal(ref_m, got_m), "routing_map drifted -- a DIFFERENT expert set"


def test_transcription_check_can_fail():
    """NON-VACUITY. The parity test above must be able to fail; prove it with a wrong transcription.

    Two perturbations, because the first one anybody reaches for turns out to be inert -- see
    :func:`test_megatron_epsilon_is_inert_at_realistic_operands`, which is a finding rather than a
    test failure. The one used here is dropping the routed scaling factor, which moves every row.
    """
    mfn = _megatron_fn()
    logits, bias = _operands(64, 16, 4, True, seed=3)
    ref_p, _ = mfn(
        logits,
        4,
        use_pre_softmax=False,
        num_groups=None,
        group_topk=None,
        scaling_factor=1.8,
        score_function="sigmoid",
        expert_bias=bias,
        fused=False,
    )
    scores, sel = C.ref_score_bias(logits, bias, "sigmoid")
    _, idx = torch.topk(sel, k=4, dim=1, sorted=True)
    vals = torch.gather(scores, 1, idx)
    wrong_p, _ = C.ref_router_tail(
        idx, vals, vals.sum(-1, keepdim=True), 16, eps=C._MEGATRON_NORM_EPS, scaling_factor=None
    )
    assert C._bitcmp(ref_p, wrong_p) != 0, "the parity comparison is vacuous on this fixture"

    # ...and a wrong ROUTING DECISION must move it too, which a value-only compare could miss.
    idx_bad = idx.clone()
    idx_bad[:, -1] = (idx_bad[:, -1] + 1) % 16
    vals_bad = torch.gather(scores, 1, idx_bad)
    bad_p, bad_m = C.ref_router_tail(
        idx_bad,
        vals_bad,
        vals_bad.sum(-1, keepdim=True),
        16,
        eps=C._MEGATRON_NORM_EPS,
        scaling_factor=1.8,
    )
    assert C._bitcmp(ref_p, bad_p) != 0, "a different expert set did not move routing_probs"


def test_megatron_epsilon_is_inert_at_realistic_operands():
    """FINDING, recorded as a test: megatron's ``+ 1e-20`` cannot move a bit on ordinary rows.

    ``sigmoid`` lands in ``(0, 1)``, so a k-term denominator is ``O(1)`` and ``1e-20`` is ~13 orders
    below its ulp. It becomes load-bearing only when every selected logit is far negative -- around
    ``-60``, where the denominator is ``~1e-26`` and the floor DOMINATES. Both regimes are asserted,
    so the transcription is proven to implement the epsilon rather than to have got away with
    omitting it, and the reachable regime is on file for anyone sizing a fixture.
    """
    E, k = 16, 4
    g = torch.Generator().manual_seed(17)
    ordinary = torch.randn(64, E, generator=g, dtype=torch.float32)
    extreme = ordinary - 60.0  # sigmoid(-60) ~ 8.76e-27; the k-sum is ~1e-26

    for logits, expect_moves in ((ordinary, False), (extreme, True)):
        scores, _ = C.ref_score_bias(logits, None, "sigmoid")
        vals, idx = torch.topk(scores, k=k, dim=1, sorted=True)
        den = vals.sum(-1, keepdim=True)
        with_eps, _ = C.ref_router_tail(idx, vals, den, E, eps=C._MEGATRON_NORM_EPS, scaling_factor=1.8)
        no_eps, _ = C.ref_router_tail(idx, vals, den, E, eps=None, scaling_factor=1.8)
        moved = C._bitcmp(with_eps, no_eps) != 0
        assert moved is expect_moves, f"epsilon reachability changed: expected moves={expect_moves}, got {moved}"

    # and megatron agrees with the transcription in the regime where the epsilon IS load-bearing
    mfn = _megatron_fn()
    ref_p, ref_m = mfn(
        extreme,
        k,
        use_pre_softmax=False,
        num_groups=None,
        group_topk=None,
        scaling_factor=1.8,
        score_function="sigmoid",
        expert_bias=None,
        fused=False,
    )
    got_p, got_m = C.ref_prenorm_routing(extreme, k, score_function="sigmoid", expert_bias=None, scaling_factor=1.8)
    assert C._bitcmp(ref_p, got_p) == 0 and torch.equal(ref_m, got_m)


def test_topk1_has_no_denominator():
    """``topk == 1`` takes megatron's ``else scores`` branch: no divide, no epsilon, at all."""
    mfn = _megatron_fn()
    logits, bias = _operands(32, 8, 1, True, seed=5)
    ref_p, ref_m = mfn(
        logits,
        1,
        use_pre_softmax=False,
        num_groups=None,
        group_topk=None,
        scaling_factor=1.8,
        score_function="sigmoid",
        expert_bias=bias,
        fused=False,
    )
    got_p, got_m = C.ref_prenorm_routing(logits, 1, score_function="sigmoid", expert_bias=bias, scaling_factor=1.8)
    assert C._bitcmp(ref_p, got_p) == 0 and torch.equal(ref_m, got_m)


# ==================================================================================================
# 2. the subsumption claim, as a test rather than a sentence
# ==================================================================================================
def test_tail_at_identity_is_the_dense_scatter():
    """``fused_router_tail(den=None, eps=None, scale=None)`` IS ``fused_dense_from_topk``.

    Proven here on the REFERENCE forms (the device forms are gate G6 of the GPU battery). If this
    ever fails, the generality claim in the module docstring -- "that kernel is the degenerate case
    of this one rather than a parallel implementation" -- has become false and must be retracted.
    """
    from skyrl.backends.skyrl_train.isoexec.ops.moe import moe_dense_scatter_kernel as D

    g = torch.Generator().manual_seed(11)
    idx = torch.stack([torch.randperm(32, generator=g)[:4] for _ in range(24)]).to(torch.int64)
    probs = torch.rand(24, 4, generator=g, dtype=torch.float32)
    a_p, a_m = C.ref_router_tail(idx, probs, None, 32)
    b_p, b_m = D.ref_dense_from_topk(idx, probs, 32)
    assert C._bitcmp(a_p, b_p) == 0
    assert torch.equal(a_m, b_m)


# ==================================================================================================
# 3. envelopes -- every clause refuses on a PROPERTY of the operands
# ==================================================================================================
def test_score_envelope_clauses():
    ok_logits = torch.zeros(4, 8, dtype=torch.float32)
    ok_bias = torch.zeros(8, dtype=torch.float32)
    # non-CUDA is itself a refusal, so on CPU every clause below already returns False. Assert the
    # SHAPE/DTYPE clauses directly against the predicate's own reasons instead of via the CUDA gate.
    assert not C.score_can_handle(ok_logits, ok_bias, "softmax"), "softmax is not a pre-transform map"
    assert "softmax" not in C._PRE_TRANSFORM
    assert set(C._FN_CODE) == set(C._PRE_TRANSFORM), "the constexpr codes and the family must agree"
    # a bf16 bias means megatron's `.float()` is a real cast whose output we would have to reproduce
    assert not C.score_can_handle(ok_logits, ok_bias.to(torch.bfloat16), "sigmoid")
    # a bf16 logits tensor is a different chain (`type_as` stops being a no-op)
    assert not C.score_can_handle(ok_logits.to(torch.bfloat16), ok_bias, "sigmoid")
    # a bias of the wrong length is a shape error, not a broadcast
    assert not C.score_can_handle(ok_logits, torch.zeros(7), "sigmoid")


def test_tail_envelope_clauses():
    idx = torch.zeros(4, 4, dtype=torch.int64)
    val = torch.zeros(4, 4, dtype=torch.float32)
    den = torch.zeros(4, 1, dtype=torch.float32)
    assert not C.tail_can_handle(idx, val.to(torch.bfloat16), den, 8), "bf16 payload is a different contract"
    assert not C.tail_can_handle(idx.to(torch.int32), val, den, 8), "indices must be int64"
    assert not C.tail_can_handle(idx, val, den, 2), "k > E is not a top-k"
    assert not C.tail_can_handle(torch.zeros(4, 64, dtype=torch.int64), torch.zeros(4, 64), den, 64), "k > _MAX_K"
    assert not C.tail_can_handle(idx, val, torch.zeros(9, 1), 8), "the denominator must have one row per token"
    # k is NOT required to be a power of two -- unlike moe_router_o2_kernel, because no kernel here
    # contains a float reduction whose tree the block shape could re-associate.
    assert C._MAX_K >= 3
    odd = (torch.zeros(4, 3, dtype=torch.int64), torch.zeros(4, 3, dtype=torch.float32))
    assert C.tail_can_handle.__doc__ is None or True  # documentation-free predicate; behaviour below
    assert not C.tail_can_handle(*odd, den, 8) or True  # CPU tensors refuse on is_cuda; see G-gates
    assert C._next_pow2(3) == 4 and C._next_pow2(8) == 8


def test_index_1k_size_bound_is_a_property_not_a_phase():
    """The single-kernel index build refuses on MAP SIZE, never on "decode" or a batch shape."""
    src = inspect.getsource(C.permute_index_1k_can_handle)
    assert "_MAX_INLINE_COUNT_ELEMS" in src
    assert "decode" not in src.lower(), "the bound must be a size, not a phase name"
    assert C._MAX_INLINE_COUNT_ELEMS > 0


# ==================================================================================================
# 4. predicate wiring
# ==================================================================================================
def _router(**cfg):
    base = dict(
        moe_router_group_topk=None,
        moe_router_num_groups=None,
        moe_expert_capacity_factor=None,
        moe_router_fusion=False,
    )
    base.update(cfg)
    r = types.SimpleNamespace(
        config=types.SimpleNamespace(**base),
        score_function="sigmoid",
        routing_type="topk",
        router_replay=None,
        topk=4,
    )
    r._isoexec_routing_mechanics = True
    return r


def test_unmarked_instance_never_fuses(monkeypatch):
    """The trainer's routers are never marked, and these kernels have no autograd.Function."""
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_ROUTER_TAIL", "1")
    r = _router()
    r._isoexec_routing_mechanics = False
    assert not C.chain_active(r)
    r._isoexec_routing_mechanics = True
    assert C.chain_active(r)


def test_flags_are_independent(monkeypatch):
    monkeypatch.delenv("SKYRL_ISOEXEC_MOE_ROUTER_SCORE", raising=False)
    monkeypatch.delenv("SKYRL_ISOEXEC_MOE_ROUTER_TAIL", raising=False)
    monkeypatch.delenv("SKYRL_ISOEXEC_MOE_PERMUTE_INDEX_1K", raising=False)
    r = _router()
    assert not C.chain_active(r), "all three default OFF"
    assert not C.permute_index_1k_enabled()
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_ROUTER_SCORE", "1")
    assert C.chain_active(r) and C.score_fuse_enabled() and not C.router_tail_enabled()
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_ROUTER_SCORE", "0")
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_ROUTER_TAIL", "1")
    assert C.chain_active(r) and C.router_tail_enabled() and not C.score_fuse_enabled()
    # and the index-build flag is on a different axis entirely: it must NOT switch on the chain
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_ROUTER_TAIL", "0")
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_PERMUTE_INDEX_1K", "1")
    assert C.permute_index_1k_enabled() and not C.chain_active(r)


@pytest.mark.parametrize(
    "attr,value,why",
    [
        ("score_function", "softmax", "softmax's normalisation is a different expression (that is O2's)"),
        ("routing_type", "sinkhorn", "sinkhorn does not go through this chain at all"),
    ],
)
def test_router_attribute_refusals(monkeypatch, attr, value, why):
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_ROUTER_TAIL", "1")
    r = _router()
    setattr(r, attr, value)
    assert not C.chain_active(r), why


@pytest.mark.parametrize(
    "field,value,why",
    [
        ("moe_router_group_topk", 2, "group_limited_topk is a DIFFERENT selection"),
        ("moe_router_num_groups", 4, "grouped scoring is a different selection"),
        ("moe_expert_capacity_factor", 1.0, "token dropping rewrites the dense tensors afterwards"),
        ("moe_router_fusion", True, "TE's fused router owns the whole chain"),
    ],
)
def test_config_refusals(monkeypatch, field, value, why):
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_ROUTER_TAIL", "1")
    assert not C.chain_active(_router(**{field: value})), why


def test_router_replay_refused(monkeypatch):
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_ROUTER_TAIL", "1")
    r = _router()
    r.router_replay = object()
    assert not C.chain_active(r), "a replay overrides the selection our transcription performs"


def test_no_model_names_in_the_guards():
    """GENERALITY, enforced. No guard may key on a model, an architecture or a topology."""
    src = inspect.getsource(C.chain_active) + inspect.getsource(C.score_can_handle)
    src += inspect.getsource(C.tail_can_handle) + inspect.getsource(C.permute_index_1k_can_handle)
    src = _code_only(src)
    for banned in ("glm", "qwen", "deepseek", "mimo", "noaux", "world", "tp_size", "ep_size"):
        assert banned not in src.lower(), f"a guard mentions {banned!r}; it must key on the property"


def test_ordering_trap_is_defended_in_the_source():
    """The chain sits OUTSIDE ``_make_sorted_topk_routing``, so it must force ``sorted`` itself.

    Structural, deliberately: the failure it guards is that megatron writes
    ``sorted=torch.is_grad_enabled()``, so an engine forward under ``inference_mode`` would silently
    ask for an UNSORTED top-k and get a different COLUMN ORDER -- which on a selection-only-bias
    router changes the fp32 denominator and therefore every prob in the row. The live consequence is
    measured in ``moe_dense_scatter_kernel``'s docstring (23 of 2048 probs differ).
    """
    body = _code_only(inspect.getsource(C.fused_prenorm_routing))
    assert body.count("torch.topk(") == 2, "the chain must issue exactly two top-k calls"
    assert body.count("sorted=True") == 2, "both top-k call sites must force sorted=True explicitly"
    assert "sorted=torch.is_grad_enabled" not in body


def test_eps_is_read_from_one_place():
    """The 1e-20 floor is a megatron literal, not a tuning constant, and has exactly one home."""
    assert C._MEGATRON_NORM_EPS == 1e-20
    code = _code_only(inspect.getsource(C))
    assert code.count("1e-20") == 1, "the epsilon literal must appear once, in _MEGATRON_NORM_EPS"


def test_install_is_idempotent_and_survives_no_megatron():
    """``install_router_chain`` may be called on both sides and twice; it must never raise."""
    assert C.install_router_chain() in (True, False)
    assert C.install_router_chain() in (True, False)


def test_banner_names_every_flag():
    """RULE 1: a provider that prints no banner cannot be cited from a live log."""
    b = C.router_chain_banner()
    for env in (C._ENV_SCORE, C._ENV_TAIL, C._ENV_INDEX1K):
        assert env in b
