"""CPU tests for the fused forward MoE permute (index build + row gather).

The permute is pure data movement, so acceptance is ``torch.equal`` on ``uint8`` views against
megatron's eager expression. The Triton kernels are checked by compiling them to TTIR for SM90.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest
import torch

from skyrl.backends.skyrl_train.isoexec.ops.moe import moe_fused_permute as P

# --- compat shims -------------------------------------------------------------------------------
# The public module ships the fused permute unconditionally, so these env names are inert here
# (setting them is a no-op); the stats accessors below stand in for stripped ones.
P.ENABLE_ENV = "SKYRL_ISOEXEC_MOE_FUSED_PERMUTE"
P.GATHER_ENV = "SKYRL_ISOEXEC_MOE_FUSED_PERMUTE_GATHER"
P.VALIDATE_ENV = "SKYRL_ISOEXEC_MOE_FUSED_PERMUTE_VALIDATE"
P.PREEMIT_ROWS_ENV = "SKYRL_ISOEXEC_MOE_PREEMIT_COMBINE_ROWS"


def _fused_permute_stats() -> dict:
    out = dict(P._COUNTS)
    out["decline_reasons"] = dict(P._DECLINE_REASONS)
    return out


def _reset_fused_permute_stats() -> None:
    for key in P._COUNTS:
        P._COUNTS[key] = 0
    P._DECLINE_REASONS.clear()


P.fused_permute_stats = _fused_permute_stats
P.reset_fused_permute_stats = _reset_fused_permute_stats
# -------------------------------------------------------------------------------------------------

from skyrl.backends.skyrl_train.isoexec.ops.moe import (  # noqa: E402
    moe_batch_invariant as MBI,
)


# =================================================================================================
# the adversarial population
# =================================================================================================
def _topk_map(num_tokens: int, num_experts: int, topk: int, seed: int) -> torch.Tensor:
    """Exact top-k routing: every token picks ``topk`` distinct experts."""
    g = torch.Generator().manual_seed(seed)
    routing_map = torch.zeros(num_tokens, num_experts, dtype=torch.bool)
    for t in range(num_tokens):
        routing_map[t, torch.randperm(num_experts, generator=g)[:topk]] = True
    return routing_map


def _skewed_map(num_tokens: int, num_experts: int, topk: int, seed: int) -> torch.Tensor:
    """Every token routes to the same topk experts: maximal skew, num_experts-topk empty."""
    routing_map = torch.zeros(num_tokens, num_experts, dtype=torch.bool)
    hot = [(seed + j) % num_experts for j in range(topk)]
    routing_map[:, hot] = True
    return routing_map


_CASES = [
    # (num_tokens, num_experts, topk, hidden, builder)
    (1, 1, 1, 1, _topk_map),  # degenerate everything
    (1, 8, 8, 3, _topk_map),  # one token, every expert, unaligned hidden
    (5, 3, 1, 2, _topk_map),  # topk == 1
    (7, 4, 4, 5, _topk_map),  # topk == num_experts (no empty expert possible)
    (13, 17, 8, 9, _topk_map),  # production top-k, prime dims
    (64, 256, 8, 7, _topk_map),  # production E/k, deliberately unaligned hidden
    (33, 16, 2, 129, _topk_map),  # hidden just past a 128 boundary
    (9, 12, 3, 4, _skewed_map),  # 9 empty experts
    (2, 64, 1, 1, _skewed_map),  # 63 empty experts, hidden 1
    (128, 5, 5, 65, _skewed_map),  # every expert full, odd hidden
]


@pytest.mark.parametrize("num_tokens,num_experts,topk,hidden,builder", _CASES)
def test_compaction_is_bitwise_megatrons_argsort_permutation(num_tokens, num_experts, topk, hidden, builder):
    """Compaction yields the same int64 permutation megatron's argsort does."""
    routing_map = builder(num_tokens, num_experts, topk, seed=num_tokens + num_experts)
    num_out_tokens = int(routing_map.sum())

    got_idx, got_expert, counts = P.compact_route_positions_reference(routing_map, num_out_tokens)
    ref_idx, ref_expert = P.megatron_permute_index_reference(routing_map, num_out_tokens)

    assert torch.equal(got_idx, ref_idx)
    assert torch.equal(got_expert, ref_expert)
    assert torch.equal(counts, routing_map.sum(dim=0).to(torch.int64))
    # expert-major, token-ascending within an expert: what the fixed-order combine and VJP rest on
    key = got_expert * num_tokens + got_idx
    assert torch.equal(key, torch.sort(key).values)
    assert torch.equal(torch.sort(key).values, torch.unique(key))


@pytest.mark.parametrize("num_tokens,num_experts,topk,hidden,builder", _CASES)
def test_preemitted_rows_are_exact_stable_argsort_inverse(num_tokens, num_experts, topk, hidden, builder):
    """Pre-emitted rows equal the stable-argsort inverse, element for element."""
    routing_map = builder(num_tokens, num_experts, topk, seed=hidden + 17)
    n = int(routing_map.sum())
    sorted_indices, route_expert, _ = P.compact_route_positions_reference(routing_map, n)
    rows = P.canonical_combine_rows_reference(routing_map, sorted_indices, route_expert)
    ref = torch.argsort(sorted_indices, stable=True).view(num_tokens, topk).to(torch.int32)
    assert torch.equal(rows, ref)


@pytest.mark.parametrize("num_tokens,num_experts,topk,hidden,builder", _CASES)
def test_gather_is_byte_identical_and_row_identity_holds(num_tokens, num_experts, topk, hidden, builder):
    routing_map = builder(num_tokens, num_experts, topk, seed=hidden)
    num_out_tokens = int(routing_map.sum())
    sorted_indices, _, _ = P.compact_route_positions_reference(routing_map, num_out_tokens)

    for dtype in (torch.bfloat16, torch.float32):
        tokens = torch.randn(num_tokens, hidden, dtype=dtype)
        got = P.permute_rows(tokens, sorted_indices)
        assert torch.equal(got.view(torch.uint8), tokens.index_select(0, sorted_indices).view(torch.uint8))
        # row identity: each routed row carries its own token's bytes, not just the right multiset
        for r in range(num_out_tokens):
            assert torch.equal(got[r].view(torch.uint8), tokens[int(sorted_indices[r])].view(torch.uint8))


def test_gather_preserves_nan_subnormal_and_signed_zero_payloads():
    """The gather is a copy, so NaN, subnormal and signed-zero payloads are not normalised."""
    routing_map = _topk_map(6, 8, 3, seed=5)
    num_out_tokens = int(routing_map.sum())
    sorted_indices, _, _ = P.compact_route_positions_reference(routing_map, num_out_tokens)
    payload = torch.tensor(
        [float("nan"), float("-nan"), float("inf"), -0.0, 0.0, 5e-324, -5e-324, 1.0],
        dtype=torch.float64,
    ).to(torch.float32)
    tokens = payload.view(1, -1).repeat(6, 1).contiguous()
    tokens[:, 0] = torch.arange(6, dtype=torch.float32)  # make rows distinguishable
    got = P.permute_rows(tokens, sorted_indices)
    assert torch.equal(got.view(torch.uint8), tokens.index_select(0, sorted_indices).view(torch.uint8))


@pytest.mark.parametrize("num_tokens,num_experts,topk,hidden,builder", _CASES)
def test_end_to_end_wrapper_is_bitwise_megatrons_eager_permute(
    monkeypatch, num_tokens, num_experts, topk, hidden, builder
):
    """The wrapper's returns match megatron's eager permute expression byte for byte."""
    monkeypatch.setenv(P.ENABLE_ENV, "1")
    routing_map = builder(num_tokens, num_experts, topk, seed=topk * 31 + hidden)
    num_out_tokens = int(routing_map.sum())
    tokens = torch.randn(num_tokens, hidden, dtype=torch.bfloat16)
    probs = torch.randn(num_tokens, num_experts, dtype=torch.float32) * routing_map

    got, got_probs, got_idx, pad, tpe = P._fused_permute(tokens, routing_map, probs, num_out_tokens)

    flat = routing_map.bool().T.contiguous().reshape(-1).argsort(descending=True, stable=True)[:num_out_tokens]
    assert torch.equal(got_idx, flat % num_tokens)
    assert torch.equal(got.view(torch.uint8), tokens.index_select(0, flat % num_tokens).view(torch.uint8))
    assert torch.equal(got_probs.view(torch.uint8), probs.T.contiguous().reshape(-1)[flat].view(torch.uint8))
    assert pad is None and tpe is None


def test_wrapper_returns_megatrons_tuple_shape_and_passes_tokens_per_expert_through(monkeypatch):
    monkeypatch.setenv(P.ENABLE_ENV, "1")
    routing_map = _topk_map(11, 13, 4, seed=3)
    tpe = torch.arange(13)
    out = P._fused_permute(torch.randn(11, 6), routing_map, None, 44, tokens_per_expert=tpe)
    assert len(out) == 5
    assert out[1] is None and out[3] is None and out[4] is tpe


def test_wrapper_preemits_and_consumer_reuses_rows_without_sort(monkeypatch):
    monkeypatch.setenv(P.ENABLE_ENV, "1")
    monkeypatch.setenv(P.PREEMIT_ROWS_ENV, "1")
    monkeypatch.setenv(P.GATHER_ENV, "kernel")
    P.reset_fused_permute_stats()
    routing_map = _skewed_map(19, 31, 8, seed=7)
    _, _, sorted_indices, _, _ = P._fused_permute(torch.randn(19, 5, dtype=torch.bfloat16), routing_map, None, 19 * 8)
    rows = P.get_preemitted_combine_rows(sorted_indices, 19)
    ref = torch.argsort(sorted_indices, stable=True).view(19, 8).to(torch.int32)
    assert torch.equal(rows, ref)
    stats = P.fused_permute_stats()
    assert stats["rows_emitted"] == 1 and stats["rows_consumed"] == 1


def test_pik_owner_reuses_the_same_preemitted_rows(monkeypatch):
    from skyrl.backends.skyrl_train.isoexec.ops.moe.moe_pik_combine_owner import (
        _build_rows,
    )

    routing_map = _topk_map(23, 29, 8, seed=12)
    sidx, experts, _ = P.compact_route_positions_reference(routing_map, 23 * 8)
    rows = P.canonical_combine_rows_reference(routing_map, sidx, experts)
    setattr(sidx, P._PREEMITTED_ROWS_ATTR, rows)
    # make reconstruction illegal: reaching it means the owner ignored the published rows
    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.isoexec.ops.moe.moe_combine_rows_kernel.stable_combine_rows",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("reconstructed rows")),
    )
    assert _build_rows(sidx, 23, 8) is rows


def test_malformed_published_rows_fail_closed(monkeypatch):
    sorted_indices = torch.arange(12, dtype=torch.int64)
    setattr(sorted_indices, P._PREEMITTED_ROWS_ATTR, torch.empty(3, 4, dtype=torch.int64))
    with pytest.raises(RuntimeError, match="malformed preemitted rows"):
        P.get_preemitted_combine_rows(sorted_indices, 3)


# =================================================================================================
# fail-closed envelope
# =================================================================================================
def _install_fake_megatron(monkeypatch, original):
    megatron = ModuleType("megatron")
    core = ModuleType("megatron.core")
    transformer = ModuleType("megatron.core.transformer")
    moe = ModuleType("megatron.core.transformer.moe")
    moe_utils = ModuleType("megatron.core.transformer.moe.moe_utils")
    token_dispatcher = ModuleType("megatron.core.transformer.moe.token_dispatcher")
    moe_utils.permute = original
    token_dispatcher.permute = original
    moe.moe_utils = moe_utils
    moe.token_dispatcher = token_dispatcher
    transformer.moe = moe
    core.transformer = transformer
    megatron.core = core
    for name, module in {
        "megatron": megatron,
        "megatron.core": core,
        "megatron.core.transformer": transformer,
        "megatron.core.transformer.moe": moe,
        "megatron.core.transformer.moe.moe_utils": moe_utils,
        "megatron.core.transformer.moe.token_dispatcher": token_dispatcher,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return moe_utils, token_dispatcher


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"fused": True}, "fused"),
        ({"drop_and_pad": True}, "drop_and_pad"),
        ({"num_out_tokens": None}, "num_out_tokens=None"),
    ],
)
def test_declines_loudly_and_falls_back_to_the_captured_binding(monkeypatch, kwargs, expected):
    monkeypatch.setenv(P.ENABLE_ENV, "1")
    P.reset_fused_permute_stats()
    seen = {}

    def original(*args, **kw):
        seen["hit"] = True
        return "FELL-BACK"

    monkeypatch.setattr(P, "_orig_permute", original)
    routing_map = _topk_map(4, 8, 2, seed=1)
    call = {"num_out_tokens": 8}
    call.update(kwargs)
    assert P._fused_permute(torch.randn(4, 3), routing_map, None, **call) == "FELL-BACK"
    assert seen.get("hit")
    # the public module dropped the decline-census increments, so check the reason directly
    full = {"num_out_tokens": None, "fused": False, "drop_and_pad": False}
    full.update(call)
    assert P._decline_reason(torch.randn(4, 3), routing_map, None, **full) == expected
    assert P.fused_permute_stats()["served"] == 0


def test_grad_without_exact_topk_declines_rather_than_guessing_a_vjp(monkeypatch):
    """A layout with no [T, topk] route rows declines instead of guessing a VJP."""
    monkeypatch.setenv(P.ENABLE_ENV, "1")
    P.reset_fused_permute_stats()
    monkeypatch.setattr(P, "_orig_permute", lambda *a, **k: "FELL-BACK")
    routing_map = torch.zeros(4, 8, dtype=torch.bool)
    routing_map[0, :3] = True
    routing_map[1, :2] = True  # 5 routed pairs over 4 tokens: not exact top-k
    x = torch.randn(4, 3, requires_grad=True)
    assert P._fused_permute(x, routing_map, None, 5) == "FELL-BACK"
    assert (
        P._decline_reason(x, routing_map, None, num_out_tokens=5, fused=False, drop_and_pad=False)
        == "grad_without_exact_topk"
    )
    # rows are emitted on every served call and are only lawful for an exact top-k layout, so a
    # no_grad non-exact-topk call raises inside row emission rather than being served
    P.reset_fused_permute_stats()
    with torch.no_grad(), pytest.raises(ValueError, match="exact-topk"):
        P._fused_permute(x, routing_map, None, 5)


def test_live_validation_raises_on_a_corrupted_permutation(monkeypatch):
    """A compaction that disagrees with the reference raises rather than degrading silently."""
    monkeypatch.setenv(P.ENABLE_ENV, "1")
    monkeypatch.setenv(P.VALIDATE_ENV, "4")
    P.reset_fused_permute_stats()
    routing_map = _topk_map(6, 8, 3, seed=2)
    good = P.compact_route_positions_reference

    def corrupted(rm, n):
        idx, exp, counts = good(rm, n)
        idx = idx.clone()
        idx[0], idx[1] = idx[1].clone(), idx[0].clone()
        return idx, exp, counts

    monkeypatch.setattr(P, "compact_route_positions", corrupted)
    with pytest.raises(RuntimeError, match="sorted_indices differ"):
        P._fused_permute(torch.randn(6, 4), routing_map, None, 18)


# =================================================================================================
# autograd wiring
# =================================================================================================
def test_custom_autograd_matches_an_exact_fp64_index_add_oracle():
    """The permute VJP equals an exact fp64 index_add oracle (integer values, so no reduction tree)."""
    num_tokens, num_experts, topk, hidden = 17, 24, 8, 11
    routing_map = _topk_map(num_tokens, num_experts, topk, seed=9)
    x = torch.randn(num_tokens, hidden, dtype=torch.float64, requires_grad=True)
    sorted_indices, _, _ = P.compact_route_positions_reference(routing_map, num_tokens * topk)
    cotangent = torch.randint(-1000, 1001, (num_tokens * topk, hidden)).to(torch.float64)

    out = P._FusedPermuteGather.apply(x, sorted_indices, routing_map, torch.empty_like(sorted_indices), False)
    out.backward(cotangent)
    ref = torch.zeros_like(x)
    ref.index_add_(0, sorted_indices, cotangent)
    assert torch.equal(x.grad, ref)


def test_probs_gradient_is_the_scatter_back_into_the_dense_map(monkeypatch):
    monkeypatch.setenv(P.ENABLE_ENV, "1")
    num_tokens, num_experts, topk = 13, 19, 4
    routing_map = _topk_map(num_tokens, num_experts, topk, seed=11)
    probs = torch.randn(num_tokens, num_experts, dtype=torch.float64, requires_grad=True)
    _, permuted_probs, sorted_indices, _, _ = P._fused_permute(
        torch.randn(num_tokens, 5, dtype=torch.float64), routing_map, probs, num_tokens * topk
    )
    cotangent = torch.randint(-100, 101, (num_tokens * topk,)).to(torch.float64)
    permuted_probs.backward(cotangent)

    _, route_expert = P.megatron_permute_index_reference(routing_map, num_tokens * topk)
    ref = torch.zeros(num_tokens * num_experts, dtype=torch.float64)
    ref.index_add_(0, sorted_indices * num_experts + route_expert, cotangent)
    assert torch.equal(probs.grad, ref.view(num_tokens, num_experts))


def test_preemitted_combine_vjp_and_gradient_accumulation_are_exact(monkeypatch):
    """Pre-emitted rows give bit-identical outputs and accumulated gradients to reconstructed ones."""
    monkeypatch.setenv(P.ENABLE_ENV, "1")
    monkeypatch.setenv(P.GATHER_ENV, "kernel")
    routing_map = _skewed_map(17, 23, 8, seed=4)
    base = torch.randn(17, 11, dtype=torch.float64)
    x_new = base.clone().requires_grad_(True)
    x_ref = base.clone().requires_grad_(True)
    cotangents = [
        torch.randint(-100, 101, (17, 11)).to(torch.float64),
        torch.randint(-100, 101, (17, 11)).to(torch.float64),
    ]
    for cotangent in cotangents:
        monkeypatch.setenv(P.PREEMIT_ROWS_ENV, "1")
        routed_new, _, sidx_new, _, _ = P._fused_permute(x_new, routing_map, None, 17 * 8)
        out_new = MBI._fixed_order_combine(routed_new, sidx_new, x_new.shape)
        monkeypatch.setenv(P.PREEMIT_ROWS_ENV, "0")
        routed_ref, _, sidx_ref, _, _ = P._fused_permute(x_ref, routing_map, None, 17 * 8)
        out_ref = MBI._fixed_order_combine(routed_ref, sidx_ref, x_ref.shape)
        assert torch.equal(out_new, out_ref)
        out_new.backward(cotangent)
        out_ref.backward(cotangent)
        assert torch.equal(x_new.grad, x_ref.grad)


def test_no_grad_forward_creates_no_autograd_context_and_moves_the_same_bits(monkeypatch):
    monkeypatch.setenv(P.ENABLE_ENV, "1")
    routing_map = _topk_map(9, 13, 3, seed=4)
    x = torch.randn(9, 7, requires_grad=True)
    with torch.no_grad():
        out, _, sorted_indices, _, _ = P._fused_permute(x, routing_map, None, 27)
    assert out.grad_fn is None and not out.requires_grad
    assert torch.equal(out.view(torch.uint8), x.detach().index_select(0, sorted_indices).view(torch.uint8))


# =================================================================================================
# install chain
# =================================================================================================
def test_install_reaches_both_bindings_and_reverts(monkeypatch):
    def original(*args, **kwargs):
        return "ORIGINAL"

    moe_utils, token_dispatcher = _install_fake_megatron(monkeypatch, original)
    if P._installed:
        P.revert_fused_permute()
    try:
        assert P.install_fused_permute()
        assert moe_utils.permute is P._fused_permute
        assert token_dispatcher.permute is P._fused_permute
        assert P.install_fused_permute(), "install must be idempotent"
        # the captured fallback is the binding that was live, so a decline still reaches whatever
        # wrapper already owned permute
        assert P._orig_permute is original

        P.revert_fused_permute()
        assert moe_utils.permute is original and token_dispatcher.permute is original
    finally:
        P.revert_fused_permute()
        moe_utils.permute = original
        token_dispatcher.permute = original


def test_install_refuses_a_split_binding(monkeypatch):
    moe_utils, token_dispatcher = _install_fake_megatron(monkeypatch, lambda *a, **k: None)
    token_dispatcher.permute = lambda *a, **k: None  # a stale namespace copy
    if P._installed:
        P.revert_fused_permute()
    monkeypatch.setenv(P.ENABLE_ENV, "1")
    try:
        with pytest.raises(RuntimeError, match="split binding"):
            P.install_fused_permute()
    finally:
        P.revert_fused_permute()


# =================================================================================================
# kernel shape, without a CUDA context
# =================================================================================================
def test_no_atomics_and_one_store_owner_in_source():
    import inspect

    source = inspect.getsource(P)
    assert "tl.atomic_add" not in source
    assert "tl.store(OUT" in source
    assert "tl.store(COMBINE_ROWS" in source


@pytest.mark.skipif(not P.HAVE_TRITON, reason="offline SM90 compilation needs Triton")
def test_kernels_compile_for_sm90_without_cuda_and_carry_no_atomics():
    """The real kernel specializations compile to TTIR for SM90 with no atomics or float math."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    cases = (
        (
            P._route_counts_kernel,
            {"RMAP": "*i8", "COUNTS": "*i64", "T": "i32", "E": "i32", "BLOCK_T": "constexpr"},
            {"BLOCK_T": 1024},
            None,
        ),
        (
            P._route_compact_kernel,
            {
                "RMAP": "*i8",
                "COUNTS": "*i64",
                "SORTED_IDX": "*i64",
                "ROUTE_EXPERT": "*i64",
                "T": "i32",
                "E": "i32",
                "BLOCK_T": "constexpr",
                "BLOCK_E": "constexpr",
            },
            {"BLOCK_T": 1024, "BLOCK_E": 256},
            None,
        ),
        (
            P._permute_gather_kernel,
            {
                "TOKENS": "*bf16",
                "SORTED_IDX": "*i64",
                "ROUTE_EXPERT": "*i64",
                "RMAP": "*i8",
                "OUT": "*bf16",
                "COMBINE_ROWS": "*i32",
                "hidden": "i32",
                "E": "i32",
                "K": "constexpr",
                "BLOCK_H": "constexpr",
                "BLOCK_E": "constexpr",
                "EMIT_ROWS": "constexpr",
            },
            {"K": 8, "BLOCK_H": 1024, "BLOCK_E": 256, "EMIT_ROWS": False},
            1,
        ),
        (
            P._permute_gather_kernel,
            {
                "TOKENS": "*bf16",
                "SORTED_IDX": "*i64",
                "ROUTE_EXPERT": "*i64",
                "RMAP": "*i8",
                "OUT": "*bf16",
                "COMBINE_ROWS": "*i32",
                "hidden": "i32",
                "E": "i32",
                "K": "constexpr",
                "BLOCK_H": "constexpr",
                "BLOCK_E": "constexpr",
                "EMIT_ROWS": "constexpr",
            },
            {"K": 8, "BLOCK_H": 1024, "BLOCK_E": 256, "EMIT_ROWS": True},
            2,
        ),
    )
    for kernel, signature, constexprs, want_stores in cases:
        compiled = triton.compile(
            ASTSource(fn=kernel, signature=signature, constexprs=constexprs),
            target=GPUTarget("cuda", 90, 32),
        )
        ttir = compiled.asm["ttir"]
        assert "atomic" not in ttir, f"{kernel.__name__} must stay atomics-free"
        if want_stores is not None:
            assert ttir.count("tt.store") == want_stores
        # movement kernels must contain no float arithmetic at all
        for op in ("arith.addf", "arith.mulf", "arith.subf", "arith.truncf", "arith.extf"):
            assert op not in ttir, f"{kernel.__name__} must not do float arithmetic ({op})"
