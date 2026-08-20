"""CPU proofs for the MoE combine's ROUND FOLD and the no-grad bypass.

WHAT IS BEING PROVEN, AND WHY IT IS NOT AUTOMATIC
-------------------------------------------------
Unlike the permute, the combine is NOT pure data movement -- it contains a k-term sum. So byte
identity is a claim that has to be discharged, not a property that comes for free. This lever
does not touch the sum. It moves the SINGLE trailing round the caller already performs
(``combine_postprocess``: ``output.view(...).to(torch.bfloat16)``) from a separate pass over a
materialized ``[T, H]`` fp32 tensor into the store of the kernel that produced it. The arithmetic
content of that claim is exactly

    reference(out_dtype=bf16)  ==  reference(out_dtype=None).to(bf16)          (bit for bit)

i.e. one round-to-nearest-even of one fp32 accumulator, whether it is applied at the store or one
op later. That is a statement about fp32 -> bf16 rounding and about NOTHING ELSE, so it is fully
provable on CPU -- including on the inputs where a sloppier implementation would differ:
subnormals, signed zeros, NaN/inf, and the exact halfway values where RTNE's tie rule bites.

The three things that could turn this into a COMPOSITION EVENT are each tested with teeth:

  1. keying ``ROUND_BF16`` on the STORE dtype instead of the ACCUMULATE dtype -- that would turn an
     fp32 sum into a per-term-rounded bf16 sum. ``test_accumulate_and_store_are_different_axes``
     shows the two schedules are genuinely different functions (a positive control), and
     ``legal_fold_dtype`` is what keeps them separate;
  2. accepting some other dtype pair (bf16 accumulate widened to fp32, fp16, ...) -- refused with a
     raise rather than silently honoured;
  3. the request leaking into megatron's own ``unpermute``, which has no such parameter -- the
     kwarg is zero-KL-private and every fallback path is checked for it.

The Triton kernel itself is GPU-only; the private repo's nightly ``moe_combine_fold_test.py`` closes
the remaining step (that the kernel's store agrees with this reference) with ``torch.equal`` on the
real operands before it times anything.
"""

from __future__ import annotations

import struct

import pytest
import torch

from skyrl.backends.skyrl_train.isoexec.ops.moe import moe_combine_backward as CB
from skyrl.backends.skyrl_train.isoexec.ops.moe import moe_combine_kernel as CK


def _combine_backward_stats():
    # inlined: the public module keeps the state but not this test accessor
    out = dict(CB._COUNTS)
    out["decline_reasons"] = dict(CB._DECLINES)
    return out


def _fixed_order_combine_reference(permuted_tokens, rows, *, permuted_probs=None, out_dtype=None):
    # inlined verbatim from the private module: the eager chain as a callable oracle (CPU, no Triton)
    x = permuted_tokens if permuted_probs is None else permuted_tokens * permuted_probs.unsqueeze(-1)
    out = x.index_select(0, rows[:, 0].contiguous().long())
    for j in range(1, rows.shape[1]):
        out = out + x.index_select(0, rows[:, j].contiguous().long())
    return out if out_dtype is None else out.to(out_dtype)



# =================================================================================================
# the adversarial fp32 population
# =================================================================================================
def _bf16_tie_values(n: int) -> torch.Tensor:
    """fp32 values sitting EXACTLY halfway between two adjacent bf16 values.

    bf16 keeps the top 16 bits of an fp32. The halfway point between bf16 ``b`` and its successor
    has the low 16 bits equal to 0x8000. Round-to-nearest-EVEN then decides by the low bit of the
    kept mantissa, so a population that alternates that bit exercises both directions of the tie
    rule -- the one place where "round twice" and "round once" can disagree if anything about the
    schedule is wrong.
    """
    out = []
    for i in range(n):
        hi = 0x3F80 + (i % 97)  # exponents/mantissas around 1.0, both tie parities
        bits = (hi << 16) | 0x8000
        out.append(struct.unpack("<f", struct.pack("<I", bits))[0])
    return torch.tensor(out, dtype=torch.float32)


def _adversarial_fp32(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    body = torch.randn(max(n - 24, 1), generator=g, dtype=torch.float32) * (10.0 ** torch.randint(-6, 6, (1,)).item())
    special = torch.tensor(
        [
            0.0,
            -0.0,
            float("nan"),
            float("inf"),
            float("-inf"),
            struct.unpack("<f", struct.pack("<I", 0x00000001))[0],  # smallest positive subnormal
            -struct.unpack("<f", struct.pack("<I", 0x00000001))[0],
            struct.unpack("<f", struct.pack("<I", 0x007FFFFF))[0],  # largest subnormal
            struct.unpack("<f", struct.pack("<I", 0x00800000))[0],  # smallest normal
            3.4028234663852886e38,  # fp32 max -- overflows bf16's mantissa, not its exponent
            -3.4028234663852886e38,
            1.0,
            -1.0,
        ],
        dtype=torch.float32,
    )
    ties = _bf16_tie_values(11)
    v = torch.cat([body, special, ties])
    return v[torch.randperm(v.numel(), generator=g)][:n] if v.numel() >= n else v.repeat(n // v.numel() + 1)[:n]


def _routing(num_tokens: int, num_experts: int, topk: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    scores = torch.rand(num_tokens, num_experts, generator=g)
    m = torch.zeros(num_tokens, num_experts, dtype=torch.bool)
    m.scatter_(1, scores.topk(topk, dim=1).indices, True)
    return m


def _permute_expression(tokens: torch.Tensor, routing_map: torch.Tensor):
    """Megatron's own unfused permute, verbatim -- the producer of ``sorted_indices``."""
    num_tokens = tokens.shape[0]
    num_out = int(routing_map.sum().item())
    flat = routing_map.bool().T.contiguous().reshape(-1).argsort(descending=True, stable=True)[:num_out]
    sorted_indices = flat % num_tokens
    return tokens.index_select(0, sorted_indices), sorted_indices


def _rows(sorted_indices: torch.Tensor, num_tokens: int) -> torch.Tensor:
    k = sorted_indices.numel() // num_tokens
    return torch.argsort(sorted_indices, stable=True).view(num_tokens, k)


SHAPES = [
    (1, 1, 1, 1),
    (3, 4, 1, 7),
    (5, 8, 8, 3),  # topk == num_experts: every token owns every expert
    (17, 64, 8, 2048),
    (64, 256, 8, 129),  # 192 empty experts, unaligned hidden
    (8, 16, 3, 65),
    (128, 32, 2, 1),
]


# =================================================================================================
# 1. THE FOLD IDENTITY -- the entire arithmetic content of the lever
# =================================================================================================
@pytest.mark.parametrize("T,E,K,H", SHAPES)
def test_fold_is_byte_identical_to_the_trailing_cast(T, E, K, H):
    routing = _routing(T, E, K, seed=11 + T)
    tokens = _adversarial_fp32(T * H, seed=23 + H).view(T, H)
    permuted, sidx = _permute_expression(tokens, routing)
    rows = _rows(sidx, T)

    unfolded = _fixed_order_combine_reference(permuted, rows)  # fp32 result, as today
    folded = _fixed_order_combine_reference(permuted, rows, out_dtype=torch.bfloat16)

    assert unfolded.dtype is torch.float32
    assert folded.dtype is torch.bfloat16
    # uint8 view so a moved bit cannot hide behind float comparison semantics (NaN != NaN).
    assert torch.equal(folded.view(torch.uint8), unfolded.to(torch.bfloat16).view(torch.uint8))


def test_fold_identity_on_bf16_tie_boundaries():
    """The population where "round once" and "round twice" would diverge if anything were wrong."""
    vals = _bf16_tie_values(512)
    permuted = vals.view(64, 8).repeat(1, 1)  # P=64 rows, H=8
    sidx = torch.arange(8).repeat_interleave(8)  # 8 tokens x k=8, already grouped
    rows = _rows(sidx, 8)
    a = _fixed_order_combine_reference(permuted, rows, out_dtype=torch.bfloat16)
    b = _fixed_order_combine_reference(permuted, rows).to(torch.bfloat16)
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    # ...and the test is not vacuous: at least one element really did land on a tie.
    exact = _fixed_order_combine_reference(permuted, rows).view(torch.int32) & 0xFFFF
    assert int((exact == 0x8000).sum()) > 0, "no element landed on a bf16 halfway point; test is vacuous"


def test_fold_preserves_nan_inf_and_signed_zero():
    permuted = torch.tensor(
        [
            [float("nan"), 1.0, -0.0, 0.0],
            [1.0, float("inf"), -0.0, -0.0],
            [1.0, 1.0, 0.0, -0.0],
            [1.0, float("-inf"), -0.0, -0.0],
        ],
        dtype=torch.float32,
    )
    sidx = torch.tensor([0, 0, 1, 1])
    rows = _rows(sidx, 2)
    a = _fixed_order_combine_reference(permuted, rows, out_dtype=torch.bfloat16)
    b = _fixed_order_combine_reference(permuted, rows).to(torch.bfloat16)
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    # -0.0 + -0.0 == -0.0 must survive the fold as a NEGATIVE zero (0x8000 in bf16, not 0x0000).
    assert (a[1, 3].view(torch.int16).item() & 0xFFFF) == 0x8000


# =================================================================================================
# 2. THE ACCUMULATE AXIS IS NOT THE STORE AXIS
# =================================================================================================
def test_accumulate_and_store_are_different_axes():
    """POSITIVE CONTROL. bf16-accumulate and fp32-accumulate-then-round are different functions.

    If they were not, keying ``ROUND_BF16`` on the store dtype would be harmless and the guard in
    ``legal_fold_dtype`` would be theatre. This constructs a case where they visibly differ, which
    is what gives every other assertion in this file its meaning.
    """
    one = torch.tensor(1.0, dtype=torch.float32)
    eps = torch.tensor(2.0**-9, dtype=torch.float32)  # below bf16's resolution at 1.0
    permuted = torch.stack([one.repeat(1)] + [eps.repeat(1) for _ in range(7)]).view(8, 1)
    sidx = torch.zeros(8, dtype=torch.int64)
    rows = _rows(sidx, 1)

    fp32_then_round = _fixed_order_combine_reference(permuted, rows).to(torch.bfloat16)
    bf16_per_term = _fixed_order_combine_reference(permuted.to(torch.bfloat16), rows)
    assert not torch.equal(fp32_then_round, bf16_per_term), "the two rounding schedules did not diverge"


def test_legal_fold_dtype_admits_only_the_one_fold():
    assert CK.legal_fold_dtype(torch.float32, None) is torch.float32
    assert CK.legal_fold_dtype(torch.float32, torch.float32) is torch.float32
    assert CK.legal_fold_dtype(torch.bfloat16, torch.bfloat16) is torch.bfloat16
    assert CK.legal_fold_dtype(torch.bfloat16, None) is torch.bfloat16
    assert CK.legal_fold_dtype(torch.float32, torch.bfloat16) is torch.bfloat16
    for acc, out in (
        (torch.bfloat16, torch.float32),  # widening a per-term-rounded sum -- a DIFFERENT function
        (torch.float32, torch.float16),
        (torch.bfloat16, torch.float16),
        (torch.float32, torch.float64),
    ):
        with pytest.raises(RuntimeError, match="COMPOSITION EVENT"):
            CK.legal_fold_dtype(acc, out)


def test_reference_matches_the_production_eager_expression():
    """The oracle used above is the SAME expression ``_fixed_order_combine`` runs, not a rewrite."""
    T, E, K, H = 31, 64, 8, 17
    routing = _routing(T, E, K, seed=5)
    tokens = _adversarial_fp32(T * H, seed=6).view(T, H)
    permuted, sidx = _permute_expression(tokens, routing)
    rows = _rows(sidx, T)

    # literal transcription of moe_batch_invariant._fixed_order_combine's tail
    out = permuted.index_select(0, rows[:, 0].contiguous())
    for j in range(1, K):
        out = out + permuted.index_select(0, rows[:, j].contiguous())
    assert torch.equal(out.view(torch.uint8), _fixed_order_combine_reference(permuted, rows).view(torch.uint8))


# =================================================================================================
# 3. PLUMBING -- the request must reach the kernel, and must never reach megatron
# =================================================================================================
def test_every_combine_binding_publishes_the_marker():
    """The pik call site refuses to send the kwarg to an unmarked binding, so the marker IS the
    contract. All four bindings that can be installed on ``moe_utils.unpermute`` must carry it."""
    from skyrl.backends.skyrl_train.isoexec.ops.moe import moe_batch_invariant as MBI

    for fn in (
        MBI._deterministic_unpermute,
        CB.differentiable_unpermute,
        CK.fused_deterministic_unpermute,
    ):
        assert getattr(fn, "_isoexec_accepts_out_dtype", False), f"{fn.__name__} lost the fold marker"


def test_eager_unpermute_honours_the_request_and_agrees_bitwise():
    from skyrl.backends.skyrl_train.isoexec.ops.moe import moe_batch_invariant as MBI

    T, E, K, H = 23, 32, 4, 33
    routing = _routing(T, E, K, seed=41)
    tokens = _adversarial_fp32(T * H, seed=42).view(T, H)
    permuted, sidx = _permute_expression(tokens, routing)

    plain = MBI._deterministic_unpermute(permuted, sidx, (T, H), routing_map=routing)
    folded = MBI._deterministic_unpermute(permuted, sidx, (T, H), routing_map=routing, isoexec_out_dtype=torch.bfloat16)
    assert plain.dtype is torch.float32 and folded.dtype is torch.bfloat16
    assert torch.equal(folded.view(torch.uint8), plain.to(torch.bfloat16).view(torch.uint8))


def test_request_is_never_forwarded_to_megatron(monkeypatch):
    """``_orig_unpermute`` is megatron's own function and has no such parameter. Every fallback
    branch must strip the kwarg; a leak here is a TypeError in the middle of a production step."""
    from skyrl.backends.skyrl_train.isoexec.ops.moe import moe_batch_invariant as MBI

    seen = {}

    def fake_orig(permuted_tokens, sorted_indices, restore_shape, **kw):
        seen.update(kw)
        return torch.zeros(int(restore_shape[0]), permuted_tokens.shape[-1], dtype=permuted_tokens.dtype)

    monkeypatch.setattr(MBI, "_orig_unpermute", fake_orig)
    permuted = torch.randn(6, 4)
    sidx = torch.tensor([0, 1, 2, 0, 1, 2])
    # fused=True forces the very first fallback; drop_and_pad and the non-topk layout take the others
    MBI._deterministic_unpermute(permuted, sidx, (3, 4), fused=True, isoexec_out_dtype=torch.bfloat16)
    assert "isoexec_out_dtype" not in seen, f"the private kwarg leaked into megatron: {seen}"
    seen.clear()
    CB.differentiable_unpermute(permuted, sidx, (3, 4), drop_and_pad=True, isoexec_out_dtype=torch.bfloat16)
    assert "isoexec_out_dtype" not in seen, f"the private kwarg leaked into megatron: {seen}"
    seen.clear()
    # non-topk layout (5 routed rows for 3 tokens) -> build_combine_rows declines
    CB.differentiable_unpermute(
        torch.randn(5, 4), torch.tensor([0, 1, 2, 0, 1]), (3, 4), isoexec_out_dtype=torch.bfloat16
    )
    assert "isoexec_out_dtype" not in seen, f"the private kwarg leaked into megatron: {seen}"


def test_decline_is_counted_and_named(monkeypatch):
    from skyrl.backends.skyrl_train.isoexec.ops.moe import moe_batch_invariant as MBI

    monkeypatch.setattr(MBI, "_orig_unpermute", lambda pt, si, rs, **kw: torch.zeros(int(rs[0]), pt.shape[-1]))
    before = _combine_backward_stats()
    CB.differentiable_unpermute(torch.randn(5, 4), torch.tensor([0, 1, 2, 0, 1]), (3, 4))
    after = _combine_backward_stats()
    assert after["declined"] > before["declined"]
    assert "layout_not_exact_topk" in after["decline_reasons"]


# =================================================================================================
# 4. THE NO-GRAD BYPASS
# =================================================================================================
def _stub_kernel(monkeypatch):
    """Stand in for the GPU-only Triton call with the pure-torch reference, so the WIRING is
    testable on CPU. Semantics are identical by ``fixed_order_combine_reference``'s definition."""

    def fake(permuted_tokens, sorted_indices, restore_shape, *, permuted_probs=None, rows=None, **kw):
        num_tokens = int(restore_shape[0])
        if rows is None:
            rows = CK.build_combine_rows(sorted_indices, num_tokens)
            if rows is None:
                return None
        acc = permuted_tokens.dtype if permuted_probs is None else torch.float32
        store = CK.legal_fold_dtype(acc, kw.get("out_dtype"))
        return _fixed_order_combine_reference(
            permuted_tokens, rows.long(), permuted_probs=permuted_probs, out_dtype=store
        )

    monkeypatch.setattr(CB, "fused_fixed_order_combine", fake)


def test_nograd_call_bypasses_the_autograd_function(monkeypatch):
    _stub_kernel(monkeypatch)
    permuted = torch.randn(24, 8)
    sidx = torch.arange(3).repeat_interleave(8)
    before = _combine_backward_stats()
    with torch.no_grad():
        out = CB.fused_combine(permuted, sidx, 3)
    after = _combine_backward_stats()
    assert after["nograd_served"] == before["nograd_served"] + 1
    assert after["forward_served"] == before["forward_served"], "the autograd.Function was still entered"
    assert out.grad_fn is None


def test_grad_enabled_call_keeps_the_autograd_edge(monkeypatch):
    _stub_kernel(monkeypatch)
    permuted = torch.randn(24, 8, requires_grad=True)
    sidx = torch.arange(3).repeat_interleave(8)
    before = _combine_backward_stats()
    out = CB.fused_combine(permuted, sidx, 3)
    after = _combine_backward_stats()
    assert after["forward_served"] == before["forward_served"] + 1
    assert after["nograd_served"] == before["nograd_served"]
    assert out.grad_fn is not None, "the graph was severed -- the 32B-frozen-params failure mode"


def test_nograd_bypass_returns_the_same_bytes(monkeypatch):
    _stub_kernel(monkeypatch)
    permuted = _adversarial_fp32(24 * 8, seed=77).view(24, 8)
    sidx = torch.arange(3).repeat_interleave(8)
    with torch.no_grad():
        fast = CB.fused_combine(permuted, sidx, 3)
    slow = CB.MoECombine.apply(permuted, sidx, 3, None, None, "auto", None)
    assert torch.equal(fast.view(torch.uint8), slow.view(torch.uint8))


def test_folded_forward_still_produces_the_fp32_row_gradient(monkeypatch):
    """The fold changes the forward's STORE dtype; the VJP of the removed ``.to(bfloat16)`` is an
    exact upcast, so ``d(permuted)`` must stay fp32 and must stay equal to the unfolded arm's."""
    _stub_kernel(monkeypatch)
    sidx = torch.arange(3).repeat_interleave(8)
    # ONE cotangent, in the dtype the FOLDED forward emits. The unfolded arm sees it through the
    # `.to(bfloat16)` this lever removes, whose VJP is an exact upcast -- so `cot.float()` is
    # literally what autograd hands the unfolded combine today. Comparing the two arms under the
    # same incoming gradient is the only comparison that means anything.
    cot = torch.randn(3, 8).to(torch.bfloat16)
    base = torch.randn(24, 8)

    def run(out_dtype, cotangent):
        p = base.detach().clone().requires_grad_(True)
        out = CB.fused_combine(p, sidx, 3, out_dtype=out_dtype)
        out.backward(cotangent)
        return p.grad

    g_plain = run(None, cot.float())
    g_folded = run(torch.bfloat16, cot)
    assert g_folded.dtype is torch.float32
    assert torch.equal(g_plain.view(torch.uint8), g_folded.view(torch.uint8))
