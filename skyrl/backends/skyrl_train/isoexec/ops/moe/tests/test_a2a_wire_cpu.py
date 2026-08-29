"""CPU proof obligations for the bf16 BACKWARD wire on the MoE combine all-to-all.

THE CLAIM UNDER TEST, in three parts, each with a control that MUST fail so a PASS is not vacuous:

  1. ROUND-TRIP IDENTITY. On fp32 values that are exactly bf16-representable -- which is what the
     backward payload is, because the VJP of ``combine_postprocess``'s single ``.to(bfloat16)`` is
     an exact upcast and everything after it on that path is a permutation -- narrow -> exchange ->
     widen is the identity, bit for bit. CONTROL: on fp32 values that are NOT bf16-representable
     (the forward payload's value class: a sum of 8 bf16 leaves) the same round trip MOVES bits.

  2. THE BIT COMPARE IS A BIT COMPARE. ``-0.0`` must survive the round trip and be recognised as
     surviving, which a float compare cannot see (``-0.0 == 0.0``). NaN must be REFUSED -- and the
     MEASURED reason is stronger than expected: torch canonicalises every fp32 NaN to bf16
     ``0xFFFF``, so not even ``0x7FC00000`` round-trips. +-inf does. A float compare could make
     none of these calls (``nan != nan`` marks every NaN as failing whether or not its bits moved).

  3. THE FORWARD IS UNTOUCHED. Structural, on the real op: the forward payload handed to the
     collective is the caller's tensor, same dtype, same bytes; only the BACKWARD payload is
     narrowed. Asserted twice -- functionally, by recording what each direction actually put on the
     wire through a stubbed exchange, and syntactically, by an AST scan of ``forward`` for any
     narrowing at all (a future edit that adds one has to delete this test to land).

Everything here runs without CUDA, without megatron and without a process group: the exchange is
stubbed so the test observes the WIRE PAYLOAD, which is the thing the design is about.

    uv run --isolated --extra dev python -m pytest <thisfile> -q
"""

from __future__ import annotations

import ast
import inspect
import struct

import pytest
import torch

from skyrl.backends.skyrl_train.isoexec.ops.moe import moe_a2a_wire as W


def _census_counts():
    # inlined: the public module keeps the state but not this test accessor
    tot = [0, 0, 0, 0]
    for t in W._CENSUS.values():
        tot = [a + b for a, b in zip(tot, t.tolist())]
    return tuple(tot)


# =============================================================================================
# helpers
# =============================================================================================
def _bits(x: torch.Tensor) -> torch.Tensor:
    """int32 view of an fp32 tensor -- the only honest way to compare floats here."""
    return x.contiguous().view(torch.int32)


def _bf16_representable(n: int, seed: int) -> torch.Tensor:
    """The BACKWARD payload's value class: fp32 boxes holding bf16 values (an exact upcast)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=g).to(torch.bfloat16).to(torch.float32)


def _leaf_sum(n: int, seed: int, leaves: int = 8) -> torch.Tensor:
    """The FORWARD payload's value class: the pik leaf-tree fc2 root, a sum of 8 bf16 leaves in
    fp32 (``moe_batched_experts._leaftree_fc2``). Not bf16-representable, and that is the point."""
    g = torch.Generator().manual_seed(seed)
    acc = torch.zeros(n, dtype=torch.float32)
    for _ in range(leaves):
        acc = acc + torch.randn(n, generator=g).to(torch.bfloat16).to(torch.float32)
    return acc


def _wire_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """narrow -> (exchange, which copies bytes and does no arithmetic) -> widen."""
    return x.to(torch.bfloat16).to(torch.float32)


class _StubGroup:
    """A process group that only has to answer ``size()``; nothing here communicates."""

    def __init__(self, n: int = 8):
        self._n = n

    def size(self) -> int:
        return self._n


class _Wire:
    """Records every payload the op hands to the collective, and returns it unchanged.

    Returning the input is exactly what an all-to-all does at world==1 and is the right stub for a
    collective that "performs no arithmetic -- every output byte is a copy of exactly one input
    byte" (the argument already used for the NCCL channel ladder). What is under test is what goes
    ON the wire, not how it is routed.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, group, x, output_split_sizes, input_split_sizes, use_nccl_stream):
        self.calls.append((tuple(x.shape), x.dtype, x.clone()))
        return x.clone()


@pytest.fixture
def wired(monkeypatch):
    """Op with the collective stubbed out and module counters reset."""
    w = _Wire()
    monkeypatch.setattr(W, "_raw_a2a", w)
    W._reset_for_test()
    yield w
    W._reset_for_test()
    W._S["mode"] = None


def _run(wire_mode: str, x: torch.Tensor, group=None):
    """One forward + backward through the real ``CombineAllToAll``; returns (out, grad_in)."""
    W._S["mode"] = wire_mode
    xin = x.clone().requires_grad_(True)
    out = W.CombineAllToAll.apply(group or _StubGroup(8), xin, None, None, False)
    out.backward(x.clone())  # an fp32 upstream gradient of the same value class
    return out, xin.grad


# =============================================================================================
# 1. the round-trip identity, and its control
# =============================================================================================
@pytest.mark.parametrize("n", [1, 7, 4096, 5510 * 8])
def test_roundtrip_is_the_identity_on_bf16_representable_fp32(n):
    x = _bf16_representable(n, seed=n)
    assert torch.equal(_bits(_wire_roundtrip(x)), _bits(x))
    assert bool(W.roundtrip_is_bitwise(x).item())


@pytest.mark.parametrize("n", [4096, 5510 * 8])
def test_control_roundtrip_moves_bits_on_the_forward_payloads_value_class(n):
    """If this ever PASSES, the harness is broken: an 8-leaf fp32 sum is not a bf16 number."""
    x = _leaf_sum(n, seed=n)
    moved = int((_bits(_wire_roundtrip(x)) != _bits(x)).sum())
    assert moved > 0, "a sum of 8 bf16 leaves came back bit-identical through bf16 -- control broken"
    assert not bool(W.roundtrip_is_bitwise(x).item())
    # and it is a LOT of them, not a rounding coincidence on a handful of elements
    assert moved > n // 2


def test_bf16_representable_is_exactly_the_set_that_round_trips():
    """Sanity on the two generators: one class is closed under the round trip, the other is not."""
    a = _bf16_representable(1 << 14, seed=1)
    b = _leaf_sum(1 << 14, seed=1)
    assert bool(W.roundtrip_is_bitwise(a).item())
    assert not bool(W.roundtrip_is_bitwise(b).item())


# =============================================================================================
# 2. the bit compare: signed zeros and NaN
# =============================================================================================
def test_signed_zero_survives_and_is_seen_to_survive():
    x = torch.tensor([-0.0, 0.0, -0.0], dtype=torch.float32)
    r = _wire_roundtrip(x)
    # compared as INTEGERS: a float compare would call every one of these equal even if the sign
    # bit had been dropped, so it could not tell a preserved -0.0 from a destroyed one.
    assert torch.equal(_bits(r), _bits(x))
    assert _bits(r)[0].item() == struct.unpack("<i", struct.pack("<f", -0.0))[0]
    assert bool(W.roundtrip_is_bitwise(x).item())

    # the control the float compare would fail: -0.0 turned into +0.0 is bit-different...
    flipped = torch.tensor([0.0, 0.0, -0.0], dtype=torch.float32)
    assert not torch.equal(_bits(flipped), _bits(x))
    # ...and float-equal, which is exactly why the check must not use ==.
    assert bool(torch.eq(flipped, x).all().item())


def test_every_nan_is_refused_because_torch_canonicalises_it():
    """MEASURED, not assumed: torch's fp32->bf16 does NOT truncate a NaN, it emits the canonical
    ``0xFFFF``. So no NaN of any payload -- not even ``0x7FC00000`` -- survives the round trip, and
    the bit compare refuses all of them.

    The operational consequence is real and belongs in the report: if a MoE gradient ever goes NaN
    while the wire is engaged, the sampled drift probe RAISES rather than propagating the NaN. That
    is the conservative answer for a lever whose only safe failure state is "stop", but it is a
    behaviour change from the flag-off path and nobody should discover it from a stack trace.
    """

    def f32(bits: int) -> torch.Tensor:
        return torch.tensor([struct.unpack("<f", struct.pack("<I", bits))[0]], dtype=torch.float32)

    for payload in (0x7FC00000, 0x7FC00001, 0xFFC00000, 0x7F800001):
        x = f32(payload)
        assert torch.isnan(x).all()
        assert _bits(_wire_roundtrip(x)).item() == -65536  # 0xFFFF0000, the canonical bf16 NaN
        assert not bool(W.roundtrip_is_bitwise(x).item()), hex(payload)

    # a float compare cannot make this call at all: `nan != nan` marks a NaN as failing whether or
    # not its bits moved, so it can neither admit nor diagnose one.
    assert not bool(torch.eq(f32(0x7FC00000), f32(0x7FC00000)).all().item())


def test_infinities_survive_and_fp32_subnormals_do_not():
    """The two remaining edges of the value class, both measured."""

    def f32(bits: int) -> torch.Tensor:
        return torch.tensor([struct.unpack("<f", struct.pack("<I", bits))[0]], dtype=torch.float32)

    for inf in (0x7F800000, 0xFF800000):  # bf16 shares fp32's exponent width, so inf round-trips
        assert torch.equal(_bits(_wire_roundtrip(f32(inf))), _bits(f32(inf)))
        assert bool(W.roundtrip_is_bitwise(f32(inf)).item())

    tiny = f32(0x00000001)  # an fp32 subnormal below bf16's 7-bit grid -> flushed, so REFUSED
    assert not bool(W.roundtrip_is_bitwise(tiny).item())
    # ...but a value that came FROM bf16 (which is what the backward payload is) never is one:
    assert bool(W.roundtrip_is_bitwise(torch.tensor([1e-40], dtype=torch.bfloat16).to(torch.float32)).item())


# =============================================================================================
# 3. the op: forward untouched, backward narrowed
# =============================================================================================
def test_forward_payload_is_untouched_and_backward_payload_is_bf16(wired):
    x = _bf16_representable(2048, seed=11).view(8, 256)
    out, grad = _run(W.MODE_ON, x)

    assert len(wired.calls) == 2
    (fshape, fdtype, fpayload), (bshape, bdtype, bpayload) = wired.calls

    # FORWARD: same shape, same dtype, same BYTES as what the caller passed in.
    assert fdtype == torch.float32 and fshape == tuple(x.shape)
    assert torch.equal(_bits(fpayload), _bits(x))
    assert torch.equal(_bits(out), _bits(x))

    # BACKWARD: narrowed on the wire...
    assert bdtype == torch.bfloat16 and bshape == tuple(x.shape)
    # ...and widened back to the forward payload's dtype, bit-identical to the fp32 wire's answer.
    assert grad.dtype == torch.float32
    assert torch.equal(_bits(grad), _bits(x))
    assert W.wire_stats()["served"] == 1
    assert W.wire_stats()["declined"] == 0
    assert W.wire_stats()["bytes_saved"] == 2 * x.numel() * 2


def test_flag_off_narrows_nothing(wired):
    x = _bf16_representable(2048, seed=12).view(8, 256)
    _run(W.MODE_OFF, x)
    assert [c[1] for c in wired.calls] == [torch.float32, torch.float32]
    assert W.wire_stats()["served"] == 0 and W.wire_stats()["declined"] == 1


def test_probe_mode_censuses_both_directions_and_changes_no_wire(wired):
    x = _leaf_sum(2048, seed=13).view(8, 256)  # forward payload: NOT representable
    W._S["mode"] = W.MODE_PROBE
    xin = x.clone().requires_grad_(True)
    out = W.CombineAllToAll.apply(_StubGroup(8), xin, None, None, False)
    out.backward(_bf16_representable(2048, seed=14).view(8, 256))  # backward payload: representable

    assert [c[1] for c in wired.calls] == [torch.float32, torch.float32], "probe mode moved a wire"
    fwd_seen, fwd_pass, bwd_seen, bwd_pass = _census_counts()
    assert (fwd_seen, fwd_pass) == (1, 0), "the forward payload must NOT be bf16-representable"
    assert (bwd_seen, bwd_pass) == (1, 1), "the backward payload MUST be bf16-representable"


def test_a_non_representable_gradient_disables_the_path_rather_than_corrupting_it(wired):
    """The first-call vote is the only fallback there is; after it, the design RAISES."""
    x = _bf16_representable(2048, seed=15).view(8, 256)
    W._S["mode"] = W.MODE_ON
    xin = x.clone().requires_grad_(True)
    out = W.CombineAllToAll.apply(_StubGroup(8), xin, None, None, False)
    out.backward(_leaf_sum(2048, seed=16).view(8, 256))  # a gradient the wire cannot carry

    assert W.wire_stats()["agreed"] is False
    assert W.wire_stats()["served"] == 0 and W.wire_stats()["declined"] == 1
    assert [c[1] for c in wired.calls] == [torch.float32, torch.float32]
    # the grad still came back exactly, because the declined path is verbatim megatron
    assert xin.grad.dtype == torch.float32


def test_bf16_grad_is_declined_structurally(wired):
    """Admission reads dtype and group size only -- both identical on every EP rank."""
    x = torch.randn(8, 256, dtype=torch.bfloat16)
    W._S["mode"] = W.MODE_ON
    xin = x.clone().requires_grad_(True)
    out = W.CombineAllToAll.apply(_StubGroup(8), xin, None, None, False)
    out.backward(torch.randn(8, 256, dtype=torch.bfloat16))
    assert W.wire_stats()["served"] == 0 and W.wire_stats()["declined"] == 1
    assert W.wire_stats()["agreed"] is None, "the group vote must not run on a declined shape"


def test_world_one_is_declined(wired):
    x = _bf16_representable(256, seed=17).view(1, 256)
    _run(W.MODE_ON, x, group=_StubGroup(1))
    assert W.wire_stats()["served"] == 0 and W.wire_stats()["declined"] == 1


# =============================================================================================
# 3b. the structural claim, syntactically
# =============================================================================================
def _fn_ast(fn) -> ast.FunctionDef:
    src = inspect.getsource(fn)
    # strip the class-level indentation so ast can parse the method on its own
    node = ast.parse(ast.unparse(ast.parse(src.lstrip().replace("\n    ", "\n"))))
    return next(n for n in ast.walk(node) if isinstance(n, ast.FunctionDef))


def test_forward_source_contains_no_narrowing():
    """A future edit that narrows the forward wire has to delete this assert to land.

    Narrowing the forward is the canary-v10 defect (round-before-sum on the top-k sum); the reason
    this file exists is that the BACKWARD is a different question with a different answer.
    """
    src = inspect.getsource(W.CombineAllToAll.forward)
    tree = ast.parse(src.replace("\n    ", "\n").lstrip())
    casts = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in ("bfloat16", "half", "float16")]
    # the ONLY bfloat16 mention allowed in forward is the census dtype filter, which reads a dtype
    # and puts nothing on the wire.
    assert len(casts) <= 1, [ast.dump(c) for c in casts]
    assert ".to(torch.bfloat16)" not in src


def test_backward_narrowing_lives_in_the_backward_helper_only():
    src = inspect.getsource(W._combine_backward)
    assert "to(torch.bfloat16)" in src
    assert "ctx.wire_dtype" in src, "the widen must restore the FORWARD payload's dtype"


def test_sampled_check_raises_and_never_falls_back():
    """The inverted convention, pinned: no `return`/fallback in the drift probe, only a raise."""
    src = inspect.getsource(W._drift_probe_or_raise)
    tree = ast.parse(src.lstrip())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    assert any(isinstance(n, ast.Raise) for n in ast.walk(fn))
    assert "MIN" in src, "the sampled check must be MIN-reduced so every rank raises together"
    # `_raw_a2a` must not appear: falling back to the fp32 exchange from here is the bug.
    assert "_raw_a2a" not in src
