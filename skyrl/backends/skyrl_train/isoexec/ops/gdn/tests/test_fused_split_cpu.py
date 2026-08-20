"""CPU+GPU tests for the fused q/k/v(+alpha/beta) split (``gdn_fused_split``).

WHAT THIS GATES. Five ATen copies per GDN layer -- ``alpha[0].contiguous()`` /
``beta[0].contiguous()`` at ``gdn_gptmodel:477`` and the three column slices of
``_split_qkv_raw`` -- measured positionally at 11.37 us/layer, 341.2 us of every decode step at
30 GDN layers, to move 2.1 MB. ``fused_split_qkvab`` writes all five contiguous outputs in one
launch. The copies themselves are irremovable (vLLM's fused_sigmoid_gating core takes no strides);
only the launch count moves.

The change is therefore allowed to be a LAUNCH-STRUCTURE change and NOTHING else, and the proof
obligation is correspondingly strict: the outputs must be BYTE-IDENTICAL to the eager chain, which
for a pure permutation of bytes means ``torch.equal`` on a ``uint8`` view -- not an ``allclose``,
and not a comparison on values that a NaN or a subnormal could paper over.

TRITON IS GPU-ONLY, SO THIS FILE HAS TWO HALVES. On CPU it proves (a) the seam declines to the
ATen chain and returns exactly the legacy answer, (b) every decline predicate fires on the shapes
the kernel refuses, (c) the deferral predicate ``defer_ab`` selects exactly the native-core
composition. On a CUDA box the same file additionally runs the REAL kernel against the eager chain
on adversarial byte populations and asserts uint8 equality, and captures/replays it in a CUDA
graph -- those are the assertions that matter, and they are skipped, never faked, without a device.

Run: uv run --isolated --extra dev python -m pytest \
       skyrl/backends/skyrl_train/isoexec/ops/gdn/tests/test_fused_split_cpu.py -q
"""

import pytest

torch = pytest.importorskip("torch")

from skyrl.backends.skyrl_train.isoexec.ops.gdn import (  # noqa: E402
    gdn_fused_split as fs,
)


def _split_stats():
    # inlined: the public module keeps the counters but not this test accessor
    return fs._served, fs._declined, fs._decline_reason

from skyrl.backends.skyrl_train.isoexec.ops.gdn.gdn_recurrent_state import (  # noqa: E402
    RecurrentGDN,
)

CUDA = torch.cuda.is_available()

# The production geometry, at TP=8 on Qwen3.5-35B-A3B: kd = num_k_heads*head_k_dim = 256,
# vd = num_v_heads*head_v_dim = 512, ha = num_value_heads/tp = 4. These are the grids the
# positional attribution named (g=256, g=256, g=512, g=4, g=4).
NUM_K_HEADS, HEAD_K_DIM = 2, 128
NUM_V_HEADS, HEAD_V_DIM = 4, 128
KD = NUM_K_HEADS * HEAD_K_DIM
VD = NUM_V_HEADS * HEAD_V_DIM
HA = NUM_V_HEADS


# ================================================================================================
# the oracle: a VERBATIM copy of the pre-change chain, kept as the reference rather than as history
# ================================================================================================
def _eager_split(y, kd, a=None, b=None):
    """``_split_qkv_raw``'s three slices + ``gdn_gptmodel``'s two ``.contiguous()`` calls."""
    q = y[:, :kd].contiguous()
    k = y[:, kd : 2 * kd].contiguous()
    v = y[:, 2 * kd :].contiguous()
    if a is None:
        return q, k, v, None, None
    return q, k, v, a.contiguous(), b.contiguous()


def _bytes(t):
    """The only comparison that is honest for a byte permutation."""
    return t.contiguous().view(torch.uint8) if t.dtype != torch.uint8 else t.contiguous()


def _pool(device="cpu"):
    """A ``RecurrentGDN`` shell carrying only the split geometry (``__new__``: no state alloc)."""
    rg = RecurrentGDN.__new__(RecurrentGDN)
    rg.num_k_heads, rg.head_k_dim = NUM_K_HEADS, HEAD_K_DIM
    rg.num_v_heads, rg.head_v_dim = NUM_V_HEADS, HEAD_V_DIM
    return rg


def _adversarial(T, D, device, seed=0):
    """A bf16 row whose BYTES span the populations a value comparison would hide.

    Built by drawing uint16 bit patterns uniformly and viewing them as bf16, so NaNs (both
    payloads), signalling NaNs, +-inf, subnormals and negative zero all appear. A copy is a copy;
    if the fused kernel ever touched a value instead of moving it, this is what catches it.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    bits = torch.randint(0, 1 << 16, (T, D), generator=g, dtype=torch.int32).to(torch.uint16)
    return bits.view(torch.bfloat16).to(device)


# ================================================================================================
# CPU half -- the seam, the decline predicates, and the deferral rule
# ================================================================================================
def test_cpu_seam_declines_and_matches_the_legacy_chain():
    """No CUDA -> ``_split_qkv_raw`` takes the ATen chain and returns exactly what it always did."""
    rg = _pool()
    y = _adversarial(7, 2 * KD + VD, "cpu", seed=1)
    q, k, v = rg._split_qkv_raw(y)
    eq, ek, ev, _, _ = _eager_split(y, KD)
    assert torch.equal(_bytes(q.reshape(7, -1)), _bytes(eq))
    assert torch.equal(_bytes(k.reshape(7, -1)), _bytes(ek))
    assert torch.equal(_bytes(v.reshape(7, -1)), _bytes(ev))
    assert q.shape == (7, NUM_K_HEADS, HEAD_K_DIM)
    assert v.shape == (7, NUM_V_HEADS, HEAD_V_DIM)


def test_cpu_seam_with_ab_returns_five_and_compacts():
    """The five-output form compacts strided alpha/beta exactly as the call site used to."""
    rg = _pool()
    y = _adversarial(5, 2 * KD + VD, "cpu", seed=2)
    wide = _adversarial(5, 3 * HA, "cpu", seed=3)
    a, b = wide[:, HA : 2 * HA], wide[:, 2 * HA :]  # strided views, as alpha[0]/beta[0] are
    assert not a.is_contiguous() and not b.is_contiguous()
    q, k, v, ac, bc = rg._split_qkv_raw(y, a, b)
    assert ac.is_contiguous() and bc.is_contiguous()
    assert torch.equal(_bytes(ac), _bytes(a.contiguous()))
    assert torch.equal(_bytes(bc), _bytes(b.contiguous()))
    eq, ek, ev, _, _ = _eager_split(y, KD)
    assert torch.equal(_bytes(q.reshape(5, -1)), _bytes(eq))
    assert torch.equal(_bytes(k.reshape(5, -1)), _bytes(ek))
    assert torch.equal(_bytes(v.reshape(5, -1)), _bytes(ev))


def test_flag_default_is_on_and_switchable(monkeypatch):
    monkeypatch.delenv(fs._ENV, raising=False)
    assert fs.fused_split_enabled() is True
    for off in ("0", "false", "no", ""):
        monkeypatch.setenv(fs._ENV, off)
        assert fs.fused_split_enabled() is False
    monkeypatch.setenv(fs._ENV, "1")
    assert fs.fused_split_enabled() is True


def test_defer_ab_selects_only_the_native_core(monkeypatch):
    """The deferral is the native-core composition's alone; everything else compacts at the seam."""

    class _S:
        pass

    native, other = _S(), _S()
    native._native_core = True
    other._native_core = False
    monkeypatch.setenv(fs._ENV, "1")
    assert fs.defer_ab(native) is True
    assert fs.defer_ab(other) is False
    assert fs.defer_ab(_S()) is False  # no attribute at all -> compact, as before
    monkeypatch.setenv(fs._ENV, "0")
    assert fs.defer_ab(native) is False


def test_contiguous_ab_is_identity_on_contiguous_input():
    """The flag-off path must launch NOTHING extra where the caller already compacted."""
    a = torch.zeros(4, HA)
    b = torch.zeros(4, HA)
    ca, cb = fs.contiguous_ab(a, b)
    assert ca is a and cb is b


@pytest.mark.parametrize(
    "make,why",
    [
        (lambda: (torch.zeros(4, 8, 2), KD, VD, None, None), "3-D y"),
        (lambda: (torch.zeros(4, 8).t(), 2, 4, None, None), "non-unit last stride"),
        (lambda: (torch.zeros(4, 8), 3, 4, None, None), "widths do not sum to the row"),
        (lambda: (torch.zeros(0, 8), 2, 4, None, None), "T=0"),
        (lambda: (torch.zeros(4, 8), 2, 4, torch.zeros(4, 2), None), "alpha without beta"),
        (lambda: (torch.zeros(4, 8), 2, 4, torch.zeros(3, 2), torch.zeros(3, 2)), "alpha rows != T"),
    ],
)
def test_decline_predicates_fire(make, why):
    """Every structural surprise DECLINES (caller falls back), none of them guesses."""
    y, kd, vd, a, b = make()
    before = _split_stats()[1]
    assert fs.fused_split_qkvab(y, kd, vd, a, b) is None, why
    assert _split_stats()[1] == before + 1


# ================================================================================================
# GPU half -- the assertion that actually matters
# ================================================================================================
@pytest.mark.skipif(not CUDA, reason="the fused split is a Triton kernel")
@pytest.mark.parametrize("T", [1, 2, 7, 63, 64, 129, 512])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_gpu_fused_split_is_bytewise_equal(T, dtype):
    """uint8 ``torch.equal`` against the eager chain -- a byte permutation has no tolerance."""
    y = _adversarial(T, 2 * KD + VD, "cuda", seed=T).to(dtype)
    wide = _adversarial(T, 3 * HA, "cuda", seed=T + 1000).to(dtype)
    a, b = wide[:, HA : 2 * HA], wide[:, 2 * HA :]
    got = fs.fused_split_qkvab(y, KD, VD, a, b)
    assert got is not None
    eq, ek, ev, ea, eb = _eager_split(y, KD, a, b)
    for name, g, e in (("q", got[0], eq), ("k", got[1], ek), ("v", got[2], ev), ("a", got[3], ea), ("b", got[4], eb)):
        assert torch.equal(_bytes(g), _bytes(e)), name


@pytest.mark.skipif(not CUDA, reason="the fused split is a Triton kernel")
def test_gpu_fused_split_without_ab():
    """The three-segment grid: same three outputs, no alpha/beta tensors allocated."""
    y = _adversarial(96, 2 * KD + VD, "cuda", seed=11)
    got = fs.fused_split_qkvab(y, KD, VD)
    assert got is not None and got[3] is None and got[4] is None
    eq, ek, ev, _, _ = _eager_split(y, KD)
    assert torch.equal(_bytes(got[0]), _bytes(eq))
    assert torch.equal(_bytes(got[1]), _bytes(ek))
    assert torch.equal(_bytes(got[2]), _bytes(ev))


@pytest.mark.skipif(not CUDA, reason="the fused split is a Triton kernel")
def test_gpu_seam_matches_the_legacy_chain_both_flag_arms(monkeypatch):
    """``_split_qkv_raw`` through the seam: fused arm == ATen arm, byte for byte, both shapes."""
    rg = _pool()
    y = _adversarial(512, 2 * KD + VD, "cuda", seed=5)
    wide = _adversarial(512, 3 * HA, "cuda", seed=6)
    a, b = wide[:, HA : 2 * HA], wide[:, 2 * HA :]
    monkeypatch.setenv(fs._ENV, "0")
    off = rg._split_qkv_raw(y, a, b)
    monkeypatch.setenv(fs._ENV, "1")
    on = rg._split_qkv_raw(y, a, b)
    for i, name in enumerate("qkvab"):
        assert torch.equal(_bytes(on[i]), _bytes(off[i])), name


@pytest.mark.skipif(not CUDA, reason="the fused split is a Triton kernel")
def test_gpu_captures_and_replays_in_a_cuda_graph():
    """Decode is graph-replayed, so the kernel has to capture: no host reads, static shapes.

    The replay must also RECOMPUTE -- a graph that baked in the first launch's outputs would pass a
    single-input test and serve stale q/k/v forever. Two different inputs, written into the same
    captured buffers, must give the two different eager answers.
    """
    T = 128
    y = torch.empty(T, 2 * KD + VD, dtype=torch.bfloat16, device="cuda")
    a_src = torch.empty(T, 3 * HA, dtype=torch.bfloat16, device="cuda")
    y.copy_(_adversarial(T, 2 * KD + VD, "cuda", seed=21))
    a_src.copy_(_adversarial(T, 3 * HA, "cuda", seed=22))
    a, b = a_src[:, HA : 2 * HA], a_src[:, 2 * HA :]

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):  # warmup outside the capture, as vLLM does
        fs.fused_split_qkvab(y, KD, VD, a, b)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fs.fused_split_qkvab(y, KD, VD, a, b)
    assert out is not None

    for seed in (21, 33):
        y.copy_(_adversarial(T, 2 * KD + VD, "cuda", seed=seed))
        a_src.copy_(_adversarial(T, 3 * HA, "cuda", seed=seed + 1))
        g.replay()
        torch.cuda.synchronize()
        eq, ek, ev, ea, eb = _eager_split(y, KD, a, b)
        for i, (name, e) in enumerate((("q", eq), ("k", ek), ("v", ev), ("a", ea), ("b", eb))):
            assert torch.equal(_bytes(out[i]), _bytes(e)), f"{name} @ seed {seed}"
