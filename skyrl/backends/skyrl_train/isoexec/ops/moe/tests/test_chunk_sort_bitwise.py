"""CPU bitwise guards for ``moe_chunk_sort`` (``SKYRL_ISOEXEC_MOE_CHUNK_SORT``).

The change is a pure re-addressing: megatron's ``torch.cat`` over a python loop of chunk slices
becomes one ``index_select``, and its VJP becomes the inverse gather. No floating-point operation
appears anywhere, so bit-equality holds BY CONSTRUCTION -- and is asserted anyway, forward and
backward, because the two forms have different VJPs and because the real failure mode is not
rounding but producing the WRONG PERMUTATION.

Same discipline as ``test_flat_stage_bitwise.py``: CPU, no CUDA, no megatron import.
"""

from __future__ import annotations

import torch

from skyrl.backends.skyrl_train.isoexec.ops.moe import moe_chunk_sort as M


def _layout(seed: int, n_splits: int, max_chunk: int = 9):
    g = torch.Generator().manual_seed(seed)
    ss = torch.randint(0, max_chunk, (n_splits,), generator=g)
    si = torch.randperm(n_splits, generator=g)
    return ss, si, int(ss.sum())


def test_gather_map_reproduces_megatron_cat_over_random_layouts():
    """Forward bits, including layouts with EMPTY chunks (the case a naive offset walk gets wrong)."""
    for seed in range(200):
        n_splits = 1 + (seed % 12)
        ss, si, n = _layout(seed, n_splits)
        if n == 0:
            continue
        x = torch.randn(n, 5)
        p = torch.randn(n)
        ref, refp = M._reference(x, ss, si, p)
        got, gotp = M.sort_chunks_gather(x, ss, si, p)
        assert torch.equal(ref, got), f"seed={seed} splits={ss.tolist()} order={si.tolist()}"
        assert torch.equal(refp, gotp)


def test_row_provenance_is_megatrons_permutation_not_merely_equal_payloads():
    """A random payload can pass on a wrong-but-symmetric map. An index marker cannot."""
    for seed in range(60):
        ss, si, n = _layout(1000 + seed, 1 + (seed % 9))
        if n == 0:
            continue
        marker = torch.arange(n, dtype=torch.int64).unsqueeze(1)
        ref, _ = M._reference(marker, ss, si, None)
        g = M.gather_map(ss, si, n, torch.device("cpu"))
        assert torch.equal(ref.reshape(-1), g)


def test_map_is_a_bijection_and_the_inverse_restores_the_input():
    for seed in range(60):
        ss, si, n = _layout(2000 + seed, 1 + (seed % 9))
        if n == 0:
            continue
        g = M.gather_map(ss, si, n, torch.device("cpu"))
        assert torch.equal(torch.sort(g).values, torch.arange(n, dtype=g.dtype))
        x = torch.randn(n, 3)
        assert torch.equal(x.index_select(0, g).index_select(0, M.inverse_map(g)), x)


def test_backward_is_bitwise_equal_to_the_cat_expressions_vjp():
    """The gate the halo report held fix 1 back for: gradients are not covered by the zero-KL gate,
    so a subtly wrong VJP would be invisible. Assert it instead."""
    for seed in range(40):
        ss, si, n = _layout(3000 + seed, 1 + (seed % 8))
        if n == 0:
            continue
        base = torch.randn(n, 6)
        pbase = torch.randn(n)
        cot = torch.randn(n, 6)
        cotp = torch.randn(n)

        xa = base.clone().requires_grad_(True)
        pa = pbase.clone().requires_grad_(True)
        ra, rap = M._reference(xa, ss, si, pa)
        torch.autograd.backward([ra, rap], [cot, cotp])

        xb = base.clone().requires_grad_(True)
        pb = pbase.clone().requires_grad_(True)
        rb, rbp = M.sort_chunks_gather(xb, ss, si, pb)
        torch.autograd.backward([rb, rbp], [cot, cotp])

        assert torch.equal(xa.grad, xb.grad), f"seed={seed}"
        assert torch.equal(pa.grad, pb.grad)


def test_signed_zeros_survive_both_directions():
    """Why the VJP is two gathers and never ``index_add_``: ``0.0 + (-0.0) == +0.0``."""
    ss = torch.tensor([2, 3])
    si = torch.tensor([1, 0])
    x = torch.tensor([[-0.0], [0.0], [-0.0], [0.0], [-0.0]])
    ref, _ = M._reference(x, ss, si, None)
    got, _ = M.sort_chunks_gather(x, ss, si, None)
    assert torch.equal(torch.signbit(ref), torch.signbit(got))

    xa = x.clone().requires_grad_(True)
    cot = torch.tensor([[-0.0], [0.0], [-0.0], [0.0], [-0.0]])
    out, _ = M.sort_chunks_gather(xa, ss, si, None)
    out.backward(cot)
    # An index_add_ VJP would turn every -0.0 in the cotangent into +0.0.
    assert torch.signbit(xa.grad).any(), "the inverse-gather VJP must preserve negative zeros"


def test_reference_matches_megatrons_source_expression():
    """``_reference`` is the thing admission compares against; if it drifts from megatron the whole
    gate is vacuous. Pin it to the literal expression at moe_utils.py:569-570.

    Note the shape megatron actually ships: the ``torch.split`` is HOISTED OUT of the comprehension.
    Any OFF-side benchmark that inlines it (as the flags.py registry quotation does) re-splits per
    chunk and overstates the baseline by ~500x."""
    ss = torch.tensor([3, 0, 2])
    si = torch.tensor([2, 0, 1])
    x = torch.arange(5 * 2, dtype=torch.float32).reshape(5, 2)
    parts = torch.split(x, ss.tolist(), dim=0)
    expect = torch.cat([parts[i] for i in si.tolist()], dim=0)
    got, _ = M._reference(x, ss, si, None)
    assert torch.equal(expect, got)


def test_admission_is_independent_of_the_callers_grad_mode():
    """REGRESSION, defect A -- the reason this op never once engaged in production.

    Gate (iii) runs ``torch.autograd.backward`` on the live operands, but EVERY MoE forward on this
    stack is under ``torch.no_grad()``: scoring wraps the whole forward
    (``megatron_worker.py:1020``), and training does too because the arm pins
    ``recompute_granularity=full`` / ``recompute_num_layers=1``, so every layer enters through
    megatron's ``CheckpointFunction.forward``, whose run_function is called under ``no_grad``
    (``tensor_parallel/random.py:580-581``). Unfixed, ``_admit`` raised "element 0 of tensors does
    not require grad", the shape was written off permanently, and the backward recompute -- which
    DOES run under ``enable_grad`` (``random.py:620``) -- arrived to find the key already poisoned.
    """
    ss = torch.tensor([3, 0, 2, 5])
    si = torch.tensor([2, 0, 3, 1])
    n = int(ss.sum())
    x = torch.randn(n, 8)
    p = torch.randn(n)

    assert M._admit(x, ss, si, p) == (True, ""), "grad-enabled admission must pass"
    with torch.no_grad():
        assert M._admit(x, ss, si, p) == (True, ""), "admission must not depend on the caller's grad mode"


def test_admission_probe_leaves_no_grad_fn_on_the_callers_tensors():
    """The probe is safe inside a caller's ``no_grad`` only because it works on detached clones."""
    ss = torch.tensor([4, 1, 3])
    si = torch.tensor([1, 2, 0])
    n = int(ss.sum())
    x = torch.randn(n, 4)
    with torch.no_grad():
        M._admit(x, ss, si, None)
        out, _ = M.sort_chunks_gather(x, ss, si, None)
    assert out.grad_fn is None and out.requires_grad is False
    assert x.grad is None and x.grad_fn is None


def test_forward_bits_are_unchanged_under_no_grad():
    """The gather must be the same bits whether or not the caller holds a grad context."""
    for seed in range(40):
        ss, si, n = _layout(4000 + seed, 1 + (seed % 8))
        if n == 0:
            continue
        x = torch.randn(n, 7)
        p = torch.randn(n)
        ref, refp = M._reference(x, ss, si, p)
        with torch.no_grad():
            got, gotp = M.sort_chunks_gather(x, ss, si, p)
        assert M._bitcmp(ref, got) == 0
        assert M._bitcmp(refp, gotp) == 0


def test_admission_key_carries_no_data_dependent_axis():
    """REGRESSION, defect B -- the row count is a PRECONDITION, not a shape class.

    ``num_tokens`` is the routed-row count of one microbatch on one layer, so it is effectively
    unique per (layer, microbatch). With it in the key the ~2-3 ms five-gate probe ran on essentially
    every one of the ~20,480 calls per forward to save ~147 us each -- a net loss. Every sibling op
    keys on the weight axes only (``mm_cublaslt`` on ``(K, N)``, ``moe_expert_cublaslt`` on
    ``(K, N)``); this one now keys on ``(num_splits, hidden, dtype)``.
    """
    ss = torch.tensor([3, 0, 2, 5])
    a = torch.randn(int(ss.sum()), 128)

    ss2 = torch.tensor([100, 40, 60, 96])
    b = torch.randn(int(ss2.sum()), 128)

    assert a.shape[0] != b.shape[0], "the two probes must differ in row count for this to prove anything"
    assert M.admission_key(a, ss) == M.admission_key(b, ss2) == (4, 128, str(torch.float32))

    # ... and the axes that DO belong in the key still separate.
    assert M.admission_key(a.to(torch.bfloat16), ss) != M.admission_key(a, ss)
    assert M.admission_key(torch.randn(int(ss.sum()), 256), ss) != M.admission_key(a, ss)
    assert M.admission_key(a, torch.tensor([5, 5])) != M.admission_key(a, ss)


def test_preconditions_accept_wellformed_vectors_and_refuse_malformed_ones():
    """What each call re-establishes now that the probe does not re-run: the two integer facts gate
    (ii)'s algebra needs. Fail-closed -- the battery's gate-7 control depends on the refusals."""
    ss = torch.tensor([3, 0, 2, 5])
    si = torch.tensor([2, 0, 3, 1])
    n = int(ss.sum())
    assert M.preconditions_ok(ss, si, n) == (True, "")

    # 1. the chunks must tile the rows
    ok, why = M.preconditions_ok(ss, si, n + 1)
    assert not ok and "tile" in why

    # 2. sorted_idxs must be a permutation -- the battery's gate-7 control
    ok, why = M.preconditions_ok(torch.tensor([512, 512]), torch.tensor([0, 0]), 1024)
    assert not ok and "permutation" in why

    # out of range is caught by the same fact
    ok, why = M.preconditions_ok(ss, torch.tensor([0, 1, 2, 9]), n)
    assert not ok and "permutation" in why

    # negative chunks, and mismatched vector lengths
    ok, why = M.preconditions_ok(torch.tensor([6, -1]), torch.tensor([1, 0]), 5)
    assert not ok and "negative" in why
    ok, why = M.preconditions_ok(ss, torch.tensor([0, 1]), n)
    assert not ok and "mismatch" in why


def test_a_malformed_call_does_not_write_off_the_shape(monkeypatch):
    """A bad ``sorted_idxs`` is a statement about that CALL. With a shape-only key, caching it would
    disable a whole shape class forever on one malformed microbatch."""
    monkeypatch.setenv(M._ENV_GATE, "1")
    M._STATE.clear()

    class _PretendCuda:
        """``chunk_sort_ready`` reads only metadata before the precondition gate, so a CPU test can
        get past the device check without a GPU. It never reaches ``_admit``: the preconditions
        refuse first, which is precisely the claim under test."""

        def __init__(self, t):
            self._t = t
            self.device = torch.device("cuda")
            self.dtype = t.dtype

        def dim(self):
            return self._t.dim()

        def numel(self):
            return self._t.numel()

        @property
        def shape(self):
            return self._t.shape

    ss = torch.tensor([512, 512])
    x = _PretendCuda(torch.randn(1024, 64))
    assert M.chunk_sort_ready(x, ss, torch.tensor([0, 0]), None) is False  # not a permutation
    assert "permutation" in M.chunk_sort_stats()[2]
    # same shape key (2, 64, float32), a row count the splits do not tile
    assert M.chunk_sort_ready(x, torch.tensor([100, 200]), torch.tensor([1, 0]), None) is False
    assert "tile" in M.chunk_sort_stats()[2]
    assert M._STATE == {}, "a per-call refusal must not poison the shape's verdict"


def test_state_table_is_bounded():
    assert M._STATE_CAP <= 4096, "an unbounded verdict dict on a call this hot is a leak"


def test_served_census_exists_because_an_install_banner_is_not_engagement():
    served, declined, reason = M.chunk_sort_stats()
    assert isinstance(served, int) and isinstance(declined, int) and isinstance(reason, str)


def test_the_installed_wrapper_counts_served_and_declined_and_stays_bit_equal(monkeypatch):
    """The wrapper is what the live arm is read from. With a shape-only key ``ADMITTED`` prints ~2
    lines per rank for a whole run, so the census -- not the INSTALLED banner -- is engagement."""
    import pytest

    td = pytest.importorskip("megatron.core.transformer.moe.token_dispatcher")

    monkeypatch.setenv(M._ENV_GATE, "1")
    monkeypatch.setattr(M, "_installed", False, raising=False)
    orig = td.sort_chunks_by_idxs
    try:
        assert M.install_chunk_sort() is True
        assert td.sort_chunks_by_idxs is not orig

        ss = torch.tensor([3, 0, 2, 5])
        si = torch.tensor([2, 0, 3, 1])
        n = int(ss.sum())
        x = torch.randn(n, 6)
        p = torch.randn(n)
        ref, refp = M._reference(x, ss, si, p)

        # CPU input -> the shipped predicate declines (not cuda); the wrapper must still be correct.
        s0, d0, _ = M.chunk_sort_stats()
        out, outp = td.sort_chunks_by_idxs(x, ss, si, probs=p)
        s1, d1, reason = M.chunk_sort_stats()
        assert (s1, d1) == (s0, d0 + 1), "a fall-through must be counted as declined"
        assert reason == "input is not on cuda"
        assert torch.equal(ref, out) and torch.equal(refp, outp)

        # ... and when the predicate admits, the gather runs and `served` moves.
        monkeypatch.setattr(M, "chunk_sort_ready", lambda *a, **k: True)
        out, outp = td.sort_chunks_by_idxs(x, ss, si, probs=p)
        s2, d2, _ = M.chunk_sort_stats()
        assert (s2, d2) == (s1 + 1, d1), "a gather must be counted as served"
        assert M._bitcmp(ref, out) == 0 and M._bitcmp(refp, outp) == 0

        # fused=True is delegated untouched and is not censused.
        td.sort_chunks_by_idxs(x, ss, si, probs=p, fused=False)
        assert M.chunk_sort_stats()[0] == s2 + 1
    finally:
        td.sort_chunks_by_idxs = orig
        M._installed = False
        M._orig_sort_chunks = None


def test_reject_prints_are_rate_limited(capsys):
    """~20,480 calls per forward per rank: an unbounded refusal print is a log flood, and a flooded
    log is an unread log."""
    M._reject_prints = 0
    for _ in range(4096):
        M._print_reject("synthetic")
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if "REJECTED" in ln]
    assert 1 <= len(lines) <= 32, f"{len(lines)} lines for 4096 refusals"
    assert "REJECTED[#1]" in out and "REJECTED[#4096]" in out
    M._reject_prints = 0


def test_bitcmp_sees_signed_zero_where_torch_equal_does_not():
    a = torch.tensor([0.0, 1.0])
    b = torch.tensor([-0.0, 1.0])
    assert torch.equal(a, b)  # torch.equal is blind to it -- mm_tiles hazard 6
    assert M._bitcmp(a, b) == 1


def test_module_reads_no_env_at_import_time_and_is_off_by_default(monkeypatch):
    """Defaults live in the readers, so a flag forwarded to a Ray actor is honoured there."""
    import inspect

    src = inspect.getsource(M)
    body = src.split("def chunk_sort_enabled")[0]
    assert "os.environ" not in body.split('"""', 2)[-1], "no env read at module scope"
    monkeypatch.delenv(M._ENV_GATE, raising=False)
    assert M.chunk_sort_enabled() is False
    monkeypatch.setenv(M._ENV_GATE, "1")
    assert M.chunk_sort_enabled() is True


def test_flag_is_registered_default_off_and_forwarded_to_the_trainer_actor():
    from skyrl.backends.skyrl_train.isoexec.core import flags as F

    flag = next(f for f in F.FLAGS if f.name == "SKYRL_ISOEXEC_MOE_CHUNK_SORT")
    assert flag.default == "0"
    assert flag.sides == ("both",)
    assert "SKYRL_ISOEXEC_MOE_CHUNK_SORT" in F.actor_forwarding_tuple(F.TRAIN)


def test_registry_declares_the_impl_as_bit_free_in_both_directions():
    from skyrl.backends.skyrl_train.isoexec.core.registry import Registry
    from skyrl.backends.skyrl_train.isoexec.ops.moe import _register

    reg = Registry()
    _register.register(reg)
    op = reg.get_op("moe.dispatch")
    impl = op.impls["chunk_sort_gather"]
    ma = impl.rounding.machine_assertable
    assert ma["bit_free"] is True
    assert ma["vjp"] == "inverse_gather"
    assert ma["bit_equal_to"] == "megatron_cat_chunk_loop"
