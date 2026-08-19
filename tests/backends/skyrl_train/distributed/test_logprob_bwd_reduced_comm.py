"""CPU proof obligations for SKYRL_ISOEXEC_LOGPROB_BWD_REDUCED_COMM.

THE CLAIM, in one line: the IsoExec FORWARD owes the generator a bitwise lse and therefore must keep
its full-vocab ``all_gather``; the BACKWARD's recompute reaches only the optimizer and therefore must
not. So the two are asserted DIFFERENTLY, and that asymmetry is the whole point:

  1. THE FORWARD IS BIT-IDENTICAL with the flag on and off. Asserted as an int32 BIT compare, not
     ``allclose`` -- ``allclose`` cannot see a last-ulp move, which is exactly the size of the defect
     the gather branch exists to prevent (~60% of rows differ in the last ulp without it). Guarded
     against vacuity by asserting the gather branch actually RAN (``fwd_gather == world``); a forward
     that quietly took the standard path would pass a bit compare against itself.
  2. THE BACKWARD AGREES TO TOLERANCE, and is expected NOT to be bitwise-equal. Both formulations are
     checked against a float64 reference so "they agree" cannot mean "they are both wrong the same
     way", and the observed gap is asserted to be of reassociation size (<1e-5), not merely finite.
  3. THE MAX SHIFT IS LOAD-BEARING and cross-rank. With logits at +200 a naive ``exp(x).sum()``
     overflows to inf in fp32 -- asserted as a CONTROL so the stability test is not vacuous -- and
     the global maximum is planted on a rank that is NOT the rank whose output is checked, so a
     formulation that took a per-shard max instead of ``all_reduce(MAX)`` would fail.
  4. THE FLAG DEFAULTS OFF and every layer of the fail-closed story is asserted: the env default,
     the ``for_backward=False`` parameter default, the forward call site not passing it (an AST scan
     -- a future edit that adds one has to delete this test to land), and the census showing
     ``served == 0`` when the flag is unset. An install banner is not engagement; ``served`` is.

TP is SIMULATED, not gloo'd: ``world`` threads each hold one vocabulary shard and the collectives are
replaced by a barrier-synchronised rendezvous. That is deliberate -- it makes the vocabulary
PARTITION the thing under test (which is what the reduction is about) and needs no process group, no
CUDA and no spawned processes. One consequence is written down rather than discovered: the module's
census counters are process-global, so under threads they count the whole group rather than one rank,
and the census assertions below are stated as ``served > 0`` / ``served == 0`` rather than exact.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/ray/default/SkyRL-IsoExec \\
        /mnt/local_storage/venvs/skyrl-isoexec-zk/bin/python -m pytest <thisfile> -q
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import textwrap
import threading

import pytest
import torch

_has_megatron = importlib.util.find_spec("megatron") is not None

pytestmark = pytest.mark.skipif(not _has_megatron, reason="megatron-core not installed")

if _has_megatron:
    from skyrl.backends.skyrl_train.distributed.megatron import model_utils as M


# =============================================================================================
# simulated tensor parallelism
# =============================================================================================
class _Rendezvous:
    """The all-to-all meeting point every fake collective goes through."""

    def __init__(self, world: int):
        self.world = world
        self.slots: list = [None] * world
        self.barrier = threading.Barrier(world, timeout=60)

    def exchange(self, rank: int, x: torch.Tensor) -> list:
        self.slots[rank] = x
        self.barrier.wait()
        out = list(self.slots)
        self.barrier.wait()  # nobody overwrites a slot until everyone has read it
        return out


class _RankGroup:
    """One rank's view of the process group.

    Rank lives on the OBJECT, not in a thread-local: autograd is free to run a Function's backward
    on a thread other than the one that called ``.backward()``, and a thread-keyed rank would then
    silently mis-attribute a shard.
    """

    def __init__(self, rv: _Rendezvous, rank: int):
        self._rv = rv
        self.rank = rank

    def size(self) -> int:
        return self._rv.world

    def exchange(self, x: torch.Tensor) -> list:
        return self._rv.exchange(self.rank, x)


def _install_fake_collectives(monkeypatch) -> None:
    """Point ``torch.distributed``'s three collectives at the rendezvous, for the test only."""

    def _world_size(group=None, *_a, **_k):
        return group.size() if isinstance(group, _RankGroup) else 1

    def _all_gather(tensor_list, tensor, group=None, async_op=False):
        for dst, src in zip(tensor_list, group.exchange(tensor.detach().clone())):
            dst.copy_(src)

    def _all_reduce(tensor, op=None, group=None, async_op=False):
        vals = group.exchange(tensor.detach().clone())
        acc = vals[0].clone()
        for v in vals[1:]:
            if op == torch.distributed.ReduceOp.MAX:
                acc = torch.maximum(acc, v)
            elif op == torch.distributed.ReduceOp.MIN:
                acc = torch.minimum(acc, v)
            else:
                acc = acc + v
        tensor.copy_(acc)

    monkeypatch.setattr(torch.distributed, "get_world_size", _world_size)
    monkeypatch.setattr(torch.distributed, "all_gather", _all_gather)
    monkeypatch.setattr(torch.distributed, "all_reduce", _all_reduce)
    # so the probe's MIN-reduced verdict really goes through the rendezvous instead of being skipped
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)


def _run_ranks(world: int, body):
    """Run ``body(rank, group)`` on ``world`` threads in lockstep; re-raise the first failure."""
    rv = _Rendezvous(world)
    groups = [_RankGroup(rv, r) for r in range(world)]
    out: list = [None] * world
    err: list = [None] * world

    def _target(r):
        try:
            out[r] = body(r, groups[r])
        except BaseException as e:  # noqa: BLE001 - re-raised below; abort so peers do not hang
            err[r] = e
            rv.barrier.abort()

    threads = [threading.Thread(target=_target, args=(r,), daemon=True) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    for e in err:
        if e is not None:
            raise e
    return out


# =============================================================================================
# fixtures / helpers
# =============================================================================================
@pytest.fixture(autouse=True)
def _isoexec_env(monkeypatch):
    """Every test here runs UNDER IsoExec (the gather branch only exists there) with the lever unset,
    and starts from a clean census."""
    monkeypatch.setenv("SKYRL_ISOEXEC", "1")
    monkeypatch.delenv(M._IX_BWD_ENV, raising=False)
    M._ix_bwd_reset_for_test()
    yield
    M._ix_bwd_reset_for_test()


def _logits(batch: int, seq: int, vocab: int, seed: int = 0, scale: float = 4.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, seq, vocab, generator=g, dtype=torch.float32) * scale


def _shards(full: torch.Tensor, world: int) -> list:
    assert full.shape[-1] % world == 0
    return [s.contiguous() for s in full.split(full.shape[-1] // world, dim=-1)]


def _bits(x: torch.Tensor) -> torch.Tensor:
    """int32 view -- the only honest way to say "the forward did not move"."""
    return x.contiguous().view(torch.int32)


def _ref_log_softmax_f64(full: torch.Tensor) -> torch.Tensor:
    """The ground truth, computed in float64 so it is not one of the two contestants."""
    x = full.double()
    shifted = x - x.amax(-1, keepdim=True)
    return shifted - shifted.exp().sum(-1, keepdim=True).log()


# ---------------------------------------------------------------------------------------------
# how big is the gather-vs-reduced gap ALLOWED to be?  (the scale-free answer)
# ---------------------------------------------------------------------------------------------
# Both branches take the same global max -- ``amax`` is exact and ``all_reduce(MAX)`` of exact
# per-shard maxima is the exact global max -- so ``t = x - logits_max`` is BITWISE identical either
# way and the entire difference lives in lse. Each branch then returns ``fl(t - lse)`` for its own
# lse, so elementwise
#
#     |gather - reduced| = |fl(t - lse_g) - fl(t - lse_r)| <= |lse_g - lse_r| + ulp(|out|)
#
# because each rounding moves its operand by at most half an ulp. Both terms are RELATIVE: the lse
# gap is ~1 ulp of lse, and ulp(|out|) scales with the logit range. There is no absolute constant in
# the statement, which is why the absolute bounds this file used to assert were properties of its own
# toy shape (V = 8*world, scale 4.0) rather than properties of the code -- and are FALSE at the live
# 35B shape, where V=248,320 and scale-20 logits put |out| near 200 and one ulp at 1.53e-5. Measured
# with the helpers below: 300 (V, TP, scale, seed) configurations, zero violations of the elementwise
# bound, and max |gap| / ulp(max|out|) = 1.000.
def _ulp(x: float) -> float:
    """The gap to the next float32 above ``|x|``."""
    t = torch.tensor(abs(x), dtype=torch.float32)
    return float(torch.nextafter(t, torch.tensor(float("inf"))) - t)


def _ulp_like(x: torch.Tensor) -> torch.Tensor:
    """Elementwise ``_ulp``."""
    a = x.abs().float()
    return torch.nextafter(a, torch.full_like(a, float("inf"))) - a


def _lse_pair(full: torch.Tensor, world: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The two log-sum-exps the two branches compute, mirrored operation for operation.

    Mirrored, not approximated: the bound above is only exact if this really is the arithmetic the
    module performs, so the gather leg does ONE ``sum`` over the rank-ordered contiguous row and the
    reduced leg sums per shard and then accumulates in rank order, exactly as the fake
    ``all_reduce(SUM)`` does.
    """
    logits_max = torch.amax(full, dim=-1, keepdim=True)
    work = full.clone()
    work.sub_(logits_max).exp_()
    lse_gather = work.sum(-1, keepdim=True).float().log()

    shard_sums = [(s - logits_max).exp().sum(-1, keepdim=True).float() for s in _shards(full, world)]
    acc = shard_sums[0].clone()
    for s in shard_sums[1:]:
        acc = acc + s
    return lse_gather, acc.log_()


def _assert_reassociation_only(gather: torch.Tensor, reduced: torch.Tensor, lse_gap: torch.Tensor, where: str):
    """The elementwise form of the bound above. Nothing here is calibrated to a shape."""
    gap = (gather - reduced).abs()
    allowed = lse_gap + _ulp_like(torch.maximum(gather.abs(), reduced.abs()))
    over = gap > allowed
    assert not bool(over.any()), (
        f"{where}: {int(over.sum())} element(s) differ by more than the lse gap plus one ulp of the "
        f"output -- worst {float(gap.max()):.3e} against an allowance of "
        f"{float(allowed[over].min()) if bool(over.any()) else 0.0:.3e}. The two branches are then "
        f"no longer the same max-shifted log-softmax reassociated."
    )


def _sweep(full: torch.Tensor, world: int, *, for_backward: bool) -> list:
    """``_compute_distributed_log_softmax`` on every rank of a simulated TP group."""
    parts = _shards(full, world)

    def body(rank, group):
        return M._compute_distributed_log_softmax(parts[rank], group=group, for_backward=for_backward)

    return _run_ranks(world, body)


# =============================================================================================
# 1. THE FORWARD DOES NOT MOVE
# =============================================================================================
@pytest.mark.parametrize("world", [2, 4])
def test_forward_is_bit_identical_flag_on_vs_off(monkeypatch, world):
    """The gate quantity's producer must be byte-for-byte unchanged by this lever.

    Asserted as an int32 bit compare. ``allclose`` would pass on a last-ulp move, and a last-ulp move
    IS the defect the gather branch exists to prevent.
    """
    _install_fake_collectives(monkeypatch)
    full = _logits(2, 6, 4 * world, seed=11)

    monkeypatch.delenv(M._IX_BWD_ENV, raising=False)
    M._ix_bwd_reset_for_test()
    off = _sweep(full, world, for_backward=False)
    # ANTI-VACUITY: the thing being compared must be the GATHER branch, not the standard path.
    assert M.ix_bwd_stats()["fwd_gather"] == world

    monkeypatch.setenv(M._IX_BWD_ENV, "1")
    M._ix_bwd_reset_for_test()
    on = _sweep(full, world, for_backward=False)
    assert M.ix_bwd_stats()["fwd_gather"] == world
    # the forward never reaches the lever at all
    assert M.ix_bwd_stats()["bwd_calls"] == 0

    for rank, (a, b) in enumerate(zip(off, on)):
        assert torch.equal(_bits(a), _bits(b)), f"forward moved on rank {rank} with the flag on"


def _softmax_calls(fn) -> list:
    """Every real CALL to ``_compute_distributed_log_softmax`` in ``fn``, as AST nodes.

    An AST scan, not a substring grep: ``save_for_backward`` contains ``for_backward``, and the
    module's own explanatory comments name the function they are explaining. A grep would report
    both as call sites.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_compute_distributed_log_softmax"
    ]


def test_forward_call_site_does_not_pass_for_backward():
    """Syntactic guard on the contract: the FORWARD must never opt in.

    An edit that makes ``ChunkedDistributedLogprob.forward`` pass ``for_backward`` has to delete this
    test to land -- and it would be trading the IsoExec gate for a wire saving.

    Asserted over EVERY forward call site rather than a single expected one: the forward has more
    than one (it branches on source dtype), and pinning the count only produces a tripwire that
    fires on unrelated refactors while the property itself -- no forward call opts in -- is what
    the gate actually depends on.
    """
    fwd = _softmax_calls(M.ChunkedDistributedLogprob.forward)
    assert fwd, "no forward log-softmax call site found; re-check this guard"
    for call in fwd:
        assert not any(kw.arg == "for_backward" for kw in call.keywords), (
            f"the forward call site at line {call.lineno} opts into the reduced-comm backward; "
            "the forward must stay bitwise"
        )

    bwd = _softmax_calls(M.ChunkedDistributedLogprob.backward)
    assert len(bwd) == 1
    passed = {kw.arg: getattr(kw.value, "value", None) for kw in bwd[0].keywords}
    assert passed.get("for_backward") is True


def test_distributed_logprob_backward_does_not_recompute():
    """Pins the premise correction: ``DistributedLogprob`` does NOT have the same structure.

    Its forward SAVES the softmax (``ctx.save_for_backward``) and its backward reads it off ``ctx``,
    so there is no ``_compute_distributed_log_softmax`` CALL there for this lever to reach and no
    all_gather for it to remove. Asserted so a future reader does not go looking for a call site that
    has never existed.
    """
    assert _softmax_calls(M.DistributedLogprob.backward) == []
    assert _softmax_calls(M.DistributedLogprob.forward), "the forward must compute it to save it"
    assert len(_softmax_calls(M.ChunkedDistributedLogprob.backward)) == 1


# =============================================================================================
# 2. THE BACKWARD AGREES TO TOLERANCE (and is not expected to be bitwise)
# =============================================================================================
def test_backward_world1_is_inert():
    """At TP=1 the gather branch does not exist (it is ``world > 1`` only), so the lever is INERT.

    This is the single-process case, and it is a finding, not a formality: a live arm at
    ``tensor_model_parallel_size=1`` cannot measure this lever at all.
    """
    world = 1
    full = _logits(2, 5, 8, seed=3)
    with pytest.MonkeyPatch.context() as mp:
        _install_fake_collectives(mp)
        mp.delenv(M._IX_BWD_ENV, raising=False)
        M._ix_bwd_reset_for_test()
        off = _sweep(full, world, for_backward=True)
        mp.setenv(M._IX_BWD_ENV, "1")
        on = _sweep(full, world, for_backward=True)
    assert torch.equal(_bits(off[0]), _bits(on[0]))
    assert M.ix_bwd_stats()["bwd_calls"] == 0
    assert M.ix_bwd_stats()["served"] == 0


@pytest.mark.parametrize("world", [2, 4])
def test_backward_reduced_comm_agrees_with_gather(monkeypatch, world):
    """Gather vs reduced-comm in the BACKWARD: reassociation-close, not equal -- and both right.

    Agreement alone would be satisfied by two identically-wrong formulations, so each is also
    compared against a float64 reference. The observed gather-vs-reduced gap is additionally asserted
    to be of REASSOCIATION size; a gap that grew to 1e-3 would still pass a loose allclose but would
    mean something other than summation order had changed.

    CORRECTED BOUND (this test used to assert ``allclose(rtol=1e-6, atol=1e-6)`` and
    ``worst < 1e-5``). Both were ABSOLUTE bounds on a quantity that scales with the logit range, so
    both stated a property of this test's toy shape rather than of the code, and ``worst < 1e-5`` is
    measurably FALSE at the live shape -- 1.526e-5 at V=248,320 with scale-20 logits, see
    ``test_backward_agreement_at_the_production_vocabulary``. They are replaced by the derived
    elementwise bound above, which is STRICTLY TIGHTER than what it replaces at every element: over a
    battery of (V, TP, scale, seed) configurations the new allowance is at most 0.238x the old
    ``allclose`` allowance, and here it is ~1.3e-6 against the old ``worst < 1e-5``.
    """
    _install_fake_collectives(monkeypatch)
    full = _logits(2, 7, 8 * world, seed=17)
    ref = _ref_log_softmax_f64(full)
    parts_ref = _shards(ref, world)

    monkeypatch.delenv(M._IX_BWD_ENV, raising=False)
    M._ix_bwd_reset_for_test()
    gather = _sweep(full, world, for_backward=True)
    assert M.ix_bwd_stats()["served"] == 0, "flag unset must not serve the reduced-comm path"
    assert M.ix_bwd_stats()["declined"] > 0

    monkeypatch.setenv(M._IX_BWD_ENV, "1")
    M._ix_bwd_reset_for_test()
    reduced = _sweep(full, world, for_backward=True)
    assert M.ix_bwd_stats()["served"] > 0, "served>0 is the ONLY engagement evidence"
    assert M.ix_bwd_stats()["declined"] == 0
    assert M.ix_bwd_stats()["probes"] > 0, "the live-operand agreement probe must have run"

    lse_gather, lse_reduced = _lse_pair(full, world)
    lse_gap = (lse_gather - lse_reduced).abs()

    worst = 0.0
    for rank in range(world):
        _assert_reassociation_only(gather[rank], reduced[rank], lse_gap, f"rank {rank}")
        worst = max(worst, float((gather[rank] - reduced[rank]).abs().max()))
        # neither formulation is allowed to be the wrong one
        for got in (gather[rank], reduced[rank]):
            assert torch.allclose(got.double(), parts_ref[rank], rtol=1e-5, atol=1e-5)

    # ... and the scale-free headline: the gap is at most one ulp of the largest output (k = 1.000
    # measured over 300 configurations). This is the bound that replaces `worst < 1e-5`.
    biggest = max(float(a.abs().max()) for a in gather)
    assert worst <= 2.0 * _ulp(biggest), (
        f"gather-vs-reduced gap {worst:.3e} is {worst / _ulp(biggest):.2f} ulp of |out|max="
        f"{biggest:.1f} -- too large to be fp32 reassociation"
    )


def test_the_two_formulations_really_do_differ_bitwise(monkeypatch):
    """ANTI-VACUITY for the forward test, and the honest statement of what this lever costs.

    If gather and reduced-comm happened to produce identical bits, the forward bit compare above
    would pass no matter what the lever did, and "the backward moves by ~1e-7" would be an unearned
    claim. They do differ -- measured here -- which is exactly why the forward is not allowed to
    take this path and the backward is.
    """
    world = 4
    _install_fake_collectives(monkeypatch)
    full = _logits(2, 7, 8 * world, seed=17)

    monkeypatch.delenv(M._IX_BWD_ENV, raising=False)
    M._ix_bwd_reset_for_test()
    gather = _sweep(full, world, for_backward=True)
    monkeypatch.setenv(M._IX_BWD_ENV, "1")
    M._ix_bwd_reset_for_test()
    reduced = _sweep(full, world, for_backward=True)

    moved = sum(int((_bits(a) != _bits(b)).sum()) for a, b in zip(gather, reduced))
    total = sum(a.numel() for a in gather)
    worst = max(float((a - b).abs().max()) for a, b in zip(gather, reduced))
    assert moved > 0, "the two formulations are bitwise identical here -- the forward test is vacuous"
    biggest = max(float(a.abs().max()) for a in gather)
    print(
        f"\nbackward gather-vs-reduced: {moved}/{total} elements moved, max |diff| = {worst:.3e} "
        f"= {worst / _ulp(biggest):.3f} ulp of |out|max={biggest:.1f}"
    )
    # The move must be of reassociation SIZE. Stated in ulp, not as `worst < _IX_BWD_PROBE_ATOL/100`:
    # that was a third absolute bound (1e-5, the same one this file used to assert twice more), and
    # it is false at any shape whose outputs reach |out| > 84. Here the ulp form allows 3.8e-6.
    assert worst <= 2.0 * _ulp(biggest), f"{worst:.3e} is {worst / _ulp(biggest):.2f} ulp of |out|max"
    # and the probe's threshold must still sit far above the observed noise at THIS magnitude
    assert worst < M._ix_bwd_probe_tol(biggest) / 100


def test_backward_agreement_at_the_production_vocabulary(monkeypatch):
    """THE SHAPE THE LEVER ACTUALLY RUNS AT: V=248,320, TP=4, wide logit scales, several seeds.

    This test exists because every numerical bound in this file was once calibrated at V=8*world=32
    with scale-4.0 logits, and a bound fitted at V=32 says nothing about V=248,320. Two things are
    asserted here that the toy shape cannot reach:

      1. THE INVARIANT STILL HOLDS at the live shape -- the elementwise lse-gap-plus-one-ulp bound,
         and the scale-free ``gap <= 2 ulp of |out|max``.
      2. THE OLD ABSOLUTE BOUND IS FALSE HERE, asserted rather than merely asserted-about: at least
         one of these configurations exceeds the ``worst < 1e-5`` this file used to require. If that
         ever stops being true this test has lost its point and should be re-derived, not relaxed --
         which is precisely the mistake it is here to prevent.

    Rows are kept modest (64) because the gap is a per-ROW coin flip, not a per-element one: within a
    row ``t = x - max`` is exact and lands on the fp32 grid, so ``(t - lse) mod ulp`` is set by lse
    alone and every element of a row rounds the same way. 64 rows is enough to see a full ulp.
    """
    world, vocab, rows = 4, 248320, 64
    _install_fake_collectives(monkeypatch)

    over_the_old_bound = []
    for scale, seed in ((4.0, 0), (20.0, 1), (40.0, 1)):
        full = _logits(1, rows, vocab, seed=seed, scale=scale)

        monkeypatch.delenv(M._IX_BWD_ENV, raising=False)
        M._ix_bwd_reset_for_test()
        gather = _sweep(full, world, for_backward=True)
        assert M.ix_bwd_stats()["served"] == 0

        monkeypatch.setenv(M._IX_BWD_ENV, "1")
        M._ix_bwd_reset_for_test()
        reduced = _sweep(full, world, for_backward=True)
        assert M.ix_bwd_stats()["served"] > 0

        lse_gather, lse_reduced = _lse_pair(full, world)
        lse_gap = (lse_gather - lse_reduced).abs()
        worst = 0.0
        for rank in range(world):
            _assert_reassociation_only(gather[rank], reduced[rank], lse_gap, f"V={vocab} s={scale} rank {rank}")
            worst = max(worst, float((gather[rank] - reduced[rank]).abs().max()))

        biggest = max(float(a.abs().max()) for a in gather)
        k = worst / _ulp(biggest) if worst else 0.0
        print(
            f"\nV={vocab} TP={world} scale={scale} seed={seed}: worst={worst:.4e} "
            f"|out|max={biggest:.2f} ulp={_ulp(biggest):.4e} k={k:.3f} lse_gap={float(lse_gap.max()):.3e}"
        )
        assert k <= 2.0, f"gap {worst:.3e} is {k:.2f} ulp of |out|max={biggest:.1f} -- not reassociation"
        # the live probe must not be anywhere near firing on a legal difference at this shape
        assert worst < M._ix_bwd_probe_tol(biggest) / 10
        if worst >= 1e-5:
            over_the_old_bound.append((scale, seed, worst))

    assert over_the_old_bound, (
        "no configuration exceeded 1e-5, so this test no longer demonstrates that the absolute "
        "bound this file used to assert is false at the production shape"
    )
    print(f"\nconfigurations exceeding the old absolute `worst < 1e-5`: {over_the_old_bound}")


@pytest.mark.parametrize("world", [2, 4])
def test_full_autograd_forward_bitwise_backward_allclose(monkeypatch, world):
    """The same asymmetry through the real call site: ``ChunkedDistributedLogprob.apply`` + backward.

    This is the test that would catch a wiring mistake -- a ``for_backward`` that never reaches the
    recompute, or one that leaks into the forward.
    """
    _install_fake_collectives(monkeypatch)
    batch, seq, chunk = 2, 12, 4
    vocab = 16 * world
    v_local = vocab // world
    full = _logits(batch, seq, vocab, seed=29)
    g = torch.Generator().manual_seed(5)
    target = torch.randint(0, vocab, (batch, seq), generator=g)
    grad_seed = torch.linspace(0.5, 1.5, steps=batch * seq).reshape(batch, seq)

    def run():
        parts = _shards(full, world)

        def body(rank, group):
            leaf = parts[rank].detach().clone().requires_grad_(True)
            out = M.ChunkedDistributedLogprob.apply(
                leaf, target, rank * v_local, (rank + 1) * v_local, chunk, group, False
            )
            out.backward(grad_seed.clone())
            return out.detach().clone(), leaf.grad.detach().clone()

        return _run_ranks(world, body)

    monkeypatch.delenv(M._IX_BWD_ENV, raising=False)
    M._ix_bwd_reset_for_test()
    off = run()
    assert M.ix_bwd_stats()["served"] == 0

    monkeypatch.setenv(M._IX_BWD_ENV, "1")
    M._ix_bwd_reset_for_test()
    on = run()
    assert M.ix_bwd_stats()["served"] > 0

    for rank in range(world):
        assert torch.equal(_bits(off[rank][0]), _bits(on[rank][0])), f"forward logprobs moved, rank {rank}"
        assert torch.allclose(off[rank][1], on[rank][1], rtol=1e-5, atol=1e-6), f"grad disagreement, rank {rank}"
        # the gradient is EXPECTED to move a little; asserting equality here is the misreading the
        # module comment warns about, so pin the size instead of the identity.
        assert float((off[rank][1] - on[rank][1]).abs().max()) < 1e-5


# =============================================================================================
# 3. THE MAX SHIFT IS LOAD-BEARING, AND IT IS CROSS-RANK
# =============================================================================================
def test_max_shift_survives_logits_that_overflow_a_naive_exp(monkeypatch):
    """The whole reason for subtracting the max, tested with a CONTROL that must fail.

    The global maximum is planted on the LAST rank while the output of rank 0 is what gets checked,
    so a formulation that shifted by its own shard's max -- instead of ``all_reduce(MAX)`` -- would
    still overflow when the peers' contributions arrive in the SUM.
    """
    world = 4
    full = torch.full((1, 3, 8 * world), 100.0)
    full[..., -1] = 200.0  # the global max lives on rank world-1
    full[0, 1, 0] = 150.0

    # CONTROL: without the shift, fp32 really does overflow. If this ever stops being true the
    # stability test below has become vacuous.
    assert torch.isinf(full.exp().sum(-1)).any()

    _install_fake_collectives(monkeypatch)
    monkeypatch.setenv(M._IX_BWD_ENV, "1")
    M._ix_bwd_reset_for_test()
    reduced = _sweep(full, world, for_backward=True)
    assert M.ix_bwd_stats()["served"] > 0

    ref = _shards(_ref_log_softmax_f64(full), world)
    for rank in range(world):
        assert torch.isfinite(reduced[rank]).all(), f"rank {rank} overflowed"
        assert torch.allclose(reduced[rank].double(), ref[rank], rtol=1e-5, atol=1e-5)


# =============================================================================================
# 3b. THE RUNTIME PROBE'S TOLERANCE IS RELATIVE, BECAUSE THE LEGAL DIFFERENCE IS
# =============================================================================================
# The probe REFUSES TO FALL BACK by design: if it fires it kills the step, and at 35B it kills the
# run. So its threshold has two failure modes of very different cost, and both are tested here --
# too tight kills a live run on a legal difference, too loose lets a formulation bug through.
def _big_range_logits(seed: int = 1064, rows: int = 8, vocab: int = 8192) -> torch.Tensor:
    """A LEGAL operand whose gather-vs-reduced gap exceeds a flat 1e-3.

    Nothing is perturbed here: a dense cluster of ordinary logits (so lse really does reassociate
    across the shard boundary) sitting above a floor at -24000 (so |out| ~ 24000, where one ulp is
    1.953e-3). Both branches remain max-shifted log-softmaxes of the same logits; they differ by
    exactly one ulp, k=1.000. This is the ``logits.div_(temperature)`` / diverging-step regime.

    The seed is not decorative. Within a row the flip is a single coin toss with p ~ lse_gap/ulp
    (~1e-4 here), so most seeds show a gap of zero; 1064 is the first seed at which the shipped
    flat-atol probe actually raises. Found by scanning 3000 seeds: 1 hit, i.e. ~4e-5 per row --
    rare per chunk, but a certainty over a run that probes thousands of times.
    """
    g = torch.Generator().manual_seed(seed)
    full = torch.full((1, rows, vocab), -24000.0)
    idx = torch.arange(0, vocab, 4)
    full[..., idx] = torch.randn(1, rows, idx.numel(), generator=g) * 4.0
    return full


def test_probe_tolerance_tracks_the_output_magnitude():
    """The tolerance FUNCTION, before any tensors are involved.

    A flat absolute atol on a difference that scales with |out| has a crossover -- at 1e-3 it is
    |out| = 8388.6 -- past which a LEGAL one-ulp reassociation is reported as drift.
    """
    atol = M._IX_BWD_PROBE_ATOL
    rel_from = M._IX_BWD_PROBE_REL_FROM

    # below the knee nothing changes: this fix does not loosen the regime the lever runs in
    for mag in (0.0, 1.0, 12.0, 200.0, 425.0, rel_from):
        assert M._ix_bwd_probe_tol(mag) == atol, mag
    # every production shape measured sits under the knee (|out|max is 41 / 204 / 425 at V=248,320
    # with scale-4 / -20 / -40 logits), so the flat term is what the live arm will actually use
    assert rel_from >= 425.0

    # above it the tolerance is proportional, and never below a comfortable multiple of one ulp
    for mag in (512.0, 1000.0, 8388.6, 16384.0, 24016.0, 40000.0, 1e5):
        tol = M._ix_bwd_probe_tol(mag)
        assert tol >= 8.0 * _ulp(mag), f"|out|={mag}: tol {tol:.3e} is only {tol / _ulp(mag):.1f} ulp"
        # ... and still far enough below an O(1) formulation bug to catch one
        assert tol <= 0.25, f"|out|={mag}: tol {tol:.3e} would not catch an O(1) bug"
    # monotone, so a bigger operand can never get a smaller allowance
    tols = [M._ix_bwd_probe_tol(m) for m in (1.0, 512.0, 1024.0, 8192.0, 65536.0)]
    assert tols == sorted(tols)

    # a non-finite reference must NOT produce an infinite (i.e. blind) tolerance
    for bad in (float("inf"), float("nan")):
        assert M._ix_bwd_probe_tol(bad) == atol

    # THE RELATIVE TERM IS A MULTIPLIER, NOT AN ADDED TERM. Driving the atol to zero must drive the
    # tolerance to zero at every magnitude -- that is what keeps the adversarial suite's
    # `test_drift_probe_actually_raises` able to prove the probe is not inert.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(M, "_IX_BWD_PROBE_ATOL", 0.0)
        assert [M._ix_bwd_probe_tol(m) for m in (1.0, 200.0, 24016.0, 1e6)] == [0.0] * 4


def test_probe_does_not_raise_on_a_legal_large_magnitude_difference(monkeypatch):
    """A LEGAL one-ulp difference at |out| ~ 24000 must not kill the run.

    This is the D7 regression: with a flat ``_IX_BWD_PROBE_ATOL`` the probe raises here, on an
    operand nothing has tampered with, and it refuses to fall back -- so a live 35B step dies for a
    difference the module's own comment calls expected. Anti-vacuity is built in: the observed probe
    diff is asserted to EXCEED the flat atol, so this test fails loudly if the construction ever
    stops reproducing the condition instead of passing for the wrong reason.
    """
    world = 4
    full = _big_range_logits()
    _install_fake_collectives(monkeypatch)

    monkeypatch.delenv(M._IX_BWD_ENV, raising=False)
    M._ix_bwd_reset_for_test()
    gather = _sweep(full, world, for_backward=True)

    monkeypatch.setenv(M._IX_BWD_ENV, "1")
    M._ix_bwd_reset_for_test()
    reduced = _sweep(full, world, for_backward=True)  # must NOT raise

    stats = M.ix_bwd_stats()
    assert stats["served"] > 0
    assert stats["probes"] > 0, "the probe must have run, or this proves nothing"

    biggest = max(float(a.abs().max()) for a in gather)
    worst = max(float((a - b).abs().max()) for a, b in zip(gather, reduced))
    print(
        f"\nlegal large-magnitude case: |out|max={biggest:.1f} ulp={_ulp(biggest):.4e} "
        f"gap={worst:.4e} k={worst / _ulp(biggest):.3f} probe_diff={stats['max_probe_diff']:.4e} "
        f"tol={M._ix_bwd_probe_tol(biggest):.4e}"
    )
    # ANTI-VACUITY: a flat 1e-3 really would have raised on this.
    assert stats["max_probe_diff"] > M._IX_BWD_PROBE_ATOL, (
        f"probe diff {stats['max_probe_diff']:.3e} no longer exceeds the flat atol "
        f"{M._IX_BWD_PROBE_ATOL:.0e}; this test has stopped exercising the defect it guards"
    )
    # and the difference is still exactly reassociation -- legal, not drift
    lse_gather, lse_reduced = _lse_pair(full, world)
    lse_gap = (lse_gather - lse_reduced).abs()
    for rank in range(world):
        _assert_reassociation_only(gather[rank], reduced[rank], lse_gap, f"rank {rank}")
    assert worst <= 2.0 * _ulp(biggest)


def _local_only_log_softmax(shard: torch.Tensor) -> torch.Tensor:
    """A GENUINE formulation bug: shift and normalize by this shard's own max and sum.

    This is the real mistake the probe exists to catch -- dropping the two cross-rank reductions
    leaves a per-shard log-softmax, which is still finite, still well-shaped and still passes any
    smoke test that only looks at one rank.
    """
    m = shard.amax(-1, keepdim=True)
    z = shard - m
    return z - z.exp().sum(-1, keepdim=True).log()


@pytest.mark.parametrize("case", ["normal", "large_magnitude"])
def test_probe_still_raises_on_a_genuine_formulation_error(monkeypatch, case):
    """The other end of D7: making the tolerance relative must not make the probe blind.

    The reduced-comm leg is replaced -- only inside the probe -- by a per-shard log-softmax, and the
    probe must raise on EVERY rank at both a normal magnitude and at the |out| ~ 24000 magnitude
    where the tolerance is at its widest (4.7e-2 there, still ~20x below an O(1) error).
    """
    world = 4
    _install_fake_collectives(monkeypatch)
    monkeypatch.setenv(M._IX_BWD_ENV, "1")
    M._ix_bwd_reset_for_test()

    orig = M._compute_distributed_log_softmax

    # Pass through the rest of the real signature so the stub cannot drift out of step with it.
    def broken(vocab_parallel_logits, group, for_backward=False, **kwargs):
        if for_backward and M._ix_bwd_probing():
            return _local_only_log_softmax(vocab_parallel_logits)
        return orig(vocab_parallel_logits, group=group, for_backward=for_backward, **kwargs)

    monkeypatch.setattr(M, "_compute_distributed_log_softmax", broken)

    full = _logits(1, 8, 8 * world, seed=17) if case == "normal" else _big_range_logits()
    parts = _shards(full, world)

    with pytest.raises(RuntimeError, match="DRIFT"):
        _run_ranks(world, lambda r, g: broken(parts[r], group=g, for_backward=True))


# =============================================================================================
# 4. FAIL CLOSED
# =============================================================================================
def test_flag_defaults_off(monkeypatch):
    """Layer 1 of fail-closed: the env default."""
    monkeypatch.delenv(M._IX_BWD_ENV, raising=False)
    assert M.ix_bwd_reduced_comm_enabled() is False
    monkeypatch.setenv(M._IX_BWD_ENV, "0")
    assert M.ix_bwd_reduced_comm_enabled() is False
    monkeypatch.setenv(M._IX_BWD_ENV, "1")
    assert M.ix_bwd_reduced_comm_enabled() is True


def test_for_backward_parameter_defaults_false():
    """Layer 2: every pre-existing caller keeps today's behaviour with no edit."""
    sig = inspect.signature(M._compute_distributed_log_softmax)
    assert sig.parameters["for_backward"].default is False


def test_flag_is_catalogued_and_its_census_default_matches_the_read_site(monkeypatch):
    """The census is only trustworthy if a flag's recorded default is the one the code actually
    falls back to. This lever is opt-in, so both halves have to say "0"."""
    from skyrl.backends.skyrl_train.isoexec.core import flags as F

    flag = F.get(M._IX_BWD_ENV)
    assert flag.default == "0"
    assert flag.disposition == F.DEPLOYMENT
    # The read site's own fallback, observed rather than restated as a literal.
    monkeypatch.delenv(M._IX_BWD_ENV, raising=False)
    assert M.ix_bwd_reduced_comm_enabled() is False


def test_served_is_zero_when_admission_is_granted_but_nothing_runs(monkeypatch):
    """``served`` must count EXECUTIONS, not admissions -- the project rule is that a banner is not
    engagement and only a served count is, so the counter has to earn that.

    The failure this pins: with the increment inside ``_ix_bwd_admit``, a lever whose verdict was
    granted but whose reduced-comm branch never ran still reported ``served`` climbing. Here
    admission is granted in full (the census advances, the banner fires, the probe runs) and then the
    verdict is turned into a decline, so the reduced-comm formulation never executes. ``served`` must
    stay at 0, and the byte savings with it -- nothing was saved, because nothing was skipped.
    """
    world = 2
    _install_fake_collectives(monkeypatch)
    monkeypatch.setenv(M._IX_BWD_ENV, "1")
    M._ix_bwd_reset_for_test()

    real_admit = M._ix_bwd_admit

    # Trailing args are positional at the call site; absorb them so the stub cannot drift.
    def admits_then_declines(vocab_parallel_logits, group, w, *args, **kwargs):
        real_admit(vocab_parallel_logits, group, w, *args, **kwargs)  # full path, verdict discarded
        return False

    monkeypatch.setattr(M, "_ix_bwd_admit", admits_then_declines)

    full = _logits(1, 6, 8 * world, seed=23)
    _sweep(full, world, for_backward=True)

    stats = M.ix_bwd_stats()
    assert stats["bwd_calls"] > 0, "the lever must have been reached, or this proves nothing"
    assert stats["probes"] > 0, "admission must have run to completion, or this proves nothing"
    assert stats["served"] == 0, (
        f"served={stats['served']} after a path that was admitted but never executed -- `served` is "
        f"counting admissions again, and it is the only engagement evidence this lever has"
    )
    assert stats["wire_bytes_saved"] == 0
    assert stats["buffer_bytes_saved"] == 0


def test_served_and_bytes_advance_once_per_executed_chunk(monkeypatch):
    """The positive half: one served chunk per reduced-comm execution, with the byte census
    describing the SHARD that would have been gathered.

    Run single-threaded (world simulated but only rank 0 stepped through a fake 1-rank exchange is
    not possible here, so use the real 2-thread harness and compare against the group total) -- the
    counters are process-global, so what is checked is the per-chunk INCREMENT, not an absolute.
    """
    world = 2
    _install_fake_collectives(monkeypatch)
    monkeypatch.setenv(M._IX_BWD_ENV, "1")
    M._ix_bwd_reset_for_test()

    full = _logits(1, 6, 8 * world, seed=23)
    parts = _shards(full, world)
    shard_bytes = parts[0].numel() * parts[0].element_size()

    _sweep(full, world, for_backward=True)
    after_one = M.ix_bwd_stats()
    assert after_one["served"] > 0
    # every served chunk contributes (world-1) shards of wire and `world` shards of buffer
    assert after_one["wire_bytes_saved"] == after_one["served"] * (world - 1) * shard_bytes
    assert after_one["buffer_bytes_saved"] == after_one["served"] * world * shard_bytes


def test_banner_prints_on_and_off_and_is_not_engagement(monkeypatch, capsys):
    """A banner is not engagement. Both states must announce themselves, and only the OFF state may
    show ``served=0`` after a backward that reached the lever."""
    world = 2
    full = _logits(1, 4, 8, seed=41)

    _install_fake_collectives(monkeypatch)
    monkeypatch.delenv(M._IX_BWD_ENV, raising=False)
    M._ix_bwd_reset_for_test()
    _sweep(full, world, for_backward=True)
    off_out = capsys.readouterr().out
    assert f"{M._IX_BWD_BANNER} OFF" in off_out
    assert "CENSUS" in off_out and "served=0" in off_out
    assert M.ix_bwd_stats()["served"] == 0

    monkeypatch.setenv(M._IX_BWD_ENV, "1")
    M._ix_bwd_reset_for_test()
    _sweep(full, world, for_backward=True)
    on_out = capsys.readouterr().out
    assert f"{M._IX_BWD_BANNER} ON" in on_out
    assert "CENSUS" in on_out
    assert M.ix_bwd_stats()["served"] > 0
