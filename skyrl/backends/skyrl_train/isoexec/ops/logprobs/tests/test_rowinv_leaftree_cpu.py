"""The rowinv leaf-tree logprob denominator is TP-invariant; rank-boundary folding is not.

This file is the INVARIANCE DERIVATION for the ``rowinv_leaftree`` sampled-logprob impl -- the
composed logprob at all four sites -- as a machine check rather than prose. It is the logprob
counterpart of ``ops/collectives/tests/test_bf16_leaf_scheme_cpu.py`` and mirrors its derivation:

THE CENTRAL FACT it pins: the denominator's leaf granularity is FIXED. The vocabulary is cut into
``G`` contiguous leaves whatever the TP degree is; TP only moves leaf OWNERSHIP (which rank's shard
holds which contiguous leaf range), never the leaf boundaries and never the combine order
(``combine_order`` is a function of G alone). Every per-leaf Kahan fp32 exp sum therefore runs over
the same K-offsets, on the same values, in the same order, and the leaf sums meet in the same tree
positions, at every TP degree dividing G -- the schedule commutes with the choice of C, and the
engine's already-gathered full row (``parallel_output=False``, group=None) is simply C=1 of the
same expression. A fold applied at the RANK boundary instead sums over V/C-offsets, which move with
C -- TP=4 folds 4 partials where TP=8 folds 8, different functions, different bits. That moving
boundary is the same defect class as ATen's fp32 vocab ``sum`` (shape-dependent schedule: the
trainer reducing [B,S,V] and the engine reducing sampled rows are different functions -- the
measured 72.2% non-bitwise tokens), which is why no execution twin of the incumbent can serve both
runtimes and the leaf tree replaces it on both at once. Asserted below on adversarial data:

  * scheme(fixed leaves, Kahan fp32 per leaf, pik combine tree, single log) is BIT-IDENTICAL
    across C in {1, 2, 4, 8} under G=8 -- including the trainer/engine pair (C=4 vs C=8) -- for
    the denominator AND for the final ``(x_t - m) - log(S)``;
  * the naive alternative -- each rank folding its shard into ONE partial before the wire --
    produces DIFFERENT bits at C=4 vs C=8, i.e. it is refused with a demonstration, not an
    assumption;
  * rows are independent in the expression: evaluating one row alone equals evaluating it inside a
    batch, bit for bit. This is the row-count-invariance HALF of the design as far as CPU algebra
    can carry it -- the expression contains no cross-row term.

Arithmetic note: what is pinned here is the EXPRESSION TREE, exact algebra over whatever the
per-element ``exp`` values are. The kernel's leaf-internal microstructure (its fixed BLOCK tiling
of the Kahan sum, one program per row) is pinned by ``ops/logprobs/rowinv.py`` and its own battery,
not by this file; a per-element Kahan in vocab-index order stands in for it below, and the claims
asserted -- fixed boundaries, fixed combine order, no cross-row and no cross-world term -- are
about the tree, which the stand-in shares by construction.

Run (CPU only):
    uv run --extra dev pytest skyrl/backends/skyrl_train/isoexec/ops/logprobs/tests/test_rowinv_leaftree_cpu.py -q
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

torch = pytest.importorskip("torch")

_HERE = os.path.dirname(os.path.abspath(__file__))
_COLLECTIVES_DIR = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "collectives")


def _load_by_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_BOOT = _load_by_path("_ix_pik_bootstrap_rowinv", os.path.join(_COLLECTIVES_DIR, "pik_bootstrap.py"))
_BOOT.ensure_pik()
from pik.plan import ReductionPlan, combine_order  # type: ignore  # noqa: E402

G = 8
V, M = 4096, 4  # leaf_k = 512; a handful of rows is enough -- rows are independent by construction


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int32)


def _data():
    g = torch.Generator().manual_seed(11)
    # Adversarial spread: after the max subtraction the exponents span ~35 binades, so the exp
    # values range from O(1) down to ~1e-35 and every reassociation or moved fold boundary has
    # low-order bits to disturb (the pik probe convention, adapted to an all-positive summand).
    x = (torch.rand(M, V, generator=g) * 80.0 - 80.0).to(torch.float32)
    x[torch.arange(M), torch.randint(0, V, (M,), generator=g)] = 0.0  # pin each row max at a known site
    target = torch.randint(0, V, (M,), generator=g)
    return x, target


def _kahan(e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Kahan fp32 sum over the LAST dim in vocab-index order; every op elementwise over rows.

    Returns (sum, compensation). Elementwise over the row dim by construction, which is what makes
    every claim below row-count-invariant: no op here can see how many rows sit alongside.
    """
    s = torch.zeros(e.shape[:-1], dtype=torch.float32)
    c = torch.zeros_like(s)
    for j in range(e.shape[-1]):
        y = e[..., j] - c
        t = s + y
        c = (t - s) - y
        s = t
    return s, c


def _tree(vals: list[torch.Tensor]) -> torch.Tensor:
    """Apply pik's fixed balanced combine tree (the real combine_order) in fp32."""
    slots = [v.clone() for v in vals]
    for dst, lhs, rhs in combine_order(len(vals)):
        slots[dst] = slots[lhs] + slots[rhs]
    return slots[0]


def _leaf_sums(e: torch.Tensor, plan: ReductionPlan) -> list[torch.Tensor]:
    """The G fixed per-leaf Kahan fp32 exp sums. Boundaries are a function of G alone."""
    lk = plan.leaf_k(V)
    return [_kahan(e[..., g * lk : (g + 1) * lk])[0] for g in range(plan.num_leaves)]


def _scheme_at_tp(x: torch.Tensor, target: torch.Tensor, plan: ReductionPlan, tp: int) -> torch.Tensor:
    """Evaluate ``(x_t - m) - log(S)`` the way TP=tp evaluates it.

    ``m`` is the global row max (all_reduce(MAX) -- exact, so world-free). Each rank owns the G/tp
    leaves its contiguous vocab shard holds (``rank_leaf_range``) and combines them with its local
    subtree; the cross-rank combine applies the REMAINING tree levels over the rank partials. At
    tp=1 (the engine's gathered full row) every leaf is local and the same full tree runs.
    """
    m = x.amax(dim=-1)  # exact: MAX has no rounding, so shard-then-all_reduce(MAX) == row amax
    e = torch.exp(x - m.unsqueeze(-1))
    leaves = _leaf_sums(e, plan)
    partials = []
    for r in range(tp):
        lo, hi = plan.rank_leaf_range(r, tp)
        partials.append(_tree(leaves[lo:hi]))
    s = _tree(partials)
    x_t = x.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return (x_t - m) - torch.log(s)


def _rank_partial_fold_at_tp(x: torch.Tensor, tp: int) -> torch.Tensor:
    """The INADMISSIBLE control: each rank folds its whole V/tp shard into ONE partial.

    The fold is a plain sequential fp32 accumulation over the shard -- the incumbent's own shape
    (a per-shard ``sum`` handed to ``all_reduce(SUM)``) -- so its boundaries sit at V/tp and move
    with the world. The partials are then combined by the same balanced tree the scheme uses, so
    any C=4 vs C=8 difference below is attributable to the moved fold boundary alone. Returns the
    raw denominator S: the comparison must happen where the wire happens, before ``log`` gets a
    chance to compress a low-order difference back onto the same grid point.

    (A Kahan-per-shard fold is deliberately NOT the control asserted here: on this data its
    compensated partials round to the same bits at C=4 and C=8, so asserting inequality for it
    would overstate. The moved boundary is a defect of the plain fold actually shipped in the
    incumbent, and that is the function this control refuses.)
    """
    m = x.amax(dim=-1)
    e = torch.exp(x - m.unsqueeze(-1))
    shard = V // tp
    partials = []
    for r in range(tp):
        s = torch.zeros(x.shape[:-1], dtype=torch.float32)
        for j in range(r * shard, (r + 1) * shard):
            s = s + e[..., j]
        partials.append(s)
    return _tree(partials)


def test_leaf_tree_denominator_is_bit_identical_across_tp():
    """One expression, one bit pattern, at C=1,2,4,8 under G=8.

    C=4 vs C=8 is the pair that matters: the Megatron trainer scores at TP=4 while the engine
    serves at TP=8 (and hands the patch a gathered full row, i.e. C=1 of the same tree). Same
    function by construction; asserted, not argued.
    """
    x, target = _data()
    plan = ReductionPlan(num_leaves=G, leaf_dtype=torch.float32)
    ref = _scheme_at_tp(x, target, plan, 1)
    for tp in (2, 4, 8):
        got = _scheme_at_tp(x, target, plan, tp)
        assert torch.equal(_bits(ref), _bits(got)), f"TP={tp} diverged from TP=1 under the fixed leaf tree"


def test_rank_boundary_folding_is_world_dependent():
    """The naive per-shard fold -- one partial per rank before the wire -- is REFUSED with a
    demonstration. At TP=4 the fold runs straight across an interior boundary that TP=8 would have
    cut, evaluating fold(shard 0..V/4) + ... where TP=8 evaluates fold(0..V/8) + fold(V/8..V/4) +
    ... -- different association, different functions. This is exactly why the impl may not "just
    sum the shard": the fold boundary must never be a function of the world size."""
    x, _ = _data()
    tp4 = _rank_partial_fold_at_tp(x, 4)
    tp8 = _rank_partial_fold_at_tp(x, 8)
    assert not torch.equal(_bits(tp4), _bits(tp8)), (
        "rank-boundary folding came out TP-invariant on adversarial data -- the impossibility "
        "control is vacuous, tighten the data"
    )


def test_rows_are_independent_in_the_expression():
    """Row i evaluated alone == row i evaluated inside the batch, bit for bit, at every C.

    The expression contains no cross-row term -- max, exp, Kahan and the combine tree are all
    elementwise over rows -- so the row count cannot reach the bits. This is the property ATen's
    shape-dependent vocab sum does NOT have, and the reason the trainer's every-position reduction
    and the engine's sampled-rows reduction disagreed under the incumbent.
    """
    x, target = _data()
    plan = ReductionPlan(num_leaves=G, leaf_dtype=torch.float32)
    for tp in (1, 4, 8):
        batch = _scheme_at_tp(x, target, plan, tp)
        for i in range(M):
            alone = _scheme_at_tp(x[i : i + 1], target[i : i + 1], plan, tp)
            assert torch.equal(_bits(batch[i : i + 1]), _bits(alone)), f"row {i} moved with the batch at TP={tp}"
