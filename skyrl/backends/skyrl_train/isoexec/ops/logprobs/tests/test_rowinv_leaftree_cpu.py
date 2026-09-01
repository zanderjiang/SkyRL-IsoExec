"""The rowinv leaf-tree logprob denominator is TP-invariant; rank-boundary folding is not.

Pins the expression tree, not the kernel's leaf-internal microstructure: leaf boundaries and
combine order are functions of G alone, so TP only moves leaf ownership. A per-element Kahan in
vocab-index order stands in for the kernel below.
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
V, M = 4096, 4  # leaf_k = 512; rows are independent by construction, so a handful is enough


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int32)


def _data():
    g = torch.Generator().manual_seed(11)
    # Adversarial spread: the exp values span ~35 binades after the max subtraction, so every
    # reassociation or moved fold boundary has low-order bits to disturb.
    x = (torch.rand(M, V, generator=g) * 80.0 - 80.0).to(torch.float32)
    x[torch.arange(M), torch.randint(0, V, (M,), generator=g)] = 0.0  # pin each row max at a known site
    target = torch.randint(0, V, (M,), generator=g)
    return x, target


def _kahan(e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Kahan fp32 sum over the last dim in vocab-index order, elementwise over rows.

    Returns (sum, compensation). No op here can see how many rows sit alongside.
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

    Each rank combines the G/tp leaves its shard holds with its local subtree; the cross-rank
    combine applies the remaining tree levels. At tp=1 every leaf is local and the full tree runs.
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
    """The inadmissible control: each rank folds its whole V/tp shard into ONE partial.

    A plain sequential fp32 accumulation, so its boundaries sit at V/tp and move with the world;
    the partials then meet the same balanced tree, isolating the moved boundary. Returns the raw
    denominator S, before ``log`` can compress a low-order difference onto the same grid point.
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

    C=4 vs C=8 is the pair that matters: the trainer scores at TP=4 while the engine serves at
    TP=8 (handing the patch a gathered full row, i.e. C=1 of the same tree).
    """
    x, target = _data()
    plan = ReductionPlan(num_leaves=G, leaf_dtype=torch.float32)
    ref = _scheme_at_tp(x, target, plan, 1)
    for tp in (2, 4, 8):
        got = _scheme_at_tp(x, target, plan, tp)
        assert torch.equal(_bits(ref), _bits(got)), f"TP={tp} diverged from TP=1 under the fixed leaf tree"


def test_rank_boundary_folding_is_world_dependent():
    """The naive per-shard fold is refused with a demonstration: TP=4 folds across an interior
    boundary TP=8 cuts, so the fold boundary must never be a function of the world size."""
    x, _ = _data()
    tp4 = _rank_partial_fold_at_tp(x, 4)
    tp8 = _rank_partial_fold_at_tp(x, 8)
    assert not torch.equal(_bits(tp4), _bits(tp8)), (
        "rank-boundary folding came out TP-invariant on adversarial data -- the impossibility "
        "control is vacuous, tighten the data"
    )


def test_rows_are_independent_in_the_expression():
    """Row i evaluated alone == row i evaluated inside the batch, bit for bit, at every C.

    The expression contains no cross-row term, so the row count cannot reach the bits.
    """
    x, target = _data()
    plan = ReductionPlan(num_leaves=G, leaf_dtype=torch.float32)
    for tp in (1, 4, 8):
        batch = _scheme_at_tp(x, target, plan, tp)
        for i in range(M):
            alone = _scheme_at_tp(x[i : i + 1], target[i : i + 1], plan, tp)
            assert torch.equal(_bits(batch[i : i + 1]), _bits(alone)), f"row {i} moved with the batch at TP={tp}"
