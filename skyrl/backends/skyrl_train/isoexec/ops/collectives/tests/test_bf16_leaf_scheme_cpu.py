"""The bf16-leaf (reduced-precision wire) scheme is TP-invariant; rank-boundary rounding is not.

This file is the INVARIANCE DERIVATION for the bf16-wire pik all-reduce, as a machine check
rather than prose (companion: ``ops/collectives/LEAF_DTYPE.md`` and the OPEN_WORK design note).

THE CENTRAL FACT it pins: the pik tree's leaf granularity is FIXED. ``ReductionPlan`` cuts K
into ``G`` contiguous leaves whatever the TP degree is; TP only moves leaf OWNERSHIP (which
rank computes which contiguous leaf range), never the leaf boundaries and never the combine
order (``combine_order`` is a function of G alone). Therefore a rounding applied AT the leaf
boundary is applied at the same K-offsets, to the same values, in the same tree positions, at
every TP degree -- it commutes with the choice of C. A rounding applied at the RANK boundary
is applied at K/C-offsets, which move with C -- TP=4 rounds 4 partials where TP=8 rounds 8,
different functions, different bits. Both halves are asserted below on adversarial data:

  * scheme(bf16 leaves, fp32 internal nodes, single root round) is BIT-IDENTICAL across
    C in {1, 2, 4, 8} under G=8 -- including the trainer/engine pair (C=4 vs C=8);
  * the naive alternative -- round each rank PARTIAL to bf16 (what "just ship the wire in
    bf16" means at TP < G) -- produces DIFFERENT bits at C=4 vs C=8, i.e. it is refused
    with a demonstration, not an assumption;
  * the scheme is a DIFFERENT function from the fp32-leaf scheme (why the leaf dtype is a
    manifest-pinned contract constant: flipping it changes bits vs previous runs, and both
    runtimes must flip together);
  * transport neutrality: evaluating the same tree from bf16 leaves that were first upcast
    to fp32 (the NCCL-path emulation -- ``_tree_all_reduce_nccl`` upcasts before moving
    bytes) equals evaluating it with bf16 loads upcast in-kernel (the P2P bf16-wire path).
    The upcast is exact, so WHERE it happens is transport detail, not function.

Arithmetic note: leaves here are torch CPU fp32 matmuls -- NOT the GPU MMA k-order, so the
absolute values differ from a live run. That is fine: every claim asserted is about the
EXPRESSION TREE (which is exact algebra over whatever the leaf values are), not about the
leaf-internal accumulation order, which is pinned separately by pik/arch.py.

Run (CPU only):
    uv run --extra dev pytest skyrl/backends/skyrl_train/isoexec/ops/collectives/tests/test_bf16_leaf_scheme_cpu.py -q
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

torch = pytest.importorskip("torch")

_HERE = os.path.dirname(os.path.abspath(__file__))
_COLLECTIVES_DIR = os.path.dirname(_HERE)


def _load_by_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_BOOT = _load_by_path("_ix_pik_bootstrap_leafscheme", os.path.join(_COLLECTIVES_DIR, "pik_bootstrap.py"))
_BOOT.ensure_pik()
from pik.plan import ReductionPlan, combine_order  # type: ignore  # noqa: E402

G = 8
K_FULL, N, M = 4096, 64, 16  # leaf_k = 512: the o_proj geometry, shrunk in N/M for CPU speed


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int32) if t.dtype == torch.float32 else t.view(torch.int16)


def _data():
    g = torch.Generator().manual_seed(7)
    # adversarial magnitudes: catastrophic-cancellation fodder so any reassociation or moved
    # rounding shows up in the bits (the pik probe convention).
    x = torch.randn(M, K_FULL, generator=g) * torch.logspace(-6, 6, K_FULL, base=10).unsqueeze(0)
    w = torch.randn(N, K_FULL, generator=g)
    return x.to(torch.bfloat16).to(torch.float32), w.to(torch.bfloat16).to(torch.float32)


def _leaves(x, w, plan: ReductionPlan, round_leaf: bool) -> list[torch.Tensor]:
    """The G fixed leaf partials. round_leaf=True applies the bf16-leaf contract's single
    leaf-boundary rounding (exact upcast back: the tree ADDS in fp32)."""
    lk = plan.leaf_k(K_FULL)
    out = []
    for g in range(plan.num_leaves):
        leaf = x[:, g * lk : (g + 1) * lk] @ w[:, g * lk : (g + 1) * lk].T  # fp32 leaf sum
        if round_leaf:
            leaf = leaf.to(torch.bfloat16).to(torch.float32)
        out.append(leaf)
    return out


def _tree(vals: list[torch.Tensor]) -> torch.Tensor:
    """Apply pik's fixed balanced combine tree (the real combine_order) in fp32."""
    slots = [v.clone() for v in vals]
    for dst, lhs, rhs in combine_order(len(vals)):
        slots[dst] = slots[lhs] + slots[rhs]
    return slots[0]


def _scheme_at_tp(x, w, plan: ReductionPlan, tp: int, *, round_leaf: bool, round_rank_partial: bool = False):
    """Evaluate the full row-parallel output the way TP=tp evaluates it.

    Each rank computes the subtree over its contiguous leaf range (``rank_leaf_range``); the
    cross-rank combine applies the REMAINING tree levels over the rank partials. With
    round_rank_partial=True the rank partial is rounded to bf16 before the cross-rank combine
    -- the INADMISSIBLE naive wire scheme, kept here as the impossibility control.
    """
    leaves = _leaves(x, w, plan, round_leaf)
    partials = []
    for r in range(tp):
        lo, hi = plan.rank_leaf_range(r, tp)
        p = _tree(leaves[lo:hi])
        if round_rank_partial:
            p = p.to(torch.bfloat16).to(torch.float32)
        partials.append(p)
    root = _tree(partials)
    return root.to(torch.bfloat16)  # the single root rounding, every TP configuration


def test_bf16_leaf_scheme_is_bit_identical_across_tp():
    """tree(bf16 leaves) at C=1,2,4,8 under G=8: one expression, one bit pattern.

    C=4 vs C=8 is the pair that matters: the Megatron trainer scores at TP=4 (rank partial =
    an fp32 INTERNAL node, m=2) while the engine decodes at TP=8 (rank partial = a bf16 LEAF,
    m=1, the halved wire). Same function by construction; asserted, not argued.
    """
    x, w = _data()
    plan = ReductionPlan(num_leaves=G, leaf_dtype=torch.bfloat16)
    ref = _scheme_at_tp(x, w, plan, 1, round_leaf=True)
    for tp in (2, 4, 8):
        got = _scheme_at_tp(x, w, plan, tp, round_leaf=True)
        assert torch.equal(_bits(ref), _bits(got)), f"TP={tp} diverged from TP=1 under bf16 leaves"


def test_rank_boundary_rounding_is_world_dependent():
    """The naive bf16 wire -- round each rank PARTIAL -- is REFUSED with a demonstration.

    At TP=8 under G=8 a rank partial IS a leaf, so rounding it is the leaf scheme. At TP=4 a
    rank partial is an INTERNAL node (leaf pair); rounding it evaluates
    bf16(L0+L1) + bf16(L2+L3) + ... where TP=8 evaluates (bf16 L0 + bf16 L1) + ... --
    different functions. This is why the wire may only narrow where m == 1, and why at
    TP < G the partial must stay fp32 (pik.linear wire_dtype rule).
    """
    x, w = _data()
    plan = ReductionPlan(num_leaves=G, leaf_dtype=torch.bfloat16)
    tp4 = _scheme_at_tp(x, w, plan, 4, round_leaf=True, round_rank_partial=True)
    tp8 = _scheme_at_tp(x, w, plan, 8, round_leaf=True, round_rank_partial=True)
    assert not torch.equal(_bits(tp4), _bits(tp8)), (
        "rank-boundary rounding came out TP-invariant on adversarial data -- the impossibility "
        "control is vacuous, tighten the data"
    )


def test_leaf_dtype_is_a_different_function():
    """fp32 leaves vs bf16 leaves differ in bits: the flip is a manifest event, not a knob."""
    x, w = _data()
    plan = ReductionPlan(num_leaves=G, leaf_dtype=torch.bfloat16)
    a = _scheme_at_tp(x, w, plan, 8, round_leaf=False)
    b = _scheme_at_tp(x, w, plan, 8, round_leaf=True)
    assert not torch.equal(_bits(a), _bits(b)), (
        "fp32-leaf and bf16-leaf schemes agreed bitwise on adversarial data -- the quality "
        "comparison below would be measuring nothing"
    )


def test_upcast_location_is_transport_detail():
    """NCCL-path emulation (upcast bf16 leaves to fp32 BEFORE the tree) == in-kernel upcast.

    ``_tree_all_reduce_nccl`` upcasts a bf16 partial before moving bytes (its wire stays fp32);
    the P2P kernel loads bf16 and upcasts in-register. Same exact embedding, same tree, same
    bits -- which is what makes the trainer (NCCL transport, m=2, fp32 internal nodes) and the
    engine (P2P transport, m=1, bf16 wire) the same function under one plan.
    """
    x, w = _data()
    plan = ReductionPlan(num_leaves=G, leaf_dtype=torch.bfloat16)
    leaves = _leaves(x, w, plan, round_leaf=True)
    # in-kernel-upcast emulation: leaves handled as bf16 storage, upcast exactly at load
    as_bf16 = [leaf.to(torch.bfloat16) for leaf in leaves]
    in_kernel = _tree([b.to(torch.float32) for b in as_bf16]).to(torch.bfloat16)
    pre_upcast = _tree(leaves).to(torch.bfloat16)
    assert torch.equal(_bits(in_kernel), _bits(pre_upcast))
