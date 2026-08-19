"""Drop-in linear layers that produce bit-identical output in the trainer (TP=1) and the rollout engine (TP>1).

Only row-parallel layers (o_proj, down_proj) need the reduction plan, since they are the only ones whose
K-reduction is split across ranks. Column-parallel layers shard N and never K, so any fixed non-split-K order
is already TP-invariant -- they only have to avoid cuBLAS, which may pick a split-K kernel depending on shape.
"""

from __future__ import annotations

import torch

from .allreduce import (
    p2p_available,
    sym_partial,
    tree_all_reduce,
    tree_all_reduce_rounded,
    tree_reduce_scatter,
)
from .gemm import ti_gemm, ti_gemm_column_parallel
from .plan import DEFAULT_PLAN, ReductionPlan


def row_parallel_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    plan: ReductionPlan = DEFAULT_PLAN,
    tp_size: int = 1,
    tp_rank: int = 0,
    k_full: int | None = None,
    group=None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """x: [..., K_local]  w: [N, K_local]  ->  [..., N]

    Internal tree nodes are always fp32 and never rounded, since rounding one would make the result depend on
    where the rank boundary fell. Leaves are fixed by the contract, so under a bf16-leaf plan a rank whose
    partial *is* a leaf (m == 1, i.e. TP == G) may put it on the wire in bf16 and stay bitwise TP-invariant.
    """
    lead, k_local = x.shape[:-1], x.shape[-1]
    x2 = x.reshape(-1, k_local)
    m_leaves = plan.leaves_per_rank(tp_size)

    # fp32 on the wire unless this rank's partial is itself a leaf.
    wire_dtype = torch.bfloat16 if (plan.bf16_leaves and m_leaves == 1) else torch.float32

    # Stage the GEMM output in peer-visible symmetric memory so the all-reduce reads it over NVLink, no copy.
    dst = None
    if tp_size > 1 and p2p_available(group):
        dst = sym_partial((x2.shape[0], w.shape[0]), x2.device, group, dtype=wire_dtype)

    part = ti_gemm(x2, w, plan=plan, tp_size=tp_size, tp_rank=tp_rank, k_full=k_full, out=dst)

    # No bias + fp32 leaves + bf16 out: fold the root rounding into the reduce kernel's store. Same bits,
    # one launch fewer. With a bias it is illegal -- the bias joins the fp32 root and the sum is rounded once.
    if tp_size > 1 and bias is None and not plan.bf16_leaves and out_dtype == torch.bfloat16:
        return tree_all_reduce_rounded(part, group=group, out_dtype=torch.bfloat16).reshape(*lead, -1)

    full = tree_all_reduce(part, group=group) if tp_size > 1 else part

    # Round the ROOT in every TP configuration, before the bias: the root is the same node however K was
    # sharded, so rounding it is TP-independent -- but doing it in only some TP configs is not.
    if plan.bf16_leaves:
        full = full.to(torch.bfloat16)

    if bias is not None:
        full = full + bias.to(full.dtype)  # after the reduction, and exactly once
    return full.to(out_dtype).reshape(*lead, -1)


# Sequence-parallel variant: same leaf-tree GEMM, combined with tree_reduce_scatter so each rank keeps its
# sequence slice of the tree root. Megatron layout: x is [s, b, K_local], scattered along s in rank order.
def row_parallel_linear_rs(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    plan: ReductionPlan = DEFAULT_PLAN,
    tp_size: int = 1,
    tp_rank: int = 0,
    k_full: int | None = None,
    group=None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """x: [S, ..., K_local]  w: [N, K_local]  ->  [S/tp, ..., N] (this rank's sequence slice).

    Equals ``row_parallel_linear(x, w, ...)`` sliced to rows [tp_rank*S/tp, (tp_rank+1)*S/tp): the same
    expression tree minus a transport-only all-gather, and rounding is elementwise so round-then-slice equals
    slice-then-round. There is deliberately no bias parameter -- Megatron adds the bias on the scattered slice
    outside the reduce, and the caller must mirror that control flow.
    """
    lead, k_local = x.shape[:-1], x.shape[-1]
    assert len(lead) >= 1 and lead[0] % tp_size == 0, (
        f"row_parallel_linear_rs: sequence dim {lead[0] if lead else None} must divide tp_size "
        f"{tp_size} (x shape {tuple(x.shape)})"
    )
    x2 = x.reshape(-1, k_local)

    # ti_gemm already emits the correct wire dtype. No symmetric-memory staging: tree_reduce_scatter is
    # NCCL-transport by design.
    part = ti_gemm(x2, w, plan=plan, tp_size=tp_size, tp_rank=tp_rank, k_full=k_full)

    if tp_size > 1:
        sliced = tree_reduce_scatter(part.view(lead[0], -1), group=group)
    else:
        sliced = part.view(lead[0], -1)

    # Round the ROOT before the out_dtype cast, same rule as row_parallel_linear; elementwise, so
    # slice-invariant.
    if plan.bf16_leaves:
        sliced = sliced.to(torch.bfloat16)

    out_lead = (lead[0] // tp_size, *lead[1:])
    return sliced.to(out_dtype).reshape(*out_lead, -1)


def column_parallel_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """x: [..., K]  w: [N_local, K]  ->  [..., N_local]. No reduction is sharded here."""
    lead, k = x.shape[:-1], x.shape[-1]
    out = ti_gemm_column_parallel(x.reshape(-1, k), w, out_dtype=out_dtype)
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out.to(out_dtype).reshape(*lead, -1)
