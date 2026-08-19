"""Patch vLLM's linear layers to use pik, making the rollout engine bitwise-identical to the trainer at any TP.

vLLM already stores ``RowParallelLinear.weight`` as ``[N, K_local]``, pik's layout, so no transpose is needed.
Two vLLM behaviours would break the contract and are replaced here: the bias is added once after the reduction
rather than folded into rank 0's GEMM (``(P0 + b) + P1 != (P0 + P1) + b`` in fp32), and the NCCL all-reduce,
whose reduction order varies with message size, is replaced by the fixed tree. Column-parallel layers shard N
and never K, so they need no tree, but they take the pinned-algo cuBLASLt path because ``torch.matmul`` picks
split-K by shape and is not batch-invariant.
"""

from __future__ import annotations

import torch

from ..allreduce import (
    p2p_available,
    sym_partial,
    tree_all_reduce,
    tree_all_reduce_rounded,
)
from ..gemm import ti_gemm, ti_gemm_column_parallel
from ..plan import DEFAULT_PLAN, ReductionPlan

_PLAN: ReductionPlan = DEFAULT_PLAN
_ORIG: dict = {}


def _row_forward(self, input_):
    from vllm.distributed import split_tensor_along_last_dim

    if self.input_is_parallel:
        x = input_
    else:
        x = split_tensor_along_last_dim(input_, num_partitions=self.tp_size)[self.tp_rank].contiguous()

    lead, k_local = x.shape[:-1], x.shape[-1]
    x2 = x.reshape(-1, k_local).contiguous()
    w = self.weight  # [N, K_local] -- already our layout
    tp, rank = self.tp_size, self.tp_rank
    m_leaves = _PLAN.leaves_per_rank(tp)

    reduce_ = self.reduce_results and tp > 1
    wire = torch.bfloat16 if (_PLAN.bf16_leaves and m_leaves == 1) else torch.float32

    dst = None
    if reduce_ and p2p_available():
        dst = sym_partial((x2.shape[0], w.shape[0]), x2.device, None, dtype=wire)

    part = ti_gemm(x2, w, plan=_PLAN, tp_size=tp, tp_rank=rank, k_full=self.input_size, out=dst)

    # Bias AFTER the reduction, exactly once -- never folded into rank 0's GEMM.
    bias = None if self.skip_bias_add else self.bias

    # No bias: fold the activation-dtype rounding into the reduce kernel's store. The root is the same node at
    # every TP size, so rounding it stays TP-independent, and the two roundings are checked per shape first.
    # With a bias this is illegal: the bias must be added to the fp32 root and the sum rounded once, otherwise
    # TP>1 would add in bf16 where a TP=1 trainer adds in fp32.
    if reduce_ and bias is None and not _PLAN.bf16_leaves and input_.dtype == torch.bfloat16:
        full = tree_all_reduce_rounded(part, out_dtype=torch.bfloat16)
    else:
        full = tree_all_reduce(part) if reduce_ else part

        if _PLAN.bf16_leaves:
            full = full.to(torch.bfloat16)  # round the ROOT -- a TP-independent node

        if bias is not None:
            full = full + bias.to(full.dtype)

    out = full.to(input_.dtype).reshape(*lead, -1)
    if not self.return_bias:
        return out
    return out, (self.bias if self.skip_bias_add else None)


def _col_forward(self, input_):
    lead, k = input_.shape[:-1], input_.shape[-1]
    x2 = input_.reshape(-1, k).contiguous()
    out = ti_gemm_column_parallel(x2, self.weight, out_dtype=input_.dtype)

    bias = None if self.skip_bias_add else self.bias
    if bias is not None:
        out = out + bias.to(out.dtype)
    out = out.reshape(*lead, -1)
    if not self.return_bias:
        return out
    return out, (self.bias if self.skip_bias_add else None)


def patch_vllm(plan: ReductionPlan = DEFAULT_PLAN) -> None:
    """Route vLLM's linear layers through pik. Idempotent."""
    global _PLAN
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        RowParallelLinear,
    )

    _PLAN = plan
    for cls, fwd in ((RowParallelLinear, _row_forward), (ColumnParallelLinear, _col_forward)):
        if cls not in _ORIG:
            _ORIG[cls] = cls.forward
        cls.forward = fwd
    # MergedColumnParallelLinear and QKVParallelLinear subclass ColumnParallelLinear and
    # do not override forward, so they are covered.


def unpatch_vllm() -> None:
    for cls, fwd in _ORIG.items():
        cls.forward = fwd
    _ORIG.clear()
