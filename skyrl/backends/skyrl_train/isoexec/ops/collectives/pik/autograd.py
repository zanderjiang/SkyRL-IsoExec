"""Differentiable TP-invariant linears.

Only the forward needs the reduction plan: the row-parallel backward reduces over N (for dX) and over M (for
dW), neither of which is sharded along K, so backward has no TP-invariance requirement and runs stock cuBLAS.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from .linear import column_parallel_linear as _col_fwd
from .linear import row_parallel_linear as _row_fwd
from .plan import DEFAULT_PLAN, ReductionPlan


class _RowParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, bias, plan, tp_size, tp_rank, k_full, group, out_dtype):
        ctx.save_for_backward(x, w)
        ctx.has_bias = bias is not None
        with torch.no_grad():
            y = _row_fwd(
                x, w, bias, plan=plan, tp_size=tp_size, tp_rank=tp_rank, k_full=k_full, group=group, out_dtype=out_dtype
            )
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        dy = dy.contiguous()
        n = dy.shape[-1]
        dy2 = dy.reshape(-1, n)
        x2 = x.reshape(-1, x.shape[-1])
        # dY is already the full (all-reduced) output gradient on every rank.
        dx = (dy2 @ w).reshape(*x.shape)  # [.., K_local]
        dw = dy2.t() @ x2  # [N, K_local]
        db = dy2.sum(0) if ctx.has_bias else None
        return dx, dw, db, None, None, None, None, None, None


class _ColParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, bias, group, out_dtype):
        ctx.save_for_backward(x, w)
        ctx.has_bias = bias is not None
        ctx.group = group
        with torch.no_grad():
            y = _col_fwd(x, w, bias, out_dtype=out_dtype)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        dy = dy.contiguous()
        dy2 = dy.reshape(-1, dy.shape[-1])
        x2 = x.reshape(-1, x.shape[-1])
        dx = (dy2 @ w).reshape(*x.shape)
        # column-parallel shards N, so dX is a partial sum over the N shards
        if dist.is_initialized() and dist.get_world_size(ctx.group) > 1:
            dist.all_reduce(dx, group=ctx.group)
        dw = dy2.t() @ x2
        db = dy2.sum(0) if ctx.has_bias else None
        return dx, dw, db, None, None


def row_parallel_linear(
    x,
    w,
    bias=None,
    *,
    plan: ReductionPlan = DEFAULT_PLAN,
    tp_size=1,
    tp_rank=0,
    k_full=None,
    group=None,
    out_dtype=torch.bfloat16,
):
    return _RowParallel.apply(x, w, bias, plan, tp_size, tp_rank, k_full, group, out_dtype)


def column_parallel_linear(x, w, bias=None, *, group=None, out_dtype=torch.bfloat16):
    return _ColParallel.apply(x, w, bias, group, out_dtype)
