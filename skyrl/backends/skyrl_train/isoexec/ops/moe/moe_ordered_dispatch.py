"""Atomics-free backward for Megatron's MoE token permutation gather.

Forward stays Megatron's unfused ``permute`` verbatim; only the final ``index_select`` gather's autograd
edge changes. One Triton program owns each token/hidden block, accumulates its fixed top-k routed rows in
fp32 and stores once, so there are no atomics and no fp32 output buffer. Backward-only: forward numerics
are untouched.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False


def build_route_rows(sorted_indices: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """``[token, slot] -> routed_row`` in fixed ascending expert-major row order."""
    n_routes = int(sorted_indices.numel())
    if n_tokens < 0 or (n_tokens == 0 and n_routes) or (n_tokens > 0 and n_routes % n_tokens):
        raise RuntimeError(f"ordered dispatch requires exact top-k: routes={n_routes}, tokens={n_tokens}")
    if n_tokens == 0:
        return torch.empty((0, 0), dtype=torch.int64, device=sorted_indices.device)
    topk = n_routes // n_tokens
    rows = torch.argsort(sorted_indices, stable=True).view(n_tokens, topk)
    expected = torch.arange(n_tokens, device=sorted_indices.device, dtype=sorted_indices.dtype).view(-1, 1)
    torch._assert_async(
        torch.eq(sorted_indices[rows], expected).all(),
        "ordered dispatch requires every token to own exactly top-k routed rows",
    )
    return rows


if HAVE_TRITON:

    @triton.jit
    def _ordered_dispatch_backward_kernel(
        GRAD_ROUTES,
        ROUTE_ROWS,
        GRAD_TOKENS,
        hidden,
        TOPK: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        token = tl.program_id(0).to(tl.int64)
        offs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = offs < hidden
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for slot in tl.static_range(TOPK):
            route = tl.load(ROUTE_ROWS + token * TOPK + slot).to(tl.int64)
            acc += tl.load(GRAD_ROUTES + route * hidden + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(GRAD_TOKENS + token * hidden + offs, acc.to(GRAD_TOKENS.dtype.element_ty), mask=mask)


def ordered_dispatch_reference(
    grad_routes: torch.Tensor,
    sorted_indices: torch.Tensor,
    n_tokens: int,
    *,
    accumulation_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Fixed-order oracle used on CPU and by tests."""
    rows = build_route_rows(sorted_indices, n_tokens)
    out = torch.zeros(n_tokens, grad_routes.shape[1], dtype=accumulation_dtype, device=grad_routes.device)
    for slot in range(rows.shape[1]):
        out += grad_routes.index_select(0, rows[:, slot]).to(accumulation_dtype)
    return out.to(grad_routes.dtype)


def ordered_dispatch_backward(grad_routes: torch.Tensor, sorted_indices: torch.Tensor, n_tokens: int) -> torch.Tensor:
    rows = build_route_rows(sorted_indices, n_tokens)
    if not grad_routes.is_cuda:
        out = ordered_dispatch_reference(grad_routes, sorted_indices, n_tokens)
    else:
        if not HAVE_TRITON:
            raise RuntimeError("ordered MoE dispatch backward requires Triton for CUDA tensors")
        hidden = grad_routes.shape[1]
        block_h = min(1024, triton.next_power_of_2(hidden))
        out = torch.empty((n_tokens, hidden), dtype=grad_routes.dtype, device=grad_routes.device)
        _ordered_dispatch_backward_kernel[(n_tokens, triton.cdiv(hidden, block_h))](
            grad_routes.contiguous(),
            rows,
            out,
            hidden,
            TOPK=rows.shape[1],
            BLOCK_H=block_h,
            num_warps=4,
        )
    return out
