"""Fused fixed-order leaf-tree COMBINE for the trainer's pik-fc2 (``SKYRL_ISOEXEC_MOE_FUSED_LEAFCOMBINE``).

``moe_batched_experts._leaftree_fc2`` promotes ``G`` leaves to fp32 and folds them with a balanced binary
tree, costing ``G`` promotions plus ``G-1`` full-sweep adds through ``G-1`` intermediate fp32 buffers. This
does the whole fold in one Triton pass that reads the leaves and writes the fp32 root.

The reduction PLAN is untouched: promotion to fp32 is exact, the streaming fold reproduces ``_tree_sum``'s
balanced tree element for element, and each program owns a disjoint set of elements, so the result is
bit-identical to the buffer tree. A fail-closed self-check at first use asserts that and disables the
provider permanently on any mismatch or raise, so the flag can only change speed, never numbers. Wired into
``_leaftree_fc2``; the cross-rank ETP tree and the engine's in-GEMM leaf tree are separate and untouched.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception:  # pragma: no cover - triton is present on every GPU stack we ship
    HAVE_TRITON = False

_ENV_GATE = "SKYRL_ISOEXEC_MOE_FUSED_LEAFCOMBINE"

# Elements per program. Bitwise-neutral -- each element's tree is independent -- so it is free to tune;
# 4096 saturates the load pipe without spilling the four live fp32 tiles of a G=8 fold out of registers.
_BLOCK = int(os.environ.get("SKYRL_ISOEXEC_MOE_LEAFCOMBINE_BLOCK", "4096"))

# Fail-closed provider state: first use runs the bitwise self-check; failure disables permanently.
_STATE = {"checked": False, "ok": False}

_MAX_LEAVES = 8


def fused_leafcombine_enabled() -> bool:
    """Default OFF. Read per call so an in-process A/B can flip it."""
    return os.environ.get(_ENV_GATE, "0") == "1"


if HAVE_TRITON:

    @triton.jit
    def leaftree_combine_kernel(
        l0_ptr,
        l1_ptr,
        l2_ptr,
        l3_ptr,
        l4_ptr,
        l5_ptr,
        l6_ptr,
        l7_ptr,
        out_ptr,
        n_elem,
        N_LEAVES: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """One pass over the flattened leaves; the balanced fold lives in registers.

        The leaves are element-aligned contiguous buffers of identical shape, so a program owning a
        contiguous element range owns the SAME range in every leaf and in the output. ``s0/s1/s2`` hold
        the pending partial of tree levels 0/1/2; every branch below is on a Python-level ``int``
        (``leaf`` comes from ``tl.static_range``), so tracing prunes them and the body is straight-line.
        """
        pid = tl.program_id(0).to(tl.int64)
        offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        mask = offs < n_elem

        for leaf in tl.static_range(N_LEAVES):
            # constexpr select of this leaf's base pointer (pruned at trace time).
            if leaf == 0:
                p = l0_ptr
            elif leaf == 1:
                p = l1_ptr
            elif leaf == 2:
                p = l2_ptr
            elif leaf == 3:
                p = l3_ptr
            elif leaf == 4:
                p = l4_ptr
            elif leaf == 5:
                p = l5_ptr
            elif leaf == 6:
                p = l6_ptr
            else:
                p = l7_ptr
            v = tl.load(p + offs, mask=mask, other=0.0).to(tl.float32)  # exact for bf16 and fp32

            # streaming balanced fold: merge by the trailing-one bits of the leaf index, earlier
            # partial on the LEFT -- character-for-character _tree_sum's tree.
            if (leaf & 1) == 1:
                v = s0 + v  # noqa: F821  (assigned on the previous unrolled iteration)
            if (leaf & 3) == 3:
                v = s1 + v  # noqa: F821
            if (leaf & 7) == 7:
                v = s2 + v  # noqa: F821
            if leaf == N_LEAVES - 1:
                result = v
            elif (leaf & 1) == 0:
                s0 = v  # noqa: F841  (consumed on a later unrolled iteration)
            elif (leaf & 3) == 1:
                s1 = v  # noqa: F841
            else:
                s2 = v  # noqa: F841

        tl.store(out_ptr + offs, result, mask=mask)


def _combine_forward(leaves) -> torch.Tensor:
    """The raw (non-differentiable) fused fold. ``leaves`` is a list of identically shaped,
    contiguous tensors; returns the fp32 balanced-tree sum of their fp32 promotions."""
    n = len(leaves)
    ref = leaves[0]
    out = torch.empty(ref.shape, device=ref.device, dtype=torch.float32)
    n_elem = ref.numel()
    if n_elem == 0:
        return out
    pad = list(leaves) + [ref] * (_MAX_LEAVES - n)  # unused pointers; the kernel prunes them
    grid = (triton.cdiv(n_elem, _BLOCK),)
    leaftree_combine_kernel[grid](
        *pad,
        out,
        n_elem,
        N_LEAVES=n,
        BLOCK=_BLOCK,
        num_warps=4,
    )
    return out


class _LeafTreeCombine(torch.autograd.Function):
    """Differentiable wrapper: forward is the fused fold, backward is the sum's VJP (broadcast the
    incoming gradient to every leaf, rounded back to the leaf dtype)."""

    @staticmethod
    def forward(ctx, *leaves):
        ctx.leaf_dtype = leaves[0].dtype
        ctx.n_leaves = len(leaves)
        return _combine_forward(leaves)

    @staticmethod
    def backward(ctx, g):
        gl = g.to(ctx.leaf_dtype)
        return tuple(gl for _ in range(ctx.n_leaves))


def fused_leaf_tree_combine(leaves) -> torch.Tensor:
    """Differentiable fp32 balanced-tree sum of ``leaves`` (lower index on the left)."""
    return _LeafTreeCombine.apply(*leaves)


def leaves_are_supported(leaves) -> bool:
    """Shape/layout preconditions the kernel assumes. Anything else -> the buffer tree."""
    n = len(leaves)
    if n < 2 or n > _MAX_LEAVES or (n & (n - 1)) != 0:
        return False
    ref = leaves[0]
    if ref.dtype not in (torch.bfloat16, torch.float16, torch.float32) or not ref.is_cuda:
        return False
    return all(t.is_contiguous() and t.dtype == ref.dtype and t.shape == ref.shape for t in leaves)


def _tree_sum_ref(nodes):
    """The production fold, copied from ``moe_batched_experts._tree_sum`` so the check compares against
    the expression it must replace rather than a paraphrase of it."""
    while len(nodes) > 1:
        nodes = [nodes[i] + nodes[i + 1] if i + 1 < len(nodes) else nodes[i] for i in range(0, len(nodes), 2)]
    return nodes[0]


def _bitcmp(x: torch.Tensor, y: torch.Tensor) -> int:
    """Bit-pattern mismatch count (torch.equal is blind to signed zero)."""
    return int((x.view(torch.int32) != y.view(torch.int32)).sum().item())


@torch.no_grad()
def _self_check(device) -> bool:
    """Bitwise: fused fold == ``[l.float() for l in leaves]`` folded by ``_tree_sum``.

    Cases cover the leaf counts a rank can own, the production rank-3 shape as well as ragged element
    counts that straddle the block boundary, both leaf dtypes, and magnitudes spread across exponents so
    that a mis-ordered fold actually shows up in the low bits.
    """
    torch.manual_seed(0)
    cases = [
        (8, (5, 128, 112), torch.bfloat16),  # production rank-3 shape, ragged h
        (8, (1, 7, 13), torch.bfloat16),  # tiny + odd: partial block, heavy masking
        (8, (4097,), torch.bfloat16),  # straddles the 4096-element block boundary
        (4, (3, 128, 128), torch.bfloat16),  # ETP=2 rank
        (2, (2, 64, 96), torch.bfloat16),  # ETP=4 rank
        (8, (5, 128, 112), torch.float32),  # fp32 leaf dtype
    ]
    for n, shape, dtype in cases:
        # deliberately spread exponents across leaves so the summation ORDER is observable.
        leaves = [
            (torch.randn(shape, device=device, dtype=torch.float32) * (10.0 ** (i - 4))).to(dtype) for i in range(n)
        ]
        ref = _tree_sum_ref([leaf.float() for leaf in leaves])
        got = _combine_forward(leaves)
        bad = _bitcmp(got, ref)
        if bad:
            print(
                f"[ISOEXEC-MOE-LEAFCOMBINE] SELF-CHECK FAIL (n_leaves={n}, shape={tuple(shape)}, "
                f"{dtype}): {bad} mismatched bit patterns vs the buffer tree. Disabling the fused "
                "leaf combine (buffer-tree fallthrough).",
                flush=True,
            )
            return False
    return True


def fused_leafcombine_ready(device) -> bool:
    """True iff the flag is on AND the one-time bitwise self-check passed on this stack.

    Fail-closed: a failed or raising self-check disables the provider permanently and the caller keeps the
    buffer tree, so the flag can never change the bits of a run, only its speed.
    """
    if not fused_leafcombine_enabled() or not HAVE_TRITON:
        return False
    if not _STATE["checked"]:
        _STATE["checked"] = True
        try:
            _STATE["ok"] = _self_check(device)
        except Exception as e:  # noqa: BLE001
            print(
                f"[ISOEXEC-MOE-LEAFCOMBINE] self-check raised ({type(e).__name__}: {e}) -- " "buffer-tree fallthrough.",
                flush=True,
            )
            _STATE["ok"] = False
        if _STATE["ok"]:
            print(
                "[ISOEXEC-MOE-LEAFCOMBINE] fused pik-fc2 leaf combine ENABLED (self-check bit-exact "
                "vs the buffer tree): the G fp32 leaf promotions and the G-1 tree adds are now one "
                "in-register pass, same leaves, same tree, same rounding.",
                flush=True,
            )
    return _STATE["ok"]
