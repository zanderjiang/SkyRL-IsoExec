"""The reduction plan: pik's single source of determinism.

K is split into G equal contiguous leaves, each leaf is summed by a plain sequential fp32 MMA accumulation,
and the leaves are combined by a fixed balanced binary tree in fp32. Any tensor-parallel size C dividing G maps
rank i onto the contiguous leaf range [i*G/C, (i+1)*G/C), which is exactly a subtree of that combine tree, so
every C evaluates the same expression. Block sizes, warps, stages, grid order, split-K along leaf boundaries,
and the all-reduce algorithm are therefore free; G, the fp32 combine dtype, the GPU architecture family, and
the weight layout are pinned and must match between engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class ReductionPlan:
    """A fixed K-reduction contract shared by every engine in the system.

    Args:
        num_leaves: G, the number of fixed K-leaves. Must be a power of two and >= the largest tensor-parallel
            size ever run. This is a contract constant: trainer and rollout engine must agree on it the same
            way they agree on the weights, and changing G changes the numerics. Set it to the largest TP size
            and no higher -- extra leaves are narrower and less efficient and buy nothing.

        leaf_dtype: the precision a leaf partial is rounded to before it enters the combine tree. fp32
            (default) is exact; bf16 halves the all-reduce payload and the trainer workspace.

    bf16 leaves stay TP-invariant because the leaf boundary is the only rounding point that does not depend on
    TP: leaves are fixed by the contract while rank boundaries move with C. Internal tree nodes must stay fp32.
    At TP=G a rank's partial is itself a leaf and goes on the wire in bf16; at TP<G it is an internal node.
    """

    num_leaves: int = 8
    leaf_dtype: "torch.dtype" = None  # type: ignore[assignment]  # None -> fp32

    def __post_init__(self) -> None:
        g = self.num_leaves
        if g < 1 or (g & (g - 1)) != 0:
            raise ValueError(f"num_leaves must be a power of two, got {g}")
        if self.leaf_dtype is None:
            import torch

            object.__setattr__(self, "leaf_dtype", torch.float32)

    @property
    def bf16_leaves(self) -> bool:
        import torch

        return self.leaf_dtype == torch.bfloat16

    def contract(self, device=None) -> str:
        """The full contract string; both engines must print the same thing.

        Architecture is part of it: a leaf summed by Hopper's wgmma (k-step 16) is not the same fp32 number as
        one summed by Blackwell's tcgen05 (k-step 32), so trainer and rollout must run on the same arch.
        """
        from .arch import arch_tag

        return f"G={self.num_leaves} leaf={self.leaf_dtype} arch={arch_tag(device)}"

    # -- validation ---------------------------------------------------------

    def validate(self, k_full: int, tp_size: int) -> None:
        """Check that a (K, TP) pair is expressible under this plan."""
        g = self.num_leaves
        if tp_size > g:
            raise ValueError(
                f"tp_size={tp_size} exceeds num_leaves={g}: every rank needs at "
                f"least one leaf. Raise ReductionPlan(num_leaves=...) to >= {tp_size} "
                f"-- but note that changes the numerics, so all engines must agree."
            )
        if g % tp_size != 0:
            raise ValueError(
                f"tp_size={tp_size} must divide num_leaves={g} so that each rank's "
                f"K-shard is exactly a subtree of the combine tree."
            )
        if k_full % g != 0:
            raise ValueError(f"K={k_full} must be divisible by num_leaves={g} for equal leaves.")

    # -- geometry -----------------------------------------------------------

    def leaf_k(self, k_full: int) -> int:
        """Size of one leaf along K."""
        return k_full // self.num_leaves

    def leaves_per_rank(self, tp_size: int) -> int:
        """m: how many leaves a single rank owns. m == 1 => plain GEMM, zero tax."""
        return self.num_leaves // tp_size

    def rank_leaf_range(self, tp_rank: int, tp_size: int) -> tuple[int, int]:
        """The contiguous leaf range owned by a rank -- a subtree of the tree."""
        m = self.leaves_per_rank(tp_size)
        return tp_rank * m, (tp_rank + 1) * m


DEFAULT_PLAN = ReductionPlan(num_leaves=8)


def combine_order(num: int) -> list[tuple[int, int, int]]:
    """The fixed balanced binary combine tree over `num` values, as a flat schedule.

    Returns (dst, lhs, rhs) triples over slot indices, applied in order, where `dst` reuses `lhs`'s slot. The
    left operand always carries the lower leaf indices, matching the binary-counter carry order used inside the
    fused kernel, so the reduce kernels and the all-reduce realize the same tree as the inline GEMM.
    """
    if num & (num - 1) != 0:
        raise ValueError(f"combine width must be a power of two, got {num}")
    sched: list[tuple[int, int, int]] = []
    stride = 1
    while stride < num:
        for i in range(0, num, 2 * stride):
            sched.append((i, i, i + stride))
        stride *= 2
    return sched
