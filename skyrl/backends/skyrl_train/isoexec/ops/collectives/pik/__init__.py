"""pik -- bitwise-identical LLM math across tensor-parallel sizes.

Determinism is a property of the reduction plan, not of the kernel: fix how K is cut into leaves and how
those leaves are combined, and every other knob (block sizes, warps, stages, grid order, split-K along leaf
boundaries, the all-reduce algorithm, the TP size itself) stays free.
"""

from .allreduce import tree_all_reduce
from .arch import ArchProfile, arch_tag
from .arch import current as current_arch
from .gemm import ti_gemm, ti_gemm_column_parallel
from .linear import column_parallel_linear, row_parallel_linear
from .plan import DEFAULT_PLAN, ReductionPlan, combine_order

__all__ = [
    "arch_tag",
    "current_arch",
    "ArchProfile",
    "ReductionPlan",
    "DEFAULT_PLAN",
    "combine_order",
    "ti_gemm",
    "ti_gemm_column_parallel",
    "tree_all_reduce",
    "row_parallel_linear",
    "column_parallel_linear",
]
