"""The bitwise decomposition table: graph-level repair of sign-sensitive lowerings."""

from __future__ import annotations

import os
from typing import Callable, Sequence


def _rewrite_neg(node, graph) -> None:
    """aten.neg(x) -> aten.mul(x, -1.0): sign-preserving at zero."""
    import torch

    node.target = torch.ops.aten.mul.Tensor
    node.args = (node.args[0], -1.0)


#: Overload-packet base name -> rewrite, which mutates the node in place.
BITWISE_DECOMPS: dict[str, Callable] = {
    "neg": _rewrite_neg,
}


def decomp_names(table: dict | None) -> list[str]:
    return sorted(table.keys()) if table else []


def decomp_enabled() -> bool:
    """Install-side switch; a with/without mismatch resolves to eager via the fingerprint."""
    return os.environ.get("SKYRL_ISOEXEC_AUTOFUSE_DECOMP", "0") == "1"


def active_table() -> dict | None:
    return dict(BITWISE_DECOMPS) if decomp_enabled() else None


def apply_bitwise_decomps(gm, table: dict | None):
    """Rewrite every table-matched node of an aten fx graph in place, returning ``(gm, applied_names)``.

    Each rewrite is exactly equal in real arithmetic (including signed zeros and NaN payloads); what
    it changes is which lowering inductor picks.
    """
    if not table:
        return gm, []
    applied: list[str] = []
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        packet = getattr(node.target, "overloadpacket", None)
        base = getattr(packet, "__name__", None) or str(node.target)
        base = base.split(".", 1)[0]
        rewrite = table.get(base)
        if rewrite is not None:
            rewrite(node, gm.graph)
            applied.append(base)
    if applied:
        gm.graph.lint()
        gm.recompile()
    return gm, sorted(set(applied))


def diff_is_sign_of_zero_only(eager_out, compiled_out) -> bool:
    """True iff every differing element is a zero whose sign bit differs -- the class the table repairs.

    A magnitude difference anywhere is a rounding divergence that no rewrite repairs, so returns False.
    """
    import torch

    e_list = eager_out if isinstance(eager_out, (list, tuple)) else [eager_out]
    c_list = compiled_out if isinstance(compiled_out, (list, tuple)) else [compiled_out]
    any_diff = False
    for e, c in zip(e_list, c_list):
        if not isinstance(e, torch.Tensor) or e.shape != c.shape or e.dtype != c.dtype:
            return False
        if not e.dtype.is_floating_point:
            continue
        width = {2: torch.int16, 4: torch.int32, 8: torch.int64}[e.dtype.itemsize]
        diff = e.contiguous().view(width) != c.contiguous().view(width)
        if not bool(diff.any()):
            continue
        any_diff = True
        both_zero = (e == 0) & (c == 0)  # torch's == treats +-0 as equal, so this selects magnitude-zero pairs
        if bool((diff & ~both_zero).any()):
            return False
    return any_diff


def compile_rewritten(fn: Callable, example_inputs: Sequence, table: dict | None, *, dynamic: bool):
    """Trace ``fn`` to aten, apply the table, and return ``(compiled_callable, applied_names)``.

    The compile target is the rewritten GraphModule, so a dynamo retrace re-traces the repaired graph
    and the table cannot be lost. An empty table compiles the original function directly.
    """
    import torch

    if not table:
        return torch.compile(fn, dynamic=dynamic), []
    from .region_gate import _trace_aten

    gm = _trace_aten(fn, example_inputs)
    gm, applied = apply_bitwise_decomps(gm, table)
    return torch.compile(gm, dynamic=dynamic), applied
