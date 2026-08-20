"""Count ATen dispatches, and attribute them to the source line that issued them.

WHY THIS EXISTS
---------------
The 2026-08-14 host-dispatch attribution found the trainer HOST-BOUND: in ``rank0_scoring_0`` the
GPU got 61% faster between two identical windows and the wall did not move. The host census that
followed says where the host time goes -- **70,941 ATen ops per microbatch produce 6,867
``cudaLaunchKernel`` calls, so 90.3% of our ATen ops issue no kernel at all** -- but a chrome trace
carries no python stacks, so it can say *which op* and never *which line*.

This module closes that gap on the CPU, before any GPU minute is spent:

    from ...tools.op_census import count_ops

    with count_ops() as c:
        f(x)
    c.by_op()      # Counter: aten op -> calls
    c.by_site()    # Counter: "file.py:LINE" -> calls   (innermost frame outside torch/)
    c.report()     # both, ranked

The counts are exact for the traced region and they are the unit the campaign is priced in: a lever
that removes host work has to remove *dispatches*, and this is what measures whether it did. Use it
in a CPU test to turn "this refactor removes ops" from a claim into an assertion:

    before = count_ops_of(lambda: old(x)).total()
    after = count_ops_of(lambda: new(x)).total()
    assert after < before

WHAT IT DOES NOT DO
-------------------
It observes; it changes nothing. ``TorchDispatchMode`` sees each op once at the dispatcher and
re-issues it unchanged via ``func(*args, **kwargs)``, so the kernels that run, and their arguments,
are exactly the ones that would have run without it -- the instrument cannot move a bit. It is
*not* free (a python frame per op), so it belongs in tests and probes, never in a shipped path.

Views and metadata ops are counted the same as compute ops on purpose: that is the whole point.
``aten::as_strided`` is 50.5% of the scoring microbatch's ATen ops and 1.86% of its host time,
while ``aten::split_with_sizes`` is 0.3% of the ops and 3.79% of the host time -- so a count alone
never justifies a change. Pair it with the per-op host cost from the trace census.
"""

from __future__ import annotations

import collections
import os
import traceback

import torch
from torch.utils._python_dispatch import TorchDispatchMode

_TORCH_PREFIX = os.path.dirname(os.path.dirname(torch.__file__)) + os.sep


class OpCensus(TorchDispatchMode):
    """Counts every ATen dispatch inside the ``with`` block, by op and by issuing source line."""

    def __init__(self, site_depth: int = 1, keep: str | None = None):
        super().__init__()
        self.ops: collections.Counter = collections.Counter()
        self.sites: collections.Counter = collections.Counter()
        self.op_sites: collections.Counter = collections.Counter()
        # elements WRITTEN per op name -- the device-traffic half of the story. An op count alone
        # cannot tell a full-pool memset from a one-row one; this can.
        self.elems: collections.Counter = collections.Counter()
        self._site_depth = int(site_depth)
        self._keep = keep

    # -- the observation itself; re-issues the op unchanged --------------------------------------
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        self.ops[name] += 1
        site = self._site()
        self.sites[site] += 1
        self.op_sites[(name, site)] += 1
        out = func(*args, **(kwargs or {}))
        try:
            outs = out if isinstance(out, (tuple, list)) else (out,)
            self.elems[name] += sum(int(t.numel()) for t in outs if isinstance(t, torch.Tensor))
        except Exception:  # noqa: BLE001 -- accounting must never change behaviour
            pass
        return out

    def _site(self) -> str:
        """Innermost frame that is not inside torch itself and not this file."""
        here = __file__
        out = []
        for fr in reversed(traceback.extract_stack()[:-2]):
            fn = fr.filename
            if fn == here or fn.startswith(_TORCH_PREFIX) or "/torch/" in fn:
                continue
            if self._keep is not None and self._keep not in fn:
                continue
            out.append(f"{os.path.basename(fn)}:{fr.lineno}")
            if len(out) >= self._site_depth:
                break
        return " < ".join(out) if out else "(unknown)"

    # -- readout ---------------------------------------------------------------------------------
    def total(self) -> int:
        return int(sum(self.ops.values()))

    def by_op(self) -> collections.Counter:
        return collections.Counter(self.ops)

    def by_site(self) -> collections.Counter:
        return collections.Counter(self.sites)

    def report(self, top: int = 20) -> str:
        lines = [f"[op-census] {self.total()} ATen dispatches, {len(self.ops)} distinct ops"]
        lines.append("  by op:")
        lines += [f"    {n:7d}  {k}" for k, n in self.ops.most_common(top)]
        lines.append("  by site:")
        lines += [f"    {n:7d}  {k}" for k, n in self.sites.most_common(top)]
        return "\n".join(lines)


def count_ops(site_depth: int = 1, keep: str | None = None) -> OpCensus:
    """``with count_ops() as c:`` -- see the module docstring."""
    return OpCensus(site_depth=site_depth, keep=keep)


def count_ops_of(fn, *args, **kwargs) -> OpCensus:
    """Run ``fn(*args, **kwargs)`` under the census and return it. The result is discarded."""
    c = OpCensus()
    with c:
        fn(*args, **kwargs)
    return c
