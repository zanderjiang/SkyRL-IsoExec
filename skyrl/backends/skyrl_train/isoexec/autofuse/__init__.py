"""Bitwise auto-fusion for the pointwise/copy op class, driven by one shared ledger.

Only regions whose every op is a copy or a single correctly-rounded IEEE-754 primitive are
eligible; anything folding a float reduction is refused, since the compiler's association order
is not ATen's. Fusion verdicts are produced offline by the gate and only consumed at install,
so the trainer and the engine cannot disagree; every missing, stale, or refused entry falls back
to eager.
"""

from __future__ import annotations
