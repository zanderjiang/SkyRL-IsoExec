"""The GDN delta-rule kernel selector: one vocabulary, one default, one parser.

``SKYRL_ISOEXEC_GDN_KERNEL`` is read at two places that must agree -- the DECLARATION site
(``models/qwen3_5.build``, which picks the profile variant the contract hashes) and the READ sites
(``ops/gdn/gdn_ops``, which pick the kernel that executes). They used to parse it differently:
case-sensitive against one literal on one side, ``.lower()`` against three on the other, different
defaults, and an unrecognized value silently falling back on one side while raising on the other.
The consequence was the worst kind: with ``SKYRL_ISOEXEC_GDN_KERNEL=CPR`` both runtimes derived
the same WRONG contract, so the weight-sync handshake MATCHED. Parsing lives here so there is
exactly one answer to what the variable says.
"""

from __future__ import annotations

import os

KERNEL_ENV = "SKYRL_ISOEXEC_GDN_KERNEL"
# Trainer-only ablation override (ops/gdn/gdn_ops.gdn_core). Documented there as "not a IsoExec
# configuration": the contract pins ONE gdn.core kernel for all four sites and cannot express a
# trainer/engine split, so a contract build refuses when this asks for one.
TRAINER_KERNEL_ENV = "SKYRL_ISOEXEC_GDN_TRAINER_KERNEL"

# "cpr" is the chunkwise-parallel-recurrent kernel: parallel within a chunk, recurrent across
# chunk boundaries.
KERNELS = ("chunk", "cpr", "recurrent")
# The DECLARATION site owns the default: models/qwen3_5 declares a recurrent variant and a cpr one,
# and no model declares a chunk composition, so "chunk" by default meant the contract named a
# function the process did not run. Unset therefore resolves to the variant that is actually
# declarable.
DEFAULT_KERNEL = "recurrent"

# Retired spellings, refused by name so an old launch script fails loudly instead of falling back.
_RENAMED = {"chunk_synced": "cpr"}


def parse_gdn_kernel(value, env: str = KERNEL_ENV) -> str:
    """Normalize one env value to a kernel name. Unknown refuses; empty/absent gives the default."""
    name = (value or "").strip().lower()
    if not name:
        return DEFAULT_KERNEL
    if name not in KERNELS:
        successor = f" {name!r} was renamed to {_RENAMED[name]!r}." if name in _RENAMED else ""
        raise ValueError(
            f"{env}={value!r} names no delta-rule kernel; the vocabulary is {list(KERNELS)}.{successor} "
            f"Refusing rather than falling back: a typo that falls back is the same typo on both "
            f"runtimes, so both derive the same wrong composition and the handshake agrees."
        )
    return name


def gdn_kernel_mode() -> str:
    """The delta-rule kernel this process runs, from the environment. Read at call time."""
    return parse_gdn_kernel(os.environ.get(KERNEL_ENV), KERNEL_ENV)


def gdn_trainer_kernel_override() -> str | None:
    """The trainer-only kernel override, or None when unset (the supported configuration)."""
    raw = os.environ.get(TRAINER_KERNEL_ENV, "")
    return parse_gdn_kernel(raw, TRAINER_KERNEL_ENV) if raw.strip() else None
