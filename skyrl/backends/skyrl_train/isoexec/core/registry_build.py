"""Assemble the process's op Registry from the per-family ``ops/<family>/_register.py`` modules.

``build_registry`` is the single entry the model manifests and the fingerprint/handshake path
consume. The family ``register`` functions reference only registry dataclasses and string ids, never
live kernels, so building a registry stays import-light and works on CPU.
"""

from __future__ import annotations

import importlib

from .registry import Registry

# Families in a fixed order, for deterministic registry construction. Listing a family here makes
# its ops INSTALLED keys that every model manifest must then cover, so "quant" is deliberately
# absent: no current target model executes block-FP8, and naming an impl no fingerprint could
# confirm is worse than leaving the family out.
_FAMILIES = ("mm", "attention", "rope", "norms", "logprobs", "gdn", "moe", "collectives")

# The isoexec package root, derived from this module's name so family imports are absolute: a
# relative import_module here misresolves the level, since the anchor is a module, not a package.
_ISOEXEC_ROOT = __name__.rsplit(".core.", 1)[0]


def build_registry(*, strict: bool = False) -> Registry:
    """Build and validate the Registry; ``strict`` raises on a missing family instead of skipping it."""
    reg = Registry()
    missing = []
    for fam in _FAMILIES:
        try:
            mod = importlib.import_module(f"{_ISOEXEC_ROOT}.ops.{fam}._register")
        except ModuleNotFoundError:
            missing.append(fam)
            continue
        mod.register(reg)
    if strict and missing:
        raise RegistryError_missing(missing)
    _mark_canonical_routes(reg)
    reg.assert_subsumption_closed()
    return reg


# Framework-default reference impls (eager torch / aten) get contract route "canonical" rather than
# the "protected" default that every other provider keeps.
_CANONICAL_IMPLS = (
    ("rope.rope", "eager"),
    ("norms.rms", "eager_zero_centered"),
    ("norms.rms", "eager_torch_rms"),
    ("norms.gated_out", "eager"),
    ("logprobs.log_softmax", "aten_reference"),
    ("gdn.gating", "eager"),
)


def _mark_canonical_routes(reg: Registry) -> None:
    import dataclasses

    for op, impl_id in _CANONICAL_IMPLS:
        if reg.has_op(op) and impl_id in reg.get_op(op).impls:
            spec = reg.get_op(op).impls[impl_id]
            reg.get_op(op).impls[impl_id] = dataclasses.replace(spec, route="canonical")


def registered_families() -> list:
    """Which families currently have a _register module."""
    present = []
    for fam in _FAMILIES:
        try:
            importlib.import_module(f"{_ISOEXEC_ROOT}.ops.{fam}._register")
            present.append(fam)
        except ModuleNotFoundError:
            pass
    return present


def RegistryError_missing(missing):
    from .registry import RegistryError

    return RegistryError(f"strict build: op families missing a _register module: {missing}")
