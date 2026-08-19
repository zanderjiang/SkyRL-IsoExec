"""Per-model composition manifests.

A model file declares structural facts (``profile.ModelProfile``); ``policy.derive_selections`` turns
those into the ``{(op, site) -> impl}`` manifest, and a model that departs from policy declares an
EXCEPTION list. No kernels live here -- only selections.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def build_for(model_path: str, registry, *, arch=None):
    """Dispatch a HF model path / id to its composition manifest builder (unfrozen).

    Both runtimes build the same complete manifest from this one function, so their hashes match by
    construction. Unknown models raise rather than silently running unmanaged.
    """
    from .resolve import resolve_model_module

    mod, how = resolve_model_module(model_path)
    logger.info("[isoexec-models] %r -> %s (matched by %s)", model_path, mod.PROFILE.model, how)
    return mod.build(registry, arch=arch)


def profile_for(model_path: str):
    """The ``ModelProfile`` for a model path/id, without building a manifest."""
    from .resolve import resolve_profile

    return resolve_profile(model_path)


__all__ = ["build_for", "profile_for"]
