"""Resolve a model path/id to its ``ModelProfile`` -- keyed on ARCHITECTURE, not on a path substring.

Resolution order, most authoritative first: a local snapshot's ``config.json`` ``architectures[0]``;
``AutoConfig`` from the HF cache (offline unless ``SKYRL_ISOEXEC_ARCH_FROM_HUB=1``); then name
patterns, which are a labelled fallback for bare HF ids with nothing on disk and are logged as a guess.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# The model modules, in dispatch order. Each exposes ``PROFILE`` (a ModelProfile) and ``build``.
# Order matters ONLY for the name-pattern fallback; architecture matching is exact and order-free.
_MODEL_MODULES = ("qwen3_5",)

_PKG = __name__.rsplit(".", 1)[0]


class ResolutionError(ValueError):
    """Raised when no registered profile claims a model, or when two claim the same architecture."""


def _load_modules() -> list:
    mods = []
    for name in _MODEL_MODULES:
        mods.append(importlib.import_module(f"{_PKG}.{name}"))
    return mods


def raw_config_json(model_path: str) -> Optional[dict]:
    """The checkpoint's raw ``config.json`` as a dict, flattened over ``text_config`` so a
    multimodal wrapper's LM fields are visible to a discriminator. ``None`` when unreadable."""
    cfg_path = os.path.join(model_path or "", "config.json")
    if not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except Exception as e:  # a malformed config is a real problem, but not this function's to raise
        logger.warning("[isoexec-resolve] could not read %s: %s", cfg_path, e)
        return None
    text_cfg = cfg.get("text_config")
    if isinstance(text_cfg, dict):
        merged = dict(text_cfg)
        merged.update({k: v for k, v in cfg.items() if k != "text_config"})
        if "architectures" not in cfg and "architectures" in text_cfg:
            merged["architectures"] = text_cfg["architectures"]
        return merged
    return cfg


def _architectures_from_config_json(model_path: str) -> Tuple[str, ...]:
    """Read ``architectures`` out of a local snapshot's config.json."""
    cfg = raw_config_json(model_path)
    if not cfg:
        return ()
    return tuple(cfg.get("architectures") or ())


def _architectures_from_hf_cache(model_id: str) -> Tuple[str, ...]:
    """Read ``architectures`` via transformers, from the local HF cache only.

    Network access is off unless ``SKYRL_ISOEXEC_ARCH_FROM_HUB=1``: this runs in every worker, and a
    resolution step that can block on the network can hang the whole job on a DNS timeout.
    """
    allow_network = os.environ.get("SKYRL_ISOEXEC_ARCH_FROM_HUB", "0") == "1"
    try:
        from transformers import AutoConfig
    except Exception:
        return ()
    try:
        cfg = AutoConfig.from_pretrained(model_id, local_files_only=not allow_network, trust_remote_code=False)
    except Exception:
        return ()
    archs = getattr(cfg, "architectures", None) or ()
    if not archs:
        text_cfg = getattr(cfg, "text_config", None)
        archs = getattr(text_cfg, "architectures", None) or ()
    return tuple(archs)


def resolve_architectures(model_path: str) -> Tuple[str, ...]:
    """The checkpoint's declared HF architecture class names, or ``()`` if none can be read offline."""
    if not model_path:
        return ()
    archs = _architectures_from_config_json(model_path)
    if archs:
        return archs
    return _architectures_from_hf_cache(model_path)


def assert_no_architecture_collisions() -> None:
    """Two profiles claiming the same HF architecture must both carry a discriminator.

    A shared class name is legitimate, but resolving it by module order is not: a collision is
    allowed only when every colliding profile can tell itself apart from the checkpoint's own config.
    """
    seen = {}
    for mod in _load_modules():
        for arch in getattr(mod.PROFILE, "architectures", ()):
            seen.setdefault(arch, []).append(mod)
    for arch, mods in seen.items():
        if len(mods) > 1 and not all(getattr(m.PROFILE, "arch_discriminator", None) for m in mods):
            raise ResolutionError(
                f"architecture {arch!r} is claimed by {[m.__name__ for m in mods]} and at least one "
                f"of them declares no arch_discriminator. Sharing an architecture key is allowed "
                f"only when every claimant can separate itself from the checkpoint's own config; "
                f"otherwise resolution would depend on module order."
            )


def resolve_model_module(model_path: str):
    """Return the ``models/*.py`` module whose profile claims this model, plus how it was matched.

    Returns ``(module, how)`` where ``how`` is ``"architecture"`` or ``"name"``. Raises
    ``ResolutionError`` when nothing claims it -- an unknown model must never run under a guessed
    composition.
    """
    mods = _load_modules()
    assert_no_architecture_collisions()

    archs = resolve_architectures(model_path)
    if archs:
        raw = raw_config_json(model_path)
        candidates = [m for m in mods if set(getattr(m.PROFILE, "architectures", ())) & set(archs)]
        # A discriminator only demotes a candidate, and only when a config is actually readable.
        if raw is not None and len(candidates) > 1:
            kept = [m for m in candidates if _claims(m, raw)]
            if kept:
                candidates = kept
        if len(candidates) == 1:
            return candidates[0], "architecture"
        if len(candidates) > 1:
            raise ResolutionError(
                f"architecture(s) {list(archs)} (model {model_path!r}) are claimed by "
                f"{[m.PROFILE.model for m in candidates]} and no discriminator separated them. "
                f"Refusing to pick: a wrong manifest is a VALID manifest naming a different "
                f"composition, and the run would be bitwise consistent with itself."
            )
        raise ResolutionError(
            f"no isoexec profile claims architecture(s) {list(archs)} (model {model_path!r}). "
            f"Add a models/*.py declaring a ModelProfile for it."
        )

    key = (model_path or "").lower()
    for mod in mods:
        for pat in getattr(mod.PROFILE, "name_patterns", ()):
            if pat in key:
                logger.warning(
                    "[isoexec-resolve] %r matched profile %s by NAME PATTERN %r -- no config.json was "
                    "readable, so the architecture key could not be used. This is the documented "
                    "fallback, not the mechanism; pass a local snapshot path to resolve exactly.",
                    model_path,
                    mod.PROFILE.model,
                    pat,
                )
                return mod, "name"
    raise ResolutionError(
        f"no isoexec composition manifest registered for model {model_path!r}, and no config.json was "
        f"readable to resolve it by architecture. Add a models/*.py with a ModelProfile "
        f"(see models/qwen3_5.py)."
    )


def _claims(mod, raw_cfg: dict) -> bool:
    """Does this module's discriminator accept the checkpoint's raw config? Profiles without a
    discriminator abstain (True)."""
    disc = getattr(mod.PROFILE, "arch_discriminator", None)
    if disc is None:
        return True
    try:
        return bool(disc(raw_cfg))
    except Exception as e:
        logger.warning("[isoexec-resolve] discriminator for %s raised (%s); abstaining", mod.PROFILE.model, e)
        return False


def resolve_profile(model_path: str):
    """The ``ModelProfile`` claiming this model."""
    mod, _how = resolve_model_module(model_path)
    return mod.PROFILE


def profile_for_architecture(arch_name: str):
    """Look a profile up by HF architecture class name alone (no path). Returns None if unclaimed."""
    for mod in _load_modules():
        if arch_name in getattr(mod.PROFILE, "architectures", ()):
            return mod.PROFILE
    return None


def registered_profiles() -> list:
    """Every registered profile."""
    return [mod.PROFILE for mod in _load_modules()]


def describe_resolution(model_path: str) -> dict:
    """Structured resolution result, for logging."""
    archs = resolve_architectures(model_path)
    try:
        mod, how = resolve_model_module(model_path)
        return {
            "model_path": model_path,
            "architectures": list(archs),
            "resolved_to": mod.PROFILE.model,
            "module": mod.__name__,
            "matched_by": how,
        }
    except ResolutionError as e:
        return {"model_path": model_path, "architectures": list(archs), "resolved_to": None, "error": str(e)}
