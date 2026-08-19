"""Per-process composition manifest: build once, freeze, hash, log, cache. The handshake seam.

Both runtimes call ``get_process_manifest(model_path)`` at startup; because both build the same
complete manifest from the same code, model and arch, the hashes match by construction and
``assert_manifest_agreement`` turns a composition split-brain into a refuse-to-run rather than a
broken gate. SKYRL_ISOEXEC_MANIFEST_STRICT (default "1") makes a mismatch fatal; "0" is warn-only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os

logger = logging.getLogger(__name__)

_MANIFEST = None  # cached frozen Manifest for this process
_HASH = None

# Named composition components that live outside the model registry but must still agree across the
# pair. They are not registry ops because an installed op key obligates every model manifest to
# cover it, which is wrong for a cross-model process capability. A side whose install never ran
# never registers its extension, so the hashes differ and weight-sync refuses. With no extensions
# registered the composite hash equals the plain manifest hash.
_EXTENSIONS: dict = {}


def register_manifest_extension(name: str, digest_fn) -> None:
    """Fold a named component into the handshake hash.

    ``digest_fn() -> str`` is called lazily at every ``manifest_hash()`` read. Idempotent per name,
    and must be registered on both sides before their handshake or the mismatch is the outcome.
    """
    _EXTENSIONS[name] = digest_fn


def _composite(base: str | None) -> str | None:
    if base is None or not _EXTENSIONS:
        return base
    ext = {}
    for name, fn in sorted(_EXTENSIONS.items()):
        try:
            ext[name] = str(fn())
        except Exception as e:  # noqa: BLE001 -- a broken extension must not fake agreement
            ext[name] = f"EXTENSION-ERROR:{type(e).__name__}"
    payload = json.dumps({"base": base, "extensions": ext}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def composite_hash(base: str | None) -> str | None:
    """``base`` folded with the registered extensions, exactly as ``manifest_hash`` folds its own."""
    return _composite(base)


def get_process_manifest(model_path: str, *, arch=None):
    """Build (once) and return this process's frozen manifest. Idempotent and cached."""
    global _MANIFEST, _HASH
    if _MANIFEST is not None:
        return _MANIFEST
    from .. import models
    from .registry_build import build_registry

    reg = build_registry(strict=True)
    m = models.build_for(model_path, reg, arch=arch).freeze()
    _MANIFEST = m
    _HASH = m.hash()
    logger.warning(
        "[ISOEXEC-MANIFEST] model=%s arch=%s hash=%s entries=%d (fn=%d/deploy=%d)",
        m.model,
        m.arch,
        _HASH,
        len(m._entries),
        len(m.function_entries()),
        len(m.deployment_entries()),
    )
    return m


def cached_manifest():
    """This process's already-built frozen manifest, or None. Never builds one -- callers deep in a
    forward have no model_path; the adapter builds it at startup."""
    return _MANIFEST


def manifest_hash(model_path: str | None = None) -> str | None:
    """The handshake hash: the frozen manifest's hash composed with any registered extensions.

    Builds the manifest if a model_path is given and it is not built yet. With no extensions this is
    exactly the manifest hash.
    """
    if _HASH is None and model_path is not None:
        get_process_manifest(model_path)
    return _composite(_HASH)


def assert_manifest_agreement(other_hash: str, *, other_side: str = "peer") -> bool:
    """Handshake check: the peer runtime's manifest hash must equal ours.

    Returns True on match. A mismatch is fatal unless SKYRL_ISOEXEC_MANIFEST_STRICT=0, which warns
    and returns False.
    """
    ours = _composite(_HASH)
    if ours is None:
        logger.warning("[ISOEXEC-MANIFEST-HANDSHAKE] local manifest not built; skipping agreement check")
        return True
    if other_hash == ours:
        logger.warning("[ISOEXEC-MANIFEST-HANDSHAKE] MATCH hash=%s (%s)", ours, other_side)
        return True
    strict = os.environ.get("SKYRL_ISOEXEC_MANIFEST_STRICT", "1").lower() not in ("", "0", "false", "no")
    msg = (
        f"[ISOEXEC-MANIFEST-HANDSHAKE] MISMATCH: local={ours} {other_side}={other_hash}. "
        "The two runtimes resolved DIFFERENT compositions -- a split-brain that would break the "
        "gate (or, worse, pass while measuring the wrong thing). Composition source of truth diverged."
    )
    if strict:
        raise RuntimeError(msg)
    logger.error(msg + " (SKYRL_ISOEXEC_MANIFEST_STRICT=0 -> warn-only)")
    return False
