"""Per-process ExecutionContract: derive once from the cached manifest, hash it, log it, assert it.

The contract-side twin of ``process_manifest``: the contract derives lazily from the manifest both
runtimes already build at startup. ``contract_hash`` folds the same registered handshake extensions
as ``manifest_hash``, and agreement is governed by the same SKYRL_ISOEXEC_MANIFEST_STRICT knob.
"""

from __future__ import annotations

import logging
import os

from .process_manifest import cached_manifest, composite_hash, get_process_manifest

logger = logging.getLogger(__name__)

_CONTRACT = None  # cached derived ExecutionContract for this process


def get_process_contract(model_path: str | None = None, *, arch=None):
    """Derive (once) this process's contract from the process manifest.

    Returns None when no manifest exists yet and no model_path is given. If ISOEXEC_CONTRACT_PATH is
    set, the first build also writes the canonical artifact there (best-effort, never fatal).
    """
    global _CONTRACT
    if _CONTRACT is not None:
        return _CONTRACT
    m = get_process_manifest(model_path, arch=arch) if model_path else cached_manifest()
    if m is None:
        return None
    from .contract_build import build_execution_contract
    from .registry_build import build_registry

    c = build_execution_contract(build_registry(strict=True), m)
    _CONTRACT = c
    ids = c.identities
    logger.warning(
        "[ISOEXEC-CONTRACT] model=%s arch=%s numerical_policy=%s semantic=%s deployment=%s entries=%d",
        c.model.family,
        m.arch,
        ids.numerical_policy,
        ids.semantic,
        ids.deployment,
        len(c.composition),
    )
    path = os.environ.get("ISOEXEC_CONTRACT_PATH")
    if path:
        try:
            from .contract_delivery import write_contract_file

            write_contract_file(c, path)
            logger.warning("[ISOEXEC-CONTRACT] artifact written to %s", path)
        except Exception as e:  # noqa: BLE001 -- artifact write is best-effort, never fatal
            logger.warning("[ISOEXEC-CONTRACT] artifact write to %s skipped: %s", path, e)
    return c


def cached_contract():
    return _CONTRACT


def contract_hash(model_path: str | None = None) -> str | None:
    """The contract handshake hash: the numerical_policy identity composed with the registered
    manifest extensions. None if no contract can be derived yet."""
    c = get_process_contract(model_path)
    return composite_hash(c.identities.numerical_policy) if c is not None else None


def assert_contract_agreement(other_hash: str, *, other_side: str = "peer") -> bool:
    """Handshake check: the peer runtime's contract hash must equal ours. Mirrors
    ``assert_manifest_agreement``, including SKYRL_ISOEXEC_MANIFEST_STRICT semantics."""
    ours = contract_hash()
    if ours is None:
        logger.warning("[ISOEXEC-CONTRACT-HANDSHAKE] local contract not built; skipping agreement check")
        return True
    if other_hash == ours:
        logger.warning("[ISOEXEC-CONTRACT-HANDSHAKE] MATCH hash=%s (%s)", ours, other_side)
        return True
    strict = os.environ.get("SKYRL_ISOEXEC_MANIFEST_STRICT", "1").lower() not in ("", "0", "false", "no")
    msg = (
        f"[ISOEXEC-CONTRACT-HANDSHAKE] MISMATCH: local={ours} {other_side}={other_hash}. "
        "The two runtimes resolved DIFFERENT numerical policies -- a split-brain that would break "
        "the gate (or, worse, pass while measuring the wrong thing)."
    )
    if strict:
        raise RuntimeError(msg)
    logger.error(msg + " (SKYRL_ISOEXEC_MANIFEST_STRICT=0 -> warn-only)")
    return False


def assert_init_info_contract(init_info, *, other_side: str = "trainer") -> bool:
    """Receiver-side handshake against the contract hash a peer stamped on init_info.

    Skips when the peer stamped nothing or the local contract cannot be derived; a genuine mismatch
    raises according to the strictness knob.
    """
    other = getattr(init_info, "contract_hash", None)
    if other is None:
        return True
    try:
        ours = contract_hash()
    except Exception as e:  # noqa: BLE001 -- local derivation failure must not break weight sync
        logger.warning("[ISOEXEC-CONTRACT-HANDSHAKE] local contract derivation failed (%s); skipping", e)
        return True
    if ours is None:
        logger.warning("[ISOEXEC-CONTRACT-HANDSHAKE] local contract not built; skipping agreement check")
        return True
    return assert_contract_agreement(other, other_side=other_side)
