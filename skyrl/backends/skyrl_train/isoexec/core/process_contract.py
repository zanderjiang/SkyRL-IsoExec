"""Per-process ExecutionContract: build once directly from registry + model selections, hash it,
log it, assert it. The weight-sync handshake seam.

Both runtimes call ``get_process_contract(model_path)`` at startup; because both build the same
complete contract from the same code, model and arch, the identities match by construction and
``assert_contract_agreement`` turns a composition split-brain into a refuse-to-run rather than a
broken gate. SKYRL_ISOEXEC_MANIFEST_STRICT (default "1") makes a mismatch fatal; "0" is warn-only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import struct

logger = logging.getLogger(__name__)

_CONTRACT = None  # cached ExecutionContract for this process
_VIEW = None  # cached {(op, site) -> {impl_id, version, pinned_constants, half}} projection

# Named composition components that live outside the model registry but must still agree across the
# pair. They are not registry ops because an installed op key obligates every model contract to
# cover it, which is wrong for a cross-model process capability. A side whose install never ran
# never registers its extension, so the hashes differ and weight-sync refuses. With no extensions
# registered the composite hash equals the plain numerical_policy identity.
_EXTENSIONS: dict = {}


def register_contract_extension(name: str, digest_fn) -> None:
    """Fold a named component into the handshake hash.

    ``digest_fn() -> str`` is called lazily at every ``contract_hash()`` read. Idempotent per name,
    and must be registered on both sides before their handshake or the mismatch is the outcome.
    """
    _EXTENSIONS[name] = digest_fn


# Historical name, kept as an alias for existing callers.
register_manifest_extension = register_contract_extension


def composite_hash(base: str | None) -> str | None:
    """``base`` folded with the registered extensions."""
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


def _decode_constant(v):
    # Contract constants carry floats as fp64 BitPatterns; the live fingerprint compares plain values.
    from ..contract import BitPattern

    if isinstance(v, BitPattern) and v.dtype == "fp64":
        return struct.unpack("<d", struct.pack("<Q", int(v.bits, 16)))[0]
    return v


def build_contract_view(contract, registry) -> dict:
    """The contract's per-(op, site) projection the fingerprint and NCCL asserts compare against."""
    from .contract_delivery import _primary_op

    view = {}
    for e in contract.composition:
        op = _primary_op(registry, e)
        for case in e.cases:
            view[(op, case)] = {
                "impl_id": e.impl.id,
                "version": e.impl.version,
                "pinned_constants": {k: _decode_constant(v) for k, v in e.constants},
                "half": e.half,
            }
    return view


def get_process_contract(model_path: str | None = None, *, arch=None):
    """Build (once) and return this process's contract. Idempotent and cached.

    Returns None when no contract exists yet and no model_path is given. If ISOEXEC_CONTRACT_PATH is
    set, the first build also writes the canonical artifact there (best-effort, never fatal).
    """
    global _CONTRACT, _VIEW
    if _CONTRACT is not None:
        return _CONTRACT
    if not model_path:
        return None
    from .. import models
    from .registry_build import build_registry

    reg = build_registry(strict=True)
    c = models.build_for(model_path, reg, arch=arch)
    _CONTRACT = c
    _VIEW = build_contract_view(c, reg)
    ids = c.identities
    logger.warning(
        "[ISOEXEC-CONTRACT] model=%s arch=%s numerical_policy=%s semantic=%s deployment=%s entries=%d",
        c.model.family,
        c.composition[0].impl.arch if c.composition else "?",
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
    """This process's already-built contract, or None. Never builds one -- callers deep in a
    forward have no model_path; the adapter builds it at startup."""
    return _CONTRACT


def cached_contract_view():
    """The cached contract's (op, site) projection, or None. Never builds."""
    return _VIEW


def contract_hash(model_path: str | None = None) -> str | None:
    """The contract handshake hash: the numerical_policy identity composed with the registered
    extensions. Builds the contract if a model_path is given and it is not built yet; None if no
    contract can be derived yet."""
    c = get_process_contract(model_path)
    return composite_hash(c.identities.numerical_policy) if c is not None else None


def assert_contract_agreement(other_hash: str, *, other_side: str = "peer") -> bool:
    """Handshake check: the peer runtime's contract hash must equal ours.

    Returns True on match. A mismatch is fatal unless SKYRL_ISOEXEC_MANIFEST_STRICT=0, which warns
    and returns False.
    """
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
