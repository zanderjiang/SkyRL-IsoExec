"""First-forward install fingerprint: what the adapter actually installed, per (op, site).

The contract says which impl should serve each (op, site); this records the module and qualname of
what really got bound, so the two can be compared. That catches the case a config hash cannot: the
flag arrived and the contract agreed, but the install never happened or bound a different impl.
Install sites call ``record_install`` / ``record_installs`` as they bind, and ``log_fingerprint``
warns on every disagreement against the contract's per-(op, site) view
(``process_contract.cached_contract_view``). Recording and logging are fail-soft: never fatal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

Key = Tuple[str, str]  # (op, site)


@dataclass
class ResolvedFingerprint:
    """What the adapter records was actually installed per (op, site): module, qualname, pinned
    constants. Stronger than hashing the config -- it catches "the flag arrived but the install did
    not happen"."""

    installed: Dict[Key, dict] = field(default_factory=dict)

    def keys(self) -> frozenset:
        return frozenset(self.installed.keys())

# The site vocabulary, as tuples an install site can spell in one word. Kept here rather than in
# models/policy.py so an op folder never imports the derivation to describe itself, and so the
# coverage test can resolve these names statically.
TRAINER_SITES = ("trainer_fwd", "trainer_score")
ENGINE_SITES = ("engine_prefill", "engine_decode")
ALL_SITES = TRAINER_SITES + ENGINE_SITES
SCORE_ONLY = ("trainer_score",)
PREFILL_ONLY = ("engine_prefill",)
DECODE_ONLY = ("engine_decode",)

# The impl_id recorded when the install a site was asked for did not happen: the flag was off, the
# rebind found nothing, the guard declined. It is a value rather than silence, so "the manifest
# names fused and nobody installed it" is visible instead of merely unrecorded.
NOT_INSTALLED = "NOT_INSTALLED"

_RECORDER: "FingerprintRecorder | None" = None
_LOGGED_TAGS: set = set()


class FingerprintRecorder:
    """Process-global record of (op, site) -> {impl_id, module, qualname}."""

    def __init__(self) -> None:
        self.installed: dict = {}

    def record(self, op: str, site: str, impl_id: str, obj=None, pinned=None) -> None:
        module = qualname = None
        if obj is not None:
            module = getattr(obj, "__module__", None) or type(obj).__module__
            qualname = getattr(obj, "__qualname__", None) or type(obj).__qualname__
        item = {"impl_id": impl_id, "module": module, "qualname": qualname}
        if pinned is not None:
            item["pinned_constants"] = dict(pinned)
        self.installed[(op, site)] = item

    def resolved(self) -> ResolvedFingerprint:
        return ResolvedFingerprint(installed=dict(self.installed))


def recorder() -> FingerprintRecorder:
    global _RECORDER
    if _RECORDER is None:
        _RECORDER = FingerprintRecorder()
    return _RECORDER


def record_install(op: str, site: str, impl_id: str, obj=None, pinned=None) -> None:
    """Record that ``op`` at ``site`` was installed as ``impl_id``; ``obj`` supplies module and
    qualname. Fail-soft: a recording error is swallowed rather than breaking an install."""
    try:
        recorder().record(op, site, impl_id, obj, pinned)
    except Exception as e:  # pragma: no cover - never fatal
        logger.warning(f"[ISOEXEC-FINGERPRINT] record_install({op},{site}) skipped: {e}")


def record_installs(op: str, sites, impl_id: str, obj=None, pinned=None) -> None:
    """``record_install`` for the common case of one impl serving several sites.

    A rebind is a process-wide fact, so which sites it serves is a property of the runtime rather
    than of the call; spelling it ``record_installs(op, ENGINE_SITES, ...)`` keeps that readable.
    """
    for site in (sites,) if isinstance(sites, str) else sites:
        record_install(op, site, impl_id, obj, pinned)


def log_fingerprint_once(view=None, tag: str = "default") -> dict:
    """``log_fingerprint``, at most once per ``tag`` in this process.

    Adapters call this both at the end of their install sequence and at first forward: the two
    moments see different things -- a lazily built state pool does not exist at install time -- and
    neither alone covers every op.
    """
    try:
        if tag in _LOGGED_TAGS:
            return []
        _LOGGED_TAGS.add(tag)
        return log_fingerprint(view, tag=tag)
    except Exception as e:  # pragma: no cover - never fatal
        logger.warning(f"[ISOEXEC-FINGERPRINT] log_fingerprint_once({tag}) skipped: {e}")
        return []


def missing_from_fingerprint(view) -> list:
    """Contract keys that nothing recorded -- the instrumentation's own blind spots.

    Distinct from a mismatch: a mismatch says the contract is wrong about an op, this says nobody
    can tell. Only INFO, because both runtimes build the complete contract, so an op whose site
    belongs to the other runtime is legitimately unrecorded here.
    """
    entries = view or {}
    rec = recorder().installed
    return sorted(k for k in entries if k not in rec)


def log_fingerprint(view=None, tag: str = "default") -> dict:
    """Log the recorded install fingerprint and return the list of disagreements with ``view``.

    ``view`` is the contract's ``{(op, site) -> {impl_id, pinned_constants, ...}}`` projection.
    Warns on every recorded (op, site) whose installed impl_id or pins disagree with the contract,
    or that the contract does not name at all. An empty list means the instrumented installs match.
    """
    rec = recorder().installed
    logger.warning("[ISOEXEC-FINGERPRINT] (%s) recorded %d instrumented install(s)", tag, len(rec))
    for key, got in sorted(rec.items()):
        logger.info(
            "[ISOEXEC-FINGERPRINT] %s::%s -> %s pins=%s (%s.%s)",
            key[0],
            key[1],
            got["impl_id"],
            got.get("pinned_constants", "unreported"),
            got["module"],
            got["qualname"],
        )
    problems = []
    if view is not None:
        entries = view or {}
        for key, got in sorted(rec.items()):
            want = entries.get(key)
            if want is None:
                problems.append(f"{key} installed as {got['impl_id']!r} but manifest names no such (op,site)")
            elif want["impl_id"] != got["impl_id"]:
                problems.append(
                    f"{key}: manifest={want['impl_id']!r} but INSTALLED={got['impl_id']!r} "
                    f"({got['module']}.{got['qualname']})"
                )
            elif "pinned_constants" in got and want["pinned_constants"] != got["pinned_constants"]:
                problems.append(
                    f"{key}: manifest pins={want['pinned_constants']!r} but INSTALLED pins="
                    f"{got['pinned_constants']!r} ({got['module']}.{got['qualname']})"
                )
        for p in problems:
            logger.error("[ISOEXEC-FINGERPRINT] MISMATCH %s", p)
        if not problems:
            logger.warning("[ISOEXEC-FINGERPRINT] all %d instrumented installs match the manifest", len(rec))
        # Blind spots at INFO: keys belonging to the other runtime legitimately appear here.
        gaps = missing_from_fingerprint(view)
        if gaps:
            logger.info("[ISOEXEC-FINGERPRINT] %d manifest key(s) unrecorded in this process: %s", len(gaps), gaps)
    return problems
