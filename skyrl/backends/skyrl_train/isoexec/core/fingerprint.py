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
#: (op, site, impl_id, pins) already attested to the obligation ledger. Guards the per-forward
#: recorders from re-reporting a fact the ledger already holds; see record_install.
_ATTESTED: set = set()


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
        return
    # The attestation ledger: a record's existence is what discharges install_attest@INSTALL
    # (NOT_INSTALLED included -- attesting a non-install is the record; the comparator decides
    # whether it violates the contract). Fail-safe, separately from the record itself.
    #
    # ATTESTED ONCE PER DISTINCT FACT. An install is a process-wide fact, but some recorders sit on
    # a per-forward path (gdn_gptmodel._state rebinds on a (data_ptr, shape) change and re-records
    # every call), and after the INSTALL phase closes each repeat is a LATE record -- which logs an
    # error AND rewrites the whole enforcement.json. Measured on a live engine: 61,834 repeats of
    # the same gdn.state attestation rewriting an 11 MB artifact per GDN layer per forward, with the
    # ledger growing unboundedly, i.e. quadratic; generation ran >=25x slower than baseline.
    # Keying on (op, site, impl_id, pins) keeps the semantics -- the record's EXISTENCE is what
    # discharges the obligation, and a genuine drift changes impl_id or the pins and so re-reports.
    try:
        key = (op, site, impl_id, repr(sorted(pinned.items())) if isinstance(pinned, dict) else repr(pinned))
        if key in _ATTESTED:
            return
        _ATTESTED.add(key)

        from . import enforce

        enforce.report(f"install_attest:{op}:{site}", enforce.INSTALL, enforce.OK, impl_id)
    except Exception as e:  # pragma: no cover - never fatal
        logger.warning(f"[ISOEXEC-FINGERPRINT] install_attest report({op},{site}) skipped: {e}")


def record_installs(op: str, sites, impl_id: str, obj=None, pinned=None) -> None:
    """``record_install`` for the common case of one impl serving several sites.

    A rebind is a process-wide fact, so which sites it serves is a property of the runtime rather
    than of the call; spelling it ``record_installs(op, ENGINE_SITES, ...)`` keeps that readable.
    """
    for site in (sites,) if isinstance(sites, str) else sites:
        record_install(op, site, impl_id, obj, pinned)


def pin_disagreements(want_pins, got_pins) -> list:
    """Per-key disagreements between the contract's pinned constants and the recorded ones.

    The contract's pins are the claim, so every key it names must be recorded and must agree; a key
    the install reports and the contract does not pin is the install's own detail. Values compare
    by the registry's pin equality, since a pin round-trips through JSON (a declared tuple arrives
    as a list) and a bool is never an int. Pure and import-light: an install site can call it
    directly to compare what it is about to bind.
    """
    from .registry import pin_values_equal

    want, got = dict(want_pins or {}), dict(got_pins or {})
    problems = []
    for key in sorted(want, key=str):
        if key not in got:
            problems.append(f"{key}: contract pins {want[key]!r}, install recorded nothing")
        elif not pin_values_equal(want[key], got[key]):
            problems.append(f"{key}: contract pins {want[key]!r}, install used {got[key]!r}")
    return problems


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
        # Each per-key verdict also lands in the obligation ledger (fail-safe): the comparator
        # itself stays log-only; the FIRST_FORWARD boundary is what refuses per the severity table.
        try:
            from . import enforce
        except Exception:  # pragma: no cover - a broken ledger must not break the comparator
            enforce = None
        _phase = enforce.FIRST_FORWARD if enforce and "first_forward" in tag else (enforce.INSTALL if enforce else None)
        entries = view or {}
        for key, got in sorted(rec.items()):
            want = entries.get(key)
            problem = None
            if want is None:
                problem = f"{key} installed as {got['impl_id']!r} but manifest names no such (op,site)"
            elif want["impl_id"] != got["impl_id"]:
                problem = (
                    f"{key}: manifest={want['impl_id']!r} but INSTALLED={got['impl_id']!r} "
                    f"({got['module']}.{got['qualname']})"
                )
            elif "pinned_constants" in got:
                # Only when the recorder reported pins: an install that reports none is a blind
                # spot (missing_from_fingerprint's business), not evidence that the pins agree.
                bad = pin_disagreements(want.get("pinned_constants"), got["pinned_constants"])
                if bad:
                    problem = (
                        f"{key}: manifest pins={want.get('pinned_constants')!r} but INSTALLED pins="
                        f"{got['pinned_constants']!r} -- {'; '.join(bad)} "
                        f"({got['module']}.{got['qualname']})"
                    )
            if problem is not None:
                problems.append(problem)
            if enforce is not None:
                oid = f"fingerprint:{key[0]}:{key[1]}"
                if problem is None:
                    enforce.report(oid, _phase, enforce.OK, f"({tag}) {got['impl_id']}")
                else:
                    enforce.report(oid, _phase, enforce.VIOLATION, problem)
        for p in problems:
            logger.error("[ISOEXEC-FINGERPRINT] MISMATCH %s", p)
        if not problems:
            logger.warning("[ISOEXEC-FINGERPRINT] all %d instrumented installs match the manifest", len(rec))
        # Blind spots at INFO: keys belonging to the other runtime legitimately appear here.
        gaps = missing_from_fingerprint(view)
        if gaps:
            logger.info("[ISOEXEC-FINGERPRINT] %d manifest key(s) unrecorded in this process: %s", len(gaps), gaps)
    return problems
