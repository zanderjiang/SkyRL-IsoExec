"""The obligation ledger: every check the contract requires, derived from the contract itself.

The inversion this module exists for: before it, a check that existed fired and a check that never
ran was silent -- the audited production run trained 39 steps under a wrong declaration with the
only trace two ERROR lines in logs nobody surfaces. Here the REQUIRED checks are derived
mechanically from the frozen contract (``derive_obligation_plan``), every existing check site
reports its verdict into a per-process ledger (``report``), and ``close_phase`` refuses when an
obligation due at a boundary has no record at all -- a check that never ran is as loud as one that
failed.

Severity is decided once, in ``SEVERITY`` + the deployment-half demotion, not per call site; the
``EXCEPTIONS`` list is the only sanctioned softness and each entry carries its reason and removal
condition. The verdict is serialized as ``enforcement.json`` next to the contract artifact and
summarized in one ``[ISOEXEC-ENFORCE]`` line.

Reporting is fail-safe by construction: a broken reporter logs and never breaks the check it
wraps. The only exception this module raises on purpose is the boundary refusal in ``close_phase``
(demoted to an ERROR log under SKYRL_ISOEXEC_MANIFEST_STRICT=0, mirroring the handshake).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .fingerprint import ENGINE_SITES, TRAINER_SITES, recorder

logger = logging.getLogger(__name__)

# Phases, in lifecycle order. PATCHING_ARCHITECTURE names thirteen; these four are the ones a real
# reporter exists for today -- adopt more only when a reporter needs them.
INSTALL = "INSTALL"
FIRST_FORWARD = "FIRST_FORWARD"
WEIGHT_SYNC = "WEIGHT_SYNC"
STEP1 = "STEP1"
PHASES = (INSTALL, FIRST_FORWARD, WEIGHT_SYNC, STEP1)

# Obligation kinds.
BUILD_VALID = "build_valid"
INSTALL_ATTEST = "install_attest"
FINGERPRINT = "fingerprint"
SERVED = "served"
DOMAIN_CHECK = "domain_check"
HOOK_EXISTS = "hook_exists"
GATE = "gate"
HANDSHAKE = "handshake"
INSTALLED_BACKSTOP = "installed_backstop"

# Report results.
OK = "ok"
VIOLATION = "violation"
SKIPPED = "skipped"
MISSING = "missing"  # synthesized at close_phase, never passed to report()

REFUSE = "refuse"
LOG = "log"

SIDES = {"trainer": TRAINER_SITES, "engine": ENGINE_SITES}

# The one severity table: (kind, phase) -> refuse | log. Function-half obligations refuse;
# deployment-half entries demote to log in ``severity()`` -- their neutrality proof, not this
# ledger, is what licenses them. ``served`` is engagement observability, never a stop condition.
SEVERITY: Dict[Tuple[str, str], str] = {
    (BUILD_VALID, INSTALL): REFUSE,
    (INSTALL_ATTEST, INSTALL): REFUSE,
    (DOMAIN_CHECK, INSTALL): REFUSE,
    (HOOK_EXISTS, INSTALL): REFUSE,
    (FINGERPRINT, FIRST_FORWARD): REFUSE,
    (INSTALLED_BACKSTOP, FIRST_FORWARD): REFUSE,
    (HANDSHAKE, WEIGHT_SYNC): REFUSE,
    (SERVED, STEP1): LOG,
    (GATE, STEP1): REFUSE,
}

# Which (op, impl_id) pairs declare a live served/engagement counter, inventoried from the
# [ISOEXEC-*] served=/ADMITTED banners in the tree. An entry absent here owes no served@STEP1
# obligation and is listed in the plan's ``no_served_counter`` instead -- the audit's G9 inversion
# (the strongest-obligation engine impls are exactly the counterless ones) stays visible.
SERVED_COUNTERS: Dict[Tuple[str, str], str] = {
    ("attention.varlen", "vllm_flash_ns1"): "IS EXECUTING banner (ops/attention/varlen_backend.py)",
    ("mm", "cublaslt_pinned"): "per-shape census (ops/mm/mm_cublaslt.py)",
    ("gdn.core", "native_fused_sigmoid"): "[ISOEXEC-GDN-SPLIT/-ROWS/-BV64] served= censuses (ops/gdn/)",
    ("gdn.conv", "causal_conv1d_fn"): "[ISOEXEC-GDN-CONV-BWD] served= census (ops/gdn/gdn_ops.py)",
    ("moe.dispatch", "index_build"): "served=/validated= census (ops/moe/moe_fused_permute.py)",
    ("moe.combine", "pik_leaf_tree"): "owner-combine served=/ADMITTED (ops/moe/moe_pik_combine_owner.py)",
    ("moe.experts", "batched_bmm"): "leaf-tree served= census (ops/moe/moe_batched_experts.py)",
    ("collectives.tree_all_reduce", "pik_tree"): "TRANSPORT RESOLVED banner + transport_counts (pik_tp_invariant.py)",
}


@dataclass(frozen=True)
class Exemption:
    """One deliberate softness: the matched obligations never refuse and count as ``excepted``.

    ``pattern`` is an fnmatch over obligation ids. Every entry owes a reason and a removal
    condition; an exemption with neither is per-site judgment sneaking back in.
    """

    pattern: str
    reason: str
    removal: str


# Seeded honestly from the enforcement audit. The first twelve are its G3 list verbatim: recorders
# whose impl_id is a literal or a flag mirror, so their record echoes the declaration's own source
# and a match proves nothing about the install ("flag arrived but install didn't happen" stays
# invisible). Their removal condition is uniformly recorder independence: derive the impl_id off
# the installed object / live predicate, then delete the entry. gdn.state's FINGERPRINT exception
# is deliberately NOT here -- its declaration was fixed (chunk_synced variant), so a gdn.state
# mismatch now refuses at FIRST_FORWARD per the table.
_SELF_ATTESTING = "recorder echoes the declaration's own source (literal/flag mirror), not the installed state"
_RECORDER_INDEPENDENCE = "make the recorder read the installed object or a live predicate, then remove"
EXCEPTIONS: Tuple[Exemption, ...] = (
    Exemption(
        "fingerprint:logprobs.lm_head_slice:*", _SELF_ATTESTING + "; structurally always-MATCH", _RECORDER_INDEPENDENCE
    ),
    Exemption(
        "fingerprint:mm:*", _SELF_ATTESTING + "; ignores the _DISABLED self-check kill-switch", _RECORDER_INDEPENDENCE
    ),
    Exemption(
        "fingerprint:attention.varlen:trainer_fwd",
        _SELF_ATTESTING + "; records even at 0 swapped layers",
        _RECORDER_INDEPENDENCE,
    ),
    Exemption(
        "fingerprint:attention.varlen:trainer_score",
        _SELF_ATTESTING + "; records even at 0 swapped layers",
        _RECORDER_INDEPENDENCE,
    ),
    Exemption("fingerprint:logprobs.log_softmax:trainer_score", _SELF_ATTESTING, _RECORDER_INDEPENDENCE),
    Exemption(
        "fingerprint:gdn.conv:trainer_*",
        _SELF_ATTESTING + "; real choice is call-time in the fla shim",
        _RECORDER_INDEPENDENCE,
    ),
    Exemption(
        "fingerprint:rope.rope:trainer_*", _SELF_ATTESTING + "; blind to the fp32-rope variant", _RECORDER_INDEPENDENCE
    ),
    Exemption("fingerprint:moe.experts:*", _SELF_ATTESTING + "; literals on both sides", _RECORDER_INDEPENDENCE),
    Exemption(
        "fingerprint:moe.dispatch:*", _SELF_ATTESTING + "; literal index_build in both hubs", _RECORDER_INDEPENDENCE
    ),
    Exemption("fingerprint:moe.weights:*", _SELF_ATTESTING, _RECORDER_INDEPENDENCE),
    Exemption(
        "fingerprint:collectives.row_parallel:*",
        _SELF_ATTESTING + "; env mirror, ignores per-layer fallbacks",
        _RECORDER_INDEPENDENCE,
    ),
    Exemption(
        "fingerprint:collectives.tree_all_reduce:*",
        _SELF_ATTESTING + " at the hubs (env mirror); the first-collective re-record and the pik plan assert are live",
        "hub recorder independence; the live re-record already narrows this to the pre-first-collective window",
    ),
    # Tree-green softener: the engine state pool exists only at the first metadata-bearing forward
    # (gdn_gptmodel builds and records it there), so its INSTALL attestation cannot exist yet.
    Exemption(
        "install_attest:gdn.state:*",
        "state pool is built and recorded at the first metadata-bearing forward, after the INSTALL boundary",
        "attest the IsoExecGDNStateLayer attach at swap_gdn_core install time, then remove",
    ),
)


def exemption_for(obligation_id: str) -> Optional[Exemption]:
    for ex in EXCEPTIONS:
        if fnmatch.fnmatchcase(obligation_id, ex.pattern):
            return ex
    return None


@dataclass(frozen=True)
class Obligation:
    obligation_id: str
    kind: str
    phase: str
    half: str = "function"


@dataclass(frozen=True)
class ObligationPlan:
    side: str
    obligations: Tuple[Obligation, ...]
    # (op, site, impl_id) triples that owe no served obligation because no counter exists.
    no_served_counter: Tuple[Tuple[str, str, str], ...]


def severity(ob: Obligation) -> str:
    # The single demotion rule: a deployment-half entry's obligations log, never refuse.
    if ob.half == "deployment":
        return LOG
    return SEVERITY[(ob.kind, ob.phase)]


def derive_obligation_plan(contract, registry, side: str) -> ObligationPlan:
    """Mechanically derive the obligation set this process owes -- no hand lists.

    Every composition entry visible on ``side`` owes install_attest@INSTALL and
    fingerprint@FIRST_FORWARD, plus served@STEP1 iff its impl is in the SERVED_COUNTERS inventory;
    every topology claim owes domain_check@INSTALL; every state claim hook_exists@INSTALL; every
    tolerance claim gate@STEP1; the identities owe handshake@WEIGHT_SYNC; the contract itself owes
    build_valid@INSTALL and the installed-completeness backstop at FIRST_FORWARD.
    """
    from .contract_delivery import _primary_op

    if side not in SIDES:
        raise ValueError(f"unknown side {side!r}; expected one of {sorted(SIDES)}")
    sites = set(SIDES[side])
    obs: List[Obligation] = [
        Obligation("build_valid:contract", BUILD_VALID, INSTALL),
        Obligation("installed_backstop:first_forward", INSTALLED_BACKSTOP, FIRST_FORWARD),
        Obligation("handshake:numerical_policy", HANDSHAKE, WEIGHT_SYNC),
    ]
    unserved: List[Tuple[str, str, str]] = []
    for e in contract.composition:
        op = _primary_op(registry, e)
        for case in e.cases:
            if case not in sites:
                continue
            obs.append(Obligation(f"install_attest:{op}:{case}", INSTALL_ATTEST, INSTALL, e.half))
            obs.append(Obligation(f"fingerprint:{op}:{case}", FINGERPRINT, FIRST_FORWARD, e.half))
            if (op, e.impl.id) in SERVED_COUNTERS:
                obs.append(Obligation(f"served:{op}:{case}", SERVED, STEP1, e.half))
            else:
                unserved.append((op, case, e.impl.id))
    for t in contract.claims.topology:
        obs.append(Obligation(f"domain_check:{t.axis}", DOMAIN_CHECK, INSTALL))
    for s in contract.claims.state:
        obs.append(Obligation(f"hook_exists:{s.state_id}", HOOK_EXISTS, INSTALL))
    for t in contract.claims.tolerances:
        obs.append(Obligation(f"gate:{t.case_pair[0]}|{t.case_pair[1]}", GATE, STEP1))
    order = {p: i for i, p in enumerate(PHASES)}
    obs.sort(key=lambda o: (order[o.phase], o.kind, o.obligation_id))
    return ObligationPlan(side=side, obligations=tuple(obs), no_served_counter=tuple(sorted(unserved)))


@dataclass(frozen=True)
class Record:
    obligation_id: str
    phase: str
    result: str
    evidence: str = ""


@dataclass
class ObligationLedger:
    """Per-process record of every reported check verdict, plus the armed per-side plans."""

    records: Dict[str, List[Record]] = field(default_factory=dict)
    plans: Dict[str, ObligationPlan] = field(default_factory=dict)
    closed: set = field(default_factory=set)  # {(side, phase)}

    def report(self, obligation_id: str, phase: str, result: str, evidence: str = "") -> None:
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r}")
        if result not in (OK, VIOLATION, SKIPPED):
            raise ValueError(f"unknown result {result!r}")
        self.records.setdefault(obligation_id, []).append(Record(obligation_id, phase, result, str(evidence)))

    def all_records(self):
        for recs in self.records.values():
            yield from recs

    def status_of(self, ob: Obligation) -> str:
        recs = self.records.get(ob.obligation_id, ())
        if not recs:
            return MISSING
        if any(r.result == VIOLATION for r in recs):
            return VIOLATION
        if all(r.result == SKIPPED for r in recs):
            return SKIPPED
        return OK


_LEDGER: Optional[ObligationLedger] = None


def ledger() -> ObligationLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = ObligationLedger()
    return _LEDGER


def _reset_for_tests() -> None:
    global _LEDGER
    _LEDGER = None


def report(obligation_id: str, phase: str, result: str, evidence: str = "") -> None:
    """Record one check verdict. Fail-safe: a reporting bug must never break the check it wraps."""
    try:
        ledger().report(obligation_id, phase, result, evidence)
    except Exception as e:  # noqa: BLE001 - never fatal
        logger.warning(f"[ISOEXEC-ENFORCE] report({obligation_id}, {phase}, {result}) skipped: {e}")


def _strict() -> bool:
    return os.environ.get("SKYRL_ISOEXEC_MANIFEST_STRICT", "1").lower() not in ("", "0", "false", "no")


def _arm(side: str) -> Optional[ObligationPlan]:
    led = ledger()
    if side in led.plans:
        return led.plans[side]
    from .process_contract import cached_contract
    from .registry_build import build_registry

    c = cached_contract()
    if c is None:
        return None
    led.plans[side] = derive_obligation_plan(c, build_registry(strict=True), side)
    return led.plans[side]


def close_phase(phase: str, side: str) -> bool:
    """Completeness at a phase boundary: every obligation due now must have a record.

    A missing record becomes a synthesized ``missing`` violation -- the property that kills the
    silent-inert class. Refuse-severity violations raise (SKYRL_ISOEXEC_MANIFEST_STRICT=0 demotes
    to ERROR + False); everything else logs. Internal errors never propagate: only the deliberate
    refusal does. Always rewrites the verdict artifact, so the last boundary leaves the final one.
    """
    refusals: List[str] = []
    try:
        led = ledger()
        plan = _arm(side)
        # Log/artifact latch: recomputation stays idempotent, but the summary line and verdict
        # rewrite happen once per boundary (per-forward delegates were spamming worker logs).
        first_close = (side, phase) not in led.closed
        led.closed.add((side, phase))
        if plan is None:
            # A process that was supposed to build a contract and did not has nothing derivable to
            # enforce -- say so loudly instead of silently closing an empty plan (the audit's
            # "build failure => silently stamp None" class).
            logger.error(
                "[ISOEXEC-ENFORCE] side=%s phase=%s closed with NO contract built; the obligation "
                "plan cannot be derived and nothing was enforced",
                side,
                phase,
            )
            _write_verdict()
            return True
        for ob in plan.obligations:
            if ob.phase != phase:
                continue
            status = led.status_of(ob)
            if status in (VIOLATION, MISSING):
                ex = exemption_for(ob.obligation_id)
                if ex is not None:
                    logger.warning(
                        "[ISOEXEC-ENFORCE] side=%s %s %s EXCEPTED (%s)", side, ob.obligation_id, status, ex.reason
                    )
                elif severity(ob) == REFUSE:
                    refusals.append(f"{ob.obligation_id} ({status})")
                else:
                    logger.error("[ISOEXEC-ENFORCE] side=%s %s %s (severity=log)", side, ob.obligation_id, status)
        if first_close:
            _write_verdict()
            _log_summary(side)
    except Exception as e:  # noqa: BLE001 - the boundary must not crash on ledger bugs
        logger.warning(f"[ISOEXEC-ENFORCE] close_phase({phase}, {side}) internal error: {e}")
        return True
    if refusals:
        msg = (
            f"[ISOEXEC-ENFORCE] side={side} phase={phase} REFUSED: {len(refusals)} required "
            f"obligation(s) violated or never checked: {'; '.join(refusals)}. The contract derives "
            "these obligations; a required check with no record is itself a violation. Fix the "
            "reporter or the install -- do not soften the table without an EXCEPTIONS entry."
        )
        if _strict():
            raise RuntimeError(msg)
        logger.error(msg + " (SKYRL_ISOEXEC_MANIFEST_STRICT=0 -> warn-only)")
        return False
    return True


def verdict_counts() -> Dict[str, int]:
    """ok/refused/logged/missing/excepted over every closed phase's obligations."""
    counts = {"ok": 0, "refused": 0, "logged": 0, "missing": 0, "excepted": 0}
    led = ledger()
    for side, plan in led.plans.items():
        for ob in plan.obligations:
            if (side, ob.phase) not in led.closed:
                continue
            status = led.status_of(ob)
            if exemption_for(ob.obligation_id) is not None:
                counts["excepted"] += 1
            elif status == OK:
                counts["ok"] += 1
            elif status == MISSING:
                counts["missing"] += 1
            elif status == VIOLATION:
                counts["refused" if severity(ob) == REFUSE else "logged"] += 1
            else:  # skipped: the reporter ran and visibly could not check
                counts["logged"] += 1
    return counts


def _log_summary(side: str) -> None:
    c = verdict_counts()
    logger.warning(
        "[ISOEXEC-ENFORCE] side=%s ok=%d refused=%d logged=%d missing=%d excepted=%d",
        side,
        c["ok"],
        c["refused"],
        c["logged"],
        c["missing"],
        c["excepted"],
    )


def verdict() -> dict:
    led = ledger()
    obligations = []
    for side, plan in led.plans.items():
        for ob in plan.obligations:
            ex = exemption_for(ob.obligation_id)
            obligations.append(
                {
                    "id": ob.obligation_id,
                    "side": side,
                    "kind": ob.kind,
                    "phase": ob.phase,
                    "half": ob.half,
                    "severity": severity(ob),
                    "status": led.status_of(ob) if (side, ob.phase) in led.closed else "open",
                    "excepted": ex.pattern if ex else None,
                    "records": [
                        {"phase": r.phase, "result": r.result, "evidence": r.evidence}
                        for r in led.records.get(ob.obligation_id, ())
                    ],
                }
            )
    return {
        "pid": os.getpid(),
        "sides": sorted(led.plans),
        "phases_closed": sorted(f"{s}:{p}" for s, p in led.closed),
        "counts": verdict_counts(),
        "obligations": obligations,
        "no_served_counter": [list(t) for side in sorted(led.plans) for t in led.plans[side].no_served_counter],
        "exceptions": [{"pattern": e.pattern, "reason": e.reason, "removal": e.removal} for e in EXCEPTIONS],
    }


def _write_verdict() -> Optional[str]:
    """Serialize the verdict next to the contract artifact (ISOEXEC_CONTRACT_PATH). Never fatal."""
    path = os.environ.get("ISOEXEC_CONTRACT_PATH")
    if not path:
        return None
    try:
        out = os.path.join(os.path.dirname(os.path.abspath(path)), "enforcement.json")
        tmp = f"{out}.tmp.{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump(verdict(), fh, indent=1, sort_keys=True)
        os.replace(tmp, out)
        return out
    except Exception as e:  # noqa: BLE001 - artifact write is best-effort
        logger.warning(f"[ISOEXEC-ENFORCE] verdict write skipped: {e}")
        return None


def install_attestation_digest() -> str:
    """The INSTALL-phase attestation digest folded into the handshake composite.

    ``CLEAN`` when this process attested no INSTALL-phase violation on a non-excepted obligation,
    else a content hash of the violations. Both sides of a correct pair digest to the literal
    ``CLEAN``, so the composite stays symmetric by construction; a side that declared identically
    but installed differently digests its violations and the pair refuses at weight sync.
    """
    try:
        viol = sorted(
            {
                (r.obligation_id, r.evidence)
                for r in ledger().all_records()
                if r.phase == INSTALL and r.result == VIOLATION and exemption_for(r.obligation_id) is None
            }
        )
        if not viol:
            return "CLEAN"
        payload = json.dumps(viol, sort_keys=True, separators=(",", ":"))
        return f"violations={len(viol)}:sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"
    except Exception as e:  # noqa: BLE001 - a broken digest must not fake agreement
        return f"DIGEST-ERROR:{type(e).__name__}"


def attest_state_hooks(contract) -> None:
    """hook_exists reporter: every StateClaim's ``path::symbol`` ref must be real code in-tree.

    Deprecated name, kept for existing callers: the resolution logic lives in the ContractAdapter's
    ``StateHookChecker`` (core/adapter.py); this delegates and reports each verdict.
    """
    try:
        from .adapter import StateHookChecker

        chk = StateHookChecker()
        for s in contract.claims.state:
            r = chk.check(s, {})
            report(chk.obligation_id(s), INSTALL, r.result, r.evidence)
    except Exception as e:  # noqa: BLE001 - never fatal
        logger.warning(f"[ISOEXEC-ENFORCE] state-hook attestation skipped: {e}")


def report_installed_backstop(side: str) -> None:
    """The FIRST_FORWARD completeness backstop: ``validate_contract_against_installed`` (the audit's
    test-only finding (b)), wired live. This side's recorded keys, completed with the other side's
    contract keys (a single process legitimately installs only its own sites), must exactly cover
    the contract. Reports, then re-raises the check's own refusal under the severity table."""
    from .contract_delivery import ContractDeliveryError, expected_installed_keys, validate_contract_against_installed
    from .process_contract import cached_contract
    from .registry_build import build_registry

    oid = "installed_backstop:first_forward"
    try:
        c = cached_contract()
        if c is None:
            report(oid, FIRST_FORWARD, SKIPPED, "no contract built")
            return
        reg = build_registry(strict=True)
        sites = set(SIDES[side])
        other = {k for k in expected_installed_keys(c, reg) if k[1] not in sites}
        installed = recorder().resolved().keys() | frozenset(other)
    except Exception as e:  # noqa: BLE001 - never fatal
        logger.warning(f"[ISOEXEC-ENFORCE] installed backstop skipped: {e}")
        return
    try:
        validate_contract_against_installed(c, reg, installed)
    except ContractDeliveryError as e:
        report(oid, FIRST_FORWARD, VIOLATION, str(e))
        if _strict():
            raise
        logger.error(f"[ISOEXEC-ENFORCE] {e} (SKYRL_ISOEXEC_MANIFEST_STRICT=0 -> warn-only)")
        return
    report(oid, FIRST_FORWARD, OK, f"{len(recorder().resolved().keys())} recorded key(s) cover the {side} contract")


# Boundary wrappers: one line at each call site, refusals propagate, everything else is fail-safe.


def install_boundary(side: str) -> bool:
    return close_phase(INSTALL, side)


def first_forward_boundary(side: str) -> bool:
    report_installed_backstop(side)
    return close_phase(FIRST_FORWARD, side)


def weight_sync_boundary(side: str) -> bool:
    return close_phase(WEIGHT_SYNC, side)
