"""The obligation ledger: every check the contract requires, derived from the contract itself.

The inversion this module exists for: a check that exists fires, but a check that never runs is
silent -- a run can train for hours under a wrong declaration with no trace but two ERROR lines in
logs nobody surfaces. Here the REQUIRED checks are derived
mechanically from the frozen contract (``derive_obligation_plan``), every existing check site
reports its verdict into a per-process ledger (``report``), and ``close_phase`` refuses when an
obligation due at a boundary has no record at all -- a check that never ran is as loud as one that
failed.

Severity is decided once, in ``SEVERITY`` + the deployment-half demotion, not per call site; the
``EXCEPTIONS`` list is the only sanctioned softness and each entry carries its reason and removal
condition. The verdict is serialized next to the contract artifact as
``enforcement.<side>.r<rank>.json`` -- one file per process, because the sixteen workers sharing a
contract directory made a single file last-writer-wins -- and summarized in one
``[ISOEXEC-ENFORCE]`` line. Readers use ``verdict_artifacts`` / ``merge_verdicts``.

Reporting is fail-safe by construction: a broken reporter logs and never breaks the check it
wraps. The only exception this module raises on purpose is the boundary refusal in ``close_phase``.
The enforcement-layer refusals -- claims, phase boundaries, the weight-sync handshake, contract
delivery, the rowinv engagement boundary, and the two op-side contract-vs-install asserts (the pik
plan check and the NCCL tuple check) -- go through ``refuse``, so ``demoted()`` (strict off, or
debug tracing armed) turns them into logged verdicts without ever weakening the ledger behind
them. Ops raise their own
``RuntimeError``s for runtime invariants (a missing kernel, an unsupported shape); those are not
contract enforcement and are not demoted.
"""

from __future__ import annotations

import fnmatch
import glob
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
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
# Synthesized at close_phase too: a refuse-severity obligation whose every record is a skip nobody
# recognized. ``skipped`` is the ledger's honest answer for "the checker ran and could not judge",
# which is exactly the silent-inert class -- so it discharges nothing unless the skip names one of
# the structural reasons below.
UNCHECKED = "unchecked"

# The skip reasons a boundary accepts as genuinely discharging a refuse-severity obligation: the
# check could not apply at all, rather than could not be performed. Evidence is spelled
# ``"<reason>: <detail>"`` so the whitelist is by reason, never by obligation or by blanket.
# Deliberately NOT here: a topology axis the caller could not obtain (an adapter that reports no
# runtime facts must not close INSTALL clean) and a checker that raised (now a CHECKER-ERROR
# violation, see core/adapter.check_all_claims).
SKIP_NO_PEER_STAMP = "no-peer-stamp"
SKIP_NO_LOCAL_CONTRACT = "no-local-contract"
RECOGNIZED_SKIPS = (SKIP_NO_PEER_STAMP, SKIP_NO_LOCAL_CONTRACT)

# Prefix of the evidence a checker that RAISED records, as a violation (fail closed).
CHECKER_ERROR = "CHECKER-ERROR"

REFUSE = "refuse"
LOG = "log"

SIDES = {"trainer": TRAINER_SITES, "engine": ENGINE_SITES}

# The one severity table: (kind, phase) -> refuse | log. Function-half obligations refuse;
# deployment-half entries demote to log in ``severity()`` -- their neutrality proof, not this
# ledger, is what licenses them. ``served`` is engagement observability, never a stop condition --
# except for the impls named in SERVED_REFUSE_IMPLS below, whose non-engagement IS a wrong
# function, not a missed optimization.
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
# obligation and is listed in the plan's ``no_served_counter`` instead, keeping visible the
# inversion that the strongest-obligation engine impls are exactly the counterless ones.
SERVED_COUNTERS: Dict[Tuple[str, str], str] = {
    ("attention.varlen", "vllm_flash_ns1"): "IS EXECUTING banner (ops/attention/varlen_backend.py)",
    ("mm", "cublaslt_pinned"): "per-shape census (ops/mm/mm_cublaslt.py)",
    ("gdn.core", "native_fused_sigmoid"): "[ISOEXEC-GDN-SPLIT/-ROWS/-BV64] served= censuses (ops/gdn/)",
    ("gdn.conv", "causal_conv1d_fn"): "[ISOEXEC-GDN-CONV-BWD] served= census (ops/gdn/gdn_ops.py)",
    ("moe.dispatch", "index_build"): "served=/validated= census (ops/moe/moe_fused_permute.py)",
    ("moe.combine", "pik_leaf_tree"): "owner-combine served=/ADMITTED (ops/moe/moe_pik_combine_owner.py)",
    ("moe.experts", "batched_bmm"): "leaf-tree served= census (ops/moe/moe_batched_experts.py)",
    ("collectives.tree_all_reduce", "pik_tree"): "TRANSPORT RESOLVED banner + transport_counts (pik_tp_invariant.py)",
    ("logprobs.log_softmax", "rowinv_leaftree"): (
        "[ISOEXEC-ROWINV-LOGPROB] served= census (ops/logprobs/rowinv.py::stats()); served>0 is the "
        "ONLY evidence of engagement -- the banner proves the flag arrived, not that the impl ran"
    ),
}

# SEVERITY ESCALATION for served@STEP1, by (op, impl_id). (SERVED, STEP1) stays LOG in the table
# above because for every other counter-bearing impl a zero census is a performance fact, not a
# correctness fact: those impls are engine-side execution twins (``bitwise_equal_to`` the
# incumbent) or single-runtime installs whose decline path falls back to a function the contract
# still licenses -- an unengaged twin leaves both runtimes computing the same bits, just slower.
#
# The impls named here are categorically different: each REPLACES the incumbent on BOTH runtimes
# at once. rowinv_leaftree declares capabilities {tp_invariant, row_count_invariant} and
# deliberately carries NO bitwise_equal_to claim -- there exists no twin whose engagement could
# stand in for it. So "contract selects it, census says served=0" on ONE side means that side
# silently fell back to a DIFFERENT fp32 function from the peer that did serve it: the PPO ratio
# leaves 1 by construction. That is the exact one-sided composition the contract machinery exists
# to refuse, and it hides behind a MATCHING contract hash: both sides compose it, only one
# executes (trainer served=4096 against engine served=0). The
# manifest proves selection; only the census proves engagement. Hence REFUSE -- for exactly these
# impls, never as a blanket flip of the SERVED kind.
SERVED_REFUSE_IMPLS: Dict[Tuple[str, str], str] = {
    ("logprobs.log_softmax", "rowinv_leaftree"): (
        "dual-runtime replacement with no bitwise_equal_to twin: a one-sided serve is a "
        "split-brain composition behind a matching contract hash"
    ),
}
# An escalated impl must have a live census to judge engagement by; without one the escalation
# could only ever refuse MISSING, which is a broken reporter, not a safety property.
assert set(SERVED_REFUSE_IMPLS) <= set(SERVED_COUNTERS), "SERVED_REFUSE_IMPLS entries owe a SERVED_COUNTERS census"


@dataclass(frozen=True)
class Exemption:
    """One deliberate softness: the matched obligations never refuse and count as ``excepted``.

    ``pattern`` is an fnmatch over obligation ids. Every entry owes a reason and a removal
    condition; an exemption with neither is per-site judgment sneaking back in.
    """

    pattern: str
    reason: str
    removal: str


# The first twelve entries are recorders
# whose impl_id is a literal or a flag mirror, so their record echoes the declaration's own source
# and a match proves nothing about the install ("flag arrived but install didn't happen" stays
# invisible). Their removal condition is uniformly recorder independence: derive the impl_id off
# the installed object / live predicate, then delete the entry. gdn.state's FINGERPRINT exception
# is deliberately NOT here -- its declaration was fixed (cpr variant), so a gdn.state
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
    # REFUSE for the served obligations of SERVED_REFUSE_IMPLS entries; None everywhere else.
    # Set only at plan derivation, mechanically from the contract's selected impl -- never at a
    # call site -- so severity stays decided in one place (the tables above).
    severity_override: Optional[str] = None


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
    # The single escalation rule: only REFUSE is honored, so an override can strengthen a table
    # entry (SERVED_REFUSE_IMPLS) but can never weaken one, and never outranks the deployment-half
    # demotion above.
    if ob.severity_override == REFUSE:
        return REFUSE
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
                override = REFUSE if (op, e.impl.id) in SERVED_REFUSE_IMPLS else None
                obs.append(Obligation(f"served:{op}:{case}", SERVED, STEP1, e.half, severity_override=override))
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


def skip_recognized(evidence: str) -> bool:
    """Whether a skip's evidence names one of the RECOGNIZED_SKIPS reasons."""
    return str(evidence).split(":", 1)[0].strip() in RECOGNIZED_SKIPS


@lru_cache(maxsize=None)
def _plan_obligation_ids(plan: ObligationPlan) -> frozenset:
    """The obligation ids one side's plan carries -- how a record resolves which side owes it."""
    return frozenset(o.obligation_id for o in plan.obligations)


def _status_of(recs) -> str:
    if not recs:
        return MISSING
    if any(r.result == VIOLATION for r in recs):
        return VIOLATION
    if all(r.result == SKIPPED for r in recs):
        return SKIPPED
    return OK


@dataclass
class ObligationLedger:
    """Per-process record of every reported check verdict, plus the armed per-side plans."""

    records: Dict[str, List[Record]] = field(default_factory=dict)
    plans: Dict[str, ObligationPlan] = field(default_factory=dict)
    closed: set = field(default_factory=set)  # {(side, phase)}
    # Records that arrived AFTER their phase closed, and close_phase's own internal errors: both
    # are states the obligation table cannot express, and both must reach the artifact.
    reopened: List[Record] = field(default_factory=list)
    internal_errors: List[str] = field(default_factory=list)

    def report(self, obligation_id: str, phase: str, result: str, evidence: str = "") -> None:
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r}")
        if result not in (OK, VIOLATION, SKIPPED):
            raise ValueError(f"unknown result {result!r}")
        rec = Record(obligation_id, phase, result, str(evidence))
        self.records.setdefault(obligation_id, []).append(rec)
        # LATE means late for the side that owes this obligation. One ledger serves both sides in a
        # colocated process, and the trainer closing INSTALL says nothing about an engine record
        # that is still on time; an obligation no armed plan claims falls back to "any close".
        owing = self._sides_owing(obligation_id)
        if any(p == phase and (not owing or side in owing) for side, p in self.closed):
            self.reopened.append(rec)

    def _sides_owing(self, obligation_id: str) -> set:
        """The armed sides whose plan carries this obligation; empty when no plan names it."""
        return {side for side, plan in self.plans.items() if obligation_id in _plan_obligation_ids(plan)}

    def all_records(self):
        for recs in self.records.values():
            yield from recs

    def status_of(self, ob: Obligation) -> str:
        return _status_of(self.records.get(ob.obligation_id, ()))

    def judged_status(self, ob: Obligation) -> str:
        """``status_of``, plus the judgment the raw status cannot carry: a refuse-severity
        obligation every record of which is an UNRECOGNIZED skip is ``unchecked``, not discharged."""
        status = self.status_of(ob)
        if status != SKIPPED or severity(ob) != REFUSE:
            return status
        recs = self.records.get(ob.obligation_id, ())
        return status if all(skip_recognized(r.evidence) for r in recs) else UNCHECKED


_LEDGER: Optional[ObligationLedger] = None
#: (obligation_id, phase, result) late records already logged + written. One rewrite per
#: distinct late fact is enough to keep enforcement.json from standing stale-green.
_LATE_SEEN: set = set()


def ledger() -> ObligationLedger:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = ObligationLedger()
    return _LEDGER


def _reset_for_tests() -> None:
    global _LEDGER, _VERDICT_PATH
    _LEDGER = None
    _VERDICT_PATH = None
    _LATE_SEEN.clear()
    _ROWINV_CASES.clear()
    _ROWINV_OK.clear()
    _ROWINV_BOUNDARY_SEEN.clear()
    try:
        from .fingerprint import _ATTESTED

        _ATTESTED.clear()
    except Exception:  # noqa: BLE001 -- reset must never raise
        pass


def report(obligation_id: str, phase: str, result: str, evidence: str = "") -> None:
    """Record one check verdict. Fail-safe: a reporting bug must never break the check it wraps."""
    try:
        led = ledger()
        late = len(led.reopened)
        led.report(obligation_id, phase, result, evidence)
        if len(led.reopened) > late:
            # A record arriving after its phase closed is real evidence, so it is kept rather than
            # refused -- but the write latch would otherwise leave enforcement.json stale-green
            # until some later phase closes for the first time. Rewrite it, marked ``reopened``.
            #
            # ONCE PER DISTINCT LATE FACT. The artifact only has to stop being stale-green, and one
            # rewrite per (obligation, phase, result) achieves that. Rewriting per CALL makes a
            # caller on a per-forward path serialize the whole ledger every time -- measured on a
            # live engine at 11 MB per GDN layer per forward, and worse as the ledger grows. The
            # log is deduped with it so the error stays readable instead of scrolling 61k times.
            late_key = (obligation_id, phase, result)
            if late_key not in _LATE_SEEN:
                _LATE_SEEN.add(late_key)
                logger.error(
                    "[ISOEXEC-ENFORCE] LATE RECORD: %s %s=%s arrived after phase %s closed; verdict "
                    "artifact rewritten with a reopened marker",
                    obligation_id,
                    phase,
                    result,
                    phase,
                )
                _write_verdict()
    except Exception as e:  # noqa: BLE001 - never fatal
        logger.warning(f"[ISOEXEC-ENFORCE] report({obligation_id}, {phase}, {result}) skipped: {e}")


def _strict() -> bool:
    return os.environ.get("SKYRL_ISOEXEC_MANIFEST_STRICT", "1").lower() not in ("", "0", "false", "no")


def debug_demoted() -> bool:
    """Debug tracing's OWN demotion condition, deliberately not an alias for strict=0.

    A debug run exists to localize a kernel mix, which means every refusal must become a logged
    verdict or the run never reaches the trace it was started for. Keeping it separate from
    SKYRL_ISOEXEC_MANIFEST_STRICT is what lets a strict production run stay strict while a traced
    run of the same config continues.
    """
    return bool(os.environ.get("SKYRL_ISOEXEC_DEBUG_TRACE"))


def demoted() -> bool:
    """True when an enforcement refusal must log and continue instead of raising."""
    return not _strict() or debug_demoted()


def refuse(msg: str) -> bool:
    """Raise ``msg`` as the deliberate refusal, or -- when demoted -- log it and return False.

    Every ENFORCEMENT refusal funnels through here -- claims, phase boundaries, the handshake,
    contract delivery, the rowinv engagement boundary, and the two op-side contract-vs-install
    asserts -- so grepping ``enforce.refuse`` is the complete list of what debug mode demotes. It is not the complete list
    of what the package raises: ops raise ``RuntimeError`` directly for their own runtime
    invariants, and those stay fatal in debug mode. The ledger is never touched: a demoted
    violation is recorded exactly as strict mode would record it, so the artifact and the verdict
    stay red and only the raise is suppressed.
    """
    if debug_demoted():
        logger.error("[ISOEXEC-DEBUG] DEMOTED by SKYRL_ISOEXEC_DEBUG_TRACE (violation still recorded): %s", msg)
        return False
    if not _strict():
        logger.error("%s (SKYRL_ISOEXEC_MANIFEST_STRICT=0 -> warn-only)", msg)
        return False
    raise RuntimeError(msg)


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
    silent-inert class; a refuse-severity obligation whose only records are unrecognized skips
    becomes ``unchecked`` for the same reason. Refuse-severity violations raise unless ``demoted``;
    everything else logs. Internal errors never propagate (only the deliberate refusal does) but do
    leave a record in the artifact. Always rewrites the verdict, so the last boundary leaves the
    final one.
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
            # enforce -- say so loudly instead of silently closing an empty plan.
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
            status = led.judged_status(ob)
            if status in (VIOLATION, MISSING, UNCHECKED):
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
        # Fail-open is deliberate (a ledger bug must not stop a run), but a WARNING in a worker log
        # is exactly the invisibility this module exists to kill: leave it in the artifact too.
        msg = f"close_phase({phase}, {side}) internal error: {type(e).__name__}: {e}"
        logger.warning(f"[ISOEXEC-ENFORCE] {msg}")
        try:
            ledger().internal_errors.append(msg)
            _write_verdict()
        except Exception as e2:  # noqa: BLE001 - the recovery path must not crash either
            logger.warning(f"[ISOEXEC-ENFORCE] internal-error record skipped: {e2}")
        return True
    if refusals:
        msg = (
            f"[ISOEXEC-ENFORCE] side={side} phase={phase} REFUSED: {len(refusals)} required "
            f"obligation(s) violated or never checked: {'; '.join(refusals)}. The contract derives "
            "these obligations; a required check with no record is itself a violation. Fix the "
            "reporter or the install -- do not soften the table without an EXCEPTIONS entry."
        )
        return refuse(msg)
    return True


def verdict_counts() -> Dict[str, int]:
    """ok/refused/logged/missing/excepted over every closed phase's obligations."""
    counts = {"ok": 0, "refused": 0, "logged": 0, "missing": 0, "excepted": 0}
    led = ledger()
    for side, plan in led.plans.items():
        for ob in plan.obligations:
            if (side, ob.phase) not in led.closed:
                continue
            status = led.judged_status(ob)
            if exemption_for(ob.obligation_id) is not None:
                counts["excepted"] += 1
            elif status == OK:
                counts["ok"] += 1
            elif status == MISSING:
                counts["missing"] += 1
            elif status == UNCHECKED:  # refuse-severity by construction
                counts["refused"] += 1
            elif status == VIOLATION:
                counts["refused" if severity(ob) == REFUSE else "logged"] += 1
            else:  # a recognized skip: the reporter ran and said why it could not check
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


def _records_json(recs):
    return [{"phase": r.phase, "result": r.result, "evidence": r.evidence} for r in recs]


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
                    "status": led.judged_status(ob) if (side, ob.phase) in led.closed else "open",
                    "excepted": ex.pattern if ex else None,
                    "records": _records_json(led.records.get(ob.obligation_id, ())),
                }
            )
    # Reported ids no plan derives. They already taint install_attestation_digest, so the pair
    # handshake sees them; enumerating only plan obligations here was what let the artifact look
    # green while the digest said otherwise.
    planned = {ob.obligation_id for plan in led.plans.values() for ob in plan.obligations}
    for oid in sorted(set(led.records) - planned):
        recs = led.records[oid]
        obligations.append(
            {
                "id": oid,
                "side": None,
                "kind": "unplanned",
                "phase": recs[0].phase,
                "half": None,
                "severity": "unplanned",
                "status": _status_of(recs),
                "excepted": exemption_for(oid).pattern if exemption_for(oid) else None,
                "records": _records_json(recs),
            }
        )
    return {
        "pid": os.getpid(),
        "sides": sorted(led.plans),
        "phases_closed": sorted(f"{s}:{p}" for s, p in led.closed),
        "counts": verdict_counts(),
        "obligations": obligations,
        "reopened": _records_json(led.reopened),
        "internal_errors": list(led.internal_errors),
        "no_served_counter": [list(t) for side in sorted(led.plans) for t in led.plans[side].no_served_counter],
        "exceptions": [{"pattern": e.pattern, "reason": e.reason, "removal": e.removal} for e in EXCEPTIONS],
    }


#: The per-process verdict file this process has already written, so a name that changes once the
#: side becomes known (see ``_verdict_side``) replaces its own earlier file instead of leaving a
#: stale sibling in the glob.
_VERDICT_PATH: Optional[str] = None


def _verdict_rank() -> str:
    """This process's rank tag for the artifact name: dist rank, else RANK/LOCAL_RANK, else pid."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return str(dist.get_rank())
    except Exception:  # noqa: BLE001 -- no torch / no process group is a pid-named artifact
        pass
    for name in ("RANK", "LOCAL_RANK"):
        v = os.environ.get(name, "").strip()
        if v.lstrip("+-").isdigit():
            return v
    return f"p{os.getpid()}"


def _verdict_side() -> str:
    sides = sorted(ledger().plans)
    if sides:
        return "+".join(sides)
    return os.environ.get("SKYRL_ISOEXEC_DEBUG_SIDE") or "unknown"


def verdict_artifacts(directory: str) -> List[str]:
    """Every per-process verdict file in a contract directory.

    ONE FILE PER PROCESS, not one per run: sixteen workers share the contract directory, and a
    single ``enforcement.json`` made the artifact last-writer-wins -- a trainer worker's green
    verdict overwrote the engine's recorded RED. Readers glob and merge.
    """
    return sorted(glob.glob(os.path.join(directory, "enforcement.*.json")))


def merge_verdicts(directory: str) -> dict:
    """Merge every per-process verdict in ``directory``. Red stays red: the worst status wins.

    Deliberately small -- summed counts, per-obligation worst status, concatenated records and the
    list of files merged. Anything richer belongs in a reader, not here.
    """
    worst = {OK: 0, SKIPPED: 1, "unplanned": 1, MISSING: 2, UNCHECKED: 3, VIOLATION: 4}
    counts: Dict[str, int] = {"ok": 0, "refused": 0, "logged": 0, "missing": 0, "excepted": 0}
    obligations: Dict[str, dict] = {}
    files, pids, errors = [], [], []
    for f in verdict_artifacts(directory):
        try:
            with open(f) as fh:
                v = json.load(fh)
        except Exception as e:  # noqa: BLE001 -- an unreadable sibling must not hide the others
            errors.append(f"{f}: {type(e).__name__}: {e}")
            continue
        files.append(os.path.basename(f))
        pids.append(v.get("pid"))
        for k in counts:
            counts[k] += int(v.get("counts", {}).get(k, 0) or 0)
        errors.extend(v.get("internal_errors", ()) or ())
        for ob in v.get("obligations", ()) or ():
            cur = obligations.get(ob["id"])
            if cur is None or worst.get(ob.get("status"), 0) > worst.get(cur.get("status"), 0):
                obligations[ob["id"]] = dict(ob)
            else:
                cur.setdefault("records", []).extend(ob.get("records", ()) or ())
    return {
        "files": files,
        "pids": pids,
        "counts": counts,
        "obligations": [obligations[k] for k in sorted(obligations)],
        "internal_errors": errors,
    }


def _write_verdict() -> Optional[str]:
    """Serialize the verdict next to the contract artifact (ISOEXEC_CONTRACT_PATH). Never fatal."""
    global _VERDICT_PATH
    path = os.environ.get("ISOEXEC_CONTRACT_PATH")
    if not path:
        return None
    try:
        payload = verdict()
    except Exception as e:  # noqa: BLE001 - a ledger too broken to serialize still owes the artifact
        # The only reason close_phase's fail-open is tolerable is that the failure is visible; a
        # verdict that cannot be built would otherwise take the internal-error record down with it.
        payload = {
            "pid": os.getpid(),
            "verdict_error": f"{type(e).__name__}: {e}",
            "internal_errors": list(getattr(ledger(), "internal_errors", ())),
        }
    try:
        name = f"enforcement.{_verdict_side()}.r{_verdict_rank()}.json"
        out = os.path.join(os.path.dirname(os.path.abspath(path)), name)
        tmp = f"{out}.tmp.{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
        os.replace(tmp, out)
        if _VERDICT_PATH is not None and _VERDICT_PATH != out:
            # The side is unknown until a plan is armed; drop the name this process used before
            # it knew, so the glob never shows one process twice.
            try:
                os.unlink(_VERDICT_PATH)
            except OSError:
                pass
        _VERDICT_PATH = out
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
        claims = list(contract.claims.state)
    except Exception as e:  # noqa: BLE001 - never fatal
        logger.warning(f"[ISOEXEC-ENFORCE] state-hook attestation skipped: {e}")
        return
    for s in claims:
        # Per claim, so one broken resolution cannot silently drop the attestation of the rest --
        # and a checker that raises fails CLOSED, as a violation, never as an absent record.
        try:
            r = chk.check(s, {})
            report(chk.obligation_id(s), INSTALL, r.result, r.evidence)
        except Exception as e:  # noqa: BLE001 - never fatal
            logger.warning(f"[ISOEXEC-ENFORCE] state-hook attestation of {s.state_id!r} failed: {e}")
            report(f"hook_exists:{s.state_id}", INSTALL, VIOLATION, f"{CHECKER_ERROR}: {type(e).__name__}: {e}")


def report_installed_backstop(side: str) -> None:
    """The FIRST_FORWARD completeness backstop: ``validate_contract_against_installed``, wired
    live. This side's recorded keys, completed with the other side's
    contract keys (a single process legitimately installs only its own sites), must exactly cover
    the contract. Reports, then re-raises the check's own refusal under the severity table."""
    from .contract_delivery import (
        ContractDeliveryError,
        expected_installed_keys,
        validate_contract_against_installed,
    )
    from .process_contract import cached_contract
    from .registry_build import build_registry

    oid = "installed_backstop:first_forward"
    try:
        c = cached_contract()
        if c is None:
            report(oid, FIRST_FORWARD, SKIPPED, f"{SKIP_NO_LOCAL_CONTRACT}: no contract built")
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
        if not demoted():
            raise  # the check's own ContractDeliveryError, per the severity table
        refuse(f"[ISOEXEC-ENFORCE] {e}")
        return
    report(oid, FIRST_FORWARD, OK, f"{len(recorder().resolved().keys())} recorded key(s) cover the {side} contract")


# -- rowinv engagement -------------------------------------------------------------------------

ROWINV_OP = "logprobs.log_softmax"
ROWINV_IMPL_ID = "rowinv_leaftree"

#: side -> the rowinv cases this side's contract selects. Cached only once a contract exists (the
#: process contract is immutable after build); () means resolved-and-not-selected, i.e. flag OFF.
_ROWINV_CASES: Dict[str, Tuple[str, ...]] = {}
#: sides whose engagement was judged OK once -- every later boundary call is one set lookup.
_ROWINV_OK: set = set()
#: side -> boundary calls seen while rowinv is selected. Call #1 is the init weight sync, which
#: legitimately precedes any forward; from call #2 on (the first post-step sync) served=0 refuses.
_ROWINV_BOUNDARY_SEEN: Dict[str, int] = {}


def rowinv_selected_cases(side: str) -> Tuple[str, ...]:
    """The cases on ``side`` where THIS process's contract selects rowinv_leaftree; () otherwise.

    Reads only the cached process contract. Flag OFF -- or no contract built yet -- returns (),
    which makes every caller an exact no-op: no obligation, no record, no refusal, so the flag-off
    path is untouched by construction.
    """
    if side in _ROWINV_CASES:
        return _ROWINV_CASES[side]
    from .process_contract import cached_contract

    c = cached_contract()
    if c is None:
        return ()  # deliberately uncached: the contract may still be built later in this process
    sites = set(SIDES.get(side, ()))
    _ROWINV_CASES[side] = tuple(
        sorted(
            case
            for e in c.composition
            if e.impl.id == ROWINV_IMPL_ID and ROWINV_OP in e.region
            for case in e.cases
            if case in sites
        )
    )
    return _ROWINV_CASES[side]


def rowinv_engagement_boundary(side: str, *, require: bool = False) -> bool:
    """Judge rowinv ENGAGEMENT on ``side`` by the live census: ``served > 0`` or refuse.

    The failure this exists for: both processes composed rowinv, so the contract hashes MATCHED
    and the handshake passed -- but only the trainer ever EXECUTED the
    impl (engine served=0 across all workers), and the run burned hours producing a one-sided
    composition the ratio math cannot survive. The manifest proves selection; only
    ``ops/logprobs/rowinv.py::stats()['served'] > 0`` proves engagement (the banner proves the
    flag arrived, which is exactly how that failure hid).

    Call it at each per-sync boundary (trainer: the pre-extraction seam of
    ``broadcast_to_inference_engines``; engine: the once-per-sync ``isoexec_reapply_cached_weights``
    seam). Cost: a latched no-op after the first OK, and never on a per-forward path; it neither
    closes a phase nor serializes the ledger. Readiness before refusing, because the init weight
    sync legitimately precedes any forward: the first boundary call with a silent census
    (calls == 0) defers; from the second call on -- by then a training step has completed on every
    deployment shape we run -- served == 0 refuses even when the dispatch was never consulted
    (calls == 0 there means the dispatch itself is unwired, the worst variant). ``require=True``
    (the preflight and tests) skips the readiness grace entirely.

    Verdicts are reported into the ledger under the same ``served:{op}:{case}`` ids the STEP1
    obligations carry, so a later ``close_phase(STEP1)`` re-judges them from the records; the
    refusal itself goes through ``refuse()``, so debug tracing / SKYRL_ISOEXEC_MANIFEST_STRICT=0
    demote it to a logged verdict with the violation still recorded. Everything but the deliberate
    refusal is fail-safe.
    """
    try:
        cases = rowinv_selected_cases(side)
    except Exception as e:  # noqa: BLE001 -- resolution must never break the sync it rides
        logger.warning(f"[ISOEXEC-ENFORCE] rowinv engagement resolution skipped ({side}): {e}")
        return True
    if not cases or side in _ROWINV_OK:
        return True
    census_error = None
    served = declined = calls = 0
    last_decline = ""
    try:
        _ROWINV_BOUNDARY_SEEN[side] = _ROWINV_BOUNDARY_SEEN.get(side, 0) + 1
        from ..ops.logprobs import rowinv

        s = rowinv.stats()  # read-only copy; the census is the single source of engagement truth
        served = int(s.get("served", 0))
        declined = int(s.get("declined", 0))
        calls = int(s.get("calls", 0))
        last_decline = str(s.get("decline_reason", ""))
    except Exception as e:  # noqa: BLE001 -- selected but unjudgeable fails CLOSED, not silent
        census_error = f"{CHECKER_ERROR}: {type(e).__name__}: {e}"
    if census_error is None:
        if served > 0:
            evidence = f"served={served} declined={declined} calls={calls}"
            for case in cases:
                report(f"served:{ROWINV_OP}:{case}", STEP1, OK, evidence)
            _ROWINV_OK.add(side)
            return True
        if not require and calls == 0 and _ROWINV_BOUNDARY_SEEN.get(side, 0) < 2:
            return True  # init-sync grace: nothing has run yet, nothing to judge yet
    evidence = census_error or (
        f"served=0 declined={declined} calls={calls} last_decline={last_decline!r} "
        "(served>0 is the only evidence of engagement; a banner proves only flag arrival)"
    )
    for case in cases:
        report(f"served:{ROWINV_OP}:{case}", STEP1, VIOLATION, evidence)
    return refuse(
        f"[ISOEXEC-ENFORCE] side={side} contract selects {ROWINV_OP}:{ROWINV_IMPL_ID} at "
        f"[{', '.join(cases)}] but this process NEVER SERVED it: {evidence}. rowinv replaces the "
        "incumbent on BOTH runtimes at once (no bitwise_equal_to twin), so a one-sided serve is a "
        "split-brain composition behind a matching contract hash -- the PPO ratio leaves 1 by "
        "construction. Rowinv is the composed default with no flag to fall back to, so the fix is "
        "the dispatch/admission on this side (see last_decline), never a one-sided opt-out."
    )


# Boundary wrappers: one line at each call site, refusals propagate, everything else is fail-safe.


def install_boundary(side: str) -> bool:
    return close_phase(INSTALL, side)


def first_forward_boundary(side: str) -> bool:
    report_installed_backstop(side)
    return close_phase(FIRST_FORWARD, side)


def weight_sync_boundary(side: str) -> bool:
    return close_phase(WEIGHT_SYNC, side)


def step1_boundary(side: str) -> bool:
    """Close STEP1 -- but only in a process whose obligation plan for ``side`` is armed.

    The only STEP1 call site is the controller's forward gate, and the controller builds no
    contract. An unguarded close there would judge an empty plan (enforcing nothing) and then
    rewrite the verdict artifact from an empty ledger, clobbering the worker verdicts that hold
    the STEP1 records. Armed processes get the full close; everyone else is a no-op.
    """
    try:
        if _arm(side) is None:
            return True
    except Exception as e:  # noqa: BLE001 -- a plan-derivation bug must not break the gate site
        logger.warning("[ISOEXEC-ENFORCE] step1_boundary(%s) plan check skipped: %s", side, e)
        return True
    return close_phase(STEP1, side)
