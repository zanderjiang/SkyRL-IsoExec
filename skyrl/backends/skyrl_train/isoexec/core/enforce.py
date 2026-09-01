"""The obligation ledger: every check the contract requires, derived from the contract itself.

Required checks are derived from the frozen contract, check sites report verdicts into a
per-process ledger, and ``close_phase`` refuses when an obligation due at a boundary has no record
at all. Severity is decided once in ``SEVERITY``; every enforcement refusal goes through ``refuse``.
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

# Phases, in lifecycle order; these four are the ones a real reporter exists for today.
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
# Synthesized at close_phase too: a refuse-severity obligation whose every record is an
# unrecognized skip. A skip discharges nothing unless it names a structural reason below.
UNCHECKED = "unchecked"

# The skip reasons a boundary accepts as discharging a refuse-severity obligation: the check could
# not apply at all, rather than could not be performed. Evidence is spelled ``"<reason>: <detail>"``
# so the whitelist is by reason, never by obligation.
SKIP_NO_PEER_STAMP = "no-peer-stamp"
SKIP_NO_LOCAL_CONTRACT = "no-local-contract"
RECOGNIZED_SKIPS = (SKIP_NO_PEER_STAMP, SKIP_NO_LOCAL_CONTRACT)

# Prefix of the evidence a checker that RAISED records, as a violation (fail closed).
CHECKER_ERROR = "CHECKER-ERROR"

REFUSE = "refuse"
LOG = "log"

SIDES = {"trainer": TRAINER_SITES, "engine": ENGINE_SITES}

# The one severity table: (kind, phase) -> refuse | log. Deployment-half entries demote to log in
# ``severity()``; ``served`` is observability except for the SERVED_REFUSE_IMPLS below.
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

# Which (op, impl_id) pairs declare a live served/engagement counter. An entry absent here owes no
# served@STEP1 obligation and is listed in the plan's ``no_served_counter`` instead.
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

# SEVERITY ESCALATION for served@STEP1, by (op, impl_id). These impls REPLACE the incumbent on both
# runtimes at once with no bitwise_equal_to twin, so a one-sided serve is a split-brain composition
# behind a matching contract hash. Never a blanket flip of the SERVED kind.
SERVED_REFUSE_IMPLS: Dict[Tuple[str, str], str] = {
    ("logprobs.log_softmax", "rowinv_leaftree"): (
        "dual-runtime replacement with no bitwise_equal_to twin: a one-sided serve is a "
        "split-brain composition behind a matching contract hash"
    ),
}
# An escalated impl needs a live census to judge engagement by; without one it could only ever
# refuse MISSING, which is a broken reporter, not a safety property.
assert set(SERVED_REFUSE_IMPLS) <= set(SERVED_COUNTERS), "SERVED_REFUSE_IMPLS entries owe a SERVED_COUNTERS census"


@dataclass(frozen=True)
class Exemption:
    """One deliberate softness: the matched obligations never refuse and count as ``excepted``.

    ``pattern`` is an fnmatch over obligation ids; every entry owes a reason and a removal condition.
    """

    pattern: str
    reason: str
    removal: str


# The fingerprint entries below are recorders whose impl_id is a literal or a flag mirror, so their
# record echoes the declaration's own source and a match proves nothing about the install.
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
    # The engine state pool exists only at the first metadata-bearing forward, so its INSTALL
    # attestation cannot exist yet.
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
    # REFUSE for the served obligations of SERVED_REFUSE_IMPLS entries; None everywhere else. Set
    # only at plan derivation, never at a call site, so severity stays decided in one place.
    severity_override: Optional[str] = None


@dataclass(frozen=True)
class ObligationPlan:
    side: str
    obligations: Tuple[Obligation, ...]
    # (op, site, impl_id) triples that owe no served obligation because no counter exists.
    no_served_counter: Tuple[Tuple[str, str, str], ...]


def severity(ob: Obligation) -> str:
    # A deployment-half entry's obligations log, never refuse.
    if ob.half == "deployment":
        return LOG
    # Only REFUSE is honored, so an override can strengthen a table entry but never weaken one,
    # and never outranks the demotion above.
    if ob.severity_override == REFUSE:
        return REFUSE
    return SEVERITY[(ob.kind, ob.phase)]


def derive_obligation_plan(contract, registry, side: str) -> ObligationPlan:
    """Mechanically derive the obligation set this process owes -- no hand lists."""
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
    """The obligation ids one side's plan carries."""
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
    # Records that arrived AFTER their phase closed, and close_phase's own internal errors: states
    # the obligation table cannot express, and both must reach the artifact.
    reopened: List[Record] = field(default_factory=list)
    internal_errors: List[str] = field(default_factory=list)

    def report(self, obligation_id: str, phase: str, result: str, evidence: str = "") -> None:
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r}")
        if result not in (OK, VIOLATION, SKIPPED):
            raise ValueError(f"unknown result {result!r}")
        rec = Record(obligation_id, phase, result, str(evidence))
        self.records.setdefault(obligation_id, []).append(rec)
        # LATE means late for the side that owes this obligation: one ledger serves both sides in a
        # colocated process. An obligation no armed plan claims falls back to "any close".
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
        """``status_of``, except that a refuse-severity obligation whose every record is an
        unrecognized skip is ``unchecked``, not discharged."""
        status = self.status_of(ob)
        if status != SKIPPED or severity(ob) != REFUSE:
            return status
        recs = self.records.get(ob.obligation_id, ())
        return status if all(skip_recognized(r.evidence) for r in recs) else UNCHECKED


_LEDGER: Optional[ObligationLedger] = None
#: (obligation_id, phase, result) late records already logged + written.
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
            # A late record is real evidence, so it is kept and the artifact rewritten rather than
            # left stale-green. Once per distinct (obligation, phase, result): rewriting per call
            # would re-serialize the whole ledger from a per-forward path.
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
    """Debug tracing's own demotion condition, deliberately not an alias for strict=0.

    A traced run must reach its trace, so every refusal becomes a logged verdict; keeping this
    separate lets a strict production run of the same config stay strict.
    """
    return bool(os.environ.get("SKYRL_ISOEXEC_DEBUG_TRACE"))


def demoted() -> bool:
    """True when an enforcement refusal must log and continue instead of raising."""
    return not _strict() or debug_demoted()


def refuse(msg: str) -> bool:
    """Raise ``msg`` as the deliberate refusal, or -- when demoted -- log it and return False.

    Every enforcement refusal funnels through here; ops' own runtime invariants raise directly and
    stay fatal. The ledger is untouched: a demoted violation is still recorded red.
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

    A missing record becomes a synthesized ``missing`` violation. Refuse-severity violations raise
    unless ``demoted``; internal errors never propagate but do reach the artifact.
    """
    refusals: List[str] = []
    try:
        led = ledger()
        plan = _arm(side)
        # Log/artifact latch: recomputation stays idempotent, but the summary line and verdict
        # rewrite happen once per boundary.
        first_close = (side, phase) not in led.closed
        led.closed.add((side, phase))
        if plan is None:
            # Nothing derivable to enforce; say so loudly rather than closing an empty plan.
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
        # Fail-open is deliberate (a ledger bug must not stop a run), but a worker-log warning is
        # too easy to miss: leave the error in the artifact too.
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
    # Reported ids no plan derives: they taint install_attestation_digest, so the artifact must
    # show them too rather than looking green while the digest says otherwise.
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


#: The verdict file this process already wrote, so a name that changes once the side becomes known
#: replaces its own earlier file instead of leaving a stale sibling in the glob.
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

    One file per process, not per run: workers share the directory, so a single file would be
    last-writer-wins. Readers glob and merge.
    """
    return sorted(glob.glob(os.path.join(directory, "enforcement.*.json")))


def merge_verdicts(directory: str) -> dict:
    """Merge every per-process verdict in ``directory``. Red stays red: the worst status wins."""
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
        # close_phase's fail-open is only tolerable while the failure stays visible, so an
        # unbuildable verdict must still carry the internal-error record.
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
            # The side is unknown until a plan is armed; drop the earlier name so the glob never
            # shows one process twice.
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

    ``CLEAN`` when no non-excepted INSTALL violation was attested, else a hash of the violations,
    so a side that declared identically but installed differently refuses at weight sync.
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

    Deprecated alias: delegates to ``adapter.StateHookChecker`` and reports each verdict.
    """
    try:
        from .adapter import StateHookChecker

        chk = StateHookChecker()
        claims = list(contract.claims.state)
    except Exception as e:  # noqa: BLE001 - never fatal
        logger.warning(f"[ISOEXEC-ENFORCE] state-hook attestation skipped: {e}")
        return
    for s in claims:
        # Per claim, so one broken resolution cannot drop the rest; a checker that raises fails
        # CLOSED, as a violation, never as an absent record.
        try:
            r = chk.check(s, {})
            report(chk.obligation_id(s), INSTALL, r.result, r.evidence)
        except Exception as e:  # noqa: BLE001 - never fatal
            logger.warning(f"[ISOEXEC-ENFORCE] state-hook attestation of {s.state_id!r} failed: {e}")
            report(f"hook_exists:{s.state_id}", INSTALL, VIOLATION, f"{CHECKER_ERROR}: {type(e).__name__}: {e}")


def report_installed_backstop(side: str) -> None:
    """The FIRST_FORWARD completeness backstop: this side's recorded keys, completed with the other
    side's contract keys, must exactly cover the contract."""
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

#: side -> the rowinv cases this side's contract selects; () means not selected, i.e. flag OFF.
_ROWINV_CASES: Dict[str, Tuple[str, ...]] = {}
#: sides whose engagement was judged OK once -- every later boundary call is one set lookup.
_ROWINV_OK: set = set()
#: side -> boundary calls seen while rowinv is selected. Call #1 is the init weight sync, which
#: legitimately precedes any forward; from call #2 on, served=0 refuses.
_ROWINV_BOUNDARY_SEEN: Dict[str, int] = {}


def rowinv_selected_cases(side: str) -> Tuple[str, ...]:
    """The cases on ``side`` where this process's contract selects rowinv_leaftree; () otherwise.

    Reads only the cached process contract; () makes every caller an exact no-op.
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

    The manifest proves selection; only the census proves engagement, and a one-sided serve is a
    split-brain composition behind a matching contract hash. Called at each per-sync boundary; the
    first call with a silent census defers (the init sync precedes any forward), ``require`` skips
    that grace.
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

    The controller (the only STEP1 call site) builds no contract, so an unguarded close would
    rewrite the verdict artifact from an empty ledger and clobber the worker verdicts.
    """
    try:
        if _arm(side) is None:
            return True
    except Exception as e:  # noqa: BLE001 -- a plan-derivation bug must not break the gate site
        logger.warning("[ISOEXEC-ENFORCE] step1_boundary(%s) plan check skipped: %s", side, e)
        return True
    return close_phase(STEP1, side)
