"""ContractAdapter: the component that systematically drives contract enforcement per runtime.

Before this module, claim checking was bespoke: topology had a hand-called function at two sites,
state claims were attested inside the contract build, the tolerance claim was consumed by the gate,
and nothing iterated the contract's claims AS A SET -- a new claim kind would validate, hash, and
then be silently unchecked at runtime (the "prose claim" failure the design forbids).

Here every claim kind owes a registered ``ClaimChecker``; ``check_all_claims`` iterates EVERY claim
group on the contract, dispatches by kind, and reports each verdict into the obligation ledger.
A claim kind with no registered checker is itself a violation (severity: refuse) -- the runtime
completion of the schema-side rule that a claim without enforcement is unrepresentable.

``ContractAdapter`` owns the per-process enforcement sequence
(``build_contract -> check_all_claims -> install -> close INSTALL``) and exposes the later
boundaries (``on_first_forward``, ``on_weight_sync``) as the thin methods runtime call sites
delegate to. Per-runtime subclasses supply ``runtime_facts()`` and wrap -- never rewrite -- their
existing install sequences (runtimes/vllm/adapter.py, runtimes/megatron/adapter.py).
"""

from __future__ import annotations

import logging
import os
import pathlib
import re
from dataclasses import dataclass
from dataclasses import fields as _dc_fields
from typing import Dict, Optional, Protocol, runtime_checkable

from . import enforce

logger = logging.getLogger(__name__)

_ISOEXEC_DIR = pathlib.Path(__file__).resolve().parents[1]


def _strict() -> bool:
    return os.environ.get("SKYRL_ISOEXEC_MANIFEST_STRICT", "1").lower() not in ("", "0", "false", "no")


@dataclass(frozen=True)
class CheckResult:
    result: str  # enforce.OK | enforce.VIOLATION | enforce.SKIPPED
    evidence: str = ""
    display: str = ""  # banner fragment (topology axes); empty when not applicable


@runtime_checkable
class ClaimChecker(Protocol):
    """One claim kind's runtime enforcement: obligation naming + the judgment itself."""

    kind: str  # the field name on contract.claims this checker owns

    def obligation_id(self, claim) -> str: ...

    def check(self, claim, facts: dict) -> CheckResult: ...


# kind (contract.claims field name) -> checker. Every field of Claims MUST have an entry here or
# check_all_claims refuses; registration is how a new claim kind buys its way into a contract.
CLAIM_CHECKERS: Dict[str, ClaimChecker] = {}


def register_claim_checker(checker: ClaimChecker) -> ClaimChecker:
    CLAIM_CHECKERS[checker.kind] = checker
    return checker


class TopologyChecker:
    """The domain comparison migrated from ``assert_topology_within_claims``, one claim at a time:
    a ``pinned`` claim demands equality with its degree, an ``invariant`` claim membership in its
    proven domain; an axis the caller could not obtain is a visible skip, never a guess."""

    kind = "topology"

    def obligation_id(self, claim) -> str:
        return f"domain_check:{claim.axis}"

    def check(self, t, facts: dict) -> CheckResult:
        side = facts.get("side", "?")
        have = facts.get(t.axis)
        if have is None:
            return CheckResult(enforce.SKIPPED, f"side={side} axis unobtainable")
        have = int(have)
        if t.kind == "pinned":
            display = f"{t.axis}={have}(pinned {t.degree})"
            problem = None if have == int(t.degree) else f"{t.axis}: deployed {have}, contract pins {t.degree}"
        else:
            display = f"{t.axis}={have}(domain {sorted(t.domain)})"
            problem = (
                None
                if have in t.domain
                else f"{t.axis}: deployed {have} is outside the proven domain {sorted(t.domain)} "
                f"(proof: {t.proof})"
            )
        if problem is not None:
            return CheckResult(enforce.VIOLATION, problem, display)
        return CheckResult(enforce.OK, f"side={side} {t.axis}={have}", display)


class StateHookChecker:
    """The hook-exists attestation migrated from the contract build: every StateClaim's
    ``path::symbol`` ref must be real code in-tree."""

    kind = "state"

    def obligation_id(self, claim) -> str:
        return f"hook_exists:{claim.state_id}"

    def check(self, s, facts: dict) -> CheckResult:
        path, _, symbol = s.ref.partition("::")
        f = _ISOEXEC_DIR / path
        if not f.is_file():
            return CheckResult(enforce.VIOLATION, f"ref file {path!r} missing")
        if not re.search(rf"^def {re.escape(symbol)}\(", f.read_text(), re.M):
            return CheckResult(enforce.VIOLATION, f"hook {symbol!r} not defined in {path}")
        return CheckResult(enforce.OK, s.ref)


class ToleranceChecker:
    """Attests the gate WIRING: the forward gate's limits must resolve from this claim
    (``trainer_utils.isoexec_gate_limits``). The STEP-time judgment stays in the gate itself --
    an ``ok`` here is deliberately NOT ledger-reported, so gate@STEP1 still requires the real run."""

    kind = "tolerances"
    report_ok = False  # never pre-discharge the STEP1 gate obligation with an INSTALL-time attest

    def obligation_id(self, claim) -> str:
        return f"gate:{claim.case_pair[0]}|{claim.case_pair[1]}"

    def check(self, t, facts: dict) -> CheckResult:
        try:
            from skyrl.train.utils.trainer_utils import (
                ISOEXEC_FORWARD_GATE_CASE_PAIR,
                isoexec_gate_limits,
            )
        except Exception as e:  # noqa: BLE001 -- a missing gate module is a visible skip
            return CheckResult(enforce.SKIPPED, f"gate module unavailable: {type(e).__name__}: {e}")
        if tuple(t.case_pair) != tuple(ISOEXEC_FORWARD_GATE_CASE_PAIR):
            return CheckResult(
                enforce.VIOLATION,
                f"no gate consumes the tolerance claim for pair {tuple(t.case_pair)}; the only "
                f"wired gate reads {tuple(ISOEXEC_FORWARD_GATE_CASE_PAIR)}",
            )
        try:
            b = dict(t.bounds)
            want = (float(b["abs_diff_mean_max"]), float(b["abs_diff_max_max"]))
        except Exception as e:  # noqa: BLE001 -- malformed bounds cannot wire a gate
            return CheckResult(enforce.VIOLATION, f"claim bounds unusable by the gate: {type(e).__name__}: {e}")
        got = isoexec_gate_limits()
        if tuple(got) != want:
            return CheckResult(
                enforce.VIOLATION,
                f"gate limits {tuple(got)} do not resolve from the claim's bounds {want}",
            )
        return CheckResult(enforce.OK, f"gate limits resolve from the claim: mean_max={want[0]} max_max={want[1]}")


for _c in (TopologyChecker(), StateHookChecker(), ToleranceChecker()):
    register_claim_checker(_c)


def check_topology_claims(contract, actual, *, side: str = "?"):
    """The aggregate topology check (banner + refusal), judgment delegated to ``TopologyChecker``.

    Behavior of the pre-adapter ``assert_topology_within_claims``, preserved exactly: same ledger
    records, same ``[ISOEXEC-CLAIMS]`` banner, same refusal message, same
    SKYRL_ISOEXEC_MANIFEST_STRICT=0 demotion. Returns ``(ok, {obligation_id: CheckResult})``.
    """
    if contract is None:
        logger.warning("[ISOEXEC-CLAIMS] side=%s no contract built; topology unchecked", side)
        return True, {}
    claims = contract.claims.topology
    if not claims:
        logger.warning("[ISOEXEC-CLAIMS] side=%s contract carries no topology claims; nothing to check", side)
        return True, {}
    checker = CLAIM_CHECKERS["topology"]
    facts = dict(actual or {})
    facts["side"] = side
    results: Dict[str, CheckResult] = {}
    checked, unchecked, problems = [], [], []
    for t in claims:
        r = checker.check(t, facts)
        results[checker.obligation_id(t)] = r
        if r.result == enforce.SKIPPED:
            unchecked.append(t.axis)
        else:
            checked.append(r.display)
            if r.result == enforce.VIOLATION:
                problems.append(r.evidence)
        enforce.report(checker.obligation_id(t), enforce.INSTALL, r.result, r.evidence)
    logger.warning(
        "[ISOEXEC-CLAIMS] side=%s checked %s%s violations=%d",
        side,
        " ".join(checked) or "(none)",
        f" unchecked={unchecked}" if unchecked else "",
        len(problems),
    )
    if not problems:
        return True, results
    msg = (
        f"[ISOEXEC-CLAIMS] side={side} deployed topology OUTSIDE the contract's claims: "
        + "; ".join(problems)
        + ". The contract's numerical_policy hashes these claims, so running outside them executes "
        "a composition the identity does not name -- the gate could agree about the wrong "
        "function. Deploy inside the claimed envelope, or extend the profile's TopologyAxisFact "
        "with NEW recorded proof (a different composition: new hash, new proof obligation)."
    )
    if _strict():
        raise RuntimeError(msg)
    logger.error(msg + " (SKYRL_ISOEXEC_MANIFEST_STRICT=0 -> warn-only)")
    return False, results


def check_all_claims(contract, facts, side: str) -> Dict[str, CheckResult]:
    """Iterate EVERY claim on the contract, dispatch each by kind to its registered checker, and
    report each verdict to the ledger under its obligation.

    A claim whose kind has no registered checker REFUSES (strict knob demotes to error-log): this
    is what makes "claim without enforcement" unrepresentable at runtime. Topology keeps its
    aggregate banner + refusal; state and tolerance violations are ledger records the phase closes
    judge under the severity table.
    """
    if contract is None:
        logger.warning("[ISOEXEC-CLAIMS] side=%s no contract built; topology unchecked", side.upper())
        return {}
    facts = dict(facts or {})
    facts.setdefault("side", side.upper())
    results: Dict[str, CheckResult] = {}
    unknown = []
    for f in _dc_fields(type(contract.claims)):
        claims = getattr(contract.claims, f.name)
        checker = CLAIM_CHECKERS.get(f.name)
        if checker is None:
            if claims:
                evidence = f"side={side} {len(claims)} claim(s) of kind {f.name!r} have no registered checker"
                enforce.report(f"claim_check:{f.name}", enforce.INSTALL, enforce.VIOLATION, evidence)
                unknown.append((f.name, len(claims)))
            continue
        if f.name == "topology":
            _ok, tres = check_topology_claims(contract, facts, side=facts["side"])
            results.update(tres)
            continue
        for claim in claims:
            oid = checker.obligation_id(claim)
            try:
                r = checker.check(claim, facts)
            except Exception as e:  # noqa: BLE001 -- a broken checker is a visible skip, never a crash
                r = CheckResult(enforce.SKIPPED, f"checker error: {type(e).__name__}: {e}")
                logger.warning("[ISOEXEC-ADAPTER] %s check skipped: %s", oid, e)
            results[oid] = r
            if r.result == enforce.OK and not getattr(checker, "report_ok", True):
                continue
            enforce.report(oid, enforce.INSTALL, r.result, r.evidence)
    if unknown:
        msg = (
            f"[ISOEXEC-ADAPTER] side={side} contract carries claim kind(s) with NO registered "
            "checker: " + ", ".join(f"{k} ({n} claim(s))" for k, n in unknown) + ". Every claim in "
            "a contract must dispatch to a ClaimChecker; a claim nothing checks is prose the "
            "identity hashes but the runtime never enforces. Register a checker for the kind or "
            "remove the claim."
        )
        if _strict():
            raise RuntimeError(msg)
        logger.error(msg + " (SKYRL_ISOEXEC_MANIFEST_STRICT=0 -> warn-only)")
    return results


class ContractAdapter:
    """Base adapter: owns one process's enforcement sequence against its contract.

    ``run_install()`` drives ``build_contract -> check_all_claims -> install() -> close INSTALL``;
    ``on_first_forward()`` and ``on_weight_sync()`` are the thin boundary methods existing call
    sites delegate to. Subclasses supply ``runtime_facts()`` and ``install()``.
    """

    build_failsoft = False  # trainer builds fail-soft (historical behavior); engine refuses

    def __init__(self, side: str, model_path: Optional[str]):
        if side not in enforce.SIDES:
            raise ValueError(f"unknown side {side!r}; expected one of {sorted(enforce.SIDES)}")
        self.side = side
        self.model_path = model_path
        self.contract = None

    # -- per-runtime obligations --

    def runtime_facts(self) -> dict:
        """Deployed facts of this runtime: ``{TP, PP, CP, SP, world, arch, ...}``."""
        raise NotImplementedError

    def install(self) -> None:
        """Run the runtime's existing install sequence (wrapped, never rewritten)."""
        raise NotImplementedError

    # -- the owned sequence --

    def build_contract(self):
        from .process_contract import get_process_contract

        try:
            self.contract = get_process_contract(self.model_path)
        except Exception as e:  # noqa: BLE001 -- fail-soft only where the pre-adapter path was
            if not self.build_failsoft:
                raise
            logger.warning(f"[ISOEXEC-CONTRACT] {self.side} contract build skipped: {e}")
            self.contract = None
        return self.contract

    def run_install(self, close: bool = True) -> bool:
        """The INSTALL sequence: build the contract BEFORE any install that asserts against it
        (the pik-ordering fix), check every claim, run the install, close the phase."""
        self.build_contract()
        check_all_claims(self.contract, self.runtime_facts(), self.side)
        self.install()
        if close:
            return enforce.install_boundary(self.side)
        return True

    # -- later boundaries --

    def on_first_forward(self) -> bool:
        """Fingerprint compare (once per side tag) + delivered-composition backstop + close."""
        from .enforce import FIRST_FORWARD, ledger

        # Full latch: after the boundary has closed once, later forwards do zero enforcement work.
        if (self.side, FIRST_FORWARD) in ledger().closed:
            return True
        try:
            from .fingerprint import log_fingerprint_once
            from .process_contract import cached_contract_view

            log_fingerprint_once(cached_contract_view(), tag=f"{self.side}_first_forward")
        except Exception as e:  # noqa: BLE001 -- never fatal; the boundary still judges completeness
            logger.warning("[ISOEXEC-ADAPTER] first-forward fingerprint log skipped: %s", e)
        return enforce.first_forward_boundary(self.side)

    def on_weight_sync(self, peer_hash_or_stamp=None) -> bool:
        """Handshake + close; per-runtime subclasses define what the argument is."""
        raise NotImplementedError


_PROCESS_ADAPTER: Optional[ContractAdapter] = None


def set_process_adapter(adapter: ContractAdapter) -> ContractAdapter:
    global _PROCESS_ADAPTER
    _PROCESS_ADAPTER = adapter
    return adapter


def process_adapter() -> Optional[ContractAdapter]:
    """This process's adapter, so sites in other files (weight sync, first forward) reach it."""
    return _PROCESS_ADAPTER


def _reset_for_tests() -> None:
    global _PROCESS_ADAPTER
    _PROCESS_ADAPTER = None
