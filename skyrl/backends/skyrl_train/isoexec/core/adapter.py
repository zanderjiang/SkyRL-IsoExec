"""ContractAdapter drives contract enforcement per runtime.

Owns the per-process sequence (``build_contract -> check_all_claims -> install -> close INSTALL``)
plus the later boundaries. Every claim kind owes a registered ``ClaimChecker`` or the run refuses.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from dataclasses import fields as _dc_fields
from typing import Dict, Optional, Protocol, runtime_checkable

from . import enforce

logger = logging.getLogger(__name__)


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
# check_all_claims refuses.
CLAIM_CHECKERS: Dict[str, ClaimChecker] = {}


def register_claim_checker(checker: ClaimChecker) -> ClaimChecker:
    CLAIM_CHECKERS[checker.kind] = checker
    return checker


class TopologyChecker:
    """A ``pinned`` claim demands equality with its degree, an ``invariant`` claim membership in
    its proven domain; an axis the caller could not obtain is a visible skip, never a guess."""

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
                else f"{t.axis}: deployed {have} is outside the proven domain {sorted(t.domain)} " f"(proof: {t.proof})"
            )
        if problem is not None:
            return CheckResult(enforce.VIOLATION, problem, display)
        return CheckResult(enforce.OK, f"side={side} {t.axis}={have}", display)


class StateHookChecker:
    """Every StateClaim's ``path::symbol`` ref must be real code in-tree."""

    kind = "state"

    def obligation_id(self, claim) -> str:
        return f"hook_exists:{claim.state_id}"

    def check(self, s, facts: dict) -> CheckResult:
        from .claim_refs import hook_ref_problem

        problem = hook_ref_problem(s.ref)
        if problem is not None:
            return CheckResult(enforce.VIOLATION, problem)
        return CheckResult(enforce.OK, s.ref)


class ToleranceChecker:
    """Attests the gate WIRING: the forward gate's limits must resolve from this claim. An ``ok``
    is deliberately NOT ledger-reported, so gate@STEP1 still requires the real run."""

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
    """The aggregate topology check (banner + refusal); returns ``(ok, {obligation_id: result})``."""
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
    enforce.refuse(msg)
    return False, results


def check_all_claims(contract, facts, side: str) -> Dict[str, CheckResult]:
    """Dispatch every claim to its registered checker and report each verdict to the ledger.

    A claim kind with no registered checker refuses; a checker that RAISES records a
    CHECKER-ERROR violation, never a skip.
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
            except Exception as e:  # noqa: BLE001 -- a broken checker never crashes the sequence
                # ...but it fails CLOSED: a skip here would let a refuse-severity obligation be
                # discharged by a checker that could not run.
                r = CheckResult(enforce.VIOLATION, f"{enforce.CHECKER_ERROR}: {type(e).__name__}: {e}")
                logger.error("[ISOEXEC-ADAPTER] %s checker RAISED, recorded as a violation: %s", oid, e)
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
        enforce.refuse(msg)
    return results


def live_pins(op: str) -> Optional[dict]:
    """The pinned constants ``op``'s install can still be READ for, or None when it cannot.

    Values come off the object the install built or what the kernels themselves read, never off
    the contract, or the comparison would be the declaration checking itself. Fail-soft.
    """
    try:
        if op == "collectives.tree_all_reduce":
            from ..ops.collectives.pik_tp_invariant import get_plan

            plan = get_plan()
            return {"leaves": int(plan.num_leaves), "leaf_dtype": "bf16" if plan.bf16_leaves else "fp32"}
        if op == "moe.combine":
            # Leaf count read from the same env the fc2 leaf-tree kernels read; fp32 leaves are
            # the impl itself, not a setting.
            return {"leaves": int(os.environ.get("SKYRL_ISOEXEC_PIK_LEAVES", "8")), "leaf_dtype": "fp32"}
        if op == "logprobs.log_softmax":
            from ..ops.logprobs import rowinv

            # Leaves off the same env the kernel reads, BLOCK off the module the install bound, so
            # a drifted tile is a recorded disagreement. Kahan fp32 is the impl, not a setting.
            return {
                "leaves": int(os.environ.get(rowinv.LEAVES_ENV, "8")),
                "block": int(rowinv.BLOCK),
                "accum": "kahan_fp32",
            }
    except Exception as e:  # noqa: BLE001 -- an unreadable pin is a blind spot, never a failed install
        logger.warning("[ISOEXEC-FINGERPRINT] live pins for %s unreadable (%s: %s)", op, type(e).__name__, e)
    return None


def log_unreported_pins(view) -> Dict[str, list]:
    """Name the recorded installs whose contract entry pins constants the record does not carry.

    Returns ``{f"{op}:{site}": [pinned keys]}``; empty when every pinned entry reported its pins.
    """
    from .fingerprint import recorder

    entries = view or {}
    gaps = {}
    for key, got in sorted(recorder().installed.items()):
        want = entries.get(key)
        if not want or not want.get("pinned_constants") or "pinned_constants" in got:
            continue
        gaps[f"{key[0]}:{key[1]}"] = sorted(want["pinned_constants"])
    if gaps:
        logger.warning(
            "[ISOEXEC-FINGERPRINT] %d install(s) report NO pins while the contract pins some, so "
            "those pins are unverified in this process: %s",
            len(gaps),
            gaps,
        )
    return gaps


class ContractAdapter:
    """Base adapter: owns one process's enforcement sequence against its contract.

    Subclasses supply ``runtime_facts()`` and ``install()``.
    """

    build_failsoft = False  # trainer builds fail-soft (historical behavior); engine refuses

    def __init__(self, side: str, model_path: Optional[str], model_fn=None):
        if side not in enforce.SIDES:
            raise ValueError(f"unknown side {side!r}; expected one of {sorted(enforce.SIDES)}")
        self.side = side
        self.model_fn = model_fn
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
        """The INSTALL sequence; the contract is built BEFORE any install that asserts against it."""
        self.build_contract()
        check_all_claims(self.contract, self.runtime_facts(), self.side)
        self.install()
        self._install_debug_trace()
        if close:
            return enforce.install_boundary(self.side)
        return True

    #: Debug tracing's ``.item()`` D2H copy poisons a CUDA-graph capture, and a replayed graph runs
    #: no Python, so decode steps produce no records at all. Refused at init instead.
    CUDAGRAPH_REFUSAL = (
        "[ISOEXEC-DEBUG] REFUSED: {side} debug tracing (SKYRL_ISOEXEC_DEBUG_TRACE={dir!r}) with "
        "SKYRL_ISOEXEC_ENABLE_CUDAGRAPH=1. A replayed CUDA graph executes no Python, so every "
        "decode step served from a captured graph would silently produce NO trace records, and a "
        "hook reached during capture would poison the capture with the digest's device-to-host "
        "copy. Fix: unset SKYRL_ISOEXEC_ENABLE_CUDAGRAPH (the default, enforce_eager=True) for "
        "debug runs, or unset SKYRL_ISOEXEC_DEBUG_TRACE to run for throughput."
    )

    def _install_debug_trace(self) -> None:
        """Arm per-region output tracing under SKYRL_ISOEXEC_DEBUG_TRACE; fail-soft.

        Engine-side tracing with CUDA-graph decode is a hard error rather than an
        ``enforce.refuse``: debug mode demotes every refusal, so this one would demote itself.
        """
        from ..debug.trace import enabled

        if not enabled():
            return
        if self.side == "engine" and os.environ.get("SKYRL_ISOEXEC_ENABLE_CUDAGRAPH") == "1":
            raise RuntimeError(
                self.CUDAGRAPH_REFUSAL.format(side=self.side, dir=os.environ.get("SKYRL_ISOEXEC_DEBUG_TRACE"))
            )
        try:
            os.environ["SKYRL_ISOEXEC_DEBUG_SIDE"] = self.side
            from ..debug import install_debug_hooks

            # With a GPTModel handle the records are layer-indexed; without one the trainer falls
            # back to layer_src="call_order".
            mf = getattr(self, "model_fn", None)
            n = install_debug_hooks(mf() if mf is not None else None)
        except Exception as e:  # noqa: BLE001 -- diagnostics never fail a run
            logger.warning("[ISOEXEC-DEBUG] %s tracing NOT armed: %s: %s", self.side, type(e).__name__, e)
            return
        logger.warning("[ISOEXEC-DEBUG] %s tracing armed: %d region hooks (enforcement demoted)", self.side, n)

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
