"""ContractAdapter drives contract enforcement per runtime.

Every claim kind owes a registered ``ClaimChecker``; ``check_all_claims`` iterates EVERY claim
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
                else f"{t.axis}: deployed {have} is outside the proven domain {sorted(t.domain)} " f"(proof: {t.proof})"
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
        from .claim_refs import hook_ref_problem

        problem = hook_ref_problem(s.ref)
        if problem is not None:
            return CheckResult(enforce.VIOLATION, problem)
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
    records, same ``[ISOEXEC-CLAIMS]`` banner, same refusal message; the refusal itself goes
    through ``enforce.refuse``. Returns ``(ok, {obligation_id: CheckResult})``.
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
    enforce.refuse(msg)
    return False, results


def check_all_claims(contract, facts, side: str) -> Dict[str, CheckResult]:
    """Iterate EVERY claim on the contract, dispatch each by kind to its registered checker, and
    report each verdict to the ledger under its obligation.

    A claim whose kind has no registered checker REFUSES (``enforce.refuse``, so strict=0 and debug
    tracing demote it): this is what makes "claim without enforcement" unrepresentable at runtime.
    A checker that RAISES records a CHECKER-ERROR violation, never a skip. Topology keeps its
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
            except Exception as e:  # noqa: BLE001 -- a broken checker never crashes the sequence
                # ...but it fails CLOSED: recording a skip here let a refuse-severity obligation be
                # discharged by a checker that could not run, which is the silent-inert class.
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

    Both runtimes' fingerprint sites record impl ids; the pins the contract hashes alongside them
    were going unrecorded, so ``fingerprint.pin_disagreements`` -- the check that catches "same impl,
    different constant" -- never ran outside tests. The values here come off the object the install
    built (pik's ReductionPlan) or the call-time predicate the kernels themselves read, never off
    the contract, or the comparison would be the declaration checking itself.

    None means the pins are not readable here: either they are call-site literals with no
    post-install state (the attention num_splits/fa_version pair, the logprob fastpath row
    threshold) or the read itself failed. Either way the record carries no pins at all and
    ``log_unreported_pins`` names it, rather than a record echoing the contract. Fail-soft
    deliberately -- this runs inside the install fingerprint, and a pin read that raised would take
    the remaining records down with it.
    """
    try:
        if op == "collectives.tree_all_reduce":
            from ..ops.collectives.pik_tp_invariant import get_plan

            plan = get_plan()
            return {"leaves": int(plan.num_leaves), "leaf_dtype": "bf16" if plan.bf16_leaves else "fp32"}
        if op == "moe.combine":
            # G is the leaf count the fc2 leaf-tree splits by (ops/moe/moe_batched_experts), read
            # from the same env those kernels read; fp32 leaves are the impl -- pik_leaf_tree IS
            # the fp32 tree, there is no setting that makes it anything else.
            return {"leaves": int(os.environ.get("SKYRL_ISOEXEC_PIK_LEAVES", "8")), "leaf_dtype": "fp32"}
        if op == "logprobs.log_softmax":
            from ..ops.logprobs import rowinv

            # G off the same env the kernel reads, BLOCK off the module the install bound -- so a
            # kernel whose tile drifted from the contract's pin is a recorded disagreement, not a
            # silently rehashed schedule. Kahan fp32 is the impl: rowinv_leaftree IS the Kahan
            # tree, there is no setting that makes it anything else.
            return {
                "leaves": int(os.environ.get(rowinv.LEAVES_ENV, "8")),
                "block": int(rowinv.BLOCK),
                "accum": "kahan_fp32",
            }
    except Exception as e:  # noqa: BLE001 -- an unreadable pin is a blind spot, never a failed install
        logger.warning("[ISOEXEC-FINGERPRINT] live pins for %s unreadable (%s: %s)", op, type(e).__name__, e)
    return None


def log_unreported_pins(view) -> Dict[str, list]:
    """Name the recorded installs whose contract entry pins constants that the record does not carry.

    ``log_fingerprint`` compares pins only when the recorder reported some, because an install that
    reports none is a blind spot rather than evidence of agreement. That is the right rule and it is
    also silent, so the blind spots never appear anywhere. This says them out loud: a site listed
    here runs ``pin_disagreements`` over nothing, and the fix is to source the value off the object
    the install bound -- not to echo the contract back at itself.

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

    ``run_install()`` drives ``build_contract -> check_all_claims -> install() -> close INSTALL``;
    ``on_first_forward()`` and ``on_weight_sync()`` are the thin boundary methods existing call
    sites delegate to. Subclasses supply ``runtime_facts()`` and ``install()``.
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
        """The INSTALL sequence: build the contract BEFORE any install that asserts against it
        (the pik-ordering fix), check every claim, run the install, close the phase."""
        self.build_contract()
        check_all_claims(self.contract, self.runtime_facts(), self.side)
        self.install()
        self._install_debug_trace()
        if close:
            return enforce.install_boundary(self.side)
        return True

    #: Debug tracing digests region outputs, which ends in a ``.item()`` D2H copy. Under CUDA
    #: graph capture that poisons the capture, and a replayed graph runs no Python at all, so the
    #: decode steps the graph serves produce no records either way -- a trace that looks clean
    #: because half the forward was never observed. Refused at init instead.
    CUDAGRAPH_REFUSAL = (
        "[ISOEXEC-DEBUG] REFUSED: {side} debug tracing (SKYRL_ISOEXEC_DEBUG_TRACE={dir!r}) with "
        "SKYRL_ISOEXEC_ENABLE_CUDAGRAPH=1. A replayed CUDA graph executes no Python, so every "
        "decode step served from a captured graph would silently produce NO trace records, and a "
        "hook reached during capture would poison the capture with the digest's device-to-host "
        "copy. Fix: unset SKYRL_ISOEXEC_ENABLE_CUDAGRAPH (the default, enforce_eager=True) for "
        "debug runs, or unset SKYRL_ISOEXEC_DEBUG_TRACE to run for throughput."
    )

    def _install_debug_trace(self) -> None:
        """Debug mode (SKYRL_ISOEXEC_DEBUG_TRACE): per-region output tracing. Every enforcement
        refusal is demoted (``enforce.demoted``) so any kernel mix runs; the trace, not the gate,
        reports the disagreement.

        Fail-soft like every other diagnostic in this package: tracing is DIAGNOSTIC disposition,
        so a hook-install failure must not be able to abort the INSTALL sequence that follows it.

        ONE exception, raised before anything is installed: debug tracing on the engine together
        with CUDA-graph decode. It does NOT go through ``enforce.refuse``, deliberately --
        ``enforce.demoted()`` is true exactly when debug tracing is armed, so a refusal whose own
        precondition is debug mode would demote itself to a log line every single time. It is
        also not the failure mode fail-soft exists for: the run would not crash, it would produce
        a trace that reads clean because the decode half of the forward was never observed. That
        is worse than no trace, so it is a hard error with its own message.
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

            # A subclass that keeps its GPTModel handle gets layer-indexed records for free;
            # without one the trainer falls back to layer_src="call_order" (see debug/trace.py).
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
