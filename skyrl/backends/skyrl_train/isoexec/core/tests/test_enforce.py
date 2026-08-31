"""The obligation ledger: plan derivation, completeness at boundaries, severity, exceptions,
the verdict artifact, and the install-attestation handshake extension.

The negative control: a reporter that never runs must surface as a ``missing`` violation at the
phase close, so a check that never ran is distinguishable from a passing one.
"""

import json
import os
import tempfile
from collections import Counter

from skyrl.backends.skyrl_train.isoexec.core import enforce
from skyrl.backends.skyrl_train.isoexec.core import fingerprint as fp
from skyrl.backends.skyrl_train.isoexec.core import process_contract as pc
from skyrl.backends.skyrl_train.isoexec.core.process_contract import build_contract_view
from skyrl.backends.skyrl_train.isoexec.core.registry_build import build_registry
from skyrl.backends.skyrl_train.isoexec.debug import trace
from skyrl.backends.skyrl_train.isoexec.models import qwen3_5

STRICT_ENV = "SKYRL_ISOEXEC_MANIFEST_STRICT"

_REG = build_registry(strict=True)


def _clean_build():
    # Build under a stripped env so the derivation reflects code defaults, not the caller's shell.
    saved = {k: v for k, v in os.environ.items() if k.startswith("SKYRL_ISOEXEC")}
    for k in saved:
        del os.environ[k]
    try:
        return qwen3_5.build(_REG, arch="sm90", profile=qwen3_5.PROFILE)
    finally:
        os.environ.update(saved)


_CONTRACT = _clean_build()


class _fresh:
    """Reset ledger + fingerprint recorder + cached contract around one test."""

    def __enter__(self):
        self._saved = (pc._CONTRACT, pc._VIEW, fp._RECORDER, set(fp._LOGGED_TAGS), os.environ.get(STRICT_ENV))
        enforce._reset_for_tests()
        fp._RECORDER = None
        fp._LOGGED_TAGS.clear()
        os.environ.pop(STRICT_ENV, None)
        pc._CONTRACT, pc._VIEW = _CONTRACT, build_contract_view(_CONTRACT, _REG)
        return self

    def __exit__(self, *exc):
        enforce._reset_for_tests()
        pc._CONTRACT, pc._VIEW, fp._RECORDER = self._saved[:3]
        fp._LOGGED_TAGS.clear()
        fp._LOGGED_TAGS.update(self._saved[3])
        if self._saved[4] is None:
            os.environ.pop(STRICT_ENV, None)
        else:
            os.environ[STRICT_ENV] = self._saved[4]
        return False


def _refuses(fn, *a, **kw) -> str:
    try:
        fn(*a, **kw)
    except (RuntimeError, ValueError) as e:  # ContractDeliveryError is a ValueError
        return str(e)
    raise AssertionError(f"{getattr(fn, '__name__', fn)} should have refused")


def _report_phase(plan, phase, result="ok"):
    for ob in plan.obligations:
        if ob.phase == phase:
            enforce.report(ob.obligation_id, phase, result, "test")


def test_plan_derivation_complete_and_deterministic():
    # The trainer's extra key is the training forward's logprob case, which rowinv covers.
    for side, n_keys in (("trainer", 30), ("engine", 40)):
        p1 = enforce.derive_obligation_plan(_CONTRACT, _REG, side)
        p2 = enforce.derive_obligation_plan(_CONTRACT, _REG, side)
        assert p1 == p2, f"{side}: plan derivation must be deterministic"
        cnt = Counter((o.kind, o.phase) for o in p1.obligations)
        # Every composition entry visible on the side owes install_attest + fingerprint...
        assert cnt[(enforce.INSTALL_ATTEST, enforce.INSTALL)] == n_keys
        assert cnt[(enforce.FINGERPRINT, enforce.FIRST_FORWARD)] == n_keys
        # ...every claim its check, the identities their handshake, the contract its build + backstop.
        assert cnt[(enforce.DOMAIN_CHECK, enforce.INSTALL)] == len(_CONTRACT.claims.topology) == 4
        assert cnt[(enforce.HOOK_EXISTS, enforce.INSTALL)] == len(_CONTRACT.claims.state) == 2
        assert cnt[(enforce.GATE, enforce.STEP1)] == len(_CONTRACT.claims.tolerances) == 1
        assert cnt[(enforce.HANDSHAKE, enforce.WEIGHT_SYNC)] == 1
        assert cnt[(enforce.BUILD_VALID, enforce.INSTALL)] == 1
        assert cnt[(enforce.INSTALLED_BACKSTOP, enforce.FIRST_FORWARD)] == 1
        # served only where the SERVED_COUNTERS inventory names the (op, impl); the rest are listed.
        assert cnt[(enforce.SERVED, enforce.STEP1)] + len(p1.no_served_counter) == n_keys
        for op, _site, impl in p1.no_served_counter:
            assert (op, impl) not in enforce.SERVED_COUNTERS


def test_plan_lists_counterless_entries():
    p = enforce.derive_obligation_plan(_CONTRACT, _REG, "engine")
    listed = {(op, impl) for op, _s, impl in p.no_served_counter}
    # The inversion: the strongest-obligation engine impls are the ones with no counter.
    assert ("moe.experts", "fused") in listed
    assert ("moe.epilogue", "fused_swiglu") in listed
    # And the counter-bearing ones owe served@STEP1.
    served = {o.obligation_id for o in p.obligations if o.kind == enforce.SERVED}
    assert "served:moe.dispatch:engine_prefill" in served
    assert "served:moe.experts:engine_prefill" not in served


def test_suppressed_reporter_is_a_missing_violation():
    """The negative control of the architecture: a required check that never ran refuses at close."""
    with _fresh():
        plan = enforce.derive_obligation_plan(_CONTRACT, _REG, "trainer")
        for ob in plan.obligations:
            if ob.phase == enforce.INSTALL and ob.obligation_id != "install_attest:mm:trainer_fwd":
                enforce.report(ob.obligation_id, enforce.INSTALL, "ok", "test")
        msg = _refuses(enforce.close_phase, enforce.INSTALL, "trainer")
        assert "install_attest:mm:trainer_fwd" in msg and "missing" in msg
        assert enforce.verdict_counts()["missing"] == 1
    with _fresh():
        os.environ[STRICT_ENV] = "0"  # warn-only demotion, mirroring the handshake knob
        assert enforce.close_phase(enforce.INSTALL, "trainer") is False


def test_severity_table_honored_both_directions():
    with _fresh():
        plan = enforce.derive_obligation_plan(_CONTRACT, _REG, "engine")
        # Function-half violation at a refuse phase refuses...
        _report_phase(plan, enforce.INSTALL)
        enforce.report("install_attest:moe.experts:engine_prefill", enforce.INSTALL, "violation", "test violation")
        msg = _refuses(enforce.close_phase, enforce.INSTALL, "engine")
        assert "install_attest:moe.experts:engine_prefill" in msg
    with _fresh():
        # ...while the deployment-half (engine nccl_pin under code defaults) demotes to log.
        plan = enforce.derive_obligation_plan(_CONTRACT, _REG, "engine")
        dep = [o for o in plan.obligations if o.half == "deployment" and o.phase == enforce.INSTALL]
        assert dep, "code-default engine contract must carry a deployment-half entry"
        assert all(enforce.severity(o) == enforce.LOG for o in dep)
        _report_phase(plan, enforce.INSTALL)
        enforce.report(dep[0].obligation_id, enforce.INSTALL, "violation", "test violation")
        assert enforce.close_phase(enforce.INSTALL, "engine") is True  # logged, not refused
        assert enforce.verdict_counts()["logged"] == 1
    with _fresh():
        # served@STEP1 is log by table even for function-half entries.
        plan = enforce.derive_obligation_plan(_CONTRACT, _REG, "trainer")
        _report_phase(plan, enforce.STEP1)
        served = next(o for o in plan.obligations if o.kind == enforce.SERVED and o.half == "function")
        enforce.report(served.obligation_id, enforce.STEP1, "violation", "counter says 0 served")
        assert enforce.close_phase(enforce.STEP1, "trainer") is True


def test_exceptions_suppress_exactly_their_target():
    with _fresh():
        plan = enforce.derive_obligation_plan(_CONTRACT, _REG, "engine")
        # gdn.state's INSTALL attestation is EXCEPTIONS-listed (recorded at first forward)...
        assert enforce.exemption_for("install_attest:gdn.state:engine_prefill") is not None
        # ...and its fingerprint deliberately is NOT: the declaration was fixed, mismatches refuse.
        assert enforce.exemption_for("fingerprint:gdn.state:engine_prefill") is None
        for ob in plan.obligations:
            if ob.phase == enforce.INSTALL and not ob.obligation_id.startswith("install_attest:gdn.state:"):
                enforce.report(ob.obligation_id, enforce.INSTALL, "ok", "test")
        # Both gdn.state attest obligations missing -> excepted, not refused.
        assert enforce.close_phase(enforce.INSTALL, "engine") is True
        assert enforce.verdict_counts()["excepted"] == 2
    with _fresh():
        # The exception must not leak: the same miss on a NON-matched obligation still refuses.
        plan = enforce.derive_obligation_plan(_CONTRACT, _REG, "engine")
        for ob in plan.obligations:
            if ob.phase == enforce.INSTALL and ob.obligation_id != "install_attest:gdn.core:engine_prefill":
                enforce.report(ob.obligation_id, enforce.INSTALL, "ok", "test")
        msg = _refuses(enforce.close_phase, enforce.INSTALL, "engine")
        assert "install_attest:gdn.core:engine_prefill" in msg
    # Every seeded exception carries its reason and removal condition.
    for ex in enforce.EXCEPTIONS:
        assert ex.reason and ex.removal, f"exception {ex.pattern} owes a reason and a removal condition"


def test_fingerprint_reporters_feed_the_ledger():
    with _fresh():
        fp.record_install("moe.router", "trainer_fwd", "deterministic")
        recs = enforce.ledger().records.get("install_attest:moe.router:trainer_fwd")
        assert recs and recs[0].result == "ok" and recs[0].evidence == "deterministic"
        # A comparator mismatch is BOTH the existing log line and a ledger violation.
        fp.record_install("moe.experts", "trainer_fwd", "fused")  # contract says batched_bmm
        problems = fp.log_fingerprint(pc.cached_contract_view(), tag="test_install")
        assert any("moe.experts" in p for p in problems), "check logic unchanged: problem list still returned"
        recs = enforce.ledger().records.get("fingerprint:moe.experts:trainer_fwd")
        assert recs and recs[-1].result == "violation"
        recs = enforce.ledger().records.get("fingerprint:moe.router:trainer_fwd")
        assert recs and recs[-1].result == "ok"


def test_topology_reporter_feeds_domain_check():
    with _fresh():
        assert pc.assert_topology_within_claims(_CONTRACT, {"TP": 4, "SP": 1, "PP": 1, "CP": 1}, side="TRAINER")
        led = enforce.ledger().records
        assert led["domain_check:TP"][-1].result == "ok"
        assert led["domain_check:PP"][-1].result == "ok"
    with _fresh():
        os.environ[STRICT_ENV] = "0"
        assert pc.assert_topology_within_claims(_CONTRACT, {"TP": 16}, side="ENGINE") is False
        led = enforce.ledger().records
        assert led["domain_check:TP"][-1].result == "violation"
        assert led["domain_check:PP"][-1].result == "skipped"  # unobtainable axis: visible, not silent


def test_handshake_reporter_and_weight_sync_close():
    with _fresh():
        h = pc.contract_hash()
        assert pc.assert_contract_agreement(h, other_side="peer") is True
        assert enforce.ledger().records["handshake:numerical_policy"][-1].result == "ok"
        assert enforce.close_phase(enforce.WEIGHT_SYNC, "engine") is True
    with _fresh():
        # Handshake that never ran: the WEIGHT_SYNC close is the completeness check.
        msg = _refuses(enforce.close_phase, enforce.WEIGHT_SYNC, "engine")
        assert "handshake:numerical_policy" in msg and "missing" in msg
    with _fresh():
        os.environ[STRICT_ENV] = "0"
        assert pc.assert_contract_agreement("0" * 64) is False
        assert enforce.ledger().records["handshake:numerical_policy"][-1].result == "violation"


def test_installed_backstop_refuses_missing_key():
    with _fresh():
        # Complete trainer install -> the backstop passes.
        for op, site in pc.cached_contract_view():
            if site.startswith("trainer"):
                fp.record_install(op, site, "x")
        enforce.report_installed_backstop("trainer")
        assert enforce.ledger().records["installed_backstop:first_forward"][-1].result == "ok"
    with _fresh():
        # Drop one contract-named key: refuse (severity refuse), and the violation is recorded.
        keys = [k for k in pc.cached_contract_view() if k[1].startswith("trainer")]
        for op, site in keys[1:]:
            fp.record_install(op, site, "x")
        msg = _refuses(enforce.report_installed_backstop, "trainer")
        assert "NOT installed" in msg
        assert enforce.ledger().records["installed_backstop:first_forward"][-1].result == "violation"


def test_gate_reporter():
    from skyrl.train.utils import trainer_utils as tu

    with _fresh():
        P = tu.ISOEXEC_FORWARD_GATE_PREFIX
        good = {f"{P}_mean": 1e-6, f"{P}_max": 1e-5, f"{P}_min": 0.0, f"{P}_std": 0.0}
        assert tu.validate_isoexec_forward_gate(good, enabled=True, scoring_audit_skipped=False) is True
        recs = enforce.ledger().records["gate:engine_decode|trainer_score"]
        assert recs[-1].result == "ok"
        # rowinv's served obligations are REFUSE-severity and must be reported before STEP1 can
        # close (test_rowinv_engagement.py owns the refusal path itself).
        for case in ("trainer_fwd", "trainer_score"):
            enforce.report(f"served:logprobs.log_softmax:{case}", enforce.STEP1, enforce.OK, "test census")
        assert enforce.close_phase(enforce.STEP1, "trainer") is True
    with _fresh():
        P = tu.ISOEXEC_FORWARD_GATE_PREFIX
        bad = {f"{P}_mean": 1e-2, f"{P}_max": 1e-1, f"{P}_min": 0.0, f"{P}_std": 0.0}
        try:
            tu.validate_isoexec_forward_gate(bad, enabled=True, scoring_audit_skipped=False)
            raise AssertionError("red gate must refuse")
        except RuntimeError:
            pass
        assert enforce.ledger().records["gate:engine_decode|trainer_score"][-1].result == "violation"
        # A recorded gate violation refuses the STEP1 close too (severity table, not the call site).
        _refuses(enforce.close_phase, enforce.STEP1, "trainer")


def test_enforcement_json_round_trips():
    with _fresh():
        with tempfile.TemporaryDirectory() as d:
            os.environ["ISOEXEC_CONTRACT_PATH"] = os.path.join(d, "contract.json")
            try:
                plan = enforce.derive_obligation_plan(_CONTRACT, _REG, "trainer")
                enforce.ledger().plans["trainer"] = plan
                _report_phase(plan, enforce.INSTALL)
                assert enforce.close_phase(enforce.INSTALL, "trainer") is True
                (out,) = enforce.verdict_artifacts(d)
                assert os.path.basename(out).startswith("enforcement.trainer.r")
                with open(out) as fh:
                    loaded = json.load(fh)
                assert loaded == json.loads(json.dumps(enforce.verdict(), sort_keys=True))
                assert loaded["counts"]["missing"] == 0 and loaded["counts"]["refused"] == 0
                ids = {o["id"] for o in loaded["obligations"]}
                assert "build_valid:contract" in ids and "handshake:numerical_policy" in ids
                assert loaded["exceptions"] and all(e["reason"] and e["removal"] for e in loaded["exceptions"])
            finally:
                os.environ.pop("ISOEXEC_CONTRACT_PATH", None)


def test_attestation_digest_changes_composite_and_detects_divergence():
    with _fresh():
        base = _CONTRACT.identities.numerical_policy
        saved = dict(pc._EXTENSIONS)
        pc._EXTENSIONS.clear()
        try:
            without = pc.composite_hash(base)
            pc.register_contract_extension("install_attestation", enforce.install_attestation_digest)
            with_ext = pc.composite_hash(base)
            assert with_ext != without, "the attestation extension must fold into the composite"

            # Two simulated processes, identical clean installs: symmetric by construction (CLEAN).
            enforce._reset_for_tests()
            fp.record_install("moe.router", "trainer_fwd", "deterministic")
            fp.log_fingerprint(pc.cached_contract_view(), tag="a_install")
            assert enforce.install_attestation_digest() == "CLEAN"
            h_trainer = pc.composite_hash(base)

            enforce._reset_for_tests()
            fp._RECORDER = None
            fp.record_install("moe.router", "engine_prefill", "fused_o2")  # its own side, same verdict
            fp.log_fingerprint(pc.cached_contract_view(), tag="b_install")
            assert enforce.install_attestation_digest() == "CLEAN"
            assert pc.composite_hash(base) == h_trainer, "identical installs must agree at weight sync"

            # Third process declared the same contract but INSTALLED differently: refuses at sync.
            enforce._reset_for_tests()
            fp._RECORDER = None
            fp.record_install("moe.router", "engine_prefill", "deterministic")  # contract says fused_o2
            fp.log_fingerprint(pc.cached_contract_view(), tag="c_install")
            digest = enforce.install_attestation_digest()
            assert digest != "CLEAN" and "sha256" in digest
            assert pc.composite_hash(base) != h_trainer, "a diverged install must break the handshake"
            # An EXCEPTED self-attesting record must NOT perturb the digest (exactly its target).
            enforce._reset_for_tests()
            fp._RECORDER = None
            fp.record_install("moe.experts", "engine_prefill", "wrong")  # excepted pattern
            fp.log_fingerprint(pc.cached_contract_view(), tag="d_install")
            assert enforce.install_attestation_digest() == "CLEAN"
        finally:
            pc._EXTENSIONS.clear()
            pc._EXTENSIONS.update(saved)


def test_contract_path_is_a_registered_flag():
    # An env var that is not a registered Flag reaches no actor.
    from skyrl.backends.skyrl_train.isoexec.core import flags

    f = flags.get("ISOEXEC_CONTRACT_PATH")
    assert f.disposition == flags.DIAGNOSTIC and f.sides == ("both",)
    assert "ISOEXEC_CONTRACT_PATH" in flags.actor_forwarding_tuple(flags.TRAIN)
    assert "ISOEXEC_CONTRACT_PATH" in flags.actor_forwarding_tuple(flags.ENGINE)


def test_phantom_entry_is_gone():
    # An identity-hashed entry with no impl, installer, or check anywhere.
    assert not _REG.has_op("attention.qwen35_context_layout")
    # 28: the two aten-order logprob entries collapsed into one rowinv entry covering four sites.
    assert len(_CONTRACT.composition) == 28
    assert all("attention.qwen35_context_layout" not in e.region for e in _CONTRACT.composition)


# -- demotion, unrecognized skips, and the states the obligation table cannot express -------------

DEBUG_ENV = "SKYRL_ISOEXEC_DEBUG_TRACE"


class _debug_trace:
    """Arm debug tracing (its own demotion condition) around one test."""

    def __init__(self, path="/tmp/isoexec-test-trace"):
        self.path = path

    def __enter__(self):
        self._saved = os.environ.get(DEBUG_ENV)
        os.environ[DEBUG_ENV] = self.path
        return self

    def __exit__(self, *exc):
        if self._saved is None:
            os.environ.pop(DEBUG_ENV, None)
        else:
            os.environ[DEBUG_ENV] = self._saved
        return False


def _install_plan(side="trainer"):
    return enforce.derive_obligation_plan(_CONTRACT, _REG, side)


def test_debug_trace_demotes_the_boundary_refusal_without_weakening_the_ledger():
    # Strict stays 1, yet a red INSTALL close returns False instead of raising -- and the
    # violation is recorded exactly as it would have been in strict mode.
    with _fresh():
        os.environ[STRICT_ENV] = "1"
        plan = _install_plan()
        _report_phase(plan, enforce.INSTALL, enforce.OK)
        enforce.report("build_valid:contract", enforce.INSTALL, enforce.VIOLATION, "injected")
        with _debug_trace():
            assert enforce.install_boundary("trainer") is False
        assert enforce.verdict_counts()["refused"] == 1
        ob = next(o for o in plan.obligations if o.obligation_id == "build_valid:contract")
        assert enforce.ledger().judged_status(ob) == enforce.VIOLATION


def test_strict_still_refuses_the_same_boundary_without_debug_trace():
    with _fresh():
        os.environ[STRICT_ENV] = "1"
        os.environ.pop(DEBUG_ENV, None)
        plan = _install_plan()
        _report_phase(plan, enforce.INSTALL, enforce.OK)
        enforce.report("build_valid:contract", enforce.INSTALL, enforce.VIOLATION, "injected")
        assert "build_valid:contract (violation)" in _refuses(enforce.install_boundary, "trainer")


def test_all_skipped_refuse_obligation_does_not_close_green():
    # An unexplained skip discharges nothing -- silent inertness via ``skipped`` rather than absence.
    with _fresh():
        os.environ[STRICT_ENV] = "1"
        os.environ.pop(DEBUG_ENV, None)
        plan = _install_plan()
        _report_phase(plan, enforce.INSTALL, enforce.SKIPPED)
        msg = _refuses(enforce.install_boundary, "trainer")
        assert "build_valid:contract (unchecked)" in msg
        assert enforce.verdict_counts()["refused"] > 0


def test_recognized_skip_reason_still_discharges_its_obligation():
    # ...and the whitelist is by reason, not blanket: the peer stamping nothing is a real answer.
    with _fresh():
        os.environ[STRICT_ENV] = "1"
        os.environ.pop(DEBUG_ENV, None)
        enforce.report(
            "handshake:numerical_policy",
            enforce.WEIGHT_SYNC,
            enforce.SKIPPED,
            f"{enforce.SKIP_NO_PEER_STAMP}: trainer stamped no contract_hash",
        )
        assert enforce.weight_sync_boundary("trainer") is True


def test_unrecognized_skip_reason_on_the_same_obligation_refuses():
    with _fresh():
        os.environ[STRICT_ENV] = "1"
        os.environ.pop(DEBUG_ENV, None)
        enforce.report("handshake:numerical_policy", enforce.WEIGHT_SYNC, enforce.SKIPPED, "felt like it")
        assert "handshake:numerical_policy (unchecked)" in _refuses(enforce.weight_sync_boundary, "trainer")


def test_zero_runtime_facts_cannot_close_install_clean():
    # An adapter that obtains no topology fact leaves every domain_check a skip; an axis nobody
    # could read is not a recognized skip reason.
    from skyrl.backends.skyrl_train.isoexec.core import adapter as ad

    with _fresh():
        os.environ[STRICT_ENV] = "1"
        os.environ.pop(DEBUG_ENV, None)
        plan = _install_plan()
        _report_phase(plan, enforce.INSTALL, enforce.OK)
        for ob in plan.obligations:
            if ob.kind == enforce.DOMAIN_CHECK:
                enforce.ledger().records.pop(ob.obligation_id, None)
        ad.check_all_claims(_CONTRACT, {}, "trainer")  # no facts at all
        msg = _refuses(enforce.install_boundary, "trainer")
        assert "domain_check:TP (unchecked)" in msg


def test_checker_error_is_recorded_as_a_violation():
    # A checker that RAISES fails closed.
    from skyrl.backends.skyrl_train.isoexec.core import adapter as ad

    with _fresh():
        os.environ[STRICT_ENV] = "1"
        os.environ.pop(DEBUG_ENV, None)
        chk = ad.CLAIM_CHECKERS["state"]
        orig = type(chk).check
        type(chk).check = lambda self, c, f: (_ for _ in ()).throw(RuntimeError("checker exploded"))
        try:
            ad.check_all_claims(_CONTRACT, {"TP": 8, "PP": 1, "CP": 1, "SP": 0}, "engine")
        finally:
            type(chk).check = orig
        recs = [r for oid, rs in enforce.ledger().records.items() if oid.startswith("hook_exists:") for r in rs]
        assert recs and all(r.result == enforce.VIOLATION for r in recs)
        assert all(r.evidence.startswith(enforce.CHECKER_ERROR) for r in recs)


def test_stray_obligation_id_is_visible_in_the_verdict():
    # O1: it already broke the attestation digest; the artifact must not look green either.
    with _fresh():
        _report_phase(_install_plan(), enforce.INSTALL, enforce.OK)
        enforce.install_boundary("trainer")
        enforce.report("bogus:not_in_plan", enforce.INSTALL, enforce.VIOLATION, "stray")
        entry = next(o for o in enforce.verdict()["obligations"] if o["id"] == "bogus:not_in_plan")
        assert entry["kind"] == "unplanned" and entry["status"] == enforce.VIOLATION
        assert enforce.install_attestation_digest().startswith("violations=1:")


def test_late_record_rewrites_the_artifact_with_a_reopened_marker():
    # O2: the write latch used to leave enforcement.json stale-green after a post-close record.
    with _fresh():
        os.environ[STRICT_ENV] = "0"
        d = tempfile.mkdtemp()
        os.environ["ISOEXEC_CONTRACT_PATH"] = os.path.join(d, "contract.json")
        try:
            plan = _install_plan()
            _report_phase(plan, enforce.INSTALL, enforce.OK)
            assert enforce.install_boundary("trainer") is True
            (out,) = enforce.verdict_artifacts(d)
            assert json.load(open(out))["counts"]["refused"] == 0
            enforce.report("build_valid:contract", enforce.INSTALL, enforce.VIOLATION, "late")
            v = json.load(open(out))
            assert v["counts"]["refused"] == 1
            assert [r["evidence"] for r in v["reopened"]] == ["late"]
        finally:
            os.environ.pop("ISOEXEC_CONTRACT_PATH", None)


def test_late_record_check_is_side_aware():
    # One ledger serves both sides in a colocated process: the trainer closing INSTALL must not
    # brand an engine record that is still on time as LATE.
    with _fresh():
        os.environ[STRICT_ENV] = "0"
        trainer = enforce._arm("trainer")
        engine = enforce._arm("engine")
        trainer_ids = {o.obligation_id for o in trainer.obligations}
        engine_only = next(
            o.obligation_id
            for o in engine.obligations
            if o.phase == enforce.INSTALL and o.obligation_id not in trainer_ids
        )
        _report_phase(trainer, enforce.INSTALL, enforce.OK)
        assert enforce.install_boundary("trainer") is True
        enforce.report(engine_only, enforce.INSTALL, enforce.OK, "on time for the engine")
        assert enforce.ledger().reopened == []
        # ...and a genuinely late one, after the engine's own INSTALL closed, still is.
        enforce.close_phase(enforce.INSTALL, "engine")
        enforce.report(engine_only, enforce.INSTALL, enforce.OK, "late")
        assert [r.evidence for r in enforce.ledger().reopened] == ["late"]


def test_close_phase_internal_error_leaves_a_record_not_only_a_log():
    # O4: fail-open is deliberate, invisible is not.
    with _fresh():
        d = tempfile.mkdtemp()
        os.environ["ISOEXEC_CONTRACT_PATH"] = os.path.join(d, "contract.json")
        orig = enforce.ObligationLedger.judged_status
        enforce.ObligationLedger.judged_status = lambda self, ob: (_ for _ in ()).throw(RuntimeError("ledger bug"))
        try:
            assert enforce.close_phase(enforce.INSTALL, "trainer") is True
        finally:
            enforce.ObligationLedger.judged_status = orig
            os.environ.pop("ISOEXEC_CONTRACT_PATH", None)
        (art,) = enforce.verdict_artifacts(d)
        errors = json.load(open(art))["internal_errors"]
        assert len(errors) == 1 and "ledger bug" in errors[0]


def test_step1_boundary_is_a_no_op_without_a_contract():
    """The controller builds no contract, so its STEP1 call must not close an empty plan.

    An unguarded close there rewrites the verdict artifact from an empty ledger, turning a
    recorded RED green.
    """
    with _fresh():
        pc._CONTRACT, pc._VIEW = None, None
        with tempfile.TemporaryDirectory() as d:
            os.environ["ISOEXEC_CONTRACT_PATH"] = os.path.join(d, "contract.json")
            try:
                assert enforce.step1_boundary("trainer") is True
                assert ("trainer", enforce.STEP1) not in enforce.ledger().closed
                assert enforce.verdict_artifacts(d) == []
            finally:
                os.environ.pop("ISOEXEC_CONTRACT_PATH", None)


def _report_step1_but_the_gate(result=enforce.OK):
    plan = enforce._arm("trainer")
    for ob in plan.obligations:
        if ob.phase == enforce.STEP1 and not ob.obligation_id.startswith("gate:"):
            enforce.report(ob.obligation_id, enforce.STEP1, result, "test census")
    return plan


def test_step1_boundary_closes_green_on_a_clean_gate():
    from skyrl.train.utils import trainer_utils as tu

    with _fresh():
        P = tu.ISOEXEC_FORWARD_GATE_PREFIX
        good = {f"{P}_mean": 1e-6, f"{P}_max": 1e-5, f"{P}_min": 0.0, f"{P}_std": 0.0}
        assert tu.validate_isoexec_forward_gate(good, enabled=True, scoring_audit_skipped=False) is True
        _report_step1_but_the_gate()
        assert enforce.step1_boundary("trainer") is True
        assert enforce.verdict_counts()["refused"] == 0


def test_red_gate_refuses_the_step1_close_strict_and_demotes_under_debug():
    """The RED gate goes through ``enforce.refuse``: identical strict behavior, demoted in debug.

    Either way the ledger record is a violation, so the demoted run's artifact is just as red.
    """
    from skyrl.train.utils import trainer_utils as tu

    P = tu.ISOEXEC_FORWARD_GATE_PREFIX
    bad = {f"{P}_mean": 1e-2, f"{P}_max": 1e-1, f"{P}_min": 0.0, f"{P}_std": 0.0}
    with _fresh():
        msg = _refuses(tu.validate_isoexec_forward_gate, dict(bad), enabled=True, scoring_audit_skipped=False)
        assert "RED before backward" in msg
        _report_step1_but_the_gate()
        assert "gate:engine_decode|trainer_score (violation)" in _refuses(enforce.step1_boundary, "trainer")
    with _fresh():
        os.environ[trace.ENV_TRACE] = "/tmp/isoexec-debug-demote-test"
        try:
            served = dict(bad)
            assert tu.validate_isoexec_forward_gate(served, enabled=True, scoring_audit_skipped=False) is True
            # Demoted, not softened: the limits are still served and the record is still a violation.
            assert served["policy/isoexec_forward_gate_max_limit"] == tu.ISOEXEC_FORWARD_GATE_MAX_MAX
            assert enforce.ledger().records["gate:engine_decode|trainer_score"][-1].result == enforce.VIOLATION
            _report_step1_but_the_gate()
            assert enforce.step1_boundary("trainer") is False  # logged verdict, no raise
            assert enforce.verdict_counts()["refused"] == 1
        finally:
            os.environ.pop(trace.ENV_TRACE, None)


def test_two_ledgers_in_one_directory_both_survive_and_red_stays_red():
    """Workers share the contract directory, so one filename would be last-writer-wins and a
    trainer's green write would replace an engine RED. Per-process names + a glob merge keep both."""
    with _fresh():
        with tempfile.TemporaryDirectory() as d:
            os.environ["ISOEXEC_CONTRACT_PATH"] = os.path.join(d, "contract.json")
            os.environ[STRICT_ENV] = "0"
            try:
                os.environ["RANK"] = "3"
                plan = enforce.derive_obligation_plan(_CONTRACT, _REG, "engine")
                enforce.ledger().plans["engine"] = plan
                enforce.report("handshake:numerical_policy", enforce.WEIGHT_SYNC, enforce.VIOLATION, "split-brain")
                enforce.close_phase(enforce.WEIGHT_SYNC, "engine")
                red = enforce.verdict_counts()["refused"]
                assert red >= 1

                # A second process, same directory: the trainer worker that used to clobber it.
                enforce._reset_for_tests()
                os.environ["RANK"] = "11"
                plan = enforce.derive_obligation_plan(_CONTRACT, _REG, "trainer")
                enforce.ledger().plans["trainer"] = plan
                _report_phase(plan, enforce.WEIGHT_SYNC, enforce.OK)
                assert enforce.close_phase(enforce.WEIGHT_SYNC, "trainer") is True
                assert enforce.verdict_counts()["refused"] == 0

                names = [os.path.basename(f) for f in enforce.verdict_artifacts(d)]
                assert names == ["enforcement.engine.r3.json", "enforcement.trainer.r11.json"]
                merged = enforce.merge_verdicts(d)
                assert merged["files"] == names and merged["counts"]["refused"] == red
                hs = next(o for o in merged["obligations"] if o["id"] == "handshake:numerical_policy")
                assert hs["status"] == enforce.VIOLATION  # red stays red across the merge
            finally:
                for k in ("ISOEXEC_CONTRACT_PATH", "RANK"):
                    os.environ.pop(k, None)


def _run():
    import traceback

    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = []
    for name, fn in fns:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed.append(name)
            traceback.print_exc()
            print(f"FAIL {name}")
    print(f"{len(fns) - len(failed)}/{len(fns)} passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    _run()
