"""Rowinv ENGAGEMENT enforcement: a one-sided serve refuses instead of running silently.

The failure this file pins: rowinv composed on both sides -- contract hashes MATCHED, the
handshake passed -- with every trainer process serving rowinv (4096/4096) and ZERO engine workers
doing so, producing a one-sided composition. The manifest proves selection; only ``ops/logprobs/rowinv.py::stats()['served'] > 0`` proves
engagement, so that census is now wired into the obligation ledger and escalated to REFUSE for
exactly this impl (SERVED_REFUSE_IMPLS: it replaces BOTH runtimes at once and carries no
bitwise_equal_to twin -- see the rationale beside the table in core/enforce.py).

Asserted here, all CPU:

  * a side whose census says served=0 REFUSES -- at the engagement boundary, and at a
    ``close_phase(STEP1)`` whose served record is missing (a check that never ran is as loud as
    one that failed);
  * both sides serving passes clean;
  * a process with NO contract derives no rowinv obligation, records nothing, and cannot refuse --
    rowinv is composed unconditionally now, so "not selected" no longer means "flag off", it means
    no contract was built here. No OTHER served obligation may gain refuse severity either;
  * the demotion path holds: SKYRL_ISOEXEC_MANIFEST_STRICT=0 and SKYRL_ISOEXEC_DEBUG_TRACE both
    turn the refusal into a logged verdict with the violation still recorded (refuse() is the
    funnel, never bypassed);
  * the init-sync grace: the FIRST boundary call with a completely silent census (calls=0)
    defers -- the init weight sync legitimately precedes any forward -- and the SECOND refuses.

Run (CPU only):
    uv run --extra dev pytest skyrl/backends/skyrl_train/isoexec/core/tests/test_rowinv_engagement.py -q
"""

import os

import pytest

from skyrl.backends.skyrl_train.isoexec.core import enforce
from skyrl.backends.skyrl_train.isoexec.core import fingerprint as fp
from skyrl.backends.skyrl_train.isoexec.core import process_contract as pc
from skyrl.backends.skyrl_train.isoexec.core.process_contract import build_contract_view
from skyrl.backends.skyrl_train.isoexec.core.registry_build import build_registry
from skyrl.backends.skyrl_train.isoexec.models import qwen3_5

pytest.importorskip("torch")  # ops/logprobs/rowinv.py imports torch; the census read needs it

from skyrl.backends.skyrl_train.isoexec.ops.logprobs import rowinv  # noqa: E402

STRICT_ENV = "SKYRL_ISOEXEC_MANIFEST_STRICT"
DEBUG_ENV = "SKYRL_ISOEXEC_DEBUG_TRACE"
OP = enforce.ROWINV_OP
SERVED_IDS = {
    "trainer": ("served:logprobs.log_softmax:trainer_fwd", "served:logprobs.log_softmax:trainer_score"),
    "engine": ("served:logprobs.log_softmax:engine_decode", "served:logprobs.log_softmax:engine_prefill"),
}

_REG = build_registry(strict=True)


def _build():
    """Build the contract under a stripped env so it asserts what the code composes."""
    saved = {k: v for k, v in os.environ.items() if k.startswith("SKYRL_ISOEXEC")}
    for k in saved:
        del os.environ[k]
    try:
        return qwen3_5.build(_REG, arch="sm90", profile=qwen3_5.PROFILE)
    finally:
        os.environ.update(saved)


_CONTRACT = _build()


class _fresh:
    """Reset ledger + fingerprint recorder and pin the cached contract around one test."""

    def __init__(self, contract):
        self._contract = contract

    def __enter__(self):
        self._saved = (
            pc._CONTRACT,
            pc._VIEW,
            fp._RECORDER,
            set(fp._LOGGED_TAGS),
            os.environ.get(STRICT_ENV),
            os.environ.get(DEBUG_ENV),
        )
        enforce._reset_for_tests()
        fp._RECORDER = None
        fp._LOGGED_TAGS.clear()
        os.environ.pop(STRICT_ENV, None)
        os.environ.pop(DEBUG_ENV, None)
        pc._CONTRACT, pc._VIEW = self._contract, build_contract_view(self._contract, _REG)
        return self

    def __exit__(self, *exc):
        enforce._reset_for_tests()
        pc._CONTRACT, pc._VIEW, fp._RECORDER = self._saved[:3]
        fp._LOGGED_TAGS.clear()
        fp._LOGGED_TAGS.update(self._saved[3])
        for env, val in ((STRICT_ENV, self._saved[4]), (DEBUG_ENV, self._saved[5])):
            if val is None:
                os.environ.pop(env, None)
            else:
                os.environ[env] = val
        return False


def _stats(monkeypatch, **overrides):
    """Pin the census the boundary reads. rowinv.py itself is consumed strictly read-only."""
    s = {"calls": 0, "served": 0, "declined": 0, "decline_reason": ""}
    s.update(overrides)
    monkeypatch.setattr(rowinv, "stats", lambda: dict(s))


def _refuses(fn, *a, **kw) -> str:
    try:
        fn(*a, **kw)
    except RuntimeError as e:
        return str(e)
    raise AssertionError(f"{getattr(fn, '__name__', fn)} should have refused")


# -- derivation and severity: the escalation is exactly rowinv-shaped, nothing else --------------


def test_composition_derives_refuse_severity_served_obligations():
    for side, ids in SERVED_IDS.items():
        plan = enforce.derive_obligation_plan(_CONTRACT, _REG, side)
        served = {o.obligation_id: o for o in plan.obligations if o.kind == enforce.SERVED}
        for oid in ids:
            assert oid in served, f"{side}: {oid} must be owed (SERVED_COUNTERS names the rowinv census)"
            assert enforce.severity(served[oid]) == enforce.REFUSE, f"{oid}: one-sided serve must refuse"
        # The escalation must not blanket-flip the SERVED kind: every non-rowinv served
        # obligation keeps its documented LOG severity.
        for oid, ob in served.items():
            if oid not in ids:
                assert enforce.severity(ob) == enforce.LOG, f"{oid} must stay engagement observability"
        # And rowinv never appears among the counterless entries.
        assert all(impl != enforce.ROWINV_IMPL_ID for _op, _case, impl in plan.no_served_counter)


def test_no_other_obligation_gains_refuse_severity():
    """The escalation is rowinv-shaped: only the rowinv served ids carry an override."""
    for side, ids in SERVED_IDS.items():
        plan = enforce.derive_obligation_plan(_CONTRACT, _REG, side)
        for o in plan.obligations:
            if o.obligation_id not in ids:
                assert o.severity_override is None, f"{o.obligation_id} must not inherit the escalation"


def test_escalated_impls_all_carry_a_census():
    assert set(enforce.SERVED_REFUSE_IMPLS) <= set(enforce.SERVED_COUNTERS)
    assert (OP, enforce.ROWINV_IMPL_ID) in enforce.SERVED_COUNTERS


# -- (a) one side served=0 refuses ----------------------------------------------------------------


def test_boundary_refuses_when_selected_and_never_served(monkeypatch):
    with _fresh(_CONTRACT):
        _stats(monkeypatch, calls=4096, declined=4096, decline_reason="kernel unavailable")
        msg = _refuses(enforce.rowinv_engagement_boundary, "engine", require=True)
        assert "NEVER SERVED" in msg and "rowinv_leaftree" in msg
        assert "last_decline='kernel unavailable'" in msg, "the refusal must carry the census, not a banner"
        for oid in SERVED_IDS["engine"]:
            recs = enforce.ledger().records[oid]
            assert recs[-1].result == enforce.VIOLATION
            assert "served=0" in recs[-1].evidence


def test_close_phase_step1_refuses_a_served_record_that_never_arrived():
    """A check that never ran is as loud as one that failed: nobody reported, STEP1 refuses."""
    with _fresh(_CONTRACT):
        plan = enforce.derive_obligation_plan(_CONTRACT, _REG, "engine")
        enforce.ledger().plans["engine"] = plan
        for ob in plan.obligations:
            if ob.phase == enforce.STEP1 and not ob.obligation_id.startswith(f"served:{OP}:"):
                enforce.report(ob.obligation_id, enforce.STEP1, enforce.OK, "test")
        msg = _refuses(enforce.close_phase, enforce.STEP1, "engine")
        assert f"served:{OP}:engine_decode" in msg and "missing" in msg


def test_one_sided_serve_trainer_ok_engine_refuses(monkeypatch):
    """The one-sided serve in miniature: the trainer's census is green, the engine's is zero."""
    with _fresh(_CONTRACT):
        _stats(monkeypatch, calls=4096, served=4096)
        assert enforce.rowinv_engagement_boundary("trainer", require=True) is True
    with _fresh(_CONTRACT):
        _stats(monkeypatch, calls=512, served=0, declined=512, decline_reason="a TP peer declined")
        _refuses(enforce.rowinv_engagement_boundary, "engine", require=True)


# -- (b) both sides serving passes ----------------------------------------------------------------


def test_both_sides_serving_pass_and_step1_closes_green(monkeypatch):
    with _fresh(_CONTRACT):
        _stats(monkeypatch, calls=64, served=64)
        for side in ("trainer", "engine"):
            assert enforce.rowinv_engagement_boundary(side, require=True) is True
            for oid in SERVED_IDS[side]:
                recs = enforce.ledger().records[oid]
                assert recs[-1].result == enforce.OK and "served=64" in recs[-1].evidence
        # STEP1 then closes green: the rowinv records exist and are ok; counterless/unreported
        # LOG-severity served obligations log, never refuse (unchanged behavior).
        for side in ("trainer", "engine"):
            plan = enforce.derive_obligation_plan(_CONTRACT, _REG, side)
            enforce.ledger().plans[side] = plan
            for ob in plan.obligations:
                if ob.kind == enforce.GATE:
                    enforce.report(ob.obligation_id, enforce.STEP1, enforce.OK, "test gate")
            assert enforce.close_phase(enforce.STEP1, side) is True


def test_boundary_is_latched_after_ok(monkeypatch):
    with _fresh(_CONTRACT):
        _stats(monkeypatch, calls=8, served=8)
        assert enforce.rowinv_engagement_boundary("engine", require=True) is True
        n = len(enforce.ledger().records[SERVED_IDS["engine"][0]])
        # Later per-sync calls are free: no new records, no census read that could refuse.
        _stats(monkeypatch, calls=8, served=0)  # even a (impossible) regressed census is not re-read
        assert enforce.rowinv_engagement_boundary("engine") is True
        assert len(enforce.ledger().records[SERVED_IDS["engine"][0]]) == n


# -- (c) no contract: no obligation, no record, no refusal ----------------------------------------


def test_boundary_is_inert_without_a_contract(monkeypatch):
    """Rowinv is composed unconditionally, so the only "not selected" left is "no contract".

    A CPU driver or a process that never built one must not be refused for a census it was never
    in a position to move.
    """
    saved = (pc._CONTRACT, pc._VIEW)
    enforce._reset_for_tests()
    pc._CONTRACT, pc._VIEW = None, None
    try:
        _stats(monkeypatch)  # served=0 everywhere -- and it must not matter
        for side in ("trainer", "engine"):
            assert enforce.rowinv_engagement_boundary(side, require=True) is True
        assert enforce.ledger().records == {}, "no contract must write nothing into the ledger"
    finally:
        enforce._reset_for_tests()
        pc._CONTRACT, pc._VIEW = saved


# -- (d) the demotion path holds ------------------------------------------------------------------


def test_strict_off_demotes_to_logged_verdict_with_violation_recorded(monkeypatch):
    with _fresh(_CONTRACT):
        os.environ[STRICT_ENV] = "0"
        _stats(monkeypatch, calls=16, declined=16, decline_reason="V % G != 0")
        assert enforce.rowinv_engagement_boundary("engine", require=True) is False
        for oid in SERVED_IDS["engine"]:
            assert enforce.ledger().records[oid][-1].result == enforce.VIOLATION
        # The ledger stays red behind the demotion: a later strict STEP1 close still refuses.
        plan = enforce.derive_obligation_plan(_CONTRACT, _REG, "engine")
        enforce.ledger().plans["engine"] = plan
        os.environ.pop(STRICT_ENV, None)
        for ob in plan.obligations:
            if ob.phase == enforce.STEP1 and not ob.obligation_id.startswith(f"served:{OP}:"):
                enforce.report(ob.obligation_id, enforce.STEP1, enforce.OK, "test")
        msg = _refuses(enforce.close_phase, enforce.STEP1, "engine")
        assert f"served:{OP}:engine_decode" in msg


def test_debug_trace_demotes_without_bypassing_refuse(monkeypatch):
    with _fresh(_CONTRACT):
        os.environ[DEBUG_ENV] = "1"
        _stats(monkeypatch, calls=16, declined=16)
        assert enforce.rowinv_engagement_boundary("engine", require=True) is False
        for oid in SERVED_IDS["engine"]:
            assert enforce.ledger().records[oid][-1].result == enforce.VIOLATION


def test_step1_close_demotes_under_strict_off():
    with _fresh(_CONTRACT):
        os.environ[STRICT_ENV] = "0"
        plan = enforce.derive_obligation_plan(_CONTRACT, _REG, "engine")
        enforce.ledger().plans["engine"] = plan
        assert enforce.close_phase(enforce.STEP1, "engine") is False  # logged verdict, no raise


# -- the init-sync grace ---------------------------------------------------------------------------


def test_first_silent_boundary_defers_second_refuses(monkeypatch):
    with _fresh(_CONTRACT):
        _stats(monkeypatch)  # calls=0: the init sync precedes any forward
        assert enforce.rowinv_engagement_boundary("engine") is True
        assert enforce.ledger().records == {}, "the grace must not discharge or taint anything"
        _refuses(enforce.rowinv_engagement_boundary, "engine")


def test_consulted_but_declining_census_refuses_on_the_first_boundary(monkeypatch):
    with _fresh(_CONTRACT):
        _stats(monkeypatch, calls=32, declined=32, decline_reason="import failed")
        _refuses(enforce.rowinv_engagement_boundary, "trainer")


def test_unreadable_census_fails_closed(monkeypatch):
    with _fresh(_CONTRACT):

        def _boom():
            raise OSError("census unreadable")

        monkeypatch.setattr(rowinv, "stats", _boom)
        msg = _refuses(enforce.rowinv_engagement_boundary, "engine", require=True)
        assert enforce.CHECKER_ERROR in msg
        for oid in SERVED_IDS["engine"]:
            assert enforce.ledger().records[oid][-1].result == enforce.VIOLATION
