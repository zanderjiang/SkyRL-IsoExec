"""State and tolerance claims: grounded declaration, hook existence, and the gate reading back.

StateClaim refs must resolve to real hook code. The gate resolves its thresholds from the
ToleranceClaim wherever a contract exists, falling back to module constants where none can be built.
"""

import dataclasses
import math
import os
import pathlib
import re
import tempfile

from skyrl.backends.skyrl_train.isoexec.contract import ToleranceClaim
from skyrl.backends.skyrl_train.isoexec.contract.identity import compute_identities
from skyrl.backends.skyrl_train.isoexec.core import enforce
from skyrl.backends.skyrl_train.isoexec.core import process_contract as pc
from skyrl.backends.skyrl_train.isoexec.core.adapter import ToleranceChecker
from skyrl.backends.skyrl_train.isoexec.core.contract_delivery import (
    write_contract_file,
)
from skyrl.backends.skyrl_train.isoexec.core.registry_build import build_registry
from skyrl.backends.skyrl_train.isoexec.debug import trace
from skyrl.backends.skyrl_train.isoexec.models import qwen3_5
from skyrl.backends.skyrl_train.isoexec.models.profile import (
    ProfileError,
    StateFact,
    ToleranceFact,
)

_ISOEXEC_DIR = pathlib.Path(__file__).resolve().parents[2]

GATE_PAIR = ("engine_decode", "trainer_score")


def _contract():
    reg = build_registry(strict=True)
    return qwen3_5.build(reg, arch="sm90", profile=qwen3_5.PROFILE)


def test_state_claims_are_declared_facts():
    c = _contract()
    facts = {s.state_id: s for s in qwen3_5.PROFILE.states}
    claims = {s.state_id: s for s in c.claims.state}
    assert set(claims) == set(facts) == {"engine_prefix_cache", "gdn_recurrent_state"}
    for sid, f in facts.items():
        s = claims[sid]
        assert (s.invalidated_by, s.replay_safe, s.ref) == (tuple(f.invalidated_by), f.replay_safe, f.ref)
    assert claims["engine_prefix_cache"].invalidated_by == ("weight_sync",)
    assert claims["gdn_recurrent_state"].invalidated_by == ("weight_sync", "sleep_wake")


def test_state_claim_hooks_exist():
    # Every ref is "path::symbol"; a StateClaim naming a nonexistent hook is refused here.
    for s in _contract().claims.state:
        path, _, symbol = s.ref.partition("::")
        f = _ISOEXEC_DIR / path
        assert f.is_file(), f"state claim {s.state_id}: ref file {path!r} missing under {_ISOEXEC_DIR}"
        src = f.read_text()
        assert re.search(
            rf"^def {re.escape(symbol)}\(", src, re.M
        ), f"state claim {s.state_id}: hook {symbol!r} not defined in {path}"


def test_profile_refuses_hookless_state_fact():
    try:
        StateFact(state_id="x", invalidated_by=("weight_sync",), replay_safe=True, ref="")
        raise AssertionError("state fact without a hook ref must refuse")
    except ProfileError as e:
        assert "hook" in str(e)


def test_profile_refuses_unknown_lifecycle_event():
    try:
        StateFact(state_id="x", invalidated_by=("the_vibes_changed",), replay_safe=True, ref="lifecycle/ordering.py::f")
        raise AssertionError("an event no boundary observes must refuse")
    except ProfileError as e:
        assert "unknown lifecycle event(s)" in str(e)


def test_tolerance_claim_matches_gate_constants():
    from skyrl.train.utils import trainer_utils as tu

    (t,) = _contract().claims.tolerances
    assert tuple(t.case_pair) == GATE_PAIR == tu.ISOEXEC_FORWARD_GATE_CASE_PAIR
    b = dict(t.bounds)
    assert float(b["abs_diff_mean_max"]) == tu.ISOEXEC_FORWARD_GATE_MEAN_MAX
    assert float(b["abs_diff_max_max"]) == tu.ISOEXEC_FORWARD_GATE_MAX_MAX


def test_profile_refuses_non_decimal_bound():
    try:
        ToleranceFact(case_pair=GATE_PAIR, bounds=(("abs_diff_mean_max", "not-a-number"),))
        raise AssertionError("non-decimal bound must refuse")
    except ProfileError as e:
        assert "decimal" in str(e)


def test_profile_refuses_non_finite_bound():
    for bound in ("nan", "inf", "-inf"):
        try:
            ToleranceFact(case_pair=GATE_PAIR, bounds=(("abs_diff_mean_max", bound),))
            raise AssertionError(f"bound {bound!r} must refuse")
        except ProfileError as e:
            assert "finite threshold" in str(e)


def _gate_metrics(mean, maximum):
    from skyrl.train.utils.trainer_utils import ISOEXEC_FORWARD_GATE_PREFIX as P

    return {f"{P}_mean": mean, f"{P}_max": maximum, f"{P}_min": 0.0, f"{P}_std": 0.0}


def test_gate_reads_limits_from_contract_claim():
    from skyrl.train.utils import trainer_utils as tu

    saved = pc._CONTRACT
    try:
        pc._CONTRACT = None  # no contract -> the module constants
        assert tu.isoexec_gate_limits() == (tu.ISOEXEC_FORWARD_GATE_MEAN_MAX, tu.ISOEXEC_FORWARD_GATE_MAX_MAX)

        c = _contract()
        pc._CONTRACT = c  # production contract -> the claim (same numbers, different authority)
        assert tu.isoexec_gate_limits() == (1.0e-5, 1.0e-4)

        # A tighter claim must tighten the gate: mean=7e-6 passes the constants but violates
        # the claim, so the gate must refuse.
        tight = ToleranceClaim(
            case_pair=GATE_PAIR, bounds=(("abs_diff_mean_max", "5.0e-6"), ("abs_diff_max_max", "1.0e-4"))
        )
        c2 = dataclasses.replace(c, claims=dataclasses.replace(c.claims, tolerances=(tight,)))
        pc._CONTRACT = dataclasses.replace(c2, identities=compute_identities(c2))
        assert tu.isoexec_gate_limits() == (5.0e-6, 1.0e-4)
        assert (
            tu.validate_isoexec_forward_gate(_gate_metrics(1.0e-6, 1.0e-5), enabled=True, scoring_audit_skipped=False)
            is True
        )
        try:
            tu.validate_isoexec_forward_gate(_gate_metrics(7.0e-6, 1.0e-5), enabled=True, scoring_audit_skipped=False)
            raise AssertionError("mean above the CLAIMED limit must refuse")
        except RuntimeError as e:
            assert "5.0e-06" in str(e)
    finally:
        pc._CONTRACT = saved


def test_cpr_variant_claims_exact_zero():
    """The cpr composition claims 0.0; the recurrent one keeps the pre-rowinv bounds.

    Live exact-zero evidence exists only for cpr, and it licenses that composition, not the model.
    """
    reg = build_registry(strict=True)
    cpr = qwen3_5.build(reg, arch="sm90", profile=qwen3_5.CPR_PROFILE)
    rec = qwen3_5.build(reg, arch="sm90", profile=qwen3_5.PROFILE)
    assert dict(cpr.claims.tolerances[0].bounds) == {"abs_diff_max_max": "0.0", "abs_diff_mean_max": "0.0"}
    assert dict(rec.claims.tolerances[0].bounds) == {"abs_diff_max_max": "1.0e-4", "abs_diff_mean_max": "1.0e-5"}


def test_zero_bounds_admit_exactly_zero_and_refuse_one_ulp():
    """A 0.0 claim is a real gate, not a vacuous one: 0.0 passes, one ULP refuses."""
    from skyrl.train.utils import trainer_utils as tu

    saved = pc._CONTRACT
    try:
        reg = build_registry(strict=True)
        c = qwen3_5.build(reg, arch="sm90", profile=qwen3_5.CPR_PROFILE)
        pc._CONTRACT = c
        assert tu.isoexec_gate_limits() == (0.0, 0.0)
        assert (
            tu.validate_isoexec_forward_gate(_gate_metrics(0.0, 0.0), enabled=True, scoring_audit_skipped=False) is True
        )
        one_ulp = math.ulp(1.0e-7)
        for mean, maximum in ((0.0, one_ulp), (one_ulp, one_ulp)):
            try:
                tu.validate_isoexec_forward_gate(
                    _gate_metrics(mean, maximum), enabled=True, scoring_audit_skipped=False
                )
                raise AssertionError(f"mean={mean} max={maximum} must refuse against a 0.0 claim")
            except RuntimeError as e:
                assert "RED before backward" in str(e)
    finally:
        pc._CONTRACT = saved


def test_zero_bounds_survive_validation_and_the_checker():
    """0.0 is a finite threshold, and the install-time ToleranceChecker resolves the gate from it."""
    from skyrl.backends.skyrl_train.isoexec.contract.validate import validate

    reg = build_registry(strict=True)
    c = qwen3_5.build(reg, arch="sm90", profile=qwen3_5.CPR_PROFILE)
    assert validate(c) == []

    saved = pc._CONTRACT
    try:
        pc._CONTRACT = c
        res = ToleranceChecker().check(c.claims.tolerances[0], {})
        assert res.result == enforce.OK, res
    finally:
        pc._CONTRACT = saved


class _limits_env:
    """Isolate one gate-limit resolution: no process contract, no cached artifact, no log latch."""

    def __init__(self, path=None):
        self.path = path

    def __enter__(self):
        from skyrl.train.utils import trainer_utils as tu

        self._saved = (pc._CONTRACT, pc._VIEW, os.environ.get(tu.ISOEXEC_CONTRACT_PATH_ENV))
        pc._CONTRACT, pc._VIEW = None, None
        tu._gate_artifact_cache.clear()
        tu._gate_limit_logged.clear()
        os.environ.pop(tu.ISOEXEC_CONTRACT_PATH_ENV, None)
        if self.path is not None:
            os.environ[tu.ISOEXEC_CONTRACT_PATH_ENV] = self.path
        return self

    def __exit__(self, *exc):
        from skyrl.train.utils import trainer_utils as tu

        pc._CONTRACT, pc._VIEW = self._saved[:2]
        tu._gate_artifact_cache.clear()
        tu._gate_limit_logged.clear()
        if self._saved[2] is None:
            os.environ.pop(tu.ISOEXEC_CONTRACT_PATH_ENV, None)
        else:
            os.environ[tu.ISOEXEC_CONTRACT_PATH_ENV] = self._saved[2]


def _gate_logs(fn):
    """(result, [ISOEXEC-GATE] lines) -- the provenance line is part of the contract here."""
    from loguru import logger

    msgs = []
    sink = logger.add(lambda m: msgs.append(m.record["message"]), level="WARNING")
    try:
        return fn(), [m for m in msgs if m.startswith("[ISOEXEC-GATE]")]
    finally:
        logger.remove(sink)


def test_gate_limits_resolve_from_the_delivered_artifact():
    """The controller builds no contract, so it must read its limits from the delivered artifact.

    Falling back to the module constants would judge the run against the wrong envelope.
    """
    from skyrl.train.utils import trainer_utils as tu

    reg = build_registry(strict=True)
    cpr = qwen3_5.build(reg, arch="sm90", profile=qwen3_5.CPR_PROFILE)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "contract.json")
        write_contract_file(cpr, path)
        with _limits_env(path):
            limits, logs = _gate_logs(tu.isoexec_gate_limits)
            assert limits == (0.0, 0.0)
            assert any("source=contract_artifact" in m and path in m for m in logs), logs
            # ...and the gate then judges by the CONTRACT: 0.0 admits 0.0 and refuses one ulp.
            metrics = _gate_metrics(0.0, math.ulp(1.0e-7))
            try:
                tu.validate_isoexec_forward_gate(metrics, enabled=True, scoring_audit_skipped=False)
                raise AssertionError("the artifact's 0.0 claim must refuse a positive difference")
            except RuntimeError as e:
                assert "RED before backward" in str(e) and "max_limit=0.0e+00" in str(e)
            assert metrics["policy/isoexec_forward_gate_max_limit"] == 0.0


def test_gate_limit_fallback_logs_its_provenance():
    """The module constants are the fallback, but never a silent one.

    A configured-but-unresolvable artifact refuses rather than falling back (debug tracing demotes
    that refusal to a named fallback).
    """
    from skyrl.train.utils import trainer_utils as tu

    fallback = (tu.ISOEXEC_FORWARD_GATE_MEAN_MAX, tu.ISOEXEC_FORWARD_GATE_MAX_MAX)
    with _limits_env():
        limits, logs = _gate_logs(tu.isoexec_gate_limits)
        assert limits == fallback
        assert any("source=module_fallback" in m and "no ISOEXEC_CONTRACT_PATH" in m for m in logs), logs

    with tempfile.TemporaryDirectory() as d:
        missing = os.path.join(d, "contract.json")
        with _limits_env(missing):
            try:
                tu.isoexec_gate_limits()
                raise AssertionError("a configured artifact that never appeared must refuse")
            except RuntimeError as e:
                assert "never appeared" in str(e) and missing in str(e)
        with _limits_env(missing):
            os.environ[trace.ENV_TRACE] = d
            try:
                limits, logs = _gate_logs(tu.isoexec_gate_limits)
                assert limits == fallback
                assert any("source=module_fallback" in m and "DEMOTED" in m for m in logs), logs
            finally:
                os.environ.pop(trace.ENV_TRACE, None)


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
