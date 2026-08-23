"""State and tolerance claims: grounded declaration, hook existence, and the gate reading back.

The StateClaims name lifecycle hooks that ALREADY run (declaration, not machinery) -- each ref must
resolve to real code. The ToleranceClaim carries the forward gate's qualified thresholds; the gate
resolves its limits from the claim wherever a contract exists, falling back to its constants (the
same numbers) where none can be built (the CPU driver).
"""

import dataclasses
import pathlib
import re

from skyrl.backends.skyrl_train.isoexec.contract import ToleranceClaim
from skyrl.backends.skyrl_train.isoexec.contract.identity import compute_identities
from skyrl.backends.skyrl_train.isoexec.core import process_contract as pc
from skyrl.backends.skyrl_train.isoexec.core.registry_build import build_registry
from skyrl.backends.skyrl_train.isoexec.models import qwen3_5
from skyrl.backends.skyrl_train.isoexec.models.profile import ProfileError, StateFact, ToleranceFact

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
    # Declaration plus a hook-exists check: every ref is "path::symbol" and the symbol is real
    # code in that file -- a StateClaim naming a nonexistent hook is refused here, in CI.
    for s in _contract().claims.state:
        path, _, symbol = s.ref.partition("::")
        f = _ISOEXEC_DIR / path
        assert f.is_file(), f"state claim {s.state_id}: ref file {path!r} missing under {_ISOEXEC_DIR}"
        src = f.read_text()
        assert re.search(rf"^def {re.escape(symbol)}\(", src, re.M), (
            f"state claim {s.state_id}: hook {symbol!r} not defined in {path}"
        )


def test_profile_refuses_hookless_state_fact():
    try:
        StateFact(state_id="x", invalidated_by=("weight_sync",), replay_safe=True, ref="")
        raise AssertionError("state fact without a hook ref must refuse")
    except ProfileError as e:
        assert "hook" in str(e)


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

        # A contract claiming a TIGHTER envelope must tighten the gate: mean=7e-6 passes the
        # constants but violates the claim -- the gate now refuses, proving it consumes the claim.
        tight = ToleranceClaim(
            case_pair=GATE_PAIR, bounds=(("abs_diff_mean_max", "5.0e-6"), ("abs_diff_max_max", "1.0e-4"))
        )
        c2 = dataclasses.replace(c, claims=dataclasses.replace(c.claims, tolerances=(tight,)))
        pc._CONTRACT = dataclasses.replace(c2, identities=compute_identities(c2))
        assert tu.isoexec_gate_limits() == (5.0e-6, 1.0e-4)
        assert (
            tu.validate_isoexec_forward_gate(
                _gate_metrics(1.0e-6, 1.0e-5), enabled=True, scoring_audit_skipped=False
            )
            is True
        )
        try:
            tu.validate_isoexec_forward_gate(_gate_metrics(7.0e-6, 1.0e-5), enabled=True, scoring_audit_skipped=False)
            raise AssertionError("mean above the CLAIMED limit must refuse")
        except RuntimeError as e:
            assert "5.0e-06" in str(e)
    finally:
        pc._CONTRACT = saved


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
