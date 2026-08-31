"""ContractAdapter: claim dispatch, unknown-kind refusal, per-runtime facts, checker goldens.

Invariant: every claim on the contract reaches a registered checker, and a claim kind with no
checker refuses rather than being silently unchecked.
"""

import dataclasses
import os
from types import SimpleNamespace

from skyrl.backends.skyrl_train.isoexec.contract import (
    Claims,
    StateClaim,
    ToleranceClaim,
)
from skyrl.backends.skyrl_train.isoexec.core import adapter as ad
from skyrl.backends.skyrl_train.isoexec.core import enforce
from skyrl.backends.skyrl_train.isoexec.core import fingerprint as fp
from skyrl.backends.skyrl_train.isoexec.core import process_contract as pc
from skyrl.backends.skyrl_train.isoexec.core.process_contract import build_contract_view
from skyrl.backends.skyrl_train.isoexec.core.registry_build import build_registry
from skyrl.backends.skyrl_train.isoexec.models import qwen3_5

STRICT_ENV = "SKYRL_ISOEXEC_MANIFEST_STRICT"

_REG = build_registry(strict=True)


def _clean_build():
    saved = {k: v for k, v in os.environ.items() if k.startswith("SKYRL_ISOEXEC")}
    for k in saved:
        del os.environ[k]
    try:
        return qwen3_5.build(_REG, arch="sm90", profile=qwen3_5.PROFILE)
    finally:
        os.environ.update(saved)


_CONTRACT = _clean_build()

# The production deployments (run_qwen35_dapo_isoexec.sh): trainer TP4/SP1, engine TP8/SP0.
TRAINER_FACTS = {"TP": 4, "SP": 1, "PP": 1, "CP": 1}
ENGINE_FACTS = {"TP": 8, "SP": 0, "PP": 1, "CP": 1}


class _fresh:
    """Reset ledger + fingerprint recorder + cached contract + process adapter around one test."""

    def __enter__(self):
        self._saved = (pc._CONTRACT, pc._VIEW, fp._RECORDER, set(fp._LOGGED_TAGS), os.environ.get(STRICT_ENV))
        enforce._reset_for_tests()
        ad._reset_for_tests()
        fp._RECORDER = None
        fp._LOGGED_TAGS.clear()
        os.environ.pop(STRICT_ENV, None)
        pc._CONTRACT, pc._VIEW = _CONTRACT, build_contract_view(_CONTRACT, _REG)
        return self

    def __exit__(self, *exc):
        enforce._reset_for_tests()
        ad._reset_for_tests()
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
    except (RuntimeError, ValueError) as e:
        return str(e)
    raise AssertionError(f"{getattr(fn, '__name__', fn)} should have refused")


def test_registry_covers_every_production_claim_kind():
    # Every field of the Claims schema owes a registered ClaimChecker.
    kinds = {f.name for f in dataclasses.fields(Claims)}
    assert kinds == {"topology", "state", "tolerances"}
    assert set(ad.CLAIM_CHECKERS) == kinds
    for kind, chk in ad.CLAIM_CHECKERS.items():
        assert isinstance(chk, ad.ClaimChecker) and chk.kind == kind
        assert callable(chk.check) and callable(chk.obligation_id)
    assert isinstance(ad.CLAIM_CHECKERS["topology"], ad.TopologyChecker)
    assert isinstance(ad.CLAIM_CHECKERS["state"], ad.StateHookChecker)
    assert isinstance(ad.CLAIM_CHECKERS["tolerances"], ad.ToleranceChecker)


def test_dispatch_completeness_every_claim_reaches_its_checker():
    with _fresh():
        results = ad.check_all_claims(_CONTRACT, ENGINE_FACTS, "engine")
        want = (
            {f"domain_check:{t.axis}" for t in _CONTRACT.claims.topology}
            | {f"hook_exists:{s.state_id}" for s in _CONTRACT.claims.state}
            | {f"gate:{t.case_pair[0]}|{t.case_pair[1]}" for t in _CONTRACT.claims.tolerances}
        )
        assert set(results) == want and len(want) == 7
        assert all(r.result == enforce.OK for r in results.values())
        led = enforce.ledger().records
        for t in _CONTRACT.claims.topology:
            assert led[f"domain_check:{t.axis}"][-1].result == "ok"
        for s in _CONTRACT.claims.state:
            assert led[f"hook_exists:{s.state_id}"][-1].result == "ok"
        # Gate wiring attests ok but is not ledger-reported: gate@STEP1 must require the real
        # STEP-time run, not an INSTALL-time attest.
        assert "gate:engine_decode|trainer_score" not in led


def test_unknown_claim_kind_is_a_refusal():
    # Guards against a claim kind with no registered checker being silently unchecked at runtime.
    ExtClaims = dataclasses.make_dataclass(
        "ExtClaims",
        [("provenance", tuple, dataclasses.field(default=()))],
        bases=(Claims,),
        frozen=True,
    )
    claims = ExtClaims(
        topology=_CONTRACT.claims.topology,
        state=_CONTRACT.claims.state,
        tolerances=_CONTRACT.claims.tolerances,
        provenance=(SimpleNamespace(claim="unhashed prose"),),
    )
    c2 = dataclasses.replace(_CONTRACT, claims=claims)
    with _fresh():
        msg = _refuses(ad.check_all_claims, c2, ENGINE_FACTS, "engine")
        assert "provenance" in msg and "no registered checker" in msg.lower()
        recs = enforce.ledger().records["claim_check:provenance"]
        assert recs[-1].result == "violation"
    with _fresh():
        # Severity refuse, demotable only by the one strictness knob (mirroring every boundary).
        os.environ[STRICT_ENV] = "0"
        ad.check_all_claims(c2, ENGINE_FACTS, "engine")  # error-logged, not raised
        assert enforce.ledger().records["claim_check:provenance"][-1].result == "violation"
    with _fresh():
        # An EMPTY unknown group carries no claims and owes nothing.
        empty = dataclasses.replace(_CONTRACT, claims=ExtClaims(topology=_CONTRACT.claims.topology, provenance=()))
        ad.check_all_claims(empty, ENGINE_FACTS, "engine")


def test_per_runtime_facts_shape():
    from skyrl.backends.skyrl_train.isoexec.runtimes.megatron.adapter import (
        MegatronContractAdapter,
    )
    from skyrl.backends.skyrl_train.isoexec.runtimes.vllm.adapter import (
        VLLMContractAdapter,
    )

    saved = os.environ.get("SKYRL_ISOEXEC_TRAINER_SP")
    try:
        os.environ["SKYRL_ISOEXEC_TRAINER_SP"] = "1"
        m = MegatronContractAdapter(
            qwen3_5.MODEL,
            megatron_config=SimpleNamespace(
                tensor_model_parallel_size=4, pipeline_model_parallel_size=1, context_parallel_size=1
            ),
            install_fn=lambda: None,
            world_size=4,
        )
        facts = m.runtime_facts()
        assert {k: facts[k] for k in ("TP", "PP", "CP", "SP")} == TRAINER_FACTS
        assert facts["world"] == 4 and "arch" in facts
        assert m.side == "trainer" and m.build_failsoft is True
    finally:
        if saved is None:
            os.environ.pop("SKYRL_ISOEXEC_TRAINER_SP", None)
        else:
            os.environ["SKYRL_ISOEXEC_TRAINER_SP"] = saved
    v = VLLMContractAdapter(
        qwen3_5.MODEL,
        vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=1, world_size=8)),
        mp=SimpleNamespace(sequence_parallel=False),
        tp_size=8,
        install_fn=lambda: None,
    )
    facts = v.runtime_facts()  # mpu uninitialized here -> CP falls back to 1
    assert {k: facts[k] for k in ("TP", "PP", "CP", "SP")} == ENGINE_FACTS
    assert facts["world"] == 8 and "arch" in facts
    assert v.side == "engine" and v.build_failsoft is False


def test_topology_checker_matches_pre_migration_golden():
    claims = {t.axis: t for t in _CONTRACT.claims.topology}
    chk = ad.CLAIM_CHECKERS["topology"]
    # Golden accept/reject on the same inputs the pre-migration function judged.
    assert chk.check(claims["TP"], dict(ENGINE_FACTS, side="ENGINE")).result == enforce.OK
    r = chk.check(claims["TP"], dict(ENGINE_FACTS, TP=16, side="ENGINE"))
    assert r.result == enforce.VIOLATION and "outside the proven domain" in r.evidence and "16" in r.evidence
    r = chk.check(claims["PP"], dict(TRAINER_FACTS, PP=2, side="TRAINER"))
    assert r.result == enforce.VIOLATION and "pins 1" in r.evidence
    r = chk.check(claims["TP"], {"side": "ENGINE"})
    assert r.result == enforce.SKIPPED and "unobtainable" in r.evidence
    # And the deprecated name still enforces identically through the delegate.
    with _fresh():
        assert pc.assert_topology_within_claims(_CONTRACT, TRAINER_FACTS, side="TRAINER") is True
        led = enforce.ledger().records
        assert led["domain_check:TP"][-1].result == "ok" and led["domain_check:SP"][-1].result == "ok"
    with _fresh():
        msg = _refuses(pc.assert_topology_within_claims, _CONTRACT, dict(ENGINE_FACTS, TP=16), side="ENGINE")
        assert "OUTSIDE the contract's claims" in msg and "proven domain" in msg
    with _fresh():
        os.environ[STRICT_ENV] = "0"
        assert pc.assert_topology_within_claims(_CONTRACT, dict(ENGINE_FACTS, TP=16), side="ENGINE") is False
        assert enforce.ledger().records["domain_check:TP"][-1].result == "violation"


def test_state_checker_matches_pre_migration_golden():
    chk = ad.CLAIM_CHECKERS["state"]
    for s in _CONTRACT.claims.state:  # every production ref is real code in-tree
        r = chk.check(s, {})
        assert r.result == enforce.OK and r.evidence == s.ref
    bogus_sym = StateClaim(state_id="x", invalidated_by=("weight_sync",), replay_safe=True, ref="core/enforce.py::nope")
    r = chk.check(bogus_sym, {})
    assert r.result == enforce.VIOLATION and "not defined" in r.evidence
    bogus_file = StateClaim(state_id="y", invalidated_by=("weight_sync",), replay_safe=True, ref="no/such.py::f")
    r = chk.check(bogus_file, {})
    assert r.result == enforce.VIOLATION and "missing" in r.evidence
    # The deprecated reporter delegates to the same checker with the same records.
    with _fresh():
        enforce.attest_state_hooks(_CONTRACT)
        led = enforce.ledger().records
        for s in _CONTRACT.claims.state:
            assert led[f"hook_exists:{s.state_id}"][-1].result == "ok"
    with _fresh():
        broken = dataclasses.replace(_CONTRACT, claims=dataclasses.replace(_CONTRACT.claims, state=(bogus_sym,)))
        enforce.attest_state_hooks(broken)
        assert enforce.ledger().records["hook_exists:x"][-1].result == "violation"


def test_tolerance_checker_attests_gate_wiring():
    from skyrl.train.utils import trainer_utils as tu

    chk = ad.CLAIM_CHECKERS["tolerances"]
    with _fresh():
        (t,) = _CONTRACT.claims.tolerances
        r = chk.check(t, {})
        assert r.result == enforce.OK and "resolve" in r.evidence
        # A tightened cached claim re-wires the gate: limits resolve from the cached claim.
        tightened = ToleranceClaim(
            case_pair=t.case_pair, bounds=(("abs_diff_mean_max", "5.0e-6"), ("abs_diff_max_max", "1.0e-4"))
        )
        c2 = dataclasses.replace(_CONTRACT, claims=dataclasses.replace(_CONTRACT.claims, tolerances=(tightened,)))
        pc._CONTRACT = c2
        assert tu.isoexec_gate_limits() == (5.0e-6, 1.0e-4)
        assert chk.check(tightened, {}).result == enforce.OK
        # ...and the ORIGINAL claim now visibly fails to wire (limits resolve elsewhere).
        assert chk.check(t, {}).result == enforce.VIOLATION
    # A tolerance pair no gate consumes is an unenforced claim: violation, not silence.
    alien = ToleranceClaim(case_pair=("engine_prefill", "trainer_fwd"), bounds=(("abs_diff_mean_max", "1.0e-5"),))
    r = chk.check(alien, {})
    assert r.result == enforce.VIOLATION and "no gate consumes" in r.evidence


def test_adapter_install_sequence_matches_pre_adapter_ledger_counts():
    """Trainer INSTALL closes with the golden ledger counts: all ok, nothing refused or missing."""
    from skyrl.backends.skyrl_train.isoexec.runtimes.megatron.adapter import (
        MegatronContractAdapter,
    )

    golden = {"ok": 37, "refused": 0, "logged": 0, "missing": 0, "excepted": 0}
    with _fresh():
        pc._CONTRACT, pc._VIEW = None, None  # the adapter builds for real: build_valid is recorded

        def _install():
            for (op, site), meta in sorted(pc.cached_contract_view().items()):
                if site.startswith("trainer"):
                    fp.record_install(op, site, meta["impl_id"])

        adapter = MegatronContractAdapter(
            qwen3_5.MODEL,
            megatron_config=SimpleNamespace(
                tensor_model_parallel_size=4, pipeline_model_parallel_size=1, context_parallel_size=1
            ),
            install_fn=_install,
            world_size=4,
        )
        ad.set_process_adapter(adapter)
        assert ad.process_adapter() is adapter
        assert adapter.run_install() is True
        assert enforce.verdict_counts() == golden
        assert ("trainer", enforce.INSTALL) in enforce.ledger().closed


def test_boundary_delegates_first_forward_and_weight_sync():
    from skyrl.backends.skyrl_train.isoexec.runtimes.megatron.adapter import (
        MegatronContractAdapter,
    )
    from skyrl.backends.skyrl_train.isoexec.runtimes.vllm.adapter import (
        VLLMContractAdapter,
    )

    with _fresh():
        # Trainer stamp flow: report + WEIGHT_SYNC close.
        m = MegatronContractAdapter(qwen3_5.MODEL, megatron_config=SimpleNamespace(), install_fn=lambda: None)
        h = pc.contract_hash()
        assert m.on_weight_sync(h) is True
        recs = enforce.ledger().records["handshake:numerical_policy"]
        assert recs[-1].result == "ok" and f"stamped composite={h}" in recs[-1].evidence
        assert ("trainer", enforce.WEIGHT_SYNC) in enforce.ledger().closed
    with _fresh():
        # Engine receiver flow: assert_init_info_contract + WEIGHT_SYNC close.
        v = VLLMContractAdapter(
            qwen3_5.MODEL, vllm_config=SimpleNamespace(), mp=SimpleNamespace(), tp_size=8, install_fn=lambda: None
        )
        good = SimpleNamespace(contract_hash=pc.contract_hash())
        assert v.on_weight_sync(good) is True
        assert enforce.ledger().records["handshake:numerical_policy"][-1].result == "ok"
        msg = _refuses(v.on_weight_sync, SimpleNamespace(contract_hash="0" * 64))
        assert "MISMATCH" in msg
    with _fresh():
        # Engine first forward: fingerprint log (once per tag) + backstop + FIRST_FORWARD close.
        for op, site in pc.cached_contract_view():
            if site.startswith("engine"):
                fp.record_install(op, site, pc.cached_contract_view()[(op, site)]["impl_id"])
        v = VLLMContractAdapter(
            qwen3_5.MODEL, vllm_config=SimpleNamespace(), mp=SimpleNamespace(), tp_size=8, install_fn=lambda: None
        )
        assert v.on_first_forward() is True
        assert "engine_first_forward" in fp._LOGGED_TAGS
        assert enforce.ledger().records["installed_backstop:first_forward"][-1].result == "ok"
        assert ("engine", enforce.FIRST_FORWARD) in enforce.ledger().closed


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
