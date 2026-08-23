"""Adapter unit-coverage gaps against the base-class surface: the process-adapter accessor
lifecycle, the fail-soft trainer build vs the refusing engine build, run_install(close=False)
gating, per-subclass facts completeness against every claimed axis, double-run idempotency of the
INSTALL sequence, and register_claim_checker re-registration semantics.
"""

import dataclasses
import os
from types import SimpleNamespace

# torch-first, as in production.
from skyrl.backends.skyrl_train.weight_sync.cuda_ipc_strategy import CudaIpcInitInfo  # noqa: F401

from skyrl.backends.skyrl_train.isoexec.contract import Claims
from skyrl.backends.skyrl_train.isoexec.core import adapter as ad
from skyrl.backends.skyrl_train.isoexec.core import arch as arch_mod
from skyrl.backends.skyrl_train.isoexec.core import enforce
from skyrl.backends.skyrl_train.isoexec.core import fingerprint as fp
from skyrl.backends.skyrl_train.isoexec.core import process_contract as pc
from skyrl.backends.skyrl_train.isoexec.core.process_contract import build_contract_view
from skyrl.backends.skyrl_train.isoexec.core.registry_build import build_registry
from skyrl.backends.skyrl_train.isoexec.models import qwen3_5
from skyrl.backends.skyrl_train.isoexec.runtimes.megatron.adapter import MegatronContractAdapter
from skyrl.backends.skyrl_train.isoexec.runtimes.vllm.adapter import VLLMContractAdapter

# CPU-only harness (a live production run owns the GPUs): real builds read core/arch.ARCH.
if arch_mod.ARCH == arch_mod.NON_ACCELERATOR_ARCH:
    arch_mod.ARCH = "sm90"

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
_VIEW = build_contract_view(_CONTRACT, _REG)


class _fresh:
    def __enter__(self):
        self._env = {k: v for k, v in os.environ.items() if k.startswith(("SKYRL_ISOEXEC", "ISOEXEC_"))}
        for k in self._env:
            del os.environ[k]
        self._saved = (pc._CONTRACT, pc._VIEW, fp._RECORDER, set(fp._LOGGED_TAGS))
        enforce._reset_for_tests()
        ad._reset_for_tests()
        fp._RECORDER = None
        fp._LOGGED_TAGS.clear()
        pc._CONTRACT, pc._VIEW = None, None
        return self

    def __exit__(self, *exc):
        enforce._reset_for_tests()
        ad._reset_for_tests()
        pc._CONTRACT, pc._VIEW, fp._RECORDER = self._saved[:3]
        fp._LOGGED_TAGS.clear()
        fp._LOGGED_TAGS.update(self._saved[3])
        for k in list(os.environ):
            if k.startswith(("SKYRL_ISOEXEC", "ISOEXEC_")):
                del os.environ[k]
        os.environ.update(self._env)
        return False


def _refuses(fn, *a, **kw) -> str:
    try:
        fn(*a, **kw)
    except (RuntimeError, ValueError) as e:
        return str(e)
    raise AssertionError(f"{getattr(fn, '__name__', fn)} should have refused")


def _mk_trainer(install_fn=lambda: None, model=qwen3_5.MODEL):
    os.environ["SKYRL_ISOEXEC_TRAINER_SP"] = "1"
    return MegatronContractAdapter(
        model,
        megatron_config=SimpleNamespace(
            tensor_model_parallel_size=4, pipeline_model_parallel_size=1, context_parallel_size=1
        ),
        install_fn=install_fn,
        world_size=4,
    )


def _mk_engine(install_fn=lambda: None, model=qwen3_5.MODEL):
    return VLLMContractAdapter(
        model,
        vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=1, world_size=8)),
        mp=SimpleNamespace(sequence_parallel=False),
        tp_size=8,
        install_fn=install_fn,
    )


def _stub_install(side):
    def _install():
        view = pc.cached_contract_view()
        for (op, site), meta in sorted(view.items()):
            if site.startswith(side):
                fp.record_install(op, site, meta["impl_id"], pinned=meta["pinned_constants"])
        fp.log_fingerprint_once(view, tag=f"{side}_install")

    return _install


def test_process_adapter_accessor_lifecycle():
    with _fresh():
        assert ad.process_adapter() is None  # unset: sites see "no adapter", never a stale one
        a = _mk_trainer()
        assert ad.set_process_adapter(a) is a and ad.process_adapter() is a
        b = _mk_engine()
        assert ad.set_process_adapter(b) is b and ad.process_adapter() is b  # last set wins
        ad._reset_for_tests()
        assert ad.process_adapter() is None
    assert ad.process_adapter() is None  # _fresh restored the module state


def test_unknown_side_is_rejected_at_construction():
    msg = _refuses(ad.ContractAdapter, "gateway", None)
    assert "unknown side" in msg and "gateway" in msg


def test_trainer_build_failsoft_engine_build_refuses():
    with _fresh():
        ran = []
        trainer = _mk_trainer(lambda: ran.append(1), model="no-such-model-zzz")
        assert trainer.build_failsoft is True
        assert trainer.run_install() is True  # the non-IsoExec path must survive a failed build
        assert trainer.contract is None and ran == [1]
        # ...but the failure is LOUD in the ledger, not a silent skip.
        assert enforce.ledger().records["build_valid:contract"][-1].result == "violation"
    with _fresh():
        ran = []
        engine = _mk_engine(lambda: ran.append(1), model="no-such-model-zzz")
        assert engine.build_failsoft is False
        _refuses(engine.run_install)  # engine refuses at build; ResolutionError is a ValueError
        assert not ran
        assert enforce.ledger().records["build_valid:contract"][-1].result == "violation"


def test_run_install_close_false_defers_the_boundary():
    with _fresh():
        trainer = ad.set_process_adapter(_mk_trainer(_stub_install("trainer")))
        assert trainer.run_install(close=False) is True
        assert ("trainer", enforce.INSTALL) not in enforce.ledger().closed
        assert enforce.verdict_counts() == {"ok": 0, "refused": 0, "logged": 0, "missing": 0, "excepted": 0}
        assert enforce.install_boundary("trainer") is True  # the caller owns the close
        assert ("trainer", enforce.INSTALL) in enforce.ledger().closed
        assert enforce.verdict_counts()["refused"] == 0 and enforce.verdict_counts()["missing"] == 0


def test_runtime_facts_cover_every_claimed_axis():
    with _fresh():
        chk = ad.CLAIM_CHECKERS["topology"]
        for adapter, want_world in ((_mk_trainer(), 4), (_mk_engine(), 8)):
            facts = adapter.runtime_facts()
            for t in _CONTRACT.claims.topology:
                assert t.axis in facts and facts[t.axis] is not None, f"{adapter.side} cannot obtain {t.axis}"
                r = chk.check(t, dict(facts, side=adapter.side.upper()))
                assert r.result == enforce.OK, f"{adapter.side} {t.axis}: {r.result} {r.evidence}"
            assert facts["world"] == want_world and facts["arch"]
        # An axis the caller cannot obtain is a visible skip, never a guess.
        t = next(c for c in _CONTRACT.claims.topology if c.axis == "TP")
        r = chk.check(t, {"side": "X"})
        assert r.result == enforce.SKIPPED and "unobtainable" in r.evidence


def test_run_install_twice_does_not_double_report():
    with _fresh():
        trainer = ad.set_process_adapter(_mk_trainer(_stub_install("trainer")))
        assert trainer.run_install() is True
        first = enforce.verdict_counts()
        assert first["refused"] == 0 and first["missing"] == 0
        assert len(enforce.ledger().records["build_valid:contract"]) == 1
        assert trainer.run_install() is True  # idempotent: cached build, re-closed boundary
        assert enforce.verdict_counts() == first
        assert len(enforce.ledger().records["build_valid:contract"]) == 1  # cached build: one attest
        plan = enforce.ledger().plans["trainer"]
        assert all(
            enforce.ledger().status_of(ob) == enforce.OK
            for ob in plan.obligations
            if ob.phase == enforce.INSTALL
        )


def test_register_claim_checker_reregistration():
    saved = dict(ad.CLAIM_CHECKERS)
    try:

        class _AltState:
            kind = "state"

            def obligation_id(self, c):
                return f"hook_exists:{c.state_id}"

            def check(self, c, facts):
                return ad.CheckResult(enforce.OK, "alt checker")

        alt = _AltState()
        assert ad.register_claim_checker(alt) is alt  # returns the checker (decorator-friendly)
        assert ad.CLAIM_CHECKERS["state"] is alt  # re-registration: last one wins
        with _fresh():
            pc._CONTRACT, pc._VIEW = _CONTRACT, _VIEW
            results = ad.check_all_claims(_CONTRACT, {"TP": 8, "SP": 0, "PP": 1, "CP": 1}, "engine")
            for s in _CONTRACT.claims.state:
                assert results[f"hook_exists:{s.state_id}"].evidence == "alt checker"

        class _Prov:
            kind = "provenance"

            def obligation_id(self, c):
                return "provenance:prose"

            def check(self, c, facts):
                return ad.CheckResult(enforce.OK, "checked")

        ad.register_claim_checker(_Prov())
        ExtClaims = dataclasses.make_dataclass(
            "ExtClaims", [("provenance", tuple, dataclasses.field(default=()))], bases=(Claims,), frozen=True
        )
        claims = ExtClaims(
            topology=_CONTRACT.claims.topology,
            state=_CONTRACT.claims.state,
            tolerances=_CONTRACT.claims.tolerances,
            provenance=(SimpleNamespace(claim="prose"),),
        )
        c2 = dataclasses.replace(_CONTRACT, claims=claims)
        with _fresh():
            pc._CONTRACT, pc._VIEW = c2, build_contract_view(c2, _REG)
            # Registration is how a kind buys its way in: the formerly refusing kind now dispatches.
            results = ad.check_all_claims(c2, {"TP": 8, "SP": 0, "PP": 1, "CP": 1}, "engine")
            assert results["provenance:prose"].result == enforce.OK
            assert "claim_check:provenance" not in enforce.ledger().records
    finally:
        ad.CLAIM_CHECKERS.clear()
        ad.CLAIM_CHECKERS.update(saved)
    assert set(ad.CLAIM_CHECKERS) == {"topology", "state", "tolerances"}


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
