"""The negative-path battery: every designed refusal, proven to refuse (or to record exactly its
designed severity), each scenario through the REAL adapter/ledger path with state restored.

Scenarios: topology outside the claimed domain; perturbed weight-sync stamp; install-vs-declaration
divergence (wrong impl, wrong pin) refusing at FIRST_FORWARD; a suppressed reporter surfacing as a
missing-violation; an unknown claim kind; attestation-digest divergence breaking the pair; every
EXCEPTIONS entry soft exactly at its target with a non-excepted neighbor still refusing; and the
SKYRL_ISOEXEC_MANIFEST_STRICT=0 demotion of each refusal to warn-only.
"""

import dataclasses
import fnmatch
import os
import pickle
from types import SimpleNamespace

# torch-first, as in production.
from skyrl.backends.skyrl_train.weight_sync.cuda_ipc_strategy import CudaIpcInitInfo

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


def _new_process_role():
    enforce._reset_for_tests()
    ad._reset_for_tests()
    fp._RECORDER = None
    fp._LOGGED_TAGS.clear()
    pc._CONTRACT, pc._VIEW = None, None


def _refuses(fn, *a, **kw) -> str:
    try:
        fn(*a, **kw)
    except (RuntimeError, ValueError) as e:
        return str(e)
    raise AssertionError(f"{getattr(fn, '__name__', fn)} should have refused")


def _mk_trainer(install_fn=lambda: None):
    os.environ["SKYRL_ISOEXEC_TRAINER_SP"] = "1"
    return MegatronContractAdapter(
        qwen3_5.MODEL,
        megatron_config=SimpleNamespace(
            tensor_model_parallel_size=4, pipeline_model_parallel_size=1, context_parallel_size=1
        ),
        install_fn=install_fn,
        world_size=4,
    )


def _mk_engine(install_fn=lambda: None, tp=8):
    return VLLMContractAdapter(
        qwen3_5.MODEL,
        vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=1, world_size=8)),
        mp=SimpleNamespace(sequence_parallel=False),
        tp_size=tp,
        install_fn=install_fn,
    )


def _stub_install(side, skip=(), wrong_impl=None, wrong_pins=None):
    wrong_impl, wrong_pins = wrong_impl or {}, wrong_pins or {}

    def _install():
        view = pc.cached_contract_view()
        for (op, site), meta in sorted(view.items()):
            if not site.startswith(side) or (op, site) in skip:
                continue
            impl = wrong_impl.get((op, site), meta["impl_id"])
            pins = wrong_pins.get((op, site), meta["pinned_constants"])
            fp.record_install(op, site, impl, pinned=pins)
        fp.log_fingerprint_once(view, tag=f"{side}_install")

    return _install


def _stamp(h):
    return pickle.loads(
        pickle.dumps(CudaIpcInitInfo(override_existing_receiver=False, model_dtype_str="bfloat16", contract_hash=h))
    )


def test_topology_outside_domain_refuses_at_install():
    with _fresh():
        ran = []
        engine = _mk_engine(lambda: ran.append(1), tp=16)  # TP=16: outside the proven domain {1,2,4,8}
        msg = _refuses(engine.run_install)
        assert "[ISOEXEC-CLAIMS]" in msg and "OUTSIDE the contract's claims" in msg
        assert "outside the proven domain" in msg and "16" in msg
        assert not ran, "the refusal must land BEFORE any install runs"
        rec = enforce.ledger().records["domain_check:TP"][-1]
        assert rec.phase == enforce.INSTALL and rec.result == "violation"
        assert "deployed 16" in rec.evidence and "proven domain" in rec.evidence


def test_perturbed_stamp_refuses_at_weight_sync():
    with _fresh():
        engine = ad.set_process_adapter(_mk_engine(_stub_install("engine")))
        assert engine.run_install() is True
        stamp = pc.contract_hash()
        bad = stamp[:-1] + ("0" if stamp[-1] != "0" else "1")
        msg = _refuses(engine.on_weight_sync, _stamp(bad))
        assert "MISMATCH" in msg and "split-brain" in msg
        rec = enforce.ledger().records["handshake:numerical_policy"][-1]
        assert rec.phase == enforce.WEIGHT_SYNC and rec.result == "violation"


def test_wrong_impl_refuses_at_first_forward():
    # gdn.state is the audit's post-promotion entry: its fingerprint exception was removed when the
    # declaration was fixed, so a mismatch now refuses at FIRST_FORWARD per the severity table.
    target = ("gdn.state", "engine_prefill")
    assert enforce.exemption_for("fingerprint:gdn.state:engine_prefill") is None
    assert enforce.SEVERITY[(enforce.FINGERPRINT, enforce.FIRST_FORWARD)] == enforce.REFUSE
    with _fresh():
        engine = ad.set_process_adapter(_mk_engine(_stub_install("engine", wrong_impl={target: "legacy_pool"})))
        assert engine.run_install() is True  # attesting the (wrong) install IS the record; INSTALL closes
        msg = _refuses(engine.on_first_forward)
        assert "fingerprint:gdn.state:engine_prefill" in msg and "violation" in msg
        rec = enforce.ledger().records["fingerprint:gdn.state:engine_prefill"][-1]
        assert rec.result == "violation" and "INSTALLED='legacy_pool'" in rec.evidence.replace('"', "'")


def test_wrong_pin_refuses_at_first_forward():
    target = ("attention.varlen", "engine_prefill")  # engine attention fingerprint is NOT excepted
    assert enforce.exemption_for("fingerprint:attention.varlen:engine_prefill") is None
    with _fresh():
        bad_pins = dict(_VIEW[target]["pinned_constants"])
        assert bad_pins["fa_version"] == 3
        bad_pins["fa_version"] = 2
        engine = ad.set_process_adapter(_mk_engine(_stub_install("engine", wrong_pins={target: bad_pins})))
        assert engine.run_install() is True
        msg = _refuses(engine.on_first_forward)
        assert "fingerprint:attention.varlen:engine_prefill" in msg
        rec = enforce.ledger().records["fingerprint:attention.varlen:engine_prefill"][-1]
        assert rec.result == "violation" and "pins" in rec.evidence


def test_suppressed_reporter_is_missing_violation_at_install_close():
    with _fresh():
        trainer = ad.set_process_adapter(_mk_trainer(_stub_install("trainer", skip={("mm", "trainer_fwd")})))
        msg = _refuses(trainer.run_install)
        assert "install_attest:mm:trainer_fwd (missing)" in msg and "never checked" in msg
        assert enforce.verdict_counts()["missing"] == 1
        assert "install_attest:mm:trainer_fwd" not in enforce.ledger().records


def test_unknown_claim_kind_refuses_through_adapter():
    ExtClaims = dataclasses.make_dataclass(
        "ExtClaims", [("provenance", tuple, dataclasses.field(default=()))], bases=(Claims,), frozen=True
    )
    with _fresh():
        claims = ExtClaims(
            topology=_CONTRACT.claims.topology,
            state=_CONTRACT.claims.state,
            tolerances=_CONTRACT.claims.tolerances,
            provenance=(SimpleNamespace(claim="unhashed prose"),),
        )
        c2 = dataclasses.replace(_CONTRACT, claims=claims)
        pc._CONTRACT, pc._VIEW = c2, build_contract_view(c2, _REG)  # this process cached a contract
        ran = []
        engine = ad.set_process_adapter(_mk_engine(lambda: ran.append(1)))
        msg = _refuses(engine.run_install)
        assert "provenance" in msg and "no registered checker" in msg.lower()
        assert not ran
        assert enforce.ledger().records["claim_check:provenance"][-1].result == "violation"


def test_attestation_divergence_breaks_the_pair():
    # Two simulated processes with IDENTICAL declarations; one installs differently. The INSTALL
    # boundary alone cannot see it (fingerprints owe FIRST_FORWARD), but the attestation digest
    # folds the divergence into the composite and the pair refuses at weight sync.
    with _fresh():
        trainer = ad.set_process_adapter(_mk_trainer(_stub_install("trainer")))
        assert trainer.run_install() is True
        assert enforce.install_attestation_digest() == "CLEAN"
        stamp = pc.contract_hash()
        t_policy = pc.cached_contract().identities.numerical_policy

        _new_process_role()
        wrong = {("moe.router", "engine_prefill"): "deterministic"}  # declared fused_o2, non-excepted
        engine = ad.set_process_adapter(_mk_engine(_stub_install("engine", wrong_impl=wrong)))
        assert engine.run_install() is True
        assert pc.cached_contract().identities.numerical_policy == t_policy, "declarations identical"
        digest = enforce.install_attestation_digest()
        assert digest != "CLEAN" and "sha256" in digest
        assert pc.contract_hash() != stamp, "a diverged install must break the composite"
        msg = _refuses(engine.on_weight_sync, _stamp(stamp))
        assert "MISMATCH" in msg


def test_every_exception_soft_at_its_target_and_nowhere_else():
    plans = {s: enforce.derive_obligation_plan(_CONTRACT, _REG, s) for s in ("trainer", "engine")}
    for ex in enforce.EXCEPTIONS:
        matches = {
            s: [ob for ob in plans[s].obligations if fnmatch.fnmatchcase(ob.obligation_id, ex.pattern)]
            for s in plans
        }
        side = "engine" if matches["engine"] else "trainer"
        matched = matches[side]
        assert matched, f"exception {ex.pattern} matches no live obligation on either side"
        phase = matched[0].phase
        targets = {ob.obligation_id for ob in matched if ob.phase == phase}
        with _fresh():
            enforce.ledger().plans[side] = plans[side]
            for ob in plans[side].obligations:
                if ob.phase == phase and ob.obligation_id not in targets:
                    enforce.report(ob.obligation_id, phase, "ok", "test")
            assert enforce.close_phase(phase, side) is True, f"{ex.pattern}: its own target must stay soft"
            assert enforce.verdict_counts()["excepted"] >= len(targets)
        neighbor = next(
            ob
            for ob in plans[side].obligations
            if ob.phase == phase
            and ob.kind == matched[0].kind
            and enforce.exemption_for(ob.obligation_id) is None
            and enforce.severity(ob) == enforce.REFUSE
        )
        with _fresh():
            enforce.ledger().plans[side] = plans[side]
            for ob in plans[side].obligations:
                if ob.phase == phase and ob.obligation_id != neighbor.obligation_id:
                    enforce.report(ob.obligation_id, phase, "ok", "test")
            msg = _refuses(enforce.close_phase, phase, side)
            assert neighbor.obligation_id in msg, f"{ex.pattern} must not soften neighbor {neighbor.obligation_id}"


def test_strictness_demotes_each_refusal_to_warn_only():
    # (a) topology outside the domain: the sequence completes, the INSTALL close returns False.
    with _fresh():
        os.environ[STRICT_ENV] = "0"
        engine = ad.set_process_adapter(_mk_engine(_stub_install("engine"), tp=16))
        assert engine.run_install() is False
        assert enforce.ledger().records["domain_check:TP"][-1].result == "violation"
    # (b) perturbed stamp: violation recorded, weight-sync close returns False instead of raising.
    with _fresh():
        os.environ[STRICT_ENV] = "0"
        engine = ad.set_process_adapter(_mk_engine(_stub_install("engine")))
        assert engine.run_install() is True
        stamp = pc.contract_hash()
        assert engine.on_weight_sync(_stamp(stamp[:-1] + ("0" if stamp[-1] != "0" else "1"))) is False
        assert enforce.ledger().records["handshake:numerical_policy"][-1].result == "violation"
    # (c) wrong impl: FIRST_FORWARD close returns False instead of raising.
    with _fresh():
        os.environ[STRICT_ENV] = "0"
        wrong = {("gdn.state", "engine_prefill"): "legacy_pool"}
        engine = ad.set_process_adapter(_mk_engine(_stub_install("engine", wrong_impl=wrong)))
        assert engine.run_install() is True
        assert engine.on_first_forward() is False
    # (d) suppressed reporter: INSTALL close returns False instead of raising.
    with _fresh():
        os.environ[STRICT_ENV] = "0"
        trainer = ad.set_process_adapter(_mk_trainer(_stub_install("trainer", skip={("mm", "trainer_fwd")})))
        assert trainer.run_install() is False
    # (e) unknown claim kind: error-logged, the sequence still runs, the violation is on record.
    with _fresh():
        os.environ[STRICT_ENV] = "0"
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
        pc._CONTRACT, pc._VIEW = c2, build_contract_view(c2, _REG)
        # Seeding the cache bypasses the build-time build_valid report; supply it as a build would.
        enforce.report("build_valid:contract", enforce.INSTALL, "ok", "seeded synthetic contract")
        engine = ad.set_process_adapter(_mk_engine(_stub_install("engine")))
        assert engine.run_install() is True
        assert enforce.ledger().records["claim_check:provenance"][-1].result == "violation"


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
