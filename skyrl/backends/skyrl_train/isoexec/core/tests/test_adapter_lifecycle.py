"""End-to-end ContractAdapter lifecycle, in-process and CPU-only: the complete trainer flow, the
complete engine flow, and the paired handshake -- every phase boundary crossed through the REAL
adapter sequence (build -> check_all_claims -> install -> INSTALL close -> first forward ->
weight sync -> gate), with only the GPU-touching installers stubbed. The final enforcement.json
and the [ISOEXEC-ENFORCE] summary counts are asserted, not just the return values.
"""

import json
import logging
import os
import pickle
import re
import tempfile
from types import SimpleNamespace

from skyrl.backends.skyrl_train.isoexec.core import adapter as ad
from skyrl.backends.skyrl_train.isoexec.core import arch as arch_mod
from skyrl.backends.skyrl_train.isoexec.core import enforce
from skyrl.backends.skyrl_train.isoexec.core import fingerprint as fp
from skyrl.backends.skyrl_train.isoexec.core import process_contract as pc
from skyrl.backends.skyrl_train.isoexec.core.registry_build import build_registry
from skyrl.backends.skyrl_train.isoexec.models import qwen3_5
from skyrl.backends.skyrl_train.isoexec.runtimes.megatron.adapter import (
    MegatronContractAdapter,
)
from skyrl.backends.skyrl_train.isoexec.runtimes.vllm.adapter import VLLMContractAdapter

# torch-first, as in production: workers load torch long before any isoexec module.
from skyrl.backends.skyrl_train.weight_sync.cuda_ipc_strategy import CudaIpcInitInfo

# CPU-only harness (a live production run owns the GPUs): the real adapter build path reads
# core/arch.ARCH, so point it at the production accelerator instead of the sentinel.
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


class _fresh:
    """Reset ledger + recorder + cached contract + process adapter AND strip every ISOEXEC env var
    (restored on exit), so the real in-test contract build reflects code defaults."""

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
    """Within one paired test: forget everything per-process, as a fresh worker process would."""
    enforce._reset_for_tests()
    ad._reset_for_tests()
    fp._RECORDER = None
    fp._LOGGED_TAGS.clear()
    pc._CONTRACT, pc._VIEW = None, None


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


def _stub_install(side):
    """The GPU work stubbed, the reporting real: record every contract-named install for this side
    exactly as the runtime installers do, then the install-time fingerprint pass."""

    def _install():
        view = pc.cached_contract_view()
        for (op, site), meta in sorted(view.items()):
            if site.startswith(side):
                fp.record_install(op, site, meta["impl_id"], pinned=meta["pinned_constants"])
        fp.log_fingerprint_once(view, tag=f"{side}_install")

    return _install


class _capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.msgs = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


def _summary_counts(msgs, side):
    line = [m for m in msgs if "[ISOEXEC-ENFORCE]" in m and f"side={side} ok=" in m][-1]
    return {k: int(v) for k, v in re.findall(r"(ok|refused|logged|missing|excepted)=(\d+)", line)}


def test_trainer_full_lifecycle_green():
    from skyrl.train.utils import trainer_utils as tu

    with _fresh():
        with tempfile.TemporaryDirectory() as d:
            os.environ["ISOEXEC_CONTRACT_PATH"] = os.path.join(d, "contract.json")
            cap = _capture()
            enforce.logger.addHandler(cap)
            try:
                adapter = ad.set_process_adapter(_mk_trainer(_stub_install("trainer")))
                assert adapter.run_install() is True
                assert pc.cached_contract().identities == _CONTRACT.identities
                assert ("trainer", enforce.INSTALL) in enforce.ledger().closed
                assert os.path.exists(os.path.join(d, "contract.json"))

                assert adapter.on_first_forward() is True
                assert "trainer_first_forward" in fp._LOGGED_TAGS
                assert enforce.ledger().records["installed_backstop:first_forward"][-1].result == "ok"

                stamp = pc.contract_hash()
                assert stamp is not None
                assert adapter.on_weight_sync(stamp) is True

                P = tu.ISOEXEC_FORWARD_GATE_PREFIX
                good = {f"{P}_mean": 1e-6, f"{P}_max": 1e-5, f"{P}_min": 0.0, f"{P}_std": 0.0}
                assert tu.validate_isoexec_forward_gate(good, enabled=True, scoring_audit_skipped=False) is True
                plan = enforce.ledger().plans["trainer"]
                for ob in plan.obligations:
                    if ob.kind == enforce.SERVED:
                        enforce.report(ob.obligation_id, enforce.STEP1, enforce.OK, "simulated counter")
                assert enforce.close_phase(enforce.STEP1, "trainer") is True

                counts = enforce.verdict_counts()
                n_exc = sum(1 for ob in plan.obligations if enforce.exemption_for(ob.obligation_id))
                assert counts == {
                    "ok": len(plan.obligations) - n_exc,
                    "refused": 0,
                    "logged": 0,
                    "missing": 0,
                    "excepted": n_exc,
                }
                assert _summary_counts(cap.msgs, "trainer") == counts

                # One verdict file per process now; this fixture is one process.
                (art,) = enforce.verdict_artifacts(d)
                with open(art) as fh:
                    loaded = json.load(fh)
                assert os.path.basename(art).startswith("enforcement.trainer.r")
                assert loaded["counts"] == counts
                assert loaded["phases_closed"] == sorted(f"trainer:{p}" for p in enforce.PHASES)
                by_id = {o["id"]: o for o in loaded["obligations"]}
                assert len(by_id) == len(plan.obligations)
                assert all(o["status"] == "ok" for o in by_id.values())
                hs = by_id["handshake:numerical_policy"]
                assert f"stamped composite={stamp}" in hs["records"][-1]["evidence"]
                assert by_id["gate:engine_decode|trainer_score"]["phase"] == enforce.STEP1
                assert loaded["exceptions"] and all(e["reason"] and e["removal"] for e in loaded["exceptions"])
            finally:
                enforce.logger.removeHandler(cap)


def test_engine_full_lifecycle_green():
    with _fresh():
        with tempfile.TemporaryDirectory() as d:
            os.environ["ISOEXEC_CONTRACT_PATH"] = os.path.join(d, "contract.json")
            cap = _capture()
            enforce.logger.addHandler(cap)
            try:
                adapter = ad.set_process_adapter(_mk_engine(_stub_install("engine")))
                assert adapter.run_install() is True
                assert ("engine", enforce.INSTALL) in enforce.ledger().closed

                assert adapter.on_first_forward() is True
                assert "engine_first_forward" in fp._LOGGED_TAGS
                assert enforce.ledger().records["installed_backstop:first_forward"][-1].result == "ok"

                stamp = pc.contract_hash()
                info = pickle.loads(
                    pickle.dumps(
                        CudaIpcInitInfo(
                            override_existing_receiver=False, model_dtype_str="bfloat16", contract_hash=stamp
                        )
                    )
                )
                assert info.contract_hash == stamp
                assert adapter.on_weight_sync(info) is True
                rec = enforce.ledger().records["handshake:numerical_policy"][-1]
                assert rec.result == "ok" and "MATCH" in rec.evidence

                plan = enforce.ledger().plans["engine"]
                closed = enforce.ledger().closed
                open_obs = [ob for ob in plan.obligations if ("engine", ob.phase) not in closed]
                assert open_obs and all(ob.phase == enforce.STEP1 for ob in open_obs)
                n_exc = sum(
                    1
                    for ob in plan.obligations
                    if ("engine", ob.phase) in closed and enforce.exemption_for(ob.obligation_id)
                )
                counts = enforce.verdict_counts()
                assert counts == {
                    "ok": len(plan.obligations) - len(open_obs) - n_exc,
                    "refused": 0,
                    "logged": 0,
                    "missing": 0,
                    "excepted": n_exc,
                }
                assert _summary_counts(cap.msgs, "engine") == counts

                (art,) = enforce.verdict_artifacts(d)
                with open(art) as fh:
                    loaded = json.load(fh)
                assert os.path.basename(art).startswith("enforcement.engine.r")
                assert loaded["counts"] == counts
                assert loaded["phases_closed"] == sorted(
                    f"engine:{p}" for p in (enforce.INSTALL, enforce.FIRST_FORWARD, enforce.WEIGHT_SYNC)
                )
                statuses = {o["id"]: o["status"] for o in loaded["obligations"]}
                assert statuses["gate:engine_decode|trainer_score"] == "open"
                assert statuses["handshake:numerical_policy"] == "ok"
            finally:
                enforce.logger.removeHandler(cap)


def test_paired_trainer_engine_flow():
    with _fresh():
        trainer = ad.set_process_adapter(_mk_trainer(_stub_install("trainer")))
        assert trainer.run_install() is True
        assert trainer.on_first_forward() is True
        stamp = pc.contract_hash()
        assert trainer.on_weight_sync(stamp) is True
        t_counts = enforce.verdict_counts()
        assert t_counts["refused"] == 0 and t_counts["missing"] == 0 and t_counts["logged"] == 0
        t_policy = pc.cached_contract().identities.numerical_policy
        assert enforce.install_attestation_digest() == "CLEAN"

        _new_process_role()
        engine = ad.set_process_adapter(_mk_engine(_stub_install("engine")))
        assert engine.run_install() is True
        assert engine.on_first_forward() is True
        # Identical declarations -> identical identity; identical clean installs -> identical composite.
        assert pc.cached_contract().identities.numerical_policy == t_policy
        assert enforce.install_attestation_digest() == "CLEAN"
        assert pc.contract_hash() == stamp
        info = pickle.loads(
            pickle.dumps(
                CudaIpcInitInfo(override_existing_receiver=False, model_dtype_str="bfloat16", contract_hash=stamp)
            )
        )
        assert engine.on_weight_sync(info) is True
        rec = enforce.ledger().records["handshake:numerical_policy"][-1]
        assert rec.result == "ok" and "MATCH" in rec.evidence and stamp in rec.evidence
        e_counts = enforce.verdict_counts()
        assert e_counts["refused"] == 0 and e_counts["missing"] == 0 and e_counts["logged"] == 0


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
