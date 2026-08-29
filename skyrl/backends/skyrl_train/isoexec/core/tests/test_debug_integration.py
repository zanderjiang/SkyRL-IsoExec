"""Debug-mode integration: flags registered, adapter arms hooks, and -- the part that decides
whether debug mode is usable at all -- EVERY refusal demoting at the real call site.

The demotion is only worth anything end to end: ``on_weight_sync`` demotes the handshake and then
closes WEIGHT_SYNC on the record the demotion just wrote, so a test that exercises
``assert_contract_agreement`` alone proves nothing about the run. The cases below drive the actual
adapters, and each one asserts BOTH halves: execution continues, and the ledger is exactly as red
as strict mode would have left it.
"""

import os
import unittest
from unittest import mock

from skyrl.backends.skyrl_train.isoexec.core import enforce
from skyrl.backends.skyrl_train.isoexec.core import flags as flags_mod
from skyrl.backends.skyrl_train.isoexec.core import process_contract
from skyrl.backends.skyrl_train.isoexec.core.tests.test_adapter_negative import (
    _CONTRACT,
    _fresh,
    _mk_engine,
    _stamp,
    _stub_install,
)

DEBUG_ENV = "SKYRL_ISOEXEC_DEBUG_TRACE"
STRICT_ENV = "SKYRL_ISOEXEC_MANIFEST_STRICT"


def _armed(env):
    """Debug tracing on, strictness left at its fail-closed default."""
    env[DEBUG_ENV] = "/tmp/isoexec-test-trace"
    env[STRICT_ENV] = "1"
    return env


def _statuses(oid):
    return [r.result for r in enforce.ledger().records.get(oid, ())]


class TestDebugFlagsRegistered(unittest.TestCase):
    def test_all_debug_flags_in_registry(self):
        names = {f.name for f in flags_mod.FLAGS}
        for name in (
            "SKYRL_ISOEXEC_DEBUG_TRACE",
            "SKYRL_ISOEXEC_DEBUG_SIDE",
            "SKYRL_ISOEXEC_DEBUG_SAMPLE",
            "SKYRL_ISOEXEC_DEBUG_LADDER",
            "SKYRL_ISOEXEC_DEBUG_RING",
            "SKYRL_ISOEXEC_DEBUG_SEGMENTS",
        ):
            self.assertIn(name, names)

    def test_trace_forwarded_to_both_actors(self):
        flag = next(f for f in flags_mod.FLAGS if f.name == "SKYRL_ISOEXEC_DEBUG_TRACE")
        self.assertEqual(set(flag.forwarded_by), {flags_mod.TRAIN, flags_mod.ENGINE})


class TestHandshakeDemotion(unittest.TestCase):
    def _mismatch(self):
        with mock.patch.object(process_contract, "contract_hash", return_value="aa" * 32):
            return process_contract.assert_contract_agreement("bb" * 32, other_side="test")

    def test_strict_refuses_without_debug(self):
        env = {"SKYRL_ISOEXEC_MANIFEST_STRICT": "1"}
        with mock.patch.dict(os.environ, env, clear=False):
            os.environ.pop("SKYRL_ISOEXEC_DEBUG_TRACE", None)
            with self.assertRaises(RuntimeError):
                self._mismatch()

    def test_debug_trace_demotes_to_warn(self):
        env = {"SKYRL_ISOEXEC_MANIFEST_STRICT": "1", "SKYRL_ISOEXEC_DEBUG_TRACE": "/tmp/t"}
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertFalse(self._mismatch())


class TestRealCallSiteDemotion(unittest.TestCase):
    """The demotion at the sites a run actually goes through, not at the helper in isolation."""

    def setUp(self):
        # These cases arm debug tracing to get the demotion, not to trace: real region hooks are
        # process-global and idempotent, so leaving them installed changes what every later test in
        # the session sees. debug/tests owns the hook installation itself.
        patcher = mock.patch("skyrl.backends.skyrl_train.isoexec.debug.install.install_debug_hooks", return_value=0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_on_weight_sync_with_a_mismatched_hash_returns_and_records(self):
        with _fresh():
            os.environ.update(_armed({}))
            engine = _mk_engine(install_fn=_stub_install("engine"))
            engine.run_install()
            # Returns rather than raising: the handshake demotes AND the WEIGHT_SYNC close that
            # reads its record demotes too (the self-defeating half of the old wiring).
            self.assertIs(engine.on_weight_sync(_stamp("ff" * 32)), False)
            self.assertIn(enforce.VIOLATION, _statuses("handshake:numerical_policy"))
            self.assertGreaterEqual(enforce.verdict_counts()["refused"], 1)

    def test_on_weight_sync_still_refuses_without_debug_trace(self):
        with _fresh():
            os.environ[STRICT_ENV] = "1"
            engine = _mk_engine(install_fn=_stub_install("engine"))
            engine.run_install()
            with self.assertRaises(RuntimeError):
                engine.on_weight_sync(_stamp("ff" * 32))

    def test_claims_violation_survives_run_install_under_debug(self):
        # A debug run started to localize a kernel mix must REACH the trace: deployed outside the
        # claimed domain, run_install used to die before install() ever ran.
        with _fresh():
            os.environ.update(_armed({}))
            ran = []
            engine = _mk_engine(install_fn=lambda: ran.append("install"), tp=16)
            self.assertIs(engine.run_install(), False)
            self.assertEqual(ran, ["install"])
            self.assertIn(enforce.VIOLATION, _statuses("domain_check:TP"))

    def test_unknown_claim_kind_demotes_under_debug(self):
        import dataclasses

        from skyrl.backends.skyrl_train.isoexec.contract import Claims
        from skyrl.backends.skyrl_train.isoexec.core import adapter as ad

        gremlin = dataclasses.make_dataclass(
            "Gremlins", [("gremlins", tuple, dataclasses.field(default=()))], bases=(Claims,), frozen=True
        )
        claims = gremlin(
            topology=_CONTRACT.claims.topology,
            state=_CONTRACT.claims.state,
            tolerances=_CONTRACT.claims.tolerances,
            gremlins=("boo",),
        )
        contract = dataclasses.replace(_CONTRACT, claims=claims)
        with _fresh():
            os.environ.update(_armed({}))
            ad.check_all_claims(contract, {"TP": 8, "PP": 1, "CP": 1, "SP": 0}, "engine")
            self.assertIn(enforce.VIOLATION, _statuses("claim_check:gremlins"))

    def test_first_forward_boundary_demotes_on_a_wrong_impl(self):
        with _fresh():
            os.environ.update(_armed({}))
            key = ("attention.varlen", "engine_prefill")
            engine = _mk_engine(install_fn=_stub_install("engine", wrong_impl={key: "IMPOSTOR"}))
            engine.run_install()
            self.assertIs(engine.on_first_forward(), False)
            self.assertIn(enforce.VIOLATION, _statuses("fingerprint:attention.varlen:engine_prefill"))

    def test_debug_hook_install_failure_never_fails_the_run(self):
        with _fresh():
            os.environ.update(_armed({}))
            engine = _mk_engine(install_fn=_stub_install("engine"))
            with mock.patch(
                "skyrl.backends.skyrl_train.isoexec.debug.install.install_debug_hooks",
                side_effect=RuntimeError("hook install failed"),
            ):
                self.assertIs(engine.run_install(), True)
            self.assertIn(("engine", enforce.INSTALL), enforce.ledger().closed)


class TestAdapterArmsHooks(unittest.TestCase):
    def test_install_debug_trace_stamps_side_and_installs(self):
        from skyrl.backends.skyrl_train.isoexec.core.adapter import ContractAdapter

        adapter = ContractAdapter.__new__(ContractAdapter)
        adapter.side = "engine"
        with mock.patch.dict(os.environ, {"SKYRL_ISOEXEC_DEBUG_TRACE": "/tmp/t"}, clear=False):
            with mock.patch(
                "skyrl.backends.skyrl_train.isoexec.debug.install.install_debug_hooks", return_value=3
            ) as hooks:
                adapter._install_debug_trace()
                self.assertEqual(os.environ.get("SKYRL_ISOEXEC_DEBUG_SIDE"), "engine")
        hooks.assert_called_once()

    def test_inert_when_unset(self):
        from skyrl.backends.skyrl_train.isoexec.core.adapter import ContractAdapter

        adapter = ContractAdapter.__new__(ContractAdapter)
        adapter.side = "trainer"
        os.environ.pop("SKYRL_ISOEXEC_DEBUG_TRACE", None)
        with mock.patch("skyrl.backends.skyrl_train.isoexec.debug.install.install_debug_hooks") as hooks:
            adapter._install_debug_trace()
        hooks.assert_not_called()


class TestCudaGraphRefusal(unittest.TestCase):
    """Debug tracing + CUDA-graph decode is refused at init, and not demotable.

    Every other refusal in debug mode demotes (enforce.demoted() is true precisely because debug
    tracing is armed). This one cannot: its precondition IS debug mode, so demoting it would make
    it a no-op every time it fired. The failure it prevents is also not a crash -- it is a trace
    that reads clean because the decode half of the forward was never recorded.
    """

    CG = "SKYRL_ISOEXEC_ENABLE_CUDAGRAPH"

    def _adapter(self, side):
        from skyrl.backends.skyrl_train.isoexec.core.adapter import ContractAdapter

        a = ContractAdapter.__new__(ContractAdapter)
        a.side = side
        return a

    def test_engine_tracing_with_cudagraph_is_a_hard_error(self):
        adapter = self._adapter("engine")
        env = {"SKYRL_ISOEXEC_DEBUG_TRACE": "/tmp/t", self.CG: "1"}
        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch(
                "skyrl.backends.skyrl_train.isoexec.debug.install.install_debug_hooks", return_value=3
            ) as hooks:
                with self.assertRaises(RuntimeError) as ctx:
                    adapter._install_debug_trace()
        msg = str(ctx.exception)
        self.assertIn("SKYRL_ISOEXEC_ENABLE_CUDAGRAPH", msg)
        self.assertIn("SKYRL_ISOEXEC_DEBUG_TRACE", msg)
        self.assertIn("NO trace records", msg)
        hooks.assert_not_called()  # refused BEFORE anything was installed

    def test_demotion_does_not_soften_it(self):
        """enforce.demoted() is true here; the refusal must still raise."""
        adapter = self._adapter("engine")
        env = {"SKYRL_ISOEXEC_DEBUG_TRACE": "/tmp/t", self.CG: "1", STRICT_ENV: "1"}
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertTrue(enforce.demoted())
            with self.assertRaises(RuntimeError):
                adapter._install_debug_trace()

    def test_cudagraph_without_tracing_is_fine(self):
        adapter = self._adapter("engine")
        with mock.patch.dict(os.environ, {self.CG: "1"}, clear=False):
            os.environ.pop("SKYRL_ISOEXEC_DEBUG_TRACE", None)
            adapter._install_debug_trace()  # no raise

    def test_trainer_side_is_unaffected(self):
        """The flag is engine-only; a trainer process that inherits it must still trace."""
        adapter = self._adapter("trainer")
        env = {"SKYRL_ISOEXEC_DEBUG_TRACE": "/tmp/t", self.CG: "1"}
        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch(
                "skyrl.backends.skyrl_train.isoexec.debug.install.install_debug_hooks", return_value=2
            ) as hooks:
                adapter._install_debug_trace()
        hooks.assert_called_once()


if __name__ == "__main__":
    unittest.main()
