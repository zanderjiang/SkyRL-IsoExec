"""The wiring between the enforcement core and the two runtimes: one owner per fact.

Each case pins a place where two files could answer the same question differently, or where a
fact the contract hashes was never read back off the running process.
"""

import os
import unittest
from unittest import mock

from skyrl.backends.skyrl_train.isoexec.core import adapter as ad
from skyrl.backends.skyrl_train.isoexec.core import arch as arch_mod
from skyrl.backends.skyrl_train.isoexec.core import flags as flags_mod
from skyrl.backends.skyrl_train.isoexec.core import gdn_kernel_env
from skyrl.backends.skyrl_train.isoexec.core.contract_delivery import _primary_op
from skyrl.backends.skyrl_train.isoexec.core.fingerprint import pin_disagreements
from skyrl.backends.skyrl_train.isoexec.core.registry_build import build_registry

if arch_mod.ARCH == arch_mod.NON_ACCELERATOR_ARCH:
    arch_mod.ARCH = "sm90"

KERNEL_ENV = gdn_kernel_env.KERNEL_ENV
TRAINER_KERNEL_ENV = gdn_kernel_env.TRAINER_KERNEL_ENV


class _env:
    """Set/clear env vars around one case."""

    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        self._saved = {k: os.environ.get(k) for k in self.kw}
        for k, v in self.kw.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        return self

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        return False


class TestGdnKernelSingleOwner(unittest.TestCase):
    """The executing sites parse the env through the same module the declaration site does."""

    def _ops(self):
        from skyrl.backends.skyrl_train.isoexec.ops.gdn import gdn_ops

        return gdn_ops

    def test_unset_executes_the_declarable_default(self):
        # "chunk" was once the executing default while no model declared a chunk composition,
        # so unset named a function the process did not run.
        with _env(**{KERNEL_ENV: None}):
            self.assertEqual(gdn_kernel_env.DEFAULT_KERNEL, "recurrent")
            self.assertEqual(self._ops().gdn_kernel_mode(), "recurrent")
            self.assertTrue(self._ops().recurrent_mode())

    def test_case_folding_and_vocabulary_are_the_shared_ones(self):
        with _env(**{KERNEL_ENV: "CPR"}):
            self.assertEqual(self._ops().gdn_kernel_mode(), "cpr")
            self.assertTrue(self._ops().cpr_mode())
        with _env(**{KERNEL_ENV: "bogus"}):
            with self.assertRaises(ValueError) as cm:
                self._ops().gdn_kernel_mode()
            self.assertIn(str(list(gdn_kernel_env.KERNELS)), str(cm.exception))

    def test_trainer_override_goes_through_the_shared_parser(self):
        with _env(**{TRAINER_KERNEL_ENV: "  "}):
            self.assertIsNone(gdn_kernel_env.gdn_trainer_kernel_override())
        with _env(**{TRAINER_KERNEL_ENV: "bogus"}):
            self.assertRaises(ValueError, gdn_kernel_env.gdn_trainer_kernel_override)
        self.assertIn("gdn_trainer_kernel_override", open(self._ops().__file__).read())


class TestFlagTable(unittest.TestCase):
    def _flag(self, name):
        return next(f for f in flags_mod.FLAGS if f.name == name)

    def test_gdn_kernel_flag_states_the_shared_default_and_vocabulary(self):
        flag = self._flag(KERNEL_ENV)
        self.assertEqual(flag.default, gdn_kernel_env.DEFAULT_KERNEL)
        for kernel in gdn_kernel_env.KERNELS:
            self.assertIn(kernel, flag.selects)

    def test_debug_segments_is_registered_and_forwarded(self):
        flag = self._flag("SKYRL_ISOEXEC_DEBUG_SEGMENTS")
        self.assertEqual(flag.disposition, flags_mod.DIAGNOSTIC)
        self.assertEqual(flag.sides, ("both",))
        self.assertEqual(set(flag.forwarded_by), {flags_mod.TRAIN, flags_mod.ENGINE})

    def test_debug_side_is_not_a_latent_split_brain(self):
        # It is stamped per process by the adapter; forwarding a launch-shell value would label the
        # engine's records with the trainer's side, which IS the bug.
        flag = self._flag("SKYRL_ISOEXEC_DEBUG_SIDE")
        self.assertFalse(flag.should_forward)
        self.assertFalse(flag.is_latent_split_brain)
        self.assertNotIn(flag.name, flags_mod.actor_forwarding_list(None))
        self.assertNotIn(flag.name, [f.name for f in flags_mod.latent_split_brain()])


class TestHookRefResolution(unittest.TestCase):
    """StateHookChecker judges through core/claim_refs, so build, runtime and CI answer alike."""

    def _check(self, ref):
        from types import SimpleNamespace

        return ad.CLAIM_CHECKERS["state"].check(SimpleNamespace(ref=ref, state_id="x"), {})

    def test_escaping_ref_is_a_violation(self):
        from skyrl.backends.skyrl_train.isoexec.core import enforce

        r = self._check("../../../../etc/passwd::root")
        self.assertEqual(r.result, enforce.VIOLATION)
        self.assertIn("escapes the isoexec package", r.evidence)

    def test_missing_file_and_missing_symbol_keep_their_messages(self):
        self.assertIn("ref file 'core/nope.py' missing", self._check("core/nope.py::f").evidence)
        self.assertIn("not defined in", self._check("core/enforce.py::not_a_symbol").evidence)

    def test_a_real_hook_resolves(self):
        from skyrl.backends.skyrl_train.isoexec.core import enforce

        self.assertEqual(self._check("core/enforce.py::close_phase").result, enforce.OK)


class TestLivePins(unittest.TestCase):
    """The pinned constants the contract hashes, read back off the running install."""

    @classmethod
    def setUpClass(cls):
        from skyrl.backends.skyrl_train.isoexec.models import qwen3_5

        cls.reg = build_registry(strict=True)
        cls.contract = qwen3_5.build(cls.reg, arch="sm90", profile=qwen3_5.PROFILE)

    def _contract_pins(self, op):
        for e in self.contract.composition:
            if _primary_op(self.reg, e) == op:
                return {k: v for k, v in e.constants}
        raise AssertionError(f"{op} not in the contract")

    def test_live_pins_cover_every_key_the_contract_pins(self):
        # A partial pin report is worse than none: pin_disagreements calls an unreported key
        # "install recorded nothing", so a live reader must answer for all of them or for none.
        from types import SimpleNamespace

        self.assertEqual(set(ad.live_pins("moe.combine")), set(self._contract_pins("moe.combine")))
        plan = SimpleNamespace(num_leaves=8, bf16_leaves=True)  # pik needs a GPU; the read does not
        with mock.patch(
            "skyrl.backends.skyrl_train.isoexec.ops.collectives.pik_tp_invariant.get_plan", return_value=plan
        ):
            got = ad.live_pins("collectives.tree_all_reduce")
        self.assertEqual(got, {"leaves": 8, "leaf_dtype": "bf16"})
        self.assertEqual(set(got), set(self._contract_pins("collectives.tree_all_reduce")))

    def test_an_unreadable_pin_degrades_to_none_instead_of_failing_the_install(self):
        with mock.patch(
            "skyrl.backends.skyrl_train.isoexec.ops.collectives.pik_tp_invariant.get_plan",
            side_effect=RuntimeError("No CUDA GPUs are available"),
        ):
            self.assertIsNone(ad.live_pins("collectives.tree_all_reduce"))

    def test_live_pins_disagree_when_the_env_moves_the_plan(self):
        with _env(SKYRL_ISOEXEC_PIK_LEAVES="4"):
            problems = pin_disagreements(self._contract_pins("moe.combine"), ad.live_pins("moe.combine"))
        self.assertEqual(len(problems), 1)
        self.assertIn("leaves", problems[0])

    def test_live_pins_is_none_where_the_pins_are_call_site_literals(self):
        self.assertIsNone(ad.live_pins("attention.varlen"))

    def test_unreported_pins_are_named_rather_than_silently_unverified(self):
        from skyrl.backends.skyrl_train.isoexec.core import fingerprint as fp
        from skyrl.backends.skyrl_train.isoexec.core.process_contract import (
            build_contract_view,
        )

        saved = fp._RECORDER
        fp._RECORDER = None
        try:
            fp.record_install("attention.varlen", "trainer_fwd", "varlen_custom")
            fp.record_install(
                "collectives.tree_all_reduce", "trainer_fwd", "pik_tree", pinned={"leaves": 8, "leaf_dtype": "bf16"}
            )
            gaps = ad.log_unreported_pins(build_contract_view(self.contract, self.reg))
        finally:
            fp._RECORDER = saved
        self.assertEqual(sorted(gaps), ["attention.varlen:trainer_fwd"])
        self.assertEqual(gaps["attention.varlen:trainer_fwd"], ["num_splits"])


class TestEngineInstallPins(unittest.TestCase):
    """The engine's GDN recorders answer for exactly the keys the contract pins.

    ``log_fingerprint`` compares pins only when the recorder reported some, so an impl id alone
    leaves ``gdn.core``'s ``kernel`` with no live cross-check. Read from source: vLLM-only site.
    """

    @classmethod
    def setUpClass(cls):
        import ast
        import pathlib

        from skyrl.backends.skyrl_train.isoexec.models import qwen3_5

        cls.reg = build_registry(strict=True)
        cls.qwen3_5 = qwen3_5
        src = pathlib.Path(__file__).resolve().parents[2] / "runtimes" / "vllm" / "gdn_gptmodel.py"
        cls.tree = ast.parse(src.read_text())
        cls.ast = ast

    def _pinned_arg(self, op):
        """The ``pinned=`` expression of the engine's ``record_installs`` call for ``op``."""
        for node in self.ast.walk(self.tree):
            if not isinstance(node, self.ast.Call) or getattr(node.func, "id", None) != "record_installs":
                continue
            first = node.args[0] if node.args else None
            if not (isinstance(first, self.ast.Constant) and first.value == op):
                continue
            for kw in node.keywords:
                if kw.arg == "pinned":
                    return kw.value
        return None

    def _dict_keys(self, node):
        """The literal key sets of ``node``, following one level of local name binding."""
        if isinstance(node, self.ast.Dict):
            return [{k.value for k in node.keys}]
        name = getattr(node, "id", None)
        sets = []
        for a in self.ast.walk(self.tree):
            if isinstance(a, self.ast.Assign) and any(getattr(t, "id", None) == name for t in a.targets):
                sets.extend(self._dict_keys(a.value))
        return sets

    def _contract_pin_keys(self, contract, op):
        for e in contract.composition:
            if _primary_op(self.reg, e) == op:
                return {k for k, _v in e.constants}
        raise AssertionError(f"{op} not in the contract")

    def test_engine_gdn_core_records_the_kernel_pin(self):
        pinned = self._pinned_arg("gdn.core")
        self.assertIsNotNone(pinned, "the engine gdn.core recorder must report pins, not an impl id alone")
        self.assertEqual(self._dict_keys(pinned), [{"kernel"}])
        for profile in self.qwen3_5.PROFILE_BY_KERNEL.values():
            c = self.qwen3_5.build(self.reg, arch="sm90", profile=profile)
            self.assertEqual(self._contract_pin_keys(c, "gdn.core"), {"kernel"})

    def test_engine_gdn_state_records_every_key_the_variant_pins(self):
        # A partial pin report reads as "install recorded nothing" for the missing key, so the
        # recorder's per-variant key sets must be the contract's per-variant key sets.
        pinned = self._pinned_arg("gdn.state")
        self.assertIsNotNone(pinned, "the engine gdn.state recorder must report pins")
        recorded = {frozenset(k) for k in self._dict_keys(pinned)}
        declared = {
            frozenset(self._contract_pin_keys(self.qwen3_5.build(self.reg, arch="sm90", profile=p), "gdn.state"))
            for p in self.qwen3_5.PROFILE_BY_KERNEL.values()
        }
        self.assertEqual(recorded, declared)


class TestCollectivePinGuards(unittest.TestCase):
    """A pin is reported only when the thing it pins was installed.

    The ReductionPlan exists whether or not pik was installed, so reporting it beside a
    ``not_installed`` impl id claims a leaf tree the process is not running.
    """

    @classmethod
    def setUpClass(cls):
        import ast
        import pathlib

        iso = pathlib.Path(__file__).resolve().parents[2]
        cls.ast = ast
        cls.sources = {
            "engine": iso / "runtimes" / "vllm" / "gptmodel_vllm.py",
            "trainer": iso.parent / "workers" / "megatron" / "megatron_worker.py",
        }

    def test_tree_all_reduce_pins_are_guarded_on_pik_enabled(self):
        for side, path in self.sources.items():
            tree = self.ast.parse(path.read_text())
            calls = [
                n
                for n in self.ast.walk(tree)
                if isinstance(n, self.ast.Call)
                and getattr(n.func, "id", None) == "record_installs"
                and n.args
                and isinstance(n.args[0], self.ast.Constant)
                and n.args[0].value == "collectives.tree_all_reduce"
            ]
            self.assertEqual(len(calls), 1, side)
            pinned = next(kw.value for kw in calls[0].keywords if kw.arg == "pinned")
            self.assertIsInstance(pinned, self.ast.IfExp, f"{side}: pins must be conditional on the install")
            self.assertEqual(getattr(pinned.test.func, "id", None), "pik_enabled", side)
            self.assertIsNone(pinned.orelse.value, side)


class TestEngineStepKey(unittest.TestCase):
    """The engine keys its trace records to the trainer's optim_step counter."""

    def _wrapper(self):
        from skyrl.backends.skyrl_train.isoexec.runtimes.vllm.gptmodel_vllm import (
            GPTModelVLLMWrapper,
        )

        return GPTModelVLLMWrapper.__new__(GPTModelVLLMWrapper)

    def test_nth_effective_sync_reports_the_optim_step_that_produced_the_weights(self):
        # Sync 1 predates the first optim_step, so it is left unkeyed exactly as the trainer's own
        # pre-optim_step forwards are; sync N thereafter carries optim_step N-1.
        w = self._wrapper()
        with _env(SKYRL_ISOEXEC_DEBUG_TRACE="/tmp/isoexec-test-trace"):
            with mock.patch("skyrl.backends.skyrl_train.isoexec.debug.trace.set_step") as ss:
                for _ in range(4):
                    w._isoexec_debug_set_step()
        self.assertEqual([c.args[0] for c in ss.call_args_list], [1, 2, 3])

    def test_inert_and_fail_soft_when_tracing_is_off(self):
        w = self._wrapper()
        with _env(SKYRL_ISOEXEC_DEBUG_TRACE=None):
            with mock.patch("skyrl.backends.skyrl_train.isoexec.debug.trace.set_step") as ss:
                w._isoexec_debug_set_step()
        ss.assert_not_called()
        with _env(SKYRL_ISOEXEC_DEBUG_TRACE="/tmp/isoexec-test-trace"):
            with mock.patch(
                "skyrl.backends.skyrl_train.isoexec.debug.trace.set_step", side_effect=RuntimeError("boom")
            ):
                w._isoexec_debug_set_step()  # must not propagate into a weight sync
                w._isoexec_debug_set_step()


class TestTransformerEngineProbe(unittest.TestCase):
    def test_a_te_that_cannot_load_reads_as_te_absent(self):
        # A partial TE install raises OSError, not ImportError; propagating it made importing the
        # package fail on exactly the CPU-only machines the trace comparator runs on.
        from skyrl.backends.skyrl_train.isoexec.runtimes.megatron import no_te_guard

        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def _broken(name, *a, **kw):
            if name == "transformer_engine":
                raise OSError("libcublas.so.13: cannot open shared object file")
            return real_import(name, *a, **kw)

        saved = no_te_guard._INSTALLED
        no_te_guard._INSTALLED = False
        try:
            with mock.patch("builtins.__import__", _broken):
                with mock.patch.object(no_te_guard, "_bridge_root", return_value=None):
                    # Widened, but never silent: present-but-broken TE is a different fact from
                    # absent TE, and on a trainer it is a broken install.
                    with self.assertLogs(no_te_guard.logger, level="ERROR") as log:
                        self.assertIs(no_te_guard.install_no_te_guard(), False)
            self.assertIn("fails to load", "\n".join(log.output))
            self.assertIn("libcublas.so.13", "\n".join(log.output))
        finally:
            no_te_guard._INSTALLED = saved

    def test_no_megatron_installed_is_nothing_to_patch(self):
        # find_spec RAISES on an absent parent package, so the widened TE probe alone would still
        # have taken the import down on a machine with neither TE nor megatron.
        from skyrl.backends.skyrl_train.isoexec.runtimes.megatron import no_te_guard

        with mock.patch("importlib.util.find_spec", side_effect=ModuleNotFoundError("No module named 'megatron'")):
            self.assertIsNone(no_te_guard._bridge_root())


if __name__ == "__main__":
    unittest.main()
