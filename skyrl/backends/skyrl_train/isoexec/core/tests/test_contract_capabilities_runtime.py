"""Runtime-seam capabilities: extension composite, weight-sync handshake, fingerprint/view
projection, and the NCCL contract check. Shares the synthetic registry of
test_contract_capabilities.py and restores every process-global per test.
"""

import logging
import os
from contextlib import contextmanager
from types import SimpleNamespace

from skyrl.backends.skyrl_train.isoexec.core import fingerprint as fp
from skyrl.backends.skyrl_train.isoexec.core import process_contract as pc
from skyrl.backends.skyrl_train.isoexec.core.contract_build import (
    DEPLOYMENT,
    build_execution_contract,
)
from skyrl.backends.skyrl_train.isoexec.core.contract_delivery import (
    expected_installed_keys,
)
from skyrl.backends.skyrl_train.isoexec.core.registry import (
    ImplSpec,
    OpSpec,
    Registry,
    RoundingSchedule,
)
from skyrl.backends.skyrl_train.isoexec.core.tests.test_contract_capabilities import (
    ENGINE,
    MM_PINS,
    SITES,
    TRAINER,
    _build,
    _env,
    _refuses,
    _registry,
)
from skyrl.backends.skyrl_train.isoexec.models.policy import entry
from skyrl.backends.skyrl_train.isoexec.ops.collectives import nccl_identity as ni


@contextmanager
def _extensions():
    saved = dict(pc._EXTENSIONS)
    pc._EXTENSIONS.clear()
    try:
        yield
    finally:
        pc._EXTENSIONS.clear()
        pc._EXTENSIONS.update(saved)


@contextmanager
def _installed_contract(c):
    saved = (pc._CONTRACT, pc._VIEW)
    pc._CONTRACT, pc._VIEW = c, None
    try:
        yield
    finally:
        pc._CONTRACT, pc._VIEW = saved


@contextmanager
def _fresh_recorder():
    saved_r, saved_tags = fp._RECORDER, set(fp._LOGGED_TAGS)
    fp._RECORDER = None
    try:
        yield
    finally:
        fp._RECORDER = saved_r
        fp._LOGGED_TAGS.clear()
        fp._LOGGED_TAGS.update(saved_tags)


class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@contextmanager
def _captured(logger):
    h, old = _LogCapture(), logger.level
    logger.addHandler(h)
    logger.setLevel(logging.DEBUG)
    try:
        yield h
    finally:
        logger.removeHandler(h)
        logger.setLevel(old)


# --- 8. extensions / composite ---


def test_composite_hash_base_semantics():
    with _extensions():
        assert pc.composite_hash(None) is None
        assert pc.composite_hash("base") == "base"  # no extensions: the identity
        pc.register_contract_extension("a", lambda: "1")
        h = pc.composite_hash("base")
        assert h != "base" and pc.composite_hash("base") == h  # deterministic
        assert pc.composite_hash(None) is None  # no contract stays no contract


def test_alias_hits_the_same_registry():
    assert pc.register_manifest_extension is pc.register_contract_extension
    with _extensions():
        pc.register_manifest_extension("a", lambda: "1")
        via_alias = pc.composite_hash("base")
    with _extensions():
        pc.register_contract_extension("a", lambda: "1")
        assert pc.composite_hash("base") == via_alias


def test_extension_error_marker_prevents_agreement():
    def boom_value():
        raise ValueError("x")

    def boom_key():
        raise KeyError("x")

    with _extensions():
        pc.register_contract_extension("ext", boom_value)
        hv = pc.composite_hash("base")
    with _extensions():
        pc.register_contract_extension("ext", boom_key)
        hk = pc.composite_hash("base")
    with _extensions():
        pc.register_contract_extension("ext", lambda: "ok")
        hok = pc.composite_hash("base")
    assert hv != "base"  # a broken extension must not fake agreement with the plain identity
    assert len({hv, hk, hok}) == 3  # sides with different errors must not agree


def test_reregistration_same_name_last_wins():
    with _extensions():
        pc.register_contract_extension("n", lambda: "one")
        pc.register_contract_extension("n", lambda: "two")
        assert len(pc._EXTENSIONS) == 1
        last = pc.composite_hash("base")
    with _extensions():
        pc.register_contract_extension("n", lambda: "two")
        assert pc.composite_hash("base") == last


# --- 9. handshake ---


def test_strictness_env_vocabulary():
    c = _build()
    with _installed_contract(c), _extensions():
        ours = pc.contract_hash()
        assert ours == c.identities.numerical_policy
        assert pc.assert_contract_agreement(ours) is True
        for lenient in ("0", "", "false", "no", "FALSE", "No"):
            with _env(SKYRL_ISOEXEC_MANIFEST_STRICT=lenient):
                assert pc.assert_contract_agreement("f" * 64) is False
        for strict in (None, "1", "true", "anything-else"):
            with _env(SKYRL_ISOEXEC_MANIFEST_STRICT=strict):
                _refuses(RuntimeError, pc.assert_contract_agreement, "f" * 64)


def test_init_info_stamp_on_synthetic_contract():
    c = _build()
    with _installed_contract(c), _extensions(), _env(SKYRL_ISOEXEC_MANIFEST_STRICT=None):
        h = pc.contract_hash()
        assert pc.assert_init_info_contract(SimpleNamespace(contract_hash=h)) is True
        _refuses(RuntimeError, pc.assert_init_info_contract, SimpleNamespace(contract_hash="0" * 64))
        assert pc.assert_init_info_contract(SimpleNamespace()) is True  # no stamp -> skip
    with _installed_contract(None):
        assert pc.assert_init_info_contract(SimpleNamespace(contract_hash="x")) is True  # not built -> skip


def test_negative_control_flag_perturbs_composite():
    c = _build()
    with (
        _installed_contract(c),
        _extensions(),
        _env(SKYRL_ISOEXEC_HANDSHAKE_NEGATIVE_CONTROL="1", SKYRL_ISOEXEC_MANIFEST_STRICT=None),
    ):
        clean = pc.contract_hash()
        # workers/worker.py registers this extension when the flag is on.
        if os.environ.get("SKYRL_ISOEXEC_HANDSHAKE_NEGATIVE_CONTROL") == "1":
            pc.register_contract_extension("negative_control", lambda: "tampered")
        perturbed = pc.contract_hash()
        assert perturbed != clean
        _refuses(RuntimeError, pc.assert_contract_agreement, clean, other_side="engine")
    with _extensions():  # registry restored: composite is the plain identity again
        assert pc.composite_hash(c.identities.numerical_policy) == c.identities.numerical_policy


# --- 10. fingerprint / view ---


def test_contract_view_projection():
    reg = _registry()
    c = _build(reg=reg)
    view = pc.build_contract_view(c, reg)
    assert set(view) == expected_installed_keys(c, reg)  # merged multi-site entries expand per site
    mm = view[("alpha.mm", "engine_decode")]
    assert (mm["impl_id"], mm["version"], mm["half"]) == ("ref", 1, "function")
    assert mm["pinned_constants"] == {"block": 128, "leaves": 4, "eps": 1e-6}  # BitPattern decoded
    dep = view[("alpha.pin", "engine_decode")]
    assert dep == {"impl_id": "unpinned", "version": 1, "pinned_constants": {}, "half": "deployment"}
    assert ("alpha.sub", "trainer_fwd") not in view


def test_log_fingerprint_mismatches_log_error_without_raising():
    reg = _registry()
    view = pc.build_contract_view(_build(reg=reg), reg)
    with _fresh_recorder():
        fp.record_install("alpha.mm", "trainer_fwd", "ref", pinned=dict(MM_PINS))  # match
        fp.record_install("alpha.mm", "trainer_score", "twin")  # impl mismatch
        fp.record_install("alpha.mm", "engine_prefill", "ref", pinned=dict(MM_PINS, block=64))  # pin mismatch
        fp.record_install("beta.rogue", "trainer_fwd", "x")  # unnamed by the contract
        with _captured(fp.logger) as cap:
            problems = fp.log_fingerprint(view, tag="mismatch-case")
    assert len(problems) == 3
    text = "\n".join(problems)
    assert "INSTALLED='twin'" in text and "INSTALLED pins=" in text and "no such (op,site)" in text
    assert len([r for r in cap.records if r.levelno >= logging.ERROR]) == 3


def test_pin_disagreements_names_every_drifted_key():
    want = {"kernel": "recurrent", "leaves": 4, "mode": ("tree", "flat"), "fused": True}
    assert fp.pin_disagreements(want, dict(want, mode=["tree", "flat"])) == []  # JSON round-trip
    assert fp.pin_disagreements(want, dict(want, fused=1)) == ["fused: contract pins True, install used 1"]
    bad = fp.pin_disagreements(want, {"kernel": "chunk", "leaves": 4, "mode": ("tree", "flat")})
    assert bad == [
        "fused: contract pins True, install recorded nothing",
        "kernel: contract pins 'recurrent', install used 'chunk'",
    ]


def test_drifted_pin_is_caught_against_the_contract_view():
    # Guards an install binding the right impl_id under a drifted pin.
    reg = _registry()
    view = pc.build_contract_view(_build(reg=reg), reg)
    with _fresh_recorder():
        fp.record_install("alpha.mm", "trainer_fwd", "ref", pinned=dict(MM_PINS, leaves=8))
        problems = fp.log_fingerprint(view, tag="drifted-pin")
    assert len(problems) == 1
    assert "leaves: contract pins 4, install used 8" in problems[0]


def test_log_fingerprint_clean_match_and_gaps():
    reg = _registry()
    view = pc.build_contract_view(_build(reg=reg), reg)
    with _fresh_recorder():
        fp.record_install("alpha.mm", "trainer_fwd", "ref", pinned=dict(MM_PINS))
        with _captured(fp.logger) as cap:
            assert fp.log_fingerprint(view, tag="clean-case") == []
        assert not [r for r in cap.records if r.levelno >= logging.ERROR]
        gaps = fp.missing_from_fingerprint(view)
    assert ("alpha.mm", "trainer_fwd") not in gaps and ("alpha.pin", "engine_decode") in gaps


# --- 11. NCCL contract check ---


def test_nccl_assert_contract_matches_manual_view():
    pins = dict(ni.PINNED_CONSTANTS)
    view = {
        ("collectives.nccl_pin", s): {
            "impl_id": "pinned",
            "version": 1,
            "pinned_constants": dict(pins),
            "half": "function",
        }
        for s in TRAINER
    }
    ni.assert_contract_matches(view, TRAINER, "pinned", pins)  # exact tuple accepted
    msg = _refuses(RuntimeError, ni.assert_contract_matches, view, TRAINER, "unpinned", ni.UNPINNED_CONSTANTS)
    assert "refusing before forward" in msg
    _refuses(RuntimeError, ni.assert_contract_matches, view, TRAINER, "pinned", dict(pins, NCCL_MAX_NCHANNELS="8"))
    _refuses(RuntimeError, ni.assert_contract_matches, None, TRAINER, "pinned", pins)  # fail-closed: no contract
    msg = _refuses(RuntimeError, ni.assert_contract_matches, view, SITES, "pinned", pins)
    assert "no collectives.nccl_pin entry" in msg


def test_nccl_assert_against_a_built_view():
    reg = Registry()
    op = OpSpec("collectives.nccl_pin", list(SITES))
    op.add_impl(ImplSpec("pinned", 1, frozenset({"sm90"}), rounding=RoundingSchedule(dict(ni.PINNED_CONSTANTS))))
    op.add_impl(ImplSpec("unpinned", 1, frozenset({"sm90"})))
    reg.register_op(op)
    sel = {("collectives.nccl_pin", s): entry("pinned", pinned=dict(ni.PINNED_CONSTANTS)) for s in TRAINER}
    sel |= {("collectives.nccl_pin", s): entry("unpinned", cls=DEPLOYMENT, proof="run-1") for s in ENGINE}
    view = pc.build_contract_view(build_execution_contract(reg, sel, arch="sm90", model="tiny-nccl"), reg)
    ni.assert_contract_matches(view, TRAINER, "pinned", ni.PINNED_CONSTANTS)
    ni.assert_contract_matches(view, ENGINE, "unpinned", {})
    _refuses(RuntimeError, ni.assert_contract_matches, view, TRAINER, "cap8", ni.CAP8_CONSTANTS)
    _refuses(RuntimeError, ni.assert_contract_matches, view, ENGINE, "unpinned", ni.UNPINNED_CONSTANTS)


def test_nccl_mismatch_demotes_under_debug_tracing():
    # A debug run must reach the trace it was started for; the ledger keeps the violation anyway.
    from skyrl.backends.skyrl_train.isoexec.core import enforce

    pins = dict(ni.PINNED_CONSTANTS)
    view = {
        ("collectives.nccl_pin", s): {
            "impl_id": "pinned",
            "version": 1,
            "pinned_constants": dict(pins),
            "half": "function",
        }
        for s in TRAINER
    }
    enforce._reset_for_tests()
    try:
        with _env(SKYRL_ISOEXEC_DEBUG_TRACE="/tmp/isoexec-test-trace"):
            ni.assert_contract_matches(view, TRAINER, "unpinned", ni.UNPINNED_CONSTANTS)
        recs = [r for oid, rs in enforce.ledger().records.items() if "nccl_pin" in oid for r in rs]
        assert recs and all(r.result == enforce.VIOLATION for r in recs)
    finally:
        enforce._reset_for_tests()


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
