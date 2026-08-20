"""The Qwen3.5 composition, built from the op registry into an ExecutionContract.

Every assertion here is about what the CODE composes by default. The derivation reads launcher
environment (the NCCL identities in particular), so every test builds under a cleared environment
and the one test that cares about a non-default composition sets it explicitly -- otherwise these
tests pass or fail according to whatever shell happened to invoke them.
"""

import os
from contextlib import contextmanager

from skyrl.backends.skyrl_train.isoexec.core.process_contract import build_contract_view
from skyrl.backends.skyrl_train.isoexec.core.registry_build import build_registry
from skyrl.backends.skyrl_train.isoexec.models import qwen3_5

_SITES = ("trainer_fwd", "trainer_score", "engine_prefill", "engine_decode")

_SUBSUMED = {("gdn." + op, site) for op in ("l2norm", "gating") for site in _SITES}


@contextmanager
def _code_defaults(**overrides):
    """Strip every IsoExec override so the composition reflects code defaults, not the caller's shell."""
    saved = {k: v for k, v in os.environ.items() if k.startswith("SKYRL_ISOEXEC")}
    for k in saved:
        del os.environ[k]
    os.environ.update(overrides)
    try:
        yield
    finally:
        for k in list(os.environ):
            if k.startswith("SKYRL_ISOEXEC"):
                del os.environ[k]
        os.environ.update(saved)


def _build():
    reg = build_registry(strict=True)
    c = qwen3_5.build(reg, arch="sm90")
    return reg, c, build_contract_view(c, reg)


def test_contract_builds_and_identities_are_stable():
    with _code_defaults():
        reg, c, _ = _build()
        c2 = qwen3_5.build(reg, arch="sm90")
    assert c.identities == c2.identities, "contract identities must be deterministic"
    assert c.identities.numerical_policy
    assert c.model.family == qwen3_5.MODEL


def test_uncovered_keys_are_exactly_the_subsumed_ops():
    with _code_defaults():
        reg, _, view = _build()
    uncovered = set(reg.installed_keys()) - set(view.keys())
    assert uncovered == _SUBSUMED, f"unexpected uncovered set: {sorted(uncovered ^ _SUBSUMED)}"

    assert not (set(view.keys()) - set(reg.installed_keys()))


def test_only_engine_nccl_unpin_is_deployment_half():
    with _code_defaults():
        _, c, view = _build()
    dep = {k for k, e in view.items() if e["half"] == "deployment"}
    assert dep == {("collectives.nccl_pin", "engine_prefill"), ("collectives.nccl_pin", "engine_decode")}, dep

    for e in c.composition:
        if e.half == "deployment":
            assert e.discharge and e.discharge.kind == "neutrality_proof" and e.discharge.ref


def test_site_asymmetries_present():
    """The trainer and the engine deliberately run different implementations of the same op."""
    with _code_defaults():
        _, _, view = _build()

    assert view[("moe.experts", "trainer_fwd")]["impl_id"] == "batched_bmm"
    assert view[("moe.experts", "engine_prefill")]["impl_id"] == "fused"
    assert view[("moe.router", "trainer_fwd")]["impl_id"] == "deterministic"
    assert view[("moe.router", "engine_prefill")]["impl_id"] == "fused_o2"


def test_trainer_nccl_pin_is_function_bearing_by_default():
    """The trainer pin is on by default and its exact tuple is part of the hashed contract."""
    with _code_defaults():
        _, _, view = _build()
    e = view[("collectives.nccl_pin", "trainer_fwd")]

    assert e["impl_id"] == "pinned"
    assert e["half"] == "function"
    assert e["pinned_constants"] == {
        "NCCL_ALGO": "allreduce:tree",
        "NCCL_MIN_NCHANNELS": "1",
        "NCCL_MAX_NCHANNELS": "1",
    }


def test_engine_nccl_pin_defaults_to_unpinned():
    """Nothing is asked of the engine's communicator unless the launcher asks for it."""
    with _code_defaults():
        _, _, view = _build()
    e = view[("collectives.nccl_pin", "engine_decode")]

    assert e["impl_id"] == "unpinned"
    assert e["half"] == "deployment"


def test_engine_nccl_pin_records_the_requested_cap():
    """A launcher that unpins the engine to a 16-channel ceiling gets that exact tuple declared."""
    with _code_defaults(SKYRL_ISOEXEC_ENGINE_NCCL_UNPIN="1", SKYRL_ISOEXEC_ENGINE_NCCL_MAX_NCHANNELS="16"):
        _, _, view = _build()
    e = view[("collectives.nccl_pin", "engine_decode")]

    assert e["impl_id"] == "engine_cap16"
    assert e["pinned_constants"] == {
        "NCCL_ALGO": "allreduce:tree",
        "NCCL_MIN_NCHANNELS": None,
        "NCCL_MAX_NCHANNELS": "16",
    }
