"""The Qwen3.5 manifest, built from the op registry.

Every assertion here is about what the CODE composes by default. The manifest reads launcher
environment (the NCCL identities in particular), so the whole module builds under a cleared
environment and the one test that cares about a non-default composition sets it explicitly --
otherwise these tests pass or fail according to whatever shell happened to invoke pytest.
"""

import os

import pytest

from skyrl.backends.skyrl_train.isoexec.core.composition import DEPLOYMENT, FUNCTION
from skyrl.backends.skyrl_train.isoexec.core.registry_build import build_registry
from skyrl.backends.skyrl_train.isoexec.models import qwen3_5

_SITES = ("trainer_fwd", "trainer_score", "engine_prefill", "engine_decode")

_SUBSUMED = {("gdn." + op, site) for op in ("l2norm", "gating") for site in _SITES}


@pytest.fixture(autouse=True)
def _code_defaults(monkeypatch):
    """Strip every IsoExec override so the manifest reflects code defaults, not the caller's shell."""
    for name in list(os.environ):
        if name.startswith("SKYRL_ISOEXEC"):
            monkeypatch.delenv(name, raising=False)


def _build():
    reg = build_registry(strict=True)
    return reg, qwen3_5.build(reg, arch="sm90").freeze()


def test_manifest_builds_and_hash_is_stable():
    reg, m = _build()
    h1 = m.hash()
    h2 = qwen3_5.build(reg, arch="sm90").freeze().hash()
    assert h1 == h2, "manifest hash must be deterministic"
    assert m.model == qwen3_5.MODEL


def test_uncovered_keys_are_exactly_the_subsumed_ops():
    reg, m = _build()
    uncovered = set(reg.installed_keys()) - set(m._entries.keys())
    assert uncovered == _SUBSUMED, f"unexpected uncovered set: {sorted(uncovered ^ _SUBSUMED)}"

    assert not (set(m._entries.keys()) - set(reg.installed_keys()))


def test_only_engine_nccl_unpin_is_deployment_half():
    reg, m = _build()
    dep = set(m.deployment_entries().keys())
    assert dep == {("collectives.nccl_pin", "engine_prefill"), ("collectives.nccl_pin", "engine_decode")}, dep

    for _k, e in m.deployment_entries().items():
        assert e.classification == DEPLOYMENT and e.neutrality_proof


def test_site_asymmetries_present():
    """The trainer and the engine deliberately run different implementations of the same op."""
    _reg, m = _build()
    e = m._entries

    assert e[("moe.experts", "trainer_fwd")].impl_id == "batched_bmm"
    assert e[("moe.experts", "engine_prefill")].impl_id == "fused"
    assert e[("moe.router", "trainer_fwd")].impl_id == "deterministic"
    assert e[("moe.router", "engine_prefill")].impl_id == "fused_o2"


def test_trainer_nccl_pin_is_function_bearing_by_default():
    """The trainer pin is on by default and its exact tuple is part of the hashed manifest."""
    _reg, m = _build()
    entry = m._entries[("collectives.nccl_pin", "trainer_fwd")]

    assert entry.impl_id == "pinned"
    assert entry.classification == FUNCTION
    assert entry.pinned_constants == {
        "NCCL_ALGO": "allreduce:tree",
        "NCCL_MIN_NCHANNELS": "1",
        "NCCL_MAX_NCHANNELS": "1",
    }


def test_engine_nccl_pin_defaults_to_unpinned():
    """Nothing is asked of the engine's communicator unless the launcher asks for it."""
    _reg, m = _build()
    entry = m._entries[("collectives.nccl_pin", "engine_decode")]

    assert entry.impl_id == "unpinned"
    assert entry.classification == DEPLOYMENT


def test_engine_nccl_pin_records_the_requested_cap(monkeypatch):
    """A launcher that unpins the engine to a 16-channel ceiling gets that exact tuple declared."""
    monkeypatch.setenv("SKYRL_ISOEXEC_ENGINE_NCCL_UNPIN", "1")
    monkeypatch.setenv("SKYRL_ISOEXEC_ENGINE_NCCL_MAX_NCHANNELS", "16")
    _reg, m = _build()
    entry = m._entries[("collectives.nccl_pin", "engine_decode")]

    assert entry.impl_id == "engine_cap16"
    assert entry.pinned_constants == {
        "NCCL_ALGO": "allreduce:tree",
        "NCCL_MIN_NCHANNELS": None,
        "NCCL_MAX_NCHANNELS": "16",
    }
