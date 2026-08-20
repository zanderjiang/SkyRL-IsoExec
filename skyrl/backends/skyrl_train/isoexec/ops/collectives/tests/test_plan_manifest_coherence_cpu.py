"""The env-built pik plan must match the manifest's pinned contract constants, fail-closed.

``pik_tp_invariant._assert_plan_matches_manifest`` is the seam that closes MODEL_PORTABILITY
audit gap 1(b) for the pik contract constants: the runtime tree is built from env vars, the
handshake hash is built from profile pins, and nothing used to connect them -- an env flip on
both sides ran a different reduction under an unchanged, MATCHING handshake. These tests pin
the check's three behaviours: refuse on disagreement, pass on agreement, skip without a
manifest (benches). CPU-only; the manifest is stubbed at the module seam the check reads.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

pmod = importlib.import_module("skyrl.backends.skyrl_train.isoexec.ops.collectives.pik_tp_invariant")
manifest_mod = importlib.import_module("skyrl.backends.skyrl_train.isoexec.core.process_contract")


def _StubManifest(pins):
    # the check reads process_contract.cached_contract_view(): {(op, site) -> {"pinned_constants": ...}}
    return {("collectives.tree_all_reduce", "engine_decode"): {"pinned_constants": pins}}


def _plan(leaves=8, bf16=False):
    return SimpleNamespace(num_leaves=leaves, bf16_leaves=bf16)


@pytest.fixture(autouse=True)
def _restore_manifest():
    saved = manifest_mod._VIEW
    yield
    manifest_mod._VIEW = saved


def test_no_manifest_skips():
    manifest_mod._VIEW = None
    pmod._assert_plan_matches_manifest("TEST", _plan())  # must not raise


def test_agreement_passes():
    manifest_mod._VIEW = _StubManifest({"leaves": 8, "leaf_dtype": "fp32"})
    pmod._assert_plan_matches_manifest("TEST", _plan(8, bf16=False))
    manifest_mod._VIEW = _StubManifest({"leaves": 8, "leaf_dtype": "bf16"})
    pmod._assert_plan_matches_manifest("TEST", _plan(8, bf16=True))


def test_leaf_dtype_split_refuses(monkeypatch):
    monkeypatch.delenv("SKYRL_ISOEXEC_MANIFEST_STRICT", raising=False)
    manifest_mod._VIEW = _StubManifest({"leaves": 8, "leaf_dtype": "fp32"})
    with pytest.raises(RuntimeError, match="leaf_dtype"):
        pmod._assert_plan_matches_manifest("TEST", _plan(8, bf16=True))


def test_leaves_split_refuses(monkeypatch):
    monkeypatch.delenv("SKYRL_ISOEXEC_MANIFEST_STRICT", raising=False)
    manifest_mod._VIEW = _StubManifest({"leaves": 8, "leaf_dtype": "fp32"})
    with pytest.raises(RuntimeError, match="leaves"):
        pmod._assert_plan_matches_manifest("TEST", _plan(4, bf16=False))


def test_strict_off_downgrades_to_warning(monkeypatch, capsys):
    monkeypatch.setenv("SKYRL_ISOEXEC_MANIFEST_STRICT", "0")
    manifest_mod._VIEW = _StubManifest({"leaves": 8, "leaf_dtype": "fp32"})
    pmod._assert_plan_matches_manifest("TEST", _plan(8, bf16=True))  # no raise
    assert "SPLIT" in capsys.readouterr().out
