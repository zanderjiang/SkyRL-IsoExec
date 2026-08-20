"""The pik all-reduce must not be able to change transport in silence.

WHAT WENT WRONG. ``pik.allreduce.tree_all_reduce`` dispatched to the symmetric-memory P2P path
and wrapped it in ``except Exception: pass``, falling through to the NCCL transport. Both
transports evaluate the identical tree, so the fallback is bitwise-invisible -- and therefore
COMPLETELY invisible, because bitwise-identical is the only property anything in this stack
checks. The engine printed "pik P2P symmetric-memory all-reduce KEPT" at install time (a
statement about ``_disable_pik_p2p`` not having been called) and nothing afterwards could tell
you whether symmetric memory had actually enrolled. Measured cost of being wrong, world=4 H100
NVLink under graph replay: 19.2 us vs 47.7 us per [128, 2048] fp32 combine, ~120 combines per
decode step.

WHAT IS GATED HERE, all on CPU with both transports stubbed -- this is a test of the DISPATCH and
its bookkeeping, not of either transport's arithmetic (that is ``pik``'s own bitwise suite and the
world-4 GPU probe):

  1. a P2P failure is counted AND warned about, with the exception in the message;
  2. it warns exactly ONCE per process (this runs per row-parallel layer per token) while the
     COUNTERS keep counting -- "once" must not decay into "once and then no idea";
  3. the transport hook fires on the first resolution and again on the first degradation, which is
     what carries the transport into the F1 install fingerprint;
  4. a hook registered late still fires (registration order vs. the first forward is not something
     an install site can control);
  5. ``pik_tp_invariant``'s hook installs a callback that records the fingerprint.

Run (CPU only, torch is the only requirement -- no triton, no megatron, no vLLM):
    uv run --isolated --extra dev python -m pytest <thisfile> -q
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
import types

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_COLLECTIVES_DIR = os.path.dirname(_HERE)


def _load_by_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ``pik_bootstrap`` top-level imports are importlib/logging/pathlib/sys/threading, so it loads
# standalone -- and ``ensure_pik`` is the ONLY supported way to get the vendored package (it must
# be registered as top-level ``pik`` or the codegen'd kernels cannot resolve their own imports).
_BOOT = _load_by_path("_ix_pik_bootstrap", os.path.join(_COLLECTIVES_DIR, "pik_bootstrap.py"))
_BOOT.ensure_pik()
import pik.allreduce as AR  # type: ignore  # noqa: E402


def _reset_transport_counters():
    # inlined: the public module keeps the state but not this test hook
    for k in AR._TRANSPORT_COUNTS:
        AR._TRANSPORT_COUNTS[k] = 0
    AR._FIRST_FALLBACK = {}
    AR._LAST_STATE = "none"


# ``pik_tp_invariant`` relative-imports its sibling, so it needs A package -- a shell whose
# ``__path__`` is ops/collectives. Its own top level is logging/os/torch + that sibling; every
# megatron import in it is function-local, which is what lets this stay a CPU test.
_PKG = "_ix_collectives_pkg"
_shell = types.ModuleType(_PKG)
_shell.__path__ = [_COLLECTIVES_DIR]
_shell.__spec__ = importlib.machinery.ModuleSpec(_PKG, loader=None, is_package=True)
_shell.__spec__.submodule_search_locations = list(_shell.__path__)
sys.modules[_PKG] = _shell
_load_by_path(f"{_PKG}.pik_bootstrap", os.path.join(_COLLECTIVES_DIR, "pik_bootstrap.py"))
PTI = _load_by_path(f"{_PKG}.pik_tp_invariant", os.path.join(_COLLECTIVES_DIR, "pik_tp_invariant.py"))


def test_ensure_pik_hot_path_does_not_rescan_sys_modules(monkeypatch):
    """Once package identity is established, the MoE hot path must be O(1)."""
    canonical = sys.modules["pik"]
    assert sys.modules[_BOOT._DOTTED_NAME] is canonical

    def fail_rescan(*_args, **_kwargs):
        raise AssertionError("steady-state ensure_pik rescanned sys.modules")

    monkeypatch.setattr(_BOOT, "_alias_loaded_children", fail_rescan)
    for _ in range(1024):
        assert _BOOT.ensure_pik() is canonical


class _FakeDist:
    """The two calls ``tree_all_reduce`` makes before it picks a transport. Nothing else in the
    dispatch touches torch.distributed, and both transports are stubbed below."""

    WORLD = 4

    @staticmethod
    def is_initialized():
        return True

    @classmethod
    def get_world_size(cls, group=None):
        return cls.WORLD


@pytest.fixture()
def ar(monkeypatch):
    """A clean per-test view of the transport bookkeeping, with both transports stubbed.

    The stubs are the point: a CPU box has neither symmetric memory nor a triton tree kernel, and
    what is under test is which one gets CALLED and what gets recorded about it.
    """
    monkeypatch.setattr(AR, "dist", _FakeDist)
    monkeypatch.setattr(AR, "p2p_available", lambda group=None: True)
    monkeypatch.setattr(AR, "_tree_all_reduce_nccl", lambda partial, group, out, root_dtype: partial + 1)
    monkeypatch.setattr(AR, "_TRANSPORT_HOOKS", [])
    _reset_transport_counters()
    yield AR
    _reset_transport_counters()


def _x():
    return torch.ones(4, 8, dtype=torch.float32)


# The stubs are NAMED because the name is part of what is under test: the hook hands the install
# fingerprint the function that ACTUALLY moved the bytes, read off the module global at call time.
# In production those are ``pik.allreduce._tree_all_reduce_p2p`` and ``pik.allreduce._tree_reduce``.
def _stub_p2p(partial, group, out, rd):
    return partial * 2


def _stub_p2p_broken(partial, group, out, rd):
    raise RuntimeError("symm_mem rendezvous: allocations from overlapping devices")


# ==================================================================================================
# 1-2: the fallback is counted, warned about ONCE, and names the exception
# ==================================================================================================
def test_p2p_success_counts_and_stays_quiet(ar, capsys):
    ar._tree_all_reduce_p2p = _stub_p2p
    out = ar.tree_all_reduce(_x())
    assert torch.equal(out, _x() * 2), "the p2p stub's result must be the one returned"
    c = ar.transport_counts()
    assert (c["p2p_calls"], c["nccl_calls"], c["p2p_fallbacks"]) == (1, 0, 0), c
    assert c["state"] == "p2p" and c["first_fallback"] is None
    assert "WARNING" not in capsys.readouterr().out


def test_fallback_warns_once_names_the_exception_and_keeps_counting(ar, capsys):
    ar._tree_all_reduce_p2p = _stub_p2p_broken

    out = ar.tree_all_reduce(_x())
    assert torch.equal(out, _x() + 1), "the NCCL transport's result must be returned on fallback"
    first = capsys.readouterr().out
    # The exception is IN the message -- "it fell back" without "why" is not actionable.
    assert first.count("symmetric-memory P2P all-reduce FAILED") == 1, first
    assert "allocations from overlapping devices" in first
    assert "RuntimeError" in first

    # ...and every subsequent call is silent, but still COUNTED. This is the half that decays:
    # a warn-once with no counter degrades into "we warned about something, once, some time ago".
    for _ in range(50):
        ar.tree_all_reduce(_x())
    assert capsys.readouterr().out == "", "the fallback warning must fire exactly once per process"

    c = ar.transport_counts()
    assert (c["p2p_calls"], c["nccl_calls"], c["p2p_fallbacks"]) == (0, 51, 51), c
    assert c["state"] == "p2p_fallback", "a degraded process must not report itself as plain nccl"
    assert c["first_fallback"]["shape"] == (4, 8)
    assert c["first_fallback"]["world"] == 4
    assert "RuntimeError" in c["first_fallback"]["exception"]


def test_explicit_nccl_backend_counts_as_nccl_not_as_a_fallback(ar):
    ar._tree_all_reduce_p2p = _stub_p2p
    ar.tree_all_reduce(_x(), backend="nccl")
    c = ar.transport_counts()
    assert (c["p2p_calls"], c["nccl_calls"], c["p2p_fallbacks"]) == (0, 1, 0), c
    assert c["state"] == "nccl"


def test_world_1_records_no_transport(ar, monkeypatch):
    """TP=1 does no reduction at all, so it must report ``none`` rather than a transport it never
    chose -- otherwise the fingerprint would report the absence of a collective as a transport."""
    monkeypatch.setattr(_FakeDist, "WORLD", 1)
    ar.tree_all_reduce(_x())
    assert ar.transport_counts()["state"] == "none"


# ==================================================================================================
# 3-4: the hook, which is what carries the transport into the install fingerprint
# ==================================================================================================
def test_hook_fires_on_first_resolution_and_on_degradation(ar):
    seen = []
    ar.register_transport_hook(lambda state, fn: seen.append((state, getattr(fn, "__name__", fn))))

    ar._tree_all_reduce_p2p = _stub_p2p
    ar.tree_all_reduce(_x())
    ar.tree_all_reduce(_x())
    assert seen == [("p2p", "_stub_p2p")], "one hook call per STATE, not per reduce"

    ar._tree_all_reduce_p2p = _stub_p2p_broken
    ar.tree_all_reduce(_x())
    ar.tree_all_reduce(_x())
    # The degraded state names the NCCL transport's arithmetic (the real, unstubbed `_tree_reduce`).
    assert seen == [("p2p", "_stub_p2p"), ("p2p_fallback", "_tree_reduce")], seen


def test_late_registration_still_sees_the_resolved_transport(ar):
    """An install site cannot order itself against the first forward, so a hook registered after
    the fact must still fire -- otherwise the fingerprint silently records nothing."""
    ar._tree_all_reduce_p2p = _stub_p2p
    ar.tree_all_reduce(_x())
    seen = []
    ar.register_transport_hook(lambda state, fn: seen.append(state))
    assert seen == ["p2p"]


def test_a_broken_hook_cannot_break_the_reduce(ar):
    ar.register_transport_hook(lambda state, fn: 1 / 0)
    ar._tree_all_reduce_p2p = _stub_p2p
    assert torch.equal(ar.tree_all_reduce(_x()), _x() * 2)
    assert ar.transport_counts()["p2p_calls"] == 1


# ==================================================================================================
# 5: pik_tp_invariant wires the hook to the F1 fingerprint
# ==================================================================================================
def test_install_transport_hook_records_the_fingerprint(ar, monkeypatch, capsys):
    recorded = []
    monkeypatch.setattr(
        PTI, "_record_transport_fingerprint", lambda side, state, fn: recorded.append((side, state, fn.__name__))
    )
    PTI._install_transport_hook("ENGINE")

    ar._tree_all_reduce_p2p = _stub_p2p
    ar.tree_all_reduce(_x())

    assert recorded == [("ENGINE", "p2p", "_stub_p2p")], recorded
    banner = capsys.readouterr().out
    assert "[ISOEXEC-ENGINE] pik tree all-reduce TRANSPORT RESOLVED: p2p" in banner
    assert "_stub_p2p" in banner and "p2p=1 nccl=0 fallbacks=0" in banner


def test_transport_status_reports_the_counts(ar):
    ar._tree_all_reduce_p2p = _stub_p2p
    ar.tree_all_reduce(_x())
    st = PTI.pik_transport_status()
    assert st["state"] == "p2p" and st["p2p_calls"] == 1, st


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))
