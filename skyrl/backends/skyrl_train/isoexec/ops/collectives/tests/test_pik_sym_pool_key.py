"""The zero-copy staging path must actually land in the buffer the reduce reads.

WHAT WENT WRONG. ``pik.allreduce`` treats ``out_dtype=None`` as "same as ``dtype``" everywhere --
``_SymPool.__init__`` resolves it that way, ``_tree_all_reduce_p2p`` resolves it that way -- except
in ``_sym_pool``'s CACHE KEY, which used the raw argument. So the two halves of the staging
fast-path asked for two different pools:

    sym_partial(...)          -> _sym_pool(..., dtype=fp32, out_dtype=None)  key (.., fp32, None)
    _tree_all_reduce_p2p(...) -> _sym_pool(..., dt_in=fp32,  dt_out=fp32)    key (.., fp32, fp32)

``row_parallel_linear`` hands ``sym_partial``'s view to ``ti_gemm`` as ``out=`` precisely so the
GEMM writes its subtree partial straight into peer-visible memory; the reduce then found
``partial.data_ptr() != inp.data_ptr()`` and copied the whole partial into the OTHER pool. The
copy that ``sym_partial``'s docstring exists to remove -- "measured: that copy alone was the
difference between 2.0x and 1.0x vs NCCL" -- has therefore been running on every row-parallel site
since ``out_dtype`` was introduced, together with a second symmetric allocation per
(device, world, dtype) that nothing ever reads. It is visible in the live GLM-4.7-Flash decode
trace as ``memcpy128`` x142.8 per step: one per pik site, immediately before that site's leading
barrier.

Nothing could catch it, because both transports and both pools evaluate the identical tree: the
result was always right. This is the same failure class as the silent NCCL fallback next door
(``test_pik_transport_observability``) -- a perf regression hiding behind a correct answer.

AND IT GREW BACK A SECOND TIME (2026-08-13). Normalizing ``None`` fixed the default case, but the
key still carried ``out_dtype`` -- so the moment ``SKYRL_ISOEXEC_PIK_FUSED_ROOT_CAST`` was admitted,
the reduce looked up (fp32, bf16) while ``sym_partial`` (which CANNOT know the root dtype;
admission decides it later, per shape) had staged into (fp32, fp32): a different pool again, and
the 4 MiB ``memcpy128`` came back at every one of the 80 decode sites (2.7 us each, rank-0 trace
``traces_fb_off``). The fix this time removes the failure CLASS: the STAGING pool is keyed by the
wire dtype alone (``_SymInPool``), the OUTPUT pool by the root dtype alone (``_SymOutPool``), and
``_sym_pool`` returns a view pairing them. Staging can no longer be un-aliased by anything the
root does.

WHAT IS GATED HERE (CPU only; the split pools are stubbed, since real ones need CUDA symmetric
memory): ``_sym_pool`` resolves ``out_dtype=None`` to ``dtype`` BEFORE keying; the staging lookup
and the reduce lookup return the same staging BUFFERS for a given wire dtype, whatever root dtype
the reduce asks for; and distinct dtype pairs still get distinct output buffers.

Run (CPU only):
    uv run --isolated --extra dev python -m pytest <thisfile> -q
"""

from __future__ import annotations

import importlib.util
import os
import sys

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


_BOOT = _load_by_path("_ix_pik_bootstrap_poolkey", os.path.join(_COLLECTIVES_DIR, "pik_bootstrap.py"))
_BOOT.ensure_pik()
import pik.allreduce as AR  # type: ignore  # noqa: E402


class _FakeInPool:
    """Stands in for _SymInPool: fake double-buffered staging, keyed upstream by wire dtype."""

    def __init__(self, n, device, group, dtype=torch.float32):
        self.cap = max(n, 1 << 20)
        self.dtype = dtype
        self.inp = [torch.zeros(self.cap, dtype=dtype) for _ in range(2)]
        self.phase = 0

    def stage(self):
        return self.inp[self.phase], None

    def flip(self):
        self.phase ^= 1


class _FakeOutPool:
    """Stands in for _SymOutPool: fake output buffer, keyed upstream by root dtype."""

    def __init__(self, n, device, group, out_dtype=torch.float32):
        self.cap = max(n, 1 << 20)
        self.out_dtype = out_dtype
        self.out = torch.zeros(self.cap, dtype=out_dtype)
        self.h_out = None


@pytest.fixture()
def stubbed(monkeypatch):
    monkeypatch.setattr(AR, "_SymInPool", _FakeInPool)
    monkeypatch.setattr(AR, "_SymOutPool", _FakeOutPool)
    monkeypatch.setattr(AR.dist, "get_world_size", lambda group=None: 4)
    AR._SYM.clear()
    AR._SYM_IN.clear()
    AR._SYM_OUT.clear()
    yield
    AR._SYM.clear()
    AR._SYM_IN.clear()
    AR._SYM_OUT.clear()


def test_none_out_dtype_shares_the_pool_with_its_resolved_dtype(stubbed):
    """The staging lookup and the reduce lookup must be the SAME pool.

    (fp32, None) is what ``sym_partial`` asks for; (fp32, fp32) is what ``_tree_all_reduce_p2p``
    resolves to when the caller passes no ``root_dtype``. If these differ, the GEMM stages into a
    buffer no peer ever reads and the reduce copies it across.
    """
    staged = AR._sym_pool(1024, "cpu", None, torch.float32, None)
    reduced = AR._sym_pool(1024, "cpu", None, torch.float32, torch.float32)
    assert staged is reduced, "staging pool and reduce pool diverged -> a full copy-in per site"
    assert len(AR._SYM_IN) == 1, f"one staging pool, {len(AR._SYM_IN)} allocated"


def test_root_cast_cannot_unalias_the_staging(stubbed):
    """THE 2026-08-13 REGRESSION: a bf16-ROOT reduce must read the fp32 staging it was handed.

    ``sym_partial`` stages (fp32, None) BEFORE root-cast admission has decided anything; a
    ROOT_CAST-admitted reduce then looks up (fp32, bf16). Under the old (wire, root)-keyed pool
    those were different buffers and every site paid a full copy-in (memcpy128 x80/step on the
    fb_off decode trace). The staging buffers must be the same object whatever the root dtype.
    """
    staged = AR._sym_pool(1024, "cpu", None, torch.float32, None)
    reduced = AR._sym_pool(1024, "cpu", None, torch.float32, torch.bfloat16)
    assert staged.stage()[0] is reduced.stage()[0], (
        "root-cast reduce reads a different staging buffer than sym_partial staged into -> "
        "a full copy-in per site whenever SKYRL_ISOEXEC_PIK_FUSED_ROOT_CAST is admitted"
    )
    assert reduced.out_dtype == torch.bfloat16 and staged.out_dtype == torch.float32
    assert staged.out is not reduced.out, "fp32 and bf16 roots must not share an output buffer"


def test_sym_partial_returns_the_buffer_the_reduce_will_stage_into(stubbed):
    """End to end at the pointer level: the whole point of ``sym_partial`` is data_ptr equality."""
    view = AR.sym_partial((16, 64), "cpu", None, dtype=torch.float32)
    inp, _ = AR._sym_pool(16 * 64, "cpu", None, torch.float32, torch.float32).stage()
    assert view.data_ptr() == inp.data_ptr(), (
        "sym_partial handed out a buffer the reduce does not read -> _tree_all_reduce_p2p takes "
        "its `partial.data_ptr() != inp.data_ptr()` copy-in branch on every single call"
    )
    # ...and the SAME pointer must serve a bf16-root reduce (the root-cast fast path)
    inp_bf16_root, _ = AR._sym_pool(16 * 64, "cpu", None, torch.float32, torch.bfloat16).stage()
    assert view.data_ptr() == inp_bf16_root.data_ptr()


def test_bf16_wire_with_fp32_root_still_gets_its_own_view(stubbed):
    """The decoupled combination keeps its own OUTPUT while SHARING the bf16 staging.

    bf16 partial + fp32 root is the MoE-combine wire saving: the wire buffer is bf16 (shared with
    every other bf16-wire site -- that is what makes ``moe_batch_invariant``'s staging zero-copy)
    and the root is fp32, which must not collapse into the bf16 output buffer.
    """
    a = AR._sym_pool(1024, "cpu", None, torch.bfloat16, torch.float32)
    b = AR._sym_pool(1024, "cpu", None, torch.bfloat16, None)
    c = AR._sym_pool(1024, "cpu", None, torch.float32, torch.float32)
    assert a is not b, "(bf16 wire, fp32 root) collapsed into (bf16, bf16)"
    assert a is not c and b is not c
    assert b.out_dtype == torch.bfloat16 and a.out_dtype == torch.float32
    assert a.stage()[0] is b.stage()[0], "same wire dtype must share the staging buffers"
    assert a.stage()[0].dtype == torch.bfloat16
    assert a.out is c.out, "fp32 roots share one output pool regardless of wire dtype"


def test_growth_still_reallocates(stubbed):
    """Splitting the pools must not disturb the capacity check."""
    small = AR._sym_pool(1024, "cpu", None, torch.float32, None)
    small_in = small.stage()[0]
    same = AR._sym_pool(2048, "cpu", None, torch.float32, None)
    assert same is small, "still inside cap -> must reuse"
    big = AR._sym_pool((1 << 20) + 1, "cpu", None, torch.float32, None)
    assert big.stage()[0] is not small_in, "beyond cap -> must reallocate the staging"
    assert AR._sym_pool(16, "cpu", None, torch.float32, torch.float32) is big, "grown pool must be the shared one"
