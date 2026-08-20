"""CPU guarantees for the ``_native`` RMSNorm admission memo (``ops/norms/native_rmsnorm_memo.py``).

The lever memoizes exactly two functions inside torch's own predicate,
``torch._native.ops.norm.rmsnorm_impl._is_supported`` and ``._smem_budget_bytes``, both of which
are pure functions of the DEVICE (plus dtype). The predicate returns a boolean, and there is
exactly one override node for ``(_fused_rms_norm, CUDA)``, so an unchanged boolean is an unchanged
kernel. Everything the predicate reads that is genuinely per-call -- alignment, contiguity, COW,
numel, shape, weight dtype/device -- is untouched.

What is CPU-testable is the memo's DISCIPLINE, which is where a defect would live:

  1. the memoized functions return exactly what the originals return, for every key;
  2. the original is called ONCE per key and not again -- the whole point;
  3. an INDEX-LESS device is never cached (``get_device_capability(torch.device("cuda"))``
     resolves against the CURRENT device, so a cached answer would be wrong after a device switch);
  4. install is idempotent, revert restores torch's own functions by identity, and the flag is
     honoured -- an unset flag must leave torch untouched;
  5. ``hits`` is a real count, so an inert install is visible as inert.

This CPU test does not establish live-CUDA predicate behavior; that requires a separate GPU
qualification on the target runtime and production widths.

Run (CPU only):
    uv run --isolated --extra dev python -m pytest \
        skyrl/backends/skyrl_train/isoexec/ops/norms/tests/test_native_rmsnorm_memo_cpu.py -q
"""

from __future__ import annotations

import pathlib
import sys
import types

import pytest
import torch

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[7]))  # repo root

from skyrl.backends.skyrl_train.isoexec.ops.norms import native_rmsnorm_memo as NM  # noqa: E402


class _FakeRmsnormImpl(types.SimpleNamespace):
    """The two functions the memo rebinds, instrumented so double evaluation is visible."""

    def __init__(self):
        super().__init__()
        self.supported_calls: list = []
        self.smem_calls: list = []

        def _is_supported(inp):
            self.supported_calls.append((inp.device, inp.dtype))
            return inp.dtype is not torch.float64  # an arbitrary but device/dtype-pure answer

        def _smem_budget_bytes(device):
            self.smem_calls.append(device)
            return 200_000 + (device.index or 0)

        self._is_supported = _is_supported
        self._smem_budget_bytes = _smem_budget_bytes


@pytest.fixture
def fake(monkeypatch):
    impl = _FakeRmsnormImpl()
    mod = types.ModuleType("torch._native.ops.norm.rmsnorm_impl")
    mod._is_supported = impl._is_supported
    mod._smem_budget_bytes = impl._smem_budget_bytes
    monkeypatch.setitem(sys.modules, "torch._native.ops.norm.rmsnorm_impl", mod)
    # make `from torch._native.ops.norm import rmsnorm_impl` resolve to it
    pkg = types.ModuleType("torch._native.ops.norm")
    pkg.rmsnorm_impl = mod
    monkeypatch.setitem(sys.modules, "torch._native.ops.norm", pkg)
    monkeypatch.setitem(sys.modules, "torch._native.ops", types.ModuleType("torch._native.ops"))
    monkeypatch.setitem(sys.modules, "torch._native", types.ModuleType("torch._native"))
    NM.revert_native_rmsnorm_memo()
    NM._reset_for_tests()
    yield impl, mod
    NM.revert_native_rmsnorm_memo()
    NM._reset_for_tests()


class _Fake:
    """A tensor stand-in carrying only what the memo's key reads."""

    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype


def test_flag_off_leaves_torch_untouched(fake, monkeypatch):
    _impl, mod = fake
    monkeypatch.delenv("SKYRL_ISOEXEC_NATIVE_NORM_MEMO", raising=False)
    before = mod._is_supported
    assert NM.install_native_rmsnorm_memo() is False
    assert mod._is_supported is before
    assert NM.native_rmsnorm_memo_counts()["installed"] is False


def test_same_answer_evaluated_once_per_key(fake, monkeypatch):
    impl, mod = fake
    monkeypatch.setenv("SKYRL_ISOEXEC_NATIVE_NORM_MEMO", "1")
    assert NM.install_native_rmsnorm_memo() is True

    dev = torch.device("cuda", 0)
    ref = impl._is_supported(_Fake(dev, torch.bfloat16))
    impl.supported_calls.clear()

    for _ in range(50):
        assert mod._is_supported(_Fake(dev, torch.bfloat16)) == ref
    assert len(impl.supported_calls) == 1, "the original must be evaluated once per (device, dtype)"
    assert NM.native_rmsnorm_memo_counts()["hits"] == 49

    # a different dtype is a different key, and still gets the ORIGINAL's answer
    assert mod._is_supported(_Fake(dev, torch.float64)) is False
    assert len(impl.supported_calls) == 2

    for _ in range(20):
        assert mod._smem_budget_bytes(dev) == 200_000
    assert len(impl.smem_calls) == 1
    assert mod._smem_budget_bytes(torch.device("cuda", 1)) == 200_001, "keyed on the INDEX"


def test_index_less_device_is_never_cached(fake, monkeypatch):
    """``get_device_capability(torch.device('cuda'))`` resolves against the CURRENT device, so a
    cached answer would survive a device switch and be wrong. It must fall through, uncounted."""
    impl, mod = fake
    monkeypatch.setenv("SKYRL_ISOEXEC_NATIVE_NORM_MEMO", "1")
    NM.install_native_rmsnorm_memo()

    anon = torch.device("cuda")
    assert anon.index is None
    for _ in range(10):
        mod._is_supported(_Fake(anon, torch.bfloat16))
        mod._smem_budget_bytes(anon)
    assert len(impl.supported_calls) == 10, "every index-less call must re-evaluate"
    assert len(impl.smem_calls) == 10
    c = NM.native_rmsnorm_memo_counts()
    assert c["hits"] == 0 and c["bypass_no_index"] == 20


def test_install_is_idempotent_and_revert_restores_by_identity(fake, monkeypatch):
    impl, mod = fake
    monkeypatch.setenv("SKYRL_ISOEXEC_NATIVE_NORM_MEMO", "1")
    orig_s, orig_b = mod._is_supported, mod._smem_budget_bytes
    assert NM.install_native_rmsnorm_memo() is True
    wrapped = mod._is_supported
    assert wrapped is not orig_s
    assert NM.install_native_rmsnorm_memo() is True
    assert mod._is_supported is wrapped, "a second install must not double-wrap"
    NM.revert_native_rmsnorm_memo()
    assert mod._is_supported is orig_s and mod._smem_budget_bytes is orig_b


def test_missing_torch_surface_is_fail_soft(monkeypatch):
    monkeypatch.setenv("SKYRL_ISOEXEC_NATIVE_NORM_MEMO", "1")
    NM.revert_native_rmsnorm_memo()
    pkg = types.ModuleType("torch._native.ops.norm")
    pkg.rmsnorm_impl = types.ModuleType("torch._native.ops.norm.rmsnorm_impl")  # no helpers at all
    monkeypatch.setitem(sys.modules, "torch._native.ops.norm", pkg)
    monkeypatch.setitem(sys.modules, "torch._native.ops.norm.rmsnorm_impl", pkg.rmsnorm_impl)
    assert NM.install_native_rmsnorm_memo() is False


def test_flag_registered_default_off_and_forwarded():
    from skyrl.backends.skyrl_train.isoexec.core.flags import ENGINE, FLAGS, TRAIN, actor_forwarding_tuple

    cat = {f.name: f for f in FLAGS}
    assert "SKYRL_ISOEXEC_NATIVE_NORM_MEMO" in cat
    assert cat["SKYRL_ISOEXEC_NATIVE_NORM_MEMO"].default == "0"
    assert "SKYRL_ISOEXEC_NATIVE_NORM_MEMO" in actor_forwarding_tuple(TRAIN)
    assert "SKYRL_ISOEXEC_NATIVE_NORM_MEMO" in actor_forwarding_tuple(ENGINE)
