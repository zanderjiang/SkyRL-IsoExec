"""CPU/source contracts for the isolated Megatron GDN scoring fast paths."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from skyrl.backends.skyrl_train.isoexec.runtimes.megatron import gdn_fla_shim as shim


def _scoring_fused_outnorm_stats():
    # inlined: the public module keeps the counters but not this test accessor
    return dict(shim._SCORING_FUSED_OUTNORM_COUNTS)


def _reset_scoring_fused_outnorm_stats():
    shim._SCORING_FUSED_OUTNORM_COUNTS["served"] = 0



@pytest.fixture(autouse=True)
def _default_off(monkeypatch):
    monkeypatch.delenv("SKYRL_ISOEXEC_GDN_SCORING_FUSED_OUTNORM", raising=False)


def _eager_subject(*, training: bool):
    class Norm(torch.nn.Module):
        def forward(self, x):
            return F.rms_norm(x, (x.shape[-1],), None, 1e-6)

    return SimpleNamespace(training=training, out_norm=Norm(), act_fn=F.silu, activation="silu")


def test_scoring_fused_outnorm_is_default_off(monkeypatch):
    assert not shim._scoring_fused_outnorm_enabled()
    monkeypatch.setenv("SKYRL_ISOEXEC_GDN_SCORING_FUSED_OUTNORM", "1")
    assert shim._scoring_fused_outnorm_enabled()


def test_scoring_fused_outnorm_source_is_eval_and_no_grad_only():
    source = inspect.getsource(shim._eager_apply_gated_norm)
    contract = inspect.getsource(shim._scoring_fused_outnorm_static_contract)
    assert "not self.training" in contract
    assert "not torch.is_grad_enabled()" in contract
    assert "ZeroCenteredTorchRMSNorm" in contract
    assert 'gate.dtype == torch.bfloat16' in contract
    assert "_SCORING_FUSED_OUTNORM_MAX_WIDTH" in contract
    assert "tuple(weight.shape) == (width,)" in contract
    assert "weight.device == gate.device" in contract
    assert "weight.dtype == gate.dtype" in contract
    assert "weight.is_contiguous()" in contract
    assert "gate3 is not None" in source
    assert "x.device == gate.device" in source
    assert "x.dtype == gate.dtype" in source
    assert "fused_gated_out_norm" in source
    assert '_SCORING_FUSED_OUTNORM_COUNTS["served"] += 1' in source


def test_scoring_gate_view_preserves_projected_token_stride_without_copy():
    projected = torch.randn(3, 5, 29)
    gate = projected[..., 7:23].view(3, 5, 2, 8)
    assert not gate.is_contiguous()

    gate3 = shim._scoring_gate_view(gate)

    assert gate3 is not None
    assert gate3.shape == (15, 2, 8)
    assert gate3.stride() == (29, 8, 1)
    assert gate3.untyped_storage().data_ptr() == gate.untyped_storage().data_ptr()
    assert gate3.storage_offset() == gate.storage_offset()


def test_scoring_gate_view_refuses_layouts_that_would_need_materialization():
    gate = torch.randn(4, 3, 5).transpose(-1, -2)
    assert shim._scoring_gate_view(gate) is None


def test_prepare_source_keeps_strided_gate_only_behind_full_scoring_contract():
    source = inspect.getsource(shim._eager_prepare_qkv)
    assert "_scoring_fused_outnorm_static_contract(self, gate)" in source
    assert "else gate.contiguous()" in source


def test_training_and_grad_enabled_eval_keep_the_eager_expression(monkeypatch):
    monkeypatch.setenv("SKYRL_ISOEXEC_GDN_SCORING_FUSED_OUTNORM", "1")
    x = torch.randn(2, 3, 4, requires_grad=True)
    gate = torch.randn_like(x)

    training_output = shim._eager_apply_gated_norm(_eager_subject(training=True), x, gate)
    eval_grad_output = shim._eager_apply_gated_norm(_eager_subject(training=False), x, gate)

    expected = (F.rms_norm(x.reshape(-1, 4), (4,), None, 1e-6) * F.silu(gate.reshape(-1, 4).float())).to(x.dtype)
    assert torch.equal(training_output, expected)
    assert torch.equal(eval_grad_output, expected)
    assert training_output.grad_fn is not None and eval_grad_output.grad_fn is not None


def test_scoring_fused_outnorm_counter_is_explicit_engagement_evidence():
    shim._SCORING_FUSED_OUTNORM_COUNTS["served"] = 7
    assert _scoring_fused_outnorm_stats() == {"served": 7}
    _reset_scoring_fused_outnorm_stats()
    assert _scoring_fused_outnorm_stats() == {"served": 0}


def test_rejected_qk_candidate_remains_unwired():
    """The H100 battery found compressed-head core slower; production must retain expansion."""
    source = inspect.getsource(shim._eager_prepare_qkv)
    assert "query = query.repeat_interleave(repeat_factor, dim=2)" in source
    assert "key = key.repeat_interleave(repeat_factor, dim=2)" in source
