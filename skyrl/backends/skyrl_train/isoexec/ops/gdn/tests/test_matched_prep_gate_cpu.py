"""CPU fallback and environment-gate checks for matched-prep launch fusion."""

from __future__ import annotations

import torch

from skyrl.backends.skyrl_train.isoexec.ops.gdn.gdn_matched_prep_gate import (
    matched_prep_fused_gate_enabled,
    maybe_matched_prep_fused_gate,
    maybe_matched_prep_fused_l2,
)
from skyrl.backends.skyrl_train.isoexec.ops.gdn import gdn_matched_prep_gate as _mpg


def matched_prep_fused_gate_stats():
    # inlined: the public module keeps the counters but not this test accessor
    return dict(_mpg._COUNTS)



def test_default_on_and_explicit_off(monkeypatch):
    monkeypatch.delenv("SKYRL_ISOEXEC_GDN_MATCHED_PREP_FUSED_GATE", raising=False)
    assert matched_prep_fused_gate_enabled()
    monkeypatch.setenv("SKYRL_ISOEXEC_GDN_MATCHED_PREP_FUSED_GATE", "0")
    assert not matched_prep_fused_gate_enabled()


def test_cpu_operands_decline_without_cuda_query(monkeypatch):
    a = torch.empty(8, 4, dtype=torch.bfloat16)
    b = torch.empty_like(a)
    A_log = torch.empty(4, dtype=torch.float32)
    dt_bias = torch.empty_like(A_log)
    k = torch.empty(1, 8, 2, 128, dtype=torch.bfloat16)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("CPU fallback must not query CUDA capability")

    monkeypatch.setattr(torch.cuda, "get_device_capability", forbidden)
    before = matched_prep_fused_gate_stats()
    assert maybe_matched_prep_fused_gate(a, b, A_log, dt_bias) is None
    assert maybe_matched_prep_fused_l2(k) is None
    after = matched_prep_fused_gate_stats()
    assert after["gate_declined"] == before["gate_declined"] + 1
    assert after["l2_declined"] == before["l2_declined"] + 1


def test_layout_and_dtype_contracts_decline_before_launch():
    a = torch.empty(8, 4, dtype=torch.bfloat16)
    A_log = torch.empty(4, dtype=torch.float32)
    assert maybe_matched_prep_fused_gate(a.t(), a.t(), torch.empty(8), torch.empty(8)) is None
    assert maybe_matched_prep_fused_gate(a.half(), a.half(), A_log, A_log) is None
    assert maybe_matched_prep_fused_l2(torch.empty(8, 2, 128, dtype=torch.float16)) is None
