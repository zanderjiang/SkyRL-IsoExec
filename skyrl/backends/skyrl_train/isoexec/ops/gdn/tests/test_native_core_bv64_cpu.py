"""CPU-side admission and fallback tests for the Hopper BV64 native-core launch."""

from __future__ import annotations

import torch

from skyrl.backends.skyrl_train.isoexec.ops.gdn.gdn_native_core_bv64 import (
    _eligible,
    maybe_native_core_bv64,
)
from skyrl.backends.skyrl_train.isoexec.ops.gdn import gdn_native_core_bv64 as _bv64


def native_core_bv64_stats():
    # inlined: the public module keeps the counters but not this test accessor
    return dict(_bv64._STATS)



def _cpu_operands():
    q = torch.empty(1, 4, 2, 128, dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty(1, 4, 4, 128, dtype=torch.bfloat16)
    a = torch.empty(4, 4, dtype=torch.bfloat16)
    b = torch.empty_like(a)
    A_log = torch.empty(4, dtype=torch.bfloat16)
    dt_bias = torch.empty_like(A_log)
    state = torch.empty(3, 4, 128, 128, dtype=torch.float32)
    indices = torch.ones(2, 2, dtype=torch.int32)
    cu = torch.tensor([0, 2, 4], dtype=torch.int32)
    return q, k, v, a, b, A_log, dt_bias, state, indices, cu


def test_cpu_declines_without_importing_or_launching_vllm_kernel():
    before = native_core_bv64_stats()
    q, k, v, a, b, A_log, dt_bias, state, indices, cu = _cpu_operands()
    assert not _eligible(q, k, v, a, b, A_log, dt_bias, state, indices, cu)
    assert (
        maybe_native_core_bv64(
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            ssm_state=state,
            state_indices=indices,
            cu_seqlens=cu,
        )
        is None
    )
    after = native_core_bv64_stats()
    assert after["served"] == before["served"]
    assert after["declined"] == before["declined"] + 1


def test_decode_and_missing_metadata_decline_before_device_checks(monkeypatch):
    q, k, v, a, b, A_log, dt_bias, state, indices, cu = _cpu_operands()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("device capability must not be queried on a fallback path")

    monkeypatch.setattr(torch.cuda, "get_device_capability", forbidden)
    assert not _eligible(q, k, v, a, b, A_log, dt_bias, state, None, None)
    assert not _eligible(q, k, v, a, b, A_log, dt_bias, state, indices, cu)
