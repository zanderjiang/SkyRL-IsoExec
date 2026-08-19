"""Geometry-gated Hopper launch for vLLM's native GDN recurrent core.

The upstream kernel fixes ``BV=32``; at ``K=V=128`` that runs four programs over the same token sequence
and key reduction where two suffice, so this path relaunches the same kernel with ``BV=64``. Only the
partition of independent V rows changes -- the floating-point expression for each output element is
unchanged. Admission is narrow (CUDA bf16 varlen prefill on SM90 at ``K=V=128``); anything else falls back
to vLLM's wrapper.
"""

from __future__ import annotations

import functools
import os

import torch

_K = 128
_V = 128
_BV = 64
_NUM_WARPS = 4
_NUM_STAGES = 3
_STATS = {"served": 0, "declined": 0}


@functools.cache
def _is_sm90(device_index: int) -> bool:
    """Cache the immutable device property instead of querying it once per GDN layer."""

    return torch.cuda.get_device_capability(device_index) == (9, 0)


def _eligible(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> bool:
    """Whether the BV64 launch is an exact structural substitute for the vendor wrapper."""

    tensors = (q, k, v, a, b, A_log, dt_bias, ssm_state)
    if cu_seqlens is None or state_indices is None:
        # Varlen prefill only; decode stays on the vendor path.
        return False
    if any(t.device.type != "cuda" for t in (*tensors, state_indices, cu_seqlens)):
        return False
    if len({t.device for t in (*tensors, state_indices, cu_seqlens)}) != 1:
        return False
    device_index = q.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    if not _is_sm90(device_index):
        return False
    if any(t.dtype != torch.bfloat16 for t in (q, k, v, a, b, A_log, dt_bias)):
        return False
    if ssm_state.dtype != torch.float32:
        return False
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or q.shape[0] != 1:
        return False
    B, T, H, K = q.shape
    if B != 1 or T == 0 or K != _K:
        return False
    if v.shape[:2] != (B, T) or v.shape[-1] != _V:
        return False
    HV = v.shape[2]
    if H == 0 or HV == 0 or HV % H != 0:
        return False
    if a.shape != (T, HV) or b.shape != (T, HV):
        return False
    if A_log.shape != (HV,) or dt_bias.shape != (HV,):
        return False
    if ssm_state.ndim != 4 or tuple(ssm_state.shape[1:]) != (HV, _V, _K):
        return False
    if state_indices.ndim not in (1, 2) or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        return False
    if state_indices.dtype not in (torch.int32, torch.int64) or cu_seqlens.dtype not in (
        torch.int32,
        torch.int64,
    ):
        return False
    if state_indices.shape[0] != cu_seqlens.numel() - 1:
        return False
    # The public wrapper would materialise copies for non-contiguous inputs; decline instead of
    # changing the allocation/launch sequence.
    if any(not t.is_contiguous() for t in (*tensors, state_indices, cu_seqlens)):
        return False
    return True


def maybe_native_core_bv64(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor | None:
    """Return the BV64 result, or ``None`` so the caller uses vLLM's unchanged wrapper."""

    if not _eligible(q, k, v, a, b, A_log, dt_bias, ssm_state, state_indices, cu_seqlens):
        _STATS["declined"] += 1
        return None

    from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update_kernel,
    )

    B, T, H, K = q.shape
    HV, V = v.shape[2:]
    N = cu_seqlens.numel() - 1
    if state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = state_indices.stride()

    # Matches vLLM's wrapper argument-for-argument; BV and hence grid.y are the only differences.
    # NK stays 1 because the admitted K is exactly BK=128.
    output = q.new_empty(1, *v.shape)
    grid = (1, 2, N * HV)
    fused_sigmoid_gating_delta_rule_update_kernel[grid](
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        beta=1.0,
        threshold=20.0,
        q=q,
        k=k,
        v=v,
        o=output,
        h0=ssm_state,
        ht=ssm_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        num_accepted_tokens=None,
        scale=K**-0.5,
        N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=_K,
        BV=_BV,
        stride_init_state_token=ssm_state.stride(0),
        stride_final_state_token=ssm_state.stride(0),
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        INPLACE_FINAL_STATE=True,
        USE_QK_L2NORM_IN_KERNEL=True,
        IS_KDA=False,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    _STATS["served"] += 1
    if _STATS["served"] == 1:
        print(
            f"[ISOEXEC-GDN-BV64] pid={os.getpid()} served=1 " f"T={T} N={N} H={H} HV={HV} K={K} V={V}",
            flush=True,
        )
    return output.squeeze(0)
