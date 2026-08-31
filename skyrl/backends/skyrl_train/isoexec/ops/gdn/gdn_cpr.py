"""Canonical CPR GDN forward for trainer and engine prefill.

The FLA state pass gives each chunk's boundary state; a recurrent scan computes within-chunk outputs
from the bf16 snapshot upcast to fp32. Both that snapshot dtype and the fp32 chain are contract.
"""

from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np
import torch

# Host-built chunk metadata: the vendored prepare_chunk_indices/offsets read lengths back from the
# device. Pure metadata, so identical index tensors give identical kernel arithmetic.
_META_CACHE: OrderedDict = OrderedDict()
_META_CACHE_MAX = 512


def host_meta_enabled() -> bool:
    """``SKYRL_ISOEXEC_GDN_CPR_HOST_META`` (default on); both paths give equal tensors, so it moves no bits."""
    return os.environ.get("SKYRL_ISOEXEC_GDN_CPR_HOST_META", "1").lower() not in ("0", "false", "no", "")


def host_chunk_meta(lens, chunk_size: int, device, dtype=torch.int32):
    """``(chunk_indices, chunk_offsets)`` for host-known ``lens``, byte-identical to the vendored pair."""
    key = (tuple(int(x) for x in lens), int(chunk_size), str(device), dtype)
    hit = _META_CACHE.get(key)
    if hit is not None:
        _META_CACHE.move_to_end(key)
        return hit
    nb = np.asarray([(int(L) + chunk_size - 1) // chunk_size for L in lens], dtype=np.int64)
    if nb.size:
        starts = np.cumsum(nb) - nb
        seq = np.repeat(np.arange(nb.size, dtype=np.int64), nb)
        ch = np.arange(int(nb.sum()), dtype=np.int64) - np.repeat(starts, nb)
        idx_np = np.stack([seq, ch], 1)
        off_np = np.concatenate([[0], np.cumsum(nb)])
    else:  # pragma: no cover - empty batch
        idx_np = np.zeros((0, 2), dtype=np.int64)
        off_np = np.zeros(1, dtype=np.int64)
    # Indices follow cu's dtype (int32) but offsets MUST be int64 (the vendored cumsum promotion);
    # an int32 offsets tensor mis-strides the fwd_h loads.
    idx = torch.from_numpy(idx_np).to(device=device, dtype=dtype)
    off = torch.from_numpy(off_np).to(device=device, dtype=torch.int64)
    _META_CACHE[key] = (idx, off)
    if len(_META_CACHE) > _META_CACHE_MAX:
        _META_CACHE.popitem(last=False)
    return idx, off


# Per-forward sequence-metadata memo, keyed on cu_seqlens identity + version counter and holding a
# strong ref: the ref makes id() unforgeable, and the version counter is shared across views.
_SEQ_CACHE: OrderedDict = OrderedDict()
_SEQ_CACHE_MAX = 8
_SEQ_STATS = {"served": 0, "built": 0, "declined": 0}


def seq_meta_cache_enabled() -> bool:
    """``SKYRL_ISOEXEC_GDN_SEQ_META_CACHE`` (default on); both paths give the same tensors."""
    return os.environ.get("SKYRL_ISOEXEC_GDN_SEQ_META_CACHE", "1").lower() not in ("0", "false", "no", "")


def seq_meta_census() -> dict:
    """``{served, built, declined}`` counters."""
    return dict(_SEQ_STATS)


def _seq_cache_key(cu_seqlens: torch.Tensor, chunk_size: int):
    """The memo key, or ``None`` when this tensor is too large to pin alive."""
    if not torch.is_tensor(cu_seqlens) or cu_seqlens.numel() > 8192:
        return None
    return (id(cu_seqlens), cu_seqlens._version, int(chunk_size), str(cu_seqlens.device))


def sequence_metadata(cu_seqlens: torch.Tensor, chunk_size: int, device):
    """``(lens, cu_fla, cu_chunked, state_indices, n_chunks)`` for a packed batch, memoized.

    ``cu_fla`` is the identity-stable clone the FLA stages consume.
    """
    key = _seq_cache_key(cu_seqlens, chunk_size) if seq_meta_cache_enabled() else None
    if key is None:
        _SEQ_STATS["declined"] += 1
        lens = packed_lens(cu_seqlens)
        cu_chunked, state_indices, n_chunks = chunked_cu_seqlens_and_indices(cu_seqlens, chunk_size, device, lens=lens)
        return lens, cu_seqlens.clone(), cu_chunked, state_indices, n_chunks
    hit = _SEQ_CACHE.get(key)
    if hit is not None:
        _SEQ_CACHE.move_to_end(key)
        _SEQ_STATS["served"] += 1
        return hit[1]
    lens = packed_lens(cu_seqlens)
    cu_chunked, state_indices, n_chunks = chunked_cu_seqlens_and_indices(cu_seqlens, chunk_size, device, lens=lens)
    entry = (lens, cu_seqlens.clone(), cu_chunked, state_indices, n_chunks)
    # The strong ref to cu_seqlens makes the id() half of the key unforgeable; never drop it.
    _SEQ_CACHE[key] = (cu_seqlens, entry)
    _SEQ_STATS["built"] += 1
    if len(_SEQ_CACHE) > _SEQ_CACHE_MAX:
        _SEQ_CACHE.popitem(last=False)
    return entry


def chunk_boundary_states(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    lens: "list[int] | None" = None,
    cu_fla: torch.Tensor | None = None,
):
    """Per-chunk boundary states via the vendored chunk STATE pass; ``(h, final_state, chunk_indices)``.

    ``h`` is ``[N, NT, HV, V, K]``: the bf16 snapshot entering chunk ``c`` of sequence ``n``.
    ``final_state`` is the fp32 accumulator at sequence end -- the chaining dtype.
    ``initial_state`` is ``[N, HV, V, K]`` fp32 entry states or None. q is unused here (o-pass only).
    ``lens`` / ``cu_fla`` are optional host lengths and the identity-stable cu clone; both move time only.
    """
    from vllm.model_executor.layers.fla.ops.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h,
    )
    from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import (
        chunk_scaled_dot_kkt_fwd,
    )
    from vllm.model_executor.layers.fla.ops.cumsum import chunk_local_cumsum
    from vllm.model_executor.layers.fla.ops.index import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
    )
    from vllm.model_executor.layers.fla.ops.solve_tril import solve_tril
    from vllm.model_executor.layers.fla.ops.wy_fast import recompute_w_u_fwd

    from .gdn_batch_invariant import pin_fla_autotune_configs

    # The chunk state pass is autotuned; batch-invariant boundary states require the pinned config.
    pin_fla_autotune_configs()

    # `prepare_chunk_indices` is @tensor_cache'd on tensor identity; clone so a recycled buffer id
    # cannot return a stale index.
    cu = cu_seqlens.clone() if cu_fla is None else cu_fla
    if lens is not None and host_meta_enabled():
        chunk_indices, chunk_offsets = host_chunk_meta(lens, chunk_size, cu.device, cu.dtype)
    else:
        chunk_indices = prepare_chunk_indices(cu, chunk_size)
        chunk_offsets = prepare_chunk_offsets(cu, chunk_size)

    g_cs = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu, chunk_indices=chunk_indices)
    # `chunk_size=` is required: this stage fixes BT for the rest of the chain (solve_tril and
    # recompute_w_u_fwd read it off `A.shape[-1]`), so omitting it breaks any C != FLA_CHUNK_SIZE.
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g=g_cs,
        cu_seqlens=cu,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )
    A = solve_tril(A=A, cu_seqlens=cu, chunk_indices=chunk_indices, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g_cumsum=g_cs, cu_seqlens=cu, chunk_indices=chunk_indices)
    h, _v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g_cs,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        cu_seqlens=cu,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    return h, final_state, chunk_indices


def packed_lens(cu_seqlens: torch.Tensor) -> list[int]:
    """Host sequence lengths of a packed batch. One device read; callers pass the result around."""
    return (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()


def chunked_cu_seqlens_and_indices(cu_seqlens: torch.Tensor, chunk_size: int, device, lens=None):
    """Split a packed batch at every chunk boundary; ``(cu_chunked, state_indices, n_chunks)``.

    ``state_indices`` is ``[M, chunk_size]``: only column 0 is set (the chunk's pool row), the rest stay 0
    so the ``idx <= 0`` guard skips the store. Rows count from 1 (row 0 is NULL) and the map must be full
    width -- a ``[M, 1]`` map faults, because the store still reads up to column ``chunk_size - 1``.
    """
    if lens is None:  # host; a trainer forward already syncs here
        lens = packed_lens(cu_seqlens)
    offsets = [0]
    idx_rows = []
    row = 1  # row 0 of the padded pool is the NULL row
    for seq_len in lens:
        pos = 0
        while pos < seq_len:
            seg = min(chunk_size, seq_len - pos)
            offsets.append(offsets[-1] + seg)
            idx_rows.append(row)
            row += 1
            pos += seg
    cu_chunked = torch.tensor(offsets, dtype=torch.int32, device=device)
    M = len(idx_rows)
    state_indices = torch.zeros(M, chunk_size, dtype=torch.int32, device=device)
    state_indices[:, 0] = torch.tensor(idx_rows, dtype=torch.int32, device=device)
    return cu_chunked, state_indices, row - 1


def _state_pool(h: torch.Tensor, n_chunks: int, device) -> torch.Tensor:
    """The fp32 state pool: row 0 is the NULL row, rows 1.. are ``h``'s boundary states in chunk order."""
    h_flat = h.reshape(-1, *h.shape[-3:])
    pool = torch.empty(n_chunks + 1, *h_flat.shape[-3:], dtype=torch.float32, device=device)
    pool[0].zero_()  # the NULL row; rows 1.. are written in full below
    pool[1 : n_chunks + 1].copy_(h_flat[:n_chunks])  # copy_ does the widening bf16 -> fp32 cast
    return pool


def native_matched_prep(k_raw: torch.Tensor, a: torch.Tensor, b: torch.Tensor, A_log, dt_bias):
    """The boundary-pass prep: eager l2norm and sigmoid gating from raw inputs.

    The canonical pin shared by trainer forward, engine prefill and decode resync, which is what makes
    their boundary states identical. It does NOT reproduce the fused kernel's intermediates bitwise.
    """
    # The middle torch.sum stays eager: its fp32 reduction tree is part of the pin.
    from .gdn_matched_prep_gate import (
        maybe_matched_prep_fused_gate,
        maybe_matched_prep_fused_l2,
    )

    kn = maybe_matched_prep_fused_l2(k_raw)
    if kn is None:
        kf = k_raw.float()
        kn = (kf * torch.rsqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)).to(k_raw.dtype)

    # The fused gate writes beta in b's wire dtype; mixed-wire callers stay on the eager expression.
    fused = maybe_matched_prep_fused_gate(a, b, A_log, dt_bias) if a.dtype == k_raw.dtype else None
    if fused is None:
        x = a.float() + dt_bias.float()
        softplus = torch.where(x <= 20.0, torch.log(1.0 + torch.exp(x)), x)
        g = -torch.exp(A_log.float()) * softplus
        beta = torch.sigmoid(b.float()).to(k_raw.dtype)
    else:
        g, beta = fused
        beta = beta.to(k_raw.dtype)
    return kn, g, beta


def gdn_native_cpr_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Canonical native trainer/prefill forward: chunk-pass boundary ``h_c`` plus one segmented
    fused_sigmoid scan with per-chunk initial states.

    q/k are raw and un-normalised (compressed GQA heads are fine); a/b are raw gating inputs ``[T, HV]``.
    """
    from .gdn_ops import gdn_native_core_kernel

    dev = q.device
    T = q.shape[1]
    lens, cu_fla, cu_chunked, state_indices, n_chunks = sequence_metadata(cu_seqlens, chunk_size, dev)
    kn, g, beta = native_matched_prep(k, a, b, A_log, dt_bias)
    h, _final, _ci = chunk_boundary_states(
        None,
        kn,
        v,
        g.reshape(1, T, -1),
        beta.reshape(1, T, -1),
        cu_seqlens,
        chunk_size,
        lens=lens,
        cu_fla=cu_fla,
    )
    pool = _state_pool(h, n_chunks, dev)
    return gdn_native_core_kernel(
        q, k, v, a, b, A_log, dt_bias, ssm_state=pool, state_indices=state_indices, cu_seqlens=cu_chunked
    )


def gdn_cpr_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Canonical trainer/prefill forward: chunk-state boundary ``h_c`` + segmented recurrent scan.

    q/k must already be L2-normalised and GQA-expanded to ``H == HV``; g/beta are raw per-token.
    Shapes: q,k ``[1, T, HV, K]``; v ``[1, T, HV, V]``; g,beta ``[1, T, HV]``; packed (B==1) with
    ``cu_seqlens`` [N+1]. Returns o ``[1, T, HV, V]``.
    """
    from .gdn_ops import gdn_recurrent_kernel

    dev = q.device
    lens, cu_fla, cu_chunked, state_indices, n_chunks = sequence_metadata(cu_seqlens, chunk_size, dev)
    h, _final, _ci = chunk_boundary_states(q, k, v, g, beta, cu_seqlens, chunk_size, lens=lens, cu_fla=cu_fla)

    # Packed varlen collapses (N, NT) chunk-major, in the order chunked_cu_seqlens_and_indices numbers rows.
    pool = _state_pool(h, n_chunks, dev)

    # One segmented recurrent scan: each chunk is a sequence starting from its h_c.
    o = gdn_recurrent_kernel(q, k, v, g, beta, ssm_state=pool, state_indices=state_indices, cu_seqlens=cu_chunked)
    return o
