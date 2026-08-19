"""Canonical chunk-synced GDN forward for trainer and engine prefill.

The FLA state pass computes each chunk's boundary state; a fused recurrent scan then computes the
within-chunk outputs, restarting from the corresponding bf16 boundary snapshot upcast to fp32, while the
cross-chunk chain retains FLA's fp32 final state. The bf16 snapshot and the fp32 chain are both part of
the arithmetic contract: substituting an algebraically equivalent state changes output bytes.
"""

from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np
import torch

# Host-built chunk metadata. The vendored `prepare_chunk_indices` / `prepare_chunk_offsets` describe
# "which (sequence, chunk) does program i handle" -- metadata only, so identical index tensors give
# identical kernel arithmetic -- but they read sequence lengths back from the device and build one
# tensor per sequence in python. Our call sites already know the lengths on the host, so the same
# tensors are built in numpy instead: no device read, and bitwise-neutral by construction.
_META_CACHE: OrderedDict = OrderedDict()
_META_CACHE_MAX = 512


def host_meta_enabled() -> bool:
    """``SKYRL_ISOEXEC_GDN_CS_HOST_META`` (default on): host-built chunk metadata.

    Off restores the vendored ``prepare_chunk_indices``/``prepare_chunk_offsets`` path; the two produce
    ``torch.equal`` tensors, so the flag moves time, never bits.
    """
    return os.environ.get("SKYRL_ISOEXEC_GDN_CS_HOST_META", "1").lower() not in ("0", "false", "no", "")


def host_chunk_meta(lens, chunk_size: int, device, dtype=torch.int32):
    """``(chunk_indices, chunk_offsets)`` for host-known sequence lengths ``lens``.

    Byte-identical to ``prepare_chunk_indices``/``prepare_chunk_offsets`` for the ``cu`` those lengths
    describe. Small LRU on ``(lens, chunk_size, device)``, since decode resync cycles through a handful
    of uniform grids.
    """
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
    # Dtypes are part of the contract: indices follow cu's dtype (int32) but offsets must be int64,
    # matching the vendored cumsum promotion. An int32 offsets tensor mis-strides the fwd_h loads.
    idx = torch.from_numpy(idx_np).to(device=device, dtype=dtype)
    off = torch.from_numpy(off_np).to(device=device, dtype=torch.int64)
    _META_CACHE[key] = (idx, off)
    if len(_META_CACHE) > _META_CACHE_MAX:
        _META_CACHE.popitem(last=False)
    return idx, off


# Per-forward sequence-metadata memo. Every quantity below is a pure function of the values in
# ``cu_seqlens`` (plus chunk_size and device), and ``cu_seqlens`` is the same object for every GDN layer
# of a microbatch, yet each layer otherwise rebuilds it -- including a pageable D2H read-back per layer.
# The memo is keyed on object identity plus version counter and holds a strong reference to the tensor:
# the strong ref makes ``id()`` unforgeable, and torch's version counter is shared across views, so any
# in-place write through any alias invalidates the entry. It returns the same tensors the rebuild would
# have produced, so it can only move time.
_SEQ_CACHE: OrderedDict = OrderedDict()
_SEQ_CACHE_MAX = 8
_SEQ_STATS = {"served": 0, "built": 0, "declined": 0}


def seq_meta_cache_enabled() -> bool:
    """``SKYRL_ISOEXEC_GDN_SEQ_META_CACHE`` (default on): memoize per-forward sequence metadata.

    Off restores the per-layer rebuild; both paths produce the same tensors, so the flag moves time only.
    """
    return os.environ.get("SKYRL_ISOEXEC_GDN_SEQ_META_CACHE", "1").lower() not in ("0", "false", "no", "")


def seq_meta_census() -> dict:
    """``{served, built, declined}`` counters."""
    return dict(_SEQ_STATS)


def _seq_cache_key(cu_seqlens: torch.Tensor, chunk_size: int):
    """The memo key, or ``None`` when this tensor must not be cached (non-tensor, or large enough that
    pinning it and its base storage alive would cost real memory)."""
    if not torch.is_tensor(cu_seqlens) or cu_seqlens.numel() > 8192:
        return None
    return (id(cu_seqlens), cu_seqlens._version, int(chunk_size), str(cu_seqlens.device))


def sequence_metadata(cu_seqlens: torch.Tensor, chunk_size: int, device):
    """``(lens, cu_fla, cu_chunked, state_indices, n_chunks)`` for a packed batch, memoized.

    ``cu_fla`` is the identity-stable clone the FLA stages consume (see the stale-``@tensor_cache``
    trap in :func:`chunk_boundary_states`); the other four are exactly what
    :func:`packed_lens` and :func:`chunked_cu_seqlens_and_indices` return.
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
    # The strong reference to cu_seqlens makes the id() half of the key unforgeable; never drop it
    # while the entry lives.
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
    """Per-chunk boundary states ``h`` via the vendored chunk STATE pass. Returns ``(h, final_state,
    chunk_indices)``.

    ``h`` has shape ``[N, NT, HV, V, K]`` (the chunk kernel's own layout): ``h[n, c]`` is the state
    ENTERING chunk ``c`` of sequence ``n`` (``h[n, 0]`` = ``initial_state`` or zero), stored as the
    kernel's bf16 SNAPSHOT. ``final_state`` (when requested) is the fp32 accumulator at sequence end
    -- the chaining dtype. ``NT`` is the total chunk count across the packed batch
    (``len(chunk_indices)``); the mapping from ``(n, c)`` to a row of the flat state pool is what
    :func:`chunked_cu_seqlens_and_indices` reproduces.

    ``initial_state``: ``[N, HV, V, K]`` per-sequence entry states (fp32 for the exact chain), or
    None for zero. The prep (``chunk_local_cumsum`` -> ``chunk_scaled_dot_kkt_fwd`` -> ``solve_tril``
    -> ``recompute_w_u_fwd``) is exactly ``chunk_gated_delta_rule_fwd``'s first half, lifted so we
    can stop before ``chunk_fwd_o``. q is unused by the state pass (it only feeds the o-pass), kept
    in the signature for call-site symmetry.

    ``lens``: the packed batch's host-known sequence lengths, if the caller has them; supplying them
    takes the chunk metadata off the vendored device-read-back path (:func:`host_chunk_meta`), with
    identical tensors either way.

    ``cu_fla``: the identity-stable clone of ``cu_seqlens`` to hand the FLA stages, if the caller already
    holds one (:func:`sequence_metadata`). Stable identity across a microbatch's GDN layers is what lets
    FLA's identity-keyed ``@tensor_cache`` hit instead of re-running a device read-back per layer.
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

    # The chunk state pass is autotuned; batch-invariant boundary states require the single pinned config.
    pin_fla_autotune_configs()

    # `prepare_chunk_indices` is @tensor_cache'd on tensor identity; clone cu_seqlens so a recycled
    # buffer id cannot return a stale index. Callers holding the per-forward clone pass it in.
    cu = cu_seqlens.clone() if cu_fla is None else cu_fla
    if lens is not None and host_meta_enabled():
        chunk_indices, chunk_offsets = host_chunk_meta(lens, chunk_size, cu.device, cu.dtype)
    else:
        chunk_indices = prepare_chunk_indices(cu, chunk_size)
        chunk_offsets = prepare_chunk_offsets(cu, chunk_size)

    g_cs = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu, chunk_indices=chunk_indices)
    # `chunk_size=` is required: this stage fixes BT for the rest of the chain (solve_tril and
    # recompute_w_u_fwd read BT off `A.shape[-1]` and take no chunk_size), and it otherwise defaults to
    # FLA_CHUNK_SIZE. Omitting it makes the pass non-chunk-local for any C != 64.
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
    """Split a packed batch at every chunk boundary. Returns ``(cu_chunked, state_indices, n_chunks)``.

    ``cu_chunked`` [M+1] cuts each sequence into ceil(len/C) segments of <= C tokens, so the recurrent
    kernel treats every chunk as its own sequence and reloads a fresh initial state at each boundary.
    ``state_indices`` is ``[M, chunk_size]``: the kernel indexes it per token (column ``i_t``) -- column 0
    is the initial-state row it loads, every column a candidate final-state store row. Only column 0 is
    set, to the chunk's flat pool row of ``h_c``; the rest stay 0 so the ``idx <= 0`` guard skips the
    store. Rows count from 1 so row 0 stays the NULL row. The map must be full width: a ``[M, 1]`` map
    faults, because the store still reads columns up to ``chunk_size - 1``.
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
    """The fp32 state pool the recurrent/native core loads from: row 0 is the NULL row, rows 1.. are the
    boundary states of ``h`` in chunk order.

    Allocated uninitialised: row 0 is explicitly zeroed and rows ``1..n_chunks`` are fully written by the
    ``copy_``, which performs the widening bf16 -> fp32 cast itself.
    """
    h_flat = h.reshape(-1, *h.shape[-3:])
    pool = torch.empty(n_chunks + 1, *h_flat.shape[-3:], dtype=torch.float32, device=device)
    pool[0].zero_()  # the NULL row; rows 1.. are written in full below
    pool[1 : n_chunks + 1].copy_(h_flat[:n_chunks])  # copy_ does the widening cast, exactly
    return pool


def native_matched_prep(k_raw: torch.Tensor, a: torch.Tensor, b: torch.Tensor, A_log, dt_bias):
    """The v2 boundary-pass prep: eager l2norm and sigmoid gating computed from raw inputs.

    The native-composition core keeps l2norm/gating in-kernel, but the boundary pass needs explicit
    ``k/g/beta``. This single definition is used by the trainer forward, engine prefill and the decode
    resync, which is what makes their boundary states identical. It is the canonical pin and does not
    reproduce the fused kernel's in-kernel intermediates bitwise: fp32 rsqrt-multiply l2norm rounded to
    the wire dtype, threshold-softplus fp32 gating (g stays fp32), sigmoid beta in the wire dtype.
    """
    # The middle torch.sum stays where it is: its fp32 reduction tree is part of the pin. Only the
    # square and post-reduction elementwise chains collapse around it.
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
        # Normally identity (the fused path admits matching wire dtypes); keeps the return contract explicit.
        beta = beta.to(k_raw.dtype)
    return kn, g, beta


def gdn_native_chunk_synced_fwd(
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
    """v2 canonical trainer/prefill forward: chunk-pass boundary ``h_c`` on the eager matched prep, plus
    one segmented native (fused_sigmoid) scan with per-chunk initial states.

    q/k are raw (un-normalised; compressed GQA heads are fine, both the fused core and the vendored state
    pass support ``Hg != HV``); a/b are the raw gating inputs ``[T, HV]``. Same segmented grid and the same
    bf16-snapshot/fp32-chain handoff as v1; only the within-chunk evaluator and boundary prep differ.
    """
    from .gdn_ops import gdn_native_core_kernel

    dev = q.device
    T = q.shape[1]
    # One device read per forward, not per layer: see :func:`sequence_metadata`.
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


def gdn_chunk_synced_fwd(
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

    q/k must already be L2-normalised and GQA-expanded to ``H == HV``, done identically on both runtimes.
    g/beta are raw per-token: the recurrent scan does its own cumulation, and only the boundary-state pass
    consumes the cumsum'd g, internally.

    Shapes: q,k ``[1, T, HV, K]``; v ``[1, T, HV, V]``; g,beta ``[1, T, HV]``; packed (B==1) with
    ``cu_seqlens`` [N+1]. Returns o ``[1, T, HV, V]``.
    """
    from .gdn_ops import gdn_recurrent_kernel

    dev = q.device
    # One device read per forward, not per layer: see :func:`sequence_metadata`.
    lens, cu_fla, cu_chunked, state_indices, n_chunks = sequence_metadata(cu_seqlens, chunk_size, dev)
    # (1) boundary states h[n, c] via the chunk state pass, in the chunk kernel's [N, NT, HV, V, K].
    h, _final, _ci = chunk_boundary_states(q, k, v, g, beta, cu_seqlens, chunk_size, lens=lens, cu_fla=cu_fla)

    # (2) flatten h into an fp32 [S, HV, V, K] pool with a leading NULL row. Packed varlen collapses
    # (N, NT) chunk-major, in exactly the order chunked_cu_seqlens_and_indices numbers the rows.
    pool = _state_pool(h, n_chunks, dev)

    # (3) One segmented recurrent scan: each chunk is a sequence starting from its h_c.
    o = gdn_recurrent_kernel(q, k, v, g, beta, ssm_state=pool, state_indices=state_indices, cu_seqlens=cu_chunked)
    return o
