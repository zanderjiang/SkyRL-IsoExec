"""A hand-written VJP for the trainer's batched expert path.

IsoExec constrains the FORWARD only, so this module keeps the forward byte for byte -- same ops, order,
dtypes and chunking -- and replaces the backward with an analytic VJP that is free to reorder anything.
The backward never touches the tile layout: the permuted rows are already expert-grouped and contiguous,
so every gradient is one jagged grouped GEMM over the unpadded ``[T, *]`` rows delimited by device-resident
offsets. That removes the per-tile weight gather, the padded ``[n_tiles, cap, *]`` staging and the stacked
weight gradient in one step.

The only tensor saved beyond the forward's own inputs is the pre-glu ``inter`` in ROW form ``[T, 2f]``;
the rest of the epilogue is recomputed from it. Gathering ``inter`` back to row form is an exact op, so
the forward's numerics are untouched. The pik-fc2 leaf tree needs no special handling: a K-split followed
by additions has the same gradient to fp accuracy, though not bitwise the leaf tree's own autograd.
"""

from __future__ import annotations

import os

import torch

from ...autofuse.bwd_compile import call_region
from .moe_batched_experts import (
    _BMM_MAX_ELEMS,
    _CAP,
    _STATIC_DECODE,
    _STATIC_MAX_ROWS,
    _fc2_n_leaves,
    _leaftree_fc2,
    _moe_pik_fc2_on,
)


# FORWARD -- a byte-for-byte transcription of moe_batched_experts._batched_experts_forward. It is COPIED
# rather than called because the VJP needs the pre-glu `inter` in ROW form and the production function does
# not return it. Every op below matches the production op in order and dtype; the only addition is the
# `inter_rows` gather, which is exact and cannot perturb `out`.
def _forward_core(x, w1p, w2p, probs, counts, E, cfg, *, pik_fc2, n_leaves, want_inter):
    """``x [T,h]`` expert-grouped, ``w1p [E,2f,h]``, ``w2p [E,h,f]`` (Megatron parameter layout).

    Returns ``(out, inter_rows, cu)``. ``inter_rows`` is the PRE-GLU fc1 output ``[T, 2f]`` (or None).
    """
    dev = x.device
    T = x.shape[0]
    cap = _CAP

    cu = torch.zeros(E + 1, device=dev, dtype=torch.long)
    cu[1:] = counts.cumsum(0)
    idx = torch.arange(T, device=dev)
    tok_expert = torch.searchsorted(cu[1:], idx, right=True).clamp_(max=E - 1)
    off = idx - cu[tok_expert]

    if _STATIC_DECODE and T <= _STATIC_MAX_ROWS:
        n_tiles = -(-T // cap) + E
        n_tiles_e = (counts + cap - 1) // cap
        tile_cu = torch.zeros(E + 1, device=dev, dtype=torch.long)
        tile_cu[1:] = n_tiles_e.cumsum(0)
        tile_idx = tile_cu[tok_expert] + off // cap
        row_idx = off % cap
        tile_expert = torch.searchsorted(tile_cu[1:], torch.arange(n_tiles, device=dev), right=True).clamp_(max=E - 1)
    else:
        n_tiles_e = (counts + cap - 1) // cap
        tile_cu = torch.zeros(E + 1, device=dev, dtype=torch.long)
        tile_cu[1:] = n_tiles_e.cumsum(0)
        n_tiles = int(tile_cu[-1])
        tile_idx = tile_cu[tok_expert] + off // cap
        row_idx = off % cap
        tile_expert = torch.repeat_interleave(torch.arange(E, device=dev), n_tiles_e)

    xp = x.new_zeros(n_tiles, cap, x.shape[-1])
    xp[tile_idx, row_idx] = x
    pp = probs.new_zeros(n_tiles, cap)
    pp[tile_idx, row_idx] = probs

    w1 = w1p.transpose(1, 2)  # [E,h,2f]
    w2 = w2p.transpose(1, 2)  # [E,f,h]

    per_tile = max(cap * x.shape[-1], w1.shape[1] * w1.shape[2], cap * w1.shape[2])
    tiles_per_chunk = max(1, _BMM_MAX_ELEMS // max(1, per_tile))

    inter_rows = x.new_empty(T, w1p.shape[1]) if want_inter else None
    outs = []
    for s in range(0, n_tiles, tiles_per_chunk):
        e = min(s + tiles_per_chunk, n_tiles)
        inter = torch.ops.aten.bmm(xp[s:e], w1[tile_expert[s:e]])  # [chunk, cap, 2f]

        if want_inter:
            # exact gather of this chunk's valid rows back to row form. `sel` is the rows whose tile
            # falls in [s, e); the scatter writes them at their own row index, so no ordering is assumed.
            sel = (tile_idx >= s) & (tile_idx < e)
            r = sel.nonzero(as_tuple=True)[0]
            inter_rows[r] = inter[tile_idx[r] - s, row_idx[r]]

        x_glu, x_linear = torch.chunk(inter, 2, dim=-1)
        if (val := cfg.activation_func_clamp_value) is not None:
            x_glu = x_glu.clamp(min=None, max=val)
            x_linear = x_linear.clamp(min=-val, max=val)
        inter = cfg.activation_func(x_glu) * (x_linear + cfg.glu_linear_offset)

        original_dtype = inter.dtype
        inter = inter * pp[s:e].unsqueeze(-1)
        inter = inter.to(original_dtype)

        outs.append(
            # stack + map, not a gathered slice: the in-GEMM leaf tree indexes the stack per
            # program, and _leaftree_fc2 runs the gather only if it falls back to a buffer path.
            _leaftree_fc2(inter, None, n_leaves, w2_stack=w2, idx=tile_expert[s:e])
            if pik_fc2
            else torch.ops.aten.bmm(inter.contiguous(), w2[tile_expert[s:e]])
        )

    out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
    return out[tile_idx, row_idx], inter_rows, cu


# BACKWARD -- jagged grouped GEMMs over the unpadded expert-grouped rows.
def _pointwise_chunk_rows() -> int:
    rows = int(os.environ.get("SKYRL_ISOEXEC_MOE_BWD_POINTWISE_ROWS", "16384"))
    if rows <= 0:
        raise ValueError("SKYRL_ISOEXEC_MOE_BWD_POINTWISE_ROWS must be positive")
    return rows


def _grouped_mm(lhs: torch.Tensor, rhs: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """One jagged grouped GEMM; ``offsets[e]`` is the exclusive end row of expert ``e``."""
    return torch.ops.aten._grouped_mm.default(lhs, rhs, offsets)


def _epilogue_values(inter: torch.Tensor, probs: torch.Tensor, cfg, out_dtype: torch.dtype):
    """One independent row chunk of forward epilogue values needed by the VJP."""
    gate, up = inter.chunk(2, dim=-1)
    val = cfg.activation_func_clamp_value
    if val is not None:
        gate_c = gate.clamp(min=None, max=val)
        up_c = up.clamp(min=-val, max=val)
    else:
        gate_c, up_c = gate, up
    sg = torch.sigmoid(gate_c.float())
    silu = gate_c.float() * sg
    lin = up_c.float() + cfg.glu_linear_offset
    h = silu * lin
    hs = (h * probs.unsqueeze(-1)).to(out_dtype)
    return gate, up, gate_c, sg, silu, lin, h, hs


def _fastbwd_epilogue_hs_region(
    inter: torch.Tensor,
    probs: torch.Tensor,
    clamp_value,
    glu_linear_offset,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """One ATen-heavy backward-only recompute chunk for the fc2 input.

    Backward-only by construction: called only from :func:`_build_hs_chunked`, and it must never be
    reused by the copied forward, since a backward seam that leaked into scoring is invisible to the
    IsoExec gate. Scalars are explicit arguments so clamp/offset changes rotate the backward-compile key
    instead of hiding in a mutable config object.
    """
    gate, up = inter.chunk(2, dim=-1)
    if clamp_value is not None:
        gate = gate.clamp(min=None, max=clamp_value)
        up = up.clamp(min=-clamp_value, max=clamp_value)
    gate_f = gate.float()
    h = (gate_f * torch.sigmoid(gate_f)) * (up.float() + glu_linear_offset)
    return (h * probs.unsqueeze(-1)).to(out_dtype)


def _fastbwd_epilogue_vjp_region(
    inter: torch.Tensor,
    probs: torch.Tensor,
    dhs: torch.Tensor,
    clamp_value,
    glu_linear_offset,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One ATen-heavy backward-only SwiGLU/probability VJP chunk.

    The grouped GEMMs and ordered dispatch stay outside this region: the compiler may fuse or reassociate
    the row-local derivative and the per-row ``dprobs`` reduction, but cannot alter expert segmentation,
    GEMM providers, route order, or any forward/no-grad arithmetic.
    """
    gate, up = inter.chunk(2, dim=-1)
    if clamp_value is not None:
        gate_c = gate.clamp(min=None, max=clamp_value)
        up_c = up.clamp(min=-clamp_value, max=clamp_value)
    else:
        gate_c, up_c = gate, up
    gate_f = gate_c.float()
    sg = torch.sigmoid(gate_f)
    silu = gate_f * sg
    lin = up_c.float() + glu_linear_offset
    h = silu * lin
    dhs_f = dhs.float()
    dprobs = (dhs_f * h).sum(-1).to(probs.dtype)
    dh = dhs_f * probs.unsqueeze(-1).float()
    dgate = dh * lin * (sg * (1.0 + gate_f * (1.0 - sg)))
    dup = dh * silu
    if clamp_value is not None:
        dgate = dgate * (gate <= clamp_value).float()
        dup = dup * (up.abs() <= clamp_value).float()
    return torch.cat([dgate, dup], dim=-1).to(out_dtype), dprobs


def _build_hs_chunked(inter_rows: torch.Tensor, probs: torch.Tensor, cfg, out_dtype: torch.dtype) -> torch.Tensor:
    """Recompute the fc2 input while bounding all fp32 temporaries by a row chunk."""
    rows, two_f = inter_rows.shape
    hs = torch.empty((rows, two_f // 2), dtype=out_dtype, device=inter_rows.device)
    chunk_rows = _pointwise_chunk_rows()
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        hs_chunk = call_region(
            "moe.fastbwd.epilogue_hs_chunk",
            _fastbwd_epilogue_hs_region,
            inter_rows[start:stop],
            probs[start:stop],
            cfg.activation_func_clamp_value,
            cfg.glu_linear_offset,
            out_dtype,
        )
        hs[start:stop].copy_(hs_chunk)
    return hs


def _build_dinter_chunked(
    inter_rows: torch.Tensor,
    probs: torch.Tensor,
    dhs: torch.Tensor,
    cfg,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the epilogue VJP in chunks and recycle custom-Function-private saved storage."""
    rows = inter_rows.shape[0]
    dinter = inter_rows.detach()
    dprobs = torch.empty_like(probs)
    chunk_rows = _pointwise_chunk_rows()
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        dinter_chunk, dprobs_chunk = call_region(
            "moe.fastbwd.epilogue_vjp_chunk",
            _fastbwd_epilogue_vjp_region,
            inter_rows[start:stop],
            probs[start:stop],
            dhs[start:stop],
            cfg.activation_func_clamp_value,
            cfg.glu_linear_offset,
            out_dtype,
        )
        dinter[start:stop].copy_(dinter_chunk)
        dprobs[start:stop].copy_(dprobs_chunk)
    return dinter, dprobs


def _segment_backward(dout, x, inter_rows, probs, w1p, w2p, offsets, host_counts, cfg=None):
    """The VJP proper. All buffers are ``[T, *]`` row-major and expert-grouped; expert ``e`` owns
    ``offsets[e-1] : offsets[e]`` (with an implicit zero at the left edge).

    Forward being differentiated (per row ``i`` of expert ``e``)::

        inter   = x_i @ w1p[e].T                                  # [2f]
        g, u    = inter.chunk(2)             (optionally clamped)
        h       = silu(g) * (u + offset)                          # [f]
        hs      = (h * probs_i).to(dtype)
        out_i   = hs @ w2p[e].T                                   # [H]

    Note what is NOT here: the tile grid, the ``[n_tiles, cap, *]`` staging and the per-tile weight
    gather. Those exist so the FORWARD's bmm gives every token a fixed block regardless of routing; the
    backward has no such constraint, so it works on the contiguous expert segments directly and produces
    the weight gradients at ``[E, ...]`` instead of at ``[n_tiles, ...]`` plus a reduction.
    """
    # Compatibility for direct probes written against the original private seam:
    # ``_segment_backward(..., cu_host, cfg)``. Production enters with device offsets plus optional host
    # counts; accepting the old form keeps a host conversion off the live path.
    if cfg is None:
        cfg = host_counts
        cu_host_compat = [int(value) for value in offsets]
        host_counts = [cu_host_compat[i + 1] - cu_host_compat[i] for i in range(len(cu_host_compat) - 1)]
        offsets = torch.tensor(cu_host_compat[1:], dtype=torch.long, device=x.device)

    dt = x.dtype
    dout = dout.to(dt)

    grouped_offsets = offsets.to(torch.int32)

    # `hs` -- the fc2 input, probs already folded in -- is RECOMPUTED from the saved pre-glu `inter`
    # rather than saved. Chunking bounds every fp32 temporary; the bf16 storage is subsequently
    # recycled for dhs after each corresponding wgrad has consumed hs.
    hs = _build_hs_chunked(inter_rows, probs, cfg, dt)

    # dhs: gradient w.r.t. the fc2 INPUT.
    # 2D x 2D partitions the contraction dimension and returns [E,H,f] wgrad; 2D x 3D
    # partitions rows and returns compact [T,f] dgrad.  No padding and no expert loop.
    dw2 = _grouped_mm(dout.t(), hs, grouped_offsets)
    dhs = _grouped_mm(dout, w2p, grouped_offsets)
    # Both grouped calls have consumed the recomputed fc2 input. Drop it before the fp32
    # epilogue VJP so the compact hs+dhs overlap is transient rather than pointwise-live.
    del hs

    # Epilogue VJP: consume each private inter_rows chunk completely, then reuse those bytes for
    # dinter. This is legal because the custom Function is inter_rows' only owner/reader.
    dinter, dprobs = _build_dinter_chunked(inter_rows, probs, dhs, cfg, dt)

    dw1 = _grouped_mm(dinter.t(), x, grouped_offsets)
    dx = _grouped_mm(dinter, w1p, grouped_offsets)

    return dx, dw1, dw2, dprobs


class _MoEExpertsFastBwd(torch.autograd.Function):
    """Bitwise-production forward, analytic jagged-grouped backward.

    The expert Parameters are passed VARIADICALLY (``*w_params`` = E fc1 weights then E fc2 weights), so
    the gradient reaches each ``linear_fc1/linear_fc2.weight`` directly as a slice of ``dw1``/``dw2``.
    There is no in-graph ``torch.stack``, hence no stacked gradient to split back out.
    """

    @staticmethod
    def forward(ctx, x, probs, counts, E, cfg, pik_fc2, n_leaves, host_counts, outer_grad_enabled, *w_params):
        w1p = torch.stack(w_params[:E])  # [E, 2f, h]  -- no_grad, so this is not in any graph
        w2p = torch.stack(w_params[E:])  # [E, h, f]
        # ``autograd.Function.forward`` itself always runs with grad mode disabled. Capture the
        # caller's mode before ``apply`` and pass it in, so scoring/checkpoint forwards do not
        # materialize an inter_rows buffer that only the recompute backward can consume.
        need = bool(outer_grad_enabled) and (
            ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or any(ctx.needs_input_grad[9:])
        )
        with torch.no_grad():
            out, inter_rows, cu = _forward_core(
                x, w1p, w2p, probs, counts, E, cfg, pik_fc2=pik_fc2, n_leaves=n_leaves, want_inter=need
            )
        # The jagged grouped operator consumes DEVICE offsets. Saving the long view and converting it
        # only inside backward means the copied gate-critical forward adds no kernel. If the caller
        # already had CPU counts, preserve those so the legacy fallback can reuse them without a D2H read.
        offsets = cu[1:]
        ctx.save_for_backward(x, probs, inter_rows, w1p, w2p, offsets)
        ctx.E, ctx.cfg, ctx.host_counts = E, cfg, host_counts
        return out

    @staticmethod
    def backward(ctx, dout):
        x, probs, inter_rows, w1p, w2p, offsets = ctx.saved_tensors
        if inter_rows is None:
            raise RuntimeError(
                "[isoexec-moe] fast-bwd: forward ran without saving `inter` but a gradient was "
                "requested. needs_input_grad was wrong -- refusing to return a silent zero gradient."
            )
        dx, dw1, dw2, dprobs = _segment_backward(
            dout.contiguous(), x, inter_rows, probs, w1p, w2p, offsets, ctx.host_counts, ctx.cfg
        )
        # Return per-expert SLICES: each expert Parameter gets its own gradient, no split step.
        return (dx, dprobs, None, None, None, None, None, None, None, *dw1.unbind(0), *dw2.unbind(0))


def moe_experts_fastbwd(x, probs, counts, E, cfg, w1_params, w2_params, *, pik_fc2=None, n_leaves=None):
    """Tensor-level API. ``w1_params``/``w2_params`` are the E per-expert Parameters (NOT stacked).

    ``x [T,h]`` expert-grouped, ``probs [T]``, ``counts [E]`` (long, device or host),
    ``w1_params[e] : [2f, h]``, ``w2_params[e] : [h, f]``. Returns ``out [T, h]`` (fp32 under pik-fc2).
    """
    if pik_fc2 is None:
        pik_fc2 = _moe_pik_fc2_on()
    if n_leaves is None:
        n_leaves = _fc2_n_leaves() if pik_fc2 else 1
    host_counts = counts.tolist() if counts.device.type == "cpu" else None
    counts = counts.to(device=x.device, dtype=torch.long)
    outer_grad_enabled = torch.is_grad_enabled()
    return _MoEExpertsFastBwd.apply(
        x,
        probs,
        counts,
        E,
        cfg,
        pik_fc2,
        n_leaves,
        host_counts,
        outer_grad_enabled,
        *w1_params,
        *w2_params,
    )


def batched_experts_forward_fastbwd(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs):
    """Drop-in ``SequentialMLP.forward`` installed only by the default-off FASTBWD admission."""

    x = permuted_local_hidden_states
    if x.shape[0] == 0:
        return x.new_zeros(0, x.shape[-1]), None

    out = moe_experts_fastbwd(
        x,
        permuted_probs,
        tokens_per_expert,
        self.num_local_experts,
        self.config,
        [e.linear_fc1.weight for e in self.local_experts],
        [e.linear_fc2.weight for e in self.local_experts],
    )
    return out, None


def install_fastbwd_experts() -> bool:
    """Install the production analytic backward."""
    from megatron.core.transformer.moe.experts import SequentialMLP

    if getattr(SequentialMLP, "_isoexec_fastbwd", False):
        return True
    SequentialMLP.forward = batched_experts_forward_fastbwd
    SequentialMLP._isoexec_fastbwd = True
    print(
        "[ISOEXEC-MOE] SequentialMLP.forward -> bitwise forward + analytic segment-GEMM backward; "
        "NOT ENGAGEMENT -- require [ISOEXEC-MOE-FASTBWD] served>0",
        flush=True,
    )
    return True
