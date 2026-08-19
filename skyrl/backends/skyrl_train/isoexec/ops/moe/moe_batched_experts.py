"""Batched expert execution for Megatron's ``SequentialMLP``: replace the per-expert python loop with a
padded batched GEMM -- scatter each expert's token block into ``[E, M_pad, h]``, one ``torch.bmm``
against the stacked fc1 weights, the identical glu/probs epilogue, one ``bmm`` against stacked fc2
weights, then gather the valid rows back in expert order.

This is NOT bitwise-equal to the sequential loop (bmm and mm are different kernels), and does not need
to be: the trainer and the engine both run this same function, so rollout, scoring and training
numerics move together. What IsoExec requires instead is determinism, per-token invariance to the
routing of other tokens, and invariance to the padding amount ``M_pad`` -- all three properties of
vLLM's ``bmm_batch_invariant``, and asserted by :func:`verify_batched_experts_invariance`.

The stacked weights are built in-graph each forward, so autograd routes expert weight grads back to
each ``linear_fc1/linear_fc2.weight`` as the standard path does. The stack is memoized under
``SKYRL_ISOEXEC_MOE_WEIGHT_CACHE``; the memoized buffer is re-attached to autograd per forward.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)


# Rows per tile. A multiple of the batch-invariant bmm kernel's BLOCK_SIZE_M (128) so tiles align
# with its row-blocks. Staged rows are bounded by T + E*CAP regardless of routing skew.
_CAP = int(os.environ.get("SKYRL_ISOEXEC_MOE_TILE_ROWS", "128"))

# Token-derived tile capacity (SKYRL_ISOEXEC_MOE_TILE_CAP; "" keeps the pinned `_CAP` above).
#
# A larger cap collapses n_tiles towards E, shrinking the per-tile weight gather (`w1[tile_expert]`);
# when n_tiles reaches E exactly the gather disappears (tile id == expert id, `tile_expert=None`).
# But a tile is also the padding quantum, so the bmm's wasted padded rows grow linearly in cap, and at
# trainer dims that costs more than the gather saves -- hence default off.
#
#   "auto" -> clamp(round_up_128(ceil(T/E)), 128, 1024). T is host-known, so this costs no sync. The
#             1024 ceiling bounds staged rows (T + E*cap) and the padded-bmm FLOPs.
#   <int>  -> rows per tile, rounded UP to a multiple of 128 (the bmm's BLOCK_SIZE_M; the tiling
#             argument rests on a token's row-block being a fixed shape).
_TILE_CAP_ENV = "SKYRL_ISOEXEC_MOE_TILE_CAP"
_TILE_CAP_WARNED = set()


def _tile_cap(T: int, E: int) -> tuple[int, bool]:
    """(rows per tile, whether the override is active). Override off -> the pinned `_CAP`, unchanged."""
    spec = os.environ.get(_TILE_CAP_ENV, "").strip()
    if not spec:
        return _CAP, False
    if spec == "auto":
        rows = -(-T // max(1, E))  # ceil(T/E): the perfectly balanced tile
        return min(1024, max(128, -(-rows // 128) * 128)), True
    cap = -(-int(spec) // 128) * 128
    if cap != int(spec) and spec not in _TILE_CAP_WARNED:
        _TILE_CAP_WARNED.add(spec)
        logger.warning("[isoexec-moe] %s=%s rounded up to %d (must be a multiple of 128)", _TILE_CAP_ENV, spec, cap)
    return max(128, cap), True


# Hard ceiling on the elements in any ONE batch-invariant bmm operand: vLLM's Triton bmm computes
# offsets in int32 and reads out of bounds past 2^31 elements. The expert bmm is split along the tile
# axis to stay under it; 2^30 leaves a 2x margin at a cost of 2-3 extra launches per layer.
_BMM_MAX_ELEMS = int(os.environ.get("SKYRL_ISOEXEC_MOE_BMM_MAX_ELEMS", str(2**30)))

# Static-shape decode: the sync-free, shape-static expert path, which is what makes CUDA-graph capture
# possible. Taken whenever the routed-row count is small enough for a bounded tile grid. Prefill and the
# trainer forward fall through to the dynamic tile path; neither is ever graph-captured.
_STATIC_DECODE = os.environ.get("SKYRL_ISOEXEC_MOE_STATIC_DECODE", "1") == "1"
# Take the static path only when T is small enough that the `ceil(T/cap) + E` tile bound is cheap.
_STATIC_MAX_ROWS = int(os.environ.get("SKYRL_ISOEXEC_MOE_STATIC_MAX_ROWS", "131072"))


# Flat tile addressing (SKYRL_ISOEXEC_MOE_FLAT_STAGE, default off). The staged-tile path addresses the
# padded buffer with a pair of index tensors (`xp[tile_idx, row_idx]`), which lands on ATen's generic
# `index_elementwise_kernel` and cannot vectorise the [h] row. Since `tile_idx * cap + row_idx` is a
# bijection onto the flattened first two axes, the same bytes are reachable as a 1-D
# `index_copy_`/`index_select` on the [n_tiles*cap, h] view, which vectorises. This is a change of
# address arithmetic only -- no floating-point operation is involved, so it is bitwise by construction.
_FLAT_STAGE = os.environ.get("SKYRL_ISOEXEC_MOE_FLAT_STAGE", "0") == "1"


def _flat_tile_offsets(tile_idx: torch.Tensor, row_idx: torch.Tensor, cap: int) -> torch.Tensor:
    """``tile_idx * cap + row_idx`` -- the row of the [n_tiles*cap, h] view each token owns."""
    return tile_idx * cap + row_idx


_TRAINER_EPILOGUE_COUNTS = {
    "served": 0,
    "rows": 0,
    "grad_fallback": 0,
    "math_fallback": 0,
    "reported": 0,
}

# The trainer gate is separate from, and stricter than, the shared engine FUSED_EPILOGUE flag.
_TRAINER_EPILOGUE_ENABLED = (
    os.environ.get("SKYRL_ISOEXEC_MOE_FUSED_EPILOGUE", "0") == "1"
    and os.environ.get("SKYRL_ISOEXEC_MOE_FUSED_EPILOGUE_TRAINER", "0") == "1"
)


#: Which of the fc2 fold's three owners actually ran, per call. ``_leaftree_fc2`` has a strictly
#: ordered precedence (in-GEMM tree > fused fold > eager buffer tree) whose safety argument is that
#: all three are bit-identical. Exactly one bucket advances per call.
_LEAFTREE_OWNER_COUNTS = {"ingemm": 0, "fused": 0, "buffer": 0, "single_leaf": 0}
_LEAFTREE_OWNER_REPORTED = 0


def leaftree_fold_stats() -> dict:
    """Per-owner execution census for the fc2 leaf fold, plus the total that served."""
    counts = dict(_LEAFTREE_OWNER_COUNTS)
    counts["served"] = sum(counts.values())
    return counts


def _report_leaftree_owner() -> None:
    """Emit the mutually-exclusive fc2 owner census at power-of-two calls."""
    global _LEAFTREE_OWNER_REPORTED
    counts = leaftree_fold_stats()
    served = counts["served"]
    if served < 1 or (served & (served - 1)) != 0 or served == _LEAFTREE_OWNER_REPORTED:
        return
    _LEAFTREE_OWNER_REPORTED = served
    print(
        "[ISOEXEC-MOE-LEAFTREE-OWNER] "
        f"pid={os.getpid()} served={served} ingemm={counts['ingemm']} fused={counts['fused']} "
        f"buffer={counts['buffer']} single_leaf={counts['single_leaf']}",
        flush=True,
    )


def _report_trainer_epilogue() -> None:
    calls = (
        _TRAINER_EPILOGUE_COUNTS["served"]
        + _TRAINER_EPILOGUE_COUNTS["grad_fallback"]
        + _TRAINER_EPILOGUE_COUNTS["math_fallback"]
    )
    if calls < 1 or (calls & (calls - 1)) != 0 or calls == _TRAINER_EPILOGUE_COUNTS["reported"]:
        return
    _TRAINER_EPILOGUE_COUNTS["reported"] = calls
    print(
        "[ISOEXEC-MOE-TRAINER-EPILOGUE] "
        f"pid={os.getpid()} served={_TRAINER_EPILOGUE_COUNTS['served']} "
        f"rows={_TRAINER_EPILOGUE_COUNTS['rows']} "
        f"grad_fallback={_TRAINER_EPILOGUE_COUNTS['grad_fallback']} "
        f"math_fallback={_TRAINER_EPILOGUE_COUNTS['math_fallback']}",
        flush=True,
    )


def _apply_trainer_epilogue(inter: torch.Tensor, probs: torch.Tensor, cfg) -> torch.Tensor:
    """Fuse only no-grad scoring/checkpoint forwards; keep the literal autograd chain otherwise."""
    # Trainer admission is gated independently of the shared engine FUSED_EPILOGUE flag.
    fused_requested = (
        os.environ.get("SKYRL_ISOEXEC_MOE_FUSED_EPILOGUE", "0") == "1"
        and os.environ.get("SKYRL_ISOEXEC_MOE_FUSED_EPILOGUE_TRAINER", "0") == "1"
    )
    if fused_requested:
        from .moe_epilogue_kernel import apply_glu_probs_epilogue, epilogue_supported

        if torch.is_grad_enabled():
            _TRAINER_EPILOGUE_COUNTS["grad_fallback"] += 1
            _report_trainer_epilogue()
        elif epilogue_supported(cfg) and inter.dtype in (torch.bfloat16, torch.float16):
            shape = inter.shape
            flat_inter = inter.reshape(-1, shape[-1])
            flat_probs = probs.reshape(-1)
            out = apply_glu_probs_epilogue(flat_inter, flat_probs).view(*shape[:-1], shape[-1] // 2)
            _TRAINER_EPILOGUE_COUNTS["served"] += 1
            _TRAINER_EPILOGUE_COUNTS["rows"] += int(flat_inter.shape[0])
            _report_trainer_epilogue()
            return out
        else:
            _TRAINER_EPILOGUE_COUNTS["math_fallback"] += 1
            _report_trainer_epilogue()

    # Verbatim Megatron local-MLP epilogue, including all observable dtype round trips.
    x_glu, x_linear = torch.chunk(inter, 2, dim=-1)
    if (val := cfg.activation_func_clamp_value) is not None:
        x_glu = x_glu.clamp(min=None, max=val)
        x_linear = x_linear.clamp(min=-val, max=val)
    out = cfg.activation_func(x_glu) * (x_linear + cfg.glu_linear_offset)
    original_dtype = out.dtype
    return (out * probs.unsqueeze(-1)).to(original_dtype)


# ETP-invariant fc2 (SKYRL_ISOEXEC_MOE_PIK_FC2). The batched fc2 bmm reduces the LOCAL moe_intermediate
# shard contiguously, so a trainer and an engine at different ETP compute a different fp expression and
# are not bitwise-equal. Instead, cut the FULL moe_intermediate into G fixed leaves
# (G = SKYRL_ISOEXEC_PIK_LEAVES) and reduce them with a balanced fp32 tree: a rank at ETP=C owns G/C
# leaves, and its local subtree composed with pik's cross-rank tree over the C subtrees yields the same
# G-leaf tree for every C.
def _moe_pik_fc2_on() -> bool:
    return os.environ.get("SKYRL_ISOEXEC_MOE_PIK_FC2") == "1"


def _fc2_n_leaves() -> int:
    """Leaves this rank owns = G // ETP. 1 at ETP=G (rank's shard IS one leaf); >1 when ETP<G."""
    if not _moe_pik_fc2_on():
        return 1
    g = int(os.environ.get("SKYRL_ISOEXEC_PIK_LEAVES", "8"))
    try:
        from megatron.core import parallel_state as mpu

        if not mpu.model_parallel_is_initialized():
            return 1
        try:
            etp = mpu.get_expert_tensor_parallel_world_size()
        except Exception:
            etp = mpu.get_tensor_model_parallel_world_size()
    except Exception:
        return 1
    etp = max(1, int(etp))
    return max(1, g // etp)


def _tree_sum(nodes):
    """Balanced binary tree sum (left = lower index), matching pik's leaf-fold order."""
    while len(nodes) > 1:
        nodes = [nodes[i] + nodes[i + 1] if i + 1 < len(nodes) else nodes[i] for i in range(0, len(nodes), 2)]
    return nodes[0]


def _w2_for_tiles(w2_slice, w2_stack, idx):
    """Materialize the per-tile fc2 weights, deferred so paths that never touch them do not pay the
    gather (the in-GEMM tree indexes the stack in-kernel). ``idx is None`` means tile id == expert id.
    """
    if w2_slice is not None:
        return w2_slice
    return w2_stack if idx is None else w2_stack[idx]


def _leaftree_fc2(inter, w2_slice, n_leaves, *, w2_stack=None, idx=None):
    """fc2 = bmm(inter, w2) reducing moe_intermediate, as a balanced fp32 leaf tree.

    ``n_leaves == 1`` returns the single leaf promoted to fp32. Otherwise moe_intermediate is split
    into ``n_leaves`` blocks, each bmm'd in bf16 by the batch-invariant kernel, promoted to fp32
    (lossless) and balanced-tree-summed. The fp32 result is a leaf or internal node that pik's
    cross-rank ``tree_all_reduce`` combines, so the trees compose bitwise across ETP.

    Exactly one of three owners performs the fold, in strict precedence, and all three are
    bit-identical:

        SKYRL_ISOEXEC_MOE_FC2_INGEMM on and self-check passed  -> in-GEMM tree (LEAFCOMBINE inert:
                                                                 no leaf buffers are left to read)
        else SKYRL_ISOEXEC_MOE_FUSED_LEAFCOMBINE on and passed -> leaf bmms + one-pass fused fold
        else                                                  -> leaf bmms + eager buffer tree

    Each step down is a fail-closed fallback (a failed self-check disables its provider permanently),
    never a mixed state. This is the single fold site: the trainer forward and the recompute paths all
    reach it with the same arguments, so recompute selects the same owner and reproduces it bit for bit.
    """
    if n_leaves <= 1:
        _LEAFTREE_OWNER_COUNTS["single_leaf"] += 1
        _report_leaftree_owner()
        return torch.ops.aten.bmm(inter.contiguous(), _w2_for_tiles(w2_slice, w2_stack, idx)).float()

    # owner 1: the in-GEMM tree, which supersedes the fold entirely
    from .moe_fc2_ingemm import (
        fc2_ingemm_ready,
        ingemm_leaftree_fc2,
        leaftree_shape_supported,
    )

    # The stack + map when the caller has them (so the gather never runs), else the slice it had.
    wg, ig = (w2_stack, idx) if w2_stack is not None else (w2_slice, None)
    if wg is not None and leaftree_shape_supported(inter, wg, n_leaves) and fc2_ingemm_ready(inter.device):
        _LEAFTREE_OWNER_COUNTS["ingemm"] += 1
        _report_leaftree_owner()
        return ingemm_leaftree_fc2(inter, wg, ig, n_leaves)

    # owners 2/3: the leaf GEMMs, then either the fused fold or the buffer tree
    w2_slice = _w2_for_tiles(w2_slice, w2_stack, idx)
    f = w2_slice.shape[1]
    lf = f // n_leaves
    ic = inter.contiguous()
    # Leaf partials straight from the GEMMs, NOT promoted here: the fused fold promotes in register
    # (exact, same values) and the buffer-tree fallback promotes below.
    leaves = [
        torch.ops.aten.bmm(
            ic[:, :, i * lf : (i + 1) * lf].contiguous(), w2_slice[:, i * lf : (i + 1) * lf, :].contiguous()
        )
        for i in range(n_leaves)
    ]
    from .moe_leafcombine_kernel import (
        fused_leaf_tree_combine,
        fused_leafcombine_ready,
        leaves_are_supported,
    )

    if leaves_are_supported(leaves) and fused_leafcombine_ready(leaves[0].device):
        _LEAFTREE_OWNER_COUNTS["fused"] += 1
        _report_leaftree_owner()
        return fused_leaf_tree_combine(leaves)
    _LEAFTREE_OWNER_COUNTS["buffer"] += 1
    _report_leaftree_owner()
    return _tree_sum([leaf.float() for leaf in leaves])


def _batched_experts_forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs):
    """Drop-in ``SequentialMLP.forward``: identical math per expert, one bmm pair for all experts."""
    x = permuted_local_hidden_states
    probs = permuted_probs  # [T]
    dev = x.device
    E = self.num_local_experts

    T = x.shape[0]
    if T == 0:
        return x.new_zeros(0, x.shape[-1]), None

    counts = tokens_per_expert.to(device=dev, dtype=torch.long)  # [E]

    # Fixed-capacity tiles, not pad-to-max: padding every expert to `counts.max()` would size the
    # staging buffer by routing skew rather than token count. Total staged rows are <= T + E*CAP.
    # CAP is a constant multiple of the bmm kernel's BLOCK_SIZE_M, so a token's row is always computed
    # by a bmm of the same shape, which is what keeps its result invariant to how many tokens share
    # its expert.
    cap, cap_override = _tile_cap(T, E)

    # Expert-major row mapping without a host round-trip. `repeat_interleave(x, counts)` has a
    # data-dependent output size and therefore syncs; `searchsorted` gives the same mapping with a
    # static [T] shape (T = x.shape[0], host-known): row i belongs to the expert whose cumulative
    # count first exceeds i. Keeping this sync-free is what makes the decode path graph-capturable.
    cu = torch.zeros(E + 1, device=dev, dtype=torch.long)
    cu[1:] = counts.cumsum(0)
    idx = torch.arange(T, device=dev)
    tok_expert = torch.searchsorted(cu[1:], idx, right=True).clamp_(max=E - 1)  # [T]
    off = idx - cu[tok_expert]  # within-expert

    # Bound on BOTH branches: the tail call passes it unconditionally, and the static-decode branch
    # deliberately does not produce it (it must stay sync-free). None means "no grouped path".
    host_tiles_per_expert = None

    if _STATIC_DECODE and T <= _STATIC_MAX_ROWS:
        # `ceil(T/cap) + E` is a static upper bound on `sum_e ceil(count_e / cap)` for ANY routing --
        # each expert wastes at most one partial tile. It depends only on host-known T and E, so the
        # shape never touches the routing, and it cannot overflow (a fixed P tiles per expert can).
        n_tiles = -(-T // cap) + E
        n_tiles_e = (counts + cap - 1) // cap  # [E]
        tile_cu = torch.zeros(E + 1, device=dev, dtype=torch.long)
        tile_cu[1:] = n_tiles_e.cumsum(0)
        tile_idx = tile_cu[tok_expert] + off // cap  # [T]
        row_idx = off % cap  # [T]
        # Which expert owns each tile. searchsorted, not repeat_interleave, whose output size is
        # data-dependent and syncs. Tiles past the used range clamp to the last expert and compute
        # garbage that `output_local` never gathers.
        tile_expert = torch.searchsorted(tile_cu[1:], torch.arange(n_tiles, device=dev), right=True).clamp_(
            max=E - 1
        )  # [n_tiles]
    else:
        # Prefill / trainer. Never graph-captured, so the exact tile count is worth the one sync:
        # the static bound would waste up to E partial tiles, which is real memory at ~1e6 rows.
        n_tiles_e = (counts + cap - 1) // cap  # [E]
        if cap_override:
            # Give every expert at least one tile, even an empty one: that is padding the bmm is
            # invariant to, and with one tile per expert `tile_expert` IS `arange(E)`, so the per-tile
            # weight gather becomes an identity copy and can be skipped entirely.
            n_tiles_e = n_tiles_e.clamp_(min=1)
        tile_cu = torch.zeros(E + 1, device=dev, dtype=torch.long)
        tile_cu[1:] = n_tiles_e.cumsum(0)
        n_tiles = int(tile_cu[-1])  # <- the sync, prefill only
        tile_idx = tile_cu[tok_expert] + off // cap  # [T]
        row_idx = off % cap  # [T]
        if cap_override and n_tiles == E:
            tile_expert = None  # tile id == expert id: no per-tile weight gather at all
        else:
            tile_expert = torch.repeat_interleave(torch.arange(E, device=dev), n_tiles_e)  # [n_tiles]
        # Tiles-per-expert on the host, free only on this branch (it already synced above). The
        # grouped cuBLASLt expert GEMM needs it; the static-decode branch must stay sync-free.

    if _FLAT_STAGE:
        # Same slots, same bytes, 1-D addressing. The buffer is built flat and reshaped after the
        # copy so autograd never sees an in-place write through a view.
        flat = _flat_tile_offsets(tile_idx, row_idx, cap)
        xp_flat = x.new_zeros(n_tiles * cap, x.shape[-1])
        xp_flat.index_copy_(0, flat, x)
        xp = xp_flat.view(n_tiles, cap, x.shape[-1])
        pp_flat = probs.new_zeros(n_tiles * cap)
        pp_flat.index_copy_(0, flat, probs)
        pp = pp_flat.view(n_tiles, cap)
    else:
        xp = x.new_zeros(n_tiles, cap, x.shape[-1])
        xp[tile_idx, row_idx] = x
        pp = probs.new_zeros(n_tiles, cap)
        pp[tile_idx, row_idx] = probs

    return _batched_experts_gemm(
        self,
        x,
        xp,
        pp,
        tile_idx,
        row_idx,
        tile_expert,
        n_tiles,
        cap,
        tiles_per_expert=host_tiles_per_expert,
    )


def _stack_expert_weights(self, role: str) -> torch.Tensor:
    """``torch.stack`` of the E expert ``<role>.weight``s -- memoized when the cache flag is on.

    All three arms return the same values, autograd-connected to the same E parameters. Gate order is
    load-bearing: the fused alias is consulted FIRST and independently of
    ``SKYRL_ISOEXEC_MOE_WEIGHT_CACHE``, because the two select different things -- the cache memoizes a
    copy (costs memory, so memory-pressured recipes turn it off) while the fused path returns an alias
    onto the parameters' own storage and is memory-neutral. Staleness is handled by
    ``fused_expert_weights``'s own fail-closed check, which returns ``None`` onto the code below.
    """
    from .moe_fused_weights import fused_expert_weights

    fused = fused_expert_weights(self, role)
    if fused is not None:
        return fused
    if os.environ.get("SKYRL_ISOEXEC_MOE_WEIGHT_CACHE", "1") == "1":
        from .moe_weight_cache import stacked_expert_weights

        return stacked_expert_weights(self, role)
    return torch.stack([getattr(e, role).weight for e in self.local_experts])


def _batched_experts_gemm(self, x, xp, pp, tile_idx, row_idx, tile_expert, n_tiles, cap, *, tiles_per_expert=None):
    """The bmm pair + epilogue over the staged tiles.

    Split out so the fused-staging path can reach it with tiles a kernel built; the final
    ``out[tile_idx, row_idx]`` gather reuses the same grid the tiles came from.
    """
    cfg = self.config

    # Stacked weights in bmm layout ([E, h, 2f] / [E, f, h]), in-graph for autograd.
    #
    # Stack the CONTIGUOUS parameters and transpose the result rather than stacking E transposed views:
    # same values, and the bmm is bitwise-identical either way (the reduction runs over k, which
    # strides do not reorder), but a stack of non-contiguous inputs cannot take torch.cat's batched
    # fast path and degenerates into one transposing copy kernel per expert.
    w1_lin = _stack_expert_weights(self, "linear_fc1")  # [E,2f,h] -- (N, K) per expert, as built
    w1 = w1_lin.transpose(1, 2)  # [E,h,2f]
    w2 = _stack_expert_weights(self, "linear_fc2").transpose(1, 2)  # [E,f,h]

    grouped = None
    # Tile-chunked bmm. vLLM's batch-invariant Triton bmm (which IsoExec forces on in place of cuBLAS)
    # computes its offsets in int32 and reads out of bounds once an operand passes 2^31 elements --
    # an illegal memory access, not an OOM, and only on the batch-invariant path. At [n_tiles, cap, h]
    # that ceiling is reached at roughly 1e6 routed rows, which the trainer forward can hit.
    #
    # Splitting along the TILE axis is bitwise-safe by the property the whole tiled design rests on:
    # the batch axis never enters a reduction, so each (batch, row-tile) program is independent of how
    # many other tiles ride along. It also cuts peak memory, since the weight gather materializes one
    # chunk at a time instead of the full [n_tiles, h, 2f].
    per_tile = max(cap * x.shape[-1], w1.shape[1] * w1.shape[2], cap * w1.shape[2])
    tiles_per_chunk = max(1, _BMM_MAX_ELEMS // max(1, per_tile))

    def _wslice(w, s, e):
        return w[s:e] if tile_expert is None else w[tile_expert[s:e]]

    # Gather-free indexed bmm (SKYRL_ISOEXEC_MOE_INDEXED_BMM). The `w[tile_expert]` gather materializes
    # per-tile expert weights for the bmm to read, which both costs time and saves a full gathered copy
    # as an activation. The indexed kernel is vLLM's batch-invariant bmm_kernel with the B base pointer
    # computed as `w + tile_expert[pid]*stride_e` in-kernel -- same values, same fixed K-order tl.dot,
    # so bitwise by construction, and asserted by a fail-closed first-use self-check that falls through
    # to the gather permanently on failure. Backward saves the stack plus indices, never the gathered
    # copy. See moe_indexed_bmm.py.
    use_indexed = False
    if tile_expert is not None:
        from .moe_indexed_bmm import indexed_bmm, indexed_bmm_ready

        use_indexed = indexed_bmm_ready(xp.device)

    pik_fc2 = _moe_pik_fc2_on()  # ETP-invariant leaf-tree fc2 (fp32 output); OFF -> original bf16 bmm
    n_leaves = _fc2_n_leaves() if pik_fc2 else 1
    outs = []
    # `grouped` is [(tile_start, tile_stop, splits, expert_start), ...]; the non-grouped arm builds the
    # same 4-tuple shape with splits/expert_start None.
    if grouped is not None:
        chunk_ranges = grouped
    else:
        chunk_ranges = [(s, min(s + tiles_per_chunk, n_tiles), None, None) for s in range(0, n_tiles, tiles_per_chunk)]
    for s, e, splits, e0 in chunk_ranges:
        # torch.ops.aten.bmm, NOT torch.bmm: enable_batch_invariant_mode rebinds the python attribute
        # torch.bmm to a raw Triton function that never records autograd. Dispatching the operator
        # routes Autograd -> the overridden batch-invariant CUDA kernel, so grads and determinism hold.
        inter = None
        if inter is None:
            if use_indexed:
                inter = indexed_bmm(xp[s:e], w1, tile_expert[s:e])  # [chunk, cap, 2f], no gather
            else:
                inter = torch.ops.aten.bmm(xp[s:e], _wslice(w1, s, e))  # [chunk, cap, 2f]

        # Trainer scoring/checkpoint no-grad calls may fuse this chain into one pass; the off arm is
        # kept literally inline so it stays byte-for-byte the original expression.
        if _TRAINER_EPILOGUE_ENABLED:
            inter = _apply_trainer_epilogue(inter, pp[s:e], cfg)
        else:
            x_glu, x_linear = torch.chunk(inter, 2, dim=-1)
            if (val := cfg.activation_func_clamp_value) is not None:
                x_glu = x_glu.clamp(min=None, max=val)
                x_linear = x_linear.clamp(min=-val, max=val)
            inter = cfg.activation_func(x_glu) * (x_linear + cfg.glu_linear_offset)
            original_dtype = inter.dtype
            inter = inter * pp[s:e].unsqueeze(-1)
            inter = inter.to(original_dtype)

        # fc2: the indexed kernel covers the plain bmm and the pik single-leaf case (same bmm,
        # promoted to fp32 after, which is lossless). The multi-leaf tree keeps the gather.
        if pik_fc2:
            if use_indexed and n_leaves <= 1:
                outs.append(indexed_bmm(inter.contiguous(), w2, tile_expert[s:e]).float())
            else:
                # Hand over the stack + map, not a gathered slice: _leaftree_fc2 materializes
                # w2[tile_expert] only on a buffer path. tile_expert None means tile id == expert id.
                w2s, w2i = (w2[s:e], None) if tile_expert is None else (w2, tile_expert[s:e])
                outs.append(_leaftree_fc2(inter, None, n_leaves, w2_stack=w2s, idx=w2i))
        elif use_indexed:
            outs.append(indexed_bmm(inter.contiguous(), w2, tile_expert[s:e]))
        else:
            outs.append(torch.ops.aten.bmm(inter.contiguous(), _wslice(w2, s, e)))
        # [chunk, cap, h] (fp32 if PIK_FC2)

    out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)  # [n_tiles, cap, h]
    if _FLAT_STAGE:
        # `reshape`, not `view`: the fc2 output is contiguous on every path today, but a future path
        # handing back a stride-aliased buffer must copy rather than silently mis-address.
        output_local = out.reshape(n_tiles * cap, out.shape[-1]).index_select(
            0, _flat_tile_offsets(tile_idx, row_idx, cap)
        )  # [T, h], expert order
    else:
        output_local = out[tile_idx, row_idx]  # [T, h], expert order
    # explicit_expert_comm: the dispatcher owns any reduction; the sequential loop did none here.
    return output_local, None


def _dispatch_postprocess_fixed_shape(self, hidden_states, probs):
    """Drop-in ``MoEAllGatherTokenDispatcher.dispatch_postprocess`` without the ``masked_select``.

    Megatron builds the permuted probs with a ``masked_select`` whose output size is data-dependent,
    forcing a device sync and blocking CUDA-graph capture. ``permute`` already returns the same
    permuted probs computed with a fixed-shape gather; the dispatcher simply discards that value.
    Bitwise-identical -- both select the same elements of ``probs.T`` in the same expert-major order,
    with no arithmetic involved.
    """
    from megatron.core.transformer.moe.moe_utils import permute

    self.hidden_shape_before_permute = hidden_states.shape
    lo, hi = self.local_expert_indices[0], self.local_expert_indices[-1] + 1
    self.local_map = self.routing_map[:, lo:hi].contiguous()
    self.local_probs = probs[:, lo:hi].contiguous()

    # `num_out_tokens` is static, not data: under top-k routing the routed-row count is
    # `num_tokens * topk`. Megatron reads it back with `.item()`, a needless D2H sync.
    topk = self.config.moe_router_topk
    num_out_tokens = self.local_map.shape[0] * topk

    # `tokens_per_expert` stays on the DEVICE where possible: megatron `.cpu()`s it once per MoE
    # layer and the expert forward pushes it straight back, and that round-trip is what forces
    # enforce_eager. The static-decode expert path derives its tile grid from host-known
    # `num_out_tokens` instead.
    #
    # SKYRL_ISOEXEC_MOE_ROUTER_O2 / SKYRL_ISOEXEC_MOE_PERMUTE_SORT both select `fused_permute_index`, a
    # counting sort replacing the column sum, `permute`'s two transposing copies, the [E*T] stable
    # argsort, the `remainder` and the prob gather. Pure integer indexing plus one gather, so exact by
    # construction. Either flag is sufficient and both predicates carry the same EP=1 guard.
    # ENGINE-ONLY: `dispatch_o2_active` requires an instance mark only the engine model build sets,
    # because with VLLM_ENABLE_V1_MULTIPROCESSING=0 this method is shared with the trainer, which needs
    # the autograd graph megatron's ops build.
    from .moe_dense_scatter_kernel import permute_sort_active

    # SKYRL_ISOEXEC_MOE_PERMUTE_INDEX_1K folds `_counts_kernel` into `_permute_index_kernel` by having
    # every program recompute the [E] count vector in registers: same outputs, one launch instead of
    # two. Admitted only below a size bound on the E-fold map re-read, else the two-kernel form.
    from .moe_router_chain_kernel import (
        fused_permute_index_1k,
        permute_index_1k_enabled,
    )
    from .moe_router_o2_kernel import (
        dispatch_o2_active,
        fused_permute_index,
        permute_can_handle,
    )

    if (dispatch_o2_active(self) or permute_sort_active(self)) and permute_can_handle(self.local_map, topk):
        _index_build = fused_permute_index_1k if permute_index_1k_enabled() else fused_permute_index
        sorted_indices, permuted_probs, tokens_per_expert = _index_build(self.local_map, self.local_probs, topk)
        if not _STATIC_DECODE or num_out_tokens > _STATIC_MAX_ROWS:
            tokens_per_expert = tokens_per_expert.cpu()
        self.reversed_local_input_permutation_mapping = sorted_indices
        self.local_probs = permuted_probs
        self.routing_map = None
        return hidden_states.index_select(0, sorted_indices), tokens_per_expert, self.local_probs

    tokens_per_expert = self.local_map.sum(dim=0).long()
    if not _STATIC_DECODE or num_out_tokens > _STATIC_MAX_ROWS:
        tokens_per_expert = tokens_per_expert.cpu()  # prefill / trainer: the tile path needs it

    permuted_local_hidden_states, permuted_probs, self.reversed_local_input_permutation_mapping, _, _ = permute(
        hidden_states,
        self.local_map,
        probs=self.local_probs,
        num_out_tokens=num_out_tokens,
        fused=self.config.moe_permute_fusion,
    )
    self.local_probs = permuted_probs
    self.routing_map = None
    return permuted_local_hidden_states, tokens_per_expert, self.local_probs


def install_fixed_shape_dispatch() -> bool:
    """Rebind the allgather dispatcher's ``dispatch_postprocess``. Idempotent.

    Installed on BOTH runtimes from ``prepare_isoexec_moe``. ``SKYRL_ISOEXEC_MOE_FIXED_DISPATCH=0``
    restores megatron's bitwise-equal ``masked_select`` version.
    """
    if os.environ.get("SKYRL_ISOEXEC_MOE_FIXED_DISPATCH", "1") != "1":
        return False
    from megatron.core.transformer.moe.token_dispatcher import (
        MoEAllGatherTokenDispatcher,
    )

    if getattr(MoEAllGatherTokenDispatcher, "_isoexec_fixed_dispatch", False):
        return True
    MoEAllGatherTokenDispatcher.dispatch_postprocess = _dispatch_postprocess_fixed_shape
    MoEAllGatherTokenDispatcher._isoexec_fixed_dispatch = True
    print(
        "[ISOEXEC-MOE] dispatch_postprocess -> fixed-shape permuted probs (was a masked_select: "
        "~47 ms/step at 35B, data-dependent shape, uncapturable)",
        flush=True,
    )
    return True


def install_batched_sequential_mlp() -> bool:
    """Rebind ``SequentialMLP.forward`` to the batched implementation. Idempotent.

    Installed from ``prepare_isoexec_moe`` on BOTH the trainer and the engine so the two runtimes
    move together.
    """
    from megatron.core.transformer.moe.experts import SequentialMLP

    if getattr(SequentialMLP, "_isoexec_batched", False):
        return True
    SequentialMLP.forward = _batched_experts_forward
    SequentialMLP._isoexec_batched = True
    print(
        "[ISOEXEC-MOE] SequentialMLP.forward -> padded batched expert GEMMs "
        "(one bmm pair per layer instead of a per-expert python loop)",
        flush=True,
    )
    return True


@torch.no_grad()
def verify_batched_experts_invariance(*, E: int = 16, h: int = 256, f: int = 64, dtype=torch.bfloat16) -> None:
    """Assert the three properties the batched path rests on. Raises on violation. GPU-only.

    (1) determinism: same inputs twice -> bitwise-equal.
    (2) routing invariance: a token's output through expert e is bitwise-identical no matter how
        many tokens share e or what other experts received.
    (3) padding invariance: changing M_pad (by inflating another expert's load) does not change
        existing rows.
    """
    dev = "cuda"
    torch.manual_seed(0)
    w1 = torch.randn(E, 2 * f, h, device=dev, dtype=dtype) * 0.05
    w2 = torch.randn(E, h, f, device=dev, dtype=dtype) * 0.05
    tok = torch.randn(h, device=dev, dtype=dtype)

    def run(counts, probe_expert, probe_slot, extra_seed):
        torch.manual_seed(extra_seed)
        T = int(sum(counts))
        cu = [0]
        for c in counts:
            cu.append(cu[-1] + c)
        x = torch.randn(T, h, device=dev, dtype=dtype)
        p = torch.rand(T, device=dev, dtype=torch.float32)
        pos = cu[probe_expert] + probe_slot
        x[pos] = tok
        p[pos] = 0.5
        M = max(counts)
        xp = x.new_zeros(E, M, h)
        pp = p.new_zeros(E, M)
        ie = torch.repeat_interleave(torch.arange(E, device=dev), torch.tensor(counts, device=dev))
        im = torch.arange(T, device=dev) - torch.tensor(cu[:-1], device=dev).repeat_interleave(
            torch.tensor(counts, device=dev)
        )
        xp[ie, im] = x
        pp[ie, im] = p
        inter = torch.bmm(xp, w1.transpose(1, 2))
        a, b = torch.chunk(inter, 2, dim=-1)
        inter = (torch.nn.functional.silu(a) * b * pp.unsqueeze(-1)).to(dtype)
        out = torch.bmm(inter, w2.transpose(1, 2))
        return out[probe_expert, probe_slot].clone()

    base = run([4] * E, 2, 1, seed0 := 7)
    again = run([4] * E, 2, 1, seed0)
    assert torch.equal(base, again), "[isoexec] batched experts: NOT deterministic"

    skew = [1] * E
    skew[2] = 4
    skew[5] = 300  # inflate another expert -> changes M_pad AND other-batch content
    moved = run(skew, 2, 1, seed0)
    # NOTE: run() places random tokens around the probe; with different counts the surrounding
    # content differs by construction, which is exactly the point -- the probe row must not care.
    assert torch.equal(base, moved), "[isoexec] batched experts: row depends on routing/padding"
    print(
        "[ISOEXEC-MOE] batched-experts invariance verified (deterministic, routing- and " "padding-invariant)",
        flush=True,
    )
