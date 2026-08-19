"""Fused-MoE expert GEMMs: vLLM's Triton MoE kernel in the ROLLOUT ENGINE ONLY.

The fused kernel reads an expert-block map and skips the padding the trainer's ``bmm`` pair pays for, but
that padding-free forward has no cheap bmm-shaped backward, so it is installed only in the engine and the
trainer keeps ``moe_batched_experts``' forward AND backward. That makes the two forwards different code,
which means they must be BITWISE-EQUAL: ``_fused_forward`` mirrors ``_batched_experts_forward`` op for op,
including applying the router weight to the intermediate BEFORE fc2 with the same ``.to(dtype)``
round-trip. The IsoExec gate compares exactly the fused rollout against the bmm scoring forward, so a
broken equality shows up on step 1.

The kernel must also be batch-invariant. The stock MoE kernel is not -- ``try_get_optimal_moe_config``
picks BLOCK_SIZE_M/N/K from the token count -- but under ``VLLM_BATCH_INVARIANT=1`` vLLM skips that lookup
and pins ``BLOCK_M 64 / BLOCK_N 64 / BLOCK_K 32, SPLIT_K 1``, so each (block, row) program reduces its own
row over K in a fixed order and the M axis never enters a reduction.

The hook is the INNER GEMM (``invoke_fused_moe_triton_kernel``) rather than vLLM's whole-layer
``fused_experts``, so megatron's dispatcher, permute and combine stay in place and ``A`` is the
already-expert-grouped permuted buffer with ``top_k = 1``. The epilogue collapses to ``silu(gate) * up``
only when ``glu_linear_offset == 0`` and there is no clamp value; that is asserted rather than assumed, and
a model setting either falls back.

SHAPE-STATIC AND SYNC-FREE outranks the speedup: a D2H readback to size the launch grid would trade the
GEMM win for the much larger CUDA-graph win on decode. The block map is built entirely on device and the
grid is sized from the host-known bound ``ceil(T / BLOCK_M) + E``; blocks past the real count early-return
against the device scalar ``num_tokens_post_padded``. The engine-only install lives in
``moe_batch_invariant.prepare_isoexec_moe``.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

# The batch-invariant config vLLM pins under VLLM_BATCH_INVARIANT=1 (fused_moe.py). We must build the
# block map at the SAME BLOCK_SIZE_M the kernel will use, so read it from vLLM rather than duplicating
# the constant -- if vLLM ever changes it, the map must follow or the kernel reads the wrong rows.
_BLOCK_M_FALLBACK = 64

_installed = False
_orig_forward = None

# Bumped on every fused expert forward. The install banner is not a usable check -- Ray does not capture
# these prints reliably -- so this counter is how the split is verified: after a run the ENGINE ranks must
# report CALL_COUNT > 0 and the TRAINER ranks CALL_COUNT == 0. A nonzero count on a trainer rank means
# fused leaked into training.
CALL_COUNT = 0


# Log the fused-weights alias state exactly ONCE, at the first forward that runs under the flag. See
# _log_fused_weights_state_once.
_LOGGED_FW_STATE = False


def _log_fused_weights_state_once(alias_active: bool) -> None:
    """Surface, once, whether the fused-weight alias is LIVE at forward time -- a capture-time check.

    CUDA-graph capture bakes whatever address the fc1/fc2 GEMM reads. If the forward being captured has
    the alias ACTIVE, the graph bakes the persistent ``[E, ...]`` buffer, and the sync-boundary eager
    refresh (moe_fused_weights.refresh_all_fused) keeps every subsequent replay reading fresh weights.
    If instead the alias is INACTIVE (torch.stack fallback -- a broken alias, or the module could not be
    fused), the graph bakes a per-forward temporary that the eager refresh CANNOT reach, so a
    capture-with-broken-alias is a silent stale-weights bug. Logging the state once makes that visible
    in the engine logs (Ray drops actor stdout otherwise, so this is a ``print``, not a logger call).
    """
    global _LOGGED_FW_STATE
    if _LOGGED_FW_STATE:
        return
    _LOGGED_FW_STATE = True
    if alias_active:
        msg = "ACTIVE (fc1/fc2 GEMMs read the persistent [E,...] buffer; a graph captured here stays refreshable)"
    else:
        msg = (
            "INACTIVE (torch.stack fallback; a CUDA graph captured NOW bakes a per-forward temp that the "
            "sync-boundary eager refresh CANNOT keep fresh)"
        )
    print(f"[ISOEXEC-MOE-FW] fused expert-weight alias at first forward: {msg}", flush=True)


_EPILOGUE_LOGGED = False


def _fused_epilogue_on() -> bool:
    """SKYRL_ISOEXEC_MOE_FUSED_EPILOGUE: fold the glu+probs epilogue into the fc1 GEMM (default OFF).

    Read at CALL time, not install time, so a live A/B can flip it between forwards. The kernel is
    bitwise-equal to the torch chain it replaces (moe_epilogue_kernel), which is why it may be
    ENGINE-ONLY: a bitwise-equal replacement needs no partner on the other side.

    Logs its value ONCE per process on first use, so a live A/B is falsifiable from the child's log --
    an env var that never reached the actor otherwise compares a baseline against itself.
    """
    on = os.environ.get("SKYRL_ISOEXEC_MOE_FUSED_EPILOGUE", "0") == "1"
    global _EPILOGUE_LOGGED
    if not _EPILOGUE_LOGGED:
        _EPILOGUE_LOGGED = True
        print(
            f"[isoexec-moe] fused fc1+glu+probs epilogue (O9): SKYRL_ISOEXEC_MOE_FUSED_EPILOGUE={int(on)}", flush=True
        )
    return on


def fused_blockmap_enabled() -> bool:
    """SKYRL_ISOEXEC_MOE_FUSED_BLOCKMAP: build the expert block map in one Triton launch (default OFF).

    Re-exported from moe_blockmap_kernel so `_block_map` can gate without importing triton when the
    flag is off (the import cost is real on the dense path, which never reaches this module's kernels).
    """
    return os.environ.get("SKYRL_ISOEXEC_MOE_FUSED_BLOCKMAP", "0") == "1"


def _bi_config(E: int, N: int, K: int) -> dict:
    """The kernel config. MUST be the batch-invariant one, and must not depend on the token count."""
    from vllm import envs
    from vllm.model_executor.layers.fused_moe.fused_moe import get_default_config

    if not envs.VLLM_BATCH_INVARIANT:
        raise RuntimeError(
            "[isoexec-moe] fused MoE requires VLLM_BATCH_INVARIANT=1. Without it vLLM picks "
            "BLOCK_SIZE_M/N/K from the TOKEN COUNT, so the K-tiling -- and the fp32 accumulation "
            "order -- changes with the batch. That is not a rounding difference; it is exactly the "
            "batch-variance IsoExec exists to eliminate."
        )
    cfg = get_default_config(M=1, E=E, N=N, K=K, topk=1, dtype=None)
    if cfg.get("SPLIT_K", 1) != 1:
        raise RuntimeError(
            f"[isoexec-moe] fused MoE needs SPLIT_K=1 (a split-K reduction is order-dependent), got {cfg}"
        )

    # BLOCK_M OVERRIDE (SKYRL_ISOEXEC_MOE_BLOCK_M) -- the decode lever, and the one tile that is safe to
    # move. vLLM's batch-invariant default is 64, which at many local experts makes every expert holding
    # even ONE row cost a full 64-row block, so decode computes padding rather than tokens.
    #
    # WHY M IS THE SAFE ONE. BLOCK_SIZE_K sets the fp32 accumulation ORDER and BLOCK_SIZE_N is only an
    # output-column split; M is neither -- each output element runs its own K-loop, so moving M relabels
    # which program owns a row and nothing else. The tuning harness still VERIFIES that with torch.equal
    # against the BLOCK_M=64 result, because Triton picks its warp-level tile split from the whole config
    # and a small M could in principle push it to split along K.
    #
    # A FIXED PER-PROCESS CONSTANT, read once from the env and applied to every token count in this
    # process. It must NEVER be selected from the runtime token count -- that is precisely the
    # batch-variance IsoExec exists to eliminate. Trainer and engine may legitimately differ, since the
    # fused path is proven bitwise against the trainer's bmm path at every M this gate accepts.
    override = os.environ.get("SKYRL_ISOEXEC_MOE_BLOCK_M")
    if override:
        m = int(override)
        if m not in (16, 32, 64, 128) or m & (m - 1):
            raise RuntimeError(f"[isoexec-moe] SKYRL_ISOEXEC_MOE_BLOCK_M must be 16/32/64/128, got {m}")
        cfg = dict(cfg, BLOCK_SIZE_M=m)
    return cfg


@torch.no_grad()
def _block_map(counts: torch.Tensor, T: int, E: int, block_m: int):
    """Expert-block map for tokens ALREADY grouped by expert. All on-device; no host readback.

    ``counts[e]`` rows of expert ``e`` sit contiguously at ``cu[e] ..``. The kernel wants:

      * ``sorted_token_ids [max_blocks * block_m]`` -- for each slot of each block, the row of ``A``
        it holds. Padding slots get ``T``, which is >= ``num_valid_tokens``, so the kernel masks them.
      * ``expert_ids [max_blocks]`` -- which expert each block belongs to.
      * ``num_tokens_post_padded`` -- a device scalar; the kernel early-returns any program above it,
        which is what lets us launch a STATICALLY sized grid.

    ``max_blocks = ceil(T / block_m) + E`` is a host-known BOUND: each expert wastes at most one
    partial block. Nothing here reads a count back to the host, so the layer stays capturable.

    Under SKYRL_ISOEXEC_MOE_FUSED_BLOCKMAP this whole function collapses into ONE Triton launch
    (moe_blockmap_kernel). The dozen torch ops below are individually trivial but their launch latency is
    flat in the row count, which dominates the expert forward at decode. The kernel computes no floating
    point; its output is the same integer tensors, asserted elementwise-identical by the tuning gate.
    """
    if fused_blockmap_enabled():
        from .moe_blockmap_kernel import fused_block_map

        return fused_block_map(counts, T, E, block_m)

    dev = counts.device
    max_blocks = (T + block_m - 1) // block_m + E

    cu = torch.cumsum(counts, 0) - counts  # exclusive start row of each expert
    nb = (counts + block_m - 1) // block_m  # blocks per expert
    bcu = torch.cumsum(nb, 0) - nb  # first block index of each expert

    blk = torch.arange(max_blocks, device=dev)
    # which expert owns block b: the last expert whose first block is <= b. searchsorted, not
    # repeat_interleave -- the latter's output size is data-dependent and would sync.
    expert_ids = torch.searchsorted(bcu, blk, right=True) - 1
    expert_ids = expert_ids.clamp_(0, E - 1)

    j = blk - bcu[expert_ids]  # this block's index WITHIN its expert
    rows = cu[expert_ids].unsqueeze(1) + j.unsqueeze(1) * block_m + torch.arange(block_m, device=dev)
    valid = rows < (cu[expert_ids] + counts[expert_ids]).unsqueeze(1)
    sorted_token_ids = torch.where(valid, rows, torch.full_like(rows, T)).reshape(-1)

    num_tokens_post_padded = (nb.sum() * block_m).reshape(1)
    return (
        sorted_token_ids.to(torch.int32),
        expert_ids.to(torch.int32),
        num_tokens_post_padded.to(torch.int32),
        max_blocks,
    )


# Leaftree wire-stage: per-geometry admission for storing the fc2 leaf value (bf16) straight into the owner
# combine's symmetric wire buffer. The claim is an identity -- truncf(extf(truncf(acc))) == truncf(acc),
# the RNE round-trip through the wider type -- but no geometry runs a new store path without re-proving it
# on ITS OWN live operands first: reference fp32 store + RNE cast vs the wire store, compared as int16 bit
# patterns, host-synced, eagerly, never under capture. A verdict is remembered per geometry; False is a
# loud permanent fallback to the fp32 path (bit-identical, it just pays the stage copy). RANK-LOCAL on
# purpose: the choice alters no rendezvous structure, so a disagreeing rank cannot hang the group.
_WIRE_ADMIT: dict = {}
_WIRE_ADMIT_COUNTS = {"served": 0, "admitted": 0, "rejected": 0, "capture_skips": 0, "errors": 0}


def leaftree_wire_counts() -> dict:
    """For the A/B arm's own log. ``served`` counts wire-path dispatches (capture/warmup entries
    under CUDA graphs -- replays are a trace question, same as fused_owner_counts)."""
    return {**_WIRE_ADMIT_COUNTS, "shapes": {str(k): v for k, v in _WIRE_ADMIT.items()}}


def _leaftree_wire_dispatch(launch, wire_c, x, P: int, H: int, E: int, K: int):
    """Store the fc2 leaf partial straight to the wire, iff this geometry is admitted.

    Returns the bf16 [P,H] wire view (already holding proven bytes), or None -- in which case the
    caller runs the fp32 path exactly as if the sink had never granted."""
    key = (P, H, E, K)
    verdict = _WIRE_ADMIT.get(key)
    if verdict is False:
        return None
    from ..collectives.pik_bootstrap import ensure_pik

    ensure_pik()
    from pik.allreduce import _capturing  # type: ignore

    if verdict is None:
        if _capturing():
            # A bit compare needs a host sync; a geometry first seen under capture keeps the fp32
            # path and gets another chance eagerly (vLLM's warmup pass runs every size eagerly).
            _WIRE_ADMIT_COUNTS["capture_skips"] += 1
            return None
        ref = x.new_empty(P, 1, H, dtype=torch.float32)
        try:
            launch(ref, False)
            launch(wire_c.view(P, 1, H), True)
            same = torch.equal(
                ref.view(P, H).to(torch.bfloat16).view(torch.int16),
                wire_c.view(torch.int16),
            )
        except Exception as e:  # noqa: BLE001
            _WIRE_ADMIT_COUNTS["errors"] += 1
            _WIRE_ADMIT[key] = False
            print(
                f"[ISOEXEC-MOE] WARNING: leaftree wire-store admission RAISED at {key}: {e!r}. "
                "This geometry permanently falls back to the fp32 store + stage copy "
                "(bit-identical; the wire-stage saving is not taken here).",
                flush=True,
            )
            return None
        _WIRE_ADMIT[key] = same
        if not same:
            _WIRE_ADMIT_COUNTS["rejected"] += 1
            diff = int((ref.view(P, H).to(torch.bfloat16).view(torch.int16) != wire_c.view(torch.int16)).sum().item())
            print(
                f"[ISOEXEC-MOE] WARNING: leaftree wire-store REFUSED at {key}: BIT PATTERNS DIFFER "
                f"{diff}/{P * H} elements against the reference fp32 store + RNE cast. Permanent "
                "per-geometry fallback to the fp32 path; the result is unaffected.",
                flush=True,
            )
            return None
        _WIRE_ADMIT_COUNTS["admitted"] += 1
        if _WIRE_ADMIT_COUNTS["admitted"] == 1:
            print(
                f"[ISOEXEC-MOE] leaftree WIRE-STORE ADMITTED (first geometry {key}): the fc2 "
                "leaf-tree now stores the wire bytes straight into the owner combine's symmetric "
                "staging buffer -- bit patterns matched the reference fp32 store + RNE cast on live "
                "operands. A decode trace should lose the [P,H] cast-copy before the owner "
                "combine's lead barrier. Read moe_fused_experts.leaftree_wire_counts().",
                flush=True,
            )
        _WIRE_ADMIT_COUNTS["served"] += 1
        return wire_c  # admission's own wire launch already staged these proven bytes
    _WIRE_ADMIT_COUNTS["served"] += 1
    launch(wire_c.view(P, 1, H), True)
    return wire_c


@torch.no_grad()
def _fused_forward(x, w1, w2, probs, counts, E: int, *, pik_fc2: bool | None = None, n_leaves: int | None = None):
    """The rollout forward: two fused Triton MoE GEMMs over the expert-grouped rows.

    ``x [T, H]`` expert-grouped, ``w1 [E, 2I, H]``, ``w2 [E, H, I]``, ``probs [T]``.

    BITWISE-EQUAL to ``moe_batched_experts._batched_experts_forward`` for the same inputs -- that equality
    is the whole license for the rollout-only split -- so the epilogue mirrors the bmm path OP FOR OP:
    ``silu(gate) * up``, then the router probs applied to the intermediate BEFORE fc2 with a
    ``.to(dtype)`` round-trip, NOT after. Applying them after is mathematically identical (the prob is a
    per-row scalar and factors out of the fc2 contraction) but bitwise different, because the bf16
    rounding lands at a different point.

    ETP-INVARIANT fc2 (``pik_fc2``, from SKYRL_ISOEXEC_MOE_PIK_FC2). A single fc2 call reduces this rank's
    whole moe_intermediate shard contiguously inside the kernel, which is a different fp expression at
    every ETP. The leaf tree does not care which GEMM computes a leaf: cut fc2 into ``n_leaves``
    K-slices, run the SAME fused kernel on each (the block map is K-independent and the batch-invariant
    config is pinned), round each leaf where the single call rounds, promote to fp32 (lossless) and fold
    with the balanced tree, mirroring ``moe_batched_experts._leaftree_fc2`` op for op. The K-slices are
    strided VIEWS, so there are no copies, no syncs and no shape changes, and CUDA-graph capture is
    unaffected. Returns fp32 ``[T, H]``: this rank's leaf-subtree partial, which the dispatcher's pik
    tree all-reduce combines across ranks.
    """
    import triton.language as tl
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        invoke_fused_moe_triton_kernel,
    )

    if pik_fc2 is None:
        from .moe_batched_experts import _moe_pik_fc2_on

        pik_fc2 = _moe_pik_fc2_on()
    if n_leaves is None:
        from .moe_batched_experts import _fc2_n_leaves

        n_leaves = _fc2_n_leaves() if pik_fc2 else 1

    # One-time visibility of the operating point: the in-kernel leaf-tree tax appears only at
    # n_leaves >= 8, and the shipped engine (ETP=TP) runs n_leaves=1, which is parity. If this ever logs
    # n_leaves >= 2 on the engine, the tax is back on the decode path -- SKYRL_ISOEXEC_MOE_BLOCK_M=16 is
    # bitwise-equal and collapses it.
    if not getattr(_fused_forward, "_leaves_logged", False):
        _fused_forward._leaves_logged = True
        print(
            f"[ISOEXEC-MOE-LEAFTREE] engine fused fc2 n_leaves={n_leaves} "
            f"({'parity operating point' if n_leaves == 1 else 'in-kernel tree tax regime -- consider SKYRL_ISOEXEC_MOE_BLOCK_M=16'})",
            flush=True,
        )

    T, H = x.shape
    two_i = w1.shape[1]
    inter_dim = two_i // 2
    cfg = _bi_config(E, two_i, H)
    block_m = cfg.get("BLOCK_SIZE_M", _BLOCK_M_FALLBACK)

    sti, eids, ntpp, _ = _block_map(counts, T, E, block_m)
    compute_type = tl.bfloat16 if x.dtype is torch.bfloat16 else tl.float16

    # FUSED fc1+EPILOGUE (SKYRL_ISOEXEC_MOE_FUSED_EPILOGUE, default OFF). The fc1 GEMM is re-tiled over the
    # OUTPUT half-width so one program holds the gate and up tiles of the same output element in registers;
    # silu, the `* up`, the probs multiply and the bf16 round-trip all run on the accumulators, and `inter`
    # is never written. BITWISE-equal to the torch chain below -- the kernel rounds at exactly the three
    # points torch rounds (fc1 C-store, silu, the mul) -- so this is a drop-in for the `hc` fc2 consumes.
    #
    # The preconditions are the ones the kernel cannot express its way around: a [T] probs vector it can
    # index by permuted row (stride 1) and a bf16/fp16 compute dtype. Anything else falls through to the
    # torch chain rather than computing something subtly different. The activation itself (silu, no clamp,
    # offset 0) is already gated upstream in `_fused_experts_forward`.
    use_fused_epi = (
        _fused_epilogue_on()
        and probs is not None
        and probs.dim() == 1
        and probs.numel() == T
        and probs.stride(0) == 1
        and x.dtype in (torch.bfloat16, torch.float16)
    )

    if use_fused_epi:
        from .moe_epilogue_kernel import invoke_fused_moe_fc1_glu_kernel

        # `hc` directly: [T, f] in the compute dtype, router weight already folded in. This IS the
        # tensor the torch chain produces at `h.contiguous()`, so every fc2 branch below (single call,
        # in-kernel leaf tree, per-leaf fallback) is untouched and consumes it unchanged.
        hc = x.new_empty(T, inter_dim)
        invoke_fused_moe_fc1_glu_kernel(
            A=x,
            B=w1,
            C=hc,
            probs=probs,
            sorted_token_ids=sti,
            expert_ids=eids,
            num_tokens_post_padded=ntpp,
            config=cfg,
            compute_type=compute_type,
        )
    else:
        # fc1: C is [M, top_k, N]; the kernel indexes it with C.stride(1)/C.stride(2). With top_k = 1
        # that is just our [T, N] output wearing a middle axis.
        inter = x.new_empty(T, 1, two_i)
        invoke_fused_moe_triton_kernel(
            A=x,
            B=w1,
            C=inter,
            A_scale=None,
            B_scale=None,
            topk_weights=None,
            sorted_token_ids=sti,
            expert_ids=eids,
            num_tokens_post_padded=ntpp,
            mul_routed_weight=False,
            top_k=1,  # each row is already one (token, expert) pair
            config=cfg,
            compute_type=compute_type,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
        )

        # swiglu, then the router weight, EXACTLY as _batched_experts_forward does it:
        #     inter = act(x_glu) * (x_linear + offset)      # offset 0 for swiglu
        #     inter = (inter * probs.unsqueeze(-1)).to(inter.dtype)   # probs BEFORE fc2, round-trip
        # gate is the first half, up the second (torch.chunk order), matching the bmm path.
        gate, up = inter.reshape(T, two_i).chunk(2, dim=-1)
        h = torch.nn.functional.silu(gate) * up
        h = (h * probs.unsqueeze(-1)).to(h.dtype)  # probs keeps its dtype so the promote+round matches

        # fc2: the router weight is already folded into `h`, so mul_routed_weight stays False.
        hc = h.contiguous()

    def _fc2(a, b, k):
        o = x.new_empty(T, 1, H)
        invoke_fused_moe_triton_kernel(
            A=a,
            B=b,
            C=o,
            A_scale=None,
            B_scale=None,
            topk_weights=None,
            sorted_token_ids=sti,
            expert_ids=eids,
            num_tokens_post_padded=ntpp,
            mul_routed_weight=False,
            top_k=1,
            config=_bi_config(E, H, k),
            compute_type=compute_type,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
        )
        return o.reshape(T, H)

    if not pik_fc2:
        return _fc2(hc, w2, inter_dim)

    # pik-fc2: leaf-tree fc2, mirroring _leaftree_fc2 -- the moe_intermediate reduction is cut into
    # fixed leaves, each rounded to bf16 exactly where the single call rounds, promoted fp32
    # (lossless) and folded with the balanced tree, so the fc2 expression is the same G-leaf tree at
    # every ETP. n_leaves == 1 (ETP == G) still promotes fp32: this rank's shard IS one leaf and the
    # dispatcher's pik tree reduces across ranks.
    if inter_dim % n_leaves != 0:
        raise RuntimeError(
            f"[isoexec-moe] pik-fc2 fused: moe_intermediate shard {inter_dim} not divisible by "
            f"n_leaves={n_leaves} (G must divide the full moe_intermediate; check SKYRL_ISOEXEC_PIK_LEAVES)"
        )

    if os.environ.get("SKYRL_ISOEXEC_MOE_FUSED_LEAFTREE", "1") == "1":
        # IN-KERNEL leaf tree (moe_fused_leaftree): one launch, the fold in registers, one fp32 store.
        # The python per-leaf variant below is bitwise-identical but pays n_leaves extra [T, H] memory
        # sweeps. `=0` falls back to the per-leaf calls for A/B and debugging.
        from .moe_fused_leaftree import invoke_fused_moe_leaftree_kernel

        def _launch(C, wire: bool):
            invoke_fused_moe_leaftree_kernel(
                A=hc,
                B=w2,
                C=C,
                sorted_token_ids=sti,
                expert_ids=eids,
                num_tokens_post_padded=ntpp,
                n_leaves=n_leaves,
                config=_bi_config(E, H, inter_dim),
                compute_type=compute_type,
                wire_store=wire,
            )

        # Wire-stage: at n_leaves == 1 the fp32 C is a lossless promotion of the bf16 leaf value, and the
        # owner combine's next act is to round it straight back while copying it into symmetric memory.
        # Store the wire bytes there directly instead, if the sink grants a buffer and this geometry has
        # proven itself on live operands. Every refusal falls open to the fp32 path.
        if n_leaves == 1 and compute_type == tl.bfloat16:
            from .moe_pik_combine_owner import owner_wire_sink

            wire_c = owner_wire_sink(T, H, x.device)
            if wire_c is not None:
                out = _leaftree_wire_dispatch(_launch, wire_c, x, T, H, E, inter_dim)
                if out is not None:
                    return out

        out = x.new_empty(T, 1, H, dtype=torch.float32)
        _launch(out, False)
        return out.reshape(T, H)

    # python per-leaf fallback: the SAME kernel once per K-slice (strided views -- the wrapper
    # passes A/B strides, so no copy), then promote + tree-fold in torch.
    from .moe_batched_experts import _tree_sum

    lf = inter_dim // n_leaves
    nodes = [_fc2(hc[:, i * lf : (i + 1) * lf], w2[:, :, i * lf : (i + 1) * lf], lf).float() for i in range(n_leaves)]
    return _tree_sum(nodes)


class _FusedExpertsAutograd(torch.autograd.Function):
    """Fused kernel forward + a reference VJP, so the fused path stays autograd-safe if ever imported into
    a grad context.

    THE BACKWARD IS NOT ON THE HOT PATH: the fused kernel runs ROLLOUT-ONLY, under ``no_grad``, and
    training keeps the batched-bmm ``SequentialMLP`` for both directions. This wrapper exists to keep the
    module importable in a grad context without autograd raising on the backward-less Triton kernel, and
    to hand back a correct gradient if it is ever exercised, so the backward simply differentiates the
    fp32 reference (:func:`_ref_forward`) of the same function.
    """

    @staticmethod
    def forward(ctx, x, w1, w2, probs, counts, E):
        with torch.no_grad():
            out = _fused_forward(x, w1, w2, probs, counts, E)
        ctx.save_for_backward(x, w1, w2, probs, counts)
        ctx.E = E
        return out

    @staticmethod
    def backward(ctx, dout):
        x, w1, w2, probs, counts = ctx.saved_tensors
        names = ("x", "w1", "w2", "probs")
        with torch.enable_grad():
            leaves = [t.detach().requires_grad_(True) for t in (x, w1, w2, probs)]
            out = _ref_forward(*leaves, counts, ctx.E)
            # allow_unused ONLY so a missing grad RAISES with the numbers instead of dying at "index 0".
            # Never zero-fill a missing grad -- that turns "this tensor left the graph" into a silent
            # zero gradient, the failure mode this stack keeps paying for.
            # dout may be fp32 (the pik-fc2 forward emits the fp32 leaf-subtree partial) while the
            # reference computes in the model dtype -- match it; the VJP is fp-accurate, not bitwise.
            grads = torch.autograd.grad(out, leaves, dout.to(out.dtype), allow_unused=True)
        missing = [n for n, g in zip(names, grads) if g is None]
        if missing:
            raise RuntimeError(
                f"[isoexec-moe] fused-MoE reference VJP produced NO gradient for {missing}. "
                f"T={x.shape[0]} n_routed={int(counts.sum())} cap={int(counts.max())} E={ctx.E}"
            )
        return (*grads, None, None)


def _fused_experts_forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs):
    """Drop-in ``SequentialMLP.forward``: the same math, two fused MoE GEMMs instead of a bmm pair.

    Contract is unchanged from the bmm path -- the dispatcher hands us tokens already grouped by
    expert and owns the combine (``explicit_expert_comm``), so we return the local per-expert output
    and no bias.
    """
    global CALL_COUNT
    CALL_COUNT += 1

    x = permuted_local_hidden_states  # [T, h], expert-grouped
    if x.shape[0] == 0:
        return x.new_zeros(0, x.shape[-1]), None

    E = self.num_local_experts
    counts = tokens_per_expert.to(device=x.device, dtype=torch.long)

    # Stacked in-graph, exactly as the bmm path does, so autograd routes the expert weight grads back
    # to each linear_fc1/linear_fc2.weight. Note NO transpose here: the fused kernel wants B as
    # [E, N, K], which is the parameter's own layout -- the bmm path transposed only because a bmm
    # wants [E, K, N].
    #
    # The stack is FREE: each expert's .weight is already a view into one contiguous [E, ...] buffer, so
    # the "stack" IS that buffer. fused_expert_weights returns None the moment the alias is broken (a
    # Parameter object replaced by weight sync, a .data rebind, a cumem sleep) -- then we fall back to the
    # real stack rather than serve a stale weight.
    # ENGINE-ONLY, and that is load-bearing: a broken alias on the trainer can mean the distributed
    # optimizer re-pointed the param into its own buffer, where re-fusing would sever that aliasing. The
    # engine has no optimizer -- a break there is weight sync materialising a param -- so re-fusing is
    # unambiguously right. This function is installed only on the engine, so we inherit that scope for
    # free; the trainer's batched path keeps torch.stack.
    w1 = w2 = None
    if not getattr(self, "_ix_fw_giveup", False):
        from .moe_fused_weights import fused_expert_weights, refuse_if_synced

        w1 = fused_expert_weights(self, "linear_fc1")  # [E, 2f, h] or None
        w2 = fused_expert_weights(self, "linear_fc2")  # [E, h, f]  or None
        if w1 is None or w2 is None:
            # First forward, or the alias was broken by the weight sync that just ran. Re-fuse from the
            # CURRENT parameter values (so this can never resurrect stale bytes) and retry once.
            # refuse_if_synced (NOT fuse_module(force=True)) caps this at ONE re-fuse per weight sync: a
            # re-fuse is far more expensive than the torch.stack it replaces, so doing it per forward
            # would be a regression rather than a missed optimisation.
            if refuse_if_synced(self) is None:
                # None also means "rate-limited, try again after the next sync". Give up for good
                # only when the module genuinely cannot be fused, i.e. no fused state survived.
                if getattr(self, "_isoexec_fused_weights", None) is None:
                    self._ix_fw_giveup = True
            else:
                w1 = fused_expert_weights(self, "linear_fc1")
                w2 = fused_expert_weights(self, "linear_fc2")
        # Capture-time visibility: if this forward is the one being CUDA-graph captured, an INACTIVE
        # alias means the graph bakes a torch.stack temp the sync-boundary eager refresh can't reach.
        _log_fused_weights_state_once(w1 is not None and w2 is not None)
    if w1 is None:
        w1 = torch.stack([e.linear_fc1.weight for e in self.local_experts])  # [E, 2f, h]
    if w2 is None:
        w2 = torch.stack([e.linear_fc2.weight for e in self.local_experts])  # [E, h, f]

    out = _FusedExpertsAutograd.apply(x, w1, w2, permuted_probs, counts, E)
    return out, None


def install_fused_experts() -> bool:
    """Rebind ``SequentialMLP.forward`` to the fused-MoE GEMM path. Idempotent.

    ENGINE ONLY. The caller (``prepare_isoexec_moe``) invokes this only for ``side == "ENGINE"``, so the
    fused rebind lands in the rollout engine and NOT in the trainer -- which is the whole rollout-only
    plan (see the module docstring). It runs AFTER ``install_batched_sequential_mlp`` in the engine, so
    this rebind wins there; the trainer, never seeing this call, keeps the batched-bmm forward.
    """
    global _installed, _orig_forward

    if _installed:
        return True
    from megatron.core.transformer.moe.experts import SequentialMLP

    _orig_forward = SequentialMLP.forward
    SequentialMLP.forward = _fused_experts_forward
    _installed = True
    print(
        "[ISOEXEC-MOE] ENGINE: SequentialMLP.forward -> FUSED MoE GEMMs (vLLM Triton, batch-invariant, "
        "bitwise == the trainer's batched bmm)",
        flush=True,
    )
    return True


# Cap on the elements of one padded [chunk, cap, *] staging tensor in the reference. Bounds peak memory
# when the routing is skewed (`cap` is the busiest expert's row count, so it scales with SKEW, not with
# tokens) and keeps every bmm under the 2^31-element ceiling of the batch-invariant Triton bmm, which
# reads out of bounds past it.
_REF_MAX_ELEMS = 1 << 26


def _ref_forward(x, w1, w2, probs, counts, E: int):
    """Differentiable reference: ONE bmm pair over expert-padded rows. The VJP, and nothing else.

    THIS IS THE BACKWARD, SO IT HAS TO BE FAST. A python loop over the experts here -- the loop
    ``moe_batched_experts`` exists to eliminate -- is correct but swamps the forward win end to end. So
    the reference is BATCHED like the path it replaces: scatter the expert-grouped rows into
    ``[E, cap, h]``, one bmm against the stacked fc1, the glu epilogue, one bmm against fc2, gather back.
    It never runs in the forward, so it does not have to be bitwise the kernel -- only the gradient of the
    same function, to fp accuracy.

    ``cap`` needs a D2H read, which is fine only HERE: the backward is never CUDA-graph captured, and the
    forward's block map stays sync-free, which is what the capture depends on.
    """
    T, H = x.shape
    dev = x.device
    n_routed = int(counts.sum().item())
    cap = int(counts.max().item())

    # EVERY row must belong to an expert. If the counts do not account for all of them, the rows past
    # `n_routed` would be silently dropped -- and `out` would never be written for them, so it would carry
    # the garbage of an uninitialized buffer in the forward and a ZERO GRADIENT here. Fail loudly instead.
    if n_routed != T:
        raise RuntimeError(
            f"[isoexec-moe] expert counts sum to {n_routed} but got {T} permuted rows. Every row must "
            "belong to exactly one expert; the fused forward's block map assumes it too."
        )

    # No token reached any expert (T == 0 is caught upstream, so this is the all-zero-counts case).
    # The output is zeros and the gradient w.r.t. x is genuinely zero -- but it must be expressed as a
    # FUNCTION of x, or autograd raises "the differentiated Tensor at index 0 appears to not have been
    # used in the graph" rather than handing back the zero.
    if cap == 0:
        return x * 0.0

    cu = torch.cumsum(counts, 0) - counts  # first row of each expert
    idx = torch.arange(T, device=dev)
    # row -> (expert, slot within that expert). searchsorted, not repeat_interleave: the latter's
    # output size is data-dependent.
    e_of = torch.searchsorted(cu + counts, idx, right=True).clamp_(max=E - 1)
    slot = idx - cu[e_of]

    # Chunk the expert axis so one staging tensor stays bounded even under a skewed routing.
    per_expert = cap * max(H, w1.shape[1])
    step = max(1, _REF_MAX_ELEMS // max(1, per_expert))

    out = x.new_zeros(T, H)
    for s in range(0, E, step):
        e = min(s + step, E)
        sel = (e_of >= s) & (e_of < e)
        if not bool(sel.any()):
            continue
        rows = idx[sel]
        xe = x.new_zeros(e - s, cap, H).index_put((e_of[rows] - s, slot[rows]), x[rows])
        # torch.ops.aten.bmm, NOT torch.bmm. `enable_batch_invariant_mode` REBINDS the python attr
        # torch.bmm to its raw Triton function, which never records autograd -- so a `torch.bmm` here
        # would silently drop x out of the graph. Dispatching the OPERATOR routes Autograd -> the
        # (overridden, batch-invariant) CUDA kernel: gradients AND determinism.
        inter = torch.ops.aten.bmm(xe, w1[s:e].transpose(1, 2))  # [chunk, cap, 2f]
        gate, up = inter.chunk(2, dim=-1)
        h = torch.nn.functional.silu(gate) * up
        o = torch.ops.aten.bmm(h, w2[s:e].transpose(1, 2))  # [chunk, cap, h]
        vals = o[e_of[rows] - s, slot[rows]] * probs[rows].unsqueeze(-1).to(o.dtype)
        out = out.index_put((rows,), vals)
    return out
