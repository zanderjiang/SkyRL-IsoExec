"""Row-count- and TP-invariant sampled-token logprob via a fixed-leaf exp-sum tree.

The full vocabulary is cut into ``G`` fixed contiguous leaves (``SKYRL_ISOEXEC_PIK_LEAVES``);
each per-row leaf sum of ``exp(x - m)`` is a Kahan fp32 Triton reduction in fixed column order,
and the G leaf sums are folded by ``pik.plan.combine_order(G)``. Per-program work depends only
on the leaf, so the result is row-count invariant; TP only moves leaf ownership between ranks
(a contiguous leaf range is a subtree of the combine tree), so it is TP-invariant too.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import torch
import torch.distributed as dist

LEAVES_ENV = "SKYRL_ISOEXEC_PIK_LEAVES"  # the same contract constant pik reads
BANNER = "[ISOEXEC-ROWINV-LOGPROB]"
PROBE_EVERY = 512
# The candidate's lse differs from ATen's by 1-2 ULP; structural bugs land far above this.
PROBE_MAX_ABS = 1e-4
BLOCK = 4096  # fixed leaf-internal step: part of the numerics contract, never autotuned
NUM_WARPS = 4  # pinned: tl.sum lowering depends on (BLOCK, num_warps); autotuning would move bits

_KERNEL = None
_KERNEL_ERROR = ""
_GROUP_ADMISSION = {}
_S = {
    "calls": 0,
    "served": 0,
    "declined": 0,
    "decline_reason": "",
    "agreed": None,
    "probes": 0,
    "tokens": 0,
    "full_rows": 0,
    "full_gather_bytes_avoided": 0,
    "leaf_wire_floats": 0,
    "hot_hits": 0,
    "reported": 0,
    "bannered": False,
}
#: Probe latch for the full-row entry. ``fused=False`` demotes only the single-kernel fast path
#: (the eager rowinv path keeps serving, so trainer==engine bits are unaffected).
_FULL_ROW = {
    "agreed": None,
    "served": 0,
    "latched_off": False,
    "latch_reason": "",
    "fused": None,
    "fused_reason": "",
    "shapes": set(),
}
#: Synthesized admission targets for the full-row entry, cached per device.
_FULL_TARGET: dict = {}


def stats() -> dict:
    """Return a copy of the engagement census; only ``served > 0`` proves ownership."""
    return dict(_S)


def _reset_stats() -> None:
    _S.update(
        {
            "calls": 0,
            "served": 0,
            "declined": 0,
            "decline_reason": "",
            "agreed": None,
            "probes": 0,
            "tokens": 0,
            "full_rows": 0,
            "full_gather_bytes_avoided": 0,
            "leaf_wire_floats": 0,
            "hot_hits": 0,
            "reported": 0,
            "bannered": False,
        }
    )
    _FULL_ROW.update(
        {
            "agreed": None,
            "served": 0,
            "latched_off": False,
            "latch_reason": "",
            "fused": None,
            "fused_reason": "",
            "shapes": set(),
        }
    )
    _FULL_TARGET.clear()


def _reset_for_test() -> None:
    global _KERNEL, _KERNEL_ERROR
    _KERNEL = None
    _KERNEL_ERROR = ""
    _GROUP_ADMISSION.clear()
    _reset_stats()


def release_group(group) -> None:
    """Release admission state before destroying one TP group."""
    _GROUP_ADMISSION.pop(_group_key(group), None)


def reset_for_teardown() -> None:
    """Release all process-group-bound state before distributed reinitialization."""
    _GROUP_ADMISSION.clear()
    _reset_stats()


def _group_key(group):
    # `None` is the world=1 (engine full-row) view and needs a stable key of its own.
    return id(group) if group is not None else "none"


def _load_kernel():
    global _KERNEL, _KERNEL_ERROR
    if _KERNEL is not None or _KERNEL_ERROR:
        return _KERNEL
    try:
        import triton
        import triton.language as tl
        from triton.language.extra import libdevice

        from ..collectives.pik_bootstrap import ensure_pik

        ensure_pik()
        from pik.plan import (
            combine_order,  # noqa: PLC0415 -- resolvable only after ensure_pik
        )

        @triton.jit
        def leaf_sum_exp(out_ptr, x_ptr, row_max_ptr, row_stride, out_stride, L: tl.constexpr, BLOCK: tl.constexpr):
            # One program per (row, local leaf). Loop bounds, step and masking depend only on the
            # leaf geometry (L, BLOCK), never on the grid, so the leaf sum is row-count invariant.
            row = tl.program_id(0)
            leaf = tl.program_id(1)
            base = row * row_stride + leaf * L
            row_max = tl.load(row_max_ptr + row)
            acc = 0.0
            comp = 0.0  # Kahan compensation
            for off in range(0, L, BLOCK):
                idx = off + tl.arange(0, BLOCK)
                value = tl.load(x_ptr + base + idx, mask=idx < L, other=0.0).to(tl.float32)
                block_sum = tl.sum(tl.where(idx < L, libdevice.exp(value - row_max), 0.0))
                y = block_sum - comp
                t = acc + y
                comp = (t - acc) - y
                acc = t
            tl.store(out_ptr + row * out_stride + leaf, acc)

        @triton.jit
        def finalize_rows(
            out_ptr,
            lse_ptr,
            leaf_ptr,
            row_max_ptr,
            x_ptr,
            target_ptr,
            sampled_ptr,
            row_stride,
            full_vocab,
            G: tl.constexpr,
            LOG2G: tl.constexpr,
            LOCAL_GATHER: tl.constexpr,
        ):
            # One program per row: fold the G leaf sums, log, gather/mask. The pairwise tl.split
            # fold IS combine_order(G), bit-compared against the eager torch fold on every probe.
            row = tl.program_id(0)
            v = tl.load(leaf_ptr + row * G + tl.arange(0, G))
            for _ in tl.static_range(LOG2G):
                lo, hi = tl.split(tl.reshape(v, (v.shape[0] // 2, 2)))
                v = lo + hi
            s = tl.sum(v)  # single element after the fold: extraction, not arithmetic
            row_max = tl.load(row_max_ptr + row)
            target = tl.load(target_ptr + row)
            valid = (target >= 0) & (target < full_vocab)
            if LOCAL_GATHER:
                # world=1: the full row is local, gather the sampled raw logit in-kernel.
                safe = tl.where(valid, target, 0)
                sampled = tl.load(x_ptr + row * row_stride + safe).to(tl.float32)
            else:
                # world>1: distributed by the exact masked all_reduce(SUM).
                sampled = tl.load(sampled_ptr + row)
            log_s = libdevice.log(s)  # bit-identical to torch.log on fp32 (probe-checked)
            tl.store(out_ptr + row, tl.where(valid, (sampled - row_max) - log_s, 0.0))
            tl.store(lse_ptr + row, row_max + log_s)

        @triton.jit
        def full_row_logprobs(
            out_ptr,
            x_ptr,
            row_stride,
            out_stride,
            L: tl.constexpr,
            G: tl.constexpr,
            LOG2G: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            # Engine full-row steady state in ONE launch: row max, G Kahan leaf sums,
            # combine_order(G) fold, and the two output subtracts -- the eager path's expression.
            row = tl.program_id(0).to(tl.int64)
            x_row = x_ptr + row * row_stride
            out_row = out_ptr + row * out_stride
            V: tl.constexpr = G * L

            # pass 1: row max. Exact and order-independent, so any schedule gives torch.amax's bits.
            m = -float("inf")
            for off in range(0, V, BLOCK):
                idx = off + tl.arange(0, BLOCK)
                v = tl.load(x_row + idx, mask=idx < V, other=-float("inf")).to(tl.float32)
                m = tl.maximum(m, tl.max(v))

            # pass 2: the G leaf sums, ascending leaf order, each the exact leaf_sum_exp body.
            vec = tl.zeros((G,), dtype=tl.float32)
            for g in tl.static_range(G):
                acc = 0.0
                comp = 0.0  # Kahan compensation, identical to leaf_sum_exp
                base = g * L
                for off in range(0, L, BLOCK):
                    idx = off + tl.arange(0, BLOCK)
                    value = tl.load(x_row + base + idx, mask=idx < L, other=0.0).to(tl.float32)
                    block_sum = tl.sum(tl.where(idx < L, libdevice.exp(value - m), 0.0))
                    y = block_sum - comp
                    t = acc + y
                    comp = (t - acc) - y
                    acc = t
                vec = tl.where(tl.arange(0, G) == g, acc, vec)

            # the combine_order(G) fold: the same pairwise tl.split schedule finalize_rows runs.
            for _ in tl.static_range(LOG2G):
                lo, hi = tl.split(tl.reshape(vec, (vec.shape[0] // 2, 2)))
                vec = lo + hi
            s = tl.sum(vec)  # single element after the fold: extraction, not arithmetic
            log_s = libdevice.log(s)  # bit-identical to torch.log on fp32 (probe-checked)

            # pass 3: the eager tail's two fp32 subtracts, elementwise.
            for off in range(0, V, BLOCK):
                idx = off + tl.arange(0, BLOCK)
                v = tl.load(x_row + idx, mask=idx < V, other=0.0).to(tl.float32)
                tl.store(out_row + idx, (v - m) - log_s, mask=idx < V)

        _KERNEL = (triton, leaf_sum_exp, finalize_rows, combine_order, full_row_logprobs)
    except Exception as error:  # noqa: BLE001 -- unavailable means the incumbent remains owner
        _KERNEL_ERROR = repr(error)
    return _KERNEL


def _banner_once(on: bool) -> None:
    if _S["bannered"]:
        return
    _S["bannered"] = True
    if on:
        print(
            f"{BANNER} sampled logprobs use the fixed-G leaf exp-sum tree "
            f"(G={os.environ.get(LEAVES_ENV, '8')}, BLOCK={BLOCK}, Kahan fp32) at all four sites -- "
            "the composed default, no flag. Unsupported layouts and a failed first-call probe "
            "(fused-fold bitwise self-check + tolerance vs the incumbent) fall back to the "
            "incumbent. Judge engagement only by served>0 in the census.",
            flush=True,
        )
    else:
        print(
            f"{BANNER} SKYRL_ISOEXEC != 1: IsoExec is not composed in this process, so the "
            "incumbent ATen full-vocabulary sum tree remains the sampled-logprob owner.",
            flush=True,
        )


def _report() -> None:
    calls = _S["calls"]
    if calls < 1 or (calls & (calls - 1)) or calls == _S["reported"]:
        return
    _S["reported"] = calls
    print(
        f"{BANNER} CENSUS pid={os.getpid()} calls={calls} served={_S['served']} "
        f"declined={_S['declined']} tokens={_S['tokens']} probes={_S['probes']} "
        f"hot_hits={_S['hot_hits']} "
        f"full_gather_avoided={_S['full_gather_bytes_avoided'] / 1e9:.2f} GB "
        f"leaf_wire_floats={_S['leaf_wire_floats']}"
        + (f" last_decline={_S['decline_reason']}" if _S["decline_reason"] else ""),
        flush=True,
    )


def _decline(reason: str) -> None:
    _S["declined"] += 1
    _S["decline_reason"] = reason
    _report()


def _hot_key(logits, target, vocab_start_index, vocab_end_index, src_dtype):
    return (
        tuple(logits.shape),
        tuple(logits.stride()),
        logits.device,
        logits.dtype,
        tuple(target.shape),
        target.dtype,
        src_dtype,
        vocab_start_index,
        vocab_end_index,
    )


def _admit(
    logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start_index: int,
    vocab_end_index: int,
    group,
    src_dtype: torch.dtype,
) -> tuple[tuple[int, int, int] | None, str]:
    _S["calls"] += 1

    # Latched hot route for admitted signatures. Any env/layout/dtype change misses the guard and
    # falls into the full admission below, which still RAISES on structural drift.
    group_cache = _GROUP_ADMISSION.get(_group_key(group))
    if group_cache is not None and not group_cache["latched_off"]:
        if (
            os.environ.get("SKYRL_ISOEXEC"),
            os.environ.get(LEAVES_ENV, "8"),
        ) == group_cache["env_sig"]:
            hot = group_cache["hot"].get(_hot_key(logits, target, vocab_start_index, vocab_end_index, src_dtype))
            if hot is not None:
                _S["hot_hits"] += 1
                return hot, ""

    on = os.environ.get("SKYRL_ISOEXEC") == "1"
    _banner_once(on)
    if group is not None and not (dist.is_available() and dist.is_initialized()):
        reason = "a process group was passed but torch.distributed is not initialized"
        _decline(reason)
        return None, reason
    if group is None:
        world, rank = 1, 0
    else:
        world = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
    shard_vocab = logits.shape[-1] if logits.ndim else 0
    full_vocab = shard_vocab * world
    try:
        g_leaves = int(os.environ.get(LEAVES_ENV, "8"))
    except ValueError:
        g_leaves = 0
    capability = torch.cuda.get_device_capability(logits.device) if logits.is_cuda else None

    reason = ""
    if not on:
        reason = "SKYRL_ISOEXEC!=1"
    elif not logits.is_cuda or logits.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        reason = f"requires CUDA fp32/bf16/fp16 logits, got device={logits.device} dtype={logits.dtype}"
    elif src_dtype not in (torch.float32, torch.bfloat16, torch.float16):
        reason = f"source dtype {src_dtype} is not fp32/bf16/fp16"
    elif logits.dtype is not torch.float32 and logits.dtype is not src_dtype:
        reason = f"low-precision logits dtype {logits.dtype} contradicts declared src_dtype {src_dtype}"
    elif logits.ndim not in (2, 3) or target.shape != logits.shape[:-1] or not logits.is_contiguous():
        reason = f"unsupported logits/target layout logits={tuple(logits.shape)} target={tuple(target.shape)}"
    elif target.dtype not in (torch.int32, torch.int64):
        reason = f"target dtype {target.dtype} is not int32/int64"
    elif target.numel() == 0:
        reason = "zero rows"
    elif vocab_start_index != rank * shard_vocab or vocab_end_index != vocab_start_index + shard_vocab:
        reason = (
            f"non-contiguous/equal vocab partition start={vocab_start_index} end={vocab_end_index} "
            f"rank={rank} shard={shard_vocab}"
        )
    elif g_leaves < 1 or (g_leaves & (g_leaves - 1)) != 0:
        reason = f"{LEAVES_ENV}={os.environ.get(LEAVES_ENV)!r} is not a positive power of two"
    elif world > g_leaves or g_leaves % world != 0:
        reason = f"world={world} does not map onto contiguous subtrees of the G={g_leaves} leaf tree"
    elif full_vocab % g_leaves != 0:
        reason = f"V={full_vocab} is not divisible by G={g_leaves}: unequal leaves change the contract, declining"
    elif capability != (9, 0):
        reason = f"unvalidated CUDA capability {capability}"
    elif _load_kernel() is None:
        reason = f"Triton/pik leaf kernel unavailable: {_KERNEL_ERROR}"

    # Immutable per-process contract: only facts selecting the COLLECTIVE SEQUENCE, so rank-local
    # divergence RAISES. Payload dtypes stay out (see below) -- collectives always run on fp32.
    contract = (
        on,
        os.environ.get("SKYRL_ISOEXEC"),
        logits.device.type,
        logits.device.index,
        world,
        rank,
        shard_vocab,
        vocab_start_index,
        vocab_end_index,
        g_leaves,
        capability,
        _KERNEL is not None,
        _KERNEL_ERROR,
    )
    signature = (
        tuple(logits.shape),
        tuple(logits.stride()),
        logits.dtype,
        src_dtype,
        tuple(target.shape),
        target.dtype,
        reason,
    )
    if group_cache is not None:
        if contract != group_cache["contract"]:
            raise RuntimeError(
                f"{BANNER} STRUCTURAL DRIFT after the TP-unanimous admission vote: "
                f"first={group_cache['contract']!r} now={contract!r}. ENV, device, world, "
                "vocabulary partition, G, capability and kernel availability are immutable for "
                "this process; refusing before entering either collective sequence. (Payload "
                "dtypes and layouts are per-call signature fields and never trip this gate.)"
            )
        if group_cache["latched_off"]:
            cached_reason = reason or "cached TP peer refusal or end-to-end probe failure"
            _decline(cached_reason)
            return None, cached_reason
        cached = group_cache["verdicts"].get(signature)
        if cached is not None and not cached:
            cached_reason = reason or "cached TP peer refusal"
            _decline(cached_reason)
            return None, cached_reason
        if cached:
            return (world, rank, g_leaves), ""
    else:
        # Retain the group object so a destroy/reinit cycle cannot reuse its Python id and inherit
        # a prior group's admission or probe verdict.
        group_cache = {
            "group": group,
            "contract": contract,
            "env_sig": (os.environ.get("SKYRL_ISOEXEC"), os.environ.get(LEAVES_ENV, "8")),
            "verdicts": {},
            "hot": {},
            "latched_off": False,
            "agreed": None,
            "served": 0,
        }
        _GROUP_ADMISSION[_group_key(group)] = group_cache

    # Candidate and incumbent issue different collectives past this point, so every new signature
    # per TP group must be unanimous; MIN makes any one rank's refusal everybody's.
    if world > 1:
        vote = torch.tensor(int(not reason), dtype=torch.int32, device=logits.device)
        dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=group)
        admitted = bool(vote.item())
    else:
        admitted = not reason
    group_cache["verdicts"][signature] = admitted
    if not admitted:
        if not reason:
            reason = "a TP peer declined the candidate signature"
        _decline(reason)
        return None, reason
    verdict = (world, rank, g_leaves)
    group_cache["hot"][_hot_key(logits, target, vocab_start_index, vocab_end_index, src_dtype)] = verdict
    return verdict, ""


def _shared_pieces(
    x: torch.Tensor,
    target: torch.Tensor,
    vocab_start_index: int,
    world: int,
    group,
    g_leaves: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return (row_max [rows], leaf_sums [rows, G], sampled [rows] or None at world=1)."""
    leaf_kernel = _KERNEL[1]
    rows, shard_vocab = x.shape
    full_vocab = shard_vocab * world
    leaf_cols = full_vocab // g_leaves
    leaves_local = g_leaves // world

    # 1) global row max: local amax then all_reduce(MAX); max is exact and order-independent.
    row_max = torch.amax(x, dim=-1)
    if world > 1:
        dist.all_reduce(row_max, op=dist.ReduceOp.MAX, group=group)

    # 2) leaf sums for this rank's leaves, one program per (row, leaf), fixed order + Kahan.
    local_sums = torch.empty((rows, leaves_local), device=x.device, dtype=torch.float32)
    leaf_kernel[(rows, leaves_local)](
        local_sums,
        x,
        row_max,
        shard_vocab,
        leaves_local,
        L=leaf_cols,
        BLOCK=BLOCK,
        num_warps=NUM_WARPS,
    )

    # 3) all G leaf sums on every rank, in global leaf order (ownership is contiguous ascending,
    #    so rank-order concat IS leaf order). all_gather moves bytes only.
    if world > 1:
        parts = [torch.empty_like(local_sums) for _ in range(world)]
        dist.all_gather(parts, local_sums, group=group)
        leaf_sums = torch.cat(parts, dim=1)
        # 5) the sampled raw logit lives on exactly one rank: mask + all_reduce(SUM) is exact.
        valid = (target >= 0) & (target < full_vocab)
        in_shard = valid & (target >= vocab_start_index) & (target < vocab_start_index + shard_vocab)
        local_target = (target - vocab_start_index).masked_fill(~in_shard, 0).to(torch.int64)
        sampled = torch.gather(x, 1, local_target.unsqueeze(1)).squeeze(1)
        sampled = sampled.masked_fill(~in_shard, 0.0)
        dist.all_reduce(sampled, op=dist.ReduceOp.SUM, group=group)
    else:
        leaf_sums = local_sums
        sampled = None
    return row_max, leaf_sums, sampled


def _finalize_fused(
    x: torch.Tensor,
    target: torch.Tensor,
    row_max: torch.Tensor,
    leaf_sums: torch.Tensor,
    sampled: torch.Tensor | None,
    full_vocab: int,
    g_leaves: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine tree, log, sampled gather/mask and lse in one launch."""
    finalize_kernel = _KERNEL[2]
    rows = x.shape[0]
    out = torch.empty(rows, device=x.device, dtype=torch.float32)
    lse = torch.empty(rows, device=x.device, dtype=torch.float32)
    finalize_kernel[(rows,)](
        out,
        lse,
        leaf_sums,
        row_max,
        x,
        target,
        sampled if sampled is not None else row_max,  # unread when LOCAL_GATHER
        x.shape[1],
        full_vocab,
        G=g_leaves,
        LOG2G=g_leaves.bit_length() - 1,
        LOCAL_GATHER=sampled is None,
        num_warps=1,
    )
    return out, lse


def _finalize_reference(
    x: torch.Tensor,
    target: torch.Tensor,
    row_max: torch.Tensor,
    leaf_sums: torch.Tensor,
    sampled: torch.Tensor | None,
    full_vocab: int,
    g_leaves: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager finalize (explicit ``combine_order(G)`` fold + ``torch.log``); probe path only.

    The contract's letter: ``_finalize_fused`` must reproduce it bit for bit.
    """
    combine_order = _KERNEL[3]
    slots = list(leaf_sums.unbind(dim=1))
    for dst, lhs, rhs in combine_order(g_leaves):
        slots[dst] = slots[lhs] + slots[rhs]
    log_s = torch.log(slots[0])
    valid = (target >= 0) & (target < full_vocab)
    if sampled is None:
        safe = target.masked_fill(~valid, 0).to(torch.int64)
        sampled = torch.gather(x, 1, safe.unsqueeze(1)).squeeze(1)
    out = ((sampled - row_max) - log_s).masked_fill(~valid, 0.0)
    return out, row_max + log_s


class _RowInvSampledLogprob(torch.autograd.Function):
    """Forward under the pinned leaf tree; rank-local backward from per-row scalars only."""

    @staticmethod
    def forward(ctx, logits, target, vocab_start_index, world, group, g_leaves):
        x = logits.reshape(-1, logits.shape[-1])
        t = target.reshape(-1)
        full_vocab = logits.shape[-1] * world
        # Kernel input dtype is pinned fp32: Triton's intra-block tl.sum order follows the load
        # element width (a bf16 load moves ~30% of leaf sums by 1 ULP), so widen transiently here.
        x_wide = x if x.dtype is torch.float32 else x.to(torch.float32)
        row_max, leaf_sums, sampled = _shared_pieces(x_wide, t, vocab_start_index, world, group, g_leaves)
        out, lse = _finalize_fused(x_wide, t, row_max, leaf_sums, sampled, full_vocab, g_leaves)
        ctx.save_for_backward(logits, target, lse)
        ctx.vocab_start_index = vocab_start_index
        ctx.full_vocab = full_vocab
        return out.reshape(target.shape)

    @staticmethod
    def backward(ctx, grad_out):
        logits, target, lse = ctx.saved_tensors
        shard_vocab = logits.shape[-1]
        x = logits.reshape(-1, shard_vocab)
        t = target.reshape(-1)
        upstream = grad_out.reshape(-1).to(torch.float32)
        valid = (t >= 0) & (t < ctx.full_vocab)
        upstream = torch.where(valid, upstream, torch.zeros_like(upstream))
        # grad_x_j = g * (delta_{j,target} - exp(x_j - lse)); local columns only, no collective.
        # x may be the original bf16/fp16 shard; `x - lse` promotes, so exp runs in fp32 either way.
        grad = torch.exp(x - lse.unsqueeze(1)).mul_(-upstream.unsqueeze(1))
        in_shard = valid & (t >= ctx.vocab_start_index) & (t < ctx.vocab_start_index + shard_vocab)
        local_target = (t - ctx.vocab_start_index).masked_fill(~in_shard, 0).to(torch.int64)
        grad.scatter_add_(
            1,
            local_target.unsqueeze(1),
            torch.where(in_shard, upstream, torch.zeros_like(upstream)).unsqueeze(1),
        )
        if grad.dtype is not logits.dtype:
            grad = grad.to(logits.dtype)  # backward owes no bitwise contract; optimizer-only
        return grad.reshape(logits.shape), None, None, None, None, None


@torch.no_grad()
def _forward_reference(
    logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start_index: int,
    world: int,
    group,
    g_leaves: int,
) -> torch.Tensor:
    """The forward with the eager finalize, for the probe's fused-fold bitwise self-check."""
    x = logits.reshape(-1, logits.shape[-1])
    t = target.reshape(-1)
    full_vocab = logits.shape[-1] * world
    x_wide = x if x.dtype is torch.float32 else x.to(torch.float32)  # same transient widen
    row_max, leaf_sums, sampled = _shared_pieces(x_wide, t, vocab_start_index, world, group, g_leaves)
    out, _lse = _finalize_reference(x_wide, t, row_max, leaf_sums, sampled, full_vocab, g_leaves)
    return out.reshape(target.shape)


def _probe_final(
    candidate: torch.Tensor,
    incumbent: torch.Tensor,
    eager_fold: torch.Tensor,
    group,
    world: int,
) -> tuple[bool, str]:
    """One unanimous vote over both probe halves: fused-vs-eager bitwise, incumbent tolerance.

    The candidate is a different fp32 lse tree than ATen's by design (~1-2 ULP), so the
    incumbent half is a tolerance gate, not bit-equality.
    """
    fused_same = torch.equal(candidate.contiguous().view(torch.int32), eager_fold.contiguous().view(torch.int32))
    finite = bool(torch.isfinite(candidate).all().item()) and bool(torch.isfinite(incumbent).all().item())
    max_abs = float((candidate.float() - incumbent.float()).abs().max().item()) if finite else float("inf")
    ok_local = fused_same and finite and max_abs <= PROBE_MAX_ABS
    if world > 1:
        ok = torch.tensor(int(ok_local), dtype=torch.int32, device=candidate.device)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
        ok_all = bool(ok.item())
    else:
        ok_all = ok_local
    _S["probes"] += 1
    if not fused_same:
        return False, "fused finalize kernel is not bit-identical to the eager combine_order fold"
    if not finite:
        return False, "non-finite final sampled output"
    if max_abs > PROBE_MAX_ABS:
        return False, f"|candidate - incumbent| max {max_abs:.3e} exceeds the {PROBE_MAX_ABS:.1e} tolerance gate"
    return ok_all, "peer probe failed"


def rowinv_sampled_logprobs(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
    group,
    src_dtype: torch.dtype,
    reference: Callable[[], torch.Tensor],
) -> torch.Tensor | None:
    """Return row/TP-invariant sampled logprobs, or ``None`` to retain the incumbent owner."""
    admitted, _reason = _admit(logits, target, vocab_start_index, vocab_end_index, group, src_dtype)
    if admitted is None:
        return None
    world, _rank, g_leaves = admitted

    result = _RowInvSampledLogprob.apply(logits, target, vocab_start_index, world, group, g_leaves)

    group_cache = _GROUP_ADMISSION[_group_key(group)]
    should_probe = group_cache["agreed"] is None or (
        group_cache["served"] > 0 and group_cache["served"] % PROBE_EVERY == 0
    )
    if should_probe:
        # Both reruns issue extra collectives; every rank reaches this branch on the same call by
        # construction, so they stay aligned.
        incumbent = reference()
        eager_fold = _forward_reference(logits, target, vocab_start_index, world, group, g_leaves)
        ok, reason = _probe_final(result.detach(), incumbent.detach(), eager_fold, group, world)
        if not ok:
            group_cache["agreed"] = False
            _S["agreed"] = False
            group_cache["latched_off"] = True
            _decline(f"end-to-end probe failed: {reason}")
            return incumbent
        group_cache["agreed"] = True
        _S["agreed"] = True

    _S["served"] += 1
    group_cache["served"] += 1
    _S["tokens"] += target.numel()
    if world > 1:
        # The incumbent materializes a [rows, V] fp32 gather per rank; the leaf wire carries
        # rows*G floats instead.
        _S["full_gather_bytes_avoided"] += target.numel() * logits.shape[-1] * world * 4
    _S["leaf_wire_floats"] += target.numel() * g_leaves
    _report()
    return result


def _probe_full_row(
    full: torch.Tensor,
    incumbent: torch.Tensor,
    x_wide: torch.Tensor,
    row_max: torch.Tensor,
    leaf_sums: torch.Tensor,
    g_leaves: int,
) -> tuple[bool, str]:
    """The full-row entry's two probe halves; world=1, so no vote to align.

    Bitwise against the sampled path (proving the full row carries the trainer's bits at any
    column), then a tolerance gate against the incumbent's differing fp32 lse tree.
    """
    rows, shard_vocab = x_wide.shape
    # Deterministic per-row probe columns spread across all G leaves (Knuth hash mod V).
    t = (torch.arange(rows, device=x_wide.device, dtype=torch.int64) * 2654435761) % shard_vocab
    fused, _lse = _finalize_fused(x_wide, t, row_max, leaf_sums, None, shard_vocab, g_leaves)
    gathered = torch.gather(full, 1, t.unsqueeze(1)).squeeze(1)
    _S["probes"] += 1
    if not torch.equal(gathered.contiguous().view(torch.int32), fused.contiguous().view(torch.int32)):
        return False, (
            "full-row gather is not bit-identical to the fused sampled finalize "
            "(eager combine_order fold / torch.log drifted from the tl.split fold / libdevice.log)"
        )
    if not (bool(torch.isfinite(full).all().item()) and bool(torch.isfinite(incumbent).all().item())):
        return False, "non-finite full-row candidate or incumbent output"
    max_abs = float((full - incumbent.float()).abs().max().item())
    if max_abs > PROBE_MAX_ABS:
        return False, f"|candidate - incumbent| max {max_abs:.3e} exceeds the {PROBE_MAX_ABS:.1e} tolerance gate"
    return True, ""


def _full_target(rows: int, device: torch.device) -> torch.Tensor:
    """The synthesized admission target for the full-row entry, cached per device.

    ``_admit`` reads only its shape/dtype; the forward never reads a value.
    """
    cached = _FULL_TARGET.get(device)
    if cached is None or cached.shape[0] < rows:
        size = max(rows, 2 * (cached.shape[0] if cached is not None else 512))
        cached = torch.zeros(size, dtype=torch.int64, device=device)
        _FULL_TARGET[device] = cached
    return cached[:rows]


def _full_row_eager(
    x_wide: torch.Tensor, admit_target: torch.Tensor, g_leaves: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The full-row contract's letter: amax + leaf kernel + eager combine_order fold + subtracts.

    The fused fast path must reproduce it bit for bit, and is demoted to this path if it does not.
    """
    row_max, leaf_sums, _sampled = _shared_pieces(x_wide, admit_target, 0, 1, None, g_leaves)
    combine_order = _KERNEL[3]
    slots = list(leaf_sums.unbind(dim=1))
    for dst, lhs, rhs in combine_order(g_leaves):
        slots[dst] = slots[lhs] + slots[rhs]
    log_s = torch.log(slots[0])
    out = (x_wide - row_max.unsqueeze(1)) - log_s.unsqueeze(1)
    return out, row_max, leaf_sums


def _full_row_fused(x_wide: torch.Tensor, g_leaves: int) -> torch.Tensor:
    """The steady-state fast path: one kernel, 3 reads + 1 write of [N, V]."""
    full_kernel = _KERNEL[4]
    rows, shard_vocab = x_wide.shape
    out = torch.empty_like(x_wide)
    full_kernel[(rows,)](
        out,
        x_wide,
        x_wide.stride(0),
        out.stride(0),
        L=shard_vocab // g_leaves,
        G=g_leaves,
        LOG2G=g_leaves.bit_length() - 1,
        BLOCK=BLOCK,
        num_warps=NUM_WARPS,
    )
    return out


def rowinv_full_logprobs(
    logits: torch.Tensor,
    *,
    src_dtype: torch.dtype,
    reference: Callable[[], torch.Tensor],
) -> torch.Tensor | None:
    """Full-row ``[N, V]`` fp32 logprobs under the same leaf-tree denominator as the sampled entry.

    The engine's V1 sampler site needs the whole row (top-k, ranks, prompt logprobs), not just the
    sampled value. Returns ``(x - m) - log(S)`` with ``m`` and ``S`` computed exactly as the
    sampled entry computes them, so the row is row-count invariant throughout. world=1 only (the
    row arrives already gathered); admission and census are shared with the sampled entry.
    Returns ``None`` to retain the incumbent. Inference-only: no autograd.
    """
    if _FULL_ROW["latched_off"]:
        _S["calls"] += 1
        _decline(f"full-row entry latched off: {_FULL_ROW['latch_reason']}")
        return None
    if logits.ndim != 2 or logits.numel() == 0:
        _S["calls"] += 1
        _decline(f"full-row entry requires a non-empty [N, V] matrix, got shape={tuple(logits.shape)}")
        return None
    rows, shard_vocab = logits.shape
    admit_target = _full_target(rows, logits.device)
    admitted, _reason = _admit(logits, admit_target, 0, shard_vocab, None, src_dtype)
    if admitted is None:
        return None
    _world, _rank, g_leaves = admitted  # group=None => world == 1, all leaves local

    with torch.no_grad():
        # Same transient exact widen as the sampled entry: kernel input dtype is pinned fp32.
        x_wide = logits if logits.dtype is torch.float32 else logits.to(torch.float32)

        use_fused = _FULL_ROW["fused"] is not False and len(_KERNEL) > 4
        should_probe = (
            _FULL_ROW["agreed"] is None
            or (_FULL_ROW["served"] > 0 and _FULL_ROW["served"] % PROBE_EVERY == 0)
            or (use_fused and rows not in _FULL_ROW["shapes"])
        )

        out = _full_row_fused(x_wide, g_leaves) if use_fused else None
        if should_probe or out is None:
            eager, row_max, leaf_sums = _full_row_eager(x_wide, admit_target, g_leaves)
            if out is not None and not torch.equal(out.view(torch.int32), eager.view(torch.int32)):
                # The fused kernel drifted from the letter: demote the optimization, keep the bits.
                _FULL_ROW["fused"] = False
                _FULL_ROW["fused_reason"] = f"fused full-row kernel != eager letter at rows={rows}"
                print(
                    f"{BANNER} full-row FUSED kernel demoted (not bit-identical to the eager "
                    f"combine_order composition at rows={rows}); the eager path keeps serving, "
                    "bits unchanged.",
                    flush=True,
                )
                out = eager
            elif out is not None:
                _FULL_ROW["fused"] = True
            else:
                out = eager
            _FULL_ROW["shapes"].add(rows)

            if _FULL_ROW["agreed"] is None or (_FULL_ROW["served"] > 0 and _FULL_ROW["served"] % PROBE_EVERY == 0):
                incumbent = reference()
                ok, why = _probe_full_row(out, incumbent, x_wide, row_max, leaf_sums, g_leaves)
                if not ok:
                    _FULL_ROW["agreed"] = False
                    _FULL_ROW["latched_off"] = True
                    _FULL_ROW["latch_reason"] = why
                    _S["agreed"] = False
                    _decline(f"full-row probe failed: {why}")
                    return incumbent
                _FULL_ROW["agreed"] = True
                if _S["agreed"] is None:
                    _S["agreed"] = True

    _S["served"] += 1
    _S["tokens"] += rows
    _S["full_rows"] += rows
    _FULL_ROW["served"] += 1
    _report()
    return out
