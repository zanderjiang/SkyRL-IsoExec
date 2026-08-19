"""Exact gather-to-owner vocabulary pipeline for scoring-only sampled logprobs.

This is an execution twin of the replicated full-vocabulary gather.  Each widened
bf16/fp16 shard is narrowed losslessly, moved to TP rank zero in rank order, and
widened into the same contiguous FP32 layout the incumbent constructs.  The caller
executes the incumbent reduction schedule on that owner and the sampled FP32 result
is broadcast.  No floating-point reduction is distributed or reassociated.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import torch
import torch.distributed as dist

from . import exact_vocab_transport

ENV = "SKYRL_ISOEXEC_EXACT_VOCAB_PIPELINE"
BANNER = "[ISOEXEC-EXACT-VOCAB-PIPELINE]"

_GROUPS: dict[int, dict] = {}
_S = {
    "calls": 0,
    "served": 0,
    "declined": 0,
    "threshold_declined": 0,
    "admission_votes": 0,
    "bytes_moved": 0,
    "reported": 0,
}


def stats() -> dict:
    result = dict(_S)
    result["transport"] = exact_vocab_transport.stats()
    return result


def _reset_for_test() -> None:
    reset_for_teardown()


def _reset_census() -> None:
    _S.update(
        {
            "calls": 0,
            "served": 0,
            "declined": 0,
            "threshold_declined": 0,
            "admission_votes": 0,
            "bytes_moved": 0,
            "reported": 0,
        }
    )


def release_group(group) -> None:
    """Release state before destroying and recreating one TP process group."""

    _GROUPS.pop(id(group), None)
    exact_vocab_transport.release_group(group)


def reset_for_teardown() -> None:
    """Release all process-group-bound state at a distributed lifecycle boundary."""

    _GROUPS.clear()
    _reset_census()
    exact_vocab_transport.reset_for_teardown()


def _report() -> None:
    calls = _S["calls"]
    if calls < 1 or calls == _S["reported"] or calls & (calls - 1):
        return
    _S["reported"] = calls
    print(
        f"{BANNER} CENSUS pid={os.getpid()} calls={calls} served={_S['served']} "
        f"declined={_S['declined']} threshold_declined={_S['threshold_declined']} "
        f"admission_votes={_S['admission_votes']} "
        f"moved={_S['bytes_moved'] / 1e9:.3f} GB",
        flush=True,
    )


def _global_rank(group, group_rank: int) -> int:
    if group is None or group is dist.group.WORLD:
        return group_rank
    getter = getattr(dist, "get_global_rank", None)
    if getter is None:
        raise RuntimeError(f"{BANNER} torch.distributed.get_global_rank is required for TP subgroups")
    return int(getter(group, group_rank))


@torch.no_grad()
def maybe_exact_vocab_pipeline(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    group,
    world: int,
    src_dtype: torch.dtype,
    finalize: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor | None:
    """Return exact sampled values or ``None`` for the replicated incumbent.

    All TP ranks call this after the outer exact-sampled admission.  A scalar MIN
    vote precedes either collective sequence, so asymmetric enablement or local
    eligibility can only select the common incumbent path.
    """

    _S["calls"] += 1
    rank = dist.get_rank(group=group)
    on = os.environ.get(ENV, "0") == "1"
    base_reason = ""
    src_element_size = torch.empty((), dtype=src_dtype).element_size()
    if not on:
        base_reason = f"{ENV}!=1"
    elif logits.dtype is not torch.float32 or src_dtype not in (torch.bfloat16, torch.float16):
        base_reason = f"requires widened bf16/fp16 FP32 logits, got {logits.dtype}/{src_dtype}"
    elif logits.ndim != 3 or target.shape != logits.shape[:-1] or logits.stride(-1) != 1:
        base_reason = f"unsupported layout logits={tuple(logits.shape)} target={tuple(target.shape)}"
    elif world != dist.get_world_size(group=group) or world <= 1:
        base_reason = f"invalid TP world={world}"

    contract = (
        on,
        logits.device.type,
        logits.device.index,
        logits.dtype,
        src_dtype,
        world,
        bool(base_reason),
    )
    state = _GROUPS.get(id(group))
    if state is None:
        # Do not retain the ProcessGroup itself: the transport weakly owns its lifecycle identity
        # and drops it when Megatron destroys the group. This cache holds only the contract.
        state = {"contract": contract, "dispatch_verdict": None}
        _GROUPS[id(group)] = state
    elif state["contract"] != contract:
        raise RuntimeError(f"{BANNER} STRUCTURAL DRIFT first={state['contract']!r} now={contract!r}")

    # The base execution contract is lifecycle-invariant, so vote once and reuse the unanimous
    # result: repeating a collective and an `item()` per vocabulary chunk would serialize the stream
    # thousands of times per step. Threshold dispatch stays a per-shape decision below.
    if state["dispatch_verdict"] is None:
        vote = torch.tensor(int(not base_reason), dtype=torch.int32, device=logits.device)
        dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=group)
        state["dispatch_verdict"] = bool(vote.item())
        _S["admission_votes"] += 1
    if not state["dispatch_verdict"]:
        _S["declined"] += 1
        _report()
        return None

    if not exact_vocab_transport.use_owner_gather(logits.numel(), src_element_size, world):
        _S["declined"] += 1
        _S["threshold_declined"] += 1
        _report()
        return None

    wire = logits.to(src_dtype).contiguous()
    owner = 0
    owner_global = _global_rank(group, owner)
    full = exact_vocab_transport.gather_rank_ordered_fp32(wire, group=group, world=world, owner=owner)
    if rank == owner:
        if full is None:
            raise RuntimeError(f"{BANNER} owner transport returned no gathered vocabulary")
        result = finalize(full, target)
    else:
        result = torch.empty(target.shape, dtype=torch.float32, device=logits.device)
    dist.broadcast(result, src=owner_global, group=group)

    _S["served"] += 1
    _S["bytes_moved"] += (world - 1) * wire.numel() * wire.element_size() + (
        world - 1
    ) * result.numel() * result.element_size()
    _report()
    return result


__all__ = ["ENV", "maybe_exact_vocab_pipeline", "release_group", "reset_for_teardown", "stats"]
