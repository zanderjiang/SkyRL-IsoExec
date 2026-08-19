"""Batched TP transport for the exact-vocabulary scoring pipeline.

The pipeline needs every low-precision vocabulary shard on one TP rank in rank order. Point-to-point
sends satisfy that byte contract but make ProcessGroupNCCL create peer communicators lazily, and
those are invisible to PyTorch's allocator while consuming the headroom the colocated inference
engine needs. This expresses the same movement as one uneven ``all_to_all_single`` on the existing
TP group, so every rank enters one collective in the same order.

NCCL is byte transport only: the owner copies the rank-major wire into the same rank-ordered FP32
layout the incumbent all-gather produces, and no floating-point reduction is moved or reassociated.

The first accepted collective per process-group lifecycle records device memory around the lazy
transport, since ``mem_get_info`` sees NCCL allocations that allocator peaks cannot. Telemetry only,
not a budget gate.
"""

from __future__ import annotations

import os
import threading
import weakref
from dataclasses import asdict, dataclass
from typing import Callable

import torch
import torch.distributed as dist

BANNER = "[ISOEXEC-EXACT-VOCAB-TRANSPORT]"
# Below this the batched transport loses to the incumbent. Keyed to the RECONSTRUCTED full wire
# rather than the local shard, so the same model and row shape decide the same way at every TP.
MIN_FULL_WIRE_BYTES = 64 * 1024**2


@dataclass(frozen=True)
class MemorySnapshot:
    device_free_mib: float
    device_total_mib: float
    device_used_mib: float
    torch_allocated_mib: float
    torch_reserved_mib: float
    absolute_nontorch_mib: float


@dataclass(frozen=True)
class OwnerGatherPlan:
    """Element-count split plan for one gather-to-owner A2A."""

    rank: int
    world: int
    owner: int
    shard_numel: int
    input_split_sizes: tuple[int, ...]
    output_split_sizes: tuple[int, ...]
    output_numel: int


@dataclass
class _GroupState:
    group_ref: Callable[[], object | None]
    members: tuple[int, ...]
    device_type: str
    device_index: int | None
    wire_dtype: torch.dtype
    calls: int = 0
    served: int = 0
    wire_bytes: int = 0
    direct_p2p_calls: int = 0
    lazy_memory_samples: int = 0
    memory_before: MemorySnapshot | None = None
    memory_after: MemorySnapshot | None = None
    reported: int = 0


_LOCK = threading.Lock()
_GROUPS: dict[int, _GroupState] = {}
_TOTAL = {
    "calls": 0,
    "served": 0,
    "wire_bytes": 0,
    "direct_p2p_calls": 0,
    "lazy_memory_samples": 0,
}


def owner_gather_plan(rank: int, world: int, owner: int, shard_numel: int) -> OwnerGatherPlan:
    """Build the all-ranks symmetric split contract without touching c10d."""

    if world <= 1:
        raise ValueError(f"world must be >1, got {world}")
    if not 0 <= rank < world or not 0 <= owner < world:
        raise ValueError(f"invalid rank/owner rank={rank} owner={owner} world={world}")
    if shard_numel < 0:
        raise ValueError(f"shard_numel must be non-negative, got {shard_numel}")
    input_splits = [0] * world
    input_splits[owner] = shard_numel
    output_splits = [shard_numel] * world if rank == owner else [0] * world
    return OwnerGatherPlan(
        rank=rank,
        world=world,
        owner=owner,
        shard_numel=shard_numel,
        input_split_sizes=tuple(input_splits),
        output_split_sizes=tuple(output_splits),
        output_numel=world * shard_numel if rank == owner else 0,
    )


def full_wire_bytes(shard_numel: int, element_size: int, world: int) -> int:
    if shard_numel < 0 or element_size < 1 or world < 1:
        raise ValueError(
            f"invalid full-wire size inputs shard_numel={shard_numel} " f"element_size={element_size} world={world}"
        )
    return shard_numel * element_size * world


def use_owner_gather(shard_numel: int, element_size: int, world: int) -> bool:
    """Canonical, TP-invariant size dispatch for the owner-gather transport."""

    return full_wire_bytes(shard_numel, element_size, world) >= MIN_FULL_WIRE_BYTES


def _members(group, world: int) -> tuple[int, ...]:
    getter = getattr(dist, "get_process_group_ranks", None)
    if getter is not None:
        return tuple(int(rank) for rank in getter(group))
    # Older torch exposes global-rank translation but not the full member list.
    global_rank = getattr(dist, "get_global_rank", None)
    if group is None or group is dist.group.WORLD:
        return tuple(range(world))
    if global_rank is None:
        raise RuntimeError(f"{BANNER} cannot identify TP process-group membership")
    return tuple(int(global_rank(group, rank)) for rank in range(world))


def _weak_group_ref(group, key: int) -> Callable[[], object | None]:
    if group is None:
        world = dist.group.WORLD
        return lambda: world
    try:

        def group_gone(ref) -> None:
            with _LOCK:
                state = _GROUPS.get(key)
                if state is not None and state.group_ref is ref:
                    _GROUPS.pop(key, None)

        return weakref.ref(group, group_gone)
    except TypeError:
        # Older pybind ProcessGroup builds are not weak-referenceable. Holding the object prevents
        # Python-id reuse; release_group()/reset_for_teardown() is the lifecycle boundary there.
        return lambda: group


def _memory_snapshot(device: torch.device) -> MemorySnapshot:
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    mib = float(1024**2)
    used = total - free
    return MemorySnapshot(
        device_free_mib=free / mib,
        device_total_mib=total / mib,
        device_used_mib=used / mib,
        torch_allocated_mib=allocated / mib,
        torch_reserved_mib=reserved / mib,
        absolute_nontorch_mib=(used - reserved) / mib,
    )


def _state(group, wire: torch.Tensor, world: int) -> _GroupState:
    key = id(dist.group.WORLD if group is None else group)
    current_group = dist.group.WORLD if group is None else group
    members = _members(group, world)
    contract = (members, wire.device.type, wire.device.index, wire.dtype)
    with _LOCK:
        state = _GROUPS.get(key)
        if state is not None and state.group_ref() is not current_group:
            # A weakly-held group was destroyed and Python reused its id.
            _GROUPS.pop(key, None)
            state = None
        if state is None:
            state = _GroupState(
                group_ref=_weak_group_ref(group, key),
                members=members,
                device_type=wire.device.type,
                device_index=wire.device.index,
                wire_dtype=wire.dtype,
            )
            _GROUPS[key] = state
        elif contract != (state.members, state.device_type, state.device_index, state.wire_dtype):
            raise RuntimeError(
                f"{BANNER} STRUCTURAL DRIFT first="
                f"{(state.members, state.device_type, state.device_index, state.wire_dtype)!r} "
                f"now={contract!r}"
            )
        state.calls += 1
        _TOTAL["calls"] += 1
    return state


def _memory_delta(before: MemorySnapshot | None, after: MemorySnapshot | None) -> dict[str, float | None]:
    if before is None or after is None:
        return {
            "incremental_nontorch_mib": None,
            "absolute_nontorch_before_mib": None,
            "absolute_nontorch_after_mib": None,
            "device_free_after_mib": None,
        }
    return {
        "incremental_nontorch_mib": after.absolute_nontorch_mib - before.absolute_nontorch_mib,
        "absolute_nontorch_before_mib": before.absolute_nontorch_mib,
        "absolute_nontorch_after_mib": after.absolute_nontorch_mib,
        "device_free_after_mib": after.device_free_mib,
    }


def stats(group=None) -> dict:
    """Return process totals and optional per-group post-lazy memory telemetry."""

    result = dict(_TOTAL)
    if group is None:
        result["groups"] = tuple(
            {
                "members": state.members,
                "calls": state.calls,
                "served": state.served,
                "wire_bytes": state.wire_bytes,
                "direct_p2p_calls": state.direct_p2p_calls,
                "lazy_memory_samples": state.lazy_memory_samples,
                "memory_before": asdict(state.memory_before) if state.memory_before else None,
                "memory_after": asdict(state.memory_after) if state.memory_after else None,
                **_memory_delta(state.memory_before, state.memory_after),
            }
            for state in sorted(_GROUPS.values(), key=lambda item: item.members)
        )
        return result
    key = id(dist.group.WORLD if group is None else group)
    state = _GROUPS.get(key)
    if state is None:
        return {}
    return {
        "members": state.members,
        "calls": state.calls,
        "served": state.served,
        "wire_bytes": state.wire_bytes,
        "direct_p2p_calls": state.direct_p2p_calls,
        "lazy_memory_samples": state.lazy_memory_samples,
        "memory_before": asdict(state.memory_before) if state.memory_before else None,
        "memory_after": asdict(state.memory_after) if state.memory_after else None,
        **_memory_delta(state.memory_before, state.memory_after),
    }


def _report(state: _GroupState) -> None:
    served = state.served
    if served < 1 or (served & (served - 1)) or served == state.reported:
        return
    state.reported = served
    memory = _memory_delta(state.memory_before, state.memory_after)
    lazy = memory["incremental_nontorch_mib"]
    absolute = memory["absolute_nontorch_after_mib"]
    lazy_text = "unavailable" if lazy is None else f"{lazy:.1f}MiB"
    absolute_text = "unavailable" if absolute is None else f"{absolute:.1f}MiB"
    print(
        f"{BANNER} CENSUS pid={os.getpid()} members={state.members} calls={state.calls} "
        f"served={served} a2a_served={served} direct_p2p_calls={state.direct_p2p_calls} "
        f"wire={state.wire_bytes / 1e9:.3f}GB lazy_nontorch={lazy_text} "
        f"absolute_nontorch_after={absolute_text}",
        flush=True,
    )


@torch.no_grad()
def gather_rank_ordered_fp32(
    wire: torch.Tensor,
    *,
    group,
    world: int,
    owner: int = 0,
    collective: Callable | None = None,
    memory_snapshot: Callable[[torch.device], MemorySnapshot] | None = None,
) -> torch.Tensor | None:
    """Gather equal low-precision shards to ``owner`` via one TP all-to-all.

    Returns the incumbent ``[..., TP * shard_vocab]`` FP32 layout on the owner and ``None``
    elsewhere. ``collective`` and ``memory_snapshot`` are CPU-test seams.
    """

    if wire.dtype not in (torch.bfloat16, torch.float16, torch.float32) or wire.ndim < 1 or wire.stride(-1) != 1:
        raise ValueError(f"{BANNER} requires contiguous-last-dim bf16/fp16/fp32 wire, got {wire.shape}/{wire.dtype}")
    wire = wire.contiguous()
    rank = int(dist.get_rank(group=group))
    if int(dist.get_world_size(group=group)) != world:
        raise RuntimeError(f"{BANNER} declared world={world} differs from process group")
    plan = owner_gather_plan(rank, world, owner, wire.numel())
    state = _state(group, wire, world)
    output = torch.empty(plan.output_numel, dtype=wire.dtype, device=wire.device)
    snapshot = _memory_snapshot if memory_snapshot is None else memory_snapshot
    sample_lazy = state.lazy_memory_samples == 0 and wire.is_cuda
    if sample_lazy:
        state.memory_before = snapshot(wire.device)
    fn = dist.all_to_all_single if collective is None else collective
    fn(
        output,
        wire.view(-1),
        output_split_sizes=list(plan.output_split_sizes),
        input_split_sizes=list(plan.input_split_sizes),
        group=group,
        async_op=False,
    )
    if sample_lazy:
        state.memory_after = snapshot(wire.device)
        state.lazy_memory_samples += 1
        _TOTAL["lazy_memory_samples"] += 1

    moved = (world - 1) * wire.numel() * wire.element_size()
    state.served += 1
    state.wire_bytes += moved
    _TOTAL["served"] += 1
    _TOTAL["wire_bytes"] += moved
    # Always zero: direct P2P is not a fallback, and a failed A2A raises before any alternative
    # collective could desynchronize peers.
    _TOTAL["direct_p2p_calls"] += state.direct_p2p_calls
    _report(state)
    if rank != owner:
        return None

    rank_major = output.view(world, *wire.shape)
    shard_vocab = wire.shape[-1]
    full = torch.empty((*wire.shape[:-1], world * shard_vocab), dtype=torch.float32, device=wire.device)
    for source in range(world):
        full[..., source * shard_vocab : (source + 1) * shard_vocab].copy_(rank_major[source])
    return full


def release_group(group) -> None:
    """Release all identity/telemetry state before destroying one process group."""

    key = id(dist.group.WORLD if group is None else group)
    with _LOCK:
        _GROUPS.pop(key, None)


def reset_for_teardown() -> None:
    """Release every process-group identity at a distributed lifecycle boundary."""

    with _LOCK:
        _GROUPS.clear()
        _TOTAL.update(
            {
                "calls": 0,
                "served": 0,
                "wire_bytes": 0,
                "direct_p2p_calls": 0,
                "lazy_memory_samples": 0,
            }
        )


def _reset_for_test() -> None:
    reset_for_teardown()


__all__ = [
    "MIN_FULL_WIRE_BYTES",
    "MemorySnapshot",
    "OwnerGatherPlan",
    "full_wire_bytes",
    "gather_rank_ordered_fp32",
    "owner_gather_plan",
    "release_group",
    "reset_for_teardown",
    "stats",
    "use_owner_gather",
]
