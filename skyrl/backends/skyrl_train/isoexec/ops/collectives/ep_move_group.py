"""Dedicated wide NCCL transport groups for expert all-to-all byte movement.

The Megatron EP process group also carries arithmetic (an int32 MIN admission vote and drift probes), so it
must keep the process-wide one-channel/tree schedule. When admitted, this module builds a second group with
the exact same ordered membership, routes only ``all_to_all_single`` onto it, and refuses any reduction
presented to the duplicate. With no admitted movement plan it creates no group and installs no wrapper.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist

from .nccl_channel_budget import MOVE_ONLY, GroupPlan

_GROUP_GETTERS = {"ep": "get_expert_model_parallel_group"}
_REDUCE_OPERATIONS = {
    "all_reduce": 2,
    "all_reduce_coalesced": 2,
    "reduce": 3,
    "reduce_scatter_tensor": 3,
    "_reduce_scatter_base": 3,
    "reduce_scatter": 3,
}


@dataclass(frozen=True)
class MovementGroup:
    """One logical pinned group and its same-membership wide transport twin."""

    name: str
    source: object
    transport: object
    ranks: tuple[int, ...]
    channels: int
    creation_nontorch_mib: float


_GROUPS: dict[str, MovementGroup] = {}
_SOURCE_ROUTES: dict[int, MovementGroup] = {}
_TRANSPORT_NAMES: dict[int, str] = {}
_COUNTS: dict[tuple[str, str], int] = {}
_ROUTER_INSTALLED = False
_ORIGINAL_DIST: dict[str, object] = {}
_WRAPPED_DIST: dict[str, object] = {}
_ORIGINAL_MPU_DESTROY = None
_WRAPPED_MPU_DESTROY = None


def validate_membership_reports(reports: list[dict], world_size: int) -> dict[str, tuple[tuple[int, ...], ...]]:
    """Validate and canonicalize rank-local logical-group membership reports."""
    if len(reports) != world_size:
        raise RuntimeError(f"membership report count {len(reports)} != WORLD {world_size}")
    errors = [report.get("error", "") for report in reports if report.get("error")]
    report_ranks = [int(report.get("rank", -1)) for report in reports]
    if sorted(report_ranks) != list(range(world_size)):
        errors.append(f"report ranks must be exactly WORLD once each, got {sorted(report_ranks)!r}")
    signatures = {tuple(report.get("plans", ())) for report in reports}
    if len(signatures) != 1:
        errors.append(f"rank-nonunanimous movement plans: {sorted(map(repr, signatures))!r}")
    canonical: dict[str, tuple[tuple[int, ...], ...]] = {}
    if not errors:
        for name, _channels, expected_world in next(iter(signatures)):
            by_rank = {int(report["rank"]): tuple(report["memberships"][name]) for report in reports}
            groups = tuple(sorted(set(by_rank.values())))
            flat = [rank for group in groups for rank in group]
            if any(len(group) != expected_world for group in groups):
                errors.append(f"{name}: membership width != planned world {expected_world}: {groups!r}")
            if sorted(flat) != list(range(world_size)) or len(flat) != len(set(flat)):
                errors.append(f"{name}: groups are not a disjoint WORLD partition: {groups!r}")
            for rank, members in by_rank.items():
                if rank not in members:
                    errors.append(f"{name}: rank {rank} reported non-owning membership {members!r}")
            canonical[name] = groups
    if errors:
        raise RuntimeError("[ISOEXEC-NCCL-MOVE] membership refusal: " + "; ".join(errors))
    return canonical


def _nontorch_mib() -> float:
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return (total - free - torch.cuda.memory_reserved()) / (1024**2)


def _readback_max_ctas(group) -> int | None:
    candidates = (group, getattr(group, "_get_backend", lambda _device: None)(torch.device("cuda")))
    for candidate in candidates:
        config = getattr(getattr(candidate, "options", None), "config", None)
        value = getattr(config, "max_ctas", None)
        if value is not None:
            return int(value)
    return None


def initialize_movement_groups(
    plans: tuple[GroupPlan, ...], control_gather: Callable[[str, object], list[object]]
) -> None:
    """Create same-membership duplicates after Megatron has constructed the logical groups."""
    move_plans = tuple(plan for plan in plans if plan.contract == MOVE_ONLY)
    if _GROUPS:
        raise RuntimeError(
            "[ISOEXEC-NCCL-MOVE] movement groups survived into a second setup lifecycle; "
            "destroy_model_parallel teardown was not run, so reusing stale process groups is refused"
        )
    if not move_plans:
        return
    if os.environ.get("NCCL_MAX_NCHANNELS", "").strip() != "1":
        raise RuntimeError(
            "[ISOEXEC-NCCL-MOVE] duplicate transport requires NCCL_MAX_NCHANNELS=1 so the "
            "logical/control EP communicator remains pinned"
        )

    from megatron.core import parallel_state as mpu

    rank = dist.get_rank()
    local_sources: dict[str, object] = {}
    memberships: dict[str, tuple[int, ...]] = {}
    local_error = ""
    try:
        for plan in move_plans:
            getter = _GROUP_GETTERS.get(plan.name)
            if getter is None:
                raise RuntimeError(f"no explicit movement owner for group {plan.name!r}")
            source = getattr(mpu, getter)()
            ranks = tuple(dist.get_process_group_ranks(source))
            if len(ranks) != plan.world:
                raise RuntimeError(
                    f"{plan.name}: source membership {ranks!r} has world {len(ranks)}, expected {plan.world}"
                )
            local_sources[plan.name] = source
            memberships[plan.name] = ranks
    except Exception as exc:  # noqa: BLE001 - every rank reports before any duplicate PG exists
        local_error = f"rank={rank} {type(exc).__name__}: {exc}"

    plan_signature = tuple((plan.name, plan.channels, plan.world) for plan in move_plans)
    reports = control_gather(
        "nccl-ep-move-membership-v1",
        {"rank": rank, "plans": plan_signature, "memberships": memberships, "error": local_error},
    )
    canonical = validate_membership_reports(reports, dist.get_world_size())

    for plan in move_plans:
        opts = dist.ProcessGroupNCCL.Options()
        opts.config.max_ctas = plan.channels
        opts.config.min_ctas = plan.channels
        local_transport = None
        before = _nontorch_mib()
        for ranks in canonical[plan.name]:
            group = dist.new_group(
                ranks=list(ranks),
                pg_options=opts,
                use_local_synchronization=False,
                group_desc=f"ISOEXEC_{plan.name.upper()}_MOVE",
            )
            if rank in ranks:
                if tuple(ranks) != memberships[plan.name]:
                    raise RuntimeError(
                        f"{plan.name}: created local membership {ranks!r} != source {memberships[plan.name]!r}"
                    )
                local_transport = group
        creation_mib = _nontorch_mib() - before
        if local_transport is None:
            raise RuntimeError(f"{plan.name}: no movement group created for rank {rank}")
        entry = MovementGroup(
            name=plan.name,
            source=local_sources[plan.name],
            transport=local_transport,
            ranks=memberships[plan.name],
            channels=plan.channels,
            creation_nontorch_mib=creation_mib,
        )
        _GROUPS[plan.name] = entry
        _SOURCE_ROUTES[id(entry.source)] = entry
        _TRANSPORT_NAMES[id(entry.transport)] = entry.name
        print(
            f"[ISOEXEC-NCCL-MOVE] CREATED group={plan.name} ranks={entry.ranks} "
            f"logical_channels=process-pin-1 transport_channels={plan.channels} "
            f"creation_nontorch={creation_mib / 1024:.3f}GiB; routing not armed until postflight",
            flush=True,
        )
    _install_destroy_hook(mpu)


def movement_groups_for_prewarm() -> tuple[MovementGroup, ...]:
    """Movement groups created locally, in deterministic plan order."""
    return tuple(_GROUPS.values())


def get_transport_group(source_group):
    """Return the transport twin for a logical group, or ``None`` when not admitted."""
    entry = _SOURCE_ROUTES.get(id(source_group))
    return None if entry is None else entry.transport


def is_movement_source(group) -> bool:
    """Whether this logical group has an A2A-only movement twin."""
    return id(group) in _SOURCE_ROUTES


def transport_evidence(name: str, prewarm_report: dict | None) -> tuple[int | None, int, float | None]:
    """Return ``(channel readback, served, full non-torch GiB)`` for one duplicate."""
    entry = _GROUPS.get(name)
    if entry is None:
        return None, 0, None
    label = f"isoexec_move:{name}"
    rows = [] if not prewarm_report else [row for row in prewarm_report.get("nontorch_groups", ()) if row[0] == label]
    if len(rows) != 1:
        return _readback_max_ctas(entry.transport), 0, None
    return _readback_max_ctas(entry.transport), 1, float(rows[0][1]) / 1024


def _group_from_call(args, kwargs, position: int):
    if "group" in kwargs:
        return kwargs["group"]
    return args[position] if len(args) > position else None


def _replace_group(args, kwargs, position: int, group):
    if "group" in kwargs:
        changed = dict(kwargs)
        changed["group"] = group
        return args, changed
    changed = list(args)
    while len(changed) <= position:
        changed.append(None)
    changed[position] = group
    return tuple(changed), kwargs


def _note(group_name: str, operation: str, *, control: bool = False) -> None:
    key = (group_name, operation)
    count = _COUNTS.get(key, 0) + 1
    _COUNTS[key] = count
    if count == 1 or count & (count - 1) == 0:
        role = "CONTROL" if control else "MOVE"
        suffix = " logical_pin=1" if control else " arithmetic_declined=0"
        print(
            f"[ISOEXEC-NCCL-{role}] CENSUS group={group_name} op={operation} served={count}{suffix}",
            flush=True,
        )


def install_router() -> None:
    """Route logical EP A2A to the duplicate and arm reduction refusal on the duplicate."""
    global _ROUTER_INSTALLED
    if not _GROUPS or _ROUTER_INSTALLED:
        return

    original_a2a = dist.all_to_all_single
    _ORIGINAL_DIST["all_to_all_single"] = original_a2a

    @functools.wraps(original_a2a)
    def routed_a2a(*args, **kwargs):
        supplied = _group_from_call(args, kwargs, 4)
        entry = _SOURCE_ROUTES.get(id(supplied))
        if entry is not None:
            args, kwargs = _replace_group(args, kwargs, 4, entry.transport)
            _note(entry.name, "all_to_all_single")
        elif id(supplied) in _TRANSPORT_NAMES:
            _note(_TRANSPORT_NAMES[id(supplied)], "all_to_all_single")
        return original_a2a(*args, **kwargs)

    dist.all_to_all_single = routed_a2a
    _WRAPPED_DIST["all_to_all_single"] = routed_a2a

    for operation, group_position in _REDUCE_OPERATIONS.items():
        original = getattr(dist, operation, None)
        if original is None:
            continue

        def make_guard(fn, name: str, position: int):
            @functools.wraps(fn)
            def guarded(*args, **kwargs):
                supplied = _group_from_call(args, kwargs, position)
                transport_name = _TRANSPORT_NAMES.get(id(supplied))
                if transport_name is not None:
                    print(
                        f"[ISOEXEC-NCCL-MOVE] DRIFT group={transport_name} op={name} "
                        "arithmetic_declined=1 -- REFUSING TO CONTINUE",
                        flush=True,
                    )
                    raise RuntimeError(
                        f"move-only NCCL group {transport_name!r} received arithmetic collective {name!r}"
                    )
                source = _SOURCE_ROUTES.get(id(supplied))
                if source is not None:
                    _note(source.name, name, control=True)
                return fn(*args, **kwargs)

            return guarded

        setattr(dist, operation, make_guard(original, operation, group_position))
        _ORIGINAL_DIST[operation] = original
        _WRAPPED_DIST[operation] = getattr(dist, operation)

    _ROUTER_INSTALLED = True
    print(
        f"[ISOEXEC-NCCL-MOVE] ROUTER ARMED groups={sorted(_GROUPS)}: only all_to_all_single "
        "is redirected; logical-group reductions remain pinned and duplicate reductions fail closed",
        flush=True,
    )


def _install_destroy_hook(mpu) -> None:
    """Tie duplicate PG and wrapper lifetime to Megatron's model-parallel lifecycle."""
    global _ORIGINAL_MPU_DESTROY, _WRAPPED_MPU_DESTROY
    if _WRAPPED_MPU_DESTROY is not None:
        raise RuntimeError("[ISOEXEC-NCCL-MOVE] destroy_model_parallel hook already installed")
    original = mpu.destroy_model_parallel

    @functools.wraps(original)
    def destroy_with_movement_cleanup(*args, **kwargs):
        teardown_movement_groups()
        return original(*args, **kwargs)

    _ORIGINAL_MPU_DESTROY = original
    _WRAPPED_MPU_DESTROY = destroy_with_movement_cleanup
    mpu.destroy_model_parallel = destroy_with_movement_cleanup


def teardown_movement_groups() -> None:
    """Restore torch APIs, destroy duplicate PGs, and clear all lifecycle state."""
    global _ROUTER_INSTALLED, _ORIGINAL_MPU_DESTROY, _WRAPPED_MPU_DESTROY
    mpu = None
    if _WRAPPED_MPU_DESTROY is not None:
        from megatron.core import parallel_state as mpu

        current_destroy = mpu.destroy_model_parallel
        if current_destroy is not _WRAPPED_MPU_DESTROY:
            raise RuntimeError("[ISOEXEC-NCCL-MOVE] destroy_model_parallel hook ownership drifted")
    for operation, wrapper in tuple(_WRAPPED_DIST.items()):
        current = getattr(dist, operation, None)
        if current is not wrapper:
            raise RuntimeError(
                f"[ISOEXEC-NCCL-MOVE] cannot safely uninstall {operation}: runtime wrapper ownership drifted"
            )

    # Mutate nothing until every ownership check passes. A refusal retains the cleanup hook and
    # all maps so a caller can remove the conflicting wrapper and retry safely.
    if mpu is not None:
        mpu.destroy_model_parallel = _ORIGINAL_MPU_DESTROY
    for operation, original in _ORIGINAL_DIST.items():
        setattr(dist, operation, original)

    # Every rank owns exactly one duplicate for each named plan and destroys the same local
    # handles in plan order. Megatron itself only drops references to its NCCL groups; ours are
    # independent and must not survive into a reinitialization.
    for entry in tuple(_GROUPS.values()):
        dist.destroy_process_group(entry.transport)

    _GROUPS.clear()
    _SOURCE_ROUTES.clear()
    _TRANSPORT_NAMES.clear()
    _COUNTS.clear()
    _ORIGINAL_DIST.clear()
    _WRAPPED_DIST.clear()
    _ROUTER_INSTALLED = False
    _ORIGINAL_MPU_DESTROY = None
    _WRAPPED_MPU_DESTROY = None

    from . import ep_nccl_channels

    ep_nccl_channels.reset_channel_plan_state()
    print("[ISOEXEC-NCCL-MOVE] TEARDOWN complete; duplicate PGs and routing state cleared", flush=True)
