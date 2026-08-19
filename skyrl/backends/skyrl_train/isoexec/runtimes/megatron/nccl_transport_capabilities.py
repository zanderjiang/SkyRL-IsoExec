"""Declared NCCL transport ownership for selective trainer prewarming.

NCCL connects some transports lazily, so an all-to-all can materialize peer-to-peer connections a
reduction-only group never needs. An active owner declares the operations it may issue on a resolved
process group, prewarm warms the union of those declarations, and afterwards the c10d wrappers refuse
an undeclared operation; nothing is inferred from a model name or group label. Default-off and
import-inert until ``SKYRL_ISOEXEC_NCCL_CAPABILITY_MODE`` is ``census`` (record only) or ``enforce``
(fail-closed); this is process-local enforcement, while ``nccl_prewarm`` runs the WORLD-unanimous
Store vote before skipping any transport.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import pathlib
from dataclasses import dataclass, field
from typing import Callable, Iterable

import torch
import torch.distributed as dist

ENV = "SKYRL_ISOEXEC_NCCL_CAPABILITY_MODE"
OFF = "off"
CENSUS = "census"
ENFORCE = "enforce"
BOUNDARY_REQUIREMENTS_ENV = "SKYRL_ISOEXEC_NCCL_TRANSPORT_BOUNDARY_REQUIREMENTS"
MAX_POST_LAZY_GROWTH_ENV = "SKYRL_ISOEXEC_NCCL_MAX_POST_LAZY_GROWTH_MIB"
MAX_ABSOLUTE_NONTORCH_ENV = "SKYRL_ISOEXEC_NCCL_MAX_ABSOLUTE_NONTORCH_MIB"
MIN_DEVICE_FREE_ENV = "SKYRL_ISOEXEC_NCCL_MIN_DEVICE_FREE_MIB"

ALL_REDUCE = "all_reduce"
ALL_REDUCE_COALESCED = "all_reduce_coalesced"
REDUCE = "reduce"
REDUCE_SCATTER_TENSOR = "reduce_scatter_tensor"
REDUCE_SCATTER_BASE = "_reduce_scatter_base"
REDUCE_SCATTER = "reduce_scatter"
ALL_GATHER_INTO_TENSOR = "all_gather_into_tensor"
ALL_GATHER_BASE = "_all_gather_base"
ALL_GATHER = "all_gather"
ALL_TO_ALL_SINGLE = "all_to_all_single"
ALL_TO_ALL = "all_to_all"
BROADCAST = "broadcast"
BARRIER = "barrier"

OPERATIONS = frozenset(
    {
        ALL_REDUCE,
        ALL_REDUCE_COALESCED,
        REDUCE,
        REDUCE_SCATTER_TENSOR,
        REDUCE_SCATTER_BASE,
        REDUCE_SCATTER,
        ALL_GATHER_INTO_TENSOR,
        ALL_GATHER_BASE,
        ALL_GATHER,
        ALL_TO_ALL_SINGLE,
        ALL_TO_ALL,
        BROADCAST,
        BARRIER,
    }
)
P2P_OPERATIONS = frozenset({ALL_TO_ALL_SINGLE, ALL_TO_ALL})

# Position of ``group`` in the public torch.distributed function.  Keeping this table explicit
# makes signature drift fail visibly in tests instead of accidentally treating a subgroup call as
# WORLD traffic.
_GROUP_POSITION = {
    ALL_REDUCE: 2,
    ALL_REDUCE_COALESCED: 2,
    REDUCE: 3,
    REDUCE_SCATTER_TENSOR: 3,
    REDUCE_SCATTER_BASE: 3,
    REDUCE_SCATTER: 3,
    ALL_GATHER_INTO_TENSOR: 2,
    ALL_GATHER_BASE: 2,
    ALL_GATHER: 2,
    ALL_TO_ALL_SINGLE: 4,
    ALL_TO_ALL: 2,
    BROADCAST: 2,
    BARRIER: 0,
}


@dataclass
class GroupCapability:
    """Mutable process-local union for one physical c10d group."""

    group: object
    members: tuple[int, ...]
    aliases: set[str] = field(default_factory=set)
    owners: dict[str, frozenset[str]] = field(default_factory=dict)
    # Attempted calls are kept separate from calls which c10d accepted. The owner table credits
    # every explicit claimant when aliases intentionally share one physical group/operation;
    # c10d has no information from which to invent exclusive call-site attribution.
    counts: dict[str, int] = field(default_factory=dict)
    served_counts: dict[str, int] = field(default_factory=dict)
    claimant_served_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def operations(self) -> frozenset[str]:
        return frozenset(operation for operations in self.owners.values() for operation in operations)

    def signature(self) -> tuple:
        return (
            self.members,
            tuple(sorted(self.aliases)),
            tuple(sorted((owner, tuple(sorted(operations))) for owner, operations in self.owners.items())),
        )


_GROUPS: dict[int, GroupCapability] = {}
_ORIGINALS: dict[str, Callable] = {}
_WRAPPERS: dict[str, Callable] = {}
_INSTALLED = False
_ARMED = False
_MPU_MODULE = None
_ORIGINAL_MPU_DESTROY = None
_WRAPPED_MPU_DESTROY = None
_ARM_MEMORY: dict[str, float] | None = None
_SERVED_BASELINE: dict[int, dict[str, int]] = {}


def mode() -> str:
    value = os.environ.get(ENV, OFF).strip().lower() or OFF
    if value not in (OFF, CENSUS, ENFORCE):
        raise RuntimeError(f"[ISOEXEC-NCCL-CAP] invalid {ENV}={value!r}; expected off, census, or enforce")
    return value


def enabled() -> bool:
    return mode() != OFF


def register_group(owner: str, alias: str, group, operations: Iterable[str]) -> None:
    """Register one active owner's complete operation set on a resolved group.

    Registration is intentionally idempotent for repeated setup helpers, but the same owner may
    not silently change its declaration within one lifecycle.
    """

    if not enabled():
        return
    if not owner.strip() or not alias.strip():
        raise ValueError("owner and alias must be non-empty")
    declared = frozenset(str(operation) for operation in operations)
    unknown = declared - OPERATIONS
    if unknown:
        raise ValueError(f"invalid operation declaration unknown={sorted(unknown)!r}")
    entry = track_group(alias, group)
    prior = entry.owners.get(owner)
    if prior is not None and prior != declared:
        raise RuntimeError(
            f"[ISOEXEC-NCCL-CAP] owner {owner!r} changed declaration {sorted(prior)!r} -> {sorted(declared)!r}"
        )
    entry.aliases.add(alias)
    entry.owners[owner] = declared


def register_mpu_manifest(mpu, path: str | os.PathLike[str]) -> dict:
    """Resolve and register a reviewed active-owner manifest against this Megatron topology."""

    if not enabled():
        return {}
    manifest_path = pathlib.Path(path)
    raw = manifest_path.read_bytes()
    rows = json.loads(raw)
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("[ISOEXEC-NCCL-CAP] capabilities manifest must be a non-empty JSON list")
    registered = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"[ISOEXEC-NCCL-CAP] manifest row {index} is not an object: {row!r}")
        unknown_fields = set(row) - {"owner", "getter", "kwargs", "operations"}
        if unknown_fields:
            raise RuntimeError(f"[ISOEXEC-NCCL-CAP] manifest row {index} has unknown fields {sorted(unknown_fields)!r}")
        try:
            owner = str(row["owner"])
            getter = str(row["getter"])
            kwargs = dict(row.get("kwargs", {}))
            operations = tuple(str(operation) for operation in row["operations"])
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"[ISOEXEC-NCCL-CAP] invalid manifest row {index}: {row!r}") from exc
        if getter == "WORLD":
            group = dist.group.WORLD
        else:
            get_group = getattr(mpu, getter, None)
            if get_group is None or not getter.startswith("get_") or not getter.endswith("_group"):
                raise RuntimeError(f"[ISOEXEC-NCCL-CAP] row {index} has invalid group getter {getter!r}")
            if getter in ("get_embedding_group", "get_position_embedding_group"):
                kwargs = {"check_initialized": False, **kwargs}
            group = get_group(**kwargs)
        if group is not None and dist.get_world_size(group) >= 2:
            alias = f"{getter}:{json.dumps(kwargs, sort_keys=True, separators=(',', ':'))}"
            register_group(owner, alias, group, operations)
            registered += 1
    if registered < 1:
        raise RuntimeError("[ISOEXEC-NCCL-CAP] manifest resolved no active multi-rank process group")
    result = {
        "path": str(manifest_path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "rows": len(rows),
        "registered_rows": registered,
    }
    print(
        f"[ISOEXEC-NCCL-CAP] MANIFEST sha256={result['sha256']} rows={len(rows)} "
        f"registered_rows={registered} path={result['path']}",
        flush=True,
    )
    return result


def track_group(alias: str, group) -> GroupCapability:
    """Add a group to runtime census without claiming any operation ownership."""

    if not enabled():
        raise RuntimeError(f"[ISOEXEC-NCCL-CAP] cannot track groups while {ENV}=off")
    if not alias.strip():
        raise ValueError("alias must be non-empty")
    members = tuple(int(rank) for rank in dist.get_process_group_ranks(group))
    key = id(group)
    entry = _GROUPS.get(key)
    if entry is None:
        entry = GroupCapability(group=group, members=members)
        _GROUPS[key] = entry
    elif entry.group is not group or entry.members != members:
        raise RuntimeError(f"[ISOEXEC-NCCL-CAP] process-group identity collision for {alias!r}")
    entry.aliases.add(alias)
    return entry


def capability_for(group) -> GroupCapability | None:
    return _GROUPS.get(id(group))


def canonical_signature() -> tuple:
    """Stable local signature used by the Store-based WORLD vote."""

    return tuple(sorted(entry.signature() for entry in _GROUPS.values()))


def declaration_signature() -> tuple:
    """Rank-independent owner/alias signature for the WORLD configuration vote."""

    return tuple(
        sorted(
            (
                tuple(sorted(entry.aliases)),
                tuple(sorted((owner, tuple(sorted(operations))) for owner, operations in entry.owners.items())),
            )
            for entry in _GROUPS.values()
        )
    )


def _group_from_call(operation: str, args: tuple, kwargs: dict):
    if "group" in kwargs:
        return kwargs["group"]
    position = _GROUP_POSITION[operation]
    return args[position] if len(args) > position else None


def _group_label(entry: GroupCapability) -> str:
    return ",".join(sorted(entry.aliases)) or repr(entry.members)


def _record_or_refuse(operation: str, group) -> tuple[GroupCapability | None, tuple[str, ...]]:
    if not _ARMED:
        return None, ()
    entry = _GROUPS.get(id(dist.group.WORLD if group is None else group))
    active_mode = mode()
    if entry is None:
        if active_mode == ENFORCE:
            raise RuntimeError(
                f"[ISOEXEC-NCCL-CAP] undeclared process group attempted {operation}; "
                "register its active owner before capability postflight"
            )
        return None, ()
    count = entry.counts.get(operation, 0) + 1
    entry.counts[operation] = count
    if active_mode == ENFORCE and operation not in entry.operations:
        raise RuntimeError(
            f"[ISOEXEC-NCCL-CAP] group={_group_label(entry)} members={entry.members} attempted "
            f"undeclared op={operation}; declared={sorted(entry.operations)!r}"
        )
    if count == 1 or count & (count - 1) == 0:
        disposition = "DECLARED" if operation in entry.operations else "OBSERVED-UNDECLARED"
        print(
            f"[ISOEXEC-NCCL-CAP] CENSUS group={_group_label(entry)} op={operation} "
            f"count={count} disposition={disposition}",
            flush=True,
        )
    owners = tuple(sorted(owner for owner, operations in entry.owners.items() if operation in operations))
    return entry, owners


def _record_served(operation: str, entry: GroupCapability | None, owners: tuple[str, ...]) -> None:
    """Record a call only after the original c10d entry point has accepted it."""

    if entry is None:
        return
    entry.served_counts[operation] = entry.served_counts.get(operation, 0) + 1
    for owner in owners:
        owner_counts = entry.claimant_served_counts.setdefault(owner, {})
        owner_counts[operation] = owner_counts.get(operation, 0) + 1


def install_runtime_guard() -> None:
    """Wrap c10d entry points.  The wrappers are inert until :func:`arm` is called."""

    global _INSTALLED
    if not enabled() or _INSTALLED:
        return
    for operation in sorted(OPERATIONS):
        original = getattr(dist, operation, None)
        if original is None:
            continue
        _ORIGINALS[operation] = original

        @functools.wraps(original)
        def wrapped(*args, __operation=operation, __original=original, **kwargs):
            entry, owners = _record_or_refuse(__operation, _group_from_call(__operation, args, kwargs))
            result = __original(*args, **kwargs)
            _record_served(__operation, entry, owners)
            return result

        _WRAPPERS[operation] = wrapped
        setattr(dist, operation, wrapped)
    _INSTALLED = True


def arm() -> None:
    """Arm census/enforcement after unanimous prewarm postflight."""

    global _ARMED, _ARM_MEMORY, _SERVED_BASELINE
    if not enabled():
        raise RuntimeError(f"[ISOEXEC-NCCL-CAP] cannot arm while {ENV}=off")
    if not _GROUPS:
        raise RuntimeError("[ISOEXEC-NCCL-CAP] cannot arm an empty capability registry")
    install_runtime_guard()
    _ARM_MEMORY = _memory_snapshot()
    _SERVED_BASELINE = {id(entry.group): dict(entry.served_counts) for entry in _GROUPS.values()}
    _ARMED = True


def _store_control_gather(tag: str, value):
    """Gather a bounded runtime verdict without issuing another guarded NCCL operation."""

    from skyrl.backends.skyrl_train.distributed.store_rendezvous import store_all_gather
    from skyrl.env_vars import SKYRL_WORKER_NCCL_TIMEOUT_IN_S

    return store_all_gather(
        dist.distributed_c10d._get_default_store(),
        dist.get_rank(),
        dist.get_world_size(),
        tag,
        value,
        SKYRL_WORKER_NCCL_TIMEOUT_IN_S,
    )


def _memory_snapshot() -> dict[str, float]:
    """Device-wide headroom plus an explicit non-Torch residency estimate."""

    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    mib = float(1024**2)
    return {
        "device_free_mib": free / mib,
        "device_total_mib": total / mib,
        "torch_reserved_mib": torch.cuda.memory_reserved() / mib,
        "absolute_nontorch_mib": (total - free - torch.cuda.memory_reserved()) / mib,
    }


def _boundary_memory_budgets() -> dict[str, float]:
    budgets = {}
    # Spell each governed read explicitly. Besides making the three independent safety rails
    # reviewable, this keeps the static flag-contract census complete without teaching it to
    # infer arbitrary loop-variable values.
    raw_budgets = {
        MAX_POST_LAZY_GROWTH_ENV: os.environ.get(MAX_POST_LAZY_GROWTH_ENV, ""),
        MAX_ABSOLUTE_NONTORCH_ENV: os.environ.get(MAX_ABSOLUTE_NONTORCH_ENV, ""),
        MIN_DEVICE_FREE_ENV: os.environ.get(MIN_DEVICE_FREE_ENV, ""),
    }
    for env_name, raw_value in raw_budgets.items():
        raw = raw_value.strip()
        try:
            value = float(raw)
        except ValueError as exc:
            raise RuntimeError(f"{env_name} must be a finite non-negative MiB value, got {raw!r}") from exc
        if not raw or not math.isfinite(value) or value < 0:
            raise RuntimeError(f"{env_name} must be a finite non-negative MiB value, got {raw!r}")
        budgets[env_name] = value
    return budgets


def _boundary_requirements(boundary: str, path: str | os.PathLike[str]) -> tuple[str, tuple[tuple[str, str, int], ...]]:
    raw = pathlib.Path(path).read_bytes()
    document = json.loads(raw)
    if not isinstance(document, dict) or boundary not in document:
        raise RuntimeError(f"boundary requirements have no {boundary!r} entry")
    rows = document[boundary]
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"boundary {boundary!r} requirements must be a non-empty list")
    requirements = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) - {"group_alias", "operation", "min_served"}:
            raise RuntimeError(f"invalid boundary={boundary!r} requirement row={index}: {row!r}")
        group_alias = str(row.get("group_alias", "")).strip()
        operation = str(row.get("operation", "")).strip()
        minimum = row.get("min_served", 1)
        if not group_alias or operation not in OPERATIONS or not isinstance(minimum, int) or minimum < 1:
            raise RuntimeError(f"invalid boundary={boundary!r} requirement row={index}: {row!r}")
        requirements.append((group_alias, operation, minimum))
    canonical = tuple(sorted(requirements))
    if len(canonical) != len(set(canonical)):
        raise RuntimeError(f"boundary {boundary!r} contains duplicate physical-group/op/min requirements")
    return hashlib.sha256(raw).hexdigest(), canonical


def validate_transport_engagement(
    boundary: str,
    *,
    requirements_path: str | os.PathLike[str] | None = None,
    _control_gather=None,
) -> dict:
    """WORLD-unanimously require real post-arm physical-group traffic at a runtime boundary.

    Each requirement is a delta since the previous successful boundary (or since arm); every
    required physical group must report the same delta and cumulative count on exactly its declared
    member ranks. The validator uses the rendezvous Store, not a guarded NCCL collective, and the
    baseline advances only on WORLD-unanimous success, so a retry cannot hide missing traffic.
    """

    global _SERVED_BASELINE
    if not enabled():
        return {}
    if not _ARMED:
        raise RuntimeError(f"[ISOEXEC-NCCL-CAP] boundary={boundary!r} reached before runtime guard armed")
    if not boundary.strip():
        raise ValueError("boundary must be non-empty")
    path = requirements_path or os.environ.get(BOUNDARY_REQUIREMENTS_ENV, "").strip()
    if not path:
        raise RuntimeError(f"[ISOEXEC-NCCL-CAP] {BOUNDARY_REQUIREMENTS_ENV} is required for boundary={boundary!r}")
    gather = _store_control_gather if _control_gather is None else _control_gather
    rank = dist.get_rank()
    world = dist.get_world_size()
    config_error = ""
    try:
        manifest_sha256, requirements = _boundary_requirements(boundary, path)
        budgets = _boundary_memory_budgets()
    except Exception as exc:  # noqa: BLE001 - vote the parse failure before any rank proceeds
        manifest_sha256, requirements, budgets = "", (), {}
        config_error = f"rank={rank} {type(exc).__name__}: {exc}"
    config_reports = gather(
        f"nccl-transport-boundary-config-{boundary}",
        {
            "rank": rank,
            "phase": "config",
            "sha256": manifest_sha256,
            "requirements": requirements,
            "budgets": tuple(sorted(budgets.items())),
            "error": config_error,
        },
    )
    config_errors = [str(report.get("error")) for report in config_reports if report.get("error")]
    config_ranks = sorted(int(report.get("rank", -1)) for report in config_reports)
    config_signatures = {
        (
            str(report.get("sha256", "")),
            tuple(report.get("requirements", ())),
            tuple(report.get("budgets", ())),
        )
        for report in config_reports
    }
    if config_ranks != list(range(world)):
        config_errors.append(f"config ranks must be WORLD exactly once, got {config_ranks!r}")
    if len(config_signatures) != 1:
        config_errors.append(f"rank-nonunanimous boundary configuration {sorted(map(repr, config_signatures))!r}")
    if config_errors:
        raise RuntimeError(
            f"[ISOEXEC-NCCL-CAP] boundary={boundary!r} config unanimous refusal: " + "; ".join(config_errors)
        )

    local_rows = []
    for group_alias, operation, minimum in requirements:
        for entry in _GROUPS.values():
            if group_alias not in entry.aliases:
                continue
            local_rows.append(
                {
                    "group_alias": group_alias,
                    "operation": operation,
                    "min_served": minimum,
                    "members": entry.members,
                    "aliases": tuple(sorted(entry.aliases)),
                    "cumulative_served": entry.served_counts.get(operation, 0),
                    "interval_delta": entry.served_counts.get(operation, 0)
                    - _SERVED_BASELINE.get(id(entry.group), {}).get(operation, 0),
                }
            )
    memory_error = ""
    memory = {}
    try:
        if _ARM_MEMORY is None:
            raise RuntimeError("arm memory baseline is missing")
        memory = _memory_snapshot()
        memory["arm_absolute_nontorch_mib"] = _ARM_MEMORY["absolute_nontorch_mib"]
        memory["post_lazy_growth_mib"] = memory["absolute_nontorch_mib"] - _ARM_MEMORY["absolute_nontorch_mib"]
        if memory["post_lazy_growth_mib"] > budgets[MAX_POST_LAZY_GROWTH_ENV]:
            raise RuntimeError(
                f"post-lazy growth={memory['post_lazy_growth_mib']:.1f} MiB exceeds "
                f"{MAX_POST_LAZY_GROWTH_ENV}={budgets[MAX_POST_LAZY_GROWTH_ENV]:.1f}"
            )
        if memory["absolute_nontorch_mib"] > budgets[MAX_ABSOLUTE_NONTORCH_ENV]:
            raise RuntimeError(
                f"absolute non-Torch={memory['absolute_nontorch_mib']:.1f} MiB exceeds "
                f"{MAX_ABSOLUTE_NONTORCH_ENV}={budgets[MAX_ABSOLUTE_NONTORCH_ENV]:.1f}"
            )
        if memory["device_free_mib"] < budgets[MIN_DEVICE_FREE_ENV]:
            raise RuntimeError(
                f"device free={memory['device_free_mib']:.1f} MiB below "
                f"{MIN_DEVICE_FREE_ENV}={budgets[MIN_DEVICE_FREE_ENV]:.1f}"
            )
    except Exception as exc:  # noqa: BLE001 - make every rank refuse the local memory failure
        memory_error = f"rank={rank} {type(exc).__name__}: {exc}"
    reports = gather(
        f"nccl-transport-boundary-served-{boundary}",
        {"rank": rank, "phase": "served", "rows": local_rows, "memory": memory, "error": memory_error},
    )
    errors = [str(report.get("error")) for report in reports if report.get("error")]
    report_ranks = sorted(int(report.get("rank", -1)) for report in reports)
    if report_ranks != list(range(world)):
        errors.append(f"served ranks must be WORLD exactly once, got {report_ranks!r}")
    by_requirement: dict[tuple[str, str, int], dict[tuple, dict[int, tuple[int, int]]]] = {
        requirement: {} for requirement in requirements
    }
    for report in reports:
        reporting_rank = int(report.get("rank", -1))
        for row in report.get("rows", ()):
            requirement = (str(row["group_alias"]), str(row["operation"]), int(row["min_served"]))
            members = tuple(int(member) for member in row["members"])
            aliases = tuple(str(alias) for alias in row["aliases"])
            key = (members, aliases)
            by_requirement.setdefault(requirement, {}).setdefault(key, {})[reporting_rank] = (
                int(row["interval_delta"]),
                int(row["cumulative_served"]),
            )
    validated = []
    for requirement in requirements:
        group_alias, operation, minimum = requirement
        groups = by_requirement.get(requirement, {})
        if not groups:
            errors.append(f"group_alias={group_alias!r} operation={operation!r} resolved no active physical group")
            continue
        for (members, aliases), counts in groups.items():
            if tuple(sorted(counts)) != tuple(sorted(members)):
                errors.append(
                    f"group_alias={group_alias!r} operation={operation!r} aliases={aliases!r} "
                    f"reporters={sorted(counts)!r} != members={members!r}"
                )
                continue
            unique_counts = set(counts.values())
            if len(unique_counts) != 1:
                errors.append(
                    f"group_alias={group_alias!r} operation={operation!r} aliases={aliases!r} "
                    f"rank-nonunanimous interval/cumulative={counts!r}"
                )
                continue
            interval_delta, cumulative_served = next(iter(unique_counts))
            if interval_delta < minimum:
                errors.append(
                    f"group_alias={group_alias!r} operation={operation!r} aliases={aliases!r} "
                    f"interval_delta={interval_delta} < required={minimum} "
                    f"cumulative_served={cumulative_served}"
                )
            validated.append(
                {
                    "group_alias": group_alias,
                    "operation": operation,
                    "min_served": minimum,
                    "interval_delta": interval_delta,
                    "cumulative_served": cumulative_served,
                    "members": members,
                    "aliases": aliases,
                }
            )
    if errors:
        raise RuntimeError(f"[ISOEXEC-NCCL-CAP] boundary={boundary!r} served unanimous refusal: " + "; ".join(errors))
    # Transactional advance: take the full physical group/op snapshot only after every rank,
    # required group, interval count, and memory budget has passed. A refused call leaves the prior
    # baseline intact, so retrying cannot hide the missing phase traffic.
    _SERVED_BASELINE = {id(entry.group): dict(entry.served_counts) for entry in _GROUPS.values()}
    result = {
        "boundary": boundary,
        "requirements_sha256": manifest_sha256,
        "memory_budgets_mib": budgets,
        "memory_by_rank": tuple(report["memory"] for report in reports),
        "groups": tuple(sorted(validated, key=repr)),
    }
    if rank == 0:
        summary = ", ".join(
            f"{row['group_alias']}:{row['operation']} delta={row['interval_delta']} "
            f"cumulative={row['cumulative_served']} members={row['members']}"
            for row in result["groups"]
        )
        print(
            f"[ISOEXEC-NCCL-CAP] BOUNDARY_PASS boundary={boundary} "
            f"requirements_sha256={manifest_sha256} :: {summary}",
            flush=True,
        )
    return result


def install_teardown_hook(mpu) -> None:
    """Bind registry/wrapper lifetime to Megatron's process-group lifetime."""

    global _MPU_MODULE, _ORIGINAL_MPU_DESTROY, _WRAPPED_MPU_DESTROY
    if _WRAPPED_MPU_DESTROY is not None:
        if mpu.destroy_model_parallel is not _WRAPPED_MPU_DESTROY:
            raise RuntimeError("[ISOEXEC-NCCL-CAP] destroy_model_parallel hook ownership drifted")
        return
    original = mpu.destroy_model_parallel

    @functools.wraps(original)
    def destroy_with_capability_cleanup(*args, **kwargs):
        reset_for_teardown()
        return original(*args, **kwargs)

    _ORIGINAL_MPU_DESTROY = original
    _WRAPPED_MPU_DESTROY = destroy_with_capability_cleanup
    _MPU_MODULE = mpu
    mpu.destroy_model_parallel = destroy_with_capability_cleanup


def projected_residency_gib(
    channels: int,
    *,
    mib_per_connection_channel: float = 10.2,
) -> dict:
    """Project full communicator residency from declared transport classes.

    A planning model, not allocator telemetry -- prewarm's non-torch delta remains the admission
    measurement. Physical group identity is charged once however many aliases refer to it.
    """

    if channels < 1 or mib_per_connection_channel <= 0:
        raise ValueError("channels and MiB coefficient must be positive")
    rows = []
    total_mib = 0.0
    for entry in sorted(_GROUPS.values(), key=lambda value: value.signature()):
        world = len(entry.members)
        # An explicitly audited-unused group has an owner declaration with an empty operation set.
        # It is retained in runtime enforcement but prewarm need not materialize any connection.
        ring_connections = 2 if world > 1 and entry.operations else 0
        p2p_connections = 2 * (world - 1) if entry.operations & P2P_OPERATIONS else 0
        charge_mib = channels * (ring_connections + p2p_connections) * mib_per_connection_channel
        total_mib += charge_mib
        rows.append(
            {
                "members": entry.members,
                "aliases": tuple(sorted(entry.aliases)),
                "owners": tuple(sorted(entry.owners)),
                "carries_p2p": bool(p2p_connections),
                "connections_per_channel": ring_connections + p2p_connections,
                "projected_mib": charge_mib,
            }
        )
    return {
        "channels": channels,
        "mib_per_connection_channel": mib_per_connection_channel,
        "projected_mib": total_mib,
        "projected_gib": total_mib / 1024,
        "groups": rows,
    }


def reset_for_teardown() -> None:
    """Restore torch.distributed functions and clear all lifecycle-bound group identities."""

    global _ARMED, _INSTALLED, _MPU_MODULE, _ORIGINAL_MPU_DESTROY, _WRAPPED_MPU_DESTROY, _ARM_MEMORY
    global _SERVED_BASELINE
    _ARMED = False
    _ARM_MEMORY = None
    _SERVED_BASELINE.clear()
    for operation, original in _ORIGINALS.items():
        if getattr(dist, operation, None) is _WRAPPERS.get(operation):
            setattr(dist, operation, original)
    _ORIGINALS.clear()
    _WRAPPERS.clear()
    _GROUPS.clear()
    _INSTALLED = False
    if _WRAPPED_MPU_DESTROY is not None:
        # When called from the hook, the wrapped function is still installed. Restore the function
        # that owns Megatron's real cleanup before invoking it. A different wrapper means another
        # subsystem changed ownership after admission; mutating through that ambiguity is unsafe.
        if _MPU_MODULE is None or _MPU_MODULE.destroy_model_parallel is not _WRAPPED_MPU_DESTROY:
            raise RuntimeError("[ISOEXEC-NCCL-CAP] cannot teardown after destroy hook ownership drift")
        _MPU_MODULE.destroy_model_parallel = _ORIGINAL_MPU_DESTROY
    _MPU_MODULE = None
    _ORIGINAL_MPU_DESTROY = None
    _WRAPPED_MPU_DESTROY = None


def _reset_for_tests() -> None:
    reset_for_teardown()
