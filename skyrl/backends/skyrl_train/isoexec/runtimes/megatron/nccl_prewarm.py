"""Force every trainer NCCL communicator into existence at a known point in the run.

NCCL allocates communicator memory lazily -- at the first collective, and again for each transport it
has not yet connected -- which under ``colocate_all`` is memory vLLM's KV-pool profile never saw and
``wake_up(kv_cache)`` can never get back. Each group is therefore warmed with six different
collectives rather than one, because an ``all_reduce`` alone connects less than 10% of a communicator
and ``all_to_all`` (SendRecv/p2p) is the expensive transport. The weight-sync broadcast group is
deliberately not warmed: it does not exist yet at init_model, and its first collective already lands
before ``wake_up(kv_cache)``.

Warming makes the trainer's NCCL cost deterministic and complete; sizing the KV pool around the
reported GiB remains a separate, explicit act. Gated on ``SKYRL_ISOEXEC_NCCL_PREWARM=1``, off by
default. Optional ``SKYRL_ISOEXEC_NCCL_CAPABILITY_MODE=census|enforce`` adds transport-ownership
accounting, where ``enforce`` requires explicit owner declarations per physical group and refuses an
unexpected all-to-all before c10d can connect it lazily.
"""

from __future__ import annotations

import math
import os

import torch
import torch.distributed as dist

# (getter name, kwargs). Order is FIXED and identical on every rank -- that is what keeps
# the collectives below in lockstep. Every resolved membership and every execution result is
# WORLD-unanimous through the CPU Store before the next entry may start; rank-local skipping is
# forbidden because it changes the collective sequence.
_MPU_GROUPS = (
    ("get_tensor_model_parallel_group", {}),
    ("get_pipeline_model_parallel_group", {}),
    ("get_model_parallel_group", {}),
    ("get_context_parallel_group", {}),
    ("get_tensor_and_context_parallel_group", {}),
    ("get_data_parallel_group", {"with_context_parallel": False}),
    ("get_data_parallel_group", {"with_context_parallel": True}),
    ("get_expert_model_parallel_group", {}),
    ("get_expert_tensor_parallel_group", {}),
    ("get_expert_tensor_and_model_parallel_group", {}),
    ("get_expert_tensor_model_pipeline_parallel_group", {}),
    ("get_expert_data_parallel_group", {}),
    ("get_embedding_group", {}),
    ("get_position_embedding_group", {}),
)

# 8 MiB is large enough to select the production transport/protocol family while keeping the two
# full-size scratch buffers below to ~16 MiB/rank in total.
_WARM_PAYLOAD_BYTES = 8 * 1024**2


def _free_mib() -> float:
    torch.cuda.synchronize()
    return torch.cuda.mem_get_info()[0] / (1024**2)


def _nontorch_mib() -> float:
    """Device-wide used bytes minus this process's torch reservation.

    This is an estimate, not process-local NCCL accounting: allocations owned by colocated engine
    processes are also visible in ``mem_get_info``.  The before/after delta is useful at this known
    quiescent point; the absolute readings are reported explicitly as estimates.
    """
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return (total - free - torch.cuda.memory_reserved()) / (1024**2)


def _warm_elements_per_peer(world: int, element_size: int) -> int:
    """Elements per peer for an equal-split payload of at least ``_WARM_PAYLOAD_BYTES``."""
    if world < 1 or element_size < 1:
        raise ValueError(f"invalid warm payload geometry world={world} element_size={element_size}")
    return math.ceil(_WARM_PAYLOAD_BYTES / (world * element_size))


def _control_plane_all_gather(tag: str, value):
    """Bounded init vote through c10d's CPU rendezvous Store, never a NCCL collective."""
    from skyrl.backends.skyrl_train.distributed.store_rendezvous import store_all_gather
    from skyrl.env_vars import SKYRL_WORKER_NCCL_TIMEOUT_IN_S

    store = dist.distributed_c10d._get_default_store()
    return store_all_gather(
        store,
        dist.get_rank(),
        dist.get_world_size(),
        tag,
        value,
        SKYRL_WORKER_NCCL_TIMEOUT_IN_S,
    )


def _unanimous_reports(control_gather, tag: str, local_report: dict, world: int) -> list[dict]:
    """Gather one phase verdict and make every rank refuse the same rank-local failure."""
    reports = control_gather(tag, local_report)
    if len(reports) != world:
        raise RuntimeError(f"[ISOEXEC-NCCL-PREWARM] {tag} refusal: report count {len(reports)} != WORLD {world}")
    ranks = sorted(int(report.get("rank", -1)) for report in reports)
    errors = [str(report.get("error")) for report in reports if report.get("error")]
    if ranks != list(range(world)):
        errors.append(f"report ranks must be exactly WORLD once each, got {ranks!r}")
    phases = {report.get("phase") for report in reports}
    if phases != {local_report.get("phase")}:
        errors.append(f"rank-nonunanimous phase signatures: {sorted(map(repr, phases))!r}")
    if errors:
        raise RuntimeError(f"[ISOEXEC-NCCL-PREWARM] {tag} unanimous refusal: " + "; ".join(errors))
    return reports


def _validate_resolved_memberships(reports: list[dict], phase: str) -> None:
    """Require every active subgroup member to report the same ordered membership before traffic."""
    by_rank = {int(report["rank"]): report for report in reports}
    errors = []
    for rank, report in by_rank.items():
        if not report.get("active"):
            continue
        members = tuple(int(peer) for peer in report.get("members", ()))
        if rank not in members:
            errors.append(f"rank {rank} reported non-owning membership {members!r}")
            continue
        for peer in members:
            peer_report = by_rank.get(peer)
            if peer_report is None or not peer_report.get("active"):
                errors.append(f"rank {rank} expects active peer {peer}, but its report is absent/inactive")
            elif tuple(int(value) for value in peer_report.get("members", ())) != members:
                errors.append(
                    f"rank {rank} membership {members!r} != peer {peer} membership "
                    f"{tuple(peer_report.get('members', ()))!r}"
                )
    if errors:
        raise RuntimeError(f"[ISOEXEC-NCCL-PREWARM] {phase} membership refusal: " + "; ".join(errors))


def _validate_resolved_capabilities(reports: list[dict], phase: str) -> None:
    """Require every member of one physical subgroup to declare identical owner operations."""
    by_rank = {int(report["rank"]): report for report in reports}
    errors = []
    for rank, report in by_rank.items():
        if not report.get("active"):
            continue
        members = tuple(int(peer) for peer in report.get("members", ()))
        capability = tuple(report.get("capability", ()))
        for peer in members:
            peer_report = by_rank.get(peer)
            if peer_report is not None and tuple(peer_report.get("capability", ())) != capability:
                errors.append(
                    f"rank {rank} capability {capability!r} != peer {peer} capability "
                    f"{tuple(peer_report.get('capability', ()))!r}"
                )
    if errors:
        raise RuntimeError(f"[ISOEXEC-NCCL-PREWARM] {phase} capability refusal: " + "; ".join(errors))


def _run_unanimous_warm(
    *,
    control_gather,
    tag: str,
    phase: str,
    rank: int,
    world: int,
    active: bool,
    warm,
) -> bool:
    """Run one local subgroup warm, then stop every WORLD rank before the next phase on error."""
    served = False
    local_error = ""
    if active:
        try:
            served = bool(warm())
            if not served:
                raise RuntimeError("warm owner declined after unanimous active membership admission")
        except Exception as exc:  # noqa: BLE001 - every rank reports before any proceeds
            local_error = f"rank={rank} {type(exc).__name__}: {exc}"
    _unanimous_reports(
        control_gather,
        tag,
        {
            "rank": rank,
            "phase": phase,
            "active": active,
            "served": served,
            "error": local_error,
        },
        world,
    )
    return served


def _warm_one(g, *, move_only: bool = False, skip_a2a: bool = False, operations=None) -> bool:
    """Fire one of every collective the trainer will ever run on ``g``.

    Sizes are small on purpose: NCCL's per-channel buffers do not scale with the payload, so this
    costs milliseconds. What matters is the SET of collectives, since each connects a different
    transport -- SendRecv (all_to_all) is the expensive one and an all_reduce does not connect it. A
    channel plan classified as move-only warms AG/A2A/broadcast only, so it does not violate the
    operation-ownership premise the runtime guard enforces.
    """
    try:
        ws = dist.get_world_size(g)
    except Exception:  # noqa: BLE001
        return False
    if ws < 2:
        return False
    if operations is not None and not operations:
        # Explicitly audited-unused physical group: retain it in the armed runtime guard, but do
        # not create a single lazy NCCL connection. Any later operation is refused before c10d.
        return True
    dev = torch.cuda.current_device()
    n = _warm_elements_per_peer(ws, torch.empty((), dtype=torch.bfloat16).element_size())
    big = torch.zeros(ws * n, device=dev, dtype=torch.bfloat16)
    out = torch.zeros(ws * n, device=dev, dtype=torch.bfloat16)
    chunk = torch.zeros(n, device=dev, dtype=torch.bfloat16)
    smallf = torch.zeros(64, device=dev, dtype=torch.float32)

    if not move_only:
        dist.all_reduce(smallf, group=g)  # fp32 ring/tree
        dist.all_reduce(big, group=g)  # bf16, larger -> other protocol
    dist.all_gather_into_tensor(out, chunk, group=g)
    if not move_only:
        dist.reduce_scatter_tensor(chunk, big, group=g)
    if not skip_a2a:
        dist.all_to_all_single(out, big, group=g)  # <- SendRecv/p2p transports
    dist.broadcast(big, src=dist.get_global_rank(g, 0), group=g)
    if not move_only:
        dist.barrier(group=g)
    torch.cuda.synchronize()
    del big, out, chunk, smallf
    return True


def _warm_move_a2a(g) -> bool:
    """Materialize only the A2A transport owned by a dedicated movement group."""
    try:
        ws = dist.get_world_size(g)
    except Exception:  # noqa: BLE001
        return False
    if ws < 2:
        return False
    n = _warm_elements_per_peer(ws, torch.empty((), dtype=torch.bfloat16).element_size())
    source = torch.zeros(ws * n, device=torch.cuda.current_device(), dtype=torch.bfloat16)
    destination = torch.empty_like(source)
    dist.all_to_all_single(destination, source, group=g)
    torch.cuda.synchronize()
    del source, destination
    return True


def _skip_a2a_for_group(g, *, movement_source: bool, capabilities, capability_mode: str) -> bool:
    """Whether prewarm may omit the expensive P2P transport for this physical group."""
    if movement_source:
        return True
    if capabilities is None or capability_mode != capabilities.ENFORCE:
        return False
    entry = capabilities.capability_for(g)
    if entry is None or not entry.owners:
        raise RuntimeError("selective prewarm reached a group without explicit active ownership")
    return not bool(entry.operations & capabilities.P2P_OPERATIONS)


def prewarm_trainer_nccl(tag: str = "policy", *, _control_gather=None) -> dict:
    """Warm every megatron process group this rank belongs to. Returns the accounting.

    Safe to call more than once (the second call allocates nothing). No-op unless
    ``SKYRL_ISOEXEC_NCCL_PREWARM=1``.
    """
    if os.environ.get("SKYRL_ISOEXEC_NCCL_PREWARM", "0") != "1":
        return {}
    if not (dist.is_available() and dist.is_initialized() and torch.cuda.is_available()):
        return {}

    from megatron.core import parallel_state as mpu

    capability_mode = os.environ.get("SKYRL_ISOEXEC_NCCL_CAPABILITY_MODE", "off").strip().lower() or "off"
    capabilities = None
    if capability_mode != "off":
        from skyrl.backends.skyrl_train.isoexec.runtimes.megatron import (
            nccl_transport_capabilities,
        )

        capabilities = nccl_transport_capabilities
        # Validate the spelling before any rank enters a collective.
        capability_mode = capabilities.mode()

    movement_setup_error = ""
    movement_requested = bool(os.environ.get("SKYRL_ISOEXEC_NCCL_CHANNEL_PLAN", "").strip())
    if movement_requested and capabilities is not None:
        raise RuntimeError(
            "[ISOEXEC-NCCL-PREWARM] capability-owned incumbent groups and duplicate movement "
            "groups are separate admission arms and cannot be enabled in one lifecycle"
        )
    if movement_requested:
        try:
            from skyrl.backends.skyrl_train.isoexec.ops.collectives.ep_move_group import (
                is_movement_source,
                movement_groups_for_prewarm,
            )

            movement_groups = movement_groups_for_prewarm()
        except Exception as exc:  # noqa: BLE001 - vote before logical-group traffic can diverge
            movement_groups = ()
            movement_setup_error = f"{type(exc).__name__}: {exc}"

            def is_movement_source(_group):
                return False

    else:
        # Default-off means import-inert for the experimental duplicate-PG module. The active
        # process-wide cap path should not depend on an unrequested transport implementation.
        movement_groups = ()

        def is_movement_source(_group):
            return False

    rank = dist.get_rank()
    world = dist.get_world_size()
    control_gather = _control_plane_all_gather if _control_gather is None else _control_gather
    if capabilities is not None:
        declaration_signature = capabilities.declaration_signature()
        declaration_error = ""
        if capability_mode == capabilities.ENFORCE and not declaration_signature:
            declaration_error = f"rank={rank} enforce mode has no active owner declarations"
        declaration_reports = _unanimous_reports(
            control_gather,
            f"nccl-prewarm-{tag}-capability-signature",
            {
                "rank": rank,
                "phase": "capability-signature",
                "signature": declaration_signature,
                "error": declaration_error,
            },
            world,
        )
        declaration_signatures = {tuple(report.get("signature", ())) for report in declaration_reports}
        if len(declaration_signatures) != 1:
            raise RuntimeError(
                "[ISOEXEC-NCCL-PREWARM] capability-signature refusal: "
                f"rank-nonunanimous declarations {sorted(map(repr, declaration_signatures))!r}"
            )
    # This vote must precede logical-group traffic. If one rank failed to import/resolve movement
    # ownership, it would otherwise warm A2A on the logical EP group while peers deliberately skip
    # it, creating the exact asymmetric collective order this guard exists to prevent.
    movement_signature = tuple((entry.name, entry.channels) for entry in movement_groups)
    signature_reports = _unanimous_reports(
        control_gather,
        f"nccl-prewarm-{tag}-movement-signature",
        {
            "rank": rank,
            "phase": "movement-signature",
            "signature": movement_signature,
            "error": (f"rank={rank} movement setup failed: {movement_setup_error}" if movement_setup_error else ""),
        },
        world,
    )
    signatures = {tuple(report.get("signature", ())) for report in signature_reports}
    if len(signatures) != 1:
        raise RuntimeError(
            "[ISOEXEC-NCCL-PREWARM] movement-signature refusal: "
            f"rank-nonunanimous duplicate plans {sorted(map(repr, signatures))!r}"
        )
    torch.cuda.empty_cache()
    absolute_nontorch_before = _nontorch_mib()
    free0 = _free_mib()
    per_group, per_group_nontorch, warmed = [], [], 0

    # WORLD first: it is the group every rank shares and the one already half-connected by
    # init_model's barrier, so its delta reports only what the barrier did NOT cover.
    for index, (name, kwargs) in enumerate((("WORLD", None),) + _MPU_GROUPS):
        phase = f"{index}:{name}{_kw(kwargs)}"
        g = None
        members = ()
        active = False
        local_error = ""
        capability_signature = ()
        try:
            if kwargs is None:
                g = dist.group.WORLD
            elif name in ("get_embedding_group", "get_position_embedding_group"):
                # In PP>2, middle stages intentionally do not belong to these endpoint groups.
                # MCore's default getter asserts there; ``check_initialized=False`` turns that
                # expected non-membership into None so it can participate in the WORLD phase vote.
                g = getattr(mpu, name)(check_initialized=False, **kwargs)
            else:
                g = getattr(mpu, name)(**kwargs)
            if g is not None:
                group_world = dist.get_world_size(g)
                members = tuple(dist.get_process_group_ranks(g))
                active = group_world >= 2
                if active and capabilities is not None:
                    if capability_mode == capabilities.CENSUS:
                        capability = capabilities.track_group(phase, g)
                    else:
                        capability = capabilities.capability_for(g)
                        if capability is None or not capability.owners:
                            raise RuntimeError(
                                f"no active owner declared transport capabilities for {phase} members={members!r}"
                            )
                        capability.aliases.add(phase)
                    capability_signature = tuple(
                        sorted((owner, tuple(sorted(operations))) for owner, operations in capability.owners.items())
                    )
                else:
                    capability_signature = ()
        except Exception as exc:  # noqa: BLE001 - vote before any rank starts subgroup traffic
            local_error = f"rank={rank} {type(exc).__name__}: {exc}"
            capability_signature = ()
        reports = _unanimous_reports(
            control_gather,
            f"nccl-prewarm-{tag}-resolve-{index}",
            {
                "rank": rank,
                "phase": phase,
                "active": active,
                "members": members,
                "capability": capability_signature,
                "error": local_error,
            },
            world,
        )
        _validate_resolved_memberships(reports, phase)
        if capabilities is not None:
            _validate_resolved_capabilities(reports, phase)
        if not active:
            # Still take the post-phase WORLD Store vote, so every rank advances in phase order even
            # when only a subset belongs to an embedding/pipeline group.
            _run_unanimous_warm(
                control_gather=control_gather,
                tag=f"nccl-prewarm-{tag}-execute-{index}",
                phase=phase,
                rank=rank,
                world=world,
                active=False,
                warm=lambda: False,
            )
            continue
        before = _free_mib()
        nontorch_before = _nontorch_mib()
        ok = _run_unanimous_warm(
            control_gather=control_gather,
            tag=f"nccl-prewarm-{tag}-execute-{index}",
            phase=phase,
            rank=rank,
            world=world,
            active=True,
            warm=lambda g=g: _warm_one(
                g,
                skip_a2a=_skip_a2a_for_group(
                    g,
                    movement_source=is_movement_source(g),
                    capabilities=capabilities,
                    capability_mode=capability_mode,
                ),
                operations=(
                    capabilities.capability_for(g).operations
                    if capabilities is not None and capability_mode == capabilities.ENFORCE
                    else None
                ),
            ),
        )
        if not ok:  # Defensive even under ``python -O``; normally the unanimous vote raised first.
            raise RuntimeError(f"[ISOEXEC-NCCL-PREWARM] {phase} returned without serving")
        warmed += 1
        label = f"{name}{_kw(kwargs)}"
        per_group.append((label, before - _free_mib(), dist.get_world_size(g)))
        per_group_nontorch.append((label, _nontorch_mib() - nontorch_before, dist.get_world_size(g)))

    # Dedicated movement groups are not part of Megatron's getter table.  Warm exactly their
    # admitted A2A owner; all logical-group reductions above stayed on the incumbent pinned group.
    for index, entry in enumerate(movement_groups):
        phase = f"movement:{index}:{entry.name}"
        reports = _unanimous_reports(
            control_gather,
            f"nccl-prewarm-{tag}-movement-resolve-{index}",
            {
                "rank": rank,
                "phase": phase,
                "active": True,
                "members": tuple(entry.ranks),
                "error": "",
            },
            world,
        )
        _validate_resolved_memberships(reports, phase)
        before = _free_mib()
        nontorch_before = _nontorch_mib()
        ok = _run_unanimous_warm(
            control_gather=control_gather,
            tag=f"nccl-prewarm-{tag}-movement-execute-{index}",
            phase=phase,
            rank=rank,
            world=world,
            active=True,
            warm=lambda entry=entry: _warm_move_a2a(entry.transport),
        )
        if not ok:
            raise RuntimeError(f"[ISOEXEC-NCCL-PREWARM] {phase} returned without serving")
        warmed += 1
        label = f"isoexec_move:{entry.name}"
        per_group.append((label, before - _free_mib(), dist.get_world_size(entry.transport)))
        lazy_nontorch_mib = _nontorch_mib() - nontorch_before
        per_group_nontorch.append(
            (
                label,
                entry.creation_nontorch_mib + lazy_nontorch_mib,
                dist.get_world_size(entry.transport),
            )
        )

    torch.cuda.empty_cache()
    compatibility_free_delta = free0 - _free_mib()
    absolute_nontorch_after = _nontorch_mib()
    incremental_nontorch = absolute_nontorch_after - absolute_nontorch_before
    if capabilities is not None and os.environ.get("SKYRL_ISOEXEC_NCCL_TRANSPORT_BOUNDARY_REQUIREMENTS", "").strip():
        # All ranks have completed all subgroup phases and their per-membership declarations were
        # unanimous.  Only now may the runtime wrapper reject an unexpected lazy transport.
        capabilities.install_teardown_hook(mpu)
        capabilities.arm()
    if rank == 0:
        rows = " ".join(f"{n}({w})={d:.0f}M" for n, d, w in per_group)
        print(
            f"[ISOEXEC-NCCL-PREWARM] rank={rank} {tag} groups_warmed={warmed} "
            f"incremental_nontorch={incremental_nontorch:.0f} MiB "
            f"({incremental_nontorch / 1024:.2f} GiB) "
            f"absolute_nontorch_estimate_before={absolute_nontorch_before:.0f} MiB "
            f"absolute_nontorch_estimate_after={absolute_nontorch_after:.0f} MiB "
            f"compatibility_free_delta={compatibility_free_delta:.0f} MiB "
            f"free_after={_free_mib() / 1024:.2f} GiB "
            f"warm_payload={_WARM_PAYLOAD_BYTES / 1024**2:.0f}MiB "
            f"NCCL_MAX_NCHANNELS={os.environ.get('NCCL_MAX_NCHANNELS')} "
            f"PIN={os.environ.get('SKYRL_ISOEXEC_NCCL_PIN', '1')} :: {rows}",
            flush=True,
        )
    return {
        # Compatibility: callers historically read ``total_mib`` as the free-memory delta around
        # this function. It is not and never was an absolute NCCL footprint.
        "total_mib": compatibility_free_delta,
        "incremental_nontorch_mib": incremental_nontorch,
        "absolute_nontorch_before_mib": absolute_nontorch_before,
        "absolute_nontorch_after_mib": absolute_nontorch_after,
        "warm_payload_bytes": _WARM_PAYLOAD_BYTES,
        "groups": per_group,
        "nontorch_groups": per_group_nontorch,
        "warmed": warmed,
    }


def _kw(kwargs) -> str:
    if not kwargs:
        return ""
    return "[" + ",".join(f"{k}={v}" for k, v in kwargs.items()) + "]"
