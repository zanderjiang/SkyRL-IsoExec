"""Price a NCCL channel per communicator, and decide whether this run can afford it.

A per-group ``max_ctas`` overrides the process-wide ``NCCL_MAX_NCHANNELS``, so channel count can be bought
where it pays and left alone elsewhere. NCCL allocates transport buffers per (channel, connection), and the
connection count depends on the collective patterns that ran on the communicator: ring/tree collectives need
2 connections regardless of world size, point-to-point patterns need ``2 * (world - 1)``. That memory matters
because NCCL buffers are plain ``cudaMalloc`` outside torch's allocator (``NCCL_CUMEM_ENABLE=0``) and stay
resident across sleep, so under ``colocate_all`` they compete one-for-one with vLLM's KV pool. The guard
therefore fails closed in three places: a pre-flight projection against a declared budget, a post-flight check
of the measured non-torch delta, and an outright refusal of groups whose NCCL reductions have not been
classified -- a different channel split can be a different summation order.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

#: MiB of NCCL transport buffer per (channel, connection), per rank. Calibrated high, so the projection
#: over-states and the guard errs toward refusing.
MIB_PER_CONNECTION_PER_CHANNEL = 10.2

#: Ring/tree collectives touch one predecessor and one successor per channel, whatever the world.
RING_CONNECTIONS = 2

#: What an unpinned communicator builds when ``NCCL_MAX_NCHANNELS`` is unset.
_DEFAULT_UNPINNED_CHANNELS = 24

# Channel count decides how NCCL splits a collective across thread blocks. For byte movement that is
# unobservable -- every output byte is a copy of one input byte. For a reduction a different split can be a
# different summation order, so the question per communicator is whether it reduces anything the forward gate
# reads, in a way whose order matters.

#: No NCCL reduction rides this communicator at all. Bit-safe by construction, unconditionally.
MOVE_ONLY = "MOVE-ONLY"

#: NCCL reduces on this communicator, but every reduction the FORWARD performs is order-immune and
#: every order-sensitive one is backward-only. Gradients reassociate (which they are allowed to do
#: -- the battery's ``BITWISE-FWD/TOL-BWD`` class); the gate does not move. Admissible, but only
#: while the composition the derivation was made under still holds -- see ``CONTRACT_PRECONDITIONS``.
FWD_IMMUNE = "FWD-IMMUNE/BWD-REASSOCIATES"

#: NCCL reduces on this communicator and nobody has derived what the forward reads. Refused.
#: The honest state for a group whose traffic has not been traced, and the default for everything
#: outside the two rows below.
UNCLASSIFIED = "REDUCES-UNCLASSIFIED"

#: Per-communicator contract, derived by attributing every collective on the group to its call site.
#:
#: ``ep`` / ``tp_ep`` are MOVE_ONLY: ``ep`` carries only ``all_to_allv``, ``tp_ep`` only the dispatcher
#: preprocess ``_allgather_base``.
#:
#: ``tp`` is FWD_IMMUNE. Its all-gathers move bytes; its ``all_to_allv`` traffic is pik's
#: ``tree_reduce_scatter``, which uses ``all_to_all_single`` as pure transport and evaluates its fixed tree
#: on-rank; all but one of its reduce-scatters are backward (gradients, which owe no bit contract), and the
#: forward one is ``VocabParallelEmbedding``, order-immune by construction because out-of-shard rows are
#: zeroed and ``x + 0`` is exact in any order. Of its all-reduces, the gated log-prob SUM has the same
#: masked-zero shape, one is a MAX, and the remaining fp32 SUMs feed the entropy metric rather than the gate.
GROUP_CONTRACT: dict[str, str] = {
    "ep": MOVE_ONLY,
    "tp_ep": MOVE_ONLY,
    "tp": FWD_IMMUNE,
}

#: The ``tp`` derivation holds only for this composition. Without pik shadowing RowParallelLinear, megatron's
#: forward all-reduce becomes a genuine SUM over nonzero partials -- order-sensitive and gate-critical -- so
#: ``tp`` falls back to UNCLASSIFIED unless these flags are set.
CONTRACT_PRECONDITIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "tp": (
        ("SKYRL_ISOEXEC_PIK", "1"),
        ("SKYRL_ISOEXEC_MOE_PIK_FC2", "1"),
        ("SKYRL_ISOEXEC_TRAINER_SP", "1"),
    ),
}


def group_contract(name: str, world: int) -> str:
    """The contract class this communicator's traffic falls under, at this world size.

    A world-2 reduction is order-free whatever the channel count, since two operands admit exactly one
    summation order, so the DP groups are immune by arithmetic rather than by derivation.
    """
    if world <= 1:
        return MOVE_ONLY
    if world == 2 and name not in GROUP_CONTRACT:
        return FWD_IMMUNE
    return GROUP_CONTRACT.get(name, UNCLASSIFIED)


def unmet_preconditions(name: str, env: dict[str, str] | None = None) -> tuple[str, ...]:
    """Which of this group's contract preconditions are NOT satisfied by the environment."""
    src = os.environ if env is None else env
    return tuple(
        f"{k}={src.get(k, '(unset)')!r} (needs {v!r})"
        for k, v in CONTRACT_PRECONDITIONS.get(name, ())
        if src.get(k) != v
    )


#: Groups whose traffic includes a point-to-point pattern, and which therefore pay ``2 * (world - 1)``
#: connections per channel rather than 2.
P2P_GROUPS = frozenset({"ep", "tp"})


def process_wide_channels() -> int:
    """The channel count a group gets with no per-group override -- i.e. what we are paying now."""
    raw = os.environ.get("NCCL_MAX_NCHANNELS", "").strip()
    try:
        n = int(raw)
    except ValueError:
        return _DEFAULT_UNPINNED_CHANNELS
    return n if n > 0 else _DEFAULT_UNPINNED_CHANNELS


def connections_per_channel(world: int, *, carries_p2p: bool) -> int:
    """Connections NCCL establishes per channel on a communicator of this world and pattern set.

    The pattern set matters as much as the world size, which is why a new movement-only communicator can be
    cheaper than widening an existing one that already carries every pattern.
    """
    if world <= 1:
        return 0
    return RING_CONNECTIONS + (2 * (world - 1) if carries_p2p else 0)


def charge_gib(world: int, channels: int, *, carries_p2p: bool, baseline_channels: int = 0) -> float:
    """GiB/rank this communicator costs at ``channels``, above what it already costs today.

    ``baseline_channels=0`` prices a NEW communicator (every channel is new memory);
    ``baseline_channels=8`` prices WIDENING one that already exists (only the delta is new).
    """
    if channels <= baseline_channels or world <= 1:
        return 0.0
    conns = connections_per_channel(world, carries_p2p=carries_p2p)
    return (channels - baseline_channels) * conns * MIB_PER_CONNECTION_PER_CHANNEL / 1024.0


@dataclass(frozen=True)
class GroupPlan:
    """One communicator's requested channel count, with everything the guard needs to judge it."""

    name: str
    channels: int
    world: int
    #: 0 for a communicator that does not exist yet; the process-wide count for one being widened.
    baseline_channels: int

    @property
    def carries_p2p(self) -> bool:
        return self.name in P2P_GROUPS

    @property
    def contract(self) -> str:
        return group_contract(self.name, self.world)

    @property
    def charge_gib(self) -> float:
        return charge_gib(
            self.world,
            self.channels,
            carries_p2p=self.carries_p2p,
            baseline_channels=self.baseline_channels,
        )


def parse_plan(spec: str, *, worlds: dict[str, int]) -> tuple[GroupPlan, ...]:
    """``"ep:24,tp:16"`` -> plans. Unknown groups and malformed entries are dropped and named in the banner,
    rather than raising, so a typo in an env var runs the incumbent configuration instead of killing the job.
    """
    plans: list[GroupPlan] = []
    base = process_wide_channels()
    for entry in (e.strip() for e in spec.split(",")):
        if not entry:
            continue
        name, _, raw = entry.partition(":")
        name = name.strip()
        try:
            channels = int(raw.strip())
        except ValueError:
            print(f"[ISOEXEC-NCCL-BUDGET] DROPPED {entry!r}: not 'group:channels'", flush=True)
            continue
        if name not in worlds:
            print(
                f"[ISOEXEC-NCCL-BUDGET] DROPPED {entry!r}: no world size known for group {name!r} "
                f"(known: {sorted(worlds)})",
                flush=True,
            )
            continue
        if channels <= 0:
            print(f"[ISOEXEC-NCCL-BUDGET] DROPPED {entry!r}: channels must be > 0", flush=True)
            continue
        plans.append(GroupPlan(name=name, channels=channels, world=worlds[name], baseline_channels=base))
    return tuple(plans)


@dataclass(frozen=True)
class Verdict:
    """The admission decision, plus every number that produced it. ``banner`` is always printed."""

    admitted: tuple[GroupPlan, ...]
    refused: tuple[tuple[GroupPlan, str], ...]
    projected_gib: float
    budget_gib: float
    banner: str

    @property
    def any_admitted(self) -> bool:
        return bool(self.admitted)


def admit(
    plans: tuple[GroupPlan, ...],
    *,
    budget_gib: float,
    free_gib: float | None = None,
    reserve_gib: float = 0.0,
    ack_reduce: bool = False,
) -> Verdict:
    """Decide which of ``plans`` this run can afford, and refuse the rest with a reason.

    The budget test is all-or-nothing on the plan as a whole, so a partially admitted plan can never produce a
    configuration nobody chose; per-group refusals only fire for properties of that group.
    """
    kept: list[GroupPlan] = []
    refused: list[tuple[GroupPlan, str]] = []

    for p in plans:
        unmet = unmet_preconditions(p.name)
        if p.world <= 1:
            refused.append((p, f"world={p.world}: there is no communicator here to widen"))
        elif p.channels <= p.baseline_channels:
            refused.append((p, f"channels={p.channels} <= today's {p.baseline_channels}: a no-op or a NARROWING"))
        elif p.contract == UNCLASSIFIED and not ack_reduce:
            refused.append(
                (
                    p,
                    f"{UNCLASSIFIED}: this group carries a NCCL REDUCTION and nobody has derived "
                    "what its FORWARD reads. Channel count decides how NCCL splits a collective, "
                    "and for a reduction a different split can be a different summation order, so "
                    "an underived group is a composition event (manifest pin + gate re-freeze), not "
                    "a flag. Trace it and add a GROUP_CONTRACT row, or set "
                    "SKYRL_ISOEXEC_NCCL_CHANNEL_ACK_REDUCE=1 for a deliberate experiment",
                )
            )
        elif p.contract == FWD_IMMUNE and unmet and not ack_reduce:
            refused.append(
                (
                    p,
                    f"{FWD_IMMUNE} was DERIVED UNDER A COMPOSITION THIS RUN IS NOT IN: "
                    + "; ".join(unmet)
                    + ". Without pik shadowing RowParallelLinear, megatron's _reduce becomes a "
                    "genuine forward SUM over four nonzero partials and this group is "
                    f"{UNCLASSIFIED} again",
                )
            )
        else:
            kept.append(p)

    projected = sum(p.charge_gib for p in kept)

    over_budget = projected > budget_gib
    short_of_free = free_gib is not None and projected + reserve_gib > free_gib
    if kept and (over_budget or short_of_free):
        why = []
        if over_budget:
            why.append(f"projected {projected:.3f} GiB/rank > budget {budget_gib:.3f}")
        if short_of_free:
            why.append(f"projected {projected:.3f} + reserve {reserve_gib:.3f} GiB > free {free_gib:.3f} GiB")
        reason = "; ".join(why)
        refused.extend((p, reason) for p in kept)
        kept = []
        projected = 0.0

    lines = [
        "[ISOEXEC-NCCL-BUDGET] per-communicator channel plan "
        f"(process-wide NCCL_MAX_NCHANNELS={os.environ.get('NCCL_MAX_NCHANNELS', '(unset)')}, "
        f"budget {budget_gib:.3f} GiB/rank"
        + (f", free {free_gib:.3f} GiB, reserve {reserve_gib:.3f} GiB" if free_gib is not None else "")
        + ")"
    ]
    for p in kept:
        lines.append(
            f"[ISOEXEC-NCCL-BUDGET]   ADMIT  {p.name:>7s} world={p.world} "
            f"{p.baseline_channels} -> {p.channels} ch  "
            f"{connections_per_channel(p.world, carries_p2p=p.carries_p2p)} conn/ch  "
            f"projected +{p.charge_gib:.3f} GiB/rank  [{p.contract}]"
        )
        if p.contract == FWD_IMMUNE:
            lines.append(
                "[ISOEXEC-NCCL-BUDGET]     ^ this group REDUCES. Admitted because every forward "
                "reduction on it is order-immune (masked-zero SUMs and a MAX) and the "
                "order-sensitive ones are BACKWARD-only: expect GRADIENTS and the entropy metric "
                "to move, and [ISOEXEC-DIFF] abs_diff_mean NOT to. If the gate moves, this "
                "classification is wrong -- refuse the lever, do not re-freeze the gate."
            )
    for p, why in refused:
        lines.append(f"[ISOEXEC-NCCL-BUDGET]   REFUSE {p.name:>7s} -> {p.channels} ch: {why}")
    if not kept:
        lines.append(
            "[ISOEXEC-NCCL-BUDGET]   RESULT: nothing admitted -- megatron keeps "
            "nccl_communicator_config_path=None, i.e. EXACTLY today's configuration."
        )
    else:
        lines.append(
            f"[ISOEXEC-NCCL-BUDGET]   RESULT: {len(kept)} group(s), projected total "
            f"+{projected:.3f} GiB/rank. Verified against the measured delta after "
            f"initialize_model_parallel; a real charge over budget KILLS THE RUN AT INIT."
        )
    return Verdict(
        admitted=tuple(kept),
        refused=tuple(refused),
        projected_gib=projected,
        budget_gib=budget_gib,
        banner="\n".join(lines),
    )


#: How far the measured charge may exceed the declared budget before the guard fires. Leaves room for a NCCL
#: version that allocates differently without letting a large surprise through to ``wake_up()``.
VERIFY_TOLERANCE = 1.5


def verify_charge(measured_gib: float, verdict: Verdict) -> str:
    """Compare the measured charge against the projection; returns a banner and raises if over budget."""
    if not verdict.any_admitted:
        return "[ISOEXEC-NCCL-BUDGET] verify skipped: nothing was admitted."
    ratio = measured_gib / verdict.projected_gib if verdict.projected_gib > 0 else float("inf")
    head = (
        f"[ISOEXEC-NCCL-BUDGET] measured charge {measured_gib:+.3f} GiB/rank vs projected "
        f"{verdict.projected_gib:+.3f} ({ratio:.2f}x), budget {verdict.budget_gib:.3f}"
    )
    if measured_gib > verdict.budget_gib * VERIFY_TOLERANCE:
        raise RuntimeError(
            f"{head}\n[ISOEXEC-NCCL-BUDGET] REFUSING TO CONTINUE: the widened communicators cost "
            f"{measured_gib:.3f} GiB/rank, more than {VERIFY_TOLERANCE}x the declared budget. Under "
            f"colocate_all this memory is un-reclaimable (NCCL_CUMEM_ENABLE=0, outside torch's "
            f"allocator, resident across sleep) and it would OOM vLLM's wake_up() at step 1 instead "
            f"of here. Lower SKYRL_ISOEXEC_NCCL_CHANNEL_PLAN, or raise "
            f"SKYRL_ISOEXEC_NCCL_CHANNEL_BUDGET_GIB only together with gpu_memory_utilization."
        )
    return head + ("  OK" if ratio <= VERIFY_TOLERANCE else "  OK (over projection, under budget)")
