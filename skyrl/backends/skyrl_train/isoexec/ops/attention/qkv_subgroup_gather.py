"""Subgroup all-gather for megatron's fused-QKV column gather -- half the bytes, byte-identical.

When num_query_groups < tp_world, megatron all-gathers the full fused-QKV width across the whole
TP group and discards (ng-1)/ng of it. Gathering over only the rank subgroup that owns the kept
window yields exactly those columns in exactly that order, so this is a byte permutation rather
than an approximation.

Engine-only: the shim records no backward, and building the subgroups is a default-group
collective, so it is admitted only when the TP group spans the whole default world.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch

FLAG_QKV_SUBGROUP_AG = "SKYRL_ISOEXEC_ATTN_QKV_SUBGROUP_AG"

# NCCL channels for the subgroup communicator; a per-communicator setting overrides
# NCCL_MAX_NCHANNELS. Below ~8 the subgroup gather is slower than the full-group gather it
# replaces. 16 is the measured knee and costs ~326 MiB/rank; 0 leaves NCCL's own tuner alone.
SUBGROUP_MAX_CTAS = 16


def qkv_subgroup_ag_enabled() -> bool:
    """On by default; ``=0`` restores the stock megatron gather byte for byte."""
    return os.environ.get(FLAG_QKV_SUBGROUP_AG, "1") == "1"


@dataclass
class SubgroupPlan:
    """Everything the shim needs, resolved once at install and frozen."""

    world: int  # tensor-parallel world size (== SelfAttention.world_size)
    num_query_groups: int  # config.num_query_groups
    ranks_per_group: int  # world // num_query_groups
    group_index: int  # this rank's `idx`, == the megatron slice index
    tp_group: Any = None  # the megatron tp ProcessGroup the plan was built for
    sub_pg: Any = None  # the cached sub-ProcessGroup {idx*rpg .. (idx+1)*rpg-1}
    tp_ranks: tuple = ()  # global ranks of tp_group, in group-rank order
    sub_ranks: tuple = ()  # global ranks of sub_pg, in group-rank order
    # tp ProcessGroups whose rank list was verified to equal ``tp_ranks``. megatron builds one
    # ProcessGroupCollection per layer, so the ``.tp`` handles need not be one object; keyed by
    # id() and held here so the shim can do an O(1) membership test without id() recycling.
    ok_groups: dict = field(default_factory=dict)


def admission_refusal(
    *,
    world: int,
    num_query_groups: int | None,
    local_width: int | None = None,
) -> str | None:
    """Return why this geometry is inadmissible, or ``None`` if it is admissible.

    Pure: no distributed state, no CUDA. Each predicate is one the byte-identity argument depends
    on, so a change to megatron's slicing formula lands as a refusal rather than a wrong answer.
    """
    if num_query_groups is None:
        return "config.num_query_groups is None (megatron asserts on this itself)"
    if world < 2:
        return f"tp world={world}: nothing to gather"
    if num_query_groups >= world:
        return (
            f"num_query_groups={num_query_groups} >= tp world={world}: megatron takes NO "
            f"all-gather in this branch, so there is nothing to shrink"
        )
    if num_query_groups < 1:
        return f"num_query_groups={num_query_groups} < 1"
    if num_query_groups == 1:
        return (
            "num_query_groups=1: the subgroup would BE the full tp group -- identical traffic, "
            "so the extra communicator buys nothing"
        )
    if world % num_query_groups != 0:
        return (
            f"tp world={world} is not a multiple of num_query_groups={num_query_groups}: the "
            f"slice window would not land on rank-shard boundaries"
        )
    if local_width is None:
        return None
    rpg = world // num_query_groups
    full = local_width * world
    if full % num_query_groups != 0:
        return (
            f"fused-qkv width {full} is not divisible by num_query_groups={num_query_groups}: "
            f"megatron's `size = width // ng` would truncate"
        )
    size = full // num_query_groups
    if size != rpg * local_width:
        # The statement that the kept window is exactly `rpg` whole rank shards; catches a change
        # to megatron's slicing formula.
        return (
            f"slice width {size} != ranks_per_group({rpg}) * shard width({local_width}): the "
            f"kept window is not a whole number of rank shards"
        )
    return None


_SUBGROUPS: dict = {}  # (tp_ranks, ngroups) -> list[pg | None], one entry per group index


def build_qkv_subgroups(tp_group, num_query_groups: int, *, max_ctas: int = SUBGROUP_MAX_CTAS):
    """Create (once) the ``num_query_groups`` sub-process-groups over ``tp_group``'s ranks.

    Collective over the DEFAULT process group: every rank must call this in the same order with the
    same arguments. Returns this rank's own sub-group, and raises on anything unexpected.
    """
    import torch.distributed as dist

    tp_ranks = tuple(dist.get_process_group_ranks(tp_group))
    key = (tp_ranks, int(num_query_groups), int(max_ctas))
    world = len(tp_ranks)
    rpg = world // num_query_groups
    my_rank = dist.get_rank()

    if key not in _SUBGROUPS:
        opts = None
        if max_ctas:
            try:
                opts = dist.ProcessGroupNCCL.Options()
                opts.config.max_ctas = int(max_ctas)
                opts.config.min_ctas = int(max_ctas)
            except Exception:  # pragma: no cover - non-NCCL builds
                opts = None
        made = []
        for g in range(num_query_groups):
            members = list(tp_ranks[g * rpg : (g + 1) * rpg])
            pg = dist.new_group(
                ranks=members,
                pg_options=opts,
                group_desc=f"isoexec_qkv_subgroup_{g}",
            )
            made.append(pg if my_rank in members else None)
        _SUBGROUPS[key] = made

    idx = tp_ranks.index(my_rank) // rpg
    return _SUBGROUPS[key][idx]


def warm_subgroup(sub_pg) -> None:
    """Force NCCL to build the communicator now, outside any CUDA-graph capture.

    NCCL creates a communicator lazily on first collective; if that lands inside a decode-graph
    capture the process hangs or dies, so this warmup is mandatory at install time.
    """
    import torch.distributed as dist

    n = dist.get_world_size(sub_pg)
    src = torch.zeros(8, dtype=torch.bfloat16, device=torch.cuda.current_device())
    dst = torch.zeros(8 * n, dtype=torch.bfloat16, device=torch.cuda.current_device())
    dist.all_gather_into_tensor(dst, src, group=sub_pg)
    torch.cuda.synchronize()


def subgroup_gather_last_dim(input_: torch.Tensor, plan: SubgroupPlan) -> torch.Tensor:
    """Full-width tensor whose ``[idx*size, (idx+1)*size)`` window holds the subgroup gather.

    The permutation is megatron's own ``_gather_along_last_dim`` called on the sub-group, not a
    re-derivation of it, so a changed convention upstream breaks here rather than sliding past.
    The rest of the buffer is left uninitialised: megatron's next lines slice out only that window.
    """
    from megatron.core.tensor_parallel.mappings import _gather_along_last_dim

    local = input_.size(-1)
    size = plan.ranks_per_group * local
    full = plan.world * local

    # width == rpg * local == size, columns in rank order == exactly the kept window.
    sub = _gather_along_last_dim(input_, plan.sub_pg)

    out = torch.empty(input_.shape[:-1] + (full,), dtype=input_.dtype, device=input_.device)
    lo = plan.group_index * size
    out[..., lo : lo + size].copy_(sub)
    return out


_INSTALLED = False
_ORIG_METHOD = None
_ORIG_AG = None
_PLAN: SubgroupPlan | None = None
_ARMED: SubgroupPlan | None = None
_REFUSED: set = set()
_COUNTS = {"armed": 0, "diverted": 0, "fallback": 0, "verified": 0}
_UNSET = object()

# How many of the first armed calls cross-check against megatron's own full gather. The claim under
# test is about shard layout, not any one shape, so a handful of eager calls settles it; checking
# per shape would put a full-width gather back into every new prefill bucket.
VERIFY_FIRST_CALLS = 16


def _refuse(reason: str) -> None:
    if reason not in _REFUSED:
        _REFUSED.add(reason)
        print(f"[ISOEXEC-QKV-AG] REFUSED (stock gather stands): {reason}", flush=True)


def _capturing() -> bool:
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:  # pragma: no cover
        return False


def _layer_plan(attn) -> SubgroupPlan | None:
    """Per-instance admission, computed once and cached on the module. ``None`` == stock path."""
    cached = getattr(attn, "_ix_qkv_ag_plan", _UNSET)
    if cached is not _UNSET:
        return cached

    plan: SubgroupPlan | None = None
    reason: str | None = None
    try:
        import torch.distributed as dist

        p = _PLAN
        if p is None:
            reason = "no plan (install refused)"
        else:
            tp = getattr(getattr(attn, "pg_collection", None), "tp", None)
            ngroups = getattr(attn.config, "num_query_groups", None)
            world = int(getattr(attn, "world_size", 0))
            reason = admission_refusal(world=world, num_query_groups=ngroups)
            if reason is None:
                if tp is None:
                    reason = "layer has no tp process group"
                elif tuple(dist.get_process_group_ranks(tp)) != p.tp_ranks:
                    reason = f"layer tp ranks {tuple(dist.get_process_group_ranks(tp))} != planned " f"{p.tp_ranks}"
                elif world != p.world or int(ngroups) != p.num_query_groups:
                    reason = (
                        f"layer geometry (world={world}, ng={ngroups}) != planned "
                        f"(world={p.world}, ng={p.num_query_groups})"
                    )
                else:
                    # Ties megatron's slice bounds to subgroup membership: its own `idx`,
                    # recomputed from the live group, must be this rank's subgroup index.
                    idx = dist.get_rank(tp) // (world // int(ngroups))
                    if idx != p.group_index:
                        reason = f"megatron slice idx={idx} != subgroup index {p.group_index}"
                    else:
                        p.ok_groups[id(tp)] = tp
                        plan = p
    except Exception as e:  # pragma: no cover
        reason = f"{type(e).__name__}: {e}"

    if plan is None and reason:
        _refuse(reason)
    attn._ix_qkv_ag_plan = plan
    return plan


def _shim_all_gather_last_dim(input_, group=None):
    """Replacement for the name megatron's attention module imported. Diverts only when ARMED."""
    global _ARMED
    plan, _ARMED = _ARMED, None
    if plan is None or plan.sub_pg is None:
        return _ORIG_AG(input_, group)
    if id(group) not in plan.ok_groups:
        # Someone else's gather rode in on our arm. Never divert on a group we did not plan for.
        _refuse("armed call presented a tp group other than the planned one")
        _COUNTS["fallback"] += 1
        return _ORIG_AG(input_, group)

    if torch.is_grad_enabled() or input_.requires_grad:
        # megatron's name wraps an autograd function whose backward is a reduce-scatter over tp;
        # the subgroup path calls the plain functional gather and records no autograd node. Correct
        # under inference_mode, silently wrong anywhere grad is live -- so fall back.
        _refuse("grad is enabled: the subgroup path records no backward, so the stock autograd gather stands")
        _COUNTS["fallback"] += 1
        return _ORIG_AG(input_, group)

    local = int(input_.size(-1))
    reason = admission_refusal(world=plan.world, num_query_groups=plan.num_query_groups, local_width=local)
    if reason is not None:
        _refuse(reason)
        _COUNTS["fallback"] += 1
        return _ORIG_AG(input_, group)

    out = subgroup_gather_last_dim(input_, plan)
    _COUNTS["diverted"] += 1

    # One-shot live-operand cross-check per layer. Skipped under capture: a refusal cannot be
    # raised from inside a graph, and by capture time this has already passed.
    if _COUNTS["verified"] < VERIFY_FIRST_CALLS and not _capturing():
        size = plan.ranks_per_group * local
        lo = plan.group_index * size
        ref = _ORIG_AG(input_, group)[..., lo : lo + size]
        got = out[..., lo : lo + size]
        if not torch.equal(ref.contiguous().view(torch.uint8), got.contiguous().view(torch.uint8)):
            raise RuntimeError(
                f"[ISOEXEC-QKV-AG] REFUSING THE RUN: the subgroup gather disagrees with megatron's "
                f"own full gather on live operands (shape={tuple(input_.shape)}, "
                f"world={plan.world}, ng={plan.num_query_groups}, idx={plan.group_index}). The "
                f"contiguous rank-ordered sharding assumption does not hold for this build."
            )
        _COUNTS["verified"] += 1
        if _COUNTS["verified"] == 1:
            print(
                f"[ISOEXEC-QKV-AG] FIRST FIRE: subgroup gather ran on live operands "
                f"(shape={tuple(input_.shape)} -> kept {size} of {plan.world * local} columns) "
                f"and matched megatron's own full gather bit for bit.",
                flush=True,
            )
    return out


def _patched_get_query_key_value_tensors(self, *args, **kwargs):
    """Thin wrapper: compute admission from the module, arm the shim for exactly this call."""
    global _ARMED
    plan = _layer_plan(self)
    if plan is None:
        return _ORIG_METHOD(self, *args, **kwargs)
    _ARMED = plan
    _COUNTS["armed"] += 1
    try:
        return _ORIG_METHOD(self, *args, **kwargs)
    finally:
        _ARMED = None


def _candidate_layers(model):
    """SelfAttention modules that WOULD take megatron's gather branch."""
    from megatron.core.transformer.attention import SelfAttention

    out = []
    for m in model.modules():
        if isinstance(m, SelfAttention) and hasattr(m, "pg_collection"):
            ng = getattr(m.config, "num_query_groups", None)
            ws = int(getattr(m, "world_size", 0))
            if ng is not None and ws and ng < ws:
                out.append(m)
    return out


def _agree(local_ok: bool) -> bool:
    """AND a per-rank verdict across the DEFAULT process group (MIN over int).

    ``new_group`` is a default-group collective, so a rank that skips it while the others enter it
    hangs the engine. Every rank must therefore reach this whatever its own answer is -- callers run
    it BEFORE testing their own predicate, including on flag-OFF builds.
    """
    import torch.distributed as dist

    try:
        if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() < 2:
            return local_ok
        if not torch.cuda.is_available():  # pragma: no cover - CPU-only builds
            return local_ok
        t = torch.tensor([1 if local_ok else 0], device=torch.cuda.current_device(), dtype=torch.int32)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return bool(t.item())
    except Exception:  # pragma: no cover - never take the engine build down over an opt-in flag
        return local_ok


def install_engine_qkv_subgroup_ag(model=None, *, side: str = "ENGINE") -> bool:
    """Patch the QKV gather to a subgroup all-gather. Idempotent. No-op unless the flag is 1.

    Must be called at engine model-build time, on every rank, before any CUDA-graph capture: it
    creates process groups (a collective) and warms their communicators. Rank-uniformity is
    enforced, not assumed -- a flag set on only some ranks would deadlock, not merely slow down.
    """
    global _INSTALLED, _ORIG_METHOD, _ORIG_AG, _PLAN

    if _INSTALLED:
        return True

    try:
        import torch.distributed as dist
        from megatron.core.transformer import attention as _attn_mod
        from megatron.core.transformer.attention import SelfAttention
    except Exception as e:  # pragma: no cover
        if qkv_subgroup_ag_enabled():
            _refuse(f"megatron attention not importable ({type(e).__name__}: {e})")
        return False

    # Entered by EVERY rank, whatever its own flag says. Must precede the flag test.
    if not _agree(qkv_subgroup_ag_enabled()):
        if qkv_subgroup_ag_enabled():
            _refuse(
                "the flag is 1 on this rank but not on every rank of the default group; "
                "new_group is a default-group collective, so a split enablement would hang. The "
                "whole group takes the stock gather. Set it identically on every rank."
            )
        return False

    # Everything down to the agreement below computes a LOCAL verdict and returns from nowhere: any
    # rank walking away while the others enter `new_group` would hang them.
    local_reason: str | None = None
    layers: list = []
    world = ngroups = rpg = gidx = 0
    tp_ranks: tuple = ()
    tp_group = None
    try:
        if not (dist.is_available() and dist.is_initialized()):
            local_reason = "torch.distributed is not initialized"
        elif not torch.cuda.is_available():
            local_reason = "no CUDA device"
        elif _capturing():
            local_reason = "install reached under CUDA-graph capture (process groups cannot be built there)"
        elif model is None:
            local_reason = "no model handed to the installer; geometry must be read from real layers"
        else:
            layers = _candidate_layers(model)
            if not layers:
                local_reason = (
                    "no SelfAttention layer takes megatron's gather branch on this model "
                    "(num_query_groups >= tp world everywhere)"
                )
            else:
                geo = {
                    (
                        int(m.world_size),
                        int(m.config.num_query_groups),
                        tuple(dist.get_process_group_ranks(m.pg_collection.tp)),
                    )
                    for m in layers
                }
                if len(geo) != 1:
                    local_reason = f"gather-branch layers disagree on geometry: {sorted(geo)}"
                else:
                    world, ngroups, tp_ranks = geo.pop()
                    local_reason = admission_refusal(world=world, num_query_groups=ngroups)
                    # Building subgroups is only safe when the tp group IS the default world, so
                    # that every rank reaches the same `new_group` sequence.
                    if local_reason is None and tuple(tp_ranks) != tuple(range(dist.get_world_size())):
                        local_reason = (
                            f"tp group ranks {tp_ranks} are not the whole default world "
                            f"(size {dist.get_world_size()}); new_group is a default-group "
                            f"collective, so the subgroups cannot be built safely from here. "
                            f"ENGINE TP=world is the supported arm."
                        )
                    if local_reason is None:
                        tp_group = layers[0].pg_collection.tp
                        rpg = world // ngroups
                        gidx = dist.get_rank(tp_group) // rpg
    except Exception as e:  # pragma: no cover
        local_reason = f"admission raised ({type(e).__name__}: {e})"

    if not _agree(local_reason is None):
        _refuse(local_reason or "another rank of the default group refused; the group stays on the stock gather")
        return False

    sub_pg = None
    sub_ranks: tuple = ()
    build_reason: str | None = None
    try:
        sub_pg = build_qkv_subgroups(tp_group, ngroups)
        if sub_pg is None:
            build_reason = "this rank is not a member of any subgroup (impossible geometry)"
        else:
            sub_ranks = tuple(dist.get_process_group_ranks(sub_pg))
            expected = tuple(tp_ranks[gidx * rpg : (gidx + 1) * rpg])
            if sub_ranks != expected:
                build_reason = f"subgroup ranks {sub_ranks} != the slice's shard range {expected}"
            else:
                warm_subgroup(sub_pg)
    except Exception as e:
        build_reason = f"subgroup build/warm failed ({type(e).__name__}: {e})"

    # Last place a rank could walk off alone: patching on some ranks and not others leaves the
    # subgroup gather and the full gather as two halves of a rendezvous that never completes.
    if not _agree(build_reason is None):
        _refuse(build_reason or "another rank failed to build its subgroup; none of us patches")
        return False

    _PLAN = SubgroupPlan(
        world=world,
        num_query_groups=ngroups,
        ranks_per_group=rpg,
        group_index=gidx,
        tp_group=tp_group,
        sub_pg=sub_pg,
        tp_ranks=tuple(tp_ranks),
        sub_ranks=sub_ranks,
    )

    _ORIG_AG = _attn_mod.all_gather_last_dim_from_tensor_parallel_region
    _ORIG_METHOD = SelfAttention.get_query_key_value_tensors
    _attn_mod.all_gather_last_dim_from_tensor_parallel_region = _shim_all_gather_last_dim
    SelfAttention.get_query_key_value_tensors = _patched_get_query_key_value_tensors
    _INSTALLED = True

    full_hint = ""
    try:
        lin = layers[0].linear_qkv
        local_w = int(getattr(lin, "output_size_per_partition", 0))
        if local_w:
            full_hint = (
                f" fused-qkv width {local_w * world} -> shard {local_w}/rank, kept window "
                f"{(local_w * world) // ngroups} == {rpg} whole shards;"
            )
    except Exception:  # pragma: no cover
        pass

    print(
        f"[ISOEXEC-{side}-QKV-AG] ADMITTED: {len(layers)} SelfAttention layer(s) take megatron's "
        f"gather branch (world={world}, num_query_groups={ngroups}). Their {world}-rank "
        f"all-gather becomes a {rpg}-rank all-gather over subgroup {gidx} = ranks {sub_ranks} "
        f"(max_ctas={SUBGROUP_MAX_CTAS}), communicator built and WARMED before any capture."
        f"{full_hint} The kept columns are the same bytes in the same order -- an all-gather "
        f"performs no arithmetic and megatron's shards are contiguous and rank-ordered. First "
        f"call per shape cross-checks against megatron's own full gather and refuses the run on "
        f"any bit of disagreement.",
        flush=True,
    )
    return True


def revert_engine_qkv_subgroup_ag() -> None:
    global _INSTALLED, _PLAN
    if not _INSTALLED:
        return
    from megatron.core.transformer import attention as _attn_mod
    from megatron.core.transformer.attention import SelfAttention

    _attn_mod.all_gather_last_dim_from_tensor_parallel_region = _ORIG_AG
    SelfAttention.get_query_key_value_tensors = _ORIG_METHOD
    _INSTALLED = False
    _PLAN = None


def qkv_subgroup_status() -> dict:
    """Counts and refusal reasons, so "did it fire" is a number rather than an inference."""
    return {
        "enabled": qkv_subgroup_ag_enabled(),
        "installed": _INSTALLED,
        "world": _PLAN.world if _PLAN else None,
        "num_query_groups": _PLAN.num_query_groups if _PLAN else None,
        "subgroup_ranks": list(_PLAN.sub_ranks) if _PLAN else None,
        "refusals": sorted(_REFUSED),
        **_COUNTS,
    }
