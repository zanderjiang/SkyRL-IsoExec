"""bf16 wire for the IsoExec full-vocab log-softmax ``all_gather`` (``SKYRL_ISOEXEC_LOGPROB_GATHER_BF16_WIRE``).

The gather branch receives a widened bf16 tensor -- the model's logits are bf16 and only a widening and a
sequence-dim slice separate them from here -- so every element has 16 zero low mantissa bits and
``float32(bfloat16(x)) == x`` holds elementwise. Narrowing the shard before the gather and widening it back is
therefore a byte-count change and not an approximation: the collective's structure, group, element order, and
output layout are untouched. Because it changes the wire dtype, admission must be purely structural (env var,
caller-declared source dtype, shard dtype, group size) and unanimous across the group -- a rank that narrowed
while its peers did not would post a mismatched collective, so the drift probe raises rather than falling back.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist

ENV = "SKYRL_ISOEXEC_LOGPROB_GATHER_BF16_WIRE"
BANNER = "[ISOEXEC-LOGPROB-GATHER-WIRE]"

#: Re-prove on live operands every N admitted calls. The cadence is a counter, so every rank probes on the
#: same call and the MIN-reduce inside the probe cannot desynchronize the group.
PROBE_EVERY = 512

#: Dtypes whose widening to fp32 is exact and whose narrowing is that widening's exact inverse on
#: already-widened values.
_EXACT_WIRE_DTYPES = (torch.bfloat16, torch.float16)

_S = {
    "calls": 0,  # gather-branch calls that reached this module at all
    "served": 0,  # calls whose payload actually travelled narrowed
    "declined": 0,
    "decline_reason": "",
    "agreed": None,  # None = not yet decided; False = MIN-reduce said no, permanently off
    "structural_agreed": None,  # first-call TP vote, before either wire collective is selected
    "structural_contract": None,  # latest group, for the human-readable census
    "wire_bytes_saved": 0,
    "buffer_bytes_saved": 0,
    "probes": 0,
    "reported": 0,
    "bannered": False,
}

# Retain the group object beside its id, so Python id reuse cannot inherit a destroyed group's verdict.
_GROUP_STRUCTURAL = {}


def enabled() -> bool:
    """``SKYRL_ISOEXEC_LOGPROB_GATHER_BF16_WIRE``; default off."""
    return os.environ.get(ENV, "0") == "1"


def wire_stats() -> dict:
    """A copy of the census counters; ``served`` is the only evidence the narrow wire carried anything."""
    return dict(_S)


def _reset_for_test() -> None:
    _GROUP_STRUCTURAL.clear()
    _S.update(
        {
            "calls": 0,
            "served": 0,
            "declined": 0,
            "decline_reason": "",
            "agreed": None,
            "structural_agreed": None,
            "structural_contract": None,
            "wire_bytes_saved": 0,
            "buffer_bytes_saved": 0,
            "probes": 0,
            "reported": 0,
            "bannered": False,
        }
    )


def roundtrip_is_bitwise(x: torch.Tensor, wire_dtype: torch.dtype) -> torch.Tensor:
    """Device-side 0/1 int32 scalar: is ``float32(wire_dtype(x))`` the same tensor as ``x``?

    Compared through an int32 view rather than ``==``, because the claim is about bit patterns, which ``==``
    gets wrong for both ``-0.0`` and NaN. NaN-vs-NaN is the one accepted exception: narrowing does not preserve
    a NaN payload, but every consumer of this buffer propagates NaN, so the payload is unobservable. Returns a
    tensor rather than a bool so the caller decides where the single host sync happens.
    """
    rt = x.to(wire_dtype).to(torch.float32)
    same_bits = rt.contiguous().view(torch.int32) == x.contiguous().view(torch.int32)
    both_nan = rt.isnan() & x.isnan()
    ok = bool((same_bits | both_nan).all().item())
    return torch.tensor([1 if ok else 0], dtype=torch.int32, device=x.device)


def _min_reduce(ok: torch.Tensor, group) -> torch.Tensor:
    """MIN the verdict over the group so it is unanimous; a no-op when no process group is initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
    return ok


def _banner_once(on: bool, world: int, src_dtype: Optional[torch.dtype]) -> None:
    """Print once, on or off, so "never reached" and "reached and declined" stay distinguishable."""
    if _S["bannered"]:
        return
    _S["bannered"] = True
    if on:
        print(
            f"{BANNER} ON ({ENV}=1, world={world}, src_dtype={src_dtype}): the IsoExec full-vocab "
            f"all_gather ships {src_dtype} and widens back to fp32 on arrival. `full` is BIT-"
            f"IDENTICAL because the payload is a widened {src_dtype} -- narrow->exchange->widen is "
            f"the identity, not an approximation. The collective's structure, group, element order "
            f"and output layout are unchanged; only the bytes in flight are. Agreement is proved on "
            f"the WHOLE shard at call 1 (MIN-reduced, one dissenter disables every rank) and re-"
            f"proved every {PROBE_EVERY} admitted calls, where it RAISES instead of falling back -- "
            f"a rank-local fallback would put fp32 on the wire while peers put {src_dtype} there. "
            f"Judge engagement by served>0 below, never by this line.",
            flush=True,
        )
    else:
        print(
            f"{BANNER} OFF ({ENV}={os.environ.get(ENV, '0')}, world={world}, src_dtype={src_dtype}): "
            f"the IsoExec full-vocab all_gather ships fp32 -- {world}x[B, chunk, V/TP] on the wire "
            f"and a world-sized fp32 staging list, for values that are exactly bf16-representable. "
            f"=1 halves both. The gather itself is the forward gate and is not removable either way.",
            flush=True,
        )


def _report() -> None:
    """One census line at 1/2/4/8/... calls."""
    n = _S["calls"]
    if n < 1 or (n & (n - 1)) != 0 or n == _S["reported"]:
        return
    _S["reported"] = n
    print(
        f"{BANNER} CENSUS pid={os.getpid()} calls={n} served={_S['served']} "
        f"declined={_S['declined']} structural_agreed={_S['structural_agreed']} agreed={_S['agreed']} "
        f"wire_saved={_S['wire_bytes_saved'] / 1e9:.2f} GB "
        f"buffer_saved={_S['buffer_bytes_saved'] / 1e9:.2f} GB probes={_S['probes']}"
        + (f" last_decline={_S['decline_reason']}" if _S["decline_reason"] else ""),
        flush=True,
    )


def _decline(reason: str) -> None:
    _S["declined"] += 1
    _S["decline_reason"] = reason
    _report()


def _admit(shard: torch.Tensor, src_dtype: Optional[torch.dtype], world: int, group=None) -> Optional[torch.dtype]:
    """Structural verdict: the wire dtype to narrow to, or ``None`` to keep the fp32 wire.

    Every term is structural -- env var, caller-declared source dtype, shard dtype, group size, and the
    unanimous per-group latch. Nothing reads a tensor value, and ``served`` is left to the caller.
    """
    _S["calls"] += 1
    configured = os.environ.get(ENV)
    on = configured == "1"
    _banner_once(on, world, src_dtype)
    # An unset env var skips the vote entirely, so an application that never configured this lever does not
    # acquire two scalar TP rendezvous; an explicit value (including 0) does participate.
    if configured is None:
        reason = f"{ENV} is unset (default 0): fp32 wire"
        _decline(reason)
        return None
    reason = ""
    wire = src_dtype
    if not on:
        reason = f"{ENV}=0: fp32 wire"
    if world <= 1:
        reason = reason or f"world={world}: no collective to narrow"
    elif not (dist.is_available() and dist.is_initialized()):
        reason = reason or "torch.distributed is not initialized: cannot prove TP unanimity"
    elif shard.dtype is not torch.float32:
        reason = reason or f"shard dtype {shard.dtype} is not float32: nothing to narrow from"
    elif src_dtype not in _EXACT_WIRE_DTYPES:
        reason = reason or (
            f"src_dtype={src_dtype} is not one of {_EXACT_WIRE_DTYPES}: the payload is not a widened "
            "low-precision tensor, so narrowing it would ROUND, not be the identity"
        )

    # Vote the immutable structural signature once per process group, with MIN+MAX so two individually
    # admissible but different wire dtypes cannot look unanimous. Chunk shape is deliberately not latched: a
    # final short chunk is normal.
    group_key = id(group)
    group_state = _GROUP_STRUCTURAL.get(group_key)
    if world > 1 and dist.is_available() and dist.is_initialized() and group_state is None:
        dtype_code = {
            None: 0,
            torch.bfloat16: 1,
            torch.float16: 2,
            torch.float32: 3,
        }
        local_contract = (
            int(not reason),
            dtype_code.get(src_dtype, -1),
            dtype_code.get(shard.dtype, -1),
            int(world),
        )
        low = torch.tensor(local_contract, dtype=torch.int64, device=shard.device)
        high = low.clone()
        dist.all_reduce(low, op=dist.ReduceOp.MIN, group=group)
        dist.all_reduce(high, op=dist.ReduceOp.MAX, group=group)
        unanimous = bool(low[0].item()) and torch.equal(low, high)
        _GROUP_STRUCTURAL[group_key] = {
            "group": group,
            "agreed": unanimous,
            "contract": local_contract if unanimous else None,
            "value_agreed": None,
            "served": 0,
        }
        group_state = _GROUP_STRUCTURAL[group_key]
        _S["structural_agreed"] = unanimous
        _S["structural_contract"] = local_contract if unanimous else None
        if not unanimous and not reason:
            reason = "a TP peer declined or reported a different immutable wire contract"
    elif group_state is not None:
        _S["structural_agreed"] = group_state["agreed"]
        _S["structural_contract"] = group_state["contract"]
        dtype_code = {None: 0, torch.bfloat16: 1, torch.float16: 2, torch.float32: 3}
        current = (
            int(not reason),
            dtype_code.get(src_dtype, -1),
            dtype_code.get(shard.dtype, -1),
            int(world),
        )
        if group_state["agreed"] and current != group_state["contract"]:
            # Scoring and policy recompute share this group with different payload provenance, so the
            # source dtype is a per-call eligibility field, not an immutable group property: a locally
            # ineligible call takes the fp32 fallback rather than tripping a drift gate.
            if reason:
                _decline(reason)
                return None
            raise RuntimeError(
                f"{BANNER} incompatible eligible wire dtype after TP admission: "
                f"first={group_state['contract']!r} now={current!r}"
            )

    if reason or (group_state is not None and not group_state["agreed"]):
        _decline(reason or "first structural agreement failed on at least one rank")
        return None
    if _S["agreed"] is False:
        _decline("first-call agreement failed on at least one rank: bf16 wire permanently disabled")
        return None
    return wire


def gather_full_vocab(
    shard: torch.Tensor,
    *,
    group,
    world: int,
    src_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """All-gather ``shard`` across ``group`` and return the ``[..., V]`` fp32 concatenation.

    The return value is bit-identical whether or not the narrow wire engages: the same ops run in the same
    order, on narrowed tensors with one widening appended, and the concatenation order and last-dim layout are
    unchanged.

    Args:
        shard: this rank's fp32 vocabulary shard, ``[..., V/world]``.
        group: the tensor-parallel process group.
        world: ``group``'s size (passed in; the caller already computed it).
        src_dtype: the dtype the caller's tensor had BEFORE it was widened to fp32, or ``None`` when
            the caller cannot make that statement. ``None`` is the fail-closed default: without a
            declared source dtype there is no identity argument and the fp32 wire stands.
    """
    shard = shard.contiguous()
    wire = _admit(shard, src_dtype, world, group)

    if wire is None:
        # fp32 is the only safe common fallback wire; widening bf16/fp16 to it is exact.
        fallback_shard = shard if shard.dtype is torch.float32 else shard.to(torch.float32)
        gathered = [torch.empty_like(fallback_shard) for _ in range(world)]
        dist.all_gather(gathered, fallback_shard, group=group)
        full = torch.cat(gathered, dim=-1)
        del gathered, fallback_shard
        return full

    # First admitted call: prove the identity on the whole shard and MIN-reduce the verdict. A dissent here
    # is recoverable, so this stage falls back on every rank at once rather than raising.
    group_state = _GROUP_STRUCTURAL[id(group)]
    if group_state["value_agreed"] is None:
        ok = _min_reduce(roundtrip_is_bitwise(shard, wire), group)
        group_state["value_agreed"] = bool(ok.item())
        _S["agreed"] = group_state["value_agreed"]
        _S["probes"] += 1
        if not group_state["value_agreed"]:
            print(
                f"{BANNER} DISABLED: float32({wire}(x)) != x on at least one rank's shard, so "
                f"narrow->exchange->widen is NOT the identity on this payload and the fp32 wire "
                f"stands on EVERY rank (the verdict is MIN-reduced, hence unanimous, hence safe). "
                f"This means the tensor reaching the gather branch is not a widened {wire} -- check "
                f"for arithmetic between the model's logits and `.to(torch.float32)`.",
                flush=True,
            )
            _decline("first-call agreement failed")
            fallback_shard = shard if shard.dtype is torch.float32 else shard.to(torch.float32)
            gathered = [torch.empty_like(fallback_shard) for _ in range(world)]
            dist.all_gather(gathered, fallback_shard, group=group)
            full = torch.cat(gathered, dim=-1)
            del gathered, fallback_shard
            return full
    elif group_state["served"] > 0 and group_state["served"] % PROBE_EVERY == 0:
        # Drift probe: raises, because by now the group has agreed to narrow and a unilateral fallback
        # would be a mismatched collective.
        ok = _min_reduce(roundtrip_is_bitwise(shard, wire), group)
        _S["probes"] += 1
        if not bool(ok.item()):
            raise RuntimeError(
                f"{BANNER} DRIFT: at group-served call {group_state['served']} the gather branch's operand is no "
                f"longer exactly {wire}-representable, so the {wire} wire would ROUND it -- the "
                f"canary-v10 round-before-sum defect, on the forward gate's own collective. "
                f"REFUSING TO FALL BACK: the group already agreed to narrow, so a rank-local "
                f"fp32 wire would be a mismatched collective, not a safe state. Set {ENV}=0 to run."
            )

    narrowed = shard.to(wire)
    gathered = [torch.empty_like(narrowed) for _ in range(world)]
    dist.all_gather(gathered, narrowed, group=group)
    packed = torch.cat(gathered, dim=-1)
    # Free the staging list before allocating the fp32 buffer: that ordering is what lowers the peak.
    del gathered, narrowed
    full = packed.to(torch.float32)
    del packed

    shard_bytes = shard.numel() * shard.element_size()
    _S["served"] += 1
    group_state["served"] += 1
    _S["wire_bytes_saved"] += (world - 1) * (shard_bytes // 2)
    _S["buffer_bytes_saved"] += world * (shard_bytes // 2)
    _report()
    return full
