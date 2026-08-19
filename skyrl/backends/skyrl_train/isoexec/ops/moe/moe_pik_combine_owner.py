"""Owner-computes MoE pik combine for the engine no-gather path.

The landed pik-fc2 combine all-reduces the full permuted expert output ``[P, H]`` fp32 over the ETP
group, so every rank computes every row's tree root and k-sums and rounds every token -- a
``world``-fold redundant compute and peer read. The per-token expression is rank-independent::

    out[t] = bf16_round( SUM_{j=0..k-1} treeroot[ rows[t,j] ] )    (fp32 sum, ascending expert)
    treeroot[r] = fp32 balanced-tree( partial_0[r], ..., partial_{world-1}[r] )   (left = lower rank)

so the T tokens are partitioned into ``world`` contiguous slices and each OWNER computes the full
chain for its own tokens, reading only its tokens' k rows from each peer. The finished bf16 rows are
then exchanged by a symmetric-memory push -- the combine kernel writes each owned row straight into
every peer's symm output buffer and one barrier completes the exchange -- which keeps NCCL out of the
hot path and leaves the whole combine CUDA-graph capturable.

BITS ARE UNCHANGED throughout: same peer partials, same tree order, same ascending-expert fp32 adds,
same single bf16 round. Only the executing rank and the transport moved.

Gated by SKYRL_ISOEXEC_MOE_PIK_OWNER_COMBINE (default off) and only ever acts on dispatchers the
engine build marked ``_isoexec_engine_nogather`` under pik-fc2 nogather; the trainer's alltoall path is
untouched, and any unsupported shape falls back to the landed combine. Shapes are static (T from
restore_shape, host-known; H, world, k constant), the barrier is stream-ordered, and there is no host
sync in the hot path.

Two optional refinements sit on top, both integer/plumbing-only:

  * SKYRL_ISOEXEC_MOE_OC_ROWS32 builds ``my_rows`` [s, k] int32 as a VIEW of the counting-sort kernel's
    own [T, k] int32 output when ``T % world == 0``, instead of int64-argsort -> pad -> slice ->
    contiguous -> int32 cast. Same permutation, same dtype the kernel already consumed.
  * SKYRL_ISOEXEC_MOE_OC_WIRE_STAGE lets the fc2 leaf-tree GEMM store the wire bytes directly into this
    module's symmetric staging buffer. At ``n_leaves == 1`` the leaftree kernel's fp32 store is a
    lossless promotion of the bf16 leaf value, so ``_stage`` can detect staged bytes by pointer
    compare and skip the copy entirely. It is rank-local by design: granting or refusing the sink
    changes no rendezvous structure, so ranks may disagree freely and a refused rank just pays the
    copy.

SKYRL_ISOEXEC_PIK_FUSED_OWNER_COMBINE (default off) folds the two barriers into the push kernel using
pik's own barrier emitter, three launches to one. See ``_oc_core`` for why fusion cannot move a bit
and ``_oc_admit`` for the per-shape live re-proof.
"""

from __future__ import annotations

import functools
import importlib.util
import os
import pathlib
import sys
from contextlib import contextmanager
from contextvars import ContextVar

import torch
import torch.distributed as dist

# The tree over `world` peers is pik's balanced tree (left = lower rank), emitted straight-line so the
# evaluation order is pinned and matches pik.codegen / the landed tree_all_reduce.

_CACHE_DIR = pathlib.Path(os.environ.get("PIK_CACHE", pathlib.Path.home() / ".cache" / "pik")) / "owner_combine"


def _ensure_canonical_pik():
    """Load PIK under its single supported runtime identity before importing a child module."""
    from ..collectives.pik_bootstrap import ensure_pik

    return ensure_pik()


def _load(name: str, src: str):
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / f"{name}.py"
    if not path.exists() or path.read_text() != src:
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(src)
        tmp.replace(path)  # atomic: concurrent ranks must not read a half-written file
    spec = importlib.util.spec_from_file_location(f"owner_combine.{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.kernel


def _tree_src(world: int, in_bf16: bool) -> str:
    """Straight-line balanced tree over in0..in{world-1} at scalar offset `base`, into t0 (fp32).
    Post-order, left = lower rank -- identical order to pik.codegen._emit / plan.combine_order."""
    cast = ".to(tl.float32)" if in_bf16 else ""
    leaf = "        t{d} = tl.load(in{lo} + base, mask=hmask, other=0.0)" + cast + "\n"

    def emit(lo, hi, depth, out):
        if hi - lo == 1:
            out.append(leaf.format(d=depth, lo=lo))
            return
        mid = (lo + hi) // 2
        emit(lo, mid, depth, out)
        emit(mid, hi, depth + 1, out)
        out.append(f"        t{depth} = t{depth} + t{depth + 1}\n")

    body: list[str] = []
    emit(0, world, 0, body)
    return "".join(body)


_TMPL = """
import triton
import triton.language as tl

@triton.jit
def kernel({ins}, rows_ptr, out_ptr, T, H, K: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    hb = tl.program_id(1)
    hoff = hb * BLOCK + tl.arange(0, BLOCK)
    hmask = hoff < H
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for j in tl.static_range(K):
        row = tl.load(rows_ptr + t * K + j)
        base = row * H + hoff
{tree}        acc = acc + t0
    tl.store(out_ptr + t * H + hoff, acc.to(tl.bfloat16), mask=hmask)
"""


# Push variant: identical arithmetic to the kernel above (same tree, same k-sum, same single bf16
# round), but instead of writing one local [s,H] slice followed by an NCCL all_gather, each rank pushes
# its finished bf16 rows into EVERY peer's symmetric output buffer at global offset (out_base + t)*H,
# and one symm-mem barrier completes every rank's output. This removes NCCL from the combine hot path
# so the whole combine is CUDA-graph capturable. Only the transport moved; bits are unchanged.


def _oc_core(world: int, k: int, in_bf16: bool) -> str:
    """The float-touching body of the owner push: offsets, the tree, the k-sum, the round, the stores.

    Shared VERBATIM by ``_PUSH_TMPL`` (barrier + kernel + barrier, 3 launches) and ``_FUSED_PUSH_TMPL``
    (both barriers folded into the kernel, 1 launch). Because this string is the only part of either
    kernel that touches a float and both templates paste it unchanged, the fused variant is
    bitwise-equal to the unfused one by construction. ``k`` is unread here -- K is a ``tl.constexpr``
    and the loop is a ``static_range`` -- but it keys the generated variant.
    """
    stores = "".join(f"    tl.store(out{i} + goff, res, mask=hmask)\n" for i in range(world))
    return f"""    t = tl.program_id(0)
    hb = tl.program_id(1)
    hoff = hb * BLOCK + tl.arange(0, BLOCK)
    hmask = hoff < H
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for j in tl.static_range(K):
        row = tl.load(rows_ptr + t * K + j)
        base = row * H + hoff
{_tree_src(world, in_bf16)}        acc = acc + t0
    res = acc.to(tl.bfloat16)
    goff = (out_base + t) * H + hoff
{stores}"""


_PUSH_TMPL = """
import triton
import triton.language as tl

@triton.jit
def kernel({ins}, rows_ptr, {outs}, out_base, T, H, K: tl.constexpr, BLOCK: tl.constexpr):
{core}"""


@functools.cache
def _owner_kernel(world: int, k: int, in_bf16: bool):
    ins = ", ".join(f"in{i}" for i in range(world))
    src = _TMPL.format(ins=ins, tree=_tree_src(world, in_bf16))
    tag = f"_owner_c{world}_k{k}{'_bf16' if in_bf16 else ''}"
    return _load(tag, src)


def _push_src(world: int, k: int, in_bf16: bool) -> str:
    ins = ", ".join(f"in{i}" for i in range(world))
    outs = ", ".join(f"out{i}" for i in range(world))
    return _PUSH_TMPL.format(ins=ins, outs=outs, core=_oc_core(world, k, in_bf16))


@functools.cache
def _owner_push_kernel(world: int, k: int, in_bf16: bool):
    tag = f"_owner_push_c{world}_k{k}{'_bf16' if in_bf16 else ''}"
    return _load(tag, _push_src(world, k, in_bf16))


# Exact shared+routed owner composition: deliberately a SEPARATE kernel family from
# ``_owner_push_kernel``, so the shipped routed owner stays byte-for-byte unchanged. It evaluates the
# existing two branches at their existing rounding points and only co-schedules them:
#
#   routed = bf16(ascending_k_sum(tree(rank routed leaves)))
#   shared = bf16(bf16(tree(rank shared leaves)) * gate_bf16)
#   result = bf16(routed + shared)
#
# ``gate_bf16`` is the already-rounded result of torch.sigmoid(shared_gate_logits); keeping the
# sigmoid outside this kernel makes the proof independent of its implementation.
# ``enable_fp_fusion=False`` is load-bearing: without it LLVM may reach through a bf16 cast and
# contract the shared multiply with the final add, deleting a rounding point.


def _named_tree_src(world: int, in_bf16: bool, *, inp: str, tmp: str, base: str, indent: str) -> str:
    """The same balanced peer tree as :func:`_tree_src`, with disjoint SSA names, so one program can
    carry both the routed and the shared tree. Both use the same recursive emitter, left = lower rank.
    """
    cast = ".to(tl.float32)" if in_bf16 else ""
    leaf = f"{indent}{tmp}{{d}} = tl.load({inp}{{lo}} + {base}, mask=hmask, other=0.0){cast}\n"

    def emit(lo, hi, depth, out):
        if hi - lo == 1:
            out.append(leaf.format(d=depth, lo=lo))
            return
        mid = (lo + hi) // 2
        emit(lo, mid, depth, out)
        emit(mid, hi, depth + 1, out)
        out.append(f"{indent}{tmp}{depth} = {tmp}{depth} + {tmp}{depth + 1}\n")

    body: list[str] = []
    emit(0, world, 0, body)
    return "".join(body)


def _shared_owner_core(world: int, k: int, routed_bf16: bool, shared_bf16: bool) -> str:
    """Float-touching body of the exact shared+routed owner push.

    The explicit bf16 temporaries are the contract, not storage conveniences: they pin the three
    existing roundings (shared root, routed root, shared gate product) before the final bf16 add.
    """
    stores = "".join(f"    tl.store(out{i} + goff, result, mask=hmask)\n" for i in range(world))
    routed_tree = _named_tree_src(world, routed_bf16, inp="rin", tmp="rt", base="rbase", indent="        ")
    shared_tree = _named_tree_src(world, shared_bf16, inp="sin", tmp="st", base="sbase", indent="    ")
    return f"""    t = tl.program_id(0)
    hb = tl.program_id(1)
    hoff = hb * BLOCK + tl.arange(0, BLOCK)
    hmask = hoff < H
    routed_acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for j in tl.static_range(K):
        row = tl.load(rows_ptr + t * K + j)
        rbase = row * H + hoff
{routed_tree}        routed_acc = routed_acc + rt0
    routed_bf = routed_acc.to(tl.bfloat16)
    global_t = out_base + t
    sbase = global_t * H + hoff
{shared_tree}    shared_root_bf = st0.to(tl.bfloat16)
    gate_bf = tl.load(gate_ptr + global_t).to(tl.bfloat16)
    shared_bf = (shared_root_bf.to(tl.float32) * gate_bf.to(tl.float32)).to(tl.bfloat16)
    result = (routed_bf.to(tl.float32) + shared_bf.to(tl.float32)).to(tl.bfloat16)
    goff = global_t * H + hoff
{stores}"""


_SHARED_PUSH_TMPL = """
import triton
import triton.language as tl

@triton.jit
def kernel({rins}, rows_ptr, {sins}, gate_ptr, {outs}, out_base, T, H,
           K: tl.constexpr, BLOCK: tl.constexpr):
{core}"""


def _shared_push_src(world: int, k: int, routed_bf16: bool, shared_bf16: bool) -> str:
    rins = ", ".join(f"rin{i}" for i in range(world))
    sins = ", ".join(f"sin{i}" for i in range(world))
    outs = ", ".join(f"out{i}" for i in range(world))
    return _SHARED_PUSH_TMPL.format(
        rins=rins,
        sins=sins,
        outs=outs,
        core=_shared_owner_core(world, k, routed_bf16, shared_bf16),
    )


@functools.cache
def _shared_owner_push_kernel(world: int, k: int, routed_bf16: bool, shared_bf16: bool):
    tag = (
        f"_shared_owner_push_c{world}_k{k}"
        f"_r{'bf16' if routed_bf16 else 'fp32'}_s{'bf16' if shared_bf16 else 'fp32'}"
    )
    return _load(tag, _shared_push_src(world, k, routed_bf16, shared_bf16))


# Fused push: the same push kernel with the leading and trailing barriers folded in.
#
# The barrier is NOT re-implemented here -- ``pik.codegen._fused_barrier`` emits it, the same function
# that emits pik's fused all-reduce barrier, so the flag layout, release/acquire scopes, epoch
# comparison and ``tl.debug_barrier()`` placement are inherited verbatim.
#
# The one thing NOT inherited is the flag index. pik's kernels are 1-D, so the flag slot is
# ``program_id(0)``. This grid is 2-D -- (owned tokens, H blocks) -- and two blocks sharing a slot
# would race on the local epoch counter and publish one epoch for two arrivals. So the slot is the
# FLAT block id, ``program_id(0) * NHB + program_id(1)``, and the flag stride NB must be at least
# ``s * NHB``.
#
# The spin cannot deadlock when the grid exceeds what the GPU holds resident: a block publishes to
# every peer before it waits on any peer, and the published flag is a monotone epoch in symmetric
# memory that persists after the block retires. A resident block b of rank R waiting on block b of
# peer P is satisfied whether P's block b is running, queued or long retired. Ranks need only run the
# same grid, which SPMD gives; an oversized grid costs throughput, not liveness.
_FUSED_PUSH_TMPL = """
import triton
import triton.language as tl

@triton.jit
def kernel({ins}, rows_ptr, {outs}, {fls}, self_fl, ep_ptr,
           my_slot0, my_slot1, wbase1, NB,
           out_base, T, H, K: tl.constexpr, BLOCK: tl.constexpr,
           NHB: tl.constexpr, WORLD: tl.constexpr):
    pid = tl.program_id(0) * NHB + tl.program_id(1)
    one = tl.arange(0, 1)
    lane = tl.arange(0, WORLD)

{lead}    tl.debug_barrier()   # every peer has arrived before any thread reads peer memory

{core}{trail}"""


def _fused_push_src(world: int, k: int, in_bf16: bool) -> str:
    _ensure_canonical_pik()
    import pik.codegen as CG  # type: ignore

    ins = ", ".join(f"in{i}" for i in range(world))
    outs = ", ".join(f"out{i}" for i in range(world))
    fls = ", ".join(f"fl{i}" for i in range(world))
    lead = CG._fused_barrier(0, world, "0", "pid")
    trail = "\n    tl.debug_barrier()   # this block's pushes precede its release\n" + CG._fused_barrier(
        1, world, "wbase1", "NB + pid"
    )
    return _FUSED_PUSH_TMPL.format(
        ins=ins, outs=outs, fls=fls, lead=lead, core=_oc_core(world, k, in_bf16), trail=trail
    )


@functools.cache
def _owner_push_fused_kernel(world: int, k: int, in_bf16: bool):
    tag = f"_owner_push_fused_c{world}_k{k}{'_bf16' if in_bf16 else ''}"
    return _load(tag, _fused_push_src(world, k, in_bf16))


# Staging: reuse pik's symmetric-memory double-buffered pool -- stage this rank's partial, barrier,
# read peers over NVLink. Falls back to NCCL all_gather when symm-mem is unavailable.


def _sym_pool(n, device, group, dtype):
    _ensure_canonical_pik()
    from pik.allreduce import (
        _sym_pool as pik_sym_pool,  # type: ignore
    )

    return pik_sym_pool(n, device, group, dtype)


# Dedicated double-buffered symmetric OUTPUT pool for the push exchange. It MUST NOT share the input
# pool: that pool's key includes dtype, so at bf16 wire a bf16 output pool would hash to the same
# buffers the kernel reads as peers and the push would clobber them mid-read. Keyed on (device, world)
# and always bf16, so it can never alias the input staging. Double-buffering keeps a rank from
# overwriting a buffer peers may still be reading: barrier N+1 is stream-ordered after every peer
# issued its read of buffer N, so buffer N is free again only at N+2.
_OUT_SYM: dict = {}

# Shared FC2 leaves must never alias routed FC2 leaves even when both are bf16.  A single owner
# kernel reads both populations after one group-wide barrier; aliasing them would turn that kernel
# into a read-after-overwrite race.  Keep an independent double-buffered allocation, keyed on group
# membership rather than world size so two same-sized process groups cannot silently share it.
_SHARED_IN_SYM: dict = {}


def _out_sym_pool(n, device, group):
    _ensure_canonical_pik()
    from pik.allreduce import _SymPool  # type: ignore

    key = (device, dist.get_world_size(group))
    pool = _OUT_SYM.get(key)
    if pool is None or pool.cap < n:
        pool = _SymPool(n, device, group, torch.bfloat16)
        _OUT_SYM[key] = pool
    return pool


def _shared_in_sym_pool(n, device, group, dtype):
    _ensure_canonical_pik()
    from pik.allreduce import _SymPool  # type: ignore

    key = (device, _group_key(group), dtype)
    pool = _SHARED_IN_SYM.get(key)
    if pool is None or pool.cap < n:
        pool = _SymPool(n, device, group, dtype)
        _SHARED_IN_SYM[key] = pool
    return pool


def _p2p_ok(group) -> bool:
    try:
        from torch.distributed import _symmetric_memory  # noqa: F401

        return dist.is_initialized() and dist.get_world_size(group) > 1
    except Exception:  # noqa: BLE001
        return False


# Wire-stage sink: producer-side staging (see the module docstring). The fc2 leaf-tree kernel asks for
# a bf16 [P,H] view of the current-phase staging buffer and stores the wire bytes straight into it; the
# exchange then detects the staged pointer and skips its cast-copy. The group cannot be known at the
# fc2 call site (the dispatcher owns it), so the sink is refused until the first owner combine records
# it. Every refusal is fail-open and names itself in _WIRE_REASONS.
_WIRE_ENV = "SKYRL_ISOEXEC_MOE_OC_WIRE_STAGE"

# ``None`` is a VALID group -- torch.distributed's default process group -- so it cannot serve as the
# "not recorded yet" sentinel or the sink would refuse forever on those configurations. The sentinel is
# a private object no caller can supply, keeping "recorded" distinct from "recorded as the default".
_WIRE_UNRECORDED = object()
_WIRE_GROUP: dict = {"group": _WIRE_UNRECORDED, "device": None}
_WIRE_COUNTS = {"grants": 0, "staged_hits": 0, "copy_falls": 0, "refusals": 0, "promotes": 0}

# Named refusal accounting. The sink's rank-local refusals are not equivalent -- "no combine has run
# yet" is expected once per process, "world != leaf count" is a configuration statement, and "the pool
# would grow under capture" is a warmup bug -- so each names itself in the counters.
_WIRE_REFUSALS: dict = {}
_WIRE_REASONS = {
    "group_not_recorded": "no owner combine has run in this process yet, so the sink does not know "
    "which process group (hence which symmetric pool) the fc2 partial will cross. Expected exactly "
    "once, at the first MoE layer of the first forward; if it repeats, the owner combine is not "
    "running at all and the wire-stage lever is INERT",
    "device_mismatch": "the recorded owner-combine device is not the device this fc2 ran on",
    "no_symm_p2p": "symmetric memory / P2P is unavailable on the recorded group (world<2, or "
    "torch.distributed._symmetric_memory did not import)",
    "bf16_wire_off": "SKYRL_ISOEXEC_MOE_PIK_BF16_WIRE=0, so the combine's wire is fp32 and there is "
    "no bf16 buffer for the leaftree to store into",
    "world_ne_leaves": "world != the pik plan's num_leaves, so this combine ships an fp32 wire "
    "(the bf16 wire is admissible only at world == G)",
    "pool_growth_under_capture": "the symmetric staging pool would have to GROW, and this call is "
    "inside a CUDA-graph capture where a symmetric allocation is a hang rather than an error. Run "
    "every shape once eagerly before capture",
    "exception": "the sink raised; see the traceback printed once above",
}


def wire_stage_enabled() -> bool:
    return os.environ.get(_WIRE_ENV, "1") == "1"


def wire_stage_counts() -> dict:
    """Per-process tally: ``staged_hits`` counts exchanges that consumed producer-staged bytes (a
    pointer match), ``copy_falls`` those that paid the copy anyway, and ``refused_by_reason`` names
    which rank-local refusal path declined (see ``_WIRE_REASONS``).
    """
    return {**_WIRE_COUNTS, "refused_by_reason": dict(_WIRE_REFUSALS)}


def _wire_refuse(reason: str, detail: str = ""):
    """Count and name a rank-local refusal, then fall open. Always returns None (the bit-identical
    fp32 path).
    """
    _WIRE_COUNTS["refusals"] += 1
    _WIRE_REFUSALS[reason] = _WIRE_REFUSALS.get(reason, 0) + 1
    if _WIRE_REFUSALS[reason] == 1:
        print(
            f"[ISOEXEC-MOE] owner_wire_sink REFUSED [{reason}]"
            f"{': ' + detail if detail else ''}. {_WIRE_REASONS.get(reason, '')}. The owner combine "
            "falls back to its stage copy: bit-identical, ~15.7us/site slower. Read "
            "moe_pik_combine_owner.wire_stage_counts().",
            flush=True,
        )
    return None


def owner_wire_sink(P: int, H: int, device):
    """A bf16 [P,H] view of the current-phase symmetric staging buffer, or None to use the fp32 path.

    ``None`` is always safe: the exchange's pointer compare simply misses and the copy runs. Refuses
    under CUDA-graph capture if the pool would have to grow, because a symmetric allocation mid-capture
    hangs rather than erroring. Every refusal is named in ``wire_stage_counts()['refused_by_reason']``.

    Pool sharing: the bf16 _SymInPool is keyed (device, world, wire dtype), so another sym_partial user
    could share it. That is safe by order plus phase -- every dense site both stages and REDUCES (which
    flips the pool) strictly before this layer's fc2, so an interleaved user leaves the exchange's
    stage() on a different phase than the sink's and the pointer compare misses into the bit-identical
    copy. A caller that staged into this pool between fc2 and the combine WITHOUT reducing would be
    silently wrong; no such path exists today, and any new sym_partial caller must re-check this.

    Pool growth is a collective while the grant decision is rank-local, but disagreement cannot
    deadlock: a granting rank grows the pool in the sink and a refusing rank grows the same pool at its
    exchange's ``_stage`` moments later, and neither needs the other in between.
    """
    if not (wire_stage_enabled() and owner_combine_enabled()):
        return None
    group, dev = _WIRE_GROUP["group"], _WIRE_GROUP["device"]
    if group is _WIRE_UNRECORDED:
        return _wire_refuse("group_not_recorded")
    if dev != device:
        return _wire_refuse("device_mismatch", f"recorded {dev}, asked for {device}")
    if not _p2p_ok(group):
        return _wire_refuse("no_symm_p2p")
    try:
        from ...ops.collectives.pik_tp_invariant import get_plan
        from .moe_batch_invariant import _pik_bf16_wire_enabled

        if not _pik_bf16_wire_enabled():
            return _wire_refuse("bf16_wire_off")
        world, leaves = dist.get_world_size(group), get_plan().num_leaves
        if world != leaves:
            return _wire_refuse("world_ne_leaves", f"world={world}, num_leaves={leaves}")
        _ensure_canonical_pik()
        from pik.allreduce import _SYM_IN, _capturing  # type: ignore

        key = (device, world, torch.bfloat16)
        pool = _SYM_IN.get(key)
        if (pool is None or pool.cap < P * H) and _capturing():
            return _wire_refuse(
                "pool_growth_under_capture", f"need {P * H} elems, have {None if pool is None else pool.cap}"
            )
        inp, _ = _sym_pool(P * H, device, group, torch.bfloat16).stage()
    except Exception:  # noqa: BLE001 -- any surprise here must degrade to the copy path, loudly once
        if "exception" not in _WIRE_REFUSALS:
            import traceback

            print("[ISOEXEC-MOE] owner_wire_sink raised:\n" + traceback.format_exc(), flush=True)
        return _wire_refuse("exception")
    _WIRE_COUNTS["grants"] += 1
    if _WIRE_COUNTS["grants"] == 1:
        print(
            f"[ISOEXEC-MOE] owner_wire_sink GRANTED (first grant, [{P},{H}] bf16 on {device}): the fc2 "
            "leaf-tree may now store the owner combine's wire bytes straight into the current-phase "
            "symmetric staging buffer. ENGAGEMENT is this line PLUS wire_stage_counts()['staged_hits']>0 "
            "-- the exchange's own pointer match -- never the flag being exported.",
            flush=True,
        )
    v = inp[: P * H].view(P, H)
    return v


# Flag pool for the in-kernel barrier: pik's ``_FlagPool`` object, reused as-is, but NOT pik's
# registry. ``pik.allreduce._flag_pool`` keys on (device, world_size), which would (a) give a
# same-sized but different-membership group a pool rendezvoused against the wrong peers -- this path
# runs on ``tp_ep_group``, which only happens to equal the TP group at EP=1 -- and (b) let the two
# paths' very different block counts thrash each other's NB.
_OC_FLAGS: dict = {}
_OC_FLAG_KEEP: list = []


def _oc_flag_pool(nb: int, device, group):
    """Flag pool for THIS group, grown geometrically, never freed while a kernel may hold it."""
    _ensure_canonical_pik()
    from pik.allreduce import _capturing, _FlagPool  # type: ignore

    key = (device, _group_key(group))
    pool = _OC_FLAGS.get(key)
    if pool is None or pool.nb < nb:
        if _capturing():
            # Allocating and rendezvous-ing inside a capture is illegal, and a half-built symmetric
            # allocation hangs rather than erroring. Refuse visibly.
            raise RuntimeError(
                f"fused owner-combine: the flag pool must grow to {nb} blocks, but this is inside a "
                "CUDA graph capture. Every shape must run once EAGERLY before capture (vLLM's "
                "cudagraph_num_of_warmups=1), so the pools are sized before the graph is recorded."
            )
        want = max(nb, 2 * pool.nb if pool is not None else 0, 1024)
        if pool is not None:
            _OC_FLAG_KEEP.append(pool)  # a launched kernel may still hold the old pointers
        pool = _FlagPool(want, device, group)
        _OC_FLAGS[key] = pool
    return pool


# Fused-path enablement, mirroring ``pik.allreduce``'s admission machinery. RANK UNIFORMITY IS A
# CORRECTNESS REQUIREMENT: the unfused path's barriers are separate symm-mem rendezvous kernels while
# the fused path's live inside the push kernel on a different flag buffer, so two ranks disagreeing
# about which they run HANG rather than run slowly -- one parked at a barrier no peer will reach, the
# other spinning on a flag no peer will publish. Every input to the decision is therefore either
# group-wide by construction (world, T, H, k, wire, BLOCK) or made group-wide by a collective: the env
# flag via ``_oc_group_on`` and each per-shape verdict via ``_agree``.
_FUSED_ENV = "SKYRL_ISOEXEC_PIK_FUSED_OWNER_COMBINE"


def _env_on(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("", "0", "false", "no", "off")


_FUSED_ON = _env_on(_FUSED_ENV)

# Per (world, s, nhb, k, wire) admission verdict. True = proven bitwise-equal to the unfused
# barrier+push+barrier exchange on LIVE operands at this exact launch geometry.
_FUSED_ADMIT: dict = {}
_FUSED_COUNTS = {"calls": 0, "admitted": 0, "rejected": 0, "errors": 0, "capture_skips": 0, "peer_vetoes": 0}
_FUSED_FIRST_REJECT: dict = {}
_FUSED_GROUP: dict = {}


def fused_owner_enabled() -> bool:
    """Whether this process is allowed to try the fused single-launch owner exchange."""
    return _FUSED_ON


def set_fused_owner_enabled(on: bool) -> None:
    """Test/bench hook. Flipping this does NOT bypass admission: every shape still proves."""
    global _FUSED_ON
    _FUSED_ON = bool(on)


def fused_owner_counts() -> dict:
    """Per-process tally for ``pik_status()``.

    ``calls`` counts dispatches onto the fused path, i.e. Python-side entries. Under CUDA graphs that
    is the capture-time (and warmup) entry, not the replay -- a replay re-runs the kernel without
    re-entering this module.
    """
    return {
        **_FUSED_COUNTS,
        "enabled": _FUSED_ON,
        "shapes": {str(k): v for k, v in _FUSED_ADMIT.items()},
        "first_reject": dict(_FUSED_FIRST_REJECT) or None,
    }


def _group_key(group):
    try:
        return tuple(dist.get_process_group_ranks(group))
    except Exception:  # noqa: BLE001 -- default group, or an older torch
        return id(group)


def _oc_group_on(group, device) -> bool:
    """Has this GROUP agreed the fused exchange may be attempted? One collective, once, ever."""
    _ensure_canonical_pik()
    from pik.allreduce import _agree, _capturing  # type: ignore

    key = _group_key(group)
    got = _FUSED_GROUP.get(key)
    if got is not None:
        return got
    if _capturing():
        _FUSED_COUNTS["capture_skips"] += 1
        return False
    got = _agree(_FUSED_ON, group, device)
    _FUSED_GROUP[key] = got
    if got != _FUSED_ON:
        print(
            f"[ISOEXEC-MOE] the FUSED owner-combine is DISABLED for process group {key}: this rank "
            f"has {_FUSED_ENV}={'on' if _FUSED_ON else 'off'} but at least one peer disagrees, and "
            "running the two launch structures side by side would deadlock. Set the flag "
            "identically on every rank of a TP group.",
            flush=True,
        )
    elif got:
        print(
            f"[ISOEXEC-MOE] FUSED owner-combine ENABLED and agreed by all "
            f"{dist.get_world_size(group) if dist.is_initialized() else 1} ranks of group {key}. "
            "Each launch geometry still has to prove itself bitwise against the "
            "barrier+push+barrier reference before it is used.",
            flush=True,
        )
    return got


def _oc_reject(key, reason: str, detail: str = "") -> None:
    """Refuse the fused exchange for this geometry, LOUDLY and once. Never silently."""
    global _FUSED_FIRST_REJECT
    _FUSED_ADMIT[key] = False
    _FUSED_COUNTS["rejected"] += 1
    if _FUSED_FIRST_REJECT:
        return
    _FUSED_FIRST_REJECT = {"key": str(key), "reason": reason, "detail": detail}
    print(
        f"[ISOEXEC-MOE] WARNING: the FUSED owner-combine was REFUSED at {key} -- {reason}. {detail} "
        "The result is unaffected (this geometry falls back to the barrier+push+barrier reference, "
        "which is what every frozen bit was proven against), but the launch-count saving is not "
        "being taken here. Read moe_pik_combine_owner.fused_owner_counts().",
        flush=True,
    )


def _oc_admit(key, ex, group, device) -> bool:
    """Prove the fused exchange bitwise-equal to the reference, at this geometry, on these operands.

    Runs both paths back to back on the live staged partial, reference first. Each exchange flips both
    double-buffered pools exactly once, so two exchanges leave both pools on the phase they started on
    and the caller's real call runs afterwards as if admission had not happened.

    The compare is an int16 bit-pattern compare on the bf16 result, not allclose: a reassociated tree
    is usually still allclose.

    The verdict is AGREED, not decided locally -- the local answer is AND-ed across the group, so a
    single dissenting rank puts the whole group on the reference path. Every exit goes through
    ``_agree``, including the exception path, so a rank whose kernel failed to launch still takes part
    in the collective its peers are running.
    """
    _ensure_canonical_pik()
    from pik.allreduce import _agree  # type: ignore

    reason = detail = ""
    ref = ex.unfused().clone()
    local_ok = True
    try:
        got = ex.fused().clone()
    except Exception as e:  # noqa: BLE001
        _FUSED_COUNTS["errors"] += 1
        local_ok, reason, detail = False, "the fused kernel raised", repr(e)
    else:
        if ref.shape != got.shape or ref.dtype != got.dtype:
            local_ok, reason = False, "shape/dtype mismatch"
            detail = f"ref={ref.shape}/{ref.dtype} got={got.shape}/{got.dtype}"
        elif not torch.equal(_bits(ref), _bits(got)):
            local_ok, reason = False, "BIT PATTERNS DIFFER"
            detail = f"{int((_bits(ref) != _bits(got)).sum().item())}/{ref.numel()} elements"

    agreed = _agree(local_ok, group, device)
    if not agreed:
        if local_ok:
            _FUSED_COUNTS["peer_vetoes"] += 1
            reason = "a PEER rank refused it"
            detail = "this rank's own compare passed; the group runs the reference path together"
        _oc_reject(key, reason, detail)
        return False
    _FUSED_ADMIT[key] = True
    _FUSED_COUNTS["admitted"] += 1
    if _FUSED_COUNTS["admitted"] == 1:
        # Positive banner, once: silence on the success path is indistinguishable from the flag never
        # reaching this process, or from every shape being first seen under capture and skipped.
        print(
            f"[ISOEXEC-MOE] FUSED owner-combine ADMITTED (first geometry {key}): this geometry now "
            "runs as ONE launch instead of barrier+push+barrier. Bit patterns were compared against "
            "the reference exchange on live operands and agreed across the group. A decode trace "
            "should show c10d::symmetric_memory::barrier_kernel drop by 2 calls per MoE layer per "
            "step. Read moe_pik_combine_owner.fused_owner_counts() for the served-call count.",
            flush=True,
        )
    return True


def _bits(t: torch.Tensor) -> torch.Tensor:
    """A tensor's BIT PATTERN as an integer tensor. ``torch.equal`` on floats is not a bit compare:
    it says -0.0 == 0.0 and NaN != NaN, so a reduction that flipped a sign bit passes."""
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    if t.dtype in (torch.bfloat16, torch.float16):
        return t.view(torch.int16)
    return t


def _served_tick(key) -> None:
    """Served-call count on the fused path. Loud early, then rare; never an install banner."""
    _FUSED_COUNTS["calls"] += 1
    n = _FUSED_COUNTS["calls"]
    if n <= 2 or n % 4096 == 0:
        print(
            f"[ISOEXEC-MOE-FUSED-OWNER] pid={os.getpid()} served={n} dispatches on the FUSED "
            f"owner-combine (one launch, no separate barriers); last geometry {key}",
            flush=True,
        )


class _Exchange:
    """The push exchange in its two launch structures, sharing every operand.

    Both members stage the same partial into the same double-buffered symmetric pool, read the same
    peer buffers, launch a kernel generated from the same ``_oc_core`` string, and copy out the same
    slice. The only difference is whether the two rendezvous are separate symm-mem barrier kernels
    around the push, or the push kernel's own prologue and epilogue.
    """

    def __init__(self, partial, my_rows, s, Tp, T, H, k, group, wire_bf16, block):
        self.partial, self.my_rows = partial, my_rows
        self.s, self.Tp, self.T, self.H, self.k = s, Tp, T, H, k
        self.group, self.wire_bf16, self.block = group, wire_bf16, block
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.dev = partial.device
        self.dt = torch.bfloat16 if wire_bf16 else torch.float32
        self.P, _ = partial.shape

    def _stage(self):
        """Stage this rank's partial and hand back the peer views plus the output pool.

        A dtype-converting ``copy_`` rounds RNE exactly like ``.to(dt)``, so the fp32 partial lands in
        the bf16 wire buffer in one pass. If the fc2 leaf-tree already stored the wire bytes into this
        very buffer (``owner_wire_sink``), the pointer compare detects it and the copy is skipped; any
        mismatch falls open to the bit-identical copy.
        """
        pool = _sym_pool(self.P * self.H, self.dev, self.group, self.dt)
        inp, h_in = pool.stage()
        if self.partial.dtype == self.dt and self.partial.data_ptr() == inp.data_ptr():
            _WIRE_COUNTS["staged_hits"] += 1  # producer already staged these exact bytes
            if _WIRE_COUNTS["staged_hits"] == 1:
                # A grant only means the sink handed out a buffer; this means the exchange found the
                # producer's bytes already in it and skipped the copy.
                print(
                    f"[ISOEXEC-MOE] owner combine CONSUMED producer-staged wire bytes (first hit, "
                    f"[{self.P},{self.H}] bf16, rank {self.rank}/{self.world}): the fc2 leaf-tree's "
                    "store IS the staging, so the [P,H] cast-copy before the lead barrier is gone.",
                    flush=True,
                )
        else:
            if self.partial.dtype == torch.bfloat16 and wire_stage_enabled():
                _WIRE_COUNTS["copy_falls"] += 1
            inp[: self.P * self.H].view(self.P, self.H).copy_(self.partial)
        peers = [h_in.get_buffer(r, (self.P * self.H,), self.dt) for r in range(self.world)]
        opool = _out_sym_pool(self.Tp * self.H, self.dev, self.group)
        out_sym, h_out = opool.stage()
        outs = [h_out.get_buffer(r, (self.Tp * self.H,), torch.bfloat16) for r in range(self.world)]
        return pool, h_in, peers, opool, out_sym, h_out, outs

    def _finish(self, pool, opool, out_sym):
        """Copy out [:T] to a fresh tensor, dropping the world-padding and detaching from the
        symmetric buffer so the double-buffered pool may reuse it next layer. CUDA-graph safe: shapes
        are static and the copy is stream-ordered.
        """
        pool.flip()
        result = out_sym[: self.Tp * self.H].view(self.Tp, self.H)[: self.T].contiguous()
        opool.flip()
        return result

    @property
    def grid(self):
        import triton

        return (self.s, triton.cdiv(self.H, self.block))

    def unfused(self):
        """The reference exchange: barrier, push, barrier -- three launches, and the default."""
        pool, h_in, peers, opool, out_sym, h_out, outs = self._stage()
        h_in.barrier()  # every rank's partial visible to peers
        pk = _owner_push_kernel(self.world, self.k, self.wire_bf16)
        pk[self.grid](
            *peers,
            self.my_rows,
            *outs,
            self.rank * self.s,
            self.Tp,
            self.H,
            K=self.k,
            BLOCK=self.block,
            num_warps=4,
        )
        h_out.barrier()  # every rank's owned slice has landed in every peer's out buffer
        return self._finish(pool, opool, out_sym)

    def fused(self):
        """The same exchange as ONE launch: the two barriers are the kernel's prologue and epilogue.

        The double-buffer safety argument still holds with the leading barrier at the front of the
        kernel rather than behind it:

          INPUT pool. Rank R's ``copy_`` for call N+1 writes buffer (N+1)%2 and is stream-ordered after
          R's call-N kernel. That kernel cleared its leading wait, so every peer had arrived at call N,
          hence retired its call-(N-1) kernel and its reads of buffer (N-1)%2 == (N+1)%2. The buffer R
          is about to overwrite is one nobody is reading -- the N+2 rule ``_SymPool`` states.

          OUTPUT pool. R's call-N+2 kernel pushes into peers' out buffer phase N%2 only after its
          leading wait clears, which requires every peer to have retired its N+1 kernel and therefore
          completed its call-N copy-out of that phase.

          TRAILING. R's kernel retires only once all its blocks passed the trailing wait, and block b
          passes only after every peer's block b published region 1, which a peer does after
          ``tl.debug_barrier()`` following its stores. So every peer push into R's out buffer has
          landed, and is visible through the release/acquire sys-scope pair, before R's copy-out.

        Nothing here touches a value: the stores are to the same addresses, each output element is
        written by exactly one (rank, block) pair, and the arithmetic is the same ``_oc_core`` string.
        """
        import triton

        nhb = triton.cdiv(self.H, self.block)
        fp = _oc_flag_pool(self.s * nhb, self.dev, self.group)
        pool, _h_in, peers, opool, out_sym, _h_out, outs = self._stage()
        # No leading barrier here: the kernel's own prologue is the barrier. The staging copy above is
        # still ordered before it, since a kernel block runs only after the copy kernel retired.
        pk = _owner_push_fused_kernel(self.world, self.k, self.wire_bf16)
        pk[self.grid](
            *peers,
            self.my_rows,
            *outs,
            *fp.peers,
            fp.mine,
            fp.epoch,
            fp.my_slot(0),
            fp.my_slot(1),
            fp.wait_base(1),
            fp.nb,
            self.rank * self.s,
            self.Tp,
            self.H,
            K=self.k,
            BLOCK=self.block,
            NHB=nhb,
            WORLD=self.world,
            num_warps=4,
        )
        return self._finish(pool, opool, out_sym)


class _SharedExchange:
    """Isolated exact shared+routed exchange.

    Both input populations are staged before one group-wide symmetric-memory barrier. The barrier is an
    acquire/release rendezvous for the process group, not for one allocation, so invoking it through
    the routed handle publishes the earlier stream-ordered copies into *both* symmetric allocations.
    Kept separate from :class:`_Exchange` so the shipped owner path's pool phase, source and launch
    sequence cannot move.
    """

    def __init__(
        self,
        routed_partial,
        shared_partial,
        gate_score,
        my_rows,
        s,
        Tp,
        T,
        H,
        k,
        group,
        routed_bf16,
        shared_bf16,
        block,
    ):
        self.routed_partial = routed_partial
        self.shared_partial = shared_partial
        self.gate_score = gate_score
        self.my_rows = my_rows
        self.s, self.Tp, self.T, self.H, self.k = s, Tp, T, H, k
        self.group = group
        self.routed_bf16, self.shared_bf16 = routed_bf16, shared_bf16
        self.block = block
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        self.dev = routed_partial.device
        self.P = routed_partial.shape[0]

    @property
    def grid(self):
        import triton

        return (self.s, triton.cdiv(self.H, self.block))

    def run(self):
        rdt = torch.bfloat16 if self.routed_bf16 else torch.float32
        sdt = torch.bfloat16 if self.shared_bf16 else torch.float32

        rpool = _sym_pool(self.P * self.H, self.dev, self.group, rdt)
        rinp, rh = rpool.stage()
        if not (self.routed_partial.dtype == rdt and self.routed_partial.data_ptr() == rinp.data_ptr()):
            rinp[: self.P * self.H].view(self.P, self.H).copy_(self.routed_partial)

        spool = _shared_in_sym_pool(self.T * self.H, self.dev, self.group, sdt)
        sinp, sh = spool.stage()
        if not (self.shared_partial.dtype == sdt and self.shared_partial.data_ptr() == sinp.data_ptr()):
            sinp[: self.T * self.H].view(self.T, self.H).copy_(self.shared_partial)

        rpeers = [rh.get_buffer(r, (self.P * self.H,), rdt) for r in range(self.world)]
        speers = [sh.get_buffer(r, (self.T * self.H,), sdt) for r in range(self.world)]
        opool = _out_sym_pool(self.Tp * self.H, self.dev, self.group)
        out_sym, oh = opool.stage()
        outs = [oh.get_buffer(r, (self.Tp * self.H,), torch.bfloat16) for r in range(self.world)]

        # Both staging copies precede this on the same CUDA stream. One group rendezvous publishes
        # both allocations; there is intentionally no shared-input-specific second barrier.
        rh.barrier()
        pk = _shared_owner_push_kernel(self.world, self.k, self.routed_bf16, self.shared_bf16)
        pk[self.grid](
            *rpeers,
            self.my_rows,
            *speers,
            self.gate_score,
            *outs,
            self.rank * self.s,
            self.Tp,
            self.H,
            K=self.k,
            BLOCK=self.block,
            num_warps=4,
            enable_fp_fusion=False,
        )
        oh.barrier()

        rpool.flip()
        spool.flip()
        result = out_sym[: self.Tp * self.H].view(self.Tp, self.H)[: self.T].contiguous()
        opool.flip()
        return result


# Grid crossover for the in-kernel barrier. That barrier publishes to every peer ONCE PER BLOCK, so a
# rank issues grid*world remote atomics per region where the device-wide symm-mem barrier issues
# world. Its cost is therefore roughly linear in the grid and flat in payload, while the device-wide
# barrier pair it replaces does not grow with the grid at all. Fusion only pays at a very small grid;
# the gate below defaults to 8 blocks. This is a REFUSAL, not a tuning knob -- no production decode
# geometry is near it, so the flag is safe to set on and simply will not engage, saying so in the log.
_MAX_BLOCKS_ENV = "SKYRL_ISOEXEC_PIK_FUSED_OWNER_MAX_BLOCKS"


def _max_fused_blocks() -> int:
    try:
        return max(1, int(os.environ.get(_MAX_BLOCKS_ENV, "8")))
    except ValueError:
        return 8


def _owner_push_exchange(ex):
    """Dispatch only. ``_Exchange.unfused`` is the default and the reference; ``_Exchange.fused`` is
    used only for a geometry proven bitwise-identical to it on live operands in this process and small
    enough that fusing is not a regression.

    Every branch here is taken identically on every rank of the group. The one input that is not
    invariant by construction, the env flag, is settled by ``_oc_group_on`` before it is read, by a
    collective every rank enters; the grid gate is a pure function of rank-invariant shape facts.
    """
    _ensure_canonical_pik()
    from pik.allreduce import _capturing  # type: ignore

    if not _oc_group_on(ex.group, ex.dev):
        return ex.unfused()

    import triton

    key = (ex.world, ex.s, triton.cdiv(ex.H, ex.block), ex.k, "bf16" if ex.wire_bf16 else "fp32")
    verdict = _FUSED_ADMIT.get(key)
    if verdict is None:
        blocks = ex.s * triton.cdiv(ex.H, ex.block)
        cap = _max_fused_blocks()
        if blocks > cap:
            _oc_reject(
                key,
                "the grid exceeds the MEASURED fusion crossover",
                f"{blocks} blocks > {cap}: the in-kernel handshake costs ~2.0 us PER BLOCK (flat in "
                f"payload, measured 8xH100), so fusing this geometry would cost ~{blocks * 2.0:.0f} us "
                f"against a device-wide barrier pair's 10-35 us. Raise {_MAX_BLOCKS_ENV} only with a "
                "measurement that says otherwise.",
            )
            return ex.unfused()
        if ex.world < 2 or (ex.world & (ex.world - 1)):
            _oc_reject(key, "world size is not a power of two", f"world={ex.world}")
            return ex.unfused()
        if _capturing():
            # A compare needs a host sync, which capture forbids. Do NOT record a verdict: the
            # geometry gets another chance the next time it is seen eagerly.
            _FUSED_COUNTS["capture_skips"] += 1
            return ex.unfused()
        try:
            verdict = _oc_admit(key, ex, ex.group, ex.dev)
        except RuntimeError as e:  # the flag pool refused to grow (capture); never a wrong answer
            _oc_reject(key, "the flag pool could not be sized", repr(e))
            return ex.unfused()
    if not verdict:
        return ex.unfused()
    _served_tick(key)
    return ex.fused()


def _owner_combine(partial, rows, T, k, group, wire_bf16, block: int = 1024):
    """Owner-computes combine.

    ``partial`` is this rank's [P,H] leaf-subtree fc2 partial (fp32); ``rows`` is the [T,k] global
    permuted-row index per token, ascending expert. Returns bf16 [T,H], bit-identical on every rank.
    """
    import triton

    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    P, H = partial.shape
    dev = partial.device
    dt = torch.bfloat16 if wire_bf16 else torch.float32
    _WIRE_GROUP["group"], _WIRE_GROUP["device"] = group, dev  # the sink needs the group; record it

    # shape-only ownership: contiguous slice, pad T up to a multiple of world
    s = -(-T // world)  # ceil, host-known
    Tp = s * world
    if rows.dtype == torch.int32 and T % world == 0:
        # The counting-sort kernel already emitted [T,k] int32 contiguous, and at a divisible T the
        # pad is vacuous, so this rank's slice is a contiguous VIEW -- no fill, copy or cast.
        my_rows = rows[rank * s : (rank + 1) * s]
    else:
        rows_p = rows.new_zeros(Tp, k)
        rows_p[:T] = rows
        my_rows = rows_p[rank * s : (rank + 1) * s].contiguous().to(torch.int32)  # [s,k]

    grid = (s, triton.cdiv(H, block))

    if _p2p_ok(group):
        # symm-mem PUSH exchange, no NCCL in the hot path: the kernel reads peers, computes each owned
        # token's tree + k-sum, and writes the finished bf16 row into every peer's symmetric output
        # buffer at its global offset. Two barriers complete it, or zero on the admitted fused path.
        return _owner_push_exchange(_Exchange(partial, my_rows, s, Tp, T, H, k, group, wire_bf16, block))

    # NCCL correctness fallback (no NVLink / no symm-mem): all-gather partials, run the kernel over
    # local copies, all-gather the finals. Same tree/k-sum/round, bit-identical to the P2P push.
    out_local = torch.empty(s, H, dtype=torch.bfloat16, device=dev)
    kern = _owner_kernel(world, k, wire_bf16)
    staged = partial.to(dt) if partial.dtype != dt else partial
    buf = torch.empty(world, P * H, dtype=dt, device=dev)
    dist.all_gather_into_tensor(buf, staged.contiguous().view(-1), group=group)
    peers = [buf[r] for r in range(world)]
    kern[grid](*peers, my_rows, out_local, s, H, K=k, BLOCK=block, num_warps=4)
    full = torch.empty(Tp, H, dtype=torch.bfloat16, device=dev)
    dist.all_gather_into_tensor(full, out_local, group=group)
    return full[:T]


def _owner_shared_combine(
    routed_partial,
    shared_partial,
    gate_score,
    rows,
    T,
    k,
    group,
    routed_bf16,
    shared_bf16,
    block: int = 1024,
):
    """Isolated shared+routed composition; ``None`` means unsupported, never a changed answer.

    Args mirror :func:`_owner_combine`, plus the shared fc2 rank-subtree partial ``[T,H]`` and the
    already-rounded bf16 shared gate ``[T]``/``[T,1]``.
    """
    if not _p2p_ok(group):
        return None
    world = dist.get_world_size(group)
    if world < 2 or (world & (world - 1)):
        return None
    if routed_partial.ndim != 2 or shared_partial.ndim != 2:
        return None
    P, H = routed_partial.shape
    if T <= 0 or P != T * k or tuple(shared_partial.shape) != (T, H):
        return None
    gate = gate_score.reshape(-1)
    if gate.numel() != T or gate.dtype != torch.bfloat16 or not gate.is_contiguous():
        return None
    rdt = torch.bfloat16 if routed_bf16 else torch.float32
    sdt = torch.bfloat16 if shared_bf16 else torch.float32
    if routed_partial.dtype not in (rdt, torch.float32) or shared_partial.dtype not in (
        sdt,
        torch.float32,
    ):
        return None

    rank = dist.get_rank(group)
    s = -(-T // world)
    Tp = s * world
    if rows.dtype == torch.int32 and T % world == 0:
        my_rows = rows[rank * s : (rank + 1) * s]
    else:
        rows_p = rows.new_zeros(Tp, k)
        rows_p[:T] = rows
        my_rows = rows_p[rank * s : (rank + 1) * s].contiguous().to(torch.int32)

    ex = _SharedExchange(
        routed_partial,
        shared_partial,
        gate,
        my_rows,
        s,
        Tp,
        T,
        H,
        k,
        group,
        routed_bf16,
        shared_bf16,
        block,
    )
    return ex.run()


# Default-off production admission for the shared+routed composition.
_SHARED_OWNER_ENV = "SKYRL_ISOEXEC_MOE_SHARED_OWNER_FUSION"
_SHARED_OWNER_ADMIT: dict = {}
_SHARED_OWNER_GROUP: dict = {}
_SHARED_OWNER_PROFILE_PROVISIONED: dict = {}
_SHARED_OWNER_COUNTS = {
    "calls": 0,
    "admitted": 0,
    "rejected": 0,
    "fallbacks": 0,
    "capture_skips": 0,
    "payloads": 0,
    "profile_provisions": 0,
    "profile_deferred": 0,
    "profile_capture_skips": 0,
}
_SHARED_OWNER_FIRST_REJECT = None
_SHARED_OWNER_MEMORY_PROFILE: ContextVar[bool] = ContextVar("isoexec_shared_owner_memory_profile", default=False)


@contextmanager
def shared_owner_memory_profile_scope():
    """Mark vLLM's peak-memory measurement without changing any model arithmetic.

    The candidate's persistent symmetric pools must be counted against vLLM's KV-cache budget, but
    its admission evaluates both compositions at once and that one-time overlap must not be charged
    as a permanent peak. Inside this scope, dispatch provisions every persistent pool at the real
    profile shape, serves the current composition, and leaves admission unresolved; the first ordinary
    eager warmup after the scope does the live bit compare.
    """
    token = _SHARED_OWNER_MEMORY_PROFILE.set(True)
    try:
        yield
    finally:
        _SHARED_OWNER_MEMORY_PROFILE.reset(token)


def shared_owner_memory_profile_active() -> bool:
    return bool(_SHARED_OWNER_MEMORY_PROFILE.get())


def shared_owner_fusion_enabled() -> bool:
    """New exact shared+routed composition.  Default OFF; read at instance-install time."""
    return _env_on(_SHARED_OWNER_ENV, "0")


def shared_owner_group_enabled(group, device) -> bool:
    """Agree the install decision before any rank changes its shared-expert launch structure.

    A one-sided instance rebind would make one rank enter the shared fc2 tree while its peers enter the
    fused owner exchange -- a hang, not a slow fallback -- so an env value never becomes an install
    decision until every rank in the TP/ETP group has voted for it.
    """
    if group is None or not dist.is_initialized():
        return False
    key = _group_key(group)
    verdict = _SHARED_OWNER_GROUP.get(key)
    if verdict is not None:
        return verdict
    _ensure_canonical_pik()
    from pik.allreduce import _agree, _capturing  # type: ignore

    local = shared_owner_fusion_enabled()
    if _capturing():
        return False
    verdict = _agree(local, group, device)
    _SHARED_OWNER_GROUP[key] = verdict
    if verdict != local:
        print(
            f"[ISOEXEC-MOE] shared+routed owner instance seam DISABLED for group {key}: this rank "
            f"has {_SHARED_OWNER_ENV}={'on' if local else 'off'} but at least one peer disagrees. "
            "No instance was rebound; set the flag identically on every TP rank.",
            flush=True,
        )
    return verdict


def shared_owner_fusion_counts() -> dict:
    """Per-process tally for the shared+routed owner fusion."""
    return {
        **_SHARED_OWNER_COUNTS,
        "enabled": shared_owner_fusion_enabled(),
        "group_agreement": {str(k): v for k, v in _SHARED_OWNER_GROUP.items()},
        "shapes": {str(k): v for k, v in _SHARED_OWNER_ADMIT.items()},
        "first_reject": _SHARED_OWNER_FIRST_REJECT,
    }


def _shared_owner_reject(key, reason: str) -> None:
    global _SHARED_OWNER_FIRST_REJECT
    _SHARED_OWNER_ADMIT[key] = False
    _SHARED_OWNER_COUNTS["rejected"] += 1
    if _SHARED_OWNER_FIRST_REJECT is None:
        _SHARED_OWNER_FIRST_REJECT = {"key": str(key), "reason": reason}
        print(
            f"[ISOEXEC-MOE] shared+routed owner fusion REFUSED at {key}: {reason}. The exact "
            "separate shared-tree + routed-owner + final-add composition remains in charge. Read "
            "moe_pik_combine_owner.shared_owner_fusion_counts().",
            flush=True,
        )


def _shared_owner_reference(routed_partial, shared_partial, gate_score, rows, T, k, group, routed_bf16):
    """The current arithmetic, with an isolated transport for the shared tree.

    The production path reduces shared fc2 *before* routed fc2 writes the owner wire; at this later
    seam the routed bytes already occupy pik's current symmetric phase, so the P2P shared tree would
    overwrite them. pik's NCCL helper moves the shared leaves through a distinct buffer while
    evaluating the identical balanced tree.
    """
    _ensure_canonical_pik()
    from pik.allreduce import _tree_all_reduce_nccl  # type: ignore

    shared_root = _tree_all_reduce_nccl(shared_partial, group, None, None)
    shared_out = shared_root * gate_score.reshape(T, 1)
    routed_out = _owner_combine(routed_partial, rows, T, k, group, routed_bf16)
    return routed_out + shared_out


def _shared_owner_provision(
    routed_partial,
    shared_partial,
    gate_score,
    T,
    k,
    group,
    routed_bf16,
    shared_bf16,
) -> tuple[bool, str]:
    """Reserve every persistent candidate allocation without executing candidate arithmetic.

    Mirrors :func:`_owner_shared_combine`'s shape/transport guards. Constructing the pools does not
    stage, flip, publish or read a value, so the reference stays the only expression evaluated during
    vLLM's memory-profile forward. Pool growth must happen INSIDE that profile: moving it after KV
    sizing would under-account persistent memory.
    """
    if not _p2p_ok(group):
        return False, "symmetric-memory P2P is unavailable"
    world = dist.get_world_size(group)
    if world < 2 or (world & (world - 1)):
        return False, f"world={world} is not a supported power-of-two group"
    if routed_partial.ndim != 2 or shared_partial.ndim != 2:
        return False, "routed/shared partials must both be rank-2"
    P, H = routed_partial.shape
    if T <= 0 or P != T * k or tuple(shared_partial.shape) != (T, H):
        return False, (
            f"shape mismatch routed={tuple(routed_partial.shape)} " f"shared={tuple(shared_partial.shape)} T={T} k={k}"
        )
    gate = gate_score.reshape(-1)
    if gate.numel() != T or gate.dtype != torch.bfloat16 or not gate.is_contiguous():
        return False, "shared gate must be contiguous bf16[T]"
    rdt = torch.bfloat16 if routed_bf16 else torch.float32
    sdt = torch.bfloat16 if shared_bf16 else torch.float32
    if routed_partial.dtype not in (rdt, torch.float32) or shared_partial.dtype not in (
        sdt,
        torch.float32,
    ):
        return False, (
            f"unsupported partial dtypes routed={routed_partial.dtype}/{rdt} " f"shared={shared_partial.dtype}/{sdt}"
        )

    s = -(-T // world)
    Tp = s * world
    _sym_pool(P * H, routed_partial.device, group, rdt)
    _shared_in_sym_pool(T * H, routed_partial.device, group, sdt)
    _out_sym_pool(Tp * H, routed_partial.device, group)
    return True, ""


def _shared_owner_dispatch(
    routed_partial,
    shared_partial,
    gate_score,
    rows,
    T,
    k,
    group,
    routed_bf16,
    shared_bf16,
):
    """Live admission and dispatch.  Every refusal returns the exact current composition."""
    _SHARED_OWNER_COUNTS["payloads"] += 1
    world = dist.get_world_size(group)
    key = (
        world,
        T,
        routed_partial.shape[-1],
        k,
        str(routed_partial.dtype),
        str(shared_partial.dtype),
    )
    verdict = _SHARED_OWNER_ADMIT.get(key)
    if verdict is False:
        _SHARED_OWNER_COUNTS["fallbacks"] += 1
        return _shared_owner_reference(routed_partial, shared_partial, gate_score, rows, T, k, group, routed_bf16)

    _ensure_canonical_pik()
    from pik.allreduce import _agree, _capturing  # type: ignore

    if verdict is None:
        if shared_owner_memory_profile_active():
            # vLLM's profile forward is immediately followed by CUDA-graph memory estimation, and
            # neither phase may validate: the eager profile pass provisions the maximum pools and
            # capture may only reuse them, never perform a host-visible agreement or compare.
            #
            # Evaluate the reference BEFORE provisioning. Its NCCL transport has temporary
            # gather/reduce buffers, and allocating candidate pools first would make those transients
            # overlap the persistent candidate floor -- the artificial peak this scope exists to remove.
            ref = _shared_owner_reference(
                routed_partial,
                shared_partial,
                gate_score,
                rows,
                T,
                k,
                group,
                routed_bf16,
            )
            if _capturing():
                _SHARED_OWNER_COUNTS["profile_capture_skips"] += 1
            elif key not in _SHARED_OWNER_PROFILE_PROVISIONED:
                local_ok, reason = True, ""
                try:
                    local_ok, reason = _shared_owner_provision(
                        routed_partial,
                        shared_partial,
                        gate_score,
                        T,
                        k,
                        group,
                        routed_bf16,
                        shared_bf16,
                    )
                except Exception as exc:  # noqa: BLE001 -- peer agreement below keeps rank parity
                    local_ok, reason = False, f"persistent-pool provisioning raised {exc!r}"
                provisioned = _agree(local_ok, group, routed_partial.device)
                if not provisioned:
                    _shared_owner_reject(
                        key,
                        reason or "a peer rank vetoed persistent-pool provisioning",
                    )
                    _SHARED_OWNER_COUNTS["fallbacks"] += 1
                    return ref
                _SHARED_OWNER_PROFILE_PROVISIONED[key] = True
                _SHARED_OWNER_COUNTS["profile_provisions"] += 1
                print(
                    f"[ISOEXEC-MOE] shared+routed owner fusion PROFILE-PROVISIONED at {key}: "
                    "every persistent routed/shared/output symmetric pool is reserved inside "
                    "vLLM's peak-memory measurement; the exact current composition is served "
                    "there and live bitwise admission remains pending for the first ordinary "
                    "eager warmup outside that measurement.",
                    flush=True,
                )
            _SHARED_OWNER_COUNTS["profile_deferred"] += 1
            return ref
        if _capturing():
            _SHARED_OWNER_COUNTS["capture_skips"] += 1
            _SHARED_OWNER_COUNTS["fallbacks"] += 1
            return _shared_owner_reference(routed_partial, shared_partial, gate_score, rows, T, k, group, routed_bf16)

        # Reference first: the candidate then consumes the same routed/shared leaves and leaves the
        # symmetric pool phases ready for the next layer. Both answers are materialized before the
        # int16 bit compare, which no reassociation can pass.
        ref = _shared_owner_reference(
            routed_partial, shared_partial, gate_score, rows, T, k, group, routed_bf16
        ).clone()
        local_ok, reason = True, ""
        try:
            got = _owner_shared_combine(
                routed_partial,
                shared_partial,
                gate_score,
                rows,
                T,
                k,
                group,
                routed_bf16,
                shared_bf16,
            )
        except Exception as exc:  # noqa: BLE001 -- agreement below keeps every rank together
            got = None
            local_ok, reason = False, f"candidate raised {exc!r}"
        if got is None:
            local_ok = False
            reason = reason or "candidate declined the live transport/shape"
        elif ref.shape != got.shape or ref.dtype != got.dtype:
            local_ok = False
            reason = f"shape/dtype mismatch ref={ref.shape}/{ref.dtype} got={got.shape}/{got.dtype}"
        elif not torch.equal(_bits(ref), _bits(got)):
            local_ok = False
            reason = f"{int((_bits(ref) != _bits(got)).sum().item())}/{ref.numel()} bit words differ"

        verdict = _agree(local_ok, group, routed_partial.device)
        if not verdict:
            _shared_owner_reject(key, reason or "a peer rank vetoed its live bit compare")
            _SHARED_OWNER_COUNTS["fallbacks"] += 1
            return ref
        _SHARED_OWNER_ADMIT[key] = True
        _SHARED_OWNER_COUNTS["admitted"] += 1
        print(
            f"[ISOEXEC-MOE] shared+routed owner fusion ADMITTED at {key}: live int16 bytes match "
            "the separate shared PIK tree + bf16 gate + routed owner + final bf16 add on every "
            "rank. TP8 H100 isolated battery: 1.912 -> 1.581 ms p50 at T=10240/H=2048/k=8 "
            "(1.211x).",
            flush=True,
        )
    else:
        try:
            got = _owner_shared_combine(
                routed_partial,
                shared_partial,
                gate_score,
                rows,
                T,
                k,
                group,
                routed_bf16,
                shared_bf16,
            )
        except Exception as exc:  # noqa: BLE001
            got = None
            reason = f"admitted candidate later raised {exc!r}"
        if got is None:
            # A shape admitted eagerly should not later lose its transport; the exact fallback still
            # holds, so make the loss loud and permanent for this process.
            _shared_owner_reject(key, reason if "reason" in locals() else "admitted candidate later declined")
            _SHARED_OWNER_COUNTS["fallbacks"] += 1
            return _shared_owner_reference(routed_partial, shared_partial, gate_score, rows, T, k, group, routed_bf16)

    _SHARED_OWNER_COUNTS["calls"] += 1
    calls = _SHARED_OWNER_COUNTS["calls"]
    if calls <= 2 or calls % 4096 == 0:
        print(
            f"[ISOEXEC-MOE-SHARED-OWNER] served={calls} exact fused shared+routed owner calls; " f"last geometry={key}",
            flush=True,
        )
    return got


def shared_fc2_subtree_partial(linear_fc2, x: torch.Tensor):
    """Run the same pik fc2 leaf GEMM but return its rank-subtree partial before reduction.

    Returns ``(partial, wire_is_bf16)`` or ``None``. Engine/no-grad only. The optional symmetric output
    is the candidate's dedicated shared-input pool, so the fc2 store becomes its staging store.
    """
    if torch.is_grad_enabled() or x.ndim < 2 or x.numel() == 0:
        return None
    if getattr(linear_fc2, "bias", None) is not None or not getattr(linear_fc2, "skip_bias_add", False):
        return None
    group = getattr(linear_fc2, "tp_group", None)
    if group is None or not dist.is_initialized():
        return None
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    k_full = int(getattr(linear_fc2, "input_size"))
    from ...ops.collectives.pik_tp_invariant import get_plan

    plan = get_plan()
    try:
        plan.validate(k_full, world)
    except Exception:
        return None
    if getattr(linear_fc2, "input_is_parallel", False):
        local_x = x
    else:
        from megatron.core.tensor_parallel.mappings import (
            scatter_to_tensor_model_parallel_region,
        )

        local_x = scatter_to_tensor_model_parallel_region(x, group=group)
    lead, k_local = local_x.shape[:-1], local_x.shape[-1]
    x2 = local_x.reshape(-1, k_local)
    out_dtype = torch.bfloat16 if (plan.bf16_leaves and plan.leaves_per_rank(world) == 1) else torch.float32
    dst = None
    if _p2p_ok(group):
        try:
            _ensure_canonical_pik()
            from pik.allreduce import _capturing  # type: ignore

            key = (x.device, _group_key(group), out_dtype)
            pool = _SHARED_IN_SYM.get(key)
            need = x2.shape[0] * int(getattr(linear_fc2, "output_size"))
            if not ((pool is None or pool.cap < need) and _capturing()):
                buf, _ = _shared_in_sym_pool(need, x.device, group, out_dtype).stage()
                dst = buf[:need].view(x2.shape[0], int(getattr(linear_fc2, "output_size")))
        except Exception:
            dst = None
    _ensure_canonical_pik()
    from pik.gemm import ti_gemm  # type: ignore

    part = ti_gemm(
        x2,
        linear_fc2.weight,
        plan=plan,
        tp_size=world,
        tp_rank=rank,
        k_full=k_full,
        out=dst,
    )
    return part.reshape(*lead, -1), out_dtype == torch.bfloat16


def _rows32_enabled() -> bool:
    return os.environ.get("SKYRL_ISOEXEC_MOE_OC_ROWS32", "1") == "1"


def _build_rows(sorted_indices, T, k):
    """``rows[t, :]`` = the k permuted-row positions of token t, ascending expert.

    Reproduces ``moe_batch_invariant._fixed_order_combine``'s ``rows`` (the same stable argsort of
    ``sorted_indices``). When enabled, the counting-sort kernel is asked for the [T,k] int32 form
    directly -- the identical permutation, minus the int64 materialization the general path converts
    away afterwards -- and falls back to the int64 path whenever the kernel declines.
    """
    from .moe_fused_permute import get_preemitted_combine_rows

    rows = get_preemitted_combine_rows(sorted_indices, T)
    if rows is not None:
        return rows
    if _rows32_enabled():
        from .moe_combine_rows_kernel import stable_combine_rows

        rows = stable_combine_rows(sorted_indices, T, dtype=torch.int32)
        if rows is not None:
            return rows
    from .moe_combine_rows_kernel import stable_argsort_order

    order = stable_argsort_order(sorted_indices, T)
    return order.view(T, k)


_patched = False


def owner_combine_enabled() -> bool:
    return os.environ.get("SKYRL_ISOEXEC_MOE_PIK_OWNER_COMBINE", "0") == "1"


def _owner_active(disp) -> bool:
    """True iff the landed nogather path applies to this dispatcher and the owner flag is on."""
    from .moe_batch_invariant import _nogather_active

    return owner_combine_enabled() and _nogather_active(disp)


def _wire_bf16(partial, group) -> bool:
    """Match ``_PikTreeAllReduce``'s wire decision exactly: bf16 only at world==G with a lossless
    leaf, else fp32. Reuses the same self-check state so the rounding decision is shared.

    A bf16 partial IS the wire: it only ever arrives from ``owner_wire_sink``, where the fc2 leaf-tree
    stored the already-rounded leaf value, so the losslessness check is vacuous by construction.
    """
    if partial.dtype == torch.bfloat16:
        return True
    from ...ops.collectives.pik_tp_invariant import get_plan
    from .moe_batch_invariant import _bf16_wire, _pik_bf16_wire_enabled

    if not (
        _pik_bf16_wire_enabled()
        and partial.dtype == torch.float32
        and dist.get_world_size(group) == get_plan().num_leaves
    ):
        return False
    if not _bf16_wire["checked"]:
        _bf16_wire["checked"] = True
        _bf16_wire["ok"] = bool((partial.to(torch.bfloat16).to(torch.float32) == partial).all())
    return _bf16_wire["ok"]


def install_moe_pik_owner_combine() -> bool:
    """Wrap the AllGather dispatcher's combine_preprocess/token_combine with the owner-computes path.

    Idempotent, and only fires for marked engine-nogather dispatchers under
    SKYRL_ISOEXEC_MOE_PIK_OWNER_COMBINE. MUST be installed AFTER ``install_moe_pik_fc2_combine``: it
    delegates the non-owner path back to whatever held the binding.
    """
    global _patched
    if _patched:
        return True
    try:
        from megatron.core.transformer.moe.token_dispatcher import (
            MoEAllGatherTokenDispatcher as D,
        )
    except Exception:  # pragma: no cover
        return False

    _prev_pre = D.combine_preprocess
    _prev_tc = D.token_combine

    def combine_preprocess(self, hidden_states):
        if _owner_active(self):
            group = self.tp_ep_group
            partial = hidden_states.contiguous()  # [P,H] fp32 leaf-subtree partial
            P = partial.shape[0]
            T = int(self.hidden_shape_before_permute[0])
            if T > 0 and P % T == 0:
                k = P // T
                rows = _build_rows(self.reversed_local_input_permutation_mapping, T, k)
                wire = _wire_bf16(partial, group)
                payload = getattr(self, "_isoexec_shared_owner_payload", None)
                self._isoexec_shared_owner_payload = None
                if shared_owner_fusion_enabled() and payload is not None:
                    shared_partial, gate_score, shared_bf16 = payload
                    shared_partial = shared_partial.reshape(T, -1)
                    out = _shared_owner_dispatch(
                        partial,
                        shared_partial,
                        gate_score,
                        rows,
                        T,
                        k,
                        group,
                        wire,
                        shared_bf16,
                    )
                else:
                    out = _owner_combine(partial, rows, T, k, group, wire)  # bf16 [T,H]
                self._isoexec_owner_done = True
                return out
        self._isoexec_owner_done = False
        if hidden_states.dtype == torch.bfloat16 and wire_stage_enabled():
            # A wire-staged bf16 partial fell through to the landed combine, which expects the fp32
            # leaf-subtree partial. The promotion is bit-identical to what fc2 would have produced
            # without the sink, since the fp32 partial always was the promotion of the bf16 leaf.
            _WIRE_COUNTS["promotes"] += 1
            if _WIRE_COUNTS["promotes"] == 1:
                print(
                    "[ISOEXEC-MOE] WARNING: a wire-staged bf16 partial reached the NON-owner combine "
                    "path; promoting to fp32 (bit-identical -- the fp32 partial is by construction "
                    "the promotion of this exact bf16 leaf) and continuing on the landed combine. "
                    "The wire-stage saving is not being taken here.",
                    flush=True,
                )
            hidden_states = hidden_states.to(torch.float32)
        return _prev_pre(self, hidden_states)

    def token_combine(self, hidden_states):
        if getattr(self, "_isoexec_owner_done", False):
            # The owner path already produced the bf16 [T,H] full-batch result on every rank.
            return hidden_states if hidden_states.dtype == torch.bfloat16 else hidden_states.to(torch.bfloat16)
        return _prev_tc(self, hidden_states)

    D.combine_preprocess = combine_preprocess
    D.token_combine = token_combine
    _patched = True
    print(
        "[ISOEXEC-MOE] pik OWNER-COMPUTES combine installed (engine nogather): each rank owns a "
        "token slice, computes its tree+topk-sum, all-gathers tiny bf16 finals -- bit-identical to "
        "the landed all-reduce combine at ~1/world the peer-read traffic",
        flush=True,
    )
    return True
