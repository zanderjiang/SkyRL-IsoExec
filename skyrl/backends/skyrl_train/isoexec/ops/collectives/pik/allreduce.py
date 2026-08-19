"""Deterministic all-reduce: the upper levels of the same combine tree.

``pik.gemm`` leaves each rank holding the root of its subtree of leaves; what remains is to combine the C rank
partials with the top ``log2(C)`` levels of the fixed balanced tree (left operand = lower rank), so that it
composes with the intra-rank tree into one global tree independent of C, and so that every rank evaluates the
same expression rather than trusting a collective's internal reduction order. NCCL is therefore never asked to
reduce: it only moves bytes (all-gather / all-to-all) and the arithmetic is always the local tree kernel, which
keeps determinism independent of NCCL's algorithm, topology, protocol, and channel count. The payload is fp32
because the tree's intermediate nodes are fp32 at TP=1, where there is no all-reduce at all.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist

from .codegen import (
    fused_p2p_allreduce_kernel,
    p2p_allreduce_kernel,
    tree_reduce_kernel,
)
from .fastlaunch import launch as fastlaunch

logger = logging.getLogger(__name__)

# Below this many bytes a single all-gather beats a two-shot: one collective, one kernel, minimal latency.
# Above it, the C-fold read amplification dominates. Tunable via SKYRL_ISOEXEC_PIK_ONESHOT_MB.
ONESHOT_MAX_BYTES = int(os.environ.get("SKYRL_ISOEXEC_PIK_ONESHOT_MB", "2")) << 20

# Same crossover for the P2P path: below it every rank reads every peer and computes the whole tensor (one
# barrier, no push); above it each rank owns a slice and pushes. It must read the same env var as
# ONESHOT_MAX_BYTES, since the engine keeps the symmetric-memory path and never reaches the NCCL constant.
# Both branches evaluate the identical tree, so this is a pure perf knob; see pik/ar_branch.py for the
# derivable alternatives to this byte threshold.
P2P_ONESHOT_MAX_BYTES = int(os.environ.get("SKYRL_ISOEXEC_PIK_ONESHOT_MB", "2")) << 20


_SYM: dict = {}
_SYM_IN: dict = {}
_SYM_OUT: dict = {}


class _SymInPool:
    """Symmetric staging buffers, double-buffered, keyed by (device, world, wire dtype).

    The two alternating buffers let the one-shot path use a single barrier instead of two: the trailing
    barrier only existed to stop a rank overwriting its staging buffer while peers were still reading it, and
    ``barrier()`` is stream-ordered, so a rank cannot clear barrier N+1 until every peer has issued its read of
    buffer N -- making buffer N safe to reuse at N+2.

    The key deliberately excludes the root dtype: the staging buffer holds the partial, and keying on the root
    dtype would split the pool so that ``sym_partial`` staged into a different buffer than the reduce reads,
    reintroducing the whole-partial copy this pool exists to remove.
    """

    def __init__(self, n: int, device, group, dtype=torch.float32):
        from torch.distributed import _symmetric_memory as symm_mem

        g = group if group is not None else dist.group.WORLD
        cap = max(n, 1 << 20)
        self.cap = cap
        self.dtype = dtype
        self.inp = [symm_mem.empty(cap, dtype=dtype, device=device) for _ in range(2)]
        self.h_in = [symm_mem.rendezvous(b, g) for b in self.inp]
        self.phase = 0

    def stage(self):
        return self.inp[self.phase], self.h_in[self.phase]

    def flip(self):
        self.phase ^= 1


class _SymOutPool:
    """Symmetric output buffer, single-buffered, keyed by the root dtype alone.

    The two-shot trailing barrier (or the fused kernel's trailing handshake) protects it, and sharing it
    across wire dtypes is safe: a site's consumer reads the buffer before any rank can pass the next site's
    leading barrier, since arrival there is stream-ordered after everything earlier.
    """

    def __init__(self, n: int, device, group, out_dtype=torch.float32):
        from torch.distributed import _symmetric_memory as symm_mem

        g = group if group is not None else dist.group.WORLD
        cap = max(n, 1 << 20)
        self.cap = cap
        self.out_dtype = out_dtype
        self.out = symm_mem.empty(cap, dtype=out_dtype, device=device)
        self.h_out = symm_mem.rendezvous(self.out, g)


class _SymPool:
    """Self-contained pool: double-buffered staging plus an output buffer.

    Not used by this file's own ``_sym_pool``, which composes the dtype-keyed pools above, but kept for the
    modules that borrow it with their own registries and aliasing rules.
    """

    def __init__(self, n: int, device, group, dtype=torch.float32, out_dtype=None):
        from torch.distributed import _symmetric_memory as symm_mem

        g = group if group is not None else dist.group.WORLD
        cap = max(n, 1 << 20)
        self.cap = cap
        self.dtype = dtype
        self.out_dtype = out_dtype if out_dtype is not None else dtype
        self.inp = [symm_mem.empty(cap, dtype=dtype, device=device) for _ in range(2)]
        self.h_in = [symm_mem.rendezvous(b, g) for b in self.inp]
        self.out = symm_mem.empty(cap, dtype=self.out_dtype, device=device)
        self.h_out = symm_mem.rendezvous(self.out, g)
        self.phase = 0

    def stage(self):
        return self.inp[self.phase], self.h_in[self.phase]

    def flip(self):
        self.phase ^= 1


class _PoolView:
    """A view pairing one staging pool with one output pool.

    Two views whose wire dtype agrees share staging buffers whatever their root dtypes are, which is what the
    zero-copy path needs.
    """

    def __init__(self, inp: _SymInPool, outp: _SymOutPool):
        self._in = inp
        self._out = outp

    @property
    def cap(self):
        return min(self._in.cap, self._out.cap)

    @property
    def dtype(self):
        return self._in.dtype

    @property
    def out_dtype(self):
        return self._out.out_dtype

    @property
    def out(self):
        return self._out.out

    @property
    def h_out(self):
        return self._out.h_out

    @property
    def inp(self):
        return self._in.inp

    @property
    def phase(self):
        return self._in.phase

    def stage(self):
        return self._in.stage()

    def flip(self):
        self._in.flip()


def _sym_in_pool(n: int, device, group, dtype) -> _SymInPool:
    key = (device, dist.get_world_size(group), dtype)
    pool = _SYM_IN.get(key)
    if pool is None or pool.cap < n:
        pool = _SymInPool(n, device, group, dtype)
        _SYM_IN[key] = pool
    return pool


def _sym_out_pool(n: int, device, group, out_dtype) -> _SymOutPool:
    key = (device, dist.get_world_size(group), out_dtype)
    pool = _SYM_OUT.get(key)
    if pool is None or pool.cap < n:
        pool = _SymOutPool(n, device, group, out_dtype)
        _SYM_OUT[key] = pool
    return pool


def _sym_pool(n: int, device, group, dtype=torch.float32, out_dtype=None) -> _PoolView:
    # out_dtype=None means "same as dtype", resolved here as everywhere else in this file.
    # The view is cached so repeated lookups (and the CPU identity tests) get one object per
    # (wire, root) pair, but the BUFFERS underneath are keyed per-dtype -- see _SymInPool.
    if out_dtype is None:
        out_dtype = dtype
    inp = _sym_in_pool(n, device, group, dtype)
    outp = _sym_out_pool(n, device, group, out_dtype)
    key = (device, dist.get_world_size(group), dtype, out_dtype)
    view = _SYM.get(key)
    if view is None or view._in is not inp or view._out is not outp:
        view = _PoolView(inp, outp)
        _SYM[key] = view
    return view


def p2p_available(group=None) -> bool:
    try:
        from torch.distributed import _symmetric_memory  # noqa: F401

        return dist.is_initialized() and dist.get_world_size(group) > 1
    except Exception:  # noqa: BLE001
        return False


def sym_partial(shape, device, group=None, dtype=torch.float32, out_dtype=None) -> torch.Tensor:
    """A view of the requested shape into this rank's current symmetric staging buffer.

    Hand this to ``ti_gemm`` as ``out=`` and the GEMM writes its subtree partial straight into peer-visible
    memory; without it the P2P path must copy the partial in, a full extra pass over the payload.

    ``out_dtype`` no longer selects a buffer -- the staging pool is keyed by the wire dtype alone -- and is
    kept only for caller compatibility.
    """
    n = 1
    for s in shape:
        n *= s
    inp, _ = _sym_pool(n, device, group, dtype, out_dtype).stage()
    return inp[:n].view(*shape)


def _p2p_shot(n: int, elt: int, world: int) -> bool:
    """True selects two-shot (own a slice and push it); False selects one-shot (read every peer, compute all).

    The budget is compared against bytes read, not payload: one-shot reads every other peer's full ``n`` on
    top of computing the whole output, so its cost grows with world size while two-shot's does not. Both
    branches evaluate the same tree, so the choice cannot move a bit. Split out so the fused single-launch
    path makes the identical branch choice as the unfused one, and delegated to ``pik.ar_branch``, which can
    consult a measured arch-keyed crossover or a warmup calibration instead of the byte threshold.
    """
    from .ar_branch import two_shot

    return two_shot(n, elt, world, P2P_ONESHOT_MAX_BYTES)


# Element-to-thread layout of the reduce kernel (``vec``), admitted per shape. Giving each thread ``vec``
# contiguous elements -- a ``(BLOCK//vec, vec)`` arange instead of a 1-D one -- lets Triton emit 128-bit
# vector accesses, which pays off on the remote-store side of the two-shot push.
#
# It is bit-safe because it moves only which thread computes an element: the element set, the load sources,
# the tree association and the single rounding are unchanged. Every (world, numel, wire, root, branch) still
# re-proves the layout against the vec=1 reference on live operands with a bit-pattern compare before first
# use, and a mismatch is a remembered per-shape fallback to vec=1. The verdict is AND-ed across the group so
# that all ranks run the same kernel, avoiding unattributable rank skew.
_AR_VEC_ENV = "SKYRL_ISOEXEC_PIK_AR_VEC"


def _ar_vec() -> int:
    try:
        v = int(os.environ.get(_AR_VEC_ENV, "4").strip() or "1")
    except ValueError:
        return 1
    return v if v in (2, 4, 8) else 1


_VEC_ADMIT: dict = {}
_VEC_COUNTS = {"calls": 0, "admitted": 0, "rejected": 0, "errors": 0, "capture_skips": 0, "peer_vetoes": 0}
_VEC_FIRST_REJECT: dict = {}


def vec_counts() -> dict:
    """Per-process tally of the vec-layout admission."""
    return {
        **_VEC_COUNTS,
        "vec": _ar_vec(),
        "shapes": {str(k): v for k, v in _VEC_ADMIT.items()},
        "first_reject": dict(_VEC_FIRST_REJECT) or None,
    }


def _vec_reject(key, reason: str, detail: str = "") -> None:
    global _VEC_FIRST_REJECT
    _VEC_ADMIT[key] = 1
    _VEC_COUNTS["rejected"] += 1
    if _VEC_FIRST_REJECT:
        return
    _VEC_FIRST_REJECT = {"key": str(key), "reason": reason, "detail": detail}
    msg = (
        f"[ISOEXEC-PIK] WARNING: the vec element-to-thread layout was REFUSED at {key} -- "
        f"{reason}. {detail} The result is unaffected (this shape runs the vec=1 reference "
        "layout, which every frozen bit was proven against); only the ~2 us/site store-side "
        "saving is not taken here. Read pik.allreduce.vec_counts()."
    )
    print(msg, flush=True)
    logger.warning(msg)


def _vec_admit(key, partial: torch.Tensor, group, root_dtype, want: int) -> int:
    """Prove the vec layout bitwise-equal to the vec=1 reference at this shape, on live operands.

    Reference first, cloned, then re-run and bit-compared, with the verdict AND-ed across the group before it
    is recorded. Returns the vec to use.
    """
    device = partial.device
    reason = detail = ""
    ref = _p2p_unfused(partial, group, None, root_dtype, _vec=1).clone()
    local_ok = True
    try:
        got = _p2p_unfused(partial, group, None, root_dtype, _vec=want).clone()
    except Exception as e:  # noqa: BLE001
        _VEC_COUNTS["errors"] += 1
        local_ok, reason, detail = False, "the vec kernel raised", repr(e)
    else:
        if not torch.equal(_bits(ref), _bits(got)):
            local_ok, reason = False, "BIT PATTERNS DIFFER"
            detail = f"{int((_bits(ref) != _bits(got)).sum().item())}/{ref.numel()} elements"
    agreed = _agree(local_ok, group, device)
    if not agreed:
        if local_ok:
            _VEC_COUNTS["peer_vetoes"] += 1
            reason = "a PEER rank refused it"
            detail = "this rank's own compare passed; the group runs vec=1 together"
        _vec_reject(key, reason, detail)
        return 1
    _VEC_ADMIT[key] = want
    _VEC_COUNTS["admitted"] += 1
    if _VEC_COUNTS["admitted"] == 1:
        msg = (
            f"[ISOEXEC-PIK] AR vec layout ADMITTED (first shape {key}, vec={want}): bit patterns "
            "equal the vec=1 reference on live operands, agreed across the group. "
            "Read pik.allreduce.vec_counts() for the per-shape verdicts."
        )
        print(msg, flush=True)
        logger.warning(msg)
    return want


def _vec_resolved(world, n, dt_in, dt_out, branch_push, partial, group, root_dtype) -> int:
    want = _ar_vec()
    if want == 1:
        return 1
    key = (world, n, str(dt_in), str(dt_out), "push" if branch_push else "local", want)
    verdict = _VEC_ADMIT.get(key)
    if verdict is not None:
        return verdict
    if _capturing():
        # A compare needs a host sync; the shape gets another chance when seen eagerly.
        _VEC_COUNTS["capture_skips"] += 1
        return 1
    return _vec_admit(key, partial, group, root_dtype, want)


def _p2p_unfused(partial: torch.Tensor, group, out: torch.Tensor | None, root_dtype=None, _vec=None):
    """One kernel: read every rank's partial over NVLink, apply the tree, and write back.

    Replaces all_to_all -> reduce kernel -> all_gather with a single pass. "Unfused" names the launch
    structure, not the arithmetic: the barriers are still their own kernels (2 launches one-shot, 3 two-shot),
    and ``_p2p_fused`` folds them in. This is the reference path the fused path must prove itself against.

    ``root_dtype`` is the dtype the tree root is emitted in, defaulting to ``partial.dtype``. A bf16 partial
    with an fp32 root halves the bytes read in shot 1 while leaving the root bit-identical to the fp32-wire
    root, since the adds and their order are unchanged and the leaf is bf16-representable.

    ``_vec`` is internal: the admission path pins the layout so its two arms are exactly the kernels being
    compared. Callers leave it None.
    """
    import triton

    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    n = partial.numel()
    shape = partial.shape
    dt_in = partial.dtype
    dt_out = root_dtype if root_dtype is not None else dt_in
    in_bf16 = dt_in == torch.bfloat16
    out_bf16 = dt_out == torch.bfloat16
    pool = _sym_pool(n, partial.device, group, dt_in, dt_out)
    inp, h_in = pool.stage()

    push = _p2p_shot(n, partial.element_size(), world)
    if _vec is None:
        _vec = _vec_resolved(world, n, dt_in, dt_out, push, partial, group, root_dtype)
        _VEC_COUNTS["calls"] += 1
        # Admission ran both paths and flipped the pool twice (net zero); re-resolve the staging view.
        inp, h_in = pool.stage()

    if partial.data_ptr() != inp.data_ptr():
        inp[:n].copy_(partial.view(-1))  # caller didn't stage; see sym_partial
    h_in.barrier()  # every rank's partial is now visible to every peer

    peers_in = [h_in.get_buffer(r, (n,), dt_in) for r in range(world)]

    if not push:
        # One-shot: each rank reads all peers and computes the whole tensor. Double-buffering makes the
        # trailing barrier unnecessary.
        dst = out.view(-1) if out is not None else torch.empty(n, device=partial.device, dtype=dt_out)
        k = p2p_allreduce_kernel(world, push=False, in_bf16=in_bf16, out_bf16=out_bf16, vec=_vec)
        k[(triton.cdiv(n, 1024),)](*peers_in, dst, 0, n, BLOCK=1024, num_warps=4)
        pool.flip()
        return dst.view(*shape)

    # Two-shot: rank r owns slice r and pushes it into every peer's out. The trailing barrier is unavoidable
    # here, because the caller reads slices that peers wrote.
    assert n % world == 0, f"two-shot P2P needs numel {n} divisible by world {world}"
    chunk = n // world
    peers_out = [pool.h_out.get_buffer(r, (n,), dt_out) for r in range(world)]
    k = p2p_allreduce_kernel(world, push=True, in_bf16=in_bf16, out_bf16=out_bf16, vec=_vec)
    k[(triton.cdiv(chunk, 1024),)](*peers_in, *peers_out, rank * chunk, chunk, BLOCK=1024, num_warps=4)
    pool.h_out.barrier()  # every slice has landed on every rank
    pool.flip()
    if out is not None:
        out.view(-1).copy_(pool.out[:n])
        return out.view(*shape)
    return pool.out[:n].view(*shape)  # no copy-out: hand back the symmetric buffer


# Fused barrier + reduce: one launch per site. pik cannot use NCCL's reduction -- that is the invariance
# contract -- so a site otherwise costs barrier + kernel [+ barrier], 2 or 3 launches against NATIVE's single
# fused ncclAllReduce. Fusion removes launches without changing which expression is evaluated: ``_p2p_shot``
# still makes the one-shot/two-shot choice for both paths, so there is no payload threshold to tune. The only
# gates are structural -- symmetric memory present, world a power of two, and a per-shape bitwise proof on
# live operands.
#
# Rank invariance is a correctness requirement here. The reference path's barriers are separate rendezvous
# kernels while the fused path's are inside the reduce kernel on a different flag buffer, so two ranks
# disagreeing about which path to run hang rather than run slowly: one parked at a barrier no peer will
# reach, the other spinning on a flag no peer will publish. Every input to the decision is therefore either
# group-wide by construction (world size, numel, dtype, the branch, the grid and hence the flag stride, and
# SPMD graph capture) or made group-wide by a collective: ``_fused_group_on`` AND-s the env flag across the
# group, and ``_agree`` AND-s each per-shape admission verdict before it is recorded. Nothing in the decision
# is data-dependent, rank-local, or timing-dependent.
_FUSED_ENV = "SKYRL_ISOEXEC_PIK_FUSED_BARRIER"


def _env_on(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("", "0", "false", "no", "off")


_FUSED_ON = _env_on(_FUSED_ENV)

# Per (world, numel, dt_in, dt_out) admission verdict: True = proven bitwise-equal to ``_p2p_unfused`` on
# live operands at this shape, False = proven different or the fused path raised. Never populated from inside
# a graph capture, since the compare needs a host sync.
_FUSED_ADMIT: dict = {}
_FUSED_COUNTS = {"calls": 0, "admitted": 0, "rejected": 0, "errors": 0, "capture_skips": 0, "peer_vetoes": 0}
_FUSED_FIRST_REJECT: dict = {}
# Per process group: has this group AGREED that the fused path may be attempted at all?
_FUSED_GROUP: dict = {}


def fused_enabled() -> bool:
    """Whether this process is allowed to try the fused single-launch collective."""
    return _FUSED_ON


def set_fused_enabled(on: bool) -> None:
    """Test/bench hook; flipping this does not bypass admission."""
    global _FUSED_ON
    _FUSED_ON = bool(on)


def fused_counts() -> dict:
    """Per-process tally of the fused collective, plus the first rejection if any."""
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


def _agree(local_ok: bool, group, device) -> bool:
    """AND a per-rank verdict across the group, so no rank acts on its own answer.

    If one rank admitted a shape and another refused it, they would run different launch structures and both
    wait forever; a MIN all-reduce takes the whole group down to the reference path instead.
    """
    if not dist.is_initialized() or dist.get_world_size(group) < 2:
        return local_ok
    t = torch.tensor([1 if local_ok else 0], device=device, dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group)
    return bool(t.item())


def _fused_group_on(group, device) -> bool:
    """Has this group agreed the fused path may be attempted? One collective, once, ever.

    ``_FUSED_ON`` comes from an env var, which can end up set on some ranks and not others, and that mismatch
    deadlocks -- so enablement is settled by a collective every rank runs whatever its own flag says, which is
    why this is called before the ``_FUSED_ON`` test. Under graph capture the handshake cannot run, so the
    group keeps the reference path and is resolved the next time it is entered eagerly.
    """
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
        msg = (
            f"[ISOEXEC-PIK] the FUSED single-launch all-reduce is DISABLED for process group "
            f"{key}: this rank has {_FUSED_ENV}={'on' if _FUSED_ON else 'off'} but at least one "
            "peer disagrees, and running the two launch structures side by side would deadlock. "
            "Set the flag identically on every rank of a TP group."
        )
        print(msg, flush=True)
        logger.warning(msg)
    elif got:
        msg = (
            f"[ISOEXEC-PIK] FUSED single-launch tree all-reduce ENABLED and agreed by all "
            f"{dist.get_world_size(group) if dist.is_initialized() else 1} ranks of group {key}. "
            "Each shape still has to prove itself bitwise against the reference path before it "
            "is used. Read pik.allreduce.fused_counts()."
        )
        print(msg, flush=True)
        logger.info(msg)
    return got


def _capturing() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001 -- older torch without the query
        return False


def _bits(t: torch.Tensor) -> torch.Tensor:
    """A tensor's bit pattern as an integer tensor.

    ``torch.equal`` on floats is not a bit compare -- it calls -0.0 and 0.0 equal and NaN unequal -- so every
    admission compare in this file goes through here.
    """
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16)
    if t.dtype == torch.float16:
        return t.view(torch.int16)
    return t


def flag_slot(region: int, world: int, rank: int, nb: int) -> int:
    """Where rank ``rank`` publishes its arrival in ANY rank's flag array, for ``region``."""
    return (region * world + rank) * nb


def flag_wait_base(region: int, world: int, nb: int) -> int:
    """Start of ``region`` in a rank's OWN flag array; the spin adds ``lane*nb + block``."""
    return region * world * nb


class _FlagPool:
    """Symmetric int32 flag matrix plus local epoch counters for the in-kernel barrier.

    Layout, sized entirely from world size, payload, and BLOCK::

        flags[region, src_rank, block]   region 0 = leading, 1 = trailing; symmetric memory
        epoch[region, block]             local, never read by a peer

    ``nb`` is the flag stride, i.e. the largest grid this rank will launch; it grows geometrically with the
    largest payload seen and stays a small fraction of the payload, so it needs no cap.

    One pool is shared by every pik site in the process, which is why the epochs are compared with
    ``v - seq >= 0`` rather than ``==``: ranks run the same SPMD program and agree on the order, but a rank
    may be a full call ahead of a peer and must not strand it on a flag that already moved. Reallocation
    retires the previous pool rather than freeing it, since a kernel launched before the growth may still be
    spinning on the old flags and symmetric allocations are not stream-aware.
    """

    def __init__(self, nb: int, device, group):
        from torch.distributed import _symmetric_memory as symm_mem

        g = group if group is not None else dist.group.WORLD
        world = dist.get_world_size(g)
        rank = dist.get_rank(g)
        self.nb = nb
        self.world = world
        self.rank = rank
        self.flags = symm_mem.empty(2 * world * nb, dtype=torch.int32, device=device)
        self.flags.zero_()
        self.h = symm_mem.rendezvous(self.flags, g)
        self.peers = [self.h.get_buffer(r, (2 * world * nb,), torch.int32) for r in range(world)]
        self.mine = self.peers[rank]
        self.epoch = torch.zeros(2 * nb, dtype=torch.int32, device=device)
        # rendezvous is collective, so ranks agree on when; this barrier makes them agree that the zeroing
        # landed before the first publish. Once per allocation, never in the hot path.
        self.h.barrier()

    # Slot a peer writes this rank's arrival into, and where this rank looks for theirs.
    def my_slot(self, region: int) -> int:
        return flag_slot(region, self.world, self.rank, self.nb)

    def wait_base(self, region: int) -> int:
        return flag_wait_base(region, self.world, self.nb)


_FLAGS: dict = {}
_FLAG_KEEP: list = []


def _flag_pool(nb: int, device, group) -> _FlagPool:
    key = (device, dist.get_world_size(group) if dist.is_initialized() else 1)
    pool = _FLAGS.get(key)
    if pool is None or pool.nb < nb:
        if _capturing():
            # Allocating and rendezvous-ing inside a capture is illegal, and a half-built symmetric
            # allocation hangs rather than erroring, so refuse visibly instead.
            raise RuntimeError(
                f"pik fused collective: the flag pool must grow to {nb} blocks, but this is "
                "inside a CUDA graph capture. Run every shape once EAGERLY before capturing "
                "(a warmup pass), so the pools are sized before the graph is recorded."
            )
        want = max(nb, 2 * pool.nb if pool is not None else 0, 1024)
        if pool is not None:
            _FLAG_KEEP.append(pool)  # a launched kernel may still hold the old pointers
        pool = _FlagPool(want, device, group)
        _FLAGS[key] = pool
    return pool


def _p2p_fused(partial: torch.Tensor, group, out: torch.Tensor | None, root_dtype=None):
    """``_p2p_unfused`` with the barriers folded into the reduce kernel: one launch.

    The arithmetic is unchanged -- same ``_p2p_shot`` branch, same generated tree (``codegen._ar_core`` is
    emitted byte-for-byte into both kernels), same intermediates, dtypes, rank order, BLOCK and num_warps.
    Only the flag traffic moves, from separate barrier kernels into this kernel's prologue and epilogue.

    The double-buffer argument still holds with the barrier at the front of the kernel: a rank passes the
    leading wait of call N+1 only once every peer has started its call-N+1 kernel, hence retired its call-N
    kernel and its reads of staging buffer N%2. The next write to that buffer is stream-ordered after the
    call-N+1 kernel, so it is free at N+2 exactly as before.
    """
    import triton

    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    n = partial.numel()
    shape = partial.shape
    dt_in = partial.dtype
    dt_out = root_dtype if root_dtype is not None else dt_in
    in_bf16 = dt_in == torch.bfloat16
    out_bf16 = dt_out == torch.bfloat16
    pool = _sym_pool(n, partial.device, group, dt_in, dt_out)
    inp, h_in = pool.stage()

    if partial.data_ptr() != inp.data_ptr():
        inp[:n].copy_(partial.view(-1))  # caller didn't stage; see sym_partial
    # No h_in.barrier() here: the kernel's own prologue is the barrier, and the copy above is still ordered
    # before it, since a block of the kernel runs only after the copy kernel retired.

    peers_in = [h_in.get_buffer(r, (n,), dt_in) for r in range(world)]
    push = _p2p_shot(n, partial.element_size(), world)

    if not push:
        grid = triton.cdiv(n, 1024)
        base, count = 0, n
        dst = out.view(-1) if out is not None else torch.empty(n, device=partial.device, dtype=dt_out)
        outs = [dst]
    else:
        assert n % world == 0, f"two-shot P2P needs numel {n} divisible by world {world}"
        chunk = n // world
        grid = triton.cdiv(chunk, 1024)
        base, count = rank * chunk, chunk
        outs = [pool.h_out.get_buffer(r, (n,), dt_out) for r in range(world)]

    fp = _flag_pool(grid, partial.device, group)
    k = fused_p2p_allreduce_kernel(world, push=push, in_bf16=in_bf16, out_bf16=out_bf16)
    k[(grid,)](
        *peers_in,
        *outs,
        *fp.peers,
        fp.mine,
        fp.epoch,
        fp.my_slot(0),
        fp.my_slot(1),
        fp.wait_base(1),
        fp.nb,
        base,
        count,
        BLOCK=1024,
        WORLD=world,
        num_warps=4,
    )
    pool.flip()
    if not push:
        return dst.view(*shape)
    if out is not None:
        out.view(-1).copy_(pool.out[:n])
        return out.view(*shape)
    return pool.out[:n].view(*shape)


def _fused_reject(key, reason: str, detail: str = "") -> None:
    """Refuse the fused path for this shape, once and loudly."""
    global _FUSED_FIRST_REJECT
    _FUSED_ADMIT[key] = False
    _FUSED_COUNTS["rejected"] += 1
    if _FUSED_FIRST_REJECT:
        return
    _FUSED_FIRST_REJECT = {"key": str(key), "reason": reason, "detail": detail}
    msg = (
        f"[ISOEXEC-PIK] WARNING: the FUSED single-launch tree all-reduce was REFUSED at "
        f"{key} -- {reason}. {detail} The result is unaffected (this shape falls back to "
        "the reference barrier+kernel path, which is what every frozen bit was proven "
        "against), but the launch-count saving is not being taken here. Read "
        "pik.allreduce.fused_counts()."
    )
    print(msg, flush=True)
    logger.warning(msg)


def _fused_admit(key, partial: torch.Tensor, group, root_dtype) -> bool:
    """Prove the fused kernel bitwise-equal to the reference at this shape, on these operands.

    A bit-pattern compare, not allclose, because a reduction that reassociated is usually still allclose; and
    per (world, numel, dtype, dtype), because the kernel is generated per world size and wire form.

    Both paths run back to back on the live partial, reference first and cloned, then re-staged. The two pool
    flips cancel, so the caller's double-buffer phase ends up where it would have been. Every rank runs this
    at the same call on the same shape, keeping the barrier sequences matched.

    The verdict is AND-ed across the group before it is recorded -- including on the exception path, so a rank
    whose kernel failed to launch still joins the collective its peers are running.
    """
    device = partial.device
    reason = detail = ""
    ref = _p2p_unfused(partial, group, None, root_dtype)
    ref = ref.clone()
    local_ok = True
    try:
        got = _p2p_fused(partial, group, None, root_dtype)
        got = got.clone()
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
        _fused_reject(key, reason, detail)
        return False
    _FUSED_ADMIT[key] = True
    _FUSED_COUNTS["admitted"] += 1
    if _FUSED_COUNTS["admitted"] == 1:
        # Announce the first admission too: silence on the success path is indistinguishable from the flag
        # never reaching this process, or from every shape being first seen under graph capture.
        msg = (
            f"[ISOEXEC-PIK] FUSED barrier+reduce ADMITTED (first shape {key}): this shape now runs "
            f"as ONE launch instead of barrier+kernel[+barrier]. Bit patterns were compared against "
            f"the reference path on live operands and agreed across the group. Read "
            f"pik.allreduce.fused_counts() for the full per-shape verdict list; a decode trace "
            f"should show c10d::symmetric_memory::barrier_kernel drop by 2 calls/step per admitted "
            f"two-shot site."
        )
        print(msg, flush=True)
        logger.warning(msg)
    return True


def _tree_all_reduce_p2p(partial: torch.Tensor, group, out: torch.Tensor | None, root_dtype=None):
    """Symmetric-memory tree all-reduce: dispatch to the reference launch structure or the fused one.

    ``_p2p_unfused`` is the default and the reference; ``_p2p_fused`` is used only for a shape proven
    bitwise-identical to it on live operands in this process. Every branch here is taken identically on every
    rank of ``group``; the one input that is not invariant by construction, the env flag, is settled by
    ``_fused_group_on`` before it is read.
    """
    world = dist.get_world_size(group)
    if not _fused_group_on(group, partial.device):
        return _p2p_unfused(partial, group, out, root_dtype)

    n = partial.numel()
    dt_out = root_dtype if root_dtype is not None else partial.dtype
    key = (world, n, str(partial.dtype), str(dt_out))
    verdict = _FUSED_ADMIT.get(key)
    if verdict is None:
        if world < 2 or (world & (world - 1)):
            _fused_reject(key, "world size is not a power of two", f"world={world}")
            return _p2p_unfused(partial, group, out, root_dtype)
        if _capturing():
            # A compare needs a host sync, which capture forbids. Record no verdict, so the shape gets
            # another chance the next time it is seen eagerly.
            _FUSED_COUNTS["capture_skips"] += 1
            return _p2p_unfused(partial, group, out, root_dtype)
        # Admission ran both paths, discarded both answers, and left the staging pool on the phase it
        # started on, so this call runs for real as if admission had not happened.
        verdict = _fused_admit(key, partial, group, root_dtype)
    if not verdict:
        return _p2p_unfused(partial, group, out, root_dtype)
    _FUSED_COUNTS["calls"] += 1
    return _p2p_fused(partial, group, out, root_dtype)


# Absorbing the root's fp32 -> bf16 round into the reduce kernel. With fp32 leaves against a bf16 residual
# stream, every row-parallel site ends with ``tree_all_reduce(part).to(bf16)``, and that trailing cast is a
# separate elementwise launch over the whole [M, N] -- elements the reduce kernel already loads and stores,
# and for which codegen already emits an ``out_bf16`` store.
#
# Rounding the root is TP-independent: the root, like a leaf and unlike an internal node, is the same node
# however K was sharded. Moving the round into the reduce's store is therefore legal provided nothing happens
# in between, which puts two conditions on the caller: there must be no bias (with one, the tail is
# ``(root + bias).to(bf16)`` in fp32, and rounding first would have TP>1 add the bias in bf16 where TP=1 adds
# it in fp32), and the round must be the next operation on the root.
#
# torch's fp32->bf16 and Triton's ``cvt.rn.bf16.f32`` are both round-to-nearest-even, but the narrowing root
# is still admitted per shape against ``reference_fp32_root.to(bf16)``, with the verdict agreed across the
# group. This gets its own flag because, unlike fusion, it moves a rounding point.
_ROOT_CAST_ENV = "SKYRL_ISOEXEC_PIK_FUSED_ROOT_CAST"
# Default on: with the staging/output pools split by dtype the copy-in this used to reintroduce is gone.
# Admission is unchanged, and a refusal simply leaves the round where it was.
_ROOT_CAST_ON = _env_on(_ROOT_CAST_ENV, "1")
_ROOT_CAST_ADMIT: dict = {}
_ROOT_CAST_COUNTS = {"absorbed": 0, "admitted": 0, "rejected": 0, "capture_skips": 0}


def set_root_cast_enabled(on: bool) -> None:
    """Test/bench hook for the root-round absorption. Admission still applies per shape."""
    global _ROOT_CAST_ON
    _ROOT_CAST_ON = bool(on)


def root_cast_counts() -> dict:
    return {**_ROOT_CAST_COUNTS, "enabled": _ROOT_CAST_ON, "shapes": {str(k): v for k, v in _ROOT_CAST_ADMIT.items()}}


def tree_all_reduce_rounded(
    partial: torch.Tensor,
    group=None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """``tree_all_reduce(partial).to(out_dtype)``, with the round inside the kernel where it is proven.

    The caller must guarantee, and this cannot check, that the round is the very next thing to happen to the
    root -- in particular that no bias comes between them, since a bias added after a bf16 round is a
    TP-dependent expression. Returns the same bits either way.
    """
    want_narrow = partial.dtype == torch.float32 and out_dtype == torch.bfloat16
    world = dist.get_world_size(group) if dist.is_initialized() else 1
    if not (_ROOT_CAST_ON and want_narrow and world > 1):
        return tree_all_reduce(partial, group=group).to(out_dtype)

    key = (world, partial.numel(), str(partial.dtype))
    verdict = _ROOT_CAST_ADMIT.get(key)
    if verdict is None:
        if _capturing():
            _ROOT_CAST_COUNTS["capture_skips"] += 1
            return tree_all_reduce(partial, group=group).to(out_dtype)
        verdict = _root_cast_admit(key, partial, group)
    if not verdict:
        return tree_all_reduce(partial, group=group).to(out_dtype)
    _ROOT_CAST_COUNTS["absorbed"] += 1
    return tree_all_reduce(partial, group=group, root_dtype=torch.bfloat16)


def _root_cast_admit(key, partial: torch.Tensor, group) -> bool:
    """Prove the in-kernel round equals the separate-kernel round at this shape, on live operands.

    The reference is ``reduce(fp32 root).to(bf16)``, the expression being replaced, rather than
    ``reduce(bf16 root)``, which would compare the new path to itself.
    """
    ref = tree_all_reduce(partial, group=group).to(torch.bfloat16).clone()
    try:
        got = tree_all_reduce(partial, group=group, root_dtype=torch.bfloat16).clone()
    except Exception as e:  # noqa: BLE001
        local_ok, why = False, f"the narrowing-root path raised: {e!r}"
    else:
        local_ok = torch.equal(_bits(ref), _bits(got))
        why = (
            ""
            if local_ok
            else (
                f"{int((_bits(ref) != _bits(got)).sum().item())}/{ref.numel()} elements differ -- "
                "the in-kernel round is NOT the same rounding as torch's"
            )
        )
    agreed = _agree(local_ok, group, partial.device)
    if not agreed:
        _ROOT_CAST_ADMIT[key] = False
        _ROOT_CAST_COUNTS["rejected"] += 1
        msg = (
            f"[ISOEXEC-PIK] WARNING: absorbing the ROOT's fp32->bf16 round into the reduce "
            f"kernel was REFUSED at {key} -- "
            f"{why or 'a PEER rank refused it'}. The result is unaffected (the round happens "
            "in its own kernel, as it always has) but the launch is not being saved here. "
            "Read pik.allreduce.root_cast_counts()."
        )
        print(msg, flush=True)
        logger.warning(msg)
        return False
    _ROOT_CAST_ADMIT[key] = True
    _ROOT_CAST_COUNTS["admitted"] += 1
    return True


def _tree_reduce(stacked: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """stacked: [C, ...] fp32 contiguous -> out: [...] fp32, combined by the tree."""
    import triton

    c = stacked.shape[0]
    n = out.numel()
    # Through the memoized launcher; with SKYRL_ISOEXEC_PIK_FASTLAUNCH off this is the plain bracket call.
    fastlaunch(
        tree_reduce_kernel(c), (triton.cdiv(n, 1024),), stacked, out, n, stacked.stride(0), BLOCK=1024, num_warps=4
    )
    return out


# Transport observability. ``tree_all_reduce`` falls back from the P2P path to NCCL on any exception, and
# because both transports evaluate the identical tree the fallback is bitwise-safe -- which is exactly why it
# is invisible. Its only symptom is latency, roughly 2x per row-parallel combine, so the fallback is loud once
# (naming the exception) and counted: ``transport_counts()`` / ``transport_state()`` make the live transport a
# readable fact rather than an inference from an install banner.
#
# A host may register a hook that fires whenever the resolved transport state changes -- first call, and again
# the first time a P2P process degrades. Pure Python, no CUDA, so it is safe inside graph capture.
_TRANSPORT_COUNTS = {"p2p_calls": 0, "nccl_calls": 0, "p2p_fallbacks": 0}
_FIRST_FALLBACK: dict = {}
_TRANSPORT_HOOKS: list = []
_LAST_STATE: str = "none"


def transport_state() -> str:
    """Which transport this process's tree all-reduce has actually been running on.

    ``p2p_fallback`` means the P2P path was selected and then threw, so the run pays NCCL latency while the
    install banner says P2P. ``p2p_and_nccl`` without a fallback is normal only when both were asked for.
    """
    c = _TRANSPORT_COUNTS
    if c["p2p_fallbacks"]:
        return "p2p_fallback"
    if c["p2p_calls"] and c["nccl_calls"]:
        return "p2p_and_nccl"
    if c["p2p_calls"]:
        return "p2p"
    if c["nccl_calls"]:
        return "nccl"
    return "none"


def transport_counts() -> dict:
    """Per-process call counts plus the first fallback's exception, if any."""
    return {**_TRANSPORT_COUNTS, "state": transport_state(), "first_fallback": dict(_FIRST_FALLBACK) or None}


def register_transport_hook(fn) -> None:
    """Call ``fn(state, impl_fn)`` whenever the resolved transport state changes.

    Fires immediately if a transport already resolved in this process, so registration order does not matter.
    Hook exceptions are swallowed.
    """
    _TRANSPORT_HOOKS.append(fn)
    if _LAST_STATE != "none":
        impl = _tree_all_reduce_p2p if _LAST_STATE == "p2p" else _tree_reduce
        try:
            fn(_LAST_STATE, impl)  # catch-up, this hook only
        except Exception as e:  # noqa: BLE001
            logger.warning("[pik] transport hook %r failed: %r", fn, e)


def _fire_transport_hooks(state: str, impl_fn) -> None:
    for fn in list(_TRANSPORT_HOOKS):
        try:
            fn(state, impl_fn)
        except Exception as e:  # noqa: BLE001 -- a hook must never break the reduce
            logger.warning("[pik] transport hook %r failed: %r", fn, e)


def _note_transport(kind: str, impl_fn) -> None:
    global _LAST_STATE
    _TRANSPORT_COUNTS[f"{kind}_calls"] += 1
    state = transport_state()
    if state != _LAST_STATE:
        _LAST_STATE = state
        _fire_transport_hooks(state, impl_fn)


def _note_p2p_fallback(exc: BaseException, partial: torch.Tensor, world: int) -> None:
    """Count the fallback and report it once per process, naming the exception."""
    global _FIRST_FALLBACK
    _TRANSPORT_COUNTS["p2p_fallbacks"] += 1
    if _FIRST_FALLBACK:
        return
    _FIRST_FALLBACK = {
        "exception": repr(exc),
        "shape": tuple(partial.shape),
        "dtype": str(partial.dtype),
        "world": world,
    }
    msg = (
        "[ISOEXEC-PIK] WARNING: pik symmetric-memory P2P all-reduce FAILED and fell back to the "
        f"NCCL transport -- {exc!r} (first seen at shape={tuple(partial.shape)} "
        f"dtype={partial.dtype} world={world}). The RESULT is unaffected (both transports "
        "evaluate the identical tree), but every subsequent all-reduce in this process pays NCCL "
        "latency (~2x at decode payloads) while the engine banner claims P2P is kept. Read "
        "pik.allreduce.transport_counts() for the running tally."
    )
    print(msg, flush=True)
    logger.warning(msg)


def tree_all_reduce(
    partial: torch.Tensor,
    group=None,
    out: torch.Tensor | None = None,
    backend: str | None = None,
    root_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Combine per-rank subtree partials into the full tree, on every rank.

    Takes this rank's ``ti_gemm`` output and returns a result bit-identical on all ranks and across all TP
    sizes. Both backends realize the same tree, so ``backend`` is a pure performance knob.

    ``root_dtype`` is the dtype of the emitted root, defaulting to ``partial.dtype``. A caller whose
    downstream re-reduces the root must request fp32: a bf16 root turns round-after-sum into sum-of-rounded.
    """
    # An fp32 partial is an internal tree node (or fp32 leaves) and must stay fp32. A bf16 partial means the
    # rank's partial is itself a leaf, legal on the wire because the leaf is the only TP-independent rounding
    # point; the tree still adds in fp32.
    assert partial.dtype in (
        torch.float32,
        torch.bfloat16,
    ), f"partial must be fp32 (internal node) or bf16 (leaf, m=1), got {partial.dtype}"
    # Legal root dtypes: the partial's own, fp32 (widening a bf16 leaf wire), or bf16 (narrowing the root).
    # Narrowing is sound because the root is the same node at every TP size, but the caller must ensure the
    # round is the next operation, which is why ``tree_all_reduce_rounded`` is the entry point for it.
    assert root_dtype in (None, partial.dtype, torch.float32, torch.bfloat16), (
        f"root_dtype={root_dtype} with partial dtype {partial.dtype}: the root may be the "
        "partial's dtype, fp32 (widened bf16 leaf wire), or bf16 (narrowed root round)"
    )
    world = dist.get_world_size(group) if dist.is_initialized() else 1
    if world == 1:
        # No reduce: the root is the partial itself. The bf16 -> fp32 upcast is exact.
        return partial.to(root_dtype) if root_dtype is not None else partial

    partial = partial.contiguous()

    backend = backend or ("p2p" if p2p_available(group) else "nccl")
    if backend == "p2p":
        try:
            # Do not pre-allocate `out`: the two-shot P2P path can hand back its symmetric output buffer
            # directly, and forcing a destination reintroduces a copy-out.
            res = _tree_all_reduce_p2p(partial, group, out, root_dtype)
        except Exception as e:  # noqa: BLE001 -- no NVLink / no symm-mem: fall back
            # Report it once, naming `e`: a silent fallback is what makes the transport unobservable.
            _note_p2p_fallback(e, partial, world)
        else:
            _note_transport("p2p", _tree_all_reduce_p2p)
            return res

    _note_transport("nccl", _tree_reduce)
    return _tree_all_reduce_nccl(partial, group, out, root_dtype)


def _tree_all_reduce_nccl(partial: torch.Tensor, group, out: torch.Tensor | None, root_dtype) -> torch.Tensor:
    """The NCCL transport: all_gather / all_to_all move bytes, ``_tree_reduce`` does the arithmetic."""
    world = dist.get_world_size(group)

    # This path runs the tree in fp32 whatever the wire dtype: upcast a bf16 leaf (exact), then round the
    # root only if the caller asked for bf16. Same adds, same order, same single rounding point as P2P.
    want_bf16_root = (root_dtype or partial.dtype) == torch.bfloat16
    if partial.dtype == torch.bfloat16:
        partial = partial.to(torch.float32)
    if out is None or out.dtype != torch.float32:
        out = torch.empty_like(partial)

    if partial.numel() * 4 <= ONESHOT_MAX_BYTES:
        # One-shot: every rank gathers every partial and evaluates the whole tree.
        buf = torch.empty((world, *partial.shape), device=partial.device, dtype=torch.float32)
        dist.all_gather_into_tensor(buf, partial, group=group)
        res = _tree_reduce(buf, out)
        return res.to(torch.bfloat16) if want_bf16_root else res

    # Two-shot: all-to-all the slices, tree-reduce the slice this rank owns, all-gather back. Same bytes on
    # the wire as a ring all-reduce, but the arithmetic is ours.
    n = partial.numel()
    assert n % world == 0, (
        f"two-shot needs numel {n} divisible by world {world}; " "pad M or N, or lower ONESHOT_MAX_BYTES"
    )
    chunk = n // world
    flat = partial.view(-1)

    recv = torch.empty((world, chunk), device=partial.device, dtype=torch.float32)
    dist.all_to_all_single(recv.view(-1), flat, group=group)  # recv[r] = rank r's slice `me`

    mine = torch.empty(chunk, device=partial.device, dtype=torch.float32)
    _tree_reduce(recv, mine)

    dist.all_gather_into_tensor(out.view(-1), mine, group=group)
    return out.to(torch.bfloat16) if want_bf16_root else out


def tree_reduce_scatter(
    partial: torch.Tensor,
    group=None,
    root_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """The two-shot tree all-reduce minus the trailing all-gather: each rank keeps its slice.

    Returns this rank's contiguous first-dim slice of the full tree root, in rank order (rank r owns rows
    ``[r*M/world, (r+1)*M/world)``), which is Megatron's sequence-parallel slicing convention -- so this is a
    drop-in replacement for ``reduce_scatter_to_sequence_parallel_region``'s NCCL SUM, with pik's fixed tree
    doing the arithmetic. All-gathering the per-rank outputs across ``group`` reproduces
    ``tree_all_reduce(partial)`` bit for bit at every payload size, which is why the two-shot structure is used
    unconditionally: payload size must never change the expression tree.

    NCCL transport only. The trainer is the only caller -- sequence parallelism is a trainer-side layout -- and
    it runs with the P2P symmetric-memory path disabled.
    """
    assert partial.dtype in (
        torch.float32,
        torch.bfloat16,
    ), f"partial must be fp32 (internal node) or bf16 (leaf, m=1), got {partial.dtype}"
    assert root_dtype in (None, partial.dtype, torch.float32), (
        f"root_dtype={root_dtype} with partial dtype {partial.dtype}: the only decoupled "
        "combination is bf16 partial (leaf wire) + fp32 root"
    )
    world = dist.get_world_size(group) if dist.is_initialized() else 1
    if world == 1:
        # No reduce and no scatter: the whole tensor is this rank's slice. The bf16 -> fp32 upcast is exact.
        return partial.to(root_dtype) if root_dtype is not None else partial

    # Megatron scatters the first dim in equal chunks, and a flat world-way split only coincides with
    # first-dim blocks when the first dim divides evenly. Refuse anything else rather than mis-slice.
    assert partial.shape[0] % world == 0, (
        f"tree_reduce_scatter: first dim {partial.shape[0]} not divisible by world {world} "
        f"(shape {tuple(partial.shape)}). Megatron sequence parallelism requires the sequence "
        "dim to divide the TP size; pad the sequence (or run SP off for this shape)."
    )

    partial = partial.contiguous()
    want_bf16_root = (root_dtype or partial.dtype) == torch.bfloat16
    if partial.dtype == torch.bfloat16:
        # Exact upcast: the tree adds in fp32 on every transport.
        partial = partial.to(torch.float32)

    n = partial.numel()
    chunk = n // world  # first-dim divisibility above implies numel divisibility
    out_shape = (partial.shape[0] // world, *partial.shape[1:])
    flat = partial.view(-1)

    # The two-shot of _tree_all_reduce_nccl without the trailing all-gather: all_to_all delivers rank r's
    # copy of this rank's slice, the tree combines them, and this rank keeps the result.
    recv = torch.empty((world, chunk), device=partial.device, dtype=torch.float32)
    dist.all_to_all_single(recv.view(-1), flat, group=group)  # recv[r] = rank r's slice `me`

    mine = torch.empty(chunk, device=partial.device, dtype=torch.float32)
    _tree_reduce(recv, mine)

    _note_transport("nccl", _tree_reduce)
    res = mine.view(out_shape)
    return res.to(torch.bfloat16) if want_bf16_root else res
