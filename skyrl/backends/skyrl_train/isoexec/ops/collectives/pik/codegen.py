"""Source generator for the fused leaf-tree GEMM and the tree all-reduce kernels.

Triton turns loops in a kernel body into real control flow, so the carry-stack cannot be walked at trace time;
instead straight-line Triton source is emitted per local-leaf count, which also pins the evaluation order. The
emitted tree is exactly ``pik.plan.combine_order`` (left operand carries the lower leaf indices); the post-order
walk keeps only ``log2(m)+1`` accumulators live, which is a register concern and not a numerical one.
"""

from __future__ import annotations

import functools
import importlib.util
import os
import pathlib
import sys

_LEAF_BODY = """
    x_ptrs_{d} = x_ptr + offs_m[:, None] * stride_xm + ({lo} * LEAF_K + offs_k)[None, :] * stride_xk
    w_ptrs_{d} = w_ptr + offs_n[:, None] * stride_wn + ({lo} * LEAF_K + offs_k)[None, :] * stride_wk
    t{d} = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _k{d} in range(k_tiles):
        a{d} = tl.load(x_ptrs_{d})
        b{d} = tl.load(w_ptrs_{d})
        t{d} = tl.dot(a{d}, tl.trans(b{d}), t{d})
        x_ptrs_{d} += BLOCK_K * stride_xk
        w_ptrs_{d} += BLOCK_K * stride_wk
{round}"""


_LOAD_BODY = "    t{d} = tl.load(p_ptr + {lo} * stride_pl + offs, mask=mask, other=0.0)\n"

# bf16 leaves: load bf16, add in fp32. Rounding happens once, at the leaf -- the only boundary that does not
# move with TP -- and every internal tree node stays fp32.
_LOAD_BODY_BF16 = "    t{d} = tl.load(p_ptr + {lo} * stride_pl + offs, mask=mask, other=0.0).to(tl.float32)\n"


_ROUND = "    t{d} = t{d}.to(tl.bfloat16).to(tl.float32)   # round the LEAF only\n"


def _emit(lo: int, hi: int, depth: int, out: list[str], leaf: str, bf16_leaf: bool = False) -> None:
    """Post-order emit of tree(leaves[lo:hi]) into variable t{depth}."""
    if hi - lo == 1:
        rnd = _ROUND.format(d=depth) if bf16_leaf else ""
        out.append(leaf.format(d=depth, lo=lo, round=rnd) if "{round}" in leaf else leaf.format(d=depth, lo=lo))
        return
    mid = (lo + hi) // 2
    _emit(lo, mid, depth, out, leaf, bf16_leaf)  # left subtree  -> t{depth}
    _emit(mid, hi, depth + 1, out, leaf, bf16_leaf)  # right subtree -> t{depth+1}
    out.append(f"    t{depth} = t{depth} + t{depth + 1}\n")  # lower indices on the left


def tree_source(m: int, leaf: str = _LEAF_BODY, bf16_leaf: bool = False) -> str:
    body: list[str] = []
    _emit(0, m, 0, body, leaf, bf16_leaf)
    return "".join(body)


_KERNEL_TMPL = """
import triton
import triton.language as tl
from pik.gemm import _tile_ids, _configs, _prune, _AUTOTUNE_KEY

@triton.autotune(configs=_configs(), key=_AUTOTUNE_KEY,
                 prune_configs_by={{"early_config_prune": _prune}})
@triton.jit
def kernel(
    x_ptr, w_ptr, o_ptr,
    M, N, LEAF_K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    NUM_LOCAL_LEAVES: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid_m, pid_n = _tile_ids(tl.program_id(0), M, N, BLOCK_M, BLOCK_N, GROUP_M)
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    k_tiles = LEAF_K // BLOCK_K

{tree}
    om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(o_ptr + om[:, None] * stride_om + on[None, :] * stride_on, t0,
             mask=(om[:, None] < M) & (on[None, :] < N))
"""


# A CTA handles `g` consecutive leaves starting at split*g, carries their subtree in registers/TMEM, and
# writes one partial. With s = m/g CTAs per output tile this interpolates between g=1 (one accumulator, but
# (2m+1) passes of workspace traffic) and g=m (log2(m)+1 accumulators, one write); the optimum is usually in
# between and is found by measurement. cuBLASLt cannot serve this: beta=1 accumulation fuses only a pair of
# leaves, because a balanced tree node is exactly two independently-rounded summands.
_SPLIT_LEAF_BODY = """
    x_ptrs_{d} = x_ptr + offs_m[:, None] * stride_xm + ((leaf0 + {lo}) * LEAF_K + offs_k)[None, :] * stride_xk
    w_ptrs_{d} = w_ptr + offs_n[:, None] * stride_wn + ((leaf0 + {lo}) * LEAF_K + offs_k)[None, :] * stride_wk
    t{d} = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _k{d} in range(k_tiles):
        a{d} = tl.load(x_ptrs_{d})
        b{d} = tl.load(w_ptrs_{d})
        t{d} = tl.dot(a{d}, tl.trans(b{d}), t{d})
        x_ptrs_{d} += BLOCK_K * stride_xk
        w_ptrs_{d} += BLOCK_K * stride_wk
{round}"""

_SPLIT_TMPL = """
import triton
import triton.language as tl
from pik.gemm import _tile_ids, _configs, _prune, _AUTOTUNE_KEY

@triton.autotune(configs=_configs(), key=_AUTOTUNE_KEY,
                 prune_configs_by={{"early_config_prune": _prune}})
@triton.jit
def kernel(
    x_ptr, w_ptr, p_ptr,
    M, N, LEAF_K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ps, stride_pm, stride_pn,
    NUM_LOCAL_LEAVES: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    split = tl.program_id(1)
    leaf0 = split * {g}
    pid_m, pid_n = _tile_ids(tl.program_id(0), M, N, BLOCK_M, BLOCK_N, GROUP_M)
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    k_tiles = LEAF_K // BLOCK_K

{tree}
    om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(p_ptr + split * stride_ps + om[:, None] * stride_pm + on[None, :] * stride_pn,
             t0, mask=(om[:, None] < M) & (on[None, :] < N))
"""


@functools.lru_cache(maxsize=None)
def split_tree_kernel(g: int, bf16_leaf: bool = False, store_bf16: bool = False):
    """Each CTA carries the subtree of `g` consecutive leaves and writes one partial.

    ``store_bf16`` is legal only when ``g == 1``: the partial is then a leaf, the one node whose rounding does
    not depend on TP. For ``g > 1`` the partial is an internal tree node and must stay fp32.
    """
    _check_pow2(g)
    assert not (store_bf16 and g > 1), "only a leaf (g==1) may be stored in bf16"
    src = _SPLIT_TMPL.format(g=g, tree=tree_source(g, _SPLIT_LEAF_BODY, bf16_leaf))
    if store_bf16:
        src = src.replace("             t0,", "             t0.to(tl.bfloat16),")
    tag = f"_split_tree_g{g}{'_bl' if bf16_leaf else ''}{'_sb' if store_bf16 else ''}"
    return _load(tag, src)


_REDUCE_TMPL = """
import triton
import triton.language as tl

@triton.jit
def kernel(
    p_ptr, o_ptr, n_elem,
    stride_pl,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem

{tree}
    tl.store(o_ptr + offs, t0, mask=mask)
"""


_CACHE_DIR = pathlib.Path(os.environ.get("PIK_CACHE", pathlib.Path.home() / ".cache" / "pik"))


def _load(name: str, src: str):
    """Materialize generated source as a real module (Triton needs inspect.getsource)."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / f"{name}.py"
    if not path.exists() or path.read_text() != src:
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(src)
        tmp.replace(path)  # atomic: concurrent ranks must not read a half-written file
    spec = importlib.util.spec_from_file_location(f"pik.{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    from .autotune_cache import register

    return register(name, mod.kernel)


def _check_pow2(m: int) -> None:
    if m < 1 or (m & (m - 1)):
        raise ValueError(f"leaf count must be a power of two, got {m}")


@functools.lru_cache(maxsize=None)
def fused_tree_kernel(m: int, bf16_leaf: bool = False):
    """Autotuned GEMM that walks `m` local leaves and combines them in-register."""
    _check_pow2(m)
    src = _KERNEL_TMPL.format(tree=tree_source(m, _LEAF_BODY, bf16_leaf))
    return _load(f"_fused_tree_m{m}{'_bl' if bf16_leaf else ''}", src)


@functools.lru_cache(maxsize=None)
def tree_reduce_kernel(m: int, in_bf16: bool = False, out_bf16: bool = False):
    """Elementwise combine of `m` leaf partials, using the same tree and generator.

    ``in_bf16``: leaves arrive rounded to bf16; the adds stay fp32, so no internal node is ever rounded.
    ``out_bf16``: round the tree's root at the store, for a result about to be cast to bf16 anyway.
    """
    _check_pow2(m)
    leaf = _LOAD_BODY_BF16 if in_bf16 else _LOAD_BODY
    src = _REDUCE_TMPL.format(tree=tree_source(m, leaf))
    if out_bf16:
        src = src.replace(
            "tl.store(o_ptr + offs, t0, mask=mask)", "tl.store(o_ptr + offs, t0.to(tl.bfloat16), mask=mask)"
        )
    tag = f"_tree_reduce_m{m}{'_bi' if in_bf16 else ''}{'_bo' if out_bf16 else ''}"
    return _load(tag, src)


# Fused P2P tree all-reduce. With symmetric memory the peer pointers are held directly, so one kernel reads
# every rank's partial over NVLink, applies the tree, and pushes the result back into every rank's output
# buffer -- one pass instead of a2a -> reduce -> all-gather, with the arithmetic still ours.

_PEER_LOAD = "    t{d} = tl.load(in{lo} + offs, mask=mask, other=0.0)\n"
_PEER_LOAD_BF16 = "    t{d} = tl.load(in{lo} + offs, mask=mask, other=0.0).to(tl.float32)\n"


def _ar_args(c: int, push: bool) -> tuple[str, str]:
    ins = ", ".join(f"in{i}" for i in range(c))
    outs = ", ".join(f"out{i}" for i in range(c)) if push else "out0"
    return ins, outs


def _ar_core(c: int, push: bool, in_bf16: bool, out_bf16: bool, vec: int = 1) -> str:
    """The arithmetic: offsets, the tree, the stores, and nothing else.

    Emitted byte-identically into every template here, so the barrier-fused variants are bitwise-equal to the
    unfused one by construction.

    ``vec`` changes only the element-to-thread layout: ``vec > 1`` shapes the same BLOCK as ``(BLOCK//vec,
    vec)`` so each thread owns ``vec`` contiguous elements and Triton emits 128-bit vector accesses. The set of
    elements, the per-element load sources, the tree association and the single rounding are unchanged.
    """
    val = "t0.to(tl.bfloat16)" if out_bf16 else "t0"
    stores = (
        "".join(f"    tl.store(out{i} + offs, {val}, mask=mask)\n" for i in range(c))
        if push
        else f"    tl.store(out0 + offs, {val}, mask=mask)\n"
    )
    leaf = _PEER_LOAD_BF16 if in_bf16 else _PEER_LOAD
    if vec == 1:
        offs = "    offs = base + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n"
    else:
        offs = (
            f"    offs = base + tl.program_id(0) * BLOCK + "
            f"(tl.arange(0, BLOCK // {vec})[:, None] * {vec} + tl.arange(0, {vec})[None, :])\n"
        )
    return f"""{offs}    mask = offs < base + n

{tree_source(c, leaf)}
{stores}"""


def _ar_tmpl(c: int, push: bool, in_bf16: bool, out_bf16: bool, vec: int = 1) -> str:
    ins, outs = _ar_args(c, push)
    return f"""
import triton
import triton.language as tl

@triton.jit
def kernel({ins}, {outs}, base, n, BLOCK: tl.constexpr):
{_ar_core(c, push, in_bf16, out_bf16, vec)}"""


@functools.lru_cache(maxsize=None)
def p2p_allreduce_kernel(c: int, push: bool, in_bf16: bool = False, out_bf16: bool | None = None, vec: int = 1):
    """Read `c` peer partials, apply the tree, and write locally (one-shot) or to every peer (two-shot push).

    ``in_bf16``: the wire payload is bf16. Legal only when each rank's partial is a leaf (m == 1), the only
    TP-independent rounding point; the adds are fp32 regardless.
    ``out_bf16``: round the tree's root to bf16 at the store; defaults to ``in_bf16``. The two are decoupled
    because the MoE combine needs ``in_bf16=True, out_bf16=False`` -- its downstream is a fixed-order top-k
    fp32 sum, so a bf16 root would turn round-after-sum into sum-of-rounded.
    ``vec``: element-to-thread layout (see ``_ar_core``).
    """
    _check_pow2(c)
    if out_bf16 is None:
        out_bf16 = in_bf16
    tag = (
        f"_p2p_ar_c{c}_{'push' if push else 'local'}{'_bi' if in_bf16 else ''}{'_bo' if out_bf16 else ''}"
        f"{f'_v{vec}' if vec != 1 else ''}"
    )
    return _load(tag, _ar_tmpl(c, push, in_bf16, out_bf16, vec))


# Fused barrier + tree all-reduce: one launch per site instead of two (one-shot) or three (two-shot). A
# barrier is a spin on a flag in symmetric memory and the P2P kernel already reads peer buffers over NVLink,
# so both fit in one kernel. ``_ar_core`` is emitted byte-identically here and in ``_ar_tmpl``, so the fused
# kernel is bitwise-equal to the unfused one by construction.
#
# The barrier is the classic per-block cross-rank handshake: block b of rank R publishes its epoch into every
# peer's flag[R][b] with release at system scope, then spins on its own flag[p][b] for every p with acquire at
# system scope. Per-block rather than device-wide is sufficient in both directions:
#   - leading: a block of this kernel running implies the previous kernel on that rank retired (stream order),
#     so peer arrival means peer partial complete;
#   - trailing (two-shot push): the grid is identical on every rank, so block b of rank R writes exactly the
#     index range block b of every peer writes, and R's kernel ends only after every peer's blocks published.
#
# Epochs are monotone and compared as ``v - seq >= 0``, not ``==``: a rank may legitimately be one call ahead
# of a peer, and an equality test would hang on a flag that already moved past. The subtraction is wrap-safe
# in int32. The counters live in device memory and are bumped by the kernel itself, which is what makes this
# CUDA-graph safe -- a replay re-runs the increment rather than replaying a frozen constant. Publish uses a
# shape-(1,) tensor op so exactly one lane issues each remote atomic.
#
# Publishing with ``sem="release", scope="sys"`` after ``tl.debug_barrier()`` is the PTX release pattern: the
# barrier orders the block's data writes before the flag write, and release is cumulative, so a peer that
# acquires the flag sees the data. Getting this wrong is a silent wrong-value failure, never a crash.
_FUSED_PUBLISH = '    tl.atomic_xchg(fl{i} + my_slot{r} + pid + one, sv{r}, sem="release", scope="sys")\n'

_FUSED_WAIT = """    w{r} = {wbase} + lane * NB + pid
    ready{r} = 0
    while ready{r} == 0:
        v{r} = tl.atomic_add(self_fl + w{r}, 0, sem="acquire", scope="sys")
        ready{r} = tl.min(tl.where(v{r} - seq{r} >= 0, 1, 0))
"""


def _fused_barrier(r: int, c: int, wbase: str, ep: str) -> str:
    """Emit one full cross-rank barrier: arrive at every peer, then wait for every peer.

    The ``tl.debug_barrier()`` between the epoch load and the epoch store is load-bearing. The load is
    unpredicated (every thread needs ``seq`` for the spin) while the store is predicated on tid==0, so without
    the barrier one warp can bump the counter before another has read it; that warp then waits on an epoch no
    peer will ever publish and the collective hangs.
    """
    pub = "".join(_FUSED_PUBLISH.format(i=i, r=r) for i in range(c))
    return (
        f"    seq{r} = tl.load(ep_ptr + {ep}) + 1\n"
        f"    tl.debug_barrier()   # every thread has read the old epoch before any bumps it\n"
        f"    tl.store(ep_ptr + {ep}, seq{r})\n"
        f"    sv{r} = seq{r} + tl.zeros((1,), tl.int32)\n"
        f"{pub}"
        f"{_FUSED_WAIT.format(r=r, wbase=wbase)}"
    )


def _fused_ar_tmpl(c: int, push: bool, in_bf16: bool, out_bf16: bool) -> str:
    ins, outs = _ar_args(c, push)
    fls = ", ".join(f"fl{i}" for i in range(c))
    lead = _fused_barrier(0, c, "0", "pid")
    if push:
        trail = "\n    tl.debug_barrier()   # this block's pushes precede its release\n" + _fused_barrier(
            1, c, "wbase1", "NB + pid"
        )
    else:
        trail = ""
    return f"""
import triton
import triton.language as tl

@triton.jit
def kernel({ins}, {outs}, {fls}, self_fl, ep_ptr,
           my_slot0, my_slot1, wbase1, NB,
           base, n, BLOCK: tl.constexpr, WORLD: tl.constexpr):
    pid = tl.program_id(0)
    one = tl.arange(0, 1)
    lane = tl.arange(0, WORLD)

{lead}    tl.debug_barrier()   # every peer has arrived before any thread reads peer memory

{_ar_core(c, push, in_bf16, out_bf16)}{trail}"""


# Device-wide fused barrier + tree all-reduce: one launch, world-count atomics. The per-block variant above
# publishes to every peer once per block, so a 128-block site issues grid * world remote atomics per region
# where a device-wide barrier issues world. Here:
#
#   * leading: block 0 alone publishes one arrival per peer (release/sys) and spins for every peer's, then
#     fans the release out through a local flag (release/gpu) every other block spins on -- L2 polling, not
#     NVLink traffic. The stream-order argument for arrival == staging complete is unchanged.
#   * trailing (two-shot only): each block, after its stores and a fence, bumps a local grid-completion
#     counter (acq_rel/gpu). The block that observes the count reach ``seq * GRID`` -- the last block of this
#     call, whichever it is on this replay -- publishes one trailing arrival per peer and spins for every
#     peer's; every other block exits, and the kernel cannot retire until that last block is released.
#
# No cooperative launch is required: block 0 is in the first wave by scheduling order, so the release always
# arrives, and an oversubscribed grid drains because pre-release blocks only spin after block 0 is resident.
#
# Epochs must advance uniformly across blocks for ``seq * GRID`` to identify the last block, so the flag/epoch
# state is keyed per grid: every call on a given state object launches the same grid and every block bumps its
# own counter. Counters are monotone, compared with serial-number arithmetic, and bumped in-kernel, so this is
# graph-safe for the same reason the per-block kernel is. ``_ar_core`` is pasted byte-identically here too.
_DW_PUBLISH = '        tl.atomic_xchg(fl{i} + my_slot{r} + one, sv, sem="release", scope="sys")\n'

_DW_WAIT = """        w{r} = {wbase} + lane * NB
        ready{r} = 0
        while ready{r} == 0:
            v{r} = tl.atomic_add(self_fl + w{r}, 0, sem="acquire", scope="sys")
            ready{r} = tl.min(tl.where(v{r} - seq >= 0, 1, 0))
"""


def _dw_fused_ar_tmpl(c: int, push: bool, in_bf16: bool, out_bf16: bool) -> str:
    ins, outs = _ar_args(c, push)
    fls = ", ".join(f"fl{i}" for i in range(c))
    pub0 = "".join(_DW_PUBLISH.format(i=i, r=0) for i in range(c))
    pub1 = "".join(_DW_PUBLISH.format(i=i, r=1) for i in range(c))
    trail = ""
    if push:
        trail = f"""
    tl.debug_barrier()   # this block's pushes precede its completion bump
    prev = tl.atomic_add(cnt_ptr + one, tl.full((1,), 1, tl.int32), sem="acq_rel", scope="gpu")
    is_last = tl.min(tl.where(prev - (seq * GRID - 1) == 0, 1, 0))
    if is_last == 1:
{pub1}{_DW_WAIT.format(r=1, wbase="WORLD * NB")}"""
    return f"""
import triton
import triton.language as tl

@triton.jit
def kernel({ins}, {outs}, {fls}, self_fl, ep_ptr, rel_ptr, cnt_ptr,
           my_slot0, my_slot1, NB,
           base, n, BLOCK: tl.constexpr, WORLD: tl.constexpr, GRID: tl.constexpr):
    pid = tl.program_id(0)
    one = tl.arange(0, 1)
    lane = tl.arange(0, WORLD)
    seq = tl.load(ep_ptr + pid) + 1
    tl.debug_barrier()   # every thread has read the old epoch before any bumps it
    tl.store(ep_ptr + pid, seq)
    sv = seq + tl.zeros((1,), tl.int32)
    if pid == 0:
{pub0}{_DW_WAIT.format(r=0, wbase="0")}        tl.atomic_xchg(rel_ptr + one, sv, sem="release", scope="gpu")
    else:
        ready9 = 0
        while ready9 == 0:
            v9 = tl.atomic_add(rel_ptr + one, tl.zeros((1,), tl.int32), sem="acquire", scope="gpu")
            ready9 = tl.min(tl.where(v9 - seq >= 0, 1, 0))
    tl.debug_barrier()   # every peer has arrived before any thread reads peer memory

{_ar_core(c, push, in_bf16, out_bf16)}{trail}"""


@functools.lru_cache(maxsize=None)
def dw_fused_p2p_allreduce_kernel(c: int, push: bool, in_bf16: bool = False, out_bf16: bool | None = None):
    """``p2p_allreduce_kernel`` with device-wide barriers folded in: one launch, world atomics.

    Signature (all pointers into symmetric memory unless noted):
      in0..in{c-1}   peer input staging buffers          (read)
      out0..out{c-1} peer output buffers (push) / out0 = local destination (one-shot)
      fl0..fl{c-1}   peer flag arrays                    (written: block 0 / last block arrive)
      self_fl        THIS rank's flag array              (spun on by block 0 / the last block)
      ep_ptr         LOCAL int32[grid] per-block epochs, bumped in-kernel (graph-safe)
      rel_ptr        LOCAL int32[1] release flag block 0 fans the leading barrier out through
      cnt_ptr        LOCAL int32[1] monotone grid-completion counter (two-shot trailing)
      my_slot0/1     rank*NB / (WORLD+rank)*NB -- where this rank's arrival lands in a peer array
      NB             flag stride (1 for this kernel's dedicated pools)
      base, n        exactly as p2p_allreduce_kernel

    Same tree, same order, and same dtypes as ``p2p_allreduce_kernel``.
    """
    _check_pow2(c)
    if out_bf16 is None:
        out_bf16 = in_bf16
    tag = f"_p2p_dwfused_ar_c{c}_{'push' if push else 'local'}" f"{'_bi' if in_bf16 else ''}{'_bo' if out_bf16 else ''}"
    return _load(tag, _dw_fused_ar_tmpl(c, push, in_bf16, out_bf16))


@functools.lru_cache(maxsize=None)
def fused_p2p_allreduce_kernel(c: int, push: bool, in_bf16: bool = False, out_bf16: bool | None = None):
    """``p2p_allreduce_kernel`` with its barrier(s) folded in: one launch per site.

    Signature (all pointers are into symmetric memory unless noted):
      in0..in{c-1}   peer input staging buffers      (read)
      out0..out{c-1} peer output buffers (push) / out0 = local destination (one-shot)
      fl0..fl{c-1}   peer flag arrays                (written: arrive)
      self_fl        THIS rank's flag array          (spun on: wait)
      ep_ptr         LOCAL int32[2*NB] epoch counters, bumped in-kernel (graph-safe)
      my_slot0/1     rank*NB / (WORLD+rank)*NB -- where this rank writes in a peer's array
      wbase1         WORLD*NB -- start of the trailing region in this rank's array
      NB             flag-array stride: blocks per (region, rank). grid must be <= NB.
      base, n        exactly as p2p_allreduce_kernel

    Same tree, same order, and same dtypes as ``p2p_allreduce_kernel``.
    """
    _check_pow2(c)
    if out_bf16 is None:
        out_bf16 = in_bf16
    tag = f"_p2p_fused_ar_c{c}_{'push' if push else 'local'}" f"{'_bi' if in_bf16 else ''}{'_bo' if out_bf16 else ''}"
    return _load(tag, _fused_ar_tmpl(c, push, in_bf16, out_bf16))
