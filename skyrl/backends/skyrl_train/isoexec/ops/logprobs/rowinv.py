"""Row-count- and TP-invariant sampled-token logprob via a fixed-leaf exp-sum tree.

WHY. ATen's fp32 full-vocabulary ``sum`` is not row-count invariant at V=248320: the trainer
reduces ``[B, chunk, V]`` while the engine reduces only the sampled rows ``[N, V]``, and the two
launch geometries pick different fp32 reduction trees, so the lse differs by 1-2 ULP and ~72% of
real Qwen3.5 tokens score non-bitwise even with bitwise-identical logits. The trainer additionally
scores at TP=4 while the engine decodes at TP=8, so the same reduction is also split across ranks
differently. Both defects are one defect -- an unpinned reduction expression -- and this module
pins it pik-style (see ``ops/collectives/pik/plan.py``):

  * the FULL vocabulary ``[0, V)`` is cut into ``G`` fixed contiguous leaves, ``G =
    SKYRL_ISOEXEC_PIK_LEAVES`` (the same contract constant pik uses, default 8). ``V % G != 0``
    DECLINES -- unequal or padded leaves would be a different function, never a silent one;
  * each per-row leaf sum of ``exp(x - m)`` is computed by ONE Triton program per (row, leaf)
    with a fixed ``BLOCK``, iterating the leaf's columns in fixed ascending order and accumulating
    in Kahan-compensated fp32. Per-program work depends only on that row's leaf, so the result is
    row-count invariant by construction, and the leaf-internal order is pinned by the kernel;
  * the ``G`` per-row leaf sums (G floats per row -- tiny) are all-gathered in rank order and
    EVERY rank folds all G with the same fixed balanced binary tree -- the expression of
    ``pik.plan.combine_order(G)`` -- in fp32. Never sequentially: the tree is the contract.

TP only moves leaf OWNERSHIP. A rank at TP=C owns a contiguous leaf range
(``rank_leaf_range``-equivalent: ``[rank*G/C, (rank+1)*G/C)``) which is exactly a subtree of the
combine tree, so every C dividing G evaluates the identical expression -- the derivation is
machine-checked in ``ops/collectives/tests/test_bf16_leaf_scheme_cpu.py``. The refuted
anti-pattern -- each rank reduces its whole shard into one partial and the C partials are combined
-- rounds at rank boundaries that move with C and is NOT invariant; ``tests/rowinv_tp_dist.py``
keeps it as a live negative control.

The remaining two cross-rank steps are exact, hence free: the row max (local ``amax`` +
``all_reduce(MAX)``; max is order-independent) and distributing the sampled raw logit (mask +
``all_reduce(SUM)`` with exactly one nonzero contribution per row). The engine passes an
already-gathered full ``[N, V]`` row (world=1 from this module's view: it computes all G leaves
locally); the trainer passes its ``[..., V/TP]`` shard; both walk the same leaves in the same
order and fold the same tree, so both produce identical bits. This also ELIMINATES the
``[rows, V]`` fp32 full-vocabulary gather -- the wire carries ``rows*G`` fp32 leaf sums.

DECODE-PATH COST. The engine calls this once per decode step at a handful of rows, where kernel
count -- not bytes -- is the price, so the steady state is exactly three launches: ``amax``, the
leaf kernel, and ONE fused finalize kernel that folds the G leaf sums, takes the log, gathers the
sampled logit (world=1) and applies the padding mask. Fusing moves WHERE the arithmetic runs, not
the expression: the finalize kernel's pairwise ``tl.split`` fold is the identical IEEE add
sequence to ``combine_order(G)`` (left operand = lower leaf indices at every node) and
``libdevice.log`` is bit-identical to ``torch.log`` on fp32 -- and neither fact is trusted: every
probe call re-runs an eager torch ``combine_order`` fold on the same leaf sums and bit-compares,
latching off onto the incumbent on any mismatch. Steady-state admission is a latched per-group hot
route (tuple compare + dict hit, no contract rebuild, no host syncs); any env or layout change
falls off the hot route into the full admission, which still RAISES on structural drift.

BACKWARD. ``grad_x_j = g * (delta_{j,target} - exp(x_j - lse))`` for local columns j: rank-local,
no collective, and only the per-row scalars ``lse = m + log(S)`` are saved -- not the ``[rows, V]``
fp32 softmax the incumbent saves. Logits are accepted in bf16/fp16 as well as fp32; the KERNEL
INPUT dtype stays pinned fp32 (in-kernel widening is REFUSED with a measurement: Triton's
intra-block ``tl.sum`` order follows the load element width, and a bf16 load moves ~30% of leaf
sums by 1 ULP at num_warps=4), so a narrow input is widened transiently inside the forward -- an
exact widen, bit-identical to a pre-widened caller -- and the backward retains only the ORIGINAL
narrow tensor (half the held bytes of a widened fp32 copy, and a free view when the caller passes
a slice of a live activation) plus the per-row scalars. The gradient reaches
only the optimizer and owes no bitwise contract; it is gated by allclose, not bit-equality.

PROBE. The candidate is deliberately a DIFFERENT fp32 function from the incumbent ATen tree, so
the end-to-end probe against ``reference()`` is a tight tolerance gate (live divergence vs ATen is
1-2 ULP of the lse; observed mean |d| ~ 7.7e-7 on real Qwen3.5 logits), not ``torch.equal``. The
bitwise claims -- row-count invariance and TP-invariance of the candidate against ITSELF -- are
gated by ``tests/rowinv_gpu.py`` and ``tests/rowinv_tp_dist.py``. Because this function's bits
differ from the incumbent's, it is composed on BOTH runtimes or neither: it is now the unflagged
default at all four sites, so the "one side only" state it used to be flag-reachable from no
longer exists.

Admission is purely structural (env, dtypes, shapes, world, vocab partition, arch, kernel
availability) and NEVER reads a tensor value: candidate and incumbent issue different collectives,
so ranks must reach the same verdict by construction, sealed by a unanimous ``all_reduce(MIN)``
vote per admitted signature per TP group. The structural facts split in two, exactly the split
``ops/collectives/logprob_gather_wire.py`` draws: facts that select the COLLECTIVE SEQUENCE (env,
device, world, vocabulary partition, G, arch capability, kernel availability) are immutable for
the process and drift there RAISES; per-call payload facts (shapes, strides, logits/src/target
dtypes) are eligibility fields of the signature, each new signature re-voted unanimously. The
dtypes MUST be per-call: every collective here runs on the transiently-widened fp32 tensors
whatever the input dtype, and one trainer process legitimately alternates bf16 (the scoring
forward under ``SKYRL_ISOEXEC_SCORING_LOGITS_BF16``) with fp32 (the training forward, which
``Float16Module`` upcasts) -- same bits either way, per the exact-widen contract above. Judge
engagement only by ``served > 0`` in the census: composed is not executed, and a structural
decline (unsupported layout, wrong arch, missing kernel) silently leaves the incumbent in charge
until ``enforce.rowinv_engagement_boundary`` refuses at the next weight sync.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import torch
import torch.distributed as dist

LEAVES_ENV = "SKYRL_ISOEXEC_PIK_LEAVES"  # the same contract constant pik reads
BANNER = "[ISOEXEC-ROWINV-LOGPROB]"
PROBE_EVERY = 512
# The candidate's lse differs from ATen's by 1-2 ULP (mean |d| ~ 7.7e-7 on real logits); a real
# structural bug (wrong leaf offset, wrong max, wrong tree) lands orders of magnitude above this.
PROBE_MAX_ABS = 1e-4
BLOCK = 4096  # fixed leaf-internal step: part of the numerics contract, never autotuned
NUM_WARPS = 4  # pinned: tl.sum lowering depends on (BLOCK, num_warps); autotuning would move bits

_KERNEL = None
_KERNEL_ERROR = ""
_GROUP_ADMISSION = {}
_S = {
    "calls": 0,
    "served": 0,
    "declined": 0,
    "decline_reason": "",
    "agreed": None,
    "probes": 0,
    "tokens": 0,
    "full_rows": 0,
    "full_gather_bytes_avoided": 0,
    "leaf_wire_floats": 0,
    "hot_hits": 0,
    "reported": 0,
    "bannered": False,
}
#: Probe latch for the FULL-ROW entry (``rowinv_full_logprobs``), separate from the per-group
#: sampled-path latch: the two entries share the admission contract but probe different claims.
#: ``fused`` is the single-kernel fast path's own verdict: ``None`` until first probed, ``True``
#: while every probe bit-matches the eager letter, ``False`` demotes THIS OPTIMIZATION ONLY -- the
#: entry keeps serving the eager rowinv path, so the trainer==engine bits are unaffected by a
#: fused-kernel regression. ``shapes`` drives an extra probe on every new row count.
_FULL_ROW = {
    "agreed": None,
    "served": 0,
    "latched_off": False,
    "latch_reason": "",
    "fused": None,
    "fused_reason": "",
    "shapes": set(),
}
#: Synthesized admission targets for the full-row entry, cached per device: the forward never
#: reads them, so one zeros tensor per (device) -- grown geometrically -- replaces a per-call
#: ``torch.zeros`` allocation+launch on the hot path.
_FULL_TARGET: dict = {}


def stats() -> dict:
    """Return a copy of the engagement census; only ``served > 0`` proves ownership."""
    return dict(_S)


def _reset_stats() -> None:
    _S.update(
        {
            "calls": 0,
            "served": 0,
            "declined": 0,
            "decline_reason": "",
            "agreed": None,
            "probes": 0,
            "tokens": 0,
            "full_rows": 0,
            "full_gather_bytes_avoided": 0,
            "leaf_wire_floats": 0,
            "hot_hits": 0,
            "reported": 0,
            "bannered": False,
        }
    )
    _FULL_ROW.update(
        {
            "agreed": None,
            "served": 0,
            "latched_off": False,
            "latch_reason": "",
            "fused": None,
            "fused_reason": "",
            "shapes": set(),
        }
    )
    _FULL_TARGET.clear()


def _reset_for_test() -> None:
    global _KERNEL, _KERNEL_ERROR
    _KERNEL = None
    _KERNEL_ERROR = ""
    _GROUP_ADMISSION.clear()
    _reset_stats()


def release_group(group) -> None:
    """Release admission state before destroying one TP group."""
    _GROUP_ADMISSION.pop(_group_key(group), None)


def reset_for_teardown() -> None:
    """Release all process-group-bound state before distributed reinitialization."""
    _GROUP_ADMISSION.clear()
    _reset_stats()


def _group_key(group):
    # `None` is the world=1 (engine full-row) view and needs a stable key of its own.
    return id(group) if group is not None else "none"


def _load_kernel():
    global _KERNEL, _KERNEL_ERROR
    if _KERNEL is not None or _KERNEL_ERROR:
        return _KERNEL
    try:
        import triton
        import triton.language as tl
        from triton.language.extra import libdevice

        from ..collectives.pik_bootstrap import ensure_pik

        ensure_pik()
        from pik.plan import (
            combine_order,  # noqa: PLC0415 -- resolvable only after ensure_pik
        )

        @triton.jit
        def leaf_sum_exp(out_ptr, x_ptr, row_max_ptr, row_stride, out_stride, L: tl.constexpr, BLOCK: tl.constexpr):
            # One program per (row, local leaf). The loop bounds, step, and masking depend only on
            # the leaf geometry (L, BLOCK) -- never on the grid -- so the leaf sum is identical for
            # any row count and for whichever rank owns the leaf.
            row = tl.program_id(0)
            leaf = tl.program_id(1)
            base = row * row_stride + leaf * L
            row_max = tl.load(row_max_ptr + row)
            acc = 0.0
            comp = 0.0  # Kahan compensation: costs nothing (memory-bound) and beats ATen's error
            for off in range(0, L, BLOCK):
                idx = off + tl.arange(0, BLOCK)
                value = tl.load(x_ptr + base + idx, mask=idx < L, other=0.0).to(tl.float32)
                block_sum = tl.sum(tl.where(idx < L, libdevice.exp(value - row_max), 0.0))
                y = block_sum - comp
                t = acc + y
                comp = (t - acc) - y
                acc = t
            tl.store(out_ptr + row * out_stride + leaf, acc)

        @triton.jit
        def finalize_rows(
            out_ptr,
            lse_ptr,
            leaf_ptr,
            row_max_ptr,
            x_ptr,
            target_ptr,
            sampled_ptr,
            row_stride,
            full_vocab,
            G: tl.constexpr,
            LOG2G: tl.constexpr,
            LOCAL_GATHER: tl.constexpr,
        ):
            # One program per row: fold the G leaf sums, log, gather/mask, done -- the whole
            # post-gather tail in one launch, which is what keeps the 1-row decode step cheap.
            # The pairwise tl.split fold IS combine_order(G): at every level the left operand
            # carries the lower leaf indices, so the IEEE add sequence is identical (bit-compared
            # against the eager torch combine_order fold on every probe call, never assumed).
            row = tl.program_id(0)
            v = tl.load(leaf_ptr + row * G + tl.arange(0, G))
            for _ in tl.static_range(LOG2G):
                lo, hi = tl.split(tl.reshape(v, (v.shape[0] // 2, 2)))
                v = lo + hi
            s = tl.sum(v)  # single element after the fold: extraction, not arithmetic
            row_max = tl.load(row_max_ptr + row)
            target = tl.load(target_ptr + row)
            valid = (target >= 0) & (target < full_vocab)
            if LOCAL_GATHER:
                # world=1: the full row is local, gather the sampled raw logit in-kernel.
                safe = tl.where(valid, target, 0)
                sampled = tl.load(x_ptr + row * row_stride + safe).to(tl.float32)
            else:
                # world>1: the sampled logit was distributed by the exact masked all_reduce(SUM).
                sampled = tl.load(sampled_ptr + row)
            log_s = libdevice.log(s)  # bit-identical to torch.log on fp32 (probe-checked)
            tl.store(out_ptr + row, tl.where(valid, (sampled - row_max) - log_s, 0.0))
            tl.store(lse_ptr + row, row_max + log_s)

        @triton.jit
        def full_row_logprobs(
            out_ptr,
            x_ptr,
            row_stride,
            out_stride,
            L: tl.constexpr,
            G: tl.constexpr,
            LOG2G: tl.constexpr,
            BLOCK: tl.constexpr,
        ):
            # The engine full-row steady state in ONE launch: one program per row computes the row
            # max, the G leaf Kahan fp32 exp-sums, the combine_order(G) fold, and writes the full
            # ``(x - m) - log(S)`` row -- 3 reads + 1 write of [N, V], the incumbent's traffic.
            # Fusing moves WHERE the arithmetic runs, never the expression:
            #   * the max is exact and order-free, so computing it in-program instead of via
            #     ``torch.amax`` is the same fp32 value bit for bit;
            #   * the leaf loop body is the LETTER of ``leaf_sum_exp``: same fp32 loads, same
            #     ``BLOCK``, same masking, same ``tl.sum`` shape at the same pinned num_warps,
            #     same Kahan y/t/comp sequence, leaves walked in ascending order;
            #   * the fold is the same pairwise ``tl.split`` schedule as ``finalize_rows`` (left
            #     operand = lower leaf indices at every level == ``combine_order(G)``);
            #   * the tail is the two elementwise fp32 subtracts of the eager statement.
            # None of this is trusted: the full [N, V] output is bit-compared against the eager
            # letter (amax + leaf_sum_exp + eager fold + broadcast subtracts) on the first call,
            # on every NEW row count, and every PROBE_EVERY serves; any mismatch demotes to the
            # eager path (bits preserved), never to the incumbent.
            row = tl.program_id(0).to(tl.int64)
            x_row = x_ptr + row * row_stride
            out_row = out_ptr + row * out_stride
            V: tl.constexpr = G * L

            # pass 1: row max. Exact and order-independent, so any schedule gives torch.amax's bits.
            m = -float("inf")
            for off in range(0, V, BLOCK):
                idx = off + tl.arange(0, BLOCK)
                v = tl.load(x_row + idx, mask=idx < V, other=-float("inf")).to(tl.float32)
                m = tl.maximum(m, tl.max(v))

            # pass 2: the G leaf sums, ascending leaf order, each the exact leaf_sum_exp body.
            vec = tl.zeros((G,), dtype=tl.float32)
            for g in tl.static_range(G):
                acc = 0.0
                comp = 0.0  # Kahan compensation, identical to leaf_sum_exp
                base = g * L
                for off in range(0, L, BLOCK):
                    idx = off + tl.arange(0, BLOCK)
                    value = tl.load(x_row + base + idx, mask=idx < L, other=0.0).to(tl.float32)
                    block_sum = tl.sum(tl.where(idx < L, libdevice.exp(value - m), 0.0))
                    y = block_sum - comp
                    t = acc + y
                    comp = (t - acc) - y
                    acc = t
                vec = tl.where(tl.arange(0, G) == g, acc, vec)

            # the combine_order(G) fold: the identical pairwise tl.split schedule finalize_rows
            # runs (elementwise adds -- value-independent of warp count or layout).
            for _ in tl.static_range(LOG2G):
                lo, hi = tl.split(tl.reshape(vec, (vec.shape[0] // 2, 2)))
                vec = lo + hi
            s = tl.sum(vec)  # single element after the fold: extraction, not arithmetic
            log_s = libdevice.log(s)  # bit-identical to torch.log on fp32 (probe-checked)

            # pass 3: the eager tail's two fp32 subtracts, elementwise over the row.
            for off in range(0, V, BLOCK):
                idx = off + tl.arange(0, BLOCK)
                v = tl.load(x_row + idx, mask=idx < V, other=0.0).to(tl.float32)
                tl.store(out_row + idx, (v - m) - log_s, mask=idx < V)

        _KERNEL = (triton, leaf_sum_exp, finalize_rows, combine_order, full_row_logprobs)
    except Exception as error:  # noqa: BLE001 -- unavailable means the incumbent remains owner
        _KERNEL_ERROR = repr(error)
    return _KERNEL


def _banner_once(on: bool) -> None:
    if _S["bannered"]:
        return
    _S["bannered"] = True
    if on:
        print(
            f"{BANNER} sampled logprobs use the fixed-G leaf exp-sum tree "
            f"(G={os.environ.get(LEAVES_ENV, '8')}, BLOCK={BLOCK}, Kahan fp32) at all four sites -- "
            "the composed default, no flag. Unsupported layouts and a failed first-call probe "
            "(fused-fold bitwise self-check + tolerance vs the incumbent) fall back to the "
            "incumbent. Judge engagement only by served>0 in the census.",
            flush=True,
        )
    else:
        print(
            f"{BANNER} SKYRL_ISOEXEC != 1: IsoExec is not composed in this process, so the "
            "incumbent ATen full-vocabulary sum tree remains the sampled-logprob owner.",
            flush=True,
        )


def _report() -> None:
    calls = _S["calls"]
    if calls < 1 or (calls & (calls - 1)) or calls == _S["reported"]:
        return
    _S["reported"] = calls
    print(
        f"{BANNER} CENSUS pid={os.getpid()} calls={calls} served={_S['served']} "
        f"declined={_S['declined']} tokens={_S['tokens']} probes={_S['probes']} "
        f"hot_hits={_S['hot_hits']} "
        f"full_gather_avoided={_S['full_gather_bytes_avoided'] / 1e9:.2f} GB "
        f"leaf_wire_floats={_S['leaf_wire_floats']}"
        + (f" last_decline={_S['decline_reason']}" if _S["decline_reason"] else ""),
        flush=True,
    )


def _decline(reason: str) -> None:
    _S["declined"] += 1
    _S["decline_reason"] = reason
    _report()


def _hot_key(logits, target, vocab_start_index, vocab_end_index, src_dtype):
    return (
        tuple(logits.shape),
        tuple(logits.stride()),
        logits.device,
        logits.dtype,
        tuple(target.shape),
        target.dtype,
        src_dtype,
        vocab_start_index,
        vocab_end_index,
    )


def _admit(
    logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start_index: int,
    vocab_end_index: int,
    group,
    src_dtype: torch.dtype,
) -> tuple[tuple[int, int, int] | None, str]:
    _S["calls"] += 1

    # -- latched hot route: after a signature is unanimously admitted, the steady state is three
    # env reads, one tuple build and one dict hit -- no contract rebuild, no rank queries, no host
    # syncs. Any env change misses the guard and falls into the full admission below, which still
    # RAISES on structural drift; any layout or dtype change is a new key and takes the full path
    # once (dtype is a per-call eligibility field, never a structural fact -- see the contract).
    group_cache = _GROUP_ADMISSION.get(_group_key(group))
    if group_cache is not None and not group_cache["latched_off"]:
        if (
            os.environ.get("SKYRL_ISOEXEC"),
            os.environ.get(LEAVES_ENV, "8"),
        ) == group_cache["env_sig"]:
            hot = group_cache["hot"].get(_hot_key(logits, target, vocab_start_index, vocab_end_index, src_dtype))
            if hot is not None:
                _S["hot_hits"] += 1
                return hot, ""

    on = os.environ.get("SKYRL_ISOEXEC") == "1"
    _banner_once(on)
    if group is not None and not (dist.is_available() and dist.is_initialized()):
        reason = "a process group was passed but torch.distributed is not initialized"
        _decline(reason)
        return None, reason
    if group is None:
        world, rank = 1, 0
    else:
        world = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
    shard_vocab = logits.shape[-1] if logits.ndim else 0
    full_vocab = shard_vocab * world
    try:
        g_leaves = int(os.environ.get(LEAVES_ENV, "8"))
    except ValueError:
        g_leaves = 0
    capability = torch.cuda.get_device_capability(logits.device) if logits.is_cuda else None

    reason = ""
    if not on:
        reason = "SKYRL_ISOEXEC!=1"
    elif not logits.is_cuda or logits.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        reason = f"requires CUDA fp32/bf16/fp16 logits, got device={logits.device} dtype={logits.dtype}"
    elif src_dtype not in (torch.float32, torch.bfloat16, torch.float16):
        reason = f"source dtype {src_dtype} is not fp32/bf16/fp16"
    elif logits.dtype is not torch.float32 and logits.dtype is not src_dtype:
        reason = f"low-precision logits dtype {logits.dtype} contradicts declared src_dtype {src_dtype}"
    elif logits.ndim not in (2, 3) or target.shape != logits.shape[:-1] or not logits.is_contiguous():
        reason = f"unsupported logits/target layout logits={tuple(logits.shape)} target={tuple(target.shape)}"
    elif target.dtype not in (torch.int32, torch.int64):
        reason = f"target dtype {target.dtype} is not int32/int64"
    elif target.numel() == 0:
        reason = "zero rows"
    elif vocab_start_index != rank * shard_vocab or vocab_end_index != vocab_start_index + shard_vocab:
        reason = (
            f"non-contiguous/equal vocab partition start={vocab_start_index} end={vocab_end_index} "
            f"rank={rank} shard={shard_vocab}"
        )
    elif g_leaves < 1 or (g_leaves & (g_leaves - 1)) != 0:
        reason = f"{LEAVES_ENV}={os.environ.get(LEAVES_ENV)!r} is not a positive power of two"
    elif world > g_leaves or g_leaves % world != 0:
        reason = f"world={world} does not map onto contiguous subtrees of the G={g_leaves} leaf tree"
    elif full_vocab % g_leaves != 0:
        reason = f"V={full_vocab} is not divisible by G={g_leaves}: unequal leaves change the contract, declining"
    elif capability != (9, 0):
        reason = f"unvalidated CUDA capability {capability}"
    elif _load_kernel() is None:
        reason = f"Triton/pik leaf kernel unavailable: {_KERNEL_ERROR}"

    # The IMMUTABLE per-process structural contract: only facts that select the COLLECTIVE
    # SEQUENCE, which candidate and incumbent issue differently, so rank-local divergence here is
    # unsafe and RAISES. The payload dtypes are deliberately NOT here: every collective this
    # module issues runs on the transiently-widened fp32 tensors whatever the input dtype (the
    # widen is exact, so the bits match a pre-widened caller), and the same trainer process
    # legitimately alternates bf16 (the scoring forward under SKYRL_ISOEXEC_SCORING_LOGITS_BF16)
    # with fp32 (the training forward, which Float16Module upcasts). As in
    # ops/collectives/logprob_gather_wire.py, dtype is a per-call eligibility field of the
    # signature below -- each new signature is re-voted TP-unanimously, never assumed.
    contract = (
        on,
        os.environ.get("SKYRL_ISOEXEC"),
        logits.device.type,
        logits.device.index,
        world,
        rank,
        shard_vocab,
        vocab_start_index,
        vocab_end_index,
        g_leaves,
        capability,
        _KERNEL is not None,
        _KERNEL_ERROR,
    )
    signature = (
        tuple(logits.shape),
        tuple(logits.stride()),
        logits.dtype,
        src_dtype,
        tuple(target.shape),
        target.dtype,
        reason,
    )
    if group_cache is not None:
        if contract != group_cache["contract"]:
            raise RuntimeError(
                f"{BANNER} STRUCTURAL DRIFT after the TP-unanimous admission vote: "
                f"first={group_cache['contract']!r} now={contract!r}. ENV, device, world, "
                "vocabulary partition, G, capability and kernel availability are immutable for "
                "this process; refusing before entering either collective sequence. (Payload "
                "dtypes and layouts are per-call signature fields and never trip this gate.)"
            )
        if group_cache["latched_off"]:
            cached_reason = reason or "cached TP peer refusal or end-to-end probe failure"
            _decline(cached_reason)
            return None, cached_reason
        cached = group_cache["verdicts"].get(signature)
        if cached is not None and not cached:
            cached_reason = reason or "cached TP peer refusal"
            _decline(cached_reason)
            return None, cached_reason
        if cached:
            return (world, rank, g_leaves), ""
    else:
        # Retain the group object so a destroy/reinitialize cycle cannot reuse its Python id and
        # inherit a prior group's admission or probe verdict.
        group_cache = {
            "group": group,
            "contract": contract,
            "env_sig": (os.environ.get("SKYRL_ISOEXEC"), os.environ.get(LEAVES_ENV, "8")),
            "verdicts": {},
            "hot": {},
            "latched_off": False,
            "agreed": None,
            "served": 0,
        }
        _GROUP_ADMISSION[_group_key(group)] = group_cache

    # Load-bearing: candidate and incumbent issue different collectives after this point, so every
    # NEW signature per TP group must be unanimous -- ranks reach this vote on the same call with
    # the same payload dtype by construction (TP shards of one tensor), so the collective aligns.
    # MIN makes any one rank's refusal everybody's.
    if world > 1:
        vote = torch.tensor(int(not reason), dtype=torch.int32, device=logits.device)
        dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=group)
        admitted = bool(vote.item())
    else:
        admitted = not reason
    group_cache["verdicts"][signature] = admitted
    if not admitted:
        if not reason:
            reason = "a TP peer declined the candidate signature"
        _decline(reason)
        return None, reason
    verdict = (world, rank, g_leaves)
    group_cache["hot"][_hot_key(logits, target, vocab_start_index, vocab_end_index, src_dtype)] = verdict
    return verdict, ""


def _shared_pieces(
    x: torch.Tensor,
    target: torch.Tensor,
    vocab_start_index: int,
    world: int,
    group,
    g_leaves: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Steps 1-3+5 of the contract: (row_max [rows], leaf_sums [rows, G], sampled [rows]|None).

    ``sampled`` is the exactly-distributed raw target logit for world>1; ``None`` at world=1,
    where the finalize kernel gathers it locally instead of paying extra launches.
    """
    leaf_kernel = _KERNEL[1]
    rows, shard_vocab = x.shape
    full_vocab = shard_vocab * world
    leaf_cols = full_vocab // g_leaves
    leaves_local = g_leaves // world

    # 1) global row max: local amax then all_reduce(MAX). Max is exact and order-independent.
    row_max = torch.amax(x, dim=-1)
    if world > 1:
        dist.all_reduce(row_max, op=dist.ReduceOp.MAX, group=group)

    # 2) leaf sums for the leaves this rank owns, one program per (row, leaf), fixed order + Kahan.
    local_sums = torch.empty((rows, leaves_local), device=x.device, dtype=torch.float32)
    leaf_kernel[(rows, leaves_local)](
        local_sums,
        x,
        row_max,
        shard_vocab,
        leaves_local,
        L=leaf_cols,
        BLOCK=BLOCK,
        num_warps=NUM_WARPS,
    )

    # 3) every rank ends up with all G leaf sums, in global leaf order (rank ownership is
    #    contiguous ascending, so rank-order concat IS leaf order). all_gather moves bytes only.
    if world > 1:
        parts = [torch.empty_like(local_sums) for _ in range(world)]
        dist.all_gather(parts, local_sums, group=group)
        leaf_sums = torch.cat(parts, dim=1)
        # 5) the sampled raw logit lives on exactly one rank: mask + all_reduce(SUM) is exact.
        valid = (target >= 0) & (target < full_vocab)
        in_shard = valid & (target >= vocab_start_index) & (target < vocab_start_index + shard_vocab)
        local_target = (target - vocab_start_index).masked_fill(~in_shard, 0).to(torch.int64)
        sampled = torch.gather(x, 1, local_target.unsqueeze(1)).squeeze(1)
        sampled = sampled.masked_fill(~in_shard, 0.0)
        dist.all_reduce(sampled, op=dist.ReduceOp.SUM, group=group)
    else:
        leaf_sums = local_sums
        sampled = None
    return row_max, leaf_sums, sampled


def _finalize_fused(
    x: torch.Tensor,
    target: torch.Tensor,
    row_max: torch.Tensor,
    leaf_sums: torch.Tensor,
    sampled: torch.Tensor | None,
    full_vocab: int,
    g_leaves: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Step 4+6 in ONE launch: combine tree, log, sampled gather/mask, lse. The hot finalize."""
    finalize_kernel = _KERNEL[2]
    rows = x.shape[0]
    out = torch.empty(rows, device=x.device, dtype=torch.float32)
    lse = torch.empty(rows, device=x.device, dtype=torch.float32)
    finalize_kernel[(rows,)](
        out,
        lse,
        leaf_sums,
        row_max,
        x,
        target,
        sampled if sampled is not None else row_max,  # unread when LOCAL_GATHER
        x.shape[1],
        full_vocab,
        G=g_leaves,
        LOG2G=g_leaves.bit_length() - 1,
        LOCAL_GATHER=sampled is None,
        num_warps=1,
    )
    return out, lse


def _finalize_reference(
    x: torch.Tensor,
    target: torch.Tensor,
    row_max: torch.Tensor,
    leaf_sums: torch.Tensor,
    sampled: torch.Tensor | None,
    full_vocab: int,
    g_leaves: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The eager statement of steps 4+6: the explicit ``combine_order(G)`` fold + ``torch.log``.

    This is the contract's letter; ``_finalize_fused`` must reproduce it bit for bit and is
    re-checked against it on every probe call. Probe-path only -- never the steady state.
    """
    combine_order = _KERNEL[3]
    slots = list(leaf_sums.unbind(dim=1))
    for dst, lhs, rhs in combine_order(g_leaves):
        slots[dst] = slots[lhs] + slots[rhs]
    log_s = torch.log(slots[0])
    valid = (target >= 0) & (target < full_vocab)
    if sampled is None:
        safe = target.masked_fill(~valid, 0).to(torch.int64)
        sampled = torch.gather(x, 1, safe.unsqueeze(1)).squeeze(1)
    out = ((sampled - row_max) - log_s).masked_fill(~valid, 0.0)
    return out, row_max + log_s


class _RowInvSampledLogprob(torch.autograd.Function):
    """Forward under the pinned leaf tree; rank-local backward from per-row scalars only."""

    @staticmethod
    def forward(ctx, logits, target, vocab_start_index, world, group, g_leaves):
        x = logits.reshape(-1, logits.shape[-1])
        t = target.reshape(-1)
        full_vocab = logits.shape[-1] * world
        # The KERNEL INPUT dtype is pinned fp32: Triton's intra-block tl.sum order depends on the
        # load element width (measured: a bf16 load that widens in-register moves ~30% of leaf
        # sums by 1 ULP at num_warps=4), so a low-precision input is widened TRANSIENTLY here --
        # the exact widen reproduces the fp32-input bits, the copy dies at call exit, and only
        # the ORIGINAL narrow tensor is retained for backward.
        x_wide = x if x.dtype is torch.float32 else x.to(torch.float32)
        row_max, leaf_sums, sampled = _shared_pieces(x_wide, t, vocab_start_index, world, group, g_leaves)
        out, lse = _finalize_fused(x_wide, t, row_max, leaf_sums, sampled, full_vocab, g_leaves)
        ctx.save_for_backward(logits, target, lse)
        ctx.vocab_start_index = vocab_start_index
        ctx.full_vocab = full_vocab
        return out.reshape(target.shape)

    @staticmethod
    def backward(ctx, grad_out):
        logits, target, lse = ctx.saved_tensors
        shard_vocab = logits.shape[-1]
        x = logits.reshape(-1, shard_vocab)
        t = target.reshape(-1)
        upstream = grad_out.reshape(-1).to(torch.float32)
        valid = (t >= 0) & (t < ctx.full_vocab)
        upstream = torch.where(valid, upstream, torch.zeros_like(upstream))
        # grad_x_j = g * (delta_{j,target} - exp(x_j - lse)); local columns only, no collective.
        # x may be the ORIGINAL bf16/fp16 shard (saved instead of a widened fp32 copy: half the
        # held bytes); `x - lse` promotes through the exact widen, so exp runs in fp32 either way.
        grad = torch.exp(x - lse.unsqueeze(1)).mul_(-upstream.unsqueeze(1))
        in_shard = valid & (t >= ctx.vocab_start_index) & (t < ctx.vocab_start_index + shard_vocab)
        local_target = (t - ctx.vocab_start_index).masked_fill(~in_shard, 0).to(torch.int64)
        grad.scatter_add_(
            1,
            local_target.unsqueeze(1),
            torch.where(in_shard, upstream, torch.zeros_like(upstream)).unsqueeze(1),
        )
        if grad.dtype is not logits.dtype:
            grad = grad.to(logits.dtype)  # backward owes no bitwise contract; optimizer-only
        return grad.reshape(logits.shape), None, None, None, None, None


@torch.no_grad()
def _forward_reference(
    logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start_index: int,
    world: int,
    group,
    g_leaves: int,
) -> torch.Tensor:
    """The whole forward with the EAGER finalize, for the probe's fused-fold bitwise self-check.

    Re-runs the exact collectives of the candidate (probe calls are rank-aligned by construction,
    so the extra sequence is symmetric), then finalizes with the explicit combine_order fold.
    """
    x = logits.reshape(-1, logits.shape[-1])
    t = target.reshape(-1)
    full_vocab = logits.shape[-1] * world
    x_wide = x if x.dtype is torch.float32 else x.to(torch.float32)  # same transient widen
    row_max, leaf_sums, sampled = _shared_pieces(x_wide, t, vocab_start_index, world, group, g_leaves)
    out, _lse = _finalize_reference(x_wide, t, row_max, leaf_sums, sampled, full_vocab, g_leaves)
    return out.reshape(target.shape)


def _probe_final(
    candidate: torch.Tensor,
    incumbent: torch.Tensor,
    eager_fold: torch.Tensor,
    group,
    world: int,
) -> tuple[bool, str]:
    """One unanimous vote over both probe halves.

    (a) BITWISE: the fused finalize kernel against the eager ``combine_order`` fold -- same
    function or the fusion is a bug. (b) TOLERANCE against the incumbent: the candidate is a
    different (better-conditioned) fp32 lse tree than ATen's by design, ~1-2 ULP apart, so
    ``PROBE_MAX_ABS`` catches structural bugs, which land orders of magnitude above that.
    """
    fused_same = torch.equal(candidate.contiguous().view(torch.int32), eager_fold.contiguous().view(torch.int32))
    finite = bool(torch.isfinite(candidate).all().item()) and bool(torch.isfinite(incumbent).all().item())
    max_abs = float((candidate.float() - incumbent.float()).abs().max().item()) if finite else float("inf")
    ok_local = fused_same and finite and max_abs <= PROBE_MAX_ABS
    if world > 1:
        ok = torch.tensor(int(ok_local), dtype=torch.int32, device=candidate.device)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
        ok_all = bool(ok.item())
    else:
        ok_all = ok_local
    _S["probes"] += 1
    if not fused_same:
        return False, "fused finalize kernel is not bit-identical to the eager combine_order fold"
    if not finite:
        return False, "non-finite final sampled output"
    if max_abs > PROBE_MAX_ABS:
        return False, f"|candidate - incumbent| max {max_abs:.3e} exceeds the {PROBE_MAX_ABS:.1e} tolerance gate"
    return ok_all, "peer probe failed"


def rowinv_sampled_logprobs(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
    group,
    src_dtype: torch.dtype,
    reference: Callable[[], torch.Tensor],
) -> torch.Tensor | None:
    """Return row/TP-invariant sampled logprobs, or ``None`` to retain the incumbent owner."""
    admitted, _reason = _admit(logits, target, vocab_start_index, vocab_end_index, group, src_dtype)
    if admitted is None:
        return None
    world, _rank, g_leaves = admitted

    result = _RowInvSampledLogprob.apply(logits, target, vocab_start_index, world, group, g_leaves)

    group_cache = _GROUP_ADMISSION[_group_key(group)]
    should_probe = group_cache["agreed"] is None or (
        group_cache["served"] > 0 and group_cache["served"] % PROBE_EVERY == 0
    )
    if should_probe:
        # The reference callback runs the incumbent end to end; its collectives are aligned
        # because every rank reaches this branch on the same call by construction. The eager-fold
        # rerun re-issues the candidate's own collective sequence, equally aligned.
        incumbent = reference()
        eager_fold = _forward_reference(logits, target, vocab_start_index, world, group, g_leaves)
        ok, reason = _probe_final(result.detach(), incumbent.detach(), eager_fold, group, world)
        if not ok:
            group_cache["agreed"] = False
            _S["agreed"] = False
            group_cache["latched_off"] = True
            _decline(f"end-to-end probe failed: {reason}")
            return incumbent
        group_cache["agreed"] = True
        _S["agreed"] = True

    _S["served"] += 1
    group_cache["served"] += 1
    _S["tokens"] += target.numel()
    if world > 1:
        # The incumbent trainer path materializes the [rows, V] fp32 full-vocabulary gather on
        # every rank; the leaf wire replaces it with rows*G floats.
        _S["full_gather_bytes_avoided"] += target.numel() * logits.shape[-1] * world * 4
    _S["leaf_wire_floats"] += target.numel() * g_leaves
    _report()
    return result


def _probe_full_row(
    full: torch.Tensor,
    incumbent: torch.Tensor,
    x_wide: torch.Tensor,
    row_max: torch.Tensor,
    leaf_sums: torch.Tensor,
    g_leaves: int,
) -> tuple[bool, str]:
    """The full-row entry's two probe halves; world=1, so no vote to align.

    (a) BITWISE against the SAMPLED path: run the fused finalize kernel (the trainer's own
    steady-state tail: tl.split fold + libdevice.log + in-kernel gather) on this call's
    row_max/leaf_sums at one deterministic column per row, and bit-compare against a gather from
    the full row. This is the load-bearing equivalence -- it proves the eager combine_order fold +
    torch.log + broadcast subtracts produce the SAME bits the trainer's sampled path produces for
    any column, so it is checked live, never assumed. (b) TOLERANCE against the incumbent
    (vLLM's log_softmax): a different fp32 lse tree by design, 1-2 ULP apart; the gate catches
    structural bugs, which land orders of magnitude above PROBE_MAX_ABS.
    """
    rows, shard_vocab = x_wide.shape
    # Deterministic per-row probe columns, spread across all G leaves (Knuth hash mod V).
    t = (torch.arange(rows, device=x_wide.device, dtype=torch.int64) * 2654435761) % shard_vocab
    fused, _lse = _finalize_fused(x_wide, t, row_max, leaf_sums, None, shard_vocab, g_leaves)
    gathered = torch.gather(full, 1, t.unsqueeze(1)).squeeze(1)
    _S["probes"] += 1
    if not torch.equal(gathered.contiguous().view(torch.int32), fused.contiguous().view(torch.int32)):
        return False, (
            "full-row gather is not bit-identical to the fused sampled finalize "
            "(eager combine_order fold / torch.log drifted from the tl.split fold / libdevice.log)"
        )
    if not (bool(torch.isfinite(full).all().item()) and bool(torch.isfinite(incumbent).all().item())):
        return False, "non-finite full-row candidate or incumbent output"
    max_abs = float((full - incumbent.float()).abs().max().item())
    if max_abs > PROBE_MAX_ABS:
        return False, f"|candidate - incumbent| max {max_abs:.3e} exceeds the {PROBE_MAX_ABS:.1e} tolerance gate"
    return True, ""


def _full_target(rows: int, device: torch.device) -> torch.Tensor:
    """The synthesized admission target for the full-row entry, cached per device.

    ``_admit`` only reads its shape/dtype (signature vocabulary and hot key); the forward never
    reads a value. One zeros tensor per device, grown geometrically and sliced, replaces a
    per-call ``torch.zeros`` allocation + fill launch on the hot path.
    """
    cached = _FULL_TARGET.get(device)
    if cached is None or cached.shape[0] < rows:
        size = max(rows, 2 * (cached.shape[0] if cached is not None else 512))
        cached = torch.zeros(size, dtype=torch.int64, device=device)
        _FULL_TARGET[device] = cached
    return cached[:rows]


def _full_row_eager(
    x_wide: torch.Tensor, admit_target: torch.Tensor, g_leaves: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The full-row contract's LETTER: amax + leaf kernel + eager combine_order fold + subtracts.

    This is the exact composition that produced the qualified 8-step bitwise-zero run; the fused
    single-kernel fast path must reproduce it bit for bit and is demoted (to THIS path, never to
    the incumbent) if a probe ever catches it not doing so.
    """
    row_max, leaf_sums, _sampled = _shared_pieces(x_wide, admit_target, 0, 1, None, g_leaves)
    combine_order = _KERNEL[3]
    slots = list(leaf_sums.unbind(dim=1))
    for dst, lhs, rhs in combine_order(g_leaves):
        slots[dst] = slots[lhs] + slots[rhs]
    log_s = torch.log(slots[0])
    out = (x_wide - row_max.unsqueeze(1)) - log_s.unsqueeze(1)
    return out, row_max, leaf_sums


def _full_row_fused(x_wide: torch.Tensor, g_leaves: int) -> torch.Tensor:
    """The steady-state fast path: ONE kernel, 3 reads + 1 write of [N, V] -- incumbent traffic."""
    full_kernel = _KERNEL[4]
    rows, shard_vocab = x_wide.shape
    out = torch.empty_like(x_wide)
    full_kernel[(rows,)](
        out,
        x_wide,
        x_wide.stride(0),
        out.stride(0),
        L=shard_vocab // g_leaves,
        G=g_leaves,
        LOG2G=g_leaves.bit_length() - 1,
        BLOCK=BLOCK,
        num_warps=NUM_WARPS,
    )
    return out


def rowinv_full_logprobs(
    logits: torch.Tensor,
    *,
    src_dtype: torch.dtype,
    reference: Callable[[], torch.Tensor],
) -> torch.Tensor | None:
    """Full-row ``[N, V]`` fp32 logprobs under the SAME leaf-tree denominator as the sampled entry.

    WHY A FULL-ROW ENTRY. The engine's V1 sampler site
    (``vllm.v1.sample.sampler.Sampler.compute_logprobs``) is not a gather: its output feeds
    ``gather_logprobs`` (top-k, ranks, sampled AND prompt logprobs), so the hook must return the
    whole fp32 row, not just the sampled value. This entry returns ``(x - m) - log(S)`` over the
    row with ``m`` and ``S`` computed EXACTLY as the sampled entry computes them: the same global
    ``amax`` row max, the same G-leaf Kahan fp32 exp-sum arithmetic (same BLOCK, same num_warps),
    the leaf sums folded ONLY by pik's ``combine_order(G)`` tree, then the two elementwise fp32
    subtracts ``(x - m) - log_s`` -- one rounding each, no reduction -- so the row is row-count
    invariant by construction from top to bottom.

    TWO REALIZATIONS OF THE ONE EXPRESSION. The steady state runs ``_full_row_fused``: a single
    Triton launch per call carrying the whole expression (row max, G Kahan leaf sums in fixed
    ascending block order, the ``tl.split`` fold that IS ``combine_order(G)``, the output
    subtracts) at the incumbent's memory traffic -- 3 reads + 1 write of [N, V] -- instead of the
    eager composition's 6 passes and ~13 launches. The eager composition (``_full_row_eager``,
    the letter: ``torch.amax`` + the leaf kernel + the eager fold + broadcast subtracts) remains
    the arbiter: on the FIRST call, on every NEW row count, and every ``PROBE_EVERY`` serves, the
    fused output is bit-compared against the eager letter over the ENTIRE [N, V] tensor, the
    sampled-path finalize is bit-compared against a gather from the row (the trainer==engine
    linkage), and the incumbent tolerance gate runs. A fused mismatch demotes the fused kernel
    for the process and KEEPS SERVING THE EAGER PATH -- the optimization dies, the bits do not.

    ENGINE world=1 ONLY: the row arrives already gathered (group=None), all G leaves are local.
    Admission reuses the sampled entry's structural contract (same env, layout, ``V % G``,
    capability, kernel checks, same census counters) with a synthesized target column. Declines
    return ``None`` -- the caller keeps the incumbent, bits unchanged. A failed probe latches this
    entry off for the process and returns the incumbent's own tensor for that call. Inference-only
    by design: no autograd (the engine never backpropagates rollout logprobs), enforced with
    ``torch.no_grad`` rather than assumed from the caller.
    """
    if _FULL_ROW["latched_off"]:
        _S["calls"] += 1
        _decline(f"full-row entry latched off: {_FULL_ROW['latch_reason']}")
        return None
    if logits.ndim != 2 or logits.numel() == 0:
        _S["calls"] += 1
        _decline(f"full-row entry requires a non-empty [N, V] matrix, got shape={tuple(logits.shape)}")
        return None
    rows, shard_vocab = logits.shape
    admit_target = _full_target(rows, logits.device)
    admitted, _reason = _admit(logits, admit_target, 0, shard_vocab, None, src_dtype)
    if admitted is None:
        return None
    _world, _rank, g_leaves = admitted  # group=None => world == 1, all leaves local

    with torch.no_grad():
        # The same transient exact widen as the sampled entry: kernel input dtype pinned fp32.
        x_wide = logits if logits.dtype is torch.float32 else logits.to(torch.float32)

        use_fused = _FULL_ROW["fused"] is not False and len(_KERNEL) > 4
        should_probe = (
            _FULL_ROW["agreed"] is None
            or (_FULL_ROW["served"] > 0 and _FULL_ROW["served"] % PROBE_EVERY == 0)
            or (use_fused and rows not in _FULL_ROW["shapes"])
        )

        out = _full_row_fused(x_wide, g_leaves) if use_fused else None
        if should_probe or out is None:
            eager, row_max, leaf_sums = _full_row_eager(x_wide, admit_target, g_leaves)
            if out is not None and not torch.equal(out.view(torch.int32), eager.view(torch.int32)):
                # The fused kernel drifted from the letter: demote the OPTIMIZATION, keep the bits.
                _FULL_ROW["fused"] = False
                _FULL_ROW["fused_reason"] = f"fused full-row kernel != eager letter at rows={rows}"
                print(
                    f"{BANNER} full-row FUSED kernel demoted (not bit-identical to the eager "
                    f"combine_order composition at rows={rows}); the eager path keeps serving, "
                    "bits unchanged.",
                    flush=True,
                )
                out = eager
            elif out is not None:
                _FULL_ROW["fused"] = True
            else:
                out = eager
            _FULL_ROW["shapes"].add(rows)

            if _FULL_ROW["agreed"] is None or (_FULL_ROW["served"] > 0 and _FULL_ROW["served"] % PROBE_EVERY == 0):
                incumbent = reference()
                ok, why = _probe_full_row(out, incumbent, x_wide, row_max, leaf_sums, g_leaves)
                if not ok:
                    _FULL_ROW["agreed"] = False
                    _FULL_ROW["latched_off"] = True
                    _FULL_ROW["latch_reason"] = why
                    _S["agreed"] = False
                    _decline(f"full-row probe failed: {why}")
                    return incumbent
                _FULL_ROW["agreed"] = True
                if _S["agreed"] is None:
                    _S["agreed"] = True

    _S["served"] += 1
    _S["tokens"] += rows
    _S["full_rows"] += rows
    _FULL_ROW["served"] += 1
    _report()
    return out
