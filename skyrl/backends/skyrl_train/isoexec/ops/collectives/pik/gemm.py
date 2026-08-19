"""TP-invariant GEMM kernels.

Layout convention, pinned by the contract and identical across engines: ``x`` is ``[M, K_local]`` and ``w`` is
``[N, K_local]``, both contiguous along K, with an fp32 ``[M, N]`` output because the leaf-combine tree lives
in fp32. Two schedules realize the same balanced binary tree over leaves: ``fused`` walks a CTA's m local
leaves in order, carrying subtree partials in a binary counter of accumulators (peak live is ``log2(m) + 1``,
and m == 1 is a plain GEMM), while ``splitk`` gives one CTA per (output tile, leaf) and applies the combine
tree in a second kernel, costing workspace traffic but winning at small M. Both are fully autotuned, since
block sizes, warps, stages, and grid order do not perturb the K-order.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .autotune_cache import register
from .fastlaunch import launch
from .plan import DEFAULT_PLAN, ReductionPlan


def _configs():
    """The autotune space, per architecture. Tuning over it cannot change results, only speed."""
    from .arch import current

    cfgs = []
    for bm, bn, bk, w, s in current().configs:
        for gm in (8,):
            cfgs.append(
                triton.Config(
                    {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": gm},
                    num_warps=w,
                    num_stages=s,
                )
            )
    return cfgs


def _prune(configs, nargs, **kwargs):
    """Keep only configs whose BLOCK_K tiles a leaf exactly.

    Masked k-tails would load exact zeros anyway, but requiring ``BLOCK_K | LEAF_K`` leaves the k-loop trip
    count as the only thing that varies between configs.
    """
    leaf_k = nargs["LEAF_K"]
    out = [c for c in configs if leaf_k % c.kwargs["BLOCK_K"] == 0]
    return out or [
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=4,
            num_stages=3,
        )
    ]


# Triton keys its config cache on the exact values of these args, so autotune_cache.register installs a cache
# that buckets M by power of two above PIK_AUTOTUNE_M_BUCKET_FLOOR. Keep M first: the bucketing indexes key[0].
_AUTOTUNE_KEY = ["M", "N", "LEAF_K", "NUM_LOCAL_LEAVES"]


@triton.jit
def _tile_ids(pid, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr):
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


# The fused-tree kernel is generated per local-leaf count; see pik/codegen.py.


@triton.autotune(configs=_configs(), key=_AUTOTUNE_KEY, prune_configs_by={"early_config_prune": _prune})
@triton.jit
def _leaf_partial_kernel(
    x_ptr,
    w_ptr,
    p_ptr,
    M,
    N,
    LEAF_K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_pl,
    stride_pm,
    stride_pn,
    NUM_LOCAL_LEAVES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    leaf = tl.program_id(1)
    pid_m, pid_n = _tile_ids(tl.program_id(0), M, N, BLOCK_M, BLOCK_N, GROUP_M)

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    k0 = leaf * LEAF_K
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + (k0 + offs_k)[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(LEAF_K // BLOCK_K):
        a = tl.load(x_ptrs)
        b = tl.load(w_ptrs)
        acc = tl.dot(a, tl.trans(b), acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(
        p_ptr + leaf * stride_pl + om[:, None] * stride_pm + on[None, :] * stride_pn,
        acc,
        mask=(om[:, None] < M) & (on[None, :] < N),
    )


register("_leaf_partial", _leaf_partial_kernel)


# The elementwise tree-reduce epilogue comes from the same emitter as the fused kernel, so the two
# schedules cannot drift apart.


_WORKSPACE: dict = {}
# Buffers replaced by a grow are retired here, never freed: a CUDA graph captured while the old buffer was
# current holds raw pointers into it, and replaying it after a free is silent corruption. A retired buffer is
# still valid scratch for the graph that captured it, and growth is monotonic, so the total stays bounded.
_WORKSPACE_RETIRED: list = []


def _workspace(leaves: int, m: int, n: int, device, dtype=torch.float32) -> torch.Tensor:
    """One growable flat buffer per (device, dtype), viewed to shape.

    Deliberately not keyed on (M, N): a serving engine sees a long tail of batch sizes, and a buffer per M
    would leak until OOM.
    """
    need = leaves * m * n
    key = (device, dtype)
    buf = _WORKSPACE.get(key)
    if buf is None or buf.numel() < need:
        if buf is not None:
            _WORKSPACE_RETIRED.append(buf)  # keep it alive for graphs that captured it
        buf = torch.empty(need, device=device, dtype=dtype)
        _WORKSPACE[key] = buf
    return buf[:need].view(leaves, m, n)


# Admission-gated small-M path (m >= 2). At small M the sequential cuBLASLt loop pays per-leaf launches on
# GEMMs too thin to fill the machine, and the beta=1 pair dependency forbids overlap. `batched_leaves`
# (SKYRL_ISOEXEC_PIK_BATCHED_LEAVES) replaces it with one cuBLASLt strided-batch call over all m leaves --
# batch dim = leaf index, pinned non-split-K algo keyed without M -- plus the usual tree_reduce fold.
#
# Equality with the sequential path is never assumed: the first call at each (impl, leaf_k, N, m, leaf dtype)
# bit-compares the candidate against it on live operands, and a mismatch is a loud, permanent, per-shape
# rejection. Because every admitted path is bit-equal, the M gates are pure perf knobs.

_SMALLM_CFG: dict | None = None  # impl -> (enabled, max_m), read once per process
_SMALLM_STATE: dict = {}  # (impl, device_index, leaf_k, N, m, bf16_leaf) -> True | reason


def _smallm_cfg() -> dict:
    global _SMALLM_CFG
    if _SMALLM_CFG is None:
        import os

        _SMALLM_CFG = {
            "batched_leaves": (
                os.environ.get("SKYRL_ISOEXEC_PIK_BATCHED_LEAVES", "0") == "1",
                int(os.environ.get("SKYRL_ISOEXEC_PIK_BATCHED_LEAVES_MAX_M", "256")),
            ),
        }
    return _SMALLM_CFG


def _run_batched_leaves(x, w, out, leaf_k: int, m_leaves: int, bf16_leaf: bool):
    from . import cublas
    from .codegen import tree_reduce_kernel

    M, N = out.shape
    p = _workspace(m_leaves, M, N, x.device, dtype=torch.bfloat16 if bf16_leaf else torch.float32)
    cublas.leaf_gemm_batched(x, w, p, leaf_k)
    n_elem = M * N
    launch(
        tree_reduce_kernel(m_leaves, in_bf16=bf16_leaf, out_bf16=False),
        (triton.cdiv(n_elem, 1024),),
        p,
        out,
        n_elem,
        p.stride(0),
        BLOCK=1024,
        num_warps=4,
    )
    return out


def _batched_precond(x, w, leaf_k: int, m_leaves: int) -> str | None:
    from . import cublas

    if not cublas.available():
        return "cuBLASLt backend unavailable"
    return None


_SMALLM_IMPLS = {
    "batched_leaves": {
        "precond": _batched_precond,
        "run": _run_batched_leaves,
        "admit_variants": (_run_batched_leaves,),
    },
}
_SMALLM_ORDER = ("batched_leaves",)


def _try_smallm(x, w, out, leaf_k: int, m_leaves: int, bf16_leaf: bool, seq_fn):
    """Run the first enabled and admitted small-M path; ``None`` means the caller runs the sequential loop.

    First use per (impl, shape) is an on-device admission: candidate and sequential path both run on the live
    operands and must agree on every bit. A mismatch or a runtime error is a permanent per-shape rejection.
    """
    cfg = _smallm_cfg()
    M = x.shape[0]
    if M == 0 or m_leaves < 2 or out.dtype != torch.float32:
        return None

    for impl in _SMALLM_ORDER:
        on, max_m = cfg[impl]
        if not on or M > max_m:
            continue
        spec = _SMALLM_IMPLS[impl]
        key = (impl, x.device.index, leaf_k, w.shape[0], m_leaves, bf16_leaf)
        st = _SMALLM_STATE.get(key)
        if st is True:
            try:
                return spec["run"](x, w, out, leaf_k, m_leaves, bf16_leaf)
            except Exception as e:  # noqa: BLE001 -- an admitted path erroring later (e.g. a
                # pinned algo invalid at an unseen M) must demote itself, not kill the run
                _SMALLM_STATE[key] = f"runtime error after admission: {e!r}"
                print(
                    f"[ISOEXEC-PIK] small-M path {impl} DEMOTED for shape (leaf_k={leaf_k}, "
                    f"N={w.shape[0]}, m={m_leaves}, bf16_leaf={bf16_leaf}) after a runtime "
                    f"error at M={M}: {e!r}. Sequential path takes over.",
                    flush=True,
                )
                continue
        if st is not None:
            continue  # rejected earlier, reason recorded

        if torch.cuda.is_current_stream_capturing():
            # Admission compares on-host, which graph capture forbids; the next eager call will admit.
            continue

        def _reject(reason: str, impl=impl, key=key) -> None:
            _SMALLM_STATE[key] = reason
            print(
                f"[ISOEXEC-PIK] small-M path {impl} REJECTED for shape (leaf_k={leaf_k}, "
                f"N={w.shape[0]}, m={m_leaves}, bf16_leaf={bf16_leaf}): {reason}. Sequential "
                f"cuBLASLt path stays in charge for this shape.",
                flush=True,
            )

        reason = spec["precond"](x, w, leaf_k, m_leaves)
        if reason is not None:
            _reject(reason)
            continue

        # Sequential reference first: it and the candidate may share _workspace, so ref must be
        # materialized before the candidate runs. Then every variant the dispatcher may later pick.
        M_, N_ = out.shape
        ref = torch.empty((M_, N_), device=x.device, dtype=torch.float32)
        seq_fn(x, w, ref, leaf_k, m_leaves)
        got = torch.empty((M_, N_), device=x.device, dtype=torch.float32)
        ok = True
        try:
            for variant in spec["admit_variants"]:
                variant(x, w, got, leaf_k, m_leaves, bf16_leaf)
                if not torch.equal(got.view(torch.int32), ref.view(torch.int32)):
                    nbad = int((got.view(torch.int32) != ref.view(torch.int32)).sum())
                    _reject(
                        f"admission bit-compare FAILED ({nbad}/{got.numel()} words differ) -- "
                        f"the fixed-k-order premise does not hold for this shape on this GPU"
                    )
                    ok = False
                    break
        except Exception as e:  # noqa: BLE001
            _reject(f"admission raised: {e!r}")
            ok = False
        if not ok:
            continue

        _SMALLM_STATE[key] = True
        print(
            f"[ISOEXEC-PIK] small-M path {impl} ADMITTED for shape (leaf_k={leaf_k}, "
            f"N={w.shape[0]}, m={m_leaves}, bf16_leaf={bf16_leaf}): bitwise-equal to the "
            f"sequential path on live operands (M={M_}). Active for M <= {max_m}.",
            flush=True,
        )
        out.copy_(got)  # a copy moves no bits
        return out
    return None


def _cublas_tree_bf16(x, w, out, leaf_k: int, m_leaves: int):
    """M-gated front door: an admitted small-M path when one is armed, else the sequential path."""
    if m_leaves >= 2:
        r = _try_smallm(x, w, out, leaf_k, m_leaves, True, _cublas_tree_bf16_seq)
        if r is not None:
            return r
    return _cublas_tree_bf16_seq(x, w, out, leaf_k, m_leaves)


def _cublas_tree_bf16_seq(x, w, out, leaf_k: int, m_leaves: int):
    """bf16 leaves: each leaf partial is rounded to bf16, the tree adds in fp32.

    The leaf is the only rounding point that does not move with TP, so this stays bitwise TP-invariant. The
    beta=1 pair trick is unavailable here, since beta=1 on a bf16 C would add in bf16 storage rather than
    producing the fp32 tree node, so all m bf16 leaves are materialized and the reduce kernel adds in fp32.
    """
    from .cublas import leaf_gemm

    def leaf(t, j):
        return t[:, j * leaf_k : (j + 1) * leaf_k]

    M, N = out.shape
    if m_leaves == 1:
        # a rank's partial IS a leaf -> it may be bf16 all the way, on the wire too
        assert out.dtype == torch.bfloat16, "m=1 bf16-leaf partial must be bf16"
        leaf_gemm(leaf(x, 0), leaf(w, 0), out, beta=0.0)
        return out

    from .codegen import tree_reduce_kernel

    p = _workspace(m_leaves, M, N, x.device, dtype=torch.bfloat16)
    for j in range(m_leaves):
        leaf_gemm(leaf(x, j), leaf(w, j), p[j], beta=0.0)

    n_elem = M * N
    launch(
        tree_reduce_kernel(m_leaves, in_bf16=True, out_bf16=(out.dtype == torch.bfloat16)),
        (triton.cdiv(n_elem, 1024),),
        p,
        out,
        n_elem,
        p.stride(0),
        BLOCK=1024,
        num_warps=4,
    )
    return out


def _cublas_tree(x, w, out, leaf_k: int, m_leaves: int):
    """M-gated front door: an admitted small-M path when one is armed, else the sequential path."""
    if m_leaves >= 2:
        r = _try_smallm(x, w, out, leaf_k, m_leaves, False, _cublas_tree_seq)
        if r is not None:
            return r
    return _cublas_tree_seq(x, w, out, leaf_k, m_leaves)


def _cublas_tree_seq(x, w, out, leaf_k: int, m_leaves: int):
    """The default backend: cuBLASLt per leaf, then the combine tree.

    The contract only requires that a leaf's K-order depend on nothing but the data, which cuBLASLt with a
    pinned non-split-K algo satisfies. Two structural shortcuts: at m == 1 (TP == G) a single cuBLASLt call
    writes straight into ``out``, with no workspace and no tree; at m >= 2, leaf 2j with beta=0 followed by
    leaf 2j+1 with beta=1 accumulate into the same fp32 buffer, which is exactly the tree node, so the bottom
    level costs nothing and the workspace is m/2 rather than m.
    """
    from .cublas import leaf_gemm

    # zero-copy strided views: leaf j is x[:, j*LK:(j+1)*LK]
    def leaf(t, j):
        return t[:, j * leaf_k : (j + 1) * leaf_k]

    if m_leaves == 1:
        leaf_gemm(leaf(x, 0), leaf(w, 0), out, beta=0.0)
        return out

    pairs = m_leaves // 2
    if pairs == 1:  # m == 2: the pair result IS the answer
        leaf_gemm(leaf(x, 0), leaf(w, 0), out, beta=0.0)
        leaf_gemm(leaf(x, 1), leaf(w, 1), out, beta=1.0)
        return out

    from .codegen import tree_reduce_kernel

    M, N = out.shape
    p = _workspace(pairs, M, N, x.device)
    for j in range(pairs):
        leaf_gemm(leaf(x, 2 * j), leaf(w, 2 * j), p[j], beta=0.0)
        leaf_gemm(leaf(x, 2 * j + 1), leaf(w, 2 * j + 1), p[j], beta=1.0)

    n_elem = M * N
    launch(
        tree_reduce_kernel(pairs), (triton.cdiv(n_elem, 1024),), p, out, n_elem, p.stride(0), BLOCK=1024, num_warps=4
    )
    return out


def _dispatch(x, w, out, leaf_k: int, m_leaves: int, sched: str, leaf_dtype=torch.float32):
    """The one place a schedule is turned into kernel launches.

    Every branch here computes the SAME arithmetic tree and returns the SAME bits;
    they differ only in where the intermediate subtree partials live (fp32 workspace,
    registers/TMEM, or a cuBLASLt beta=1 accumulator).
    """
    M, _ = x.shape
    N, _ = w.shape

    if sched == "cublas":
        if leaf_dtype == torch.bfloat16:
            return _cublas_tree_bf16(x, w, out, leaf_k, m_leaves)
        return _cublas_tree(x, w, out, leaf_k, m_leaves)

    if sched == "fused":
        from .codegen import fused_tree_kernel

        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)  # noqa: E731
        launch(
            fused_tree_kernel(m_leaves, leaf_dtype == torch.bfloat16),
            grid,
            x,
            w,
            out,
            M,
            N,
            leaf_k,
            x.stride(0),
            x.stride(1),
            w.stride(0),
            w.stride(1),
            out.stride(0),
            out.stride(1),
            NUM_LOCAL_LEAVES=m_leaves,
        )
        return out

    if sched.startswith("split"):
        # "splitk" splits all the way (s = m); "split<s>" splits s ways, each CTA carrying m/s leaves.
        from .codegen import split_tree_kernel, tree_reduce_kernel

        s = m_leaves if sched == "splitk" else int(sched[5:])
        assert m_leaves % s == 0 and (s & (s - 1)) == 0, f"bad split {s} for m={m_leaves}"
        g = m_leaves // s

        bl = leaf_dtype == torch.bfloat16
        store_bf16 = bl and g == 1  # a g==1 partial is a LEAF; leaves may be bf16
        p = _workspace(s, M, N, x.device, dtype=torch.bfloat16 if store_bf16 else torch.float32)
        grid = lambda META: (  # noqa: E731
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
            s,
        )
        launch(
            split_tree_kernel(g, bl, store_bf16),
            grid,
            x,
            w,
            p,
            M,
            N,
            leaf_k,
            x.stride(0),
            x.stride(1),
            w.stride(0),
            w.stride(1),
            p.stride(0),
            p.stride(1),
            p.stride(2),
            NUM_LOCAL_LEAVES=m_leaves,
        )
        if s == 1:
            out.copy_(p[0])
            return out
        n_elem = M * N
        launch(
            tree_reduce_kernel(s, in_bf16=store_bf16, out_bf16=(out.dtype == torch.bfloat16)),
            (triton.cdiv(n_elem, 1024),),
            p,
            out,
            n_elem,
            p.stride(0),
            BLOCK=1024,
            num_warps=4,
        )
        return out

    raise ValueError(f"unknown schedule {sched!r}")


_SCHED_CACHE: dict = {}
_BF16_LEAVES = [False]  # set by ti_gemm from the plan; see _pick_schedule


def _sched_key(M, N, leaf_k, m_leaves):
    # Bucket M by power of two so a serving engine does not re-time per batch size. Every schedule produces
    # the same bits, so a mis-bucketed choice costs time and never correctness.
    mb = 1 << max(0, (M - 1).bit_length())
    return (mb, N, leaf_k, m_leaves)


def _sched_candidates(m_leaves: int) -> list[str]:
    """Every schedule the picker may choose from, in bench order.

    Every power-of-two split factor: s=1 is the fused tree, s=m is full split-K, and the
    interesting ones are usually in between.
    """
    from . import cublas

    cands = ["fused"] + [f"split{s}" for s in (1, 2, 4, 8, 16, 32) if s <= m_leaves and m_leaves % s == 0]
    if cublas.available():
        cands.insert(0, "cublas")
    return cands


def _bench_schedules(x, w, out, leaf_k: int, m_leaves: int, cands: list[str]) -> str:
    """Wall-clock pick among candidate schedules."""
    best, best_t = None, float("inf")
    for s in cands:
        try:
            t = triton.testing.do_bench(
                lambda s=s: _dispatch(
                    x, w, out, leaf_k, m_leaves, s, out.dtype if out.dtype == torch.bfloat16 else torch.float32
                ),
                warmup=5,
                rep=20,
                quantiles=(0.5,),
            )
        except Exception:  # noqa: BLE001 -- a schedule may not support this shape
            continue
        if t < best_t:
            best, best_t = s, t
    return best or "splitk"


def _pick_schedule(x, w, out, leaf_k: int, m_leaves: int) -> str:
    """Pick a schedule for this shape bucket by timing the candidates once, then cache the winner.

    All schedules realize the same combine tree by construction; whether they are also bitwise identical
    across backends (cuBLASLt leaf vs Triton ``tl.dot`` chain) is a property of the (arch, toolchain) pair.
    Because the pick is per process, the trainer and the engine may land on different schedules, which is safe
    only while that cross-backend equality holds.

    Which one wins is shape-dependent: m == 1 is a single GEMM; large leaf_k with big M favours cuBLAS; small
    leaf_k with big M favours the fused Triton kernel, which keeps the tree in TMEM and touches no workspace;
    small M favours split-K, whose leaf parallelism fills the machine.
    """

    key = _sched_key(x.shape[0], w.shape[0], leaf_k, m_leaves)
    hit = _SCHED_CACHE.get(key)
    if hit is not None:
        return hit

    cands = _sched_candidates(m_leaves)
    if m_leaves == 1 and "cublas" in cands:
        # Degenerates to one plain GEMM; nothing to measure.
        best = "cublas"
    else:
        best = _bench_schedules(x, w, out, leaf_k, m_leaves, cands)

    _SCHED_CACHE[key] = best
    return best


def ti_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    plan: ReductionPlan = DEFAULT_PLAN,
    tp_size: int = 1,
    tp_rank: int = 0,
    k_full: int | None = None,
    schedule: str | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Row-parallel TP-invariant GEMM returning this rank's subtree partial.

    ``x: [M, K_local]``, ``w: [N, K_local]`` -> ``[M, N]``. At tp_size == 1 this is already the final result;
    above it, feed the output to ``pik.allreduce.tree_all_reduce``, which applies the remaining tree levels.
    """
    assert x.ndim == 2 and w.ndim == 2, "expected 2-D x [M,K] and w [N,K]"
    M, k_local = x.shape
    N, k_local_w = w.shape
    assert k_local == k_local_w, f"K mismatch: x has {k_local}, w has {k_local_w}"

    k_full = k_full if k_full is not None else k_local * tp_size
    plan.validate(k_full, tp_size)
    assert (
        k_local == k_full // tp_size
    ), f"x/w have K_local={k_local} but plan says {k_full}//{tp_size}={k_full // tp_size}"

    leaf_k = plan.leaf_k(k_full)
    m_leaves = plan.leaves_per_rank(tp_size)

    # With bf16 leaves, a rank whose partial is itself a leaf (m == 1) stays bf16 all the way, wire included.
    # With m > 1 the partial is an internal tree node, and internal nodes are always fp32: rounding one would
    # make the result depend on where the rank boundary fell.
    if out is None:
        out_dtype = torch.bfloat16 if (plan.bf16_leaves and m_leaves == 1) else torch.float32
        out = torch.empty((M, N), device=x.device, dtype=out_dtype)

    _BF16_LEAVES[0] = plan.bf16_leaves
    sched = schedule or _pick_schedule(x, w, out, leaf_k, m_leaves)
    return _dispatch(x, w, out, leaf_k, m_leaves, sched, plan.leaf_dtype)


def ti_gemm_column_parallel(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
):
    """Column-parallel GEMM (qkv / gate / up / lm_head).

    K is never sharded here, so there is no tree and any fixed non-split-K order is already TP-invariant. The
    only requirement is that the kernel never picks split-K, which ``torch.matmul`` may do shape-dependently;
    this is one cuBLASLt call with the pinned non-split-K algo, writing the output dtype directly.
    """
    from . import cublas

    if cublas.available():
        if out is None:
            out = torch.empty((x.shape[0], w.shape[0]), device=x.device, dtype=out_dtype)
        return cublas.leaf_gemm(x, w, out, beta=0.0)

    # no cuBLASLt: fall back to Triton (fp32 out; caller casts)
    return ti_gemm(x, w, plan=ReductionPlan(num_leaves=1), tp_size=1, schedule="fused", out=out)
