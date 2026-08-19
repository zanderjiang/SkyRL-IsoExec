"""Gather-free indexed expert bmm (``SKYRL_ISOEXEC_MOE_INDEXED_BMM``, default OFF).

``_batched_experts_gemm`` normally materializes the per-tile expert weights ``w[tile_expert]``, which costs
a large copy, an ``IndexBackward`` scatter, and -- because ``aten::bmm`` saves both operands -- keeps the
gathered copy alive for the whole step. Here the ``tile_expert`` map is handed to the kernel instead and
the B base pointer becomes ``b_ptr + tile_expert[pid_b] * stride_bb``, so the gathered copy never exists.
``bmm_kernel_indexed`` is vLLM's batch-invariant ``bmm_kernel`` line for line with only that one change,
so every ``tl.load`` sees identical bits and the output is bit-identical; a fail-closed self-check asserts
that at first use and disables the provider permanently on any mismatch or raise.

Gradients carry no bitwise contract -- IsoExec constrains the forward only. ``_IndexedBmm`` saves the STACK
rather than a gathered copy; dgrad reuses the same indexed kernel against a stride-swapped view, and wgrad
is a per-tile bmm followed by a deterministic fp32 segment sum that relies on ``tile_expert`` being
expert-major. Wired into ``_batched_experts_gemm`` for fc1, plain fc2 and the single-leaf pik-fc2 case; the
multi-leaf pik-fc2 tree keeps the gather, and ``tile_expert is None`` paths never had one.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False


_ENV_GATE = "SKYRL_ISOEXEC_MOE_INDEXED_BMM"

# Transient budget for the wgrad's per-tile [chunk, K, N] product (elements). Deliberately far below
# moe_batched_experts._BMM_MAX_ELEMS: this module exists to cut peak memory, and the per-tile product is
# pure transient. Defined locally because moe_batched_experts imports THIS module (no cycle).
_DW_CHUNK_ELEMS = int(os.environ.get("SKYRL_ISOEXEC_MOE_INDEXED_BMM_DW_ELEMS", str(2**28)))

# Fail-closed provider state (mm_cublaslt discipline): first use runs the bitwise self-check; any
# failure or raise disables the provider PERMANENTLY and callers fall through to the gather path.
_STATE = {"checked": False, "ok": False}


def indexed_bmm_enabled() -> bool:
    """Default OFF until live-gated. Read per call so an in-process A/B can flip it."""
    return os.environ.get(_ENV_GATE, "0") == "1"


if HAVE_TRITON:

    @triton.jit
    def bmm_kernel_indexed(
        a_ptr,  # (*, ) pointer to A, (B, M, K)
        b_ptr,  # (*, ) pointer to B, (E, K, N) -- the expert-weight STACK
        idx_ptr,  # (*, ) pointer to the int64 [B] tile -> expert map
        c_ptr,  # (*, ) pointer to C, (B, M, N)
        B,  # int, batch size
        M,  # int, output rows
        N,  # int, output cols
        K,  # int, reduction dim
        stride_ab,
        stride_am,
        stride_ak,
        stride_bb,
        stride_bk,
        stride_bn,
        stride_cb,
        stride_cm,
        stride_cn,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        A_LARGE: tl.constexpr,
        B_LARGE: tl.constexpr,
        C_LARGE: tl.constexpr,
    ):
        """vLLM's batch-invariant ``bmm_kernel``, line for line, except that the B operand's batch index
        is ``idx[pid_b]`` instead of ``pid_b``: (B, M, K) x (E, K, N)[idx] -> (B, M, N).

        Each program computes one (batch_idx, tile_m, tile_n) tile, accumulating along K in a fixed order
        to preserve batch invariance. ONLY the ``b_batch_ptr`` line differs from the original; every load
        mask, the K schedule, the fp32 accumulator and the single rounding store are the same code.
        """
        pid_b = tl.program_id(0)
        pid = tl.program_id(1)

        if pid_b >= B:
            return

        # number of tiles along M / N
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

        if pid_m >= num_pid_m or pid_n >= num_pid_n:
            return

        # offs_m / offs_n: raw global row/col indices for this tile
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        # masks for valid logical rows/cols within (M, N)
        mask_m = offs_m < M  # [BLOCK_SIZE_M]
        mask_n = offs_n < N  # [BLOCK_SIZE_N]

        if A_LARGE or B_LARGE or C_LARGE:
            offs_m = offs_m.to(tl.int64)
            offs_n = offs_n.to(tl.int64)

        offs_m = tl.where(mask_m, offs_m, 0)
        offs_n = tl.where(mask_n, offs_n, 0)

        # hint for triton contiguous memory
        offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # base pointers for current batch, shape-wise:
        #   a_batch_ptr points to A[pid_b, 0, 0]
        #   b_batch_ptr points to B[idx[pid_b], 0, 0]   <- THE ONE CHANGE vs vLLM's bmm_kernel
        #   c_batch_ptr points to C[pid_b, 0, 0]
        a_batch_ptr = a_ptr + pid_b * stride_ab
        b_batch_ptr = b_ptr + tl.load(idx_ptr + pid_b) * stride_bb
        c_batch_ptr = c_ptr + pid_b * stride_cb

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        # number of K-blocks this tile iterates over
        k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
        offs_k_mask = tl.arange(0, BLOCK_SIZE_K)

        for ki in range(k_tiles):
            if A_LARGE or B_LARGE:
                # offs_k: [BLOCK_SIZE_K], global K indices
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
            else:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

            # a_ptrs: [BLOCK_SIZE_M, BLOCK_SIZE_K]
            #   element (i, j) points to A[pid_b, offs_m[i], offs_k[j]]
            a_ptrs = a_batch_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
            # b_ptrs: [BLOCK_SIZE_K, BLOCK_SIZE_N]
            #   element (i, j) points to B[idx[pid_b], offs_k[i], offs_n[j]]
            b_ptrs = b_batch_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

            # valid K lanes for this block
            k_valid = offs_k_mask < (K - ki * BLOCK_SIZE_K)
            # A mask within (M, K): [BLOCK_SIZE_M, BLOCK_SIZE_K]
            a_mask = mask_m[:, None] & k_valid[None, :]
            # B mask within (K, N): [BLOCK_SIZE_K, BLOCK_SIZE_N]
            b_mask = k_valid[:, None] & mask_n[None, :]

            # a: [BLOCK_SIZE_M, BLOCK_SIZE_K] from A[offs_m, offs_k]
            a = tl.load(
                a_ptrs,
                mask=a_mask,
                other=0.0,
            )
            # b: [BLOCK_SIZE_K, BLOCK_SIZE_N] from B[offs_k, offs_n]
            b = tl.load(
                b_ptrs,
                mask=b_mask,
                other=0.0,
            )
            accumulator = tl.dot(a, b, accumulator)

        # c_m / c_n: [BLOCK_SIZE_M] / [BLOCK_SIZE_N], row/col indices for C
        c_m = offs_m
        c_n = offs_n
        if C_LARGE:
            c_m = c_m.to(tl.int64)
            c_n = c_n.to(tl.int64)

        # c_ptrs: [BLOCK_SIZE_M, BLOCK_SIZE_N]
        #   element (i, j) points to C[pid_b, c_m[i], c_n[j]]
        c_ptrs = c_batch_ptr + stride_cm * c_m[:, None] + stride_cn * c_n[None, :]
        # mask out elements that fall outside logical (M, N) range
        c_mask = mask_m[:, None] & mask_n[None, :]
        # cast FP32 accumulator back to original dtype of C
        c = accumulator.to(c_ptr.dtype.element_ty)
        tl.store(c_ptrs, c, mask=c_mask)

    @triton.jit
    def _dw_segsum_kernel(
        DWT,  # [T, K*N] contiguous per-tile wgrad products (flattened per tile)
        CU,  # [E+1] int64 tile offsets per expert WITHIN this dwt chunk
        DW,  # [E, K*N] accumulator in the WEIGHT dtype (read-modify-write; stream-ordered launches)
        KN,
        BLOCK: tl.constexpr,
    ):
        """``DW[e] += sum_t in [CU[e], CU[e+1]) DWT[t]`` -- fixed ascending-tile order, no atomics.

        One program per (expert, element-block). Valid because ``tile_expert`` is expert-major on every
        producing path (asserted device-side by the caller), so an expert's tiles form one contiguous run.
        Accumulates in fp32 REGISTERS within a launch and rounds to the weight dtype once per sub-chunk,
        which is at least the gather path's accuracy without allocating an ``[E, K, N]`` fp32 buffer.
        """
        e = tl.program_id(0).to(tl.int64)
        pid = tl.program_id(1)

        t0 = tl.load(CU + e)
        t1 = tl.load(CU + e + 1)
        # Experts with no tile in THIS sub-chunk exit before touching DW. That keeps total DW traffic at
        # about one pass however many sub-chunks the transient budget forces: the map is expert-major, so
        # each sub-chunk holds a contiguous few experts, and without this guard every launch would
        # read-modify-write the whole [E, K*N] buffer.
        if t1 > t0:
            offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
            m = offs < KN
            acc = tl.zeros((BLOCK,), dtype=tl.float32)
            for t in range(t0, t1):
                acc += tl.load(DWT + t * KN + offs, mask=m, other=0.0).to(tl.float32)
            p = DW + e * KN + offs
            prev = tl.load(p, mask=m, other=0.0).to(tl.float32)
            tl.store(p, (prev + acc).to(DW.dtype.element_ty), mask=m)


def bmm_launch_config(dtype) -> dict:
    """vLLM's batch-invariant bmm block sizes / launch params for ``dtype``.

    Block sizes MUST be the original's for bit-identity; ``_fp16_block_size_n`` is read off the vLLM
    module because ``init_batch_invariance`` sizes it per-platform at install time. Exposed (rather
    than inlined in ``bmm_indexed``) so ``moe_fc2_ingemm``'s leaf-tree kernel takes the identical
    tiling from one place -- identical blocks are a PRECONDITION of its bit-identity claim, so the
    two kernels must not be able to drift apart.
    """
    from vllm.model_executor.layers import batch_invariant as bi

    configs = {
        torch.bfloat16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": getattr(bi, "_fp16_block_size_n", 128),
            "BLOCK_SIZE_K": 64,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float32: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
            "num_stages": 3,
            "num_warps": 8,
        },
    }
    return configs[dtype]


def bmm_indexed(a: torch.Tensor, b: torch.Tensor, idx: torch.Tensor, *, out=None) -> torch.Tensor:
    """(B, M, K) x (E, K, N)[idx] -> (B, M, N); the launcher is ``bmm_batch_invariant`` verbatim
    except that B's batch dim is E (indexed through ``idx``) and the grid batch dim is ``a``'s."""
    assert HAVE_TRITON, "moe_indexed_bmm requires triton"
    if not (a.ndim == 3 and b.ndim == 3):
        raise ValueError(f"bmm_indexed expects 3D tensors, got shapes {a.shape} and {b.shape}")
    if idx.ndim != 1 or idx.shape[0] != a.shape[0]:
        raise ValueError(f"idx must be [{a.shape[0]}], got {tuple(idx.shape)}")
    if a.shape[2] != b.shape[1]:
        raise ValueError(f"Incompatible inner dimensions for matmul: got {a.shape} and {b.shape}.")
    if a.dtype != b.dtype:
        raise ValueError(f"Incompatible dtypes: got {a.dtype} and {b.dtype}.")
    if idx.dtype != torch.long:
        idx = idx.long()
    idx = idx.contiguous()

    B, M, K = a.shape
    _, _, N = b.shape
    dtype = a.dtype

    if out is None:
        c = torch.empty((B, M, N), device=a.device, dtype=dtype)
    else:
        assert out.shape == (B, M, N), "out tensor has incorrect shape"
        assert out.dtype == dtype and out.device == a.device, "out tensor mismatch"
        c = out

    cfg = bmm_launch_config(dtype)
    # grid = (B, num_tiles_per_matrix)
    grid = (
        B,
        triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"]),
    )

    bmm_kernel_indexed[grid](
        a,
        b,
        idx,
        c,
        B,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        **cfg,
    )

    return c


# autograd: save the STACK, not a gathered copy
def indexed_bmm_backward(a, w, idx, g, *, need_da: bool, need_dw: bool):
    """VJP of ``out[t] = a[t] @ w[idx[t]]`` given ``g`` ALREADY in the operand dtype.

    Exposed rather than inlined in ``_IndexedBmm.backward`` because ``moe_fc2_ingemm``'s leaf-tree forward
    has literally this backward: the per-leaf bmm VJPs concatenate over a K-partition into exactly the
    full-K dgrad/wgrad below. Sharing it keeps the wgrad's expert-major segment-sum contract in one place.
    """
    da = dw = None

    if need_da:
        # da[t] = g[t] @ w[idx[t]]^T -- the same indexed kernel against the stride-swapped
        # stack view. Deterministic, gather-free, no bitwise contract to honor.
        da = bmm_indexed(g, w.transpose(1, 2), idx)

    if need_dw:
        # dw[e] = sum over e's tiles of a[t]^T @ g[t]. tile_expert is expert-major on every
        # producing path (trainer repeat_interleave, static-decode searchsorted, fused-stage
        # TileGrid), so each expert's tiles are a contiguous run: a fixed-order segment sum,
        # no atomics, deterministic. Guarded device-side (no host sync).
        if idx.numel() > 1:
            torch._assert_async((idx[1:] >= idx[:-1]).all())
        E, K, N = w.shape
        B = a.shape[0]
        kn = K * N
        # Accumulate straight into the WEIGHT-dtype grad: fp32 registers within a launch, one round
        # per sub-chunk touching that expert, which is at least the gather path's accuracy (it sums
        # EVERY tile in bf16) without a full [E, K, N] fp32 staging buffer.
        dw = torch.zeros(E, K, N, device=w.device, dtype=w.dtype)
        step = max(1, _DW_CHUNK_ELEMS // max(1, kn))
        arange_e = torch.arange(E + 1, device=w.device)
        block = 1024
        for s in range(0, B, step):
            e_ = min(s + step, B)
            # per-tile product through the (overridden, batch-invariant) bmm operator; the
            # [chunk, K, N] product is transient -- freed before the next chunk.
            dwt = torch.ops.aten.bmm(a[s:e_].transpose(1, 2), g[s:e_]).contiguous()
            cu = torch.searchsorted(idx[s:e_], arange_e)
            _dw_segsum_kernel[(E, triton.cdiv(kn, block))](
                dwt.view(e_ - s, kn),
                cu,
                dw.view(E, kn),
                kn,
                BLOCK=block,
            )
            del dwt

    return da, dw


class _IndexedBmm(torch.autograd.Function):
    """``out[t] = a[t] @ w[idx[t]]`` with the stack indexed in-kernel, forward AND backward.

    Saves ``(a, w, idx)`` rather than the gather path's ``(a, w_gathered)``, which keeps the
    ``[n_tiles, K, N]`` copy out of the activation set and the gather's IndexBackward out of the backward.
    """

    @staticmethod
    def forward(ctx, a, w, idx):
        ctx.save_for_backward(a, w, idx)
        return bmm_indexed(a, w, idx)

    @staticmethod
    def backward(ctx, g):
        a, w, idx = ctx.saved_tensors
        da, dw = indexed_bmm_backward(
            a,
            w,
            idx,
            g.contiguous(),
            need_da=ctx.needs_input_grad[0],
            need_dw=ctx.needs_input_grad[1],
        )
        return da, dw, None


def indexed_bmm(a: torch.Tensor, w: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Differentiable ``bmm(a, w[idx])`` that never materializes ``w[idx]``."""
    return _IndexedBmm.apply(a, w, idx)


# Fail-closed install self-check: any mismatch or raise disables the provider permanently.
def _bitcmp(x: torch.Tensor, y: torch.Tensor) -> int:
    """Bit-pattern mismatch count (torch.equal is blind to signed zero)."""
    vx = x.view(torch.int16) if x.dtype in (torch.bfloat16, torch.float16) else x.view(torch.int32)
    vy = y.view(torch.int16) if y.dtype in (torch.bfloat16, torch.float16) else y.view(torch.int32)
    return int((vx != vy).sum().item())


@torch.no_grad()
def _self_check(device) -> bool:
    """Bitwise: ``bmm_indexed(a, w, idx)`` == ``bmm_batch_invariant(a, w[idx])`` -- the EXACT
    production callee on the EXACT production layout (a stack stacked contiguous then transposed,
    so B is strided), across ragged shapes, repeated/absent experts and both dtypes."""
    from vllm.model_executor.layers.batch_invariant import bmm_batch_invariant

    torch.manual_seed(0)
    cases = [
        # (E, B, M, K, N, dtype) -- M=cap-like, K/N deliberately off the block sizes
        (8, 19, 128, 160, 112, torch.bfloat16),  # n_tiles > E, ragged K/N
        (16, 16, 128, 256, 384, torch.bfloat16),  # n_tiles == E
        (8, 3, 64, 96, 40, torch.bfloat16),  # n_tiles < E, small M
        (8, 19, 128, 160, 112, torch.float32),  # fp32 config arm
    ]
    for E, B, M, K, N, dtype in cases:
        stack = torch.randn(E, N, K, device=device, dtype=dtype) * 0.05  # param layout [E, N, K]
        w = stack.transpose(1, 2)  # bmm layout [E, K, N], strides (N*K, 1, K) -- as production
        a = torch.randn(B, M, K, device=device, dtype=dtype) * 0.05
        idx = torch.randint(0, E, (B,), device=device, dtype=torch.long)  # repeats + gaps
        ref = bmm_batch_invariant(a, w[idx])  # the gather path, verbatim
        got = bmm_indexed(a, w, idx)
        bad = _bitcmp(got, ref)
        if bad:
            print(
                f"[ISOEXEC-MOE-IBMM] SELF-CHECK FAIL (E={E},B={B},M={M},K={K},N={N},{dtype}): "
                f"{bad} mismatched bit patterns vs the gather path. Disabling indexed bmm "
                "(gather fallthrough).",
                flush=True,
            )
            return False
    return True


def indexed_bmm_ready(device) -> bool:
    """True iff the flag is on AND the one-time bitwise self-check passed on this stack.

    Fail-closed: a failed or raising self-check disables the provider permanently; callers keep
    the gather path, so the flag can never make a run WRONG, only faster.
    """
    if not indexed_bmm_enabled() or not HAVE_TRITON:
        return False
    if not _STATE["checked"]:
        _STATE["checked"] = True
        try:
            _STATE["ok"] = _self_check(device)
        except Exception as e:  # noqa: BLE001
            print(
                f"[ISOEXEC-MOE-IBMM] self-check raised ({type(e).__name__}: {e}) -- gather fallthrough.",
                flush=True,
            )
            _STATE["ok"] = False
        if _STATE["ok"]:
            print(
                "[ISOEXEC-MOE-IBMM] indexed expert bmm ENABLED (self-check bit-exact vs the gather "
                "path): w[tile_expert] is now indexed in-kernel, never materialized.",
                flush=True,
            )
    return _STATE["ok"]
