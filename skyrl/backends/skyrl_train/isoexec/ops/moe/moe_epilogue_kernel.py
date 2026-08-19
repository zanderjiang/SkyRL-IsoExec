"""Fuse the MoE glu+probs epilogue into the fc1 GEMM, so ``inter`` never reaches HBM.

The fc1 GEMM is re-tiled over the OUTPUT half-width ``f``, so one program owns the gate tile at columns
``c`` and the up tile at ``c + f`` in two accumulators. That is the entire reason this fuses: the native
tiling can put ``gate[:, j]`` and ``up[:, j]`` in different CTAs, where no epilogue can join them.

The epilogue rounds exactly where torch rounds -- silu rounds to the tensor dtype, the ``* up`` multiply
rounds again, and the probs multiply keeps its ``.to(dtype)`` round-trip before fc2 -- so the output is
bitwise-equal to the four-launch torch chain and to the trainer's bmm path. Moving the probs multiply after
fc2 would be mathematically identical but bitwise different.

Only ``silu(gate) * up`` with no clamp and a zero linear offset is expressible; :func:`epilogue_supported`
is the host-side gate and the callers raise rather than compute something subtly different. The block map,
the ``num_tokens_post_padded`` early return and the grid bound are the existing host-known ones, so nothing
is read back to the host and CUDA-graph capture is unaffected. Gated on
``SKYRL_ISOEXEC_MOE_FUSED_EPILOGUE``, default OFF; :func:`apply_glu_probs_epilogue` and
:func:`apply_glu_probs_epilogue_silu_and_mul` are standalone fallbacks for a caller handed ``inter``
rather than ``x``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# BLOCK_SIZE_N for the fused-epilogue fc1. Free to retune: it only splits OUTPUT columns across programs
# and never enters a reduction, so 32 and 64 are torch.equal. BLOCK_SIZE_M/K are NOT free (M feeds the
# host block map, K sets the accumulation order). All three are fixed constants, never keyed on the token
# count, so batch-invariance is untouched. 32 fills the machine at decode and small M; 64 wins once the
# grid already saturates, which is what the host branch in invoke_fused_moe_fc1_glu_kernel selects.
_FC1_BLOCK_N = 32
_FC1_NUM_WARPS = 4
_FC1_NUM_STAGES = 3

_EPI_BLOCK_T = 32
_EPI_BLOCK_F = 128
_EPI_NUM_WARPS = 4


def epilogue_supported(cfg) -> bool:
    """Host-side gate: can the fused epilogue express this model's glu?

    Only ``silu(gate) * (up + 0)`` with no clamp. A clamp value, a non-zero linear offset or a different
    activation keeps the torch chain -- falling back is the point, since computing something subtly
    different would move a bit.
    """
    return (
        getattr(cfg, "activation_func", None) is torch.nn.functional.silu
        and getattr(cfg, "activation_func_clamp_value", None) is None
        and getattr(cfg, "glu_linear_offset", 0.0) == 0.0
    )


@triton.jit
def fused_moe_fc1_glu_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    probs_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    F,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    compute_type: tl.constexpr,
    RAW_INTER: tl.constexpr = False,
):
    """fc1 for expert-grouped rows, tiled over the ``f`` OUTPUT columns, with the glu+probs epilogue
    applied to the accumulators.

    ``B`` is the Megatron fc1 parameter ``[E, 2f, K]``: rows ``[0, f)`` are the gate projection and
    ``[f, 2f)`` the up projection -- the same split ``torch.chunk(inter, 2, dim=-1)`` makes. One program
    owns column tile ``c`` of BOTH halves, so it can join them without a round-trip. ``C`` is ``[T, f]``
    in the compute dtype: the ``h`` the fc2 GEMM consumes, router weight already folded in (so fc2 keeps
    ``mul_routed_weight=False``).

    ``RAW_INTER`` (TESTS ONLY) skips the epilogue and stores the two accumulators to a ``[T, 2f]`` ``C``,
    i.e. exactly what the native fc1 writes, which is what lets a test assert the GEMM's own arithmetic is
    unchanged. It is a constexpr, so the production specialization is byte-identical to a kernel without
    the branch.
    """
    # pid -> (pid_m, pid_n): verbatim from vLLM's fused_moe_kernel (grouped for L2 reuse). No
    # numerical effect -- each program computes an independent output tile.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(F, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token = tl.load(sorted_token_ids_ptr + pid_m * BLOCK_SIZE_M + offs).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    # our block map (moe_fused_experts._block_map) never emits -1 experts, so no zero-fill branch.
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    # the gate columns of this tile, and the up columns f rows further down the same parameter. The
    # `% F` is the native kernel's out-of-range guard for the LOAD only; the store is masked by
    # offs_cn < F, so a wrapped column can never reach C.
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % F
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_token[:, None] * stride_am + offs_k[None, :] * stride_ak)
    bg_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    bu_ptrs = bg_ptrs + F * stride_bn

    acc_g = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for kb in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        kmask = offs_k < K - kb * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=token_mask[:, None] & kmask[None, :], other=0.0)
        bg = tl.load(bg_ptrs, mask=kmask[:, None], other=0.0)
        bu = tl.load(bu_ptrs, mask=kmask[:, None], other=0.0)
        # the A tile is loaded ONCE and fed to both dots -- the second half of the win, on top of
        # never storing `inter`: the native tiling reloads it for the up columns' program.
        acc_g += tl.dot(a, bg)
        acc_u += tl.dot(a, bu)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        bg_ptrs += BLOCK_SIZE_K * stride_bk
        bu_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if RAW_INTER:  # tests only: reproduce the native kernel's [T, 2f] store, no epilogue
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        cmask = token_mask[:, None] & (offs_cn[None, :] < F)
        tl.store(c_ptrs, acc_g.to(compute_type), mask=cmask)
        tl.store(c_ptrs + F * stride_cn, acc_u.to(compute_type), mask=cmask)
        return

    # ---- the epilogue, rounding exactly where torch rounds
    # 1. the fc1 C-store round the native kernel would have applied on its way to HBM.
    gf = acc_g.to(compute_type).to(tl.float32)
    uf = acc_u.to(compute_type).to(tl.float32)
    # 2. F.silu: fp32 `x / (1 + expf(-x))`, rounded to the tensor dtype.
    s = (gf / (1.0 + tl.exp(-gf))).to(compute_type).to(tl.float32)
    # 3. `* up`: bf16 x bf16 -> fp32 multiply -> round.
    h = (s * uf).to(compute_type).to(tl.float32)
    # 4. `* probs` then `.to(dtype)`: probs promoted to fp32 (exact from either dtype it arrives in),
    #    multiply, one round -- the round-trip that keeps this bitwise the bmm path.
    p = tl.load(probs_ptr + offs_token, mask=token_mask, other=0.0).to(tl.float32)
    out = (h * p[:, None]).to(compute_type)

    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, out, mask=token_mask[:, None] & (offs_cn[None, :] < F))


def invoke_fused_moe_fc1_glu_kernel(
    A: torch.Tensor,  # [T, K] expert-grouped rows (bf16)
    B: torch.Tensor,  # [E, 2f, K] stacked fc1 weights (bf16), Megatron layout
    C: torch.Tensor,  # [T, f] bf16 out: h, router weight already folded in
    probs: torch.Tensor,  # [T] router weight per permuted row (fp32 or bf16)
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    config: dict,
    compute_type: tl.dtype,
    raw_inter: bool = False,
) -> None:
    """Launch wrapper, mirroring vLLM's ``invoke_fused_moe_triton_kernel`` for our top_k=1 usage
    (including its small-batch EM shrink). Strides are passed through, so A/B may be strided views.

    Bitwise-equivalent to running the native fc1 into ``inter`` and then the torch chain::

        gate, up = inter.reshape(T, 2f).chunk(2, dim=-1)
        h = torch.nn.functional.silu(gate) * up
        C = (h * probs.unsqueeze(-1)).to(h.dtype)
    """
    two_f = B.size(1)
    if two_f % 2 != 0:
        raise RuntimeError(f"[isoexec-moe] fc1-glu kernel: fc1 output {two_f} is not a gate/up pair")
    F = two_f // 2
    want = (A.size(0), two_f if raw_inter else F)
    if C.shape != want:
        raise RuntimeError(f"[isoexec-moe] fc1-glu kernel: C must be {want}, got {tuple(C.shape)}")
    if probs.numel() != A.size(0) or probs.stride(0) != 1:
        raise RuntimeError("[isoexec-moe] fc1-glu kernel: probs must be a contiguous [T] vector")
    assert sorted_token_ids.stride(0) == 1

    K = B.size(2)
    BLOCK_M = config["BLOCK_SIZE_M"]
    EM = sorted_token_ids.size(0)
    if A.size(0) < BLOCK_M:
        EM = min(EM, A.size(0) * BLOCK_M)

    # Host branch on the trace-time-known row-block count. Bit-neutral (BLOCK_N splits OUTPUT columns
    # only, never a reduction) and graph-safe; 32 fills the machine at decode/small-M, 64 wins once the
    # grid already saturates.
    BLOCK_N = _FC1_BLOCK_N if (EM // BLOCK_M) < 512 else 64
    grid = (triton.cdiv(EM, BLOCK_M) * triton.cdiv(F, BLOCK_N),)
    fused_moe_fc1_glu_kernel[grid](
        A,
        B,
        C,
        probs,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        F,
        K,
        EM,
        A.size(0),  # num_valid_tokens: top_k = 1, each row is one (token, expert) pair
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        compute_type=compute_type,
        RAW_INTER=raw_inter,
        num_warps=_FC1_NUM_WARPS,
        num_stages=_FC1_NUM_STAGES,
    )


@triton.jit
def glu_probs_epilogue_kernel(
    inter_ptr,
    probs_ptr,
    out_ptr,
    T,
    F,
    stride_it,
    stride_if,
    stride_ot,
    stride_of,
    BLOCK_T: tl.constexpr,
    BLOCK_F: tl.constexpr,
    compute_type: tl.constexpr,
):
    """``chunk -> silu -> * up -> * probs -> cast`` in one pass over ``inter [T, 2f]``.

    Same rounding points as the fused-fc1 epilogue above and therefore bitwise-identical to it and to
    the torch chain -- the only difference is that ``gate``/``up`` are loaded from HBM instead of read
    out of the accumulators.
    """
    pid_t = tl.program_id(0)
    pid_f = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_f = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)
    mask = (offs_t[:, None] < T) & (offs_f[None, :] < F)

    base = inter_ptr + offs_t[:, None].to(tl.int64) * stride_it + offs_f[None, :].to(tl.int64) * stride_if
    gf = tl.load(base, mask=mask, other=0.0).to(tl.float32)
    uf = tl.load(base + F * stride_if, mask=mask, other=0.0).to(tl.float32)

    s = (gf / (1.0 + tl.exp(-gf))).to(compute_type).to(tl.float32)
    h = (s * uf).to(compute_type).to(tl.float32)
    p = tl.load(probs_ptr + offs_t, mask=offs_t < T, other=0.0).to(tl.float32)
    out = (h * p[:, None]).to(compute_type)

    o_ptrs = out_ptr + offs_t[:, None].to(tl.int64) * stride_ot + offs_f[None, :].to(tl.int64) * stride_of
    tl.store(o_ptrs, out, mask=mask)


def apply_glu_probs_epilogue(inter: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """One-pass fused replacement for the 4-launch torch chain, given a materialized ``inter``.

    ``inter [T, 2f]`` (or ``[T, 1, 2f]``, the fc1 output's shape), ``probs [T]``. Returns contiguous
    ``h [T, f]`` in ``inter``'s dtype. Bitwise-equal to::

        gate, up = inter.reshape(T, 2f).chunk(2, dim=-1)
        h = torch.nn.functional.silu(gate) * up
        (h * probs.unsqueeze(-1)).to(h.dtype)
    """
    inter = inter.reshape(inter.shape[0], -1)
    T, two_f = inter.shape
    if two_f % 2 != 0:
        raise RuntimeError(f"[isoexec-moe] glu epilogue: fc1 output {two_f} is not a gate/up pair")
    if probs.numel() != T or probs.stride(0) != 1:
        raise RuntimeError("[isoexec-moe] glu epilogue: probs must be a contiguous [T] vector")
    F = two_f // 2
    out = inter.new_empty(T, F)
    if T == 0:
        return out
    compute_type = tl.bfloat16 if inter.dtype is torch.bfloat16 else tl.float16
    grid = (triton.cdiv(T, _EPI_BLOCK_T), triton.cdiv(F, _EPI_BLOCK_F))
    glu_probs_epilogue_kernel[grid](
        inter,
        probs,
        out,
        T,
        F,
        inter.stride(0),
        inter.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_T=_EPI_BLOCK_T,
        BLOCK_F=_EPI_BLOCK_F,
        compute_type=compute_type,
        num_warps=_EPI_NUM_WARPS,
    )
    return out


def apply_glu_probs_epilogue_silu_and_mul(inter: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """The cheap intermediate: vLLM's ``silu_and_mul`` for the glu, torch for the probs round-trip.

    ``torch.ops._C.silu_and_mul`` is bitwise-equal to ``F.silu(gate) * up`` at bf16 -- it reproduces the
    same double rounding. Strictly worse than the two fused paths above, kept as the zero-risk step.
    """
    inter = inter.reshape(inter.shape[0], -1)
    h = inter.new_empty(inter.shape[0], inter.shape[1] // 2)
    torch.ops._C.silu_and_mul(h, inter.contiguous())
    return (h * probs.unsqueeze(-1)).to(h.dtype)
