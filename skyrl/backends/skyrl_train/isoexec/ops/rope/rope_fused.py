"""Fused attention RoPE for the engine: one Triton kernel per call, plus a cos/sin hoist.

Stock megatron recomputes ``cos``/``sin`` from a ``freqs`` tensor that is identical for every layer
and both projections, then spends six more launches on the rotate-half and multiply-add. This
computes cos/sin once per forward and fuses the rest into a single kernel.

THE REFERENCE IS MEGATRON'S STOCK ``_apply_rotary_pos_emb_bshd``, whose multiply-add is a bf16
THREE-round chain: bf16(t*cos_), bf16(rot*sin_), bf16(sum). Staying in fp32 across the add is more
accurate and a different function, which would force the trainer to move with it. The repo also
carries an fp32 RoPE patch with exactly that one-round chain, and this module refuses to install
over it rather than silently matching the wrong reference.

ENGINE-ONLY, by three independent guards: the installer is called only from the vLLM model wrapper,
the fused path fires only for a ``freqs`` tensor marked by ``_PositionIndexedRoPE``, and it refuses
whenever grad is enabled -- a raw Triton call carries no ``grad_fn`` and would sever a backward.

Flag: ``SKYRL_ISOEXEC_FUSED_ROPE=1``, default off.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

FLAG = "SKYRL_ISOEXEC_FUSED_ROPE"

# Pinned launch geometry. Not a bitwise contract -- this kernel has no reduction, so the tile cannot
# reorder any addition -- but pinned regardless, since nothing in this tree autotunes.
_MB = 4  # rows (= tokens x heads) per program
_NUM_WARPS = 4

# Populated by `_cos_sin_for`, read by `hoist_report`. A hoist that never fires is bitwise-perfect
# and worth nothing, so whether it fires has to be a counter rather than an assumption.
HOIST_STATS = {"calls": 0, "computes": 0}

_LOGGED = False
_FALLBACK_LOGGED = False
_HOIST_LOGGED = False
_ORIG_BSHD = None
_INSTALLED = False

# Both attributes live on the `freqs` tensor -- the engine mark and the hoisted cos/sin cache -- so
# they die with it and there is no invalidation boundary to miss.
_MARK_ATTR = "_ix_engine_rope"
_CACHE_ATTR = "_ix_rope_cos_sin"


def fused_rope_enabled() -> bool:
    """``SKYRL_ISOEXEC_FUSED_ROPE=1``, default off.

    Logged once per process at first read, so the value the engine actor saw appears in its own log
    rather than only the value the launcher exported.
    """
    global _LOGGED
    on = os.environ.get(FLAG, "0") == "1"
    if not _LOGGED:
        _LOGGED = True
        print(
            f"[ISOEXEC-ROPE] {FLAG}={os.environ.get(FLAG, '<unset>')} -> fused RoPE {'ON' if on else 'OFF'}",
            flush=True,
        )
    return on


@triton.jit
def _rope_kernel(
    t_ptr,
    cos_ptr,
    sin_ptr,
    o_ptr,
    n_rows,
    HB,
    H,
    st_s,
    st_b,
    st_h,
    D: tl.constexpr,
    R: tl.constexpr,
    BD: tl.constexpr,
    MB: tl.constexpr,
):
    """One fused pass over a ``[S, B, H, D]`` tensor: rotate the first ``R`` lanes, copy the rest.

    The input is addressed by explicit ``[S, B, H]`` strides and only its last dim must be unit
    stride: query/key reach RoPE from a split of the fused qkv projection whose contiguity depends
    on qk_layernorm and the rank's head counts, so a full-contiguity requirement would silently fall
    back to eager on some shard geometries. The output is contiguous, as ``torch.cat`` returns.
    """
    # int64 row index: the offset is row index times stride, and an int32 index silently wraps past
    # 2^31 rather than faulting.
    rows = tl.program_id(0).to(tl.int64) * MB + tl.arange(0, MB)[:, None]
    d = tl.arange(0, BD)[None, :]
    rmask = rows < n_rows
    mask = rmask & (d < D)
    rot_mask = rmask & (d < R)

    # row -> (s, b, h). HB = B*H.
    s = rows // HB
    rem = rows % HB
    base = s * st_s + (rem // H) * st_b + (rem % H) * st_h
    x = tl.load(t_ptr + base + d, mask=mask, other=0)

    # `_rotate_half` is `cat((-x2, x1))` over the first R lanes: lane d takes lane d+R/2 negated for
    # d < R/2 and lane d-R/2 as-is above. A fixed-distance gather, not a reduction.
    half: tl.constexpr = R // 2
    part = tl.where(d < half, d + half, d - half)
    xp = tl.load(t_ptr + base + part, mask=rot_mask, other=0)
    # Negation must be `* -1.0`, never `-x`: Triton lowers unary minus to `0.0 - x`, which yields
    # +0.0 where torch's neg gives -0.0, and that sign propagates through the add into the store.
    rot = tl.where(d < half, xp.to(tl.float32) * -1.0, xp.to(tl.float32))

    # `freqs` is [S, 1, 1, R]: one angle row per sequence position, broadcast over batch and heads.
    c = tl.load(cos_ptr + s * R + d, mask=rot_mask, other=0)
    sn = tl.load(sin_ptr + s * R + d, mask=rot_mask, other=0)

    # The three rounds. ATen's bf16 elementwise ops compute in fp32 and round once at the store, and
    # stock megatron writes `(t * cos_) + (rot * sin_)` as three such ops.
    dt = t_ptr.dtype.element_ty
    p1 = (x.to(tl.float32) * c.to(tl.float32)).to(dt)
    p2 = (rot * sn.to(tl.float32)).to(dt)  # `rot` is already fp32, exactly representing the bf16 value
    y = (p1.to(tl.float32) + p2.to(tl.float32)).to(dt)

    tl.store(o_ptr + rows * D + d, tl.where(d < R, y, x), mask=mask)


def fused_rope_bshd(t: torch.Tensor, cos_: torch.Tensor, sin_: torch.Tensor) -> torch.Tensor:
    """Bitwise-equal to stock ``_apply_rotary_pos_emb_bshd``'s tail, given hoisted ``cos_``/``sin_``.

    ``t`` is ``[S, B, H, D]`` with unit stride on the last dim only; ``cos_``/``sin_`` are
    ``[S, 1, 1, R]`` contiguous in ``t``'s dtype. Returns a contiguous ``[S, B, H, D]``.
    """
    S, B, H, D = t.shape
    R = cos_.shape[-1]
    if t.stride(-1) != 1 or not cos_.is_contiguous() or not sin_.is_contiguous():
        # The feature axis uses an implicit unit stride and cos/sin implicit row strides; a
        # violation would not fault, it would silently rotate the wrong elements.
        raise RuntimeError(f"t last dim must be unit-stride and cos/sin contiguous, got {t.stride()} / {cos_.stride()}")
    if cos_.shape != sin_.shape or cos_.shape[0] != S or cos_.dtype != t.dtype:
        raise ValueError(f"cos/sin {tuple(cos_.shape)}/{cos_.dtype} do not match t {tuple(t.shape)}/{t.dtype}")
    if R % 2 or R > D:
        raise ValueError(f"rotary_dim {R} must be even and <= head_dim {D}")

    n_rows = S * B * H
    out = torch.empty(S, B, H, D, dtype=t.dtype, device=t.device)
    _rope_kernel[(triton.cdiv(n_rows, _MB),)](
        t,
        cos_,
        sin_,
        out,
        n_rows,
        B * H,
        H,
        t.stride(0),
        t.stride(1),
        t.stride(2),
        D=D,
        R=R,
        BD=triton.next_power_of_2(D),
        MB=_MB,
        num_warps=_NUM_WARPS,
        enable_fp_fusion=False,  # load-bearing: FFMA contraction elides one product's down-cast
    )
    return out


# The hoist: cos/sin computed once per forward, cached on the freqs tensor.
def _cos_sin_for(freqs: torch.Tensor, dtype: torch.dtype, mscale: float):
    """``(cos(freqs)*mscale).to(dtype)`` and its sine, computed at most once per ``freqs`` object.

    The cache lives on the marked tensor in the view chain, so it dies with the forward that created
    it -- there is no registry to keep coherent and no boundary a CUDA-graph replay can skip. It must
    be the marked tensor and not the argument: megatron hands each layer its own
    ``rotary_pos_emb[0:q_len]`` slice, so caching on the argument would cache per layer. The row
    count joins the key so a shorter slice can never be served a longer slice's angles.
    """
    HOIST_STATS["calls"] += 1
    _host = engine_marked_host(freqs)
    host = freqs if _host is None else _host  # not `or`: a Tensor has no unambiguous truth value
    cache = getattr(host, _CACHE_ATTR, None)
    key = (dtype, float(mscale), int(freqs.shape[0]))
    if cache is not None and cache[0] == key:
        return cache[1], cache[2]
    HOIST_STATS["computes"] += 1
    # Exactly stock's two expressions. `* mscale` is kept even at mscale == 1.0 because eager emits
    # it; the multiply is exact and costs one launch per forward.
    cos_ = (torch.cos(freqs) * mscale).to(dtype)
    sin_ = (torch.sin(freqs) * mscale).to(dtype)
    try:
        setattr(host, _CACHE_ATTR, (key, cos_, sin_))
    except AttributeError:  # pragma: no cover - a tensor subclass without __dict__
        pass
    return cos_, sin_


def hoist_report() -> str:
    c, n = HOIST_STATS["computes"], HOIST_STATS["calls"]
    return f"cos/sin computed {c} time(s) over {n} rope call(s)" + (f" -- {n / max(c, 1):.1f}x hoist" if n else "")


# The live marked freqs, replaced on every mark, so at most one such tensor is held. The strong
# reference is deliberate: it is what makes the storage comparison in `engine_marked_host` safe.
_LAST_MARKED: "torch.Tensor | None" = None


def mark_engine_rope(freqs: torch.Tensor) -> torch.Tensor:
    """Stamp the engine mark on a ``freqs`` tensor, from ``_PositionIndexedRoPE.forward``.

    This mark, not the global patch, is what makes the fusion engine-only: an unmarked ``freqs``
    falls through to the original function in any process.
    """
    global _LAST_MARKED
    try:
        setattr(freqs, _MARK_ATTR, True)
    except AttributeError:  # pragma: no cover
        pass
    _LAST_MARKED = freqs
    return freqs


def engine_marked_host(freqs) -> "torch.Tensor | None":
    """The marked tensor ``freqs`` descends from, or None. Shared by both rope ops.

    Three tests in order: the attribute itself, then the ``_base`` view walk, then storage identity.
    The ``_base`` walk alone is not enough. megatron hands each layer its own
    ``rotary_pos_emb[0:q_len]`` slice, which does not carry the attribute, and vLLM runs the forward
    inside ``torch.inference_mode()`` -- a tensor allocated there is an inference tensor, for which
    torch skips view tracking entirely, so the slice reports ``_base is None`` and there is no chain
    to walk. Without the third test the detector returns False on every engine call and the fusion is
    dead code that no bitwise test can see.

    The storage test is not an identity-keyed cache: ``_LAST_MARKED`` holds a strong reference, so
    that storage cannot be freed and reused while it is held, and the claim is "this tensor shares
    storage with one I am holding" rather than "this pointer equals a number I wrote down". Dtype and
    device are compared too, so a reinterpreting view cannot sneak in.
    """
    if getattr(freqs, _MARK_ATTR, False):
        return freqs
    cur, seen = freqs, 0
    while cur is not None and seen < 4:  # bounded: real chains are depth 0 or 1
        if getattr(cur, _MARK_ATTR, False):
            return cur
        cur = getattr(cur, "_base", None)
        seen += 1
    m = _LAST_MARKED
    if (
        m is not None
        and isinstance(freqs, torch.Tensor)
        and freqs.dtype == m.dtype
        and freqs.device == m.device
        and freqs.untyped_storage().data_ptr() == m.untyped_storage().data_ptr()
    ):
        return m
    return None


def engine_mark_reaches(freqs) -> bool:
    """True iff ``freqs`` is, or descends from, the tensor ``_PositionIndexedRoPE`` marked."""
    return engine_marked_host(freqs) is not None


def _eligible(t, freqs, rotary_interleaved, mla_rotary_interleaved) -> bool:
    return (
        engine_mark_reaches(freqs)  # the engine mark, reached through a slice
        and not torch.is_grad_enabled()  # the fused path records no autograd node
        and not t.requires_grad
        and not rotary_interleaved
        and not mla_rotary_interleaved
        and t.is_cuda
        and t.dtype in (torch.bfloat16, torch.float16)
        and t.ndim == 4
        and t.stride(-1) == 1
        and freqs.ndim == 4
        and freqs.shape[0] == t.shape[0]
        and freqs.shape[1] == 1
        and freqs.shape[2] == 1
        and freqs.is_contiguous()
        and freqs.shape[-1] % 2 == 0
        and freqs.shape[-1] <= t.shape[-1]
    )


def _fused_apply_rotary_pos_emb_bshd(
    t,
    freqs,
    rotary_interleaved: bool = False,
    mla_rotary_interleaved: bool = False,
    mscale: float = 1.0,
    multi_latent_attention=None,
):
    """Drop-in for ``rope_utils._apply_rotary_pos_emb_bshd``. Falls through unless engine-marked."""
    if multi_latent_attention is not None:  # preserve upstream deprecation shim
        mla_rotary_interleaved = multi_latent_attention
    if not _eligible(t, freqs, rotary_interleaved, mla_rotary_interleaved):
        global _FALLBACK_LOGGED
        if not _FALLBACK_LOGGED and engine_mark_reaches(freqs):
            # A marked freqs that still falls back means the engine is unfused everywhere, which
            # looks exactly like a run where the flag never arrived.
            _FALLBACK_LOGGED = True
            print(
                f"[ISOEXEC-ROPE] fused RoPE FELL BACK to eager: t={tuple(t.shape)}/{t.dtype} "
                f"contig={t.is_contiguous()} freqs={tuple(freqs.shape)} interleaved={rotary_interleaved}"
                f"/{mla_rotary_interleaved} grad={torch.is_grad_enabled()}",
                flush=True,
            )
        return _ORIG_BSHD(
            t,
            freqs,
            rotary_interleaved=rotary_interleaved,
            mla_rotary_interleaved=mla_rotary_interleaved,
            mscale=mscale,
        )
    cos_, sin_ = _cos_sin_for(freqs, t.dtype, mscale)
    out = fused_rope_bshd(t, cos_, sin_)
    global _HOIST_LOGGED
    if not _HOIST_LOGGED and HOIST_STATS["calls"] >= 40:
        # Printed once so a live run shows whether the hoist is actually hoisting.
        _HOIST_LOGGED = True
        print(f"[ISOEXEC-ROPE] hoist active: {hoist_report()}", flush=True)
    return out


def install_engine_fused_rope(*, fp32_rope_installed: bool = False) -> bool:
    """Patch ``rope_utils._apply_rotary_pos_emb_bshd``. Engine process only; no-op if already
    installed or the flag is off.

    Refuses when the fp32 RoPE patch is in place: that patch is a different rounding chain (one fp32
    round) and this kernel matches stock's bf16 three-round chain. ``fp32_rope_installed`` carries
    that state in from the runtime adapter, which owns the patch, so this op never imports a runtime.
    """
    global _ORIG_BSHD, _INSTALLED
    if not fused_rope_enabled() or _INSTALLED:
        return _INSTALLED
    from megatron.core.models.common.embeddings import rope_utils

    if fp32_rope_installed:
        raise RuntimeError(
            "[ISOEXEC-ROPE] SKYRL_ISOEXEC_FUSED_ROPE=1 but the fp32 RoPE patch is installed. This "
            "kernel is bitwise-equal to megatron's STOCK bf16 rope (the local-spec production "
            "path), not to the fp32 patch. Refusing rather than matching the wrong reference."
        )
    _ORIG_BSHD = rope_utils._apply_rotary_pos_emb_bshd
    rope_utils._apply_rotary_pos_emb_bshd = _fused_apply_rotary_pos_emb_bshd
    _INSTALLED = True
    print(
        "[ISOEXEC-ROPE] fused RoPE installed on the ENGINE (K5 kernel + K5a cos/sin hoist). "
        f"tile rows/warps = {_MB}/{_NUM_WARPS}. Fires only for a freqs tensor marked by "
        "_PositionIndexedRoPE, so any other caller -- and every trainer process -- is untouched.",
        flush=True,
    )
    return True


def revert_engine_fused_rope() -> None:
    """Undo :func:`install_engine_fused_rope`; refuses unless this op is the outermost binding.

    The MLA fused rope chains on top of this one and captures this wrapper as its delegate, so
    reverting out of order would drop the MLA fusion while its own state still claimed it installed.
    Bitwise-invisible either way, which is why it is checked rather than documented.
    """
    global _INSTALLED
    if not _INSTALLED:
        return
    from megatron.core.models.common.embeddings import rope_utils

    cur = rope_utils._apply_rotary_pos_emb_bshd
    cur = getattr(cur, "__wrapped__", cur)
    if cur is not _fused_apply_rotary_pos_emb_bshd:
        raise RuntimeError(
            "[ISOEXEC-ROPE] refusing to revert out of order: rope_utils._apply_rotary_pos_emb_bshd is "
            f"{getattr(cur, '__qualname__', cur)!r}, not this op's wrapper -- something (most likely "
            "the MLA fused rope) is chained OVER this one. Revert the outer patch first."
        )
    rope_utils._apply_rotary_pos_emb_bshd = _ORIG_BSHD
    _INSTALLED = False
