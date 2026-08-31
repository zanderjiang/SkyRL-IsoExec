"""Fused attention RoPE for the engine: one Triton kernel per call, plus a cos/sin hoist.

Bitwise-equal to megatron's stock ``_apply_rotary_pos_emb_bshd``, whose multiply-add is a bf16
three-round chain. Engine-only and inference-only. Flag: ``SKYRL_ISOEXEC_FUSED_ROPE=1``, off.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

FLAG = "SKYRL_ISOEXEC_FUSED_ROPE"

# Pinned launch geometry; nothing in this tree autotunes.
_MB = 4  # rows (= tokens x heads) per program
_NUM_WARPS = 4

# Populated by `_cos_sin_for`, read by `hoist_report`.
HOIST_STATS = {"calls": 0, "computes": 0}

_LOGGED = False
_FALLBACK_LOGGED = False
_HOIST_LOGGED = False
_ORIG_BSHD = None
_INSTALLED = False

# Both attributes live on the `freqs` tensor, so they die with it: no invalidation boundary.
_MARK_ATTR = "_ix_engine_rope"
_CACHE_ATTR = "_ix_rope_cos_sin"


def fused_rope_enabled() -> bool:
    """``SKYRL_ISOEXEC_FUSED_ROPE=1``, default off; logged once per process at first read."""
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

    Addressed by explicit ``[S, B, H]`` strides; only the last dim must be unit stride, since qkv
    splits are not always fully contiguous. The output is contiguous.
    """
    # int64 row index: an int32 offset would silently wrap past 2^31 rather than faulting.
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

    # `_rotate_half` as a fixed-distance gather: lane d takes lane d+R/2 negated for d < R/2,
    # lane d-R/2 as-is above.
    half: tl.constexpr = R // 2
    part = tl.where(d < half, d + half, d - half)
    xp = tl.load(t_ptr + base + part, mask=rot_mask, other=0)
    # Negation must be `* -1.0`, never `-x`: Triton lowers unary minus to `0.0 - x`, which gives
    # +0.0 where torch's neg gives -0.0, and that sign reaches the store.
    rot = tl.where(d < half, xp.to(tl.float32) * -1.0, xp.to(tl.float32))

    # `freqs` is [S, 1, 1, R]: one angle row per sequence position, broadcast over batch and heads.
    c = tl.load(cos_ptr + s * R + d, mask=rot_mask, other=0)
    sn = tl.load(sin_ptr + s * R + d, mask=rot_mask, other=0)

    # The three rounds: stock megatron writes `(t * cos_) + (rot * sin_)` as three bf16 elementwise
    # ops, each computing in fp32 and rounding once.
    dt = t_ptr.dtype.element_ty
    p1 = (x.to(tl.float32) * c.to(tl.float32)).to(dt)
    p2 = (rot * sn.to(tl.float32)).to(dt)  # `rot` is already fp32, exact for the bf16 value
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
        # Implicit unit/row strides: a violation would not fault, it would rotate wrong elements.
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

    Cached on the MARKED tensor, not the argument: megatron hands each layer its own
    ``rotary_pos_emb[0:q_len]`` slice, so caching on the argument would cache per layer. The row
    count joins the key so a shorter slice is never served a longer slice's angles.
    """
    HOIST_STATS["calls"] += 1
    _host = engine_marked_host(freqs)
    host = freqs if _host is None else _host  # not `or`: a Tensor has no unambiguous truth value
    cache = getattr(host, _CACHE_ATTR, None)
    key = (dtype, float(mscale), int(freqs.shape[0]))
    if cache is not None and cache[0] == key:
        return cache[1], cache[2]
    HOIST_STATS["computes"] += 1
    # Exactly stock's two expressions; `* mscale` is kept even at 1.0 because eager emits it.
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


# The live marked freqs. The strong reference is deliberate: it keeps the storage alive, which is
# what makes the storage comparison in `engine_marked_host` safe.
_LAST_MARKED: "torch.Tensor | None" = None


def mark_engine_rope(freqs: torch.Tensor) -> torch.Tensor:
    """Stamp the engine mark on a ``freqs`` tensor; this is what makes the fusion engine-only."""
    global _LAST_MARKED
    try:
        setattr(freqs, _MARK_ATTR, True)
    except AttributeError:  # pragma: no cover
        pass
    _LAST_MARKED = freqs
    return freqs


def engine_marked_host(freqs) -> "torch.Tensor | None":
    """The marked tensor ``freqs`` descends from, or None. Shared by both rope ops.

    Three tests: the attribute, the ``_base`` view walk, then storage identity. The walk alone is
    not enough -- under ``torch.inference_mode()`` torch skips view tracking, so a layer's
    ``rotary_pos_emb[0:q_len]`` slice reports ``_base is None``. Dtype and device are compared too.
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
            # A marked freqs that still falls back means the engine is unfused everywhere.
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

    Refuses when the fp32 RoPE patch is in place: that patch is a different rounding chain (one
    fp32 round) and this kernel matches stock's bf16 three-round chain.
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

    The MLA fused rope chains over this one, so reverting out of order would drop the MLA fusion
    while its own state still claimed it installed.
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
