"""Shape-keyed output tiles for the batch-invariant GEMM (``matmul_kernel_persistent``).

This re-tiles vLLM's kernel and changes nothing else -- same ``@triton.jit`` function object, same
``BLOCK_SIZE_K``, no split-K -- so it is bitwise-free by construction: ``BLOCK_SIZE_M``/``N`` decide
only which ``(m, n)`` share a CTA, while ``BLOCK_SIZE_K`` and split-K are what set the K-reduction
order. It is not a replacement for the batch-invariant mm override and must never become one.

The tables are keyed ``(K, N, dtype)`` and never ``M``, so batch invariance holds structurally
rather than by measurement; the trainer therefore runs the same tiles as the engine. The opt-in
decode M-bucket is the one exception, and every entry in it is certified in-process against stock
at both bucket sides before use.

Flags: ``SKYRL_ISOEXEC_MM_TILES=1`` (default off) and ``SKYRL_ISOEXEC_MM_TILES_DECODE_BUCKET=1``
(default off, requires the former).
"""

from __future__ import annotations

import os

import torch

_TILE_LOG_ONCE = False

# The pinned M-free tiles, keyed (K, N, dtype). No autotune: autotune would pick a config from the
# problem size at runtime, which is what batch invariance forbids. BLOCK_SIZE_K is pinned to vLLM's
# production value and is not a tuning knob -- it is the only constant here that affects the
# K-reduction order. Tiles are chosen against the production B layout (W.t()) and must not regress
# the trainer's large M, which is why several rows keep stock BLOCK_M/BLOCK_N and move only warps.
_TILE_TABLE: dict[tuple[int, int, torch.dtype], dict] = {
    (2048, 128, torch.bfloat16): dict(
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, num_stages=3, num_warps=4
    ),  # shared-expert fc1
    (2048, 1, torch.bfloat16): dict(
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, num_stages=4, num_warps=1
    ),  # shared_expert_gate linear -- N=1, so stock's BLOCK_N=128 masks off 127/128 columns
    (2048, 256, torch.float32): dict(
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, GROUP_SIZE_M=8, num_stages=3, num_warps=4
    ),  # router gating GEMM -- stock tiling, num_warps only
    (2048, 1032, torch.bfloat16): dict(
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, num_stages=4, num_warps=4
    ),  # GDN in_proj
    (512, 2048, torch.bfloat16): dict(
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, num_stages=4, num_warps=4
    ),  # GDN out_proj + attn o_proj (same key)
    (2048, 1024, torch.bfloat16): dict(
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, num_stages=4, num_warps=4
    ),  # attn qkv
    (64, 2048, torch.bfloat16): dict(
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, num_stages=2, num_warps=4
    ),  # shared-expert fc2
    # GLM-4.7-Flash's fp32 router: the same pathology as the Qwen `gate` row at a different width.
    # Every other GLM dense shape is bf16 and goes to the cuBLASLt provider; the fp32 router is
    # structurally excluded there, so it is re-tiled here instead.
    (2048, 64, torch.float32): dict(
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32, GROUP_SIZE_M=8, num_stages=4, num_warps=4
    ),  # GLM-4.7-Flash router gating GEMM (64 experts, fp32)
}

# The decode M-bucket (opt-in), consulted only when M < _DECODE_M_MAX. This is the module's one
# deliberate exception to M-free keying, so every entry must be bit-identical to stock at EVERY M:
# never add one without re-running the cross-M gate for that exact (K, N, dtype). Some entries are
# dead on any given model (a shape the model never runs, or one pik's RowParallelLinear intercepts
# before aten::mm); a dead entry costs one dict miss that never happens, so they are left in place.
_DECODE_M_MAX = 1024
# Entries transferred by analogy from an adjacent width rather than swept at their own; named so the
# install banner can say so, and so nobody reads a shipped constant as a measurement.
_UNSWEPT: set[tuple[int, int, torch.dtype]] = {
    (2048, 1544, torch.bfloat16),
    (2048, 1152, torch.bfloat16),
}
_DECODE_TILE_TABLE: dict[tuple[int, int, torch.dtype], dict] = {
    # The two live column-parallel widths, and a FALLBACK rather than the primary plan: mm_cublaslt
    # installs over this module and takes these shapes outright when it proves bit-equal to Triton.
    # Their tiles are transferred from the two adjacent widths below, not swept at 1544/1152; both
    # are certified in-process (bitwise vs stock, plus the cross-M gate) before use, so the worst
    # case of an unswept entry is that it is slower, never that it moves a bit.
    (2048, 1544, torch.bfloat16): dict(
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, num_stages=4, num_warps=4
    ),  # GDN in_proj_qkvzba @ tp=8
    (2048, 1152, torch.bfloat16): dict(
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, num_stages=4, num_warps=4
    ),  # attn qkv @ tp=8 (9216/8, output gate included)
    (2048, 1032, torch.bfloat16): dict(
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, num_stages=4, num_warps=4
    ),  # GDN in_proj
    (512, 2048, torch.bfloat16): dict(
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, num_stages=4, num_warps=4
    ),  # GDN out_proj + attn o_proj (same shape)
    (2048, 1024, torch.bfloat16): dict(
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, num_stages=4, num_warps=4
    ),  # attn qkv
    (2048, 256, torch.float32): dict(
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=16, BLOCK_SIZE_K=32, GROUP_SIZE_M=8, num_stages=4, num_warps=4
    ),  # router gating GEMM
    (2048, 128, torch.bfloat16): dict(
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=16, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, num_stages=4, num_warps=4
    ),  # shared-expert fc1
    (64, 2048, torch.bfloat16): dict(
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, num_stages=2, num_warps=4
    ),  # shared-expert fc2
}

# vLLM's stock config, reproduced so we can assert we never change BLOCK_SIZE_K.
_PROD_BLOCK_K = {torch.bfloat16: 64, torch.float16: 64, torch.float32: 32}


def mm_tiles_enabled() -> bool:
    return os.environ.get("SKYRL_ISOEXEC_MM_TILES", "0") == "1"


def mm_decode_bucket_enabled() -> bool:
    return os.environ.get("SKYRL_ISOEXEC_MM_TILES_DECODE_BUCKET", "0") == "1"


def _validate_table() -> None:
    """A tile entry that moves BLOCK_SIZE_K is a IsoExec bug, not a perf choice."""
    for table in (_TILE_TABLE, _DECODE_TILE_TABLE):
        for (K, N, dtype), cfg in table.items():
            assert cfg["BLOCK_SIZE_K"] == _PROD_BLOCK_K[dtype], (
                f"tile entry ({K}, {N}, {dtype}) sets BLOCK_SIZE_K={cfg['BLOCK_SIZE_K']} but production "
                f"is {_PROD_BLOCK_K[dtype]}.  BLOCK_SIZE_K changes the K-REDUCTION ORDER and is not a "
                f"tuning knob -- only BLOCK_SIZE_M/N/num_warps/num_stages may differ from stock."
            )


def _run_tiled(bi, a, b, cfg, bias=None):
    """Launch ``matmul_kernel_persistent`` with caller-supplied tile constants (the wrapper's body,
    factored out so the install-time self-check runs the EXACT code the hot path will run)."""
    import triton

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    num_sms = bi.num_compute_units(a.device.index)
    grid = (min(num_sms, triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"])),)
    bi.matmul_kernel_persistent[grid](
        a,
        b,
        c,
        bias,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        NUM_SMS=num_sms,
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        HAS_BIAS=bias is not None,
        **cfg,
    )
    return c


def _bitcmp(x: torch.Tensor, y: torch.Tensor) -> int:
    """Bit-pattern mismatch count. ``torch.equal`` treats ``-0.0 == +0.0`` and so would be blind to
    the class of divergence a re-tiling could plausibly introduce."""
    w = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}[x.dtype]
    return int((x.view(w) != y.view(w)).sum().item())


def _certify_cross_m(bi, K: int, N: int, dtype: torch.dtype, cfg: dict) -> tuple[bool, str]:
    """The cross-M gate a decode-bucket entry owes, run in-process on the machine that will use it.

    A row computed under the bucket's tile at a decode M must be bit-identical to the same row
    computed under stock at an M above the bucket boundary, because the engine decodes below it and
    the trainer scores above it. Fail-closed: an entry that cannot prove this is dropped.
    """
    dev = "cuda"
    stock = getattr(bi, "_isoexec_stock_matmul_persistent", bi.matmul_persistent)
    m_hi = 2 * _DECODE_M_MAX  # unambiguously the trainer bucket
    for m_lo in (1, 64, 320, _DECODE_M_MAX - 1):
        for scale in (0.05, 2.0**-9):
            W = torch.randn(N, K, device=dev, dtype=dtype) * scale
            b = W.t()
            a_hi = torch.randn(m_hi, K, device=dev, dtype=dtype) * scale
            a_lo = a_hi[:m_lo].contiguous()
            got = _run_tiled(bi, a_lo, b, cfg)  # what the bucket runs at decode
            ref = stock(a_hi, b, bias=None)  # what the trainer runs above the boundary
            bad = _bitcmp(ref[:m_lo], got)
            if bad:
                return False, (
                    f"cross-M: {bad} elements differ between decode-tile@M={m_lo} and "
                    f"stock@M={m_hi} on the SAME rows (scale {scale})"
                )
    return True, ""


def _certify_entry(bi, K: int, N: int, dtype: torch.dtype, cfg: dict) -> tuple[bool, str]:
    """Prove this entry bitwise-neutral against stock, on this machine, before it is used.

    Re-tiling is bitwise-free by construction, but this is the per-process gate that makes the
    argument checkable where it is relied on. The second (underflow) population is where two output
    tilings would first disagree if the construction argument were wrong.
    """
    dev = "cuda"
    stock = getattr(bi, "_isoexec_stock_matmul_persistent", bi.matmul_persistent)
    base = dict(_STOCK_CFG[dtype])
    for M in (64, 320, 2048):
        for scale in (0.05, 2.0**-9):
            a = torch.randn(M, K, device=dev, dtype=dtype) * scale
            W = torch.randn(N, K, device=dev, dtype=dtype) * scale
            b = W.t()  # production layout: F.linear passes W.t(), so B arrives K-contiguous
            ref = _run_tiled(bi, a, b, base)
            got = _run_tiled(bi, a, b, cfg)
            bad = _bitcmp(ref, got)
            if bad:
                return False, f"{bad} elements differ vs stock tiling at M={M} (scale {scale})"
            # Also against whatever `stock` currently resolves to, so a table entry can never
            # diverge from the function the unlisted shapes take.
            if _bitcmp(stock(a, b, bias=None), got):
                return False, f"differs from the live stock matmul_persistent at M={M}"
    return True, ""


# vLLM's production constants, reproduced so an entry can be compared against them directly.
_STOCK_CFG = {
    torch.bfloat16: dict(
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, num_stages=3, num_warps=8
    ),
    torch.float16: dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, num_stages=3, num_warps=8),
    torch.float32: dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, GROUP_SIZE_M=8, num_stages=3, num_warps=8),
}


def install_mm_tiles() -> bool:
    """Point ``matmul_persistent`` at a tile-aware launcher. Idempotent; True if installed.

    The seam is ``batch_invariant.matmul_persistent`` itself, not a second ``aten::mm``/``aten::addmm``
    registration: vLLM's registered wrappers resolve that name through module globals at call time,
    so rebinding it reaches both through the one existing registration and leaves this module
    order-independent with respect to the aten install.
    """
    global _TILE_LOG_ONCE

    if not mm_tiles_enabled():
        return False
    if not _TILE_TABLE:
        print(
            "[ISOEXEC-MM-TILES] SKYRL_ISOEXEC_MM_TILES=1 but the tile table is EMPTY -- "
            "leaving vLLM's stock tiles in place. Run examples/isoexec/nightly/mm_tile_sweep.py.",
            flush=True,
        )
        return False

    _validate_table()

    from vllm.model_executor.layers import batch_invariant as bi

    from . import mm_shapes

    stock = getattr(bi, "_isoexec_stock_matmul_persistent", bi.matmul_persistent)
    if getattr(bi.matmul_persistent, "_isoexec_tiled", False):
        return True  # already installed

    decode_bucket = mm_decode_bucket_enabled()

    # Per-entry certification, fail-closed: an entry that cannot prove itself bitwise-neutral on
    # this machine is dropped and its shape keeps vLLM's stock tiling.
    live_tiles, live_decode, dropped = {}, {}, []
    for table, live in ((_TILE_TABLE, live_tiles), (_DECODE_TILE_TABLE, live_decode)):
        if table is _DECODE_TILE_TABLE and not decode_bucket:
            continue
        for (K, N, dtype), cfg in table.items():
            try:
                ok, why = _certify_entry(bi, K, N, dtype, cfg)
                # A decode-bucket entry additionally owes the cross-M gate: it is the one place here
                # where a row's tiling depends on the batch it rode in.
                if ok and table is _DECODE_TILE_TABLE:
                    ok, why = _certify_cross_m(bi, K, N, dtype, cfg)
            except Exception as e:  # noqa: BLE001
                ok, why = False, f"certification raised {type(e).__name__}: {e}"
            if ok:
                live[(K, N, dtype)] = cfg
            else:
                dropped.append(f"({K},{N},{dtype}) [{'decode' if table is _DECODE_TILE_TABLE else 'm-free'}]: {why}")

    def matmul_persistent_tiled(a, b, bias=None):
        key = (a.shape[1], b.shape[1], a.dtype)
        cfg = None
        # Decode M-bucket first (opt-in): shape-only boundary, and every entry is bit-identical to
        # stock at every M, so which bucket ran cannot move a bit.
        if decode_bucket and a.shape[0] < _DECODE_M_MAX:
            cfg = live_decode.get(key)
        if cfg is None:
            cfg = live_tiles.get(key)
        if cfg is None:
            return stock(a, b, bias=bias)  # every unlisted shape keeps vLLM's exact behaviour
        return _run_tiled(bi, a, b, cfg, bias)

    matmul_persistent_tiled._isoexec_tiled = True
    bi._isoexec_stock_matmul_persistent = stock
    bi.matmul_persistent = matmul_persistent_tiled

    if not _TILE_LOG_ONCE:
        _TILE_LOG_ONCE = True
        for reason in dropped:
            print(f"[ISOEXEC-MM-TILES] ENTRY DROPPED (fail-closed, stock tiling) {reason}", flush=True)
        print(
            f"[ISOEXEC-MM-TILES] shape-keyed tiles installed for {len(live_tiles)}/{len(_TILE_TABLE)} "
            f"(K,N,dtype) shapes (BLOCK_SIZE_K pinned to stock; output tiling only). Every entry "
            f"CERTIFIED bit-identical to stock live on this device at M in {{64,320,2048}} x 2 "
            f"populations, {len(dropped)} dropped. Flag SKYRL_ISOEXEC_MM_TILES=1. "
            f"Decode M-bucket (SKYRL_ISOEXEC_MM_TILES_DECODE_BUCKET): "
            f"{'ON, ' + str(len(live_decode)) + ' shapes, M<' + str(_DECODE_M_MAX) + ', each additionally cross-M gated' if decode_bucket else 'off'}.",
            flush=True,
        )
        unswept_live = sorted(k for k in (set(live_tiles) | set(live_decode)) if k in _UNSWEPT)
        if unswept_live:
            print(
                "[ISOEXEC-MM-TILES] PROVENANCE: "
                + "  ".join(f"({K},{N})" for K, N, _d in unswept_live)
                + " are TRANSFERRED tiles, not swept at their own width -- bitwise-certified and "
                "cross-M gated above, but their SPEED is an argument, not a measurement. Run "
                "examples/isoexec/nightly/mm_tile_sweep.py --model to replace or retire them.",
                flush=True,
            )
        covered = set(live_tiles) | set(live_decode)
        mm_shapes.report_install_coverage("MM_TILES", lambda key: key in covered, flag="SKYRL_ISOEXEC_MM_TILES")
    return True
