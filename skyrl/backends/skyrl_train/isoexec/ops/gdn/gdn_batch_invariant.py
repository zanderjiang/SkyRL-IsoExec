"""Pin GDN launch geometry that participates in the IsoExec arithmetic contract.

FLA chunk kernels otherwise autotune independently in the trainer and engine processes, which can select
different reduction orders. ``pin_fla_autotune_configs`` selects the same declared config in both runtimes;
``pin_gdn_rmsnorm_rows_per_block`` fixes the gated-norm tile height to one row so its reduction order cannot
depend on batch size. Both are on by default, and ``SKYRL_ISOEXEC_GDN_CONFIG_INDEX`` must match across runtimes.
"""

from __future__ import annotations

import importlib
import logging
import os

logger = logging.getLogger(__name__)

# Every autotuned Triton kernel reachable from `chunk_gated_delta_rule`. Missing entries are
# tolerated (vLLM moves kernels between versions) but reported: an unpinned kernel silently
# reintroduces per-process autotuning.
_FLA_KERNELS: tuple[tuple[str, str], ...] = (
    ("chunk_scaled_dot_kkt", "chunk_scaled_dot_kkt_fwd_kernel"),
    ("solve_tril", "solve_tril_16x16_kernel"),
    ("solve_tril", "merge_16x16_to_32x32_inverse_kernel"),
    ("solve_tril", "merge_16x16_to_64x64_inverse_kernel"),
    ("wy_fast", "recompute_w_u_fwd_kernel"),
    ("chunk_delta_h", "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"),
    ("chunk_o", "chunk_fwd_kernel_o"),
    ("cumsum", "chunk_local_cumsum_scalar_kernel"),
    ("cumsum", "chunk_local_cumsum_vector_kernel"),
)

_pinned = False


def gdn_pin_enabled() -> bool:
    return os.environ.get("SKYRL_ISOEXEC_GDN_PIN_CONFIGS", "1") == "1"


def _config_index() -> int:
    return int(os.environ.get("SKYRL_ISOEXEC_GDN_CONFIG_INDEX", "0"))


def _autotuner(module_name: str, kernel_name: str):
    """Return the Autotuner behind a (possibly Heuristics-wrapped) Triton kernel, or None."""
    try:
        mod = importlib.import_module(f"vllm.model_executor.layers.fla.ops.{module_name}")
    except Exception as e:  # pragma: no cover - vLLM without the vendored FLA ops
        logger.info("[isoexec-gdn] no fla.ops.%s (%s)", module_name, e)
        return None
    kernel = getattr(mod, kernel_name, None)
    if kernel is None:
        return None
    # triton.heuristics wraps triton.autotune wraps JITFunction; unwrap until we find `.configs`.
    obj = kernel
    for _ in range(4):
        if hasattr(obj, "configs") and hasattr(obj, "cache"):
            return obj
        obj = getattr(obj, "fn", None)
        if obj is None:
            return None
    return None


def pin_fla_autotune_configs() -> int:
    """Pin every FLA chunk kernel to one statically-chosen config. Idempotent; returns the count."""
    global _pinned

    if _pinned:
        return 0
    if not gdn_pin_enabled():
        print(
            "[ISOEXEC-GDN] SKYRL_ISOEXEC_GDN_PIN_CONFIGS=0 -> leaving Triton autotune ON. "
            "GDN kernels are then nondeterministic and NOT batch-invariant (baseline A/B only).",
            flush=True,
        )
        return 0

    idx = _config_index()
    pinned, missing = [], []
    for module_name, kernel_name in _FLA_KERNELS:
        at = _autotuner(module_name, kernel_name)
        if at is None:
            missing.append(f"{module_name}.{kernel_name}")
            continue
        configs = list(at.configs)
        if not configs:
            missing.append(f"{module_name}.{kernel_name}(no configs)")
            continue
        chosen = configs[idx] if idx < len(configs) else configs[0]
        at.configs = [chosen]
        at.cache.clear()
        pinned.append(f"{kernel_name}:{chosen.kwargs}/w{chosen.num_warps}/s{chosen.num_stages}")

    if missing:
        # Loud, not fatal: an unpinned kernel means per-process autotuning is back for that op.
        logger.warning("[isoexec-gdn] could not pin: %s", ", ".join(missing))
    _pinned = True
    print(
        f"[ISOEXEC-GDN] pinned {len(pinned)} FLA autotune configs (index={idx}) -> deterministic, "
        f"batch-invariant GDN. Unpinned: {len(missing)}",
        flush=True,
    )
    logger.info("[isoexec-gdn] pinned configs: %s", "; ".join(pinned))
    return len(pinned)


_norm_pinned = False


def pin_gdn_rmsnorm_rows_per_block() -> bool:
    """Make ``RMSNormGated`` (the GDN output norm) batch-invariant. Idempotent.

    ``layernorm_guard.layer_norm_fwd`` derives its tile height from the row count M, and the kernel
    reduces a ``[ROWS_PER_BLOCK, BLOCK_N]`` tile with ``tl.sum(x, axis=1)``. The tile shape therefore
    decides the order of the fp32 reduction and the last bit of ``rstd``, so decode (small M) and
    prefill (large M) can disagree on the same input row. Pinning the height to one row removes the
    M dependence; the cost is a taller grid at prefill on a memory-bound kernel.
    """
    global _norm_pinned

    if _norm_pinned:
        return True
    try:
        from vllm.model_executor.layers.fla.ops import layernorm_guard as lg
    except Exception as e:  # pragma: no cover
        logger.warning("[isoexec-gdn] cannot pin RMSNormGated tile height: %s", e)
        return False

    # A correctness pin, not a tuning knob: any other value reintroduces M-dependent rounding.
    rows = 1
    lg.calc_rows_per_block = lambda M, device: rows
    _norm_pinned = True
    print(
        f"[ISOEXEC-GDN] pinned RMSNormGated tile height to {rows} row(s) -> batch-invariant "
        "gated norm (was M-dependent: 1 row at decode, 4 at prefill)",
        flush=True,
    )
    return True


def verify_gdn_batch_invariance(*, heads: int = 8, k_dim: int = 128, v_dim: int = 128) -> None:
    """Assert determinism, cross-sequence invariance and prefix invariance. Raises on violation."""
    import torch
    from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule
    from vllm.model_executor.layers.fla.ops.index import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
    )
    from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd
    from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE as C

    dev = "cuda"

    def make(n, seed):
        torch.manual_seed(seed)
        return dict(
            # q,k must be l2-normalized or the delta rule is not a contraction and the state blows up
            q=l2norm_fwd(torch.randn(1, n, heads, k_dim, dtype=torch.bfloat16, device=dev)),
            k=l2norm_fwd(torch.randn(1, n, heads, k_dim, dtype=torch.bfloat16, device=dev)),
            v=torch.randn(1, n, heads, v_dim, dtype=torch.bfloat16, device=dev),
            g=-torch.nn.functional.softplus(torch.randn(1, n, heads, device=dev)).float(),
            beta=torch.rand(1, n, heads, dtype=torch.bfloat16, device=dev).sigmoid(),
        )

    def call(t, cu=None):
        ci = co = None
        if cu is not None:
            ci, co = prepare_chunk_indices(cu, C), prepare_chunk_offsets(cu, C)
        return chunk_gated_delta_rule(
            q=t["q"],
            k=t["k"],
            v=t["v"],
            g=t["g"],
            beta=t["beta"],
            cu_seqlens=cu,
            chunk_indices=ci,
            chunk_offsets=co,
            use_qk_l2norm_in_kernel=False,
        )[0]

    # Needs NT >= 5 chunks: fewer than that fits in one wave and hides racy autotune configs.
    lens = [150, 37, 64, 201]
    seqs = [make(n, 10 + i) for i, n in enumerate(lens)]
    packed = {key: torch.cat([s[key] for s in seqs], dim=1) for key in seqs[0]}
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device=dev)

    a, b = call(packed, cu), call(packed, cu)
    if not torch.equal(a, b):
        raise RuntimeError(
            f"[isoexec-gdn] chunk_gated_delta_rule is NONDETERMINISTIC "
            f"(max |diff| {float((a - b).abs().max()):.3e}). A Triton autotune config is racy; "
            "pin a different SKYRL_ISOEXEC_GDN_CONFIG_INDEX."
        )

    off = 0
    for i, n in enumerate(lens):
        alone = call(seqs[i], torch.tensor([0, n], dtype=torch.int32, device=dev))
        if not torch.equal(a[0, off : off + n], alone[0]):
            d = float((a[0, off : off + n] - alone[0]).abs().max())
            raise RuntimeError(
                f"[isoexec-gdn] NOT cross-sequence invariant: sequence {i} (len {n}) changes by "
                f"{d:.3e} depending on its varlen batch companions."
            )
        off += n

    seq = make(256, 7)
    full = call(seq)
    for t in (0, 1, 63, 64, 65, 127, 255):
        pref = call({key: seq[key][:, : t + 1] for key in seq})
        if not torch.equal(pref[0, t], full[0, t]):
            d = float((pref[0, t] - full[0, t]).abs().max())
            raise RuntimeError(
                f"[isoexec-gdn] NOT prefix invariant at t={t} ({d:.3e}): a token's output depends on "
                "later tokens. Chunk-consistent decode cannot be bitwise."
            )
