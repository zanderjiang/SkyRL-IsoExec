"""Cache the MoE router's fp32 weight cast, refreshed once per weight sync instead of once per forward.

IsoExec pins ``moe_router_dtype="fp32"``, so megatron's router gating re-casts the ``[E, H]`` bf16 weight
on every forward even though it changes only at a weight sync. The cached buffer holds exactly
``weight.to(torch.float32)`` in the same contiguous layout, so ``torch.mm`` sees identical operands.

Installed per INSTANCE on the engine's ``GPTModel`` only: the trainer's cast is grad-carrying, and under
``VLLM_ENABLE_V1_MULTIPROCESSING=0`` it shares this process, so a class-level rebind would hand the trainer
a detached copy and deliver zero gradient to the router weight. Buffers are allocated once and refreshed in
place at the sync seam -- a reallocation would strand a captured decode graph's baked pointer, so it raises
instead. Everything else (flag off, TE present, non-fp32 router dtype, a bias, grad enabled, ...) declines
back to megatron's own ``gating``.
"""

from __future__ import annotations

import os

import torch

BANNER = "[ISOEXEC-MOE-ROUTERCAST]"
_ENV = "SKYRL_ISOEXEC_MOE_ROUTER_CAST_CACHE"

# The registry of live fp32 buffers, keyed by id(router). Strong refs to the routers themselves keep
# the id() keys sound (no id recycling) and are what `refresh_all` walks at the sync seam.
_ENTRIES: dict[int, "_Entry"] = {}

_served = 0
_declined = 0
_refreshed = 0
_reported = 0
_decline_reason = ""


def cast_cache_enabled() -> bool:
    """Read per call, so an in-process A/B can flip it. Default ON."""
    return os.environ.get(_ENV, "1").lower() not in ("", "0", "false", "no")


class _Entry:
    """One router's persistent fp32 weight buffer. Allocated once, refreshed in place, never moved."""

    __slots__ = ("router", "weight_shape", "buf")

    def __init__(self, router, weight: torch.Tensor):
        self.router = router
        self.weight_shape = tuple(weight.shape)
        self.buf = weight.to(torch.float32)
        if not self.buf.is_contiguous():  # pragma: no cover - .to() of a contiguous param is contiguous
            self.buf = self.buf.contiguous()

    def refresh(self, weight: torch.Tensor) -> None:
        """In place, always. A reallocation here is a graph-safety bug, so it raises instead."""
        if tuple(weight.shape) != self.weight_shape:
            raise RuntimeError(
                f"{BANNER} GRAPH-UNSAFE refresh: router weight changed shape "
                f"{self.weight_shape} -> {tuple(weight.shape)}. A captured decode graph holds the "
                "address of the old fp32 buffer; refusing to reallocate under it."
            )
        if weight.device != self.buf.device:
            raise RuntimeError(
                f"{BANNER} GRAPH-UNSAFE refresh: router weight moved to {weight.device}, "
                f"fp32 buffer lives on {self.buf.device}."
            )
        self.buf.copy_(weight)  # the cast, exactly as `weight.to(torch.float32)` performs it


def router_supported(router) -> bool:
    """Can this ``TopKRouter`` instance take the cached cast? Every clause is a decline reason."""
    try:
        from megatron.core.transformer.moe import moe_utils
    except Exception:  # pragma: no cover
        return False
    cfg = getattr(router, "config", None)
    w = getattr(router, "weight", None)
    return bool(
        cfg is not None
        and w is not None
        # The TE branch hands the bf16 weight straight to te_general_gemm and never casts it, so there
        # is nothing to cache and the fallback would change the algebra.
        and getattr(moe_utils, "te_general_gemm", None) is None
        and getattr(cfg, "moe_router_dtype", None) == "fp32"
        and getattr(router, "bias", None) is None
        and w.dim() == 2
        and w.dtype == torch.bfloat16
        and w.is_cuda
    )


def _entry_for(router):
    e = _ENTRIES.get(id(router))
    if e is None:
        e = _Entry(router, router.weight)
        _ENTRIES[id(router)] = e
    return e


def _cached_cast_gating(self, input: torch.Tensor) -> torch.Tensor:
    """Replaces ``Router.gating`` on ENGINE instances: verbatim megatron, minus the repeated weight cast.

    The cached buffer holds exactly ``weight.to(fp32)``, contiguous ``[E, H]`` as ``.to()`` produces it, so
    ``.t()`` has the same strides and ``torch.mm`` sees the same operands as megatron's
    ``torch.mm(inp.to(fp32), weight.to(fp32).t())``.
    """
    global _served, _declined, _decline_reason
    w = self.weight
    if not cast_cache_enabled() or (torch.is_grad_enabled() and (input.requires_grad or w.requires_grad)):
        _declined += 1
        _decline_reason = "flag off" if not cast_cache_enabled() else "grad enabled"
        _maybe_report()
        return self._ix_castcache_orig_gating(input)
    if w.dtype != torch.bfloat16 or w.dim() != 2 or self.bias is not None or self.config.moe_router_dtype != "fp32":
        _declined += 1
        _decline_reason = f"w.dtype={w.dtype} w.dim={w.dim()} bias={self.bias is not None}"
        _maybe_report()
        return self._ix_castcache_orig_gating(input)

    e = _entry_for(self)
    inp_shape = input.shape
    flat = input.view(-1, inp_shape[-1])
    out = torch.mm(flat.to(torch.float32), e.buf.t())
    _served += 1
    _maybe_report()
    return out.view(*inp_shape[:-1], -1)


def install_engine_router_cast_cache(gpt_modules) -> int:
    """Rebind ``gating`` on the ENGINE GPTModel's ``TopKRouter`` instances. Returns the count.

    INSTANCE-level, so the trainer -- which builds the identical classes and, under
    ``VLLM_ENABLE_V1_MULTIPROCESSING=0``, shares this process -- is untouched by construction. The
    ``hasattr`` guard makes it idempotent: a second call must not capture the already-cached bound method
    as "the original".
    """
    on = cast_cache_enabled()
    try:
        from megatron.core.transformer.moe.router import TopKRouter
    except Exception:  # pragma: no cover
        return 0

    n = skipped = 0
    for m in gpt_modules.modules():
        if not isinstance(m, TopKRouter):
            continue
        if on and not hasattr(m, "_ix_castcache_orig_gating") and router_supported(m):
            m._ix_castcache_orig_gating = m.gating
            m.gating = _cached_cast_gating.__get__(m, type(m))
            _entry_for(m)  # allocate NOW, at build time, so no forward ever allocates
            n += 1
        else:
            skipped += 1

    # Printed flag on or off: this line is the only in-band evidence that the env var reached the engine
    # actor. If it says OFF on a run that exported ON, the allowlist did not forward it.
    status = "ON -- one cast per weight sync" if on else "OFF -- cast remains per forward"
    print(
        f"{BANNER} pid={os.getpid()} cached fp32 cast on {n} router(s) (skipped {skipped}); " f"{_ENV}={status}",
        flush=True,
    )
    return n


def refresh_all(reason: str = "") -> int:
    """Re-cast every registered router weight IN PLACE. Called from the engine's weight-sync seam.

    EAGER, not lazy: the decode path is CUDA-graph replayed with the buffer's address baked in, so a
    refresh that waited for the next Python-level forward would never run for a replayed step.
    """
    global _refreshed
    n = 0
    for e in list(_ENTRIES.values()):
        e.refresh(e.router.weight)
        n += 1
    _refreshed += 1
    if n:
        print(f"{BANNER} pid={os.getpid()} refreshed {n} router fp32 buffer(s) in place{reason}", flush=True)
    return n


def invalidate_all(reason: str = "") -> int:
    """The name the sync seam calls. A cache with no staleness stamp invalidates by REFRESHING."""
    return refresh_all(reason)


def drop_all() -> int:
    """Release every buffer (teardown / offload). A later forward re-allocates on first use."""
    n = len(_ENTRIES)
    _ENTRIES.clear()
    return n


def _maybe_report() -> None:
    """One line at 1/10/100/... served+declined calls."""
    global _reported
    total = _served + _declined
    if total < 1 or (total & (total - 1)) != 0 or total == _reported:
        return
    _reported = total
    print(
        f"{BANNER} pid={os.getpid()} served={_served} declined={_declined} refreshes={_refreshed}"
        + (f" last_decline={_decline_reason}" if _declined else ""),
        flush=True,
    )
