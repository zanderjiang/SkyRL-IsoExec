"""Debug-mode hook installation: wrap installed region impls with digest capture.

Reaches the regions the way the adapters do -- by rebinding the module/method attribute that
holds the *currently installed* impl -- so whatever kernel each side actually runs (IsoExec's or
native) is what gets traced. Nothing here edits adapter or worker files; the call sites named in
``INTEGRATION.md`` invoke :func:`install_debug_hooks` after their normal isoexec install.

Everything is a no-op unless ``SKYRL_ISOEXEC_DEBUG_TRACE`` is set. Idempotent; safe to call more
than once and safe to call when megatron / the GDN ops are not importable in this process.

Concrete regions installed:

  moe.router  ``topk_routing_with_score_function`` in megatron's ``moe_utils`` / ``router`` /
              ``token_dispatcher`` namespaces (the same binding IsoExec's deterministic router
              patch installs into; both runtimes execute megatron code on the unified-GPTModel
              route, so the one installer covers trainer and engine).
  gdn.core    the GDN delta-rule core: ``gdn_core`` (the trainer's door), the native fused
              variants, and the engine-facing kernels ``gdn_recurrent_kernel`` /
              ``gdn_native_core_kernel`` -- rebound in ``gdn_ops`` and in every already-imported
              caller namespace. The re-entrancy guard in ``trace`` collapses nested hits to one
              record per outermost call. :func:`install_gdn_layer_hooks` additionally wraps the
              swapped engine GDN layers so records carry a real layer index.
"""

from __future__ import annotations

import sys
from typing import Callable, Dict, Optional

import torch

from . import trace

_wrapper_cache: Dict[int, Callable] = {}


def _shared_wrap(region: str, fn: Callable, **kw) -> Callable:
    """One wrapper per underlying function, so multi-namespace rebinding shares one counter."""
    if getattr(fn, trace._WRAP_ATTR, None) is not None:
        return fn
    w = _wrapper_cache.get(id(fn))
    if w is None or getattr(w, "_isoexec_debug_inner", None) is not fn:
        w = trace.wrap_region(region, fn, **kw)
        _wrapper_cache[id(fn)] = w
    return w


def _gdn_case(args, kwargs, out) -> str:
    tr = trace.get_tracer()
    side = tr.side if tr else "unknown"
    if side != "engine":
        return tr.default_case() if tr else "unknown"
    cu = kwargs.get("cu_seqlens")
    if isinstance(cu, torch.Tensor) and cu.numel() >= 2:
        nseq = cu.numel() - 1
        return "engine_decode" if int(cu[-1]) == nseq else "engine_prefill"
    return "engine"


def _install_moe_router() -> int:
    try:
        from megatron.core.transformer.moe import moe_utils, router, token_dispatcher
    except Exception:
        return 0
    n = 0
    for ns in (moe_utils, router, token_dispatcher):
        cur = getattr(ns, "topk_routing_with_score_function", None)
        if cur is None or getattr(cur, trace._WRAP_ATTR, None) is not None:
            continue
        setattr(ns, "topk_routing_with_score_function", _shared_wrap("moe.router", cur))
        n += 1
    return n


_GDN_TARGETS = (
    "gdn_core",
    "gdn_native_core",
    "gdn_native_chunk_synced",
    "gdn_recurrent_kernel",
    "gdn_native_core_kernel",
)
_GDN_NAMESPACES = (
    "skyrl.backends.skyrl_train.isoexec.ops.gdn.gdn_ops",
    "skyrl.backends.skyrl_train.isoexec.ops.gdn.gdn_recurrent_state",
    "skyrl.backends.skyrl_train.isoexec.ops.gdn.gdn_chunk_synced",
    "skyrl.backends.skyrl_train.isoexec.ops.gdn.gdn_chunk_synced_state",
)


def _install_gdn_core() -> int:
    try:
        from ..ops.gdn import gdn_ops  # noqa: F401 -- ensures the defining module is loaded
    except Exception:
        return 0
    n = 0
    for mod_name in _GDN_NAMESPACES:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        for name in _GDN_TARGETS:
            cur = getattr(mod, name, None)
            if cur is None or not callable(cur) or getattr(cur, trace._WRAP_ATTR, None) is not None:
                continue
            setattr(mod, name, _shared_wrap("gdn.core", cur, case_fn=_gdn_case))
            n += 1
    return n


def install_gdn_layer_hooks(gpt_modules) -> int:
    """Engine, post-``swap_gdn_core``: wrap each swapped GDN layer's forward, layer index known.

    Call site (integration): right after ``swap_gdn_core(...)`` returns. Records land as region
    ``gdn.core`` with a real ``layer``; the module-level kernel wraps then stay silent underneath
    via the re-entrancy guard.
    """
    if not trace.enabled():
        return 0
    n = 0
    for layer in getattr(getattr(gpt_modules, "decoder", None), "layers", []) or []:
        gdn = getattr(layer, "self_attention", None)
        if gdn is None or getattr(gdn, "_isoexec_state", None) is None:
            continue
        cur = gdn.forward
        if getattr(cur, trace._WRAP_ATTR, None) is not None:
            continue
        idx = getattr(layer, "layer_number", None)
        layer_idx = idx - 1 if isinstance(idx, int) else None
        gdn.forward = trace.wrap_region(
            "gdn.core", cur, case_fn=_gdn_case, layer_fn=lambda a, k, li=layer_idx: li
        )
        n += 1
    return n


def install_debug_hooks() -> int:
    """Install every region hook reachable in this process. Returns bindings wrapped (0 = off)."""
    if not trace.enabled():
        return 0
    n = _install_moe_router() + _install_gdn_core()
    tr = trace.get_tracer()
    print(
        f"[ISOEXEC-DEBUG] tracing ON: {n} region binding(s) wrapped, side={tr.side}, "
        f"sample=1/{tr.sample}, ladder={'on' if tr.ladder else 'off'} -> {tr.path}",
        flush=True,
    )
    return n
