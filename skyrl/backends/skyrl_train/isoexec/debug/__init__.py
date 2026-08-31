"""Debug mode: per-region output tracing that localizes train/inference mismatch.

Regions are digested on-device (``thash``), recorded per process (``trace``/``install``), compared
offline (``compare``). Inert unless ``SKYRL_ISOEXEC_DEBUG_TRACE`` is set; re-exports are lazy and
this module imports nothing, so the comparator CLI stays stdlib-only (no torch, no CUDA).
"""

_LAZY = {
    "install_debug_hooks": (".install", "install_debug_hooks"),
    "install_gdn_layer_hooks": (".install", "install_gdn_layer_hooks"),
    "install_layer_context_hooks": (".install", "install_layer_context_hooks"),
    "hooks_coverage": (".install", "coverage"),
    "digest_backend": (".thash", "digest_backend"),
    "preload_digest": (".thash", "preload"),
    "segment_axis": (".thash", "segment_axis"),
    "layer_context": (".trace", "layer_context"),
    "tensor_digest": (".thash", "tensor_digest"),
    "digest_ladder": (".thash", "digest_ladder"),
    "ladder_for": (".thash", "ladder_for"),
    "segment_digests": (".thash", "segment_digests"),
    "wrap_region": (".trace", "wrap_region"),
    "set_step": (".trace", "set_step"),
    "flush": (".trace", "flush"),
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib

        mod, attr = _LAZY[name]
        return getattr(importlib.import_module(mod, __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
