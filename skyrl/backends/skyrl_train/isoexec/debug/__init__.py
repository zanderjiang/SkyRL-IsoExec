"""Debug mode: per-region output tracing that localizes train/inference mismatch.

Enforcement mode proves agreement; debug mode explains disagreement. With
``SKYRL_ISOEXEC_DEBUG_TRACE`` set, region outputs are digested on-device (``thash``), recorded
per process (``trace``, hooks in ``install``), and compared offline (``compare``) into a map of
where divergence first arises and roughly how large it is. Entirely inert when the env is unset.
See ``INTEGRATION.md`` for the call sites and flag registration this package deliberately does
not perform itself.

Re-exports are lazy (PEP 562, matching the isoexec package) and this module imports nothing, so
the comparator CLI stays stdlib-only: it must run on a machine with no torch and no CUDA/TE. The
fully qualified ``python -m skyrl...debug.compare`` form cannot deliver that -- ``-m`` imports the
``isoexec`` package first and its ``__init__`` installs runtime guards -- so the supported offline
invocations run ``compare.py`` by path or put ``isoexec/`` on PYTHONPATH. See ``compare.py``.
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
