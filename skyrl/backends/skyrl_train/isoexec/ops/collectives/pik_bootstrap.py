"""Make the vendored ``pik`` package importable under its canonical top-level name.

pik's runtime code generator emits Triton kernels that import from the hard-coded top-level name ``pik``, so
the vendored package is registered in ``sys.modules`` as ``pik`` -- loaded exactly once, via an importlib spec
whose submodule search path is the vendored directory. Importing it under its long dotted name instead would
create a second copy of every submodule and split pik's global autotune-cache / workspace registries, so
callers must go through :func:`ensure_pik`.
"""

from __future__ import annotations

import importlib.util
import logging
import pathlib
import sys
import threading

logger = logging.getLogger(__name__)

_PIK_DIR = pathlib.Path(__file__).parent / "pik"
_DOTTED_NAME = "skyrl.backends.skyrl_train.isoexec.ops.collectives.pik"
_LOCK = threading.Lock()


def _alias_loaded_children(source_prefix: str, target_prefix: str) -> None:
    """Alias already-loaded vendored children without executing their files a second time."""
    for name, child in list(sys.modules.items()):
        if name == source_prefix or name.startswith(source_prefix + "."):
            alias = target_prefix + name[len(source_prefix) :]
            existing = sys.modules.get(alias)
            if existing is not None and existing is not child:
                raise ImportError(f"PIK module identity split: {name!r} and {alias!r} resolve to different objects")
            sys.modules[alias] = child


def ensure_pik():
    """Import (once) and return the vendored pik package, registered as top-level ``pik``.

    Idempotent and thread-safe. If a *different* ``pik`` is already importable (e.g. a real
    pip install of the upstream repo), we defer to it -- it is the same library -- rather than
    shadow it.
    """
    mod = sys.modules.get("pik")
    if mod is not None:
        # Called from the MoE combine hot path: a root identity check avoids walking sys.modules every call.
        dotted = sys.modules.get(_DOTTED_NAME)
        if dotted is mod:
            return mod
        _alias_loaded_children("pik", _DOTTED_NAME)
        return mod
    with _LOCK:
        mod = sys.modules.get("pik")
        if mod is not None:
            if sys.modules.get(_DOTTED_NAME) is not mod:
                _alias_loaded_children("pik", _DOTTED_NAME)
            return mod
        # Adopt a package already imported under the long dotted name rather than building a second PIK registry.
        dotted = sys.modules.get(_DOTTED_NAME)
        if dotted is not None:
            sys.modules["pik"] = dotted
            _alias_loaded_children(_DOTTED_NAME, "pik")
            logger.info("[isoexec] adopted preloaded vendored PIK as canonical top-level 'pik'")
            return dotted
        init = _PIK_DIR / "__init__.py"
        if not init.exists():
            raise ImportError(f"vendored pik package not found at {_PIK_DIR}")
        # submodule_search_locations makes `pik` a package whose children resolve against the vendored dir.
        spec = importlib.util.spec_from_file_location("pik", init, submodule_search_locations=[str(_PIK_DIR)])
        mod = importlib.util.module_from_spec(spec)
        # Register before exec so pik's own relative imports see a single, consistent module object mid-import.
        sys.modules["pik"] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            sys.modules.pop("pik", None)
            raise
        _alias_loaded_children("pik", _DOTTED_NAME)
        logger.info("[isoexec] vendored pik registered as top-level 'pik' from %s", _PIK_DIR)
        return mod


def pik_arch_selfcheck(verbose: bool = True) -> bool:
    """Check on this GPU that no GEMM tiling knob moves the bits of a non-split-K K-reduction.

    Returns True on PASS; the library's invariance guarantee does not hold if this fails.
    """
    ensure_pik()
    from pik.arch import _verify, current  # type: ignore

    ok = _verify()
    if verbose:
        print(f"[ISOEXEC-PIK] arch self-check on {current()}: {'PASS' if ok else 'FAIL'}", flush=True)
    return ok
