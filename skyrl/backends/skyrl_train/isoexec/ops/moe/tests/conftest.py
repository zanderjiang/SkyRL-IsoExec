"""Pytest bootstrap that keeps the ops/moe CPU tests runnable without megatron.

Same mechanism, same reason as ``core/tests/conftest.py``: importing
``skyrl.backends.skyrl_train.isoexec`` runs ``install_no_te_guard()``, which imports megatron --
absent on a CPU box -- and pytest imports every parent package while resolving this directory's
dotted module names. A lightweight stub for that one parent package, installed before the climb,
lets ``...isoexec.ops.moe.<module>`` resolve to the real files while the heavy ``__init__`` never
runs. It exists only in the test process and changes no file and no production behaviour.

If the real package imports fine (a GPU run under --extra megatron) the stub is not installed, so
these tests run against exactly the production import graph there.
"""

import importlib.machinery
import os
import sys
import types

_ISOEXEC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ISOEXEC_DIR not in sys.path:
    sys.path.insert(0, _ISOEXEC_DIR)

_PARENT = "skyrl.backends.skyrl_train.isoexec"
if _PARENT not in sys.modules:
    try:
        import importlib.util

        _real_ok = importlib.util.find_spec(_PARENT) is not None
        if _real_ok:
            import megatron.bridge  # noqa: F401  -- probe the actual dependency
    except Exception:
        _real_ok = False

    if not _real_ok:
        _stub = types.ModuleType(_PARENT)
        _stub.__path__ = [_ISOEXEC_DIR]
        _stub.__spec__ = importlib.machinery.ModuleSpec(_PARENT, loader=None, is_package=True)
        _stub.__spec__.submodule_search_locations = [_ISOEXEC_DIR]
        sys.modules[_PARENT] = _stub
