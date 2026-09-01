"""Guard that keeps megatron-bridge importable when TransformerEngine is absent.

Rewrites three hard TE imports in place. It must NOT register a stub ``transformer_engine`` module:
that would flip HAVE_TE checks to True and trigger deeper TE imports that then fail.
"""

import importlib.util
import logging
import os

logger = logging.getLogger(__name__)

_INSTALLED = False
MARKER = "# [isoexec-no-te-guard]"

_TE_PYTORCH_GUARD = (
    "try:\n"
    "    import transformer_engine.pytorch as te  " + MARKER + "\n"
    "except ModuleNotFoundError:  " + MARKER + "\n"
    "    import types as _ix_types  " + MARKER + "\n"
    "    _ix_ph = type('_TEUnavailable', (), {})  " + MARKER + "\n"
    "    te = _ix_types.SimpleNamespace(Linear=_ix_ph, LayerNormLinear=_ix_ph, "
    "ops=_ix_types.SimpleNamespace(Sequential=_ix_ph))  " + MARKER + "\n"
)
_TEX_GUARD = (
    "try:\n"
    "    import transformer_engine_torch as tex  " + MARKER + "\n"
    "except ModuleNotFoundError:  " + MARKER + "\n"
    "    tex = None  " + MARKER + "\n"
)
_PATCHES = (
    ("peft/lora_layers.py", "import transformer_engine.pytorch as te", _TE_PYTORCH_GUARD),
    ("peft/lora.py", "import transformer_engine.pytorch as te", _TE_PYTORCH_GUARD),
    ("diffusion/models/wan/utils.py", "import transformer_engine_torch as tex", _TEX_GUARD),
)


def _bridge_root():
    # Resolve the path WITHOUT importing megatron.bridge (its __init__ is what crashes). find_spec
    # raises rather than returning None when the parent package is absent.
    try:
        spec = importlib.util.find_spec("megatron.bridge")
    except (ImportError, AttributeError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    return os.path.dirname(spec.origin)


def install_no_te_guard() -> bool:
    global _INSTALLED
    if _INSTALLED:
        return True
    # Gate on the real precondition (TE genuinely absent), not on SKYRL_ISOEXEC_LOCAL_SPEC, which is
    # not reliably forwarded to Ray actors before they import megatron.bridge.
    try:
        import transformer_engine  # noqa: F401  -- real TE present, nothing to do

        return False
    except ImportError:
        pass
    except OSError as e:
        # A partial TE install raises OSError (missing .so) rather than ImportError; a TE that
        # cannot load counts as absent, but log loudly since on a trainer it means a broken install.
        logger.error(
            "[isoexec] TransformerEngine is installed but fails to load (%s: %s); treating it as "
            "absent and installing the no-TE guard. Expected on a CPU-only comparator box; on a "
            "GPU trainer this is a broken TE install and the run will not use TE.",
            type(e).__name__,
            e,
        )

    root = _bridge_root()
    if root is None:
        return False
    changed = 0
    for rel, needle, replacement in _PATCHES:
        path = os.path.join(root, rel)
        try:
            src = open(path).read()
        except OSError:
            continue
        if MARKER in src:
            continue
        line = needle + "\n"
        if line not in src:
            continue
        new_src = src.replace(line, replacement, 1)
        # Atomic write so a concurrent importer never sees a torn file.
        tmp = f"{path}.zk.{os.getpid()}.tmp"
        with open(tmp, "w") as fh:
            fh.write(new_src)
        os.replace(tmp, path)
        changed += 1
    _INSTALLED = True
    if changed:
        print(f"[isoexec] no-TE guard: patched {changed} megatron-bridge module(s) for genuine TE absence", flush=True)
    return True
