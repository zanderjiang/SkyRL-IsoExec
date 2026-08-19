"""In-process guard that keeps megatron-bridge importable when TransformerEngine is absent.

Three megatron-bridge modules (peft/lora_layers, peft/lora, diffusion/models/wan/utils) hard-import
TransformerEngine at module load with no try/except, so ``from megatron.bridge import AutoBridge``
crashes when TE is not installed. This guard rewrites those three imports in place, atomically and
idempotently, and deliberately does NOT register a stub ``transformer_engine`` module -- a stub would
flip megatron-core/-bridge HAVE_TE checks to True and trigger deeper TE imports that then fail. Call
``install_no_te_guard()`` before the first ``import megatron.bridge``; it is a no-op when TE is
importable.
"""

import importlib.util
import os

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
    # Resolve the path WITHOUT importing megatron.bridge (its __init__ is what crashes).
    spec = importlib.util.find_spec("megatron.bridge")
    if spec is None or not spec.origin:
        return None
    return os.path.dirname(spec.origin)


def install_no_te_guard() -> bool:
    global _INSTALLED
    if _INSTALLED:
        return True
    # Gate on the real precondition -- TE genuinely absent -- not on SKYRL_ISOEXEC_LOCAL_SPEC, which
    # is not reliably forwarded to Ray actors: the actor builds its own uv env and imports
    # megatron.bridge at worker module load, before the config-driven env is applied.
    try:
        import transformer_engine  # noqa: F401  -- real TE present, nothing to do

        return False
    except ImportError:
        pass

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
