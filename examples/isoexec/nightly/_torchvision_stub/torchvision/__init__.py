"""Minimal torchvision import stub for the text-only IsoExec runtime.

vLLM's Qwen3.5 module imports its VL support chain unconditionally.  The
production environment has no torch-2.14-compatible torchvision wheel, and the
text-only DAPO path needs only the import-time surface below.
"""

__version__ = "0.0.0-isoexec-stub"

from . import io, ops, transforms  # noqa: F401,E402


def _make(name):
    raise RuntimeError(f"torchvision stub: {name} is not available")
