from .v2 import _Any  # noqa: F401
from . import functional, v2  # noqa: F401,E402
from .functional import InterpolationMode  # noqa: F401,E402


def __getattr__(name):
    return _Any
