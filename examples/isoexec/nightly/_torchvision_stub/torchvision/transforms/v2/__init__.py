from . import functional  # noqa: F401


class _Any:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        raise RuntimeError("torchvision stub: transform called")


def __getattr__(name):
    if name == "InterpolationMode":
        from ..functional import InterpolationMode

        return InterpolationMode
    return _Any
