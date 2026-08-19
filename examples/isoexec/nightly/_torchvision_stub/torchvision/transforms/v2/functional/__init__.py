from torchvision.transforms.functional import InterpolationMode  # noqa: F401


def __getattr__(name):
    def _f(*a, **k):
        raise RuntimeError(f"torchvision stub: transforms.v2.functional.{name}")

    return _f
