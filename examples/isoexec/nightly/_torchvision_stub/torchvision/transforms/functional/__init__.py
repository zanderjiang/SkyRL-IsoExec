from enum import Enum


class InterpolationMode(str, Enum):
    """Members used by transformers' import-time resampling map."""

    NEAREST = "nearest"
    NEAREST_EXACT = "nearest-exact"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"


def __getattr__(name):
    def _f(*a, **k):
        raise RuntimeError(f"torchvision stub: transforms.functional.{name}")

    return _f
