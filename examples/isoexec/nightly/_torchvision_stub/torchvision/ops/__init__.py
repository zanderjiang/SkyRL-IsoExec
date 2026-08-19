def __getattr__(name):
    def _f(*a, **k):
        raise RuntimeError(f"torchvision stub: ops.{name}")

    return _f
