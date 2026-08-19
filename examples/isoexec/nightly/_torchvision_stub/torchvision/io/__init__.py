def __getattr__(name):
    def _f(*a, **k):
        raise RuntimeError(f"torchvision stub: io.{name}")

    return _f
