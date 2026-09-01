"""Stdlib-only package entry point: ``python -m debug TRACE_A TRACE_B``.

Requires ``isoexec/`` on PYTHONPATH; imports only this package's lazy ``__init__`` and ``compare``,
never the ``isoexec`` package, whose ``__init__`` needs torch/TransformerEngine.
"""

from .compare import main

if __name__ == "__main__":
    raise SystemExit(main())
