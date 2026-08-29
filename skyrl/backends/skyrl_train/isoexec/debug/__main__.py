"""Stdlib-only package entry point: ``python -m debug TRACE_A TRACE_B``.

Reachable with ``isoexec/`` on PYTHONPATH, which is the offline comparison path: it imports this
package's lazy ``__init__`` and ``compare`` only, never the ``isoexec`` package whose ``__init__``
installs runtime guards that need torch/TransformerEngine. See ``compare.py`` for the full list
of supported invocations.
"""

from .compare import main

if __name__ == "__main__":
    raise SystemExit(main())
