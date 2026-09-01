"""Resolve the refs a claim names to real in-tree code.

A ref is the only thing standing between a declared claim and prose: a ``StateClaim``'s
``path::symbol`` hook and a proof pointer are what make the claim checkable at all. Both
resolutions live here so the build path, the runtime claim checkers and CI answer the same way.
Two rules the obvious implementations get wrong: a ref is containment-checked, because a
``../``-prefixed path reaches any file on the host and attesting one attests nothing about this
package; and the symbol check PARSES the file, because ``^def hook(`` also matches a line inside a
docstring, a comment or a string literal.
"""

from __future__ import annotations

import ast
import pathlib

ISOEXEC_DIR = pathlib.Path(__file__).resolve().parents[1]

# Where a gate pointer may resolve: the colocated op gates. Read lazily and once per process.
_GATE_DIRS = ("ops",)
_gate_sources: dict[str, str] | None = None


def _contained(path: str) -> pathlib.Path | None:
    """``path`` resolved inside the package, or None when it escapes (or cannot be resolved)."""
    try:
        resolved = (ISOEXEC_DIR / path).resolve()
        resolved.relative_to(ISOEXEC_DIR)
    except (ValueError, OSError):
        return None
    return resolved


def _defines(source: str, symbol: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol
        for node in tree.body
    )


def hook_ref_problem(ref: str) -> str | None:
    """Why ``path::symbol`` does not name real in-tree code; None when it does."""
    path, _, symbol = ref.partition("::")
    f = _contained(path)
    if f is None:
        return f"ref path {path!r} escapes the isoexec package, so it names no code this repo ships"
    if not f.is_file():
        return f"ref file {path!r} missing"
    if not _defines(f.read_text(errors="replace"), symbol):
        return f"hook {symbol!r} not defined in {path}"
    return None


def _gates() -> dict[str, str]:
    global _gate_sources
    if _gate_sources is None:
        _gate_sources = {}
        for top in _GATE_DIRS:
            for p in sorted((ISOEXEC_DIR / top).glob("*/tests/*.py")):
                _gate_sources[str(p.relative_to(ISOEXEC_DIR))] = p.read_text(errors="replace")
    return _gate_sources


def proof_ref_problem(ref: str) -> str | None:
    """Why a proof pointer resolves to no gate; None when it does.

    Accepts the two forms the tree uses: a repo-relative path to the gate file that measured the
    claim, and a gate/test name followed by the property it proved (``"<name>: <what it shows>"``).
    The name half must appear in a colocated gate's filename or source -- a pointer nobody can
    follow is the claim's evidence being asserted rather than recorded.
    """
    name = ref.strip().split(":")[0].split()[0] if ref.strip() else ""
    if not name:
        return "empty proof ref"
    f = _contained(name)
    if f is not None and f.is_file():
        return None
    if any(name in path or name in src for path, src in _gates().items()):
        return None
    return f"proof ref {ref!r} resolves to no gate: {name!r} names no file and appears in no ops/*/tests source"
