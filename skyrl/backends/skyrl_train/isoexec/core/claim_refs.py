"""Resolve the refs a claim names to real in-tree code.

Refs are containment-checked (a ``../`` path reaches any file on the host) and symbols are
resolved by parsing, not by regex, which would also match text inside docstrings and comments.
"""

from __future__ import annotations

import ast
import pathlib

ISOEXEC_DIR = pathlib.Path(__file__).resolve().parents[1]

# Where a gate pointer may resolve: the colocated op gates. Read lazily, once per process.
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

    Accepts a repo-relative path to the gate file, or ``"<gate name>: <what it shows>"``.
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
