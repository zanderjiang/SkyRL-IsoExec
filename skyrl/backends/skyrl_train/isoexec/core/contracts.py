"""Test-harness base: the gate primitives every op's colocated tests instantiate.

Three obligations every gate inherits from here. Comparisons are on integer views of the tensors,
never ``torch.equal``, which reports +0.0 == -0.0. A declared hazard must be proven to have fired,
or the pass is vacuous. And backward connectivity is checked live on the first backward as exact
set equality, since a per-group presence check passes a partially severed graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Sequence


class VacuousTestError(AssertionError):
    """Raised when a declared hazard did not actually fire, making the pass vacuous."""


class ConnectivityError(AssertionError):
    """Raised when the live first-backward grad set != the declared trainable set."""


# A float tensor is reinterpreted as the same-width signed integer so the comparison sees the raw
# bit pattern, including the sign bit of a zero.
def _int_view(t):
    import torch

    _MAP = {
        torch.float64: torch.int64,
        torch.float32: torch.int32,
        torch.float16: torch.int16,
        torch.bfloat16: torch.int16,
    }
    if t.dtype in _MAP:
        return t.contiguous().view(_MAP[t.dtype])
    # already an integer / bool dtype: its bits are its value
    return t.contiguous()


def bitwise_equal(a, b) -> bool:
    """True iff ``a`` and ``b`` are bit-for-bit identical.

    Compares integer reinterpretations, not ``torch.equal``, which returns True for +0.0 == -0.0
    even though the sign bit differs. A dtype or shape mismatch is not bitwise-equal.
    """
    import torch

    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("bitwise_equal compares torch.Tensors")
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(torch.equal(_int_view(a), _int_view(b)))


def assert_hazard_exercised(name: str, evidence) -> None:
    """Assert a declared hazard actually fired.

    ``evidence`` is a count, bool or other truthy witness the test computes from the actual data --
    subnormals in the reference, a NULL lane being present. A falsey witness raises
    ``VacuousTestError``, since any assertion downstream of a hazard that never happened is vacuous.
    """
    ok: bool
    if isinstance(evidence, bool):
        ok = evidence
    elif isinstance(evidence, (int, float)):
        ok = evidence > 0
    else:
        # tensors, sized containers, etc.: truthiness / non-empty
        try:
            ok = bool(len(evidence) > 0)
        except TypeError:
            ok = bool(evidence)
    if not ok:
        raise VacuousTestError(
            f"hazard {name!r} declared but NOT exercised (evidence={evidence!r}); the test is "
            f"vacuous. Supply inputs that make the hazard actually fire (Section 5)."
        )


@dataclass
class GateResult:
    """A recorded gate outcome.

    ``bitwise`` says whether the comparison was a bit-pattern one -- a numeric-only pass is
    degraded mode and must be labeled as such. ``hazards`` are the hazards this gate exercised.
    """

    name: str
    passed: bool
    bitwise: bool = False
    hazards: Sequence[str] = ()
    detail: Dict[str, object] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed


def check_hazard_coverage(opspec, exercised_names: Iterable[str]) -> None:
    """Fail if any hazard the op declared was not exercised by its gates.

    ``exercised_names`` is the union of hazards the op's gate battery actually fired. The declared
    list is a floor, so an unexercised hazard is a gate gap, not an acceptable state.
    """
    declared = set()
    impls = getattr(opspec, "impls", None)
    if impls:
        for impl in impls.values():
            declared |= set(getattr(impl, "hazards", ()))
    else:
        declared |= set(getattr(opspec, "hazards", ()))
    missing = declared - set(exercised_names)
    if missing:
        raise VacuousTestError(
            f"op {getattr(opspec, 'name', opspec)!r} declares hazard(s) {sorted(missing)} that "
            f"NO gate exercised; coverage is a floor, not an aspiration (Section 5)."
        )


def collect_grad_set(module) -> set:
    """The set of parameter names whose ``.grad is not None`` -- the live-connectivity witness."""
    return {name for name, p in module.named_parameters() if getattr(p, "grad", None) is not None}


def assert_grad_set_equality(module, expected_names: Iterable[str]) -> None:
    """Assert the live first-backward grad set equals the declared trainable set, exactly.

    A per-group presence check passes a partially severed graph, so only set equality catches it.
    ``ConnectivityError`` names both sides of the symmetric difference. Run this on the first
    backward of a real run: a process-global rebind is invisible to an offline toy test.
    """
    have = collect_grad_set(module)
    want = set(expected_names)
    severed = want - have  # declared trainable but received no grad -> the severed-backward bug
    surprise = have - want  # got a grad but not declared trainable -> a wiring surprise
    if severed or surprise:
        raise ConnectivityError(
            "first-backward grad set != declared trainable set:\n"
            f"  SEVERED (trainable, no grad): {sorted(severed)}\n"
            f"  SURPRISE (grad, not declared): {sorted(surprise)}\n"
            "count-vs-baseline is the degraded-mode minimum; exact set equality is the contract."
        )


def assert_grad_count(module, expected_count: int) -> None:
    """Degraded-mode minimum for when the trainable name set cannot be enumerated: assert the count
    of grad-receiving params against a recorded baseline. Prefer ``assert_grad_set_equality``."""
    have = len(collect_grad_set(module))
    if have != expected_count:
        raise ConnectivityError(
            f"first-backward grad count {have} != recorded baseline {expected_count} "
            f"(degraded-mode connectivity check; prefer assert_grad_set_equality)."
        )
