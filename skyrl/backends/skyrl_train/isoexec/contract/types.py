"""Frozen dataclasses for the ExecutionContract. Data only — no I/O, no registry."""

from collections.abc import Mapping
from dataclasses import dataclass, field

ROUTES = frozenset({"reference_preserving", "composition_defining", "protected", "canonical"})
DISCHARGE_KINDS = frozenset({"bitwise_equal_to", "equivalence_proof", "neutrality_proof"})
COVERAGE_KINDS = frozenset({"enumerated_shapes", "symbolic_domain"})
HALVES = frozenset({"function", "deployment"})
TOPOLOGY_KINDS = frozenset({"pinned", "invariant"})

# The lifecycle events a StateClaim may name. An event is listed only if some consumer reads it:
# "weight_sync" is the enforce.WEIGHT_SYNC boundary and the flush-on-sync ordering assert,
# "sleep_wake" the wake/sleep ordering asserts (lifecycle/ordering.py) that the engine's state
# rebind rides on. An event nothing observes is prose the identity hashes, so it is not declarable.
STATE_EVENTS = frozenset({"weight_sync", "sleep_wake"})


def _freeze_map(m) -> tuple:
    # Normalize a Mapping or iterable of pairs to a sorted tuple of pairs.
    items = m.items() if isinstance(m, Mapping) else tuple(m)
    return tuple(sorted((str(k), v) for k, v in items))


@dataclass(frozen=True)
class ModelRef:
    family: str
    architectures: tuple[str, ...]
    profile_ref: str


@dataclass(frozen=True)
class Identities:
    semantic: str
    numerical_policy: str
    deployment: str


@dataclass(frozen=True)
class ExecutionCase:
    id: str
    runtime_role: str
    grad_mode: str
    state_mode: str
    shape_domain: str
    constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class BitPattern:
    # The only legal carrier for a float constant: hex bits, never decimal.
    bits: str
    dtype: str


@dataclass(frozen=True)
class ImplRef:
    id: str
    version: int
    arch: str


@dataclass(frozen=True)
class EquivalenceProof:
    kind: str
    ref: str


@dataclass(frozen=True)
class Coverage:
    kind: str
    description: str


@dataclass(frozen=True)
class CompositionEntry:
    region: tuple[str, ...]  # sorted group of logical op ids — one partition unit
    cases: tuple[str, ...]
    impl: ImplRef
    route: str
    constants: tuple = ()  # accepts a Mapping; stored as sorted (key, value) pairs
    artifact: str | None = None
    reference: ImplRef | None = None
    coverage: Coverage | None = None
    discharge: EquivalenceProof | None = None
    half: str = "function"

    def __post_init__(self):
        object.__setattr__(self, "constants", _freeze_map(self.constants))


@dataclass(frozen=True)
class TopologyClaim:
    axis: str
    kind: str
    degree: int | None = None
    collective_plan: str | None = None
    domain: tuple[int, ...] = ()
    proof: str | None = None


@dataclass(frozen=True)
class StateClaim:
    state_id: str
    invalidated_by: tuple[str, ...]
    replay_safe: bool
    ref: str


@dataclass(frozen=True)
class ToleranceClaim:
    case_pair: tuple[str, str]
    bounds: tuple = ()  # accepts a Mapping; values are decimal strings (thresholds)
    attributed_to: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "bounds", _freeze_map(self.bounds))


@dataclass(frozen=True)
class Claims:
    topology: tuple[TopologyClaim, ...] = ()
    state: tuple[StateClaim, ...] = ()
    tolerances: tuple[ToleranceClaim, ...] = ()


@dataclass(frozen=True)
class ExecutionContract:
    schema_version: str
    model: ModelRef
    identities: Identities
    cases: tuple[ExecutionCase, ...]
    composition: tuple[CompositionEntry, ...]
    claims: Claims = field(default_factory=Claims)
