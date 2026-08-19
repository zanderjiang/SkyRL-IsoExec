"""Op / impl registration: the registry the composition manifest resolves against.

An op's identity is its rounding schedule, so this makes "the same function at every site" a
declared, machine-checkable object. Sites and hazards come from controlled vocabularies, an impl's
rounding schedule is split into a machine-assertable half (which is also the set of pins a manifest
may hand it) and a documentary half, and ``subsumes`` tells the adapter which sub-ops a fused impl
absorbs. The manifest has no layer dimension: an op is assumed to resolve identically at every layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

# The common site vocabulary. An op declares only the sites it has; absence of a site means the op
# has no such site, never a default. Open set: ``register_site`` extends it.
_BASE_SITES = frozenset(
    {
        "trainer_fwd",  # checkpoint forward (no-grad) plus its grad-enabled recompute
        "trainer_score",  # no_grad scoring forward
        "engine_prefill",  # eager, resumable
        "engine_decode",  # CUDA-graph-capturable: host-free, shape-static, address-stable
    }
)

# The hazard vocabulary. A test that declares one of these must prove it actually fired, or the
# pass is vacuous. A floor, not a ceiling.
HAZARDS = frozenset(
    {
        "null_lanes",  # NULL / padded lanes (graph-replay inertness)
        "t_zero",  # T=0 sequence
        "non_contiguous",  # non-contiguous inputs
        "e_mismatch",  # E != expected (expert count)
        "profiling_shapes",  # the shapes vLLM profiles engine init with
        "tie_boundaries",  # top-k ties
        "subnormals",  # subnormal inputs (FTZ paths)
        "signed_zero",  # +0.0 vs -0.0 (Triton's -x -> 0.0 - x yields +0.0)
    }
)


class RegistryError(ValueError):
    """Raised on any registry-vocabulary violation: unknown site, unknown hazard, duplicate
    op/impl, or an impl whose declared sites are not a subset of the op's sites."""


# Contract route vocabulary, kept literal so this module stays standalone.
_ROUTES = frozenset({"reference_preserving", "composition_defining", "protected", "canonical"})


class _PerModel:
    """Sentinel for a ``machine_assertable`` key that the impl declares but whose value is a model
    fact supplied by the manifest pin.

    Not the same as omitting the key: an omitted key means the impl knows nothing about that pin,
    and ``Manifest.validate_pins`` refuses a pin the impl never declared.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # appears verbatim in the refusal message
        return "PER_MODEL"


PER_MODEL = _PerModel()


class OneOf:
    """A ``machine_assertable`` value constrained to an enumerated set.

    Between ``PER_MODEL`` (any value) and a literal (exactly one): the impl is reachable under
    several pinned values and no others, so a manifest pinning anything else names a function that
    will never install.
    """

    __slots__ = ("values",)

    def __init__(self, *values) -> None:
        self.values = tuple(values)

    def __contains__(self, value) -> bool:
        return any(_pin_values_equal(v, value) for v in self.values)

    def __repr__(self) -> str:
        return f"OneOf{self.values!r}"


def _pin_values_equal(a, b) -> bool:
    """Equality for a declared schedule value vs a manifest pin.

    Sequence-insensitive, because a manifest round-trips through JSON and a declared tuple arrives
    as a list, and bool-strict, so a pin of ``1`` against a declared ``True`` is a mismatch.
    """
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_pin_values_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_pin_values_equal(a[k], b[k]) for k in a)
    return a == b


@dataclass(frozen=True)
class RoundingSchedule:
    """An impl's rounding schedule as data, split by checkability.

    ``machine_assertable`` holds structural facts a debug run can assert at the op boundary
    (boundary dtypes, autotune pin index, block sizes, constexpr values); each value is a literal,
    ``PER_MODEL``, or ``OneOf(...)``, and the declared keys are exactly the pins the impl accepts.
    ``documentary`` is free text that still moves bits -- reduction orders, formula variants. A
    change to either half is a new impl version, and therefore a new composition hash.
    """

    machine_assertable: Dict[str, object] = field(default_factory=dict)
    documentary: str = ""

    def check_pin(self, key: str, value) -> Optional[str]:
        """Judge one manifest pin against this schedule; ``None`` means consistent.

        Otherwise the reason, which the caller prefixes with the (op, site) and the model. An
        undeclared key is refused because a pin the impl never declared hashes and cross-matches
        while constraining nothing; a contradicted value means the manifest names a function the
        impl does not implement.
        """
        if key not in self.machine_assertable:
            return (
                "the impl's machine_assertable rounding schedule declares no such key "
                f"(declared: {sorted(str(k) for k in self.machine_assertable)}). A pin the impl "
                "never declared constrains nothing -- declare it (a literal, PER_MODEL, or "
                "OneOf(...)) or drop the pin."
            )
        declared = self.machine_assertable[key]
        if declared is PER_MODEL:
            return None
        if isinstance(declared, OneOf):
            return None if value in declared else f"the impl declares {key}={declared!r}"
        if not _pin_values_equal(declared, value):
            return f"the impl declares {key}={declared!r}"
        return None


@dataclass(frozen=True)
class StateInvalidation:
    """The rebind contract for a stateful op.

    ``condition`` describes what invalidates the state; ``hook(state, ctx) -> bool`` is the
    host-only, capture-safe check the adapter runs every forward, returning True while the state is
    still valid.
    """

    condition: str
    hook: Optional[Callable[..., bool]] = None


@dataclass(frozen=True)
class ImplSpec:
    """A single implementation of an op, at a pinned version.

    ``version`` is bumped on any rounding-schedule change. ``supported_archs`` is matched against
    the manifest's arch. ``capabilities`` are what the impl permits: ops declare capabilities,
    models declare choices, the adapter enforces compatibility. ``subsumes`` names the ops this
    impl absorbs, which is what lets the adapter route them to passthrough. ``hazards`` is the floor
    of hazards this impl's parity tests must exercise.
    """

    impl_id: str
    version: int
    supported_archs: frozenset
    rounding: RoundingSchedule = field(default_factory=RoundingSchedule)
    capabilities: Dict[str, object] = field(default_factory=dict)
    subsumes: Sequence[str] = ()
    state_invalidation: Optional[StateInvalidation] = None
    hazards: Sequence[str] = ()
    # Contract route class: "protected" (owned/pinned provider) unless declared otherwise.
    route: str = "protected"

    def __post_init__(self) -> None:
        if self.route not in _ROUTES:
            raise RegistryError(f"impl {self.impl_id!r} declares unknown route {self.route!r}")
        unknown = set(self.hazards) - HAZARDS
        if unknown:
            raise RegistryError(
                f"impl {self.impl_id!r} declares unknown hazard(s) {sorted(unknown)}; "
                f"the hazard vocabulary is {sorted(HAZARDS)}"
            )
        if not self.supported_archs:
            raise RegistryError(f"impl {self.impl_id!r} must declare at least one supported arch")


@dataclass
class OpSpec:
    """A mathematical function with a pinned rounding schedule, packaged as one op.

    ``sites`` are the sites this op has; declaring one here is what licenses a manifest entry for
    (name, site), and an unlisted site means "no such site". An op whose sites do not all resolve
    to one impl must carry an equivalence proof, which is enforced downstream of the registry.
    """

    name: str
    sites: List[str]
    impls: Dict[str, ImplSpec] = field(default_factory=dict)
    # Documents that this op is assumed to resolve identically at every layer.
    per_layer_uniform: bool = True

    def add_impl(self, impl: ImplSpec) -> "OpSpec":
        if impl.impl_id in self.impls:
            raise RegistryError(f"op {self.name!r} already has impl {impl.impl_id!r}")
        self.impls[impl.impl_id] = impl
        return self


class Registry:
    """The site vocabulary and the op/impl table. Both runtimes build the same registry -- it is
    code, delivered in the package -- and the manifest then selects from it."""

    def __init__(self) -> None:
        self._sites = set(_BASE_SITES)
        self._ops: Dict[str, OpSpec] = {}

    @property
    def sites(self) -> frozenset:
        return frozenset(self._sites)

    def register_site(self, name: str) -> None:
        """Extend the site vocabulary. Idempotent; names must be non-empty identifiers."""
        if not name or not name.replace("_", "").isalnum():
            raise RegistryError(f"invalid site name {name!r}: must be a non-empty identifier")
        self._sites.add(name)

    def _validate_sites(self, sites: Sequence[str], who: str) -> None:
        unknown = set(sites) - self._sites
        if unknown:
            raise RegistryError(
                f"{who} declares unknown site(s) {sorted(unknown)}; register them first via "
                f"register_site(). Known sites: {sorted(self._sites)}"
            )

    def register_op(self, op: OpSpec) -> OpSpec:
        if op.name in self._ops:
            raise RegistryError(f"op {op.name!r} already registered")
        self._validate_sites(op.sites, f"op {op.name!r}")
        self._ops[op.name] = op
        return op

    def get_op(self, name: str) -> OpSpec:
        if name not in self._ops:
            raise RegistryError(f"unknown op {name!r}")
        return self._ops[name]

    def has_op(self, name: str) -> bool:
        return name in self._ops

    @property
    def ops(self) -> Dict[str, OpSpec]:
        return dict(self._ops)

    def installed_keys(self) -> frozenset:
        """The (op, site) pairs this registry declares: the universe a manifest must name exactly,
        since every installed key carries an explicit entry and absence never means "default"."""
        return frozenset((op.name, site) for op in self._ops.values() for site in op.sites)

    def assert_subsumption_closed(self) -> None:
        """Every op named in an impl's ``subsumes`` must itself be registered, or the adapter cannot
        derive its passthrough."""
        for op in self._ops.values():
            for impl in op.impls.values():
                for sub in impl.subsumes:
                    if sub not in self._ops:
                        raise RegistryError(
                            f"impl {op.name}:{impl.impl_id} subsumes unregistered op {sub!r}; "
                            f"the adapter cannot derive its passthrough"
                        )
