"""Build the ExecutionContract directly from a registry and per-(op, site) selections.

The one composition object: ``models/policy.build_selections`` emits the selections, this validates
them against the registry (unknown op / undeclared site / unregistered impl / contradicted pins all
refuse), then projects them into contract entries -- sites whose resolved selection is identical
merge into one entry, a region is the op plus the impls it subsumes, and float pins encode as fp64
bit patterns. A DEPLOYMENT selection is discharged by its neutrality proof; a FUNCTION op that
resolves to different impls across sites needs a declared ``bitwise_equal_to`` or
``equivalence_proof`` or it refuses.
"""

from __future__ import annotations

import dataclasses
import json
import struct
from collections.abc import Mapping
from typing import Dict, Optional

from ..contract import (
    BitPattern,
    Claims,
    CompositionEntry,
    EquivalenceProof,
    ExecutionCase,
    ExecutionContract,
    Identities,
    ImplRef,
    ModelRef,
    StateClaim,
    ToleranceClaim,
    TopologyClaim,
    compute_identities,
    validate_or_raise,
)
from .arch import ARCH, is_accelerator_arch
from .registry import Registry

# Per-selection classification vocabulary: FUNCTION halves are hashed, DEPLOYMENT halves are
# proven bitwise-neutral and logged but not hashed.
FUNCTION = "function"
DEPLOYMENT = "deployment"

CASES = (
    ExecutionCase(
        "engine_decode",
        "engine",
        "no_grad",
        "continued",
        "shape_static",
        ("cudagraph_capturable", "host_free", "address_stable"),
    ),
    ExecutionCase("engine_prefill", "engine", "no_grad", "resumable", "variable"),
    ExecutionCase("trainer_fwd", "trainer", "grad", "checkpoint_recompute", "variable", ("includes_grad_recompute",)),
    ExecutionCase("trainer_score", "trainer", "no_grad", "fresh", "variable"),
)


class ContractBuildError(ValueError):
    pass


class PinValidationError(ContractBuildError):
    """Raised when a selection's pinned constants disagree with the selected impl's declared
    ``machine_assertable`` rounding schedule (``validate_pins``)."""


@dataclasses.dataclass(frozen=True)
class _Sel:
    """Normalized selection value; identity is impl@version x arch."""

    impl_id: str
    version: int
    pinned_constants: dict
    classification: str
    neutrality_proof: Optional[str]


def _norm_selection(op: str, site: str, val) -> _Sel:
    # Accepts the derivation's Selection record (duck-typed) or a plain mapping.
    if isinstance(val, Mapping):
        sel = _Sel(
            impl_id=val["impl_id"],
            version=val.get("version", 1),
            pinned_constants=dict(val.get("pinned_constants") or {}),
            classification=val.get("classification", FUNCTION),
            neutrality_proof=val.get("neutrality_proof"),
        )
    else:
        sel = _Sel(
            impl_id=val.impl_id,
            version=val.version,
            pinned_constants=dict(val.pinned_constants or {}),
            classification=val.classification,
            neutrality_proof=val.neutrality_proof,
        )
    # Same refusals as the record's construction time, re-checked for mapping inputs.
    if sel.classification not in (FUNCTION, DEPLOYMENT):
        raise ContractBuildError(
            f"({op}, {site}): unknown classification {sel.classification!r}; "
            f"must be one of ['{DEPLOYMENT}', '{FUNCTION}']"
        )
    if sel.classification == DEPLOYMENT and not sel.neutrality_proof:
        raise ContractBuildError(
            f"({op}, {site}): deployment classification requires a recorded neutrality_proof (a run "
            "id / gate result pointer). A neutrality proof licenses exactly the entry it was "
            "measured on; refusing to classify without one."
        )
    if sel.classification == FUNCTION and sel.neutrality_proof:
        raise ContractBuildError(
            f"({op}, {site}): a function-half entry must not carry a neutrality_proof; either it "
            "is proven neutral (classify it deployment) or it is not (leave it function)."
        )
    return sel


def validate_pins(registry: Registry, selections: Dict, model: Optional[str] = None) -> None:
    """Refuse any pinned constant the selected impl does not declare, or contradicts.

    Every other check compares the composition against itself or against the other runtime, so none
    of them can see a composition that is internally consistent, delivered intact, identical on both
    sides -- and names the wrong function. An unchecked pin is exactly that: it hashes, it matches
    across runtimes, and it constrains nothing, so the gate goes green on the wrong model. The
    impl's schedule is the authority, since it is a claim about the code while the profile is a
    claim about the model. A failure means the profile or the declaration is wrong, never that the
    pin should be dropped or the schedule loosened.

    Raises ``PinValidationError`` naming every offending pin at once.
    """
    problems = []
    who = f"model {model!r}" if model else "selections"
    for (op, site), val in sorted(selections.items()):
        e = _norm_selection(op, site, val)
        if not registry.has_op(op):
            if e.pinned_constants:
                problems.append(f"({op}, {site}): op is not registered, so its pins cannot be checked")
            continue
        impl = registry.get_op(op).impls.get(e.impl_id)
        if impl is None:
            if e.pinned_constants:
                problems.append(f"({op}, {site}): impl {e.impl_id!r} is not registered on the op")
            continue
        for key in sorted(e.pinned_constants, key=str):
            value = e.pinned_constants[key]
            reason = impl.rounding.check_pin(key, value)
            if reason:
                problems.append(f"({op}, {site}) impl {e.impl_id}@v{impl.version}: pin {key}={value!r} -- {reason}")
    if problems:
        raise PinValidationError(
            f"{who}: pinned constants disagree with the selected impls' declared rounding "
            f"schedules:\n  " + "\n  ".join(problems) + "\n\n"
            "A pinned constant is a claim about the FUNCTION the composition names. An unchecked "
            "claim hashes identically on both runtimes, so the IsoExec gate stays GREEN while "
            "the composition names the wrong function -- the failure mode this check exists "
            "for. Fix the profile (it describes a model this impl does not implement) or fix "
            "the impl's machine_assertable schedule (it is stale); do not delete the pin and "
            "do not weaken the schedule to make this pass."
        )


_validate_pins = validate_pins  # keep callable under the same-named keyword below


def _encode_pin(op: str, key: str, value):
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return BitPattern("0x%016x" % struct.unpack("<Q", struct.pack("<d", value))[0], "fp64")
    raise ContractBuildError(f"({op}) pin {key!r}: non-scalar pin values are not projectable yet")


def _impl_spec(registry: Registry, op: str, impl_id: str):
    if registry.has_op(op):
        return registry.get_op(op).impls.get(impl_id)
    return None


def _group_discharge(registry: Registry, op: str, entries: dict) -> Optional[EquivalenceProof]:
    """The claim discharging an op whose sites resolve to more than one impl."""
    impl_ids = sorted({e.impl_id for e in entries.values()})
    if len(impl_ids) <= 1:
        return None
    for impl_id in impl_ids:
        spec = _impl_spec(registry, op, impl_id)
        if spec and spec.capabilities.get("bitwise_equal_to") in impl_ids:
            return EquivalenceProof("bitwise_equal_to", spec.capabilities["bitwise_equal_to"])
    for impl_id in impl_ids:
        spec = _impl_spec(registry, op, impl_id)
        if spec and spec.capabilities.get("equivalence_proof"):
            return EquivalenceProof("equivalence_proof", spec.capabilities["equivalence_proof"])
    proofs = {e.neutrality_proof for e in entries.values() if e.classification == DEPLOYMENT}
    if len(proofs) == 1:
        return EquivalenceProof("neutrality_proof", proofs.pop())
    raise ContractBuildError(
        f"op {op!r} resolves to {impl_ids} across sites but no selected impl declares "
        f"bitwise_equal_to / equivalence_proof and no single neutrality proof licenses the group"
    )


def derive_topology_claims(topology) -> tuple:
    """Project the profile's declared ``TopologyAxisFact``s into contract ``TopologyClaim``s.

    Pure projection, no invention: every degree/domain/proof is the profile's recorded fact
    (``models/profile.TopologyAxisFact`` already refuses an invariant axis with no proven domain or
    no proof ref). Sorted by axis so declaration order never moves the hash.
    """
    claims = []
    for t in sorted(topology or (), key=lambda t: t.axis):
        claims.append(
            TopologyClaim(
                axis=t.axis,
                kind=t.kind,
                degree=t.degree,
                collective_plan=t.collective_plan,
                domain=tuple(t.domain),
                proof=t.proof,
            )
        )
    return tuple(claims)


def derive_state_claims(states) -> tuple:
    """Profile ``StateFact``s -> contract ``StateClaim``s, verbatim and sorted by state_id."""
    return tuple(
        StateClaim(
            state_id=s.state_id,
            invalidated_by=tuple(s.invalidated_by),
            replay_safe=bool(s.replay_safe),
            ref=s.ref,
        )
        for s in sorted(states or (), key=lambda s: s.state_id)
    )


def derive_tolerance_claims(tolerances) -> tuple:
    """Profile ``ToleranceFact``s -> contract ``ToleranceClaim``s, verbatim and sorted by pair."""
    return tuple(
        ToleranceClaim(
            case_pair=tuple(t.case_pair),
            bounds=tuple(t.bounds),
            attributed_to=tuple(t.attributed_to),
        )
        for t in sorted(tolerances or (), key=lambda t: t.case_pair)
    )


def build_execution_contract(
    registry: Registry,
    selections: Dict,
    arch: str = ARCH,
    model: Optional[str] = None,
    profile=None,
    topology=(),
    states=(),
    tolerances=(),
    validate_pins: bool = True,
    allow_non_accelerator_arch: bool = False,
) -> ExecutionContract:
    """Derive the contract from ``{(op, site) -> selection}``. The single build path.

    Every selection key must name a site the registry's op declares. ``validate_pins=True`` is the
    derivation-time gate on pins and the only supported production setting -- turning it off
    re-opens the hole where a composition hashes and cross-matches while naming the wrong function.
    Completeness against what was actually installed is a separate, adapter-time check
    (``contract_delivery.validate_contract_against_installed``).
    """
    if not allow_non_accelerator_arch and not is_accelerator_arch(arch):
        raise ContractBuildError(
            f"refusing to build a contract whose arch is {arch!r}: that is the non-accelerator "
            f"sentinel (core/arch.NON_ACCELERATOR_ARCH), not a real accelerator key. Folding it "
            f"into the contract identity silently mis-keys every gate signature away from the "
            f"real accelerator table (e.g. sm90), and the file/env hash cross-check would still "
            f"PASS -- the exact split-brain this contract exists to prevent. This usually means "
            f"the contract was built in a process with no visible CUDA device (a launcher/driver); "
            f"build it where the accelerator is visible, or pass an explicit arch. CPU-only tests "
            f"that intentionally build a sentinel contract must pass allow_non_accelerator_arch=True."
        )

    normalized: Dict = {}
    for key, val in selections.items():
        op, site = key
        if not registry.has_op(op):
            raise ContractBuildError(f"selection names unknown op {op!r} (not in registry)")
        op_spec = registry.get_op(op)
        if site not in op_spec.sites:
            raise ContractBuildError(
                f"selection names site {site!r} for op {op!r}, but that op declares only "
                f"{sorted(op_spec.sites)} (absence means 'no such site')"
            )
        sel = _norm_selection(op, site, val)
        if sel.impl_id not in op_spec.impls:
            raise ContractBuildError(
                f"selection for ({op!r}, {site!r}) names impl {sel.impl_id!r} not registered "
                f"on the op (known: {sorted(op_spec.impls)})"
            )
        normalized[(op, site)] = sel
    if validate_pins:
        _validate_pins(registry, normalized, model=model)

    by_op: Dict[str, Dict[str, _Sel]] = {}
    for (op, site), e in normalized.items():
        by_op.setdefault(op, {})[site] = e

    composition = []
    for op in sorted(by_op):
        sites = by_op[op]
        group_proof = _group_discharge(registry, op, sites)
        # Merge sites whose resolved selection is identical into one CompositionEntry.
        buckets: Dict[str, list] = {}
        for site, e in sites.items():
            k = json.dumps(
                [e.impl_id, e.version, e.pinned_constants, e.classification, e.neutrality_proof],
                sort_keys=True,
                default=str,
            )
            buckets.setdefault(k, []).append(site)
        for k in sorted(buckets):
            merged_sites = tuple(sorted(buckets[k]))
            e = sites[merged_sites[0]]
            spec = _impl_spec(registry, op, e.impl_id)
            region = tuple(sorted((op,) + tuple(spec.subsumes if spec else ())))
            deployment = e.classification == DEPLOYMENT
            discharge = EquivalenceProof("neutrality_proof", e.neutrality_proof) if deployment else group_proof
            composition.append(
                CompositionEntry(
                    region=region,
                    cases=merged_sites,
                    impl=ImplRef(e.impl_id, e.version, arch),
                    route=spec.route if spec else "protected",
                    constants={key: _encode_pin(op, key, v) for key, v in e.pinned_constants.items()},
                    discharge=discharge,
                    half="deployment" if deployment else "function",
                )
            )
    composition.sort(key=lambda c: (c.region, c.cases, c.impl.id))

    case_ids = {site for _, site in normalized}
    cases = tuple(c for c in CASES if c.id in case_ids)
    archs = tuple(getattr(profile, "architectures", ()) or ())
    contract = ExecutionContract(
        schema_version="1",
        model=ModelRef(model or "", archs, f"models/{model or 'unknown'}"),
        identities=Identities("", "", ""),
        cases=cases,
        composition=tuple(composition),
        claims=Claims(
            topology=derive_topology_claims(topology),
            state=derive_state_claims(states),
            tolerances=derive_tolerance_claims(tolerances),
        ),
    )
    contract = dataclasses.replace(contract, identities=compute_identities(contract))
    required = {
        c.id: frozenset(op for e in contract.composition if c.id in e.cases for op in e.region) for c in contract.cases
    }
    validate_or_raise(contract, required_ops=required)
    return contract
