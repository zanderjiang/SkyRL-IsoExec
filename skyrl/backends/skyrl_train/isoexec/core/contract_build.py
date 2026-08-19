"""Project a frozen Manifest into an ExecutionContract.

Pure projection, no selection: sites whose resolved entry is identical merge into one entry, a
region is the op plus the impls it subsumes, and float pins encode as fp64 bit patterns. A
DEPLOYMENT entry is discharged by its neutrality proof; a FUNCTION op that resolves to different
impls across sites needs a declared ``bitwise_equal_to`` or ``equivalence_proof`` or it refuses.
"""

from __future__ import annotations

import dataclasses
import json
import struct
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
    compute_identities,
    validate_or_raise,
)
from .composition import DEPLOYMENT, Manifest
from .registry import Registry

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


def build_execution_contract(registry: Registry, manifest: Manifest, profile=None) -> ExecutionContract:
    """Derive the contract from a frozen manifest. Pure projection: no selection, no behavior."""
    if not manifest.frozen:
        raise ContractBuildError("refusing to project an unfrozen manifest")

    by_op: Dict[str, Dict[str, object]] = {}
    for (op, site), e in manifest.entries().items():
        by_op.setdefault(op, {})[site] = e

    composition = []
    for op in sorted(by_op):
        sites = by_op[op]
        group_proof = _group_discharge(registry, op, sites)
        # Merge sites whose resolved entry is identical into one CompositionEntry.
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
                    impl=ImplRef(e.impl_id, e.version, manifest.arch),
                    route=spec.route if spec else "protected",
                    constants={key: _encode_pin(op, key, v) for key, v in e.pinned_constants.items()},
                    discharge=discharge,
                    half="deployment" if deployment else "function",
                )
            )
    composition.sort(key=lambda c: (c.region, c.cases, c.impl.id))

    case_ids = {s for _, s in manifest.keys()}
    cases = tuple(c for c in CASES if c.id in case_ids)
    archs = tuple(getattr(profile, "architectures", ()) or ())
    contract = ExecutionContract(
        schema_version="1",
        model=ModelRef(manifest.model or "", archs, f"models/{manifest.model or 'unknown'}"),
        identities=Identities("", "", ""),
        cases=cases,
        composition=tuple(composition),
        claims=Claims(),
    )
    contract = dataclasses.replace(contract, identities=compute_identities(contract))
    required = {
        c.id: frozenset(op for e in contract.composition if c.id in e.cases for op in e.region) for c in contract.cases
    }
    validate_or_raise(contract, required_ops=required)
    return contract
