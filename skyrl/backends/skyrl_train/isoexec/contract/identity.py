"""The three identity hashes and their projections."""

import hashlib

from .serialize import (
    _enc_state,
    _enc_tolerance,
    _enc_topology,
    canonical_bytes,
    encode_constants,
)
from .types import (
    CompositionEntry,
    EquivalenceProof,
    ExecutionContract,
    Identities,
    ImplRef,
)


def _impl(i: ImplRef | None):
    return None if i is None else {"id": i.id, "version": i.version, "arch": i.arch}


def _discharge(d: EquivalenceProof | None):
    return None if d is None else {"kind": d.kind, "ref": d.ref}


def _project_entry(e: CompositionEntry) -> dict:
    # Bit-relevant fields only; cases sorted so entry-internal order never matters.
    return {
        "region": list(e.region),
        "cases": sorted(e.cases),
        "impl": _impl(e.impl),
        "route": e.route,
        "constants": encode_constants(e.constants),
        "artifact": e.artifact,
        "reference": _impl(e.reference),
        "discharge": _discharge(e.discharge),
    }


def _sorted_projections(entries) -> list[dict]:
    return sorted(entries, key=canonical_bytes)


def function_half(contract: ExecutionContract) -> dict:
    """Everything that can move bits: function entries, topology claims, case ids."""
    return {
        "schema_version": contract.schema_version,
        "cases": sorted(c.id for c in contract.cases),
        "entries": _sorted_projections(_project_entry(e) for e in contract.composition if e.half == "function"),
        "topology": sorted((_enc_topology(t) for t in contract.claims.topology), key=canonical_bytes),
    }


def deployment_half(contract: ExecutionContract) -> dict:
    """Comparison-worthy facts proven not to move bits."""
    return {
        "entries": _sorted_projections(_project_entry(e) for e in contract.composition if e.half == "deployment"),
        "state": sorted((_enc_state(s) for s in contract.claims.state), key=canonical_bytes),
        "tolerances": sorted((_enc_tolerance(t) for t in contract.claims.tolerances), key=canonical_bytes),
    }


def semantic_inputs(contract: ExecutionContract) -> dict:
    """What the model means: vocabulary and case structure, independent of partitioning."""
    ops = sorted({op for e in contract.composition for op in e.region})
    return {
        "model": {
            "family": contract.model.family,
            "architectures": sorted(contract.model.architectures),
            "profile_ref": contract.model.profile_ref,
        },
        "logical_ops": ops,
        "cases": sorted(
            ({"id": c.id, "shape_domain": c.shape_domain} for c in contract.cases),
            key=canonical_bytes,
        ),
    }


def _sha256(plain) -> str:
    return hashlib.sha256(canonical_bytes(plain)).hexdigest()


def compute_identities(contract: ExecutionContract) -> Identities:
    """Recompute all three hashes from content; ignores the stored identities field."""
    return Identities(
        semantic=_sha256(semantic_inputs(contract)),
        numerical_policy=_sha256(function_half(contract)),
        deployment=_sha256(deployment_half(contract)),
    )
