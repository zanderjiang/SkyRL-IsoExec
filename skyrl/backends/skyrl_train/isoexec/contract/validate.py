"""Structural invariants over a frozen contract. Build-time / CI checkpoint."""

import re
from collections.abc import Mapping

from .identity import compute_identities
from .serialize import SUPPORTED_SCHEMA_VERSIONS, SerializationError
from .types import (
    COVERAGE_KINDS,
    DISCHARGE_KINDS,
    HALVES,
    ROUTES,
    TOPOLOGY_KINDS,
    BitPattern,
    ExecutionContract,
)

_HEX = re.compile(r"^0x[0-9a-fA-F]+$")


class ValidationError(Exception):
    pass


def validate(contract: ExecutionContract, required_ops: Mapping[str, frozenset[str]] | None = None) -> list[str]:
    """Return human-readable violations; empty list means valid."""
    v: list[str] = []
    case_ids = {c.id for c in contract.cases}

    if contract.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        v.append(f"unsupported schema_version {contract.schema_version!r}")

    if len(case_ids) != len(contract.cases):
        v.append("duplicate case ids")

    seen_keys: set[tuple] = set()
    for e in contract.composition:
        name = f"entry {e.region}@{e.impl.id}"

        if not e.region:
            v.append(f"{name}: empty region")
        if tuple(sorted(e.region)) != e.region:
            v.append(f"{name}: region tuple must be sorted")
        if e.route not in ROUTES:
            v.append(f"{name}: unknown route {e.route!r}")
        if e.half not in HALVES:
            v.append(f"{name}: unknown half {e.half!r}")
        if e.discharge is not None and e.discharge.kind not in DISCHARGE_KINDS:
            v.append(f"{name}: unknown discharge kind {e.discharge.kind!r}")
        if e.coverage is not None and e.coverage.kind not in COVERAGE_KINDS:
            v.append(f"{name}: unknown coverage kind {e.coverage.kind!r}")

        for cid in e.cases:
            if cid not in case_ids:
                v.append(f"{name}: unknown case {cid!r}")
            key = (e.region, cid)
            if key in seen_keys:
                v.append(f"{name}: duplicate (region, case) claim for {cid!r}")
            seen_keys.add(key)

        if e.route == "composition_defining" and e.artifact is None:
            v.append(f"{name}: composition_defining requires an artifact")
        if e.route == "reference_preserving" and e.reference is None:
            v.append(f"{name}: reference_preserving requires a reference")
        if e.half == "deployment" and (e.discharge is None or e.discharge.kind != "neutrality_proof"):
            v.append(f"{name}: deployment-half entry requires a neutrality_proof discharge")

        for k, c in e.constants:
            if isinstance(c, float):
                v.append(f"{name}: constant {k!r} is a raw float")
            if isinstance(c, BitPattern) and not _HEX.match(c.bits):
                v.append(f"{name}: constant {k!r} has invalid bit pattern {c.bits!r}")

    # Duplicate ownership: within one case, each logical op belongs to at most one entry.
    for cid in case_ids:
        owned: dict[str, tuple] = {}
        for e in contract.composition:
            if cid not in e.cases:
                continue
            for op in e.region:
                if op in owned and owned[op] != e.region:
                    v.append(f"case {cid!r}: op {op!r} owned by both {owned[op]} and {e.region}")
                owned[op] = e.region
        if required_ops is not None and cid in required_ops:
            missing = required_ops[cid] - set(owned)
            if missing:
                v.append(f"case {cid!r}: uncovered required op(s) {sorted(missing)}")

    # Asymmetry discharge: one region, several impls across cases -> every entry proves it.
    by_region: dict[tuple, list] = {}
    for e in contract.composition:
        by_region.setdefault(e.region, []).append(e)
    for region, entries in by_region.items():
        if len({e.impl.id for e in entries}) > 1:
            for e in entries:
                if e.discharge is None:
                    v.append(f"entry {region}@{e.impl.id}: asymmetric region requires a discharge")

    for t in contract.claims.tolerances:
        for cid in t.case_pair:
            if cid not in case_ids:
                v.append(f"tolerance {t.case_pair}: unknown case {cid!r}")

    axes = [t.axis for t in contract.claims.topology]
    if len(axes) != len(set(axes)):
        v.append("duplicate topology axes")
    for t in contract.claims.topology:
        if t.kind not in TOPOLOGY_KINDS:
            v.append(f"topology {t.axis!r}: unknown kind {t.kind!r}")
        elif t.kind == "pinned" and (t.degree is None or t.collective_plan is None):
            v.append(f"topology {t.axis!r}: pinned requires degree and collective_plan")
        elif t.kind == "invariant" and (not t.domain or t.proof is None):
            v.append(f"topology {t.axis!r}: invariant requires a non-empty domain and a proof")

    try:
        computed = compute_identities(contract)
    except SerializationError as exc:
        v.append(f"identity recomputation failed: {exc}")
    else:
        if contract.identities != computed:
            v.append("stored identities do not match recomputed identities")

    return v


def validate_or_raise(contract: ExecutionContract, required_ops: Mapping[str, frozenset[str]] | None = None) -> None:
    """Raise ValidationError listing every violation."""
    violations = validate(contract, required_ops)
    if violations:
        raise ValidationError("; ".join(violations))
