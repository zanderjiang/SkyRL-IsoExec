"""Canonical JSON encode/decode. Sorted keys, no JSON floats, unknown fields refuse."""

import json

from .types import (
    BitPattern,
    Claims,
    CompositionEntry,
    Coverage,
    EquivalenceProof,
    ExecutionCase,
    ExecutionContract,
    Identities,
    ImplRef,
    ModelRef,
    StateClaim,
    ToleranceClaim,
    TopologyClaim,
)

SUPPORTED_SCHEMA_VERSIONS = {"1"}


class SerializationError(Exception):
    pass


class UnknownFieldError(SerializationError):
    pass


def _assert_no_floats(obj, path="$"):
    if isinstance(obj, float):
        raise SerializationError(f"raw float at {path}; use BitPattern or a decimal string")
    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_no_floats(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _assert_no_floats(v, f"{path}[{i}]")


def canonical_bytes(plain) -> bytes:
    """Canonical JSON bytes of a plain structure: sorted keys, compact, UTF-8, no floats."""
    _assert_no_floats(plain)
    return json.dumps(plain, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _enc_opt(value, enc):
    return None if value is None else enc(value)


def _enc_impl(i: ImplRef) -> dict:
    return {"id": i.id, "version": i.version, "arch": i.arch}


def _enc_discharge(d: EquivalenceProof) -> dict:
    return {"kind": d.kind, "ref": d.ref}


def _enc_coverage(c: Coverage) -> dict:
    return {"kind": c.kind, "description": c.description}


def _enc_const(v):
    if isinstance(v, BitPattern):
        return {"bits": v.bits, "dtype": v.dtype}
    return v


def encode_constants(pairs: tuple) -> dict:
    return {k: _enc_const(v) for k, v in pairs}


def _enc_entry(e: CompositionEntry) -> dict:
    return {
        "region": list(e.region),
        "cases": list(e.cases),
        "impl": _enc_impl(e.impl),
        "route": e.route,
        "constants": encode_constants(e.constants),
        "artifact": e.artifact,
        "reference": _enc_opt(e.reference, _enc_impl),
        "coverage": _enc_opt(e.coverage, _enc_coverage),
        "discharge": _enc_opt(e.discharge, _enc_discharge),
        "half": e.half,
    }


def _enc_case(c: ExecutionCase) -> dict:
    return {
        "id": c.id,
        "runtime_role": c.runtime_role,
        "grad_mode": c.grad_mode,
        "state_mode": c.state_mode,
        "shape_domain": c.shape_domain,
        "constraints": list(c.constraints),
    }


def _enc_topology(t: TopologyClaim) -> dict:
    return {
        "axis": t.axis,
        "kind": t.kind,
        "degree": t.degree,
        "collective_plan": t.collective_plan,
        "domain": list(t.domain),
        "proof": t.proof,
    }


def _enc_state(s: StateClaim) -> dict:
    return {
        "state_id": s.state_id,
        "invalidated_by": list(s.invalidated_by),
        "replay_safe": s.replay_safe,
        "ref": s.ref,
    }


def _enc_tolerance(t: ToleranceClaim) -> dict:
    return {
        "case_pair": list(t.case_pair),
        "bounds": dict(t.bounds),
        "attributed_to": list(t.attributed_to),
    }


def _enc_contract(c: ExecutionContract) -> dict:
    return {
        "schema_version": c.schema_version,
        "model": {
            "family": c.model.family,
            "architectures": list(c.model.architectures),
            "profile_ref": c.model.profile_ref,
        },
        "identities": {
            "semantic": c.identities.semantic,
            "numerical_policy": c.identities.numerical_policy,
            "deployment": c.identities.deployment,
        },
        "cases": [_enc_case(x) for x in c.cases],
        "composition": [_enc_entry(x) for x in c.composition],
        "claims": {
            "topology": [_enc_topology(x) for x in c.claims.topology],
            "state": [_enc_state(x) for x in c.claims.state],
            "tolerances": [_enc_tolerance(x) for x in c.claims.tolerances],
        },
    }


def to_canonical_json(contract: ExecutionContract) -> bytes:
    """Serialize a contract to its canonical byte form."""
    return canonical_bytes(_enc_contract(contract))


def _fields(d: dict, required: tuple[str, ...], ctx: str) -> None:
    unknown = set(d) - set(required)
    if unknown:
        raise UnknownFieldError(f"{ctx}: unknown field(s) {sorted(unknown)}")
    missing = set(required) - set(d)
    if missing:
        raise SerializationError(f"{ctx}: missing field(s) {sorted(missing)}")


def _dec_impl(d: dict, ctx: str) -> ImplRef:
    _fields(d, ("id", "version", "arch"), ctx)
    return ImplRef(id=d["id"], version=d["version"], arch=d["arch"])


def _dec_opt(d, dec, ctx):
    return None if d is None else dec(d, ctx)


def _dec_discharge(d: dict, ctx: str) -> EquivalenceProof:
    _fields(d, ("kind", "ref"), ctx)
    return EquivalenceProof(kind=d["kind"], ref=d["ref"])


def _dec_coverage(d: dict, ctx: str) -> Coverage:
    _fields(d, ("kind", "description"), ctx)
    return Coverage(kind=d["kind"], description=d["description"])


def _dec_const(v, ctx: str):
    if isinstance(v, dict):
        _fields(v, ("bits", "dtype"), ctx)
        return BitPattern(bits=v["bits"], dtype=v["dtype"])
    return v


def _dec_entry(d: dict, ctx: str) -> CompositionEntry:
    _fields(
        d,
        ("region", "cases", "impl", "route", "constants", "artifact", "reference", "coverage", "discharge", "half"),
        ctx,
    )
    return CompositionEntry(
        region=tuple(d["region"]),
        cases=tuple(d["cases"]),
        impl=_dec_impl(d["impl"], f"{ctx}.impl"),
        route=d["route"],
        constants={k: _dec_const(v, f"{ctx}.constants.{k}") for k, v in d["constants"].items()},
        artifact=d["artifact"],
        reference=_dec_opt(d["reference"], _dec_impl, f"{ctx}.reference"),
        coverage=_dec_opt(d["coverage"], _dec_coverage, f"{ctx}.coverage"),
        discharge=_dec_opt(d["discharge"], _dec_discharge, f"{ctx}.discharge"),
        half=d["half"],
    )


def _dec_case(d: dict, ctx: str) -> ExecutionCase:
    _fields(d, ("id", "runtime_role", "grad_mode", "state_mode", "shape_domain", "constraints"), ctx)
    return ExecutionCase(
        id=d["id"],
        runtime_role=d["runtime_role"],
        grad_mode=d["grad_mode"],
        state_mode=d["state_mode"],
        shape_domain=d["shape_domain"],
        constraints=tuple(d["constraints"]),
    )


def _dec_topology(d: dict, ctx: str) -> TopologyClaim:
    _fields(d, ("axis", "kind", "degree", "collective_plan", "domain", "proof"), ctx)
    return TopologyClaim(
        axis=d["axis"],
        kind=d["kind"],
        degree=d["degree"],
        collective_plan=d["collective_plan"],
        domain=tuple(d["domain"]),
        proof=d["proof"],
    )


def _dec_state(d: dict, ctx: str) -> StateClaim:
    _fields(d, ("state_id", "invalidated_by", "replay_safe", "ref"), ctx)
    return StateClaim(
        state_id=d["state_id"],
        invalidated_by=tuple(d["invalidated_by"]),
        replay_safe=d["replay_safe"],
        ref=d["ref"],
    )


def _dec_tolerance(d: dict, ctx: str) -> ToleranceClaim:
    _fields(d, ("case_pair", "bounds", "attributed_to"), ctx)
    pair = d["case_pair"]
    if len(pair) != 2:
        raise SerializationError(f"{ctx}: case_pair must have exactly 2 members")
    return ToleranceClaim(
        case_pair=(pair[0], pair[1]),
        bounds=d["bounds"],
        attributed_to=tuple(d["attributed_to"]),
    )


def _reject_float(s: str):
    raise SerializationError(f"raw JSON float {s!r}; use BitPattern or a decimal string")


def from_canonical_json(data: bytes) -> ExecutionContract:
    """Strict decode: unknown fields refuse, floats refuse, schema version checked."""
    d = json.loads(data.decode("utf-8"), parse_float=_reject_float, parse_constant=_reject_float)
    _fields(d, ("schema_version", "model", "identities", "cases", "composition", "claims"), "contract")
    if d["schema_version"] not in SUPPORTED_SCHEMA_VERSIONS:
        raise SerializationError(f"unsupported schema_version {d['schema_version']!r}")
    _fields(d["model"], ("family", "architectures", "profile_ref"), "model")
    _fields(d["identities"], ("semantic", "numerical_policy", "deployment"), "identities")
    _fields(d["claims"], ("topology", "state", "tolerances"), "claims")
    return ExecutionContract(
        schema_version=d["schema_version"],
        model=ModelRef(
            family=d["model"]["family"],
            architectures=tuple(d["model"]["architectures"]),
            profile_ref=d["model"]["profile_ref"],
        ),
        identities=Identities(
            semantic=d["identities"]["semantic"],
            numerical_policy=d["identities"]["numerical_policy"],
            deployment=d["identities"]["deployment"],
        ),
        cases=tuple(_dec_case(x, f"cases[{i}]") for i, x in enumerate(d["cases"])),
        composition=tuple(_dec_entry(x, f"composition[{i}]") for i, x in enumerate(d["composition"])),
        claims=Claims(
            topology=tuple(_dec_topology(x, f"topology[{i}]") for i, x in enumerate(d["claims"]["topology"])),
            state=tuple(_dec_state(x, f"state[{i}]") for i, x in enumerate(d["claims"]["state"])),
            tolerances=tuple(_dec_tolerance(x, f"tolerances[{i}]") for i, x in enumerate(d["claims"]["tolerances"])),
        ),
    )
