import struct

from skyrl.backends.skyrl_train.isoexec.contract import (
    BitPattern,
    from_canonical_json,
    to_canonical_json,
    validate,
)
from skyrl.backends.skyrl_train.isoexec.core.contract_build import (
    build_execution_contract,
)
from skyrl.backends.skyrl_train.isoexec.core.registry_build import build_registry
from skyrl.backends.skyrl_train.isoexec.models import qwen3_5
from skyrl.backends.skyrl_train.isoexec.models.policy import build_selections


def _build():
    reg = build_registry(strict=True)
    sel = build_selections(qwen3_5.PROFILE, qwen3_5.EXCEPTIONS)
    # claims as in models/qwen3_5.build: these tests pin the PRODUCTION contract, claims included.
    p = qwen3_5.PROFILE
    return reg, sel, build_execution_contract(
        reg,
        sel,
        arch="sm90",
        model=qwen3_5.MODEL,
        topology=p.topology,
        states=p.states,
        tolerances=p.tolerances,
    )


def _primary_op(reg, entry):
    # The region member whose OpSpec registers this entry's impl.
    owners = [op for op in entry.region if reg.has_op(op) and entry.impl.id in reg.get_op(op).impls]
    assert len(owners) == 1, (entry.region, entry.impl.id, owners)
    return owners[0]


def _decode(v):
    if isinstance(v, BitPattern):
        return struct.unpack("<d", struct.pack("<Q", int(v.bits, 16)))[0]
    return v


def test_round_trip_correspondence():
    reg, sel, c = _build()
    # selections -> contract: every (op, site) is covered by exactly one entry, same selection.
    for (op, site), s in sel.items():
        hits = [e for e in c.composition if site in e.cases and _primary_op(reg, e) == op]
        assert len(hits) == 1, (op, site, hits)
        e = hits[0]
        assert (e.impl.id, e.impl.version, e.impl.arch) == (s.impl_id, s.version, "sm90")
        assert {k: _decode(v) for k, v in e.constants} == s.pinned_constants
        assert (e.half == "deployment") == (s.classification == "deployment")
    # contract -> selections: every (primary op, case) is a selection key; no extras.
    keys = {(_primary_op(reg, e), site) for e in c.composition for site in e.cases}
    assert keys == set(sel.keys())


def test_contract_validates_clean():
    _, _, c = _build()
    assert validate(c) == []


def test_numerical_policy_is_stable():
    _, _, c1 = _build()
    _, _, c2 = _build()
    assert c1.identities == c2.identities, "derived identities must be deterministic"
    assert c1.identities.numerical_policy


def test_deployment_half_matches_selections():
    reg, sel, c = _build()
    dep = {(_primary_op(reg, e), site) for e in c.composition if e.half == "deployment" for site in e.cases}
    assert dep == {k for k, s in sel.items() if s.classification == "deployment"}
    for e in c.composition:
        if e.half == "deployment":
            assert e.discharge and e.discharge.kind == "neutrality_proof" and e.discharge.ref


def test_subsumed_ops_join_the_region():
    reg, _, c = _build()
    gdn = [e for e in c.composition if "gdn.core" in e.region]
    assert gdn and all(e.region == ("gdn.core", "gdn.gating", "gdn.l2norm") for e in gdn)


def test_serialization_round_trips_byte_stable():
    _, _, c = _build()
    b = to_canonical_json(c)
    assert to_canonical_json(from_canonical_json(b)) == b
