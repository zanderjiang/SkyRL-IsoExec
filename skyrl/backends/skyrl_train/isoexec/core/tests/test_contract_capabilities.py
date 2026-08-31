"""Manifest-era capabilities on the ExecutionContract build path.

Runs on tiny synthetic registries so the assertions are about the machinery, not the qwen3.5
content the sibling test files pin down. Process-global state is restored around every test.
"""

import dataclasses
import os
import struct
import tempfile
from contextlib import contextmanager
from types import SimpleNamespace

from skyrl.backends.skyrl_train.isoexec.contract import (
    BitPattern,
    deployment_half,
    function_half,
    to_canonical_json,
    validate,
)
from skyrl.backends.skyrl_train.isoexec.core import fingerprint as fp
from skyrl.backends.skyrl_train.isoexec.core import process_contract as pc
from skyrl.backends.skyrl_train.isoexec.core.contract_build import (
    DEPLOYMENT,
    ContractBuildError,
    PinValidationError,
    build_execution_contract,
    validate_pins,
)
from skyrl.backends.skyrl_train.isoexec.core.contract_delivery import (
    CONTRACT_HASH_ENV,
    ContractDeliveryError,
    expected_installed_keys,
    load_contract,
    validate_contract_against_installed,
    write_contract_file,
)
from skyrl.backends.skyrl_train.isoexec.core.registry import (
    PER_MODEL,
    ImplSpec,
    OneOf,
    OpSpec,
    Registry,
    RegistryError,
    RoundingSchedule,
)
from skyrl.backends.skyrl_train.isoexec.models.policy import Selection, entry

TRAINER = ("trainer_fwd", "trainer_score")
ENGINE = ("engine_prefill", "engine_decode")
SITES = TRAINER + ENGINE

MM_PINS = {"block": 128, "leaves": 4, "eps": 1e-6}
ARCHS = frozenset({"sm90", "sm100"})


def _registry():
    reg = Registry()
    mm = OpSpec("alpha.mm", list(SITES))
    mm.add_impl(
        ImplSpec(
            "ref",
            1,
            ARCHS,
            rounding=RoundingSchedule({"block": 128, "leaves": PER_MODEL, "mode": OneOf("tree", "flat"), "eps": 1e-6}),
        )
    )
    # Default-selected impls are proven on both archs so the arch-rotation test can build on
    # sm100; "twin" stays sm90-only, for the arch-admission refusal.
    mm.add_impl(
        ImplSpec(
            "twin",
            1,
            frozenset({"sm90"}),
            rounding=RoundingSchedule({"block": 128}),
            capabilities={"bitwise_equal_to": "ref"},
        )
    )
    reg.register_op(mm)
    core = OpSpec("alpha.core", list(SITES))
    core.add_impl(ImplSpec("fused_all", 1, ARCHS, subsumes=("alpha.sub",)))
    reg.register_op(core)
    sub = OpSpec("alpha.sub", list(SITES))
    sub.add_impl(ImplSpec("eager", 1, ARCHS))
    reg.register_op(sub)
    pin = OpSpec("alpha.pin", list(SITES))
    pin.add_impl(ImplSpec("pinned", 1, ARCHS, rounding=RoundingSchedule({"NCCL": PER_MODEL})))
    pin.add_impl(ImplSpec("unpinned", 1, ARCHS))
    reg.register_op(pin)
    env = OpSpec("alpha.env", list(SITES))
    env.add_impl(ImplSpec("std", 1, ARCHS))
    reg.register_op(env)
    return reg


def _selections():
    s = {}
    for site in SITES:
        s[("alpha.mm", site)] = entry("ref", pinned=dict(MM_PINS))
        s[("alpha.core", site)] = entry("fused_all")
    for site in TRAINER:
        s[("alpha.pin", site)] = entry("pinned", pinned={"NCCL": "tree"})
    for site in ENGINE:
        s[("alpha.pin", site)] = entry("unpinned", cls=DEPLOYMENT, proof="gate-run-7")
    return s


def _with(op, sites, e):
    sel = _selections()
    for s in sites:
        sel[(op, s)] = e
    return sel


def _build(reg=None, sel=None, **kw):
    kw.setdefault("arch", "sm90")
    kw.setdefault("model", "tiny")
    return build_execution_contract(reg or _registry(), _selections() if sel is None else sel, **kw)


def _refuses(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        return str(e)
    raise AssertionError(f"expected {exc.__name__} from {getattr(fn, '__name__', fn)}")


@contextmanager
def _env(**pairs):
    saved = {k: os.environ.get(k) for k in pairs}
    try:
        for k, v in pairs.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        yield
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


# --- 1. registry-validated build ---


def test_build_refuses_unknown_op():
    msg = _refuses(ContractBuildError, _build, sel={("beta.nope", "trainer_fwd"): entry("ref")})
    assert "unknown op" in msg and "beta.nope" in msg


def test_build_refuses_undeclared_site():
    reg = Registry()
    op = OpSpec("alpha.mm", ["trainer_fwd"])
    op.add_impl(ImplSpec("ref", 1, frozenset({"sm90"})))
    reg.register_op(op)
    msg = _refuses(ContractBuildError, _build, reg=reg, sel={("alpha.mm", "engine_decode"): entry("ref")})
    assert "declares only" in msg and "engine_decode" in msg


def test_build_refuses_unregistered_impl():
    msg = _refuses(ContractBuildError, _build, sel={("alpha.mm", "trainer_fwd"): entry("ghost")})
    assert "ghost" in msg and "not registered" in msg


def test_registry_refuses_duplicates():
    reg = _registry()
    _refuses(RegistryError, reg.register_op, OpSpec("alpha.mm", list(SITES)))
    _refuses(RegistryError, reg.get_op("alpha.mm").add_impl, ImplSpec("ref", 2, frozenset({"sm90"})))


def test_same_key_reassignment_is_a_single_entry():
    # Selections are keyed by (op, site): a re-assigned key last-wins, so no duplicate can exist.
    sel = _selections()
    sel[("alpha.mm", "trainer_fwd")] = entry("ref", pinned={"block": 128})
    c = _build(sel=sel)
    hits = [e for e in c.composition if "alpha.mm" in e.region and "trainer_fwd" in e.cases]
    assert len(hits) == 1 and dict(hits[0].constants) == {"block": 128}
    assert validate(c) == []


def test_minimal_build_succeeds():
    c = _build()
    assert validate(c) == []
    assert {e.impl.arch for e in c.composition} == {"sm90"}
    assert {x.id for x in c.cases} == set(SITES)


def test_cases_follow_the_selected_sites():
    c = _build(sel={("alpha.mm", s): entry("ref") for s in TRAINER})
    assert {x.id for x in c.cases} == set(TRAINER)


def test_asymmetric_impls_without_claim_refuse():
    sel = _selections()
    for site in ENGINE:
        sel[("alpha.pin", site)] = entry("unpinned")  # function half, no claim anywhere
    msg = _refuses(ContractBuildError, _build, sel=sel)
    assert "resolves to" in msg and "alpha.pin" in msg


def test_asymmetric_impls_with_bitwise_twin_accepted():
    c = _build(sel=_with("alpha.mm", ENGINE, entry("twin", pinned={"block": 128})))
    e = next(e for e in c.composition if e.impl.id == "twin")
    assert e.discharge and (e.discharge.kind, e.discharge.ref) == ("bitwise_equal_to", "ref")


def _mm_impl(reg, impl_id, **kw):
    """Replace alpha.mm's twin with a variant, keeping its schedule."""
    reg.get_op("alpha.mm").impls[impl_id] = ImplSpec(impl_id, 1, ARCHS, rounding=RoundingSchedule({"block": 128}), **kw)
    return reg


def test_self_referential_bitwise_claim_refuses():
    reg = _mm_impl(_registry(), "twin", capabilities={"bitwise_equal_to": "twin"})
    sel = _with("alpha.mm", ENGINE, entry("twin", pinned={"block": 128}))
    msg = _refuses(ContractBuildError, _build, reg=reg, sel=sel)
    assert "bitwise_equal_to ITSELF" in msg and "twin" in msg


def test_dangling_bitwise_referent_names_the_referent():
    reg = _mm_impl(_registry(), "twin", capabilities={"bitwise_equal_to": "does_not_exist"})
    sel = _with("alpha.mm", ENGINE, entry("twin", pinned={"block": 128}))
    msg = _refuses(ContractBuildError, _build, reg=reg, sel=sel)
    assert "does_not_exist" in msg and "not a registered impl" in msg


def test_pairwise_claim_does_not_discharge_a_third_impl():
    reg = _mm_impl(_registry(), "twin", capabilities={"bitwise_equal_to": "ref"})
    _mm_impl(reg, "third")
    sel = _selections()
    sel[("alpha.mm", "engine_prefill")] = entry("twin", pinned={"block": 128})
    sel[("alpha.mm", "engine_decode")] = entry("third", pinned={"block": 128})
    msg = _refuses(ContractBuildError, _build, reg=reg, sel=sel)
    assert "third" in msg and "carry no discharge" in msg


def test_unresolvable_equivalence_proof_refuses():
    reg = _mm_impl(_registry(), "twin", capabilities={"equivalence_proof": "TODO(nobody): write this"})
    sel = _with("alpha.mm", ENGINE, entry("twin", pinned={"block": 128}))
    msg = _refuses(ContractBuildError, _build, reg=reg, sel=sel)
    assert "resolves to no gate" in msg


def test_equivalence_proof_naming_a_real_gate_discharges():
    proof = "ops/gdn/tests/gdn_native_kernel_parity_test_gpu.py: split-exactness at any boundary"
    reg = _mm_impl(_registry(), "twin", capabilities={"equivalence_proof": proof})
    c = _build(reg=reg, sel=_with("alpha.mm", ENGINE, entry("twin", pinned={"block": 128})))
    e = next(e for e in c.composition if e.impl.id == "twin")
    assert e.discharge and e.discharge.kind == "equivalence_proof"


def test_empty_selections_refuse():
    msg = _refuses(ContractBuildError, _build, sel={})
    assert "empty selection map" in msg


def test_dangling_subsumes_refuses_on_the_build_path():
    reg = _registry()
    reg.get_op("alpha.core").impls["fused_all"] = ImplSpec("fused_all", 1, ARCHS, subsumes=("alpha.sub", "alpha.ghost"))
    msg = _refuses(ContractBuildError, _build, reg=reg)
    assert "alpha.ghost" in msg and "subsumption-closed" in msg


# --- 2. classification rules ---


def test_selection_deployment_requires_proof():
    msg = _refuses(ContractBuildError, Selection, "ref", classification=DEPLOYMENT)
    assert "neutrality_proof" in msg


def test_selection_function_refuses_proof():
    msg = _refuses(ContractBuildError, Selection, "ref", neutrality_proof="run-1")
    assert "must not carry" in msg


def test_mapping_selections_recheck_classification():
    # Dict-shaped selections bypass Selection.__post_init__; the build re-checks them.
    for bad in (
        {"impl_id": "ref", "classification": DEPLOYMENT},
        {"impl_id": "ref", "neutrality_proof": "run-1"},
        {"impl_id": "ref", "classification": "advisory"},
    ):
        sel = _selections()
        sel[("alpha.mm", "trainer_fwd")] = bad
        _refuses(ContractBuildError, _build, sel=sel)


def test_halves_partition_the_hash_inputs():
    c = _build()
    fn = {e["impl"]["id"] for e in function_half(c)["entries"]}
    dep = {e["impl"]["id"] for e in deployment_half(c)["entries"]}
    assert dep == {"unpinned"} and "unpinned" not in fn
    assert fn == {"ref", "fused_all", "pinned"}


def test_classification_is_per_entry():
    c = _build()
    halves = {e.half: set(e.cases) for e in c.composition if e.region == ("alpha.pin",)}
    assert halves == {"function": set(TRAINER), "deployment": set(ENGINE)}


# --- 3. validate_pins ---


def test_pin_undeclared_key_refuses_naming_it():
    msg = _refuses(
        PinValidationError,
        validate_pins,
        _registry(),
        {("alpha.mm", "trainer_fwd"): entry("ref", pinned={"nope": 1})},
        model="tiny",
    )
    assert "nope" in msg and "declares no such key" in msg and "alpha.mm" in msg


def test_pin_literal_match_and_mismatch():
    reg = _registry()
    validate_pins(reg, {("alpha.mm", "trainer_fwd"): entry("ref", pinned={"block": 128})})
    msg = _refuses(
        PinValidationError, validate_pins, reg, {("alpha.mm", "trainer_fwd"): entry("ref", pinned={"block": 64})}
    )
    assert "block=64" in msg and "block=128" in msg


def test_pin_per_model_accepts_anything():
    reg = _registry()
    for v in (0, "8", True, [1, 2], None):
        validate_pins(reg, {("alpha.mm", "trainer_fwd"): entry("ref", pinned={"leaves": v})})


def test_pin_oneof_membership():
    reg = _registry()
    for v in ("tree", "flat"):
        validate_pins(reg, {("alpha.mm", "trainer_fwd"): entry("ref", pinned={"mode": v})})
    msg = _refuses(
        PinValidationError, validate_pins, reg, {("alpha.mm", "trainer_fwd"): entry("ref", pinned={"mode": "hybrid"})}
    )
    assert "OneOf" in msg


def test_pin_violations_reported_together():
    sel = {
        ("alpha.mm", "trainer_fwd"): entry("ref", pinned={"block": 64, "mode": "hybrid"}),
        ("alpha.mm", "trainer_score"): entry("ghost", pinned={"k": 1}),
        ("beta.ghost", "trainer_fwd"): entry("x", pinned={"k": 1}),
    }
    msg = _refuses(PinValidationError, validate_pins, _registry(), sel, model="tiny")
    for needle in ("block=64", "mode='hybrid'", "'ghost' is not registered", "beta.ghost"):
        assert needle in msg, needle


def test_validate_pins_off_skips_the_gate():
    sel = _with("alpha.mm", ("trainer_fwd",), entry("ref", pinned=dict(MM_PINS, undeclared=1)))
    _refuses(PinValidationError, _build, sel=sel)
    c = _build(sel=sel, validate_pins=False)
    assert validate(c) == []


def test_float_pins_round_trip_bitpatterns():
    reg = _registry()
    c = _build(reg=reg)
    consts = dict(next(e for e in c.composition if e.region == ("alpha.mm",)).constants)
    bp = consts["eps"]
    assert isinstance(bp, BitPattern) and bp.dtype == "fp64"
    assert struct.unpack("<d", struct.pack("<Q", int(bp.bits, 16)))[0] == 1e-6
    view = pc.build_contract_view(c, reg)
    assert view[("alpha.mm", "trainer_fwd")]["pinned_constants"]["eps"] == 1e-6
    to_canonical_json(c)  # a raw float in the constants would refuse here


# --- 4. arch gate ---


def test_non_accelerator_arch_refuses():
    for arch in ("cpu", ""):
        _refuses(ContractBuildError, _build, arch=arch)


def test_allow_non_accelerator_escape():
    c = _build(arch="cpu", allow_non_accelerator_arch=True)
    assert {e.impl.arch for e in c.composition} == {"cpu"} and validate(c) == []


def test_impl_outside_its_supported_archs_refuses():
    # twin is proven on sm90 only, and evidence is arch-scoped.
    sel = _with("alpha.mm", ENGINE, entry("twin", pinned={"block": 128}))
    msg = _refuses(ContractBuildError, _build, sel=sel, arch="sm100")
    assert "twin" in msg and "supported_archs" in msg and "sm100" in msg


def test_arch_rotates_numerical_policy():
    a, b = _build(arch="sm90").identities, _build(arch="sm100").identities
    assert a.numerical_policy != b.numerical_policy
    assert a.semantic == b.semantic


# --- 5. identity axes ---


def test_build_twice_is_deterministic():
    a, b = _build(), _build()
    assert a.identities == b.identities
    assert to_canonical_json(a) == to_canonical_json(b)


def test_insertion_order_never_moves_identity():
    sel = _selections()
    rev = dict(reversed(list(sel.items())))
    assert list(rev) != list(sel)
    assert _build(sel=rev).identities == _build(sel=sel).identities


def test_impl_change_rotates_numerical_policy():
    base = _build().identities
    ids = _build(sel=_with("alpha.mm", SITES, entry("twin", pinned={"block": 128}))).identities
    assert ids.numerical_policy != base.numerical_policy
    assert ids.semantic == base.semantic  # same op vocabulary


def test_version_bump_rotates_numerical_policy():
    ids = _build(sel=_with("alpha.mm", SITES, entry("ref", version=2, pinned=dict(MM_PINS)))).identities
    assert ids.numerical_policy != _build().identities.numerical_policy


def test_constant_change_rotates_numerical_policy():
    ids = _build(sel=_with("alpha.mm", SITES, entry("ref", pinned=dict(MM_PINS, leaves=8)))).identities
    assert ids.numerical_policy != _build().identities.numerical_policy


def test_deployment_only_change_rotates_deployment_not_function():
    # Symmetric impl at every site: the neutrality proof lives only in the deployment half.
    def sel(proof):
        s = _selections()
        for site in TRAINER:
            s[("alpha.env", site)] = entry("std")
        for site in ENGINE:
            s[("alpha.env", site)] = entry("std", cls=DEPLOYMENT, proof=proof)
        return s

    a, b = _build(sel=sel("gate-run-7")).identities, _build(sel=sel("gate-run-8")).identities
    assert a.deployment != b.deployment
    assert a.numerical_policy == b.numerical_policy
    assert a.semantic == b.semantic


def test_group_proof_is_function_bearing_for_asymmetric_ops():
    # When sites resolve to different impls the neutrality proof becomes the group discharge
    # carried by the function entries too, so it rotates numerical_policy.
    ids = _build(sel=_with("alpha.pin", ENGINE, entry("unpinned", cls=DEPLOYMENT, proof="gate-run-8"))).identities
    assert ids.numerical_policy != _build().identities.numerical_policy


def test_flip_function_to_deployment_moves_the_entry():
    base = _build()
    flipped = _build(sel=_with("alpha.mm", ENGINE, entry("ref", pinned=dict(MM_PINS), cls=DEPLOYMENT, proof="run-9")))

    def keys(half):
        return {(e["impl"]["id"], tuple(e["cases"])) for e in half["entries"]}

    assert ("ref", tuple(sorted(SITES))) in keys(function_half(base))
    assert ("ref", tuple(sorted(TRAINER))) in keys(function_half(flipped))
    assert ("ref", tuple(sorted(ENGINE))) in keys(deployment_half(flipped))
    assert ("ref", tuple(sorted(ENGINE))) not in keys(deployment_half(base))
    assert flipped.identities.numerical_policy != base.identities.numerical_policy
    assert flipped.identities.deployment != base.identities.deployment
    assert flipped.identities.semantic == base.identities.semantic


def test_semantic_rotates_on_model_and_vocabulary():
    base = _build().identities
    assert _build(model="tiny-2").identities.semantic != base.semantic
    grown = _selections()
    for site in TRAINER:
        grown[("alpha.env", site)] = entry("std")  # new logical op in the vocabulary
    assert _build(sel=grown).identities.semantic != base.semantic
    prof = SimpleNamespace(architectures=("SomethingForCausalLM",))
    assert _build(profile=prof).identities.semantic != base.semantic


# --- 6. immutability ---


def test_contract_is_frozen():
    c = _build()
    e = c.composition[0]
    for obj, attr in (
        (c, "schema_version"),
        (c, "composition"),
        (e, "impl"),
        (e.impl, "id"),
        (e, "constants"),
        (c.identities, "numerical_policy"),
        (c.model, "family"),
    ):
        _refuses(dataclasses.FrozenInstanceError, setattr, obj, attr, "x")
    assert isinstance(c.composition, tuple) and isinstance(e.constants, tuple) and isinstance(e.cases, tuple)
    _refuses(dataclasses.FrozenInstanceError, setattr, Selection("ref"), "impl_id", "x")


# --- 7. delivery ---


def test_delivery_round_trip_and_env_cross_check():
    c = _build()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tiny.json")
        h = write_contract_file(c, path)
        assert h == c.identities.numerical_policy
        with _env(**{CONTRACT_HASH_ENV: None}):
            assert load_contract(path).identities == c.identities  # absent env: self-consistency only
        with _env(**{CONTRACT_HASH_ENV: h}):
            load_contract(path)
        with _env(**{CONTRACT_HASH_ENV: "0" * 64}):
            msg = _refuses(ContractDeliveryError, load_contract, path)
            assert "cross-check FAILED" in msg


def test_tampered_impl_id_refuses():
    c = _build()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tiny.json")
        write_contract_file(c, path)
        raw = open(path, "rb").read()
        assert raw.count(b'"id":"ref"')
        open(path, "wb").write(raw.replace(b'"id":"ref"', b'"id":"reX"'))
        with _env(**{CONTRACT_HASH_ENV: None}):
            msg = _refuses(ContractDeliveryError, load_contract, path)
            assert "recomputed" in msg


def test_installed_validation_fingerprint_and_drift():
    reg = _registry()
    c = _build(reg=reg)
    named = expected_installed_keys(c, reg)
    ok = fp.ResolvedFingerprint(installed={k: {"impl_id": "x"} for k in named})
    validate_contract_against_installed(c, reg, ok)  # exact set, via the ResolvedFingerprint branch
    short = fp.ResolvedFingerprint(installed={k: {} for k in sorted(named)[1:]})
    msg = _refuses(ContractDeliveryError, validate_contract_against_installed, c, reg, short)
    assert "NOT installed" in msg
    msg = _refuses(ContractDeliveryError, validate_contract_against_installed, c, reg, set(named) | {("alpha.mm", "x")})
    assert "UNNAMED" in msg


def test_subsumed_op_never_expects_an_installed_key():
    reg = _registry()
    c = _build(reg=reg)
    named = expected_installed_keys(c, reg)
    assert {("alpha.core", s) for s in SITES} <= named
    assert not [k for k in named if k[0] == "alpha.sub"]
    assert next(e for e in c.composition if e.impl.id == "fused_all").region == ("alpha.core", "alpha.sub")
    # The registry still declares alpha.sub sites, so a registry-as-installed check refuses them.
    msg = _refuses(ContractDeliveryError, validate_contract_against_installed, c, reg, reg)
    assert "alpha.sub" in msg


def _run():
    import traceback

    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = []
    for name, fn in fns:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed.append(name)
            traceback.print_exc()
            print(f"FAIL {name}")
    print(f"{len(fns) - len(failed)}/{len(fns)} passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    _run()
