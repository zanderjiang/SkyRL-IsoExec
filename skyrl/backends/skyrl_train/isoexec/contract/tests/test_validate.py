"""Each structural invariant has a passing and a failing case."""

import dataclasses
import unittest

from skyrl.backends.skyrl_train.isoexec.contract import (
    BitPattern,
    CompositionEntry,
    EquivalenceProof,
    ImplRef,
    StateClaim,
    ToleranceClaim,
    TopologyClaim,
    ValidationError,
    compute_identities,
    validate,
    validate_or_raise,
)
from skyrl.backends.skyrl_train.isoexec.contract.tests.fixtures import (
    ALL_CASES,
    ENGINE,
    base_contract,
)


def _rehash(c):
    return dataclasses.replace(c, identities=compute_identities(c))


def _with_entry(c, entry):
    return _rehash(dataclasses.replace(c, composition=c.composition + (entry,)))


def _with_claims(c, **kw):
    return _rehash(dataclasses.replace(c, claims=dataclasses.replace(c.claims, **kw)))


def _replace_entry(c, idx, **kw):
    e = dataclasses.replace(c.composition[idx], **kw)
    return _rehash(dataclasses.replace(c, composition=c.composition[:idx] + (e,) + c.composition[idx + 1 :]))


class TestValidate(unittest.TestCase):
    def setUp(self):
        self.c = base_contract()

    def _assert_violation(self, c, fragment):
        violations = validate(c)
        self.assertTrue(any(fragment in v for v in violations), f"expected {fragment!r} in {violations}")

    def test_fixture_is_valid(self):
        self.assertEqual(validate(self.c), [])
        validate_or_raise(self.c)

    def test_duplicate_ownership(self):
        # gdn.core is already owned by the fused group in every case.
        e = CompositionEntry(region=("gdn.core",), cases=ALL_CASES, impl=ImplRef("rogue", 1, "sm90"), route="canonical")
        self._assert_violation(_with_entry(self.c, e), "owned by both")

    def test_coverage_gap_with_required_ops(self):
        required = {"engine_decode": frozenset({"moe.combine", "attention.core"})}
        violations = validate(self.c, required)
        self.assertTrue(any("uncovered required op" in v and "attention.core" in v for v in violations))
        covered = {"engine_decode": frozenset({"moe.combine"})}
        self.assertEqual(validate(self.c, covered), [])

    def test_missing_discharge_on_asymmetric_region(self):
        idx = next(
            i for i, e in enumerate(self.c.composition) if e.region == ("moe.weights",) and "trainer_fwd" in e.cases
        )
        self._assert_violation(_replace_entry(self.c, idx, discharge=None), "asymmetric region requires a discharge")

    def test_deployment_without_neutrality_proof(self):
        idx = next(i for i, e in enumerate(self.c.composition) if e.half == "deployment")
        c2 = _replace_entry(self.c, idx, discharge=EquivalenceProof("equivalence_proof", "gates/x"))
        self._assert_violation(c2, "requires a neutrality_proof")

    def test_route_b_without_artifact(self):
        self._assert_violation(_replace_entry(self.c, 0, artifact=None), "composition_defining requires an artifact")

    def test_route_a_without_reference(self):
        idx = next(i for i, e in enumerate(self.c.composition) if e.route == "reference_preserving")
        self._assert_violation(_replace_entry(self.c, idx, reference=None), "reference_preserving requires a reference")

    def test_bad_bit_pattern_hex(self):
        idx = next(i for i, e in enumerate(self.c.composition) if e.route == "reference_preserving")
        c2 = _replace_entry(self.c, idx, constants={"scale": BitPattern("1.0", "fp32")})
        self._assert_violation(c2, "invalid bit pattern")

    def test_raw_float_constant(self):
        import dataclasses as dc

        e = dc.replace(self.c.composition[0], constants={"leaves": 4, "eps": 1e-6})
        c2 = dc.replace(self.c, composition=(e,) + self.c.composition[1:])
        self._assert_violation(c2, "raw float")

    def test_unknown_case_ref(self):
        e = CompositionEntry(
            region=("rope.rope",), cases=("engine_sample",), impl=ImplRef("x", 1, "sm90"), route="canonical"
        )
        self._assert_violation(_with_entry(self.c, e), "unknown case 'engine_sample'")

    def test_duplicate_region_case_claim(self):
        e = CompositionEntry(
            region=("moe.combine",),
            cases=ENGINE,
            impl=ImplRef("pik_leaf_tree", 2, "sm90"),
            route="composition_defining",
            artifact="sha256:1f2e3d4c",
            discharge=EquivalenceProof("equivalence_proof", "gates/x"),
        )
        self._assert_violation(_with_entry(self.c, e), "duplicate (region, case)")

    def test_duplicate_topology_axis(self):
        t = self.c.claims.topology + (TopologyClaim("TP", "pinned", degree=8, collective_plan="nccl"),)
        c2 = _rehash(dataclasses.replace(self.c, claims=dataclasses.replace(self.c.claims, topology=t)))
        self._assert_violation(c2, "duplicate topology axes")

    def test_pinned_requires_plan(self):
        t = self.c.claims.topology + (TopologyClaim("EP", "pinned", degree=8),)
        c2 = _rehash(dataclasses.replace(self.c, claims=dataclasses.replace(self.c.claims, topology=t)))
        self._assert_violation(c2, "pinned requires degree and collective_plan")

    def test_invariant_requires_proof(self):
        t = self.c.claims.topology + (TopologyClaim("EP", "invariant", domain=(1, 8)),)
        c2 = _rehash(dataclasses.replace(self.c, claims=dataclasses.replace(self.c.claims, topology=t)))
        self._assert_violation(c2, "invariant requires a non-empty domain and a proof")

    def test_unsorted_region_tuple(self):
        e = CompositionEntry(
            region=("norms.rms", "attention.core"), cases=ALL_CASES, impl=ImplRef("x", 1, "sm90"), route="canonical"
        )
        self._assert_violation(_with_entry(self.c, e), "region tuple must be sorted")

    def test_empty_region(self):
        e = CompositionEntry(region=(), cases=ALL_CASES, impl=ImplRef("x", 1, "sm90"), route="canonical")
        self._assert_violation(_with_entry(self.c, e), "empty region")

    def test_stale_identities(self):
        c2 = dataclasses.replace(self.c, composition=self.c.composition[:-1])
        self._assert_violation(c2, "stored identities do not match")
        with self.assertRaises(ValidationError):
            validate_or_raise(c2)

    def test_unknown_route(self):
        self._assert_violation(_replace_entry(self.c, 1, route="freestyle"), "unknown route")

    def test_empty_composition(self):
        self._assert_violation(_rehash(dataclasses.replace(self.c, composition=())), "empty composition")

    def test_empty_discharge_ref(self):
        idx = next(i for i, e in enumerate(self.c.composition) if e.discharge is not None)
        c2 = _replace_entry(self.c, idx, discharge=EquivalenceProof("equivalence_proof", "  "))
        self._assert_violation(c2, "discharge with an empty ref")

    def test_pinned_degree_below_one(self):
        for degree in (0, -4):
            t = (TopologyClaim("EP", "pinned", degree=degree, collective_plan="none"),)
            self._assert_violation(_with_claims(self.c, topology=t), "is not a deployable degree")

    def test_invariance_over_one_degree(self):
        t = (TopologyClaim("EP", "invariant", domain=(8,), proof="gates/ep"),)
        self._assert_violation(_with_claims(self.c, topology=t), "is a tautology")

    def test_empty_proof_ref(self):
        t = (TopologyClaim("EP", "pinned", degree=8, collective_plan="none", proof=""),)
        self._assert_violation(_with_claims(self.c, topology=t), "empty proof ref")

    def test_non_finite_tolerance_bound(self):
        for bound in ("nan", "inf", "-inf"):
            claim = ToleranceClaim(case_pair=("engine_decode", "trainer_score"), bounds={"max": bound})
            self._assert_violation(_with_claims(self.c, tolerances=(claim,)), "is not a finite threshold")

    def test_tolerance_without_bounds(self):
        claim = ToleranceClaim(case_pair=("engine_decode", "trainer_score"))
        self._assert_violation(_with_claims(self.c, tolerances=(claim,)), "no bounds")

    def test_state_claim_unknown_event(self):
        claim = StateClaim("kv_cache", ("the_vibes_changed",), True, "lifecycle/kv_rebind")
        self._assert_violation(_with_claims(self.c, state=(claim,)), "unknown lifecycle event(s)")

    def test_unknown_schema_version(self):
        self._assert_violation(_rehash(dataclasses.replace(self.c, schema_version="99")), "unsupported schema_version")


if __name__ == "__main__":
    unittest.main()
