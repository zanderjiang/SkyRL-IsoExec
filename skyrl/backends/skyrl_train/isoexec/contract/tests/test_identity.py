"""Identity semantics: rotate on bit-relevant change, ignore everything else."""

import dataclasses
import unittest

from skyrl.backends.skyrl_train.isoexec.contract import (
    CompositionEntry,
    EquivalenceProof,
    ImplRef,
    compute_identities,
)
from skyrl.backends.skyrl_train.isoexec.contract.tests.fixtures import (
    ALL_CASES,
    base_contract,
)


def _replace_entry(c, idx, **kw):
    e = dataclasses.replace(c.composition[idx], **kw)
    comp = c.composition[:idx] + (e,) + c.composition[idx + 1 :]
    return dataclasses.replace(c, composition=comp)


class TestIdentity(unittest.TestCase):
    def setUp(self):
        self.c = base_contract()
        self.ids = compute_identities(self.c)

    def test_rotates_on_constant_change(self):
        c2 = _replace_entry(self.c, 0, constants={"leaves": 8, "leaf_dtype": "fp32"})
        self.assertNotEqual(compute_identities(c2).numerical_policy, self.ids.numerical_policy)

    def test_rotates_on_impl_version_bump(self):
        c2 = _replace_entry(self.c, 0, impl=ImplRef("pik_leaf_tree", 3, "sm90"))
        self.assertNotEqual(compute_identities(c2).numerical_policy, self.ids.numerical_policy)

    def test_rotates_on_route_change(self):
        c2 = _replace_entry(self.c, 1, route="canonical")
        self.assertNotEqual(compute_identities(c2).numerical_policy, self.ids.numerical_policy)

    def test_rotates_on_repartition(self):
        # Fuse moe.combine and moe.gate_scale into one group: same vocabulary, new partition.
        merged = CompositionEntry(
            region=("moe.combine", "moe.gate_scale"),
            cases=ALL_CASES,
            impl=ImplRef("fused_combine_gate", 1, "sm90"),
            route="composition_defining",
            artifact="sha256:feedbeef",
        )
        keep = tuple(e for e in self.c.composition if e.region not in (("moe.combine",), ("moe.gate_scale",)))
        c2 = dataclasses.replace(self.c, composition=keep + (merged,))
        ids2 = compute_identities(c2)
        self.assertNotEqual(ids2.numerical_policy, self.ids.numerical_policy)
        # The op vocabulary is unchanged, so the model still MEANS the same thing.
        self.assertEqual(ids2.semantic, self.ids.semantic)

    def test_stable_under_deployment_change(self):
        dep_idx = next(i for i, e in enumerate(self.c.composition) if e.half == "deployment")
        c2 = _replace_entry(
            self.c,
            dep_idx,
            impl=ImplRef("nccl_pinned", 1, "sm90"),
            discharge=EquivalenceProof("neutrality_proof", "gates/other_proof"),
        )
        ids2 = compute_identities(c2)
        self.assertEqual(ids2.numerical_policy, self.ids.numerical_policy)
        self.assertNotEqual(ids2.deployment, self.ids.deployment)

    def test_stable_under_entry_reordering(self):
        c2 = dataclasses.replace(self.c, composition=tuple(reversed(self.c.composition)))
        self.assertEqual(compute_identities(c2), self.ids)

    def test_stable_under_case_order_within_entry(self):
        c2 = _replace_entry(self.c, 0, cases=tuple(reversed(ALL_CASES)))
        self.assertEqual(compute_identities(c2).numerical_policy, self.ids.numerical_policy)

    def test_ignores_stored_identities(self):
        c2 = dataclasses.replace(self.c, identities=dataclasses.replace(self.ids, semantic="x"))
        self.assertEqual(compute_identities(c2), self.ids)


if __name__ == "__main__":
    unittest.main()
