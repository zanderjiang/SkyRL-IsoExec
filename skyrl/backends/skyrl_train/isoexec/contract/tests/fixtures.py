"""A small realistic contract for the tests."""

import dataclasses

from skyrl.backends.skyrl_train.isoexec.contract import (
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
    compute_identities,
)

ALL_CASES = ("engine_decode", "engine_prefill", "trainer_fwd", "trainer_recompute", "trainer_score")
TRAINER = ("trainer_fwd", "trainer_recompute", "trainer_score")
ENGINE = ("engine_prefill", "engine_decode")


def base_contract() -> ExecutionContract:
    cases = (
        ExecutionCase("trainer_fwd", "trainer", "grad", "fresh", "packed_thd"),
        ExecutionCase("trainer_recompute", "trainer", "grad", "recompute", "packed_thd"),
        ExecutionCase("trainer_score", "trainer", "no_grad", "fresh", "packed_thd"),
        ExecutionCase("engine_prefill", "engine", "no_grad", "resumable", "varlen"),
        ExecutionCase(
            "engine_decode",
            "engine",
            "no_grad",
            "continued",
            "T=1,B<=512",
            ("cudagraph_capturable", "host_free", "address_stable"),
        ),
    )
    composition = (
        # Route B: one sealed artifact serves every case on both sides.
        CompositionEntry(
            region=("moe.combine",),
            cases=ALL_CASES,
            impl=ImplRef("pik_leaf_tree", 2, "sm90"),
            route="composition_defining",
            constants={"leaves": 4, "leaf_dtype": "fp32"},
            artifact="sha256:1f2e3d4c",
        ),
        # Fused GDN group: one protected manual kernel owning three logical ops.
        CompositionEntry(
            region=("gdn.core", "gdn.gating", "norms.l2"),
            cases=ALL_CASES,
            impl=ImplRef("native_fused_sigmoid", 1, "sm90"),
            route="protected",
        ),
        # Deliberately asymmetric weights layout; both entries carry the proof.
        CompositionEntry(
            region=("moe.weights",),
            cases=TRAINER,
            impl=ImplRef("stacked_bmm", 1, "sm90"),
            route="canonical",
            discharge=EquivalenceProof("equivalence_proof", "gates/moe_weights_byte_equiv"),
        ),
        CompositionEntry(
            region=("moe.weights",),
            cases=ENGINE,
            impl=ImplRef("fused_buffer", 2, "sm90"),
            route="canonical",
            discharge=EquivalenceProof("equivalence_proof", "gates/moe_weights_byte_equiv"),
        ),
        # Route A: compiled twin, bit-equal to a named reference over admitted shapes.
        CompositionEntry(
            region=("moe.gate_scale",),
            cases=ALL_CASES,
            impl=ImplRef("autofuse_gate_scale", 1, "sm90"),
            route="reference_preserving",
            constants={"scale": BitPattern("0x3f800000", "fp32")},
            reference=ImplRef("eager_gate_scale", 1, "sm90"),
            coverage=Coverage("enumerated_shapes", "ledger v10 shapes"),
        ),
        # Deployment-half: proven not to move bits, engine-side only.
        CompositionEntry(
            region=("collectives.nccl_pin",),
            cases=ENGINE,
            impl=ImplRef("nccl_unpinned", 1, "sm90"),
            route="canonical",
            half="deployment",
            discharge=EquivalenceProof("neutrality_proof", "gates/nccl_pin_engine_neutral"),
        ),
    )
    claims = Claims(
        topology=(
            TopologyClaim("TP", "invariant", domain=(1, 4, 8), proof="gates/tp_invariance"),
            TopologyClaim("PP", "pinned", degree=1, collective_plan="none"),
            TopologyClaim("CP", "pinned", degree=1, collective_plan="none"),
        ),
        state=(StateClaim("kv_cache", ("weight_sync", "sleep_wake"), True, "lifecycle/kv_rebind"),),
        tolerances=(
            ToleranceClaim(
                case_pair=("engine_decode", "trainer_score"),
                bounds={"mean": "1e-6", "max": "5e-6"},
                attributed_to=("attention.softmax_decomposition",),
            ),
        ),
    )
    c = ExecutionContract(
        schema_version="2",
        model=ModelRef("qwen3_5", ("Qwen3_5MoeForConditionalGeneration",), "profiles/qwen3_5"),
        identities=Identities("", "", ""),
        cases=cases,
        composition=composition,
        claims=claims,
    )
    return dataclasses.replace(c, identities=compute_identities(c))
