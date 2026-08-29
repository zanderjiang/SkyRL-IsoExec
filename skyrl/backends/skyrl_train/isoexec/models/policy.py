"""The derivation: ``ModelProfile`` -> ``{(op, site) -> Selection}``. Model-independent.

The site policy, stated once: the trainer runs the grad-capable reference at both of its sites, and
the engine runs the fused twin wherever one exists and declares
``capabilities["bitwise_equal_to"]`` naming that reference. Where no such twin exists both sides run
the same impl and the op owes no equivalence proof; ``policy_matches_registry_capabilities``
re-derives that rule from the registry so the two cannot drift.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from ..core.contract_build import DEPLOYMENT, FUNCTION, ContractBuildError
from .profile import SCORE_SOFTMAX, ModelProfile, ProfileError, RouterProfile

Key = Tuple[str, str]  # (op, site)


@dataclass(frozen=True)
class Selection:
    """One resolved (op, site) selection; identity is impl@version x arch.

    ``pinned_constants`` (autotune index, boundary dtypes, block sizes, leaf counts) move bits, so
    they are hashed into the contract's function half. ``neutrality_proof`` is mandatory for a
    DEPLOYMENT selection: it points at the gate run that proved this selection bitwise-neutral.
    """

    impl_id: str
    version: int = 1
    pinned_constants: Dict[str, object] = field(default_factory=dict)
    classification: str = FUNCTION
    neutrality_proof: Optional[str] = None

    def __post_init__(self) -> None:
        if self.classification not in (FUNCTION, DEPLOYMENT):
            raise ContractBuildError(
                f"unknown classification {self.classification!r}; must be one of " f"['{DEPLOYMENT}', '{FUNCTION}']"
            )
        if self.classification == DEPLOYMENT and not self.neutrality_proof:
            raise ContractBuildError(
                "deployment classification requires a recorded neutrality_proof (a run id / gate "
                "result pointer). A neutrality proof licenses exactly the entry it was measured "
                "on; refusing to classify without one."
            )
        if self.classification == FUNCTION and self.neutrality_proof:
            # A function-half selection is hashed regardless, so carrying a proof is a classification bug.
            raise ContractBuildError(
                "a function-half entry must not carry a neutrality_proof; either it is proven "
                "neutral (classify it deployment) or it is not (leave it function)."
            )


# Site vocabulary.
TRAINER = ("trainer_fwd", "trainer_score")
ENGINE = ("engine_prefill", "engine_decode")
ALL_SITES = TRAINER + ENGINE

# The engine-unpin neutrality proof is entry-scoped: the trainer selection stays FUNCTION-classified
# because changing it changes the trainer composition and reassociates backward reductions.
NCCL_ENGINE_UNPIN_PROOF = "engine-unpin measured bitwise-neutral (perf-attribution A/B)"

# Sentinel for an exception that DELETES a derived key (the model has no such site after all).
DROP = object()


def entry(impl: str, version: int = 1, pinned: Optional[dict] = None, cls: str = FUNCTION, proof=None) -> Selection:
    """Build one ``Selection``. Public: model files spell their exceptions with it."""
    return Selection(
        impl_id=impl, version=version, pinned_constants=pinned or {}, classification=cls, neutrality_proof=proof
    )


_e = entry  # internal shorthand used by the derivation below


# Eligibility predicates: ops declare capabilities, models declare choices.


def router_is_fusable(router: RouterProfile) -> bool:
    """Is the engine's fused ``moe.router`` (``fused_o2``) eligible for this router shape?

    The kernel carries the same guard internally and declines when it fails; consulting it here keeps
    the contract from naming an impl that will decline at install time. There is no fused sigmoid twin
    at the same bits, and there cannot be one.
    """
    return router.score_function == SCORE_SOFTMAX and not router.expert_bias


def mm_impl_for_process() -> str:
    """Which global GEMM provider this process will install.

    Flag-conditional on purpose: if only one process enables ``SKYRL_ISOEXEC_MM_CUBLASLT`` the two
    contract hashes diverge and the weight-sync handshake refuses to run, rather than silently
    mixing GEMM providers across the two forwards.
    """
    return "cublaslt_pinned" if os.environ.get("SKYRL_ISOEXEC_MM_CUBLASLT", "0") == "1" else "triton_batch_invariant"


#: The fixed per-leaf tile of the rowinv Kahan exp sum. A contract constant, not a tuning knob:
#: it must equal the BLOCK ``ops/logprobs/rowinv.py`` compiles with, and the registry declares the
#: same literal, so a drifted kernel refuses at pin validation instead of hashing a schedule the
#: process does not run.
ROWINV_BLOCK = 4096


def derive_selections(profile: ModelProfile) -> Dict[Key, Selection]:
    """``ModelProfile`` -> the complete ``{(op, site): Selection}`` selection set.

    Every ``(op, site)`` this architecture installs gets an explicit entry, and an op the architecture
    does not run gets no key at all -- absence means "no such site", never "default".
    """
    s: Dict[Key, Selection] = {}

    def put(op: str, sites: Tuple[str, ...], **kw) -> None:
        for site in sites:
            s[(op, site)] = _e(**kw)

    # Global GEMM.
    put("mm", ALL_SITES, impl=mm_impl_for_process())

    # Attention.
    # torch FA3 varlen on the trainer (grad-capable reference), vLLM's FA3 varlen on the engine at the
    # same num_splits=1 pin -- that pin is the contract, since split-K is what makes decode != prefill.
    put("attention.varlen", TRAINER, impl="varlen_custom", pinned={"num_splits": 1})
    put("attention.varlen", ENGINE, impl="vllm_flash_ns1", pinned={"num_splits": 1, "fa_version": 3})

    # Rotary.
    put("rope.rope", TRAINER, impl="eager")
    put("rope.rope", ENGINE, impl="fused")

    # Norms.
    # The fused engine twin exists only for the zero-centred form: it rebinds forward on
    # ZeroCenteredTorchRMSNorm instances, of which a plain-RMS model builds none.
    if profile.zero_centered_norms:
        put("norms.rms", TRAINER, impl="eager_zero_centered")
        put("norms.rms", ENGINE, impl="fused")
    else:
        put("norms.rms", ALL_SITES, impl="eager_torch_rms")

    # Logprobs.
    # The leaf-tree impl is ONE function at all four sites -- that is the whole point (aten's
    # vocab sum is neither row-count- nor TP-invariant, so no twin of it can serve both runtimes),
    # so there is no engine-only variant and no site keeps the incumbent. Unconditional: this is
    # the composition, not a choice, and the aten-order selection it replaced is gone rather than
    # flag-reachable. FUNCTION-classified (the default): it moves bits, so it is hashed, never
    # proof-carried.
    put(
        "logprobs.log_softmax",
        ALL_SITES,
        impl="rowinv_leaftree",
        # The pins ARE the function: G fixes the leaf boundaries and the combine tree, BLOCK the
        # per-leaf tile, and the accumulator names the per-leaf summation.
        pinned={"leaves": profile.pik_leaves, "block": ROWINV_BLOCK, "accum": "kahan_fp32"},
    )
    # The lm_head row slice is engine-only either way: the trainer scores every position, the
    # engine sampled rows -- a pure row selection, orthogonal to which denominator schedule runs.
    put("logprobs.lm_head_slice", ENGINE, impl="sampled_rows")

    # GatedDeltaNet.
    if profile.has_gdn:
        # The native fused core subsumes gdn.l2norm and gdn.gating, so those get no standalone entry;
        # the registry's `subsumes` field drives the adapter's derived passthrough.
        put("gdn.core", ALL_SITES, impl="native_fused_sigmoid", pinned={"kernel": profile.gdn_kernel})
        # Distinct kernels per site, parity-proven: the fn form at prefill/trainer, update at decode.
        put("gdn.conv", TRAINER + ("engine_prefill",), impl="causal_conv1d_fn")
        put("gdn.conv", ("engine_decode",), impl="causal_conv1d_update")
        # Engine-only state, and which object owns it is decided by the GDN mode: under
        # `cpr` the engine refuses GDN_NATIVE_STATE=1 and CprGDN raises on a
        # handed-in state tensor, so `native_kv_cache` is not selectable there.
        if profile.gdn_kernel == "cpr":
            put(
                "gdn.state",
                ENGINE,
                impl="cpr_pool",
                pinned={
                    "ssm_state_dtype": profile.ssm_cache_dtype,
                    # The trainer's segmented scan and this pool's boundary pass are the same
                    # function only at the same C, so C is pinned rather than assumed.
                    "chunk_size": profile.gdn_chunk_size,
                },
            )
        else:
            # fp32 SSM cache pinned (the boundary-dtype guard).
            put("gdn.state", ENGINE, impl="native_kv_cache", pinned={"ssm_cache_dtype": profile.ssm_cache_dtype})
        # The GDN gated output norm: fused engine twin of the eager reference.
        put("norms.gated_out", TRAINER, impl="eager")
        put("norms.gated_out", ENGINE, impl="fused")

    # MoE.
    if profile.has_moe:
        r = profile.router
        if router_is_fusable(r):
            # Site-asymmetric, membership-equivalent. No pins: the softmax router's shape lives in
            # the config, not in the rounding schedule.
            put("moe.router", TRAINER, impl="deterministic")
            put("moe.router", ENGINE, impl="fused_o2")
        else:
            # One impl at every site, and the pins are part of the function: expert bias, top-k,
            # scaling factor and epsilon each move bits; group_topk=None records a bypassed path.
            put(
                "moe.router",
                ALL_SITES,
                impl="deterministic_sigmoid_bias",
                pinned={
                    "topk": r.topk,
                    "experts": r.num_experts,
                    "scaling_factor": r.scaling_factor,
                    "group_topk": r.group_topk,
                    "eps": r.eps,
                },
            )
        put("moe.dispatch", ALL_SITES, impl="index_build")
        # Site-asymmetric, max|diff|=0 proven. Fusing both sides loses on the backward: the fused
        # forward is padding-free and every bmm-shaped backward must rebuild that padding.
        put("moe.experts", TRAINER, impl="batched_bmm")
        put("moe.experts", ENGINE, impl="fused")
        # Deliberately NOT ``_pik(profile)``: the MoE combine's leaves are per-rank fc2 partials and
        # stay fp32 whatever the dense plan's leaf dtype is (its bf16 wire is a lossless narrowing,
        # not a leaf rounding). Sharing _pik here would refuse a sound dense bf16-leaf deployment.
        put("moe.combine", ALL_SITES, impl="pik_leaf_tree", pinned={"leaves": profile.pik_leaves, "leaf_dtype": "fp32"})
        put("moe.weights", ENGINE, impl="fused_buffer")
        put("moe.epilogue", ENGINE, impl="fused_swiglu")
        put("moe.blockmap", ENGINE, impl="fused")

    # Collectives.
    # Present iff TP>1: at TP=1 they do not install, so they must carry no entry at all.
    if profile.tensor_parallel:
        put("collectives.tree_all_reduce", ALL_SITES, impl="pik_tree", pinned=_pik(profile))
        put("collectives.row_parallel", ALL_SITES, impl="pik_tree")
        # Trainer pinning is an explicit FUNCTION composition choice. Read the same flag as the
        # runtime installer so an unpinned arm is represented in the contract rather than tolerated
        # as a fingerprint mismatch; the default stays pinned. The engine entry is DEPLOYMENT-neutral.
        from ..ops.collectives.nccl_identity import (
            PINNED,
            constants_for_impl,
            requested_engine_identity,
            requested_trainer_identity,
        )

        try:
            trainer_nccl = requested_trainer_identity()
        except ValueError as exc:
            raise ProfileError(f"{profile.model!r}: invalid trainer NCCL composition request: {exc}") from exc
        # Always declare the exact installed tuple: the trainer pin is FUNCTION-half, so its
        # constants belong in the hash (the live fingerprint compares them verbatim).
        trainer_pins = constants_for_impl(trainer_nccl)
        if trainer_nccl != PINNED:
            requested = trainer_pins
            admission = next(
                (item for item in profile.trainer_nccl_admissions if item.effective_constants() == requested),
                None,
            )
            if admission is None:
                raise ProfileError(
                    f"{profile.model!r}: trainer NCCL {trainer_nccl!r} requested with effective "
                    f"ALGO/MIN/MAX={requested!r}, but the profile has no admission for that exact tuple"
                )
            from ..ops.collectives.nccl_channel_budget import unmet_preconditions

            missing = unmet_preconditions(admission.premise_contract)
            if missing:
                raise ProfileError(
                    f"{profile.model!r}: trainer NCCL {trainer_nccl!r} requires centralized "
                    f"{admission.premise_contract!r} composition premises; unmet: {', '.join(missing)}"
                )
        put("collectives.nccl_pin", TRAINER, impl=trainer_nccl, pinned=trainer_pins)

        try:
            engine_nccl = requested_engine_identity()
        except ValueError as exc:
            raise ProfileError(f"{profile.model!r}: invalid engine NCCL composition request: {exc}") from exc
        engine_pins = constants_for_impl(engine_nccl) if engine_nccl != "unpinned" else None
        put(
            "collectives.nccl_pin",
            ENGINE,
            impl=engine_nccl,
            pinned=engine_pins,
            cls=DEPLOYMENT,
            proof=NCCL_ENGINE_UNPIN_PROOF,
        )

    return s


def _pik(profile: ModelProfile) -> dict:
    return {"leaves": profile.pik_leaves, "leaf_dtype": profile.pik_leaf_dtype}


def apply_exceptions(selections: Dict[Key, Selection], exceptions: Optional[dict]) -> Dict[Key, Selection]:
    """Apply a model file's exception list on top of the derived selections.

    An exception key is ``(op, site)`` or ``(op, "*")`` (all sites the derivation produced for that
    op); the value is a ``Selection``, or ``DROP`` to delete the key. Every exception changes the
    contract's numerical_policy identity, and therefore the gate signature key.
    """
    if not exceptions:
        return selections
    out = dict(selections)
    for key, val in exceptions.items():
        op, site = key
        # For an exact site the key must already exist: an exception overrides, and may never
        # introduce a site the derivation says the architecture does not have.
        targets = [k for k in out if k[0] == op] if site == "*" else [(op, site)] if (op, site) in out else []
        if not targets:
            raise ValueError(
                f"exception names ({op!r}, {site!r}) but the derivation produced no such key. An "
                f"exception may only override or drop something policy actually derived -- if the "
                f"model needs a key policy does not derive, the PROFILE is wrong, not the contract."
            )
        for t in targets:
            if val is DROP:
                out.pop(t, None)
            else:
                out[t] = val
    return out


def build_selections(profile: ModelProfile, exceptions: Optional[dict] = None) -> Dict[Key, Selection]:
    """The one call a ``models/*.py`` file makes: derive from the profile, then apply its exceptions."""
    return apply_exceptions(derive_selections(profile), exceptions)


def policy_matches_registry_capabilities(profile: ModelProfile, registry) -> list:
    """Re-derive the site policy from the registry and report where this module disagrees.

    An op whose sites do not all resolve to one impl must carry an equivalence claim, in one of three
    forms, kept distinct so a pair is never pushed into the wrong one:

      1. ``capabilities["bitwise_equal_to"] = <a trainer impl id>`` -- a byte-equal engine twin.
      2. ``capabilities["equivalence_proof"] = "<pointer to the proving gate>"`` -- distinct kernels
         exact for a reason other than kernel equality (``gdn.conv``'s fn/update pair, which does
         NOT agree bitwise and rests on split-exactness at any token boundary).
      3. A DEPLOYMENT-classified engine entry carrying a ``neutrality_proof``, scoped to that entry.

    Returns a list of human-readable disagreements (empty == agreement).
    """
    problems = []
    sel = derive_selections(profile)
    ops = {op for op, _ in sel}
    for op in sorted(ops):
        if not registry.has_op(op):
            problems.append(f"{op}: selected but not registered")
            continue
        spec = registry.get_op(op)
        tr = {sel[(op, s)].impl_id for s in TRAINER if (op, s) in sel}
        en = {sel[(op, s)].impl_id for s in ENGINE if (op, s) in sel}
        for impl_id in sorted(tr | en):
            if impl_id not in spec.impls:
                problems.append(f"{op}: selects unregistered impl {impl_id!r}")
        if not tr or not en or tr == en:
            continue  # symmetric, or single-sided: no equivalence obligation to check

        # Form 3 first: a deployment-classified engine entry is licensed by its own neutrality proof.
        engine_entries = [sel[(op, s)] for s in ENGINE if (op, s) in sel]
        if engine_entries and all(e.classification == DEPLOYMENT and e.neutrality_proof for e in engine_entries):
            continue

        # Forms 1 and 2: some engine impl carries a claim (byte-equality, or a named proof).
        claimed = False
        for impl_id in en:
            impl = spec.impls.get(impl_id)
            if not impl:
                continue
            if impl.capabilities.get("bitwise_equal_to") in tr:
                claimed = True
            elif impl.capabilities.get("equivalence_proof"):
                claimed = True
        if not claimed:
            problems.append(
                f"{op}: sites resolve to different impls (trainer={sorted(tr)} engine={sorted(en)}) "
                f"but no engine impl declares capabilities['bitwise_equal_to'] naming a trainer impl, "
                f"nor capabilities['equivalence_proof'], nor is the engine entry deployment-classified "
                f"with a neutrality proof. An op whose sites do not all resolve to one "
                f"impl MUST carry the equivalence proof."
            )
    return problems
