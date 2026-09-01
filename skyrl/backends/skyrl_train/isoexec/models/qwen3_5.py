"""Qwen3.5-35B-A3B (hybrid GDN + MoE) composition, declared as a ``ModelProfile``.

The six structural facts here are what ``policy.derive_selections`` turns into the contract; there
are no exceptions to policy for this model. Sites are TF=trainer_fwd, TS=trainer_score,
EP=engine_prefill, ED=engine_decode.
"""

from __future__ import annotations

from .policy import build_selections
from .profile import (
    SCORE_SOFTMAX,
    ModelProfile,
    RouterProfile,
    StateFact,
    ToleranceFact,
    TopologyAxisFact,
    TrainerNcclAdmission,
)

MODEL = "qwen3.5-35b-a3b"

PROFILE = ModelProfile(
    model=MODEL,
    # The HF classes megatron-bridge registers for this family. These are the dispatch key; the name
    # patterns below are only the no-config.json fallback, and a missing class here disables the census.
    architectures=(
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3NextForCausalLM",
    ),
    name_patterns=("qwen3.5", "qwen3_5", "a3b", "qwen3-next"),
    has_gdn=True,
    has_moe=True,
    zero_centered_norms=True,
    tensor_parallel=True,
    router=RouterProfile(score_function=SCORE_SOFTMAX, expert_bias=False),
    gdn_kernel="recurrent",
    ssm_cache_dtype="float32",
    trainer_nccl_admissions=(
        TrainerNcclAdmission(
            algo=None,
            min_channels=None,
            max_channels="8",
            premise_contract="tp",
            evidence=(
                "Aug-3 PIN0 measurements used an earlier PIK composition. Aug-15 U.7 observed the "
                "current PIK+MOE_PIK_FC2+TRAINER_SP composition with SP on, but was not a clean "
                "one-flag PIN A/B. Admission combines those forward observations with the "
                "centralized nccl_channel_budget.py ownership derivation; backward is curve-gated."
            ),
        ),
    ),
    # Proven parallelism envelopes -> contract TopologyClaims. Each value is grounded in a
    # colocated recorded gate, not asserted here:
    #   TP  -- the pik leaf-tree row-parallel output is bitwise-identical to the TP=1 reference at
    #          every world in the domain (divisors of pik_leaves=8), asserted by the named gate.
    #   SP  -- trainer sequence parallelism on/off is the same expression: RS == AR-then-slice,
    #          bitwise, on both transports, at the tested TP sizes.
    #   PP  -- pinned 1: no pipeline entry anywhere in the composition; the engine adapter forces
    #          pipeline_model_parallel_size=1 (runtimes/vllm/gptmodel_vllm.py).
    #   CP  -- pinned 1: the GDN packed-meta shim keeps every cp_size!=1 path unreachable until
    #          the CP=1 tautology is discharged (the named gate pins that unreachability).
    # EP  -- the composition is EP-invariant: the two sides may run different expert-parallel
    #        degrees (live: trainer EP=8, engine EP=1) without moving forward bits. Not declared
    #        as a TopologyAxisFact here only because an `invariant` axis requires a repo-relative
    #        proof ref naming the gate that measured it (profile.TopologyAxisFact refuses an
    #        unproven domain); add the axis once that gate has a path in this tree.
    topology=(
        TopologyAxisFact(
            axis="TP",
            kind="invariant",
            domain=(1, 2, 4, 8),
            proof="ops/collectives/tests/test_bf16_leaf_scheme_cpu.py",
        ),
        TopologyAxisFact(
            axis="SP",
            kind="invariant",
            domain=(0, 1),
            proof="ops/collectives/tests/test_tree_reduce_scatter_cpu.py",
        ),
        TopologyAxisFact(axis="PP", kind="pinned", degree=1, collective_plan="none"),
        TopologyAxisFact(
            axis="CP",
            kind="pinned",
            degree=1,
            collective_plan="none",
            proof="ops/gdn/tests/test_packed_meta_cache_cpu.py",
        ),
    ),
    # Engine state lifecycle -> StateClaims. DECLARATIONS of hooks that already run, each ref
    # naming the implementing code (checked for existence by the colocated core test):
    #   engine_prefix_cache -- flush-on-sync invariant (lifecycle ordering check; the hard backstop
    #       is the prefix-hit-without-held-state REFUSE in ops/gdn/gdn_recurrent_state.py).
    #   gdn_recurrent_state -- sleep/wake re-allocates kv storage, so the state core rebinds on a
    #       (data_ptr, shape) key change every call; weight sync bumps the sync epoch.
    states=(
        StateFact(
            state_id="engine_prefix_cache",
            invalidated_by=("weight_sync",),
            replay_safe=True,
            ref="lifecycle/ordering.py::check_prefix_cache_flush_on_sync",
        ),
        StateFact(
            state_id="gdn_recurrent_state",
            invalidated_by=("weight_sync", "sleep_wake"),
            replay_safe=True,
            ref="runtimes/vllm/gdn_engine_patch.py::_get_layer_state",
        ),
    ),
    # The controller's forward acceptance gate (rollout logprobs vs identical-weight trainer
    # scoring) -> a ToleranceClaim. Values are the qualified thresholds the gate enforced
    # (skyrl/train/utils/trainer_utils.py); the gate now READS them back from this claim wherever
    # a contract exists, so the envelope is contract data, not a controller constant.
    #
    # These are the RECURRENT variant's bounds, and they are the pre-rowinv envelope: the live
    # evidence for the exact-zero claim (below, on CPR_PROFILE) was produced under
    # SKYRL_ISOEXEC_GDN_KERNEL=cpr, and evidence licenses only the composition that
    # produced it. Tighten this to 0.0 when a recurrent-variant run serves it, not before.
    tolerances=(
        ToleranceFact(
            case_pair=("engine_decode", "trainer_score"),
            bounds=(("abs_diff_max_max", "1.0e-4"), ("abs_diff_mean_max", "1.0e-5")),
        ),
    ),
    notes="hybrid GDN+MoE; trainer alltoall/EP=8/ETP=1 against engine no-gather/EP=1/ETP=8",
)

# No departures from policy. Kept explicit so absence of overrides is machine-visible.
EXCEPTIONS: dict = {}

# The second live variant. ``gdn_kernel`` decides TWO ops: the ``gdn.core`` kernel pin, and the
# ``gdn.state`` impl -- cpr owns its own pools and the engine refuses ``GDN_NATIVE_STATE=1``
# under it, so ``native_kv_cache`` is not selectable there. Declaring the variant here gives it a
# contract entry and an identity of its own.
#
# ``build()`` selects between the two by ``SKYRL_ISOEXEC_GDN_KERNEL`` when no profile is passed:
# both runtimes read the same forwarded env var, so a matched pair derives the same variant and a
# one-sided flip is a hash mismatch that refuses at the handshake.
#
# It also carries the only tolerance claim proven at EXACT ZERO. The leaf-tree logprob makes the
# engine's and the trainer's denominator the same expression at every row count and TP degree, so
# the gate's admitted envelope is not "small" but "no difference at all": measured on this
# composition over 18 gate steps, every step
# reported mean=0.000e+00 max=0.000e+00 with exact=n/n on ~4M tokens per step. A single ULP now
# fails the gate -- which is the point: with the denominator pinned there is no floor left for a
# real regression to hide under.
CPR_PROFILE = PROFILE.with_overrides(
    gdn_kernel="cpr",
    tolerances=(
        ToleranceFact(
            case_pair=("engine_decode", "trainer_score"),
            bounds=(("abs_diff_max_max", "0.0"), ("abs_diff_mean_max", "0.0")),
        ),
    ),
)

# The kernel names this model has a DECLARED variant for. "chunk" is absent by construction, not by
# omission: the native fused core's schedule admits kernel=OneOf("recurrent", "cpr")
# (ops/gdn/_register.py), so a chunk composition has no impl to name and cannot be pinned.
PROFILE_BY_KERNEL = {"recurrent": PROFILE, "cpr": CPR_PROFILE}

# UN-SHARDED widths, derived from the checkpoint's config (NOT back-derived from the legacy
# cuBLASLt shape table, which omits the gate projections).
HIDDEN = 2048
LAYERS = 40
GDN_IN_PROJ = 12352  # in_proj_qkvzba: q 16*128 + k 16*128 + v 32*128 + z 32*128 + b 32 + a 32
ATTN_QKV = 9216  # q 16*256 + k 2*256 + v 2*256 + gate 16*256 -- Qwen3.5 gates its attention output
MOE_FFN = 512  # shared expert; SwiGLU makes fc1 2x this
NUM_EXPERTS = 256
VOCAB = 248320  # vocab_size from the checkpoint's config.json
# `GemmSite.calls` is passes per forward PER TOKEN, not launches: `SKYRL_ISOEXEC_SPLIT_LM_HEAD=1`
# chunks this projection over the token axis, so the chunks partition tokens and each crosses once.
LM_HEAD_PASSES_PER_TOKEN = 1
# The hybrid's layer mix, used only to rank the coverage banner (3 GDN : 1 attention is the shipped ratio).
GDN_LAYERS = 30
ATTN_LAYERS = 10


def gemm_census(*, tp: int, etp: int) -> list:
    """Every dense 2-D GEMM this model runs through ``batch_invariant.matmul_persistent``.

    The row-parallel sites (GDN out_proj, attn o_proj, shared-expert fc2) are absent: under
    ``SKYRL_ISOEXEC_PIK=1`` pik rebinds ``RowParallelLinear.forward`` and calls cuBLASLt directly,
    never reaching ``aten::mm``. The routed experts are absent because the fused MoE path consumes
    them as batched/indexed kernels.
    """
    from ..ops.mm.mm_shapes import GemmSite, dtype_of

    bf16, fp32 = dtype_of("bf16"), dtype_of("fp32")
    tp, etp = int(tp), int(etp)
    del etp  # no matmul_persistent site here is ETP-divided; the shared expert rides attention TP
    return [
        GemmSite("gdn.in_proj_qkvzba", HIDDEN, GDN_IN_PROJ // tp, bf16, "tp", GDN_LAYERS),
        GemmSite("attn.qkv", HIDDEN, ATTN_QKV // tp, bf16, "tp", ATTN_LAYERS),
        GemmSite("moe.router", HIDDEN, NUM_EXPERTS, fp32, "none", LAYERS),
        GemmSite("shexp.gate", HIDDEN, 1, bf16, "none", LAYERS),
        GemmSite("shexp.fc1", HIDDEN, 2 * MOE_FFN // tp, bf16, "tp", LAYERS),
        GemmSite("lm_head", HIDDEN, VOCAB // tp, bf16, "tp", LM_HEAD_PASSES_PER_TOKEN),
    ]


def build(registry, *, arch=None, profile=None):
    """Build the Qwen3.5 ExecutionContract against a registry.

    ``profile`` selects the variant explicitly; by default it follows ``SKYRL_ISOEXEC_GDN_KERNEL``
    through the same parser the read sites use (``core/gdn_kernel_env``), so the declaration
    matches what installs. Same derivation, different hash, so a one-sided flip refuses to run.
    """
    from ..core.arch import ARCH
    from ..core.contract_build import ContractBuildError, build_execution_contract
    from ..core.gdn_kernel_env import (
        KERNEL_ENV,
        TRAINER_KERNEL_ENV,
        gdn_kernel_mode,
        gdn_trainer_kernel_override,
    )

    if profile is None:
        kernel = gdn_kernel_mode()
        if kernel not in PROFILE_BY_KERNEL:
            raise ContractBuildError(
                f"{KERNEL_ENV}={kernel!r} is the kernel the GDN read sites will run, but this "
                f"model declares no profile variant for it (declared: {sorted(PROFILE_BY_KERNEL)}). "
                f"Declaring one of the others anyway would hash a composition that is not the one "
                f"executing, identically on both runtimes. Export {KERNEL_ENV}=recurrent or "
                f"{KERNEL_ENV}=cpr, or pass the variant explicitly."
            )
        profile = PROFILE_BY_KERNEL[kernel]
    override = gdn_trainer_kernel_override()
    if override is not None and override != profile.gdn_kernel:
        raise ContractBuildError(
            f"{TRAINER_KERNEL_ENV}={override!r} runs a different delta-rule kernel on the trainer "
            f"than the {profile.gdn_kernel!r} this contract declares, and the contract pins ONE "
            f"gdn.core kernel for all four sites -- it has no way to say 'trainer runs a different "
            f"function'. Both sides would still hash identically, so nothing downstream would "
            f"catch it. That ablation is not a IsoExec configuration; unset {TRAINER_KERNEL_ENV}."
        )
    return build_execution_contract(
        registry,
        build_selections(profile, EXCEPTIONS),
        arch=arch or ARCH,
        model=MODEL,
        topology=profile.topology,
        states=profile.states,
        tolerances=profile.tolerances,
    )
