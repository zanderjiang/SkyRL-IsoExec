"""Qwen3.5-35B-A3B (hybrid GDN + MoE) composition, declared as a ``ModelProfile``.

``policy.derive_selections`` turns these structural facts into the contract; no exceptions here.
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
    # The HF classes megatron-bridge registers for this family; the dispatch key. The name patterns
    # below are only the no-config.json fallback.
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
    # Proven parallelism envelopes -> contract TopologyClaims, each grounded in the named gate. EP is
    # invariant in practice but undeclared: an `invariant` axis needs a proof ref, and it has none.
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
    # Engine state lifecycle -> StateClaims. Declarations of hooks that already run; each ref names
    # the implementing code.
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
    # The controller's forward acceptance gate -> a ToleranceClaim it reads back from here. These
    # are the recurrent variant's pre-rowinv bounds; the exact-zero evidence below is cpr-only.
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

# The second live variant, with its own contract identity: ``gdn_kernel`` decides both the
# ``gdn.core`` pin and the engine's ``gdn.state``. Its tolerance claim is proven at exact zero.
CPR_PROFILE = PROFILE.with_overrides(
    gdn_kernel="cpr",
    tolerances=(
        ToleranceFact(
            case_pair=("engine_decode", "trainer_score"),
            bounds=(("abs_diff_max_max", "0.0"), ("abs_diff_mean_max", "0.0")),
        ),
    ),
)

# The kernels this model declares a variant for. "chunk" is absent by construction: the fused core's
# schedule admits only recurrent/cpr, so a chunk composition has no impl to name.
PROFILE_BY_KERNEL = {"recurrent": PROFILE, "cpr": CPR_PROFILE}

# Un-sharded widths, from the checkpoint's config.
HIDDEN = 2048
LAYERS = 40
GDN_IN_PROJ = 12352  # in_proj_qkvzba: q 16*128 + k 16*128 + v 32*128 + z 32*128 + b 32 + a 32
ATTN_QKV = 9216  # q 16*256 + k 2*256 + v 2*256 + gate 16*256 -- Qwen3.5 gates its attention output
MOE_FFN = 512  # shared expert; SwiGLU makes fc1 2x this
NUM_EXPERTS = 256
VOCAB = 248320  # vocab_size from the checkpoint's config.json
# `GemmSite.calls` is passes per forward per token, not launches.
LM_HEAD_PASSES_PER_TOKEN = 1
# The hybrid's layer mix, used only to rank the coverage banner.
GDN_LAYERS = 30
ATTN_LAYERS = 10


def gemm_census(*, tp: int, etp: int) -> list:
    """Every dense 2-D GEMM this model runs through ``batch_invariant.matmul_persistent``.

    Row-parallel sites are absent (pik calls cuBLASLt directly), as are the routed experts (the
    fused MoE path consumes them as batched/indexed kernels).
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
    through the same parser the read sites use, so a one-sided flip is a hash mismatch that refuses.
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
