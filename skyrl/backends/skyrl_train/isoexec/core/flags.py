"""The single flag / override table, and the forwarding channels that carry each flag.

A flag reaching one runtime but not the other is a split-brain composition, so every env override
is catalogued with its code default, reading sides, forwarding channels and manifest disposition.
``default`` is the literal the CODE falls back to; use ``default_dynamic`` when it is computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

# Disposition vocabulary.
FUNCTION = "function"
DEPLOYMENT = "deployment"
DIAGNOSTIC = "diagnostic"
DEAD = "dead"

# The three forwarding channels: two Ray-actor allowlists plus ``ISOEXEC_VLLM_ENV``, a dict applied
# in-process before vLLM init.
TRAIN = "train"  # skyrl/train/utils/utils.py  (colocated path; reaches all actors)
ENGINE = "engine"  # skyrl/backends/skyrl_train/inference_engines/utils.py (engine path)
VLLM_ENV = "vllm_env"  # runtimes/vllm/vllm_patches.py ISOEXEC_VLLM_ENV


@dataclass(frozen=True)
class Flag:
    """One env override.

    ``sides`` is which runtimes READ the value; ``forwarded_by`` is the current forwarding reality,
    while ``should_forward`` is whether the flag needs forwarding to be correct at all.
    """

    name: str
    default: str
    sides: Tuple[str, ...]  # subset of {"trainer", "engine", "both"}
    selects: str
    disposition: str
    forwarded_by: Tuple[str, ...] = ()  # subset of {TRAIN, ENGINE, VLLM_ENV}
    should_forward: bool = True
    notes: str = ""
    default_dynamic: bool = False  # code default is COMPUTED, not a literal (see below)

    @property
    def is_latent_split_brain(self) -> bool:
        """True when the flag needs forwarding but no channel forwards it."""
        return self.should_forward and not self.forwarded_by


# Ordered by concern rather than alphabetically, so the split-brain pairs sit together.

FLAGS: List[Flag] = [
    # Master switch.
    Flag(
        "SKYRL_ISOEXEC",
        "0",
        ("both",),
        "master IsoExec enable; gates every forwarding block",
        FUNCTION,
        (TRAIN, ENGINE),
        notes="special-cased: presence gates the whole allowlist loop",
    ),
    # VLLM_ENV channel: delivered by the ISOEXEC_VLLM_ENV dict, not by either Ray-actor allowlist.
    Flag(
        "VLLM_USE_AOT_COMPILE",
        "0",
        ("engine",),
        "disable vLLM AOT compile (batch-invariant setup)",
        DEPLOYMENT,
        (VLLM_ENV,),
        notes="ISOEXEC_VLLM_ENV dict only; in-process pre-init",
    ),
    Flag(
        "NCCL_ALGO",
        "",
        ("both",),
        "deterministic NCCL all-reduce (allreduce:tree) matching the trainer pin",
        FUNCTION,
        (TRAIN, VLLM_ENV),
        notes="ENGINE receives it through ISOEXEC_VLLM_ENV; the TRAINER through a direct env_vars[] assignment gated "
        "on the NCCL pin, not the allowlist loop.",
    ),
    Flag(
        "NCCL_MIN_NCHANNELS",
        "",
        ("both",),
        "NCCL min channels=1 (determinism pin)",
        FUNCTION,
        (TRAIN, VLLM_ENV),
        notes="ENGINE via ISOEXEC_VLLM_ENV; TRAINER via a direct env_vars[] assignment gated on the NCCL pin.",
    ),
    Flag(
        "NCCL_MAX_NCHANNELS",
        "",
        ("both",),
        "NCCL max channels=1 (determinism pin)",
        FUNCTION,
        (TRAIN, VLLM_ENV),
        notes="ENGINE via ISOEXEC_VLLM_ENV; TRAINER via a direct env_vars[] assignment gated on the NCCL pin.",
    ),
    # Spec / attention backend: function-half, must match on both sides.
    Flag(
        "SKYRL_ISOEXEC_LOCAL_SPEC",
        "0",
        ("both",),
        "local no-TE Megatron layer spec on both runtimes",
        FUNCTION,
        (TRAIN, ENGINE),
    ),
    Flag(
        "SKYRL_ISOEXEC_SELECTIVE_TE",
        "0",
        ("trainer",),
        "capability-driven selective TE spec: TE standard ops, local IsoExec rows and experts",
        FUNCTION,
        (TRAIN,),
        notes="Version-pinned and fail-closed; the first admitted profile is hybrid GDN+MoE. The engine stays on the "
        "existing IsoExec composition.",
    ),
    Flag(
        "SKYRL_ISOEXEC_TE_PRIMITIVES",
        "0",
        ("trainer",),
        "TE multi-tensor trainer primitives with the exact local-spec model unchanged",
        FUNCTION,
        (TRAIN,),
        notes="Requires LOCAL_SPEC=1, SELECTIVE_TE=0 and the pinned TE package trio, and fails closed if any TE model "
        "module owns the instantiated graph. Admits only the TE multi-tensor L2/grad-norm implementation.",
    ),
    Flag(
        "VLLM_BATCH_INVARIANT",
        "0",
        ("both",),
        "vLLM batch-invariant kernel path",
        FUNCTION,
        (TRAIN, ENGINE, VLLM_ENV),
        notes="also hard-set to 1 in ISOEXEC_VLLM_ENV (in-process, pre-init)",
    ),
    Flag(
        "SKYRL_ISOEXEC_BATCH_INVARIANT",
        "1",
        ("trainer",),
        "TRAINER half of the batch-invariant kernel path (vLLM's enable_batch_invariant_mode, "
        "applied to the megatron worker's non-attention ops)",
        FUNCTION,
        (TRAIN,),
        notes="Read inside the policy Ray actor, so it must be forwarded for a launch-shell value to take effect. "
        "VLLM_BATCH_INVARIANT is its engine twin; both must move together or an arm measures a split-brain "
        "instead of a contract.",
    ),
    Flag(
        "VARLEN_FORCE_NUM_SPLITS_1",
        "1",
        ("trainer",),
        "varlen attention num_splits=1 -- the core IsoExec pin, ON, and LOAD-BEARING",
        FUNCTION,
        (TRAIN,),
        notes="num_splits=1 is what makes the varlen kernel query-length invariant, i.e. bitwise decode==prefill at "
        "every length; with splits>1 the trainer forward stops matching the rollout. Trainer-side only -- the "
        "engine reaches num_splits=1 through the VLLM_BATCH_INVARIANT-gated flash_attn patch instead.",
    ),
    Flag(
        "SKYRL_ISOEXEC_VARLEN_VLLM_FLASH",
        "1",
        ("engine",),
        "engine attention fast path: vLLM flash_attn_varlen_func(ns1, fa3) instead of torch varlen",
        DEPLOYMENT,
        (TRAIN,),
        notes="Bitwise-interchangeable with the torch kernel, so a pure perf knob; 0 falls back to torch "
        "varlen_attn_out.",
    ),
    Flag(
        "SKYRL_ISOEXEC_MM_TILES_DECODE_BUCKET",
        "0",
        ("both",),
        "opt-in M<1024 decode bucket in mm_tiles (occupancy fix for skinny decode GEMMs)",
        DEPLOYMENT,
        (TRAIN,),
        notes="An explicit exception to mm_tiles M-free keying; every entry is proven bit-identical to stock at every "
        "M. Requires SKYRL_ISOEXEC_MM_TILES=1.",
    ),
    Flag(
        "SKYRL_ISOEXEC_MM_CUBLASLT",
        "0",
        ("both",),
        "pinned non-split-K cuBLASLt dense GEMM (signature-moving; bf16 5 shapes / 6 sites)",
        DEPLOYMENT,
        (TRAIN,),
        notes="Not bit-equal to Triton by contract, so a flip requires a gate re-freeze. Installs at the one site "
        "both runtimes call, after mm_tiles; the fp32 router/lm_head and the N=1 gate stay Triton, and a failed "
        "init self-check falls through to Triton permanently.",
    ),
    Flag(
        "SKYRL_ISOEXEC_MODEL_PATH",
        "",
        ("both",),
        "the model path/id every process resolves its ModelProfile (and dense-GEMM census) from",
        DIAGNOSTIC,
        (TRAIN,),
        notes="Names the model the process is already running so the shape-table coverage report can say which live "
        "shapes a table covers. It selects nothing and cannot move a bit.",
    ),
    Flag(
        "SKYRL_ISOEXEC_MM_FWD_ONLY",
        "0",
        ("trainer",),
        "scope the batch-invariant mm/addmm override to the FORWARD (backward -> cuBLAS)",
        DEPLOYMENT,
        (TRAIN,),
        notes="Backward-only and therefore bitwise-neutral by construction: the predicate returns Triton unless a "
        "backward graph task is executing on this thread, which no forward is. Checkpoint recompute forwards "
        "are explicitly re-pinned to Triton, since they must match the original forward bitwise and the live "
        "gate cannot see them. Fails closed -- no hook means no scoping.",
    ),
    Flag(
        "SKYRL_ISOEXEC_MM_FWD_ONLY_BMM",
        "0",
        ("trainer",),
        "bmm half: scope the batch-invariant aten::bmm override to the FORWARD",
        DEPLOYMENT,
        (TRAIN,),
        notes="The bmm half of SKYRL_ISOEXEC_MM_FWD_ONLY, kept independent because the two install on different "
        "Libraries at different moments. Same backward-only discriminator and recompute re-entry hook, so "
        "bitwise-neutral by construction; a failed post-install verification de-registers and leaves vLLM "
        "unscoped bmm in place.",
    ),
    Flag(
        "SKYRL_ISOEXEC_VARLEN_OUT",
        "0",
        ("both",),
        "trainer core_attn matches engine paged varlen_attn_out",
        FUNCTION,
        (TRAIN,),
    ),
    Flag("SKYRL_ISOEXEC_VARLEN_PAGED", "0", ("both",), "paged varlen attention selection", FUNCTION, (TRAIN,)),
    Flag(
        "SKYRL_ISOEXEC_SPLIT_LM_HEAD",
        "0",
        ("engine",),
        "apply lm_head to sampled rows only",
        FUNCTION,
        (TRAIN,),
        notes="Engine-read; the ladder harness re-injects it from IX_SPLIT_LM_HEAD, a known alias trap.",
    ),
    Flag("SKYRL_ISOEXEC_SCORING_FORWARD", "1", ("trainer",), "no_grad scoring-mode forward", FUNCTION, (TRAIN,)),
    # GDN.
    Flag("SKYRL_ISOEXEC_GDN", "0", ("both",), "install fla shim (trainer) + GDN engine patch", FUNCTION, (TRAIN,)),
    Flag(
        "SKYRL_ISOEXEC_GDN_KERNEL",
        "recurrent",
        ("both",),
        "delta-rule kernel: chunk|cpr|recurrent (MUST agree)",
        FUNCTION,
        (TRAIN,),
        notes="Default and vocabulary are core/gdn_kernel_env's, which both the declaration site (models/qwen3_5) "
        "and the executing sites (ops/gdn/gdn_ops) now parse through. 'chunk' was the executing default while no "
        "model declares a chunk composition, so unset named a function the process did not run.",
    ),
    Flag(
        "SKYRL_ISOEXEC_GDN_TRAINER_KERNEL",
        "",
        ("trainer",),
        "override trainer delta-rule kernel only (ablation)",
        FUNCTION,
        (TRAIN,),
        notes="Not an IsoExec configuration: the contract pins ONE gdn.core kernel for all four sites, so a build "
        "refuses when this asks for a trainer/engine split. Same vocabulary, same parser.",
    ),
    Flag(
        "SKYRL_ISOEXEC_GDN_NATIVE_KERNELS",
        "0",
        ("both",),
        "vLLM native fused GDN kernels on both runtimes",
        FUNCTION,
        (TRAIN, ENGINE),
    ),
    Flag(
        "SKYRL_ISOEXEC_GDN_NATIVE_STATE",
        "0",
        ("engine",),
        "GDN state in vLLM mamba kv_cache (forces fp32 cache)",
        FUNCTION,
        (TRAIN, ENGINE),
    ),
    Flag(
        "SKYRL_ISOEXEC_GDN_CHUNKED_PREFILL",
        "0",
        ("engine",),
        "GDN continuation-chunk resume from carried state",
        FUNCTION,
        (TRAIN, ENGINE),
    ),
    Flag(
        "SKYRL_ISOEXEC_GDN_NATIVE_CONV",
        "0",
        ("both",),
        "cpr: native conv fn/update pair (MUST agree)",
        FUNCTION,
        (TRAIN, ENGINE),
    ),
    Flag(
        "SKYRL_ISOEXEC_GDN_MATCHED_PREP_FUSED_GATE",
        "1",
        ("both",),
        "byte-exact native_matched_prep elementwise fusion with size-gated exact L2 chain",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Default ON. Strict CUDA shape/dtype/layout admission; OFF restores the canonical eager expression. The "
        "L2 candidate keeps torch.sum itself so the pinned fp32 reduction association cannot change.",
    ),
    Flag(
        "SKYRL_ISOEXEC_GDN_ANALYTIC_CONV_BWD",
        "0",
        ("trainer",),
        "packed analytic native-conv VJP (backward-only, default OFF pending live proof)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    Flag(
        "SKYRL_ISOEXEC_GDN_CONV_BWD_ROWS",
        "8192",
        ("trainer",),
        "analytic native-conv VJP row chunk size (backward-only)",
        DIAGNOSTIC,
        (TRAIN,),
    ),
    Flag(
        "SKYRL_ISOEXEC_GDN_CPR_FUSED_SCATTER",
        "1",
        ("engine",),
        "one-launch open-chunk buffer scatter (data movement)",
        DEPLOYMENT,
        (TRAIN, ENGINE),
    ),
    Flag(
        "SKYRL_ISOEXEC_GDN_CPR_FUSED_ROWS",
        "1",
        ("engine",),
        "one-launch slot->row resolution, both widths (integer index movement)",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Integer index arithmetic only -- clamp, gather, width cast -- so there is nothing to round. OFF "
        "restores the ATen chain, bit-identical.",
    ),
    Flag(
        "SKYRL_ISOEXEC_SAMPLER_TEMP_SKIP",
        "1",
        ("engine",),
        "skip V1 Sampler.apply_temperature when every temperature is exactly 1.0",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="The IsoExec recipe pins temperature=1.0 and x / 1.0 is the identity on every fp32 bit pattern, so "
        "skipping the divide leaves logits, sampled tokens and logprobs byte-identical. The predicate is read "
        "off vLLM host mirror so it costs no sync, and a drift probe re-derives it on the device periodically "
        "and raises on disagreement.",
    ),
    Flag(
        "SKYRL_ISOEXEC_GDN_FUSED_SPLIT",
        "1",
        ("engine",),
        "one-launch q/k/v + alpha/beta materialisation for the GDN native core (byte movement)",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="One Triton kernel writes the five contiguous outputs the ATen chain produced in five launches. No "
        "float arithmetic at all -- masked load, masked store, source dtype -- so the proof is byte equality, "
        "not an allclose. OFF restores the ATen chain, bit-identical.",
    ),
    Flag(
        "SKYRL_ISOEXEC_GDN_CPR_MIN_PAGES",
        "0",
        ("engine",),
        "minimal vLLM mamba pages under cpr (private pools carry the state)",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Engine memory only, moves no bits: cpr never reads vLLM GDN state pages (it uses them as a "
        "slot-id source), so they shrink and the freed KV pool returns to attention. Scoped to cpr "
        "without native state; recurrent and native-state keep full pages.",
    ),
    Flag(
        "SKYRL_ISOEXEC_GDN_CPR_APC",
        "0",
        ("engine",),
        "prefix caching under cpr via the boundary-state store",
        FUNCTION,
        (TRAIN, ENGINE),
        notes="Engine-only but FUNCTION-half: it changes which tokens the GDN layers scan for a request. Requires "
        "SKYRL_ISOEXEC_GDN_CHUNKED_PREFILL=1 and cpr. A mode, not a boolean: 1 is in-process admission "
        "and is refused unless world_size==1; 2/shm/shared uses a shared membership index and works at any "
        "topology.",
    ),
    Flag(
        "SKYRL_ISOEXEC_GDN_CPR_APC_MB",
        "1024",
        ("engine",),
        "device-memory ceiling for the CPR-APC boundary-state store",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Capacity only: an eviction costs a cache hit, never a wrong resume, because the admission clamp reads "
        "the store itself. In shared-index mode the store refuses new checkpoints when full rather than "
        "evicting, since the mirror must never advertise an entry a worker dropped.",
    ),
    Flag(
        "SKYRL_ISOEXEC_GDN_CPR_SLEEP",
        "0",
        ("engine",),
        "CPR state arena is DISCARDED at engine sleep and re-created at wake",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Engine memory and time only: the arena contents are dead across a generate/train/generate boundary, "
        "and the wake path zeroes the arena and resets every row map so the pool equals a fresh build. It "
        "removes the D2H backup and H2D restore of that dead state and holds the arena released through the "
        "weight-broadcast window.",
    ),
    Flag(
        "SKYRL_ISOEXEC_GDN_PIN_CONFIGS",
        "1",
        ("both",),
        "pin every autotuned FLA kernel to configs[0], identically in every process -- ON, and "
        "CORRECTNESS MACHINERY, not a tuning knob",
        FUNCTION,
        (TRAIN,),
        notes="Correctness machinery, not a tuning knob. The OFF path hands the kernel back to the Triton autotuner, "
        "which can select a racy config that returns different results run to run on identical inputs; "
        "autotuning is also per-process, so trainer and engine can land on different configs and therefore "
        "different reduction orders. Set 0 only to reproduce an unpinned baseline, never on a gate run.",
    ),
    Flag(
        "SKYRL_ISOEXEC_GDN_CONFIG_INDEX",
        "0",
        ("both",),
        "which entry of each kernel's own config list PIN_CONFIGS pins",
        FUNCTION,
        (TRAIN,),
        notes="Only for bisecting a bad config on a new stack; it MUST match on trainer and engine or the two pin "
        "different reduction orders.",
    ),
    Flag(
        "SKYRL_ISOEXEC_GDN_EAGER_PREP",
        "1",
        ("trainer",),
        "keep g/beta + qkv prep out of torch.compile",
        FUNCTION,
        (TRAIN,),
    ),
    Flag(
        "SKYRL_ISOEXEC_GDN_FLA_BACKWARD",
        "0",
        ("trainer",),
        "FLA fused Triton VJP (backward is free)",
        DEPLOYMENT,
        (TRAIN,),
        notes="Trainer-only; the backward carries no batch-invariance contract.",
    ),
    Flag(
        "SKYRL_ISOEXEC_FLA_SOURCE",
        "",
        ("trainer",),
        "path to a local flash-linear-attention checkout",
        DEPLOYMENT,
        (TRAIN,),
        notes="TRAIN allowlist only; read trainer-side. Empty means the installed ``fla`` package is used "
        "instead of a source tree; set it only to point at a checkout that is not on sys.path.",
    ),
    Flag(
        "SKYRL_ISOEXEC_GDN_FUSED_OUTNORM",
        "0",
        ("engine",),
        "fused gated out-norm (engine-only, bitwise-equal)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    Flag(
        "SKYRL_ISOEXEC_GDN_SCORING_FUSED_OUTNORM",
        "0",
        ("trainer",),
        "no-grad Megatron scoring gated out-norm fusion (bitwise-equal)",
        DEPLOYMENT,
        (TRAIN,),
        notes="Default OFF. The scoring shim also requires eval mode, disabled autograd, CUDA bf16, contiguous "
        "operands, silu/swish and ZeroCenteredTorchRMSNorm; training and grad paths stay eager.",
    ),
    Flag(
        "FLA_TILELANG",
        "1",
        ("trainer",),
        "FLA TileLang backend (0 on B200; arch-specific)",
        DEPLOYMENT,
        (TRAIN,),
        notes="Forwarded by a direct env_vars[] assignment in the megatron branch rather than the allowlist loop. "
        "Arch-scoped.",
    ),
    # MoE: function-half where it moves bits or must agree across sides.
    Flag(
        "SKYRL_ISOEXEC_MOE_DETERMINISTIC",
        "1",
        ("both",),
        "fixed-order combine + sorted router top-k",
        FUNCTION,
        (TRAIN,),
        notes="Known asymmetry: the engine-side install gates on a bare equality against 1, so with the variable "
        "unset the trainer installs the deterministic MoE matmul and the engine plugin does not. Every shipped "
        "launcher exports 1 explicitly.",
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_PIK_FC2",
        "0",
        ("both",),
        "ETP-invariant MoE fc2 leaf-tree (contract constant)",
        FUNCTION,
        (TRAIN, ENGINE),
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_BLOCK_M",
        "",
        ("both",),
        "fixed BLOCK_SIZE_M for expert GEMMs (pinned constant)",
        FUNCTION,
        (TRAIN,),
        notes="MUST be a fixed per-process constant, never token-count-derived",
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_RECOMPUTE",
        "1",
        ("trainer",),
        "honour recompute_modules=[...,moe] (kill switch)",
        DEPLOYMENT,
        (TRAIN,),
        notes="Bitwise-neutral: recompute re-runs the same kernels on the same inputs under a restored RNG state. "
        'Inert unless recompute_modules contains "moe".',
    ),
    Flag(
        "SKYRL_ISOEXEC_TRAINER_EP_BALANCE",
        "0",
        ("trainer",),
        "length-balanced DP striping + microbatch sort -- kills the EP-skew per-layer barrier tax",
        DEPLOYMENT,
        (TRAIN,),
        notes="Two sites, one flag: a within-mini-batch length stripe across DP ranks in the trainer driver and a "
        "shard length sort before microbatch chunking in the policy actor, so it must reach both. Scheduling "
        "only -- the per-token contract is untouched, but the fp32 grad-accumulation grouping changes, the same "
        "class as changing the micro-batch size. Declines on token-based batching, non-DP-divisible mini- "
        "batches and a missing attention mask.",
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_FUSED_COMBINE",
        "0",
        ("engine",),
        "fused top-k combine (engine, bitwise-equal, marked)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_PERMUTE_SORT",
        "0",
        ("engine",),
        "counting-sort permute index build, decoupled from the fused-router flag",
        DEPLOYMENT,
        (TRAIN,),
        notes="Selects the same counting-sort index build SKYRL_ISOEXEC_MOE_ROUTER_O2 gates, without also asking for "
        "the fused router, so it serves routers the O2 guard declines. Score-function-agnostic: a counting sort "
        "over a boolean map plus a gather, no arithmetic. EP=1 only, since at EP>1 the counting sort would "
        "leave the tail uninitialised.",
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_ROUTER_O2",
        "0",
        ("engine",),
        "fused router + permute-sort (engine-only, bitwise)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_FUSED_EPILOGUE",
        "0",
        ("engine",),
        "fused fc1+glu+probs epilogue (engine-only, bitwise)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_ROUTER_CAST_CACHE",
        "1",
        ("engine",),
        "cache the router's fp32 weight cast between weight syncs (engine-only, bitwise)",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="The fp32 cast VALUE is contract -- it is what makes routing batch-invariant under "
        "moe_router_dtype=fp32 -- and only its recomputation is waste; a widening cast is exact, so the cached "
        "bytes are the recomputed bytes. Engine-only by instance rebind, because the trainer cast is grad- "
        "carrying and a detached cached copy would zero the router weight gradient. The buffer is re-cast in "
        "place at the weight-sync epoch bump, since a captured decode graph holds its address.",
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_PREAMBLE_O12",
        "0",
        ("engine",),
        "MoE preamble + shared-expert (engine-only, bitwise)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_FUSED_BLOCKMAP",
        "0",
        ("engine",),
        "fused expert block map (integer, no fp)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_COMBINE_SORT",
        "0",
        ("engine",),
        "stable counting-sort combine (integer permutation)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_DENSE_SCATTER",
        "0",
        ("engine",),
        "one-kernel dense (routing_probs, routing_map) build from the top-k form",
        DEPLOYMENT,
        (TRAIN,),
        notes="DEPLOYMENT, not FUNCTION: it does not touch the score chain, only the dense (routing_probs, "
        "routing_map) build after it. No arithmetic -- tl.where is a select, so NaN, subnormal and -0.0 "
        "payloads move as bit patterns. Engine-only on marked instances, because the kernel has no "
        "autograd.Function.",
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_ROUTER_SCORE",
        "0",
        ("engine",),
        "one-kernel pre-transform score function + expert-bias add",
        DEPLOYMENT,
        (TRAIN,),
        notes="DEPLOYMENT because every operation is a single correctly-rounded fp32 primitive on one element (the "
        "non-FTZ sigmoid / sqrt / log1p from core.triton_nonftz), so there is no association order to get "
        'wrong. Serves the property "score function is an elementwise pre-selection map", not a model; engine- '
        "only on marked instances, admitted per shape against megatron own function.",
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_ROUTER_TAIL",
        "0",
        ("engine",),
        "one-kernel normalisation tail (eps + divide + scale) fused into the dense build",
        DEPLOYMENT,
        (TRAIN,),
        notes="The denominator is an INPUT: ATen row-sum runs unchanged upstream and its output is consumed, never "
        "re-associated -- that is what separates this from a re-associated normalisation. Three roundings, each "
        "with one correct answer: IEEE add, div.rn.f32 via inline PTX, IEEE mul. With no denominator, eps or "
        "scale it reduces to the dense-scatter kernel, which is asserted rather than claimed.",
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_PERMUTE_INDEX_1K",
        "0",
        ("engine",),
        "counting-sort index build in ONE kernel (counts folded into the permute pass)",
        DEPLOYMENT,
        (TRAIN,),
        notes="Requires SKYRL_ISOEXEC_MOE_PERMUTE_SORT (or _ROUTER_O2) to select the counting sort at all -- this only "
        "chooses which form of it runs -- and refuses above a size bound, falling back to the two-kernel form. "
        "Integer-exact and order-free; the probs are gathered, never arithmetic-ed.",
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_COMBINE_SORT_BLOCK_P",
        "1024",
        ("engine",),
        "combine-sort tile knob (occupancy)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_COMBINE_SORT_WARPS",
        "4",
        ("engine",),
        "combine-sort warp knob (occupancy)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_COMBINE_SORT_MAX_WORK",
        "33554432",
        ("engine",),
        "combine-sort work cap (occupancy)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_COMBINE_BLOCK_H",
        "512",
        ("engine",),
        "combine tile H (reduction-free, neutral)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    Flag("SKYRL_ISOEXEC_MOE_COMBINE_WARPS", "4", ("engine",), "combine warps (neutral)", DEPLOYMENT, (TRAIN,)),
    Flag(
        "SKYRL_ISOEXEC_MOE_FUSED_LEAFTREE",
        "1",
        ("engine",),
        "in-kernel fc2 leaf tree (engine-only)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_FUSED_COMBINE_TRAINER",
        "0",
        ("trainer",),
        "opt trainer INTO fused combine (memory, not speed)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_COMBINE_FOLD_ROUND",
        "0",
        ("trainer",),
        "fold the pik alltoall combine's trailing bf16 round into the fused combine's store "
        "(BYTE-IDENTICAL: one RTNE of the same fp32 accumulator, moved from a [T,H] tensor into a "
        "register; default OFF pending GPU proof)",
        DEPLOYMENT,
        (TRAIN,),
        notes="Not an arithmetic change: the accumulate dtype stays keyed on the row dtype, the sum still runs in "
        "fp32 with no intermediate rounding, and there is still exactly one round-to-nearest-even -- it just "
        "happens in a register instead of through a [T,H] fp32 tensor. Any other dtype pair raises rather than "
        "accepting a second rounding schedule. Fail-closed: the request only goes to a binding that publishes "
        "support, and the caller own cast is kept unconditionally.",
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_FUSED_EPILOGUE_TRAINER",
        "0",
        ("trainer",),
        "admit no-grad trainer standalone expert epilogue (bitwise, default OFF pending live proof)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_BMM_MAX_ELEMS",
        "1073741824",
        ("trainer",),
        "chunk expert bmm below the int32 element limit",
        DEPLOYMENT,
        (TRAIN,),
        notes="the 2^31 bmm-overflow guard",
    ),
    Flag("SKYRL_ISOEXEC_MOE_TILE_ROWS", "128", ("both",), "expert tile row count", DEPLOYMENT, (TRAIN,)),
    Flag(
        "SKYRL_ISOEXEC_MOE_FLAT_STAGE",
        "0",
        ("both",),
        "1-D addressing for the staged expert tiles (bitwise; 1.9x fwd+bwd on the stage/unstage pair)",
        DEPLOYMENT,
        (TRAIN,),
        notes="Swaps xp[tile_idx, row_idx] for index_copy_ / index_select.",
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_CHUNK_SORT",
        "0",
        ("both",),
        "megatron's sort_chunks_by_idxs cat-loop replaced by ONE row gather (bitwise fwd AND bwd; "
        "admitted per shape on live operands)",
        DEPLOYMENT,
        (TRAIN,),
        notes="The chunk reordering is a bijection on rows, so the identical bytes come out of one 1-D index_select, "
        "and because the map is bijective the VJP is the inverse gather rather than an index_add_ -- "
        "deliberately, since a scatter-add formulation flips signed zeros. Admitted per (splits, hidden, dtype, "
        "rows) at first use on the live operands: forward bits, row provenance against a marker, backward bits, "
        "round-trip bijectivity, determinism. A failing key runs megatron cat and no bits move.",
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_WEIGHT_CACHE",
        "1",
        ("both",),
        "memoize the stacked expert weights across forwards instead of re-stacking every forward",
        DEPLOYMENT,
        (TRAIN,),
        notes="Pure memoization -- same bytes, same layout, same kernels -- so bitwise-neutral by construction; the "
        "only question is staleness. Validity is a per-parameter (id, data_ptr, _version) probe plus a global "
        "epoch bumped at the optimizer step and the weight sync, because a write through param.data or a DDP "
        "param buffer leaves _version untouched. A new weight-mutation site must call invalidate_all().",
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_TILE_CAP",
        "",
        ("both",),
        "rows per expert tile derived from the batch ('auto') or pinned to an int; '' = today's 128",
        DEPLOYMENT,
        (TRAIN,),
        notes="Companion to SKYRL_ISOEXEC_MOE_TILE_ROWS, which pins the cap for decode; the trainer forward is the "
        'opposite regime. "auto" derives the cap from the host-known token count with no sync, and when the '
        "grid lands at one tile per expert the per-tile weight gather is skipped entirely. Default off: the cap "
        "sets the bmm M and so changes the launch grid of the GEMM the gate signature is computed from.",
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_INDEXED_BMM",
        "0",
        ("both",),
        "expert bmm indexes the weight stack in-kernel (w + tile_expert[pid]*stride) instead of "
        "materializing w[tile_expert] before the GEMM",
        DEPLOYMENT,
        (TRAIN,),
        notes="The indexed kernel is vLLM bmm_kernel with only the B-operand base pointer computed from tile_expert "
        "in-kernel: identical loads, identical fixed-order K reduction, so the forward is bitwise the gather "
        "path by construction and a fail-closed first-use self-check falls through to the gather permanently on "
        "any mismatch. Grads are NOT bitwise against the gather path (the tile reduction rounds per sub-chunk), "
        "which carries no bitwise contract. Scope: fc1, plain fc2 and single-leaf pik-fc2; the multi-leaf tree "
        "keeps the gather.",
    ),
    Flag(
        "SKYRL_ISOEXEC_EP_A2A_CHANNELS",
        "0",
        ("trainer",),
        "per-communicator NCCL channel count for the MoE expert all-to-all group ('ep'); 0 = off, "
        "keep the process-wide NCCL_MAX_NCHANNELS for every group",
        DEPLOYMENT,
        (TRAIN,),
        notes="NCCL_MAX_NCHANNELS is process-wide, so it buys channels on every communicator to speed up essentially "
        "one collective and charges memory for the rest; a per-group count overrides the process-wide env. It "
        "cannot move a bit: an all-to-all performs no arithmetic, so channel count decides how bytes are "
        "routed, never what they are. The all-reduce pin is untouched, and the module warns loudly if the flag "
        "is pointed at a reducing group.",
    ),
    Flag(
        "SKYRL_ISOEXEC_EP_A2A_CHANNEL_GROUPS",
        "",
        ("trainer",),
        "comma-separated megatron process-group names to apply EP_A2A_CHANNELS to (default 'ep', the "
        "expert all-to-all)",
        DIAGNOSTIC,
        (TRAIN,),
        notes="The expert all-to-all group is the only one the byte-copy legality argument covers. The expert tensor- "
        "parallel and expert data-parallel groups carry reductions, so widening them is a reduction-order "
        "change needing its own bitwise A/B; the module warns loudly rather than refusing, so an experiment is "
        "possible but never accidental.",
    ),
    Flag(
        "SKYRL_ISOEXEC_NCCL_CHANNEL_PLAN",
        "",
        ("trainer",),
        "budgeted per-communicator NCCL channel plan, 'group:channels' comma-separated (e.g. 'ep:24'); "
        "empty = off, keep the process-wide NCCL_MAX_NCHANNELS for every group",
        DEPLOYMENT,
        (TRAIN,),
        notes="Supersedes SKYRL_ISOEXEC_EP_A2A_CHANNELS and ignores it when both are set. A plan is admitted only "
        "within SKYRL_ISOEXEC_NCCL_CHANNEL_BUDGET_GIB, and widening a group that carries an NCCL reduction "
        "additionally requires SKYRL_ISOEXEC_NCCL_CHANNEL_ACK_REDUCE. NCCL allocates per (channel, connection) "
        "and the connection count depends on the patterns a communicator ran, not just its world size.",
    ),
    Flag(
        "SKYRL_ISOEXEC_NCCL_CHANNEL_BUDGET_GIB",
        "0",
        ("trainer",),
        "GiB/rank of extra un-reclaimable NCCL buffer memory the channel plan may spend; 0 refuses " "every plan",
        DEPLOYMENT,
        (TRAIN,),
        notes="The budget is a declaration, not a measurement, because only the launcher knows how much the "
        "gpu_memory_utilization setting left free. 0 refuses every plan.",
    ),
    Flag(
        "SKYRL_ISOEXEC_NCCL_CHANNEL_ACK_REDUCE",
        "0",
        ("trainer",),
        "allow SKYRL_ISOEXEC_NCCL_CHANNEL_PLAN to widen a group that carries a NCCL REDUCTION "
        "(tp/dp/mp/ep_tp/ep_dp) -- a composition event, never a production setting",
        DIAGNOSTIC,
        (TRAIN,),
        notes="Channel count decides how NCCL splits a collective, and for a reduction a different split is a "
        "different summation order -- so widening a reducing group needs a manifest pin and a gate re-freeze, "
        "not a flag. Kept as an explicit acknowledgement rather than a hard ban so the A/B stays possible.",
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_FUSED_LEAFCOMBINE",
        "0",
        ("both",),
        "pik-fc2 leaf tree folded in ONE in-register pass instead of G fp32 leaf buffers + G-1 "
        "eager adds (same leaves, same tree, same rounding)",
        DEPLOYMENT,
        (TRAIN,),
        notes="Same leaves, same tree, same rounding: the G leaf products are folded in one in-register pass instead "
        "of G fp32 leaf buffers plus G-1 eager adds.",
    ),
    *[
        Flag(
            name,
            "",
            ("both",),
            f"fc2 in-GEMM leaf-tree {what} -- SCHEDULE knob, empty = inherit vLLM's bmm config",
            DEPLOYMENT,
            (TRAIN, ENGINE),
            notes="A SCHEDULE knob, not an arithmetic one: the reduction runs over K only, so M and N index "
            "independent output elements and a BLOCK along either cannot change the accumulation. BLOCK_SIZE_K is "
            "deliberately NOT exposed, because it sets the K-tiling and therefore the accumulation order. An empty "
            "value is today's geometry byte for byte, and an override still passes the same fail-closed first-use "
            "self-check at the overridden geometry, so it can only make a run faster or slower. BLOCK_N also accepts "
            "'auto', which derives the widest power-of-two tile whose live fp32 accumulators stay inside the measured "
            "register-spill frontier, using no model dimension.",
        )
        for name, what in (
            ("SKYRL_ISOEXEC_MOE_FC2_INGEMM_BLOCK_M", "BLOCK_SIZE_M override"),
            ("SKYRL_ISOEXEC_MOE_FC2_INGEMM_BLOCK_N", "BLOCK_SIZE_N override"),
            ("SKYRL_ISOEXEC_MOE_FC2_INGEMM_WARPS", "num_warps override"),
            ("SKYRL_ISOEXEC_MOE_FC2_INGEMM_STAGES", "num_stages override"),
        )
    ],
    Flag(
        "SKYRL_ISOEXEC_MOE_FC2_INGEMM",
        "0",
        ("both",),
        "pik-fc2 leaf tree run INSIDE one indexed expert GEMM -- no weight gather, no per-leaf "
        "staging copies, no leaf buffers, no separate combine",
        DEPLOYMENT,
        (TRAIN,),
        notes="Unifies SKYRL_ISOEXEC_MOE_INDEXED_BMM, which scoped itself out of the multi-leaf tree, and "
        "SKYRL_ISOEXEC_MOE_FUSED_LEAFCOMBINE, which left the GEMM half alone. Bitwise, and the subtle part is "
        "where the rounding lives: the buffer path leaf is a bf16 tensor, so the kernel materializes each leaf "
        "bf16 round in register before folding -- an in-GEMM tree over the raw fp32 accumulators would be "
        "strictly more accurate and therefore wrong.",
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_LEAFCOMBINE_BLOCK",
        "4096",
        ("both",),
        "elements per program in the fused leaf-tree combine (bitwise-neutral tuning knob)",
        DEPLOYMENT,
        (TRAIN,),
        notes="Companion tuning knob to SKYRL_ISOEXEC_MOE_FUSED_LEAFCOMBINE, read at import time. Each program folds a "
        "disjoint contiguous range of elements and every element tree is independent, so the block size cannot "
        "reach the arithmetic.",
    ),
    Flag(
        "SKYRL_ISOEXEC_MM_TILES",
        "0",
        ("both",),
        "shape-keyed decode GEMM re-tiling (neutral by construction)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    # pik / collectives.
    Flag("SKYRL_ISOEXEC_PIK", "0", ("both",), "pik TP/EP-invariant row-parallel reduction", FUNCTION, (TRAIN, ENGINE)),
    Flag(
        "SKYRL_ISOEXEC_PIK_SELFCHECK",
        "0",
        ("both",),
        "force the pik arch self-check before enabling the row-parallel reduction",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="The check refuses to enable pik on a GPU where a GEMM tiling knob moves the bits of "
        "a non-split-K K-reduction, i.e. where pik's TP-invariance premise does not hold.",
    ),
    Flag(
        "SKYRL_ISOEXEC_COMPAT_STRICT",
        "0",
        ("both",),
        "raise instead of warn when the megatron/vLLM compatibility surface has moved",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Read by the megatron and vLLM compat checks. Default WARN keeps a surface change "
        "from taking down a run; =1 is the CI/bring-up polarity.",
    ),
    Flag(
        "SKYRL_ISOEXEC_AUTOFUSE_DECOMP",
        "0",
        ("both",),
        "offer the bitwise decomposition table to autofuse region rewriting",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Rollout knob, not a safety one: the region fingerprint makes a with/without "
        "mismatch resolve to eager rather than mis-serve.",
    ),
    Flag(
        "SKYRL_ISOEXEC_TRAINER_SP",
        "0",
        ("trainer",),
        "Megatron sequence parallelism in the IsoExec trainer (pik tree_reduce_scatter combine)",
        FUNCTION,
        (TRAIN,),
        notes="Read in the policy Ray actor at three gates that must agree: the worker keeps sequence_parallel on "
        "instead of the IsoExec force-off and refuses without SKYRL_ISOEXEC_PIK=1, pik routes SP row-parallel "
        "layers through its tree reduce-scatter instead of the native NCCL reduce-scatter (which would silently "
        "break TP-invariance), and the norm shim lifts the SP refusal.",
    ),
    Flag(
        "SKYRL_ISOEXEC_PIK_LEAVES",
        "8",
        ("both",),
        "leaf-tree fan-out G (contract constant, MUST match)",
        FUNCTION,
        (TRAIN, ENGINE),
    ),
    Flag(
        "SKYRL_ISOEXEC_PIK_LEAF_DTYPE",
        "bf16",
        ("both",),
        "leaf dtype (contract constant, MUST match)",
        FUNCTION,
        (TRAIN, ENGINE),
        notes="A contract constant that must match on both runtimes. It is not a tuning knob: leaf dtype changes the "
        "policy function, so logprobs from a bf16-leaf run and an fp32-leaf run are different numbers and the "
        "two are not comparable.",
    ),
    Flag(
        "SKYRL_ISOEXEC_PIK_P2P",
        "1",
        ("both",),
        "symmetric-memory P2P vs NCCL transport (bitwise-equal)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    Flag(
        "SKYRL_ISOEXEC_PIK_ONESHOT_MB",
        "2",
        ("both",),
        "one-shot/two-shot crossover MiB (both paths equal)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    Flag(
        "SKYRL_ISOEXEC_PIK_BATCHED_LEAVES",
        "0",
        ("both",),
        "ONE cuBLASLt strided-batch call over all m leaves for the m>=2 small-M row-parallel sites",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="One strided-batch cuBLASLt call over all m leaves plus the shipped tree fold, with one pinned non- "
        "split-K algorithm per (leaf_k, N, lda, ldb, m); the key excludes M so a single kernel serves every "
        "batch size. Bitwise-equal to the sequential path, which admission enforces.",
    ),
    Flag(
        "SKYRL_ISOEXEC_PIK_BATCHED_LEAVES_MAX_M",
        "256",
        ("both",),
        "M-gate for the batched-leaves path: above it the sequential cuBLASLt loop stays in charge",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="A pure perf knob: the two paths are bitwise-equal by admission, so crossing the threshold cannot move "
        "a bit.",
    ),
    Flag(
        "SKYRL_ISOEXEC_NATIVE_NORM_MEMO",
        "0",
        ("both",),
        "memoize the DEVICE-derived half of torch _native's RMSNorm admission predicate",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Host-time only. It memoizes the two device-property lookups inside torch _fused_rms_norm admission "
        "predicate, which depend on nothing but the device index, by rebinding the module globals the predicate "
        "resolves through -- no re-registration and no change to the kernel that runs.",
    ),
    Flag(
        "SKYRL_ISOEXEC_PIK_FASTLAUNCH",
        "0",
        ("both",),
        "memoize Triton's per-call launch derivation for pik's PINNED kernels (host-time only)",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Host-time only: it memoizes what Triton re-derives on every launch (specialization binder, cache key, "
        "grid callable), all of which are constant for a pinned kernel at a fixed shape class, which pik "
        "kernels are by construction.",
    ),
    Flag(
        "SKYRL_ISOEXEC_PIK_FUSED_BARRIER",
        "0",
        ("both",),
        "fold pik's symmetric-memory barriers INTO the tree all-reduce kernel: 1 launch per site",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Cannot move a bit: the reduction core is generated once and pasted into both templates, and every "
        "(world, numel, wire, root) re-proves it on live operands with a bit-pattern compare before use, "
        "falling back per shape on any mismatch. Rank uniformity is a hard requirement and is enforced rather "
        "than assumed, because the two launch structures rendezvous differently -- disagreement hangs rather "
        "than slows down.",
    ),
    Flag(
        "SKYRL_ISOEXEC_PIK_AR_VEC",
        "4",
        ("both",),
        "element-to-thread layout of the p2p reduce kernel: vec CONTIGUOUS elements per thread",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="The layout moves which thread computes an element and nothing else: the element set, each element load "
        "sources, the tree association and the single rounding are unchanged, and every (world, numel, wire, "
        "root, branch) re-proves the layout against the vec=1 reference. Contiguous elements per thread is what "
        "lets Triton emit 128-bit accesses.",
    ),
    Flag(
        "SKYRL_ISOEXEC_PIK_FUSED_ROOT_CAST",
        "1",
        ("both",),
        "round the tree ROOT to bf16 in the reduce kernel's store instead of a separate cast kernel",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Deliberately separate from SKYRL_ISOEXEC_PIK_FUSED_BARRIER: fusion is bit-neutral, this MOVES a "
        "rounding point, so it is a different claim with a different proof. Legal because the root, unlike an "
        "internal node, is the same node at every TP size, so rounding it cannot make the expression TP- "
        "dependent. Applied only where the round is provably the next operation -- no bias, fp32 leaves, bf16 "
        "output -- and the guard is asserted identical in both row-parallel twins so they cannot drift.",
    ),
    Flag(
        "SKYRL_ISOEXEC_NCCL_PIN",
        "1",
        ("both",),
        "NCCL algo/channel composition; trainer=FUNCTION, engine=DEPLOYMENT",
        FUNCTION,
        (TRAIN, ENGINE),
        notes="Per-entry split: the trainer selection is FUNCTION because backward reductions reassociate, the engine "
        "side is neutral. The runtime effect is trainer-side, but both runtimes build the same complete "
        "manifest, so the trainer intent must also reach the engine actor and each nested engine worker.",
    ),
    Flag(
        "SKYRL_ISOEXEC_NCCL_MAX_NCHANNELS",
        "",
        ("trainer",),
        "trainer NCCL channel CAP when the pin is off (perf/memory trade)",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Read in the driver to set NCCL_MAX_NCHANNELS in the trainer ray runtime env, and it must additionally "
        "reach both runtimes because each independently derives the same complete manifest, including trainer "
        "sites. Only meaningful when SKYRL_ISOEXEC_NCCL_PIN=0. Memory is essentially linear in channels, while "
        "latency saturates long before memory does.",
    ),
    Flag(
        "SKYRL_ISOEXEC_NCCL_PREWARM",
        "0",
        ("trainer",),
        "force every trainer NCCL communicator to allocate at init_model (memory accounting)",
        DEPLOYMENT,
        (TRAIN,),
        notes="NCCL allocates lazily per communicator AND per transport, so under colocate_all the trainer "
        "communicators appear after vLLM has sized its KV pool. This fires every collective type on every "
        "megatron group at the end of init_model so the cost is paid once, up front, and printed as an input to "
        "the memory budget. Bitwise-neutral -- it only touches scratch buffers -- and firing one all_reduce is "
        "not enough.",
    ),
    Flag(
        "SKYRL_ISOEXEC_NCCL_CAPABILITY_MODE",
        "off",
        ("trainer",),
        "declared NCCL transport ownership: off, census, or fail-closed enforce",
        DEPLOYMENT,
        (TRAIN,),
        notes="Default off preserves the complete prewarm. census tracks subsequent c10d operation ownership without "
        "skipping transports; enforce requires every active physical Megatron group to have unanimous explicit "
        "owner declarations, omits the expensive A2A warm only where no owner declares it, then refuses an "
        "unexpected A2A before c10d can lazily allocate it.",
    ),
    Flag(
        "SKYRL_ISOEXEC_NCCL_CAPABILITY_MANIFEST",
        "",
        ("trainer",),
        "reviewed active-owner operation manifest for NCCL capability enforce mode",
        DEPLOYMENT,
        (TRAIN,),
        notes="Required only when CAPABILITY_MODE=enforce. JSON rows name an active owner, a Megatron process-group "
        "getter plus kwargs, and the exact public c10d operations it may issue; resolved membership and owner "
        "signatures are Store-voted before any transport is skipped.",
    ),
    Flag(
        "SKYRL_ISOEXEC_NCCL_INCUMBENT_CONFIG",
        "",
        ("trainer",),
        "reviewed Megatron NCCL YAML applied to incumbent physical process groups",
        DEPLOYMENT,
        (TRAIN,),
        notes="Applies a reviewed YAML through MCore native communicator-config path, then reads the raw min/max CTA "
        "values and members back off the original TP/EP ProcessGroupNCCL objects. The census creates no group, "
        "traffic or memory gate. Pair with NCCL_INCUMBENT_CONFIG_SHA256 to bind the production source.",
    ),
    Flag(
        "SKYRL_ISOEXEC_NCCL_INCUMBENT_CONFIG_SHA256",
        "",
        ("trainer",),
        "expected SHA-256 of the MCore incumbent communicator YAML",
        DEPLOYMENT,
        (TRAIN,),
        notes="When set together with INCUMBENT_CONFIG, every trainer rank hashes the consumed YAML and refuses "
        "before model forward on drift.",
    ),
    Flag(
        "SKYRL_ISOEXEC_NCCL_INCUMBENT_CONTRACT",
        "",
        ("trainer",),
        "hash-bound physical-group/readback/memory contract for incumbent NCCL YAML",
        DEPLOYMENT,
        (TRAIN,),
        notes="Default OFF. JSON binds the YAML, capability manifest, transport boundary "
        "requirements, process-wide channel environment, requested group getters, exact raw "
        "min/max CTA values, and post-prewarm memory limits.",
    ),
    Flag(
        "SKYRL_ISOEXEC_NCCL_TRANSPORT_BOUNDARY_REQUIREMENTS",
        "",
        ("trainer",),
        "physical process-group operation counts required at real scoring/backward boundaries",
        DEPLOYMENT,
        (TRAIN,),
        notes="Required in capability enforce mode for runtime engagement gates. Requirements are "
        "group alias plus operation plus minimum served count; they use physical served counts, "
        "not claimant labels, and are WORLD Store-voted before a boundary may continue.",
    ),
    Flag(
        "SKYRL_ISOEXEC_NCCL_MAX_POST_LAZY_GROWTH_MIB",
        "",
        ("trainer",),
        "maximum device-global non-Torch growth after prewarm at a real transport boundary",
        DEPLOYMENT,
        (TRAIN,),
        notes="Required only by the default-off transport boundary gate. This is an explicit "
        "launcher safety rail, not a framework default; every rank reports and Store-votes its "
        "measurement.",
    ),
    Flag(
        "SKYRL_ISOEXEC_NCCL_MAX_ABSOLUTE_NONTORCH_MIB",
        "",
        ("trainer",),
        "maximum device-global non-Torch residency at a real transport boundary",
        DEPLOYMENT,
        (TRAIN,),
        notes="Required only by the default-off transport boundary gate. The estimate subtracts "
        "the trainer process's Torch reservation but includes colocated processes and custom "
        "libraries, so production launchers must set a workload-calibrated value.",
    ),
    Flag(
        "SKYRL_ISOEXEC_NCCL_MIN_DEVICE_FREE_MIB",
        "",
        ("trainer",),
        "minimum raw device headroom at a real transport boundary",
        DEPLOYMENT,
        (TRAIN,),
        notes="Required only by the default-off transport boundary gate. It is checked per rank "
        "alongside absolute non-Torch residency and does not replace the post-train engine wake "
        "gate.",
    ),
    Flag(
        "SKYRL_ISOEXEC_ENGINE_NCCL_UNPIN",
        "0",
        ("engine",),
        "unpin engine NCCL (proven neutral, -16% generate)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    Flag(
        "SKYRL_ISOEXEC_ENGINE_NCCL_MAX_NCHANNELS",
        "8",
        ("engine",),
        "channel CEILING for the engine unpin (empty = fully unpinned ~24ch)",
        DEPLOYMENT,
        (TRAIN,),
        notes="A fully unpinned engine NCCL holds a large block of non-sleepable buffers through the whole training "
        "window; a channel ceiling keeps most of the latency win at a fraction of that memory. Bitwise-neutral "
        "for the same reason the unpin is: pik owns the invariant reduction. Set empty to restore the full "
        "unpin.",
    ),
    Flag(
        "SKYRL_ISOEXEC_PIN_WORKER_DEVICE",
        "1",
        ("engine",),
        "pin each vLLM worker to its own GPU before plugin import (kills the GPU-0 context leak)",
        DEPLOYMENT,
        (TRAIN,),
        notes="Read in the vLLM worker subprocess, which inherits the engine actor env, which comes from the ray.init "
        "job runtime_env -- i.e. the TRAIN channel. Forwarding is required: a launch-shell export that is in "
        "neither allowlist is unset in both the actor and its subprocess, which makes the flag unsettable "
        "rather than merely unforwarded.",
    ),
    Flag(
        "SKYRL_ISOEXEC_ENGINE_EMPTY_CACHE",
        "1",
        ("engine",),
        "release the engine's cached blocks at sleep so the trainer's floor measurement sees them",
        DEPLOYMENT,
        (TRAIN,),
        notes="Read in the engine Ray actor sleep path, so it needs the TRAIN channel to be settable from a launch "
        "shell. It frees only cached, unallocated blocks and so cannot move a bit, but it does change the "
        "allocator layout the trainer then trains into, which is why it stays independently gateable.",
    ),
    Flag(
        "SKYRL_ISOEXEC_SLEEP_SKIP_WEIGHTS_BACKUP",
        "0",
        ("engine",),
        "engine sleep discards the sync-covered weight bytes instead of backing them up D2H",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Read in the vLLM worker. In a colocated run the launch-shell value must cross TWO Ray runtime-env "
        "boundaries: TRAIN delivers it to the controller that builds the engine runtime env, then ENGINE "
        "delivers it to the engine actor and its nested workers -- marking only ENGINE leaves the controller "
        "observing it as unset, so the stock D2H backup is silently retained.",
    ),
    # Parallelism topology / resharding.
    Flag(
        "SKYRL_ISOEXEC_ENGINE_TP",
        "",
        ("trainer",),
        "engine TP degree for weight-sync reshard target",
        DEPLOYMENT,
        (TRAIN,),
    ),
    Flag("SKYRL_ISOEXEC_ENGINE_EP", "", ("trainer",), "engine EP degree for reshard target", DEPLOYMENT, (TRAIN,)),
    # Engine-arg policy: proven bitwise-safe, hence deployment-half.
    Flag(
        "SKYRL_ISOEXEC_ENABLE_PREFIX_CACHE",
        "0",
        ("engine",),
        "APC (bitwise-safe under num_splits=1)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    Flag(
        "SKYRL_ISOEXEC_ENABLE_CHUNKED_PREFILL", "0", ("engine",), "chunked prefill (bitwise-safe)", DEPLOYMENT, (TRAIN,)
    ),
    Flag(
        "SKYRL_ISOEXEC_ENABLE_CUDAGRAPH",
        "0",
        ("engine",),
        "CUDA graphs (bitwise-safe under num_splits=1)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    Flag("SKYRL_ISOEXEC_NO_CHUNKED_PREFILL", "0", ("engine",), "force no chunked prefill", DEPLOYMENT, (TRAIN, ENGINE)),
    Flag("SKYRL_ISOEXEC_MAX_MODEL_LEN", "0", ("engine",), "engine max_model_len", DEPLOYMENT, (TRAIN, ENGINE)),
    Flag("SKYRL_ISOEXEC_MAX_BATCHED_TOKENS", "0", ("engine",), "prefill token budget (packing)", DEPLOYMENT, (TRAIN,)),
    Flag(
        "SKYRL_ISOEXEC_FUSED_ADD_NORM",
        "0",
        ("both",),
        "fused residual-add + zero-centered RMSNorm (within-layer seam, both runtimes)",
        DEPLOYMENT,
        (TRAIN,),
        notes="Bit-equal at the shipped hidden size: the in-kernel add rounds exactly like the eager add, then the "
        "norm chain runs on registers. NOT bit-equal at other widths, which must be re-classified FUNCTION "
        "before use there. Class-patches the transformer layer forward halves with a hard fallback.",
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_PIK_OWNER_COMBINE",
        "0",
        ("engine",),
        "owner-computes MoE combine (token-slice owner does tree+topk-sum, all-gather bf16 finals)",
        DEPLOYMENT,
        (TRAIN,),
        notes="Bit-identical to the all-reduce combine -- only the executing rank changes: same peer partials, same "
        "pik tree order, same ascending-expert fp32 k-sum, same single bf16 round. Engine no-gather path only.",
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_SHARED_OWNER_FUSION",
        "0",
        ("engine",),
        "exact shared+routed owner composition: one exchange and one output barrier",
        DEPLOYMENT,
        (TRAIN,),
        notes="Requires SKYRL_ISOEXEC_MOE_PIK_OWNER_COMBINE=1 and the engine no-gather dispatcher. One owner "
        "composition replaces the separate shared reduction, shared gate launch, routed owner exchange and "
        "final add, while preserving the shared tree, both bf16 root rounds, the ascending-k sum and the final "
        "bf16 add. Each live geometry is admitted by a bit-pattern comparison with a group-wide verdict before "
        "use; the trainer is unchanged.",
    ),
    Flag(
        "SKYRL_ISOEXEC_PIK_FUSED_OWNER_COMBINE",
        "0",
        ("engine",),
        "fold the owner-combine's two symm-mem barriers INTO its push kernel: 1 launch",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="The owner combine stages, barriers, pushes and barriers itself rather than entering pik generic path, "
        "so SKYRL_ISOEXEC_PIK_FUSED_BARRIER cannot reach it; this folds those two barriers into the push kernel. "
        "Bitwise-equal by admission, and enablement must be uniform across ranks or the differing launch "
        "structures hang instead of slowing down.",
    ),
    Flag(
        "SKYRL_ISOEXEC_PIK_FUSED_OWNER_MAX_BLOCKS",
        "8",
        ("engine",),
        "grid gate for the fused owner-combine: above this many blocks, fusion is refused",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Not a tuning knob but the measured crossover between the per-block in-kernel barrier and the device- "
        "wide barrier pair. Both paths are bitwise-equal by admission, so moving it cannot move a bit; raising "
        "it without a measurement re-arms a known regression.",
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_OC_ROWS32",
        "1",
        ("engine",),
        "owner-combine my_rows as an int32 VIEW of the counting sort's [T,k] output",
        DEPLOYMENT,
        (TRAIN,),
        notes="An integer-only view of the counting sort output, so it is bit-free by construction; it removes the "
        "separate rows-build launches from the owner-combine preamble.",
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_OC_WIRE_STAGE",
        "1",
        ("engine",),
        "fc2 leaf-tree stores the owner-combine wire bytes straight into symm staging",
        DEPLOYMENT,
        (TRAIN,),
        notes="The fc2 leaf tree stores the owner-combine wire bytes straight into symmetric staging instead of "
        "producing them and copying them in. Refusals are named rather than silent, and a corrupted-wire "
        "control is part of the battery.",
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_PIK_BF16_WIRE",
        "1",
        ("engine",),
        "MoE pik combine ships bf16 leaves on the wire, FP32 root (m_leaves==1 only)",
        DEPLOYMENT,
        (TRAIN,),
        notes="Bit-identical by construction plus a composition battery (tree, then top-k sum, then round). The root "
        "stays fp32 because the downstream fixed-order sum dtype follows its input, so a bf16 root would change "
        "it. A first-call losslessness self-check falls back permanently; fires only when world == G.",
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_A2A_BF16_WIRE",
        "0",
        ("trainer",),
        "BACKWARD leg of the MoE combine all-to-all ships bf16 and widens back to fp32; forward "
        "wire untouched. 'probe' = census only (both directions tested, no wire changed)",
        DEPLOYMENT,
        (TRAIN,),
        notes="Installed on the EP>1 trainer combine dispatcher, which the engine allgather flow never reaches. "
        "DEPLOYMENT because no stored float value moves and the direction that could move one is untouched: the "
        "forward payload is an fp32 leaf-tree root that is not bf16-representable, so only the backward leg "
        'narrows. "probe" runs the census in both directions without changing any wire.',
    ),
    Flag(
        "SKYRL_ISOEXEC_SCORING_LOGITS_BF16",
        "0",
        ("trainer",),
        "the SCORING forward passes Megatron's own fp32_output=False, so the logits reach the logprob "
        "head in model precision and ChunkedDistributedLogprob widens per chunk as designed",
        DEPLOYMENT,
        (TRAIN,),
        notes="Read in the policy Ray actor, so it must be forwarded or a launch-shell value is a silent no-op. "
        "DEPLOYMENT because it moves WHERE an exact widening happens, not whether it happens: the logprob head "
        "widens per chunk as designed instead of the pipeline output being upcast wholesale.",
    ),
    Flag(
        "SKYRL_ISOEXEC_LOGPROB_GATHER_BF16_WIRE",
        "0",
        ("trainer",),
        "the IsoExec full-vocab log-softmax all_gather ships the low-precision dtype its payload was "
        "widened FROM (bf16/fp16) and widens back on arrival; the gathered fp32 buffer is unchanged",
        DEPLOYMENT,
        (TRAIN,),
        notes="DEPLOYMENT because narrow, exchange, widen is the identity on this payload rather than an "
        "approximation: the fp32 the gather branch sees is itself a widened bf16, so shipping the dtype it was "
        "widened from and widening back on arrival reproduces the gathered fp32 buffer exactly. Both the "
        "forward and the backward recompute call sites declare the source dtype.",
    ),
    Flag(
        "SKYRL_ISOEXEC_EXACT_SAMPLED_LOGPROBS",
        "0",
        ("trainer",),
        "the scoring-only IsoExec logprob head gathers the sampled raw target before the required "
        "full-vocabulary buffer is overwritten, fuses fp32 sub+exp, and never materializes the "
        "local-vocabulary logprob tensor",
        DEPLOYMENT,
        (TRAIN,),
        notes="Reached only from the scoring-only logprob head (inference_only=True). The full-vocabulary all_gather, "
        "amax, fp32 sum tree, log and sampled arithmetic are retained exactly; the twin removes one full- "
        "vocabulary pass by fusing sub and exp, captures the sampled raw value before the buffer is "
        "overwritten, and drops the local-shard result passes and the selected-value all_reduce.",
    ),
    Flag(
        "SKYRL_ISOEXEC_EXACT_VOCAB_PIPELINE",
        "0",
        ("trainer",),
        "scoring-only exact sampled-logprob path gathers low-precision vocabulary shards to one "
        "TP owner in rank order and broadcasts only the sampled FP32 result",
        DEPLOYMENT,
        (TRAIN,),
        notes="Read only inside the exact-sampled path after its unanimous admission. The owner reconstructs the same "
        "rank-ordered fp32 full row and runs the existing amax / fused sub-exp / sum / log schedule, so no "
        "floating reduction is distributed; the size dispatch keeps the replicated all-gather below a wire-size "
        "threshold and selects owner A2A above it.",
    ),
    Flag(
        "SKYRL_ISOEXEC_LOGPROB_BWD_REDUCED_COMM",
        "0",
        ("trainer",),
        "the log-softmax BACKWARD recompute drops the IsoExec full-vocab all_gather + cat and uses "
        "all_reduce(MAX)+all_reduce(SUM) on [B,chunk,1]; the FORWARD still gathers",
        DEPLOYMENT,
        (TRAIN,),
        notes="The forward is left statement for statement alone -- only the backward recompute takes the reduced- "
        "communication path -- so the gate, a forward quantity, cannot see this flag.",
    ),
    Flag(
        "SKYRL_ISOEXEC_SCORING_AUDIT_INTERVAL",
        "1",
        ("trainer",),
        "sampled gating: run the redundant scoring forward only on step 1 and every Nth step",
        DEPLOYMENT,
        (TRAIN,),
        notes="Read in the trainer driver. It moves no bits in any forward: N>1 simply suppresses the pre-update "
        "scoring gate on skipped steps, which are marked as unaudited rather than faked.",
    ),
    Flag(
        "SKYRL_ISOEXEC_SYNC_PIPELINE",
        "1",
        ("trainer",),
        "overlap the engine-side apply of chunk k with the trainer-side extract+pack of chunk k+1",
        DEPLOYMENT,
        (TRAIN,),
        notes="Transport scheduling only: same chunks, same packing, same per-tensor copies, exactly one apply in "
        "flight, and buffer retirement still gated on that apply returning.",
    ),
    Flag(
        "SKYRL_ISOEXEC_SYNC_BUCKET_MB",
        "512",
        ("trainer",),
        "IPC in-flight chunk size (transport, neutral)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    Flag(
        "SKYRL_ISOEXEC_ENGINE_LOAD_WEIGHTS",
        "1",
        ("engine",),
        "engine loads its own weights (bypass native sync)",
        DEPLOYMENT,
        (TRAIN, ENGINE),
    ),
    Flag(
        "SKYRL_ISOEXEC_ENGINE_ATTN_SKIP_GDN",
        "1",
        ("engine",),
        "do NOT register a paged Attention on linear-attention (GDN) layers",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Moves no bits and frees KV pool. On a hybrid model the GDN layers sit in the self_attention slot "
        "without a core_attention, so the generic swap creates a paged Attention on them that registers a KV- "
        "cache spec and is never called. The trainer twin swap has always refused them.",
    ),
    Flag(
        "SKYRL_ISOEXEC_ATTN_QKV_SUBGROUP_AG",
        "1",
        ("engine",),
        "megatron's fused-QKV gather runs over the num_query_groups subgroup, not the whole tp group",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Transport only -- an all-gather performs no arithmetic, so this cannot move a bit. When "
        "num_query_groups is smaller than the TP world, megatron gathers the whole fused-QKV width across the "
        "whole TP group and then keeps one slice of it; gathering over the query-group subgroup delivers the "
        "same bytes.",
    ),
    Flag(
        "SKYRL_ISOEXEC_FUSED_ROPE",
        "0",
        ("engine",),
        "fused attention RoPE (engine-only, bitwise-equal)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    # MLA.
    # Backward tuning knobs: trainer-only, and the backward carries no bitwise contract.
    Flag(
        "SKYRL_ISOEXEC_MOE_BWD_POINTWISE_ROWS",
        "16384",
        ("trainer",),
        "analytic expert VJP pointwise row chunk size (backward-only)",
        DIAGNOSTIC,
        (TRAIN,),
        notes="Backward-only chunk size; larger routed batches stay bounded to this many rows per fp32 temporary "
        "chunk.",
    ),
    # Shipping-path kill switches and off-by-default alternatives.
    Flag(
        "SKYRL_ISOEXEC_GDN_PADDED_DECODE",
        "0",
        ("engine",),
        "run the GDN chunk kernel over the FULL C rows of every open-chunk buffer on a static cu "
        "grid, instead of ragged per-slot fills",
        DEPLOYMENT,
        (TRAIN,),
        notes="Bitwise-neutral by the proven prefix-invariance property -- rows past the fill are stale buffer "
        "content and rows below it cannot see them -- so DEPLOYMENT, not FUNCTION. It costs roughly twice the "
        "GDN decode FLOPs and buys fully static decode shapes, the prerequisite for CUDA-graph capture, which "
        "is why it stays off rather than being deleted.",
    ),
    Flag(
        "SKYRL_ISOEXEC_GDN_SLOT_MAP_SIZE",
        "65536",
        ("engine",),
        "entries in the GDN slot->buffer map (512 KiB/layer at the default)",
        DEPLOYMENT,
        (TRAIN,),
        notes="A sizing constant of the shipped decode path, not an experiment. It must exceed the engine state-slot "
        "count and is sized once, because a captured CUDA graph holds this tensor address; the pools also take "
        "a construction-time hint from vLLM bound block space and raise at first step if that exceeds the built "
        "map.",
    ),
    Flag(
        "SKYRL_ISOEXEC_GDN_GROUP_ALIAS",
        "1",
        ("engine",),
        "alias every GDN layer's metadata to the canonical mamba KV-cache group (one slot-id space)",
        DEPLOYMENT,
        (TRAIN,),
        notes="A correctness mechanism, not a knob. ENGINE_ATTN_SKIP_GDN changes vLLM grouping heuristic and can "
        "split the GDN layers across several mamba KV-cache groups with separate block tables; without the "
        "alias the layers disagree on state slot ids and decode folds the unmapped ones onto null row 0. 0 "
        'refuses a multi-group geometry loudly; "unsafe" is the diagnostic repro arm.',
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_STATIC_DECODE",
        "1",
        ("both",),
        "sync-free, shape-static expert path whenever the routed-row count fits a bounded tile grid",
        DEPLOYMENT,
        (TRAIN,),
        notes="The shipped decode path and the CUDA-graph unlock. Prefill and the trainer forward fall through to the "
        "dynamic tile path by design. Installs on both runtimes.",
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_STATIC_MAX_ROWS",
        "131072",
        ("both",),
        "routed-row ceiling under which STATIC_DECODE takes the bounded-tile path",
        DEPLOYMENT,
        (TRAIN,),
        notes="A sizing constant of the shipped path, read at two sites that must not drift apart -- which is why it "
        "needs one forwarded value rather than two defaults.",
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_FIXED_DISPATCH",
        "1",
        ("both",),
        "fixed-shape permuted probs in dispatch_postprocess instead of megatron's masked_select",
        DEPLOYMENT,
        (TRAIN,),
        notes="Installed on both runtimes. The two forms are bitwise-equal, so the only difference is speed: the "
        "masked_select it replaces has a data-dependent, uncapturable shape.",
    ),
    # A flag absent from this table is in neither actor allowlist, so a launch-shell export reaches
    # only the launcher process. Every governed env read owes an entry here.
    #
    # Trainer-side kill switches read inside the policy actor.
    Flag(
        "SKYRL_ISOEXEC_VARLEN_ATTN",
        "1",
        ("trainer",),
        "LOCAL-SPEC path: swap the trainer's core_attention to the torch varlen_attn kernel the "
        "engine uses (num_splits=1, causal via window=(-1,0))",
        FUNCTION,
        (TRAIN,),
        notes="Reachable only under LOCAL_SPEC=1. The local-spec default is torch SDPA, a different kernel that "
        "leaves large per-token rollout-vs-train outliers; this swap is what drives the bitwise arm abs_diff to "
        "0.",
    ),
    Flag(
        "SKYRL_ISOEXEC_SPEC_PROVIDER",
        "0",
        ("trainer",),
        "generic IsoExec spec provider (subclasses LocalSpecProvider) instead of the hand-built spec",
        FUNCTION,
        (TRAIN,),
        notes="Read in the policy actor. Its CPU-level claim is the strongest available offline, so the live gate is "
        "what promotes it -- which requires that a launch shell can turn it on.",
    ),
    Flag(
        "SKYRL_ISOEXEC_MOE_INDEXED_BMM_DW_ELEMS",
        "268435456",
        ("both",),
        "transient element budget for the indexed-bmm wgrad's per-tile [chunk, K, N] product",
        DEPLOYMENT,
        (TRAIN,),
        notes="Companion of MOE_INDEXED_BMM and deliberately far below MOE_BMM_MAX_ELEMS: that module exists for peak "
        "memory, and a large budget hands part of the saved gather back as backward transients. Pure transient "
        "sizing, bitwise-neutral.",
    ),
    Flag(
        "SKYRL_ISOEXEC_EP_NCCL_DIR",
        "",
        ("trainer",),
        "directory the per-group NCCL channel YAML is written to",
        DEPLOYMENT,
        (TRAIN,),
        notes="Companion of EP_A2A_CHANNELS / EP_A2A_CHANNEL_GROUPS, read in the policy actor at setup_distributed. "
        "The code default is tempfile.gettempdir(), which is per-process rather than a literal -- and that is "
        "exactly why it must be forwardable: every local rank has to agree on the file.",
        default_dynamic=True,
    ),
    Flag(
        "SKYRL_ISOEXEC_PIK_AR_CROSSOVER",
        "legacy",
        ("both",),
        "which one-shot/two-shot crossover pik's all-reduce uses: legacy | arch | calibrate",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Read on both sides and raises on an unknown value, so an unforwarded export would leave the two "
        "runtimes on different crossovers -- a different branch of the same tree, but the arm would be "
        "measuring a split.",
    ),
    # Engine-side and both-side flags read in the vLLM actor or its worker subprocess.
    Flag(
        "SKYRL_ISOEXEC_COMPILE",
        "0",
        ("engine",),
        "opt-in to torch.compile under the compile guard; absent => compilation is forced OFF",
        FUNCTION,
        (TRAIN, ENGINE),
        notes="Consulted by the engine actor and reported by the worker subprocess, whose probe exists to surface an "
        "actor/worker disagreement. FUNCTION: inductor lowering that escapes the dispatcher moves bits, which "
        "is what the compile guard required-config check exists to prevent.",
    ),
    Flag(
        "SKYRL_ISOEXEC_GDN_CPR_HOST_META",
        "1",
        ("engine",),
        "host-built GDN chunk metadata instead of the vendored prepare_chunk_indices/offsets",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Bitwise-neutral by assertion rather than assumption -- the two paths produce equal tensors -- so it "
        "can only move time.",
    ),
    Flag(
        "SKYRL_ISOEXEC_GDN_SEQ_META_CACHE",
        "1",
        ("both",),
        "memoize the per-forward GDN sequence metadata (lens / cu clone / chunked cu / state "
        "indices) instead of rebuilding it in every GDN layer",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="A pure host-side memo of pure functions of cu_seqlens, keyed on the object identity plus its version "
        "counter with a strong reference held, so an identity match implies the same unmutated object and there "
        "is no value comparison to get wrong. It issues no device op a rebuild would not, so it can only move "
        "time.",
    ),
    Flag(
        "SKYRL_ISOEXEC_PACKED_META_CACHE",
        "1",
        ("both",),
        "memoize the packed-sequence HOST READS (cu_seqlens .tolist()/.cpu()) per forward instead " "of once per layer",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Memoizes the host reads of a packed thd forward -- the cu_seqlens .tolist() / .cpu() calls that each "
        "land in pageable host memory and therefore block -- once per forward instead of once per layer. Host- "
        "side only: no device op changes.",
    ),
    Flag(
        "SKYRL_ISOEXEC_PACKED_META_DIV_HOIST",
        "1",
        ("train",),
        "serve `_unpack_sequence(x, cu_seqlens_q // cp_size)`'s host list from a per-forward proof "
        "instead of reading it once per GDN layer",
        DEPLOYMENT,
        (TRAIN,),
        notes="A surgical kill switch for the one part of SKYRL_ISOEXEC_PACKED_META_CACHE that is not identity-keyed: "
        "the divisor expression allocates a fresh tensor at each call site, so no identity memo can hit it and "
        "the value is proved per forward instead. Setting 0 restores the per-layer host read and leaves the "
        "identity-keyed memo engaged.",
    ),
    Flag(
        "SKYRL_ISOEXEC_GDN_VALIDATE_ONCE",
        "0",
        ("train",),
        "run GatedDeltaNet's cp-divisibility validation once per forward instead of 4x per layer",
        DEPLOYMENT,
        (TRAIN,),
        notes="Deliberately off, and the distinction is the point: the rest of SKYRL_ISOEXEC_PACKED_META_CACHE removes "
        "memcpys and no kernels, while this one deletes the per-layer validation launches. Those kernels write "
        "only temporaries the host reads and discards, but the raw kernel sequence does get shorter, which is a "
        "full-treatment change rather than a host-only one.",
    ),
    Flag(
        "SKYRL_ISOEXEC_ARCH_FROM_HUB",
        "0",
        ("both",),
        "allow model-architecture resolution to hit the network instead of the local HF cache only",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="Off by default, and the default is the safety property: manifest construction runs on the launcher and "
        "in every worker, so a resolution step that can block on a network call is one that can hang the whole "
        "job on a DNS timeout. Registered on both channels because both sides resolve.",
    ),
    Flag(
        "SKYRL_ISOEXEC_MANIFEST_STRICT",
        "1",
        ("both",),
        "manifest-handshake mismatch is FATAL (0 => warn and continue)",
        DIAGNOSTIC,
        (TRAIN, ENGINE),
        notes="Fail-closed by default. A mismatch means the two runtimes resolved different compositions, so 0 is a "
        "debugging-only escape and must reach BOTH actors, or the handshake is strict on one side and advisory "
        "on the other.",
    ),
    Flag(
        "ISOEXEC_CONTRACT_PATH",
        "",
        ("both",),
        "path where every contract-building process writes contract.json (and enforce.py writes "
        "the enforcement.json verdict beside it)",
        DIAGNOSTIC,
        (TRAIN, ENGINE),
        notes="A launcher-exported value reaches no actor unless the name is in this table: "
        "actor_forwarding_tuple carries only registered flags, so an unregistered name means no worker "
        "writes contract.json. Registered on both channels so the artifact write hooks fire. Non-SKYRL_ "
        "name kept: existing launch scripts and contract_delivery already spell it.",
    ),
    Flag(
        "SKYRL_ISOEXEC_HANDSHAKE_NEGATIVE_CONTROL",
        "0",
        ("trainer",),
        "admission-battery negative control: perturb the trainer handshake composite",
        DIAGNOSTIC,
        (TRAIN,),
        notes="ON registers a trainer-only manifest extension so the engine receiver MUST refuse at weight sync "
        "(workers/worker.py stamp site). Proves the handshake detects a mismatch; never ON in production.",
    ),
    Flag(
        "SKYRL_ISOEXEC_DEBUG_TRACE",
        "",
        ("both",),
        "debug mode master switch: trace directory for per-region output digests",
        DIAGNOSTIC,
        (TRAIN, ENGINE),
        notes="Set -> the ContractAdapter installs region hooks on both sides and EVERY enforcement refusal demotes "
        "to logged-and-continue (core/enforce.demoted), so any kernel mix runs and the trace localizes the "
        "divergence. The ledger is untouched: every demoted violation is still recorded and the verdict stays "
        "red. Its own condition, not an alias for MANIFEST_STRICT=0. Run the engine eager: replayed CUDA graphs "
        "execute no Python and record nothing.",
    ),
    Flag(
        "SKYRL_ISOEXEC_DEBUG_SIDE",
        "",
        ("both",),
        "trace record label; stamped per-process by the ContractAdapter, not by hand",
        DIAGNOSTIC,
        (),
        should_forward=False,
        notes="The one debug env that must NOT be forwarded: each process stamps its own side in "
        "ContractAdapter._install_debug_trace, so carrying a launch-shell value to both actors would label the "
        "engine's records 'trainer'. Forwarding it would BE the bug, hence should_forward=False rather than a "
        "latent split-brain entry.",
    ),
    Flag(
        "SKYRL_ISOEXEC_DEBUG_SAMPLE",
        "",
        ("both",),
        "record every Nth forward (step-keyed via debug.set_step)",
        DIAGNOSTIC,
        (TRAIN, ENGINE),
    ),
    Flag(
        "SKYRL_ISOEXEC_DEBUG_LADDER",
        "",
        ("both",),
        "also record the mantissa-truncation k-ladder per region",
        DIAGNOSTIC,
        (TRAIN, ENGINE),
    ),
    Flag(
        "SKYRL_ISOEXEC_DEBUG_RING",
        "",
        ("both",),
        "in-memory record buffer size before a flush",
        DIAGNOSTIC,
        (TRAIN, ENGINE),
    ),
    Flag(
        "SKYRL_ISOEXEC_DEBUG_REGIONS",
        "",
        ("both",),
        "comma allow list of regions to trace (default: every hooked region except mm)",
        DIAGNOSTIC,
        (TRAIN, ENGINE),
    ),
    Flag(
        "SKYRL_ISOEXEC_DEBUG_DIGEST",
        "",
        ("both",),
        "digest backend override: 'eager' forces the non-triton path",
        DIAGNOSTIC,
        (TRAIN, ENGINE),
        notes="Both sides must run the same backend only for perf parity -- the digests are bit-identical "
        "across backends by test; forwarded so a pinned choice reaches both actors.",
    ),
    Flag(
        "SKYRL_ISOEXEC_DEBUG_SEGMENTS",
        "",
        ("both",),
        "row-segment digests: one digest per N rows of dim 0",
        DIAGNOSTIC,
        (TRAIN, ENGINE),
        notes="Lets the comparator separate a fault localized to some rows from a whole-tensor round-off/reduction "
        "-order difference, which the k-ladder cannot. Unregistered it reached no actor, so it worked only for "
        "single-process runs.",
    ),
    Flag(
        "SKYRL_ISOEXEC_LIFECYCLE_ASSERTS",
        "1",
        ("both",),
        "lifecycle ordering invariant checks ([ISOEXEC-LIFECYCLE] INVARIANT VIOLATED)",
        DIAGNOSTIC,
        (TRAIN, ENGINE),
        notes="An invariant check that is on in one runtime and off in the other is worse than off in both, so it is "
        "forwarded on both channels.",
    ),
    Flag(
        "SKYRL_ISOEXEC_AUTOFUSE_FAST_CFG",
        "0",
        ("both",),
        "build a compiled region's inductor-config pin ONCE instead of on every call",
        DEPLOYMENT,
        (TRAIN, ENGINE),
        notes="A compiled region rebuilds its inductor-config pin on every call; this builds it once. Host-time only "
        "-- the pin content is unchanged, so the compiled code and its bits are unchanged.",
    ),
    Flag(
        "SKYRL_ISOEXEC_AUTOFUSE",
        "1",
        ("both",),
        "install ledger-ADMITTED compiled pointwise/copy regions (bit-proven per site+shape)",
        DEPLOYMENT,
        (TRAIN,),
        notes="The flag only lets the ledger be consulted: a site without an ADMITTED entry under the exact "
        "(region_sig, shape, arch, torch, config) fingerprint stays eager on both sides, so a stale or foreign "
        "ledger is inert rather than wrong. The neutrality proof is the per-shape bitwise gate verdict recorded "
        "in the ledger.",
    ),
    Flag(
        "SKYRL_ISOEXEC_AUTOFUSE_LEDGER",
        "",
        ("both",),
        "path of the fusion-decision ledger (default ~/.cache/skyrl-isoexec/autofuse.json)",
        DEPLOYMENT,
        (TRAIN,),
        notes="Must resolve to the SAME file contents on trainer and engine: the ledger digest is part of the "
        "autofuse manifest pin, so divergence refuses at weight sync. The default path is generic rather than "
        "current -- a ledger whose fingerprint does not match this tree simply misses on every entry, which is "
        "safe but also inert, so the install banner reports how many entries match this (arch, torch, config) "
        "triple.",
    ),
    Flag(
        "SKYRL_ISOEXEC_BWD_COMPILE",
        "0",
        ("train",),
        "compile admitted BACKWARD-only regions (autofuse/bwd_compile.py); decision comes from a "
        "persisted ledger, never from a runtime benchmark",
        FUNCTION,
        (TRAIN,),
        notes="TRAIN-only by construction: every registered site is reachable exclusively from an "
        "autograd.Function.backward, so the engine never executes one and the forward gate is untouched. "
        "FUNCTION because a compiled region reassociates backward reductions -- legal, since the backward "
        "carries no bitwise contract, but it moves gradients and the gate compares two no_grad forwards, so it "
        "cannot see that. Admission is an fp64-oracle battery, not a bit check.",
    ),
    Flag(
        "SKYRL_ISOEXEC_BWD_COMPILE_LEDGER",
        "",
        ("train",),
        "path of the backward-region compile ledger (default ~/.cache/skyrl-isoexec/bwd_compile.json)",
        DEPLOYMENT,
        (TRAIN,),
    ),
    Flag(
        "SKYRL_ISOEXEC_BWD_COMPILE_ROLE",
        "reader",
        ("train",),
        "reader (default) consumes verdicts and can never write one; writer is the offline battery",
        DEPLOYMENT,
        (TRAIN,),
        notes="a production step must NEVER be a writer: a verdict written from a live step would "
        "make the pinned artifact a function of that step's shapes, which is exactly the "
        "run-to-run gradient drift the ledger exists to remove. record() raises in a reader.",
    ),
]


def actor_forwarding_list(channel: str | None = None) -> List[str]:
    """The flag names this table says a channel forwards, sorted and de-duplicated.

    ``channel=None`` gives every flag needing forwarding minus ``SKYRL_ISOEXEC``, which gates the
    whole loop rather than sitting in it.
    """
    if channel is not None:
        return sorted({f.name for f in FLAGS if channel in f.forwarded_by})
    return sorted({f.name for f in FLAGS if f.should_forward and f.name != "SKYRL_ISOEXEC"})


# Flags forwarded by a direct ``env_vars["X"] = ...`` assignment rather than the generic allowlist
# loop; ``actor_forwarding_tuple``, the loop's iterable, excludes them.
_DIRECT_FORWARDS = frozenset(
    {
        "SKYRL_ISOEXEC",
        "FLA_TILELANG",
        "NCCL_ALGO",
        "NCCL_MIN_NCHANNELS",
        "NCCL_MAX_NCHANNELS",
    }
)


def actor_forwarding_tuple(channel: str) -> List[str]:
    """Flags forwarded by a channel's generic loop; direct assignments are excluded."""
    return sorted(set(actor_forwarding_list(channel)) - _DIRECT_FORWARDS)


def latent_split_brain() -> List[Flag]:
    """Flags read remotely that require forwarding but declare no channel."""
    return [f for f in FLAGS if f.is_latent_split_brain]


def by_disposition(disposition: str) -> List[Flag]:
    return [f for f in FLAGS if f.disposition == disposition]


def get(name: str) -> Flag:
    for f in FLAGS:
        if f.name == name:
            return f
    raise KeyError(name)
