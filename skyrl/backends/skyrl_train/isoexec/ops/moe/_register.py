"""Registry entries for the MoE op family.

MoE is the main case of legitimate SITE-ASYMMETRY: expert weights are fused ENGINE-ONLY -- the rollout
runs vLLM's padding-free fused MoE GEMM while the trainer runs the padded batched ``bmm`` -- because
fusing both sides loses on the backward. IsoExec survives the split only because the two forwards are
BITWISE-EQUAL, so a model manifest (not this module) maps the trainer sites to the ``bmm`` impl and the
engine sites to the ``fused`` impl.

DECLARATIVE metadata only: this module wires no behavior and imports no kernels. It records what the
shipped MoE kernels are, per the header verdict in each source file; not-shipped kernels are recorded in
``documentary`` with no manifest claim.
"""

from __future__ import annotations

from ...core.registry import (
    PER_MODEL,
    ImplSpec,
    OpSpec,
    RoundingSchedule,
    StateInvalidation,
)


def register(reg) -> None:
    _register_router(reg)
    _register_dispatch(reg)
    _register_experts(reg)
    _register_combine(reg)
    _register_weights(reg)
    _register_epilogue(reg)
    _register_blockmap(reg)


# moe.router -- fp32 router gating GEMM + sorted top-k + dense (routing_probs, routing_map).
def _register_router(reg) -> None:
    op = OpSpec(
        name="moe.router",
        sites=["trainer_fwd", "trainer_score", "engine_prefill", "engine_decode"],
    )
    # `deterministic`: the shipped BOTH-SIDES router. fp32 gating (moe_router_dtype=fp32), torch.topk
    # forced sorted=True in both grad and no-grad forwards, softmax over the top-k scores, dense
    # index_put_ scatter to (routing_probs, routing_map). Shipped: SKYRL_ISOEXEC_MOE_DETERMINISTIC=1.
    op.add_impl(
        ImplSpec(
            impl_id="deterministic",
            version=1,
            supported_archs=frozenset({"sm90"}),
            rounding=RoundingSchedule(
                machine_assertable={
                    "router_dtype": "fp32",
                    "topk_sorted": True,  # forced True in BOTH grad modes (was sorted=is_grad_enabled)
                    "score_function": "softmax",
                    # membership/probs are order-free wrt tied-group column order, but with
                    # pre_softmax=False the post-top-k softmax runs over the returned order, so the
                    # sorted=True forcing is load-bearing exactly there.
                    "pre_softmax_dependent": True,
                },
                documentary=(
                    "fp32 router: torch.topk(k, sorted=True) both sides, softmax over top-k scores, "
                    "dense index_put_ scatter. Reference impl at all four sites. Batch-invariant via "
                    "the global mm/addmm override (the fp32 gating GEMM is NOT cuBLAS row-invariant: "
                    "measured 4.3e-5 row drift). The fused_topk router kernel "
                    "(SKYRL_ISOEXEC_MOE_FUSED_ROUTER) is a BOTH-SIDES bitwise-on-membership alternative "
                    "but is NOT WIRED (default OFF, absent from the shipped recipe): recorded here, "
                    "not selected."
                ),
            ),
            capabilities={"cuda_graph": True},
            hazards=["tie_boundaries"],
        )
    )
    # `fused_o2`: O2 fused router (top-k + softmax + dense scatter + the dispatcher permute sort) in
    # three Triton kernels. ENGINE-ONLY, on MARKED instances (mark_engine_router_o2). Bitwise-equal to
    # `deterministic` BY MEMBERSHIP: the DENSE (routing_probs, routing_map) output is bitwise-identical
    # to torch's, ties included (tie rule = lower expert index wins; the tied-group column permutation
    # never leaves the kernel), so a bitwise-equal replacement is safe ONE-SIDED -- the strengthening
    # that lets it be engine-only. ENV SKYRL_ISOEXEC_MOE_ROUTER_O2, shipped ENGINE-only.
    op.add_impl(
        ImplSpec(
            impl_id="fused_o2",
            version=1,
            supported_archs=frozenset({"sm90"}),
            rounding=RoundingSchedule(
                machine_assertable={
                    "router_dtype": "fp32",
                    "tie_rule": "lower_expert_index_wins",
                    "emits_dense": ["routing_probs[T,E]", "routing_map[T,E]"],
                    "engine_only_marked_instance": True,  # mark_engine_router_o2; unmarked -> megatron
                },
                documentary=(
                    "ENGINE-ONLY (marked instances). Bitwise-equal-by-MEMBERSHIP to `deterministic`: "
                    "amax/denominator over a tied group are order-free and the scatter is keyed on "
                    "expert index, so the DENSE router output is byte-identical to the eager sequence "
                    "ties included -- a bitwise-equal replacement is safe one-sided (the equivalence "
                    "proof this site-asymmetry owes; moe_router_fused_test.py). RECONCILIATION: the "
                    "MoE design note that 'the router is NEVER fused' warns that re-deriving top-k "
                    "could break tie ORDER = a different model. This impl does NOT re-derive "
                    "membership (proved bitwise), and it emits the dense map whose tied-column order "
                    "no consumer can observe. Class patch installed both sides; delegates to megatron "
                    "for every unmarked (i.e. trainer) instance."
                ),
            ),
            capabilities={"cuda_graph": True, "engine_only": True, "bitwise_equal_to": "deterministic"},
            # profiling_shapes: the non-finite gating rows at vLLM engine init crashed this kernel and
            # the offline gate never saw them. tie_boundaries: the k=8 membership boundary.
            hazards=["tie_boundaries", "profiling_shapes"],
        )
    )
    # `deterministic_sigmoid_bias`: the DeepSeek `noaux_tc` routing GLM-4.7-Flash uses. Same op --
    # logits -> (routing_probs, routing_map) -- with a different score function and a SELECTION-ONLY
    # bias, so it is a distinct impl rather than a pinned constant on `deterministic`:
    #
    #     scores  = sigmoid(logits.float())                              fp32
    #     sel     = topk(scores + expert_bias.float(), k, sorted=True)    <- bias steers SELECTION
    #     probs   = gather(scores, sel)                                   <- but NOT the weights
    #     probs   = probs / (probs.sum(-1, keepdim=True) + 1e-20)         <- norm_topk_prob
    #     probs   = probs * routed_scaling_factor                         <- 1.8 on GLM-4.7-Flash
    #
    # NO FUSED ENGINE TWIN OF THE SCORE CHAIN. The normalization's denominator is an ATen row-sum over
    # [T, k] fp32, and ATen's reduction order matches none of the candidate chains, so a Triton twin
    # cannot reproduce it bit for bit and would be a change requiring BOTH runtimes to move. It also
    # buys nothing IsoExec needs: the eager chain is ALREADY batch-invariant (bitwise prefix-invariant
    # over T for both the row-sum and torch.sigmoid). Invariance is the requirement; fusion was only the
    # perf bonus, and here it is unavailable.
    #
    # A SECOND, INDEPENDENT REASON THE ROUTER MAY NOT BE RE-DERIVED HERE. `fused_o2` is engine-only
    # because of a theorem: inside an exactly-tied group, permuting the selected columns permutes EQUAL
    # values, so the softmax denominator's summand sequence is unchanged. THAT THEOREM IS FALSE FOR THIS
    # ROUTER. Selection runs on `scores + expert_bias` while the denominator sums
    # `gather(scores, top_indices)` -- a DIFFERENT tensor. Two experts can tie exactly in the selection
    # key while their raw sigmoid scores differ, so permuting them permutes UNEQUAL entries through
    # ATen's fixed positional sum tree and the denominator moves in the last ulp. Tie ORDER, not just
    # membership, is part of this router's bitwise contract.
    #
    # WHAT IS NOT REFUSED -- THE MECHANICS. Everything above is about ARITHMETIC. The routing block also
    # contains kernels that perform NO floating-point arithmetic at all: two `zeros_like`, an `.int()`,
    # two out-of-place `.scatter()` clones, the scatter pair, a `.bool()`, a bool->int64 promotion, a
    # column sum, a boolean transpose, a cub radix argsort, a `remainder`, and a transpose+gather of
    # already-computed fp32 words. `moe_dense_scatter_kernel.py` replaces those with 3 Triton launches
    # while every kernel of the score chain still runs, in the same order, launched by megatron's own
    # code -- the fused path takes `topk_routing_with_score_function`'s own `dense_output=True` early
    # return, so there is no second transcription of the sigmoid, the bias add, the top-k, the gather,
    # the row-sum, the epsilon, the divide or the scaling factor anywhere. Flags
    # SKYRL_ISOEXEC_MOE_DENSE_SCATTER / SKYRL_ISOEXEC_MOE_PERMUTE_SORT, both engine-only on marked
    # instances, and both CONDITIONAL ON ep_size == 1, which the shipped engine sets and both predicates
    # re-check at runtime. The neutrality proof is scoped per (op, site) and covers four entries:
    # (moe.router, engine_decode), (moe.router, engine_prefill), (moe.dispatch, engine_decode),
    # (moe.dispatch, engine_prefill), against BOTH of megatron's dense branches (index_put_ AND
    # .scatter) so it does not rest on which one `torch.are_deterministic_algorithms_enabled()` picks.
    # trainer_fwd / trainer_score are NOT claimed and need no proof: the class patches fire only on
    # instances the ENGINE model build marked.
    #
    # expert_bias is a BUFFER, not a parameter -- it rides the weight sync via
    # native_weight_sync.SYNCED_BUFFER_SUFFIXES in its native fp32 (bf16-rounding a bias that decides
    # top-k membership is a different model, not a rounding difference).
    op.add_impl(
        ImplSpec(
            impl_id="deterministic_sigmoid_bias",
            version=1,
            supported_archs=frozenset({"sm90"}),
            rounding=RoundingSchedule(
                machine_assertable={
                    "router_dtype": "fp32",
                    "topk_sorted": True,
                    "score_function": "sigmoid",
                    "expert_bias": "selection_only",
                    "expert_bias_dtype": "fp32",
                    "norm_topk_prob": True,
                    # The four keys below are MANIFEST PIN KEYS (policy.derive_selections spells them
                    # topk/experts/scaling_factor/eps out of the RouterProfile), so they are declared
                    # under exactly those names -- a declaration the manifest cannot be checked against
                    # is not a declaration.
                    "eps": 1e-20,  # the norm_topk_prob denominator epsilon; fixed by the chain
                    "topk": PER_MODEL,  # k -- a model fact (4 on GLM-4.7-Flash)
                    "experts": PER_MODEL,  # E -- a model fact (64 on GLM-4.7-Flash)
                    "scaling_factor": PER_MODEL,  # routed_scaling_factor (1.8 on GLM-4.7-Flash)
                    # NOT per-model: this impl runs the group-limited path BYPASSED. A profile that
                    # pins a group_topk (a DeepSeek-shaped router with n_group > 1) is describing a
                    # routing chain this function does not contain, and validate_pins refuses it
                    # rather than deriving a clean manifest that names the wrong router.
                    "group_topk": None,  # n_group == 1 -> group_limited_topk bypassed (see below)
                },
                documentary=(
                    "megatron topk_routing_with_score_function, score_function='sigmoid' + "
                    "expert_bias branch, deterministic-algorithms dense build. ONE impl at all four "
                    "sites -- no site asymmetry, so no equivalence proof owed. The n_group=1 / "
                    "topk_group=1 config is normalized to group_topk=None by "
                    "runtimes/megatron/mla_spec.force_glm47_flash_provider: with a single group "
                    "group_limited_topk masks nothing (masked_scores is a value-identical copy) and "
                    "runs the same torch.topk, so the bypass is bitwise-identical and drops a "
                    "view/topk/sum/masked_fill chain per layer. Batch-invariance of the fp32 gating "
                    "GEMM comes from the global mm override, as for `deterministic`."
                ),
            ),
            capabilities={"cuda_graph": True},
            # tie_boundaries: the k=4 membership boundary, now decided by score+bias rather than
            # score. subnormals: sigmoid saturates toward 0 for very negative logits, so the
            # denominator can carry subnormal summands on a cold router.
            hazards=["tie_boundaries", "subnormals"],
        )
    )
    reg.register_op(op)


# moe.dispatch -- permute/index build: (sorted_indices, permuted_probs, tokens_per_expert). Bit-free.
def _register_dispatch(reg) -> None:
    op = OpSpec(
        name="moe.dispatch",
        sites=["trainer_fwd", "trainer_score", "engine_prefill", "engine_decode"],
    )
    # `index_build`: megatron's stable-sort permute (expert-major, token-ascending) + the fixed-shape
    # permuted-probs gather. Pure data movement -- a stable argsort of a bool/index key and a gather --
    # no arithmetic, so bit-free and both-sided by construction.
    op.add_impl(
        ImplSpec(
            impl_id="index_build",
            version=1,
            supported_archs=frozenset({"sm90"}),
            rounding=RoundingSchedule(
                machine_assertable={
                    "permutation": "expert_major_token_ascending",
                    "bit_free": True,  # permutation + gather, no fp arithmetic
                    "outputs": ["sorted_indices", "permuted_probs", "tokens_per_expert"],
                },
                documentary=(
                    "Bit-free permute/dispatch metadata: stable argsort of the routing key (unique "
                    "permutation) + fixed-shape permuted-probs gather (bitwise-equal to the "
                    "dispatcher's masked_select). Both sides. The counting-sort replacement for the "
                    "argsort/remainder/gather index build (moe_router_o2_kernel.fused_permute_index) "
                    "is likewise recorded here and SELECTED on the GLM recipes; it is reachable under "
                    "SKYRL_ISOEXEC_MOE_ROUTER_O2 and, independently, under "
                    "SKYRL_ISOEXEC_MOE_PERMUTE_SORT (which is what the GLM launchers export; the "
                    "per-(op,site) neutrality proof is "
                    "examples/isoexec/nightly/moe_routing_site_parity.py, covering engine_decode AND "
                    "engine_prefill). The split from the router flag exists "
                    "because this half never looks at a score function -- it counting-sorts a boolean "
                    "map and gathers fp32 words -- while the O2 ROUTER half correctly declines a "
                    "sigmoid+expert_bias router, so on GLM-4.7-Flash the coupled flag left a "
                    "score-function-agnostic index build unreachable. Both predicates carry "
                    "the same load-bearing EP=1 guard, which the proof is conditional on."
                ),
            ),
            capabilities={"cuda_graph": True},
            # e_mismatch: rows-per-expert / expert-count layout. null_lanes: padded/empty tile slots
            # that store zeros in the (not-shipped) fused staging path and its graph-replay inertness.
            hazards=["e_mismatch", "null_lanes"],
        )
    )
    # `chunk_sort_gather`: the alltoall dispatcher's inter-rank chunk reordering as ONE row gather
    # instead of megatron's `torch.cat` over a python loop of `num_splits` slices. Recorded and NOT
    # SELECTED (SKYRL_ISOEXEC_MOE_CHUNK_SORT, default OFF). See moe_chunk_sort.py.
    op.add_impl(
        ImplSpec(
            impl_id="chunk_sort_gather",
            version=1,
            supported_archs=frozenset({"sm90"}),
            rounding=RoundingSchedule(
                machine_assertable={
                    "permutation": "chunk_reorder_bijection",
                    "bit_free": True,  # index_select forward, index_select backward -- no add anywhere
                    "vjp": "inverse_gather",  # NOT index_add_: 0.0 + -0.0 would flip signed zeros
                    "bit_equal_to": "megatron_cat_chunk_loop",  # enforced per shape at first use
                },
                documentary=(
                    "The ONE member of TransformerEngine's fused permutation family that is copy-only "
                    "in BOTH directions, and it is reached here with no TE and no new kernel. "
                    "Megatron's non-fused sort_chunks_by_idxs is a python loop over num_splits slices "
                    "concatenated with torch.cat (CatArrayBatchedCopy x2 per MoE layer); the "
                    "reordering is a bijection on rows, so the identical bytes are one index_select "
                    "and the VJP is the inverse gather. Admitted per (splits, hidden, dtype, rows) at first use on the "
                    "LIVE operands -- forward bits, row provenance against an index marker, backward "
                    "bits through both forms with one shared cotangent, round-trip bijectivity, "
                    "determinism -- so a failing shape runs megatron's cat and no bits move. TE's own "
                    "_moe_chunk_sort.backward has the same shape (_sort_chunks_by_map_kernel with "
                    "FORWARD=False). Its siblings are NOT reachable and deliberately so: fused "
                    "`permute`'s VJP and fused `unpermute`'s forward are both an fp32 accumulation "
                    "over the top-k contributions, i.e. arithmetic, not packaging."
                ),
            ),
            capabilities={"cuda_graph": True, "bitwise_equal_to": "megatron_cat_chunk_loop"},
            hazards=["e_mismatch"],
        )
    )
    reg.register_op(op)


# moe.experts -- the expert GEMMs. SITE-ASYMMETRIC: trainer bmm vs engine fused.
def _register_experts(reg) -> None:
    op = OpSpec(
        name="moe.experts",
        sites=["trainer_fwd", "trainer_score", "engine_prefill", "engine_decode"],
    )
    # `batched_bmm`: the TRAINER reference. Padded batched GEMM pair (torch.bmm on stacked [E,...]
    # weights) replacing the 256-iteration SequentialMLP python loop; probs folded into the epilogue
    # before fc2. Batch-invariant via aten::bmm override. Runs forward AND backward on the trainer.
    op.add_impl(
        ImplSpec(
            impl_id="batched_bmm",
            version=1,
            supported_archs=frozenset({"sm90"}),
            rounding=RoundingSchedule(
                machine_assertable={
                    "kernel": "aten::bmm_batch_invariant",
                    "grouped_gemm": False,  # SequentialMLP pin; grouped GEMM is batch-variant
                    "tile_rows": 128,  # SKYRL_ISOEXEC_MOE_TILE_ROWS, multiple of bmm BLOCK_SIZE_M=128
                    "bmm_max_elems": 2**30,  # split the bmm tile axis under the int32 2^31 OOB limit
                },
                documentary=(
                    "TRAINER path (also the engine fallback when fused is off). NOT bitwise-equal to "
                    "the sequential loop and need not be -- both runtimes ran the SAME function until "
                    "the engine fused split; per-token invariance to other tokens' routing and to the "
                    "M_pad padding reduces to aten::bmm batch-invariance. Under SKYRL_ISOEXEC_MOE_PIK_FC2 "
                    "the fc2 bmm reduces the moe_intermediate as a fixed fp32 leaf tree (leaves=8) so a "
                    "trainer at ETP=C matches an engine at ETP!=C bitwise. Weight materialization "
                    "(torch.stack) is part of THIS impl -- hence moe.weights has no trainer site."
                ),
            ),
            capabilities={"cuda_graph": True},
            hazards=["e_mismatch", "non_contiguous"],
        )
    )
    # `fused`: the ENGINE path. vLLM fused_moe Triton GEMM (inner invoke_fused_moe_triton_kernel on
    # expert-grouped rows, top_k=1), shape-static and sync-free (CUDA-graph unlock). Under pik-fc2 the
    # SAME leaf tree runs IN-KERNEL (moe_fused_leaftree). BITWISE-EQUAL to batched_bmm -- the
    # site-asymmetry's equivalence proof -- so it is installed engine-only and self-checked on step 1.
    op.add_impl(
        ImplSpec(
            impl_id="fused",
            version=1,
            supported_archs=frozenset({"sm90"}),
            rounding=RoundingSchedule(
                machine_assertable={
                    "kernel": "vllm_fused_moe_triton_kernel",
                    "block_sizes": {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1},
                    "top_k": 1,  # each permuted row is already one (token, expert) pair
                    "shape_static": True,  # grid from a host-known bound; num_tokens_post_padded gates
                    "pik_leaftree_in_kernel": True,  # moe_fused_leaftree when SKYRL_ISOEXEC_MOE_PIK_FC2=1
                },
                documentary=(
                    "ENGINE-ONLY. Bitwise-equal to batched_bmm's forward (moe_fused_vs_bmm_test.py "
                    "max|diff|=0.0; the router prob applied to the intermediate before fc2 with the "
                    "same .to(dtype) round). Engine-only because fusing the trainer side too LOSES on "
                    "the backward (padding-free forward, but every bmm-shaped backward rebuilds the "
                    "padding -> net slower); the rollout has no backward. Batch-invariance from "
                    "VLLM_BATCH_INVARIANT=1 pinning BLOCK_M/N/K + SPLIT_K=1 (no tuned-config lookup, "
                    "no split-K). Under pik-fc2 the fc2 leaf tree runs in-kernel (moe_fused_leaftree), "
                    "which is ~1.9-2.4x native BY CONSTRUCTION -- the fp32 partial store + tree "
                    "liveness ARE the ETP-invariance contract, not a perf bug; the ratio is "
                    "routing-dependent."
                ),
            ),
            capabilities={"cuda_graph": True, "engine_only": True, "bitwise_equal_to": "batched_bmm"},
            # null_lanes: graph-padded blocks past num_tokens_post_padded early-return (inert on replay).
            hazards=["e_mismatch", "non_contiguous", "null_lanes"],
        )
    )
    # `grouped_cublaslt`: the TRAINER fc1 as a real GROUPED GEMM -- one PINNED non-split-K cuBLASLt call
    # per local expert over its contiguous staged rows, no padding-tile bmm, no weight gather. It
    # replaces only the fc1 GEMM inside batched_bmm and is admitted per (K,N) ONLY IF bit-equal to it, so
    # the composition's rounding schedule is unchanged by construction.
    # ENV SKYRL_ISOEXEC_MOE_EXPERT_CUBLASLT, default OFF.
    op.add_impl(
        ImplSpec(
            impl_id="grouped_cublaslt",
            version=1,
            supported_archs=frozenset({"sm90"}),
            rounding=RoundingSchedule(
                machine_assertable={
                    "kernel": "cublasLtMatmul (pinned)",
                    "split_k": 1,  # read back off the pinned algo; a splitK!=1 pin is refused
                    "reduction_scheme": "NONE",
                    "algo_key_excludes": ["M", "group_sizes"],  # why a row's K-order cannot move
                    "scope": "fc1",  # fc2 is the pik leaf tree; see the module docstring
                    "bit_equal_to": "batched_bmm",  # enforced per (K,N) at first use, else not taken
                },
                documentary=(
                    "TRAINER-ONLY, DEFAULT OFF, and BITWISE-NEUTRAL BY CONSTRUCTION. The launcher pin "
                    "moe_grouped_gemm=false rests on 'grouped GEMM's tile schedule depends on the "
                    "per-expert token counts'; that is a property of split-K and of size-keyed "
                    "heuristic algo selection, not of grouping. TE 2.11's te_general_grouped_gemm is "
                    "itself one cublasLtMatmul per expert over 4 streams (nvte_multi_tensor_gemm; the "
                    "CUTLASS grouped kernel is compiled in but dark behind NVTE_USE_CUTLASS_GROUPED_"
                    "GEMM=1) -- i.e. this impl minus the pin. Admission runs five live-operand gates "
                    "per (K,N): bit-equality vs batched_bmm's staged-tile bmm, GROUP-SIZE invariance, "
                    "M-invariance within an expert, determinism, and the splitK/reduction read-back. "
                    "A shape that fails is dropped loudly and permanently and the caller runs "
                    "batched_bmm unchanged, so the engine's `fused` impl needs no counterpart and the "
                    "site asymmetry's equivalence proof is untouched. Gradients carry no bitwise "
                    "contract: dgrad/wgrad are one aten::mm each per expert, which also removes the "
                    "fp32 tile-axis segment-sum moe_indexed_bmm's wgrad needs."
                ),
            ),
            capabilities={"cuda_graph": False, "trainer_only": True, "bitwise_equal_to": "batched_bmm"},
            hazards=["e_mismatch", "non_contiguous"],
        )
    )
    reg.register_op(op)


# moe.combine -- top-k combine (unpermute + weighted sum of each token's k expert rows).
def _register_combine(reg) -> None:
    op = OpSpec(
        name="moe.combine",
        sites=["trainer_fwd", "trainer_score", "engine_prefill", "engine_decode"],
    )
    # `fixed_order`: the deterministic combine that replaces megatron's scatter_add_ (atomicAdd, NON-
    # deterministic under top-k>1). Sum each token's k rows in ASCENDING EXPERT order with exactly ONE
    # bf16 round after the sum. Both sides (SKYRL_ISOEXEC_MOE_DETERMINISTIC=1). The fused Triton variant
    # is moe_combine_kernel.py (SKYRL_ISOEXEC_MOE_FUSED_COMBINE=1).
    op.add_impl(
        ImplSpec(
            impl_id="fixed_order",
            version=1,
            supported_archs=frozenset({"sm90"}),
            rounding=RoundingSchedule(
                machine_assertable={
                    "accumulation_order": "ascending_expert_index",
                    "one_bf16_round_after_sum": True,
                    # bf16 sequential accumulation when expert output is bf16 (pik off); plain fp32
                    # sum + single round when expert output is fp32 (pik on).
                    "sum_dtype": "matches_expert_output_dtype",
                },
                documentary=(
                    "Fixed-order top-k combine: no atomics, no cross-token reduction (replaced CUDA "
                    "scatter_add_ atomicAdd, which is run-to-run nondeterministic and batch-variant at "
                    "top-k>1). Both sides. The fused Triton combine (moe_combine_kernel.py) is a "
                    "BITWISE-EQUAL accelerated variant, installed ENGINE-ONLY by instance mark "
                    "(mark_engine_fused_combine; an unconditional rebind would hand the trainer a "
                    "grad_fn-less Triton call and sever its MoE backward). Bitwise-equal -> safe "
                    "one-sided; SKYRL_ISOEXEC_MOE_COMBINE_SORT further swaps the stable argsort for a "
                    "counting sort, torch.equal-gated."
                ),
            ),
            capabilities={"cuda_graph": True},
            hazards=["tie_boundaries", "e_mismatch"],
        )
    )
    # `pik_leaf_tree`: the ETP-INVARIANT combine (SKYRL_ISOEXEC_MOE_PIK_FC2=1, SHIPPED). Reduce the expert
    # fc2 output across the ETP (tp_ep) group with pik's FIXED TREE *before* the top-k combine, so
    # token_combine only scatters. Pairs with the leaf-tree fc2 (moe.experts), and removes the
    # trainer-ETP != engine-ETP residual.
    op.add_impl(
        ImplSpec(
            impl_id="pik_leaf_tree",
            version=1,
            supported_archs=frozenset({"sm90"}),
            rounding=RoundingSchedule(
                machine_assertable={
                    # G = SKYRL_ISOEXEC_PIK_LEAVES; 8 on every shipped recipe, but a per-deployment
                    # fact (>= max TP either side runs), so the MODEL pins the value and the impl
                    # declares only that it IS pinned. Same key name as collectives:pik_tree.
                    "leaves": PER_MODEL,
                    # NOT per-model: fp32 leaves are the exact tree (SKYRL_ISOEXEC_PIK_LEAF_DTYPE,
                    # default fp32).
                    "leaf_dtype": "fp32",
                    "reduce_before_topk": True,  # ETP tree-reduce in combine_preprocess, then scatter
                    "tree": "balanced_fixed_fp32",  # bitwise for any ETP dividing G
                    "topk_sum_dtype": "fp32_then_one_bf16_round",
                },
                documentary=(
                    "ETP-invariant combine paired with the leaf-tree fc2 (moe.experts): a rank at ETP=C "
                    "owns G/C of G=8 fixed leaves, its local subtree composes with pik's cross-rank "
                    "tree into the SAME G-leaf tree for every C -> bitwise for trainer-ETP != engine-ETP "
                    "(the 35B 0.0105 residual). Both dispatchers carry it and are ENGINEERED bitwise-"
                    "equal: allgather (engine) sums the top-k in fp32 in combine_preprocess then rounds "
                    "to bf16 in token_combine; alltoall (EP>1 trainer) carries fp32 rows through the "
                    "alltoall and rounds AFTER the fp32 top-k sum -- rounding before the sum was a "
                    "measured 1-2 ULP/token bug. Proven offline pik_moe_fc2_{leaftree,"
                    "topk_leaftree}_test.py (maxabsdiff 0.0). The paired leaf-tree fc2 costs ~1.9-2.4x "
                    "native BY CONSTRUCTION (fp32 partial store + tree liveness = the invariance "
                    "contract), routing-dependent -- see moe.experts:fused. WARNS on ETP>1 alltoall "
                    "(reduce_scatter is not tree-invariant -- not a IsoExec target layout)."
                ),
            ),
            capabilities={"cuda_graph": True, "etp_invariant": True},
            hazards=["tie_boundaries", "e_mismatch"],
        )
    )
    reg.register_op(op)


# moe.weights -- fused [E, ...] expert-weight buffer. ENGINE sites only.
def _register_weights(reg) -> None:
    # No trainer site: the trainer's stacked-bmm weight materialization (torch.stack) is part of
    # moe.experts:batched_bmm, so "no trainer site" is correct ("no such site", not a default).
    op = OpSpec(name="moe.weights", sites=["engine_prefill", "engine_decode"])
    # `fused_buffer`: one contiguous [E, *param] buffer per (layer, role), each expert's .weight rebound
    # to a VIEW. "stacking" becomes handing back the buffer -- no per-forward copy. Bitwise trivial (same
    # bytes, one storage). See moe_fused_weights.py for the invariant and its failure modes.
    op.add_impl(
        ImplSpec(
            impl_id="fused_buffer",
            version=1,
            supported_archs=frozenset({"sm90"}),
            rounding=RoundingSchedule(
                machine_assertable={
                    "buffer_shape": "[E, *param_shape]",
                    "params_are_views": True,  # param.data = buf[i]; Parameter object preserved
                    "single_storage": True,  # exactly one copy of the bytes -> view cannot go stale
                    "reserved_pool": "large",  # per-expert tensors in SMALL pool; fuse strands segments
                },
                documentary=(
                    "ENGINE-ONLY fused expert-weight buffer -- removes the per-forward torch.stack "
                    "(CatArrayBatchedCopy, the top decode-time kernel). No trainer site: "
                    "the trainer's stack is inside moe.experts:batched_bmm. Bitwise trivial (identical "
                    "bytes). GRAPH-SAFETY: a CUDA-graph replay reads capture-time pointers, so the "
                    "buffer MUST be refreshed IN PLACE on every weight sync (eager per-sync refresh, "
                    "bump_sync_epoch); a stale replay reads wrong weights -> kill the run. See "
                    "state_invalidation."
                ),
            ),
            capabilities={"cuda_graph": True, "engine_only": True},
            state_invalidation=StateInvalidation(
                condition=(
                    "The fused [E,...] buffer's aliasing is invalidated by: (1) Parameter OBJECT "
                    "replacement (vllm_worker _set_on_module on meta/shape-mismatch), (2) .data rebind "
                    "without object replacement (distributed optimizer / checkpoint loaders), (3) "
                    "storage freed/moved (vLLM sleep+wake cumem, meta materialization -> data_ptr()==0), "
                    "(4) shape/dtype drift at the same address. On any violation the reader "
                    "(fused_expert_weights) returns None and the caller falls back to torch.stack -- it "
                    "does NOT auto-refuse (re-binding could sever a legitimate optimizer's aliasing). A "
                    "CUDA-graph replay additionally requires an in-place per-weight-sync refresh."
                ),
                # Declared callable only; the adapter invokes it -- this registry does not.
                hook=None,
            ),
            hazards=[],
        )
    )
    reg.register_op(op)


# moe.epilogue -- SwiGLU-in-GEMM epilogue (glu + probs folded into the fc1 GEMM). ENGINE sites only.
def _register_epilogue(reg) -> None:
    # Distinct wired op: folds the silu*up*probs epilogue INTO the fc1 GEMM epilogue so the [T,2f]
    # intermediate never reaches HBM. ENGINE-only (lives inside moe_fused_experts._fused_forward, which
    # only the engine installs). The non-fused elementwise epilogue is part of moe.experts.
    # ENV SKYRL_ISOEXEC_MOE_FUSED_EPILOGUE; shipped ENGINE-only.
    op = OpSpec(name="moe.epilogue", sites=["engine_prefill", "engine_decode"])
    op.add_impl(
        ImplSpec(
            impl_id="fused_swiglu",
            version=1,
            supported_archs=frozenset({"sm90"}),
            rounding=RoundingSchedule(
                machine_assertable={
                    "folded_into": "fc1_gemm_epilogue",
                    "retile_over_output_half_f": True,  # gate[:,c] and up[:,c] live in one program
                    "block_size_k_pinned": True,  # K-loop identical to native; only N re-tiled
                    "intermediate_materialized": False,  # [T,2f] never written to HBM (13S -> 1S)
                },
                documentary=(
                    "ENGINE-ONLY SwiGLU+probs epilogue fused into the fc1 GEMM. Pure elementwise -- no "
                    "reduction -- so NO invariance tax (unlike moe_fused_leaftree's fc2), parity-or-"
                    "better and bitwise-equal to the hand-written silu(gate)*up*probs chain: each "
                    "accumulator runs the identical K-loop the native kernel runs for those columns, "
                    "BLOCK_SIZE_N only splits output columns and never enters a reduction. The two "
                    "chunk views are non-contiguous (grid-8 unvectorized in the eager path)."
                ),
            ),
            capabilities={"cuda_graph": True, "engine_only": True},
            hazards=["non_contiguous"],
        )
    )
    reg.register_op(op)


# moe.blockmap -- the fused-expert-GEMM block map (sorted_token_ids/expert_ids/num_tokens_post_padded).
def _register_blockmap(reg) -> None:
    # Distinct wired op: replaces _block_map's ~12 tiny torch ops with ONE Triton launch, building the
    # metadata the fused expert GEMMs consume. ENGINE-only (feeds moe.experts:fused). Bitwise trivial --
    # integer tensors only, asserted elementwise-identical (a bug feeds wrong rows, not a last-bit
    # perturbation). Shape-static and sync-free, so CUDA-graph capture is unaffected.
    # ENV SKYRL_ISOEXEC_MOE_FUSED_BLOCKMAP; shipped ENGINE-only.
    op = OpSpec(name="moe.blockmap", sites=["engine_prefill", "engine_decode"])
    op.add_impl(
        ImplSpec(
            impl_id="fused",
            version=1,
            supported_archs=frozenset({"sm90"}),
            rounding=RoundingSchedule(
                machine_assertable={
                    "integer_only": True,  # computes no floating point at all
                    "outputs": ["sorted_token_ids", "expert_ids", "num_tokens_post_padded"],
                    "block_m_from_vllm": True,  # must match the kernel's BLOCK_SIZE_M or it reads wrong rows
                    "shape_static": True,  # max_blocks = ceil(T/BLOCK_M) + E, host-known
                    "launches": 1,  # 12 torch ops -> 1 Triton launch (per-program E-wide cumsum in regs)
                },
                documentary=(
                    "ENGINE-ONLY fused expert block map. Bitwise in the STRONG sense: integer tensors "
                    "identical (asserted elementwise) -- there is no fp arithmetic to perturb. "
                    "Shape-static and sync-free (the non-negotiable CUDA-graph constraint): grid sized "
                    "from a host-known bound, blocks past the real count early-return on the device "
                    "scalar num_tokens_post_padded (graph-replay inert padded lanes)."
                ),
            ),
            capabilities={"cuda_graph": True, "engine_only": True},
            # e_mismatch: expert-count layout in the block map. null_lanes: padded blocks past the real
            # count (graph-replay inertness) -- a metadata bug feeds the wrong rows, not a last bit.
            hazards=["e_mismatch", "null_lanes"],
        )
    )
    reg.register_op(op)
