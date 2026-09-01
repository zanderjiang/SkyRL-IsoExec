"""Declarative registry entries for the logprobs family.

The last op on the IsoExec path: the sampled-token log-probability that generation returns and the
trainer scores against.

``logprobs.log_softmax`` is the manual fp32 log-softmax and gather. vLLM's own fused Triton kernel
inlines log(softmax(.)) and never calls aten, so it bypasses the batch-invariant
``aten::_log_softmax`` and diverges from the trainer on a few tokens; the engine is patched to run
the trainer's exact formulation, ``(x - amax(x)) - log(sum(exp(x - amax(x))))`` in fp32, then gather.

``logprobs.lm_head_slice`` applies the output layer to the sampled rows only. vLLM samples one token
per sequence, so running lm_head and the full fp32 vocab gather on every token is wasted prefill.
Engine-only, and a pure row selection: it changes which rows are computed, not the logprob value.
"""

from __future__ import annotations

from ...core.registry import PER_MODEL, ImplSpec, OpSpec, RoundingSchedule


def register(reg) -> None:
    # logprobs.log_softmax -- manual fp32 amax/exp/sum/log/gather, one function at engine + trainer.
    # ``rowinv_leaftree`` is the SELECTED impl at all four sites and the only one any composition
    # names. The two aten-order impls below stay REGISTERED because they still execute: rowinv
    # declines structurally (unsupported layout, wrong arch, missing kernel) and the caller falls
    # back to them, so deregistering would leave the registry unable to describe code that runs.
    # Registration is not selection -- only selected entries reach the contract and the identity
    # hashes -- so an unselected impl costs nothing but keeps the fallback nameable.
    reg.register_op(
        OpSpec(
            name="logprobs.log_softmax",
            sites=["trainer_fwd", "trainer_score", "engine_prefill", "engine_decode"],
        )
        .add_impl(
            ImplSpec(
                impl_id="aten_reference",
                version=1,
                supported_archs=frozenset({"sm90"}),
                rounding=RoundingSchedule(
                    machine_assertable={
                        "compute_dtype": "fp32",
                        # This reduction order is the bitwise contract, matching the trainer's own
                        # distributed log-softmax and the batch-invariant aten op it dispatches to.
                        "reduction": ["amax", "sub", "exp", "sum", "log", "sub", "gather"],
                    },
                    documentary=(
                        "manual log_softmax = (x - amax(x)) - log(sum(exp(x - amax(x)))) in float32, "
                        "then gather -- NOT torch.log_softmax (a different kernel, not bitwise-equal "
                        "to the trainer's manual formulation). Must reproduce aten's exact reduction "
                        "order; do not re-order the reduction. Same input logits (bitwise forward) + "
                        "same ops => bitwise rollout==train logprobs."
                    ),
                ),
            )
        )
        .add_impl(
            # THE DEFAULT: the row-count- and TP-invariant leaf-tree denominator, one function at
            # ALL FOUR sites, composed unconditionally. Not an execution twin of aten_reference --
            # it REPLACES the incumbent on both runtimes at once, so it carries no
            # bitwise_equal_to claim; running one side on the incumbent is not a legal asymmetry
            # but a broken composition, which is why no flag can produce it any more.
            ImplSpec(
                impl_id="rowinv_leaftree",
                version=1,
                supported_archs=frozenset({"sm90"}),
                rounding=RoundingSchedule(
                    machine_assertable={
                        "compute_dtype": "fp32",
                        # G = SKYRL_ISOEXEC_PIK_LEAVES; 8 on every shipped recipe, but a
                        # per-deployment fact (>= max TP either side runs), so the MODEL pins the
                        # value and the impl declares only that it IS pinned. Same key name as
                        # collectives:pik_tree / moe.combine:pik_leaf_tree.
                        "leaves": PER_MODEL,
                        "block": 4096,  # the fixed per-leaf tile of the Kahan exp sum (rowinv.py BLOCK)
                        "accum": "kahan_fp32",
                        "reduction": [
                            "amax",
                            "all_reduce_max",  # exact: MAX has no rounding, so it is world-free
                            "sub",
                            "exp",
                            "leaf_kahan_sum",  # per-leaf, fixed BLOCK, vocab-index order
                            "leaf_tree_combine",  # pik combine_order(G), fp32 internal nodes
                            "log",
                            "gather",
                            "sub",  # (x_target - m) - log(S)
                            "sub",
                        ],
                        "leaf_boundaries": "V/G fixed",  # cut by G alone, never by rows or world
                        "combine": "pik_combine_order",
                    },
                    documentary=(
                        "logprob = (x_target - m) - log(S) with m = all_reduce(MAX) row max and S "
                        "assembled from G FIXED contiguous vocab leaves (Kahan fp32 exp sums, fixed "
                        "BLOCK) combined ONLY by pik's combine_order(G) tree. The leaf boundaries "
                        "and combine order are functions of G alone -- a rank at TP=C owns G/C "
                        "whole leaves and its local subtree composes into the SAME G-leaf tree at "
                        "every C dividing G, and the engine's already-gathered full row (world=1, "
                        "group=None) is simply C=1 of that same tree. That is what aten's schedule "
                        "is not: its fp32 vocab sum reduces in a shape-dependent order, so the "
                        "trainer's [B,S,V] reduction and the engine's sampled-rows reduction are "
                        "different functions (measured 72.2% non-bitwise tokens on real logits), "
                        "and its per-shard sum + all_reduce is a rank-boundary schedule that moves "
                        "with TP. Declines (caller keeps the incumbent) rather than approximating; "
                        "derivation pinned by ops/logprobs/tests/test_rowinv_leaftree_cpu.py."
                    ),
                ),
                capabilities={"tp_invariant": True, "row_count_invariant": True},
                hazards=["subnormals"],
            )
        )
    )

    # logprobs.lm_head_slice -- apply lm_head to the sampled rows only. Engine-only.
    reg.register_op(
        OpSpec(
            name="logprobs.lm_head_slice",
            sites=["engine_prefill", "engine_decode"],
        ).add_impl(
            ImplSpec(
                impl_id="sampled_rows",
                version=1,
                supported_archs=frozenset({"sm90"}),
                rounding=RoundingSchedule(
                    machine_assertable={
                        # Pure row selection: the same lm_head matmul and fp32 vocab gather, applied
                        # only to the sampled rows. Does not change the logprob value.
                        "rows": "sampled_only",
                        "gather_output": True,  # output layer must still gather the full vocab row
                    },
                    documentary=(
                        "SKYRL_ISOEXEC_SPLIT_LM_HEAD: flip post_process off after build so "
                        "compute_logits applies lm_head to the rows vLLM actually samples (last token "
                        "per sequence) instead of every token, removing nearly all of the prefill "
                        "logprob work. Bitwise-neutral -- selects rows, does not alter the logprob "
                        "math."
                    ),
                ),
                capabilities={"split_lm_head": True},
            )
        )
    )
