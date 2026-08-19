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

from ...core.registry import ImplSpec, OpSpec, RoundingSchedule


def register(reg) -> None:
    # logprobs.log_softmax -- manual fp32 amax/exp/sum/log/gather, one function at engine + trainer.
    reg.register_op(
        OpSpec(
            name="logprobs.log_softmax",
            sites=["trainer_score", "engine_prefill", "engine_decode"],
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
                        "order; do not re-order the reduction (see aten_reference_fused_exp for the "
                        "legal traffic optimization). Same input logits (bitwise forward) + same ops "
                        "=> bitwise rollout==train logprobs."
                    ),
                ),
            )
        )
        .add_impl(
            # Engine fast path: the same math at roughly half the HBM traffic. The sub+exp is fused
            # into one Triton pass via libdevice.exp (bitwise-equal to aten's .exp(), re-verified by
            # a first-call self-check that permanently falls back on mismatch), and the sampled
            # elements are gathered BEFORE the final subtract so the full [N,V] is never
            # materialized. The vocab amax/sum/log stay in aten verbatim: aten's fp32 sum tree over
            # the vocab is a specific balanced structure that Triton cannot reproduce, so keeping the
            # tree is what bounds how far this can fuse.
            ImplSpec(
                impl_id="aten_reference_fused_exp",
                version=1,
                supported_archs=frozenset({"sm90"}),
                rounding=RoundingSchedule(
                    machine_assertable={
                        "compute_dtype": "fp32",
                        "reduction": ["amax", "fused_sub_exp", "sum", "log", "gather", "sub", "sub"],
                        "exp": "libdevice_exp",  # == aten .exp(); self-checked at first call
                        "vocab_sum": "aten_verbatim",  # the trainer's exact tree, never re-tiled
                        "gather_first": True,  # (logits.gather - amax) - lse, same per-element rounds
                        "min_rows_fastpath": 16,  # below this the launch overhead dominates
                    },
                    documentary=(
                        "ENGINE-ONLY execution twin of aten_reference (vllm_patches "
                        "_fast_compute_token_logprobs), bitwise-equal by construction + first-call "
                        "self-check. Falls back to aten_reference for N<=16, non-contiguous rows, "
                        "no-triton stacks, or a failed self-check. Trainer_score keeps "
                        "aten_reference (the trainer never runs this code path)."
                    ),
                ),
                capabilities={"engine_only": True, "bitwise_equal_to": "aten_reference"},
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
