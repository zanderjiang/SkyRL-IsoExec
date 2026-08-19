"""Declarative registry entries for the attention family.

``attention.varlen`` (causal varlen FlashAttention) carries two impls: ``varlen_custom`` (torch FA3
varlen, grad-capable, used by the trainer and as the reference) and ``vllm_flash_ns1`` (engine-only
fast path, bitwise-equal to it). The ``num_splits=1`` pin is the core attention pin: one
KV-reduction split makes each query row independent of the launch geometry, hence bitwise
decode==prefill and trainer==engine at all lengths. FA3 activation is equally load-bearing.
"""

from __future__ import annotations

from ...core.registry import ImplSpec, OpSpec, RoundingSchedule


def register(reg) -> None:
    reg.register_op(
        OpSpec(
            name="attention.varlen",
            sites=["trainer_fwd", "trainer_score", "engine_prefill", "engine_decode"],
        )
        .add_impl(
            ImplSpec(
                impl_id="varlen_custom",
                version=1,
                supported_archs=frozenset({"sm90"}),
                rounding=RoundingSchedule(
                    machine_assertable={
                        # The core pin: one KV-reduction split => query-length invariant.
                        "num_splits": 1,
                        # causal = unlimited left, zero right.
                        "window_size": [-1, 0],
                        "flash_impl": "FA3",  # activate_flash_attention_impl("FA3"); dispatch must match
                    },
                    documentary=(
                        "torch.nn.attention.varlen varlen_attn / varlen_attn_out, num_splits=1, "
                        "window=(-1,0), FA3. ONE kernel at every site (engine CUSTOM backend + "
                        "trainer TorchVarlenCoreAttn), so no equivalence proof owed. FA3 activation "
                        "is load-bearing: the trainer builds no vLLM engine, so torch's flash impl "
                        "stays unset and the SAME varlen_attn call dispatches to a non-FA3 kernel "
                        "(1-ULP core_attention diff => the ~0.014 rollout-vs-train residual) unless "
                        "activate_flash_attention_impl('FA3') is called trainer-side too."
                    ),
                ),
                capabilities={"cuda_graph": True},
                hazards=["non_contiguous"],
            )
        )
        .add_impl(
            # Engine fast path: vLLM's FA3 varlen kernel at the same num_splits=1 pin, bitwise-equal
            # to varlen_custom. Engine-only because the trainer needs autograd (torch varlen).
            ImplSpec(
                impl_id="vllm_flash_ns1",
                version=1,
                supported_archs=frozenset({"sm90"}),
                rounding=RoundingSchedule(
                    machine_assertable={
                        "num_splits": 1,
                        "fa_version": 3,
                        "window_size": [-1, 0],
                        "causal": True,
                    },
                    documentary=(
                        "vllm.vllm_flash_attn.flash_attn_varlen_func(num_splits=1, fa_version=3): "
                        "same FA3 num_splits=1 reduction as varlen_custom, and faster. Engine "
                        "sites only (varlen_backend fast path, gate="
                        "SKYRL_ISOEXEC_VARLEN_VLLM_FLASH default 1; OFF falls back to the torch "
                        "kernel -- pure perf knob, bitwise-interchangeable). Trainer keeps "
                        "varlen_custom (grad-capable). Fixing num_splits>1 instead is not legal: "
                        "split placement follows launch geometry, so decode!=prefill at d~1e-3 "
                        "once splits engage."
                    ),
                ),
                capabilities={"cuda_graph": True, "engine_only": True, "bitwise_equal_to": "varlen_custom"},
                hazards=["non_contiguous"],
            )
        )
    )
    reg.register_op(
        OpSpec(
            name="attention.qwen35_context_layout",
            sites=["trainer_fwd", "trainer_score"],
        ).add_impl(
            ImplSpec(
                impl_id="qwen35_context_layout_sm90a",
                version=1,
                supported_archs=frozenset({"sm90"}),
                rounding=RoundingSchedule(
                    machine_assertable={
                        "input_shape": [4, 32, 256],
                        "output_shape": [32, 1, 1024],
                        "operation": "byte_exact_context_permutation",
                        "block": 1024,
                        "num_warps": 8,
                    },
                    documentary=(
                        "Manual Qwen-3.5 context-layout op at Megatron "
                        "DotProductAttention._format_context. CUDA-13 sm90a cubin, prepared "
                        "driver launch, exact source/shape/stride guards, first-call bitwise "
                        "check, and eager fallback for grad-enabled or foreign shapes."
                    ),
                ),
                capabilities={"cuda_graph": True, "manual_priority": True},
                # Shape/source specialization lives in the machine contract and installer guards;
                # the hazard vocabulary only describes operand probes.
                hazards=["non_contiguous"],
            )
        )
    )
