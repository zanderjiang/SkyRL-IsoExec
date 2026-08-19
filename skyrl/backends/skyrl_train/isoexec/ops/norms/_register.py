"""Declarative registry entries for the norm family; no behavior lives here.

Two ops -- ``norms.rms`` (the RMSNorm the no-TE local spec installs in both runtimes, in a
zero-centred and a plain gamma form) and ``norms.gated_out`` (the GDN gated output norm) -- each
with an eager reference twin and a bitwise-equal fused engine kernel. The fused impls are
engine-only and installed per instance on the engine's model, never on the class the trainer builds.

For both fused kernels the reduction tile IS the bitwise contract: quack's ``threads_per_row``
ladder fixes the fp32 reduction tree, and a wrong tile costs a handful of elements in ten million
that no ``allclose`` shows. That, ``enable_fp_fusion=False``, the rsqrt-approx.f32 rstd, and the
down-cast chain are the machine_assertable pins.
"""

from __future__ import annotations

from ...core.registry import ImplSpec, OpSpec, RoundingSchedule


def register(reg) -> None:
    # norms.rms -- RMSNorm: rms_norm(x) rounded to bf16, then multiplied by gamma in bf16.
    reg.register_op(
        OpSpec(
            name="norms.rms",
            sites=["trainer_fwd", "trainer_score", "engine_prefill", "engine_decode"],
        )
        .add_impl(
            # The reference twin, installed and called in both runtimes. F.rms_norm reaches torch's
            # vendored quack CuTe-DSL kernel, whose launch config -- hence reduction tree -- is
            # chosen from N alone, so it is batch-invariant; the gamma multiply is elementwise.
            ImplSpec(
                impl_id="eager_zero_centered",
                version=1,
                supported_archs=frozenset({"sm90"}),
                rounding=RoundingSchedule(
                    machine_assertable={
                        # F.rms_norm reaches torch's vendored quack kernel, not vLLM's
                        # batch-invariant override list (aten::rms_norm is absent from it).
                        "rms_backend": "quack_cute_dsl",
                        "downcast_chain": ["fp32_reduce", "bf16_norm", "bf16_gamma_mul"],
                        "gamma_form": "1_plus_weight",  # zero-centred: gamma stored near 0
                    },
                    documentary=(
                        "y = rms_norm(x) * (1.0 + weight). rstd is rsqrt-MULTIPLY (quack fastmath "
                        "rsqrt.approx.f32), not 1/sqrt-divide. `1.0 + weight` is recomputed inside "
                        "forward every call -- must stay in forward trainer-side or weight leaves "
                        "the autograd graph. rms(x) rounds to bf16 BEFORE gamma is applied."
                    ),
                ),
            )
        )
        .add_impl(
            # Plain RMSNorm -- `y = rms_norm(x) * weight`, for models whose bridge does not set
            # `layernorm_zero_centered_gamma`, so `ZeroCenteredTorchRMSNorm` is never constructed.
            #
            # A separate impl rather than a pinned constant on `eager_zero_centered`, because the two
            # are different functions: `* (1.0 + w)` performs an fp add per element that `* w` does
            # not, so they differ in the last bits for the same stored gamma.
            #
            # It has no fused engine twin, which is a wiring fact rather than a claim:
            # `install_engine_fused_norms` rebinds forward on `ZeroCenteredTorchRMSNorm` instances
            # only, so on such a model it swaps none and both runtimes run the eager kernel.
            ImplSpec(
                impl_id="eager_torch_rms",
                version=1,
                supported_archs=frozenset({"sm90"}),
                rounding=RoundingSchedule(
                    machine_assertable={
                        "rms_backend": "quack_cute_dsl",
                        "downcast_chain": ["fp32_reduce", "bf16_norm", "bf16_gamma_mul"],
                        "gamma_form": "weight",  # not 1_plus_weight
                    },
                    documentary=(
                        "megatron WrappedTorchNorm -> torch.nn.RMSNorm -> F.rms_norm(x, (N,), w, eps). "
                        "Same vendored quack CuTe-DSL kernel as eager_zero_centered, whose launch "
                        "config (hence reduction tree) is chosen from N alone => batch-invariant; the "
                        "difference is only that gamma multiplies as stored instead of as `1 + w`. "
                        "ONE impl at all four sites on GLM-4.7-Flash, so no equivalence proof owed."
                    ),
                ),
            )
        )
        .add_impl(
            # Fused Triton kernel, bitwise-equal to the eager twin. Engine-only, per-instance.
            ImplSpec(
                impl_id="fused",
                version=1,
                supported_archs=frozenset({"sm90"}),
                rounding=RoundingSchedule(
                    machine_assertable={
                        # The reduction tile is the bitwise contract: quack's threads_per_row ladder
                        # fixes the fp32 reduction tree, chosen per width and never by row count.
                        "reduction_tile": "quack_threads_per_row_ladder",
                        "tile_depends_on_rows": False,
                        "max_n": 16384,  # above this quack goes clustered; the kernel refuses
                        "enable_fp_fusion": False,  # FFMA contraction gives addcmul, not eager
                        "rstd": "rsqrt_approx_f32",  # tl.rsqrt; not libdevice.rsqrt_rn nor 1/sqrt
                        "downcast_chain": ["fp32_reduce", "bf16_norm", "bf16_gamma_mul"],
                    },
                    documentary=(
                        "3 launches -> 1, bitwise-equal to eager_zero_centered. ENGINE-ONLY: "
                        "installed per-INSTANCE via install_engine_fused_norms (never on the class), "
                        "so the trainer keeps eager. `1.0 + weight` is formed IN REGISTERS from the "
                        "live parameter -- NO cached gamma, so nothing can go stale across a "
                        "weight sync or a CUDA-graph replay. rsqrt-multiply-vs-divide: tl.rsqrt "
                        "matches, libdevice.rsqrt_rn and 1/sqrt do not."
                    ),
                ),
                # `bitwise_equal_to` is load-bearing: the manifest resolves norms.rms asymmetrically
                # on a zero-centred model and CI re-derives that asymmetry from this field.
                capabilities={"cuda_graph": True, "engine_only": True, "bitwise_equal_to": "eager_zero_centered"},
                hazards=["subnormals"],
            )
        )
    )

    # norms.gated_out -- GDN gated output norm: the RMSNorm chain, then * SiLU(gate) in fp32.
    reg.register_op(
        OpSpec(
            name="norms.gated_out",
            sites=["trainer_fwd", "trainer_score", "engine_prefill", "engine_decode"],
        )
        .add_impl(
            # The eager reference, present in both runtimes.
            ImplSpec(
                impl_id="eager",
                version=1,
                supported_archs=frozenset({"sm90"}),
                rounding=RoundingSchedule(
                    machine_assertable={
                        "downcast_chain": [
                            "fp32_reduce",
                            "bf16_norm",
                            "bf16_gamma_mul",
                            "fp32_gate_mul",
                            "bf16_store",
                        ],
                        "silu_form": "x_div_1_plus_exp_neg_x",  # ATen fp32 SiLU, not x*sigmoid(x)
                    },
                    documentary=(
                        "RMSNorm(x)*(1+w) then multiply by SiLU(gate) computed in fp32, one final "
                        "round to bf16. SiLU is x/(1+exp(-x)) (ATen), the OPPOSITE of the conv "
                        "site's x*sigmoid(x). Covers both GDN chunk and recurrent modes."
                    ),
                ),
            )
        )
        .add_impl(
            # Fused Triton kernel, bitwise-equal to eager. Engine-only, installed from the GDN
            # inference forward, which only the engine binds.
            ImplSpec(
                impl_id="fused",
                version=1,
                supported_archs=frozenset({"sm90"}),
                rounding=RoundingSchedule(
                    machine_assertable={
                        "reduction_tile": "quack_threads_per_row_ladder",
                        "tile_depends_on_rows": False,
                        "max_n": 16384,
                        "enable_fp_fusion": False,
                        "rstd": "rsqrt_approx_f32",
                        # The RMSNorm chain's down-casts, plus one after the fp32 gate multiply.
                        "downcast_chain": [
                            "fp32_reduce",
                            "bf16_norm",
                            "bf16_gamma_mul",
                            "fp32_gate_mul",
                            "bf16_store",
                        ],
                        "silu_form": "x_div_1_plus_exp_neg_x",
                        "exp": "libdevice_exp",  # not tl.exp (ex2.approx)
                        "div": "triton_nonftz_div_rn",  # libdevice.div_rn flushes a subnormal quotient
                    },
                    documentary=(
                        "~7 launches -> 1, bitwise-equal to eager. ENGINE-ONLY, per-instance out of "
                        "_gdn_inference_forward -- never rebind GatedDeltaNet._apply_gated_norm "
                        "(runs in BOTH processes, severs the backward). The divide MUST be the "
                        "non-FTZ div.rn.f32: for gate below ~-88 the SiLU quotient goes fp32 "
                        "subnormal and libdevice.div_rn (linked __CUDA_FTZ) flushes it; ATen keeps "
                        "it. libdevice.exp is audited safe on subnormals. Serves chunk + recurrent."
                    ),
                ),
                # Equivalence claim declared; see the note on norms.rms:fused.
                capabilities={"cuda_graph": True, "engine_only": True, "bitwise_equal_to": "eager"},
                hazards=["subnormals"],
            )
        )
    )
