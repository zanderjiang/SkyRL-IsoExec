"""Declarative registry entries for the RoPE family; no behavior lives here.

``rope.rope`` is the rotate-half rotary position embedding on query/key, with two impls: ``eager``
(megatron's stock ``_apply_rotary_pos_emb_bshd``, the reference twin both runtimes take on the
local-spec path) and ``fused`` (the hoisted-cos/sin kernel, engine-only and bitwise-equal to it).

Two pins carry the bitwise contract: cos/sin are evaluated in fp32 and only then rounded to the
input dtype, and the multiply-add is a bf16 three-round chain rather than one fp32 round -- staying
in fp32 across the add is more accurate and a different function.
"""

from __future__ import annotations

from ...core.registry import ImplSpec, OpSpec, RoundingSchedule


def register(reg) -> None:
    reg.register_op(
        OpSpec(
            name="rope.rope",
            sites=["trainer_fwd", "trainer_score", "engine_prefill", "engine_decode"],
        )
        .add_impl(
            # Megatron stock rope -- the reference twin, run in both runtimes on the local-spec path.
            ImplSpec(
                impl_id="eager",
                version=1,
                supported_archs=frozenset({"sm90"}),
                rounding=RoundingSchedule(
                    machine_assertable={
                        # cos/sin trig in fp32 from fp32 freqs, then rounded to the input dtype.
                        "cos_sin_compute_dtype": "fp32",
                        "mul_add_opmath": "bf16_three_rounds",  # not one fp32 round -- a different fn
                    },
                    documentary=(
                        "stock _apply_rotary_pos_emb_bshd. rotate-half order: lane d takes lane "
                        "d+R/2 NEGATED for d<R/2 and lane d-R/2 as-is above (cat((-x2, x1)) over the "
                        "first R lanes; upper D-R lanes copied). Three-round bf16 multiply-add: "
                        "bf16(t*cos_) + bf16(rot*sin_), rounded again. NOTE the fp32 RoPE patch "
                        "(megatron_patches) is a DIFFERENT rounding chain (one fp32 round) used only "
                        "off the local-spec path -- not this impl."
                    ),
                ),
            )
        )
        .add_impl(
            # Fused kernel plus cos/sin hoist, bitwise-equal to stock. Engine-only.
            ImplSpec(
                impl_id="fused",
                version=1,
                supported_archs=frozenset({"sm90"}),
                rounding=RoundingSchedule(
                    machine_assertable={
                        "cos_sin_compute_dtype": "fp32",  # hoisted (cos(freqs)*mscale) in fp32 -> dtype
                        "mul_add_opmath": "bf16_three_rounds",
                        "enable_fp_fusion": False,  # load-bearing: fusion contracts a product into the add
                        "rotate_half_negation": "mul_neg_one",  # x * -1.0, never -x (signed zero)
                        # Pure elementwise plus a fixed-distance gather: no reduction, so the tile is
                        # not a bitwise contract. Pinned anyway (no autotune).
                        "tile_is_bitwise_contract": False,
                    },
                    documentary=(
                        "6 launches -> 1 + a cos/sin hoist (computed ONCE per forward, cached ON the "
                        "freqs tensor so it dies with it -- no invalidation boundary). Bitwise-equal "
                        "to eager (stock). ENGINE-ONLY via a mark on freqs from _PositionIndexedRoPE; "
                        "refuses to install over the fp32 RoPE patch (different chain). Same "
                        "rotate-half order as eager; negation is x*-1.0."
                    ),
                ),
                # `bitwise_equal_to` is load-bearing, not decoration: the manifest resolves rope.rope
                # asymmetrically (eager on the trainer, this on the engine) and the policy check
                # re-derives that asymmetry from this field, so an undeclared claim fails CI.
                capabilities={"cuda_graph": True, "engine_only": True, "bitwise_equal_to": "eager"},
                hazards=["signed_zero", "subnormals", "non_contiguous"],
            )
        )
    )
