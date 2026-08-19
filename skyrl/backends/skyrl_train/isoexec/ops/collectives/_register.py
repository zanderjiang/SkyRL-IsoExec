"""Declarative registry entries for the ``collectives`` op family.

Declares the rounding schedules the composition manifest resolves against for three ops:
``collectives.tree_all_reduce`` (pik's fixed balanced binary tree, with NCCL moving bytes only),
``collectives.row_parallel`` (Megatron ``RowParallelLinear.forward`` rebound onto the pik leaf-tree GEMM), and
``collectives.nccl_pin`` (the NCCL channel/algo pin, whose five exact identities may resolve differently on the
trainer and the engine -- which is why the manifest is (op, site)-keyed). No kernel behavior lives here.
"""

from __future__ import annotations

from ...core.registry import PER_MODEL, ImplSpec, OneOf, OpSpec, RoundingSchedule

_ALL_SITES = ["trainer_fwd", "trainer_score", "engine_prefill", "engine_decode"]


def register(reg) -> None:
    # pik one-shot/two-shot IPC tree reduction: NCCL (or symm-mem P2P) moves bytes only, the arithmetic is
    # pik's Triton tree kernel realizing plan.combine_order. Both runtimes build the same ReductionPlan, so
    # this is one impl across all four sites.
    reg.register_op(
        OpSpec(
            name="collectives.tree_all_reduce",
            sites=list(_ALL_SITES),
        ).add_impl(
            ImplSpec(
                impl_id="pik_tree",
                version=1,
                supported_archs=frozenset({"sm90"}),
                rounding=RoundingSchedule(
                    machine_assertable={
                        # Contract constants: trainer and engine must agree, and these names must match the
                        # manifest pin keys or validate_pins cannot check them. G >= max(trainer TP, engine TP)
                        # is a per-deployment fact, so the model pins the value and this declares only that it
                        # is pinned.
                        "leaves": PER_MODEL,
                        # Both values are TP-invariant -- the leaf grid is fixed by G, so rounding at the
                        # leaf boundary commutes with the TP degree -- but they are different functions, so
                        # the selected value is a hashed manifest pin and a one-sided flip is refused at
                        # weight sync. The plan is checked against the pin at install (pik_tp_invariant).
                        "leaf_dtype": OneOf("fp32", "bf16"),
                        # combine_order(): balanced binary tree, left operand carries lower leaf
                        # indices; internal (non-leaf) tree nodes accumulate in fp32.
                        "combine_tree": "balanced_binary_left_lower",
                        "combine_accum_dtype": "fp32",
                        # NCCL is transport only (all_gather / all_to_all_single) -- a byte
                        # permutation with NO reduction order, hence channel-count-insensitive.
                        "nccl_role": "byte_movement_only",
                    },
                    documentary=(
                        "reduce-before-topk / FP32-sum-then-ONE-bf16-round is load-bearing. "
                        "K is cut into G fixed contiguous leaves, "
                        "each leaf summed by a non-split-K GEMM, leaves combined by a FIXED "
                        "balanced binary tree in fp32; a rank at TP=C owns a contiguous SUBTREE, so "
                        "the expression tree is identical for every C that divides G -> bit-identical "
                        "across a trainer/engine TP mismatch. NCCL never performs the reduction "
                        "(only all_gather/all_to_all byte movement), so its algo/channel choice "
                        "cannot perturb the result. one-shot vs two-shot (and P2P symm-mem vs NCCL "
                        "transport) both realize the same tree -- pure perf knobs, verified "
                        "bitwise-equal."
                    ),
                ),
                capabilities={"cuda_graph": True, "tp_invariant": True},
                # t_zero: 0-token / empty GEMM falls back to native (cuBLASLt rejects M=0).
                # non_contiguous: tree_all_reduce calls partial.contiguous() before staging.
                # null_lanes: decode graph-replay padded lanes must reduce inertly.
                hazards=["t_zero", "non_contiguous", "null_lanes"],
            )
        )
    )

    # pik TP-invariant RowParallelLinear.forward rebind: F.linear + reduce_from_tensor_model_parallel_region
    # becomes the pik leaf-tree GEMM + tree_all_reduce, so the K-reduction is the same expression tree at every
    # TP. One class is patched and both runtimes execute it, so one impl across all sites.
    reg.register_op(
        OpSpec(
            name="collectives.row_parallel",
            sites=list(_ALL_SITES),
        ).add_impl(
            ImplSpec(
                impl_id="pik_tree",
                version=1,
                supported_archs=frozenset({"sm90"}),
                rounding=RoundingSchedule(
                    machine_assertable={
                        # Only row-parallel layers shard K (o_proj/linear_proj, down_proj/linear_fc2);
                        # column-parallel shards N and is already TP-invariant (left native).
                        "patched_layers": ["o_proj", "down_proj", "linear_fc2"],
                        # Same contract constants and pin-key names as tree_all_reduce above: the two
                        # declarations describe the same plan object.
                        "leaves": PER_MODEL,
                        "leaf_dtype": OneOf("fp32", "bf16"),
                        # forward reduces over the pik tree; backward reduces over N/M, NEVER K, so it
                        # carries no invariance requirement and runs stock cuBLAS.
                        "backward_reduces_k": False,
                        "reduction": "collectives.tree_all_reduce",
                    },
                    documentary=(
                        "RowParallelLinear.forward rebound onto pik's leaf-tree GEMM + fixed-tree "
                        "all-reduce so row-parallel K-reduction is bitwise-identical across ANY "
                        "trainer/engine TP that divides G. Fixed-function fallbacks to native go to "
                        "explicit_expert_comm, is_expert@ETP=1 (whole non-split-K "
                        "GEMM, already identical), 0-token GEMMs, and any (K, TP) the plan cannot "
                        "express (warned once). sequence_parallel (TRAINER, "
                        "SKYRL_ISOEXEC_TRAINER_SP=1): same leaf-tree GEMM combined by "
                        "tree_reduce_scatter -- the two-shot tree minus its trailing all-gather, "
                        "each rank keeping its sequence slice; bitwise == AR-then-slice, so the "
                        "expression tree (and this impl's identity) is unchanged. With the flag "
                        "off, SP falls back to native (warned once, breaks invariance) -- but "
                        "megatron_worker forces SP off in that case anyway. Same "
                        "ReductionPlan/tree as tree_all_reduce."
                    ),
                ),
                capabilities={"cuda_graph": True, "tp_invariant": True},
                # t_zero: 0-token expert GEMM -> native fallback (input_.numel()==0 branch).
                # non_contiguous: dy.contiguous() in backward; x/weight reshaped.
                # e_mismatch: MoE expert fc2 under ETP is routed here, on the expert-count path.
                hazards=["t_zero", "non_contiguous", "e_mismatch"],
            )
        )
    )

    # The NCCL determinism pin, registered as five exact ALGO/MIN/MAX tuples. The same op resolves to
    # different impls per side: pinned on the trainer sites (where it is function-bearing) and unpinned on the
    # engine sites (where it is bitwise-neutral). The manifest, not this module, selects which side gets which.
    reg.register_op(
        OpSpec(
            name="collectives.nccl_pin",
            sites=list(_ALL_SITES),
        )
        .add_impl(
            ImplSpec(
                impl_id="cap8",
                version=1,
                supported_archs=frozenset({"sm90"}),
                rounding=RoundingSchedule(
                    machine_assertable={
                        "NCCL_ALGO": None,
                        "NCCL_MIN_NCHANNELS": None,
                        "NCCL_MAX_NCHANNELS": "8",
                    },
                    documentary=(
                        "NCCL algorithm/floor removed with an exact eight-channel ceiling. This is "
                        "not the uncapped implementation: it has a different persistent-memory "
                        "charge and launch geometry. Selecting it on trainer sites requires profile "
                        "evidence for that composition; on engine sites it stays an entry-scoped "
                        "DEPLOYMENT choice under the engine channel-count neutrality argument."
                    ),
                ),
                capabilities={"deterministic_allreduce": False, "max_channels": 8},
                hazards=["null_lanes"],
            )
        )
        .add_impl(
            ImplSpec(
                impl_id="engine_cap8",
                version=1,
                supported_archs=frozenset({"sm90"}),
                rounding=RoundingSchedule(
                    machine_assertable={
                        "NCCL_ALGO": "allreduce:tree",
                        "NCCL_MIN_NCHANNELS": None,
                        "NCCL_MAX_NCHANNELS": "8",
                    },
                    documentary=(
                        "vLLM retains allreduce:tree while IsoExec removes the one-channel floor "
                        "and keeps an eight-channel ceiling. This exact engine tuple is a distinct "
                        "identity from the trainer's cap8 tuple."
                    ),
                ),
                capabilities={"deterministic_allreduce": True, "max_channels": 8},
                hazards=["null_lanes"],
            )
        )
        .add_impl(
            ImplSpec(
                impl_id="engine_cap16",
                version=1,
                supported_archs=frozenset({"sm90"}),
                rounding=RoundingSchedule(
                    machine_assertable={
                        "NCCL_ALGO": "allreduce:tree",
                        "NCCL_MIN_NCHANNELS": None,
                        "NCCL_MAX_NCHANNELS": "16",
                    },
                    documentary=(
                        "vLLM retains allreduce:tree while IsoExec removes the one-channel floor "
                        "and uses a sixteen-channel ceiling. This is an exact deployment identity, "
                        "distinct from both the trainer's cap8 tuple and the engine_cap8 tuple; "
                        "the manifest and runtime census must agree on all three constants."
                    ),
                ),
                capabilities={"deterministic_allreduce": True, "max_channels": 16},
                hazards=["null_lanes"],
            )
        )
        .add_impl(
            ImplSpec(
                impl_id="pinned",
                version=1,
                supported_archs=frozenset({"sm90"}),
                rounding=RoundingSchedule(
                    machine_assertable={
                        # Trainer: env_vars in train/utils/utils.py, gated on SKYRL_ISOEXEC_NCCL_PIN.
                        # Engine: ISOEXEC_VLLM_ENV in runtimes/vllm/vllm_patches.py, set pre-init in-process.
                        "NCCL_MIN_NCHANNELS": "1",
                        "NCCL_MAX_NCHANNELS": "1",
                        "NCCL_ALGO": "allreduce:tree",
                    },
                    documentary=(
                        "The NCCL single-channel + tree-algo determinism pin. THIS IS THE SIDE-ASYMMETRIC "
                        "OP: the composition in use is pinned on TRAINER sites, unpinned on ENGINE sites. On "
                        "the TRAINER the pin is FUNCTION: unpinning moves the IsoExec gate at the 8th "
                        "significant digit (6.590784096260904e-07 -> 6.590783527826716e-07) because a "
                        "genuine trainer-side dist.all_reduce (VocabParallelEmbedding / column-parallel "
                        "backward) is pin-sensitive -- so the pin stays ON on the trainer and this impl is "
                        "what the manifest MUST select for trainer_fwd/trainer_score. gate=SKYRL_ISOEXEC_NCCL_PIN "
                        "(default 1)."
                    ),
                ),
                capabilities={"deterministic_allreduce": True},
                hazards=["null_lanes"],
            )
        )
        .add_impl(
            ImplSpec(
                impl_id="unpinned",
                version=1,
                supported_archs=frozenset({"sm90"}),
                rounding=RoundingSchedule(
                    machine_assertable={
                        # channel pins cleared / vLLM's re-pin wrapped away (neutralize_vllm_nccl_channel_pin).
                        "NCCL_ALGO": None,
                        "NCCL_MIN_NCHANNELS": None,
                        "NCCL_MAX_NCHANNELS": None,
                    },
                    documentary=(
                        "NCCL channel pin removed. On the ENGINE this is bitwise-NEUTRAL (DEPLOYMENT): 11 of "
                        "the 12 NCCL primitives in a decode step are AllGathers (byte permutations -- channel "
                        "count cannot change a bit) and the 12th is an order-free bf16 VocabParallelEmbedding "
                        "AllReduce (one non-zero addend per element); pik's own reductions never touch NCCL. "
                        "The engine's removal is entry-scoped DEPLOYMENT-proven neutral. An uncapped trainer "
                        "is NOT admitted by current profile evidence: the observations available either "
                        "predate the current PIK composition or were not a clean one-flag A/B of this flag. "
                        "The deployment/function label remains per-(op, site), not op-global."
                    ),
                ),
                capabilities={"deterministic_allreduce": False},
                hazards=["null_lanes"],
            )
        )
    )
