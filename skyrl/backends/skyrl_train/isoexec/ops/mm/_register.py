"""Registry entries for the global matmul op.

``mm`` is the process-global batch-invariant matmul override. ``mm_tiles`` is the same op's
shape-keyed output-tiling policy (BLOCK_M/BLOCK_N only, K-reduction order untouched), so it is
metadata on the one impl rather than a second op. All four sites share this impl.
"""

from __future__ import annotations

from ...core.registry import ImplSpec, OpSpec, RoundingSchedule


def register(reg) -> None:
    reg.register_op(
        OpSpec(
            name="mm",
            sites=["trainer_fwd", "trainer_score", "engine_prefill", "engine_decode"],
        )
        .add_impl(
            ImplSpec(
                impl_id="triton_batch_invariant",
                version=1,
                supported_archs=frozenset({"sm90"}),
                rounding=RoundingSchedule(
                    machine_assertable={
                        # SM90 cuBLAS is not M-invariant, so the override must stay process-global.
                        "aten_override": ["aten::mm", "aten::addmm"],
                        "split_k": False,
                        "block_size_k_pinned": True,
                        "output_tiles_shape_keyed": True,  # BM/BN only; K-order untouched
                    },
                    documentary=(
                        "batch-invariant persistent matmul; fixed BLOCK_SIZE_K, no split-K, so "
                        "element (m,n) reduces in one fixed K order regardless of M (batch). "
                        "mm_tiles retiles BM/BN per shape without touching the K reduction."
                    ),
                ),
                capabilities={"cuda_graph": True},
                hazards=["signed_zero", "non_contiguous"],
            )
        )
        .add_impl(
            # Opt-in (default OFF): pinned non-split-K cuBLASLt for a handful of bf16 dense shapes;
            # everything else falls through to triton_batch_invariant. Signature-moving by contract
            # -- not declared bit-equal to the Triton kernel -- so enabling it on only one side
            # diverges the model-manifest hash and the weight-sync handshake refuses to run.
            ImplSpec(
                impl_id="cublaslt_pinned",
                version=1,
                supported_archs=frozenset({"sm90"}),
                rounding=RoundingSchedule(
                    machine_assertable={
                        "split_k": False,  # hard-asserted at pin time (throws on violation)
                        "reduction_scheme": "NONE",
                        "m_buckets": {"decode": "probe@512", "trainer": "probe@8192"},
                        "cross_bucket_bit_identical": True,  # pin@512 == pin@8192 (proven)
                        "workspace": "32MiB_at_init",  # never allocated in the hot path (graph-safe)
                        "shapes_bf16_only": [[2048, 1032], [512, 2048], [2048, 1024], [2048, 128], [64, 2048]],
                    },
                    documentary=(
                        "Holds M-invariance and cross-bucket bit-identity (pin@512 == pin@8192), "
                        "run-to-run determinism, graph capture == eager, and the same result under "
                        "every fallthrough routing and either mm_tiles composition order. An "
                        "init-time self-check per process (M-invariance / cross-bucket / "
                        "determinism / split-K probe) permanently disables the provider on any "
                        "failure, so cuBLAS-version drift degrades to no-perf-win, never to a "
                        "broken gate. Installed via enable_moe_deterministic_ops (the one site "
                        "both runtimes call)."
                    ),
                ),
                capabilities={"cuda_graph": True, "engine_only": False},
                hazards=["signed_zero", "non_contiguous"],
            )
        )
    )
