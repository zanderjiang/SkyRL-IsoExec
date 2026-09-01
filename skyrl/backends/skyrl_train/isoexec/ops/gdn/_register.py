"""Registry entries for the GatedDeltaNet (GDN) op family.

Declarative metadata only: no behavior, no kernel imports. Transcribes the rounding schedules
implemented in ``ops/gdn/*`` into the registry's checkable form.
"""

from __future__ import annotations

from ...core.registry import (
    ImplSpec,
    OneOf,
    OpSpec,
    RoundingSchedule,
    StateInvalidation,
)

_SM90 = frozenset({"sm90"})
_ALL_SITES = ["trainer_fwd", "trainer_score", "engine_prefill", "engine_decode"]


def register(reg) -> None:
    # gdn.core -- the delta-rule scan. Each shipped mode resolves all four sites to the same impl,
    # so no intra-mode equivalence proof is owed.
    reg.register_op(
        OpSpec(name="gdn.core", sites=list(_ALL_SITES))
        # vLLM ``fused_sigmoid_gating_delta_rule_update``: l2norm, GQA and gating are in-kernel.
        .add_impl(
            ImplSpec(
                impl_id="native_fused_sigmoid",
                version=1,
                supported_archs=_SM90,
                rounding=RoundingSchedule(
                    machine_assertable={
                        "boundary_dtypes": {"in": "bf16", "state": "fp32", "out": "bf16"},
                        "use_qk_l2norm_in_kernel": True,
                        "gating_in_kernel_fp32": True,  # g=-exp(fp32 A_log)*softplus(a+dt_bias)
                        "gqa_map": "i_h = i_hv // (HV // H)",  # in-kernel GQA
                        "scale": "K**-0.5",
                        "inplace_final_state": True,
                        # Manifest pin key (= SKYRL_ISOEXEC_GDN_KERNEL). Only these two modes reach the
                        # native core; kernel="chunk" selects the sibling impl, so it must not appear here.
                        "kernel": OneOf("recurrent", "cpr"),
                    },
                    documentary=(
                        "The native-composition core (both runtimes; all four sites). vLLM "
                        "fused_sigmoid_gating_delta_rule_update: in-kernel rsqrt-multiply l2norm, "
                        "GQA fan-out, and fp32 exp(A_log) sigmoid gating. Differs BITWISE from the "
                        "eager isoexec composition (in-kernel rsqrt-mult l2norm vs l2norm_fwd, "
                        "bf16- vs fp32-exp gating, bias-in-accumulator conv), so the flag must flip "
                        "BOTH runtimes in one run. Proved by gdn_native_kernel_parity_test: one "
                        "varlen call == T sequential decode calls bitwise incl the fp32 state "
                        "round-trip; a chunk split at ANY token boundary is exact -> chunked "
                        "prefill + align-mode APC are exact. No autotune to pin (fused single "
                        "kernel)."
                    ),
                ),
                capabilities={"cuda_graph": True, "apc": True, "chunked_prefill": True},
                subsumes=["gdn.l2norm", "gdn.gating"],
                # null_lanes: padded decode lanes fold to block 0 (kernel skips state_index<=0).
                # non_contiguous: transposed conv_state view. t_zero: varlen empty lanes.
                hazards=["null_lanes", "non_contiguous", "t_zero"],
            )
        )
        # RECURRENT scan: fused_recurrent_gated_delta_rule; l2norm + gating stay OUTSIDE, so it
        # subsumes neither. Grid (1,NV,N*HV) keeps every reduction inside one sequence.
        .add_impl(
            ImplSpec(
                impl_id="recurrent",
                version=1,
                supported_archs=_SM90,
                rounding=RoundingSchedule(
                    machine_assertable={
                        "boundary_dtypes": {"in": "bf16", "state": "fp32", "out": "bf16"},
                        "use_qk_l2norm_in_kernel": False,
                        "scale": "K**-0.5",  # scale=None -> K**-0.5
                        "inplace_final_state": True,
                        "autotune": False,  # invariant by construction, nothing to pin
                        "do_not_specialize": ["N", "T"],  # decode(T=1) == prefill(T=P) same compiled kernel
                    },
                    documentary=(
                        "Recurrent delta-rule scan (all four sites in SKYRL_ISOEXEC_GDN_KERNEL="
                        "recurrent). One program per (sequence, v-head, v-block) => batch "
                        "composition cannot change a row; state carried in fp32 REGISTERS through a "
                        "plain for-loop => prefix invariance by definition; NO autotune. Always "
                        "driven through the continuous-batching ssm_state_indices path (never the "
                        "token-offset path). Prefill->decode chains bitwise "
                        "iff the fp32 ssm state round-trips exactly through memory."
                    ),
                ),
                capabilities={"cuda_graph": True, "chunked_prefill": True},
                # null_lanes: kernel skips any lane with state index <= 0 (NULL_BLOCK_ID guard).
                # non_contiguous: packed varlen [1,T,...] inputs. t_zero: varlen.
                hazards=["null_lanes", "non_contiguous", "t_zero"],
            )
        )
        # Chunk-parallel: chunk_gated_delta_rule with autotune pinned to configs[0] by
        # gdn_batch_invariant.pin_fla_autotune_configs; l2norm stays outside.
        .add_impl(
            ImplSpec(
                impl_id="chunk",
                version=1,
                supported_archs=_SM90,
                rounding=RoundingSchedule(
                    machine_assertable={
                        "boundary_dtypes": {"in": "bf16", "state": "fp32", "out": "bf16"},
                        "use_qk_l2norm_in_kernel": False,
                        "scale": "K**-0.5",  # q scaled by k_dim**-0.5
                        # Load-bearing pin: FLA chunk kernels forced to configs[0] in every process,
                        # which also avoids the racy chunk_scaled_dot_kkt BK=64/num_warps=4/num_stages>=2.
                        "autotune_pin_index": 0,
                        "chunk_size": "FLA_CHUNK_SIZE",
                    },
                    documentary=(
                        "Chunked-parallel delta rule (all four sites in chunk mode; the trainer's "
                        "training kernel). Autotune pinned to configs[0] by an identical rule in "
                        "every process -> deterministic + cross-sequence + prefix invariant, exact "
                        "state chaining (verify_gdn_batch_invariance). Decode reproduces it bitwise "
                        "by pinning the recurrent state to the chunk grid and re-running this kernel "
                        "over the open chunk. No model declares a chunk composition, so a contract "
                        "build refuses it; the mode survives as the trainer-only ablation door "
                        "SKYRL_ISOEXEC_GDN_TRAINER_KERNEL=chunk."
                    ),
                ),
                capabilities={"cuda_graph": False, "padded_decode_for_graph": True},
                # non_contiguous: y slices need .contiguous() before l2norm_fwd.
                # null_lanes: padded-decode scratch row / NULL_BLOCK_ID=0.
                hazards=["non_contiguous", "null_lanes", "t_zero"],
            )
        )
        # CHUNK-SYNCED: boundary states from the chunk state pass (wy prep + fwd_h, autotune pinned),
        # within-chunk outputs from the recurrent scan starting at bf16(h_c). l2norm + gating outside.
        .add_impl(
            ImplSpec(
                impl_id="cpr",
                version=1,
                supported_archs=_SM90,
                rounding=RoundingSchedule(
                    machine_assertable={
                        "boundary_dtypes": {"in": "bf16", "state": "fp32", "out": "bf16"},
                        "use_qk_l2norm_in_kernel": False,
                        "scale": "K**-0.5",
                        "chunk_size": "FLA_CHUNK_SIZE",
                        "autotune_pin_index": 0,  # the state pass IS the autotuned chunk prep
                        # the two dtype pins the whole design rests on:
                        "boundary_snapshot_dtype": "bf16",  # h_c as stored/loaded by every site
                        "boundary_chain_dtype": "fp32",  # entry-state handoff between state passes
                    },
                    documentary=(
                        "One canonical function, three evaluations: fwd_h chains chunk-entry states "
                        "in fp32 and stores bf16 snapshots; every within-chunk scan (trainer "
                        "segment, prefill segment, decode token) starts from bf16(h_c) upcast to "
                        "fp32. Proved by gdn_cpr_h1_test: trainer forward == "
                        "decode sim with fp32-chained resyncs, bitwise on ragged lengths. Defined "
                        "ONLY on the isoexec eager/fused-prep composition -- the chunk state pass "
                        "does not reproduce the NATIVE fused core's states even on kernel-matched "
                        "prep (2.1-3.3e-04, gdn_cpr_native_h1_test) -- so it excludes "
                        "native kernels and native state (CprGDN raises on both). "
                        "cuda_graph False in v1: the decode resync host-syncs the crossing mask."
                    ),
                ),
                capabilities={"cuda_graph": False, "chunked_prefill": True},
                hazards=["null_lanes", "non_contiguous", "t_zero"],
            )
        )
    )

    # gdn.l2norm -- standalone q/k L2 norm, subsumed by native_fused_sigmoid; registered so the
    # subsumes edge closes.
    reg.register_op(
        OpSpec(name="gdn.l2norm", sites=list(_ALL_SITES)).add_impl(
            ImplSpec(
                impl_id="l2norm_fwd",
                version=1,
                supported_archs=_SM90,
                rounding=RoundingSchedule(
                    machine_assertable={
                        "eps": 1e-6,  # L2NORM_EPS
                        "formula": "x * rsqrt(sum(x^2, -1) + eps)",  # rsqrt-multiply
                        "row_local": True,
                    },
                    documentary=(
                        "Delegates to vLLM's l2norm_fwd -- the SAME Triton kernel imported by both "
                        "runtimes, row-local so batch/prefix invariant. "
                        "Forward keeps the kernel (bitwise both sides); the autograd backward "
                        "differentiates x*rsqrt(sum(x^2)+eps) because l2norm_fwd carries no autograd "
                        "history and would silently deliver ZERO grad to q,k. "
                        "Absorbed IN-KERNEL by gdn.core:native_fused_sigmoid (rsqrt-multiply on the "
                        "fp32 upcast) -- when that core is selected the adapter passthroughs this op."
                    ),
                ),
                capabilities={"cuda_graph": True},
                # non_contiguous: q/k are non-contiguous slices of the post-conv tensor.
                hazards=["non_contiguous"],
            )
        )
    )

    # gdn.gating -- g = -exp(A_log)*softplus(a+dt_bias) fp32; beta = sigmoid(b). Subsumed by the
    # native fused core. Elementwise -> invariant by construction.
    reg.register_op(
        OpSpec(name="gdn.gating", sites=list(_ALL_SITES)).add_impl(
            ImplSpec(
                impl_id="eager",
                version=1,
                supported_archs=_SM90,
                rounding=RoundingSchedule(
                    machine_assertable={
                        "g_dtype": "fp32",
                        "beta_dtype": "input",  # sigmoid(b) in b's dtype
                        # A_log.exp() in the PARAMETER dtype (bf16), NOT upcast first, matching megatron:
                        # exp(bf16(x)).float() != exp(float(x)).
                        "A_log_exp_in_param_dtype": True,
                        "formula": "g = -A_log.exp() * softplus(a.float()+dt_bias.float()); beta = b.sigmoid()",
                    },
                    documentary=(
                        "Mirrors megatron GatedDeltaNet._compute_g_and_beta exactly. "
                        "All-elementwise over (token, head) => no reduction "
                        "order, batch/prefix invariant. Absorbed IN-KERNEL (fp32) by "
                        "gdn.core:native_fused_sigmoid; passthrough'd when that core is selected."
                    ),
                ),
                capabilities={"cuda_graph": True},
                hazards=["subnormals"],  # softplus/exp underflow path
            )
        )
    )

    # gdn.conv -- width-4 causal depthwise conv. DISTINCT kernels prefill vs decode; owes an
    # equivalence proof. Three impls (op has all four sites; each impl's served sites in documentary).
    reg.register_op(
        OpSpec(name="gdn.conv", sites=list(_ALL_SITES))
        # NATIVE prefill kernel (serves trainer_fwd, trainer_score, engine_prefill).
        .add_impl(
            ImplSpec(
                impl_id="causal_conv1d_fn",
                version=1,
                supported_archs=_SM90,
                rounding=RoundingSchedule(
                    machine_assertable={
                        "width": 4,
                        "activation": "silu",
                        "bias_in_accumulator": True,  # vs eager bias-after-taps
                        "varlen": True,  # one launch for all N prompts via query_start_loc
                    },
                    documentary=(
                        "vLLM causal_conv1d_fn -- native prefill conv (serves trainer_fwd, "
                        "trainer_score and engine_prefill; the trainer reaches it via "
                        "gdn_native_conv). EQUIVALENCE PROOF OWED: this (prefill) and "
                        "causal_conv1d_update (decode) do NOT agree bitwise natively; the native "
                        "composition's exactness rests on gdn_native_kernel_parity_test (chunk split "
                        "at any token boundary exact). Differs from the elementwise impl in rounding "
                        "only (bias preloaded into the fp32 accumulator vs added after the taps)."
                    ),
                ),
                capabilities={"chunked_prefill": True, "apc": True},
                hazards=["non_contiguous", "null_lanes"],
            )
        )
        # NATIVE decode kernel (serves engine_decode).
        .add_impl(
            ImplSpec(
                impl_id="causal_conv1d_update",
                version=1,
                supported_archs=_SM90,
                rounding=RoundingSchedule(
                    machine_assertable={
                        "width": 4,
                        "activation": "silu",
                        "in_place_window_slide": True,  # slides the conv window at the block rows
                    },
                    documentary=(
                        "vLLM causal_conv1d_update -- native decode conv (engine_decode only). "
                        "Host-free + shape-static so it captures "
                        "into a CUDA graph. Does NOT agree bitwise with causal_conv1d_fn: see the "
                        "equivalence-proof note on causal_conv1d_fn."
                    ),
                ),
                # `equivalence_proof`, NOT `bitwise_equal_to`: the fn/update pair does not agree
                # bitwise natively, so a byte-equality claim here would be false.
                capabilities={
                    "cuda_graph": True,
                    "equivalence_proof": (
                        "gdn_native_kernel_parity_test: chunk split at ANY token boundary is exact. "
                        "The fn/update pair does NOT agree bitwise natively; the native "
                        "composition's exactness rests on "
                        "split-exactness, not on kernel equality."
                    ),
                },
                # null_lanes: padded replay lanes fold to row 0 (the conv slide touches null garbage).
                hazards=["null_lanes", "non_contiguous"],
            )
        )
        # ELEMENTWISE shifted-sum conv (all four sites): one conv on both sides, so the native
        # prefill/decode kernel mismatch cannot arise.
        .add_impl(
            ImplSpec(
                impl_id="elementwise_shifted_sum",
                version=1,
                supported_archs=_SM90,
                rounding=RoundingSchedule(
                    machine_assertable={
                        "width": 4,
                        "activation": "silu",
                        "fp32_accumulate_round_once": True,
                        "bias_after_taps": True,  # bias added after the tap sum
                    },
                    documentary=(
                        "gdn_causal_conv / gdn_causal_conv_batched: a sum of W "
                        "shifted, scaled copies + bias + SiLU, fp32-accumulated and rounded ONCE. "
                        "Every op is elementwise over the token axis, so y[t] is a pure function of "
                        "x[t-3..t] -- batch/prefix invariance by construction, no equivalence proof "
                        "needed. Used identically at prefill and decode by the private-pool "
                        "recurrent and CPR paths, which is "
                        "how those paths dodge the native prefill/decode conv mismatch."
                    ),
                ),
                capabilities={"cuda_graph": True, "chunked_prefill": True},
                hazards=["non_contiguous", "null_lanes", "t_zero"],
            )
        )
    )

    # gdn.state -- recurrent-state ownership. ENGINE sites only. Declares the rebind discipline.
    reg.register_op(
        OpSpec(name="gdn.state", sites=["engine_prefill", "engine_decode"])
        # SHIPPED (GDN_NATIVE_STATE=1): state lives in vLLM's OWN mamba kv_cache blocks, indexed
        # directly by the engine block id.
        .add_impl(
            ImplSpec(
                impl_id="native_kv_cache",
                version=1,
                supported_archs=_SM90,
                rounding=RoundingSchedule(
                    machine_assertable={
                        # MANDATORY: mamba_ssm_cache_dtype=auto resolves to bf16, which rounds the state
                        # on every read/write and breaks the fp32 prefill->decode round-trip.
                        "ssm_cache_dtype": "float32",
                        "conv_state_orientation": "(num_blocks, D, W-1)",  # SD/DS transpose
                        "block_id_is_row": True,  # identity index; block-id space == num_gpu_blocks
                        "null_block_id": 0,  # out-of-range/NULL lanes fold to block 0
                    },
                    documentary=(
                        "State IS vLLM's mamba kv_cache; block id indexes it directly (measured "
                        "rows=2576, running-max id=1960 -- real ids in range). Padded CUDA-graph "
                        "replay lanes carry sentinel ids (NULL_BLOCK_ID negative, or non-live during "
                        "capture) and are folded onto block 0 (recurrent kernel skips index<=0). The "
                        "fp32 cache is set by the engine build (mamba_ssm_cache_dtype=float32)."
                    ),
                ),
                capabilities={"cuda_graph": True, "apc": True, "chunked_prefill": True},
                # THE generalized rebind contract: vLLM binds a throwaway profiling cache first, then
                # the real one; a stale binding fires an out-of-range device-side assert (no traceback).
                state_invalidation=StateInvalidation(
                    condition=(
                        "vLLM rebinds kv_cache (throwaway profiling cache of "
                        "max_cudagraph_capture_size blocks, then the real one); rebind when "
                        "(data_ptr,shape) of the bound ssm tensor changes"
                    ),
                    hook=None,
                ),
                # null_lanes: graph-padded lanes fold to block 0. profiling_shapes: the throwaway
                # profiling cache bound before the real one. non_contiguous: transposed conv_state view.
                hazards=["null_lanes", "profiling_shapes", "non_contiguous"],
            )
        )
        # Fallback (GDN_NATIVE_STATE=0): a private max_num_seqs-sized pool behind a slot->row map.
        # Row 0 is the null row; the pool is sized once and never reallocated.
        .add_impl(
            ImplSpec(
                impl_id="private_pool",
                version=1,
                supported_archs=_SM90,
                rounding=RoundingSchedule(
                    machine_assertable={
                        "ssm_state_dtype": "float32",  # NOT bf16
                        "capacity": "max_num_seqs",
                        "null_row": 0,  # row 0 never handed out; unknown slots resolve to it
                        "slot_map_size": 65536,  # _SLOT_MAP_SIZE default
                    },
                    documentary=(
                        "Private pool because for a HYBRID model vLLM pads the mamba page up to the "
                        "attention page, so the block-id space is sized by the shared KV pool, not "
                        "the state tensor (measured at 35B: prefill handed slot 516 for a 512-row "
                        "tensor -> device-side assert, worker abort). A device slot->row map "
                        "translates; row 0 is the NULL_BLOCK_ID row (padded replay lanes, unknown "
                        "slots). LRU eviction read off a device clock."
                    ),
                ),
                capabilities={"cuda_graph": True, "chunked_prefill": True},
                # Sized ONCE and never reallocated: a captured graph holds the tensor's address, so a
                # realloc would replay against a stale pointer. Overflow raises rather than grows.
                state_invalidation=StateInvalidation(
                    condition=(
                        "slot->row map + state pool are allocated once and never reallocated (a "
                        "captured CUDA graph holds the tensor address); a slot id exceeding the map "
                        "raises rather than reallocating -- raise SKYRL_ISOEXEC_GDN_SLOT_MAP_SIZE"
                    ),
                    hook=None,
                ),
                # null_lanes: unknown/NULL/out-of-range slots clamp onto row 0.
                hazards=["null_lanes", "non_contiguous"],
            )
        )
        # cpr (GDN_KERNEL=cpr, which requires GDN_NATIVE_STATE=0). Unlike `private_pool` it also holds
        # the fp32 entry state and the open-chunk buffers the boundary pass re-reads every C tokens.
        .add_impl(
            ImplSpec(
                impl_id="cpr_pool",
                version=1,
                supported_archs=_SM90,
                rounding=RoundingSchedule(
                    machine_assertable={
                        "ssm_state_dtype": "float32",
                        # The chunk-pass accumulator at the last crossed boundary; must never round
                        # through bf16, which is what makes the bf16 snapshots exact.
                        "entry_state_dtype": "float32",
                        # At a boundary the running state becomes bf16(final) upcast to fp32 while the
                        # fp32 final becomes the next entry state; trainer segments load that snapshot.
                        "boundary_snapshot_dtype": "bf16",
                        "boundary_chain_dtype": "fp32",
                        # Manifest-pinned so the two runtimes cannot pick different C: the trainer's
                        # segmented scan and this boundary pass are the same function only at the same C.
                        "chunk_size": 64,  # FLA_CHUNK_SIZE
                        "capacity": "max_num_seqs",
                        "null_row": 0,  # rows = capacity + 1; row 0 never handed out
                        # 65536 by default, 2^20 under GDN_CPR_MIN_PAGES=1: that flag multiplies vLLM's
                        # block-id space and every live slot id must fit the map.
                        "slot_map_size": OneOf(65536, 1048576),
                        # CprGDN refuses a handed-in vLLM state tensor, and the engine refuses
                        # to start in this mode with GDN_NATIVE_STATE=1.
                        "native_state": False,
                    },
                    documentary=(
                        "cpr's OWN state pool: running ssm state (fp32) + conv window "
                        "(bf16) as in the recurrent pool, PLUS a per-row fp32 entry state, an "
                        "absolute-position counter, and four open-chunk buffers (k/v/g/beta, "
                        "[rows, C, ...]) holding the raw values the boundary pass re-reads: RAW "
                        "compressed-head k and raw a/b, with native_matched_prep recomputed "
                        "at resync, so the boundary pass consumes exactly what the trainer's does. "
                        "Every large tensor is allocated in "
                        "the current CUDA memory-pool context, so the whole footprint releases "
                        "at engine sleep and the trainer gets its recompute headroom back; the tiny "
                        "position counter is a plain alloc because its host mirrors must survive "
                        "sleep. Sized ONCE and never reallocated -- a captured CUDA graph holds the "
                        "addresses. Under GDN_CPR_APC=1 prefill additionally materialises the fp32 "
                        "entry + conv window at the boundaries it closes and stashes them for the "
                        "boundary-state store (CprBoundaryStore); that store is a CACHE, a miss is "
                        "never a correctness event, so it moves no bits and adds no key here."
                    ),
                ),
                # cuda_graph: decode is capturable because the boundary resync is hoisted to the host
                # driver that runs before the step's forward (lazy_resync).
                capabilities={"cuda_graph": True, "chunked_prefill": True, "apc": "cpr_boundary_store"},
                state_invalidation=StateInvalidation(
                    condition=(
                        "pool + slot->row map + open-chunk buffers are allocated once at build and "
                        "never reallocated (captured graphs hold their addresses); a slot id "
                        "exceeding the map raises rather than growing. The boundary-state store is "
                        "invalidated by a WEIGHT SYNC -- a cached prefix state computed by the "
                        "previous policy is silent and plausible -- so the engine patch clears it "
                        "wherever vLLM resets its own prefix cache (cpr_apc_reset)"
                    ),
                    hook=None,
                ),
                # null_lanes: unknown/NULL slots clamp onto row 0, as in the parent pool.
                # t_zero: varlen prefill segments; non_contiguous: the buffered [rows, C, ...] views.
                hazards=["null_lanes", "non_contiguous", "t_zero"],
            )
        )
    )
