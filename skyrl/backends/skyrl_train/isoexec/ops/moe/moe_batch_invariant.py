"""Install the shared, deterministic IsoExec MoE recipe on Megatron and vLLM.

The package requires per-token results to be independent of unrelated batch rows and of runtime grad
mode. This module pins supported provider settings, builds the local no-TE layer spec, replaces
Megatron's atomic combine with a stable ordered combine, forces router top-k ordering, and installs
the qualified optional kernels in a fixed ownership order.

Expert routing bias is supported only when the native weight-sync buffer registry carries
``mlp.router.expert_bias`` in FP32. Input jitter and token dropping remain disallowed because they
make routing depend on runtime randomness or batch capacity.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

# How many distinct (num_tokens, topk) shapes to structurally validate before trusting the
# fixed-order combine's index layout. Each validation costs one device->host sync.
_VALIDATE_FIRST_N_CALLS = 8

_orig_unpermute = None
_orig_topk_routing = None
_deterministic_enabled = False
_validated_calls = 0


def moe_deterministic_enabled() -> bool:
    """True unless explicitly disabled. Only consulted on the MoE local-spec IsoExec path."""
    return os.environ.get("SKYRL_ISOEXEC_MOE_DETERMINISTIC", "1") == "1"


def provider_is_moe(provider) -> bool:
    return bool(getattr(provider, "num_moe_experts", None))


def _set_if_present(provider, name, value, changes):
    if hasattr(provider, name):
        old = getattr(provider, name)
        if old != value:
            changes.append(f"{name}: {old!r} -> {value!r}")
        setattr(provider, name, value)


def force_isoexec_moe_config(provider, *, side: str) -> None:
    """Pin ``provider`` onto the batch-invariant MoE recipe. Must run on BOTH trainer and engine.

    ``side`` is only used for the log line ("TRAINER" / "ENGINE").
    """
    if not provider_is_moe(provider):
        return

    if getattr(provider, "moe_router_enable_expert_bias", False):
        # Expert bias is a buffer, so its FP32 native-sync registration is a hard precondition.
        # Rounding it to bf16 can change top-k membership.
        from ...sync.native_weight_sync import SYNCED_BUFFER_SUFFIXES, is_synced_buffer

        if not is_synced_buffer("mlp.router.expert_bias"):
            raise ValueError(
                "[isoexec] moe_router_enable_expert_bias=True but 'mlp.router.expert_bias' is not in "
                f"native_weight_sync.SYNCED_BUFFER_SUFFIXES ({SYNCED_BUFFER_SUFFIXES}); the engine's "
                "routing bias would diverge from the trainer's after the first step."
            )
        changes_bias = "expert_bias ON (buffer rides the sync in native fp32)"
    else:
        changes_bias = None
    if getattr(provider, "moe_input_jitter_eps", None) is not None:
        raise ValueError("[isoexec] moe_input_jitter_eps must be None (it randomizes routing).")

    changes: list[str] = []
    if changes_bias:
        changes.append(changes_bias)
    # OLMoE's provider sets persist_layer_norm=True (a fused-kernel choice, not model math); the
    # local spec's torch norm asserts it off. Dense providers leave it False, so only MoE hits it.
    _set_if_present(provider, "persist_layer_norm", False, changes)
    # SequentialMLP: each expert is a plain MLP whose F.linear the batch-invariant aten override
    # makes independent of how many tokens routed to it. Grouped GEMM is batch-variant.
    _set_if_present(provider, "moe_grouped_gemm", False, changes)
    # Fused bias+SwiGLU must stay off. The batched expert path computes the activation itself and so
    # cannot reproduce BiasSwiGLUFunction's bits; with the fusion on it falls back to Megatron's stock
    # SequentialMLP.forward, whose `tokens_per_expert.tolist()` is a D2H sync that makes engine init
    # fail under CUDA-graph capture. Both sides run this function, so trainer and engine move together.
    _set_if_present(provider, "bias_activation_fusion", False, changes)
    # Token dispatcher. At TP=EP=1 the allgather dispatcher's collectives are no-ops. At EP>1 it
    # requires every EP rank to carry the same token count, which the scoring/training forward does not
    # (each EP rank gets a different microbatch), and it deadlocks at the first MoE layer. The alltoall
    # dispatcher exchanges only each expert's tokens and handles variable per-rank counts; its combine
    # order is made EP-invariant by enable_moe_deterministic_ops.
    _ep = getattr(provider, "expert_model_parallel_size", 1) or 1
    if os.environ.get("SKYRL_ISOEXEC_PIK") == "1" and _ep > 1:
        _set_if_present(provider, "moe_token_dispatcher_type", "alltoall", changes)
    else:
        _set_if_present(provider, "moe_token_dispatcher_type", "allgather", changes)
    _set_if_present(provider, "moe_router_dtype", "fp32", changes)
    # every MoE fusion is a TE kernel (absent on this stack) and/or batch-variant.
    _set_if_present(provider, "moe_permute_fusion", False, changes)
    _set_if_present(provider, "moe_permute_fusion_into_hybridep", False, changes)
    _set_if_present(provider, "moe_router_fusion", False, changes)
    _set_if_present(provider, "moe_enable_deepep", False, changes)
    # token dropping makes an expert's output depend on the *capacity*, i.e. on the batch size.
    _set_if_present(provider, "moe_expert_capacity_factor", None, changes)
    _set_if_present(provider, "moe_pad_expert_input_to_capacity", False, changes)
    # routing replay + forced load balancing rewrite the router's decisions.
    _set_if_present(provider, "moe_enable_routing_replay", False, changes)
    _set_if_present(provider, "moe_router_force_load_balancing", False, changes)
    _set_if_present(provider, "moe_router_force_biased", None, changes)
    # Shared-expert overlap changes ownership of the shared linears' TP communication, bypassing
    # the PIK RowParallelLinear reduction. Keep it disabled unless that composed collective gets
    # its own exact implementation.
    _set_if_present(provider, "moe_shared_expert_overlap", False, changes)
    # EP / ETP pins. Without pik these MUST be EP=1, ETP==TP: EP>1 turns the top-k expert combine into
    # a cross-rank reduce-scatter whose grouping differs from a flat ascending sum, and a mismatched ETP
    # shards the expert K-reduction differently. With SKYRL_ISOEXEC_PIK=1 the config's EP/ETP is allowed,
    # because the EP-invariant fixed-order combine makes EP>1 match EP=1 and pik's leaf tree makes
    # ETP-mismatched expert fc2 K-reductions match.
    if os.environ.get("SKYRL_ISOEXEC_PIK") == "1":
        changes.append(
            f"[pik] KEEP EP={getattr(provider, 'expert_model_parallel_size', 1)} "
            f"ETP={getattr(provider, 'expert_tensor_parallel_size', None)} "
            f"(TP={getattr(provider, 'tensor_model_parallel_size', 1)}) -- EP/ETP mismatch allowed"
        )
    else:
        _set_if_present(provider, "expert_model_parallel_size", 1, changes)
        _tp = getattr(provider, "tensor_model_parallel_size", 1) or 1
        _set_if_present(provider, "expert_tensor_parallel_size", _tp, changes)

    print(
        f"[ISOEXEC-{side}] MoE IsoExec recipe pinned "
        f"(experts={provider.num_moe_experts} topk={getattr(provider, 'moe_router_topk', '?')} "
        f"pre_softmax={getattr(provider, 'moe_router_pre_softmax', '?')}): "
        + ("; ".join(changes) if changes else "already conformant"),
        flush=True,
    )


def _carry_custom_self_attention(block_spec, attn_module) -> None:
    """Carry a provider's custom ``self_attention.module`` across a generically-built block spec.

    Some providers replace ``self_attention.module`` in their own spec (OLMoE applies q/k RMSNorm over
    ``num_heads * head_dim`` rather than per-head); a generic build would silently install megatron's
    stock ``SelfAttention`` and be a different model. Only slots currently holding the stock
    ``SelfAttention`` are replaced -- on a hybrid block the GDN layers hold ``GatedDeltaNet`` there.
    """
    if attn_module is None:
        return
    from megatron.core.transformer.attention import SelfAttention

    n = 0
    for layer in getattr(block_spec, "layer_specs", []) or []:
        sa = getattr(getattr(layer, "submodules", None), "self_attention", None)
        if sa is not None and getattr(sa, "module", None) is SelfAttention:
            sa.module = attn_module
            n += 1
    if n:
        print(f"[ISOEXEC-SPEC] carried custom SelfAttention {attn_module.__name__} onto {n} layer(s)", flush=True)


_SPEC_BANNERED: set = set()


def _spec_banner(branch: str, config) -> None:
    """Print, once per (branch, shape), which local-spec branch actually built the layer spec.

    ``te`` is reported because TransformerEngine being importable silently changes what the branches
    below build: ``transformer_block.LayerNormImpl`` becomes ``TENorm`` and ``TESpecProvider`` stops
    declining.
    """
    import importlib.util

    try:
        te = importlib.util.find_spec("transformer_engine") is not None
    except (ImportError, ValueError):
        te = False
    key = (
        branch,
        getattr(config, "num_layers", None),
        getattr(config, "num_moe_experts", None),
        bool(getattr(config, "multi_latent_attention", False)),
    )
    if key in _SPEC_BANNERED:
        return
    _SPEC_BANNERED.add(key)
    print(
        f"[ISOEXEC-SPEC] local layer spec branch={branch} layers={key[1]} moe_experts={key[2]} "
        f"mla={key[3]} transformer_engine_importable={te}",
        flush=True,
    )


def make_isoexec_local_layer_spec(provider):
    """Return a ``config -> ModuleSpec`` callable for the IsoExec local spec.

    Dense providers delegate to megatron-bridge's ``local_layer_spec`` unchanged (the dense IsoExec
    path must not shift). MoE providers get ``get_gpt_layer_local_spec(num_experts=...,
    moe_grouped_gemm=False)`` -> MoELayer(TopKRouter, SequentialMLP) built from local modules.

    Some providers replace ``self_attention.module`` in their own spec (OLMoE applies q/k RMSNorm
    over ``num_heads * head_dim`` rather than per-head). The local spec would silently build the
    generic ``SelfAttention`` with per-head norms -- a different model. So carry the original
    spec's ``self_attention.module`` across.
    """
    orig_spec = getattr(provider, "transformer_layer_spec", None)

    def _isoexec_local_layer_spec(config):
        from megatron.bridge.models.gpt_provider import local_layer_spec
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

        # Opt-in, default off: one generic call covering every shape megatron's config can express
        # (dense / GQA / MLA / MoE / per-layer dense-or-sparse / GDN hybrid) through megatron's own
        # `BackendSpecProvider` seam. Subsumes the three branches below and supports `moe_layer_freq`
        # on the hybrid path and PP>1, which they do not.
        from ...runtimes.megatron import spec_provider as _ix_spec

        # Layering: these build a Megatron layer spec, which is host-framework wiring rather than op
        # math, so they live under runtimes/megatron. The import is kept LAZY (call-time only) so this
        # op module still imports with no runtime dependency.
        from ...runtimes.megatron.gdn_hybrid_spec import (
            is_hybrid_gdn,
            make_isoexec_hybrid_local_spec,
        )

        if _ix_spec.enabled():
            spec = _ix_spec.build_decoder_block_spec(config)
            _carry_custom_self_attention(spec, _original_self_attention_module(orig_spec, config))
            _spec_banner("isoexec_spec_provider", config)
            return spec

        # Hybrid GDN models interleave GatedDeltaNet and softmax layers; `local_layer_spec` would
        # build dense attention for all of them, loading none of the checkpoint's GDN weights.
        if is_hybrid_gdn(config):
            _spec_banner("gdn_hybrid_local", config)
            return make_isoexec_hybrid_local_spec(config)

        if not provider_is_moe(config):
            _spec_banner("bridge_local_layer_spec (dense)", config)
            return local_layer_spec(config)

        spec = get_gpt_layer_local_spec(
            num_experts=config.num_moe_experts,
            moe_grouped_gemm=False,
            qk_layernorm=config.qk_layernorm,
            normalization=config.normalization,
        )
        attn_module = _original_self_attention_module(orig_spec, config)
        if attn_module is not None:
            spec.submodules.self_attention.module = attn_module
            print(f"[ISOEXEC-SPEC] MoE local spec keeps custom SelfAttention {attn_module.__name__}", flush=True)
        _spec_banner("gpt_layer_local_spec (moe)", config)
        return spec

    _isoexec_local_layer_spec.__name__ = "isoexec_local_layer_spec"
    return _isoexec_local_layer_spec


# Providers whose own layer spec swaps in a custom SelfAttention. Keyed by the spec function's
# __name__ so we never have to *call* the original spec -- these specs are built on
# `default_layer_spec`, which resolves to the TransformerEngine spec and would fail here.
_CUSTOM_SELF_ATTENTION_BY_SPEC: dict[str, tuple[str, str]] = {}


def _original_self_attention_module(orig_spec, config):
    """The custom ``self_attention.module`` the model's own spec would have used, or None."""
    import importlib

    entry = _CUSTOM_SELF_ATTENTION_BY_SPEC.get(getattr(orig_spec, "__name__", ""))
    if entry is None:
        return None
    module_path, class_name = entry
    return getattr(importlib.import_module(module_path), class_name)


def _fixed_order_combine(permuted_tokens, sorted_indices, restore_shape):
    """Sum each token's k expert rows in ascending-expert order. No atomics, no cross-token reduce.

    ``permute`` lays rows out expert-major, so for a fixed token its k rows appear at ascending row
    positions in ascending expert order. A *stable* argsort of ``sorted_indices`` therefore groups
    each token's rows together, ascending expert within the group. Gathering column j across all
    tokens and adding the k columns in order j=0..k-1 gives every token the same summation order
    regardless of how many other tokens are in the batch -- which is the bitwise invariant.

    Returns ``None`` when the layout is not the plain "every token routes to exactly k experts"
    case (e.g. token dropping), so the caller can fall back.
    """
    global _validated_calls

    num_tokens = int(restore_shape[0])
    n = int(sorted_indices.numel())
    if num_tokens == 0 or n == 0 or n % num_tokens != 0:
        return None
    k = n // num_tokens

    # The fused permute producer can emit this exact inverse while it already gathers routed rows,
    # removing every post-FC2 metadata launch. Malformed published metadata raises rather than
    # silently falling back to the sort and hiding an inert arm.
    from .moe_fused_permute import get_preemitted_combine_rows

    rows = get_preemitted_combine_rows(sorted_indices, num_tokens)
    if rows is None:
        # Exact counting-sort replacement, with the stable argsort as fallback.
        from .moe_combine_rows_kernel import stable_argsort_order

        rows = stable_argsort_order(sorted_indices, num_tokens).view(num_tokens, k)
    if _validated_calls < _VALIDATE_FIRST_N_CALLS:
        _validated_calls += 1
        expected = torch.arange(num_tokens, device=sorted_indices.device).repeat_interleave(k)
        if not torch.equal(sorted_indices[rows.view(-1)].to(expected.dtype), expected):
            raise RuntimeError(
                "[isoexec] MoE combine: tokens do not each route to exactly topk experts "
                f"(num_tokens={num_tokens}, permuted_rows={n}). Token dropping / capacity padding "
                "must be disabled for bitwise IsoExec."
            )

    out = permuted_tokens.index_select(0, rows[:, 0].contiguous())
    for j in range(1, k):
        out = out + permuted_tokens.index_select(0, rows[:, j].contiguous())
    return out


_fold_refused = set()


def _fold_refused_once(name: str) -> None:
    """The combine binding does not accept the round-fold request. Print once per binding name."""
    if name in _fold_refused:
        return
    _fold_refused.add(name)
    print(
        f"[ISOEXEC-MOE-COMBINE-FOLD] REFUSED: unpermute binding {name!r} does not accept "
        "isoexec_out_dtype; the round fold is INERT on this install. Running the unchanged path.",
        flush=True,
    )


def _deterministic_unpermute(
    permuted_tokens,
    sorted_indices,
    restore_shape,
    probs=None,
    routing_map=None,
    fused=False,
    drop_and_pad=False,
    isoexec_out_dtype=None,
    **kwargs,
):
    """Drop-in ``moe_utils.unpermute`` whose combine is deterministic and batch-invariant.

    ``isoexec_out_dtype`` is the caller's trailing cast pushed down one level; on this eager path it
    changes only where the cast is written. It is IsoExec-private and never forwarded to megatron.
    """
    if fused or drop_and_pad:
        return _orig_unpermute(
            permuted_tokens,
            sorted_indices,
            restore_shape,
            probs=probs,
            routing_map=routing_map,
            fused=fused,
            drop_and_pad=drop_and_pad,
            **kwargs,
        )

    # `input_dtype` is used only for the trailing cast, so redirecting it moves nothing else.
    input_dtype = permuted_tokens.dtype if isoexec_out_dtype is None else isoexec_out_dtype
    if probs is not None:
        assert routing_map is not None, "Mask must be provided to permute the probs."
        permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

    out = _fixed_order_combine(permuted_tokens, sorted_indices, restore_shape)
    if out is None:
        return _orig_unpermute(
            permuted_tokens,
            sorted_indices,
            restore_shape,
            probs=None,
            routing_map=routing_map,
            fused=False,
            drop_and_pad=False,
            **kwargs,
        ).to(dtype=input_dtype)
    return out.to(dtype=input_dtype)


# Published contract marker: this binding accepts the IsoExec-private `isoexec_out_dtype` kwarg.
# The pik alltoall `combine_postprocess` checks for it before sending the round-fold request, so an
# install-order change can only make the fold INERT (and say so), never raise.
_deterministic_unpermute._isoexec_accepts_out_dtype = True


def _make_sorted_topk_routing(orig):
    """Force ``sorted=True`` in the router's ``torch.topk``, in both grad and no-grad forwards.

    megatron-core calls ``torch.topk(scores, k, dim=1, sorted=torch.is_grad_enabled())`` from a
    closure inside ``topk_routing_with_score_function``, so there is no argument to thread through.
    Swapping ``torch.topk`` for the duration of the routing call is the smallest intervention that
    does not fork upstream code (the forward is single-threaded).
    """

    def _sorted_topk_routing(*args, **kwargs):
        real_topk = torch.topk

        def forced(*a, **kw):
            kw["sorted"] = True
            return real_topk(*a, **kw)

        torch.topk = forced
        try:
            return orig(*args, **kwargs)
        finally:
            torch.topk = real_topk

    return _sorted_topk_routing


_matmul_invariance_lib = None


def _install_moe_matmul_invariance() -> None:
    """Install vLLM's Triton persistent-matmul overrides on the PRIMITIVE GEMM ops (mm/addmm).

    On SM90+ ``enable_batch_invariant_mode`` only pins the cuBLAS workspace config, which gives
    run-to-run determinism but not batch invariance: cuBLAS still selects a different kernel/tiling
    for M=1 than for M=512. The MoE router's fp32 gating mm is measurably not row-invariant, and
    expert GEMMs run at per-expert token counts. Dense models never reach this module.

    Only the primitives are overridden. ``aten::matmul`` and ``aten::linear`` are
    CompositeImplicitAutograd: giving them a CUDA kernel makes them autograd leaves, and torch then
    wants ``aten::matmul_backward`` / ``aten::linear_backward``, which exist only for
    Meta/NestedTensor, so the training backward dies with NotImplementedError.

    Overriding mm/addmm instead loses nothing: matmul/linear decompose into mm/addmm/bmm (bmm is
    already overridden by ``enable_batch_invariant_mode`` on all CUDA platforms), so every GEMM
    still lands on a Triton batch-invariant kernel, and both runtimes decompose identically ->
    engine(no_grad) and trainer(grad) forwards stay bitwise-equal. Verified: fp32 router-GEMM row
    is M-invariant across batch sizes; F.linear/matmul backward work at 2D and 3D, with and
    without bias.
    """
    global _matmul_invariance_lib
    if _matmul_invariance_lib is not None:
        return
    from vllm.model_executor.layers import batch_invariant as bi
    from vllm.platforms import current_platform

    if current_platform.is_device_capability_family(80):
        return  # vLLM's enable_batch_invariant_mode already overrides matmuls on SM80.

    # The override must stay GLOBAL. Restricting it to the fp32 router and leaving bf16 GEMMs on
    # cuBLAS passes every offline gate but breaks IsoExec live, and is not faster.
    lib = torch.library.Library("aten", "IMPL")

    # Forward-only scoping (SKYRL_ISOEXEC_MM_FWD_ONLY=1, default off). Every FORWARD GEMM on both
    # runtimes still runs the batch-invariant kernel; only the trainer's dgrad/wgrad, which the
    # IsoExec gate does not constrain, fall through to cuBLAS. install() is fail-closed and returns
    # False (leaving the unscoped override below) on a flag-off, a failed self-check, or a hook that
    # will not install.
    from ..mm.mm_fwd_scope import install as _install_mm_fwd_scope

    if not _install_mm_fwd_scope(lib):
        lib.impl("aten::mm", bi.mm_batch_invariant, "CUDA")
        lib.impl("aten::addmm", bi.addmm_batch_invariant, "CUDA")
    _matmul_invariance_lib = lib
    print(
        "[ISOEXEC-MOE] Triton batch-invariant matmul overrides installed GLOBALLY (mm/addmm only; "
        "matmul/linear decompose onto them and must stay autograd-differentiable). Yes, this is "
        "2.6-4x slower than cuBLAS per GEMM -- restricting it to the fp32 router was TRIED and it "
        "BROKE IsoExec (6.9e-07 -> 8.8e-03, live 35B A/B) for ZERO speedup. See the comment above "
        "before you 'optimize' it again.",
        flush=True,
    )


def enable_moe_deterministic_ops() -> bool:
    """Patch the MoE combine and router top-k for bitwise decode==prefill. Idempotent."""
    global _orig_unpermute, _orig_topk_routing, _deterministic_enabled

    if _deterministic_enabled:
        return True
    if not moe_deterministic_enabled():
        print(
            "[ISOEXEC-MOE] SKYRL_ISOEXEC_MOE_DETERMINISTIC=0 -> leaving scatter_add_ combine and "
            "grad-dependent top-k in place (baseline A/B; NOT bitwise)",
            flush=True,
        )
        return False

    _install_moe_matmul_invariance()

    # Pinned-cuBLASLt dense-GEMM provider (SKYRL_ISOEXEC_MM_CUBLASLT=1, default off). Installed here,
    # the one function both runtimes call, so both sides resolve the SAME kernel: it is NOT bit-equal
    # to Triton, so a one-sided install would break the gate. Must come AFTER any mm_tiles install so
    # cuBLASLt is outermost and wins its shapes. Idempotent; a failed init self-check falls through to
    # Triton permanently rather than raising.
    from ..mm.mm_cublaslt import install_mm_cublaslt

    install_mm_cublaslt()

    # Fused add+RMSNorm on the within-layer seam (SKYRL_ISOEXEC_FUSED_ADD_NORM, default off). The
    # in-kernel add rounds exactly where eager does. Same shared-site both-runtimes discipline as
    # install_mm_cublaslt above; hard fallback on unowned configs.
    if os.environ.get("SKYRL_ISOEXEC_FUSED_ADD_NORM", "0") == "1":
        from ..norms.fused_add_rmsnorm import install as _install_fused_add_norm

        _install_fused_add_norm()

    from megatron.core.transformer.moe import moe_utils, router, token_dispatcher

    _orig_unpermute = moe_utils.unpermute
    _orig_topk_routing = moe_utils.topk_routing_with_score_function
    sorted_topk_routing = _make_sorted_topk_routing(_orig_topk_routing)

    # `token_dispatcher` and `router` bind these at import, so patch every namespace that holds them.
    moe_utils.unpermute = _deterministic_unpermute
    token_dispatcher.unpermute = _deterministic_unpermute
    moe_utils.topk_routing_with_score_function = sorted_topk_routing
    router.topk_routing_with_score_function = sorted_topk_routing

    # Forward permute index build: the trainer-reachable counterpart of the engine-only counting sort,
    # since the alltoall dispatcher calls module-level `moe_utils.permute`. Integer permute + gather.
    from .moe_fused_permute import install_fused_permute

    install_fused_permute()

    _deterministic_enabled = True
    print(
        "[ISOEXEC-MOE] deterministic ops installed: fixed-order expert combine (was CUDA "
        "scatter_add_ atomics) + sorted router top-k (was sorted=is_grad_enabled)",
        flush=True,
    )
    return True


def revert_moe_deterministic_ops() -> None:
    """Restore megatron-core's originals (used by the unit test's A/B)."""
    global _deterministic_enabled, _validated_calls, _matmul_invariance_lib

    if not _deterministic_enabled:
        return
    from megatron.core.transformer.moe import moe_utils, router, token_dispatcher

    moe_utils.unpermute = _orig_unpermute
    token_dispatcher.unpermute = _orig_unpermute
    moe_utils.topk_routing_with_score_function = _orig_topk_routing
    router.topk_routing_with_score_function = _orig_topk_routing
    from .moe_fused_permute import revert_fused_permute

    revert_fused_permute()
    if _matmul_invariance_lib is not None:
        # De-register the aten overrides too, or an "unpatched" baseline still runs on Triton matmuls.
        _matmul_invariance_lib._destroy()
        _matmul_invariance_lib = None
    _deterministic_enabled = False
    _validated_calls = 0


def lift_moe_tp_sp_training_veto() -> None:
    """Allow legacy SP-off MoE training configurations to pass Megatron's performance veto.

    The optimized trainer keeps sequence parallelism enabled, so this wrapper delegates directly
    to Megatron on that path.  It remains as a compatibility fallback for explicitly SP-off
    configurations, where Megatron's parent ``MoELayer.training`` guard is temporarily suppressed.
    Router and expert child modules retain their training state.

    Megatron also keys selective MoE activation recomputation on the parent flag.  When that
    feature is configured, this wrapper reapplies the same checkpoint around the wrapped forward;
    ``SKYRL_ISOEXEC_MOE_RECOMPUTE=0`` disables that compatibility repair.
    """
    from megatron.core import tensor_parallel
    from megatron.core.transformer.moe.moe_layer import MoELayer

    if getattr(MoELayer, "_isoexec_sp_veto_lifted", False):
        return
    _orig = MoELayer.forward
    _recompute_ok = os.environ.get("SKYRL_ISOEXEC_MOE_RECOMPUTE", "1") == "1"

    def _fwd(self, *args, **kwargs):
        if self.training and self.attn_tp_group.size() > 1 and not self.config.sequence_parallel:

            def _run(*a):
                # The flip MUST live inside the run-function, not around the checkpoint call:
                # CheckpointFunction re-invokes this during the BACKWARD to recompute, long after a
                # forward-scoped `finally` would have restored training=True.
                object.__setattr__(self, "training", False)
                try:
                    return _orig(self, *a, **kwargs)
                finally:
                    object.__setattr__(self, "training", True)

            # Re-apply the checkpoint the flip suppresses. fp8/fp4 would need te_checkpoint instead,
            # so fall through to the un-checkpointed call rather than silently swap mechanisms.
            if (
                _recompute_ok
                and getattr(self, "moe_layer_recompute", False)
                and not (self.config.fp8 or self.config.fp4)
            ):
                if not getattr(MoELayer, "_isoexec_moe_recompute_logged", False):
                    MoELayer._isoexec_moe_recompute_logged = True
                    print(
                        "[ISOEXEC-MOE] moe activation recompute RE-APPLIED over MoELayer.forward "
                        "(recompute_modules contains 'moe'; megatron's own gate is suppressed by "
                        "the SP-veto training flip). SKYRL_ISOEXEC_MOE_RECOMPUTE=0 to disable.",
                        flush=True,
                    )
                # kwargs stay bound in the closure: they carry no grad, and CheckpointFunction only
                # saves/detaches positional args, matching megatron's own positional call.
                return tensor_parallel.checkpoint(_run, False, *args)
            return _run(*args)
        return _orig(self, *args, **kwargs)

    MoELayer.forward = _fwd
    MoELayer._isoexec_sp_veto_lifted = True
    print(
        "[ISOEXEC-MOE] installed TP-without-SP compatibility wrapper "
        "(inactive when sequence parallelism is enabled)",
        flush=True,
    )


def _moe_pik_fc2_on() -> bool:
    return os.environ.get("SKYRL_ISOEXEC_MOE_PIK_FC2") == "1"


# bf16-wire / fp32-root fast path (kill switch SKYRL_ISOEXEC_MOE_PIK_BF16_WIRE=0). When world == G
# each rank's combine partial is one leaf, already rounded through bf16 by the leaf-tree fc2, so bf16
# on the wire is lossless and halves the combine's bytes. The ROOT must stay fp32: the downstream
# unpermute top-k sum takes its dtype from its input, so a bf16 root would be sum-of-rounded rather
# than fp32-sum-then-one-round. The fp32 root is bit-identical to the fp32-wire root by construction.
# Guarded by world==G only (an internal fp32 node at ETP<G is not bf16-representable, so the trainer
# stays fp32-wire) plus a first-call losslessness self-check with permanent fallback.
_bf16_wire = {"checked": False, "ok": False}


def _pik_bf16_wire_enabled() -> bool:
    return os.environ.get("SKYRL_ISOEXEC_MOE_PIK_BF16_WIRE", "1") == "1"


class _PikTreeAllReduce(torch.autograd.Function):
    """Differentiable pik fixed-tree all-reduce over the ETP (tp_ep) group.

    Forward sums the per-rank fc2 leaf-subtree partials with pik's balanced tree, so the result is
    bitwise identical for any ETP dividing G. Backward passes the gradient through, matching
    Megatron's ``reduce_from_tensor_model_parallel_region`` convention.
    """

    @staticmethod
    def forward(ctx, x, group):
        import torch.distributed as dist

        from ...ops.collectives.pik_tp_invariant import ensure_pik, get_plan

        ensure_pik()
        from pik.allreduce import (  # type: ignore
            p2p_available,
            sym_partial,
            tree_all_reduce,
        )

        x = x.contiguous()
        if (
            _pik_bf16_wire_enabled()
            and x.dtype == torch.float32
            and dist.get_world_size(group) == get_plan().num_leaves
        ):
            if not _bf16_wire["checked"]:
                _bf16_wire["checked"] = True
                _bf16_wire["ok"] = bool((x.to(torch.bfloat16).to(torch.float32) == x).all())
                print(
                    f"[ISOEXEC-MOE] pik combine bf16-wire/fp32-root self-check "
                    f"{'PASS (lossless leaf; shot-1 wire bytes halved)' if _bf16_wire['ok'] else 'FAIL -> fp32 wire permanently'}",
                    flush=True,
                )
            if _bf16_wire["ok"]:
                # bf16 leaf on the wire, fp32 root out. The bf16 leaf is staged straight into
                # peer-visible memory in one fp32->bf16 pass, so `_tree_all_reduce_p2p` recognises its
                # own staging buffer by data_ptr and skips the copy-in. `staged.copy_(x)` is RNE
                # fp32->bf16, bit-identical to `x.to(bf16)`, and is capture-safe (the pool is warmed
                # before capture).
                if p2p_available(group):
                    staged = sym_partial(x.shape, x.device, group, dtype=torch.bfloat16, out_dtype=torch.float32)
                    staged.copy_(x)
                    return tree_all_reduce(staged, group=group, root_dtype=torch.float32)
                return tree_all_reduce(x.to(torch.bfloat16), group=group, root_dtype=torch.float32)
        return tree_all_reduce(x, group=group)

    @staticmethod
    def backward(ctx, g):
        return g, None


_pik_fc2_combine_patched = False


def _nogather_active(disp) -> bool:
    """The engine no-gather fast path applies to THIS dispatcher call iff every condition holds.

    With SP off and TP>1 the allgather dispatcher gathers hidden/probs/routing_map across the tp_ep
    group, but each rank already holds the full batch, so the gather produces TP identical copies and
    every rank does TP times the work. Under pik-fc2 the gather buys nothing, because expert outputs
    are combined by pik's ``tree_all_reduce`` to a bit-identical result on every rank. Bitwise-neutral
    by construction: every gathered copy's rows were identical. ENGINE-ONLY (marked instances) -- the
    trainer's gathered path has autograd semantics tied to the gather's backward.
    """
    return (
        _moe_pik_fc2_on()
        and getattr(disp, "_isoexec_engine_nogather", False)
        and getattr(disp, "tp_size", 1) > 1
        and getattr(disp, "ep_size", 1) == 1
        and not getattr(disp.config, "sequence_parallel", False)
    )


def mark_engine_dispatchers_nogather(model) -> int:
    """Mark every allgather token dispatcher in the ENGINE model for the no-gather fast path.

    Called after the engine GPTModel is built, never from the trainer. The mark is what scopes the
    patch: the class-level patches below fire for both runtimes in a colocated process, and only
    marked instances take the fast path.
    """
    try:
        from megatron.core.transformer.moe.token_dispatcher import (
            MoEAllGatherTokenDispatcher,
        )
    except Exception:  # pragma: no cover
        return 0
    n = 0
    for m in model.modules():
        disp = getattr(m, "token_dispatcher", None)
        if isinstance(disp, MoEAllGatherTokenDispatcher):
            disp._isoexec_engine_nogather = True
            n += 1
    if n:
        print(
            f"[ISOEXEC-MOE] ENGINE no-gather dispatch marked on {n} MoE dispatchers "
            f"(SP off, EP=1: skip the tp_ep token gather -> 1/TP the expert rows, combine traffic "
            f"and unpermute; pik tree output is replicated so no scatter either)",
            flush=True,
        )
    return n


def install_moe_pik_fc2_combine() -> bool:
    """ETP-invariant MoE combine (SKYRL_ISOEXEC_MOE_PIK_FC2).

    Reduces the expert output across the ETP (tp_ep) group with pik's fixed tree *before* the top-k
    combine, so trainer-ETP != engine-ETP stays bitwise. ``token_combine`` then only SCATTERS --
    a reduce_scatter there would double-reduce. Pairs with the leaf-tree fc2 in moe_batched_experts.
    Idempotent; only fires when SKYRL_ISOEXEC_MOE_PIK_FC2=1 at call time. Marked ENGINE dispatchers
    additionally skip the token gather entirely (see :func:`_nogather_active`).
    """
    global _pik_fc2_combine_patched
    if _pik_fc2_combine_patched:
        return True
    try:
        from megatron.core.transformer.moe.token_dispatcher import (
            MoEAllGatherTokenDispatcher as D,
        )
    except Exception as e:  # pragma: no cover
        logger.info("[isoexec] pik-fc2 combine unavailable (%s)", e)
        return False

    _orig_td = D.token_dispatch
    _orig_pre = D.combine_preprocess
    _orig_tc = D.token_combine

    def token_dispatch(self, hidden_states, probs):
        if _nogather_active(self):
            # SP off: every rank already holds the full batch, so the gather would only make
            # tp_size identical copies. Keep the local routing_map/probs/hidden as-is.
            return hidden_states, probs
        return _orig_td(self, hidden_states, probs)

    def combine_preprocess(self, hidden_states):
        # hidden_states is this rank's fp32 leaf-subtree partial; reduce it across the ETP group with
        # pik's tree to a full fc2 per row, BEFORE unpermute + top-k.
        if _moe_pik_fc2_on() and getattr(self, "tp_size", 1) > 1:
            hidden_states = _PikTreeAllReduce.apply(hidden_states, self.tp_ep_group)
        return _orig_pre(self, hidden_states)

    def token_combine(self, hidden_states):
        if _nogather_active(self):
            # No gather happened, so nothing to scatter: the pik tree already left the bit-identical
            # full-batch result on every rank. Just cast to the residual dtype.
            return hidden_states.to(torch.bfloat16)
        if _moe_pik_fc2_on() and getattr(self, "tp_size", 1) > 1:
            # Already ETP-reduced in combine_preprocess: just SCATTER (undo the dispatch all-gather,
            # which concatenated tokens in rank order, so rank r keeps contiguous chunk r) and cast to
            # the bf16 residual dtype on both sides identically.
            import torch.distributed as dist

            tp = self.tp_size
            r = dist.get_rank(self.tp_ep_group)
            n = hidden_states.shape[0] // tp
            return hidden_states[r * n : (r + 1) * n].to(torch.bfloat16).contiguous()
        return _orig_tc(self, hidden_states)

    D.token_dispatch = token_dispatch
    D.combine_preprocess = combine_preprocess
    D.token_combine = token_combine
    _pik_fc2_combine_patched = True
    print("[ISOEXEC-MOE] pik-fc2 ETP-invariant combine installed (reduce-before-topk + scatter)", flush=True)
    return True


_pik_fc2_alltoall_patched = False


def install_moe_pik_fc2_alltoall_combine() -> bool:
    """pik-fc2 counterpart for the ALLTOALL dispatcher (the EP>1 trainer side).

    With SKYRL_ISOEXEC_MOE_PIK_FC2=1 the batched experts emit fp32 full-fc2 rows, and the two
    dispatchers must round at the same point in the expression tree: the top-k sum runs in fp32 and
    exactly one bf16 round follows it. Rounding on the other side of the sum is not bitwise-equal.
    EP>1 with ETP>1 would additionally need the ETP reduce_scatter replaced by pik's tree; that
    layout is not a IsoExec target and warns loudly if seen.
    """
    global _pik_fc2_alltoall_patched
    if _pik_fc2_alltoall_patched:
        return True
    try:
        from megatron.core.transformer.moe.token_dispatcher import (
            MoEAlltoAllTokenDispatcher as A,
        )
    except Exception as e:  # pragma: no cover
        logger.info("[isoexec] pik-fc2 alltoall combine unavailable (%s)", e)
        return False

    _orig_pre_a2a = A.combine_preprocess
    _orig_post_a2a = A.combine_postprocess

    def combine_preprocess(self, hidden_states):
        # Keep the pik-fc2 fp32 rows fp32 through the unsort + alltoall. Casting to bf16 here is the
        # wrong side of the sum: the allgather flow sums top-k on fp32 rows and casts afterwards, so
        # rounding first costs a uniform ULP-scale error on every token.
        if _moe_pik_fc2_on() and hidden_states.dtype == torch.float32 and getattr(self, "tp_size", 1) > 1:
            print(
                "[ISOEXEC-MOE] WARNING: pik-fc2 with alltoall dispatcher at ETP>1 -- the ETP "
                "reduce_scatter is NOT tree-invariant; IsoExec will not hold in this layout",
                flush=True,
            )
        return _orig_pre_a2a(self, hidden_states)

    def combine_postprocess(self, permutated_local_input_tokens):
        if not (_moe_pik_fc2_on() and permutated_local_input_tokens.dtype == torch.float32):
            return _orig_post_a2a(self, permutated_local_input_tokens)
        # Mirror the engine's expression tree: top-k sum on fp32 full-fc2 rows (referenced through the
        # module so the patched unpermute applies), THEN one bf16 round, THEN the shared-expert add.
        from megatron.core.transformer.moe import token_dispatcher as _td

        from .moe_combine_kernel import fold_round_enabled, note_fold_request

        # Round fold (SKYRL_ISOEXEC_MOE_COMBINE_FOLD_ROUND, default off). The `.to(torch.bfloat16)`
        # below is the single round the pik expression tree owes after the fp32 top-k sum. Folding it
        # into the combine's store does not move it -- it is still one RNE of the same fp32
        # accumulator -- it only avoids writing the fp32 [T, H] result to HBM and reading it back. The
        # `.to` is kept unconditionally, so a binding that ignores the request stays correct.
        #
        # The kwarg is only sent to a binding that published `_isoexec_accepts_out_dtype`; megatron's
        # own `unpermute` would raise TypeError. An unmarked binding declines loudly instead.
        kw = {}
        if fold_round_enabled():
            if getattr(_td.unpermute, "_isoexec_accepts_out_dtype", False):
                note_fold_request()
                kw["isoexec_out_dtype"] = torch.bfloat16
            else:
                _fold_refused_once(getattr(_td.unpermute, "__name__", repr(_td.unpermute)))
        output = _td.unpermute(
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.routing_map,
            fused=self.config.moe_permute_fusion,
            drop_and_pad=self.drop_and_pad,
            **kw,
        )
        output = output.view(self.hidden_shape).to(torch.bfloat16)
        if self.shared_experts is not None:
            output = output + self.shared_experts.get_output()
        return output

    A.combine_preprocess = combine_preprocess
    A.combine_postprocess = combine_postprocess
    _pik_fc2_alltoall_patched = True
    print("[ISOEXEC-MOE] pik-fc2 alltoall combine installed (fp32-topk-sum, round-after-sum == engine)", flush=True)
    return True


def prepare_isoexec_moe(provider, *, side: str) -> bool:
    """Everything the MoE IsoExec path needs on one side. Returns True when MoE was engaged.

    Call after the provider exists and before ``finalize()``/model build. No-op for dense providers,
    so the validated dense path is untouched.
    """
    if not provider_is_moe(provider):
        return False
    force_isoexec_moe_config(provider, side=side)
    enable_moe_deterministic_ops()

    # Fused router + permute sort (SKYRL_ISOEXEC_MOE_ROUTER_O2, default off): three Triton kernels
    # replacing torch.topk, the batch-invariant softmax decomposition, the dense index_put_ pair and
    # the dispatcher's stable [E*T] argsort. Bitwise-equal, so a one-sided install is safe. The class
    # patch is installed on both sides but delegates to megatron for any instance the ENGINE model
    # build did not mark, because with VLLM_ENABLE_V1_MULTIPROCESSING=0 both runtimes share this
    # process and a class-level rebind would sever the trainer's MoE backward.
    from .moe_router_o2_kernel import install_router_o2

    install_router_o2()

    # Routing mechanics (SKYRL_ISOEXEC_MOE_DENSE_SCATTER / SKYRL_ISOEXEC_MOE_PERMUTE_SORT, default off),
    # for models whose score chain O2 cannot serve. Replaces only the INTEGER and ORDERING work around
    # the score function -- no floating-point arithmetic at all. INSTALL ORDER: must come after
    # _make_sorted_topk_routing, because the wrapper closes over whatever
    # `topk_routing_with_score_function` is bound to and the `sorted=True` forcing must already be in
    # place for the top-k inside it. ENGINE-ONLY on marked instances.
    from .moe_dense_scatter_kernel import install_dense_scatter

    install_dense_scatter()

    # Router chain ends (SKYRL_ISOEXEC_MOE_ROUTER_SCORE / _ROUTER_TAIL, default off): the score
    # function + bias add and the normalisation tail + dense build. INSTALL ORDER: after
    # install_dense_scatter, so this wrapper is outermost and every call it declines still reaches the
    # mechanics wrapper and then _make_sorted_topk_routing -- which is why the transcription inside
    # must pass `sorted=True` explicitly, since it sits outside that forcing. The top-k (tie order is
    # load-bearing under a selection-only bias) and the fp32 row-sum are not replaced.
    from .moe_router_chain_kernel import install_router_chain

    install_router_chain()

    lift_moe_tp_sp_training_veto()
    # padded batched expert GEMMs;
    # must be installed on BOTH sides -- it changes numerics vs the loop, identically for both.
    from .moe_batched_experts import (
        install_batched_sequential_mlp,
        install_fixed_shape_dispatch,
    )

    install_batched_sequential_mlp()
    # Fixed-shape permuted probs instead of the dispatcher's masked_select. Bitwise-equal (a pure
    # gather of the same elements in the same order), so it does not need both sides to move together,
    # but it is installed on both anyway for speed.
    install_fixed_shape_dispatch()

    # One-gather chunk sort (SKYRL_ISOEXEC_MOE_CHUNK_SORT, default off). With `moe_permute_fusion`
    # pinned off, megatron's `sort_chunks_by_idxs` is a python loop over slices joined by torch.cat.
    # The reordering is a bijection on rows, so this rebinds it to one index_select with the inverse
    # gather as the VJP. Safe on BOTH sides: bit-equal forward and backward by construction (two pure
    # copies, no add anywhere -- the VJP is deliberately NOT index_add_, since 0.0 + -0.0 == +0.0
    # would flip signed zeros), and admitted per shape on the live operands before it is taken.
    from .moe_chunk_sort import install_chunk_sort

    install_chunk_sort()

    # Analytic MoE backward -- TRAINER ONLY: the engine has no backward, and its forward is already
    # the fused kernel. The win is also dim-dependent (it loses at small moe_intermediate per expert),
    # so enable it only for EP>1/ETP=1-style trainer parallelism. Its FORWARD is a byte-for-byte
    # transcription of _batched_experts_forward and must be re-checked whenever that forward changes,
    # or the copy silently drifts. Gradients are free to differ; IsoExec constrains only the forward.
    if side == "TRAINER":
        from .moe_backward_kernel import install_fastbwd_experts

        install_fastbwd_experts()

    # ETP-invariant fc2 (SKYRL_ISOEXEC_MOE_PIK_FC2): leaf-tree expert fc2 plus the reduce-before-topk
    # combine, so trainer-ETP != engine-ETP stays bitwise.
    if _moe_pik_fc2_on():
        install_moe_pik_fc2_combine()
        # The EP>1 trainer side uses the ALLTOALL dispatcher (allgather deadlocks on unequal per-rank
        # token counts) and needs the matching round-after-sum arithmetic.
        install_moe_pik_fc2_alltoall_combine()
        # bf16 BACKWARD wire for that same alltoall combine (SKYRL_ISOEXEC_MOE_A2A_BF16_WIRE, default
        # off). INSTALL ORDER: immediately after the pik-fc2 alltoall combine, because it is that
        # arithmetic which makes the narrowing lossless -- combine_postprocess applies the single bf16
        # round, so the gradient flowing back into token_combine's alltoall is an exact upcast of bf16
        # values. FORWARD UNCHANGED: narrowing the forward wire is not lossless. Trainer-only; it
        # RAISES on a failed self-check rather than falling back, for collective safety.
        from .moe_a2a_wire import install_moe_a2a_bf16_wire

        install_moe_a2a_bf16_wire()
        # Owner-computes combine (SKYRL_ISOEXEC_MOE_PIK_OWNER_COMBINE, default off): a bit-identical
        # restructure where each rank computes tree + topk-sum + round for its own token slice and
        # all-gathers the bf16 finals. INSTALL ORDER: must come after the pik-fc2 installs, since it
        # wraps that binding and delegates every non-owner path back to it.
        from .moe_pik_combine_owner import install_moe_pik_owner_combine

        install_moe_pik_owner_combine()

    # Fused MoE GEMMs -- ENGINE ONLY. The fused kernel is faster than the batched bmm, but its speed
    # comes from a padding-free forward and every bmm-shaped backward must rebuild that padding, which
    # makes fused-in-training a net loss. The rollout has no backward, so fused goes only there and the
    # trainer keeps the batched forward+backward. IsoExec survives the asymmetry only because the fused
    # forward is BITWISE the batched-bmm forward. INSTALL ORDER: last, and engine-only, so its
    # SequentialMLP.forward rebind wins over the batched one. Under SKYRL_ISOEXEC_MOE_PIK_FC2 the fused
    # fc2 runs the same leaf tree as the bmm path, so fused stays enabled with mismatched ETP.
    if side == "ENGINE":
        from .moe_fused_experts import install_fused_experts

        install_fused_experts()

        # Fused block map (SKYRL_ISOEXEC_MOE_FUSED_BLOCKMAP, default off) replaces `_block_map`'s tiny
        # integer torch ops with one Triton launch. Nothing to install -- `_block_map` re-reads the
        # flag per call. This line only logs the flag's value as the engine actor actually sees it,
        # which is how a flag that failed to reach the actor becomes visible.
        print(
            "[ISOEXEC-MOE] ENGINE: fused block map (O5) "
            f"{'ON -- 31 -> 1 launch/layer' if os.environ.get('SKYRL_ISOEXEC_MOE_FUSED_BLOCKMAP', '0') == '1' else 'OFF (31 torch ops/layer)'}",
            flush=True,
        )

    # Fused combine (SKYRL_ISOEXEC_MOE_FUSED_COMBINE, default off): one Triton launch per layer in
    # place of the top-k combine loop's k index_selects and k-1 out-of-place adds.
    #
    # The bitwise rule it must honour: the top-k sum accumulates in FP32, IN K-ORDER, with EXACTLY ONE
    # bf16 round AFTER the sum, never per term. The kernel walks the same k rows in the same
    # ascending-expert order in registers and rounds once at the store; `enable_fp_fusion=False` keeps
    # LLVM from contracting the weighted branch into an FFMA.
    #
    # INSTALL ORDER: last, and engine-scoped by instance mark. `unpermute` is a module-level function,
    # so the install is a process-global rebind, and with VLLM_ENABLE_V1_MULTIPROCESSING=0 the trainer
    # shares this process -- an unconditional rebind hands the trainer a raw Triton call with no
    # grad_fn and severs its MoE backward while the forward-only gate stays green. Installing last is
    # what makes the scope wrappers outermost, i.e. outside the pik-fc2 combine patches above.
    # Engine-only is safe precisely because the kernel is bitwise-equal to the combine it replaces.
    if os.environ.get("SKYRL_ISOEXEC_MOE_FUSED_COMBINE", "0") == "1" or (
        os.environ.get("SKYRL_ISOEXEC_MOE_FUSED_COMBINE_TRAINER", "0") == "1"
    ):
        from .moe_combine_kernel import install_fused_combine

        install_fused_combine(side=side)

    # The combine's stable argsort. Nothing to install: _fixed_order_combine and build_combine_rows
    # both re-read SKYRL_ISOEXEC_MOE_COMBINE_SORT per call; this only logs the flag as the actor sees it.
    if side == "ENGINE":
        from .moe_combine_rows_kernel import combine_sort_enabled

        print(
            "[ISOEXEC-MOE] ENGINE: combine stable sort (O7) "
            f"{'ON -- counting sort' if combine_sort_enabled() else 'OFF (radixSortKVInPlace, grid 1x32)'}",
            flush=True,
        )
    return True
