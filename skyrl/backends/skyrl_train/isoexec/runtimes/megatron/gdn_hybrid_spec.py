"""A hybrid (GatedDeltaNet + softmax attention) Megatron layer spec built from local modules only.

Megatron's own hybrid builder asserts ``transformer_impl == "transformer_engine"`` and asks the
backend for TE's fused layernorm+linear, neither of which this TE-free stack has. Building the spec
directly means ``in_proj`` is a plain ``ColumnParallelLinear`` with a separate ``input_layernorm``,
which is also why the Qwen3.5 bridge mapping has to be retargeted -- see
``patch_qwen35_bridge_for_local_spec``.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


def is_hybrid_gdn(config) -> bool:
    """True when this provider/config describes a GatedDeltaNet hybrid (Qwen3.5, Qwen3-Next)."""
    return getattr(config, "experimental_attention_variant", None) == "gated_delta_net"


def _gdn_layer_spec(config, base):
    """A local GDN layer: ``base``'s TransformerLayer with self_attention -> GatedDeltaNet."""
    from megatron.core.models.backends import LocalSpecProvider
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet, GatedDeltaNetSubmodules
    from megatron.core.transformer.spec_utils import ModuleSpec

    backend = LocalSpecProvider()
    rms_norm = config.normalization == "RMSNorm"
    spec = ModuleSpec(
        module=GatedDeltaNet,
        submodules=GatedDeltaNetSubmodules(
            # TE would fuse the input layernorm into in_proj (TELayerNormColumnParallelLinear).
            # Locally there is no fused module, so in_proj is a plain ColumnParallelLinear and the
            # TransformerLayer keeps its own `input_layernorm`.
            in_proj=backend.column_parallel_linear(),
            out_norm=backend.layer_norm(rms_norm=rms_norm, for_qk=False),
            out_proj=backend.row_parallel_linear(),
        ),
        metainfo={"fuse_input_layernorm": False},
    )
    layer = ModuleSpec(module=base.module, submodules=type(base.submodules)(**vars(base.submodules)))
    layer.submodules.self_attention = spec
    return layer


def make_isoexec_hybrid_local_spec(config):
    """``TransformerBlockSubmodules`` with GDN on the linear-attention layers, SelfAttention elsewhere.

    Both layer kinds keep their own ``input_layernorm`` and ``pre_mlp_layernorm`` (no TE fusion).
    """
    from megatron.core.models.backends import LocalSpecProvider
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_linear_attention_pattern,
    )
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer.transformer_block import TransformerBlockSubmodules

    if config.pipeline_model_parallel_size != 1:
        raise NotImplementedError("[isoexec-gdn] hybrid local spec is PP=1 only (no layer slicing)")

    backend = LocalSpecProvider()
    rms_norm = config.normalization == "RMSNorm"
    pattern = get_linear_attention_pattern(config)  # 1 = linear attention (GDN), 0 = softmax

    base = get_gpt_layer_local_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=False,
        qk_layernorm=config.qk_layernorm,
        normalization=config.normalization,
    )
    layer_specs = [_gdn_layer_spec(config, base) if p == 1 else base for p in pattern]

    n_gdn = sum(pattern)
    print(
        f"[ISOEXEC-SPEC] hybrid local spec: {n_gdn} GatedDeltaNet + {len(pattern) - n_gdn} "
        f"attention layers (no TransformerEngine)",
        flush=True,
    )
    return TransformerBlockSubmodules(
        layer_specs=layer_specs, layer_norm=backend.layer_norm(rms_norm=rms_norm, for_qk=False)
    )


_bridge_patched = False
_bridge_patch_mode: str | None = None

# TE fuses the input layernorm into the following linear, so megatron-bridge's Qwen3.5 mapping names
# it as that linear's `layer_norm_weight`. Under the local spec the layernorms are separate modules.
_LOCAL_SPEC_RENAMES = {
    "self_attention.linear_qkv.layer_norm_weight": "input_layernorm.weight",
    "self_attention.in_proj.layer_norm_weight": "input_layernorm.weight",
    "mlp.linear_fc1.layer_norm_weight": "pre_mlp_layernorm.weight",
}

# The IsoExec MoE recipe pins SequentialMLP, whose per-expert params are
# `mlp.experts.local_experts.<i>.linear_fcX.weight`, while the Qwen3.5 MoE bridge only declares the
# grouped-GEMM names `mlp.experts.linear_fcX.weight<i>`. Without this rename nothing matches and every
# expert silently stays at its random init. It is a pure rename, so the FusedExpertMapping classes
# keep working.
_SEQUENTIAL_MLP_RENAMES = {
    "mlp.experts.linear_fc1.weight*": "mlp.experts.local_experts.*.linear_fc1.weight",
    "mlp.experts.linear_fc2.weight*": "mlp.experts.local_experts.*.linear_fc2.weight",
}


_chunked_patched = False


def patch_chunked_mapping_index_device() -> bool:
    """Keep ``ChunkedMapping``'s shard indices on the CPU, where the HF weights are. Idempotent.

    ``get_shard_idx`` builds its indices with a bare ``torch.arange``, and inside vLLM's model loader
    the default device is ``cuda``, so indexing the CPU-resident ``hf_weights`` raises. Only the GDN
    and Mamba mappings inherit this; the trainer never hits it.
    """
    global _chunked_patched

    if _chunked_patched:
        return True
    try:
        import torch
        from megatron.bridge.models.conversion.param_mapping import ChunkedMapping
    except Exception as e:  # pragma: no cover
        logger.info("[isoexec-gdn] ChunkedMapping unavailable (%s)", e)
        return False

    def _wrap(orig):
        def _cpu_idx(self, config, local_tp):
            return [i.cpu() if torch.is_tensor(i) else i for i in orig(self, config, local_tp)]

        return _cpu_idx

    # Each subclass (GDNConv1dMapping, MambaConv1dMapping, ...) defines its own get_shard_idx and
    # shadows the base, so patching only the base silently does nothing.
    classes = [ChunkedMapping, *ChunkedMapping.__subclasses__()]
    n = 0
    for cls in classes:
        if "get_shard_idx" in cls.__dict__:
            cls.get_shard_idx = _wrap(cls.__dict__["get_shard_idx"])
            n += 1

    _chunked_patched = True
    logger.info("[isoexec-gdn] pinned get_shard_idx to CPU indices on %d ChunkedMapping class(es)", n)
    return True


def checkpoint_is_vl_named(hf_config) -> bool:
    """True when the checkpoint stores the LM under ``model.language_model.`` (VL architecture).

    Every released Qwen3.5 checkpoint does; a hand-exported LM-only checkpoint would not. Read the
    architecture BEFORE ``maybe_force_qwen35_text_bridge`` rewrites it.
    """
    archs = list(getattr(hf_config, "architectures", []) or [])
    return any(a.endswith("ForConditionalGeneration") for a in archs)


def _patch_qwen35_bridge(*, hf_lm_prefix: str | None, mode: str, renames) -> bool:
    """Apply one mutually-exclusive Qwen3.5 mapping mode before the registry is built."""
    global _bridge_patched, _bridge_patch_mode

    if _bridge_patched:
        if _bridge_patch_mode != mode:
            raise RuntimeError(
                f"[isoexec-gdn] Qwen3.5 bridge already patched for {_bridge_patch_mode}; "
                f"refusing incompatible {mode} mapping"
            )
        return True
    if os.environ.get("SKYRL_ISOEXEC_GDN") != "1":
        return False
    try:
        from megatron.bridge.models.qwen import qwen35_bridge as qb
    except Exception as e:  # pragma: no cover
        logger.info("[isoexec-gdn] qwen35_bridge unavailable (%s)", e)
        return False

    def _wrap(fn):
        def inner(hf_prefix="model.", megatron_prefix=""):
            if hf_lm_prefix and hf_prefix == "model.":
                hf_prefix = hf_lm_prefix
            mappings = fn(hf_prefix=hf_prefix, megatron_prefix=megatron_prefix)
            for m in mappings:
                mp = getattr(m, "megatron_param", None)
                if not mp:
                    continue
                for old, new in renames:
                    if mp.endswith(old):
                        m.megatron_param = mp[: -len(old)] + new
                        # __init__ validated the original pattern; re-check ours (the wildcard count
                        # must still line up with hf_param, or resolution silently mispairs captures).
                        if hasattr(m, "_validate_patterns"):
                            m._validate_patterns()
                        break
            return mappings

        return inner

    n = 0
    for name in ("_get_dense_lm_mappings", "_get_moe_lm_mappings"):
        for cls in (getattr(qb, "Qwen35Bridge", None), getattr(qb, "Qwen35MoEBridge", None)):
            fn = getattr(cls, name, None) if cls else None
            if fn is None:
                continue
            setattr(cls, name, staticmethod(_wrap(fn.__func__ if hasattr(fn, "__func__") else fn)))
            n += 1

    patch_chunked_mapping_index_device()
    _bridge_patched = True
    _bridge_patch_mode = mode
    print(
        f"[ISOEXEC-SPEC] retargeted {n} Qwen3.5 bridge mapping table(s) for {mode} "
        f"(hf_lm_prefix={hf_lm_prefix or 'model.'})",
        flush=True,
    )
    return True


def patch_qwen35_bridge_for_local_spec(*, hf_lm_prefix: str | None = None) -> bool:
    """Retarget the Qwen3.5 bridge's weight mapping at the no-TE local spec. Idempotent.

    Fixes two independent and otherwise silent mismatches: TE-fused layernorm names
    (``...in_proj.layer_norm_weight`` vs the local ``...input_layernorm.weight``), and the HF prefix --
    the text bridge builds ``model.layers.*`` while released checkpoints store
    ``model.language_model.layers.*``, so pass ``hf_lm_prefix="model.language_model."`` for those.
    Must run before the bridge's mapping registry is built, i.e. before ``AutoBridge.from_hf_pretrained``.
    """
    return _patch_qwen35_bridge(
        hf_lm_prefix=hf_lm_prefix,
        mode="local-spec",
        renames=(*_LOCAL_SPEC_RENAMES.items(), *_SEQUENTIAL_MLP_RENAMES.items()),
    )


def patch_qwen35_bridge_for_selective_te(*, hf_lm_prefix: str | None = None) -> bool:
    """Retarget separate exact norms and routed experts for selective TE.

    Selective TE leaves gate-critical norms as separate local modules and routed experts as a local
    ``SequentialMLP``, so it needs the same two structural rewrites as local-spec mode. The two modes
    are mutually exclusive at runtime.
    """
    return _patch_qwen35_bridge(
        hf_lm_prefix=hf_lm_prefix,
        mode="selective-te",
        renames=(*_LOCAL_SPEC_RENAMES.items(), *_SEQUENTIAL_MLP_RENAMES.items()),
    )
