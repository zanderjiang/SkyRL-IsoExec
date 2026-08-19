"""A model-general trainer-side decoder-block spec, built without TransformerEngine.

Megatron already derives block structure from config alone (dense / GQA / MLA / MoE / hybrid GDN),
but its hybrid path asserts ``transformer_impl == "transformer_engine"`` and asks the backend for
TE's fused layernorm+linear, so this module supplies a local backend and, for the duration of one
spec build, substitutes those two TE-hardcoded builders with their honest no-TE answers. This is not
the shipped path: it is enabled by ``SKYRL_ISOEXEC_SPEC_PROVIDER=1`` and the default remains
``gdn_hybrid_spec``.
"""

from __future__ import annotations

import contextlib
import logging
import os

logger = logging.getLogger(__name__)

ENABLE_ENV = "SKYRL_ISOEXEC_SPEC_PROVIDER"


def enabled() -> bool:
    """Is the generic spec-provider path turned on? Default OFF (see the module docstring)."""
    return os.environ.get(ENABLE_ENV, "0") == "1"


def make_provider():
    """Build the ``IsoExecSpecProvider`` class; deferred because megatron is imported lazily.

    Subclasses ``LocalSpecProvider`` because the IsoExec stack's module choices are the local ones;
    what it adds is an explicit statement of the two facts megatron's hybrid path assumes otherwise.
    """
    from megatron.core.models.backends import LocalSpecProvider

    class IsoExecSpecProvider(LocalSpecProvider):
        """The IsoExec trainer backend: local modules, nothing fused into a layernorm.

        ``core_attention`` deliberately stays ``DotProductAttention``: the attention kernel is
        installed by rebinding inside ``ops/attention``, not by the spec. The spec answers structure
        only.
        """

        # Explicit rather than inherited: with no TransformerEngine there is no fused
        # layernorm+linear module, so every layer keeps its own `input_layernorm` /
        # `pre_mlp_layernorm`. Megatron's hybrid builder reads exactly these two to decide whether to
        # emit `IdentityOp` in those slots.
        def fuse_layernorm_and_linear(self) -> bool:
            return False

        def column_parallel_layer_norm_linear(self):
            return None

    return IsoExecSpecProvider


def _gdn_module_spec_no_te(config, backend=None):
    """``get_gated_delta_net_module_spec``'s no-TE answer.

    Identical to megatron's except that ``in_proj`` is a plain ``ColumnParallelLinear`` and
    ``fuse_input_layernorm`` is False, so the assembly loop emits a real ``input_layernorm`` rather
    than ``IdentityOp``.
    """
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet, GatedDeltaNetSubmodules
    from megatron.core.transformer.spec_utils import ModuleSpec

    if backend is None:
        backend = make_provider()()
    rms_norm = config.normalization == "RMSNorm"
    return ModuleSpec(
        module=GatedDeltaNet,
        submodules=GatedDeltaNetSubmodules(
            in_proj=backend.column_parallel_linear(),
            out_norm=backend.layer_norm(rms_norm=rms_norm, for_qk=False),
            out_proj=backend.row_parallel_linear(),
        ),
        metainfo={"fuse_input_layernorm": False},
    )


def _self_attention_module_spec_no_te(config, backend=None):
    """``_get_self_attention_module_spec``'s no-TE answer: the local layer spec's self-attention.

    Megatron's version calls ``get_gpt_layer_with_transformer_engine_spec`` unconditionally. Using
    the local twin keeps the two layer kinds of a hybrid identical except for the mixer, which is
    what makes the bridge's parameter mapping resolve for both.
    """
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

    layer_spec = get_gpt_layer_local_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        multi_latent_attention=config.multi_latent_attention,
        normalization=config.normalization,
    )
    attention = layer_spec.submodules.self_attention
    # The assembly loop reads this to decide `input_layernorm` vs `IdentityOp`. Locally nothing is
    # fused, so it must be False -- and it must be SET, because the loop indexes metainfo directly.
    attention.metainfo = dict(getattr(attention, "metainfo", None) or {})
    attention.metainfo["fuse_input_layernorm"] = False
    return attention


@contextlib.contextmanager
def _no_te_builders():
    """Substitute the two TE-hardcoded builders for the duration of one spec build.

    Scoped and restored in ``finally`` rather than patched process-globally, so the blast radius is
    a single call.
    """
    from megatron.core.models.gpt import (
        experimental_attention_variant_module_specs as ev,
    )

    saved = (ev.get_gated_delta_net_module_spec, ev._get_self_attention_module_spec)
    ev.get_gated_delta_net_module_spec = _gdn_module_spec_no_te
    ev._get_self_attention_module_spec = _self_attention_module_spec_no_te
    try:
        yield
    finally:
        ev.get_gated_delta_net_module_spec, ev._get_self_attention_module_spec = saved


def build_decoder_block_spec(config, *, vp_stage=None, pp_rank=None):
    """Build the decoder block for any shape megatron's config can express.

    Dense / GQA / MLA / MoE / per-layer-dense-or-sparse go through
    ``get_gpt_decoder_block_spec(use_transformer_engine=False)``; hybrids (GatedDeltaNet) go through
    megatron's experimental assembly loop with this backend and the two no-TE builders above.
    """
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_transformer_layer_with_experimental_attention_variant_spec,
        is_linear_attention_variant,
    )
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
    from megatron.core.transformer.transformer_block import (
        TransformerBlockSubmodules,
        get_num_layers_to_build,
    )
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    variant = getattr(config, "experimental_attention_variant", None)
    if not is_linear_attention_variant(variant) and variant is None:
        # Non-hybrid: megatron's own builder covers MLA, MoE, per-layer moe_layer_freq and PP.
        spec = get_gpt_decoder_block_spec(config, use_transformer_engine=False, vp_stage=vp_stage, pp_rank=pp_rank)
        logger.info(
            "[ISOEXEC-SPEC] generic block spec: %d layer(s) via get_gpt_decoder_block_spec "
            "(mla=%s, moe_experts=%s, no TransformerEngine)",
            len(spec.layer_specs),
            getattr(config, "multi_latent_attention", False),
            getattr(config, "num_moe_experts", None),
        )
        return spec

    backend = make_provider()()
    with _no_te_builders():
        layer_specs = get_transformer_layer_with_experimental_attention_variant_spec(config=config, backend=backend)

    # Slice for this pipeline stage exactly as megatron's block wrapper does, reusing megatron's own
    # offsets so PP>1 works. PP=1 short-circuits BEFORE touching the offset helpers on purpose:
    # `get_num_layers_to_build` / `get_transformer_layer_offset` read megatron's global
    # `parallel_state`, which is not initialized everywhere this builder runs (the engine builds its
    # GPTModel inside a vLLM worker whose megatron state covers TP only, and the offline spec tests
    # have no distributed state at all). At PP=1 the answer is unconditionally "all layers, offset 0".
    if (
        getattr(config, "pipeline_model_parallel_layout", None) is None
        and int(getattr(config, "pipeline_model_parallel_size", 1) or 1) == 1
    ):
        local_ids = range(len(layer_specs))
    elif getattr(config, "pipeline_model_parallel_layout", None) is not None:
        from megatron.core.transformer.enums import LayerType

        local_ids = config.pipeline_model_parallel_layout.get_layer_id_list(
            layer_type=LayerType.decoder, vp_stage=vp_stage, pp_rank=pp_rank
        )
    else:
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage, pp_rank=pp_rank)
        local_ids = range(offset, offset + get_num_layers_to_build(config, vp_stage=vp_stage, pp_rank=pp_rank))
    layer_specs = [layer_specs[i] for i in local_ids]

    rms_norm = config.normalization == "RMSNorm"
    logger.info(
        "[ISOEXEC-SPEC] generic hybrid block spec: %d local layer(s) of %d, variant=%s, "
        "moe_experts=%s (no TransformerEngine)",
        len(layer_specs),
        config.num_layers,
        variant,
        getattr(config, "num_moe_experts", None),
    )
    return TransformerBlockSubmodules(
        layer_specs=layer_specs, layer_norm=backend.layer_norm(rms_norm=rms_norm, for_qk=False)
    )


def describe_spec(spec) -> list:
    """A comparable, printable description of a block spec: per layer, what fills every slot.

    Classes and their bound configuration, not instances. ``functools.partial`` is unwrapped rather
    than stringified because megatron fills the ``mlp`` slot with ``partial(MoELayer, ...)`` /
    ``partial(MLP, ...)``, and a description that stopped at "partial" would render a dense layer and
    an MoE layer identically.
    """
    import functools

    def describe(x, depth=0):
        if x is None or depth > 6:
            return None
        if isinstance(x, functools.partial):
            return {
                "partial": describe(x.func, depth + 1),
                "keywords": {k: describe(v, depth + 1) for k, v in sorted(x.keywords.items())},
            }
        if isinstance(x, type):
            return x.__name__
        mod = getattr(x, "module", None)
        if mod is not None:  # a ModuleSpec
            return {"module": describe(mod, depth + 1), "submodules": _describe_submodules(x, depth + 1)}
        subs = getattr(x, "submodules", None)
        if subs is not None or hasattr(x, "__dataclass_fields__"):  # a *Submodules dataclass
            return {k: describe(v, depth + 1) for k, v in sorted(vars(x).items())}
        return getattr(x, "__name__", type(x).__name__)

    def _describe_submodules(ms, depth):
        subs = getattr(ms, "submodules", None)
        if subs is None:
            return None
        return {k: describe(v, depth) for k, v in sorted(vars(subs).items())}

    return [describe(ls) for ls in spec.layer_specs]


def assert_structurally_equivalent(spec_a, spec_b, *, label_a="generic", label_b="shipped") -> None:
    """Assert two block specs name the same classes in the same slots, layer by layer.

    Structural equivalence only -- it says nothing about whether the built models' forwards are
    bitwise identical.
    """
    da, db = describe_spec(spec_a), describe_spec(spec_b)
    if len(da) != len(db):
        raise AssertionError(f"layer count differs: {label_a}={len(da)} {label_b}={len(db)}")
    for i, (x, y) in enumerate(zip(da, db)):
        if x != y:
            raise AssertionError(f"layer {i} differs:\n  {label_a}={x}\n  {label_b}={y}")
