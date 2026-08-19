"""Machine-checkable inventory of the vLLM symbols this adapter patches, vendors, reads or registers.

These are all internal vLLM surfaces that can move between versions, usually silently.
``check_vllm_compat`` runs at import time, defaults to WARN, and raises only under
``SKYRL_ISOEXEC_COMPAT_STRICT=1``.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

VALIDATED_VLLM_VERSION = "0.22.1rc1.dev436+g6f573f486.d20260621"

VLLM_PROVENANCE = "docstrings additionally cite 0.20.2 (batch_invariant/gpu_worker) and 0.23 (fused_moe config)"

PATCH = "patch"
VENDOR = "vendor"
READ = "read"
REGISTER = "register"


@dataclass(frozen=True)
class VLLMSymbol:
    """One vLLM symbol the adapter depends on, and how (``kind``: patch / vendor / read / register).

    ``accepts`` and ``fields`` drive light signature and attribute checks; ``optional`` means an
    absent symbol is tolerated because the call site is already guarded.
    """

    module: str
    attr: str = ""
    kind: str = VENDOR
    accepts: Tuple[str, ...] = ()
    fields: Tuple[str, ...] = ()
    optional: bool = False
    since: str = VALIDATED_VLLM_VERSION
    used_by: str = ""
    note: str = ""

    @property
    def fqname(self) -> str:
        return f"{self.module}:{self.attr}" if self.attr else self.module


VLLM_SYMBOLS: List[VLLMSymbol] = [
    VLLMSymbol(
        "vllm.model_executor.layers.batch_invariant",
        "override_envs_for_invariance",
        PATCH,
        used_by="vllm_patches.neutralize_vllm_nccl_channel_pin",
        note="wrapped to drop the NCCL channel pin it re-sets (SKYRL_ISOEXEC_ENGINE_NCCL_UNPIN=1)",
    ),
    VLLMSymbol(
        "vllm.v1.attention.backends.flash_attn",
        "flash_attn_varlen_func",
        PATCH,
        accepts=("num_splits",),
        used_by="vllm_patches.apply_flash_num_splits_patch",
        note="wrapped to inject num_splits=1 on the main decode path under VLLM_BATCH_INVARIANT",
    ),
    VLLMSymbol(
        "vllm.v1.worker.gpu.sample.logprob",
        "compute_token_logprobs",
        PATCH,
        used_by="vllm_patches.patch_vllm_logprobs_batch_invariant",
        note="replaced with an aten log_softmax formulation matching the trainer bitwise",
    ),
    VLLMSymbol(
        "vllm.v1.attention.backends.gdn_attn",
        "GDNAttentionBackend",
        PATCH,
        fields=("supports_batch_invariance",),
        used_by="gdn_engine_patch.lift_gdn_batch_invariance_veto",
        note="supports_batch_invariance rebound to a classmethod returning True (lifts the GDN veto)",
    ),
    VLLMSymbol(
        "vllm.v1.attention.backends.gdn_attn",
        "GDNAttentionMetadata",
        READ,
        fields=(
            "num_decodes",
            "num_prefills",
            "num_decode_tokens",
            "num_actual_tokens",
            "non_spec_state_indices_tensor",
            "prefill_state_indices",
            "prefill_query_start_loc",
            "prefill_has_initial_state",
            "spec_sequence_masks",
        ),
        used_by="gdn_engine_patch.gdn_metadata / _step_cpu / *_core",
        note="the per-forward metadata whose field names the GDN plumbing reads by hand",
    ),
    VLLMSymbol(
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
        "QwenGatedDeltaNetAttention",
        PATCH,
        fields=("_forward_core",),
        used_by="gdn_engine_patch.install_gdn_engine_patch",
        note="_forward_core + __init__ rebound; enable_packed_recurrent_decode set False (write target)",
    ),
    VLLMSymbol(
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
        "QwenGatedDeltaNetAttention.__init__",
        PATCH,
        accepts=("config", "vllm_config", "prefix", "gqa_interleaved_layout"),
        used_by="gdn_engine_patch.install_gdn_engine_patch",
        note="our _init wrapper forwards exactly these positional args to the original __init__",
    ),
    VLLMSymbol(
        "vllm.model_executor.layers.fla.ops.utils",
        "FLA_CHUNK_SIZE",
        VENDOR,
        used_by="gdn_engine_patch / gdn_ops / gdn_gptmodel / gdn_batch_invariant",
        note="the chunk width the chunk-consistent decode grid is built on",
    ),
    VLLMSymbol(
        "vllm.model_executor.layers.mamba.mamba_utils",
        "is_conv_state_dim_first",
        VENDOR,
        used_by="gdn_engine_patch._get_layer_state (native state orientation)",
    ),
    VLLMSymbol(
        "vllm.forward_context",
        "get_forward_context",
        READ,
        used_by="gdn_engine_patch.gdn_metadata / _step_cpu",
    ),
    VLLMSymbol(
        "vllm.model_executor.layers.mamba.abstract",
        "MambaBase",
        VENDOR,
        used_by="gdn_gptmodel.IsoExecGDNStateLayer (isinstance-visible KV-cache layer)",
        note="subclassed so vLLM's get_layers_from_vllm_config isinstance filter sees the layer",
    ),
    VLLMSymbol(
        "vllm.model_executor.layers.mamba.mamba_utils",
        "MambaStateShapeCalculator",
        VENDOR,
        fields=("gated_delta_net_state_shape",),
        used_by="gptmodel_vllm/gdn_gptmodel.get_mamba_state_shape_from_config",
    ),
    VLLMSymbol(
        "vllm.model_executor.layers.mamba.mamba_utils",
        "MambaStateDtypeCalculator",
        VENDOR,
        fields=("gated_delta_net_state_dtype",),
        used_by="gptmodel_vllm/gdn_gptmodel.get_mamba_state_dtype_from_config",
    ),
    VLLMSymbol(
        "vllm.v1.attention.backends.registry",
        "MambaAttentionBackendEnum",
        READ,
        used_by="gdn_gptmodel (mamba backend enum)",
    ),
    VLLMSymbol(
        "vllm.model_executor.models.registry",
        "ModelRegistry",
        REGISTER,
        fields=("register_model",),
        used_by="gptmodel_vllm.register_gptmodel_to_vllm",
        note="the GPTModel-backed wrapper is registered under VLLM_MODEL_NAME",
    ),
    VLLMSymbol(
        "vllm.model_executor.models.config",
        "MODELS_CONFIG_MAP",
        PATCH,
        used_by="gptmodel_vllm.register_gptmodel_to_vllm (hybrid GDN config pass)",
        note="setdefault our arch -> HybridAttentionMambaModelConfig",
    ),
    VLLMSymbol(
        "vllm.model_executor.models.config",
        "HybridAttentionMambaModelConfig",
        VENDOR,
        used_by="gptmodel_vllm.register_gptmodel_to_vllm",
        note="sets cache_config.mamba_block_size; selected by architecture name",
    ),
    VLLMSymbol(
        "vllm.config",
        "get_current_vllm_config",
        READ,
        used_by="gptmodel_vllm (capture max_num_seqs at build time)",
    ),
    VLLMSymbol(
        "vllm.model_executor.layers.attention",
        "Attention",
        READ,
        used_by="gptmodel_vllm (attention layer construction)",
    ),
    VLLMSymbol(
        "vllm.model_executor.layers.attention.attention",
        "get_attention_context",
        READ,
        used_by="varlen_backend",
    ),
    VLLMSymbol("vllm.v1.attention.backend", "AttentionCGSupport", READ, used_by="varlen_backend"),
    VLLMSymbol("vllm.v1.attention.backend", "AttentionType", READ, used_by="varlen_backend"),
    VLLMSymbol(
        "vllm.v1.attention.backends.flash_attn",
        "FlashAttentionBackend",
        VENDOR,
        used_by="varlen_backend.PyTorchVarlenAttentionBackend (subclass)",
    ),
    VLLMSymbol(
        "vllm.v1.attention.backends.flash_attn",
        "FlashAttentionImpl",
        VENDOR,
        used_by="varlen_backend.PyTorchVarlenAttentionImpl (subclass)",
    ),
    VLLMSymbol(
        "vllm.v1.attention.backends.flash_attn",
        "FlashAttentionMetadata",
        VENDOR,
        used_by="varlen_backend",
    ),
    VLLMSymbol(
        "vllm.v1.attention.backends.flash_attn",
        "FlashAttentionMetadataBuilder",
        VENDOR,
        used_by="varlen_backend.PyTorchVarlenAttentionMetadataBuilder (subclass)",
    ),
    VLLMSymbol(
        "vllm.v1.attention.backends.registry",
        "AttentionBackendEnum",
        REGISTER,
        fields=("CUSTOM",),
        used_by="varlen_backend.register_varlen_custom_backend",
        note="@register_backend(AttentionBackendEnum.CUSTOM) -- attention_backend='CUSTOM'",
    ),
    VLLMSymbol(
        "vllm.v1.attention.backends.registry",
        "register_backend",
        REGISTER,
        used_by="varlen_backend.register_varlen_custom_backend",
    ),
    VLLMSymbol(
        "vllm.compilation.breakable_cudagraph",
        "eager_break_during_capture",
        VENDOR,
        optional=True,
        used_by="varlen_backend (already try/except-guarded; enforce_eager path skips it)",
    ),
    VLLMSymbol(
        "vllm.vllm_flash_attn",
        "flash_attn_varlen_func",
        READ,
        optional=True,
        used_by="varlen_backend vllm_flash_ns1 fast path (already try/except-guarded; "
        "torch varlen_attn_out fallback is bitwise-interchangeable)",
    ),
    VLLMSymbol(
        "vllm.model_executor.layers.fla.ops.fused_recurrent",
        "fused_recurrent_gated_delta_rule",
        VENDOR,
        accepts=("q", "k", "v", "g", "beta", "scale", "initial_state"),
        used_by="gdn_ops (recurrent decode/prefill)",
    ),
    VLLMSymbol(
        "vllm.model_executor.layers.fla.ops.fused_sigmoid_gating",
        "fused_sigmoid_gating_delta_rule_update",
        VENDOR,
        accepts=("A_log", "a", "b", "dt_bias", "q", "k", "v"),
        used_by="gdn_ops (native fused gating update)",
    ),
    VLLMSymbol(
        "vllm.model_executor.layers.mamba.ops.causal_conv1d",
        "causal_conv1d_fn",
        VENDOR,
        used_by="gdn_ops / gdn_recurrent_state (varlen conv prefill)",
    ),
    VLLMSymbol(
        "vllm.model_executor.layers.mamba.ops.causal_conv1d",
        "causal_conv1d_update",
        VENDOR,
        used_by="gdn_recurrent_state (native decode conv window slide)",
    ),
    VLLMSymbol(
        "vllm.model_executor.layers.fla.ops.chunk",
        "chunk_gated_delta_rule",
        VENDOR,
        used_by="gdn_ops / gdn_batch_invariant (chunked-parallel training kernel)",
    ),
    VLLMSymbol(
        "vllm.model_executor.layers.fla.ops.index",
        "prepare_chunk_indices",
        VENDOR,
        used_by="gdn_ops / gdn_batch_invariant",
    ),
    VLLMSymbol(
        "vllm.model_executor.layers.fla.ops.index",
        "prepare_chunk_offsets",
        VENDOR,
        used_by="gdn_ops / gdn_batch_invariant",
    ),
    VLLMSymbol(
        "vllm.model_executor.layers.fla.ops.l2norm",
        "l2norm_fwd",
        VENDOR,
        used_by="gdn_ops / gdn_batch_invariant",
    ),
    VLLMSymbol(
        "vllm.model_executor.layers.fla.ops.layernorm_guard",
        "calc_rows_per_block",
        PATCH,
        used_by="gdn_batch_invariant.pin_gdn_rmsnorm_rows_per_block",
        note="rebound to a constant so RMSNormGated's fp32 tile is M-invariant (decode==prefill)",
    ),
    VLLMSymbol(
        "vllm.envs",
        "VLLM_BATCH_INVARIANT",
        READ,
        used_by="moe_fused_experts._bi_config (guard)",
    ),
    VLLMSymbol(
        "vllm.model_executor.layers.fused_moe.fused_moe",
        "get_default_config",
        VENDOR,
        accepts=("M", "E", "N", "K", "topk", "dtype"),
        used_by="moe_fused_experts._bi_config",
    ),
    VLLMSymbol(
        "vllm.model_executor.layers.fused_moe.fused_moe",
        "invoke_fused_moe_triton_kernel",
        VENDOR,
        used_by="moe_fused_experts (leaf-tree fc2 kernel, vendored from vLLM 0.23 fused_moe_kernel)",
        note="config block sizes are also vendored as constants in moe_preamble_o12 (configs[float32])",
    ),
    VLLMSymbol(
        "vllm.model_executor.layers.batch_invariant",
        "mm_batch_invariant",
        PATCH,
        used_by="moe_batch_invariant._install_moe_matmul_invariance",
        note="registered as aten::mm CUDA impl globally (load-bearing; see moe_batch_invariant)",
    ),
    VLLMSymbol(
        "vllm.model_executor.layers.batch_invariant",
        "addmm_batch_invariant",
        PATCH,
        used_by="moe_batch_invariant._install_moe_matmul_invariance",
        note="registered as aten::addmm CUDA impl globally",
    ),
    VLLMSymbol(
        "vllm.model_executor.layers.batch_invariant",
        "matmul_persistent",
        PATCH,
        used_by="mm_tiles.install_mm_tiles",
        note="wrapped with a re-tiled variant for skinny decode GEMMs (bitwise-neutral)",
    ),
    VLLMSymbol(
        "vllm.model_executor.layers.batch_invariant",
        "num_compute_units",
        VENDOR,
        used_by="mm_tiles / moe_preamble_o12",
    ),
    VLLMSymbol(
        "vllm.model_executor.layers.batch_invariant",
        "_compute_pid",
        VENDOR,
        used_by="moe_preamble_o12 (leaf-tree GEMM pid mapping)",
    ),
    VLLMSymbol(
        "vllm.platforms",
        "current_platform",
        READ,
        fields=("is_device_capability_family",),
        used_by="moe_batch_invariant (SM80 skip)",
    ),
    VLLMSymbol(
        "vllm.device_allocator.cumem",
        "CuMemAllocator",
        VENDOR,
        fields=("instance", "use_memory_pool"),
        used_by="moe_fused_weights (sleepable-pool alloc; already try/except-guarded)",
        optional=True,
    ),
    VLLMSymbol(
        "vllm.distributed",
        "split_tensor_along_last_dim",
        READ,
        used_by="ops/collectives/pik/integrations/vllm_patch",
        optional=True,
        note="imported inside the pik integration (SKYRL_ISOEXEC_PIK path)",
    ),
]


def _resolve(sym: VLLMSymbol):
    try:
        mod = importlib.import_module(sym.module)
    except Exception as e:
        return None, f"module import failed: {type(e).__name__}: {e}"
    if not sym.attr:
        return mod, None
    obj = mod
    walked = sym.module
    for part in sym.attr.split("."):
        walked = f"{walked}.{part}"
        if not hasattr(obj, part):
            return None, f"attribute '{part}' missing on {walked.rsplit('.', 1)[0]}"
        obj = getattr(obj, part)
    return obj, None


def _check_accepts(obj, accepts: Tuple[str, ...]) -> Optional[str]:
    if not accepts:
        return None
    try:
        params = inspect.signature(obj).parameters
    except (TypeError, ValueError):
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return None
    missing = [a for a in accepts if a not in params]
    if missing:
        return f"no longer accepts kwargs {missing} (has {sorted(params)})"
    return None


def _check_fields(obj, fields: Tuple[str, ...]) -> Optional[str]:
    if not fields:
        return None
    df = getattr(obj, "__dataclass_fields__", {})
    ann = getattr(obj, "__annotations__", {})
    missing = [f for f in fields if not (hasattr(obj, f) or f in df or f in ann)]
    if missing:
        return f"missing attribute(s)/field(s) {missing}"
    return None


def _check_symbol(sym: VLLMSymbol) -> Optional[str]:
    try:
        obj, problem = _resolve(sym)
        if problem is not None:
            if sym.optional:
                return None
            return f"{sym.fqname} [{sym.kind}]: {problem}  (used by {sym.used_by})"
        for check in (_check_accepts(obj, sym.accepts), _check_fields(obj, sym.fields)):
            if check is not None:
                return f"{sym.fqname} [{sym.kind}]: {check}  (used by {sym.used_by})"
        return None
    except Exception as e:
        return f"{sym.fqname} [{sym.kind}]: CHECK-ERROR {type(e).__name__}: {e}"


def check_vllm_compat(strict: bool = False) -> List[str]:
    problems: List[str] = []
    for sym in VLLM_SYMBOLS:
        problem = _check_symbol(sym)
        if problem:
            problems.append(problem)

    effective_strict = strict or os.environ.get("SKYRL_ISOEXEC_COMPAT_STRICT") == "1"
    n = len(VLLM_SYMBOLS)
    if not problems:
        logger.info("[ISOEXEC-COMPAT] vLLM surface OK (%d symbols, validated vs %s)", n, VALIDATED_VLLM_VERSION)
    else:
        header = (
            f"[ISOEXEC-COMPAT] vLLM surface has {len(problems)}/{n} problem(s) "
            f"(validated vs {VALIDATED_VLLM_VERSION}; running vllm=%s):"
        )
        try:
            import vllm

            running = getattr(vllm, "__version__", "unknown")
        except Exception:
            running = "unknown"
        logger.warning(header, running)
        for p in problems:
            logger.warning("[ISOEXEC-COMPAT]   - %s", p)
        if effective_strict:
            raise RuntimeError(
                f"[ISOEXEC-COMPAT] {len(problems)} vLLM symbol(s) changed under the adapter "
                f"(SKYRL_ISOEXEC_COMPAT_STRICT=1):\n  " + "\n  ".join(problems)
            )
    return problems
