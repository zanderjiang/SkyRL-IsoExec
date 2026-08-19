"""vLLM attention backend registered as ``CUSTOM`` that runs varlen FlashAttention at
``num_splits=1``, making the paged decode path query-length invariant and bitwise-equal to
full-sequence prefill. Adapted from TorchTitan's torchtitan/experiments/rl/models/attention.py.

Installation requires importing this module before the vLLM engine is constructed (registration is
an import side effect), passing ``attention_backend="CUSTOM"``, and running the engine in-process
so the registration is visible. Requires torch>=2.14 and FA3 (SM 9.0+); on older stacks this module
is not importable and is never imported.
"""

import logging
import os
from typing import Any

import torch
import torch.nn.attention.varlen  # noqa: F401 -> force-load the lazy varlen submodule (varlen_attn_out)
from torch.nn.attention import (
    activate_flash_attention_impl,
    current_flash_attention_impl,
)
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

try:
    from vllm.compilation.breakable_cudagraph import eager_break_during_capture
except Exception:  # pragma: no cover - enforce_eager path doesn't need it

    def eager_break_during_capture(fn):
        return fn


logger = logging.getLogger(__name__)

# Force num_splits=1 so the varlen kernel is query-length-invariant (bitwise decode==prefill).
_FORCE_NUM_SPLITS_1 = os.environ.get("VARLEN_FORCE_NUM_SPLITS_1", "1") == "1"

# vLLM's flash_attn_varlen_func at num_splits=1/fa_version=3 is bitwise-identical to torch
# varlen_attn_out(num_splits=1) and faster. Engine-side only; the trainer keeps torch varlen for
# autograd, so trainer==engine is preserved. Pure perf knob -- OFF falls back to the torch kernel.
_USE_VLLM_FLASH_NS1 = os.environ.get("SKYRL_ISOEXEC_VARLEN_VLLM_FLASH", "1") == "1"
try:
    from vllm.vllm_flash_attn import (
        flash_attn_varlen_func as _vllm_flash_attn_varlen_func,
    )
except Exception:  # pragma: no cover - torch fallback stays available
    _vllm_flash_attn_varlen_func = None


def _has_sm90() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


@register_backend(AttentionBackendEnum.CUSTOM)
class PyTorchVarlenAttentionBackend(FlashAttentionBackend):
    @staticmethod
    def get_name():
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return PyTorchVarlenAttentionImpl

    @staticmethod
    def get_builder_cls():
        class PyTorchVarlenAttentionMetadataBuilder(FlashAttentionMetadataBuilder):
            _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

        return PyTorchVarlenAttentionMetadataBuilder


class PyTorchVarlenAttentionImpl(FlashAttentionImpl):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.out_transform = None  # no epilogue
        self.enable_gqa = self.num_heads > self.num_kv_heads
        if _has_sm90():
            if current_flash_attention_impl() != "FA3":
                activate_flash_attention_impl("FA3")
        else:
            logger.warning("FA3 not available (requires SM 9.0+), falling back to FA2.")

    @eager_break_during_capture
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        assert self.vllm_flash_attn_version is not None, "FlashAttention version not detected."
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not yet supported for FlashAttentionImpl")

        if not getattr(PyTorchVarlenAttentionImpl, "_FWD_LOGGED", False):
            PyTorchVarlenAttentionImpl._FWD_LOGGED = True
            _vfa_live = (
                _USE_VLLM_FLASH_NS1
                and _vllm_flash_attn_varlen_func is not None
                and current_flash_attention_impl() == "FA3"
                and _FORCE_NUM_SPLITS_1
            )
            _kernel_name = (
                "vllm.vllm_flash_attn.flash_attn_varlen_func[fa3]"
                if _vfa_live
                else "torch.nn.attention.varlen.varlen_attn_out"
            )
            _msg = (
                "[isoexec-varlen] PyTorchVarlenAttentionImpl.forward IS EXECUTING "
                f"({_kernel_name}, num_splits="
                f"{1 if _FORCE_NUM_SPLITS_1 else 'auto'})"
            )
            print(_msg, flush=True)

        # Re-read live per-layer metadata + kv_cache from forward context
        attn_metadata, _, kv_cache, _ = get_attention_context(layer.layer_name)

        if attn_metadata is None:
            return output.fill_(0)

        attn_type = self.attn_type
        num_actual_tokens = attn_metadata.num_actual_tokens

        assert attn_type not in (
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER,
        ), "Encoder-only attention not supported yet."

        key_cache, value_cache = kv_cache.unbind(1)

        assert not self.kv_cache_dtype.startswith("fp8"), "FP8 KV cache not supported."
        assert not attn_metadata.use_cascade, "Cascade not supported yet."

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        assert self.dcp_world_size == 1, "DCP not supported yet."

        if not attn_metadata.causal:
            raise RuntimeError("Non-causal attention not supported yet.")

        if self.sliding_window == (-1, -1):
            sliding_window_size = (-1, 0)
        else:
            sliding_window_size = self.sliding_window

        assert self.alibi_slopes is None, "Alibi slopes not supported yet."

        if current_flash_attention_impl() == "FA3":
            cu_seqlens_k = None
        else:
            num_seqs = seqused_k.shape[0]
            cu_seqlens_k = torch.zeros(num_seqs + 1, dtype=torch.int32, device=query.device)
            cu_seqlens_k[1:] = torch.cumsum(seqused_k, dim=0)

        extra_kwargs: dict[str, Any] = {}
        fa_impl = current_flash_attention_impl()
        # Also forced for FA2/None regardless of the flag: a NaN workaround.
        force_ns1 = fa_impl in (None, "FA2") or _FORCE_NUM_SPLITS_1
        if force_ns1:
            extra_kwargs["num_splits"] = 1

        # Fast path, taken only under FA3 + forced ns1 -- the configuration the bitwise proof covers.
        if _USE_VLLM_FLASH_NS1 and _vllm_flash_attn_varlen_func is not None and fa_impl == "FA3" and force_ns1:
            return _vllm_flash_attn_varlen_func(
                query[:num_actual_tokens],
                key_cache,
                value_cache,
                max_seqlen_q,
                cu_seqlens_q,
                max_seqlen_k,
                cu_seqlens_k,
                seqused_k=seqused_k,
                causal=True,
                softmax_scale=self.scale,
                window_size=list(sliding_window_size),
                block_table=block_table,
                out=output[:num_actual_tokens],
                num_splits=1,
                fa_version=3,
            )

        if self.enable_gqa:
            extra_kwargs["enable_gqa"] = True

        result = torch.nn.attention.varlen.varlen_attn_out(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            scale=self.scale,
            window_size=sliding_window_size,
            block_table=block_table,
            seqused_k=seqused_k,
            **extra_kwargs,
        )
        return result


def register_varlen_custom_backend() -> bool:
    """Report whether torch's varlen FlashAttention kernel exists; registration is an import side effect."""
    return hasattr(torch.nn.attention.varlen, "varlen_attn_out")
