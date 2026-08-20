"""Adapt Megatron ``GatedDeltaNet`` layers for execution inside vLLM.

``IsoExecGDNStateLayer`` registers each GDN layer with vLLM so the scheduler supplies Mamba metadata
and slot identities, while ``swap_gdn_core`` keeps Megatron's projections, weights, normalization and
output modules and routes only the stateful core through the IsoExec implementation. In chunk-synced
mode the vLLM Mamba tensors are minimal slot-name pages and the real state lives in
``ChunkSyncedGDN``; native-state recurrent mode binds the vLLM tensors directly. Engine-only, enabled
by ``SKYRL_ISOEXEC_GDN=1``.
"""

from __future__ import annotations

import logging
import os

import torch
from torch import nn

logger = logging.getLogger(__name__)


def _is_gdn(module) -> bool:
    return type(module).__name__ == "GatedDeltaNet"


_state_cls = None


def state_layer_cls():
    """Build ``IsoExecGDNStateLayer`` on first use.

    It must genuinely subclass ``MambaBase``: vLLM enumerates KV-cache layers with an ``isinstance``
    filter, so a duck-typed layer is invisible and the engine runs with an empty
    ``GDNAttentionMetadata``. Built lazily so this module stays importable in the trainer process,
    which has no vLLM.
    """
    global _state_cls

    if _state_cls is not None:
        return _state_cls

    from vllm.model_executor.layers.mamba.abstract import MambaBase
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateDtypeCalculator,
        MambaStateShapeCalculator,
    )
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

    class IsoExecGDNStateLayer(nn.Module, MambaBase):
        """vLLM-visible GDN state layer: owns the GDN core, borrows Megatron's weights.

        In chunk mode the ``kv_cache`` tensors vLLM allocates are unused -- what is needed is the
        bookkeeping the allocation buys: a per-request slot id and the prefill/decode split, delivered
        as ``GDNAttentionMetadata``. Recurrent mode can use the blocks directly
        (``SKYRL_ISOEXEC_GDN_NATIVE_STATE=1``); see ``gdn_recurrent_state.RecurrentGDN``.
        """

        def __init__(self, *, vllm_config, prefix: str, gdn):
            super().__init__()
            self.prefix = prefix
            # A plain list, not an attribute: assigning `gdn` directly would register Megatron's
            # GatedDeltaNet as a submodule of this layer and duplicate every parameter.
            self._gdn = [gdn]

            self.model_config = vllm_config.model_config
            self.cache_config = vllm_config.cache_config
            self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
            if vllm_config.speculative_config:
                raise ValueError("[isoexec-gdn] speculative decoding is incompatible with chunk-consistent decode")

            # Megatron shards GDN heads over ITS tensor-parallel group; vLLM must agree, which is why
            # the IsoExec recipe pins Megatron TP == inference TP.
            self.tp_size = gdn.tp_size
            self.num_k_heads = gdn.num_key_heads
            self.num_v_heads = gdn.num_value_heads
            self.head_k_dim = gdn.key_head_dim
            self.head_v_dim = gdn.value_head_dim
            self.conv_kernel_size = gdn.conv_kernel_dim

            self.kv_cache = (torch.tensor([]), torch.tensor([]))
            self._cc = None
            # Build the chunk-synced core EAGERLY at model-build time: a lazy first build would land
            # on the CUDA-graph capture run, and the sleepable-pool allocation aborts the worker if it
            # happens under stream capture. Weights are re-read every call anyway.
            from ...ops.gdn.gdn_ops import chunk_synced_mode as _cs_mode

            if _cs_mode():
                from vllm.model_executor.layers.fla.ops.utils import (
                    FLA_CHUNK_SIZE as _C,
                )

                from ...ops.gdn.gdn_chunk_synced_state import (
                    build_chunk_synced_gdn as _build_cs,
                )

                _cw = gdn.conv1d.weight.squeeze(1)
                self._cc = _build_cs(
                    max_num_seqs=self.max_num_seqs,
                    chunk_size=_C,
                    conv_weight=_cw,
                    conv_bias=gdn.conv1d.bias if gdn.conv_bias else None,
                    A_log=gdn.A_log,
                    dt_bias=gdn.dt_bias,
                    num_k_heads=self.num_k_heads // self.tp_size,
                    head_k_dim=self.head_k_dim,
                    num_v_heads=self.num_v_heads // self.tp_size,
                    head_v_dim=self.head_v_dim,
                    activation=gdn.activation,
                    dtype=_cw.dtype,
                    device=_cw.device,
                )
                self._cc.lazy_resync = True
                logger.info(
                    "[isoexec-gdn] %s: chunk-synced core built EAGERLY (pre-capture), capacity=%d",
                    prefix,
                    self.max_num_seqs,
                )
            # (data_ptr, shape) of the ssm tensor `_cc` was built against. vLLM binds kv_cache more
            # than once on the CUDA-graph path -- a throwaway profiling cache, then the real one -- so
            # `_state` compares this key per forward and rebuilds against the real tensors.
            self._native_kv_key = None
            self._min_profiling_blocks = vllm_config.compilation_config.max_cudagraph_capture_size or 1

            ctx = vllm_config.compilation_config.static_forward_context
            if prefix in ctx:
                raise ValueError(f"Duplicate layer name: {prefix}")
            ctx[prefix] = self

        @property
        def mamba_type(self):
            return MambaAttentionBackendEnum.GDN_ATTN

        def get_state_shape(self):
            # Chunk mode keeps all state in ChunkSyncedGDN's private pools and uses the vLLM pages
            # only as a slot-id source, so they shrink to bytes. MUST agree with
            # GPTModelVLLMWrapper.get_mamba_state_shape_from_config: vLLM sizes and pads pages from
            # that at config time and carves the runtime tensors from this spec, and a full-size spec
            # over a minimally-padded page fails page unification.
            from ...ops.gdn.gdn_ops import GDN_CS_MIN_STATE_SHAPES, gdn_cs_min_pages

            if gdn_cs_min_pages():
                return GDN_CS_MIN_STATE_SHAPES
            return MambaStateShapeCalculator.gated_delta_net_state_shape(
                self.tp_size,
                self.num_k_heads,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
                self.conv_kernel_size,
                0,
            )

        def get_state_dtype(self):
            return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
                self.model_config.dtype,
                self.cache_config.mamba_cache_dtype,
                self.cache_config.mamba_ssm_cache_dtype,
            )

        def _native_state_tensors(self):
            """vLLM's own mamba ``kv_cache`` (conv, ssm), oriented for RecurrentGDN.

            ``kv_cache`` is bound as ``(conv_state, ssm_state)``. Conv is stored SD
            ``(num_blocks, W-1, D)`` by default or DS ``(num_blocks, D, W-1)``; RecurrentGDN wants DS,
            so transpose the last two dims under SD.
            """
            from vllm.model_executor.layers.mamba.mamba_utils import (
                is_conv_state_dim_first,
            )

            kv = self.kv_cache
            if not (len(kv) > 1 and kv[1].numel()):
                raise RuntimeError(
                    "[isoexec-gdn] SKYRL_ISOEXEC_GDN_NATIVE_STATE=1 but the vLLM mamba kv_cache is not "
                    "bound (empty). The state layer must be built AFTER bind_kv_cache."
                )
            ssm_state = kv[1]
            dim_first = is_conv_state_dim_first()
            conv_state = kv[0] if dim_first else kv[0].transpose(-1, -2)
            cc = self.cache_config
            # On the CUDA-graph path this build runs twice: once against the throwaway profiling
            # cache, then against the real one. The profiling bind's tensors never see a real request,
            # so it is flagged in the log and skipped by the adequacy warning below.
            profiling_bind = (
                ssm_state.shape[0] == self._min_profiling_blocks
                and getattr(cc, "num_gpu_blocks", None) == self._min_profiling_blocks
            )
            # vLLM indexes the state tensor with block_table[:, 0], whose id space is not obviously
            # ssm_state.shape[0] rows; log every number that could set it.
            logger.info(
                "[isoexec-gdn] %s: NATIVE state ssm=%s %s conv raw=%s -> oriented=%s (layout=%s) | "
                "GEOM num_gpu_blocks=%s block_size=%s mamba_block_size=%s page_padded=%s "
                "cache_mode=%s ssm_dtype=%s max_num_seqs=%s profiling_bind=%s",
                self.prefix,
                tuple(ssm_state.shape),
                ssm_state.dtype,
                tuple(kv[0].shape),
                tuple(conv_state.shape),
                "DS" if dim_first else "SD",
                getattr(cc, "num_gpu_blocks", None),
                getattr(cc, "block_size", None),
                getattr(cc, "mamba_block_size", None),
                getattr(cc, "mamba_page_size_padded", None),
                getattr(cc, "mamba_cache_mode", None),
                getattr(cc, "mamba_ssm_cache_dtype", None),
                self.max_num_seqs,
                profiling_bind,
            )
            # Mamba block ids stride by 2, so the id space reaches ~2*max_num_seqs. A state tensor
            # with fewer rows can hand a real request an id past the end, and under CUDA graphs the
            # per-step overflow check host-syncs and cannot run -- the state would corrupt silently.
            # Build time is the only capture-safe place to catch it.
            need = 2 * self.max_num_seqs
            if not profiling_bind and ssm_state.shape[0] < need:
                logger.warning(
                    "[isoexec-gdn] %s: NATIVE pool has %d blocks but ~%d may be needed "
                    "(2x max_num_seqs=%d, mamba block ids stride ~2). Real requests can overflow the "
                    "state tensor -> SILENT corruption under CUDA graphs. Lower max_num_seqs, raise "
                    "gpu_memory_utilization, or reduce CUDA-graph memory.",
                    self.prefix,
                    ssm_state.shape[0],
                    need,
                    self.max_num_seqs,
                )
            return ssm_state, conv_state

        def _state(self):
            from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE

            from ...ops.gdn.gdn_chunk_synced_state import (
                ChunkSyncedGDN,
                build_chunk_synced_gdn,
            )
            from ...ops.gdn.gdn_ops import chunk_synced_mode, recurrent_mode
            from ...ops.gdn.gdn_recurrent_state import (
                build_recurrent_gdn,
                native_state_enabled,
            )

            gdn = self._gdn[0]
            # Re-read every call: native weight sync rebinds `.data`, and a captured tensor would pin
            # the pre-update weights and silently roll back the policy.
            conv_weight = gdn.conv1d.weight.squeeze(1)  # [D, W]
            conv_bias = gdn.conv1d.bias if gdn.conv_bias else None

            # Native state: re-check the kv_cache binding every call. vLLM binds kv_cache twice on
            # the CUDA-graph path, and a core built against the discarded profiling cache walks off
            # the end of the tensor once the scheduler issues ids from the real pool. Rebuilding is
            # cheap and settles at the first forward after the real bind; warmup and capture both run
            # after initialize_kv_cache, so captured graphs pin the real tensors. Host-only reads,
            # safe under capture.
            if self._cc is not None and getattr(self._cc, "_native", False):
                kv = self.kv_cache
                key = (kv[1].data_ptr(), tuple(kv[1].shape)) if len(kv) > 1 and kv[1].numel() else None
                if key != self._native_kv_key:
                    logger.info(
                        "[isoexec-gdn] %s: NATIVE kv_cache rebound (ssm rows %s -> %s), rebuilding state core",
                        self.prefix,
                        self._native_kv_key[1][0] if self._native_kv_key else None,
                        key[1][0] if key else None,
                    )
                    self._cc = None

            if self._cc is None and chunk_synced_mode():
                # Private pool only: the entry states + open-chunk buffers live beside our pool, and
                # they must be sized by the concurrency cap, not by vLLM's slot count.
                self._cc = build_chunk_synced_gdn(
                    max_num_seqs=self.max_num_seqs,
                    chunk_size=FLA_CHUNK_SIZE,
                    conv_weight=conv_weight,
                    conv_bias=conv_bias,
                    A_log=gdn.A_log,
                    dt_bias=gdn.dt_bias,
                    num_k_heads=self.num_k_heads // self.tp_size,
                    head_k_dim=self.head_k_dim,
                    num_v_heads=self.num_v_heads // self.tp_size,
                    head_v_dim=self.head_v_dim,
                    activation=gdn.activation,
                    dtype=conv_weight.dtype,
                    device=conv_weight.device,
                )
                # the engine always runs the lazy driver (installed in swap_gdn_core): decode stays
                # pure device work; boundaries are serviced pre-forward by the metadata-builder wrap.
                self._cc.lazy_resync = True
                logger.info(
                    "[isoexec-gdn] %s: chunk-synced decode+prefill (LAZY resync every %d), capacity=%d",
                    self.prefix,
                    FLA_CHUNK_SIZE,
                    self.max_num_seqs,
                )
            elif self._cc is None and recurrent_mode():
                ssm_state = conv_state = None
                if native_state_enabled():
                    ssm_state, conv_state = self._native_state_tensors()
                    self._native_kv_key = (ssm_state.data_ptr(), tuple(ssm_state.shape))
                self._cc = build_recurrent_gdn(
                    max_num_seqs=self.max_num_seqs,
                    ssm_state=ssm_state,
                    conv_state=conv_state,
                    conv_weight=conv_weight,
                    conv_bias=conv_bias,
                    A_log=gdn.A_log,
                    dt_bias=gdn.dt_bias,
                    num_k_heads=self.num_k_heads // self.tp_size,
                    head_k_dim=self.head_k_dim,
                    num_v_heads=self.num_v_heads // self.tp_size,
                    head_v_dim=self.head_v_dim,
                    activation=gdn.activation,
                    dtype=conv_weight.dtype,
                    device=conv_weight.device,
                )
                logger.info(
                    "[isoexec-gdn] %s: recurrent decode+prefill, capacity=%d, state=%s",
                    self.prefix,
                    self.max_num_seqs,
                    "vLLM-native" if ssm_state is not None else "private-pool",
                )
            # First-forward fingerprint, outside both mode branches so every composition reports.
            # The impl id is read off the object that was actually built, not off the mode flag.
            try:
                from ...core.fingerprint import (
                    ENGINE_SITES,
                    log_fingerprint_once,
                    record_installs,
                )
                from ...core.process_contract import cached_contract_view

                if isinstance(self._cc, ChunkSyncedGDN):
                    # Its own fp32 entry states + open-chunk buffers; vLLM's state pages are a
                    # slot-id source only (and are refused outright as a state tensor).
                    _state_impl = "chunk_synced_pool"
                else:
                    _state_impl = "native_kv_cache" if getattr(self._cc, "_native", False) else "private_pool"
                record_installs("gdn.state", ENGINE_SITES, _state_impl, self._cc)
                log_fingerprint_once(cached_contract_view(), tag="engine_first_forward")
            except Exception as _e:  # pragma: no cover - never fatal
                logger.warning(f"[ISOEXEC-FINGERPRINT] gdn.state record skipped: {_e}")
            self._cc.conv_weight, self._cc.conv_bias = conv_weight, conv_bias
            self._cc.A_log, self._cc.dt_bias = gdn.A_log, gdn.dt_bias
            return self._cc

        @torch.no_grad()
        def forward(self, mixed_qkv, a, b):
            """``mixed_qkv [T, D]`` pre-conv, ``a``/``b`` ``[T, Hv]`` -> ``o [T, Hv, Dv]`` or None."""
            from .gdn_engine_patch import gdn_layer_core, gdn_metadata

            md = gdn_metadata(self.prefix)
            if md is None:  # V1 profiling run: no state, and nothing to warm (configs are pinned)
                return None
            return gdn_layer_core(self._state(), md, mixed_qkv, a, b)

    _state_cls = IsoExecGDNStateLayer
    return _state_cls


def _gdn_inference_forward(self, hidden_states, attention_mask=None, **kwargs):
    """Replacement ``GatedDeltaNet.forward`` for the in-vLLM GPTModel. Returns ``(out, bias)``.

    Mirrors Megatron's own forward step for step (``in_proj`` -> split -> conv/chunk core ->
    ``_apply_gated_norm`` -> ``out_proj``), with the conv+chunk core served by ChunkConsistentGDN
    instead of one full-sequence chunk call. CP/SP are not supported here (the IsoExec recipe runs
    CP=1), so the all-to-alls Megatron does around the core are skipped rather than faked.
    """
    if self.cp_size != 1 or self.sp_size != 1:
        raise NotImplementedError("[isoexec-gdn] in-vLLM GDN requires cp_size == sp_size == 1")

    # vLLM hands the wrapper [total_tokens] and it reshapes to Megatron sbhd [T, b=1, H].
    s, b, _ = hidden_states.shape
    if b != 1:
        raise NotImplementedError(f"[isoexec-gdn] in-vLLM GDN expects batch 1, got {b}")

    qkvzba, _ = self.in_proj(hidden_states)  # [T, 1, in_proj_dim]
    qkvzba = qkvzba.transpose(0, 1)  # [1, T, ...]
    qkv, gate, beta, alpha = torch.split(
        qkvzba,
        [
            self.qk_dim_local_tp * 2 + self.v_dim_local_tp,
            self.v_dim_local_tp,
            self.num_value_heads // self.tp_size,
            self.num_value_heads // self.tp_size,
        ],
        dim=-1,
    )

    # Contiguity of qkv/alpha/beta is `gdn_layer_core`'s job -- the native core folds it into its
    # fused q/k/v split, every other path compacts there. See ops/gdn/gdn_fused_split.py.
    core = self._isoexec_state.forward(qkv[0], alpha[0], beta[0])
    if core is None:  # profiling run: shape-correct zeros, no state touched
        core = qkv.new_zeros(s, self.num_value_heads // self.tp_size, self.value_head_dim)

    # Fused gated output norm. This call site is engine-only (`swap_gdn_core` binds it, the trainer
    # never does), so the kernel cannot reach the trainer the way rebinding `_apply_gated_norm` would;
    # it sits above the chunk/recurrent split, so it serves both GDN modes.
    gate4 = gate.reshape(1, s, -1, self.value_head_dim)
    out_norm = getattr(self, "out_norm", None)
    if getattr(out_norm, "_ix_fused_norm", False) and getattr(self, "activation", None) == "silu":
        from ...ops.norms.fused_outnorm import fused_gated_out_norm

        # `out_norm.weight` is read live inside the kernel: nothing to invalidate at a weight sync.
        norm_out = fused_gated_out_norm(core.contiguous(), gate4[0], out_norm.weight, out_norm.eps)
    else:
        norm_out = self._apply_gated_norm(core.unsqueeze(0), gate4)
    norm_out = norm_out.reshape(1, s, -1).transpose(0, 1).contiguous()  # back to sbhd
    return self.out_proj(norm_out)


def swap_gdn_core(gpt_modules, *, vllm_config) -> int:
    """Attach a vLLM state layer to every Megatron ``GatedDeltaNet`` and swap in the inference path.

    Returns the number of GDN layers swapped. Must run BEFORE vLLM allocates the KV cache, i.e.
    during model construction, so the state layers are in ``static_forward_context`` when the KV
    cache manager enumerates them.
    """
    if os.environ.get("SKYRL_ISOEXEC_GDN") != "1":
        return 0

    from ...ops.gdn.gdn_batch_invariant import (
        pin_fla_autotune_configs,
        pin_gdn_rmsnorm_rows_per_block,
    )
    from ...ops.gdn.gdn_ops import chunk_synced_mode
    from .gdn_engine_patch import (
        install_chunk_synced_lazy_driver,
        lift_gdn_batch_invariance_veto,
    )

    pin_fla_autotune_configs()
    pin_gdn_rmsnorm_rows_per_block()
    if chunk_synced_mode():
        # chunk_synced always runs the lazy resync driver on the engine: it removes the per-step
        # host sync even eager, and is what makes decode capturable under CUDA graphs.
        install_chunk_synced_lazy_driver()
    # The GDN layers are batch-invariant under chunk-consistent decode, so let the rest of the model
    # (softmax attention, GEMMs, log_softmax) be made invariant too.
    lift_gdn_batch_invariance_veto()

    cls = state_layer_cls()
    n = 0
    for layer in getattr(gpt_modules.decoder, "layers", []):
        gdn = getattr(layer, "self_attention", None)
        if gdn is None or not _is_gdn(gdn):
            continue
        layer_id = getattr(layer, "layer_number", n + 1) - 1
        prefix = f"decoder.layers.{layer_id}.self_attention"
        # `_isoexec_state` on a plain attribute, not a submodule: native weight sync walks
        # `gpt.named_parameters()` and must not see anything new.
        object.__setattr__(gdn, "_isoexec_state", cls(vllm_config=vllm_config, prefix=prefix, gdn=gdn))
        gdn.forward = _gdn_inference_forward.__get__(gdn, type(gdn))
        n += 1

    if n:
        from ...ops.gdn.gdn_ops import gdn_kernel_mode

        print(
            f"[ISOEXEC-GDN] swapped {n} Megatron GatedDeltaNet layer(s) -> {gdn_kernel_mode()} core "
            "(vLLM-registered mamba state)",
            flush=True,
        )
    # Fingerprint the engine's gdn.core / gdn.conv install. The mode is read from the same call-time
    # predicate the kernels use, so a manifest that disagrees with the exported flag is caught.
    try:
        from ...core.fingerprint import (
            DECODE_ONLY,
            ENGINE_SITES,
            NOT_INSTALLED,
            PREFILL_ONLY,
            record_installs,
        )
        from ...ops.gdn.gdn_ops import (
            chunk_synced_mode as _cs,
        )
        from ...ops.gdn.gdn_ops import (
            gdn_kernel_mode as _mode,
        )
        from ...ops.gdn.gdn_ops import (
            gdn_native_conv_enabled as _nconv,
        )
        from ...ops.gdn.gdn_ops import (
            gdn_native_kernels_enabled as _nk,
        )
        from ...ops.gdn.gdn_ops import (
            recurrent_mode as _rec,
        )

        if n:
            _core = "native_fused_sigmoid" if (_nk() and (_rec() or _cs())) else _mode()
            record_installs("gdn.core", ENGINE_SITES, _core, _gdn_inference_forward)
            # Distinct kernels per site: the fn form at prefill, the update form at decode.
            _conv_native = _nconv() or _cs()
            record_installs("gdn.conv", PREFILL_ONLY, "causal_conv1d_fn" if _conv_native else "elementwise_shifted_sum")
            record_installs(
                "gdn.conv", DECODE_ONLY, "causal_conv1d_update" if _conv_native else "elementwise_shifted_sum"
            )
        else:
            record_installs("gdn.core", ENGINE_SITES, NOT_INSTALLED)
    except Exception as _e:  # pragma: no cover - never fatal
        logger.warning(f"[ISOEXEC-FINGERPRINT] gdn.core/conv record skipped: {_e}")
    return n
