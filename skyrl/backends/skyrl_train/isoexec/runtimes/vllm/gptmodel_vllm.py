"""Run Megatron's GPTModel inside vLLM (unified-model IsoExec route).

vLLM becomes pure runtime (scheduler, paged KV cache, sampling) while the compute is the same
bridge-built Megatron GPTModel the trainer runs, so per-op inputs are bitwise-identical.
``MegatronCoreAttnToVLLM`` replaces ``SelfAttention.core_attention`` with vLLM's paged Attention,
``GPTModelVLLMWrapper`` implements vLLM's model interface over the swapped model, and
``register_gptmodel_to_vllm`` registers it under our custom architecture name.

Scope: matched TP (Megatron TP == vLLM TP), single-node TP only (the worker world must be the TP
group), enforce_eager, and the column-parallel output layer gathering logits for vLLM's sampler.
Weights arrive via native sync rather than HF, so ``load_weights`` is effectively a no-op.
"""

from __future__ import annotations

import itertools
import logging
import os

import torch
from torch import nn

logger = logging.getLogger(__name__)

_DENSE_MODEL_NAME = "MegatronGPTModelForCausalLM"
_HYBRID_MODEL_NAME = "MegatronGPTModelHybridForCausalLM"


# Each capability tuple needs its own class under its own registered name: vLLM caches a model's
# `_ModelInfo` (which carries `is_hybrid`, `supports_mrope`, ...) by module+class name, so one class
# cannot answer differently on different runs -- the first run's answer sticks. `model_classes` maps
# the tuple to a class; the names above are the canonical names for the shipped tuples and must not
# change.
def _vllm_model_name() -> str:
    from .model_classes import resolve_model_name

    return resolve_model_name()


VLLM_MODEL_NAME = _vllm_model_name()
TORCHTITAN_LIKE_CONFIG_FORMAT = "megatron_gptmodel"


# Register the engine-side bitwise kernels in the WORKER process at import time. Each rank is its
# own Ray worker actor and does not inherit the backend/logprob patches applied in the engine actor,
# and the worker lazily imports this module to build the model -- so registering here, before vLLM
# resolves the attention backend and before sampling, is what makes the worker use CUSTOM varlen
# (num_splits=1, hence bitwise decode==prefill at all lengths) and the aten-logprob path. Without it
# the worker silently falls back to vLLM's flash backend, whose split-K heuristic breaks
# decode==prefill on long sequences. Idempotent; no-op off the local-spec path.
if os.environ.get("SKYRL_ISOEXEC_LOCAL_SPEC") == "1":
    try:
        from skyrl.backends.skyrl_train.isoexec.ops.attention import (
            varlen_backend as _ix_varlen,
        )
        from skyrl.backends.skyrl_train.isoexec.runtimes.vllm.vllm_patches import (
            patch_vllm_logprobs_batch_invariant as _ix_patch_lp,
        )
        from skyrl.backends.skyrl_train.isoexec.runtimes.vllm.vllm_patches import (
            patch_vllm_sampler_temperature as _ix_patch_temp,
        )

        _ix_custom_ok = _ix_varlen.register_varlen_custom_backend()
        _ix_patch_lp()
        _ix_patch_temp()
        logger.info("[isoexec] worker-side init: CUSTOM varlen available=%s, logprob patch applied", _ix_custom_ok)
    except Exception as _ix_e:  # pragma: no cover
        logger.warning("[isoexec] worker-side CUSTOM/logprob registration failed: %s", _ix_e)


class MegatronCoreAttnToVLLM(nn.Module):
    """Replacement for ``SelfAttention.core_attention`` that uses vLLM's paged Attention.

    Megatron calls ``core_attention(query, key, value, attention_mask, attn_mask_type=...,
    attention_bias=..., packed_seq_params=...)`` with q/k/v in ``[sq, b, np, hn]`` (sbhd) and
    expects ``[sq, b, np*hn]`` back. Under vLLM, b==1 and vLLM owns the KV cache + causal mask,
    so we drop the batch dim, hand ``[tokens, heads, hn]`` to ``vllm.Attention``, and reshape
    back. q/k-norm and RoPE were already applied upstream by SelfAttention.
    """

    _layer_counter = itertools.count()

    def __init__(self, *, num_heads: int, num_kv_heads: int, head_dim: int, scale: float, layer_id: int = None):
        super().__init__()
        from vllm.config import get_current_vllm_config
        from vllm.model_executor.layers.attention import Attention

        vllm_config = get_current_vllm_config()
        cache_config = getattr(vllm_config, "cache_config", None)
        # ``layer_id`` is passed in only by the GDN-skipping path, which must produce the same
        # prefix -- the vLLM KV-cache layer name -- as the unskipped path does for that layer.
        layer_id = next(MegatronCoreAttnToVLLM._layer_counter) if layer_id is None else layer_id
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.vllm_attn = Attention(
            num_heads=num_heads,
            head_size=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            cache_config=cache_config,
            quant_config=None,
            prefix=f"decoder.layers.{layer_id}.self_attention.core_attention",
        )

    def forward(
        self,
        query,
        key,
        value,
        attention_mask=None,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
        **extra_kwargs,
    ):
        # `**extra_kwargs`: MLA's `_run_core_attention` forwards its caller's kwargs through. See
        # the same note in ops/attention/megatron_varlen_attn.TorchVarlenCoreAttn.forward.
        # Megatron sbhd: [sq, b, np, hn]. vLLM's Attention wants FLATTENED 2D
        # [num_tokens, np*hn] (matches native Qwen3: self.attn(q,k,v) with q [T, q_size]).
        sq, b = query.shape[0], query.shape[1]
        q = query.reshape(sq * b, self.num_heads * self.head_dim).contiguous()
        k = key.reshape(sq * b, self.num_kv_heads * self.head_dim).contiguous()
        v = value.reshape(sq * b, self.num_kv_heads * self.head_dim).contiguous()
        out = self.vllm_attn(q, k, v)  # [num_tokens, np*hn]
        return out.reshape(sq, b, self.num_heads * self.head_dim)


def swap_core_attention(gpt_modules, *, num_heads, num_kv_heads, head_dim, scale):
    """Replace every decoder layer's ``self_attention.core_attention`` with the vLLM adapter.

    ``SKYRL_ISOEXEC_ENGINE_ATTN_SKIP_GDN`` (default on) skips the linear-attention layers. On a
    hybrid model they occupy the ``self_attention`` slot without a ``core_attention``, so the
    assignment would create a full paged ``Attention`` that registers itself in
    ``static_forward_context`` and is never called. The cost is not the unused modules but their
    KV-cache specs: vLLM sets ``group_size`` to the largest layer type and divides available memory
    by it, so phantom attention layers cut maximum concurrency proportionally.

    Skipping moves no bits -- the modules are never invoked, hold no parameters, and ``layer_id`` is
    still advanced for every layer carrying a ``self_attention``, so each surviving layer keeps the
    exact prefix (its vLLM KV-cache layer name) it had before. On a model with nothing to skip the
    predicate is false for every Megatron attention class, so dense models take the identical path.
    """
    skip_gdn = os.environ.get("SKYRL_ISOEXEC_ENGINE_ATTN_SKIP_GDN", "1").lower() not in ("", "0", "false", "no")
    modules = gpt_modules if isinstance(gpt_modules, (list, tuple)) else [gpt_modules]
    n = 0
    n_skipped = 0
    for m in modules:
        inner = m.module if hasattr(m, "module") else m
        for layer in inner.decoder.layers:
            sa = getattr(layer, "self_attention", None)
            if sa is None:
                continue
            # Advanced for EVERY layer that reaches here, skipped or not, so the surviving layers'
            # prefixes are exactly the ones the OFF path assigns them.
            layer_id = next(MegatronCoreAttnToVLLM._layer_counter)
            if skip_gdn and not hasattr(sa, "core_attention"):
                n_skipped += 1  # linear-attention layer: it never calls core_attention
                continue
            sa.core_attention = MegatronCoreAttnToVLLM(
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                scale=scale,
                layer_id=layer_id,
            )
            n += 1
    logger.info("[isoexec] swapped core_attention -> vLLM paged Attention on %d layers", n)
    if skip_gdn or n_skipped:
        # fd 1, not logger.info: vLLM filters INFO out of engine subprocesses.
        try:
            os.write(
                1,
                f"[ISOEXEC-ATTN] pid={os.getpid()} paged Attention registered on {n} layer(s), "
                f"{n_skipped} linear-attention layer(s) SKIPPED (SKIP_GDN={int(skip_gdn)}). "
                f"vLLM's KV group_size is max(layers of one type); check the engine's own "
                f"'Maximum concurrency for ... tokens per request' line against it.\n".encode(),
            )
        except Exception:  # pragma: no cover - fd 1 closed
            pass
    # Fingerprint the engine's `attention.varlen` install from the layer COUNT, not the flag:
    # "zero layers were swapped" is a real failure mode and must not read as a successful install.
    from ...core.fingerprint import ENGINE_SITES, NOT_INSTALLED, record_installs

    record_installs(
        "attention.varlen",
        ENGINE_SITES,
        "vllm_flash_ns1" if n else NOT_INSTALLED,
        MegatronCoreAttnToVLLM.forward,
    )
    return n


class _PositionIndexedRoPE(nn.Module):
    """Wraps Megatron's RotaryEmbedding so the returned RoPE is indexed by vLLM's ABSOLUTE
    positions instead of sequence-index 0..L-1. Required for paged decode (1-token inputs whose
    true position is N). For prefill (positions==0..L-1) it reproduces the original exactly."""

    def __init__(self, orig, max_pos, per_forward_cache: bool = False):
        super().__init__()
        self._orig = orig
        with torch.no_grad():
            self._emb_full = orig(max_pos)  # [max_pos, 1, 1, dim]
        self._positions = None
        # Per-forward cache, for models whose layers each call RoPE themselves rather than taking a
        # model-level freqs object: without it every layer allocates a distinct freqs tensor and the
        # fused kernel's cos/sin hoist, which caches on that tensor, never amortizes. Correct because
        # `positions` is fixed for the forward (set_positions runs once, before any layer) and the
        # cache is replaced by the next set_positions, so there is no staleness window. Off by
        # default so the dense path keeps the object identity its own hoist keys on.
        self._per_forward_cache = per_forward_cache
        self._cached = None

    def set_positions(self, positions):
        self._positions = positions
        self._cached = None

    def forward(self, max_seq_len, *args, **kwargs):
        # Stamp the engine mark on the freqs tensor. This module is only ever constructed inside a
        # vLLM worker, so the mark exists only in the engine process -- that is what keeps the fused
        # RoPE off the trainer even though its install point is a module global. The mark also
        # carries the hoisted cos/sin cache, which dies with the tensor. See rope_fused.py.
        from skyrl.backends.skyrl_train.isoexec.ops.rope.rope_fused import (
            mark_engine_rope,
        )

        if self._positions is not None:
            if self._cached is not None:
                return self._cached
            # Advanced indexing allocates, so this is the fresh per-forward base the hoist keys on
            # and the tensor `mark_engine_rope` records as the live marked freqs. A consumer that
            # slices it cannot be matched by walking `_base`: this runs under
            # `torch.inference_mode()`, and torch skips view tracking for inference tensors -- the
            # detector matches on shared storage instead (ops/rope/rope_fused.engine_marked_host).
            out = mark_engine_rope(self._emb_full.to(self._positions.device)[self._positions])  # [T,1,1,dim]
            if self._per_forward_cache:
                self._cached = out
            return out
        return mark_engine_rope(self._orig(max_seq_len, *args, **kwargs))

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)  # _orig (submodule), _positions, etc.
        except AttributeError:
            return getattr(self._orig, name)  # delegate get_rotary_seq_len/get_cos_sin/...


class GPTModelVLLMWrapper(nn.Module):
    """vLLM model whose compute is a Megatron GPTModel (attention swapped to vLLM paged).

    Built by the `register_gptmodel_to_vllm` closure, which captures the model path. Weights arrive
    via native sync, so ``load_weights`` only reports param names for vLLM's safety check.
    """

    # A config declaring `rope_parameters.mrope_section` makes vLLM require SupportsMRoPE before it
    # will build positions at all, while the Megatron text bridge uses plain 1-D RoPE. For a
    # text-only request all three M-RoPE sections are the absolute position, so satisfying the
    # protocol is cheaper than rewriting the HF config; `forward` collapses [3, T] back to one row.
    supports_mrope = True

    # `is_hybrid` is deliberately NOT set here: vLLM caches `_ModelInfo` by module+class name, so an
    # env-dependent answer on this class would be baked in by one run and poison the next. The
    # hybrid flag lives on the subclass below, which has its own cache entry.

    # vLLM reconciles the attention page size with the mamba page size by reading the state geometry
    # from these two classmethods on the model CLASS, before any layer exists. Without them the page
    # sizes never unify and the KV cache manager refuses the config.
    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateShapeCalculator,
        )

        # Under chunk_synced the GDN state lives in ChunkSyncedGDN's private pools and these pages
        # are only a slot-id source, so a bytes-sized state is reported here and the hybrid config
        # pass stops inflating every page to cover state nothing reads. MUST agree with
        # gdn_gptmodel.IsoExecGDNStateLayer.get_state_shape, the runtime spec.
        from ...ops.gdn.gdn_ops import GDN_CS_MIN_STATE_SHAPES, gdn_cs_min_pages

        if gdn_cs_min_pages():
            return GDN_CS_MIN_STATE_SHAPES

        hf = vllm_config.model_config.hf_text_config
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            vllm_config.parallel_config.tensor_parallel_size,
            hf.linear_num_key_heads,
            hf.linear_num_value_heads,
            hf.linear_key_head_dim,
            hf.linear_value_head_dim,
            hf.linear_conv_kernel_dim,
            0,  # chunk-consistent decode forbids speculative decoding
        )

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateDtypeCalculator,
        )

        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    def get_mrope_input_positions(self, input_tokens, mm_features=None, **kwargs):
        if mm_features:
            raise RuntimeError("[isoexec] the GPTModel path is text-only; got multimodal features")
        n = len(input_tokens)
        pos = torch.arange(n, dtype=torch.long).unsqueeze(0).repeat(3, 1)
        return pos, 0  # delta 0: decode positions are the plain absolute positions

    def __init__(self, *, vllm_config, prefix="", model_path=None, load_weights=None):
        super().__init__()
        from megatron.bridge import AutoBridge
        from megatron.core.transformer.enums import AttnBackend

        from skyrl.backends.skyrl_train.isoexec import (
            apply_megatron_isoexec_patches,
            make_isoexec_local_layer_spec,
            prepare_isoexec_moe,
        )

        # vLLM string-registration instantiates us with only (vllm_config, prefix); derive the
        # model path from the engine config in that case.
        if model_path is None:
            model_path = vllm_config.model_config.model
        # load_weights: bridge loads real HF weights at init (standalone). Under SkyRL the trainer
        # overwrites them via native sync, but loading at init is harmless (env override available).
        if load_weights is None:
            load_weights = os.environ.get("SKYRL_ISOEXEC_ENGINE_LOAD_WEIGHTS", "1") == "1"
        # Same `fla` facade the trainer installs: the engine builds the SAME GPTModel, so its GDN
        # layers must run the same ops. Idempotent -- normally already done by the isoexec package
        # __init__, which runs before this module's body.
        if os.environ.get("SKYRL_ISOEXEC_GDN") == "1":
            from skyrl.backends.skyrl_train.isoexec import install_fla_shim

            install_fla_shim()
        b = AutoBridge.from_hf_pretrained(model_path, trust_remote_code=True)
        # Qwen3.5 registers as a VL architecture; the VL bridge does not build a plain GPTModel (and
        # cannot pack sequences). Force the text bridge, exactly as megatron_worker.init_configs does,
        # so the engine and the trainer construct the SAME GPTModel.
        if os.environ.get("SKYRL_ISOEXEC_GDN") == "1":
            from skyrl.backends.skyrl_train.isoexec.runtimes.megatron.gdn_hybrid_spec import (
                checkpoint_is_vl_named,
                patch_qwen35_bridge_for_local_spec,
            )
            from skyrl.backends.skyrl_train.workers.megatron.model_bridges import (
                maybe_force_qwen35_text_bridge,
            )

            # Read VL-ness BEFORE the sentinel rewrite: the released Qwen3.5 checkpoints are
            # VL-architected and store the LM under `model.language_model.`, but the text bridge we
            # are about to select builds HF names as `model.`.
            vl = checkpoint_is_vl_named(b.hf_pretrained.config)
            patch_qwen35_bridge_for_local_spec(hf_lm_prefix="model.language_model." if vl else None)
            if maybe_force_qwen35_text_bridge(b, b.hf_pretrained.config):
                print("[ISOEXEC-WRAP] forced Qwen3.5 TEXT bridge (GPTModel + GDN, not the VL model)", flush=True)
        mp = b.to_megatron_provider(load_weights=load_weights)
        # IsoExec needs the engine's GPTModel sharded exactly as the trainer's, so Megatron TP must
        # equal vLLM TP. Megatron's model-parallel state does not exist in a vLLM worker, so it is
        # built over the group vLLM already made: for single-node TP the worker world IS the TP group.
        tp = int(vllm_config.parallel_config.tensor_parallel_size)
        self._tp_size = tp
        from megatron.core import parallel_state as mpu
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        # Megatron's parallel layers read the GLOBAL mpu, so an already-initialized Megatron TP that
        # differs from the engine's tp (mismatched-TP IsoExec) would silently shard the engine model
        # at the trainer's width -- reset it. This is safe only because the engine is a distinct
        # process; sharing one process and one global mpu with the trainer would need mpu swapped
        # around the sleep/wake boundary instead.
        _cur = mpu.get_tensor_model_parallel_world_size() if mpu.model_parallel_is_initialized() else None
        if _cur is not None and _cur != tp:
            print(
                f"[ISOEXEC-WRAP] Megatron mpu is TP={_cur} but engine tp={tp}; resetting to engine tp "
                f"(mismatched-TP IsoExec)",
                flush=True,
            )
            mpu.destroy_model_parallel()
            _cur = None
        if tp > 1 and not torch.distributed.is_initialized():
            raise RuntimeError("[isoexec] vLLM did not initialize torch.distributed for TP>1")
        if tp > 1:
            world = torch.distributed.get_world_size()
            if world != tp:
                raise NotImplementedError(
                    f"[isoexec] engine TP={tp} but torch.distributed world={world}; the IsoExec "
                    "GPTModel path assumes the vLLM worker world is exactly the TP group "
                    "(single-node TP, no PP/DP inside the engine)."
                )
        if _cur is None:
            if not torch.distributed.is_initialized():
                # tp==1 single-GPU engine with no vLLM-created world: a trivial 1-rank group so
                # Megatron's mpu (and its ColumnParallel/RowParallel world=1) can initialize.
                torch.distributed.init_process_group(
                    backend="nccl", world_size=1, rank=0, store=torch.distributed.HashStore()
                )
            mpu.initialize_model_parallel(tensor_model_parallel_size=tp)
            model_parallel_cuda_manual_seed(0)
            print(
                f"[ISOEXEC-WRAP] Megatron model-parallel initialized (tp={tp}, "
                f"rank={torch.distributed.get_rank() if torch.distributed.is_initialized() else 0})",
                flush=True,
            )

        mp.tensor_model_parallel_size = tp
        mp.pipeline_model_parallel_size = 1
        mp.expert_model_parallel_size = 1  # EP>1 makes the expert combine a nondeterministic collective
        mp.expert_tensor_parallel_size = tp  # experts sharded over the TP group (ETP == TP at EP=1)
        mp.sequence_parallel = False
        # GPTModel's output layer is column-parallel: with parallel_output=True each rank returns its
        # vocab shard. vLLM's sampler wants the full row. Gather inside Megatron.
        mp.parallel_output = False
        mp.pipeline_dtype = torch.bfloat16
        mp.apply_rope_fusion = False
        mp.attention_backend = AttnBackend.flash
        mp.gradient_accumulation_fusion = False
        # Force Megatron's LOCAL layer spec so the engine GPTModel runs the same plain-torch ops as
        # the local-spec trainer. Bitwise IsoExec comes from batch invariance plus num_splits=1
        # attention over that forward, not from the TE-targeted megatron patches.
        self._local_spec = os.environ.get("SKYRL_ISOEXEC_LOCAL_SPEC") == "1"
        if self._local_spec:
            # MoE providers resolve to get_gpt_layer_local_spec(num_experts=..., grouped_gemm=False)
            # -> MoELayer(TopKRouter, SequentialMLP); dense providers to bridge's local_layer_spec.
            mp.transformer_layer_spec = make_isoexec_local_layer_spec(mp)
            print("[ISOEXEC-WRAP] forced Megatron LOCAL layer spec (no TransformerEngine)", flush=True)
        # Disable MTP to match the trainer, which drops it for training: if the engine keeps MTP
        # layers and the trainer does not, the two GPTModels are different models.
        if getattr(mp, "mtp_num_layers", None):
            print(
                f"[ISOEXEC-WRAP] disabling MTP (mtp_num_layers={mp.mtp_num_layers} -> None) to match trainer",
                flush=True,
            )
            mp.mtp_num_layers = None
        # Mirror the trainer's RoPE-base workaround: transformers v5 moves rope_theta into
        # rope_parameters, so megatron-bridge reads the now-missing config.rope_theta and silently
        # falls back to rotary_base=10000. Both sides must apply it or the two RoPE bases differ.
        _hf = vllm_config.model_config.hf_config
        _rp = getattr(_hf, "rope_parameters", None) or getattr(_hf, "rope_scaling", None)
        if isinstance(_rp, dict) and _rp.get("rope_theta"):
            mp.rotary_base = _rp["rope_theta"]
        elif getattr(_hf, "rope_theta", None):
            mp.rotary_base = _hf.rope_theta
        print(
            f"[ISOEXEC-WRAP] rotary_base set to {getattr(mp, 'rotary_base', '?')} (hf rope_theta="
            f"{getattr(_hf, 'rope_theta', None)}, rope_parameters={_rp})",
            flush=True,
        )
        # Pin the identical MoE recipe the trainer pins and install the deterministic combine /
        # sorted router top-k. Both sides must apply it. No-op on dense providers.
        if self._local_spec:
            prepare_isoexec_moe(mp, side="ENGINE")
        mp.finalize()
        gpt = mp.provide_distributed_model(wrap_with_ddp=False)
        self._gpt_list = gpt
        self.gpt = gpt[0].module if hasattr(gpt[0], "module") else gpt[0]

        # Engine-only no-gather MoE dispatch: with SP off the allgather dispatcher's token gather
        # makes TP identical copies of the batch, and under pik-fc2 the tree already leaves the
        # bit-identical reduced output on every rank. Trainer models are never marked.
        if os.environ.get("SKYRL_ISOEXEC_MOE_PIK_FC2") == "1" and self._local_spec:
            from skyrl.backends.skyrl_train.isoexec.ops.moe.moe_batch_invariant import (
                mark_engine_dispatchers_nogather,
            )

            mark_engine_dispatchers_nogather(self.gpt)

        # Engine-only fused router + permute sort. The class patches fire only on MARKED instances,
        # so the trainer -- which can share this process and needs the autograd graph megatron's
        # eager ops build -- keeps the native path. Marking is unconditional so the [ISOEXEC-MOE] line
        # below reports the flag's value as seen at engine build.
        if self._local_spec:
            from skyrl.backends.skyrl_train.isoexec.ops.moe.moe_router_o2_kernel import (
                mark_engine_router_o2,
            )

            mark_engine_router_o2(self.gpt)

        # The dense-scatter and router-chain kernels fire only on marked engine instances, by the
        # same mechanism and for the same reason. Marking is unconditional; each kernel stays gated
        # on its own flag.
        if self._local_spec:
            from skyrl.backends.skyrl_train.isoexec.ops.moe.moe_dense_scatter_kernel import (
                mark_engine_routing_mechanics,
            )

            mark_engine_routing_mechanics(self.gpt)

        # Engine-only fused MoE top-k combine. Same instance-mark mechanism and the same reason:
        # `unpermute` is a module-level function, so the binding is process-global. The raw Triton
        # call has no grad_fn, so unmarked (trainer) dispatchers keep the eager fixed-order combine.
        if self._local_spec:
            from skyrl.backends.skyrl_train.isoexec.ops.moe.moe_combine_kernel import (
                mark_engine_fused_combine,
            )

            mark_engine_fused_combine(self.gpt)

        # Engine-only MoE preamble + shared-expert fusions (shared-expert GLU, shared_expert_gate,
        # and the router's fp32 casts folded into the gating GEMM). These are INSTANCE rebinds on
        # this model's own modules -- the seam that keeps them off a trainer sharing this process.
        # The post-attention residual add is deliberately not fused: it has no instance to mark.
        if self._local_spec:
            from skyrl.backends.skyrl_train.isoexec.ops.moe.moe_preamble_o12 import (
                install_engine_moe_preamble,
            )

            install_engine_moe_preamble(self.gpt)

            # After the preamble install, so each engine-local MoE component is marked before the
            # router cache inspects it. The cache owns only the fp32 weight cast; the cast VALUE is
            # contract and only its recomputation is waste, invalidated by the in-place re-cast at
            # moe_fused_weights.bump_sync_epoch.
            from skyrl.backends.skyrl_train.isoexec.ops.moe.moe_router_cast_cache import (
                install_engine_router_cast_cache,
            )

            install_engine_router_cast_cache(self.gpt)

        # SKYRL_ISOEXEC_SPLIT_LM_HEAD (default off): Megatron's post_process stage runs the output
        # layer on every token and gathers the full vocab in fp32, but vLLM samples only the last
        # token of each sequence. Flipping post_process after the build is safe -- TransformerBlock
        # decides its final_layernorm at __init__ and then dispatches on `self.final_layernorm is not
        # None`, never re-reading post_process -- so the decoder keeps its final layernorm and
        # GPTModel._postprocess returns hidden states early.
        self._split_lm_head = os.environ.get("SKYRL_ISOEXEC_SPLIT_LM_HEAD") == "1"
        if self._split_lm_head:
            core = self.gpt
            while not hasattr(core, "output_layer") and hasattr(core, "module"):
                core = core.module
            if not hasattr(core, "output_layer"):
                raise RuntimeError("[isoexec] SPLIT_LM_HEAD: could not locate GPTModel.output_layer")
            self._gpt_core = core
            # The output layer must still gather the full vocab row (parallel_output=False), else
            # compute_logits would hand vLLM's sampler a TP shard.
            assert getattr(
                core.output_layer, "gather_output", False
            ), "[isoexec] SPLIT_LM_HEAD needs a gathering output layer (parallel_output=False)"
            # sequence_parallel on the output layer would mean the input must arrive SP-sharded;
            # this engine sets mp.sequence_parallel=False, so assert rather than silently mis-shard.
            assert not getattr(
                core.output_layer, "sequence_parallel", False
            ), "[isoexec] SPLIT_LM_HEAD: output_layer.sequence_parallel=True is unsupported"
            # The decoder's final layernorm must have survived the flip -- verify, do not assume.
            _dec_ln = getattr(getattr(core, "decoder", None), "final_layernorm", None)
            assert _dec_ln is not None, "[isoexec] SPLIT_LM_HEAD: decoder lost its final_layernorm"
            core.post_process = False
            print(
                f"[ISOEXEC-WRAP] SPLIT_LM_HEAD=1: post_process -> False (forward returns hidden "
                f"states, compute_logits applies lm_head); final_layernorm={type(_dec_ln).__name__} "
                f"share_emb={core.share_embeddings_and_output_weights}",
                flush=True,
            )
        else:
            self._gpt_core = None
            print("[ISOEXEC-WRAP] SPLIT_LM_HEAD=0: fused lm_head in forward (default)", flush=True)

        # Numeric recipe. On the local-spec stack the model is already plain torch and batch
        # invariance comes from VLLM_BATCH_INVARIANT=1, while apply_megatron_isoexec_patches targets
        # TE kernels that do not exist there -- so it runs only on the TE stack.
        if not self._local_spec:
            # vLLM already registered the aten ops under VLLM_BATCH_INVARIANT=1; add only the TE
            # GEMM/RMSNorm + RoPE patches here.
            apply_megatron_isoexec_patches(skip_aten_registration=True)

        # Engine half of the TP/EP-invariant row-parallel (pik) pair; the trainer applies the
        # identical patch. It makes the row-parallel K-reduction follow the same fixed leaf tree on
        # both sides, so the engine may run a different TP than the trainer with KL still exactly 0.
        if os.environ.get("SKYRL_ISOEXEC_PIK") == "1":
            from skyrl.backends.skyrl_train.isoexec.ops.collectives.pik_tp_invariant import (
                apply_pik_tp_invariant,
            )

            apply_pik_tp_invariant(side="ENGINE")

        # swap attention -> vLLM paged
        cfg = self.gpt.config
        head_dim = getattr(cfg, "kv_channels", cfg.hidden_size // cfg.num_attention_heads)
        # Megatron shards attention heads across TP, so core_attention sees per-rank head counts
        # (including the kv-replication case num_query_groups < TP); vLLM expects local counts too.
        from skyrl.backends.skyrl_train.isoexec.ops.attention.megatron_varlen_attn import (
            isoexec_local_head_counts,
        )

        local_q, local_kv = isoexec_local_head_counts(cfg, self._tp_size)
        swap_core_attention(
            self.gpt,
            num_heads=local_q,
            num_kv_heads=local_kv,
            head_dim=head_dim,
            scale=head_dim**-0.5,
        )

        # Hybrid models: the GatedDeltaNet layers swap_core_attention skips get a vLLM-registered
        # mamba state layer and chunk-consistent decode. MUST happen during model construction:
        # vLLM's KV cache manager enumerates static_forward_context after __init__ and before
        # allocating state.
        if os.environ.get("SKYRL_ISOEXEC_GDN") == "1":
            from skyrl.backends.skyrl_train.isoexec.runtimes.vllm.gdn_gptmodel import (
                swap_gdn_core,
            )

            n_gdn = swap_gdn_core(self.gpt, vllm_config=vllm_config)
            if n_gdn == 0:
                # Zero GatedDeltaNet layers means every layer came out dense: a different model
                # from the one the checkpoint describes, which would build, run, and even be bitwise
                # decode==prefill while generating gibberish. Refuse.
                raise RuntimeError(
                    "[isoexec-gdn] SKYRL_ISOEXEC_GDN=1 but the Megatron GPTModel has no GatedDeltaNet "
                    "layers. The no-TE local layer spec built dense attention for every layer. A "
                    "hybrid no-TE spec (GDN on 3 of 4 layers) is required."
                )

        # Fuse `F.rms_norm(x) * (1.0 + weight)` into one kernel and hoist the add to the weight-sync
        # boundary. Instance-level rebinds on THIS model, so the trainer's identical norm class is
        # untouched -- hoisting gamma trainer-side would detach `weight`'s gradient path, which a
        # forward-only IsoExec gate cannot see. Self-gates on SKYRL_ISOEXEC_GDN_FUSED_OUTNORM.
        from skyrl.backends.skyrl_train.isoexec.ops.norms.fused_outnorm import (
            install_engine_fused_norms,
        )

        # The count distinguishes "the fused twin installed" from "the flag was on and it rebound
        # nothing"; the install fingerprint below reports it.
        self._ix_fused_norms = install_engine_fused_norms(self.gpt)

        # RoPE-by-absolute-position: GPTModel computes RoPE for sequence-index 0..L-1, but vLLM
        # paged decode feeds 1-token inputs whose true position is N. Index a precomputed RoPE
        # cache by vLLM's `positions` so decode rotates at the right angle (else decode != prefill).
        max_pos = int(getattr(vllm_config.model_config, "max_model_len", 8192))
        self._rope = _PositionIndexedRoPE(self.gpt.rotary_pos_emb, max_pos)
        self.gpt.rotary_pos_emb = self._rope

        # Fuse the attention RoPE and hoist cos/sin out of the per-layer recompute. Installed here,
        # in the vLLM worker process only: the patch point is a module global, so the trainer must
        # never reach this line, and the fused path additionally requires the mark
        # _PositionIndexedRoPE stamps above. Self-gates on SKYRL_ISOEXEC_FUSED_ROPE.
        from skyrl.backends.skyrl_train.isoexec.ops.rope.rope_fused import (
            install_engine_fused_rope,
        )
        from skyrl.backends.skyrl_train.isoexec.runtimes.megatron.megatron_patches import (
            is_rope_fp32_installed,
        )

        # The op refuses to fuse over the megatron fp32-rope patch (different rounding chain). That
        # guard state lives in the megatron runtime, so the adapter reads it and passes it down --
        # ops must never import a runtime.
        install_engine_fused_rope(fp32_rope_installed=is_rope_fp32_installed())

        # Replace megatron's fused-QKV gather over the full TP group with an all-gather over the
        # subgroup whose shards are the only columns this rank keeps. When `num_query_groups <
        # tp_world` megatron gathers every qkv column and then slices; contiguous rank-ordered
        # ColumnParallelLinear sharding makes the subgroup gather byte-identical to
        # gather-then-slice. Installed at model build because it creates and WARMS a sub-process
        # group, and the gather it replaces runs inside the decode CUDA graphs -- a communicator
        # built lazily under capture is fatal. Self-gates on the geometry it reads off the layers.
        from skyrl.backends.skyrl_train.isoexec.ops.attention.qkv_subgroup_gather import (
            install_engine_qkv_subgroup_ag,
        )

        install_engine_qkv_subgroup_ag(self.gpt, side="ENGINE")

        # Patch vLLM's sampler logprob kernel to the trainer's exact formula HERE, in the worker
        # process where the sampler runs -- doing it in the engine actor does not reach the worker.
        # The fused Triton logprob kernel otherwise bypasses aten and diverges from the trainer.
        if self._local_spec:
            try:
                from skyrl.backends.skyrl_train.isoexec.runtimes.vllm.vllm_patches import (
                    patch_vllm_logprobs_batch_invariant,
                    patch_vllm_sampler_temperature,
                )

                patch_vllm_logprobs_batch_invariant()
                patch_vllm_sampler_temperature()
                _logprob_patched = True
            except Exception as _e:  # pragma: no cover
                _logprob_patched = False
                print(f"[ISOEXEC-WRAP] logprob patch failed: {type(_e).__name__}: {_e}", flush=True)
        else:
            _logprob_patched = False
        # The install sequence is finished: record what each family actually bound and compare it
        # against this process's manifest once. Always build the worker's manifest (idempotent,
        # cached) -- the contract handshake at create_receiver derives from it, and a worker without
        # one silently skips the check.
        from ...core.process_manifest import get_process_manifest

        get_process_manifest(model_path)
        if (
            os.environ.get("SKYRL_ISOEXEC_ENGINE_NCCL_UNPIN", "0") == "1"
            and self._tp_size > 1
            and os.environ.get("SKYRL_ISOEXEC_NCCL_TRANSPORT_BOUNDARY_REQUIREMENTS", "").strip()
        ):
            _assert_engine_nccl_manifest(model_path)
        _record_engine_install_fingerprint(self, cfg, logprob_patched=_logprob_patched)

    def embed_input_ids(self, input_ids):
        # vLLM VllmModel protocol requires this exact name.
        return self.gpt.embedding(input_ids=input_ids.unsqueeze(0), position_ids=None)

    def get_input_embeddings(self, input_ids):
        return self.embed_input_ids(input_ids)

    def forward(self, input_ids=None, positions=None, inputs_embeds=None, **kwargs):
        # With `rope_parameters.mrope_section` in the HF config vLLM feeds MRoPE positions of shape
        # [3, T], while the Megatron text bridge uses plain 1-D RoPE. For a text-only request all
        # three rows are identical, so collapse to the temporal row and refuse anything else rather
        # than silently rotating at the wrong angle.
        if positions is not None and positions.ndim == 2 and positions.shape[0] == 3:
            # `torch.equal` returns a python bool, so it is a D2H sync and is illegal under stream
            # capture. Skipping it while capturing is safe: the guard is about the request TYPE,
            # which is fixed for a text-only engine, and every real forward still checks.
            if not torch.cuda.is_current_stream_capturing() and not (
                torch.equal(positions[0], positions[1]) and torch.equal(positions[0], positions[2])
            ):
                raise RuntimeError(
                    "[isoexec] MRoPE sections differ -- this request carries image/video positions. "
                    "The IsoExec GPTModel path is text-only."
                )
            positions = positions[0]

        # vLLM varlen [total_tokens] -> Megatron [b=1, seq]. GPTModel applies RoPE internally;
        # attention is the swapped vLLM paged layer (ignores attention_mask, uses vLLM metadata).
        tokens = input_ids.unsqueeze(0)
        pos = positions.unsqueeze(0)
        self._rope.set_positions(positions.reshape(-1))  # absolute positions for RoPE
        out = self.gpt(input_ids=tokens, position_ids=pos, attention_mask=None)
        # post_process=True  -> logits, already transposed to [b, s, vocab] (gpt_model.py:765).
        # post_process=False -> hidden states in Megatron layout [s, b, h], NOT transposed
        #                       (_postprocess returns early). b == 1 here either way, so the same
        #                       reshape yields token-major rows in both cases.
        if out.dim() == 3:
            out = out.reshape(-1, out.shape[-1])
        return out

    def compute_logits(self, hidden_states, sampling_metadata=None):
        if not self._split_lm_head:
            return hidden_states  # forward already produced the logits
        # Replicates GPTModel._postprocess on the sampled rows only. ColumnParallelLinear is generic
        # over leading dims, so the 2-D row block is a valid input. Bitwise equality with the fused
        # path relies on the matmul being M-invariant, which batch invariance provides.
        core = self._gpt_core
        output_weight = core.shared_embedding_or_output_weight() if core.share_embeddings_and_output_weights else None
        # If anything upstream upcast the decoder output, come back down: bf16 -> fp32 -> bf16 is
        # exactly lossless, so this cannot perturb the bits.
        w_dtype = (output_weight if output_weight is not None else core.output_layer.weight).dtype
        if hidden_states.dtype != w_dtype:
            hidden_states = hidden_states.to(w_dtype)
        logits, _ = core.output_layer(hidden_states, weight=output_weight, runtime_gather_output=True)
        return core._scale_logits(logits)

    def load_weights(self, weights_iter):
        # Native sync: copy any native-named incoming params straight into self.gpt. At vLLM build
        # time the names are HF-checkpoint names and all miss, which is harmless because the bridge
        # already populated self.gpt. Always return the full param-name set so vLLM's "all weights
        # initialized" check passes.
        all_names = {"gpt." + n for n, _ in self.gpt.named_parameters()}
        dst = dict(self.gpt.named_parameters())
        dst.update(dict(self.gpt.named_buffers()))
        loaded, missed = 0, []
        with torch.no_grad():
            for name, tensor in weights_iter:
                dest = dst.get(name)
                if dest is None and name.startswith("module."):
                    dest = dst.get(name[len("module.") :])  # tolerate residual wrapper prefix
                if dest is None:
                    if len(missed) < 3:
                        missed.append(name)
                    continue
                d = dest.full_tensor() if hasattr(dest, "full_tensor") else dest
                if tuple(d.shape) != tuple(tensor.shape):
                    continue
                dest.copy_(tensor.to(dest.dtype))
                loaded += 1
        with torch.no_grad():
            _wn = float(next((p for n, p in self.gpt.named_parameters() if "weight" in n)).float().norm())
        print(
            f"[ISOEXEC-WRAP] load_weights: copied {loaded} native tensors into gpt "
            f"(non-native skipped, e.g. {missed}); first_w_norm={_wn:.3f}",
            flush=True,
        )
        return all_names


class GPTModelVLLMHybridWrapper(GPTModelVLLMWrapper):
    """The wrapper for GatedDeltaNet hybrids (Qwen3.5). Identical compute; only the flag differs.

    ``is_hybrid`` tells vLLM to reconcile the attention and mamba page sizes and to set
    ``cache_config.mamba_block_size``. It is read off the CLASS before any instance exists, and the
    answer is cached to disk per class name -- hence a separate class rather than an env-dependent
    attribute on the dense one.
    """

    is_hybrid = True

    @classmethod
    def get_mamba_state_copy_func(cls):
        """(conv, ssm) slice layout of one mamba block, used by the model runner's per-step state
        copy. Our state layers use the same shape calculator as the native model, so the native GDN
        copy specs are correct. Only consulted when ``cache_config.mamba_cache_mode == 'align'``.
        """
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateCopyFuncCalculator,
        )

        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()


def _engine_nccl_runtime_identity():
    """Read the engine's installed owner plus the exact effective process tuple."""
    from ...ops.collectives.nccl_identity import PINNED
    from ...ops.collectives.nccl_identity import (
        constants_for_impl as nccl_constants_for_impl,
    )
    from ...ops.collectives.nccl_identity import (
        effective_identity as effective_nccl_identity,
    )

    neutralized = False
    try:
        import vllm.model_executor.layers.batch_invariant as bi

        neutralized = bool(getattr(bi, "_isoexec_channel_pin_neutralized", False))
    except Exception:
        pass
    return effective_nccl_identity() if neutralized else (PINNED, nccl_constants_for_impl(PINNED))


def _assert_engine_nccl_manifest(model_path: str) -> None:
    """Build and validate the active engine process's cap-aware manifest.

    vLLM Ray workers are distinct processes, so a manifest cached by the engine actor or EngineCore
    is not visible here -- build it idempotently from the model path rather than relying on
    ``cached_manifest()``, which fails in every nested worker.
    """
    from ...core.process_manifest import get_process_manifest
    from ...ops.collectives.nccl_identity import assert_manifest_matches

    impl_id, constants = _engine_nccl_runtime_identity()
    manifest = get_process_manifest(model_path)
    assert_manifest_matches(manifest, ("engine_prefill", "engine_decode"), impl_id, constants)


def _record_engine_install_fingerprint(wrapper, cfg, *, logprob_patched: bool) -> None:
    """Record what the ENGINE adapter just installed, per op family, then log the comparison.

    Recorded here because the adapter is the one place that knows WHICH SIDE it is on: the trainer
    can share this process, so an op module recording its own rebind would have to guess. Each
    impl_id is read off the state the install actually reached -- a swap count, a post-install flag,
    a live predicate -- never off the manifest or a launcher's intent, and a family that did not
    install records ``NOT_INSTALLED`` rather than staying silent. Fail-soft: a fingerprint bug must
    never break an engine build.
    """
    try:
        from ...core.fingerprint import (
            ENGINE_SITES,
            NOT_INSTALLED,
            log_fingerprint_once,
            record_installs,
        )
        from ...core.process_manifest import cached_manifest
        from ...ops.mm.mm_cublaslt import mm_cublaslt_enabled

        record_installs("mm", ENGINE_SITES, "cublaslt_pinned" if mm_cublaslt_enabled() else "triton_batch_invariant")

        from ...ops.rope import rope_fused as _rf

        record_installs("rope.rope", ENGINE_SITES, "fused" if _rf._INSTALLED else "eager")

        # The fused engine twin exists only for the zero-centred form, so a plain-RMS model records
        # the single impl it runs at every site rather than a twin.
        from ...ops.norms.fused_outnorm import fused_outnorm_enabled

        _fused_norms = bool(getattr(wrapper, "_ix_fused_norms", 0))
        if bool(getattr(cfg, "layernorm_zero_centered_gamma", False)):
            record_installs("norms.rms", ENGINE_SITES, "fused" if _fused_norms else "eager_zero_centered")
        else:
            record_installs("norms.rms", ENGINE_SITES, "eager_torch_rms")
        if bool(getattr(cfg, "experimental_attention_variant", None) == "gated_delta_net"):
            record_installs(
                "norms.gated_out", ENGINE_SITES, "fused" if (_fused_norms and fused_outnorm_enabled()) else "eager"
            )

        record_installs(
            "logprobs.log_softmax",
            ENGINE_SITES,
            "aten_reference_fused_exp" if logprob_patched else NOT_INSTALLED,
        )
        record_installs("logprobs.lm_head_slice", ENGINE_SITES, "sampled_rows")

        # Collectives exist only at TP>1; at TP=1 they have no site and get no record.
        if int(getattr(wrapper, "_tp_size", 1) or 1) > 1:
            from ...ops.collectives.pik_tp_invariant import pik_enabled

            _pik = "pik_tree" if pik_enabled() else NOT_INSTALLED
            record_installs("collectives.tree_all_reduce", ENGINE_SITES, _pik)
            record_installs("collectives.row_parallel", ENGINE_SITES, _pik)
            # "Did the unpin actually happen" is the fact a reader needs, and it is otherwise
            # visible only as a print inside a worker.
            _nccl_impl, _nccl_constants = _engine_nccl_runtime_identity()
            record_installs(
                "collectives.nccl_pin",
                ENGINE_SITES,
                _nccl_impl,
                pinned=_nccl_constants,
            )

        # MoE: a no-op on a dense provider, which then legitimately records nothing.
        if bool(getattr(cfg, "num_moe_experts", 0)) and wrapper._local_spec:
            from ...ops.moe.moe_blockmap_kernel import fused_blockmap_enabled
            from ...ops.moe.moe_router_o2_kernel import router_o2_enabled

            _sigmoid = getattr(cfg, "moe_router_score_function", "softmax") == "sigmoid" or bool(
                getattr(cfg, "moe_router_enable_expert_bias", False)
            )
            if _sigmoid:
                # No fused sigmoid twin exists at the same bits: one impl, both sides.
                record_installs("moe.router", ENGINE_SITES, "deterministic_sigmoid_bias")
            else:
                record_installs("moe.router", ENGINE_SITES, "fused_o2" if router_o2_enabled() else "deterministic")
            record_installs("moe.dispatch", ENGINE_SITES, "index_build")
            record_installs("moe.experts", ENGINE_SITES, "fused")
            from ...ops.moe.moe_batch_invariant import _moe_pik_fc2_on

            record_installs("moe.combine", ENGINE_SITES, "pik_leaf_tree" if _moe_pik_fc2_on() else NOT_INSTALLED)
            record_installs("moe.weights", ENGINE_SITES, "fused_buffer")
            from ...ops.moe.moe_fused_experts import _fused_epilogue_on

            record_installs("moe.epilogue", ENGINE_SITES, "fused_swiglu" if _fused_epilogue_on() else NOT_INSTALLED)
            record_installs("moe.blockmap", ENGINE_SITES, "fused" if fused_blockmap_enabled() else NOT_INSTALLED)

        # gdn.* records itself where it binds: its state core is built at the first
        # metadata-bearing forward, after this point, and that is when the pool becomes a fact.
        log_fingerprint_once(cached_manifest(), tag="engine_install")
    except Exception as e:  # pragma: no cover - never fatal
        logger.warning(f"[ISOEXEC-FINGERPRINT] engine install fingerprint skipped: {e}")


def _wrapper_import_path() -> str:
    """vLLM's ``module:ClassName`` string form for this process's capability tuple."""
    from .model_classes import import_path_for, resolve_capabilities

    return import_path_for(resolve_capabilities())


_WRAPPER_IMPORT_PATH = _wrapper_import_path()


def register_gptmodel_to_vllm(model_path: str | None = None):
    """Register the GPTModel-backed wrapper with vLLM. Call before engine init.

    Uses vLLM's STRING registration form (``module:ClassName``) so the registration survives
    across vLLM's mp/async worker subprocesses (each worker lazily imports the class). The
    wrapper derives the model path from ``vllm_config`` at build time, so no closure is needed.
    Set the engine's ``hf_overrides={"architectures": [VLLM_MODEL_NAME]}`` so vLLM builds this
    class instead of the model's native architecture.
    """
    from vllm.model_executor.models.registry import ModelRegistry

    from .model_classes import (
        needs_hybrid_config_pass,
        resolve_capabilities,
        synthesize,
    )

    caps = resolve_capabilities(model_path)
    base = {
        _HYBRID_MODEL_NAME: GPTModelVLLMHybridWrapper,
        _DENSE_MODEL_NAME: GPTModelVLLMWrapper,
    }.get(VLLM_MODEL_NAME)
    if base is None:
        # A capability tuple with no shipped class: generate one. Only the class-level protocol
        # answers vLLM reads before instantiation differ; the compute is the same.
        base = synthesize(caps)
    if model_path is not None:  # legacy closure form (single-process / standalone tests)

        class _Wrapper(base):
            def __init__(self, *, vllm_config, prefix=""):
                super().__init__(model_path=model_path, vllm_config=vllm_config, prefix=prefix)

        _Wrapper.__name__ = VLLM_MODEL_NAME
        _Wrapper.__qualname__ = VLLM_MODEL_NAME
        ModelRegistry.register_model(VLLM_MODEL_NAME, _Wrapper)
    else:  # cross-process string form (SkyRL mp/async workers)
        ModelRegistry.register_model(VLLM_MODEL_NAME, _WRAPPER_IMPORT_PATH)
    logger.info("[isoexec] registered %s into vLLM ModelRegistry", VLLM_MODEL_NAME)

    # Hybrid (GDN) models need vLLM's hybrid config pass: it is what sets
    # `cache_config.mamba_block_size` (without it `MambaBase.get_kv_cache_spec` asserts) and lets
    # the platform reconcile the attention and mamba page sizes. vLLM selects that pass by
    # ARCHITECTURE NAME, and `hf_overrides` has just replaced the architecture with ours.
    if needs_hybrid_config_pass(caps):
        from vllm.model_executor.models.config import (
            MODELS_CONFIG_MAP,
            HybridAttentionMambaModelConfig,
        )

        MODELS_CONFIG_MAP.setdefault(VLLM_MODEL_NAME, HybridAttentionMambaModelConfig)
        logger.info("[isoexec] %s -> HybridAttentionMambaModelConfig (mamba_block_size)", VLLM_MODEL_NAME)
    return VLLM_MODEL_NAME


def find_inprocess_gptmodel(llm):
    """Reach the in-process GPTModelVLLMWrapper inside a vLLM LLM (VLLM_ENABLE_V1_MULTIPROCESSING=0)
    so the trainer can native-sync weights into the rollout model each step."""
    seen = set()

    def walk(o, d=0):
        if id(o) in seen or d > 8:
            return None
        seen.add(id(o))
        if type(o).__name__ == VLLM_MODEL_NAME or hasattr(o, "gpt"):
            return o
        for a in (
            "llm_engine",
            "engine_core",
            "model_executor",
            "driver_worker",
            "model_runner",
            "model",
            "worker",
            "engine",
        ):
            if hasattr(o, a):
                try:
                    r = walk(getattr(o, a), d + 1)
                except Exception:
                    r = None
                if r is not None:
                    return r
        return None

    return walk(llm)
