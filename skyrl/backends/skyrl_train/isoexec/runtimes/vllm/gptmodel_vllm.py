"""Run Megatron's GPTModel inside vLLM (unified-model IsoExec route).

vLLM becomes pure runtime (scheduler, paged KV cache, sampling) while the compute is the same
bridge-built Megatron GPTModel the trainer runs. Scope: the vLLM worker world must be exactly the
engine TP group (no DP/PP inside the engine), and a column-parallel output layer gathering logits
for the sampler.
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


# Each capability tuple needs its own class under its own registered name: vLLM caches `_ModelInfo`
# by module+class name, so one class cannot answer differently across runs. The names above are the
# canonical names for the shipped tuples and must not change.
def _vllm_model_name() -> str:
    from .model_classes import resolve_model_name

    return resolve_model_name()


VLLM_MODEL_NAME = _vllm_model_name()
TORCHTITAN_LIKE_CONFIG_FORMAT = "megatron_gptmodel"


# Register the engine-side bitwise kernels in the WORKER process at import time: a worker does not
# inherit the engine actor's patches, and this must run before vLLM resolves the attention backend.
# Otherwise the worker falls back to vLLM's flash backend, whose split-K heuristic breaks
# decode==prefill on long sequences.
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

    Megatron passes q/k/v as sbhd ``[sq, b, np, hn]`` and expects ``[sq, b, np*hn]`` back; under
    vLLM b==1 and vLLM owns the KV cache and causal mask. q/k-norm and RoPE are applied upstream.
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
        # `**extra_kwargs`: MLA's `_run_core_attention` forwards its caller's kwargs through.
        # Megatron sbhd [sq, b, np, hn] -> vLLM's flattened 2D [num_tokens, np*hn].
        sq, b = query.shape[0], query.shape[1]
        q = query.reshape(sq * b, self.num_heads * self.head_dim).contiguous()
        k = key.reshape(sq * b, self.num_kv_heads * self.head_dim).contiguous()
        v = value.reshape(sq * b, self.num_kv_heads * self.head_dim).contiguous()
        out = self.vllm_attn(q, k, v)  # [num_tokens, np*hn]
        return out.reshape(sq, b, self.num_heads * self.head_dim)


def swap_core_attention(gpt_modules, *, num_heads, num_kv_heads, head_dim, scale):
    """Replace every decoder layer's ``self_attention.core_attention`` with the vLLM adapter.

    ``SKYRL_ISOEXEC_ENGINE_ATTN_SKIP_GDN`` (default on) skips linear-attention layers: their phantom
    KV-cache specs would inflate vLLM's group_size and cut maximum concurrency. Skipping moves no
    bits -- ``layer_id`` still advances per layer, so surviving layers keep their exact prefixes.
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
            # Advanced for EVERY layer, skipped or not, so surviving layers keep the prefixes the
            # OFF path assigns them.
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
    # Fingerprint from the layer COUNT, not the flag: "zero layers swapped" is a real failure mode
    # and must not read as a successful install.
    from ...core.fingerprint import ENGINE_SITES, NOT_INSTALLED, record_installs

    record_installs(
        "attention.varlen",
        ENGINE_SITES,
        "vllm_flash_ns1" if n else NOT_INSTALLED,
        MegatronCoreAttnToVLLM.forward,
    )
    return n


class _PositionIndexedRoPE(nn.Module):
    """Wraps Megatron's RotaryEmbedding so RoPE is indexed by vLLM's ABSOLUTE positions rather than
    sequence-index 0..L-1. Required for paged decode; identical to the original at prefill."""

    def __init__(self, orig, max_pos, per_forward_cache: bool = False):
        super().__init__()
        self._orig = orig
        with torch.no_grad():
            self._emb_full = orig(max_pos)  # [max_pos, 1, 1, dim]
        self._positions = None
        # Per-forward cache for models whose layers each call RoPE themselves; without it the fused
        # kernel's cos/sin hoist never amortizes. Safe because `positions` is fixed for the forward.
        self._per_forward_cache = per_forward_cache
        self._cached = None

    def set_positions(self, positions):
        self._positions = positions
        self._cached = None

    def forward(self, max_seq_len, *args, **kwargs):
        # Stamp the engine mark on the freqs tensor. This module only ever exists inside a vLLM
        # worker, so the mark is what keeps the fused RoPE off the trainer despite its global
        # install point. It also carries the hoisted cos/sin cache, which dies with the tensor.
        from skyrl.backends.skyrl_train.isoexec.ops.rope.rope_fused import (
            mark_engine_rope,
        )

        if self._positions is not None:
            if self._cached is not None:
                return self._cached
            # Advanced indexing allocates, so this is the fresh per-forward base the hoist keys on.
            # Views of it cannot be matched by walking `_base` (inference tensors skip view
            # tracking), so the detector matches on shared storage instead.
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

    # A config declaring `rope_parameters.mrope_section` makes vLLM require SupportsMRoPE, while the
    # Megatron text bridge uses plain 1-D RoPE. For text-only requests all three sections are the
    # absolute position, so satisfying the protocol is cheaper than rewriting the HF config.
    supports_mrope = True

    # `is_hybrid` is deliberately NOT set here: vLLM caches `_ModelInfo` by module+class name, so an
    # env-dependent answer would be baked in by one run. It lives on the subclass below instead.

    # vLLM reads the state geometry from these two classmethods on the CLASS, before any layer
    # exists; without them the attention and mamba page sizes never unify.
    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateShapeCalculator,
        )

        # Under cpr these pages are only a slot-id source, so report a bytes-sized state and stop
        # the hybrid config pass inflating every page. MUST agree with
        # gdn_gptmodel.IsoExecGDNStateLayer.get_state_shape, the runtime spec.
        from ...ops.gdn.gdn_ops import GDN_CPR_MIN_STATE_SHAPES, gdn_cpr_min_pages

        if gdn_cpr_min_pages():
            return GDN_CPR_MIN_STATE_SHAPES

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
        # The bridge loads real HF weights at init; the trainer overwrites them via native sync.
        if load_weights is None:
            load_weights = os.environ.get("SKYRL_ISOEXEC_ENGINE_LOAD_WEIGHTS", "1") == "1"
        # Same `fla` facade the trainer installs: the engine builds the SAME GPTModel, so its GDN
        # layers must run the same ops. Idempotent.
        if os.environ.get("SKYRL_ISOEXEC_GDN") == "1":
            from skyrl.backends.skyrl_train.isoexec import install_fla_shim

            install_fla_shim()
        b = AutoBridge.from_hf_pretrained(model_path, trust_remote_code=True)
        # Qwen3.5 registers as a VL architecture, whose bridge builds no plain GPTModel. Force the
        # text bridge as megatron_worker.init_configs does, so both sides build the SAME GPTModel.
        if os.environ.get("SKYRL_ISOEXEC_GDN") == "1":
            from skyrl.backends.skyrl_train.isoexec.runtimes.megatron.gdn_hybrid_spec import (
                checkpoint_is_vl_named,
                patch_qwen35_bridge_for_local_spec,
            )
            from skyrl.backends.skyrl_train.workers.megatron.model_bridges import (
                maybe_force_qwen35_text_bridge,
            )

            # Read VL-ness BEFORE the sentinel rewrite: VL checkpoints store the LM under
            # `model.language_model.`, but the text bridge builds HF names as `model.`.
            vl = checkpoint_is_vl_named(b.hf_pretrained.config)
            patch_qwen35_bridge_for_local_spec(hf_lm_prefix="model.language_model." if vl else None)
            if maybe_force_qwen35_text_bridge(b, b.hf_pretrained.config):
                print("[ISOEXEC-WRAP] forced Qwen3.5 TEXT bridge (GPTModel + GDN, not the VL model)", flush=True)
        mp = b.to_megatron_provider(load_weights=load_weights)
        # Megatron's mpu is built over the group vLLM already made: the worker world IS the TP group.
        tp = int(vllm_config.parallel_config.tensor_parallel_size)
        self._tp_size = tp
        from megatron.core import parallel_state as mpu
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        # Megatron's parallel layers read the GLOBAL mpu, so an already-initialized TP differing
        # from the engine's would silently shard the engine model at the trainer's width -- reset
        # it. Safe only because the engine is a distinct process.
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
                # tp==1 with no vLLM-created world: a trivial 1-rank group so Megatron's mpu can
                # initialize.
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
        # The column-parallel output layer must gather: vLLM's sampler wants the full vocab row.
        mp.parallel_output = False
        mp.pipeline_dtype = torch.bfloat16
        mp.apply_rope_fusion = False
        mp.attention_backend = AttnBackend.flash
        mp.gradient_accumulation_fusion = False
        # Force Megatron's LOCAL layer spec so the engine GPTModel runs the same plain-torch ops as
        # the local-spec trainer.
        self._local_spec = os.environ.get("SKYRL_ISOEXEC_LOCAL_SPEC") == "1"
        if self._local_spec:
            mp.transformer_layer_spec = make_isoexec_local_layer_spec(mp)
            print("[ISOEXEC-WRAP] forced Megatron LOCAL layer spec (no TransformerEngine)", flush=True)
        # Disable MTP to match the trainer, which drops it for training; otherwise the two sides
        # build different models.
        if getattr(mp, "mtp_num_layers", None):
            print(
                f"[ISOEXEC-WRAP] disabling MTP (mtp_num_layers={mp.mtp_num_layers} -> None) to match trainer",
                flush=True,
            )
            mp.mtp_num_layers = None
        # Mirror the trainer's RoPE-base workaround: transformers v5 moves rope_theta into
        # rope_parameters, so megatron-bridge silently falls back to rotary_base=10000.
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
        # Pin the identical MoE recipe the trainer pins. No-op on dense providers.
        if self._local_spec:
            prepare_isoexec_moe(mp, side="ENGINE")
        mp.finalize()
        gpt = mp.provide_distributed_model(wrap_with_ddp=False)
        self._gpt_list = gpt
        self.gpt = gpt[0].module if hasattr(gpt[0], "module") else gpt[0]

        # Engine-only no-gather MoE dispatch: with SP off the allgather dispatcher just makes TP
        # copies of the batch, and pik-fc2 already leaves the identical reduced output on each rank.
        if os.environ.get("SKYRL_ISOEXEC_MOE_PIK_FC2") == "1" and self._local_spec:
            from skyrl.backends.skyrl_train.isoexec.ops.moe.moe_batch_invariant import (
                mark_engine_dispatchers_nogather,
            )

            mark_engine_dispatchers_nogather(self.gpt)

        # Engine-only fused router + permute sort. The class patches fire only on MARKED instances,
        # so a trainer sharing this process keeps the native (autograd-capable) path.
        if self._local_spec:
            from skyrl.backends.skyrl_train.isoexec.ops.moe.moe_router_o2_kernel import (
                mark_engine_router_o2,
            )

            mark_engine_router_o2(self.gpt)

        # The dense-scatter and router-chain kernels use the same instance mark; each stays gated on
        # its own flag.
        if self._local_spec:
            from skyrl.backends.skyrl_train.isoexec.ops.moe.moe_dense_scatter_kernel import (
                mark_engine_routing_mechanics,
            )

            mark_engine_routing_mechanics(self.gpt)

        # Engine-only fused MoE top-k combine: `unpermute` is module-level, so the binding is
        # process-global and unmarked (trainer) dispatchers keep the eager fixed-order combine.
        if self._local_spec:
            from skyrl.backends.skyrl_train.isoexec.ops.moe.moe_combine_kernel import (
                mark_engine_fused_combine,
            )

            mark_engine_fused_combine(self.gpt)

        # Engine-only MoE preamble + shared-expert fusions, as INSTANCE rebinds on this model's own
        # modules -- the seam that keeps them off a trainer sharing this process.
        if self._local_spec:
            from skyrl.backends.skyrl_train.isoexec.ops.moe.moe_preamble_o12 import (
                install_engine_moe_preamble,
            )

            install_engine_moe_preamble(self.gpt)

            # After the preamble install, so each MoE component is marked before the router cache
            # inspects it. The cache owns only the fp32 weight cast, not its value.
            from skyrl.backends.skyrl_train.isoexec.ops.moe.moe_router_cast_cache import (
                install_engine_router_cast_cache,
            )

            install_engine_router_cast_cache(self.gpt)

        # SKYRL_ISOEXEC_SPLIT_LM_HEAD (default off): Megatron's post_process runs the output layer on
        # every token while vLLM samples only the last. Flipping post_process after the build is safe
        # -- TransformerBlock fixes its final_layernorm at __init__ and never re-reads the flag.
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

        # apply_megatron_isoexec_patches targets TE kernels, which the local-spec stack does not
        # have; there, batch invariance comes from VLLM_BATCH_INVARIANT=1 instead.
        if not self._local_spec:
            # vLLM already registered the aten ops; add only the TE GEMM/RMSNorm + RoPE patches.
            apply_megatron_isoexec_patches(skip_aten_registration=True)

        # The ContractAdapter owns this worker's enforcement sequence. Ordering is load-bearing: the
        # contract must be built BEFORE any install that asserts against it, since those checks read
        # cached_contract_view() and silently skip on None.
        def _isoexec_install():
            # Engine half of the TP/EP-invariant row-parallel (pik) pair: both sides follow the same
            # fixed leaf tree, so the engine may run a different TP with KL still exactly 0.
            if os.environ.get("SKYRL_ISOEXEC_PIK") == "1":
                from skyrl.backends.skyrl_train.isoexec.ops.collectives.pik_tp_invariant import (
                    apply_pik_tp_invariant,
                )

                apply_pik_tp_invariant(side="ENGINE")

            # swap attention -> vLLM paged
            cfg = self.gpt.config
            head_dim = getattr(cfg, "kv_channels", cfg.hidden_size // cfg.num_attention_heads)
            # Megatron shards attention heads across TP, so core_attention sees per-rank head counts;
            # vLLM expects local counts too.
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

            # Hybrid models: the GDN layers swap_core_attention skipped get a vLLM-registered mamba
            # state layer. MUST happen during model construction, before the KV cache manager
            # enumerates static_forward_context.
            if os.environ.get("SKYRL_ISOEXEC_GDN") == "1":
                from skyrl.backends.skyrl_train.isoexec.runtimes.vllm.gdn_gptmodel import (
                    swap_gdn_core,
                )

                n_gdn = swap_gdn_core(self.gpt, vllm_config=vllm_config)
                if n_gdn == 0:
                    # Zero GDN layers means every layer came out dense: a different model from the
                    # one the checkpoint describes, which would run happily and generate gibberish.
                    raise RuntimeError(
                        "[isoexec-gdn] SKYRL_ISOEXEC_GDN=1 but the Megatron GPTModel has no GatedDeltaNet "
                        "layers. The no-TE local layer spec built dense attention for every layer. A "
                        "hybrid no-TE spec (GDN on 3 of 4 layers) is required."
                    )

            # Fuse `F.rms_norm(x) * (1.0 + weight)` and hoist the add to the weight-sync boundary.
            # Instance-level rebinds only: hoisting gamma trainer-side would detach `weight`'s
            # gradient path. Self-gates on SKYRL_ISOEXEC_GDN_FUSED_OUTNORM.
            from skyrl.backends.skyrl_train.isoexec.ops.norms.fused_outnorm import (
                install_engine_fused_norms,
            )

            # The count distinguishes an installed twin from a flag that rebound nothing.
            self._ix_fused_norms = install_engine_fused_norms(self.gpt)

            # RoPE-by-absolute-position: GPTModel computes RoPE for sequence-index 0..L-1, but paged
            # decode feeds 1-token inputs whose true position is N, so index by vLLM's `positions`.
            max_pos = int(getattr(vllm_config.model_config, "max_model_len", 8192))
            self._rope = _PositionIndexedRoPE(self.gpt.rotary_pos_emb, max_pos)
            self.gpt.rotary_pos_emb = self._rope

            # Fuse the attention RoPE and hoist cos/sin out of the per-layer recompute. The patch
            # point is a module global, so the trainer must never reach this line.
            from skyrl.backends.skyrl_train.isoexec.ops.rope.rope_fused import (
                install_engine_fused_rope,
            )
            from skyrl.backends.skyrl_train.isoexec.runtimes.megatron.megatron_patches import (
                is_rope_fp32_installed,
            )

            # The op refuses to fuse over the megatron fp32-rope patch (different rounding chain).
            # The adapter reads that guard state and passes it down: ops must never import a runtime.
            install_engine_fused_rope(fp32_rope_installed=is_rope_fp32_installed())

            # Replace megatron's fused-QKV gather over the full TP group with a subgroup all-gather,
            # byte-identical to gather-then-slice under contiguous rank-ordered sharding. Installed
            # at model build because it WARMS a sub-process group, and a communicator built lazily
            # under CUDA-graph capture is fatal.
            from skyrl.backends.skyrl_train.isoexec.ops.attention.qkv_subgroup_gather import (
                install_engine_qkv_subgroup_ag,
            )

            install_engine_qkv_subgroup_ag(self.gpt, side="ENGINE")

            # Sampler patches HERE, in the worker process where the sampler runs; the engine actor
            # does not reach it. Two logprob sites are patched but only the V1
            # Sampler.compute_logprobs hook executes -- the V2 rebind is inert on this runner.
            if self._local_spec:
                try:
                    from skyrl.backends.skyrl_train.isoexec.runtimes.vllm.vllm_patches import (
                        patch_vllm_logprobs_batch_invariant,
                        patch_vllm_sampler_logprobs_rowinv,
                        patch_vllm_sampler_temperature,
                    )

                    patch_vllm_logprobs_batch_invariant()
                    patch_vllm_sampler_logprobs_rowinv()
                    patch_vllm_sampler_temperature()
                    _logprob_patched = True
                except Exception as _e:  # pragma: no cover
                    _logprob_patched = False
                    print(f"[ISOEXEC-WRAP] logprob patch failed: {type(_e).__name__}: {_e}", flush=True)
            else:
                _logprob_patched = False
            # Record what each family actually bound and compare against this process's contract.
            # Always build the worker's contract: the create_receiver handshake reads it, and a
            # worker without one silently skips the check.
            from ...core.process_contract import get_process_contract

            get_process_contract(model_path)
            if (
                os.environ.get("SKYRL_ISOEXEC_ENGINE_NCCL_UNPIN", "0") == "1"
                and self._tp_size > 1
                and os.environ.get("SKYRL_ISOEXEC_NCCL_TRANSPORT_BOUNDARY_REQUIREMENTS", "").strip()
            ):
                _assert_engine_nccl_manifest(model_path)
            _record_engine_install_fingerprint(self, cfg, logprob_patched=_logprob_patched)

        # run_install: build_contract -> check_all_claims -> _isoexec_install() -> INSTALL boundary.
        # gdn.state is EXCEPTIONS-listed there; it is recorded at first forward instead.
        from ...core.adapter import set_process_adapter
        from .adapter import VLLMContractAdapter

        set_process_adapter(
            VLLMContractAdapter(
                model_path,
                vllm_config=vllm_config,
                mp=mp,
                tp_size=self._tp_size,
                install_fn=_isoexec_install,
                model_fn=lambda: self.gpt,
            )
        ).run_install()

    def embed_input_ids(self, input_ids):
        # vLLM VllmModel protocol requires this exact name.
        return self.gpt.embedding(input_ids=input_ids.unsqueeze(0), position_ids=None)

    def get_input_embeddings(self, input_ids):
        return self.embed_input_ids(input_ids)

    def forward(self, input_ids=None, positions=None, inputs_embeds=None, **kwargs):
        # vLLM feeds MRoPE positions of shape [3, T] while the text bridge uses 1-D RoPE. For a
        # text-only request all three rows are identical, so collapse and refuse anything else.
        if positions is not None and positions.ndim == 2 and positions.shape[0] == 3:
            # `torch.equal` is a D2H sync and illegal under stream capture. Skipping it there is
            # safe: the guard is about the request TYPE, and every real forward still checks.
            if not torch.cuda.is_current_stream_capturing() and not (
                torch.equal(positions[0], positions[1]) and torch.equal(positions[0], positions[2])
            ):
                raise RuntimeError(
                    "[isoexec] MRoPE sections differ -- this request carries image/video positions. "
                    "The IsoExec GPTModel path is text-only."
                )
            positions = positions[0]

        # vLLM varlen [total_tokens] -> Megatron [b=1, seq]. The swapped paged attention ignores
        # attention_mask and uses vLLM's own metadata.
        tokens = input_ids.unsqueeze(0)
        pos = positions.unsqueeze(0)
        self._rope.set_positions(positions.reshape(-1))  # absolute positions for RoPE
        out = self.gpt(input_ids=tokens, position_ids=pos, attention_mask=None)
        # post_process=True gives logits as [b, s, vocab]; False gives hidden states as [s, b, h].
        # b == 1 either way, so the same reshape yields token-major rows.
        if out.dim() == 3:
            out = out.reshape(-1, out.shape[-1])
        return out

    def compute_logits(self, hidden_states, sampling_metadata=None):
        if not self._split_lm_head:
            return hidden_states  # forward already produced the logits
        # Replicates GPTModel._postprocess on the sampled rows only. Bitwise equality with the fused
        # path relies on the matmul being M-invariant, which batch invariance provides.
        core = self._gpt_core
        output_weight = core.shared_embedding_or_output_weight() if core.share_embeddings_and_output_weights else None
        # Come back down if anything upstream upcast: bf16 -> fp32 -> bf16 is exactly lossless.
        w_dtype = (output_weight if output_weight is not None else core.output_layer.weight).dtype
        if hidden_states.dtype != w_dtype:
            hidden_states = hidden_states.to(w_dtype)
        logits, _ = core.output_layer(hidden_states, weight=output_weight, runtime_gather_output=True)
        return core._scale_logits(logits)

    def load_weights(self, weights_iter):
        # Native sync: copy native-named incoming params straight into self.gpt. At build time the
        # names are HF-checkpoint names and all miss, which is harmless. Always return the full
        # param-name set so vLLM's "all weights initialized" check passes.
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
        if loaded:
            self._isoexec_debug_set_step()
        return all_names

    def _isoexec_debug_set_step(self) -> None:
        """Key this engine's trace records to the trainer's optim_step counter (debug mode only).

        The engine has no step of its own, so the weight-sync count stands in. The train loop syncs
        once before the first optim_step, so the Nth sync carries the weights of optim_step N-1;
        that offset is applied here. The first sync stays unkeyed, matching the trainer.
        """
        try:
            from ...debug.trace import enabled

            if not enabled():
                return
            from ...debug import set_step

            self._isoexec_weight_syncs = syncs = getattr(self, "_isoexec_weight_syncs", 0) + 1
            if syncs > 1:
                set_step(syncs - 1)
        except Exception as e:  # noqa: BLE001 -- diagnostics never fail a weight sync
            logger.warning("[ISOEXEC-DEBUG] engine set_step skipped: %s: %s", type(e).__name__, e)


class GPTModelVLLMHybridWrapper(GPTModelVLLMWrapper):
    """The wrapper for GatedDeltaNet hybrids (Qwen3.5). Identical compute; only the flag differs.

    ``is_hybrid`` is read off the CLASS before any instance exists and cached to disk per class
    name, hence a separate class rather than an env-dependent attribute on the dense one.
    """

    is_hybrid = True

    @classmethod
    def get_mamba_state_copy_func(cls):
        """(conv, ssm) slice layout of one mamba block; only consulted under mamba_cache_mode
        'align'. The native GDN specs apply since our state layers use the same shape calculator."""
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
    """Build and validate the active engine process's cap-aware contract.

    vLLM Ray workers are distinct processes, so a contract cached by the engine actor is not visible
    here; build it from the model path rather than relying on ``cached_contract_view()``.
    """
    from ...core.process_contract import cached_contract_view, get_process_contract
    from ...ops.collectives.nccl_identity import assert_contract_matches

    impl_id, constants = _engine_nccl_runtime_identity()
    get_process_contract(model_path)
    assert_contract_matches(cached_contract_view(), ("engine_prefill", "engine_decode"), impl_id, constants)


def _record_engine_install_fingerprint(wrapper, cfg, *, logprob_patched: bool) -> None:
    """Record what the ENGINE adapter just installed, per op family, then log the comparison.

    Recorded here because the adapter is the one place that knows which side it is on. Each impl_id
    is read off the state the install actually reached, never off the manifest or a launcher's
    intent; a family that did not install records ``NOT_INSTALLED``. Fail-soft.
    """
    try:
        from ...core.adapter import live_pins, log_unreported_pins
        from ...core.fingerprint import (
            ENGINE_SITES,
            NOT_INSTALLED,
            log_fingerprint_once,
            record_installs,
        )
        from ...core.process_contract import cached_contract_view
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

        # ENGINE logprobs: attest the site that EXECUTES on this runner, never a patch's intent. The
        # record is read off the live V1 Sampler class; anything else records NOT_INSTALLED so the
        # comparator sees the disagreement rather than an attested fiction.
        try:
            from .vllm_patches import sampler_logprobs_hook_state

            _lp_state = sampler_logprobs_hook_state()
        except Exception as _lp_e:  # pragma: no cover - no evidence means no claim
            _lp_state = {"v1_hook_installed": False, "rowinv_available": False, "error": repr(_lp_e)}
        _rowinv_live = bool(_lp_state.get("v1_hook_installed")) and bool(_lp_state.get("rowinv_available"))
        record_installs(
            "logprobs.log_softmax",
            ENGINE_SITES,
            "rowinv_leaftree" if _rowinv_live else NOT_INSTALLED,
            # Pins read off the live install, not echoed from the contract.
            pinned=live_pins("logprobs.log_softmax") if _rowinv_live else None,
        )
        if not _rowinv_live:
            print(
                "[ISOEXEC-WRAP] logprobs.log_softmax attested NOT_INSTALLED at the ENGINE sites: "
                "the V1 Sampler.compute_logprobs hook is "
                f"{'absent' if not _lp_state.get('v1_hook_installed') else 'bound but the rowinv module is unimportable'} "
                f"(v2_patch_installed={bool(logprob_patched)} is NOT evidence -- that module never "
                f"executes on the V1 runner; state={_lp_state}). The engine serves vLLM's stock "
                "log_softmax.",
                flush=True,
            )
        record_installs("logprobs.lm_head_slice", ENGINE_SITES, "sampled_rows")

        # Collectives exist only at TP>1; at TP=1 they have no site and get no record.
        if int(getattr(wrapper, "_tp_size", 1) or 1) > 1:
            from ...ops.collectives.pik_tp_invariant import pik_enabled

            _pik = "pik_tree" if pik_enabled() else NOT_INSTALLED
            # Pins off the ReductionPlan the install actually built, not the env vars that asked for
            # it, so a plan built differently from the flag stays visible.
            record_installs(
                "collectives.tree_all_reduce",
                ENGINE_SITES,
                _pik,
                pinned=live_pins("collectives.tree_all_reduce") if pik_enabled() else None,
            )
            record_installs("collectives.row_parallel", ENGINE_SITES, _pik)
            # Whether the unpin actually happened is otherwise visible only as a worker print.
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

            record_installs(
                "moe.combine",
                ENGINE_SITES,
                "pik_leaf_tree" if _moe_pik_fc2_on() else NOT_INSTALLED,
                pinned=live_pins("moe.combine") if _moe_pik_fc2_on() else None,
            )
            record_installs("moe.weights", ENGINE_SITES, "fused_buffer")
            from ...ops.moe.moe_fused_experts import _fused_epilogue_on

            record_installs("moe.epilogue", ENGINE_SITES, "fused_swiglu" if _fused_epilogue_on() else NOT_INSTALLED)
            record_installs("moe.blockmap", ENGINE_SITES, "fused" if fused_blockmap_enabled() else NOT_INSTALLED)

        # gdn.* records itself where it binds: its state core is built at the first
        # metadata-bearing forward, after this point.
        _view = cached_contract_view()
        log_fingerprint_once(_view, tag="engine_install")
        log_unreported_pins(_view)
    except Exception as e:  # pragma: no cover - never fatal
        logger.warning(f"[ISOEXEC-FINGERPRINT] engine install fingerprint skipped: {e}")


def _wrapper_import_path() -> str:
    """vLLM's ``module:ClassName`` string form for this process's capability tuple."""
    from .model_classes import import_path_for, resolve_capabilities

    return import_path_for(resolve_capabilities())


_WRAPPER_IMPORT_PATH = _wrapper_import_path()


def register_gptmodel_to_vllm(model_path: str | None = None):
    """Register the GPTModel-backed wrapper with vLLM. Call before engine init.

    Uses vLLM's STRING registration form so the registration survives across worker subprocesses.
    Set ``hf_overrides={"architectures": [VLLM_MODEL_NAME]}`` so vLLM builds this class.
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
        # answers differ; the compute is the same.
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

    # Hybrid (GDN) models need vLLM's hybrid config pass to set `cache_config.mamba_block_size` and
    # reconcile page sizes. vLLM selects it by ARCHITECTURE NAME, which `hf_overrides` just changed.
    if needs_hybrid_config_pass(caps):
        from vllm.model_executor.models.config import (
            MODELS_CONFIG_MAP,
            HybridAttentionMambaModelConfig,
        )

        MODELS_CONFIG_MAP.setdefault(VLLM_MODEL_NAME, HybridAttentionMambaModelConfig)
        logger.info("[isoexec] %s -> HybridAttentionMambaModelConfig (mamba_block_size)", VLLM_MODEL_NAME)
    return VLLM_MODEL_NAME


def find_inprocess_gptmodel(llm):
    """Reach the in-process GPTModelVLLMWrapper inside a vLLM LLM (VLLM_ENABLE_V1_MULTIPROCESSING=0)."""
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
