import os
import shutil
from collections import defaultdict
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

# isort: off
# ---------------------------------------------------------------------------------------------
# ORDER-CRITICAL BLOCK -- the import sorter must not touch it (hence `isort: off` / `isort: on`).
#
# The isoexec package MUST be imported before ANY megatron import below. Two separate things break
# if it is not:
#
#   1. The no-TransformerEngine import guard. megatron-bridge imports the peft/LoRA chain EAGERLY
#      at `megatron.bridge.__init__` (diffusion -> bailing -> model_bridge -> peft_bridge ->
#      canonical_lora -> lora_layers -> `import transformer_engine.pytorch`), so on the no-TE
#      stack the AutoBridge import CRASHES at module load unless the guard has already patched
#      those files.
#   2. The `fla` shim, needed by GDN models (Qwen3.5 / Qwen3-Next).
#      `megatron.core.ssm.gated_delta_net` evaluates `HAVE_FLA` at MODULE IMPORT TIME. If it is
#      imported first, HAVE_FLA is baked False permanently and `GatedDeltaNet.__init__` raises
#      "FLA is not installed. Please install it with `pip install flash-linear-attention`" --
#      even though the shim installs correctly moments later and every other symbol resolves.
#      That failure mode is confusing precisely because the shim logs success.
#
# This ordering has regressed once already: ruff's isort (`extend-select = ["I"]` with
# known-first-party = ["skyrl"]) sorts third-party `megatron` ABOVE first-party `skyrl`, which
# reintroduces the bug the comment below was written to prevent. Safe on the production TE stack
# -- the guard is a no-op when TE is present.
# ---------------------------------------------------------------------------------------------
import skyrl.backends.skyrl_train.isoexec  # noqa: F401,E402

import megatron.core.parallel_state as mpu  # noqa: E402
from megatron.bridge import AutoBridge  # noqa: E402

# NOTE: megatron-bridge's LoRA layers hard-import `transformer_engine` at module load
# (peft/lora_layers.py). On the no-TE nightly/IsoExec stack TE is intentionally absent
# (so megatron-core's HAVE_TE graceful fallback engages), so these are imported lazily
# inside configure_lora() — the only place they are used — to keep module import TE-free.
from megatron.core.optimizer import ChainedOptimizer, DistributedOptimizer  # noqa: E402
from megatron.core.optimizer_param_scheduler import (
    OptimizerParamScheduler,  # noqa: E402
)

# isort: on

import ray  # noqa: E402
import torch  # noqa: E402
import torch.distributed  # noqa: E402
import torch.nn as nn  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from loguru import logger  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from transformers import AutoConfig  # noqa: E402

from skyrl.backends.skyrl_train.distributed.dispatch import MeshRank, WorkerOutput
from skyrl.backends.skyrl_train.distributed.megatron.megatron_strategy import (
    MegatronStrategy,
)
from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
    broadcast_object_across_pp_ranks,
    freeze_moe_router,
    get_model_config,
    get_moe_metrics,
    print_model_size,
)
from skyrl.backends.skyrl_train.distributed.megatron.optimizer import (
    get_megatron_optimizer,
    get_megatron_optimizer_param_scheduler,
    init_megatron_optim_config,
)
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    SKYRL_LORA_ADAPTER_NAME,
)
from skyrl.backends.skyrl_train.training_batch import (
    TrainingInputBatch,
    TrainingOutputBatch,
)
from skyrl.backends.skyrl_train.utils.ep_balance import (
    ep_balance_enabled,
    sort_shard_rows_by_length,
)
from skyrl.backends.skyrl_train.utils.profiler import Profiler
from skyrl.backends.skyrl_train.weight_sync import (
    LoraLoadRequest,
    WeightChunk,
    WeightExtractor,
)
from skyrl.backends.skyrl_train.workers.megatron.adapter_store import (
    AdapterStore,
    LoraSignature,
    iter_opts,
)
from skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper import (
    MegatronModelWrapper,
)
from skyrl.backends.skyrl_train.workers.worker import (
    CriticWorkerBase,
    PolicyWorkerBase,
    RefWorkerBase,
)
from skyrl.backends.skyrl_train.workers.worker_utils import (
    BaseBatchIterator,
    BatchIterator,
    TokenBasedBatchIterator,
    all_reduce_metrics,
    get_microbatch_iterator,
    reduce_metrics,
    resolve_forward_token_budget,
)
from skyrl.env_vars import SKYRL_WORKER_NCCL_TIMEOUT_IN_S
from skyrl.train.config.config import MegatronDDPConfig, get_config_as_dict
from skyrl.train.utils.utils import str_to_torch_dtype, update_model_config
from skyrl.utils.tok import get_tokenizer

if TYPE_CHECKING:
    from skyrl.backends.skyrl_train.inference_engines.base import (
        InferenceEngineInterface,
    )
    from skyrl.train.config.config import InferenceEngineConfig

import skyrl.backends.skyrl_train.workers.megatron.model_bridges as _model_bridges  # noqa: F401  # register extra bridges
from skyrl.backends.skyrl_train.workers.megatron.model_bridges import (
    maybe_force_qwen35_text_bridge,
)


def _isoexec_validate_nccl_transport_boundary(boundary: str) -> dict:
    """Fail closed on missing real physical-group traffic when enforcement is active."""

    mode = os.environ.get("SKYRL_ISOEXEC_NCCL_CAPABILITY_MODE", "off").strip().lower() or "off"
    if mode == "off" or not os.environ.get("SKYRL_ISOEXEC_NCCL_TRANSPORT_BOUNDARY_REQUIREMENTS", "").strip():
        return {}
    from skyrl.backends.skyrl_train.isoexec.runtimes.megatron.nccl_transport_capabilities import (
        validate_transport_engagement,
    )

    return validate_transport_engagement(boundary)


class MegatronWeightExtractor(WeightExtractor):
    """Extracts weights from Megatron model-parallel models.

    Uses Megatron's bridge to export weights in HuggingFace format.

    Args:
        bridge: Megatron AutoBridge instance for weight conversion
        actor_module: The actor module to extract weights from
        enable_bucketing: If True, group parameters into size-based buckets for packing
        bucket_size_threshold_GB: Size threshold in GB for bucketing (only used if enable_bucketing=True)
        training_dtype: Training dtype for size calculation (only used if enable_bucketing=True)
    """

    def __init__(
        self,
        bridge,
        actor_module,
        enable_bucketing: bool = False,
        bucket_size_threshold_GB: float = 1.0,
        training_dtype: torch.dtype = torch.bfloat16,
    ):
        self.bridge = bridge
        self.actor_module = actor_module
        self.enable_bucketing = enable_bucketing
        self.bucket_size_threshold_GB = bucket_size_threshold_GB
        self.training_dtype = training_dtype

        # Defer bucket init to first extract_weights call.
        # At __init__ time the model may be CPU-offloaded (colocate_all),
        # so param.numel()==0 and bucketing collapses to a single bucket.
        # By the time extract_weights runs, the dispatch has already
        # called prepare_for_weight_sync → _ensure_on_gpu.
        self.bucket_index_groups = None
        self._buckets_initialized = False

    def _init_param_buckets(self):
        """Compute bucket boundaries (index groups) from parameter sizes.

        Only the bucket *structure* (which task indices go in which bucket) is
        persisted.  The actual ``WeightConversionTask`` objects are rebuilt on
        every ``extract_weights`` call so that mapping objects start with clean
        PP-collective caches, avoiding stale cached state across offload/reload
        and training cycles.

        Tasks that participate in grouped export (e.g., fused MoE expert
        weights) are collected first and placed into dedicated buckets so that
        all tasks sharing the same ``group_key`` end up in a single
        ``export_hf_weights`` call.  The bridge's
        ``_accumulate_grouped_export`` requires every task for a group to be
        present in one call; splitting them across buckets causes expert
        weights to never be yielded.
        """
        weight_conversion_tasks = self.bridge.get_conversion_tasks(self.actor_module)

        def calculate_size_in_bytes(param, tp_size, ep_size):
            if param is None:
                size_in_bytes = None
            else:
                prec_to_bytes = {
                    torch.bfloat16: 2,
                    torch.float32: 4,
                }
                scale = prec_to_bytes[self.training_dtype] / prec_to_bytes[param.dtype]
                size_in_bytes = param.element_size() * param.numel() * tp_size * ep_size * scale
            return broadcast_object_across_pp_ranks(size_in_bytes)

        sizes = [
            calculate_size_in_bytes(
                task.param_weight,
                task.mapping.tp_size,
                task.mapping.ep_size if task.mapping.is_expert else 1,
            )
            for task in weight_conversion_tasks
        ]

        # ---- Separate grouped-export tasks from regular tasks ----
        # Grouped-export tasks (is_grouped_export=True, e.g. FusedGatedExpertMapping /
        # FusedExpertMapping for MoE expert weights) must ALL be present in a single
        # export_hf_weights call for the bridge's _accumulate_grouped_export to produce
        # the fused tensor.  Collect them by group_key and give each group its own bucket.
        grouped_task_indices: dict[str, list[int]] = {}  # group_key -> list of task indices
        regular_task_indices: list[int] = []

        for idx, task in enumerate(weight_conversion_tasks):
            if getattr(task.mapping, "is_grouped_export", False):
                gk = getattr(task.mapping, "group_key", None)
                grouped_task_indices.setdefault(gk, []).append(idx)
            else:
                regular_task_indices.append(idx)

        self.bucket_index_groups: list[list[int]] = []

        # Pack grouped-export tasks into buckets by size, keeping each
        # group_key's tasks together (they must not be split across calls).
        curr_size = 0
        threshold = self.bucket_size_threshold_GB * 1024**3
        for gk, indices in grouped_task_indices.items():
            group_size = sum(sizes[idx] for idx in indices if sizes[idx] is not None)
            if not self.bucket_index_groups or curr_size + group_size > threshold:
                self.bucket_index_groups.append([])
                curr_size = 0
            self.bucket_index_groups[-1].extend(indices)
            curr_size += group_size

        # Bucket regular (non-grouped) tasks by size as before.
        if regular_task_indices:
            self.bucket_index_groups.append([])
            curr_size = 0
            for idx in regular_task_indices:
                size = sizes[idx]
                if curr_size + size > threshold:
                    self.bucket_index_groups.append([])
                    curr_size = 0
                self.bucket_index_groups[-1].append(idx)
                curr_size += size

    def get_weight_metadata(self, dtype: torch.dtype) -> dict:
        """Return weight metadata without keeping tensors in memory.

        On first call, runs export_hf_weights to discover HF names and shapes
        (tensors are discarded immediately). Result is cached for subsequent calls.
        TODO (aaron): find a better way to get all metadata without materializing tensors.
        """
        if hasattr(self, "_weight_metadata_cache"):
            return self._weight_metadata_cache

        self._ensure_buckets_initialized()
        names = []
        dtype_names = []
        shapes = []
        dtype_name = str(dtype).split(".")[-1]

        # SkyRL-IsoExec: native metadata (no HF) so it matches extract_weights' native chunks AND
        # the engine GPTModel's own param names 1:1. Must be consistent with extract_weights below.
        if os.environ.get("SKYRL_ISOEXEC") == "1":
            # NATIVE metadata. Under mismatched TP (engine TP < trainer TP) the SENT tensors are
            # resharded to the engine's full layout, so report the resharded shapes -- computed
            # ANALYTICALLY (no all-gather here; metadata may run rank-0-only). Must match what
            # extract_weights(reshard=True) sends. See native_weight_sync.native_resharded_metadata.
            from skyrl.backends.skyrl_train.isoexec.sync.native_weight_sync import (
                native_resharded_metadata,
            )

            for name, dt_name, shape in native_resharded_metadata(self.actor_module, dtype=dtype):
                names.append(name)
                dtype_names.append(dt_name)
                shapes.append(list(shape))
            self._weight_metadata_cache = {"names": names, "dtype_names": dtype_names, "shapes": shapes}
            print(f"[ISOEXEC-SENDER] get_weight_metadata NATIVE: {len(names)} names (e.g. {names[:2]})", flush=True)
            return self._weight_metadata_cache

        # Collect parameter metadata in the same order
        # as provided by `.extract_weights`.
        if not self.enable_bucketing:
            for name, tensor in self.bridge.export_hf_weights(
                self.actor_module,
                show_progress=False,
                conversion_tasks=None,
            ):
                names.append(name)
                dtype_names.append(dtype_name)
                shapes.append(list(tensor.shape))
                del tensor
        else:
            # Build fresh tasks each sync so mapping objects have clean
            # PP-collective caches; reuse the pre-computed bucket structure.
            fresh_tasks = self.bridge.get_conversion_tasks(self.actor_module)
            for index_group in self.bucket_index_groups:
                bucket_tasks = [fresh_tasks[i] for i in index_group]
                for name, tensor in self.bridge.export_hf_weights(
                    self.actor_module,
                    show_progress=False,
                    conversion_tasks=bucket_tasks,
                ):
                    names.append(name)
                    shapes.append(list(tensor.shape))
                    dtype_names.append(dtype_name)
                    del tensor

        self._weight_metadata_cache = {"names": names, "dtype_names": dtype_names, "shapes": shapes}
        return self._weight_metadata_cache

    def _ensure_buckets_initialized(self):
        """Lazily initialize param buckets on first use (model must be on GPU)."""
        if self._buckets_initialized:
            return
        if self.enable_bucketing:
            self._init_param_buckets()
        self._buckets_initialized = True

    def extract_weights(self, dtype: torch.dtype):
        """Extract weights from Megatron model.

        Args:
            dtype: Target dtype for inference

        Yields:
            WeightChunk objects (one per parameter, or one per bucket if bucketing enabled)
        """
        self._ensure_buckets_initialized()
        device = torch.cuda.current_device()

        # SkyRL-IsoExec: the rollout runs the SAME GPTModel, so sync NATIVE params (no HF
        # conversion). The receiver copies them straight into GPTModelVLLMWrapper.gpt by name.
        # At TP>1 named_parameters are the per-rank shards, which is exactly what the colocated
        # engine rank needs (the CUDA-IPC transport routes by physical GPU).
        #
        # BUCKET many params per chunk: each chunk costs a pack + IPC handle + all_gather_object +
        # collective_rpc + barriers across all engine workers. One-chunk-per-param was fine for
        # dense 0.8B (230 params) but Qwen3.5-35B-A3B's SequentialMLP has 20,943 -- measured ~90
        # minutes per sync. The IPC request natively carries (names, shapes, sizes) lists, so
        # packing is purely a sender-side choice.
        if os.environ.get("SKYRL_ISOEXEC") == "1":
            from skyrl.backends.skyrl_train.isoexec.sync.native_weight_sync import (
                extract_native_weights,
                is_synced_buffer,
            )

            bucket_bytes = int(os.environ.get("SKYRL_ISOEXEC_SYNC_BUCKET_MB", "512")) * 1024 * 1024
            names, dtypes_l, shapes, tensors, cur = [], [], [], [], 0
            _n_params, _n_chunks = 0, 0

            def _flush():
                nonlocal names, dtypes_l, shapes, tensors, cur, _n_chunks
                if not names:
                    return None
                chunk = WeightChunk(names=names, dtypes=dtypes_l, shapes=shapes, tensors=tensors)
                names, dtypes_l, shapes, tensors, cur = [], [], [], [], 0
                _n_chunks += 1
                return chunk

            # reshard=True: under mismatched TP (engine TP < trainer TP) gather each TP-sharded
            # param across the trainer TP group to the engine's full layout (all-gather collective;
            # runs on every trainer rank here). No-op at matched TP. See native_weight_sync.
            for name, tensor in extract_native_weights(self.actor_module, dtype=dtype, reshard=True):
                # PRESERVE THE NATIVE DTYPE OF AN ALLOWLISTED BUFFER. extract_native_weights
                # deliberately yields those un-cast (native_weight_sync.py: "NATIVE DTYPE for an
                # allowlisted buffer") because `mlp.router.expert_bias` is fp32 and is ADDED TO THE
                # ROUTING SCORES -- bf16-rounding it flips top-k membership, which is a different
                # model rather than a rounding difference. Casting everything to `dtype` here threw
                # that guarantee away one line later: the receiver copies into an fp32 destination,
                # so the bf16 value was merely upcast and the rounding was already baked in.
                # MEASURED on GLM-4.7-Flash: engine expert_bias abs-sum 579.312500 vs trainer
                # 578.344131, and the routing-map index checksum differed by 206 of 7699 at layer 1
                # -- the engine was routing tokens to different experts, worth 0.043 on the gate.
                # `native_resharded_metadata` already advertises these buffers in their native
                # dtype, so sending bf16 here also disagreed with the metadata the receiver sizes on.
                t_dtype = tensor.dtype if is_synced_buffer(name) else dtype
                tensor = tensor.to(device=device, dtype=t_dtype, non_blocking=True)
                sz = tensor.numel() * tensor.element_size()
                # Keep every bucket DTYPE-HOMOGENEOUS as well as size-bounded: the CUDA-IPC
                # transport packs a chunk into ONE flat buffer of a single dtype, so a native-fp32
                # buffer sharing a bucket with bf16 params would be rounded at pack time no matter
                # what this loop yields. Flushing on a dtype change costs one extra small chunk per
                # sync (the allowlisted buffers are a few KiB in total).
                if names and (cur + sz > bucket_bytes or t_dtype != tensors[0].dtype):
                    yield _flush()
                names.append(name)
                dtypes_l.append(str(t_dtype))
                shapes.append(list(tensor.shape))
                tensors.append(tensor)
                cur += sz
                _n_params += 1
                if _n_params % 3000 == 0:
                    print(
                        f"[ISOEXEC-SENDER-MEM] n={_n_params} alloc={torch.cuda.memory_allocated() / 2**30:.1f}GiB "
                        f"reserved={torch.cuda.memory_reserved() / 2**30:.1f}GiB",
                        flush=True,
                    )
            last = _flush()
            if last is not None:
                yield last
            print(
                f"[ISOEXEC-SENDER] extract_weights NATIVE yielded {_n_params} params in "
                f"{_n_chunks} bucketed chunks (<= {bucket_bytes >> 20} MiB each)",
                flush=True,
            )
            return

        if not self.enable_bucketing:
            # No bucketing: yield one chunk per parameter
            hf_params_generator = self.bridge.export_hf_weights(
                self.actor_module,
                show_progress=False,
                conversion_tasks=None,
            )

            for name, tensor in hf_params_generator:
                tensor = tensor.to(device=device, dtype=dtype, non_blocking=True)

                yield WeightChunk(
                    names=[name],
                    dtypes=[str(dtype)],
                    shapes=[list(tensor.shape)],
                    tensors=[tensor],
                )
        else:
            # Build fresh tasks each sync so mapping objects have clean
            # PP-collective caches; reuse the pre-computed bucket structure.
            fresh_tasks = self.bridge.get_conversion_tasks(self.actor_module)

            for index_group in self.bucket_index_groups:
                bucket_tasks = [fresh_tasks[i] for i in index_group]
                hf_params_generator = self.bridge.export_hf_weights(
                    self.actor_module,
                    show_progress=False,
                    conversion_tasks=bucket_tasks,
                )

                # Collect all parameters in this bucket into one chunk
                names = []
                dtypes_list = []
                shapes = []
                tensors = []

                for name, tensor in hf_params_generator:
                    # Move to device and convert dtype
                    tensor = tensor.to(device=device, dtype=dtype, non_blocking=True)

                    names.append(name)
                    dtypes_list.append(str(dtype))
                    shapes.append(list(tensor.shape))
                    tensors.append(tensor)

                # Yield one chunk containing all parameters in this bucket
                if tensors:
                    yield WeightChunk(
                        names=names,
                        dtypes=dtypes_list,
                        shapes=shapes,
                        tensors=tensors,
                    )


class MegatronWorker:
    def init_configs(
        self,
        model_path,
        megatron_config,
        model_config_kwargs,
        transformer_config_kwargs,
        bf16=True,
        flash_attn=False,
        lora_config=None,
        language_model_only=False,
    ):
        """
        Initialize the Megatron-Bridge bridge and provider objects + hf_config and tokenizer
        """
        tokenizer = get_tokenizer(model_path, trust_remote_code=True)
        hf_config_original = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        override_config_kwargs = {
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        override_config_kwargs.update(model_config_kwargs.get("model_config", {}))
        hf_config = update_model_config(hf_config_original, override_config_kwargs=override_config_kwargs)

        transformer_config_kwargs = (
            transformer_config_kwargs
            if isinstance(transformer_config_kwargs, dict)
            else OmegaConf.to_container(transformer_config_kwargs, resolve=True)
        )

        if not self.cfg.gradient_checkpointing:
            for key in ("recompute_granularity", "recompute_method", "recompute_num_layers"):
                transformer_config_kwargs[key] = None

        # SkyRL-IsoExec: Qwen3.5's GatedDeltaNet layers need the `fla` facade
        # (isoexec.runtimes.megatron.gdn_fla_shim)
        # or GatedDeltaNet.__init__ raises ImportError. The authoritative install happens at
        # `import skyrl.backends.skyrl_train.isoexec` (line ~18, before `from megatron.bridge import
        # AutoBridge`), because megatron binds chunk_gated_delta_rule at import time. This call is
        # the idempotent belt-and-braces for anyone who imports megatron.bridge first.
        if os.environ.get("SKYRL_ISOEXEC_GDN") == "1":
            from skyrl.backends.skyrl_train.isoexec import install_fla_shim
            from skyrl.backends.skyrl_train.isoexec.runtimes.megatron.gdn_hybrid_spec import (
                checkpoint_is_vl_named,
                patch_qwen35_bridge_for_local_spec,
            )

            install_fla_shim()
            # The released Qwen3.5 checkpoints store the LM under `model.language_model.`. Local
            # spec retargets TE-fused norm names to the separate exact norm owner and local
            # SequentialMLP experts.
            _hf_lm_prefix = "model.language_model." if checkpoint_is_vl_named(hf_config) else None
            patch_qwen35_bridge_for_local_spec(hf_lm_prefix=_hf_lm_prefix)

        bridge = AutoBridge.from_hf_pretrained(model_path, trust_remote_code=True)

        # For Qwen3.5, language_model_only routes to the native GPTModel + GDN
        # path (which supports sample packing) instead of the VL Qwen3VLModel
        # (which doesn't). Must run before to_megatron_provider; no-op otherwise.
        if language_model_only and maybe_force_qwen35_text_bridge(bridge, hf_config):
            logger.info(
                "language_model_only=True: forcing Qwen3.5 text->GPTModel bridge "
                "(native GDN thd packing path; vision tower dropped)"
            )

        provider = bridge.to_megatron_provider()

        # Disable MTP for training: its aux loss is unused, and under full
        # recompute its checkpointed forward passes packed_seq_params positionally
        # into tensor_parallel.checkpoint (tensors only), breaking packed-sequence
        # backward. Mirrors the MTP-disable in model_bridges.py.
        if getattr(provider, "mtp_num_layers", None):
            logger.info(f"Disabling MTP for training (mtp_num_layers={provider.mtp_num_layers} -> None)")
            provider.mtp_num_layers = None

        # Workaround for megatron-bridge CONFIG_MAPPING dropping None values:
        # MLA models like Moonlight-16B have q_lora_rank=None (no Q compression),
        # but CONFIG_MAPPING skips None so the MCoreMLATransformerConfig default
        # (512) is used instead, causing the wrong model architecture to be built.
        # see: https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c8eb587c5fd43163dbcd9c40980225b3fe1981f8/src/megatron/bridge/recipes/moonlight/moonlight_16b.py#L60
        if hasattr(provider, "q_lora_rank") and hasattr(hf_config, "q_lora_rank"):
            provider.q_lora_rank = hf_config.q_lora_rank

        # Workaround for transformers v5 moving rope_theta into rope_parameters
        # (previously it was a top-level config attribute). megatron-bridge's
        # CONFIG_MAPPING reads config.rope_theta which no longer exists in v5,
        # causing it to fall back to the default rotary_base of 10000.
        rope_params = getattr(hf_config, "rope_parameters", None) or getattr(hf_config, "rope_scaling", None)
        if isinstance(rope_params, dict) and "rope_theta" in rope_params:
            provider.rotary_base = rope_params["rope_theta"]

        provider.tensor_model_parallel_size = megatron_config.tensor_model_parallel_size
        provider.pipeline_model_parallel_size = megatron_config.pipeline_model_parallel_size
        provider.pipeline_dtype = torch.bfloat16 if bf16 else torch.float32
        provider.context_parallel_size = megatron_config.context_parallel_size
        provider.expert_model_parallel_size = megatron_config.expert_model_parallel_size
        provider.expert_tensor_parallel_size = megatron_config.expert_tensor_parallel_size
        provider.sequence_parallel = megatron_config.tensor_model_parallel_size > 1
        # SkyRL-IsoExec: sequence parallelism historically forced OFF for two recorded reasons.
        # One was real, one was not:
        #   * "trainer's per-token activations no longer match the engine's" -- a red herring.
        #     SP shards along the TOKEN axis; every token's full hidden vector lives on exactly
        #     one rank, all norms are per-token, and the only arithmetic the layout touches is
        #     the row-parallel combine. The engine never sees trainer activations, only logits/
        #     logprobs -- and those are per-token too.
        #   * the no-TE `WrappedTorchNorm` refuses SP ("sequence parallel not supported by torch
        #     LayerNorm") -- a grad-bookkeeping gap, not an arithmetic one (per-token norms are
        #     layout-independent; what is missing is the `sequence_parallel` param mark that
        #     makes finalize_model_grads SUM per-slice grads). Lifted by sp_norm_lift below.
        # SKYRL_ISOEXEC_TRAINER_SP=1 keeps SP ON by replacing the ONE arithmetic hazard -- the
        # NCCL reduce-scatter that would replace the row-parallel all-reduce -- with pik's
        # tree_reduce_scatter: the proven two-shot tree MINUS its trailing all-gather, so each
        # rank keeps its sequence slice of the SAME expression tree. SP-on trainer logits are
        # then BITWISE the SP-off trainer logits (the admission standard; see the
        # trainer_sp_battery). Requires SKYRL_ISOEXEC_PIK=1: without pik the row-parallel RS is
        # NCCL arithmetic and SP-on is silently non-invariant, so refuse rather than run wrong.
        if os.environ.get("SKYRL_ISOEXEC") == "1" and provider.sequence_parallel:
            if os.environ.get("SKYRL_ISOEXEC_TRAINER_SP") == "1":
                if os.environ.get("SKYRL_ISOEXEC_PIK") != "1":
                    raise RuntimeError(
                        "[ISOEXEC-TRAINER] SKYRL_ISOEXEC_TRAINER_SP=1 requires SKYRL_ISOEXEC_PIK=1: "
                        "without pik, sequence parallelism replaces the row-parallel all-reduce "
                        "with an NCCL reduce-scatter (different summation tree), which breaks "
                        "bitwise IsoExec. Enable pik or unset SKYRL_ISOEXEC_TRAINER_SP."
                    )
                from skyrl.backends.skyrl_train.isoexec.ops.norms.sp_norm_lift import (
                    install_trainer_sp_norm_lift,
                )

                install_trainer_sp_norm_lift()
                print(
                    "[ISOEXEC-TRAINER] sequence_parallel KEPT ON (SKYRL_ISOEXEC_TRAINER_SP=1): "
                    "row-parallel combines go through pik tree_reduce_scatter (bitwise vs SP-off), "
                    "norm/residual segments sequence-sharded across TP, no-TE norm SP refusal "
                    "lifted with grads marked for TP summation",
                    flush=True,
                )
            else:
                provider.sequence_parallel = False
                _tp = int(megatron_config.tensor_model_parallel_size)
                _ep = int(megatron_config.expert_model_parallel_size or 1)
                # An UNSET expert_tensor_parallel_size follows TP in megatron -- it is NOT 1. Only
                # an explicit 1 leaves the experts un-sharded across the TP group, so only that
                # value earns the warning below. Guessing here would make the banner lie.
                _etp_raw = megatron_config.expert_tensor_parallel_size
                _etp = int(_etp_raw) if _etp_raw is not None else _tp
                print(
                    "[ISOEXEC-TRAINER] sequence_parallel forced OFF (SKYRL_ISOEXEC_TRAINER_SP!=1; "
                    "the pik tree_reduce_scatter path exists but is opt-in)",
                    flush=True,
                )
                # THE PRICE OF THAT LINE, stated where it is paid. With SP off every TP rank holds
                # the FULL token set through the MoE, so at ETP=1 the router, the expert GEMMs and
                # the EP all-to-all payload are all TP-INVARIANT -- raising TP does not divide the
                # per-rank MoE work, it only divides DP, which MULTIPLIES the microbatch count per
                # rank (mb/rank = mini_batch * n_samples / dp, utils.py:104-105). So trainer cost
                # scales ~linearly with TP instead of staying flat. MEASURED on b32/n8 DAPO 35B,
                # EP=8/ETP=1, only TP differing: policy_train 698.6s at TP=2 (dp=4,
                # logs/prof_pair/isoexec.log) vs 1291.1s at TP=4 (dp=2, logs/fix2_tp4/run.log) =
                # 1.85x, while per-MICROBATCH GPU time was flat (10.07 -> 10.57 s, rank-0 traces).
                # The paired NATIVE run at the SAME TP=4 runs SP ON and its EP all-to-all payload is
                # 1/TP of ours (16,752 vs 67,008 rows/call), which is why its per-rank cost DOES
                # fall with TP. Do not read this banner as harmless: at ETP=1 it is the single
                # largest term in the trainer's TP scaling.
                if _tp > 1 and _etp == 1 and _ep > 1:
                    print(
                        f"[ISOEXEC-TRAINER] MESH ECONOMICS WARNING: TP={_tp} EP={_ep} ETP={_etp} with "
                        f"sequence_parallel OFF. Every TP rank carries the full token set through the "
                        f"MoE, so per-rank expert work and EP all-to-all bytes are TP-invariant while "
                        f"DP (and therefore the microbatch count per rank) scales as 1/TP -- expect "
                        f"policy_train ~{_tp / 2:.1f}x the TP=2 figure for the same batch. Either run "
                        f"TP=2 or validate + enable SKYRL_ISOEXEC_TRAINER_SP=1 (needs the trainer_sp "
                        f"battery extended to this model's MoE/GDN layers first).",
                        flush=True,
                    )
        provider.attention_backend = "flash" if flash_attn else "fused"
        provider.variable_seq_lengths = True
        provider.masked_softmax_fusion = True
        # SkyRL-IsoExec: the engine (GPTModelVLLMWrapper) builds with apply_rope_fusion=False so the
        # fp32-RoPE IsoExec patch applies. The trainer MUST match -- otherwise it uses the fused
        # (bf16) RoPE kernel which BYPASSES the patch, so trainer RoPE != engine RoPE and the
        # rollout_train logprob diff inflates. Force it off (and log) under IsoExec.
        if os.environ.get("SKYRL_ISOEXEC") == "1":
            provider.apply_rope_fusion = False
            # variable_seq_lengths=False is the FORWARD-parity force: the varlen/THD attention
            # path diverges from the engine's per-sequence vLLM paged attention -> inflates
            # rollout_train. masked_softmax_fusion likewise touches the forward. Requires
            # remove_microbatch_padding=false (padded BSHD microbatches).
            # gradient_accumulation_fusion=False is NOT forward-parity -- it is a pure backward
            # knob (megatron layers.py:482-508 reads it only to stash main_grad in ctx). It is
            # pinned False as TE/apex CONTAINMENT (bridge fusions.py:34 defaults it True whenever
            # TE or apex imports) and because without apex the
            # fused path raises at model build. Analysed 2026-08-12: enabling would reach only
            # ~8.5% of AccumulateGrad anyway (experts ride the batched-bmm path).
            provider.variable_seq_lengths = False
            provider.masked_softmax_fusion = False
            provider.gradient_accumulation_fusion = False
            # NOTE: transformer_config_kwargs is applied AFTER this block and can override these
            # keys silently -- the resolved-values banner below (post-kwargs) is the truthful one.
            print(
                f"[ISOEXEC-TRAINER] forced apply_rope_fusion=False variable_seq_lengths=False "
                f"masked_softmax_fusion=False gradient_accumulation_fusion=False "
                f"(pre-transformer_config_kwargs; see RESOLVED banner); "
                f"attention_backend={provider.attention_backend}",
                flush=True,
            )
        # SkyRL-IsoExec nightly (no-TE) stack: megatron-bridge's default dense layer spec is
        # hard-wired to TransformerEngine, which is intentionally absent here. Force Megatron's
        # LOCAL spec (plain torch SDPA / RMSNorm / F.linear) so the trainer GPTModel is the same
        # batch-invariant local-spec model the engine serves -- the basis for bitwise IsoExec.
        # local_layer_spec(config) -> get_gpt_layer_local_spec(normalization=config.normalization,
        # qk_layernorm=config.qk_layernorm, ...), so RMSNorm / no-qk-norm follow the HF config.
        # For a MoE provider this resolves to get_gpt_layer_local_spec(num_experts=...,
        # moe_grouped_gemm=False) -> MoELayer(TopKRouter, SequentialMLP) built from local modules,
        # keeping the model's own SelfAttention class; dense providers are unchanged.
        if os.environ.get("SKYRL_ISOEXEC_LOCAL_SPEC") == "1":
            from skyrl.backends.skyrl_train.isoexec import make_isoexec_local_layer_spec

            provider.transformer_layer_spec = make_isoexec_local_layer_spec(provider)
            print("[ISOEXEC-TRAINER] forced Megatron LOCAL layer spec (no TransformerEngine)", flush=True)
        # Apply explicit MoE config fields to the provider.
        # These replace the previously hardcoded values and can be further
        # overridden by transformer_config_kwargs if needed.
        provider.moe_token_dispatcher_type = megatron_config.moe_token_dispatcher_type
        provider.moe_router_load_balancing_type = megatron_config.moe_router_load_balancing_type
        provider.moe_aux_loss_coeff = megatron_config.moe_aux_loss_coeff
        provider.moe_router_dtype = megatron_config.moe_router_dtype
        provider.moe_grouped_gemm = megatron_config.moe_grouped_gemm
        if megatron_config.moe_router_score_function is not None:
            provider.moe_router_score_function = megatron_config.moe_router_score_function
        if megatron_config.moe_router_enable_expert_bias is not None:
            provider.moe_router_enable_expert_bias = megatron_config.moe_router_enable_expert_bias
        provider.moe_enable_routing_replay = megatron_config.moe_enable_routing_replay

        # Apply any additional transformer config kwargs (can override the above).
        for k, v in transformer_config_kwargs.items():
            setattr(provider, k, v)
        # Truthful (post-kwargs) banner for the IsoExec forced keys: the force block above prints
        # BEFORE kwargs application, so a kwargs override would make that banner lie (found
        # 2026-08-12 -- DEFAULT_TRANSFORMER_CONFIG_KWARGS already carries
        # gradient_accumulation_fusion, so the kwargs loop always rewrites it).
        if os.environ.get("SKYRL_ISOEXEC") == "1":
            print(
                f"[ISOEXEC-TRAINER] RESOLVED (post-kwargs): apply_rope_fusion={provider.apply_rope_fusion} "
                f"variable_seq_lengths={provider.variable_seq_lengths} "
                f"masked_softmax_fusion={provider.masked_softmax_fusion} "
                f"gradient_accumulation_fusion={provider.gradient_accumulation_fusion}",
                flush=True,
            )

        # SkyRL-IsoExec: pin the MoE recipe LAST so neither megatron_config nor
        # transformer_config_kwargs can reintroduce a batch-variant op (grouped GEMM, permute
        # fusion, token dropping). Also installs the deterministic combine / sorted router top-k.
        # Must be byte-for-byte the same forcing the engine applies in gptmodel_vllm. No-op dense.
        if os.environ.get("SKYRL_ISOEXEC_LOCAL_SPEC") == "1":
            from skyrl.backends.skyrl_train.isoexec import prepare_isoexec_moe

            prepare_isoexec_moe(provider, side="TRAINER")

        provider.finalize()

        self.provider = provider
        self.bridge = bridge

        # strategy.hf_config is the on-disk source-of-truth used by
        # save_hf_configs and must NOT carry runtime overrides like
        # mtp_num_layers=0; assign the un-mutated AutoConfig here.
        self.strategy.hf_config = hf_config_original
        self.tokenizer = tokenizer
        self.enable_router_replay = megatron_config.moe_enable_routing_replay

    def configure_lora(self, lora_config, lora_type: Optional[str] = "lora"):
        # Lazy import: megatron-bridge LoRA layers hard-import transformer_engine (absent on
        # the no-TE IsoExec stack). Only reached when LoRA/PEFT is actually configured.
        from megatron.bridge.peft.canonical_lora import CanonicalLoRA
        from megatron.bridge.peft.lora import LoRA

        if lora_type == "lora":
            self.lora_cls = LoRA(
                target_modules=(
                    ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
                    if lora_config.target_modules == "all-linear"
                    else lora_config.target_modules
                ),
                dim=lora_config.rank,
                alpha=lora_config.alpha,
                dropout=lora_config.dropout,
                lora_A_init_method=lora_config.init_method,
                lora_B_init_method="zero",
                exclude_modules=[] if lora_config.exclude_modules is None else lora_config.exclude_modules,
                lora_dtype=torch.bfloat16 if self.cfg.bf16 else torch.float32,
            )
        elif lora_type == "canonical_lora":
            self.lora_cls = CanonicalLoRA(
                target_modules=(
                    [
                        "linear_q",
                        "linear_k",
                        "linear_v",
                        "linear_proj",
                        "linear_fc1_up",
                        "linear_fc1_gate",
                        "linear_fc2",
                    ]
                    if lora_config.target_modules == "all-linear"
                    else lora_config.target_modules
                ),
                dim=lora_config.rank,
                alpha=lora_config.alpha,
                dropout=lora_config.dropout,
                lora_A_init_method=lora_config.init_method,
                lora_B_init_method="zero",
                exclude_modules=[] if lora_config.exclude_modules is None else lora_config.exclude_modules,
            )

    def make_megatron_module(
        self,
        wrap_with_ddp: bool = True,
        ddp_config: Optional[Union[MegatronDDPConfig, Dict[str, Any]]] = None,
        lora_config: Optional[Dict[str, Any]] = None,
        lora_type: Optional[str] = "lora",
        bf16: bool = True,
    ) -> List[nn.Module]:
        """
        Creates a megatron GPTModel (optionally DDP wrapped) using the bridge.
        """
        from megatron.core.distributed.distributed_data_parallel_config import (
            DistributedDataParallelConfig,
        )

        if lora_config is not None:
            self.configure_lora(lora_config, lora_type)

            def lora_pre_wrap_hook(model):
                lora_model = self.lora_cls(model, training=True)
                self.lora_cls.set_params_to_save(lora_model)

                return lora_model

            self.provider.register_pre_wrap_hook(lora_pre_wrap_hook)

        default_ddp_config = DistributedDataParallelConfig()
        if wrap_with_ddp:
            default_ddp_config.use_distributed_optimizer = True
        if ddp_config is not None:
            for k, v in get_config_as_dict(ddp_config).items():
                setattr(default_ddp_config, k, v)
        model = self.provider.provide_distributed_model(
            ddp_config=default_ddp_config, wrap_with_ddp=wrap_with_ddp, bf16=bf16
        )
        return model

    def _forward_logprobs(self, data: TrainingInputBatch) -> torch.Tensor:
        """Run a Megatron inference forward over ``data`` and return per-sample logprobs.

        Passes the full mini batch to ``MegatronModelWrapper.forward``. Supports token-based
        micro-batching via ``max_tokens_per_microbatch`` -- or ``max_tokens_per_microbatch_forward``
        when that is set, since this phase is ``no_grad`` and can afford larger bins than the
        training step (padding micro-batches to a uniform size as Megatron's pipeline schedule
        requires, then reordering back to input order).

        Returns:
            CPU tensor of shape ``[batch_size, response_length]`` in original sample order.
        """
        from skyrl.backends.skyrl_train.utils.replay_utils import clear_router_replay

        # The FORWARD token budget: `max_tokens_per_microbatch_forward` when set, else the training
        # budget verbatim (`resolve_forward_token_budget`, shared with the FSDP/base worker so there
        # is exactly one rule and exactly one packer). This phase is `no_grad` and keeps no
        # activations, so it can carry bins the training step could not.
        forward_token_budget = resolve_forward_token_budget(self.cfg)
        use_token_batching = forward_token_budget > 0

        if use_token_batching:
            microbatch_iterator = get_microbatch_iterator(
                data,
                micro_batch_size=self.cfg.micro_forward_batch_size_per_gpu,
                max_tokens_per_microbatch=forward_token_budget,
            )
        else:
            microbatch_iterator = None

        # Build micro-batch dicts expected by policy.forward_mini_batch
        micro_dicts = []
        device = torch.cuda.current_device()
        if microbatch_iterator is not None:
            micro_batches = microbatch_iterator
        else:
            micro_batches = data.chunk(self.cfg.micro_forward_batch_size_per_gpu)

        for micro in micro_batches:
            micro.to(device)
            attention_mask = micro["attention_mask"]
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            rollout_expert_indices = micro.get("rollout_expert_indices")
            if rollout_expert_indices is not None:
                rollout_expert_indices = rollout_expert_indices.to(torch.int32)
            micro_dict = {
                "sequences": micro["sequences"],
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "num_actions": micro.metadata["response_length"],
                "rollout_expert_indices": (rollout_expert_indices if self.enable_router_replay else None),
                "sub_seq_lengths": micro.get("sub_seq_lengths"),
            }
            micro_dicts.append(micro_dict)

        if use_token_batching:
            # Pad microbatches to uniform batch size for Megatron compatibility
            max_micro_bsz = max(m["sequences"].shape[0] for m in micro_dicts) if micro_dicts else 1
            for i, m in enumerate(micro_dicts):
                micro_dicts[i] = self._pad_microbatch_to_size(m, max_micro_bsz)
            mbs = max_micro_bsz
        else:
            mbs = micro_dicts[0]["sequences"].shape[0] if micro_dicts else 1

        self.model.eval()
        seq_len = micro_dicts[0]["sequences"].shape[1]

        # ENGAGEMENT LINE for the scoring phase, the mirror of the train step's "sequence packing"
        # line (see forward_backward). It exists because an install/config banner is not engagement:
        # `trainer.max_tokens_per_microbatch(_forward)` in a launcher only proves the knob was
        # *passed*, and the number that decides this phase's cost -- how many microbatches the rank
        # actually ran -- was previously visible nowhere in an arm log. Reading a packed count
        # here is what makes a scoring-time claim
        # checkable after the fact. Gated on the first PP/TP/CP rank so it is one line per DP rank.
        if (
            micro_dicts
            and mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == 0
            and mpu.get_context_parallel_rank() == 0
        ):
            real_tokens = int(sum(int(m["attention_mask"].sum().item()) for m in micro_dicts))
            budget_note = f"budget={forward_token_budget}" if use_token_batching else "budget=off"
            if use_token_batching and forward_token_budget != self.cfg.max_tokens_per_microbatch:
                budget_note += f" (forward-only; train budget={self.cfg.max_tokens_per_microbatch})"
            logger.info(
                f"scoring packing | dp_rank={mpu.get_data_parallel_rank()} "
                f"microbatches_this_step={len(micro_dicts)} seq_len={seq_len} tokens={real_tokens} "
                f"micro_batch_size={mbs} {budget_note}"
            )

        with torch.no_grad():
            log_probs = self.model.forward(
                micro_batches=micro_dicts,
                seq_len=seq_len,
                micro_batch_size=mbs,
                temperature=self.cfg.algorithm.temperature,
            )

        # This boundary counts only real scoring traffic: capability counters arm after prewarm,
        # and the validator uses the CPU rendezvous Store rather than adding another NCCL op.
        _isoexec_validate_nccl_transport_boundary("scoring_end")

        log_probs = log_probs.to("cpu")

        if use_token_batching and microbatch_iterator is not None:
            # Need to strip padded samples and reorder back to original order
            output = TrainingOutputBatch({"output": log_probs})
            output.metadata = data.metadata
            # The output from Megatron is concatenated across microbatches.
            # We need to extract only the real (non-padded) samples and reorder.
            output = self._reorder_megatron_forward_output(output, microbatch_iterator, micro_dicts, mbs)
        else:
            output = TrainingOutputBatch({"output": log_probs})
            output.metadata = data.metadata

        clear_router_replay()
        return output["output"]

    def _reorder_megatron_forward_output(
        self, output: TrainingOutputBatch, microbatch_iterator, micro_dicts, padded_mbs
    ) -> TrainingOutputBatch:
        """Reorder forward output from token-based microbatching back to original sample order."""
        if not isinstance(microbatch_iterator, TokenBasedBatchIterator):
            return output

        # With PP > 1 only the last pipeline stage produces real per-sample logprobs;
        # other stages return a dummy placeholder (e.g. [1, 1]). There is nothing to
        # reorder there, and indexing it by microbatch would raise — so return as-is,
        # matching how the non-token-batched path leaves the placeholder untouched.
        if not mpu.is_pipeline_last_stage(ignore_virtual=True):
            return output

        log_probs = output["output"]  # shape: [total_padded_samples, num_actions]

        # Split by padded_mbs, take only real samples, reorder
        all_log_probs = log_probs.split(padded_mbs, dim=0)

        # Build original-order tensor
        batch_size = microbatch_iterator.data.batch_size
        num_actions = log_probs.shape[1]
        reordered = torch.zeros((batch_size, num_actions), dtype=log_probs.dtype, device=log_probs.device)

        for mb_idx, original_indices in enumerate(microbatch_iterator._microbatches):
            mb_log_probs = all_log_probs[mb_idx]
            for sample_idx, original_idx in enumerate(original_indices):
                reordered[original_idx] = mb_log_probs[sample_idx]

        result = TrainingOutputBatch({"output": reordered})
        result.metadata = output.metadata
        return result

    def _pad_microbatch_to_size(self, micro_dict: dict, target_batch_size: int) -> dict:
        """Pad a forward or forward_backward micro-batch dict to target_batch_size with dummy samples.

        Padded samples have loss_mask/action_mask=0 so they don't contribute to the loss
        (forward micro-batches carry neither key, so this is inert there). This is needed
        because Megatron's forward_backward_func requires uniform micro_batch_size across all
        microbatches (especially with PP > 1). Scalar keys (``num_actions``,
        ``num_microbatches``, ``num_real_microbatches``) are passed through unchanged.

        Defined on the base worker so the shared ``_forward_logprobs`` path works for
        policy, ref, and critic workers alike.
        """
        current_bsz = micro_dict["sequences"].shape[0]
        if current_bsz >= target_batch_size:
            return micro_dict

        pad_count = target_batch_size - current_bsz
        device = micro_dict["sequences"].device

        padded = {}
        for key, value in micro_dict.items():
            if key in ("num_actions", "num_microbatches", "num_real_microbatches"):
                padded[key] = value
                continue
            if value is None:
                padded[key] = None
                continue
            if isinstance(value, torch.Tensor):
                if key == "loss_mask":
                    # Pad with zeros so padded samples don't contribute to loss
                    pad_tensor = torch.zeros((pad_count, *value.shape[1:]), dtype=value.dtype, device=device)
                elif key == "attention_mask":
                    # Give each dummy row a single valid token, so the row is non-degenerate:
                    # it avoids a fully-masked row (NaN in dense attention's softmax) and a
                    # zero-length cu_seqlens segment (rejected by the packed/THD kernel).
                    # The row is still excluded from the loss via loss_mask/action_mask=0.
                    pad_tensor = torch.zeros((pad_count, *value.shape[1:]), dtype=value.dtype, device=device)
                    pad_tensor[:, 0] = 1
                elif key == "position_ids":
                    # position_ids for padded samples
                    seq_len = value.shape[1]
                    pad_tensor = torch.arange(seq_len, device=device).unsqueeze(0).expand(pad_count, -1)
                elif key == "action_mask":
                    # action_mask should be zeros for padded samples
                    pad_tensor = torch.zeros((pad_count, *value.shape[1:]), dtype=value.dtype, device=device)
                else:
                    pad_tensor = torch.zeros((pad_count, *value.shape[1:]), dtype=value.dtype, device=device)
                padded[key] = torch.cat([value, pad_tensor], dim=0)
            else:
                padded[key] = value

        return padded

    def save_hf_model(self, export_dir: str, tokenizer):
        # Save model in HuggingFace safetensors format
        self.strategy.save_hf_model(
            self.bridge,
            self.model,
            export_dir,
            tokenizer=tokenizer,
        )

    def _get_module_for_offload(self):
        # The underlying offloadable module is `self.actor_module` instead of `self.model`.
        return self.actor_module


class MegatronPolicyWorkerBase(MegatronWorker, PolicyWorkerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: MegatronModelWrapper = None
        self.actor_module: List[nn.Module] = None
        self.scheduler: OptimizerParamScheduler = None
        self.optimizer: DistributedOptimizer = None
        self.profiler: Profiler = None
        self._is_lora = self.cfg.policy.model.lora.rank > 0
        # Per-worker store of LoRA adapter snapshots. Allocated only for the
        # LoRA path; FFT runs single-tenant exactly as before.
        self.adapter_store: Optional[AdapterStore] = AdapterStore() if self._is_lora else None

    def init_worker_process_group(self):
        """
        Override DistributedTorchRayActor.init_worker_process_group to use megatron distributed setup to create the mesh.
        """
        if not torch.distributed.is_initialized():
            # Ensure CUDA device is set before process group init — required when
            # using split "cpu:gloo,cuda:nccl" backend to avoid 'invalid device ordinal'
            # errors during NCCL communicator creation in subgroups.
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            # Default torch dist pg init timeout is 10 minutes (600 seconds)
            torch.distributed.init_process_group(
                backend="cpu:gloo,cuda:nccl", timeout=timedelta(seconds=SKYRL_WORKER_NCCL_TIMEOUT_IN_S)
            )

        # Explicitly wrap torch.distributed.broadcast in torch.no_grad() to avoid a warning in Megatron training where the
        # autograd engine tries to track gradients through the default Torch kernel. This fixes a deprecated behaviour in
        # PyTorch, preventing potential silent errors in future versions.

        if not getattr(torch.distributed, "_skyrl_broadcast_no_grad_patched", False):
            _orig_broadcast = torch.distributed.broadcast

            def _broadcast_no_grad(*args, **kwargs):
                with torch.no_grad():
                    return _orig_broadcast(*args, **kwargs)

            torch.distributed.broadcast = _broadcast_no_grad
            torch.distributed._skyrl_broadcast_no_grad_patched = True

        self.strategy = MegatronStrategy(
            megatron_config=self.cfg.policy.megatron_config,
            optimizer_config=self.cfg.policy.optimizer_config,
            seed=self.cfg.seed,
            is_lora=self._is_lora,
            node_local_rank=self._local_rank,
        )
        self.strategy.setup_distributed()

        self.mesh_rank = MeshRank(
            dp=mpu.get_data_parallel_rank(),
            sp=mpu.get_context_parallel_rank(),
            tp=mpu.get_tensor_model_parallel_rank(),
            pp=mpu.get_pipeline_model_parallel_rank(),
            world_size=self._world_size,
            dp_size=mpu.get_data_parallel_world_size(),
            pp_size=mpu.get_pipeline_model_parallel_world_size(),
        )

    def init_model(self, model_path, num_training_steps: int = 1e9):
        """
        Initialize the model, optimizer, and scheduler for the policy worker.
        """
        # isoexec Phase 2: the ContractAdapter drives the trainer's enforcement sequence -- build
        # the composition contract (fail-soft, as before), check EVERY contract claim against the
        # deployed trainer facts, run the install path below, close the INSTALL boundary. Both
        # runtimes build the SAME complete (op,site)->impl contract from the same code+model+arch,
        # so their identities match by construction; a differing [ISOEXEC-CONTRACT] hash across the
        # two process logs is a composition split-brain (the #1 failure mode).
        if os.environ.get("SKYRL_ISOEXEC"):
            # NCCL pin A/B EVIDENCE, read from the WORKER's own environment (the ray runtime env is
            # what the driver *asked* for; this is what the process actually got, which is the only
            # thing NCCL reads at ncclCommInitRank). Two previous live A/Bs of this pin were invalid
            # self-comparisons because a re-pin happened behind the measurement -- so an unpinned arm
            # is only believable with this banner in the log. Engine side prints [ISOEXEC-NCCL] from
            # vllm_patches.neutralize_vllm_nccl_channel_pin.
            print(
                "[ISOEXEC-NCCL] trainer worker env: "
                f"NCCL_ALGO={os.environ.get('NCCL_ALGO')} "
                f"NCCL_MIN_NCHANNELS={os.environ.get('NCCL_MIN_NCHANNELS')} "
                f"NCCL_MAX_NCHANNELS={os.environ.get('NCCL_MAX_NCHANNELS')} "
                f"(SKYRL_ISOEXEC_NCCL_PIN={os.environ.get('SKYRL_ISOEXEC_NCCL_PIN', '1')})",
                flush=True,
            )

        def _isoexec_install():
            # initialize the bridge and provider objects
            self.init_configs(
                model_path,
                self.cfg.policy.megatron_config,
                self.cfg.policy.megatron_config.model_config_kwargs,
                self.cfg.policy.megatron_config.transformer_config_kwargs,
                bf16=self.cfg.bf16,
                flash_attn=self.cfg.flash_attn,
                language_model_only=self.cfg.policy.language_model_only,
            )

            if self.enable_router_replay:
                from skyrl.backends.skyrl_train.utils.replay_utils import (
                    patch_topk_router_layer_number,
                )

                patch_topk_router_layer_number()

            # Freeze MoE router params before optimizer build.
            # Megatron's DistributedOptimizer reads requires_grad at construction.
            if self.cfg.policy.megatron_config.freeze_moe_router:
                if self._rank == 0:
                    logger.info("freeze_moe_router=True: freezing MoE router params")
                self.provider.register_pre_wrap_hook(freeze_moe_router)

            # wrap with DDP for training
            wrap_with_ddp = not self.cfg.policy.inference_only_init
            self.actor_module = self.make_megatron_module(
                wrap_with_ddp=wrap_with_ddp,
                ddp_config=self.cfg.policy.megatron_config.ddp_config if wrap_with_ddp else None,
                lora_config=self.cfg.policy.model.lora if self._is_lora else None,
                lora_type=self.cfg.policy.megatron_config.lora_config.lora_type,
                bf16=self.cfg.bf16,
            )
            # SkyRL-IsoExec: match the trainer's attention kernel to the engine's. The trainer runs the
            # local (no-TE) spec, where torch SDPA is a different kernel from the engine's and leaves
            # large per-token rollout-vs-train logprob outliers; swapping core_attention to the same
            # torch varlen kernel, alongside the batch-invariant non-attention ops, puts the pre-update
            # scoring gate at the numerical floor.
            if os.environ.get("SKYRL_ISOEXEC") == "1":
                if (
                    os.environ.get("SKYRL_ISOEXEC_LOCAL_SPEC") == "1"
                    and os.environ.get("SKYRL_ISOEXEC_VARLEN_ATTN", "1") == "1"
                ):
                    from skyrl.backends.skyrl_train.isoexec.ops.attention.megatron_varlen_attn import (
                        enable_trainer_batch_invariant,
                        swap_trainer_core_attention_varlen,
                    )

                    if os.environ.get("SKYRL_ISOEXEC_BATCH_INVARIANT", "1") == "1":
                        enable_trainer_batch_invariant()
                    swap_trainer_core_attention_varlen(self.actor_module)
                else:
                    print(
                        "[ISOEXEC-TRAINER] SKIPPED trainer attention match "
                        "(requires SKYRL_ISOEXEC_LOCAL_SPEC=1 and SKYRL_ISOEXEC_VARLEN_ATTN=1)",
                        flush=True,
                    )

            # SkyRL-IsoExec: bitwise auto-fusion sites (isoexec/autofuse/sites.py). Applies on both the
            # TE and local-spec paths; inert one-liner unless SKYRL_ISOEXEC_AUTOFUSE=1 (default 0).
            # Decisions are CONSUMED from the shared fusion ledger (never made here), each admitted
            # shape re-proves bit-equality against eager on live operands once per process, and the
            # manifest-handshake pin (registered at isoexec package import, Tier-0.3) turns a
            # trainer<->engine flag/ledger split-brain into a weight-sync refusal. Wrapped: a wiring
            # failure demotes to eager per site and must never break worker init.
            if os.environ.get("SKYRL_ISOEXEC") == "1":
                try:
                    from skyrl.backends.skyrl_train.isoexec.autofuse.sites import (
                        install_autofuse_sites,
                        selected_autofuse_requires_exact_install,
                    )

                    install_autofuse_sites("trainer")
                except Exception as _af_e:  # pragma: no cover - fail-to-eager, never fail the worker
                    if (
                        "selected_autofuse_requires_exact_install" in locals()
                        and selected_autofuse_requires_exact_install()
                    ):
                        raise RuntimeError(
                            "selected AUTOFUSE ledger has admitted artifacts but trainer installation failed"
                        ) from _af_e
                    print(f"[ISOEXEC-AUTOFUSE] trainer install skipped on error: {_af_e}", flush=True)

            # SkyRL-IsoExec: TP/EP-INVARIANT row-parallel (pik). Lets the trainer keep its production
            # tensor-parallel size while the rollout engine runs a different one and the rollout<->train
            # KL stays exactly 0 -- the row-parallel K-reduction follows a fixed leaf tree independent of
            # TP (see isoexec/pik_tp_invariant.py). No-op unless SKYRL_ISOEXEC_PIK=1. Applies on both the TE
            # and selective-TE/local-spec paths: selective TE deliberately assigns every row-parallel
            # role to megatron.core RowParallelLinear, which is the exact class this patch intercepts.
            if os.environ.get("SKYRL_ISOEXEC") == "1" and os.environ.get("SKYRL_ISOEXEC_PIK") == "1":
                from skyrl.backends.skyrl_train.isoexec.ops.collectives.pik_tp_invariant import (
                    apply_pik_tp_invariant,
                )

                apply_pik_tp_invariant(side="TRAINER")

            # TE-primitives mode keeps the exact LOCAL_SPEC model unchanged. Admission walks the final
            # module tree after every IsoExec rebind, refuses any TE model owner, and instruments only
            # Megatron's already-selected TE multi-tensor gradient-norm primitive.
            if os.environ.get("SKYRL_ISOEXEC_TE_PRIMITIVES", "0") == "1":
                from skyrl.backends.skyrl_train.isoexec.runtimes.megatron.te_primitives import (
                    admit,
                )

                admit(self.actor_module)

            # THE TRAINER'S INSTALL FINGERPRINT (design obligation 2, worklist F1). The IsoExec install
            # sequence for this process is finished above; record what it actually bound and log the
            # comparison against this process's manifest. Until F1 the fingerprint had ONE call site in
            # the whole tree -- inside the vLLM RecurrentGDN branch -- so the trainer side of every
            # composition was dark, and `validate_against_installed` had nothing to validate.
            if os.environ.get("SKYRL_ISOEXEC") == "1":
                if (
                    os.environ.get("SKYRL_ISOEXEC_NCCL_PIN", "1") == "0"
                    and int(self.cfg.policy.megatron_config.tensor_model_parallel_size) > 1
                    and os.environ.get("SKYRL_ISOEXEC_NCCL_TRANSPORT_BOUNDARY_REQUIREMENTS", "").strip()
                ):
                    from skyrl.backends.skyrl_train.isoexec.core.process_contract import (
                        cached_contract_view,
                    )
                    from skyrl.backends.skyrl_train.isoexec.ops.collectives.nccl_identity import (
                        assert_contract_matches,
                        effective_identity,
                    )

                    _nccl_impl, _nccl_constants = effective_identity()
                    assert_contract_matches(
                        cached_contract_view(),
                        ("trainer_fwd", "trainer_score"),
                        _nccl_impl,
                        _nccl_constants,
                    )
                self._isoexec_record_trainer_fingerprint()

            if self._local_rank == 0 and not os.path.exists(
                model_path
            ):  # if not local path, try downloading model weights from huggingface
                snapshot_download(model_path)  # will be no-op if already downloaded
            torch.distributed.barrier()

            if self._rank == 0:
                print_model_size(self.actor_module[0])

            # create profiler
            if self.cfg.policy.megatron_config.torch_profiler_config.enable:
                self.profiler = Profiler(self.cfg.policy.megatron_config.torch_profiler_config)

            # create optimizer (skipped for inference-only flows; Megatron's
            # DistributedOptimizer eagerly materializes fp32 master + AdamW state
            # on GPU, which OOMs large MoE models on memory-constrained nodes)
            if self.cfg.policy.inference_only_init:
                self.optimizer = None
                self.scheduler = None
            else:
                optim_config = init_megatron_optim_config(
                    self.cfg.policy.optimizer_config, self.cfg.policy.megatron_config.optimizer_config_kwargs
                )
                self.optimizer = get_megatron_optimizer(self.actor_module, optim_config)
                if os.environ.get("SKYRL_ISOEXEC_TE_PRIMITIVES", "0") == "1":
                    from skyrl.backends.skyrl_train.isoexec.runtimes.megatron.te_primitives import (
                        census_optimizer,
                    )

                    census_optimizer(self.optimizer, optim_config)

                # create scheduler
                self.scheduler = get_megatron_optimizer_param_scheduler(
                    optimizer=self.optimizer,
                    config=self.cfg.policy.optimizer_config,
                    num_training_steps=num_training_steps,
                )

            # create worker model
            self.model = MegatronModelWrapper(
                config=self.cfg,
                actor_module=self.actor_module,
                actor_optimizer=self.optimizer,
                policy_loss_fn=self.policy_loss_fn,
            )

            self.empty_cuda_cache = self.cfg.policy.megatron_config.empty_cuda_cache

            # Enable expandable_segments after init so model weights stay in IPC-compatible
            # standard CUDA memory; only subsequent activations use expandable segments.
            self._set_expandable_segments(True)

            # SkyRL-IsoExec: pay the trainer's NCCL communicator memory HERE, once, visibly.
            # NCCL allocates per communicator AND per transport at first use, so without this
            # the TP/EP/DP communicators materialize mid-step -- after vLLM has sized its KV
            # pool -- and `wake_up(kv_cache)` OOMs. Only matters when the channel pin is off
            # (pinned it is ~0.1 GiB/rank); the banner it prints is the input to the
            # gpu_memory_utilization budget. No-op unless SKYRL_ISOEXEC_NCCL_PREWARM=1.
            _nccl_prewarm_report = None
            if os.environ.get("SKYRL_ISOEXEC_NCCL_PREWARM", "0") == "1":
                _capability_mode = os.environ.get("SKYRL_ISOEXEC_NCCL_CAPABILITY_MODE", "off").strip().lower() or "off"
                if _capability_mode == "enforce":
                    from skyrl.backends.skyrl_train.isoexec.runtimes.megatron.nccl_transport_capabilities import (
                        register_mpu_manifest,
                    )

                    _manifest_path = os.environ.get("SKYRL_ISOEXEC_NCCL_CAPABILITY_MANIFEST", "").strip()
                    if not _manifest_path:
                        raise RuntimeError(
                            "[ISOEXEC-NCCL-CAP] enforce mode requires "
                            "SKYRL_ISOEXEC_NCCL_CAPABILITY_MANIFEST from an admitted full-step census"
                        )
                    register_mpu_manifest(mpu, _manifest_path)
                from skyrl.backends.skyrl_train.isoexec.runtimes.megatron.nccl_prewarm import (
                    prewarm_trainer_nccl,
                )

                _nccl_prewarm_report = prewarm_trainer_nccl(tag="policy")
            # A budgeted plan is not admitted until every lazy NCCL transport has materialized, the
            # ProcessGroupNCCL option reads back exactly, and the measured non-torch charge fits.
            self.strategy.verify_nccl_channels(_nccl_prewarm_report)

        # run_install: contract build (fail-soft) -> check_all_claims (REFUSING, strict default:
        # a topology outside the claims must stop the run, not warn into it) -> the install path
        # above -> INSTALL boundary of the obligation ledger (every check the contract derives for
        # the trainer side must have a record by the end of init_model -- a required check that
        # never ran refuses here as loudly as one that failed). Refusals propagate; internal
        # ledger errors never do.
        if os.environ.get("SKYRL_ISOEXEC"):
            from skyrl.backends.skyrl_train.isoexec.core.adapter import (
                set_process_adapter,
            )
            from skyrl.backends.skyrl_train.isoexec.runtimes.megatron.adapter import (
                MegatronContractAdapter,
            )

            set_process_adapter(
                MegatronContractAdapter(
                    model_path,
                    megatron_config=self.cfg.policy.megatron_config,
                    install_fn=_isoexec_install,
                    world_size=self._world_size,
                    model_fn=lambda: getattr(self, "actor_module", None),
                )
            ).run_install(close=os.environ.get("SKYRL_ISOEXEC") == "1")
        else:
            _isoexec_install()

    def forward(
        self,
        data: TrainingInputBatch,
        loss_fn: Optional[str] = None,
        loss_fn_config: Optional[Dict[str, Any]] = None,
    ) -> WorkerOutput:
        """Forward pass.

        - Without ``loss_fn``: runs Megatron's pipeline inference and returns a
          :class:`WorkerOutput` with per-sample ``loss_fn_outputs`` (``logprobs``
          key) and empty ``metrics``.
        - With ``loss_fn`` (e.g., ``"cross_entropy"``): runs the SFT loss through Megatron's
          pipeline schedule with ``forward_only=True`` (no backward) and returns a
        :class:`WorkerOutput` with per-sample ``loss_fn_outputs`` plus scalar
        ``metrics`` (including ``"loss"``).
        """
        from skyrl.backends.skyrl_train.utils.replay_utils import clear_router_replay

        if loss_fn is None:
            # Megatron inference forward path: emit per-sample logprobs. Token-based
            # micro-batching (when `max_tokens_per_microbatch > 0`) is handled inside
            # `_forward_logprobs`, which also reorders back to the original sample order.
            log_probs = self._forward_logprobs(data)
            loss_fn_outputs = [{"logprobs": log_probs[i].tolist()} for i in range(log_probs.shape[0])]
            return WorkerOutput(loss_fn_outputs=loss_fn_outputs, metrics={})

        self.model.eval()

        micro_batch_size = self.cfg.micro_forward_batch_size_per_gpu
        all_metrics = defaultdict(list)
        all_loss_fn_outputs: List[Dict[str, Any]] = []

        # Move data to GPU
        data.to(torch.cuda.current_device())

        # Build micro-batch dicts expected by forward_backward_mini_batch
        micro_buffer = []
        for experience in BatchIterator(data, micro_batch_size, drop_last=False):
            sequences = experience.sequences
            attention_mask = experience.attention_mask
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            rollout_expert_indices = experience.rollout_expert_indices
            if rollout_expert_indices is not None:
                rollout_expert_indices = rollout_expert_indices.to(torch.int32)
            micro_buffer.append(
                {
                    "sequences": sequences,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "num_actions": experience.num_actions,
                    "old_action_log_probs": experience.action_log_probs,
                    "base_action_log_probs": experience.base_action_log_probs,
                    "advantages": experience.advantages,
                    "loss_mask": experience.loss_mask,
                    "rollout_action_logprobs": experience.rollout_logprobs,
                    "action_mask": experience.action_mask,
                    "rollout_expert_indices": rollout_expert_indices if self.enable_router_replay else None,
                    "sub_seq_lengths": experience.sub_seq_lengths,
                }
            )

        for m_batch in micro_buffer:
            m_batch["num_microbatches"] = len(micro_buffer)

        if not micro_buffer:
            return WorkerOutput()

        seq_len = micro_buffer[0]["sequences"].shape[1]
        micro_bsz = micro_buffer[0]["sequences"].shape[0]

        with torch.no_grad():
            metrics_list = self.model.forward_backward_mini_batch(
                micro_batches=micro_buffer,
                seq_len=seq_len,
                micro_batch_size=micro_bsz,
                temperature=self.cfg.algorithm.temperature,
                loss_fn=loss_fn,
                loss_fn_config=loss_fn_config,
                forward_only=True,
            )

        if self.empty_cuda_cache:
            torch.cuda.empty_cache()

        # Aggregate metrics across micro-batches
        for metrics in metrics_list:
            if metrics is None:
                continue
            if "loss_fn_outputs" in metrics:
                all_loss_fn_outputs.extend(metrics.pop("loss_fn_outputs"))
            for k, v in metrics.items():
                all_metrics[k].append(v)

        resolved_loss_name = loss_fn or self.cfg.algorithm.policy_loss_type
        sum_loss_metrics = resolved_loss_name != "cross_entropy"

        status = reduce_metrics(all_metrics, sum_loss_metrics=sum_loss_metrics)
        group = mpu.get_data_parallel_group(with_context_parallel=False)
        status = all_reduce_metrics(status, self.strategy, group=group, sum_loss_metrics=sum_loss_metrics)

        clear_router_replay()
        return WorkerOutput(loss_fn_outputs=all_loss_fn_outputs, metrics=status)

    def _isoexec_record_trainer_fingerprint(self) -> None:
        """Record what the TRAINER adapter installed, per op family, then log the comparison.

        WHY HERE AND NOT IN THE OP MODULES. Under ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` the engine
        shares this process, and several IsoExec installs are class-level rebinds that are scoped to
        one runtime by an instance MARK rather than by which process ran them. An op module
        therefore cannot know whose sites its rebind serves; this worker can, because it IS the
        trainer. Each impl_id is read off what the install actually reached -- a live predicate or a
        post-install flag -- never off the manifest.

        Fail-soft: a fingerprint bug must never break a trainer build.
        """
        try:
            from skyrl.backends.skyrl_train.isoexec.core.adapter import (
                live_pins,
                log_unreported_pins,
            )
            from skyrl.backends.skyrl_train.isoexec.core.fingerprint import (
                NOT_INSTALLED,
                TRAINER_SITES,
                log_fingerprint_once,
                record_installs,
            )
            from skyrl.backends.skyrl_train.isoexec.core.process_contract import (
                cached_contract_view,
            )

            cfg = getattr(self.actor_module[0], "config", None) if getattr(self, "actor_module", None) else None
            _g = lambda name, default=None: getattr(cfg, name, default)  # noqa: E731

            # -- the global GEMM provider ---------------------------------------------------------
            from skyrl.backends.skyrl_train.isoexec.ops.mm.mm_cublaslt import (
                mm_cublaslt_enabled,
            )

            record_installs(
                "mm", TRAINER_SITES, "cublaslt_pinned" if mm_cublaslt_enabled() else "triton_batch_invariant"
            )

            # -- attention / rotary ---------------------------------------------------------------
            record_installs("attention.varlen", TRAINER_SITES, "varlen_custom")
            record_installs("rope.rope", TRAINER_SITES, "eager")

            # -- norms: the trainer runs the grad-capable reference at both of its sites ----------
            record_installs(
                "norms.rms",
                TRAINER_SITES,
                "eager_zero_centered" if bool(_g("layernorm_zero_centered_gamma", False)) else "eager_torch_rms",
            )
            if _g("experimental_attention_variant", None) == "gated_delta_net":
                record_installs("norms.gated_out", TRAINER_SITES, "eager")
                # -- GDN: the MODE is what the manifest pins, read from the same call-time
                # predicate the kernels use -- which is how a manifest derived from a profile field
                # gets caught disagreeing with the flag the launcher actually exported.
                from skyrl.backends.skyrl_train.isoexec.ops.gdn.gdn_ops import (
                    cpr_mode,
                    gdn_kernel_mode,
                    gdn_native_kernels_enabled,
                    recurrent_mode,
                )

                _native = gdn_native_kernels_enabled() and (recurrent_mode() or cpr_mode())
                # The pin carries the kernel the native impl runs: `native_fused_sigmoid` alone
                # cannot distinguish a recurrent build from a cpr one, and the contract
                # pins exactly that.
                record_installs(
                    "gdn.core",
                    TRAINER_SITES,
                    "native_fused_sigmoid" if _native else gdn_kernel_mode(),
                    pinned={"kernel": gdn_kernel_mode()},
                )
                record_installs("gdn.conv", TRAINER_SITES, "causal_conv1d_fn")

            # -- logprobs -------------------------------------------------------------------------
            # The leaf-tree impl is one function at BOTH trainer sites, unconditionally. Record
            # what this process will actually arm, never what the contract asked for: a missing
            # module means the model_utils hook shim declines every call, so the incumbent serves
            # and NOT_INSTALLED is what gets recorded (the contract still names rowinv, and the
            # fingerprint comparator is what makes that disagreement visible).
            try:
                import skyrl.backends.skyrl_train.isoexec.ops.logprobs.rowinv  # noqa: F401

                _rowinv = True
            except ImportError:
                _rowinv = False
            record_installs(
                "logprobs.log_softmax",
                TRAINER_SITES,
                "rowinv_leaftree" if _rowinv else NOT_INSTALLED,
                # Pins read off the live install (rowinv.BLOCK, the kernel's own env read), not
                # echoed from the contract, so pin_disagreements can actually disagree.
                pinned=live_pins("logprobs.log_softmax") if _rowinv else None,
            )

            # -- collectives (present iff TP>1) ---------------------------------------------------
            if int(_g("tensor_model_parallel_size", 1) or 1) > 1:
                from skyrl.backends.skyrl_train.isoexec.ops.collectives.pik_tp_invariant import (
                    pik_enabled,
                )

                _pik = "pik_tree" if pik_enabled() else NOT_INSTALLED
                # Pins off the ReductionPlan the install actually built, not off the env vars that
                # asked for it, so "the flag arrived and the plan was built differently" is visible.
                record_installs(
                    "collectives.tree_all_reduce",
                    TRAINER_SITES,
                    _pik,
                    pinned=live_pins("collectives.tree_all_reduce") if pik_enabled() else None,
                )
                record_installs("collectives.row_parallel", TRAINER_SITES, _pik)
                # The trainer entry is FUNCTION-classified because this flag selects the trainer
                # composition (and backward reduction schedule).  Clean unpinned forward A/Bs keep
                # the low-e-07 gate; the manifest must still name what was actually installed.
                from skyrl.backends.skyrl_train.isoexec.ops.collectives.nccl_identity import (
                    effective_identity as effective_nccl_identity,
                )

                _nccl_impl, _nccl_constants = effective_nccl_identity()
                record_installs(
                    "collectives.nccl_pin",
                    TRAINER_SITES,
                    _nccl_impl,
                    pinned=_nccl_constants,
                )

            # -- MoE (nothing on a dense model, which then legitimately records nothing) ----------
            if bool(_g("num_moe_experts", 0)):
                _sigmoid = _g("moe_router_score_function", "softmax") == "sigmoid" or bool(
                    _g("moe_router_enable_expert_bias", False)
                )
                record_installs(
                    "moe.router", TRAINER_SITES, "deterministic_sigmoid_bias" if _sigmoid else "deterministic"
                )
                record_installs("moe.dispatch", TRAINER_SITES, "index_build")
                record_installs("moe.experts", TRAINER_SITES, "batched_bmm")
                from skyrl.backends.skyrl_train.isoexec.ops.moe.moe_batch_invariant import (
                    _moe_pik_fc2_on,
                )

                record_installs(
                    "moe.combine",
                    TRAINER_SITES,
                    "pik_leaf_tree" if _moe_pik_fc2_on() else NOT_INSTALLED,
                    pinned=live_pins("moe.combine") if _moe_pik_fc2_on() else None,
                )

            _view = cached_contract_view()
            log_fingerprint_once(_view, tag="trainer_install")
            log_unreported_pins(_view)
        except Exception as _e:  # pragma: no cover - never fatal
            logger.warning(f"[ISOEXEC-FINGERPRINT] trainer install fingerprint skipped: {_e}")

    def forward_backward(
        self,
        data: TrainingInputBatch,
        loss_fn: Optional[str] = None,
        loss_fn_config: Optional[Dict[str, Any]] = None,
    ) -> WorkerOutput:
        """
        Perform forward and backward passes for a batch, handling micro-batching internally.

        The batch is split into micro batches based on micro_train_batch_size_per_gpu,
        or by token count if max_tokens_per_microbatch is configured.
        Megatron Core's forward_backward_func handles gradient accumulation internally.

        Args:
            data: TrainingInputBatch (already DP-sharded by WorkerDispatch/MeshDispatch)
            loss_fn: Optional loss function name (e.g., "cross_entropy", "ppo").
                     If provided, overrides the config's policy_loss_type.
            loss_fn_config: Optional config overrides for the loss function.

        Returns:
        :class:`WorkerOutput` with per-sample ``loss_fn_outputs`` and scalar
        ``metrics`` (all-reduced across DP).
        """
        from skyrl.backends.skyrl_train.utils.replay_utils import clear_router_replay

        self.model.train()
        for chunk in self.actor_module:
            # if use distributed optimizer, zero grad buffer will be handled by optimizer
            chunk.zero_grad_buffer()

        all_metrics = defaultdict(list)

        use_token_batching = self.cfg.max_tokens_per_microbatch > 0

        # Move data to GPU
        data.to(torch.cuda.current_device())

        # SkyRL-IsoExec EP-skew balance (SKYRL_ISOEXEC_TRAINER_EP_BALANCE, default OFF): sort this
        # rank's shard by real length before microbatch chunking, so the i-th microbatch pairs like
        # lengths and rank-to-rank microbatch cost profiles align -- at EP>1 every MoE layer's
        # dispatch is an all-rank rendezvous per microbatch, and probe B measured its wait at 81.4%
        # of all trainer kernel time under generation-order chunking. Idempotent after the
        # trainer-side stripe (utils/ep_balance.py, which also carries the full mechanism + the
        # bitwise legality argument). Sample-count chunking only: token-based batching does its own
        # bin packing. Per-token forwards are batch-invariant (the pinned IsoExec property), so only
        # the fp32 grad-accumulation grouping moves -- trajectory-rounding class, gate unaffected.
        if not use_token_batching and ep_balance_enabled():
            data, _epbal_stats = sort_shard_rows_by_length(data, self.cfg.micro_train_batch_size_per_gpu)
            if _epbal_stats is not None and (
                mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_pipeline_model_parallel_rank() == 0
                and mpu.get_context_parallel_rank() == 0
            ):
                print(
                    f"[ISOEXEC-EPBAL] dp_rank={mpu.get_data_parallel_rank()} worker length-sort "
                    f"{'APPLIED' if _epbal_stats['applied'] else 'ALREADY-SORTED'}: "
                    f"mb_maxlen_sum {_epbal_stats['mb_maxlen_sum_before']} -> "
                    f"{_epbal_stats['mb_maxlen_sum_after']} "
                    f"(mbs={self.cfg.micro_train_batch_size_per_gpu})",
                    flush=True,
                )

        if use_token_batching:
            microbatch_iterator = get_microbatch_iterator(
                data,
                micro_batch_size=self.cfg.micro_train_batch_size_per_gpu,
                max_tokens_per_microbatch=self.cfg.max_tokens_per_microbatch,
            )
        else:
            microbatch_iterator = None

        # Build micro-batch dicts expected by forward_backward_mini_batch.
        # Token-based batching yields TrainingInputBatch microbatches (converted to
        # Experience here); sample-based BatchIterator yields Experience directly.
        micro_buffer = []

        if microbatch_iterator is not None:
            experiences = (BaseBatchIterator.batch_to_experience(mb) for mb in microbatch_iterator)
        else:
            experiences = BatchIterator(data, self.cfg.micro_train_batch_size_per_gpu, drop_last=False)

        for replay_ordinal, experience in enumerate(experiences):
            attention_mask = experience.attention_mask
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            rollout_expert_indices = experience.rollout_expert_indices
            if rollout_expert_indices is not None:
                rollout_expert_indices = rollout_expert_indices.to(torch.int32)
            micro_buffer.append(
                {
                    "sequences": experience.sequences,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "num_actions": experience.num_actions,
                    "old_action_log_probs": experience.action_log_probs,
                    "base_action_log_probs": experience.base_action_log_probs,
                    "advantages": experience.advantages,
                    "loss_mask": experience.loss_mask,
                    "rollout_action_logprobs": experience.rollout_logprobs,
                    "action_mask": experience.action_mask,
                    "rollout_expert_indices": rollout_expert_indices if self.enable_router_replay else None,
                    # used with global sequence packing (None when token-based batching is active)
                    "sub_seq_lengths": experience.sub_seq_lengths,
                    "is_padding_batch": (
                        experience.metadata.get("is_padding_batch", False) if experience.metadata else False
                    ),
                }
            )

        # Count microbatches that carry real (non-padding) samples. Token-based batching
        # appends fully-padding microbatches (loss_mask all zero) so every DP rank runs the
        # same number of forward passes; those contribute 0 to KL/entropy and to mean metrics
        # but would otherwise inflate the denominators. `num_real_microbatches` lets the loss
        # normalize KL/entropy over real microbatches only.
        num_real_microbatches = sum(1 for m in micro_buffer if m["loss_mask"].sum().item() > 0)
        for m_batch in micro_buffer:
            m_batch["num_microbatches"] = len(micro_buffer)
            m_batch["num_real_microbatches"] = num_real_microbatches

        if not micro_buffer:
            return WorkerOutput()

        seq_len = micro_buffer[0]["sequences"].shape[1]

        if use_token_batching:
            # With token-based batching, microbatches may have different batch sizes.
            # Megatron's forward_backward_func requires uniform micro_batch_size,
            # so pad all microbatches to the max batch size across microbatches.
            max_micro_bsz = max(m["sequences"].shape[0] for m in micro_buffer)
            micro_buffer = [self._pad_microbatch_to_size(m, max_micro_bsz) for m in micro_buffer]
            micro_bsz = max_micro_bsz
        else:
            micro_bsz = micro_buffer[0]["sequences"].shape[0]

        # Gate on first PP/TP/CP rank so we emit exactly one line per DP rank
        # (matches how status all-reduce treats metrics as identical within a DP group).
        if (
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == 0
            and mpu.get_context_parallel_rank() == 0
        ):
            real_tokens = int(sum(int(mb["attention_mask"].sum().item()) for mb in micro_buffer))
            num_microbatches = len(micro_buffer)
            dp_rank = mpu.get_data_parallel_rank()
            logger.info(
                f"sequence packing | dp_rank={dp_rank} microbatches_this_step={num_microbatches} "
                f"seq_len={seq_len} tokens={real_tokens}"
            )

        metrics_list = self.model.forward_backward_mini_batch(
            micro_batches=micro_buffer,
            seq_len=seq_len,
            micro_batch_size=micro_bsz,
            temperature=self.cfg.algorithm.temperature,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
        )

        # The backward has completed and main_grad is populated. Refuse before metrics/optimizer
        # if any configured physical owner failed to serve its expected real collective.
        _isoexec_validate_nccl_transport_boundary("policy_backward_end")

        if self.empty_cuda_cache:
            torch.cuda.empty_cache()

        # Aggregate metrics across micro-batches
        all_loss_fn_outputs = []  # Handle separately from scalar metrics
        for m_batch, metrics in zip(micro_buffer, metrics_list):
            # Extract loss_fn_outputs before reduce_metrics (it's not a scalar metric)
            if "loss_fn_outputs" in metrics:
                all_loss_fn_outputs.extend(metrics.pop("loss_fn_outputs"))
            # Skip fully-padding microbatches: their metrics (clip_ratio=0, policy_entropy=0,
            # ...) are meaningless and would drag down the mean-reduced metrics. Summed
            # metrics (e.g. policy_loss) are unaffected since padding contributes 0, but
            # excluding them here keeps both reductions correct.
            if m_batch["is_padding_batch"]:
                continue
            for k, v in metrics.items():
                all_metrics[k].append(v)

        # TODO: SFT path still averages metrics across microbatches and workers.
        # This needs to be unified with the RL path which sums.
        resolved_loss_name = loss_fn or self.cfg.algorithm.policy_loss_type
        sum_loss_metrics = resolved_loss_name != "cross_entropy"

        # Reduce across microbatches and all-reduce metrics across DP ranks
        # (metrics should be identical within DP groups, i.e., across TP/PP/SP ranks)
        # NOTE: Sum loss metrics because scaling is already applied at the advantage level
        status = reduce_metrics(all_metrics, sum_loss_metrics=sum_loss_metrics)
        if self.optimizer is not None:
            status["policy_lr"] = self.optimizer.param_groups[0]["lr"]

        # Token-based batching diagnostics: total microbatches this rank ran and how many
        # were purely-padding (added to equalize the microbatch count across DP ranks).
        # Added before all-reduce so they are averaged across DP (num_microbatches is
        # identical on every rank; num_padding_microbatches reports the per-rank average).
        if use_token_batching:
            status["num_microbatches"] = float(len(micro_buffer))
            status["num_padding_microbatches"] = float(len(micro_buffer) - num_real_microbatches)

        group = mpu.get_data_parallel_group(with_context_parallel=False)
        status = all_reduce_metrics(status, self.strategy, group=group, sum_loss_metrics=sum_loss_metrics)

        # Collect MoE aux metrics averaged across microbatches (all-reduced across ranks
        # inside get_moe_metrics) aggregating after per-microbatch scalar metrics.
        total_num_microbatches = len(micro_buffer)
        model_config = get_model_config(self.actor_module[0])
        num_moe_experts = getattr(model_config, "num_moe_experts", None)
        moe_metrics: Dict[str, Any] = {}
        if num_moe_experts is not None and num_moe_experts > 1:
            moe_loss_scale = 1.0 / max(1, total_num_microbatches)
            moe_metrics = get_moe_metrics(
                loss_scale=moe_loss_scale,
                per_layer_logging=self.cfg.policy.megatron_config.moe_per_layer_logging,
            )
            # moe_metrics will only be non-empty if "moe_router_load_balancing_type" is set to "aux_loss", "seq_aux_loss", or "global_aux_loss"
            if moe_metrics:
                for k, v in moe_metrics.items():
                    status[k] = v

        clear_router_replay()

        return WorkerOutput(loss_fn_outputs=all_loss_fn_outputs, metrics=status)

    def optim_step(self) -> Optional[float]:
        """
        Perform optimizer step.

        Note: Unlike FSDP workers, Megatron doesn't need manual gradient scaling here
        because Megatron Core's forward_backward_func handles loss scaling internally.

        Returns:
            The gradient norm (before scaling, after clipping), or None if unavailable.
        """
        if self.optimizer is None:
            raise RuntimeError("optim_step called but policy.inference_only_init=True (no optimizer constructed)")
        grad_norm = self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler, name="actor")

        if os.environ.get("SKYRL_ISOEXEC") == "1":
            self._isoexec_force_fresh_model_params()
        if os.environ.get("SKYRL_ISOEXEC_DEBUG_TRACE"):
            from skyrl.backends.skyrl_train.isoexec.debug import set_step

            self._isoexec_debug_step = getattr(self, "_isoexec_debug_step", 0) + 1
            set_step(self._isoexec_debug_step)

        # Reset counter for next accumulation cycle
        self._micro_batches_accumulated = 0

        if grad_norm is not None:
            grad_norm = grad_norm.detach().cpu().item() if hasattr(grad_norm, "item") else grad_norm
        return grad_norm

    @torch.no_grad()
    def _isoexec_param_buffer_witness(self):
        """A CHEAP WHOLE-BUFFER fingerprint of the bf16 model parameter data, or ``None``.

        Megatron's DDP keeps every parameter of a chunk inside a handful of FLAT
        ``param_data`` buffers, so a full-model fingerprint is a few big reductions rather than
        the ~21k per-parameter reductions ``_isoexec_param_abs_sum`` costs (which is exactly why
        that one is DEBUG-gated). Two independent reductions per buffer -- the signed sum and the
        absolute sum -- so a change that preserves one has to preserve the other too.

        ``None`` means "cannot fingerprint this model" (not DDP-bucketed, empty/offloaded
        buffers). Every caller must treat ``None`` as "assume changed": the fingerprint exists to
        license a SKIP, so absence of evidence must never license one.
        """
        try:
            from megatron.core.distributed import DistributedDataParallel as _DDP
        except Exception:
            return None
        out = []
        for chunk in self.actor_module:
            if not isinstance(chunk, _DDP):
                return None
            bufs = list(getattr(chunk, "buffers", []) or []) + list(getattr(chunk, "expert_parallel_buffers", []) or [])
            if not bufs:
                return None
            for b in bufs:
                t = getattr(b, "param_data", None)
                if t is None or t.numel() == 0 or t.device.type != "cuda":
                    return None
                # OOM fix (2026-08-14, killed production_v1 at the step-2 sync): `.float()`
                # materialized an fp32 copy of the whole flat buffer (15 GiB request against
                # 13.8 free). Reduce in fixed 256M-element slices accumulating in fp64 instead —
                # nothing larger than one slice is ever materialized, the slice grid is fixed so
                # the value is call-to-call deterministic, and the witness only ever compares
                # against values produced by this same function.
                v = t.view(-1)
                _CH = 1 << 28
                s_sum = 0.0
                s_abs = 0.0
                for i in range(0, v.numel(), _CH):
                    sl = v[i : i + _CH]
                    s_sum += float(sl.sum(dtype=torch.float64))
                    s_abs += float(sl.abs().sum(dtype=torch.float64))
                out.append((t.numel(), s_sum, s_abs))
        return tuple(out) if out else None

    @torch.no_grad()
    def _isoexec_params_provably_unchanged(self) -> bool:
        """True iff every rank agrees the model param buffer is bit-stable since the last
        ``_isoexec_force_fresh_model_params``. COLLECTIVE: the answer must be identical on every
        rank, because the routine it gates contains collectives (``start_param_sync``) and a
        split decision would deadlock. Fail-closed in every direction -- no stamp, no
        fingerprint, no distributed group, any exception -> False -> the routine runs."""
        try:
            prev = getattr(self, "_ix_fresh_witness", None)
            now = self._isoexec_param_buffer_witness() if prev is not None else None
            local = 1 if (prev is not None and now is not None and now == prev) else 0
            if torch.distributed.is_initialized():
                flag = torch.tensor([local], device=torch.cuda.current_device(), dtype=torch.int32)
                torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
                local = int(flag.item())
            return bool(local)
        except Exception as e:  # a bug in the GUARD must never skip the fix
            print(f"[ISOEXEC-FORCE-FRESH] witness check errored, running the full refresh: {e!r}", flush=True)
            return False

    @torch.no_grad()
    def _isoexec_sync_replicated_params(self):
        """SkyRL-IsoExec: re-sync TP-REPLICATED params (router, layernorms, biases -- anything
        without the `tensor_model_parallel` mark) to TP-rank-0's copy before weight extraction.

        With sequence_parallel off, Megatron never all-reduces the grads of replicated params:
        it assumes every TP rank computes bitwise-identical grads for them. Nondeterministic
        backward kernels (FLA/GDN atomic adds) break that, so after the first real optimizer
        step the per-rank replicas drift a few bf16 ULPs apart. Matched trainer/engine TP hides
        the drift (each engine rank inherits its co-located trainer rank's replica, so rollout
        and scoring stay self-consistent). Mismatched TP exposes it: the pik sender sources
        replicated params from DIFFERENT trainer ranks for the two engines (grank r and r+4 both
        produce engine-tp-rank r's payload), so the engines diverge from each other AND from the
        scoring forward -- observed live at 35B sync #1 as pairwise [ISOEXEC-CKSUM] SENDER
        mismatches (~2e-3 in 7.7e7) while every REAPPLY matched its own SENDER exactly, i.e.
        transport faithful, sources different. A drifted ROUTER replica flips top-k on rare
        tokens -> the min=0 / mean~0.02 / max~11 gate signature.

        Broadcasting rank-0's replica TP-wide right before extraction makes both engines and
        every subsequent trainer forward read one canonical copy. The fp32 masters stay per-rank
        and re-drift each step; this re-canonicalizes at every sync, which is exactly when
        consistency matters for IsoExec."""
        from megatron.core import parallel_state as mpu

        if mpu.get_tensor_model_parallel_world_size() == 1:
            return
        tp_group = mpu.get_tensor_model_parallel_group()
        src = torch.distributed.get_global_rank(tp_group, 0)
        # COALESCED. This used to issue ONE NCCL broadcast per replicated param -- on a 35B that
        # is hundreds of latency-bound collectives (layernorms, routers, biases: every one a few
        # KiB) inside the weight-sync critical path, where the cost is entirely per-call launch
        # overhead rather than bytes. Flatten per dtype, broadcast once per dtype, copy back.
        # BITWISE IDENTICAL: a broadcast installs rank-0's bytes verbatim either way, and
        # flatten/unflatten is a pure memcpy -- no arithmetic touches a value.
        named = []
        for m in self.actor_module:
            inner = m
            for _ in range(4):
                if hasattr(inner, "module"):
                    inner = inner.module
                else:
                    break
            for name, p in inner.named_parameters():
                if getattr(p, "tensor_model_parallel", False):
                    continue
                if p.device.type == "meta" or p.numel() == 0:
                    continue
                named.append((name, p))
        by_dtype: dict = {}
        for name, p in named:
            by_dtype.setdefault((p.dtype, p.device), []).append((name, p))
        for (dt, dev), group in by_dtype.items():
            flat = torch.empty(sum(p.numel() for _, p in group), dtype=dt, device=dev)
            off = 0
            for _, p in group:
                n = p.numel()
                flat[off : off + n].copy_(p.data.reshape(-1))
                off += n
            torch.distributed.broadcast(flat, src=src, group=tp_group)
            off = 0
            for _, p in group:
                n = p.numel()
                # view_as (not reshape on the DEST): reshape would hand back a temporary for a
                # non-contiguous param and the copy would land nowhere.
                p.data.copy_(flat[off : off + n].view_as(p.data))
                off += n

    @torch.no_grad()
    def _isoexec_force_fresh_model_params(self, *, skip_if_unchanged: bool = False):
        """SkyRL-IsoExec: guarantee the model param buffer holds THIS step's updated weights
        before anything else (weight-sync extraction, model offload) reads it.

        ``skip_if_unchanged=True`` (the SYNC-time, belt-and-braces call): return immediately when
        the whole-buffer fingerprint is bit-stable since the last real refresh. This routine runs
        TWICE per step -- once at ``optim_step``, where it is the actual fix, and once here to
        guard against the offload/reload cycle in between. MEASURED on the v10 Qwen3.5-35B-A3B
        arm: the sync-time call is 4.66 s of the 34.5 s ``sync_weights``, and every single
        ``[ISOEXEC-POSTSTEP]`` line in that run reads ``pre == post`` bitwise -- the copy is
        provably a no-op, because ``offload_megatron_model_to_cpu``/``load_megatron_model_to_gpu``
        move the buffer verbatim and cannot revert it to an older value. The guard keeps the fix
        for the configuration that needed it (a CPU-offloaded optimizer whose async H2D copy-back
        lands late, live DP8 run xl6cg6rr): there the fingerprint MOVES between the optim-time
        stamp and this call, and the full refresh runs exactly as before.

        Live DP8 run xl6cg6rr: under the precision-aware CPU-offloaded optimizer with
        overlap_cpu_optimizer_d2h_h2d=true, the bytes extracted for weight sync lagged the
        optimizer by ~2 steps (sender abs-sum checksums at syncs #1-#3 all byte-equal to the
        initial weights; first change only at sync #4; the per-sync delta ramps like the lr
        warmup shifted by two steps), while the next scoring forward saw fresher values ->
        rollout-vs-train logprob mismatch ~0.01 from step 3 onward. Transport itself was
        verified byte-faithful (all 31 receiver post-sync totals == sender checksums).

        Force the main->model param copy and a synchronous param all-gather (both idempotent
        if optimizer.step already did them) and drain CUDA streams. The pre/post checksums
        make (residual) staleness observable per step: pre != post pins the staleness to
        optimizer.step itself; pre == post but SENDER != POSTSTEP at the next sync pins it
        to the offload/reload cycle between train and sync.
        """
        if skip_if_unchanged and self._isoexec_params_provably_unchanged():
            return

        for _opt in iter_opts(self.optimizer):
            # STEP 1 (CPU-offload only): the fresh weights live in the INNER HybridDeviceOptimizer's
            # CPU master (self.optimizer is the OUTER DistributedOptimizer; the hybrid optimizer is at
            # `_opt.optimizer`, as megatron_utils reaches it). Its copy-back to the outer fp32 master
            # runs ASYNC on _h2d_stream, so `_copy_main_params_to_model_params` below copies a STALE
            # master -> the extracted model buffer never changes (SENDER bitwise-constant across every
            # sync -> engine rolls out stale -> step-2+ gate ~0.02). Force the CPU-master -> fp32-master
            # copy SYNCHRONOUSLY here. No-op on the 7B (no offload -> cpu_copys_map_gpu_param absent).
            for hopt in filter(None, (getattr(_opt, "optimizer", None), _opt)):
                cpu2gpu = getattr(hopt, "cpu_copys_map_gpu_param", None)
                if cpu2gpu:
                    # BATCHED copies. The per-parameter loop this replaces issued one H2D transfer
                    # per tensor -- at 35B/EP8 that is tens of thousands of latency-bound PCIe ops
                    # per call, twice per sync. _foreach_copy_ submits the same copies as one
                    # multi-tensor op: bitwise-identical data movement, a fraction of the launches.
                    dsts = list(cpu2gpu.values())
                    srcs = [c.data for c in cpu2gpu.keys()]
                    torch._foreach_copy_([d.data for d in dsts], srcs)
                    on_gpu = getattr(hopt, "gpu_params_map_cpu_copy", {}) or {}
                    pairs = [
                        (p, f) for p, f in (getattr(hopt, "param_to_fp32_param", None) or {}).items() if p not in on_gpu
                    ]
                    if pairs:
                        torch._foreach_copy_([p.data for p, _ in pairs], [f.data for _, f in pairs])
                    torch.cuda.synchronize()
            # STEP 2: copy the (now-fresh) fp32 master -> bf16 model params.
            copy_fn = getattr(_opt, "_copy_main_params_to_model_params", None)
            if copy_fn is not None:
                try:
                    copy_fn()
                except Exception as e:
                    print(f"[ISOEXEC-POSTSTEP] _copy_main_params_to_model_params failed: {e}", flush=True)
        for chunk in self.actor_module:
            sync_fn = getattr(chunk, "start_param_sync", None)
            if sync_fn is not None:
                try:
                    sync_fn(force_sync=True)
                except Exception as e:
                    print(f"[ISOEXEC-POSTSTEP] start_param_sync(force_sync=True) failed: {e}", flush=True)
        # Drain the param all-gather so the weight-sync extraction (and the next scoring forward) read
        # the fully-settled model buffer, not an in-flight one.
        torch.cuda.synchronize()
        # Stamp the whole-buffer fingerprint the SYNC-time call compares against
        # (`skip_if_unchanged`). Two big reductions over the flat DDP param buffers -- ~ms against
        # the seconds this routine costs -- and it is what makes the skip evidence-based rather
        # than an assumption about the offload path.
        self._ix_fresh_witness = self._isoexec_param_buffer_witness()

    def get_lr(self) -> Optional[float]:
        """
        Get current learning rate from optimizer.

        Handles both regular optimizers and ChainedOptimizer. Returns None when
        the worker was initialized with ``policy.inference_only_init=True``.
        """
        if self.optimizer is None:
            return None
        if isinstance(self.optimizer, ChainedOptimizer):
            return self.optimizer.chained_optimizers[0].param_groups[0]["lr"]
        return self.optimizer.param_groups[0]["lr"]

    def set_lr(self, learning_rate: float) -> None:
        """
        Set learning rate for the optimizer.

        Handles both regular optimizers and ChainedOptimizer (used with
        distributed optimizer). Updates all param_groups across all
        underlying optimizers.

        Note: This bypasses the scheduler. The next scheduler.step() call
        will override this value unless the scheduler is configured for
        constant LR. No-op when ``policy.inference_only_init=True``.
        """
        if self.optimizer is None:
            return
        if isinstance(self.optimizer, ChainedOptimizer):
            # ChainedOptimizer wraps multiple optimizers (e.g., for different param groups)
            for opt in self.optimizer.chained_optimizers:
                for param_group in opt.param_groups:
                    param_group["lr"] = learning_rate
        else:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = learning_rate

    async def init_weight_sync_state(self, inference_engine_client, inference_engine_cfg: "InferenceEngineConfig"):
        # Call super first to set _transfer_strategy_cls and create sender/receivers
        await super().init_weight_sync_state(inference_engine_client, inference_engine_cfg)

        # Initialize weight extractor with bucketing enabled for all strategies
        self.weight_extractor = MegatronWeightExtractor(
            bridge=self.bridge,
            actor_module=self.actor_module,
            enable_bucketing=True,
            bucket_size_threshold_GB=inference_engine_cfg.weight_transfer_threshold_cuda_ipc_GB,
            training_dtype=torch.bfloat16 if self.cfg.bf16 else torch.float32,
        )

    async def _save_lora_adapters_and_sync(
        self, lora_sync_path, inference_engine_client, lora_name: str = SKYRL_LORA_ADAPTER_NAME
    ):
        """Export LoRA adapter weights via Megatron-Bridge and tell the inference engine to load them.

        All ranks participate in the collective export (TP/PP/EP gathering is
        handled internally by the bridge).  Only rank 0 writes to disk and
        sends the ``LoraLoadRequest``.
        """
        import json

        from megatron.bridge.models.conversion.peft_bridge import (
            build_adapter_config_dict,
            infer_target_modules_from_adapter_weights,
        )
        from safetensors.torch import save_file

        adapter_state = {}
        for name, tensor in self.bridge.export_adapter_weights(self.actor_module, cpu=True, show_progress=False):
            adapter_state[f"base_model.model.{name}"] = tensor.clone().float()

        if torch.distributed.get_rank() == 0:
            os.makedirs(lora_sync_path, exist_ok=True)

            target_modules = infer_target_modules_from_adapter_weights(adapter_state.keys())
            base_model_name_or_path = str(
                getattr(self.bridge.hf_pretrained, "model_name_or_path", "")
                or getattr(self.bridge.hf_pretrained, "name_or_path", "")
            )
            adapter_config = build_adapter_config_dict(
                self.lora_cls,
                target_modules=target_modules,
                base_model_name_or_path=base_model_name_or_path,
            )

            save_file(adapter_state, os.path.join(lora_sync_path, "adapter_model.safetensors"))
            with open(os.path.join(lora_sync_path, "adapter_config.json"), "w", encoding="utf-8") as f:
                json.dump(adapter_config, f, ensure_ascii=False, indent=4)

            # Send LoRA disk loading request to inference engine.
            from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
                RemoteInferenceClient,
            )

            if isinstance(inference_engine_client, RemoteInferenceClient):
                await inference_engine_client.load_lora_adapter(lora_name, lora_sync_path)
            else:
                lora_request = LoraLoadRequest(lora_path=lora_sync_path, lora_name=lora_name)
                await inference_engine_client.update_named_weights(lora_request)

        torch.distributed.barrier()

    async def broadcast_to_inference_engines(
        self,
        inference_engine_client: "InferenceEngineInterface",
        inference_engine_cfg: "InferenceEngineConfig",
        model_id: Optional[str] = None,
    ):
        use_prefix_cache = inference_engine_cfg.enable_prefix_caching
        generator_dtype = str_to_torch_dtype(inference_engine_cfg.model_dtype)
        cache_reset_task = None

        # Clear prefix cache for synchronous training or for async training if `clear_kv_cache_on_weight_sync` is set
        _ix_should_flush = (
            use_prefix_cache
            and torch.distributed.get_rank() == 0
            and (not self.cfg.fully_async.enabled or self.cfg.fully_async.clear_kv_cache_on_weight_sync)
        )
        if _ix_should_flush:
            # clear prefix cache
            cache_reset_task = inference_engine_client.reset_prefix_cache(reset_running_requests=True)

        torch.cuda.empty_cache()

        if self._is_lora and not self.cfg.policy.megatron_config.lora_config.merge_lora:
            # AdapterStore.swap_to has already made `model_id` the live adapter
            # before we get here; sync that adapter to vLLM under its own name
            # so sample(model=<model_id>) routes correctly. Single-tenant
            # (model_id=None) keeps the legacy shared path + name.
            lora_name, lora_sync_path = self._resolve_lora_sync_target(model_id)
            await self._save_lora_adapters_and_sync(lora_sync_path, inference_engine_client, lora_name=lora_name)
        else:
            # SkyRL-IsoExec: drain ALL CUDA streams before reading params for extraction, so a
            # pending async param update (CPU-offloaded optimizer H2D copy-back, buffer reload)
            # can never make the extracted bytes lag the values the next scoring forward sees.
            # Also park the DDP grad buffer on CPU for the duration of the send: the broadcast now
            # runs with the engine KV pool already allocated (all wake_up calls precede the sync —
            # see save_weights_for_sampler), and the idle ~14GB grad buffer is what pushed the
            # packed-chunk staging allocations over the 80GB budget (2 OOMs at step-1 sync).
            # Post-optimizer-step grad values are dead; restore symmetrically so the dispatch
            # offload state machine is unaffected.
            _ix_grads_parked = False
            if os.environ.get("SKYRL_ISOEXEC") == "1":
                # Rowinv ENGAGEMENT boundary (trainer side), before the expensive extraction: a
                # contract that selects rowinv_leaftree while this process's census never served
                # it must refuse at the first post-step sync, not after hours of one-sided
                # composition (contract hashes MATCH in that failure -- both sides carry the flag,
                # only one executes). Exact no-op while the flag is off; the init sync (no forward
                # has run yet) is granted inside the boundary. Deliberate refusals propagate;
                # everything else is fail-safe inside the call.
                from skyrl.backends.skyrl_train.isoexec.core.enforce import (
                    rowinv_engagement_boundary,
                )

                rowinv_engagement_boundary("trainer")
                torch.cuda.synchronize()
                # Re-freshen the model buffer from the (CPU-offloaded) optimizer master RIGHT BEFORE
                # extraction -- belt-and-braces against any offload/reload between the optimizer step
                # and this sync reverting the model params to a stale value. Without this the SENDER
                # checksum was bitwise-constant across every sync (fully stale extraction). Idempotent.
                if getattr(self, "optimizer", None) is not None:
                    self._isoexec_force_fresh_model_params(skip_if_unchanged=True)
                # Re-canonicalize TP-replicated params (router/layernorms) to rank-0's copy so
                # the mismatched-TP sender exports ONE replica to every engine and the scoring
                # forward reads the same bytes. See _isoexec_sync_replicated_params.
                self._isoexec_sync_replicated_params()
                from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
                    load_megatron_grads_to_gpu,
                    offload_megatron_grads_to_cpu,
                )

                offload_megatron_grads_to_cpu(self.actor_module)
                torch.cuda.synchronize()
                _ix_grads_parked = True
            # Extract and send weights using the sender created at init time.
            # Disable expandable_segments around the send: under colocate_all the
            # CUDA-IPC path calls cudaIpcGetMemHandle, which is incompatible with the
            # VMM addresses expandable segments uses.
            try:
                with self._expandable_segments_disabled_for_sync():
                    weight_metadata = self.weight_extractor.get_weight_metadata(generator_dtype)
                    # extract+reshard (the all-gather reshard runs inside extract_weights) + pack +
                    # broadcast/send are fused in this generator-driven call.
                    await self._weight_transfer_sender.send_chunks(
                        self.weight_extractor.extract_weights(generator_dtype),
                        weight_metadata=weight_metadata,
                    )
            finally:
                if _ix_grads_parked:
                    load_megatron_grads_to_gpu(self.actor_module)

        if cache_reset_task is not None:
            await cache_reset_task
        # LIFECYCLE ASSERT (flush-prefix-cache-on-sync): with prefix caching on and the flush
        # condition met, reset_prefix_cache MUST have been invoked this sync. Observation-only,
        # fail-soft (isoexec/lifecycle/ordering.py).
        from skyrl.backends.skyrl_train.isoexec.lifecycle import ordering as _ix_order

        _ix_order.check_prefix_cache_flush_on_sync(use_prefix_cache, _ix_should_flush, cache_reset_task is not None)
        torch.cuda.empty_cache()
        torch.distributed.barrier()

    def _set_pad_token_id(self, pad_token_id):
        # this already gets set in the init_model method
        pass

    # ------------------------------------------------------------------
    # Multi-LoRA / AdapterStore Ray-callable methods
    # ------------------------------------------------------------------

    def prime_optimizer_state(self) -> None:
        """Materialise DistributedOptimizer state (exp_avg / exp_avg_sq).

        Adam's state tensors are allocated lazily on the first non-trivial
        step; without priming, the pristine snapshot would miss them.
        Megatron exposes ``_init_optimizer_states_with_dummy_values()`` which
        zero-fills grads + steps once + zero_grads, leaving the model weights
        unchanged.
        """
        if not self._is_lora:
            raise RuntimeError("prime_optimizer_state is only used on the LoRA path")
        for _opt in iter_opts(self.optimizer):
            init_fn = getattr(_opt, "_init_optimizer_states_with_dummy_values", None)
            if init_fn is not None:
                init_fn()

    def register_pristine_adapter(self) -> None:
        """Capture the current (freshly-initialised) LoRA state as the
        pristine slot. Must be called once per worker, after
        prime_optimizer_state.
        """
        if self.adapter_store is None:
            raise RuntimeError("AdapterStore not initialised (FFT path)")
        signature = LoraSignature.from_lora_config(
            self.cfg.policy.model.lora,
            lora_type=self.cfg.policy.megatron_config.lora_config.lora_type,
        )
        self.adapter_store.register_pristine(self.actor_module, self.optimizer, signature)

    def register_adapter(self, model_id: str) -> None:
        """Register a new LoRA adapter slot. The first call uses the live
        state as the slot; subsequent calls seed from pristine.
        """
        if self.adapter_store is None:
            raise RuntimeError("AdapterStore not initialised (FFT path)")
        signature = self.adapter_store.signature
        if signature is None:
            raise RuntimeError("register_adapter called before register_pristine_adapter")
        self.adapter_store.create(model_id, self.actor_module, self.optimizer, signature)

    def delete_adapter(self, model_id: str) -> None:
        if self.adapter_store is None:
            raise RuntimeError("AdapterStore not initialised (FFT path)")
        self.adapter_store.delete(model_id)
        # Drop the per-tenant safetensors subdir written by
        # _save_lora_adapters_and_sync. Rank 0 wrote it; rank 0 cleans it.
        # Other ranks no-op. Best-effort — log on failure but don't propagate.
        if self._rank == 0:
            _, lora_sync_path = self._resolve_lora_sync_target(model_id)
            base_sync_path = self.cfg.policy.model.lora.lora_sync_path
            if lora_sync_path != base_sync_path:
                try:
                    shutil.rmtree(lora_sync_path)
                except FileNotFoundError:
                    pass  # already gone, fine
                except OSError as e:
                    logger.warning(f"Failed to remove lora_sync subdir {lora_sync_path}: {e}")

    def swap_to_adapter(self, model_id: str) -> None:
        """Make ``model_id`` the live adapter on this worker. No-op if it
        already is. Issues local tensor.copy_()s + dp_group barriers.
        """
        if self.adapter_store is None:
            return  # FFT path: no-op
        self.adapter_store.swap_to(model_id, self.actor_module, self.optimizer)


class MegatronRefWorkerBase(MegatronWorker, RefWorkerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: MegatronModelWrapper = None
        self.actor_module: List[nn.Module] = None

    def forward(self, data: TrainingInputBatch) -> WorkerOutput:
        """Run inference forward pass.

        Returns a :class:`WorkerOutput` whose ``loss_fn_outputs`` carries one
        per-sample dict with key ``"logprobs"``. Token-based micro-batching (when
        ``max_tokens_per_microbatch > 0``) is handled inside ``_forward_logprobs``.
        """
        log_probs = self._forward_logprobs(data)
        loss_fn_outputs = [{"logprobs": log_probs[i].tolist()} for i in range(log_probs.shape[0])]
        return WorkerOutput(loss_fn_outputs=loss_fn_outputs, metrics={})

    def init_worker_process_group(self):
        """
        Override DistributedTorchRayActor.init_worker_process_group to use megatron distributed setup to create the mesh.
        """
        if not torch.distributed.is_initialized():
            # Ensure CUDA device is set before process group init — required when
            # using split "cpu:gloo,cuda:nccl" backend to avoid 'invalid device ordinal'
            # errors during NCCL communicator creation in subgroups.
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            # Default torch dist pg init timeout is 10 minutes (600 seconds)
            torch.distributed.init_process_group(
                backend="cpu:gloo,cuda:nccl", timeout=timedelta(seconds=SKYRL_WORKER_NCCL_TIMEOUT_IN_S)
            )

        self.strategy = MegatronStrategy(
            megatron_config=self.cfg.ref.megatron_config,
            optimizer_config=None,
            seed=self.cfg.seed,
            node_local_rank=self._local_rank,
        )
        self.strategy.setup_distributed()

        self.mesh_rank = MeshRank(
            dp=mpu.get_data_parallel_rank(),
            sp=mpu.get_context_parallel_rank(),
            tp=mpu.get_tensor_model_parallel_rank(),
            pp=mpu.get_pipeline_model_parallel_rank(),
            world_size=self._world_size,
            dp_size=mpu.get_data_parallel_world_size(),
            pp_size=mpu.get_pipeline_model_parallel_world_size(),
        )

    def init_model(self, model_path, num_training_steps: int = 1e9):
        """
        Initialize the model for the ref worker.
        """
        # initialize the bridge and provider objects
        self.init_configs(
            model_path,
            self.cfg.ref.megatron_config,
            self.cfg.ref.megatron_config.model_config_kwargs,
            self.cfg.ref.megatron_config.transformer_config_kwargs,
            bf16=self.cfg.bf16,
            flash_attn=self.cfg.flash_attn,
            language_model_only=self.cfg.ref.language_model_only,
        )

        self.actor_module = self.make_megatron_module(
            wrap_with_ddp=False,
            ddp_config=None,
            bf16=self.cfg.bf16,
        )

        # download model weights from huggingface (need to be done for ref worker as well, else errors when colocate_all=False)
        if self._local_rank == 0 and not os.path.exists(
            model_path
        ):  # if not local path, try downloading model weights from huggingface
            snapshot_download(model_path)  # will be no-op if already downloaded
        torch.distributed.barrier()

        # load weights
        if self._rank == 0:
            print_model_size(self.actor_module[0])

        # create worker model
        self.model = MegatronModelWrapper(config=self.cfg, actor_module=self.actor_module)

        self._set_expandable_segments(True)

    def _set_pad_token_id(self, pad_token_id):
        # this already gets set in the init_model method
        pass


class MegatronCriticWorkerBase(MegatronWorker, CriticWorkerBase):
    def __init__(self, **kwargs):
        raise NotImplementedError()


PolicyWorker = ray.remote(num_gpus=1)(MegatronPolicyWorkerBase)
RefWorker = ray.remote(num_gpus=1)(MegatronRefWorkerBase)
CriticWorker = ray.remote(num_gpus=1)(MegatronCriticWorkerBase)
