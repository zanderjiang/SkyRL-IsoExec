import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from skyrl.backends.skyrl_train.weight_sync.transfer_strategy import (
        WeightSyncInitInfo,
    )
import asyncio
import time
from dataclasses import dataclass
from http import HTTPStatus
from types import SimpleNamespace
from uuid import uuid4

import ray
import vllm
from loguru import logger
from vllm import SamplingParams
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponse,
)
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse
from vllm.entrypoints.openai.models.serving import (
    BaseModelPath,
    OpenAIModelRegistry,
    OpenAIServingModels,
)
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest

from skyrl.backends.skyrl_train.inference_engines.base import (
    InferenceEngineInput,
    InferenceEngineInterface,
    InferenceEngineOutput,
)
from skyrl.backends.skyrl_train.inference_engines.vllm.utils import pop_openai_kwargs

# Backward compatibility: WorkerWrap has moved to inference_servers.vllm_worker
# This alias preserves the old import path for existing scripts/configs.
# TODO (Kourosh): Remove this alias once all references are updated.
from skyrl.backends.skyrl_train.inference_servers.vllm_worker import (
    WorkerWrap,  # noqa: F401, E402
)
from skyrl.backends.skyrl_train.weight_sync import WeightLoader, WeightUpdateRequest


@dataclass
class Logprob:
    logprob: float
    rank: int
    token_id: str


def _cudagraph_enabled() -> bool:
    """Whether the engine should use its admitted CUDA-graph configuration."""
    return os.environ.get("SKYRL_ISOEXEC_ENABLE_CUDAGRAPH") == "1"


# Env-var name prefixes that must reach vLLM's RAY worker actors. Not IsoExec specific in form --
# any stack that configures its workers through the environment needs this -- but IsoExec is what
# makes it load-bearing today. See _prepare_ray_executor_env.
_RAY_WORKER_ENV_PREFIXES = ("SKYRL_", "VARLEN_")

# The load-bearing vars that match NO prefix -- neither vLLM's built-ins (VLLM_ LMCACHE_ NCCL_ UCX_
# HF_ HUGGING_FACE_) nor ours. They are set on the engine actor's Ray runtime_env or the job's, so
# under `mp` (spawn: the child inherits os.environ) and under RayExecutorV2 (whole-environ copy)
# they arrive for free; only the v1 allowlist drops them. Every one of these changes behaviour
# silently rather than loudly if it goes missing: the allocator config, the fatal-signal handler
# that is the only forensics when a worker dies, the Megatron/TE kernel selectors.
_RAY_WORKER_ENV_VARS = (
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTHONFAULTHANDLER",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "FLA_TILELANG",
    "NVTE_FUSED_ATTN",
)


def _prepare_ray_executor_env() -> None:
    """Make the ``ray`` executor's worker ACTORS see the environment this engine actor holds.

    THE ASYMMETRY THIS EXISTS TO CLOSE. A worker that is a plain subprocess of this engine actor
    inherits ``os.environ`` wholesale -- including everything set *inside* this process
    (``apply_vllm_isoexec_env``: VLLM_BATCH_INVARIANT, the NCCL pins) and every ``SKYRL_ISOEXEC_*``
    flag. Ray worker ACTORS inherit nothing implicitly, and HOW they get an environment depends on
    WHICH ray executor vLLM picks -- there are two, and they do not agree:

      * ``RayExecutorV2`` (``v1/executor/ray_executor_v2.py:205``, a SUBCLASS of MultiprocExecutor)
        ships the driver's WHOLE ``os.environ`` to each worker actor (``ray_env_utils.py:8-18``
        -> ``initialize_worker``, ``ray_executor_v2.py:319,394``, setdefault semantics). Nothing to
        fix; this branch is a no-op for it.
      * ``RayDistributedExecutor`` (``v1/executor/ray_executor.py:64``) copies from an explicit
        ALLOWLIST instead: ``ray/ray_env.py:55-95``, vLLM's own registered vars plus the prefixes
        ``VLLM_ LMCACHE_ NCCL_ UCX_ HF_ HUGGING_FACE_``. ``SKYRL_*`` and
        ``VARLEN_FORCE_NUM_SPLITS_1`` match NONE of those.

    Ray's runtime_env inheritance would probably carry them anyway (job-level ``env_vars`` are
    inherited by child actors), but "probably" is exactly the wrong guarantee here: a flag that
    fails to arrive does not raise, it silently selects a DIFFERENT kernel in that worker -- vLLM's
    default attention backend instead of the num_splits=1 CUSTOM varlen -- and the only symptom is a
    logprob gate that moves. vLLM offers a first-class, documented lever for exactly this
    (``VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY``, ``envs.py:1314``, read lazily at executor init), so
    use it and stop depending on inheritance semantics. Additive: any operator-supplied value is
    preserved.

    Model-agnostic by construction -- the bulk of it forwards a PREFIX, not a list of flags, so a
    new recipe or a new ``SKYRL_ISOEXEC_*`` knob needs no change here. ``_RAY_WORKER_ENV_VARS`` is
    the short residue of vars that match no prefix at all and must be named.
    """

    def _extend_csv(var: str, additions) -> list[str]:
        """Additive, idempotent, order-preserving merge into a comma-separated env var."""
        vals = [tok.strip() for tok in os.environ.get(var, "").split(",")]
        vals = [tok for tok in vals if tok]
        for a in additions:
            if a not in vals:
                vals.append(a)
        os.environ[var] = ",".join(vals)
        return vals

    prefixes = _extend_csv("VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY", _RAY_WORKER_ENV_PREFIXES)
    # `..._VARS_TO_COPY` is the individual-name twin of the prefix lever (``envs.py:1321``, merged
    # with vLLM's built-ins at ``ray/ray_env.py:93``). Only forward what this process actually has,
    # so the banner reports reality rather than intent.
    named = _extend_csv("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", [v for v in _RAY_WORKER_ENV_VARS if v in os.environ])
    logger.info(
        f"[isoexec] ray executor: forwarding env prefixes {prefixes} and vars {named} to vLLM ray worker actors"
    )


def setup_envvars_for_vllm(kwargs, bundle_indices, standalone_harness: bool = False):
    """Prepare the process environment for a vLLM engine build."""
    noset_visible_devices = kwargs.pop("noset_visible_devices")
    mp_cuda_visible_devices = kwargs.pop("mp_cuda_visible_devices", None)

    if os.environ.get("SKYRL_ISOEXEC") == "1" and not standalone_harness:
        _backend = kwargs.get("distributed_executor_backend")
        if _backend not in (None, "ray", "uni"):
            raise ValueError(
                f"[isoexec] distributed_executor_backend={_backend!r} is not supported: the IsoExec "
                "stack is ray-only (the mp path was removed 2026-08-13, after ray was proven on "
                "three arms). Set generator.inference_engine.distributed_executor_backend=ray, and "
                "keep VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1 so vLLM resolves ray to RayExecutorV2 "
                "(the RayDistributedExecutor alternative silently loses async scheduling)."
            )

    if kwargs.get("distributed_executor_backend") == "mp" and mp_cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = mp_cuda_visible_devices
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
        logger.info(f"mp backend: setting CUDA_VISIBLE_DEVICES={mp_cuda_visible_devices}")
    elif kwargs.get("distributed_executor_backend") in ("ray", "mp"):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
    elif noset_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])

    if kwargs.get("distributed_executor_backend") == "ray":
        _prepare_ray_executor_env()

    num_gpus = kwargs.pop("num_gpus")
    if bundle_indices is not None:
        os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(num_gpus)
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
        logger.info(f"creating LLM with bundle_indices={bundle_indices}")

    # SkyRL-IsoExec: enable vLLM batch-invariant numerics so the rollout engine matches the
    # Megatron trainer bitwise. Gated by env so it is fully reversible / opt-in. Must run
    # before the vLLM engine is constructed (this function does exactly that).
    if os.environ.get("SKYRL_ISOEXEC") == "1":
        from skyrl.backends.skyrl_train.isoexec import (
            apply_vllm_isoexec_env,
            isoexec_engine_arg_overrides,
        )
        from skyrl.backends.skyrl_train.isoexec.runtimes.vllm.compile_guard import (
            assert_compilation_admissible,
        )
        from skyrl.backends.skyrl_train.isoexec.runtimes.vllm.gptmodel_vllm import (
            VLLM_MODEL_NAME,
            register_gptmodel_to_vllm,
        )
        from skyrl.backends.skyrl_train.isoexec.runtimes.vllm.vllm_patches import (
            neutralize_vllm_nccl_channel_pin,
        )

        apply_vllm_isoexec_env()
        # Must run BEFORE the engine is constructed: vLLM's init_batch_invariance re-pins
        # NCCL_MIN/MAX_NCHANNELS=1 in-process (gpu_worker.py:1173, before
        # init_distributed_environment at :1191), so clearing them in the environment alone
        # cannot win -- which is why two live pin A/Bs read as null. No-op unless
        # SKYRL_ISOEXEC_ENGINE_NCCL_UNPIN=1 (the ENGINE-only flag; SKYRL_ISOEXEC_NCCL_PIN is the
        # TRAINER's and does NOT gate this). See vllm_patches for the measurement.
        neutralize_vllm_nccl_channel_pin()
        # CRITICAL for IsoExec: force prefix caching + chunked prefill OFF (and enforce_eager ON).
        # Prefix caching reuses KV computed in a DIFFERENT batch context than the trainer's clean
        # single-sequence forward, and chunked prefill splits a prompt across forward steps -- both
        # break batch-invariance, so the rollout (decode) logprobs drift ~0.01 from a clean recompute
        # even though VLLM_BATCH_INVARIANT=1 makes the kernels deterministic. (SkyRL config defaults
        # both to True; without this override the rollout_train_logprobs_abs_diff floors at ~0.0104
        # instead of the ~1e-5 cross-runtime floor.) These cannot be set via env -> mutate kwargs.
        # ROLLOUT-ACCEL A/B (env-gated, default OFF -> unchanged conservative config). The IsoExec
        # engine defaults force prefix caching, chunked prefill, and CUDA graphs OFF. These three
        # flags re-enable them INDIVIDUALLY so we can A/B whether the num_splits=1 CUSTOM varlen
        # backend keeps rollout==train bitwise with each on. The check is policy_kl /
        # minibatch_rollout_logprobs_abs_diff staying at the ~1e-6 floor. Per-feature so a
        # cudagraph-capture failure on the custom model can be bisected out without a code change.
        _accel_prefix = os.environ.get("SKYRL_ISOEXEC_ENABLE_PREFIX_CACHE") == "1"
        _accel_chunked = os.environ.get("SKYRL_ISOEXEC_ENABLE_CHUNKED_PREFILL") == "1"
        _accel_graph = _cudagraph_enabled()
        # G2: GDN recurrent chunked prefill. Continuation chunks resume from the prompt's carried
        # ssm/conv state (RecurrentGDN.prefill), so chunked prefill is bitwise-safe under RECURRENT
        # GDN -- unlike the CHUNK kernel, whose ssm_state is a chunk-BOUNDARY state that cannot be
        # resumed mid-prompt. Requires GDN on AND recurrent mode; default OFF. This is a prerequisite
        # for G3 (prefix caching), not a standalone speedup: gsm8k prompts fit one chunk.
        from skyrl.backends.skyrl_train.isoexec.ops.gdn.gdn_ops import (
            chunk_synced_mode as _gdn_chunk_synced_mode,
        )
        from skyrl.backends.skyrl_train.isoexec.ops.gdn.gdn_ops import (
            recurrent_mode as _gdn_recurrent_mode,
        )

        _gdn_chunked = (
            os.environ.get("SKYRL_ISOEXEC_GDN_CHUNKED_PREFILL") == "1"
            and os.environ.get("SKYRL_ISOEXEC_GDN") == "1"
            and (_gdn_recurrent_mode() or _gdn_chunk_synced_mode())
        )
        for _k, _v in isoexec_engine_arg_overrides().items():
            if _accel_graph and _k == "enforce_eager":
                _v = False
            if _accel_prefix and _k == "enable_prefix_caching":
                _v = True
            if kwargs.get(_k) != _v:
                logger.info(f"[isoexec] forcing engine arg {_k}={_v} (was {kwargs.get(_k)})")
            kwargs[_k] = _v
        # Chunked prefill. Default IsoExec forces it OFF (vLLM then requires
        # max_num_batched_tokens >= max_model_len, so cap max_model_len and bump the budget). The
        # accel flag re-enables it (chunking lifts that constraint); max_model_len is still capped
        # to prompt+response for the generation-length bound.
        if _gdn_chunked:
            # G2 GDN chunked prefill. Chunking LIFTS vLLM's max_num_batched_tokens >= max_model_len
            # requirement (the budget becomes the chunk size), so unlike the NO_CHUNKED branch below
            # we are free to leave the budget under max_model_len. We keep the SAME cap the non-chunked
            # path uses (max_model_len, or SKYRL_ISOEXEC_MAX_BATCHED_TOKENS if set) purely as the
            # full-vocab fp32-logit profiling-OOM guard documented below -- NOT because vLLM demands
            # it. max_model_len is still capped for the generation-length bound. enable_prefix_caching
            # stays hard OFF (that is G3, gated separately in the GDN block).
            _mml = int(os.environ.get("SKYRL_ISOEXEC_MAX_MODEL_LEN", "0") or 0)
            if _mml:
                kwargs["max_model_len"] = _mml
            _need = _mml or int(kwargs.get("max_model_len") or 0)
            if _need:
                kwargs["max_num_batched_tokens"] = (
                    int(os.environ.get("SKYRL_ISOEXEC_MAX_BATCHED_TOKENS", "0") or 0) or _need
                )
            kwargs["enable_chunked_prefill"] = True
            logger.info(
                "[isoexec] GDN CHUNKED PREFILL ON: enable_chunked_prefill=True "
                f"max_model_len={kwargs.get('max_model_len')} "
                f"max_num_batched_tokens={kwargs.get('max_num_batched_tokens')}"
            )
        elif os.environ.get("SKYRL_ISOEXEC_NO_CHUNKED_PREFILL") == "1" and not _accel_chunked:
            _mml = int(os.environ.get("SKYRL_ISOEXEC_MAX_MODEL_LEN", "0") or 0)
            if _mml:
                kwargs["max_model_len"] = _mml
            _need = _mml or int(kwargs.get("max_model_len") or 0)
            if _need:
                # EXACTLY max_model_len, not max(default, need): vLLM sizes its PROFILING forward by
                # max_num_batched_tokens, and the IsoExec wrapper materializes full-vocab fp32 logits
                # for every profile row (16384 rows x 248320 vocab = ~16 GB/rank on Qwen3.5) -- at
                # TP=4's 17.5 GB weight shard that OOMed engine init. Short prompts still pack many
                # per prefill within max_model_len. Overridable for long-prompt workloads.
                kwargs["max_num_batched_tokens"] = (
                    int(os.environ.get("SKYRL_ISOEXEC_MAX_BATCHED_TOKENS", "0") or 0) or _need
                )
            kwargs["enable_chunked_prefill"] = False
            logger.info(
                "[isoexec] forcing enable_chunked_prefill=False "
                f"max_model_len={kwargs.get('max_model_len')} "
                f"max_num_batched_tokens={kwargs.get('max_num_batched_tokens')}"
            )
        elif _accel_chunked:
            _mml = int(os.environ.get("SKYRL_ISOEXEC_MAX_MODEL_LEN", "0") or 0)
            if _mml:
                kwargs["max_model_len"] = _mml
            kwargs["enable_chunked_prefill"] = True
            logger.info(
                f"[isoexec] ROLLOUT-ACCEL: chunked_prefill=True prefix_cache={_accel_prefix} "
                f"cudagraph={_accel_graph} max_model_len={kwargs.get('max_model_len')}"
            )
        # Run Megatron's GPTModel inside vLLM (unified model) so the rollout == trainer. String
        # registration survives mp/async worker subprocesses (each lazily imports the wrapper).
        register_gptmodel_to_vllm()  # cross-process string form
        # THE COMPILE GUARD. Deliberately here -- AFTER registration (so the model class it asks
        # about is the one vLLM will build) and OUTSIDE every model/feature branch. The `mode=0`
        # statement further down is inside `if SKYRL_ISOEXEC_GDN == "1":`, which GLM never enters:
        # the GLM arm's own banner shows `mode: <CompilationMode.VLLM_COMPILE: 3>, backend:
        # 'inductor'`, and nothing compiles only because vLLM refuses an undecorated model class
        # (`vllm.py:2192`, x8 in that arm's log). One `@support_torch_compile` would turn inductor
        # on model-wide with no gate, and a compiled GEMM silently bypasses the batch-invariant
        # matmul (`aten::mm` functional override vs inductor's `aten.mm.out` extern call). This
        # call makes that impossible: no opt-in -> mode is pinned to NONE; opt-in -> the
        # dispatcher-preserving configuration and the pin itself must both verify, or it raises.
        # NO-OP ON EVERY SHIPPING PATH TODAY (no class is compile-eligible -> "inert").
        assert_compilation_admissible(kwargs)
        # isoexec Phase 2: build + log the composition contract hashes on the ENGINE side. Must
        # equal the trainer's [ISOEXEC-CONTRACT] identities (same code+model+arch -> same hash); a
        # mismatch across the two logs is a composition split-brain. Build+log only, never fatal
        # (behavior-preserving); the fatal assert runs worker-side at weight-sync receiver init.
        if os.environ.get("SKYRL_ISOEXEC"):
            try:
                from skyrl.backends.skyrl_train.isoexec.core.process_contract import (
                    get_process_contract,
                )

                get_process_contract(kwargs.get("model") or "")
            except Exception as _e:
                logger.warning(f"[isoexec] engine contract build skipped: {_e}")
        hf_overrides = dict(kwargs.get("hf_overrides") or {})
        hf_overrides["architectures"] = [VLLM_MODEL_NAME]
        kwargs["hf_overrides"] = hf_overrides
        # Nightly bitwise path (no-TE local spec): select the CUSTOM PyTorch-varlen attention
        # backend (num_splits=1 -> bitwise decode==prefill at all lengths). Importing the module
        # registers @register_backend(CUSTOM); run in-process so it is visible to the engine. This
        # is what drives rollout_train_logprobs_abs_diff to a true 0 (vs the ~0.01 floor of vLLM's
        # default flash backend, whose split-K heuristic makes long-context decode != prefill).
        if os.environ.get("SKYRL_ISOEXEC_LOCAL_SPEC") == "1":
            from skyrl.backends.skyrl_train.isoexec.ops.attention import (
                varlen_backend,  # noqa: F401
            )
            from skyrl.backends.skyrl_train.isoexec.runtimes.vllm.vllm_patches import (
                patch_vllm_logprobs_batch_invariant,
                patch_vllm_sampler_temperature,
            )

            if varlen_backend.register_varlen_custom_backend():
                kwargs["attention_backend"] = "CUSTOM"
                logger.info("[isoexec] using CUSTOM varlen attention backend (num_splits=1, bitwise decode==prefill)")
            else:
                logger.warning(
                    "[isoexec] torch.nn.attention.varlen unavailable; "
                    "CUSTOM backend NOT selected (IsoExec will not be bitwise)"
                )
            # The forward is bitwise, but vLLM's v2 sampler computes the ROLLOUT logprob with a fused
            # Triton kernel that bypasses aten log_softmax -> diverges from the trainer's log_softmax on
            # a few tokens. Route the generator through aten log_softmax (== trainer) for bitwise
            # rollout==train. In-process engine (VLLM_ENABLE_V1_MULTIPROCESSING=0) so this reaches the sampler.
            patch_vllm_logprobs_batch_invariant()
            # V1 Sampler.apply_temperature divides [B, V] fp32 by 1.0 every decode step (432 us at
            # B=512). x/1.0 is bit-preserving, so skipping it moves nothing.
            patch_vllm_sampler_temperature()
        # GatedDeltaNet (Qwen3.5 hybrid): 3 of every 4 layers are linear attention, whose decode
        # kernel disagrees with its own prefill kernel by ~1.7e-2 in logprob. Route both phases
        # through the training chunk kernel (chunk-consistent decode). This redefines ssm_state as a
        # chunk-BOUNDARY state, so prefix caching / chunked prefill / spec decode / CUDA graphs must
        # be off -- assert that rather than degrade quietly. (Those three are bitwise-safe for the
        # softmax layers and worth 4.6x rollout; GDN support for them is a follow-up.)
        if os.environ.get("SKYRL_ISOEXEC_GDN") == "1":
            from skyrl.backends.skyrl_train.isoexec.ops.gdn.gdn_ops import (
                recurrent_mode,
            )
            from skyrl.backends.skyrl_train.isoexec.runtimes.vllm.gdn_engine_patch import (
                assert_engine_args_compatible,
                install_gdn_engine_patch,
            )

            # G4 NATIVE KERNELS: the GDN core runs vLLM's own fused kernels (causal_conv1d_fn/update
            # + fused_sigmoid_gating core) against vLLM's native state, on the trainer AND the
            # engine. Resume-at-any-boundary is bitwise (gdn_native_kernel_parity_test), so chunked
            # prefill and prefix caching (align mode) both turn ON -- this is what closes the 15x
            # prefill gap to stock vLLM. Requires native state (fp32 blocks) + recurrent mode.
            _gdn_native_kernels = os.environ.get("SKYRL_ISOEXEC_GDN_NATIVE_KERNELS") == "1"
            _gdn_cs_mode = _gdn_chunk_synced_mode()
            if (
                _gdn_native_kernels
                and not _gdn_cs_mode
                and (os.environ.get("SKYRL_ISOEXEC_GDN_NATIVE_STATE") != "1" or not _gdn_recurrent_mode())
            ):
                raise ValueError(
                    "[isoexec] SKYRL_ISOEXEC_GDN_NATIVE_KERNELS=1 requires "
                    "SKYRL_ISOEXEC_GDN_NATIVE_STATE=1 and SKYRL_ISOEXEC_GDN_KERNEL=recurrent (the "
                    "native kernels index vLLM's own state blocks directly) -- OR "
                    "SKYRL_ISOEXEC_GDN_KERNEL=chunk_synced (v2: fused core + matched-prep boundary "
                    "pass against the private pool)."
                )
            if _gdn_native_kernels and _gdn_cs_mode and os.environ.get("SKYRL_ISOEXEC_GDN_NATIVE_STATE") == "1":
                raise ValueError(
                    "[isoexec] chunk_synced v2 keeps its own state pool (entry states + open-chunk "
                    "buffers live beside it); unset SKYRL_ISOEXEC_GDN_NATIVE_STATE."
                )
            if _gdn_native_kernels and not _gdn_cs_mode:
                kwargs["enable_prefix_caching"] = True
                kwargs["enable_chunked_prefill"] = True
                logger.info(
                    "[isoexec] GDN NATIVE KERNELS: prefix caching + chunked prefill ON "
                    f"(mamba align mode), max_num_batched_tokens={kwargs.get('max_num_batched_tokens')}"
                )
            else:
                # Prefix caching stays hard OFF outside G4. Chunked prefill is forced OFF too,
                # EXCEPT under G2 (recurrent GDN + SKYRL_ISOEXEC_GDN_CHUNKED_PREFILL): there the
                # continuation-chunk resume is bitwise-safe, and _gdn_chunked already set it True.
                kwargs["enable_prefix_caching"] = False
                if not _gdn_chunked:
                    kwargs["enable_chunked_prefill"] = False

            # G3 step 1: move recurrent GDN state ownership into vLLM's OWN mamba kv_cache blocks
            # (RecurrentGDN native-state path). The ssm cache MUST be fp32 -- the default
            # mamba_ssm_cache_dtype=auto resolves to bf16, which would round the recurrent state
            # every step and break the fp32 prefill->decode round-trip the whole design rests on
            # (conv stays bf16). Default OFF: the private pool is untouched. RecurrentGDN asserts the
            # resolved dtype at first forward, so a missed forwarding of this flag fails loud.
            if os.environ.get("SKYRL_ISOEXEC_GDN_NATIVE_STATE") == "1":
                kwargs["mamba_ssm_cache_dtype"] = "float32"
                logger.info(
                    "[isoexec] GDN NATIVE STATE=1: mamba_ssm_cache_dtype=float32 "
                    "(recurrent state lives in vLLM kv_cache; fp32 round-trip)"
                )

            # enforce_eager is NOT forced here any more. It used to be hard-set True unconditionally,
            # which silently defeated SKYRL_ISOEXEC_ENABLE_CUDAGRAPH -- the A/B flag had never once
            # been exercised. Chunk-consistent decode is now shape-static and sync-free (padded decode
            # + persistent index buffers + a masked chunk roll), and so is the MoE expert path, so the
            # graph is capturable. Keep the default eager; let the flag decide.
            kwargs.setdefault("enforce_eager", True)
            if _cudagraph_enabled():
                kwargs["enforce_eager"] = False
                # padded decode is what makes the CHUNK decode shapes static; graphs cannot capture
                # its ragged per-slot gather. RECURRENT decode is one token per sequence -- already
                # static, and it never reads the host -- so it needs no padding, and forcing the flag
                # would only advertise a code path (ChunkConsistentGDN.decode_dev) that mode never
                # enters. CHUNK_SYNCED decode is device-pure like recurrent (the boundary resync
                # runs host-driven in the metadata builder, between replays -- the LAZY driver
                # that assert_engine_args_compatible installs), so it needs no padding either.
                from skyrl.backends.skyrl_train.isoexec.ops.gdn.gdn_ops import (
                    chunk_synced_mode,
                )

                if not recurrent_mode() and not chunk_synced_mode():
                    os.environ["SKYRL_ISOEXEC_GDN_PADDED_DECODE"] = "1"
                # NO inductor, and capture PURE-DECODE batches only. enforce_eager=False alone gives
                # vLLM's default (mode=VLLM_COMPILE + FULL_AND_PIECEWISE), which is wrong twice over:
                #   * inductor REPLACES the eager ops the trainer runs with fused generated kernels --
                #     numerics that are not bitwise vs the trainer, and shape-specialized compiled
                #     kernels are exactly what produced the TP=2 gated-norm bug (report §9.4);
                #   * PIECEWISE captures MIXED prefill-decode batches, but chunk-consistent prefill
                #     mutates per-request host state (LRU, fill counts) that a graph replay would
                #     skip -- the 2026-07-12 GRAPHS3 run captured those graphs and desynced the
                #     scheduler at step 1 (KeyError in update_from_output).
                # FULL_DECODE_ONLY + CompilationMode.NONE is the decode_dev contract: the eager
                # model, with pure-decode steps replayed as one full graph; everything else eager.
                _cc = dict(kwargs.get("compilation_config") or {})
                _cc.setdefault("mode", 0)  # CompilationMode.NONE -- the model runs as-is
                _cc.setdefault("cudagraph_mode", "FULL_DECODE_ONLY")
                # Cover the REAL decode batch width. vLLM's default max capture size is 512, and a
                # decode batch wider than the largest captured graph dispatches EAGER -- with
                # n_samples_per_prompt x train_batch concurrent sequences (e.g. 640 on GSM8K) the
                # whole bulk of the rollout ran eager and the graphs only served the tail. Capture
                # up to max_num_seqs so every pure-decode step replays a graph.
                _mns = int(kwargs.get("max_num_seqs") or 0)
                if _mns > 512:
                    _cc.setdefault("max_cudagraph_capture_size", _mns)
                # ADMISSION NEEDS ONE EAGER PASS PER CAPTURE SHAPE, and this vLLM gives it none.
                # `CompilationConfig.cudagraph_num_of_warmups` defaults to **0** (vllm/config/
                # compilation.py:624), so `_warmup_and_capture` (v1/worker/gpu_model_runner.py:6586)
                # goes straight to the capture run. pik's two admission-gated paths -- the fused
                # barrier+reduce and the root-cast absorption -- both REFUSE to admit under capture
                # (a bit-pattern compare needs a host sync: pik/allreduce.py:708-712 and the
                # `tree_all_reduce_rounded` twin) and deliberately record no verdict, so every
                # decode shape is first seen inside its own capture, falls back to the reference
                # `_p2p_unfused`, and that fallback is what gets BAKED INTO THE GRAPH. Turning the
                # flag on would then measure exactly zero and look like a refutation of the kernel
                # rather than of the plumbing.
                #
                # One warmup replay per captured batch descriptor fixes it: the warmup runs with
                # `cudagraph_runtime_mode=NONE`, so `_capturing()` is False, admission runs on the
                # real shape, and the capture that follows records the ADMITTED path. Capture also
                # walks descriptors largest-first, so the fused kernel's flag pool reaches its final
                # size on the first warmup and never has to grow inside a capture (which raises by
                # design, pik/allreduce.py:514-522).
                #
                # Derived, not a new knob: both source flags default OFF, so production keeps
                # `cudagraph_num_of_warmups` absent and this branch is byte-identical to today.
                # AUTOFUSE rides the same warmup requirement: SiteDispatcher skips ledger
                # resolution under stream capture (host-side bit-compares are impossible there),
                # so without a warmup pass every captured decode shape freezes the EAGER/original
                # path into its graph permanently -- installed-but-serving-eager, the exact v10
                # engagement gap. On v10 this only worked because PIK_FUSED_BARRIER happened to be
                # on; a GLM arm exporting AUTOFUSE alone must not silently lose engagement.
                if (
                    os.environ.get("SKYRL_ISOEXEC_PIK_FUSED_BARRIER", "0") not in ("", "0", "false", "no", "off")
                    # ROOT_CAST and AUTOFUSE default ON since 2026-08-13, so this branch is now
                    # the normal path rather than an arm's opt-in: the warmup replay is what makes
                    # an admission-gated feature reach the capture ADMITTED instead of frozen eager.
                    or (os.environ.get("SKYRL_ISOEXEC_PIK_FUSED_ROOT_CAST", "1") not in ("", "0", "false", "no", "off"))
                    # wave-10: the fused OWNER-COMBINE is admission-gated the same way and would
                    # fail the same way. Its flag pool additionally must be SIZED before capture
                    # (growing a symmetric allocation inside a capture raises by design), and
                    # capture walks descriptors largest-first, so one warmup per descriptor both
                    # admits the geometry and reaches the final flag stride on the first pass.
                    or (
                        os.environ.get("SKYRL_ISOEXEC_PIK_FUSED_OWNER_COMBINE", "0")
                        not in ("", "0", "false", "no", "off")
                    )
                    or os.environ.get("SKYRL_ISOEXEC_AUTOFUSE", "1") == "1"
                ):
                    _cc.setdefault("cudagraph_num_of_warmups", 1)
                    logger.info(
                        "[isoexec] admission-gated path requested (pik fused barrier/root-cast or "
                        "autofuse): cudagraph_num_of_warmups=1 so every captured decode shape gets "
                        "one EAGER pass to be admitted on, before the capture that would freeze "
                        "the reference path in."
                    )
                kwargs["compilation_config"] = _cc
                logger.info(
                    f"[isoexec] CUDA graphs ON: enforce_eager=False, GDN padded decode forced, compilation_config={_cc}"
                )
            assert_engine_args_compatible(kwargs)
            install_gdn_engine_patch()

        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        logger.info(f"[isoexec] vLLM will run GPTModel (arch={VLLM_MODEL_NAME}) via hf_overrides")


class BaseVLLMInferenceEngine(InferenceEngineInterface):
    """Base class containing shared logic between sync and async VLLM engines."""

    def __init__(self, *args, bundle_indices: list = None, **kwargs):
        # Redirect infrastructure output to log file before any engine initialization.
        # Done here in the base class so all subclasses get it automatically.
        from skyrl.train.utils.ray_logging import redirect_actor_output_to_file

        redirect_actor_output_to_file()

        if os.environ.get("SKYRL_RAY_PY_EXECUTABLE"):
            from skyrl.backends.skyrl_train.isoexec.runtimes.megatron.te_primitives import (
                runtime_identity,
            )

            runtime_identity("engine-actor")

        setup_envvars_for_vllm(kwargs, bundle_indices)
        vllm_v1_disable_multiproc = kwargs.pop("vllm_v1_disable_multiproc", False)
        if vllm_v1_disable_multiproc:
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        # Store common attributes
        self._tp_size = kwargs.get("tensor_parallel_size", 1)
        self._pp_size = kwargs.get("pipeline_parallel_size", 1)
        self._dp_size = kwargs.get("data_parallel_size", 1)
        self._is_lora = kwargs.get("enable_lora", False)

        # Let subclass create the appropriate engine
        self.llm = self._create_engine(*args, **kwargs)

        # Weight loader is created by subclass after engine initialization
        self._weight_loader = None

    def tp_size(self):
        return self._tp_size

    def pp_size(self):
        return self._pp_size

    def dp_size(self):
        return self._dp_size

    def _create_engine(self, *args, **kwargs):
        """Abstract method for subclasses to implement engine creation."""
        raise NotImplementedError("Subclasses must implement _create_engine")

    def _preprocess_prompts(self, input_batch: InferenceEngineInput):
        """Common prompt preprocessing logic."""
        prompts = input_batch.get("prompts")
        prompt_token_ids = input_batch.get("prompt_token_ids")
        request_sampling_params = input_batch.get("sampling_params")

        assert (
            prompts is None and prompt_token_ids is not None
        ), "VLLMInferenceEngine only accepts `prompt_token_ids`, not `prompts`."

        sampling_params = (
            SamplingParams(**request_sampling_params) if request_sampling_params is not None else SamplingParams()
        )

        return prompt_token_ids, sampling_params

    def _postprocess_outputs(self, outputs):
        """Common output processing logic."""
        responses: List[str] = []
        stop_reasons: List[str] = []
        response_ids: List[List[int]] = []
        response_logprobs: Optional[List[List[float]]] = []
        prompt_logprobs_list: List[Optional[List[float]]] = []
        rollout_expert_indices: Optional[List[List[List[List[int]]]]] = []

        for output in outputs:
            # TODO(tgriggs): Support n>1 sampling.
            assert (
                len(output.outputs) == 1
            ), "Each prompt should have only one responses. n>1 sampling is supported by copying prompts."
            resp = output.outputs[0]
            responses.append(resp.text)
            stop_reasons.append(resp.finish_reason)
            response_ids.append(resp.token_ids)
            _logprobs = None
            if resp.logprobs:
                _logprobs = []
                for i, token_logprobs in enumerate(resp.logprobs):
                    token_logprobs: Dict[str, Logprob]
                    token_id = resp.token_ids[i]
                    logprob = token_logprobs[token_id].logprob
                    _logprobs.append(logprob)
                    del token_logprobs
            response_logprobs.append(_logprobs)

            # Extract per-prompt-token logprobs (from RequestOutput, not CompletionOutput).
            # Returns logprob of each prompt token given prior context, skipping position 0
            # (which has no prior context). This matches the JAX backend which computes
            # logits_to_logprobs(all_logits[:, :-1, :], input_ids[:, 1:]) → length prompt_len - 1.
            _prompt_logprobs = None
            if output.prompt_logprobs is not None:
                _prompt_logprobs = []
                for i, pos_logprobs in enumerate(output.prompt_logprobs):
                    if pos_logprobs is None:
                        # First position has no prior context; skip it (matching JAX backend).
                        # Only first position can be None
                        continue
                    else:
                        token_id = output.prompt_token_ids[i]
                        if token_id not in pos_logprobs:
                            raise RuntimeError(
                                f"vLLM prompt_logprobs missing actual token at position {i} "
                                f"(token_id={token_id}). This violates vLLM's contract that "
                                f"the actual prompt token is always returned regardless of rank."
                            )
                        _prompt_logprobs.append(pos_logprobs[token_id].logprob)
            prompt_logprobs_list.append(_prompt_logprobs)

            _routed_experts = None
            if resp.routed_experts is not None:
                if hasattr(resp.routed_experts, "tolist"):
                    _routed_experts = resp.routed_experts.tolist()
                else:
                    _routed_experts = resp.routed_experts
            rollout_expert_indices.append(_routed_experts)

        if len(response_logprobs) and response_logprobs[0] is None:
            response_logprobs = None  # hack: assume uniform sampling params

        if len(prompt_logprobs_list) and prompt_logprobs_list[0] is None:
            prompt_logprobs_list = None  # hack: assume uniform sampling params

        if len(rollout_expert_indices) > 0 and rollout_expert_indices[0] is None:
            rollout_expert_indices = None  # hack: assume uniform sampling params

        return InferenceEngineOutput(
            responses=responses,
            stop_reasons=stop_reasons,
            response_ids=response_ids,
            response_logprobs=response_logprobs,
            prompt_logprobs=prompt_logprobs_list,
            rollout_expert_indices=rollout_expert_indices,
        )

    def _get_engine(self):
        """Get the underlying engine for RPC calls."""
        return self.llm.engine if hasattr(self.llm, "engine") else self.llm

    @staticmethod
    def _get_unfinished_request_ids(output_processor) -> list:
        """Get unfinished request IDs suitable for abort/abort_request calls.

        In vllm 0.16.0+, request_states is keyed by internal IDs (with a random suffix),
        while abort() expects external IDs by default. We use external_req_ids when
        available and fall back to request_states keys for older vllm versions.
        """
        if hasattr(output_processor, "external_req_ids"):
            return list(output_processor.external_req_ids.keys())
        return list(output_processor.request_states.keys())

    def reset_prefix_cache(self, reset_running_requests: bool = False):
        """Reset the prefix cache. Subclasses override for async version."""
        return self.llm.llm_engine.reset_prefix_cache(reset_running_requests=reset_running_requests)

    async def pause_generation(self, clear_cache: bool = False) -> None:
        raise NotImplementedError("pause_generation is only supported for AsyncVLLMInferenceEngine.")

    async def resume_generation(self) -> None:
        raise NotImplementedError("resume_generation is only supported for AsyncVLLMInferenceEngine.")


class VLLMInferenceEngine(BaseVLLMInferenceEngine):
    """Synchronous VLLM engine."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._weight_loader = VLLMWeightLoader(self.llm, is_async=False)

    def _create_engine(self, *args, **kwargs):
        # Pipeline parallelism requires AsyncLLMEngine
        if kwargs.get("pipeline_parallel_size", 1) > 1:
            raise ValueError(
                "Pipeline parallelism is only supported with AsyncVLLMInferenceEngine. "
                "Please set `generator.inference_engine.async_engine=true` in your config."
            )
        # Pop enable_ray_prometheus_stats - only supported for async engine
        enable_ray_prometheus_stats = kwargs.pop("enable_ray_prometheus_stats", False)
        if enable_ray_prometheus_stats:
            logger.warning(
                "enable_ray_prometheus_stats is only supported with AsyncVLLMInferenceEngine. "
                "Set `generator.inference_engine.async_engine=true` to enable Ray Prometheus stats logging."
            )
        # RULE 1 evidence for the engine mode. Until 2026-08-10 no arm printed which engine it ran,
        # so "this arm runs async_engine=false" could only be reconstructed from the launch line --
        # and the offline entrypoint's silent `disable_log_stats=True` (entrypoints/llm.py:235-236)
        # meant the engine emitted nothing else to identify itself either. `inproc_core` is the
        # load-bearing half: with VLLM_ENABLE_V1_MULTIPROCESSING=0 the sync engine gets
        # `InprocClient` (v1/engine/core_client.py:105), so EngineCore -- scheduler, update_from_output
        # -- runs on THIS thread, interleaved with detokenization. With it unset/1 the sync engine
        # gets `SyncMPClient` and a separate EngineCore process instead. One grep now tells an arm's
        # reader which of those it was looking at.
        if os.environ.get("SKYRL_ISOEXEC") == "1":
            print(
                f"[ISOEXEC-ENGINE-MODE] async_engine=false (vllm.LLM) in pid {os.getpid()}: "
                f"inproc_core={os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING') == '0'} "
                f"(VLLM_ENABLE_V1_MULTIPROCESSING={os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING')})",
                flush=True,
            )
        return vllm.LLM(*args, **kwargs)

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        prompt_token_ids, sampling_params = self._preprocess_prompts(input_batch)

        # Check if LoRA is enabled and create LoRA requests
        lora_requests = None
        if self._is_lora:
            lora_int_ids = list(self.llm.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                batch_size = len(prompt_token_ids)
                # dummy_lora_path for placeholder (actual loading done in add_lora())
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/dummy_lora_path")
                ] * batch_size

        outputs = await asyncio.to_thread(
            self.llm.generate,
            prompts=[TokensPrompt(prompt_token_ids=r) for r in prompt_token_ids],
            sampling_params=sampling_params,
            lora_request=lora_requests,
        )

        return self._postprocess_outputs(outputs)

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Only supported in AsyncVLLMInferenceEngine."""
        raise NotImplementedError("`chat_completion` is only supported in AsyncVLLMInferenceEngine.")

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Only supported in AsyncVLLMInferenceEngine."""
        raise NotImplementedError("`completion` is only supported in AsyncVLLMInferenceEngine.")

    async def wake_up(self, *args: Any, **kwargs: Any):
        await asyncio.to_thread(self.llm.wake_up, tags=kwargs.get("tags", None))

    async def sleep(self, *args: Any, **kwargs: Any):
        engine = self._get_engine().llm_engine
        output_processor = engine.output_processor
        if output_processor.has_unfinished_requests():
            logger.warning(
                "Calling sleep() with unfinished requests in vLLM engine. This is unexpected since all "
                "generation should be done before sleep() is called. Check for potential failures or "
                "dangling requests in your Generator/Env. Aborting all unfinished requests."
            )
            unfinished_request_ids = self._get_unfinished_request_ids(output_processor)
            await asyncio.to_thread(engine.abort_request, unfinished_request_ids)

        # SkyRL-IsoExec: level 1 IS required -- but for the NON-parameter state. Level 2 (discard)
        # zeroes pool-resident derived tensors that are neither named_parameters nor synced (e.g.
        # RoPE inv_freq), which broke generation catastrophically (step-2 DIFF ~1.0 nat, 96% of
        # tokens). Level 1 restores that state correctly; its flaw -- restoring the stale step-0
        # PARAMS over the fresh sync -- is fixed by isoexec_reapply_cached_weights, which the
        # dispatch calls after the final wake_up to overwrite the params with the synced bytes.
        import os as _os

        level = 1 if (self._is_lora or _os.environ.get("SKYRL_ISOEXEC") == "1") else kwargs.get("level", 2)
        # LIFECYCLE ASSERT (sleep tag-scoping): a kv_cache-scoped sleep must not also drop 'weights'
        # at level 1 (stale-theta0 restore). Observation-only, fail-soft. tags=None (full sleep) is
        # allowed -- that path is covered by isoexec_reapply_cached_weights after the final wake.
        from skyrl.backends.skyrl_train.isoexec.lifecycle import ordering as _ix_order

        _ix_order.check_sleep_tag_scoping(kwargs.get("tags", None), level)
        # Decompose the engine_sleep stage (GLM measured it at 9.09 s live vs 2.8 s for a bare
        # standalone llm.sleep -- the delta was never attributed). One INFO line per sleep.
        import time as _time

        _t0 = _time.perf_counter()
        await asyncio.to_thread(self.llm.sleep, level=level)
        _t_sleep = _time.perf_counter() - _t0
        # The training window starts here, and the trainer measures its non-activation floor
        # DEVICE-wide ((total-free)-reserved, dyn_recompute.measure_sample) -- anything this
        # worker merely CACHES is charged to the trainer's backward budget. Hand it back
        # (~0.4 GiB/worker measured; 2026-08-04 floor forensics).
        # NOTE (2026-08-09): CuMemAllocator.sleep() itself ends with gc.collect() +
        # torch.cuda.empty_cache() in every worker, so this RPC is expected to release ~0 GiB.
        # The per-worker GiB is now logged below: if it reads 0.00 across live steps, the RPC is
        # proven dead weight and SKYRL_ISOEXEC_ENGINE_EMPTY_CACHE=0 is ledger-safe by that same
        # evidence (the floor was already emptied by the sleep call itself).
        _t_rel = 0.0
        _released = None
        if _os.environ.get("SKYRL_ISOEXEC_ENGINE_EMPTY_CACHE", "1") == "1":
            try:
                _t0 = _time.perf_counter()
                _released = await asyncio.to_thread(
                    self._get_engine().llm_engine.collective_rpc, "isoexec_release_cached_blocks"
                )
                _t_rel = _time.perf_counter() - _t0
            except Exception as e:
                logger.info(f"[isoexec] release_cached_blocks failed (non-fatal): {e}")
        logger.info(
            f"[isoexec] engine_sleep decomposition: llm.sleep={_t_sleep:.2f}s "
            f"release_rpc={_t_rel:.2f}s released_gib="
            f"{[f'{g:.3f}' for g in _released] if _released is not None else 'skipped'}"
        )

    async def isoexec_reapply_cached_weights(self):
        """SkyRL-IsoExec: re-apply the last synced weights after the final wake_up (which clobbers
        them on the nightly stack). See WorkerWrap.isoexec_reapply_cached_weights."""
        engine = self._get_engine()
        return await asyncio.to_thread(engine.collective_rpc, "isoexec_reapply_cached_weights")

    async def init_weight_update_communicator(self, init_info: "WeightSyncInitInfo"):
        import pickle

        # The CPU engine actor is a controller, not a manifest/SASS participant.  The concrete
        # WorkerWrap method receiving this payload performs the trainer/engine manifest handshake
        # inside every GPU worker before it constructs the weight receiver.

        engine = self._get_engine()
        # Pickle the init_info to preserve type through collective_rpc
        pickled_init_info = pickle.dumps(init_info)
        return await asyncio.to_thread(
            engine.collective_rpc,
            "init_weight_update_communicator",
            args=(pickled_init_info,),
        )

    async def _load_lora_from_disk(self, lora_path: str, lora_name: str = ""):
        """Load LoRA adapters from disk using vLLM's native add_lora method.

        When ``lora_name`` is empty (legacy single-tenant), a numeric name is
        generated. Multi-tenant callers pass ``lora_name`` so subsequent
        ``model=<lora_name>`` sampling routes to the right adapter.
        """
        lora_id = int(time.time_ns() % 0x7FFFFFFF)
        name = lora_name or f"{lora_id}"
        lora_request = LoRARequest(lora_name=name, lora_int_id=lora_id, lora_path=lora_path)
        result = self.llm.llm_engine.add_lora(lora_request)
        return result

    async def update_named_weights(self, request: WeightUpdateRequest):
        from skyrl.backends.skyrl_train.weight_sync import LoraLoadRequest

        # Handle LoRA disk loading request
        if isinstance(request, LoraLoadRequest):
            return await self._load_lora_from_disk(request.lora_path, lora_name=request.lora_name)

        if not len(request):
            raise ValueError("Weight update request must not be empty")

        # Use the weight loader to coordinate weight transfer
        return await self._weight_loader.load_weights(request)

    async def teardown(self):
        await self._teardown_weight_receiver()

    async def reset_prefix_cache(self, reset_running_requests: bool = False):
        return await asyncio.to_thread(
            self.llm.llm_engine.reset_prefix_cache, reset_running_requests=reset_running_requests
        )

    async def _teardown_weight_receiver(self):
        engine = self._get_engine()
        return await asyncio.to_thread(engine.collective_rpc, "teardown_weight_receiver")

    async def start_weight_update(self, is_checkpoint_format: bool = True):
        engine = self._get_engine()
        return await asyncio.to_thread(
            engine.collective_rpc,
            "skyrl_start_weight_update",
            args=(is_checkpoint_format,),
        )

    async def finish_weight_update(self):
        engine = self._get_engine()
        return await asyncio.to_thread(engine.collective_rpc, "skyrl_finish_weight_update")


def _assert_isoexec_async_engine_supported() -> None:
    if os.environ.get("SKYRL_ISOEXEC") != "1":
        return

    print(
        f"[ISOEXEC-ENGINE-MODE] async_engine=true (vllm.AsyncLLM) in pid {os.getpid()}: EngineCore "
        f"runs in its own spawned process; scheduler + output processing are no longer serialized "
        f"on this thread. final_only={os.environ.get('SKYRL_VLLM_ASYNC_FINAL_ONLY', '1')}",
        flush=True,
    )


class AsyncVLLMInferenceEngine(BaseVLLMInferenceEngine):
    """Asynchronous VLLM engine."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._weight_loader = VLLMWeightLoader(self.llm, is_async=True)

    def _create_engine(self, *args, **kwargs):
        openai_kwargs = pop_openai_kwargs(kwargs)

        _assert_isoexec_async_engine_supported()

        # Logging kwargs
        enable_ray_prometheus_stats = kwargs.pop("enable_ray_prometheus_stats", False)
        enable_log_requests = kwargs.pop("enable_log_requests", False)
        max_log_len = kwargs.pop("max_log_len", None)

        engine_args = vllm.AsyncEngineArgs(enable_log_requests=enable_log_requests, kv_cache_metrics=True, **kwargs)

        stat_loggers = None
        if enable_ray_prometheus_stats:
            stat_loggers = self._create_ray_prometheus_stat_loggers()

        engine = vllm.AsyncLLMEngine.from_engine_args(engine_args, stat_loggers=stat_loggers)

        model_path = kwargs.get("model")
        # Use served_model_name if provided (from generator.inference_engine.served_model_name config),
        # otherwise fall back to model_path. This allows using a different model name
        # in HTTP endpoint requests than the actual model path.
        # See: https://github.com/NovaSky-AI/SkyRL/pull/238#discussion_r2326561295
        served_model_name = kwargs.get("served_model_name", None)
        model_name = served_model_name if served_model_name is not None else model_path

        base_model_paths = [BaseModelPath(name=model_name, model_path=model_path)]
        models = OpenAIServingModels(engine, base_model_paths)

        # Build request logger for debugging (off by default).
        # Enable via: generator.inference_engine.engine_init_kwargs.enable_log_requests=true
        # Optionally limit logged chars: generator.inference_engine.engine_init_kwargs.max_log_len=256
        request_logger = None
        if enable_log_requests:
            from vllm.entrypoints.logger import RequestLogger

            request_logger = RequestLogger(max_log_len=max_log_len)

        chat_template = openai_kwargs.pop("chat_template", None)

        from vllm.renderers import renderer_from_config

        model_registry = OpenAIModelRegistry(
            model_config=engine.model_config,
            base_model_paths=base_model_paths,
        )
        renderer = renderer_from_config(engine.vllm_config)
        openai_serving_render = OpenAIServingRender(
            model_config=engine.model_config,
            renderer=renderer,
            model_registry=model_registry,
            request_logger=request_logger,
            chat_template=chat_template,
            chat_template_content_format="auto",
        )

        self.openai_serving_chat = OpenAIServingChat(
            engine_client=engine,
            models=models,
            response_role="assistant",
            openai_serving_render=openai_serving_render,
            request_logger=request_logger,
            chat_template=chat_template,
            chat_template_content_format="auto",
            **openai_kwargs,
        )

        # TODO(Charlie): revisit kwargs `return_tokens_as_token_ids`,
        # `enable_prompt_tokens_details`, `enable_force_include_usage`.
        self.openai_serving_completion = OpenAIServingCompletion(
            engine_client=engine,
            models=models,
            openai_serving_render=openai_serving_render,
            request_logger=request_logger,
        )
        return engine

    def _create_ray_prometheus_stat_loggers(self):
        """Create Ray Prometheus stat loggers for vLLM metrics."""
        try:
            from vllm.v1.metrics.ray_wrappers import RayPrometheusStatLogger

            logger.info("Enabling RayPrometheusStatLogger for vLLM inference engine metrics")
            return [RayPrometheusStatLogger]
        except ImportError:
            logger.warning(
                "RayPrometheusStatLogger not available in this vLLM version. "
                "For Ray-integrated metrics, upgrade to vLLM >= 0.9.0. "
                "Stat logging will be disabled."
            )
            return None

    async def _load_lora_from_disk(self, lora_path: str, lora_name: str = ""):
        """Load LoRA adapters from disk using vLLM's native add_lora method.

        When ``lora_name`` is empty (legacy single-tenant), a numeric name is
        generated. Multi-tenant callers pass ``lora_name`` so subsequent
        ``model=<lora_name>`` sampling routes to the right adapter.
        """
        lora_id = int(time.time_ns() % 0x7FFFFFFF)
        name = lora_name or f"{lora_id}"
        lora_request = LoRARequest(lora_name=name, lora_int_id=lora_id, lora_path=lora_path)
        result = await self.llm.add_lora(lora_request)
        return result

    async def _collect_outputs(self, prompt_token_ids, request_id: str, sampling_params: SamplingParams):
        """Collect outputs for a single prompt."""
        # Check if LoRA is enabled and create LoRA request
        final_output = None
        lora_request = None

        if self._is_lora:
            lora_int_ids = list(await self.llm.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                # dummy_lora_path for placeholder (actual loading done in add_lora())
                lora_request = LoRARequest(
                    lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/dummy_lora_path"
                )

        async for request_output in self.llm.generate(
            prompt=TokensPrompt(prompt_token_ids=prompt_token_ids),
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=lora_request,
        ):
            final_output = request_output

        return final_output

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        """Generate responses using vLLM's async engine."""
        prompt_token_ids, sampling_params = self._preprocess_prompts(input_batch)

        # FINAL_ONLY: the one place the async engine was strictly WORSE than the offline one.
        #
        # ``SamplingParams.output_kind`` defaults to CUMULATIVE (vllm/sampling_params.py:301). Under
        # CUMULATIVE the output processor builds a fresh ``RequestOutput`` -- prompt ids, every token
        # id so far, the detokenized text so far, the logprob list -- for EVERY in-flight request on
        # EVERY engine step, and sets a per-request asyncio Event to wake its collector task
        # (v1/engine/output_processor.py:62-66). At the concurrency this stack runs (max_num_seqs
        # 512 per engine) that is 512 object graphs + 512 event/task wakeups per step, on the one
        # frontend thread, and it grows with the response length because CUMULATIVE re-emits the
        # whole sequence each time. vLLM's own offline entrypoint refuses to pay it: it forces
        # FINAL_ONLY at entrypoints/offline_utils.py:559-561 with the comment "We only care about
        # the final output", after which make_request_output returns None on every non-final step
        # (v1/engine/output_processor.py:282-285).
        #
        # ``_collect_outputs`` below does ``final_output = request_output`` in a loop -- it KEEPS
        # ONLY THE LAST ONE. So every intermediate object built under CUMULATIVE is constructed,
        # queued, woken on, and thrown away. FINAL_ONLY yields exactly the same final object (same
        # token ids, same text, same logprobs; the enum only controls whether INTERMEDIATE outputs
        # are emitted), so this is equivalence-preserving by construction and cannot move a token.
        #
        # Escape hatch for anyone who needs streaming semantics out of this method:
        # SKYRL_VLLM_ASYNC_FINAL_ONLY=0 restores vLLM's CUMULATIVE default.
        if os.environ.get("SKYRL_VLLM_ASYNC_FINAL_ONLY", "1") == "1":
            from vllm.sampling_params import RequestOutputKind

            sampling_params.output_kind = RequestOutputKind.FINAL_ONLY

        tasks = []
        for prompt in prompt_token_ids:
            # Schedule the collection of outputs for each prompt.
            # Avoid duplicate request_ids
            request_id = str(uuid4().hex)
            task = asyncio.create_task(self._collect_outputs(prompt, request_id, sampling_params))
            tasks.append(task)
        outputs = await asyncio.gather(*tasks)

        return self._postprocess_outputs(outputs)

    async def wake_up(self, *args: Any, **kwargs: Any):
        await self.llm.wake_up(tags=kwargs.get("tags", None))

    async def sleep(self, *args: Any, **kwargs: Any):
        engine = self._get_engine()
        output_processor = engine.output_processor
        # make sure that the engine is alive
        engine.engine_core.ensure_alive()
        if output_processor.has_unfinished_requests():
            logger.warning(
                "Calling sleep() with unfinished requests in vLLM engine. This is unexpected since all "
                "generation should be done before sleep() is called. Check for potential failures or "
                "dangling requests in your Generator/Env. Aborting all unfinished requests."
            )
            unfinished_request_ids = self._get_unfinished_request_ids(output_processor)
            await engine.abort(unfinished_request_ids)

        # TODO(team): remove once vllm fixes this
        # otherwise waking it up will output gibberish: https://github.com/vllm-project/vllm/issues/17103
        await self.reset_prefix_cache()
        # SkyRL-IsoExec: level 1 IS required -- but for the NON-parameter state. Level 2 (discard)
        # zeroes pool-resident derived tensors that are neither named_parameters nor synced (e.g.
        # RoPE inv_freq), which broke generation catastrophically (step-2 DIFF ~1.0 nat, 96% of
        # tokens). Level 1 restores that state correctly; its flaw -- restoring the stale step-0
        # PARAMS over the fresh sync -- is fixed by isoexec_reapply_cached_weights, which the
        # dispatch calls after the final wake_up to overwrite the params with the synced bytes.
        import os as _os

        level = 1 if (self._is_lora or _os.environ.get("SKYRL_ISOEXEC") == "1") else kwargs.get("level", 2)
        # LIFECYCLE ASSERT (sleep tag-scoping): a kv_cache-scoped sleep must not also drop 'weights'
        # at level 1 (stale-theta0 restore). Observation-only, fail-soft. tags=None (full sleep) is
        # allowed -- that path is covered by isoexec_reapply_cached_weights after the final wake.
        from skyrl.backends.skyrl_train.isoexec.lifecycle import ordering as _ix_order

        _ix_order.check_sleep_tag_scoping(kwargs.get("tags", None), level)
        import time as _time

        _t0 = _time.perf_counter()
        await self.llm.sleep(level=level)
        _t_sleep = _time.perf_counter() - _t0
        # See the sync twin above: hand the caching allocator's reserved-but-free segments back
        # so the trainer's device-wide floor measurement is not charged for engine cache.
        # (Expected ~0 GiB: CuMemAllocator.sleep already empty_caches -- see the sync twin.)
        _t_rel = 0.0
        _released = None
        if _os.environ.get("SKYRL_ISOEXEC_ENGINE_EMPTY_CACHE", "1") == "1":
            try:
                _t0 = _time.perf_counter()
                _released = await self._get_engine().collective_rpc("isoexec_release_cached_blocks")
                _t_rel = _time.perf_counter() - _t0
            except Exception as e:
                logger.info(f"[isoexec] release_cached_blocks failed (non-fatal): {e}")
        logger.info(
            f"[isoexec] engine_sleep decomposition: llm.sleep={_t_sleep:.2f}s "
            f"release_rpc={_t_rel:.2f}s released_gib="
            f"{[f'{g:.3f}' for g in _released] if _released is not None else 'skipped'}"
        )

    async def isoexec_reapply_cached_weights(self):
        """SkyRL-IsoExec: re-apply the last synced weights after the final wake_up (which clobbers
        them on the nightly stack). See WorkerWrap.isoexec_reapply_cached_weights."""
        engine = self._get_engine()
        return await engine.collective_rpc("isoexec_reapply_cached_weights")

    async def init_weight_update_communicator(self, init_info: "WeightSyncInitInfo"):
        import pickle

        # Manifest agreement is checked by every concrete WorkerWrap below; this process only
        # coordinates the RPC and deliberately owns no CUDA/SASS participant identity.

        engine = self._get_engine()
        # Pickle the init_info to preserve type through collective_rpc
        pickled_init_info = pickle.dumps(init_info)
        return await engine.collective_rpc(
            "init_weight_update_communicator",
            args=(pickled_init_info,),
        )

    async def update_named_weights(self, request: WeightUpdateRequest):
        from skyrl.backends.skyrl_train.weight_sync import LoraLoadRequest

        # Check for LoRA disk loading request
        if isinstance(request, LoraLoadRequest):
            return await self._load_lora_from_disk(request.lora_path, lora_name=request.lora_name)

        if not len(request):
            raise ValueError("Weight update request must not be empty")

        # Use the weight loader to coordinate weight transfer
        return await self._weight_loader.load_weights(request)

    async def teardown(self):
        await self._teardown_weight_receiver()

    async def reset_prefix_cache(self, reset_running_requests: bool = False):
        engine = self._get_engine()
        await engine.reset_prefix_cache(reset_running_requests=reset_running_requests)

    async def _teardown_weight_receiver(self):
        engine = self._get_engine()
        return await engine.collective_rpc("teardown_weight_receiver")

    async def start_weight_update(self, is_checkpoint_format: bool = True):
        engine = self._get_engine()
        return await engine.collective_rpc(
            "skyrl_start_weight_update",
            args=(is_checkpoint_format,),
        )

    async def finish_weight_update(self):
        engine = self._get_engine()
        return await engine.collective_rpc("skyrl_finish_weight_update")

    # ----------------------------------------
    # Methods for handling OpenAI API requests
    # ----------------------------------------

    async def _handle_openai_request(self, request_payload: Dict[str, Any], endpoint: str) -> Dict[str, Any]:
        """Handle OpenAI API request."""
        assert endpoint in ["/chat/completions", "/completions"]

        body = request_payload.get("json", {})
        headers = request_payload.get("headers", {})

        # 1. Build request
        try:
            if endpoint == "/chat/completions":
                request = ChatCompletionRequest(**body)
            else:
                request = CompletionRequest(**body)
            assert request.stream is False, "Streaming is not supported in SkyRL yet, please set stream to False."
        except Exception as e:
            return ErrorResponse(
                error=ErrorInfo(
                    message=str(e),
                    type=HTTPStatus.BAD_REQUEST.phrase,
                    code=HTTPStatus.BAD_REQUEST.value,
                ),
            ).model_dump()

        # 2. Call vllm engine
        try:
            # Create a minimal request-like object with attributes used by vLLM
            minimal_request = _MinimalRequest(headers)
            if endpoint == "/chat/completions":
                generator = await self.openai_serving_chat.create_chat_completion(request, minimal_request)
                assert isinstance(generator, (ChatCompletionResponse, ErrorResponse))
            else:
                generator = await self.openai_serving_completion.create_completion(request, minimal_request)
                assert isinstance(generator, (CompletionResponse, ErrorResponse))
            return generator.model_dump()

        except Exception as e:
            # Handle it here so we can surface the error from a ray worker.

            # Determine appropriate HTTP status code based on error message to mimic vllm serve error
            # handling. Here, we handle context length errors, which should return 400 according to
            # vllm serve error handling, so that downstream users can handle these properly rather
            # than seeing a 500 SkyRL INTERNAL_SERVER_ERROR. For instance, LiteLLM can wraps them as
            # BadRequestError, enabling Harbor to detect ContextLengthExceededError.
            # NOTE(Charlie): This is hacky. With the refactored inference stack, we
            # should be able to directly reuse the error handling from the served vllm.
            error_message = str(e).lower()
            is_context_length_error = "context length" in error_message or "maximum input length" in error_message

            if is_context_length_error:
                http_status = HTTPStatus.BAD_REQUEST
            else:
                http_status = HTTPStatus.INTERNAL_SERVER_ERROR

            return ErrorResponse(
                error=ErrorInfo(
                    message=str(e),
                    type=http_status.phrase,
                    code=http_status.value,
                ),
            ).model_dump()

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI-compatible HTTP endpoint for handling `/chat/completions` in Python vLLM engine.

        Accepts a JSON-serializable payload: {"json": <request-body>, "headers": <headers-dict>}.
        Constructs a minimal request-like object for vLLM's openai_serving_chat.
        Returns a plain dict, either a ChatCompletionResponse or an ErrorResponse, both defined
        in vllm.entrypoints.openai.protocol.
        """
        return await self._handle_openai_request(request_payload, endpoint="/chat/completions")

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI-compatible HTTP endpoint for handling `/completions` in Python vLLM engine.

        Accepts a JSON-serializable payload: {"json": <request-body>, "headers": <headers-dict>}.
        Constructs a minimal request-like object for vLLM's openai_serving_completion.
        Returns a plain dict, either a CompletionResponse or an ErrorResponse, both defined
        in vllm.entrypoints.openai.protocol.
        """
        return await self._handle_openai_request(request_payload, endpoint="/completions")

    async def pause_generation(self, clear_cache: bool = False) -> None:
        """Pause generation using vLLM's native keep mode, freezing in-flight requests."""
        engine = self._get_engine()
        await engine.pause_generation(mode="keep", clear_cache=clear_cache)
        logger.info("pause_generation(mode='keep') finished")

    async def resume_generation(self) -> None:
        """Resume generation after a keep-mode pause."""
        engine = self._get_engine()
        await engine.resume_generation()
        logger.info("resume_generation() finished")


class _MinimalRequest:
    """
    Minimal request-like object for vLLM's openai_serving_chat and openai_serving_completion.

    We cannot use the original user Request object because it cannot be serialized and hence
    cannot be a ray method argument. Instead we take the original request's headers and
    reconstruct an instance of _MinimalRequest to mimic the FastAPI Request object.

    The fields depend on what vLLM accesses internally.
    """

    def __init__(self, headers):
        self.headers = headers  # Expect a mapping with .get support
        self.state = SimpleNamespace()  # vLLM sets raw_request.state.request_metadata


class VLLMWeightLoader(WeightLoader):
    """Loads weights into vLLM engine, managing RPC coordination.

    This loader encapsulates the collective_rpc calls to workers.
    Workers create the appropriate receiver locally for the actual weight transfer.
    """

    def __init__(self, engine: Any, is_async: bool = False) -> None:
        """Initialize the loader.

        Args:
            engine: The vLLM engine (LLM or AsyncLLMEngine).
            is_async: Whether this is for AsyncVLLMInferenceEngine.
        """
        self._engine = engine.engine if hasattr(engine, "engine") else engine
        self._is_async = is_async

    async def load_weights(self, request: WeightUpdateRequest) -> None:
        """Load weights by coordinating RPC to workers.

        Sends the request to workers via collective_rpc. Workers create
        the receiver locally and use it to receive and load weights.

        Args:
            request: Weight update request.
        """
        import pickle

        # Pickle the request to preserve type through collective_rpc
        pickled_request = pickle.dumps(request)

        if self._is_async:
            await self._engine.collective_rpc(
                "load_weights",
                args=(pickled_request,),
            )
        else:
            await asyncio.to_thread(
                self._engine.collective_rpc,
                "load_weights",
                args=(pickled_request,),
            )


VLLMRayActor = ray.remote(VLLMInferenceEngine)
AsyncVLLMRayActor = ray.remote(AsyncVLLMInferenceEngine)
