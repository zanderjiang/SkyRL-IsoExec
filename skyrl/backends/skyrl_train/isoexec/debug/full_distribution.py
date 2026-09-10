"""Compare full fp32 raw-logprob rows by token-history identity in debug traces.

Only row fingerprints leave each process. Unsupported execution modes fail closed.
"""

from __future__ import annotations

import functools
import hashlib
import os
import struct
from contextvars import ContextVar
from typing import Any, Optional, Sequence

import torch

from . import trace

ENV = "SKYRL_ISOEXEC_DEBUG_FULL_DISTRIBUTION"
REGION = "logprobs.full_raw_distribution"

_HISTORY_DOMAIN = b"skyrl-isoexec/full-raw-logprobs/history/v1\0"
_SECOND_DIGEST_SEED = 0x9E3779B97F4A7C15
_TRAINER_ROWS_PER_CHUNK = 4
_ENGINE_HISTORY_KEYS: ContextVar[Optional[tuple[Optional[str], ...]]] = ContextVar(
    "isoexec_full_distribution_history_keys", default=None
)
_ENGINE_GRAMMAR_ACTIVE: ContextVar[bool] = ContextVar("isoexec_full_distribution_grammar", default=False)
_ENGINE_HASH_CACHE: dict[tuple[int, str], tuple[object, int, Any]] = {}
_ACTIVE_WEIGHT_VERSION: Optional[int] = None
_LAST_WEIGHT_VERSION: Optional[int] = None


def enabled() -> bool:
    return os.environ.get(ENV, "0") == "1"


def clear_weight_version() -> None:
    """Make capture fail closed while a new weight-sync transaction is incomplete."""
    global _ACTIVE_WEIGHT_VERSION
    _ACTIVE_WEIGHT_VERSION = None


def set_weight_version(weight_version: int) -> None:
    """Activate the sender-minted weight-sync transaction ID in this process."""
    global _ACTIVE_WEIGHT_VERSION, _LAST_WEIGHT_VERSION
    if not enabled():
        raise RuntimeError(f"{ENV}=1 is required to set the full-distribution weight version")
    if not trace.enabled():
        raise RuntimeError(f"{ENV}=1 requires {trace.ENV_TRACE} to be set")
    if type(weight_version) is not int or weight_version < 0:
        raise RuntimeError(f"{REGION} weight version must be a non-negative int, got {weight_version!r}")
    if trace.get_tracer() is None:
        raise RuntimeError(f"{REGION} could not initialize the debug tracer")
    if _LAST_WEIGHT_VERSION is not None and weight_version != _LAST_WEIGHT_VERSION + 1:
        raise RuntimeError(
            f"{REGION} weight version must advance exactly once per completed sync, "
            f"got previous={_LAST_WEIGHT_VERSION} next={weight_version}"
        )
    _LAST_WEIGHT_VERSION = weight_version
    _ACTIVE_WEIGHT_VERSION = weight_version


def history_key(token_ids: Sequence[int]) -> str:
    """Return an unambiguous, content-only identity for one next-token history."""
    digest = hashlib.sha256()
    digest.update(_HISTORY_DOMAIN)
    for token_id in token_ids:
        _update_history(digest, token_id)
    return digest.hexdigest()


def _update_history(digest, token_id: int) -> None:
    value = int(token_id)
    if value < 0 or value > 0xFFFFFFFF:
        raise ValueError(f"token id {value} is outside canonical uint32 range")
    digest.update(struct.pack("<I", value))


def _record_rows(logprobs: torch.Tensor, keys: Sequence[Optional[str]], *, case: str) -> int:
    if logprobs.ndim != 2 or logprobs.shape[0] != len(keys):
        raise RuntimeError(
            f"{REGION} requires one [V] row per history key, got tensor={tuple(logprobs.shape)} keys={len(keys)}"
        )
    if logprobs.dtype is not torch.float32:
        raise RuntimeError(f"{REGION} requires fp32 logprobs, got {logprobs.dtype}")
    if _ACTIVE_WEIGHT_VERSION is None:
        raise RuntimeError(f"{REGION} has no active completed weight-sync transaction")
    rows = [
        (f"weight:{_ACTIVE_WEIGHT_VERSION}/history:{key}", logprobs[i]) for i, key in enumerate(keys) if key is not None
    ]
    return trace.record_named_tensors(
        REGION,
        rows,
        case=case,
        extra_seeds=(_SECOND_DIGEST_SEED,),
    )


def trainer_action_rows(
    logits: torch.Tensor,
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    num_actions: int,
) -> tuple[torch.Tensor, list[int], list[str]]:
    """Return the response-logit view plus active indices and history identities."""
    if logits.ndim != 3:
        raise RuntimeError(f"{REGION} trainer logits must be [B,S,V], got {tuple(logits.shape)}")
    batch, seq_len, _vocab = logits.shape
    expected_sequence_shape = (batch, seq_len)
    if tuple(sequences.shape) != expected_sequence_shape or tuple(attention_mask.shape) != expected_sequence_shape:
        raise RuntimeError(
            f"{REGION} sequence/mask shape must match logits [B,S]: "
            f"logits={tuple(logits.shape)} sequences={tuple(sequences.shape)} attention={tuple(attention_mask.shape)}"
        )
    if not isinstance(num_actions, int) or num_actions <= 0 or num_actions >= seq_len:
        raise RuntimeError(f"{REGION} requires 0 < num_actions < S, got num_actions={num_actions}, S={seq_len}")
    if tuple(loss_mask.shape) != (batch, num_actions):
        raise RuntimeError(f"{REGION} loss_mask must be [B,R]={(batch, num_actions)}, got {tuple(loss_mask.shape)}")

    response_start = seq_len - num_actions
    action_logits = logits[:, response_start - 1 : seq_len - 1, :]
    active = loss_mask.to(device=action_logits.device) > 0
    target_is_present = attention_mask[:, response_start:].to(device=active.device, dtype=torch.bool)
    if bool((active & ~target_is_present).any().item()):
        raise RuntimeError(f"{REGION} loss_mask marks a padded response token active")

    sequence_rows = sequences.detach().to(device="cpu")
    attention_rows = attention_mask.detach().to(device="cpu", dtype=torch.bool)
    active_rows = active.detach().to(device="cpu")
    active_indices: list[int] = []
    active_keys: list[str] = []
    for batch_index in range(batch):
        digest = hashlib.sha256()
        digest.update(_HISTORY_DOMAIN)
        for position in range(seq_len):
            present = bool(attention_rows[batch_index, position])
            if present and position >= response_start:
                response_offset = position - response_start
                key = digest.copy().hexdigest()
                if bool(active_rows[batch_index, response_offset]):
                    active_indices.append(batch_index * num_actions + response_offset)
                    active_keys.append(key)
            if present:
                _update_history(digest, sequence_rows[batch_index, position])
    return action_logits, active_indices, active_keys


def _full_logprobs_for_trace(logits: torch.Tensor) -> torch.Tensor:
    from ..ops.logprobs.rowinv import diagnostic_full_logprobs

    return diagnostic_full_logprobs(logits).detach()


def record_trainer_action_distributions(
    *,
    logits: torch.Tensor,
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    num_actions: int,
    temperature: float,
    packed: bool,
    tp_size: int,
    pp_size: int,
    cp_size: int,
    dp_size: int,
    rowinv_before: Optional[dict],
) -> int:
    """Record trainer scoring rows after enforcing the MVP semantic domain."""
    if not enabled():
        return 0
    if not trace.enabled():
        raise RuntimeError(f"{ENV}=1 requires {trace.ENV_TRACE} to be set")
    if os.environ.get("SKYRL_ISOEXEC") != "1":
        raise RuntimeError(f"{ENV}=1 requires SKYRL_ISOEXEC=1")
    if temperature != 1.0:
        raise RuntimeError(f"{REGION} requires temperature=1.0, got {temperature}")
    if packed:
        raise RuntimeError(f"{REGION} does not yet support packed trainer sequences")
    sizes = (tp_size, pp_size, cp_size, dp_size)
    if sizes != (1, 1, 1, 1):
        raise RuntimeError(f"{REGION} currently requires TP=PP=CP=DP=1, got {sizes}")
    if loss_mask is None:
        raise RuntimeError(f"{REGION} requires loss_mask to identify active action rows")
    if rowinv_before is None:
        raise RuntimeError(f"{REGION} requires a pre-forward rowinv census")

    from ..ops.logprobs.rowinv import stats as rowinv_stats

    rowinv_after = rowinv_stats()
    if (
        rowinv_after["served"] <= rowinv_before["served"]
        or rowinv_after["declined"] != rowinv_before["declined"]
        or rowinv_after["agreed"] is False
    ):
        raise RuntimeError(f"{REGION} requires the real trainer sampled-logprob path to be served by rowinv")

    action_logits, active_indices, active_keys = trainer_action_rows(
        logits, sequences, attention_mask, loss_mask, num_actions
    )
    recorded = 0
    for start in range(0, len(active_keys), _TRAINER_ROWS_PER_CHUNK):
        stop = min(start + _TRAINER_ROWS_PER_CHUNK, len(active_keys))
        indices = torch.tensor(active_indices[start:stop], dtype=torch.int64, device=action_logits.device)
        batch_indices = torch.div(indices, num_actions, rounding_mode="floor")
        response_indices = torch.remainder(indices, num_actions)
        chunk_logits = action_logits[batch_indices, response_indices].contiguous()
        full_logprobs = _full_logprobs_for_trace(chunk_logits)
        recorded += _record_rows(full_logprobs, active_keys[start:stop], case="trainer_score")
        del full_logprobs, chunk_logits, batch_indices, response_indices, indices
    return recorded


def _engine_parallel_sizes(runner) -> tuple[int, int, int]:
    config = getattr(getattr(runner, "vllm_config", None), "parallel_config", None)
    if config is None:
        raise RuntimeError(f"{REGION} cannot inspect vLLM parallel_config")
    return tuple(
        int(getattr(config, name, 1))
        for name in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size")
    )


def engine_history_keys(
    runner,
    logits: Optional[torch.Tensor],
    spec_decode_metadata,
    *,
    grammar_active: bool,
) -> tuple[Optional[str], ...]:
    """Snapshot logical row identities before vLLM mutates its persistent input batch."""
    if logits is None or logits.ndim != 2:
        raise RuntimeError(
            f"{REGION} engine logits must be [N,V], got {None if logits is None else tuple(logits.shape)}"
        )
    if _engine_parallel_sizes(runner) != (1, 1, 1):
        raise RuntimeError(f"{REGION} currently requires vLLM TP=PP=DP=1")
    if bool(getattr(runner, "use_async_scheduling", False)):
        raise RuntimeError(f"{REGION} does not support vLLM async scheduling")
    if spec_decode_metadata is not None:
        raise RuntimeError(f"{REGION} does not support speculative decoding")
    if grammar_active:
        raise RuntimeError(f"{REGION} cannot compare logits after an engine-only grammar mask")
    sampler_mode = getattr(getattr(runner, "sampler", None), "logprobs_mode", None)
    if sampler_mode != "raw_logprobs":
        raise RuntimeError(f"{REGION} requires live vLLM logprobs_mode='raw_logprobs', got {sampler_mode!r}")

    input_batch = runner.input_batch
    num_reqs = int(input_batch.num_reqs)
    if logits.shape[0] != num_reqs:
        raise RuntimeError(f"{REGION} expected one sampler row per request, got rows={logits.shape[0]} reqs={num_reqs}")
    temperatures = input_batch.temperature_cpu[:num_reqs]
    if any(float(value) != 1.0 for value in temperatures):
        raise RuntimeError(f"{REGION} requires every live request temperature to be 1.0")
    sampling_metadata = input_batch.sampling_metadata
    if sampling_metadata.max_num_logprobs is None and not sampling_metadata.logprob_token_ids:
        raise RuntimeError(f"{REGION} requires rollout logprobs to be enabled")
    if bool((input_batch.request_lora_mapping[:num_reqs] != 0).any()):
        raise RuntimeError(f"{REGION} does not yet include LoRA identity")

    discard = runner.discard_request_mask.np[:num_reqs]
    request_ids = input_batch.req_ids[:num_reqs]
    live_cache_keys = {(id(runner), request_id) for request_id in request_ids}
    for cache_key in [key for key in _ENGINE_HASH_CACHE if key[0] == id(runner) and key not in live_cache_keys]:
        _ENGINE_HASH_CACHE.pop(cache_key, None)
    keys: list[Optional[str]] = []
    for row in range(num_reqs):
        if bool(discard[row]):
            keys.append(None)
            continue
        length = int(input_batch.num_tokens_no_spec[row])
        if not bool(input_batch.is_token_ids[row, :length].all()):
            raise RuntimeError(f"{REGION} currently supports text token inputs only")
        request = getattr(runner, "requests", {}).get(request_ids[row])
        if request is None or getattr(request, "mm_features", None):
            raise RuntimeError(f"{REGION} currently supports text-only requests")
        cache_key = (id(runner), request_ids[row])
        cached = _ENGINE_HASH_CACHE.get(cache_key)
        if cached is None or cached[0] is not request or cached[1] > length:
            cached_length = 0
            digest = hashlib.sha256()
            digest.update(_HISTORY_DOMAIN)
        else:
            _cached_request, cached_length, cached_digest = cached
            digest = cached_digest.copy()
        for token_id in input_batch.token_ids_cpu[row, cached_length:length]:
            _update_history(digest, token_id)
        _ENGINE_HASH_CACHE[cache_key] = (request, length, digest.copy())
        keys.append(digest.hexdigest())
    return tuple(keys)


def _wrap_engine_classes(model_runner_cls, sampler_cls) -> int:
    """Install coordinated vLLM hooks after resolving all bindings."""
    current_sample_tokens = getattr(model_runner_cls, "sample_tokens", None)
    current_sample = getattr(model_runner_cls, "_sample", None)
    current_stream_update = getattr(model_runner_cls, "_update_streaming_request", None)
    descriptor = sampler_cls.__dict__.get("compute_logprobs")
    current_compute = descriptor.__func__ if isinstance(descriptor, staticmethod) else descriptor
    if not all(callable(fn) for fn in (current_sample_tokens, current_sample, current_stream_update, current_compute)):
        raise RuntimeError(f"{REGION} could not resolve the pinned vLLM sampling hooks")

    markers = (
        getattr(current_sample_tokens, "_isoexec_full_distribution_grammar", False),
        getattr(current_sample, "_isoexec_full_distribution_context", False),
        getattr(current_stream_update, "_isoexec_full_distribution_stream_reset", False),
        getattr(current_compute, "_isoexec_full_distribution_record", False),
    )
    if all(markers):
        return 0
    if any(markers):
        raise RuntimeError(f"{REGION} found a partially installed engine hook set")

    @functools.wraps(current_sample_tokens)
    def sample_tokens(runner, grammar_output):
        token = _ENGINE_GRAMMAR_ACTIVE.set(grammar_output is not None)
        try:
            return current_sample_tokens(runner, grammar_output)
        finally:
            _ENGINE_GRAMMAR_ACTIVE.reset(token)

    @functools.wraps(current_sample)
    def sample(runner, logits, spec_decode_metadata):
        keys = engine_history_keys(
            runner,
            logits,
            spec_decode_metadata,
            grammar_active=_ENGINE_GRAMMAR_ACTIVE.get(),
        )
        token = _ENGINE_HISTORY_KEYS.set(keys)
        try:
            return current_sample(runner, logits, spec_decode_metadata)
        finally:
            _ENGINE_HISTORY_KEYS.reset(token)

    @functools.wraps(current_stream_update)
    def update_streaming_request(runner, req_id, new_req_data):
        _ENGINE_HASH_CACHE.pop((id(runner), req_id), None)
        return current_stream_update(runner, req_id, new_req_data)

    @functools.wraps(current_compute)
    def compute_logprobs(logits):
        output = current_compute(logits)
        keys = _ENGINE_HISTORY_KEYS.get()
        if keys is not None:
            _record_rows(output, keys, case="engine")
        return output

    sample_tokens._isoexec_full_distribution_grammar = True
    sample._isoexec_full_distribution_context = True
    update_streaming_request._isoexec_full_distribution_stream_reset = True
    compute_logprobs._isoexec_full_distribution_record = True
    model_runner_cls.sample_tokens = sample_tokens
    model_runner_cls._sample = sample
    model_runner_cls._update_streaming_request = update_streaming_request
    sampler_cls.compute_logprobs = staticmethod(compute_logprobs)
    return 4


def install_engine_hooks() -> int:
    """Install full-row capture around the pinned vLLM V1 sampling path."""
    try:
        from vllm.v1.sample.sampler import Sampler
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as error:  # noqa: BLE001 -- requested diagnostics fail closed in the adapter
        raise RuntimeError(f"{REGION} cannot import the pinned vLLM V1 sampling path") from error
    return _wrap_engine_classes(GPUModelRunner, Sampler)


def arm(side: str) -> int:
    """Declare the diagnostic as required and install its engine-side hooks."""
    global _ACTIVE_WEIGHT_VERSION, _LAST_WEIGHT_VERSION
    if not enabled():
        return 0
    if not trace.enabled():
        raise RuntimeError(f"{ENV}=1 requires {trace.ENV_TRACE} to be set")
    tracer = trace.get_tracer()
    if tracer is None:
        raise RuntimeError(f"{REGION} could not initialize the debug tracer")
    if tracer.sample != 1:
        raise RuntimeError(f"{REGION} currently requires {trace.ENV_SAMPLE}=1")
    if not tracer.wants(REGION):
        raise RuntimeError(f"{ENV}=1 requires {REGION!r} in {trace.ENV_REGIONS or 'the default region set'}")
    tracer.regions_hooked.add(REGION)
    tracer.write_manifest()
    _ACTIVE_WEIGHT_VERSION = None
    _LAST_WEIGHT_VERSION = None
    if side == "trainer":
        return 0
    if side == "engine":
        _ENGINE_HASH_CACHE.clear()
        return install_engine_hooks()
    raise RuntimeError(f"{REGION} requires debug side 'trainer' or 'engine', got {side!r}")


def _reset_for_tests() -> None:
    global _ACTIVE_WEIGHT_VERSION, _LAST_WEIGHT_VERSION
    _ACTIVE_WEIGHT_VERSION = None
    _LAST_WEIGHT_VERSION = None
    _ENGINE_HASH_CACHE.clear()


__all__ = [
    "ENV",
    "REGION",
    "arm",
    "clear_weight_version",
    "enabled",
    "engine_history_keys",
    "history_key",
    "record_trainer_action_distributions",
    "set_weight_version",
    "trainer_action_rows",
]
