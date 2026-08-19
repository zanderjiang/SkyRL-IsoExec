"""Executable ordering asserts for the IsoExec weight-sync lifecycle.

Each function encodes one ordering invariant of the sync path. The checks are observation-only (they
read a flag set at a companion seam and never mutate state, so removing every call leaves runtime
behavior identical), fail-soft (a violation or a bug in the check degrades to a
``[ISOEXEC-LIFECYCLE]`` warning), and gated by ``SKYRL_ISOEXEC_LIFECYCLE_ASSERTS=0``.

State is per-process module level: each invariant's mark and check run in the same process, so no
cross-actor coordination is needed. This module must not import ``core/flags.py``.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

_ASSERTS_ENV = "SKYRL_ISOEXEC_LIFECYCLE_ASSERTS"


def lifecycle_asserts_enabled() -> bool:
    """True unless explicitly disabled (``SKYRL_ISOEXEC_LIFECYCLE_ASSERTS=0``). Default ON."""
    return os.environ.get(_ASSERTS_ENV, "1") != "0"


def _warn(msg: str) -> None:
    text = f"[ISOEXEC-LIFECYCLE] INVARIANT VIOLATED: {msg}"
    # Ray drops actor module-logger INFO but forwards stdout; emit both so it is never lost.
    logger.warning(text)
    print(text, flush=True)


def _guard(check, *args, **kwargs) -> None:
    """Run a check fail-soft: a violated assert -> warning; a bug in the check -> ignored warning.

    A lifecycle observation must NEVER be able to crash a live training run."""
    if not lifecycle_asserts_enabled():
        return
    try:
        check(*args, **kwargs)
    except AssertionError as e:
        _warn(str(e))
    except Exception as e:  # noqa: BLE001 -- a bug in the CHECK must degrade, not propagate
        logger.warning(
            "[ISOEXEC-LIFECYCLE] check %s errored (ignored, not a run failure): %r",
            getattr(check, "__name__", check),
            e,
        )


# Invariant 1: reapply after the final wake. Seam: worker_dispatch.save_weights_for_sampler. The
# required order is wake_up(["weights"]) -> send -> wake_up(["kv_cache"]) -> reapply. A wake after
# the sync restores the never-updated step-0 CPU backup and clobbers the synced weights; moving the
# broadcast after the wake instead OOMs, because the staging buffer no longer fits beside the KV pool.
_final_wake = {"seen_kv_cache": False, "last_tags": None}


def mark_wake(tags: Optional[Sequence[str]]) -> None:
    """Record a wake_up at every wake seam; only a ``kv_cache`` wake counts as the final wake."""
    _final_wake["last_tags"] = list(tags) if tags is not None else None
    if tags is not None and "kv_cache" in tags:
        _final_wake["seen_kv_cache"] = True


def check_reapply_after_final_wake() -> None:
    """Assert (fail-soft) the final wake_up(['kv_cache']) ran before this reapply, consuming the flag."""

    def _c():
        assert _final_wake["seen_kv_cache"], (
            "isoexec_reapply_cached_weights is running BEFORE the final wake_up(['kv_cache']) "
            f"(last wake tags observed = {_final_wake['last_tags']}). Reapply must run AFTER the "
            "last wake or the wake clobbers the just-synced weights with stale step-0 bytes; "
            "moving the broadcast after the wake instead OOMs. See save_weights_for_sampler."
        )

    _guard(_c)
    _final_wake["seen_kv_cache"] = False


# Invariant 2: sleep tag scoping. Seam: vllm_engine.sleep. A kv_cache-scoped sleep must not also
# offload 'weights', which at level 1 silently reverts the policy to the step-0 backup. A full sleep
# (tags=None) is fine: the reapply after the final wake overwrites the restored stale weights.
def check_sleep_tag_scoping(tags: Optional[Sequence[str]], level: Optional[int] = None) -> None:
    """Assert (fail-soft) a kv_cache-scoped sleep does not also carry the 'weights' tag."""

    def _c():
        if tags is None:
            return  # full sleep: covered by the reapply-after-final-wake contract instead
        t = set(tags)
        if "kv_cache" in t:
            assert "weights" not in t, (
                f"sleep(level={level}, tags={list(tags)}) scopes the KV cache but ALSO offloads "
                "'weights'. At level 1 the weights pool is restored from the stale step-0 backup, "
                "so this reverts the policy to theta_0 with no reapply to fix it. Scope a kv_cache "
                "sleep to kv_cache only."
            )

    _guard(_c)


# Invariant 3: drain before the IPC RPC returns. Seam: vllm_worker.load_weights. Received tensors
# VIEW the sender's IPC-mapped chunk, which the sender reuses the instant the RPC returns, so the
# receiver must both torch.cuda.synchronize() (drain in-flight D2H copies) and torch.cuda.ipc_collect()
# (release the mapping, or the sender leaks its chunk buffers until it OOMs).
def check_ipc_drained_before_return(did_cuda_synchronize: bool, did_ipc_collect: bool) -> None:
    """Assert (fail-soft) both IPC drains ran before load_weights hands control back to the sender."""

    def _c():
        assert did_cuda_synchronize, (
            "load_weights (IsoExec) is about to return WITHOUT torch.cuda.synchronize(): the "
            "receiver views the sender's IPC-mapped chunk, and returning before the drain lets the "
            "sender reuse that buffer while async pinned-D2H copies are still in flight -> corruption."
        )
        assert did_ipc_collect, (
            "load_weights (IsoExec) is about to return WITHOUT torch.cuda.ipc_collect(): the opened "
            "IPC mapping is not released, so the sender cannot reclaim its packed chunk buffer "
            "(observed +8.7GiB per 3000 params -> OOM at ~62GiB mid-extraction)."
        )

    _guard(_c)


# Invariant 4: flush the prefix cache on sync. Seam: megatron_worker.broadcast_to_inference_engines.
# Otherwise generation resumes from prefix blocks computed under pre-sync weights.
def check_prefix_cache_flush_on_sync(use_prefix_cache: bool, should_flush: bool, flush_invoked: bool) -> None:
    """Assert (fail-soft) reset_prefix_cache was invoked when this sync was supposed to flush it."""

    def _c():
        if use_prefix_cache and should_flush:
            assert flush_invoked, (
                "weight sync ran with prefix caching enabled and the flush condition met, but "
                "reset_prefix_cache was NOT invoked -- stale prefix blocks would resume generation "
                "from pre-sync weights."
            )

    _guard(_c)


def kv_rebind_contract() -> str:
    """Pointer to invariant 5, which is implemented in ``runtimes/vllm/gdn_engine_patch.py`` rather
    than duplicated here.

    vLLM's colocate sleep/wake re-allocates the mamba kv_cache storage on each wake, so the GDN state
    core must re-check ``(kv_cache.data_ptr(), kv_cache.shape)`` on every call and rebuild when it
    changes; a cached reference would write the policy update onto freed storage."""
    return (
        "skyrl/backends/skyrl_train/isoexec/runtimes/vllm/gdn_engine_patch.py"
        "::_get_layer_state (_isoexec_gdn_kv_key rebind-on-(data_ptr,shape)-change)"
    )
