"""Pin a vLLM worker subprocess to its own GPU before anything can touch CUDA.

``WorkerWrapperBase.init_worker`` loads general plugins -- including ours -- before the
``init_device()`` RPC calls ``torch.cuda.set_device(local_rank)``, so any CUDA touch during plugin
import leaves a permanent primary context on GPU 0 in every worker. This moves that context to the
right device rather than adding one; it is a silent no-op outside a vLLM worker and is disabled with
``SKYRL_ISOEXEC_PIN_WORKER_DEVICE=0``.
"""

from __future__ import annotations

import os
import sys

_pinned = False


def _find_worker_local_rank() -> int | None:
    """``local_rank`` of the vLLM worker whose ``init_worker`` is on our call stack, or None.

    ``kwargs["local_rank"]`` is authoritative; ``rpc_rank`` is the fallback. The class test must stay
    a subclass test: ``RayDistributedExecutor`` rpcs ``init_worker`` on a ``WorkerWrapperBase``
    subclass, and an exact-name test silently turns this pin into a no-op there.
    """
    frame = sys._getframe(1)
    depth = 0
    base = getattr(sys.modules.get("vllm.v1.worker.worker_base"), "WorkerWrapperBase", None)
    while frame is not None and depth < 40:
        if frame.f_code.co_name == "init_worker":
            slf = frame.f_locals.get("self")
            if slf is not None and (
                isinstance(slf, base)
                if base is not None
                else any(c.__name__ == "WorkerWrapperBase" for c in type(slf).__mro__)
            ):
                kw = frame.f_locals.get("kwargs")
                if isinstance(kw, dict) and isinstance(kw.get("local_rank"), int):
                    return kw["local_rank"]
                rpc_rank = getattr(slf, "rpc_rank", None)
                if isinstance(rpc_rank, int):
                    return rpc_rank
        frame = frame.f_back
        depth += 1
    return None


def pin_worker_cuda_device() -> int | None:
    """Idempotently ``set_device`` this vLLM worker's own GPU. Returns the device, or None."""
    global _pinned
    if _pinned or os.environ.get("SKYRL_ISOEXEC_PIN_WORKER_DEVICE", "1") != "1":
        return None
    try:
        local_rank = _find_worker_local_rank()
        if local_rank is None:
            return None
        import torch

        if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
            return None
        already = torch.cuda.is_initialized()
        torch.cuda.set_device(local_rank)
        _pinned = True
        print(
            f"[ISOEXEC-DEVICE] pid {os.getpid()} pinned to cuda:{local_rank} before plugin import "
            f"(cuda_was_already_initialized={already})",
            flush=True,
        )
        return local_rank
    except Exception:
        # Fail-open: never let the pin break worker bring-up.
        return None
