"""Debug-mode capture layer: record per-region output digests to a JSONL trace.

Gated entirely on ``SKYRL_ISOEXEC_DEBUG_TRACE`` (the trace output directory). Unset -> nothing
installs, :func:`wrap_region` returns its argument unchanged, zero overhead. NOTE: this env (and
the companions below) currently reaches only the launching shell's process; to reach the Ray
trainer/engine actors it must be registered as a Flag in ``core/flags.py`` and forwarded by the
TRAIN and ENGINE channels -- see ``INTEGRATION.md``. Not done here: ``core/flags.py`` is owned by
a concurrent change.

Env knobs (all read once, at first use):
  SKYRL_ISOEXEC_DEBUG_TRACE   trace directory (master switch)
  SKYRL_ISOEXEC_DEBUG_SIDE    "trainer" | "engine" (labels records; also the file name)
  SKYRL_ISOEXEC_DEBUG_SAMPLE  N: record every Nth forward (via set_step) or, absent a step
                              signal, every Nth region call. Default 1.
  SKYRL_ISOEXEC_DEBUG_LADDER  "1": also record the mantissa-truncation ladder (k-ladder)
  SKYRL_ISOEXEC_DEBUG_RING    in-memory buffer size before a flush (default 4096 records)

Records are buffered in memory and appended to ``{dir}/{side}-{host}-{pid}.jsonl`` when the
buffer fills, on :func:`flush`, and at interpreter exit. Capture never runs re-entrantly: while
one region records, nested wrapped regions pass straight through, so a composite region yields
one record, not one per sub-op.

CUDA-graph caveat: a replayed graph does not run Python, so decode steps served from a captured
graph produce no records. Run debug traces with eager decode (debug mode does not need the
production execution config -- that is the point).
"""

from __future__ import annotations

import atexit
import json
import os
import socket
import threading
import time
from typing import Callable, List, Optional

import torch

from . import thash

ENV_TRACE = "SKYRL_ISOEXEC_DEBUG_TRACE"
ENV_SIDE = "SKYRL_ISOEXEC_DEBUG_SIDE"
ENV_SAMPLE = "SKYRL_ISOEXEC_DEBUG_SAMPLE"
ENV_LADDER = "SKYRL_ISOEXEC_DEBUG_LADDER"
ENV_RING = "SKYRL_ISOEXEC_DEBUG_RING"

_WRAP_ATTR = "_isoexec_debug_region"

_lock = threading.Lock()
_tracer: Optional["Tracer"] = None


def enabled() -> bool:
    return bool(os.environ.get(ENV_TRACE))


class Tracer:
    def __init__(self, trace_dir: str) -> None:
        self.dir = trace_dir
        self.side = os.environ.get(ENV_SIDE, "unknown")
        self.sample = max(1, int(os.environ.get(ENV_SAMPLE, "1")))
        self.ladder = os.environ.get(ENV_LADDER, "0") == "1"
        self.ring = max(1, int(os.environ.get(ENV_RING, "4096")))
        self.step: Optional[int] = None
        self.path = os.path.join(self.dir, f"{self.side}-{socket.gethostname()}-{os.getpid()}.jsonl")
        self._buf: List[dict] = []
        self._counts: dict = {}
        self._tl = threading.local()
        self._wlock = threading.Lock()
        os.makedirs(self.dir, exist_ok=True)

    # -- gating ----------------------------------------------------------------------------
    def enter(self) -> bool:
        """Claim the capture scope for this thread; False when already inside a region."""
        if getattr(self._tl, "depth", 0):
            return False
        self._tl.depth = 1
        return True

    def exit(self) -> None:
        self._tl.depth = 0

    def bump(self, region: str) -> int:
        with _lock:
            n = self._counts.get(region, 0) + 1
            self._counts[region] = n
        return n

    def sampled(self, call: int) -> bool:
        if self.step is not None:
            return self.step % self.sample == 0
        return (call - 1) % self.sample == 0

    def default_case(self) -> str:
        if self.side == "trainer":
            return "trainer_fwd" if torch.is_grad_enabled() else "trainer_score"
        if self.side == "engine":
            return "engine"
        return "unknown"

    # -- recording -------------------------------------------------------------------------
    def record(self, region: str, case: str, out, *, layer: Optional[int], call: int) -> None:
        for idx, t in thash.iter_tensor_outputs(out):
            rec = {
                "v": 1,
                "region": region,
                "case": case,
                "side": self.side,
                "layer": layer,
                "step": self.step,
                "call": call,
                "out": idx,
                "shape": list(t.shape),
                "dtype": str(t.dtype).replace("torch.", ""),
                "ts": round(time.time(), 3),
            }
            if self.ladder:
                lad = thash.digest_ladder(t)
                rec["digest"] = lad.pop("full")
                rec["ladder"] = lad
            else:
                rec["digest"] = f"{thash.tensor_digest(t):016x}"
            with _lock:
                self._buf.append(rec)
                full = len(self._buf) >= self.ring
            if full:
                self.flush()

    def flush(self) -> None:
        with _lock:
            buf, self._buf = self._buf, []
        if not buf:
            return
        with self._wlock, open(self.path, "a") as f:
            f.write("".join(json.dumps(rec, separators=(",", ":")) + "\n" for rec in buf))


def get_tracer() -> Optional[Tracer]:
    global _tracer
    if not enabled():
        return None
    if _tracer is None:
        with _lock:
            if _tracer is None:
                t = Tracer(os.environ[ENV_TRACE])
                atexit.register(t.flush)
                _tracer = t
    return _tracer


def set_step(step: int) -> None:
    """Advance the forward/step key; wired by integration (see INTEGRATION.md), optional."""
    t = get_tracer()
    if t is not None:
        t.step = step


def flush() -> None:
    if _tracer is not None:
        _tracer.flush()


def _reset_for_tests() -> None:
    global _tracer
    with _lock:
        _tracer = None


def wrap_region(
    region: str,
    fn: Callable,
    *,
    case_fn: Optional[Callable] = None,
    layer_fn: Optional[Callable] = None,
) -> Callable:
    """Wrap an installed region impl so each (sampled, outermost) call records its output digests.

    Identity passthrough when tracing is disabled (``wrap_region(r, f) is f``). Wrapping a
    wrapper is a no-op. ``case_fn(args, kwargs, out) -> str`` and ``layer_fn(args, kwargs) ->
    int|None`` are only invoked for calls that actually record.
    """
    if not enabled():
        return fn
    if getattr(fn, _WRAP_ATTR, None) is not None:
        return fn

    def wrapped(*args, **kwargs):
        tr = get_tracer()
        if tr is None or not tr.enter():
            return fn(*args, **kwargs)
        try:
            out = fn(*args, **kwargs)
            call = tr.bump(region)
            if tr.sampled(call):
                try:
                    case = case_fn(args, kwargs, out) if case_fn else tr.default_case()
                    layer = layer_fn(args, kwargs) if layer_fn else None
                    tr.record(region, case, out, layer=layer, call=call)
                except Exception as e:  # noqa: BLE001 -- tracing must never break the forward
                    print(f"[ISOEXEC-DEBUG] record failed for {region}: {type(e).__name__}: {e}", flush=True)
            return out
        finally:
            tr.exit()

    wrapped.__name__ = getattr(fn, "__name__", region)
    wrapped.__qualname__ = getattr(fn, "__qualname__", region)
    setattr(wrapped, _WRAP_ATTR, region)
    setattr(wrapped, "_isoexec_debug_inner", fn)
    return wrapped
