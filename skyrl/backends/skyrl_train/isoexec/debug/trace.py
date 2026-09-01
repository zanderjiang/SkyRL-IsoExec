"""Debug-mode capture layer: record per-region output digests to a JSONL trace.

Gated entirely on ``SKYRL_ISOEXEC_DEBUG_TRACE``; unset means nothing installs and
:func:`wrap_region` returns its argument unchanged. Records land in
``{dir}/{side}-r{rank}-{host}-{pid}.jsonl`` with a ``manifest-*.json`` sidecar per process.

Env knobs (all read once, at first use):
  SKYRL_ISOEXEC_DEBUG_TRACE     trace directory (master switch)
  SKYRL_ISOEXEC_DEBUG_SIDE      "trainer" | "engine" -- labels records and names the file
  SKYRL_ISOEXEC_DEBUG_SAMPLE    N: record every Nth step (via set_step) or Nth call. Default 1
  SKYRL_ISOEXEC_DEBUG_LADDER    "1": also record the mantissa-truncation ladder
  SKYRL_ISOEXEC_DEBUG_SEGMENTS  R: also record one digest per R rows of the first non-unit dim
  SKYRL_ISOEXEC_DEBUG_REGIONS   region allow list; default is all but :data:`HIGH_VOLUME_REGIONS`
  SKYRL_ISOEXEC_DEBUG_RING      in-memory buffer size before a flush (default 4096 records)
  SKYRL_ISOEXEC_DEBUG_DIGEST    "auto" (default) | "eager" | "triton"; both sides must agree
"""

from __future__ import annotations

import atexit
import json
import os
import socket
import threading
import time
from typing import Callable, List, Optional, Set, Tuple

import torch

from . import thash

FORMAT_VERSION = 3

ENV_TRACE = "SKYRL_ISOEXEC_DEBUG_TRACE"
ENV_SIDE = "SKYRL_ISOEXEC_DEBUG_SIDE"
ENV_SAMPLE = "SKYRL_ISOEXEC_DEBUG_SAMPLE"
ENV_LADDER = "SKYRL_ISOEXEC_DEBUG_LADDER"
ENV_SEGMENTS = "SKYRL_ISOEXEC_DEBUG_SEGMENTS"
ENV_RING = "SKYRL_ISOEXEC_DEBUG_RING"
ENV_REGIONS = "SKYRL_ISOEXEC_DEBUG_REGIONS"

_WRAP_ATTR = "_isoexec_debug_region"

# Hooked only on request: `mm` fires once per projection per layer per microbatch on both sides.
HIGH_VOLUME_REGIONS = frozenset({"mm"})

# Regions that run once per transformer layer, so a per-step call ordinal is a meaningful layer
# index when no module context published one. Everything else records layer=null.
CALL_ORDER_LAYER_REGIONS = frozenset(
    {
        "attention.varlen",
        "gdn.conv",
        "gdn.core",
        "gdn.gating",
        "gdn.l2norm",
        "gdn.state",
        "moe.blockmap",
        "moe.combine",
        "moe.dispatch",
        "moe.epilogue",
        "moe.experts",
        "moe.router",
        "moe.weights",
        "norms.gated_out",
        "norms.rms",
        "rope.rope",
    }
)

_lock = threading.Lock()
_tracer: Optional["Tracer"] = None


def enabled() -> bool:
    return bool(os.environ.get(ENV_TRACE))


def resolve_rank() -> Tuple[int, str]:
    """(rank, how it was determined). Distributed rank when there is one, else env, else pid."""
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), "dist"
    except Exception:  # noqa: BLE001 -- a broken/absent process group must not break tracing
        pass
    for name in ("RANK", "LOCAL_RANK"):
        v = os.environ.get(name, "").strip()
        if v.lstrip("+-").isdigit():
            return int(v), f"env:{name}"
    return os.getpid(), "pid"


def _capturing() -> bool:
    """True while this thread's stream is capturing a CUDA graph (where that is knowable)."""
    try:
        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001 -- an old/CPU torch simply cannot be capturing
        return False


class Tracer:
    def __init__(self, trace_dir: str) -> None:
        self.dir = trace_dir
        self.side = os.environ.get(ENV_SIDE, "unknown")
        self.sample = max(1, int(os.environ.get(ENV_SAMPLE, "1")))
        self.ladder = os.environ.get(ENV_LADDER, "0") == "1"
        self.segment_rows = max(0, int(os.environ.get(ENV_SEGMENTS, "0")))
        self.ring = max(1, int(os.environ.get(ENV_RING, "4096")))
        self.regions_spec = os.environ.get(ENV_REGIONS, "").strip()
        self.rank, self.rank_src = resolve_rank()
        self.step: Optional[int] = None
        self.steps_seen: set = set()
        self.steps_recorded: set = set()
        self.n_records = 0
        self.capture_skipped = 0
        self.regions_hooked: Set[str] = set()
        self._seq = 0
        stem = f"{self.side}-r{self.rank}-{socket.gethostname()}-{os.getpid()}"
        self.path = os.path.join(self.dir, f"{stem}.jsonl")
        self.manifest_path = os.path.join(self.dir, f"manifest-{stem}.json")
        self._buf: List[dict] = []
        self._counts: dict = {}
        self._step_counts: dict = {}
        self._tl = threading.local()
        self._wlock = threading.Lock()
        os.makedirs(self.dir, exist_ok=True)
        self.write_manifest()

    # -- gating ----------------------------------------------------------------------------
    def wants(self, region: str) -> bool:
        """Region allow list. Default = everything but :data:`HIGH_VOLUME_REGIONS`."""
        spec = self.regions_spec
        if not spec:
            return region not in HIGH_VOLUME_REGIONS
        names = {s.strip() for s in spec.lstrip("+").split(",") if s.strip()}
        if "all" in names:
            return True
        if spec.startswith("+"):
            return region in names or region not in HIGH_VOLUME_REGIONS
        return region in names

    def enter(self, region: str) -> bool:
        """Claim the capture scope for ``region`` on this thread; False when already inside it."""
        depths = getattr(self._tl, "depths", None)
        if depths is None:
            depths = self._tl.depths = {}
        if depths.get(region):
            return False
        depths[region] = 1
        return True

    def exit(self, region: str) -> None:
        getattr(self._tl, "depths", {}).pop(region, None)

    def push_layer(self, layer: Optional[int]) -> Optional[int]:
        prev = getattr(self._tl, "layer", None)
        self._tl.layer = layer
        return prev

    def current_layer(self) -> Optional[int]:
        return getattr(self._tl, "layer", None)

    def bump(self, region: str) -> int:
        with _lock:
            n = self._counts.get(region, 0) + 1
            self._counts[region] = n
            self._step_counts[region] = self._step_counts.get(region, 0) + 1
        return n

    def step_call(self, region: str) -> int:
        return self._step_counts.get(region, 0)

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

    def resolve_layer(self, region: str, hooked: Optional[int]) -> Tuple[Optional[int], Optional[str]]:
        """(layer, layer_src): module hook, else per-step call ordinal, else unknown."""
        if hooked is not None:
            return hooked, "module"
        ctx = self.current_layer()
        if ctx is not None:
            return ctx, "module"
        if region in CALL_ORDER_LAYER_REGIONS:
            return max(0, self.step_call(region) - 1), "call_order"
        return None, None

    # -- recording -------------------------------------------------------------------------
    def _base(self, region: str, case: str, path: str, layer: Optional[int], src, call: int) -> dict:
        return {
            "v": FORMAT_VERSION,
            "region": region,
            "case": case,
            "side": self.side,
            "rank": self.rank,
            "rank_src": self.rank_src,
            "layer": layer,
            "layer_src": src,
            "step": self.step,
            "call": call,
            "out": path,
            "ts": round(time.time(), 6),
        }

    def _digest_record(self, rec: dict, t: torch.Tensor) -> dict:
        """Fill ``rec`` with shape/dtype/digest, or with an ``unrecordable`` reason."""
        rec["shape"] = list(t.shape)
        rec["dtype"] = str(t.dtype).replace("torch.", "")
        try:
            if self.ladder:
                lad = thash.digest_ladder(t)
                rec["digest"] = lad.pop("full")
                rec["ladder"] = lad
            else:
                rec["digest"] = f"{thash.tensor_digest(t):016x}"
            if self.segment_rows and t.dim() > 0:
                rec["segments"] = thash.segment_digests(t, rows_per_segment=self.segment_rows)
                rec["seg_rows"] = self.segment_rows
                rec["seg_axis"] = thash.segment_axis(t)
        except Exception as e:  # noqa: BLE001 -- an undigestable output is reported, never dropped
            rec.pop("digest", None)
            rec.pop("ladder", None)
            rec["unrecordable"] = f"{type(e).__name__}: {e}"
        return rec

    def record(self, region: str, case: str, out, *, layer: Optional[int], layer_src, call: int) -> None:
        recs: List[dict] = []
        for path, t in thash.iter_tensor_outputs(out):
            rec = self._base(region, case, path, layer, layer_src, call)
            if t is None:
                rec["unrecordable"] = f"nested output deeper than {thash.MAX_OUTPUT_DEPTH} levels"
                recs.append(rec)
            else:
                recs.append(self._digest_record(rec, t))
        if not recs:
            rec = self._base(region, case, "0", layer, layer_src, call)
            rec["unrecordable"] = f"no tensor outputs (got {type(out).__name__})"
            recs.append(rec)
        with _lock:
            for rec in recs:  # per-process execution order, exact where the clock is not
                self._seq += 1
                rec["seq"] = self._seq
            self._buf.extend(recs)
            self.n_records += len(recs)
            self.steps_recorded.add(self.step)
            full = len(self._buf) >= self.ring
        if full:
            self.flush()

    def write_manifest(self) -> None:
        """Per-process sidecar: what this trace covers, so the comparator can be honest about it."""
        man = {
            "v": FORMAT_VERSION,
            "side": self.side,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "rank": self.rank,
            "rank_src": self.rank_src,
            "sample": self.sample,
            "ladder": self.ladder,
            "segment_rows": self.segment_rows,
            "digest_backend": os.environ.get(thash.ENV_BACKEND, "auto"),
            "regions_hooked": sorted(self.regions_hooked),
            "capture_skipped": self.capture_skipped,
            "step_signal": bool(self.steps_seen),
            "steps_seen": sorted(s for s in self.steps_seen if s is not None),
            "steps_recorded": sorted(s for s in self.steps_recorded if s is not None),
            "records": self.n_records,
        }
        tmp = self.manifest_path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(man, f, separators=(",", ":"))
            os.replace(tmp, self.manifest_path)
        except OSError:  # a sidecar, rewritten on every flush -- never fail a run over it
            pass

    def flush(self) -> None:
        with _lock:
            buf, self._buf = self._buf, []
        if buf:
            with self._wlock, open(self.path, "a") as f:
                f.write("".join(json.dumps(rec, separators=(",", ":")) + "\n" for rec in buf))
        self.write_manifest()


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
    """Advance the forward/step key; also restarts the per-step call counters.

    With ``SKYRL_ISOEXEC_DEBUG_SAMPLE > 1`` it must be wired on BOTH sides, or they sample by
    different keys (step vs call ordinal) and select disjoint records.
    """
    t = get_tracer()
    if t is not None:
        with _lock:
            t._step_counts = {}
        t.step = step
        t.steps_seen.add(step)


def flush() -> None:
    if _tracer is not None:
        _tracer.flush()


def _reset_for_tests() -> None:
    global _tracer
    with _lock:
        _tracer = None


class layer_context:
    """Publish the layer index of the module whose forward is running, for the doors inside it.

    Reentrant and thread-local: an inner hook restores whatever the outer one had.
    """

    __slots__ = ("_layer", "_prev", "_tr")

    def __init__(self, layer: Optional[int]) -> None:
        self._layer = layer
        self._prev = None
        self._tr = None

    def __enter__(self):
        self._tr = get_tracer()
        if self._tr is not None:
            self._prev = self._tr.push_layer(self._layer)
        return self

    def __exit__(self, *exc):
        if self._tr is not None:
            self._tr.push_layer(self._prev)
        return False


def wrap_region(
    region: str,
    fn: Callable,
    *,
    case_fn: Optional[Callable] = None,
    layer_fn: Optional[Callable] = None,
    out_fn: Optional[Callable] = None,
) -> Callable:
    """Wrap an installed region impl so each (sampled, outermost-per-region) call records digests.

    Identity passthrough when tracing is disabled; wrapping a wrapper is a no-op. ``case_fn``,
    ``layer_fn`` and ``out_fn`` run only for calls that record (``out_fn`` covers doors that write
    into a caller-supplied buffer and return ``None``). Recording is skipped while a CUDA graph is
    capturing -- the digest's ``.item()`` D2H would poison the capture -- and counted into
    ``capture_skipped``. Every ``_isoexec_*`` attribute of ``fn`` is copied onto the wrapper
    because installers probe the live binding for capability markers.
    """
    if not enabled():
        return fn
    if getattr(fn, _WRAP_ATTR, None) is not None:
        return fn

    def wrapped(*args, **kwargs):
        tr = get_tracer()
        if tr is None or not tr.enter(region):
            return fn(*args, **kwargs)
        try:
            out = fn(*args, **kwargs)
            call = tr.bump(region)
            if tr.sampled(call) and not _capturing():
                try:
                    case = case_fn(args, kwargs, out) if case_fn else tr.default_case()
                    layer, src = tr.resolve_layer(region, layer_fn(args, kwargs) if layer_fn else None)
                    payload = out_fn(args, kwargs, out) if out_fn else out
                    tr.record(region, case, payload, layer=layer, layer_src=src, call=call)
                except Exception as e:  # noqa: BLE001 -- tracing must never break the forward
                    print(f"[ISOEXEC-DEBUG] record failed for {region}: {type(e).__name__}: {e}", flush=True)
            elif tr.sampled(call):
                with _lock:
                    tr.capture_skipped += 1
            return out
        finally:
            tr.exit(region)

    wrapped.__name__ = getattr(fn, "__name__", region)
    wrapped.__qualname__ = getattr(fn, "__qualname__", region)
    wrapped.__doc__ = getattr(fn, "__doc__", None)
    wrapped.__wrapped__ = fn
    for attr in dir(fn):
        if attr.startswith("_isoexec"):
            try:
                setattr(wrapped, attr, getattr(fn, attr))
            except Exception:  # noqa: BLE001 -- a marker we cannot copy is not worth a failed run
                pass
    setattr(wrapped, _WRAP_ATTR, region)
    setattr(wrapped, "_isoexec_debug_inner", fn)
    return wrapped
