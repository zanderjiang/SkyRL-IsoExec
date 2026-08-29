"""Debug-mode capture layer: record per-region output digests to a JSONL trace.

Gated entirely on ``SKYRL_ISOEXEC_DEBUG_TRACE`` (the trace output directory). Unset -> nothing
installs, :func:`wrap_region` returns its argument unchanged, zero overhead. NOTE: this env (and
the companions below) currently reaches only the launching shell's process; to reach the Ray
trainer/engine actors it must be registered as a Flag in ``core/flags.py`` and forwarded by the
TRAIN and ENGINE channels -- see ``INTEGRATION.md``. Not done here: ``core/flags.py`` is owned by
a concurrent change.

Env knobs (all read once, at first use):
  SKYRL_ISOEXEC_DEBUG_TRACE     trace directory (master switch)
  SKYRL_ISOEXEC_DEBUG_SIDE      "trainer" | "engine" (labels records; also the file name)
  SKYRL_ISOEXEC_DEBUG_SAMPLE    N: record every Nth forward (via set_step) or, absent a step
                                signal, every Nth region call. Default 1. MUST be driven by
                                ``set_step`` on BOTH sides or the two sides select disjoint
                                record sets; the manifest records which, and the comparator
                                refuses to call a disjoint comparison clean.
  SKYRL_ISOEXEC_DEBUG_LADDER    "1": also record the mantissa-truncation ladder (k-ladder). One
                                fused pass since the triton digest backend landed, so this is
                                ~2-3x a plain digest, not one pass per rung.
  SKYRL_ISOEXEC_DEBUG_SEGMENTS  R: also record one digest per R rows of the tensor's first
                                non-unit dim, so the comparator can say WHICH rows diverge and
                                can tell a localized fault from a whole-tensor round-off
                                difference. Off (0) by default; one extra kernel per record.
  SKYRL_ISOEXEC_DEBUG_REGIONS   comma-separated allow list of regions to hook. Default: every
                                region except the ones in :data:`HIGH_VOLUME_REGIONS` (``mm``
                                fires on every projection of every layer and dwarfs the rest by
                                two orders of magnitude). ``all`` opts back in; a leading ``+``
                                adds to the default set (``+mm``).
  SKYRL_ISOEXEC_DEBUG_RING      in-memory buffer size before a flush (default 4096 records)
  SKYRL_ISOEXEC_DEBUG_DIGEST    "auto" (default) | "eager" | "triton" -- digest backend, see
                                ``thash``. Both sides must agree; the manifest records it.

Records are buffered in memory and appended to ``{dir}/{side}-r{rank}-{host}-{pid}.jsonl`` when
the buffer fills, on :func:`flush`, and at interpreter exit. Alongside them each process writes
``{dir}/manifest-{side}-r{rank}-{host}-{pid}.json``: the sampling parameters, the rank and how it
was determined, and which steps were seen versus actually recorded. The comparator needs the
manifest to distinguish "verified clean" from "not observed".

Causal order: every record carries ``ts`` (microsecond wall clock) and ``seq`` (a per-process
counter). The comparator orders divergences by ``(ts, seq)`` so the FIRST DIVERGENCE it reports is
the one that executed first, exactly within a process and approximately across them.

Rank: every record carries one, so traces from many processes in one directory align per rank
instead of by pid sort order. It is the torch.distributed rank when a process group is
initialized, else ``RANK``/``LOCAL_RANK`` from the environment, else the pid as a last resort;
``rank_src`` says which. Resolution happens once, when the tracer is built -- at the documented
call sites (after model build / after ``swap_gdn_core``) the process group is already up.

Layer: the comparator groups by ``(region, layer, out, rank)``, so a layer index that means
different things on the two sides is worse than none. It is resolved in one order on BOTH sides
and ``layer_src`` records which rung answered: ``"module"`` -- a module-level hook published the
index of the layer whose forward is running (:func:`layer_context`, installed from the decoder
walk, and the megatron ``layer_number - 1`` it reads is global, pipeline offset included);
``"call_order"`` -- for the per-layer regions in :data:`CALL_ORDER_LAYER_REGIONS`, the region's
call ordinal within the current step, which equals the layer index for a single-microbatch
forward and drifts by a whole multiple of the layer count otherwise; ``null`` -- unknown.

Record format version is :data:`FORMAT_VERSION`; the comparator refuses anything else rather
than silently mis-reading an older schema.

Capture nests per REGION: while a ``gdn.core`` call is recording, a nested ``gdn.core`` binding
(the same op reached through a second namespace, or a composite that calls its own kernel) passes
straight through, but a nested ``norms.rms`` still records. That is what makes a 20-region hook
table useful; a single global guard would let the outermost door swallow every inner one.

CUDA graphs: a hook reached *during capture* must not record -- the digest ends in a ``.item()``
D2H copy, which poisons the capture. :func:`wrap_region` checks
``torch.cuda.is_current_stream_capturing()`` before recording and counts the skips into
``capture_skipped`` in the manifest, so tracing can never break a run that captures graphs. It is
still a misconfiguration: a replayed graph runs no Python at all, so decode steps served from a
captured graph produce no records either way. ``ContractAdapter._install_debug_trace`` refuses at
init when the engine arms tracing with ``SKYRL_ISOEXEC_ENABLE_CUDAGRAPH=1``; this check is the
defensive backstop for capture reached by any other route. The comparator reports a resulting
hole as an ``absent`` divergence and names this cause when the absent side is the engine.
"""

from __future__ import annotations

import atexit
import json
import os
import socket
import threading
import time
from typing import Callable, Dict, List, Optional, Set, Tuple

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
# index when no module context published one. Everything else records layer=null instead of a
# number that would mean different things on the two sides.
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
        """(layer, layer_src) -- the same ladder on both sides, see the module docstring."""
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
    """Advance the forward/step key; wired by integration (see INTEGRATION.md).

    Optional only in the sense that it can be left unwired on BOTH sides. With
    ``SKYRL_ISOEXEC_DEBUG_SAMPLE > 1`` it must be wired on both, or the two sides sample by
    different keys (step vs call ordinal) and select disjoint records. It also restarts the
    per-step call counters that back ``layer_src="call_order"``.
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

    Reentrant and thread-local: an inner hook restores whatever the outer one had. Used by
    ``install.install_layer_context_hooks`` so the same kernel door reports a real layer index on
    the trainer and on the engine.
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

    Identity passthrough when tracing is disabled (``wrap_region(r, f) is f``). Wrapping a
    wrapper is a no-op. ``case_fn(args, kwargs, out) -> str``, ``layer_fn(args, kwargs) ->
    int|None`` and ``out_fn(args, kwargs, out) -> object`` are only invoked for calls that
    actually record; ``out_fn`` exists for doors that write into a caller-supplied buffer and
    return ``None``, so the digest can be taken from the buffer they filled.

    Every ``_isoexec_*`` attribute of ``fn`` is copied onto the wrapper: several installers probe
    the live binding for capability markers (``_isoexec_accepts_out_dtype``, ``_isoexec_tiled``)
    and a wrapper that hid them would silently change what the run executes.
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
