"""Allocate chunk-synced state in a discard-at-sleep CUDA VMM arena.

When enabled, each layer's state tensors are views into one aligned allocation carrying the private
``isoexec_gdn_cs`` tag, which the worker sleep hook discards without a host backup; wake re-maps it,
zeros it and resets all row/lifecycle bookkeeping. Any unsupported allocator surface or ambiguous
allocation layout declines to the ordinary state allocation. ``SKYRL_ISOEXEC_GDN_CS_SLEEP=1`` enables
the feature.
"""

from __future__ import annotations

import os
import time

# A private cumem tag: not "weights" (backed up D2H at level-1 sleep) and not "kv_cache" (owned by
# vLLM's cache-initialization lifecycle). This one is discarded at sleep but woken by us.
CS_SLEEP_TAG = "isoexec_gdn_cs"

FLAG = "SKYRL_ISOEXEC_GDN_CS_SLEEP"

# torch caching-allocator constants this module's safety argument depends on.
_ROUND_LARGE = 2 << 20  # kRoundLarge: large allocations are rounded up to this
_MIN_LARGE_ALLOC = 10 << 20  # kMinLargeAlloc: below this a 20 MiB kLargeBuffer segment is used

# Layers whose state lives in a tagged arena: [(layer, flat_uint8_tensor, nbytes)].
_ARENAS: list = []
_installed = False
_asleep = False
_accounted = False


def cs_sleep_enabled() -> bool:
    """Whether the chunk-synced state arena is discarded at sleep and re-created at wake.

    Read at call time, so a test can toggle it per process.
    """
    return os.environ.get(FLAG, "0").lower() not in ("", "0", "false", "no")


def _say(msg: str) -> None:
    """Write to fd 1 rather than ``logger.info``: vLLM filters INFO out of engine subprocesses."""
    try:
        os.write(1, f"[ISOEXEC-CS-SLEEP] pid={os.getpid()} {msg}\n".encode())
    except Exception:  # pragma: no cover - fd 1 closed
        pass


def _round_up(n: int, m: int) -> int:
    return (n + m - 1) // m * m


def alloc_cs_arena(specs, device):
    """Allocate one tagged, discard-at-sleep arena for a layer and carve it into state tensors.

    ``specs`` is ``[(attr_name, shape, dtype), ...]`` in allocation order; returns
    ``{attr_name: tensor}``, or ``None`` so the caller allocates the tensors the ordinary way. Every
    rejection path here is a performance fallback, never a correctness one.
    """
    if not cs_sleep_enabled():
        return None
    import torch

    try:
        # The cumem VMM path must never run under stream capture (it aborts the worker).
        if torch.cuda.is_current_stream_capturing():
            return None
        from vllm.device_allocator.cumem import CuMemAllocator

        alloc = CuMemAllocator.instance  # not get_instance(): no sleep mode -> no arena
        if alloc is None:
            _say(f"INERT: {FLAG}=1 but no CuMemAllocator instance (engine built without sleep mode)")
            return None
        # Everything below reads cumem internals (``current_tag`` and ``pointer_to_data``), so
        # probe the surface here: a vLLM that renamed either must disable the feature, not raise
        # mid-construction.
        for _attr in ("current_tag", "pointer_to_data"):
            if not hasattr(alloc, _attr):
                _say(f"INERT: CuMemAllocator has no {_attr!r} -- vLLM's cumem surface moved")
                return None
    except Exception as e:  # pragma: no cover - vLLM absent
        _say(f"INERT: cumem unavailable ({e!r})")
        return None

    # Layout: 256 B-aligned slices of one flat uint8 buffer.
    offsets: list[tuple[str, int, int, tuple, "torch.dtype"]] = []
    cur = 0
    for name, shape, dtype in specs:
        n = 1
        for d in shape:
            n *= int(d)
        nbytes = n * torch.empty(0, dtype=dtype).element_size()
        cur = _round_up(cur, 256)  # keeps every view's storage offset dtype- and 16 B-aligned
        offsets.append((name, cur, nbytes, tuple(shape), dtype))
        cur += nbytes
    total = _round_up(cur, _ROUND_LARGE)
    if total < _MIN_LARGE_ALLOC:
        # A sub-10 MiB request lands in a 20 MiB kLargeBuffer segment whose remainder is splittable,
        # so a later weight allocation could end up inside memory we mark discard-at-sleep.
        _say(f"arena declined: {total / 2**20:.1f} MiB < {_MIN_LARGE_ALLOC / 2**20:.0f} MiB (splittable segment)")
        return None

    old_tag = alloc.current_tag
    alloc.current_tag = CS_SLEEP_TAG
    try:
        # A current_tag swap, not a nested ``use_memory_pool``: model build already holds an active
        # cumem MemPool context, so only the tag recorded by the malloc callback needs to change.
        flat = torch.zeros(total, dtype=torch.uint8, device=device)
    finally:
        alloc.current_tag = old_tag

    # Verify exclusivity: base pointer, our tag, exact size.
    try:
        data = alloc.pointer_to_data.get(flat.data_ptr())
        bad = None
        if data is None:
            bad = "not a cumem allocation base (served from a cached block)"
        elif data.tag != CS_SLEEP_TAG:
            bad = f"allocation carries tag {data.tag!r}"
        elif int(data.handle[1]) != total:
            bad = f"allocation is {int(data.handle[1])} B for a {total} B request (splittable remainder)"
    except Exception as e:  # the AllocationData layout moved: decline rather than trust it
        bad = f"could not verify the allocation against cumem's table ({e!r})"
    if bad is not None:
        del flat
        torch.cuda.empty_cache()
        _say(f"arena declined: {bad}")
        return None

    out = {}
    for name, off, nbytes, shape, dtype in offsets:
        out[name] = flat[off : off + nbytes].view(dtype).view(shape)
    out["_flat"] = flat
    return out


def register_layer(layer, flat, capacity: int) -> None:
    """Record a layer whose state lives in a tagged arena, and hook the worker on first use."""
    first = not _ARENAS
    _ARENAS.append((layer, flat, int(flat.numel())))
    if first:
        # Printed at build time, before any sleep, so its absence is diagnostic.
        _say(
            f"ARENA: {flat.numel() / 2**20:.1f} MiB per GDN layer at capacity={capacity} slots "
            f"({flat.numel() / max(capacity, 1) / 2**20:.3f} MiB/slot/layer), tag={CS_SLEEP_TAG!r}"
        )
    install_worker_sleep_hooks()


def arena_bytes() -> int:
    return sum(n for _l, _f, n in _ARENAS)


def _reset_layer(layer, zero_arena: bool) -> None:
    """Drop every meaning a row index could have, leaving the pool as freshly built.

    ``zero_arena`` is only ever true at wake: at sleep the arena is about to be unmapped, and after
    the sleep its pages are gone, so touching it would be an illegal access.
    """
    if zero_arena:
        layer._flat_arena.zero_()  # the state tensors are views of it
    layer.pos.zero_()
    for attr in ("slot2row", "last_used"):
        t = getattr(layer, attr, None)
        if t is not None:
            t.zero_()
    clock = getattr(layer, "_clock", None)
    if clock is not None:
        clock.zero_()
    layer._slot2row.clear()
    # The batched slot-map outbox goes with the map it feeds: a pending edit surviving this reset
    # would flush after the zeroing above and re-introduce a slot->row meaning that was destroyed.
    pending = getattr(layer, "_slot_pending", None)
    if pending is not None:
        pending.clear()
    # Row count read off the pool rather than re-derived from ``capacity``: the pool holds
    # ``capacity + 1`` rows (row 0 is the null row), and a second copy of that convention could drift.
    rows = int(layer.pos.numel())
    layer._free = list(range(1, rows))
    # Inverse ownership indices must be reset with the maps they describe; a stale one would
    # mis-name a row after the next wake even though the device/host maps were cleared.
    if hasattr(layer, "_row2slot"):
        layer._row2slot.clear()
    if hasattr(layer, "_free_set"):
        layer._free_set = set(layer._free)
    layer._entry_pos.clear()
    row_pos = getattr(layer, "_row_pos", None)
    if row_pos is not None:
        row_pos.clear()
    if hasattr(layer, "_driver_managed"):
        layer._driver_managed = False
    if hasattr(layer, "_last_used_host"):
        layer._last_used_host[:] = [0] * rows
        layer._clock_host = 0
        layer._host_lru_valid = True
    layer._apc_pending.clear()


def install_worker_sleep_hooks() -> None:
    """Patch ``gpu_worker.Worker.sleep``/``wake_up`` so the arena is dropped and re-created.

    Installed from the first arena allocation, i.e. during model build in the worker process and long
    before any sleep. Idempotent, and refuses loudly (leaving stock behavior) on a surface mismatch.
    """
    global _installed
    if _installed:
        return
    try:
        import inspect

        from vllm.device_allocator.cumem import CuMemAllocator
        from vllm.v1.worker import gpu_worker as _gw

        _ = (_gw.Worker.sleep, _gw.Worker.wake_up, CuMemAllocator.wake_up)
        # The replacements below are ``sleep(self, level=1)`` / ``wake_up(self, tags=None)``; if
        # vLLM's signatures move, refuse here rather than raise TypeError at the first sleep.
        _sig_sleep = list(inspect.signature(_gw.Worker.sleep).parameters)
        _sig_wake = list(inspect.signature(_gw.Worker.wake_up).parameters)
        if _sig_sleep != ["self", "level"] or _sig_wake != ["self", "tags"]:
            _say(
                f"NOT INSTALLED: Worker.sleep{tuple(_sig_sleep)} / Worker.wake_up{tuple(_sig_wake)} "
                "do not match the (self, level) / (self, tags) surface these hooks wrap"
            )
            return
    except Exception as e:  # pragma: no cover - fail to stock
        _say(f"NOT INSTALLED: worker/cumem surface mismatch ({e!r})")
        return
    if getattr(_gw.Worker.sleep, "_ix_cs_sleep", False):
        _installed = True
        return

    _orig_sleep = _gw.Worker.sleep
    _orig_wake = _gw.Worker.wake_up

    def cs_sleep(self, level: int = 1):
        global _asleep
        # Runs first and unconditionally -- before the double-sleep guard, the arena release and
        # vLLM's own sleep -- because it also retracts the shared boundary index advertised to the
        # scheduler. Exceptions propagate: an index advertising checkpoints no worker holds must
        # never pass silently.
        from .gdn_chunk_synced_state import CS_APC_STORE, cs_apc_store_invalidate

        apc_gib = CS_APC_STORE.nbytes / 2**30 if CS_APC_STORE is not None else 0.0
        cs_apc_store_invalidate()
        if _asleep:
            # Already released: the pages are unmapped, so the reset below would fault.
            _say("WARNING: sleep called while already asleep; arena release skipped (already released)")
            return _orig_sleep(self, level)
        _accounting_banner()
        t0 = time.perf_counter()
        gib = arena_bytes() / 2**30
        # Host-side bookkeeping is dropped before the pages go away, so no window exists in which
        # a map claims a row whose bytes are unmapped.
        for layer, _f, _n in _ARENAS:
            _reset_layer(layer, zero_arena=False)
        _asleep = True
        _say(
            f"RELEASE: {len(_ARENAS)} layers, {gib:.3f} GiB arena discarded at level-{level} sleep "
            f"(tag={CS_SLEEP_TAG!r}, NO D2H backup), {apc_gib:.3f} GiB boundary store cleared, "
            f"maps reset in {time.perf_counter() - t0:.3f}s"
        )
        out = _orig_sleep(self, level)
        _check_not_backed_up(level)
        return out

    def cs_wake_up(self, tags=None):
        global _asleep
        # ``tags is None`` means wake everything, and vLLM's allocator already re-maps our tag in
        # that case; calling wake_up([tag]) again would map an already-mapped handle. So for
        # tags=None only the bookkeeping is reset.
        woken_by_vllm = tags is None
        out = _orig_wake(self, tags)
        # The arena's lifetime matches vLLM's kv_cache lifetime: it returns in the wake that
        # re-maps the KV pool, so the weight-broadcast window runs with the arena still released --
        # which is where the trainer's NCCL buffers and the gathered weights are both resident.
        if not _asleep or not (tags is None or "kv_cache" in tags):
            return out
        t0 = time.perf_counter()
        try:
            from vllm.device_allocator.cumem import CuMemAllocator

            alloc = CuMemAllocator.instance
            if alloc is not None and not woken_by_vllm:
                alloc.wake_up([CS_SLEEP_TAG])
        except Exception as e:
            _say(f"FATAL: arena re-map failed ({e!r})")
            raise
        _asleep = False
        for layer, _f, _n in _ARENAS:
            _reset_layer(layer, zero_arena=True)  # remapped pages carry undefined contents
        _say(
            f"RESTORE: {len(_ARENAS)} layers, {arena_bytes() / 2**30:.3f} GiB "
            f"{'re-mapped by vLLM (tags=None)' if woken_by_vllm else 're-mapped'} and zeroed "
            f"in {time.perf_counter() - t0:.3f}s (tags={tags})"
        )
        return out

    cs_sleep._ix_cs_sleep = True
    cs_wake_up._ix_cs_sleep = True
    _gw.Worker.sleep = cs_sleep
    _gw.Worker.wake_up = cs_wake_up
    _installed = True
    _say(f"INSTALLED: Worker.sleep/wake_up hooked; arena tag={CS_SLEEP_TAG!r} is discard-at-sleep")


_backup_checked = False


def _check_not_backed_up(level: int) -> None:
    """Verify once that vLLM's sleep discarded our tag instead of copying it D2H.

    The private tag relies on ``Worker.sleep`` offloading only ``weights`` at level 1, which is an
    assumption about vLLM. Performance only: a backed-up arena is still correct, because wake zeroes
    it and resets every map regardless of what was restored.
    """
    global _backup_checked
    if _backup_checked or not _ARENAS:
        return
    _backup_checked = True
    try:
        from vllm.device_allocator.cumem import CuMemAllocator

        alloc = CuMemAllocator.instance
        if alloc is None:
            return
        backed = sum(
            int(d.handle[1])
            for d in alloc.pointer_to_data.values()
            if d.tag == CS_SLEEP_TAG and getattr(d, "cpu_backup_tensor", None) is not None
        )
    except Exception as e:  # pragma: no cover - diagnostic only
        _say(f"could not verify discard-at-sleep ({e!r})")
        return
    if backed:
        _say(
            f"WARNING: {backed / 2**30:.3f} GiB of the {CS_SLEEP_TAG!r} arena was BACKED UP to host "
            f"at level-{level} sleep -- this vLLM's sleep offloads our tag too, so the D2H/H2D copy "
            "this feature exists to remove is still being paid. Correctness is unaffected."
        )
    else:
        _say(f"verified: {CS_SLEEP_TAG!r} arena discarded at level-{level} sleep with no host backup")


def _accounting_banner() -> None:
    """One accounting line per process, printed before the first release."""
    global _accounted
    if _accounted or not _ARENAS:
        return
    _accounted = True
    n = len(_ARENAS)
    b = arena_bytes()
    _say(
        f"ACCOUNTING: {n} chunk-synced layers x {b / n / 2**20:.1f} MiB = {b / 2**30:.3f} GiB arena; "
        f"this is what is NOT copied D2H at each sleep and NOT copied H2D at each wake, and what "
        f"stays released through the weight broadcast"
    )
