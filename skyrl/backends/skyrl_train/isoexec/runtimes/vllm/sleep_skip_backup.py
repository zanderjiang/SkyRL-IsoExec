"""Skip the sleep-time D2H backup of weight bytes that the next weight sync overwrites before any read.

Only the residual, non-sync-covered byte ranges of each offloaded allocation are backed up; they are
restored right after ``wake_up`` remaps the handles. Gated on
``SKYRL_ISOEXEC_SLEEP_SKIP_WEIGHTS_BACKUP=1``, and fails to stock: if the vLLM cumem/worker surface
does not match, nothing is installed and vLLM's own sleep path runs unchanged.
"""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)

FLAG = "SKYRL_ISOEXEC_SLEEP_SKIP_WEIGHTS_BACKUP"

_state: dict = {
    "covered": (),
    "partial": {},
    "pieces": {},
}


def merge_intervals(intervals) -> list:
    ivs = sorted((int(a), int(b)) for a, b in intervals if int(b) > int(a))
    out: list = []
    for a, b in ivs:
        if out and a <= out[-1][1]:
            if b > out[-1][1]:
                out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return [tuple(x) for x in out]


def subtract_interval(span, covered) -> list:
    s, e = int(span[0]), int(span[1])
    out = []
    cur = s
    for a, b in covered:
        if b <= cur:
            continue
        if a >= e:
            break
        if a > cur:
            out.append((cur, min(a, e)))
        cur = min(max(cur, b), e)
        if cur >= e:
            break
    if cur < e:
        out.append((cur, e))
    return out


def _wanted_shape_dtype(entry):
    shape = getattr(entry, "shape", None)
    dtype = getattr(entry, "dtype", None)
    if shape is not None and dtype is not None:
        return tuple(shape), dtype
    if isinstance(entry, (tuple, list)) and len(entry) == 2:
        return tuple(entry[0]), entry[1]
    return None


def covered_intervals_from_named(cache, params, bufs) -> list:
    out = []
    for name, entry in cache.items():
        dest = params.get(name)
        if dest is None:
            dest = bufs.get(name)
        if dest is None:
            continue
        try:
            want = _wanted_shape_dtype(entry)
            if want is None:
                continue
            if dest.device.type != "cuda" or getattr(dest, "is_meta", False):
                continue
            if tuple(dest.shape) != want[0] or dest.dtype != want[1]:
                continue
            if not dest.is_contiguous():
                continue
            nbytes = dest.numel() * dest.element_size()
            if nbytes <= 0:
                continue
            ptr = dest.data_ptr()
        except Exception:
            continue
        out.append((ptr, ptr + nbytes))
    return merge_intervals(out)


def coverage_record(worker):
    cache = getattr(worker, "_ix_cached_weights", None)
    if cache:
        return cache
    meta = getattr(worker, "_ix_synced_meta", None)
    if meta:
        return meta
    return None


def covered_intervals_for_worker(worker) -> list:
    record = coverage_record(worker)
    if not record:

        return []
    model = getattr(getattr(worker, "model_runner", None), "model", None)
    if model is None:
        return []
    target = model.gpt if hasattr(model, "gpt") else model
    params = dict(target.named_parameters())
    bufs = dict(target.named_buffers())
    return covered_intervals_from_named(record, params, bufs)


def _sleep_impl(alloc, offload_tags, covered, state, *, memcpy, alloc_pinned, unmap) -> dict:
    covered = tuple(covered or ())
    partial = state["partial"]
    partial.clear()
    pieces = state["pieces"]

    total_b = full_b = residual_b = skipped_b = 0
    n_full = n_partial = n_ranges = 0

    for ptr, data in alloc.pointer_to_data.items():
        handle = data.handle
        size = handle[1]
        total_b += size
        if data.tag in offload_tags:
            residual = subtract_interval((ptr, ptr + size), covered) if covered else [(ptr, ptr + size)]
            if len(residual) == 1 and residual[0] == (ptr, ptr + size):

                buf = alloc_pinned(size)
                memcpy(buf.data_ptr(), ptr, size)
                data.cpu_backup_tensor = buf
                full_b += size
                n_full += 1
            else:
                plist = []
                res_here = 0
                for s, e in residual:
                    key = (ptr, s - ptr, e - s)
                    buf = pieces.get(key)
                    if buf is None:
                        buf = alloc_pinned(e - s)
                        pieces[key] = buf
                    memcpy(buf.data_ptr(), s, e - s)
                    plist.append((s - ptr, buf))
                    res_here += e - s
                partial[ptr] = (data, plist)
                residual_b += res_here
                skipped_b += size - res_here
                n_partial += 1
                n_ranges += len(plist)
        unmap(handle)

    return {
        "total_b": total_b,
        "full_b": full_b,
        "residual_b": residual_b,
        "skipped_b": skipped_b,
        "n_full": n_full,
        "n_partial": n_partial,
        "n_ranges": n_ranges,
    }


def _partial_restore(alloc, tags, memcpy, state) -> int:
    partial = state["partial"]
    restored = 0
    for ptr in list(partial.keys()):
        data, plist = partial[ptr]
        live = alloc.pointer_to_data.get(ptr)
        if live is not data:
            del partial[ptr]
            continue
        if tags is None or data.tag in tags:
            for off, buf in plist:
                memcpy(ptr + off, buf.data_ptr(), buf.numel())
                restored += 1
            del partial[ptr]
    return restored


def install_sleep_skip_weights_backup() -> None:
    if os.environ.get(FLAG) != "1":
        return
    try:
        from vllm.device_allocator import cumem as _cm
        from vllm.v1.worker import gpu_worker as _gw

        _ = (
            _cm.CuMemAllocator.default_tag,
            _cm.unmap_and_release,
            _cm.create_and_map,
            _cm.is_pin_memory_available,
            _cm.libcudart,
            _gw.Worker.sleep,
        )
    except Exception as e:
        logger.warning("[ISOEXEC-SLEEP-SKIP] cumem/worker surface mismatch, NOT installed (%s)", e)
        return
    if getattr(_cm.CuMemAllocator.sleep, "_ix_skip_backup", False):
        return

    import torch

    state = _state

    _orig_worker_sleep = _gw.Worker.sleep

    def worker_sleep(self, level: int = 1):
        if level == 1:
            try:
                state["covered"] = tuple(covered_intervals_for_worker(self))
            except Exception as e:
                state["covered"] = ()
                logger.warning("[ISOEXEC-SLEEP-SKIP] coverage failed, stock backup this sleep (%s)", e)
        try:
            return _orig_worker_sleep(self, level)
        finally:
            state["covered"] = ()

    worker_sleep._ix_skip_backup = True
    _gw.Worker.sleep = worker_sleep

    def alloc_pinned(nbytes: int):
        return torch.empty(nbytes, dtype=torch.uint8, device="cpu", pin_memory=_cm.is_pin_memory_available())

    def memcpy(dst: int, src: int, nbytes: int):
        _cm.libcudart.cudaMemcpy(dst, src, nbytes)

    def sleep_skip(self, offload_tags=None):
        import gc

        if offload_tags is None:
            offload_tags = (_cm.CuMemAllocator.default_tag,)
        elif isinstance(offload_tags, str):
            offload_tags = (offload_tags,)
        assert isinstance(offload_tags, tuple)

        t0 = time.perf_counter()
        stats = _sleep_impl(
            self,
            offload_tags,
            state["covered"],
            state,
            memcpy=memcpy,
            alloc_pinned=alloc_pinned,
            unmap=lambda h: _cm.unmap_and_release(h),
        )
        os.write(
            1,
            (
                f"[ISOEXEC-SLEEP-SKIP] pid={os.getpid()} freed {stats['total_b'] / 1024**3:.2f} GiB: "
                f"skipped {stats['skipped_b'] / 1024**3:.2f} GiB (sync-covered), backed up "
                f"{stats['residual_b'] / 1024**3:.2f} GiB residual in {stats['n_ranges']} ranges "
                f"/ {stats['n_partial']} handles + {stats['full_b'] / 1024**3:.2f} GiB full "
                f"in {stats['n_full']} handles, in {time.perf_counter() - t0:.2f}s\n"
            ).encode(),
        )
        gc.collect()
        torch.cuda.empty_cache()

    _orig_wake_up = _cm.CuMemAllocator.wake_up

    def wake_up_skip(self, tags=None):
        _orig_wake_up(self, tags)
        _partial_restore(self, tags, memcpy, state)

    sleep_skip._ix_skip_backup = True
    wake_up_skip._ix_skip_backup = True
    _cm.CuMemAllocator.sleep = sleep_skip
    _cm.CuMemAllocator.wake_up = wake_up_skip
    logger.info("[ISOEXEC-SLEEP-SKIP] installed (sync-covered weight bytes are discarded at sleep, not backed up)")
