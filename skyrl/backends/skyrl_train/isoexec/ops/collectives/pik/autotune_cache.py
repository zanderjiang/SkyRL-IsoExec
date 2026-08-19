"""Persist autotuned Triton configs to disk and bucket M in the tuning key.

Triton's autotuner cache is per-process, so without persistence every rank re-tunes from scratch at startup.
Above ``PIK_AUTOTUNE_M_BUCKET_FLOOR`` (default 384) M is rounded up to a power of two for config selection, so
that engines running a different token count every prefill step stop missing the cache; decode-sized M values
keep exact keys because they run inside CUDA graphs captured at discrete sizes. Neither is a correctness knob:
block sizes, warps, and stages cannot perturb the K-order, so a cache miss or a mis-bucketed config costs time
at most, never a bit.
"""

from __future__ import annotations

import atexit
import json
import os
import pathlib
import threading

import triton

_CACHE_DIR = pathlib.Path(os.environ.get("PIK_CACHE", pathlib.Path.home() / ".cache" / "pik"))
_CACHE_FILE = _CACHE_DIR / "autotune.json"

_REGISTRY: dict[str, triton.runtime.Autotuner] = {}
_LOCK = threading.Lock()
_LOADED: dict = {}
_dirty = False

# M <= floor keeps exact keys (CUDA-graph capture territory); above it M is bucketed. 0 disables bucketing.
_BUCKET_FLOOR = int(os.environ.get("PIK_AUTOTUNE_M_BUCKET_FLOOR", "384"))


def _canon(key):
    """Canonicalize an Autotuner cache key by bucketing M above the floor.

    ``_AUTOTUNE_KEY`` in gemm.py fixes element 0 as M; non-conforming keys pass through untouched.
    """
    if (
        _BUCKET_FLOOR > 0
        and isinstance(key, tuple)
        and key
        and isinstance(key[0], int)
        and not isinstance(key[0], bool)
        and key[0] > _BUCKET_FLOOR
    ):
        return (1 << (key[0] - 1).bit_length(),) + key[1:]
    return key


class _BucketedCache(dict):
    """A dict that canonicalizes keys so every M in a bucket shares one tuned config.

    ``items()`` yields the canonical keys, so the disk cache converges to a finite key set instead of growing
    one entry per batch size ever seen.
    """

    __slots__ = ()

    def __contains__(self, key):
        return dict.__contains__(self, _canon(key))

    def __getitem__(self, key):
        return dict.__getitem__(self, _canon(key))

    def __setitem__(self, key, value):
        dict.__setitem__(self, _canon(key), value)


def _cfg_to_dict(c: triton.Config) -> dict:
    return {"kwargs": c.kwargs, "num_warps": c.num_warps, "num_stages": c.num_stages}


def _dict_to_cfg(d: dict) -> triton.Config:
    return triton.Config(d["kwargs"], num_warps=d["num_warps"], num_stages=d["num_stages"])


def _read() -> dict:
    global _LOADED
    if not _LOADED:
        try:
            _LOADED = json.loads(_CACHE_FILE.read_text())
        except Exception:  # noqa: BLE001 -- a corrupt cache must never be fatal
            _LOADED = {"_": {}}
    return _LOADED


def register(name: str, kernel):
    """Attach an autotuner to the on-disk cache, seeding it with any known winners.

    The config cache is swapped for the M-bucketing one before seeding, so exact-M disk entries collapse into
    their buckets; the first entry seen for a bucket wins, which is a perf detail rather than a correctness one.
    """
    if not isinstance(kernel, triton.runtime.Autotuner):
        return kernel
    with _LOCK:
        _REGISTRY[name] = kernel
        if not isinstance(kernel.cache, _BucketedCache):
            fresh = _BucketedCache()
            for k, v in kernel.cache.items():
                fresh[k] = v
            kernel.cache = fresh
        for key_s, cfg in _read().get(name, {}).items():
            try:
                key = tuple(json.loads(key_s))
                if key not in kernel.cache:  # never clobber an in-process winner
                    kernel.cache[key] = _dict_to_cfg(cfg)
            except Exception:  # noqa: BLE001
                continue
    return kernel


def save() -> None:
    """Dump every autotuner's winners. Cheap; called at exit."""
    out = dict(_read())
    changed = False
    with _LOCK:
        for name, kern in _REGISTRY.items():
            entry = dict(out.get(name, {}))
            for key, cfg in kern.cache.items():
                key_s = json.dumps(list(key))
                if key_s not in entry:
                    entry[key_s] = _cfg_to_dict(cfg)
                    changed = True
            out[name] = entry
    if not changed:
        return
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _CACHE_FILE.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(out, indent=1))
        tmp.replace(_CACHE_FILE)  # atomic: ranks tune concurrently and race here
    except Exception:  # noqa: BLE001
        pass


atexit.register(save)
