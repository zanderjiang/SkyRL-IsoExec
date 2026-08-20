"""CPU guarantees for the pik autotune M-bucketing (the GLM ~3.85s decode-admission stall).

The mechanism being pinned: Triton's Autotuner caches the winning config per EXACT key
tuple whose element 0 is M (``_AUTOTUNE_KEY`` in gemm.py). A vLLM engine runs eager
prefill at essentially unique token counts, so without bucketing every admission step
is a cache miss = a full config-space benchmark. ``autotune_cache._BucketedCache``
collapses all M above ``PIK_AUTOTUNE_M_BUCKET_FLOOR`` into power-of-two buckets so a
process pays at most one sweep per (site, bucket).

What can be pinned without a GPU is the KEY ALGEBRA -- which is the entire fix; the
kernels and their bits are untouched by construction:

  1. Exact identity at and below the floor (captured-decode territory is untouched).
  2. Power-of-two bucketing above the floor, on lookup, store, and containment --
     i.e. the exact three operations Triton's Autotuner.run performs.
  3. Floor 0 disables bucketing (the A/B arm).
  4. ``register`` installs the cache on a real ``triton.runtime.Autotuner`` and
     legacy exact-M disk entries seed their bucket without clobbering in-process
     winners; ``items()`` yields canonical keys only (the disk cache converges).

Run (CPU only):
    uv run --isolated --extra dev python -m pytest \
        skyrl/backends/skyrl_train/isoexec/ops/collectives/tests/test_autotune_m_bucket_cpu.py -q
"""

from __future__ import annotations

import importlib
import pathlib
import sys

import pytest

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[7]))  # repo root

triton = pytest.importorskip("triton")


def _fresh_cache_module(monkeypatch, floor: str | None):
    if floor is not None:
        monkeypatch.setenv("PIK_AUTOTUNE_M_BUCKET_FLOOR", floor)
    else:
        monkeypatch.delenv("PIK_AUTOTUNE_M_BUCKET_FLOOR", raising=False)
    # a throwaway PIK_CACHE so _read() never sees a real autotune.json
    import tempfile

    monkeypatch.setenv("PIK_CACHE", tempfile.mkdtemp(prefix="pik_test_"))
    from skyrl.backends.skyrl_train.isoexec.ops.collectives.pik import autotune_cache

    return importlib.reload(autotune_cache)


_TAIL = (2048, 640, 2, "torch.bfloat16", "torch.bfloat16", "torch.float32")


def test_exact_below_floor_bucketed_above(monkeypatch):
    ac = _fresh_cache_module(monkeypatch, "384")
    c = ac._BucketedCache()

    # 1) captured-decode territory: exact keys, no aliasing
    c[(96,) + _TAIL] = "cfg96"
    assert (96,) + _TAIL in c
    assert (104,) + _TAIL not in c
    assert (384,) + _TAIL not in c  # floor itself is still exact

    # 2) eager-prefill territory: every M in (1024, 2048] shares one entry
    c[(1046,) + _TAIL] = "cfg2048"
    for m in (1025, 1174, 1603, 2048):
        assert (m,) + _TAIL in c, m
        assert c[(m,) + _TAIL] == "cfg2048"
    assert (1024,) + _TAIL not in c  # own bucket (2^10 maps to itself)
    assert (2049,) + _TAIL not in c  # next bucket

    # store canonicalizes too: items() yields the bucket key, not the exact M
    assert list(c.keys()) == [(96,) + _TAIL, (2048,) + _TAIL]


def test_floor_zero_disables(monkeypatch):
    ac = _fresh_cache_module(monkeypatch, "0")
    c = ac._BucketedCache()
    c[(1046,) + _TAIL] = "cfg"
    assert (1046,) + _TAIL in c
    assert (1047,) + _TAIL not in c


def test_non_conforming_keys_pass_through(monkeypatch):
    ac = _fresh_cache_module(monkeypatch, "384")
    c = ac._BucketedCache()
    # a key that does not start with an int M must not be rewritten
    c[("torch.float32", 4096)] = "x"
    assert ("torch.float32", 4096) in c
    assert list(c.keys()) == [("torch.float32", 4096)]


def test_register_installs_and_seeds(monkeypatch):
    ac = _fresh_cache_module(monkeypatch, "384")

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK": 64}, num_warps=4, num_stages=2),
        ],
        key=["M"],
    )
    @triton.jit
    def k(x_ptr, M, BLOCK: "triton.language.constexpr"):  # pragma: no cover - never launched
        pass

    assert isinstance(k, triton.runtime.Autotuner)

    # a pre-existing in-process winner at an exact M above the floor...
    # (a real Config: register() puts k in the module registry, and the atexit
    # save() serializes every cached value)
    inproc_winner = triton.Config({"BLOCK": 96}, num_warps=8, num_stages=3)
    k.cache[(1046,) + _TAIL] = inproc_winner
    # ...and a legacy exact-M disk entry landing in the SAME bucket
    import json

    ac._LOADED = {
        "k": {
            json.dumps([1174, *_TAIL]): {
                "kwargs": {"BLOCK": 64},
                "num_warps": 4,
                "num_stages": 2,
            },
            json.dumps([100, *_TAIL]): {
                "kwargs": {"BLOCK": 32},
                "num_warps": 4,
                "num_stages": 2,
            },
        }
    }
    ac.register("k", k)

    assert isinstance(k.cache, ac._BucketedCache)
    # the in-process winner survived the seeding (never clobbered)
    assert k.cache[(2048,) + _TAIL] is inproc_winner
    assert k.cache[(1603,) + _TAIL] is inproc_winner
    # the sub-floor disk entry seeded at its exact key
    assert (100,) + _TAIL in k.cache
    assert (101,) + _TAIL not in k.cache
    # idempotent: re-register must not rebuild the cache object
    cache_obj = k.cache
    ac.register("k", k)
    assert k.cache is cache_obj
    # keep the module-level registry clean for the atexit save()
    ac._REGISTRY.clear()
