"""Memoize packed-sequence host metadata for the duration of a forward.

Several GDN and attention layers derive the same sequence lengths, terminal offset, FLA clone and
native-convolution metadata from one ``cu_seqlens`` tensor. Entries are keyed by tensor identity, version
counter, device and dtype and hold a strong reference to the tensor, so an in-place update or a recycled
buffer cannot reuse stale metadata. The memo removes host reads only; kernel arguments and order are unchanged.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from types import SimpleNamespace

import torch

BANNER = "[ISOEXEC-PACKED-META]"

_ENV_GATE = "SKYRL_ISOEXEC_PACKED_META_CACHE"

# ``key -> (strong_ref_to_cu, {derived_key: value})``. Bounded LRU.
_CACHE: OrderedDict = OrderedDict()
_CACHE_MAX = 8
_MAX_NUMEL = 8192  # decline to pin anything larger alive

_STATS = {"served": 0, "built": 0, "declined": 0}


def _versioned_key(cu: torch.Tensor):
    """Return the identity/version key, or ``None`` for inference tensors.

    Inference tensors carry no version counter (reading ``_version`` raises), and memoizing one by
    identity alone would be unsound because inference-mode in-place writes bump nothing. Declining
    is the only safe behavior.
    """

    try:
        version = cu._version
    except RuntimeError:
        if torch.is_inference(cu):
            return None
        raise
    return (id(cu), version, str(cu.device), str(cu.dtype))


def packed_meta_cache_enabled() -> bool:
    """``SKYRL_ISOEXEC_PACKED_META_CACHE`` (default on). Off restores the per-layer host read."""
    return os.environ.get(_ENV_GATE, "1").lower() not in ("0", "false", "no", "")


def packed_meta_census() -> dict:
    """``{served, built, declined}`` counters."""
    return dict(_STATS)


def _entry(cu: torch.Tensor) -> dict | None:
    """The derived-value dict for ``cu``, or ``None`` when this tensor must not be memoized.

    Declines non-tensors and tensors large enough that pinning them (and their base storage) alive
    would cost real memory.
    """
    if not packed_meta_cache_enabled():
        return None
    if not torch.is_tensor(cu) or cu.numel() > _MAX_NUMEL:
        return None
    key = _versioned_key(cu)
    if key is None:
        return None
    hit = _CACHE.get(key)
    if hit is not None:
        _CACHE.move_to_end(key)
        return hit[1]
    # The strong reference to `cu` is what keeps the id() half of the key unforgeable; never drop
    # it while the entry lives.
    derived: dict = {}
    _CACHE[key] = (cu, derived)
    if len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return derived


def is_memoized(cu: torch.Tensor, name="list") -> bool:
    """True when ``cu`` already holds ``name``. A peek that must not create an entry.

    Probing with :func:`_entry` instead would insert one entry per freshly-computed ``cu // cp_size``
    and evict the real ``cu_seqlens`` entry out of the bounded LRU.
    """
    if not packed_meta_cache_enabled() or not torch.is_tensor(cu) or cu.numel() > _MAX_NUMEL:
        return False
    key = _versioned_key(cu)
    if key is None:
        return False
    hit = _CACHE.get(key)
    return hit is not None and name in hit[1]


def _get(cu: torch.Tensor, name, build):
    """Memoized ``build()`` under ``name``, or a plain ``build()`` when the tensor is declined."""
    derived = _entry(cu)
    if derived is None:
        _STATS["declined"] += 1
        return build()
    if name in derived:
        _STATS["served"] += 1
        return derived[name]
    value = build()
    derived[name] = value
    _STATS["built"] += 1
    return value


def cu_list(cu: torch.Tensor) -> list:
    """``cu.tolist()``, memoized. A pure host read: no kernel is issued either way."""
    return _get(cu, "list", cu.tolist)


def cu_last(cu: torch.Tensor) -> int:
    """``int(cu[-1])``, memoized off the same host read as :func:`cu_list`."""
    return _get(cu, "last", lambda: cu_list(cu)[-1])


def seq_lens(cu: torch.Tensor, div: int = 1) -> list:
    """``((cu[1:] - cu[:-1]) // div).tolist()``, memoized and computed on the host from :func:`cu_list`.

    Exact: ``cu`` is a monotone non-negative integer offset vector, so host subtraction and floor
    division agree elementwise with the device expression, and the two launches disappear.
    """
    if div < 1:
        raise ValueError(f"{BANNER} seq_lens: div must be >= 1, got {div}")

    def build():
        vals = cu_list(cu)
        return [(vals[i + 1] - vals[i]) // div for i in range(len(vals) - 1)]

    return _get(cu, ("lens", int(div)), build)


def fla_stable_clone(cu: torch.Tensor) -> torch.Tensor:
    """An identity-stable clone of ``cu``, memoized, as the key for FLA's ``@tensor_cache``.

    FLA caches ``prepare_chunk_indices``/``prepare_chunk_offsets`` on tensor identity, so a fresh
    per-layer clone misses every time. A clone that is stable for this ``cu_seqlens`` object lets
    those caches hit across layers while a recycled buffer -- different id or bumped ``_version`` --
    still gets its own clone.
    """
    return _get(cu, "fla_clone", cu.clone)


def causal_conv1d_metadata(cu: torch.Tensor, *, cu_host=None, include_launch_args: bool = True):
    """vLLM causal-conv launch metadata and immutable arguments, built once per packed forward.

    vLLM otherwise derives this inside every GDN layer via ``cu.diff().to("cpu")``, draining the
    compute stream once per layer. ``cache_indices`` and ``has_initial_state`` ride along because the
    trainer's stateless call would otherwise rebuild those same immutable tensors per layer. Returns
    ``None`` when the cache is off, which restores vLLM's per-call construction exactly.
    """
    if not packed_meta_cache_enabled():
        return None
    if cu_host is not None:
        try:
            cu_host = [int(v) for v in cu_host]
        except (TypeError, ValueError):
            return None
        if len(cu_host) != cu.numel() or not cu_host or cu_host[0] != 0:
            return None

    def build():
        from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

        # Engine callers pass the scheduler-owned host offsets, removing even the memo's first
        # device read; trainer callers omit them and use the memoized `cu_list` path.
        cu_cpu = torch.tensor(cu_host if cu_host is not None else cu_list(cu), dtype=cu.dtype, device="cpu")
        nums, batch_ptr, token_offsets = compute_causal_conv1d_metadata(cu_cpu, device=cu.device)
        result = SimpleNamespace(
            nums_dict=nums,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_offsets,
        )
        if include_launch_args:
            # Stateless trainer convention: row zero is the null row and sequence i owns row i+1.
            # Both are read-only kernel arguments, so sharing them across layers changes no state.
            result.cache_indices = torch.arange(1, cu.numel(), dtype=torch.int32, device=cu.device)
            result.has_initial_state = torch.zeros(cu.numel() - 1, dtype=torch.bool, device=cu.device)
        return result

    kind = "causal_conv1d_metadata" if include_launch_args else "causal_conv1d_core_metadata"
    return _get(cu, kind, build)
