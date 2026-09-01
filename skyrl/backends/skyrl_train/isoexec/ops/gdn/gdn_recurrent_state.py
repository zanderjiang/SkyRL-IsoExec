"""Recurrent GDN state and engine prefill/decode implementation.

Private mode maps slot ids onto a bounded fp32 state pool with row 0 reserved for padded graph lanes;
native-state mode binds vLLM's mamba tensors and indexes them by block id. Chunked-prefill resume
fails closed unless scheduler metadata or the slot map proves the carried state is present.
"""

from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np
import torch

from .gdn_ops import (
    gdn_causal_conv,
    gdn_causal_conv_batched,
    gdn_gate_and_beta,
    gdn_l2norm,
    gdn_native_core_kernel,
    gdn_native_kernels_enabled,
    gdn_recurrent_kernel,
)

# Entries in the slot->row map; must exceed the engine's slot-id space. Sized once at construction
# (a captured CUDA graph holds this tensor's address) and read then, not at import time.
_SLOT_MAP_ENV = "SKYRL_ISOEXEC_GDN_SLOT_MAP_SIZE"


def _slot_map_size() -> int:
    v = os.environ.get(_SLOT_MAP_ENV)
    if v:
        return int(v)
    # CPR_MIN_PAGES shrinks the page size, multiplying vLLM's block-id space; every id the mamba
    # group hands a live request must still fit this map (out-of-range live ids raise in _assign).
    from .gdn_ops import gdn_cpr_min_pages

    return (1 << 20) if gdn_cpr_min_pages() else 65536


# Chunked-prefill resume, default off. The signal is a slot already present in ``self._slot2row``
# before ``_assign``; off treats every prompt as fresh regardless of re-presented slots.
_CHUNKED_PREFILL_ENV = "SKYRL_ISOEXEC_GDN_CHUNKED_PREFILL"


def chunked_prefill_enabled() -> bool:
    """True iff continuation chunks of a prefill should resume from the prompt's carried state."""
    return os.environ.get(_CHUNKED_PREFILL_ENV, "0").lower() not in ("", "0", "false", "no")


# State ownership: off (default) uses the private max_num_seqs pool behind a slot->row map, on puts
# the state in vLLM's mamba kv_cache blocks indexed by block id. The compute is identical either way.
_NATIVE_STATE_ENV = "SKYRL_ISOEXEC_GDN_NATIVE_STATE"


def native_state_enabled() -> bool:
    """True iff the recurrent GDN state should live in vLLM's ``kv_cache`` blocks, not our pool."""
    return os.environ.get(_NATIVE_STATE_ENV, "0").lower() not in ("", "0", "false", "no")


# Counters for the two host reads that would otherwise sync every GDN layer in a prefill: the LRU
# snapshot and cpr's position gather (bumped in gdn_cpr_state).
_HOST_BOOKKEEPING_STATS = {
    "lru_mirror": 0,
    "lru_device_fallback": 0,
    "pos_mirror": 0,
    "pos_device_fallback": 0,
    "driver_steps": 0,
    # Prefill suffix position mirror.
    "driver_position_layer_calls": 0,
    "driver_position_rows": 0,
}


def build_recurrent_gdn(
    *,
    max_num_seqs: int,
    ssm_state: torch.Tensor | None = None,
    conv_state: torch.Tensor | None = None,
    **kw,
) -> "RecurrentGDN":
    """Build a :class:`RecurrentGDN` sized by the scheduler's concurrency cap.

    ``ssm_state``/``conv_state`` (both or neither) hand the layer vLLM's mamba kv_cache tensors;
    ``conv_state`` must already be oriented ``(num_blocks, D, W-1)`` and ``ssm_state`` must be fp32.
    """
    return RecurrentGDN(capacity=max_num_seqs, ssm_state=ssm_state, conv_state=conv_state, **kw)


def flush_slot_maps(layers) -> int:
    """Flush every layer's slot-map outbox with one host->device upload for the whole stack.

    Per-layer outboxes are never assumed equal: ``_assign`` flushes on its own schedule, and a wrong
    shared key set would point a slot at another request's row.
    """
    pend = [(ly, arr) for ly in layers if not ly._native for arr in (ly._pending_arrays(),) if arr is not None]
    if not pend:
        return 0
    total = sum(len(k) for _, (k, _) in pend)
    keys = np.empty(total, dtype=np.int64)
    vals = np.empty(total, dtype=np.int64)
    spans = []
    off = 0
    for ly, (k, v) in pend:
        n = len(k)
        keys[off : off + n] = k
        vals[off : off + n] = v
        spans.append((ly, off, off + n))
        off += n
    dev = spans[0][0].slot2row.device
    ki = torch.from_numpy(keys).to(dev, non_blocking=True)
    vi = torch.from_numpy(vals).to(dev, non_blocking=True)
    for ly, a, b in spans:
        ly.slot2row[ki[a:b]] = vi[a:b]
        ly._slot_pending.clear()
    return total


class RecurrentGDN:
    """One GDN layer's inference core in recurrent mode: an owned state pool behind a slot->row map.

    vLLM's slot ids cannot index a private tensor directly -- for a hybrid model the block-id space is
    sized by the shared KV pool, so raw ids run past the pool's row count. Row 0 is the null row and is
    never handed out; unknown slots and padded graph lanes resolve to it and the kernel skips index <= 0.
    The map is a device tensor, which is what lets a pure-decode step capture into a CUDA graph.
    """

    def __init__(
        self,
        *,
        capacity: int,
        conv_weight: torch.Tensor,  # [D, W]
        conv_bias: torch.Tensor | None,
        A_log: torch.Tensor,  # [Hv]
        dt_bias: torch.Tensor,  # [Hv]
        num_k_heads: int,
        head_k_dim: int,
        num_v_heads: int,
        head_v_dim: int,
        activation: str | None = "silu",
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cuda",
        ssm_state: torch.Tensor | None = None,
        conv_state: torch.Tensor | None = None,
        slot_map_hint: int = 0,
    ):
        if num_v_heads % num_k_heads:
            raise ValueError(f"num_v_heads {num_v_heads} must be a multiple of num_k_heads {num_k_heads}")

        W = conv_weight.shape[-1]
        D = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
        self.capacity = capacity

        # Native state: vLLM owns allocation and eviction, so there is no private pool, slot map,
        # free list or device clock, and _rows is the identity.
        self._native = ssm_state is not None or conv_state is not None
        if self._native:
            if ssm_state is None or conv_state is None:
                raise ValueError("[isoexec-gdn] native state needs BOTH ssm_state and conv_state (vLLM's kv_cache)")
            # fp32 is mandatory: `mamba_ssm_cache_dtype=auto` resolves to bf16, which rounds the state
            # every read/write and breaks the fp32 prefill->decode round-trip.
            if ssm_state.dtype != torch.float32:
                raise RuntimeError(
                    f"[isoexec-gdn] native mamba ssm cache resolved to {ssm_state.dtype}, not float32. "
                    "Set mamba_ssm_cache_dtype=float32 at engine build -- bf16 rounds the recurrent "
                    "state every step and breaks the fp32 prefill->decode round-trip."
                )
            # conv_state arrives already oriented (num_blocks, D, W-1); the resulting non-contiguous
            # view is fine, since every consumer on this path is stride-aware.
            if conv_state.shape[-1] != W - 1:
                raise RuntimeError(
                    f"[isoexec-gdn] native conv_state width {conv_state.shape[-1]} != W-1 = {W - 1}; "
                    "the (num_blocks, D, W-1) orientation is wrong (SD/DS transpose)."
                )
            self.ssm_state = ssm_state
            self.conv_state = conv_state
            # vLLM's native fused kernels (conv fn/update + fused_sigmoid_gating core). Read once at
            # build; the state layer rebuilds this object on any kv_cache rebind.
            self._native_kernels = gdn_native_kernels_enabled()
        else:
            self._native_kernels = False
            rows = capacity + 1  # row 0 = the null row; requests live at 1..capacity

            # fp32, not bf16: prefill->decode chaining rests on the state surviving the round trip
            # through memory unchanged, and bf16 would round it at every such boundary.
            self.ssm_state = torch.zeros(rows, num_v_heads, head_v_dim, head_k_dim, dtype=torch.float32, device=device)
            self.conv_state = torch.zeros(rows, D, W - 1, dtype=dtype, device=device)

            # Engine slot id -> our row; unknown slots map to row 0. Sized once and never reallocated,
            # because a captured graph reads this tensor's address on every replay.
            self.slot2row = torch.zeros(max(_slot_map_size(), int(slot_map_hint)), dtype=torch.long, device=device)
            self.last_used = torch.zeros(rows, dtype=torch.long, device=device)
            # A device clock: `last_used[rows] = <python int>` would stage the scalar through pageable
            # host memory -- a sync, and uncapturable.
            self._clock = torch.zeros((), dtype=torch.long, device=device)

            self._slot2row: OrderedDict[int, int] = OrderedDict()  # host mirror, prefill-only
            self._free: list[int] = list(range(1, rows))  # row 0 is never handed out
            # Exact inverse indices for the two collections above; every lifecycle mutation updates
            # both views together, making alias/free-list validation O(1).
            self._row2slot: dict[int, int] = {}
            self._free_set: set[int] = set(self._free)

            # Outbox for batched slot-map writes: ``slot2row[i] = v`` on a CUDA tensor stages a scalar
            # through pageable host memory. ``_slot2row`` stays the authoritative host view.
            self._slot_pending: dict[int, int] = {}

        self.conv_weight = conv_weight
        self.conv_bias = conv_bias
        self.A_log = A_log
        self.dt_bias = dt_bias
        self.num_k_heads = num_k_heads
        self.head_k_dim = head_k_dim
        self.num_v_heads = num_v_heads
        self.head_v_dim = head_v_dim
        self.activation = activation

    # -- helpers ------------------------------------------------------------------------
    def _split_qkv(self, y: torch.Tensor):
        """post-conv ``[T, D]`` -> q,k ``[T, Hv, Dk]`` (GQA-expanded, L2-normed), v ``[T, Hv, Dv]``.

        Same operations in the same order as Megatron's ``_prepare_qkv_for_gated_delta_rule``.
        """
        T = y.shape[0]
        kd = self.num_k_heads * self.head_k_dim
        q, k, v = y[:, :kd].contiguous(), y[:, kd : 2 * kd].contiguous(), y[:, 2 * kd :].contiguous()
        q = q.view(T, self.num_k_heads, self.head_k_dim)
        k = k.view(T, self.num_k_heads, self.head_k_dim)
        rep = self.num_v_heads // self.num_k_heads
        q, k = gdn_l2norm(q), gdn_l2norm(k)
        if rep > 1:
            q = q.repeat_interleave(rep, dim=1)
            k = k.repeat_interleave(rep, dim=1)
        v = v.view(T, self.num_v_heads, self.head_v_dim)
        return q, k, v

    def _split_qkv_raw(self, y: torch.Tensor, a: torch.Tensor | None = None, b: torch.Tensor | None = None):
        """post-conv ``[T, D]`` -> RAW q,k ``[T, H, Dk]`` (compressed heads), v ``[T, HV, Dv]``.

        The native-kernel path: no l2norm and no GQA expansion, since both happen in-kernel.
        Passing ``a``/``b`` compacts them in the same launch and returns ``(q, k, v, a, b)``.
        """
        T = y.shape[0]
        kd = self.num_k_heads * self.head_k_dim
        want_ab = a is not None or b is not None

        from .gdn_fused_split import fused_split_enabled, fused_split_qkvab

        if fused_split_enabled() and y.is_cuda:
            vd = self.num_v_heads * self.head_v_dim
            got = fused_split_qkvab(y, kd, vd, a if want_ab else None, b if want_ab else None)
            if got is not None:
                fq, fk, fv, fa, fb = got
                fq = fq.view(T, self.num_k_heads, self.head_k_dim)
                fk = fk.view(T, self.num_k_heads, self.head_k_dim)
                fv = fv.view(T, self.num_v_heads, self.head_v_dim)
                return (fq, fk, fv, fa, fb) if want_ab else (fq, fk, fv)

        q = y[:, :kd].contiguous().view(T, self.num_k_heads, self.head_k_dim)
        k = y[:, kd : 2 * kd].contiguous().view(T, self.num_k_heads, self.head_k_dim)
        v = y[:, 2 * kd :].contiguous().view(T, self.num_v_heads, self.head_v_dim)
        if want_ab:
            return q, k, v, a.contiguous(), b.contiguous()
        return q, k, v

    def _gate_and_beta(self, a: torch.Tensor, b: torch.Tensor):
        """``g``/``beta`` gating, fused or eager. Identical bits either way."""
        return gdn_gate_and_beta(a, b, self.A_log, self.dt_bias)

    def _rows(self, slots: torch.Tensor) -> torch.Tensor:
        """Engine slot ids -> our rows, on the device. Unknown slots land on row 0 (the null row).

        Branch-free and host-free so it captures into a CUDA graph. The bounds clamp is load-bearing:
        replay padding carries ids that are not live blocks, and an unclamped index is a device-side
        assert that aborts the worker with no traceback.
        """
        if self._native:
            s = slots.long()
            nb = self.ssm_state.shape[0]
            # >= nb (stray/uninitialised padding) -> 0; then negatives (NULL_BLOCK_ID sentinel) -> 0.
            return torch.where(s < nb, s, torch.zeros_like(s)).clamp_(min=0)
        safe = slots.long().clamp_(0, self.slot2row.numel() - 1)
        return self.slot2row[safe]

    def _rows_pair(self, slots: torch.Tensor):
        """``(rows int64, rows int32)`` -- both widths decode callers need, from one launch if possible.

        Falls back to the bit-identical ATen chain; ``_native`` always keeps it, its clamp differing.
        """
        if not self._native:
            from .gdn_cpr_rows import cpr_resolve_rows, fused_rows_enabled

            if fused_rows_enabled() and slots.is_cuda:
                pair = cpr_resolve_rows(slots, self.slot2row)
                if pair is not None:
                    return pair
        rows = self._rows(slots)
        return rows, rows.to(torch.int32)

    def _assign(self, slot: int) -> int:
        """A row for a slot starting a fresh prefill. Evicts the true LRU row if none are free."""
        return self._assign_many((slot,))[0]

    def _ownership_indices(self) -> tuple[dict[int, int], set[int]]:
        """Exact inverse ownership/free indices, reconstructed and validated once for legacy objects."""

        smap = self._slot2row
        free = self._free
        owner = getattr(self, "_row2slot", None)
        free_set = getattr(self, "_free_set", None)
        if owner is None or free_set is None:
            owner = {}
            for slot, row in smap.items():
                prior = owner.setdefault(row, slot)
                if prior != slot:
                    raise RuntimeError(
                        f"[isoexec-gdn] recurrent row {row} is also mapped by slots {prior} and "
                        f"{slot}; two requests already alias one state row."
                    )
            free_set = set(free)
            if len(free_set) != len(free):
                raise RuntimeError("[isoexec-gdn] state free-list rows contain a duplicate")
            overlap = free_set.intersection(owner)
            if overlap:
                raise RuntimeError(
                    f"[isoexec-gdn] free-list rows {sorted(overlap)} are still mapped; allocating "
                    "one would alias two live recurrent states."
                )
            self._row2slot = owner
            self._free_set = free_set
        self._assert_ownership_index_shape(owner, free_set)
        return owner, free_set

    def _assert_ownership_index_shape(self, owner: dict[int, int], free_set: set[int]) -> None:
        """O(1) tripwire for a mutation that bypassed the maintained indices."""

        if len(owner) != len(self._slot2row):
            raise RuntimeError(
                "[isoexec-gdn] a recurrent row is also mapped, or the slot ownership map was "
                "mutated outside its lifecycle methods"
            )
        if len(free_set) != len(self._free):
            raise RuntimeError(
                "[isoexec-gdn] free-list rows contain a duplicate, or the free list was mutated "
                "outside its lifecycle methods"
            )

    def _assign_many(self, slots) -> list[int]:
        """Rows for a whole prefill batch -- same rows, same order as ``[_assign(s) for s in slots]``.

        Eviction candidates must be the K smallest ``last_used`` entries of ONE snapshot with claimed
        rows excluded; ``argmin`` re-run K times against a deferred clock would hand one row to two live
        requests. The stable sort on ``(last_used, row)`` matches argmin's first-minimum tie-break.
        """
        n = len(slots)
        if n == 0:
            return []
        smap = self._slot2row
        free = self._free
        owner, free_set = self._ownership_indices()
        limit = self.slot2row.numel()
        slot_ints = [int(s) for s in slots]
        bad_slots = [s for s in slot_ints if s < 0 or s >= limit]
        if bad_slots:
            raise RuntimeError(
                f"[isoexec-gdn] state slot {bad_slots[0]} exceeds the slot map ({limit}). "
                "Raise SKYRL_ISOEXEC_GDN_SLOT_MAP_SIZE. It cannot be grown: a realloc would move "
                "the tensor out from under a captured CUDA graph."
            )

        # A driver-managed pool has no legal eviction. Refuse the WHOLE batch before popping any
        # map/free entry, so exhaustion cannot leave a partially rebound pool behind.
        if getattr(self, "_driver_managed", False):
            missing = {s for s in slot_ints if s not in smap}
            if len(missing) > len(free):
                raise RuntimeError(
                    f"[isoexec-gdn] driver-managed state pool exhausted: capacity={self.capacity}, "
                    f"mapped={len(smap)}, free={len(free)}, missing={len(missing)} while assigning "
                    f"{n} prefill slots. Every mapped row is lifecycle-owned; evicting one could "
                    "steal a live or temporarily unscheduled request. The scheduler must release "
                    "finished/preempted requests before admitting another, and max_num_seqs must "
                    "cover its live concurrency."
                )

        rows: list[int] = []
        claimed: set[int] = set()
        lru_order: list[int] | None = None  # rows by (last_used, row); built at the FIRST eviction
        lru_at = 0

        for s in slot_ints:
            row = smap.get(s)
            if row is not None:
                indexed_slot = owner.get(row)
                if indexed_slot != s:
                    raise RuntimeError(
                        f"[isoexec-gdn] recurrent row {row} is also mapped or owned by slot "
                        f"{indexed_slot}, not the requested slot {s}"
                    )
                smap.pop(s)
                owner.pop(row)
            if row is None:
                if free:
                    row = free.pop()
                    if row not in free_set:
                        raise RuntimeError(f"[isoexec-gdn] free-list row {row} is absent from the maintained free set")
                    free_set.remove(row)
                    if row in owner:
                        raise RuntimeError(f"[isoexec-gdn] free-list row {row} is still mapped by slot {owner[row]}")
                else:
                    # The driver owns request lifetime exactly, so a full driver-managed pool has no
                    # legal victim; offline callers have no such lifecycle and keep the LRU fallback.
                    if getattr(self, "_driver_managed", False):
                        raise RuntimeError(
                            f"[isoexec-gdn] driver-managed state pool exhausted: capacity="
                            f"{self.capacity}, mapped={len(smap)}, free=0 while assigning {n} "
                            "prefill slots. Every mapped row is lifecycle-owned; evicting one "
                            "could steal a live or temporarily unscheduled request. The scheduler "
                            "must release finished/preempted requests before admitting another, "
                            "and max_num_seqs must cover its live concurrency."
                        )
                    if lru_order is None:
                        # No lifecycle oracle offline: one device-truth read for the whole batch.
                        lu = self.last_used.tolist()
                        _HOST_BOOKKEEPING_STATS["lru_device_fallback"] += 1
                        lru_order = sorted(range(1, len(lu)), key=lambda r: (lu[r], r))
                    while lru_at < len(lru_order) and lru_order[lru_at] in claimed:
                        lru_at += 1
                    if lru_at >= len(lru_order):
                        raise RuntimeError(
                            f"[isoexec-gdn] prefill batch of {n} slots exceeds the {len(lru_order)} "
                            "evictable state rows in this pool. Every row is already claimed by THIS "
                            "batch, so an eviction would hand one row to two live requests. The pool "
                            "is sized by max_num_seqs; the scheduler must not exceed it."
                        )
                    row = lru_order[lru_at]
                    lru_at += 1
                    victim = owner.pop(row, None)
                    if victim is not None:
                        smap.pop(victim, None)
            prior = owner.get(row)
            if prior is not None:
                raise RuntimeError(
                    f"[isoexec-gdn] recurrent row {row} is already owned by slot {prior}; assigning "
                    f"slot {s} would alias live states"
                )
            smap[s] = row
            owner[row] = s
            claimed.add(row)
            self._set_slot(s, row)
            rows.append(row)

        self.flush_slot_map()

        # One clock scatter. Dedup keeps the last tick per row: a slot repeated inside one batch would
        # otherwise be a duplicate index_put_, which torch leaves nondeterministic.
        ticks: dict[int, int] = {}
        for j, r in enumerate(rows):
            ticks[r] = j + 1
        m = len(ticks)
        buf = np.empty(2 * m, dtype=np.int64)
        buf[:m] = list(ticks.keys())
        buf[m:] = list(ticks.values())
        t = torch.from_numpy(buf).to(self.last_used.device, non_blocking=True)
        self.last_used[t[:m]] = self._clock + t[m:]
        self._clock += n
        self._assert_ownership_index_shape(owner, free_set)
        return rows

    def _set_slot(self, slot: int, row: int) -> None:
        """Record a slot->row edit on the HOST. The device tensor moves in ``flush_slot_map``."""
        self._slot_pending[int(slot)] = int(row)

    def flush_slot_map(self) -> int:
        """Push pending slot-map edits to the device in one scatter. Returns the number applied.

        Ordering is the contract: it must run OUTSIDE CUDA-graph capture (the scatter must not be
        captured, though later edits still land since the tensor is never reallocated) and BEFORE any
        read of ``slot2row`` -- a missed flush hands a decode lane another request's row, silently.
        """
        if self._native:  # native pools keep no slot map (rebind/release are no-ops there)
            return 0
        n = len(self._slot_pending)
        if n == 0:
            return 0
        keys, vals = self._pending_arrays()
        buf = np.empty(2 * n, dtype=np.int64)
        buf[:n] = keys
        buf[n:] = vals
        t = torch.from_numpy(buf).to(self.slot2row.device, non_blocking=True)
        self.slot2row[t[:n]] = t[n:]
        self._slot_pending.clear()
        return n

    def _pending_arrays(self):
        """(keys, values) of this layer's outbox as numpy, or None. Host-only."""
        n = len(self._slot_pending)
        if n == 0:
            return None
        return (
            np.fromiter(self._slot_pending.keys(), dtype=np.int64, count=n),
            np.fromiter(self._slot_pending.values(), dtype=np.int64, count=n),
        )

    def release_slot(self, slot: int) -> int | None:
        """Hand a finished request's row back to the free list; returns the freed row, or None.

        Host-side and never called under CUDA-graph capture.
        """
        if self._native:
            return None
        owner, free_set = self._ownership_indices()
        slot = int(slot)
        row = self._slot2row.get(slot)
        if row is None:
            return None
        indexed_slot = owner.get(row)
        if indexed_slot != slot:
            raise RuntimeError(
                f"[isoexec-gdn] cannot release slot {slot}: its row {row} is also mapped or owned "
                f"by slot {indexed_slot}. Freeing it would put a still-mapped recurrent state "
                "on the free list."
            )
        if row in free_set:
            raise RuntimeError(f"[isoexec-gdn] released row {row} is already on the free list")
        self._slot2row.pop(slot)
        owner.pop(row)
        if 0 <= slot < self.slot2row.numel():
            self._set_slot(slot, 0)
        self._free.append(row)
        free_set.add(row)
        self._on_row_released(row)
        self._assert_ownership_index_shape(owner, free_set)
        return row

    def require_row(self, slot: int) -> int:
        """The mapped row for a slot known to belong to a live request -- or a loud raise.

        The host half of the padding contract: the device path must fold unknown ids onto null row 0,
        which would silently share a state row if a live id were ever unmapped. A miss is unrecoverable.
        """
        row = self._slot2row.get(int(slot))
        if row is None:
            raise RuntimeError(
                f"[isoexec-gdn] LIVE state slot {int(slot)} has no mapped row (host map holds "
                f"{len(self._slot2row)} entries; device map {self.slot2row.numel()} slots; "
                f"capacity={self.capacity}). The device decode would silently fold this lane onto "
                "null row 0 (the CUDA-graph padding contract), i.e. share one state row across "
                "requests -- the 2026-08-12 v9 corruption class. Causes: a slot-id space the "
                "bookkeeping never mapped (multiple mamba KV-cache groups without the metadata "
                "alias -- see gdn_engine_patch.resolve_gdn_groups), a block id past the map "
                "(raise SKYRL_ISOEXEC_GDN_SLOT_MAP_SIZE), or a row evicted while its request lived."
            )
        return row

    def rebind_slot(self, old: int, new: int) -> None:
        """Move a live request's row from one engine slot id to another (mamba ALIGN mode).

        Only the name changes; the row and its state are untouched. Any stale mapping on ``new``
        belongs to a dead request (vLLM just allocated the block) and is released.
        """
        if self._native or int(old) == int(new):
            return
        owner, free_set = self._ownership_indices()
        old, new = int(old), int(new)
        row = self._slot2row.get(old)
        if row is None:
            return
        indexed_slot = owner.get(row)
        if indexed_slot != old:
            raise RuntimeError(f"[isoexec-gdn] cannot rebind slot {old}: row {row} is owned by {indexed_slot}")
        new_row = self._slot2row.get(new)
        if new_row is not None and owner.get(new_row) != new:
            raise RuntimeError(
                f"[isoexec-gdn] cannot rebind onto slot {new}: its row {new_row} is also mapped "
                f"or owned by slot {owner.get(new_row)}"
            )
        if row in free_set:
            raise RuntimeError(f"[isoexec-gdn] cannot rebind row {row}: it is already free")
        self._slot2row.pop(old)
        owner.pop(row)
        if 0 <= old < self.slot2row.numel():
            self._set_slot(old, 0)
        self.release_slot(new)
        if row in free_set or row in owner:
            raise RuntimeError(
                f"[isoexec-gdn] cannot rebind row {row}: it became free or re-owned while moving "
                f"slot {old} to {new}"
            )
        self._slot2row[new] = row
        owner[row] = new
        self._set_slot(new, row)
        self._assert_ownership_index_shape(owner, free_set)

    def _on_row_released(self, row: int) -> None:
        """Hook: drop any side state a subclass keeps for ``row``. No-op in recurrent mode."""

    # -- public API ---------------------------------------------------------------------
    @torch.no_grad()
    def _continuation_mask(self, slots_cpu, has_initial_state) -> list[bool]:
        """Which prefill rows resume carried state, and which start from zero.

        Slot-map membership AND vLLM's ``prefill_has_initial_state`` must agree; membership alone is a
        silent-corruption bug, since a reissued block id still looks like a continuation. A ``None``
        mask means no scheduler opinion (offline harnesses) and falls back to membership.
        """
        resume = chunked_prefill_enabled()
        held = [int(s) in self._slot2row for s in slots_cpu]
        if has_initial_state is None:
            return [resume and h for h in held]
        raw = has_initial_state if isinstance(has_initial_state, (list, tuple)) else has_initial_state.tolist()
        want = [bool(v) for v in raw]
        for i, (w, h) in enumerate(zip(want, held)):
            if w and not h:
                # A prefix-cache hit: APC reuses attention KV blocks, but this private pool is invisible
                # to vLLM's accounting, so the cached prefix's state was never computed.
                raise RuntimeError(
                    f"[isoexec-gdn] vLLM reports an initial state for prefill row {i} (slot "
                    f"{int(slots_cpu[i])}) but the private state pool holds none -- a prefix-cache "
                    "hit. The private-pool modes cannot resume a prefix they did not themselves "
                    "scan. Under cpr, set SKYRL_ISOEXEC_GDN_CPR_APC=1 so the boundary-state "
                    "store serves the hit (its admission clamp then guarantees the row is adopted "
                    "before this call); otherwise run the G4 native composition "
                    "(GDN_KERNEL=recurrent + GDN_NATIVE_KERNELS=1 + GDN_NATIVE_STATE=1)."
                )
        return [resume and w and h for w, h in zip(want, held)]

    def prefill(
        self,
        slots: torch.Tensor,  # [N] device, engine slot ids
        slots_cpu: list[int],
        x: torch.Tensor,  # [T, D] packed, pre-conv
        a: torch.Tensor,  # [T, Hv]
        b: torch.Tensor,  # [T, Hv]
        qsl: list[int],  # [N+1] token offsets into the packed prefill region
        has_initial_state: torch.Tensor | None = None,  # [N] device bool; native-kernels only
        conv_metadata=None,  # shared vLLM causal-conv metadata; None keeps vendor fallback
        prefill_query_start_loc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Prefill every prompt in this batch from position 0. Returns ``o [T, Hv, Dv]``.

        One varlen core launch for all N prompts; the conv still runs per sequence, because a width-4
        causal conv must not reach across a packed boundary.
        """
        N = len(slots_cpu)
        dev = x.device
        W = self.conv_weight.shape[-1]
        # Chunked-prefill resume: membership must be read BEFORE _assign mutates the map, and fresh and
        # continuation rows can mix in one batched call, so it is decided per row.
        resume = chunked_prefill_enabled()
        if self._native:
            # Native resume is metadata-driven; native storage without native kernels has no host slot
            # map to consult, so it refuses.
            if resume and not self._native_kernels:
                raise NotImplementedError(
                    "[isoexec-gdn] native mamba state + chunked-prefill resume needs "
                    "SKYRL_ISOEXEC_GDN_NATIVE_KERNELS=1 (metadata-driven resume)"
                )
            is_cont = [False] * N
            # The block id is the row: index vLLM's kv_cache directly.
            rows = [int(s) for s in slots_cpu]
            rows_t = slots.long()
            nb = self.ssm_state.shape[0]
            bad = [r for r in rows if r < 0 or r >= nb]
            if bad:
                raise RuntimeError(
                    f"[isoexec-gdn] NATIVE prefill: mamba block id(s) {bad[:8]} out of range for the "
                    f"ssm state tensor with {nb} rows (conv {self.conv_state.shape[0]}). The block-id "
                    "space equals num_gpu_blocks (measured), so in a correctly-bound run every real id "
                    "is in range. This fires when the layer holds a STALE kv_cache binding -- e.g. the "
                    "minimal graph-profiling cache vLLM binds before the real one -- which the "
                    "per-forward rebind check in the state layer should have replaced. Check for the "
                    "'NATIVE kv_cache rebound' log line."
                )
        else:
            is_cont = self._continuation_mask(slots_cpu, has_initial_state)

            # Prefill is the only way a request enters a row and is never captured, so all host-side
            # bookkeeping happens here, once for the whole batch.
            rows = self._assign_many(slots_cpu)
            rows_t = torch.tensor(rows, dtype=torch.long, device=dev)

        if self._native_kernels:
            # Native prefill: one varlen conv launch, then one varlen core launch. Fresh rows must be
            # zeroed first -- the core always loads column 0's row, and a recycled block holds old state.
            from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
                causal_conv1d_fn,
            )

            rows32 = rows_t.to(torch.int32)
            cu = prefill_query_start_loc
            if cu is None:
                cu = torch.tensor(qsl, dtype=torch.int32, device=dev)
            has = has_initial_state
            if has is None:
                has = torch.zeros(N, dtype=torch.bool, device=dev)
            y = causal_conv1d_fn(
                x.transpose(0, 1),
                self.conv_weight,
                self.conv_bias,
                activation=self.activation,
                conv_states=self.conv_state,
                has_initial_state=has,
                cache_indices=rows32,
                query_start_loc=cu,
                metadata=conv_metadata,
            ).transpose(0, 1)

            self.ssm_state[rows_t[~has]] = 0.0

            lens = [qsl[i + 1] - qsl[i] for i in range(N)]
            # Power-of-2 row stride: the kernel takes it as a tl.constexpr, so an unpadded allocation
            # would recompile for every distinct max prompt length.
            Tmax = max(lens)
            pad = max(64, 1 << (Tmax - 1).bit_length())
            idx_cpu = torch.zeros(N, pad, dtype=torch.int32)
            for i, (row, n) in enumerate(zip(rows, lens)):
                idx_cpu[i, 0] = row
                idx_cpu[i, n - 1] = row
            idx = idx_cpu.to(dev, non_blocking=True)[:, :Tmax]

            q, k, v = self._split_qkv_raw(y)
            o = gdn_native_core_kernel(
                q.unsqueeze(0),  # [1, T, H, Dk] varlen
                k.unsqueeze(0),
                v.unsqueeze(0),
                a,
                b,
                self.A_log,
                self.dt_bias,
                ssm_state=self.ssm_state,
                state_indices=idx,
                cu_seqlens=cu,
            )
            return o[0]

        ys, convs = [], []
        for i in range(N):
            s, e = qsl[i], qsl[i + 1]
            # Continuation: resume the conv window from this row's stored last W-1 pre-conv inputs.
            conv_init = self.conv_state[rows[i]] if is_cont[i] else None
            y, cs = gdn_causal_conv(
                x[s:e],
                self.conv_weight,
                self.conv_bias,
                initial_state=conv_init,
                activation=self.activation,
                return_final_state=True,
            )
            ys.append(y)
            convs.append(cs)

        y = torch.cat(ys, dim=0) if N > 1 else ys[0]
        q, k, v = self._split_qkv(y)
        g, beta = self._gate_and_beta(a, b)

        lens = [qsl[i + 1] - qsl[i] for i in range(N)]
        cu = torch.tensor(qsl, dtype=torch.int32, device=dev)

        # Store once per sequence at its last token; other columns are 0 (skipped), but column 0 must
        # carry a real row. The power-of-2 row stride keeps the tl.constexpr stride from recompiling.
        Tmax = max(lens)
        _pad = max(64, 1 << (Tmax - 1).bit_length())
        idx_cpu = torch.zeros(N, _pad, dtype=torch.int32)
        for i, (row, n) in enumerate(zip(rows, lens)):
            idx_cpu[i, 0] = row
            idx_cpu[i, n - 1] = row
        idx = idx_cpu.to(dev, non_blocking=True)[:, :Tmax]

        # Zero only the fresh rows: a continuation row must keep the state the previous chunk left
        # there, which the kernel loads as the initial state at column 0 of `idx`.
        fresh = [r for r, c in zip(rows, is_cont) if not c]
        if fresh:
            self.ssm_state[torch.tensor(fresh, dtype=torch.long, device=dev)] = 0.0

        o = gdn_recurrent_kernel(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            g.unsqueeze(0),
            beta.unsqueeze(0),
            ssm_state=self.ssm_state,
            state_indices=idx,
            cu_seqlens=cu,
        )

        # conv state = the last W-1 pre-conv inputs, left-zero-padded for a shorter prompt, which is
        # exactly what gdn_causal_conv's final_state already is.
        if W - 1 != self.conv_state.shape[-1]:  # pragma: no cover - shape contract
            raise RuntimeError(f"[isoexec-gdn] conv state width {self.conv_state.shape[-1]} != {W - 1}")
        self.conv_state[rows_t] = torch.stack(convs, dim=0).to(self.conv_state.dtype)
        return o[0]

    @torch.no_grad()
    def decode(
        self,
        slots: torch.Tensor,  # [N] device, engine slot ids
        x: torch.Tensor,  # [N, D] the new token, pre-conv
        a: torch.Tensor,  # [N, Hv]
        b: torch.Tensor,  # [N, Hv]
    ) -> torch.Tensor:
        """One decode step for N requests. Returns ``o [N, Hv, Dv]``.

        Host-free with no data-dependent shapes, so it captures into a CUDA graph as-is. Padded replay
        lanes fold onto row 0 and are skipped. Every torch index here must be in-bounds by construction:
        an out-of-bounds one is a device-side assert that aborts the worker with no python traceback.
        """
        N = x.shape[0]
        rows, rows32 = self._rows_pair(slots)

        if self._native_kernels:
            # Native decode, matching vLLM's non-spec decode: an in-place conv window slide plus the
            # fused core. Host-free and shape-static; padded lanes fold to row 0, which both ignore.
            from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
                causal_conv1d_update,
            )

            y = causal_conv1d_update(
                x,
                self.conv_state,
                self.conv_weight,
                self.conv_bias,
                self.activation,
                conv_state_indices=rows32,
            )
            q, k, v = self._split_qkv_raw(y)
            o = gdn_native_core_kernel(
                q.unsqueeze(1),  # [N, T=1, H, Dk]
                k.unsqueeze(1),
                v.unsqueeze(1),
                a,
                b,
                self.A_log,
                self.dt_bias,
                ssm_state=self.ssm_state,
                state_indices=rows32,
                cu_seqlens=None,
            )
            return o[:, 0]

        cs = self.conv_state[rows]  # [N, D, W-1]
        y = gdn_causal_conv_batched(
            x.unsqueeze(1),  # [N, 1, D]
            self.conv_weight,
            self.conv_bias,
            initial_state=cs,
            activation=self.activation,
        )
        q, k, v = self._split_qkv(y.reshape(N, -1))
        g, beta = self._gate_and_beta(a, b)

        o = gdn_recurrent_kernel(
            q.unsqueeze(1),  # [N, T=1, Hv, Dk]
            k.unsqueeze(1),
            v.unsqueeze(1),
            g.unsqueeze(1),
            beta.unsqueeze(1),
            ssm_state=self.ssm_state,
            state_indices=rows.to(torch.int32),  # one token per sequence, so the row IS the index map
            cu_seqlens=None,
        )

        # Slide the conv window: drop the oldest pre-conv input, append this token.
        self.conv_state[rows] = torch.cat([cs[..., 1:], x.unsqueeze(-1).to(cs.dtype)], dim=-1)
        if not self._native:
            # LRU bookkeeping for the private pool only; native mode has no clock to update.
            self._clock += 1
            self.last_used[rows] = self._clock  # GPU->GPU; keeps _assign's LRU true
        return o[:, 0]
