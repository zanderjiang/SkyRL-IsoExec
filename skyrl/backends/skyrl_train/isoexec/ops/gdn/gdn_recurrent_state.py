"""Recurrent GDN state and engine prefill/decode implementation.

The recurrent core advances one token at a time with FP32 state and fixed launch geometry. Private
mode maps scheduler slot ids onto a bounded state pool with row zero reserved for padded graph
lanes. Native-state mode binds vLLM's Mamba state tensors directly and uses metadata-provided block
rows. Chunk-synced mode subclasses this machinery and adds open buffers plus boundary resync.

The causal-convolution state stores the preceding ``W-1`` pre-convolution inputs. Chunked prefill
can resume only when the scheduler metadata or the private slot map proves that carried state is
present; otherwise the implementation fails closed rather than silently starting from zero.
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

# Entries in the slot->row map. Must exceed the engine's state-slot id space; sized once per layer
# object at construction, because a captured CUDA graph holds this tensor's address and a realloc
# would move it out from under a replay. Read at construction time, not import time, so a value set
# between import and engine build is honoured.
_SLOT_MAP_ENV = "SKYRL_ISOEXEC_GDN_SLOT_MAP_SIZE"


def _slot_map_size() -> int:
    v = os.environ.get(_SLOT_MAP_ENV)
    if v:
        return int(v)
    # CPR_MIN_PAGES shrinks the page size, multiplying vLLM's block-id space; every id the mamba
    # group hands a live request must still fit this map (out-of-range live ids raise in _assign).
    from .gdn_ops import gdn_cpr_min_pages

    return (1 << 20) if gdn_cpr_min_pages() else 65536


# Chunked-prefill resume, default off. vLLM may split one prompt across several forward passes; chunk
# 2+ must then resume from chunk 1's ending state rather than restart from zero. The signal is a slot
# already present in ``self._slot2row`` before ``_assign``. Off means every prompt is treated as fresh
# (state zeroed, conv restarted) regardless of re-presented slots.
_CHUNKED_PREFILL_ENV = "SKYRL_ISOEXEC_GDN_CHUNKED_PREFILL"


def chunked_prefill_enabled() -> bool:
    """True iff continuation chunks of a prefill should resume from the prompt's carried state."""
    return os.environ.get(_CHUNKED_PREFILL_ENV, "0").lower() not in ("", "0", "false", "no")


# State ownership. Off (default): the private ``max_num_seqs``-sized pool below, indexed by a
# slot->row map. On: the recurrent state lives in vLLM's own mamba ``kv_cache`` blocks, indexed
# directly by vLLM's block id, which makes every block id in-bounds by construction (all hybrid KV
# groups share one ``num_blocks``) and lets ``NULL_BLOCK_ID=0`` absorb padded/unscheduled lanes that
# the recurrent kernel already skips. The compute is identical either way; only where the state lives
# moves. Read at call time so it can be toggled per process.
_NATIVE_STATE_ENV = "SKYRL_ISOEXEC_GDN_NATIVE_STATE"


def native_state_enabled() -> bool:
    """True iff the recurrent GDN state should live in vLLM's ``kv_cache`` blocks, not our pool."""
    return os.environ.get(_NATIVE_STATE_ENV, "0").lower() not in ("", "0", "false", "no")


# Counters for the two host reads that would otherwise synchronize every GDN layer in an engine
# prefill: the LRU snapshot (only when the private pool is full) and cpr's position gather.
# The position counters are bumped in gdn_cpr_state.
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

    ``ssm_state`` / ``conv_state`` (both or neither) hand the layer vLLM's own mamba ``kv_cache``
    tensors. When given, no private pool is allocated and the block id indexes them directly.
    ``conv_state`` must already be oriented ``(num_blocks, D, W-1)`` (the caller applies vLLM's SD/DS
    transpose); ``ssm_state`` must be fp32.
    """
    return RecurrentGDN(capacity=max_num_seqs, ssm_state=ssm_state, conv_state=conv_state, **kw)


def flush_slot_maps(layers) -> int:
    """Flush every layer's slot-map outbox with one host->device upload for the whole stack.

    Per-layer flushing costs three device ops per layer; concatenating every layer's edits makes the
    floor two H2D uploads plus one scatter per layer. The per-layer outboxes are deliberately not
    assumed equal -- ``_assign`` flushes on its own schedule, and a wrong shared key set would point a
    slot at another request's row -- so the concatenation is exact for any per-layer contents.
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

    vLLM's state slot ids cannot index a private state tensor directly: for a hybrid model it pads the
    mamba page up to the attention page, so the block-id space is sized by the shared KV pool rather
    than by the state tensor, and raw slot ids run past the pool's row count (a device-side assert that
    aborts the worker). The state is therefore addressed by our own row index, with a slot map doing the
    translation:

      * ``capacity`` rows back at most ``max_num_seqs`` concurrently-running requests, however many
        state slots vLLM has minted.
      * Row 0 is the null row and is never handed to a request. Every slot the map does not know
        resolves to it, including the padded lanes of a CUDA-graph replay. The recurrent kernel skips
        any lane whose state index is ``<= 0``, so a padded lane computes nothing and the conv
        gather/scatter touches row-0 garbage that no live request owns.
      * The map is a device tensor, so decode reads it on the GPU and never touches the host, which is
        what lets a pure-decode step capture into a CUDA graph.
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

        # Native state: the state tensors are vLLM's own mamba kv_cache blocks, indexed directly by
        # the engine's block id. No private pool, slot->row map, LRU/free-list or device clock --
        # vLLM owns allocation and eviction, and _rows is the identity on this path.
        self._native = ssm_state is not None or conv_state is not None
        if self._native:
            if ssm_state is None or conv_state is None:
                raise ValueError("[isoexec-gdn] native state needs BOTH ssm_state and conv_state (vLLM's kv_cache)")
            # An fp32 ssm cache is mandatory: the default `mamba_ssm_cache_dtype=auto` resolves to
            # bf16, which rounds the state on every read/write and breaks the fp32 prefill->decode
            # round-trip the recurrent design rests on.
            if ssm_state.dtype != torch.float32:
                raise RuntimeError(
                    f"[isoexec-gdn] native mamba ssm cache resolved to {ssm_state.dtype}, not float32. "
                    "Set mamba_ssm_cache_dtype=float32 at engine build -- bf16 rounds the recurrent "
                    "state every step and breaks the fp32 prefill->decode round-trip."
                )
            # conv_state arrives already oriented (num_blocks, D, W-1); the caller applies vLLM's
            # SD/DS transpose. The resulting non-contiguous view is fine: the fused conv-decode kernel
            # is stride-aware and the prefill write/read are advanced-index ops that honor strides.
            if conv_state.shape[-1] != W - 1:
                raise RuntimeError(
                    f"[isoexec-gdn] native conv_state width {conv_state.shape[-1]} != W-1 = {W - 1}; "
                    "the (num_blocks, D, W-1) orientation is wrong (SD/DS transpose)."
                )
            self.ssm_state = ssm_state
            self.conv_state = conv_state
            # Run vLLM's native fused kernels (conv fn/update + fused_sigmoid_gating core with
            # in-kernel l2norm/gating). Read once at build; the state layer rebuilds this object on any
            # kv_cache rebind, so the flag is re-read then. Native-state only.
            self._native_kernels = gdn_native_kernels_enabled()
        else:
            self._native_kernels = False
            rows = capacity + 1  # row 0 = the null row; requests live at 1..capacity

            # fp32, not bf16: prefill->decode chaining rests on the state surviving a round trip
            # through memory unchanged. Training runs one continuous scan with the state in fp32
            # registers; the rollout writes it out at end of prefill and reloads it per decoded token,
            # and bf16 would round it at each of those boundaries.
            self.ssm_state = torch.zeros(rows, num_v_heads, head_v_dim, head_k_dim, dtype=torch.float32, device=device)
            self.conv_state = torch.zeros(rows, D, W - 1, dtype=dtype, device=device)

            # Engine slot id -> our row. Sized once and never reallocated: a captured graph reads this
            # tensor's address on every replay. Unknown slots (never prefilled, NULL, out of range) map
            # to row 0, which is what the padded lanes of a replay need. ``slot_map_hint`` is the
            # caller's measurement of vLLM's actual block-id space, so the map covers it by
            # construction rather than by the env default's guess; ``_assign`` still raises per-slot.
            self.slot2row = torch.zeros(max(_slot_map_size(), int(slot_map_hint)), dtype=torch.long, device=device)
            self.last_used = torch.zeros(rows, dtype=torch.long, device=device)
            # A device clock: `last_used[rows] = <python int>` would stage the scalar through pageable
            # host memory -- a sync, and uncapturable.
            self._clock = torch.zeros((), dtype=torch.long, device=device)

            self._slot2row: OrderedDict[int, int] = OrderedDict()  # host mirror, prefill-only
            self._free: list[int] = list(range(1, rows))  # row 0 is never handed out
            # Exact inverse indices for the two ownership collections above. Lifecycle mutations go
            # through _assign_many/release_slot/rebind_slot, which update both views together, making
            # alias/free-list validation O(1). `_ownership_indices` lazily reconstructs them for
            # callers that never populated them.
            self._row2slot: dict[int, int] = {}
            self._free_set: set[int] = set(self._free)

            # Batched slot-map writes. ``slot2row[i] = v`` on a CUDA tensor stages one scalar through
            # pageable host memory, and under mamba ALIGN mode (forced on whenever prefix caching is
            # enabled) a request's state block rotates every block_size tokens, so the driver rebinds
            # every layer of every rotating request each decode step. Edits instead land in a numpy
            # mirror and reach the device in one scatter per layer per step (``flush_slot_map``), fired
            # by the driver in its pre-forward host window. ``_slot2row`` stays the authoritative host
            # view; this is only the outbox.
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

        Same operations in the same order as Megatron's ``_prepare_qkv_for_gated_delta_rule``: the two
        runtimes must round identically, and the GQA expansion happens before the kernel because that
        is where Megatron does it.
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

        The native-kernel path: no l2norm, no GQA expansion -- both happen IN-KERNEL
        (``fused_sigmoid_gating_delta_rule_update`` maps ``i_h = i_hv // (HV // H)`` and
        rsqrt-normalises on the fp32 upcast), exactly as native vLLM's ``_forward_core`` does.

        WITH ``a``/``b`` (the gptmodel's ``alpha[0]``/``beta[0]`` views, possibly strided) the same
        launch also compacts those, and the return is ``(q, k, v, a, b)`` -- five contiguous outputs
        from one kernel instead of five ATen copies. See ``gdn_fused_split``: the copies are
        irremovable (the vendor core takes no strides), the LAUNCHES are not. Without them the
        return is the historical ``(q, k, v)``.
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

        Branch-free and host-free so it captures into a CUDA graph. The bounds clamp keeps a stray id
        from walking off the map: graph replays pad the batch with slot ids that are not live blocks,
        and vLLM's block-id space is wider than any tensor we own, so an unclamped index becomes a
        device-side assert that aborts the worker with no traceback.

        Under native state the block id is the row, but padded replay lanes still carry sentinels
        (NULL_BLOCK_ID, or ids that are not live blocks during capture), so out-of-range lanes fold onto
        block 0: vLLM reserves it as its null block, the recurrent kernel skips a lane whose state index
        is <= 0, and the conv slide touches only null garbage.
        """
        if self._native:
            s = slots.long()
            nb = self.ssm_state.shape[0]
            # >= nb (stray/uninitialised padding) -> 0; then negatives (NULL_BLOCK_ID sentinel) -> 0.
            return torch.where(s < nb, s, torch.zeros_like(s)).clamp_(min=0)
        safe = slots.long().clamp_(0, self.slot2row.numel() - 1)
        return self.slot2row[safe]

    def _rows_pair(self, slots: torch.Tensor):
        """``(rows int64, rows int32)`` -- both widths every decode caller needs.

        The captured decode graph's consumers of ``rows`` want int32 (``causal_conv1d_update``'s
        ``conv_state_indices``, the GDN core's ``state_indices``) while the buffer scatter and the
        ``last_used`` write want int64, so the ATen composition is five single-block kernels per layer.
        ``gdn_cpr_rows`` does the whole chain in one program and emits both widths; this method declines
        to the bit-identical ATen chain when that kernel is off or the shapes are unexpected. The
        ``_native`` branch always keeps the ATen form -- its clamp is a different function.
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
        """A row for a slot starting a fresh prefill. Evicts the true LRU row if none are free.

        The single-slot door onto :meth:`_assign_many`, where the bookkeeping lives. For one slot the
        two are the same three device ops in the same order, preserving the write-through contract that
        single-slot callers depend on.
        """
        return self._assign_many((slot,))[0]

    def _ownership_indices(self) -> tuple[dict[int, int], set[int]]:
        """Exact inverse ownership/free indices, reconstructed once for legacy objects.

        Objects built through ``__init__`` create both indices with the pool; the lazy branch runs the
        full validation once for objects that did not, after which every mutation maintains the
        invariant incrementally.
        """

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

        The batch is served with one ``flush_slot_map``, one host read of ``last_used`` and one
        ``last_used`` scatter plus clock bump, instead of one of each per slot. Two of those hoists have
        ordering requirements:

        * Batching the flush is safe because the device map is not read again until decode, which runs
          before prefill within a mixed batch and behind the driver's own ``flush_slot_maps`` between
          steps. The outbox is keyed by slot, so an assignment still overwrites a rebind staged earlier
          in the same step.
        * The eviction candidates must be the K smallest ``last_used`` entries of one snapshot with
          every already-claimed row excluded -- not ``argmin`` re-run K times, which against a deferred
          clock update would hand the same row to two live requests. The order is a stable sort on
          ``(last_used, row)``, matching ``argmin``'s first-minimum tie-break, and the clock is monotone
          so a row bumped by this batch always sorts after one that was not.

        Slots, rows and clock ticks are int64 bookkeeping read only as indices; no kernel reads them as
        values, so nothing here can round.
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

        # A driver-managed pool has exact release ownership and therefore no legal eviction.
        # Refuse the WHOLE batch before popping a map/free entry so exhaustion cannot leave a
        # partially rebound pool behind.
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
                    # The CPR driver owns request lifetime exactly (finished and preempted
                    # rows return to `_free`, unscheduled live requests keep their mappings), so a full
                    # driver-managed pool has no legal victim and LRU eviction would steal an
                    # unscheduled live request's row. Offline callers have no such lifecycle and keep
                    # the LRU fallback below.
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
                        # Offline recurrent use has no scheduler lifecycle oracle: take one
                        # device-truth read for the whole prefill batch rather than a host mirror.
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

        Ordering is the contract:

        * It must run outside CUDA-graph capture. The captured decode graphs read ``slot2row`` by
          address on every replay, so an edit staged later still lands (the tensor is never
          reallocated), but the scatter itself must not be captured. The driver's pre-forward host
          window is that place.
        * It must run before any read of ``slot2row``. A missed flush hands a decode lane another
          request's row -- silent, not an error. Every read path either follows the driver's flush or
          flushes itself (``_assign_many``, once per prefill batch, outside capture).

        Idempotent and free when nothing is pending. Keys and values ride one numpy buffer so the edit
        crosses as a single ``non_blocking`` HtoD rather than two pageable copies with two stream syncs.
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
        """Hand a finished request's row back to the free list. Returns the freed row, or None.

        Without this, LRU eviction is the only thing that ever releases a row, and a recycled block id
        looks like a live continuation. Once the driver sees a request finish, the row goes back and the
        block id maps to nothing. Host-side and never called under CUDA-graph capture.
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

        The host half of the padding contract. The device path (``_rows``) must fold unknown ids onto
        null row 0, because on the device a graph-replay padding lane is indistinguishable from a real
        id -- but that fold would silently share one state row across requests if a live id were ever
        unmapped. The host can tell the difference: an id taken from the scheduler's own batch is never
        padding, so the driver passes every such id through here before the forward and a miss raises.
        A miss is unrecoverable (a mid-sequence state cannot be rebuilt), so there is no soft path.
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

        Under align mode vLLM rotates a request's state block every ``block_size`` tokens, so the
        slot id in the metadata is not stable for a request's lifetime the way it is under cache
        mode "none". The request, its row and its state are unchanged -- only the name changes.
        Any stale mapping on ``new`` belongs to a dead request (vLLM just allocated the block) and
        is released.
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

        Two independent facts must agree, and neither alone is sufficient:

        * ``slot in self._slot2row`` -- we still hold a row for this slot, whose carried conv/ssm (and,
          under cpr, entry state and open-chunk buffers) the resume reads.
        * vLLM's ``prefill_has_initial_state[i]`` -- the scheduler says this request continues a prefill
          it already started; only the scheduler knows a request's ``num_computed_tokens``.

        Membership alone is a silent-corruption bug: entries die only by LRU eviction, so when vLLM
        reissues a finished request's block id to a new request, membership still says "continuation"
        and the new prompt resumes a dead request's state.

        ``has_initial_state=None`` means no scheduler opinion was supplied (the offline harnesses drive
        continuations by slot map on purpose) and falls back to membership. It may arrive as the device
        mask or as a host sequence of bools; prefer the host form, since reading the device mask here is
        a D2H sync inside every GDN layer's prefill. The values are the same either way.
        """
        resume = chunked_prefill_enabled()
        held = [int(s) in self._slot2row for s in slots_cpu]
        if has_initial_state is None:
            return [resume and h for h in held]
        raw = has_initial_state if isinstance(has_initial_state, (list, tuple)) else has_initial_state.tolist()
        want = [bool(v) for v in raw]
        for i, (w, h) in enumerate(zip(want, held)):
            if w and not h:
                # vLLM wants a mid-sequence resume but we hold no state for that slot: a prefix-cache
                # hit. APC reuses attention KV blocks, while this private GDN pool is invisible to
                # vLLM's cache accounting, so the state for the cached prefix was never computed.
                # Unservable, and starting from zero would silently drift.
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

        One varlen kernel launch for all N prompts: the recurrent grid is ``(1, NV, N*HV)``, so a single
        prompt occupies only a handful of programs, and batching the prompts is the kernel's only
        parallelism at prefill. The conv still runs per sequence, because a width-4 causal conv must not
        reach across a packed boundary.
        """
        N = len(slots_cpu)
        dev = x.device
        W = self.conv_weight.shape[-1]
        # Chunked-prefill resume: a slot already in the map before _assign is a continuation chunk of a
        # prompt vLLM split across passes, and _assign hands it its existing row back, so it resumes
        # from that row's carried ssm/conv state. Membership must be captured before _assign mutates the
        # map. Fresh and continuation rows can mix in one batched call, so this is decided per row.
        resume = chunked_prefill_enabled()
        if self._native:
            # Under native kernels, resume is metadata-driven: ``has_initial_state`` marks the
            # continuation/APC-hit rows and the vLLM conv + core kernels load the carried state from the
            # block rows. The env resume flag only applies to the private-pool path; native storage
            # without native kernels refuses, having no host slot map to consult.
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

            # Claim a row per prompt. Prefill is the only way a request enters a row and is never
            # captured, so all host-side bookkeeping (LRU, free list, device map) happens here, once
            # for the whole batch.
            rows = self._assign_many(slots_cpu)
            rows_t = torch.tensor(rows, dtype=torch.long, device=dev)

        if self._native_kernels:
            # Native prefill: vLLM's prefill shape with the chunk kernel swapped for the same fused
            # recurrent core decode uses. One varlen conv launch for all N prompts (kernel-side
            # per-sequence boundaries and state read/write at the block rows), then one varlen core
            # launch. ``has_initial_state`` rows resume from the carried conv/ssm state; fresh rows are
            # zeroed first, because the core always loads the row named in column 0 of the grid and a
            # recycled block may hold a dead request's state.
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
            # Power-of-2-padded row stride: the kernel takes the grid's row stride as a tl.constexpr,
            # so a [N, max(lens)] allocation would recompile for every distinct max prompt length.
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
            # Continuation: resume the width-W conv window from this row's stored last W-1 pre-conv
            # inputs (bitwise, since final_state is the raw inputs). Fresh rows start at 0.
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

        # Store the state once per sequence, at its last token; every other column is 0, which the
        # kernel reads as a null index and skips. Column 0 must still carry a real row: it is what the
        # initial-state load reads, and a lane whose column 0 is <= 0 computes nothing. Rows are >= 1 by
        # construction, since row 0 is the null row and _assign never hands it out.
        # Power-of-2-padded row stride: the kernel takes the grid's row stride as a tl.constexpr, so a
        # [N, max(lens)] allocation would recompile for every distinct max length.
        Tmax = max(lens)
        _pad = max(64, 1 << (Tmax - 1).bit_length())
        idx_cpu = torch.zeros(N, _pad, dtype=torch.int32)
        for i, (row, n) in enumerate(zip(rows, lens)):
            idx_cpu[i, 0] = row
            idx_cpu[i, n - 1] = row
        idx = idx_cpu.to(dev, non_blocking=True)[:, :Tmax]

        # Zero only the fresh rows: a fresh prompt inherits nothing and its row may be recycled, while
        # a continuation row must keep the state the previous chunk left there, which the kernel loads
        # as the initial state at column 0 of `idx`.
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

        Reads nothing back to the host and has no data-dependent shapes, so it captures into a CUDA
        graph as-is; a recurrent step is just one token and one state update.

        Padded lanes: a graph replay runs the captured kernels on a batch padded out to the capture
        width, whose state-index entries are not live requests (NULL_BLOCK_ID, or non-live ids during
        capture). ``_rows`` folds all of them onto row 0, so the kernel skips them (it returns for any
        state index <= 0) and the conv gather/scatter touches row-0 garbage no live request owns. Every
        torch index on this path must be in-bounds by construction: an out-of-bounds one is a
        device-side assert that aborts the worker with no python traceback.
        """
        N = x.shape[0]
        rows, rows32 = self._rows_pair(slots)

        if self._native_kernels:
            # Native decode, matching vLLM's non-spec decode: causal_conv1d_update slides the conv
            # window in place at the block rows, and the fused core does l2norm, GQA mapping, fp32
            # gating and the state update in one kernel against vLLM's own state. Host-free and
            # shape-static; padded replay lanes fold to row 0, which both kernels ignore.
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
            # LRU bookkeeping for the private pool only: in native mode vLLM owns allocation and
            # eviction, so there is no clock or last_used to update.
            self._clock += 1
            self.last_used[rows] = self._clock  # GPU->GPU; keeps _assign's LRU true
        return o[:, 0]
