"""Engine state for the canonical CPR GDN function.

Decode advances the fused recurrent core one token at a time and records the prepared inputs in an
open-chunk buffer. At each chunk boundary the FLA state pass recomputes the boundary from the fp32
entry state: the running recurrent state receives the bf16 snapshot upcast to fp32, while the next
entry retains the fp32 final state. That rounding order is shared with the trainer forward, and prefill
and chunked-prefill continuation preserve the same running state, boundary entry and open tail.
"""

from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np
import torch

from .gdn_cpr import chunk_boundary_states
from .gdn_ops import gdn_causal_conv, gdn_recurrent_kernel
from .gdn_recurrent_state import _HOST_BOOKKEEPING_STATS, RecurrentGDN


def _dev_idx(vals, dtype, dev) -> torch.Tensor:
    """Python sequence -> device index tensor, without ``torch.tensor(list, device=cuda)``.

    ``torch.tensor(<python list>, device=cuda)`` walks the list element by element through the Python
    C-API to build a CPU tensor before copying; going through numpy skips that walk. These are integer
    index tensors, so there is no arithmetic here that could round differently.
    """
    npdt = {torch.long: np.int64, torch.int64: np.int64, torch.int32: np.int32, torch.bool: np.bool_}[dtype]
    return torch.from_numpy(np.asarray(vals, dtype=npdt)).to(dev, non_blocking=True)


def entry_read(layer: "CprGDN", rows) -> torch.Tensor:
    """Rows of ``layer``'s fp32 entry state as a device ``[M, HV, V, K]`` tensor."""
    return layer.entry_state[_dev_idx(rows, torch.long, layer.ssm_state.device)]


def entry_write(layer: "CprGDN", rows, vals: torch.Tensor) -> None:
    """Write ``vals`` ``[M, HV, V, K]`` fp32 into ``layer``'s entry state at ``rows``."""
    layer.entry_state[_dev_idx(rows, torch.long, layer.ssm_state.device)] = vals


def entry_write_one(layer: "CprGDN", row: int, val: torch.Tensor) -> None:
    """Single-row write (the eager prefill's per-sequence handoff, CPR-APC adoption)."""
    layer.entry_state[row] = val


def entry_zero(layer: "CprGDN", rows) -> None:
    """Zero ``rows`` -- a fresh prompt's entry state, which must be the trainer's zero h[0]."""
    layer.entry_state[_dev_idx(rows, torch.long, layer.ssm_state.device)] = 0.0


def entry_read_stacked(layers: list["CprGDN"], rows, rows_t: torch.Tensor) -> torch.Tensor:
    """``cat([ly.entry_state[rows] for ly in layers])`` -- layer-major ``[L*M, HV, V, K]`` fp32."""
    return torch.cat([ly.entry_state[rows_t] for ly in layers], dim=0)


def entry_write_stacked(layers: list["CprGDN"], rows, rows_t: torch.Tensor, final: torch.Tensor) -> None:
    """Write a layer-major ``[L*M, HV, V, K]`` fp32 result back to every layer's entry state."""
    M = len(rows)
    for j, ly in enumerate(layers):
        ly.entry_state[rows_t] = final[j * M : (j + 1) * M]


# Every live CprGDN in this process, in construction (layer) order. The lazy-resync driver
# iterates this once per decode step, on the host and before the forward, which is what lets decode()
# itself capture into a CUDA graph.
CPR_LAYERS: list["CprGDN"] = []


# APC (automatic prefix caching) under cpr: the boundary-state cache.
#
# At an absolute position that is an exact multiple of C, a slot's state collapses to two tensors. The
# open-chunk buffer is empty (pos % C == 0), and the running scan state is a pure function of the fp32
# entry state (`ssm = bf16(entry).fp32`, what both `_resync` and prefill's boundary-ender branch write),
# so the whole state is:
#
#     entry_state  fp32 [HV, V, K]      the fp32 chain value at that boundary
#     conv_state   bf16 [D, W-1]        the last W-1 pre-conv inputs
#
# That collapse is what makes an APC hit servable: a hit becomes indistinguishable from a
# chunked-prefill continuation whose earlier chunks were scanned by a different request, and prefill is
# bitwise prefix-invariant, so the same prefix is the same bits.
#
# The store is byte-capped and LRU-evicting -- one checkpoint is L_gdn * (HV*V*K*4 + D*(W-1)*2) bytes,
# and two are kept per prompt. A miss is never a correctness event, only a smaller hit.
_CPR_APC_ENV = "SKYRL_ISOEXEC_GDN_CPR_APC"
_CPR_APC_MB_ENV = "SKYRL_ISOEXEC_GDN_CPR_APC_MB"


def cpr_apc_enabled() -> bool:
    """True iff cpr should serve vLLM prefix-cache hits from the boundary-state store.

    Read at call time like every other mode flag. Default off, in which case `_continuation_mask` keeps
    its loud raise on a cache hit and the engine refuses `enable_prefix_caching` outright.
    """
    return os.environ.get(_CPR_APC_ENV, "0").lower() not in ("", "0", "false", "no")


def cpr_apc_shared_mode() -> bool:
    """True iff CPR-APC runs in SHARED-INDEX mode (``SKYRL_ISOEXEC_GDN_CPR_APC`` in {2, shm, shared}).

    Mode "1" is the in-process design: admission reads the store directly, which is legal only at
    world_size == 1. Shared mode keeps the store and adoption per-worker and adds a scheduler-readable
    membership mirror in /dev/shm, which makes admission possible when the scheduler's process holds no
    model. ``cpr_apc_enabled()`` is true for both: only where admission reads membership differs. In
    shared mode the store is also append-only within an epoch (``put`` refuses when full) so the
    published index can never advertise an evicted checkpoint.
    """
    return os.environ.get(_CPR_APC_ENV, "0").strip().lower() in ("2", "shm", "shared")


def cpr_apc_budget_bytes() -> int:
    """Device-memory ceiling for the boundary-state store. Default 1 GiB."""
    return int(float(os.environ.get(_CPR_APC_MB_ENV, "1024") or 1024) * 2**20)


# Fired by ``CprBoundaryStore.clear()``. Shared mode registers the shm publisher's ``invalidate`` here
# (rank 0 only), so any wholesale drop of the local store retracts every published entry before the
# scheduler could admit against it. Exceptions propagate: a failed retraction while the index still
# advertises must never pass silently.
CPR_APC_ON_CLEAR: list = []


class CprBoundaryStore:
    """Prefix-keyed LRU of chunk-boundary GDN checkpoints, stacked across ALL GDN layers.

    Key: an opaque bytes digest of the token prefix ending at the checkpoint's position, built by the
    driver. Value: one (entry, conv) pair per layer, stacked on dim 0 so an adopt is one indexed write
    per layer rather than a python loop over layers.

    The store's lifetime is tied to vLLM's own prefix cache: a weight sync makes every cached state
    stale, so the engine patch wraps `KVCacheManager.reset_prefix_cache` to `clear()` this too. If this
    store outlived vLLM's, a prefix computed by the previous policy would be silently resumed.
    """

    def __init__(self, budget_bytes: int, evict: bool = True):
        self.budget = int(budget_bytes)
        # evict=False (shared-index mode) makes the store append-only within an epoch: a full store
        # refuses new puts rather than evicting, so a checkpoint the scheduler's mirror advertises
        # cannot vanish mid-epoch. A refusal costs only a future cache hit.
        self.evict = bool(evict)
        self._d: OrderedDict[bytes, tuple[int, torch.Tensor, torch.Tensor]] = OrderedDict()
        self._bytes = 0
        self.hits = self.misses = self.stores = self.evictions = 0
        self.full_refusals = 0
        self._said_oversize = False

    def __contains__(self, key: bytes) -> bool:
        return key in self._d

    def __len__(self) -> int:
        return len(self._d)

    @property
    def nbytes(self) -> int:
        return self._bytes

    def get(self, key: bytes) -> tuple[int, torch.Tensor, torch.Tensor] | None:
        v = self._d.get(key)
        if v is None:
            self.misses += 1
            return None
        self._d.move_to_end(key)
        self.hits += 1
        return v

    def put(self, key: bytes, pos: int, entry: torch.Tensor, conv: torch.Tensor) -> bool:
        """Store one checkpoint. Returns True iff a NEW entry was stored (the publisher's cue)."""
        if key in self._d:
            self._d.move_to_end(key)
            return False
        n = entry.numel() * entry.element_size() + conv.numel() * conv.element_size()
        if n > self.budget:  # a single checkpoint bigger than the whole budget: refuse, do not thrash
            if not self._said_oversize:
                self._said_oversize = True
                # A model whose GDN state exceeds the whole budget can never cache; say so, or CPR-APC
                # looks installed and is permanently inert.
                print(
                    f"[ISOEXEC-CPR-APC] STORE OVERSIZE: one boundary checkpoint is "
                    f"{n / 2**20:.1f} MiB but the whole budget is {self.budget / 2**20:.1f} MiB -- "
                    "no checkpoint can ever be stored and CPR-APC will never serve a hit. Raise "
                    "SKYRL_ISOEXEC_GDN_CPR_APC_MB or set SKYRL_ISOEXEC_GDN_CPR_APC=0.",
                    flush=True,
                )
            return False
        if self._bytes + n > self.budget and not self.evict:
            self.full_refusals += 1
            if self.full_refusals == 1:
                # Fail-closed: in shared mode the store is append-only within a weight-sync epoch, so
                # once full no new checkpoint is stored or published until the next reset.
                print(
                    f"[ISOEXEC-CPR-APC] STORE FULL at {self._bytes / 2**20:.1f}/"
                    f"{self.budget / 2**20:.1f} MiB ({len(self._d)} checkpoints): append-only mode "
                    "refuses new checkpoints instead of evicting (the shared index must never "
                    "advertise an evicted entry). Hit rate decays until the next weight sync. "
                    "Raise SKYRL_ISOEXEC_GDN_CPR_APC_MB if this repeats.",
                    flush=True,
                )
            return False
        while self._bytes + n > self.budget and self._d:
            _k, (_p, e, c) = self._d.popitem(last=False)
            self._bytes -= e.numel() * e.element_size() + c.numel() * c.element_size()
            self.evictions += 1
        self._d[key] = (int(pos), entry, conv)
        self._bytes += n
        self.stores += 1
        return True

    def clear(self) -> None:
        self._d.clear()
        self._bytes = 0
        for hook in list(CPR_APC_ON_CLEAR):  # retract the published index BEFORE anyone can admit
            hook()

    def stats(self) -> dict:
        return {
            "entries": len(self._d),
            "MiB": round(self._bytes / 2**20, 1),
            "hits": self.hits,
            "misses": self.misses,
            "stores": self.stores,
            "evictions": self.evictions,
            "full_refusals": self.full_refusals,
        }


# How many rows of one prefill call may stash checkpoints. The stash is cloned fp32 state held for
# exactly one step, so an unbounded one is a real memory spike. Capping it costs only cache-fill rate:
# the useful checkpoints are one per distinct prompt.
_APC_PENDING_MAX = 16

CPR_APC_STORE: CprBoundaryStore | None = None


def cpr_apc_store() -> CprBoundaryStore:
    """The process-wide boundary-state store (built on first use, sized by the budget flag)."""
    global CPR_APC_STORE
    if CPR_APC_STORE is None:
        CPR_APC_STORE = CprBoundaryStore(cpr_apc_budget_bytes(), evict=not cpr_apc_shared_mode())
    return CPR_APC_STORE


def cpr_apc_reset() -> None:
    """Drop every cached boundary state. MUST run whenever vLLM's own prefix cache is reset."""
    if CPR_APC_STORE is not None:
        CPR_APC_STORE.clear()


def cpr_apc_store_invalidate() -> None:
    """Wholesale-drop the store because its tensors are about to become unreadable, e.g. an engine
    sleep path releasing the pools. Alias of :func:`cpr_apc_reset`, named for the contract: if the
    checkpoint tensors ever move into a releasable pool, this must run before the release, or the
    shared-mode mirror keeps advertising checkpoints no worker can serve."""
    cpr_apc_reset()


_APC_ADOPT_WRITE_BATCH = 64


def cpr_apc_adopt_many(
    layers: list["CprGDN"],
    items: list[tuple[int, int, bytes]],
) -> list[int] | None:
    """Install a scheduler step's cached boundary states in one batched transaction.

    ``items`` are ``(slot, position, prefix_key)`` in scheduler order. The return value is the assigned
    row for every item, or ``None`` when any checkpoint is not servable. All checkpoint predicates are
    checked before row ownership is mutated, so a miss cannot leave a half-adopted batch behind.

    ``_assign_many`` is called once per layer and chooses the same rows in the same order as repeated
    ``_assign``. Float semantics are unchanged: entry and conv are exact copies, and the running state
    still applies the same fp32 -> bf16 -> fp32 cast to every scalar independently. The write batch is
    bounded so stacking every source at once cannot create a large transient.
    """
    if not items:
        return []
    if not layers:
        return None
    l0 = layers[0]
    C = l0.chunk_size
    store = cpr_apc_store()
    checkpoints: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _slot, pos, key in items:
        if pos <= 0 or pos % C:
            return None
        got = store.get(key)
        if got is None:
            return None
        ckpt_pos, entry, conv = got
        if ckpt_pos != pos or entry.shape[0] != len(layers) or conv.shape[0] != len(layers):
            return None
        checkpoints.append((entry, conv))

    slots = [int(slot) for slot, _pos, _key in items]
    rows_by_layer = [ly._assign_many(slots) for ly in layers]
    rows = rows_by_layer[0]
    if any(other != rows for other in rows_by_layer[1:]):
        raise RuntimeError(
            "[isoexec-gdn] CPR-APC batched adoption found non-lockstepped recurrent pools; "
            "continuing would install different requests on different layer rows"
        )

    dev = l0.ssm_state.device
    pos_all = [int(pos) for _slot, pos, _key in items]
    for j, ly in enumerate(layers):
        for start in range(0, len(items), _APC_ADOPT_WRITE_BATCH):
            stop = min(start + _APC_ADOPT_WRITE_BATCH, len(items))
            chunk_rows = rows[start:stop]
            r = _dev_idx(chunk_rows, torch.long, dev)
            entry = torch.stack([checkpoints[i][0][j] for i in range(start, stop)], dim=0)
            conv = torch.stack([checkpoints[i][1][j] for i in range(start, stop)], dim=0)
            entry_write(ly, chunk_rows, entry)
            ly.ssm_state[r] = entry.to(torch.bfloat16).to(torch.float32)
            ly.conv_state[r] = conv.to(ly.conv_state.dtype)
        r_all = _dev_idx(rows, torch.long, dev)
        ly.pos[r_all] = _dev_idx(pos_all, torch.long, dev)
        for row, pos in zip(rows, pos_all, strict=True):
            ly._row_pos[row] = pos
            ly._entry_pos[row] = pos
            ly._apc_pending.pop(row, None)
    return rows


def cpr_apc_adopt(layers: list["CprGDN"], slot: int, pos: int, key: bytes) -> bool:
    """Install one cached boundary state; compatibility door onto batched adoption.

    After this returns True the slot is in every layer's map with ``pos``/``_entry_pos`` set to the hit
    position, so the prefill that follows takes the ordinary continuation path (``seq_r == 0``,
    open-chunk buffer unread, running state already ``bf16(entry)``). Returns False on a miss, a
    non-boundary position or a layer-count mismatch; a False is never a correctness event, it just means
    the prompt prefills cold.
    """
    rows = cpr_apc_adopt_many(layers, [(int(slot), int(pos), key)])
    return rows is not None


def _alloc_state(shape, dtype, device):
    """Allocate ordinary zeroed state in the caller's current CUDA memory-pool context."""
    return torch.zeros(shape, dtype=dtype, device=device)


def layer_batched_resync(layers: list["CprGDN"], slot_pos: list[tuple[int, int]]) -> int:
    """Fire pending boundary resyncs for ALL layers in ONE stacked state-pass call.

    All layers march in lockstep, so their independent closed chunks are stacked as one packed
    state-pass call. Sequence-local kernels preserve each lane's arithmetic and the FP32 finals are
    split back by layer.

    Dedup: (row, pos) via layer 0's ``_entry_pos`` -- the mirrors are lockstepped, and every
    layer's mirror is updated. Returns the number of (layer-shared) rows resynced.
    """
    if not layers:
        return 0
    l0 = layers[0]
    C = l0.chunk_size
    rows_b: list[int] = []
    poss: list[int] = []
    for slot, pos in slot_pos:
        if pos <= 0 or pos % C:
            continue
        row = l0._slot2row.get(int(slot))
        if row is None or l0._entry_pos.get(row, 0) >= pos:
            continue
        rows_b.append(row)
        poss.append(pos)
    if not rows_b:
        return 0
    dev = l0.ssm_state.device
    rows_t = _dev_idx(rows_b, torch.long, dev)
    M = len(rows_b)
    L = len(layers)
    HV, Kd, Vd = l0.num_v_heads, l0.head_k_dim, l0.head_v_dim

    if l0._native_core:
        from .gdn_cpr import native_matched_prep

        Hk = l0.num_k_heads
        k_raw = torch.cat([ly.k_buf[rows_t] for ly in layers], dim=0).reshape(1, L * M * C, Hk, Kd)
        v = torch.cat([ly.v_buf[rows_t] for ly in layers], dim=0).reshape(1, L * M * C, HV, Vd)
        a_raw = torch.cat([ly.g_buf[rows_t] for ly in layers], dim=0).reshape(L * M * C, HV)
        b_raw = torch.cat([ly.b_buf[rows_t] for ly in layers], dim=0).reshape(L * M * C, HV)
        # A_log/dt_bias differ per layer: expand each layer's params across its M*C rows so the single
        # eager prep computes every layer with its own parameters.
        A_log = torch.cat([ly.A_log.expand(M * C, HV).reshape(M * C, HV) for ly in layers], dim=0)
        dt_bias = torch.cat([ly.dt_bias.expand(M * C, HV).reshape(M * C, HV) for ly in layers], dim=0)
        k, g, beta = native_matched_prep(k_raw, a_raw, b_raw, A_log, dt_bias)
        g = g.reshape(1, L * M * C, HV)
        beta = beta.reshape(1, L * M * C, HV)
    else:
        k = torch.cat([ly.k_buf[rows_t] for ly in layers], dim=0).reshape(1, L * M * C, HV, Kd)
        v = torch.cat([ly.v_buf[rows_t] for ly in layers], dim=0).reshape(1, L * M * C, HV, Vd)
        g = torch.cat([ly.g_buf[rows_t] for ly in layers], dim=0).reshape(1, L * M * C, HV)
        beta = torch.cat([ly.b_buf[rows_t] for ly in layers], dim=0).reshape(1, L * M * C, HV)
    init = entry_read_stacked(layers, rows_b, rows_t)
    cu = torch.arange(0, (L * M + 1) * C, C, dtype=torch.int32, device=dev)
    # Uniform grid (L*M sequences of exactly C): hand the lengths in so the chunk metadata is built on
    # the host instead of through the vendored device read-back, which also keeps a `.tolist()` sync out
    # of a driver designed to have none.
    _h, final, _ci = chunk_boundary_states(
        None, k, v, g, beta, cu, C, initial_state=init, output_final_state=True, lens=[C] * (L * M)
    )
    final = final.to(torch.float32)  # [L*M, HV, V, K]
    snap = final.to(torch.bfloat16).to(torch.float32)
    entry_write_stacked(layers, rows_b, rows_t, final)
    for j, ly in enumerate(layers):
        ly.ssm_state[rows_t] = snap[j * M : (j + 1) * M]
        for r, p in zip(rows_b, poss):
            ly._entry_pos[r] = p
    return M


def cpr_arena_specs(*, rows, C, HV, K, V, Hk, dt, gdt, conv_shape) -> list:
    """``[(attr, shape, dtype), ...]`` for the CPR DEVICE arena, in allocation order.

    Extracted from ``CprGDN.__init__`` so allocation order remains explicit.
    """
    specs = [
        ("ssm_state", (rows, HV, V, K), torch.float32),
        ("conv_state", tuple(conv_shape), dt),
        ("entry_state", (rows, HV, V, K), torch.float32),
        ("k_buf", (rows, C, Hk, K), dt),
        ("v_buf", (rows, C, HV, V), dt),
        ("g_buf", (rows, C, HV), gdt),
        ("b_buf", (rows, C, HV), dt),
    ]
    return specs


def build_cpr_gdn(*, max_num_seqs: int, chunk_size: int, **kw) -> "CprGDN":
    """Build a :class:`CprGDN` sized by the scheduler's concurrency cap."""
    return CprGDN(capacity=max_num_seqs, chunk_size=chunk_size, **kw)


class CprGDN(RecurrentGDN):
    """RecurrentGDN + per-slot open-chunk buffers, fp32 entry states, and the boundary resync."""

    _entry_state_dev = None

    @property
    def entry_state(self) -> torch.Tensor:
        """The fp32 boundary chain, ``[rows, HV, V, K]``.

        A view into the CPR device arena.
        """
        return self._entry_state_dev

    @entry_state.setter
    def entry_state(self, t: torch.Tensor) -> None:
        self._entry_state_dev = t

    def __init__(self, *, capacity: int, chunk_size: int, ssm_state=None, conv_state=None, **kw):
        if ssm_state is not None or conv_state is not None:
            raise ValueError(
                "[isoexec-gdn] cpr mode owns its state pool (entry states + open-chunk "
                "buffers live beside it); native vLLM state is not supported. Unset "
                "SKYRL_ISOEXEC_GDN_NATIVE_STATE."
            )
        super().__init__(capacity=capacity, **kw)
        rows = capacity + 1  # row 0 = null, as in the parent pool
        C = int(chunk_size)
        HV, K, V = self.num_v_heads, self.head_k_dim, self.head_v_dim
        dev = self.ssm_state.device
        dt = self.conv_state.dtype  # bf16: the kernels' input dtype
        self.chunk_size = C
        # With native kernels, the fused-sigmoid core evaluates within-chunk scans and the native
        # conv pair handles convolution. Boundary prep is recomputed from buffered raw compressed
        # k and raw a/b. Chunk-synced state remains in the private pool.
        from .gdn_ops import gdn_native_kernels_enabled

        self._native_core = gdn_native_kernels_enabled()
        # ``GDN_CPR_SLEEP`` places these seven tensors in one verified private-tag arena. Otherwise
        # they follow the CUDA allocation context active during model construction. Allocation is
        # complete before graph capture.
        # Open-chunk buffers: the eager core stores prepared values (k expanded to HV, g fp32), the
        # native core stores raw ones (compressed-head k, raw a in the g slot, raw b) and
        # native_matched_prep recomputes the boundary prep from them at resync, as the trainer does.
        # The fp32 entry state is the chunk-pass accumulator at the slot's last crossed boundary; it is
        # what makes the bf16 snapshots exact and must never round through bf16.
        _Hk = self.num_k_heads if self._native_core else HV
        _gdt = dt if self._native_core else torch.float32
        _specs = cpr_arena_specs(
            rows=rows,
            C=C,
            HV=HV,
            K=K,
            V=V,
            Hk=_Hk,
            dt=dt,
            gdt=_gdt,
            conv_shape=self.conv_state.shape,
        )
        from .gdn_cpr_sleep import alloc_cpr_arena, register_layer

        _arena = alloc_cpr_arena(_specs, dev)
        if _arena is None:
            self.ssm_state = _alloc_state((rows, HV, V, K), torch.float32, dev)
            self.conv_state = _alloc_state(tuple(self.conv_state.shape), dt, dev)
            self.entry_state = _alloc_state((rows, HV, V, K), torch.float32, dev)
            self.k_buf = _alloc_state((rows, C, _Hk, K), dt, dev)
            self.v_buf = _alloc_state((rows, C, HV, V), dt, dev)
            self.g_buf = _alloc_state((rows, C, HV), _gdt, dev)
            self.b_buf = _alloc_state((rows, C, HV), dt, dev)
            self._flat_arena = None
        else:
            self._flat_arena = _arena.pop("_flat")
            for _name, _t in _arena.items():
                setattr(self, _name, _t)
        # Absolute position (tokens consumed) per row; pos % C is how much of the open chunk is full.
        # Kept out of the sleep arena: it is small, and the wake-time reset rewrites it in lockstep with
        # the host mirrors, so its bytes are restored by sleep rather than dropped.
        self.pos = torch.zeros(rows, dtype=torch.int64, device=dev)
        if self._flat_arena is not None:
            register_layer(self, self._flat_arena, capacity)

        # Lazy-resync mode (the CUDA-graph path). Off: decode() detects boundary crossings itself (one
        # host sync per step) and resyncs, self-contained but uncapturable. On: decode() is pure device
        # work, and a host driver that knows every lane's position from the scheduler's metadata calls
        # host_resync() before the step's forward, between graph replays. Bits are identical either way:
        # the resync for the chunk closed at step t may run any time before the lane's next token, and
        # the buffer column it would overwrite at t+1 is written after the pre-forward driver runs.
        self.lazy_resync = False
        # Host mirror: row -> absolute position of its fp32 entry state (the last boundary at which
        # entry_state[row] was set). The driver uses it to fire each boundary exactly once, since
        # prefill may already have resynced a prompt that ended on a boundary.
        self._entry_pos: dict[int, int] = {}
        # Host mirror of `pos`, fed from the scheduler-owned arrays by the lazy driver and maintained
        # by prefill/APC/reset; it removes `self.pos[rows].tolist()` from every engine prefill layer.
        # Missing rows fall back to the device read, so callers that bypass the driver still work.
        self._row_pos: dict[int, int] = {}
        # True only after the vLLM lazy driver has supplied scheduler-owned positions. Setting
        # `lazy_resync=True` in an offline harness is not enough: such a caller has no host oracle for
        # non-boundary decode rows and must keep the device-read fallback.
        self._driver_managed = False
        # Native vLLM conv pair (causal_conv1d_fn prefill / _update decode) instead of the eager
        # per-sequence loop: one varlen launch. The two convs round differently, so the trainer shim
        # must flip with it. Read once at build.
        from .gdn_cpr_scatter import fused_scatter_enabled
        from .gdn_ops import gdn_native_conv_enabled

        # The native core implies the native conv pair.
        self._native_conv = gdn_native_conv_enabled() or self._native_core
        # One-launch open-chunk buffer scatter + pos advance (pure data movement; default on).
        self._fused_scatter = fused_scatter_enabled()
        # APC (prefix caching). Off costs nothing on any path. On, prefill also materialises the fp32
        # entry state and conv window at the last one or two chunk boundaries it closes and stashes them
        # as row -> [(absolute boundary pos, entry fp32 [HV,V,K], conv [D,W-1]), ...] for the driver.
        self._apc = cpr_apc_enabled()
        self._apc_pending: dict[int, list[tuple[int, torch.Tensor, torch.Tensor]]] = {}
        CPR_LAYERS.append(self)

    # -- internals ------------------------------------------------------------------------
    def _reset_rows(self, rows) -> None:
        """Zero the CPR side state for rows starting a FRESH prompt.

        ``rows`` is a host list of row indices, not a device tensor: prefill decides freshness on the
        host and every caller already has the list.
        """
        entry_zero(self, rows)
        self.pos[_dev_idx(rows, torch.long, self.pos.device)] = 0
        for r in rows:
            self._row_pos[int(r)] = 0
        if self._apc_pending:
            for r in rows:
                self._apc_pending.pop(int(r), None)

    def _on_row_released(self, row: int) -> None:
        """A released row keeps no CPR side state: entry position, pending checkpoints."""
        self._entry_pos.pop(int(row), None)
        self._row_pos.pop(int(row), None)
        self._apc_pending.pop(int(row), None)

    def note_slot_positions(self, slot_pos) -> None:
        """Mirror scheduler positions for this step's PREFILL rows, host-only.

        ``_row_pos`` is consumed only by ``_prefill_pos0``. Decode advances the device ``pos`` but
        never consumes this mirror, so the engine deliberately omits decode rows from ``slot_pos``.
        ``None`` means the driver could not positively validate the decode/prefill split: clear the
        mirror so a possible continuation takes the exact device-read fallback instead of trusting
        a position that decode may have advanced.

        Calling this method also latches scheduler lifecycle ownership. Once latched, a full pool
        must be relieved by finished/preempted release; ``_assign_many`` may not LRU-evict a mapped
        row because it can belong to a temporarily unscheduled live request.
        """

        self._driver_managed = True
        if slot_pos is None:
            self._row_pos.clear()
            return
        for slot, pos in slot_pos:
            row = self._slot2row.get(int(slot))
            if row is not None:
                self._row_pos[row] = int(pos)

    def _prefill_pos0(self, rows, rows_t) -> list[int]:
        """Starting positions from the host mirror, with the old D2H path as exact fallback."""

        if all(int(r) in self._row_pos for r in rows):
            _HOST_BOOKKEEPING_STATS["pos_mirror"] += 1
            return [self._row_pos[int(r)] for r in rows]
        pos0 = [int(v) for v in self.pos[rows_t].tolist()]
        for r, p in zip(rows, pos0):
            self._row_pos[int(r)] = p
        _HOST_BOOKKEEPING_STATS["pos_device_fallback"] += 1
        return pos0

    # -- APC boundary checkpoints (host-visible, prefill-only) -----------------------------
    def _apc_prev_finals(self, k, v, g, beta, cu_l, ms, init):
        """fp32 entry states at the boundary one before each pass sequence's last one.

        ``chunk_boundary_states`` exposes intermediate boundaries only as the bf16 snapshots ``h``, and
        resuming a request from a bf16-rounded entry would not reproduce a cold prefill, so the fp32
        value must come from a pass that ends at the boundary being checkpointed. Re-running the same
        pass truncated by one chunk is the cheapest way to get it and leaves the production pass
        untouched. It is exact because the state pass is a chunk-sequential fp32 scan: its final over
        the first m-1 chunks equals its own internal entry to chunk m-1.

        Costs about one extra boundary pass per prefill call, only under CPR-APC.
        """
        C = self.chunk_size
        sel = [j for j, m in enumerate(ms) if m >= 2]
        if not sel:
            return {}
        dev = k.device
        idx: list[int] = []
        cu2 = [0]
        for j in sel:
            s, e = cu_l[j], cu_l[j + 1] - C
            idx.extend(range(s, e))
            cu2.append(cu2[-1] + (e - s))
        ti = _dev_idx(idx, torch.long, dev)
        sel_t = _dev_idx(sel, torch.long, dev)
        cu2_t = _dev_idx(cu2, torch.int32, dev)
        _h, final, _ci = chunk_boundary_states(
            None,
            k[:, ti],
            v[:, ti],
            g[:, ti],
            beta[:, ti],
            cu2_t,
            C,
            initial_state=init[sel_t],
            output_final_state=True,
            lens=[cu2[a + 1] - cu2[a] for a in range(len(sel))],
        )
        return {j: final[a].to(torch.float32) for a, j in enumerate(sel)}

    def _apc_capture(self, x, rows, qsl, pos0, seq_r, seq_m, pass_seqs, finals, prev_finals):
        """Stash (boundary pos, fp32 entry, conv window) for the last one/two boundaries closed.

        The conv half of a checkpoint is the last ``W-1`` pre-conv inputs at that position, carried by
        this call's own ``x``, so a boundary is checkpointable only when the whole window lies inside
        the call. Boundaries that fail that test are skipped, never approximated.
        """
        W = self.conv_weight.shape[-1]
        C = self.chunk_size
        for j, i in enumerate(pass_seqs):
            if len(self._apc_pending) >= _APC_PENDING_MAX:
                break
            row = rows[i]
            base = pos0[i] - seq_r[i]  # absolute position of this call's first closing boundary - C
            cks: list[tuple[int, torch.Tensor, torch.Tensor]] = []
            cands = [(base + seq_m[i] * C, finals[i])]
            if j in prev_finals:
                cands.append((base + (seq_m[i] - 1) * C, prev_finals[j]))
            for p, entry in cands:
                t = qsl[i] + (p - pos0[i])  # row index in x of the token AFTER the boundary
                if p <= 0 or (p - pos0[i]) < W - 1 or t > qsl[i + 1]:
                    continue
                cks.append((int(p), entry.detach().clone(), x[t - (W - 1) : t].transpose(0, 1).contiguous()))
            if cks:
                self._apc_pending[row] = cks

    # -- lazy-resync driver API (host, pre-forward, outside any CUDA graph) ----------------
    def host_resync(self, slot_pos: list[tuple[int, int]]) -> int:
        """Resync every (engine slot, absolute position) pair that sits on an unserviced boundary.

        Called by the metadata-builder wrap once per decode step, before the forward. ``pos`` is the
        lane's tokens-consumed count, which the scheduler knows on the host, so there is no device
        readback. ``_entry_pos`` dedups against prefill's own boundary handoff and against repeated
        driver calls, so the state pass fires exactly once per boundary per row. Returns rows resynced.
        """
        rows_b = []
        for slot, pos in slot_pos:
            if pos <= 0 or pos % self.chunk_size:
                continue
            row = self._slot2row.get(int(slot))
            if row is None:
                continue
            if self._entry_pos.get(row, 0) >= pos:
                continue  # this boundary was already serviced (by prefill or an earlier call)
            rows_b.append(row)
            self._entry_pos[row] = pos
        if rows_b:
            self._resync(rows_b)
        return len(rows_b)

    def _resync(self, rows) -> None:
        """Boundary resync for rows whose position just reached a multiple of C.

        Re-runs the chunk state pass over each row's full buffer (exactly C tokens) from its fp32 entry
        state, then applies the handoff: running state <- bf16(final) upcast to fp32 (the snapshot the
        trainer's next scan segment loads), entry <- final (the fp32 chain).

        ``rows`` is a host list; the device index tensor the buffer gathers need is built here, once.
        """
        M = len(rows)
        if M == 0:
            return
        C = self.chunk_size
        dev = self.ssm_state.device
        rows_b = _dev_idx(rows, torch.long, dev)
        v = self.v_buf[rows_b].reshape(1, M * C, self.num_v_heads, self.head_v_dim)
        if self._native_core:
            # Buffers hold raw compressed-head k and raw a/b: recompute the boundary prep eagerly with
            # the same shared function the trainer forward uses.
            from .gdn_cpr import native_matched_prep

            k_raw = self.k_buf[rows_b].reshape(1, M * C, self.num_k_heads, self.head_k_dim)
            a_raw = self.g_buf[rows_b].reshape(M * C, self.num_v_heads)
            b_raw = self.b_buf[rows_b].reshape(M * C, self.num_v_heads)
            k, g, beta = native_matched_prep(k_raw, a_raw, b_raw, self.A_log, self.dt_bias)
            g = g.reshape(1, M * C, self.num_v_heads)
            beta = beta.reshape(1, M * C, self.num_v_heads)
        else:
            k = self.k_buf[rows_b].reshape(1, M * C, self.num_v_heads, self.head_k_dim)
            g = self.g_buf[rows_b].reshape(1, M * C, self.num_v_heads)
            beta = self.b_buf[rows_b].reshape(1, M * C, self.num_v_heads)
        cu = torch.arange(0, (M + 1) * C, C, dtype=torch.int32, device=dev)
        _h, final, _ci = chunk_boundary_states(
            None,
            k,
            v,
            g,
            beta,
            cu,
            C,
            initial_state=entry_read(self, rows),
            output_final_state=True,
            lens=[C] * M,  # uniform grid: host-built chunk metadata (see host_chunk_meta)
        )
        final = final.to(torch.float32)
        self.ssm_state[rows_b] = final.to(torch.bfloat16).to(torch.float32)
        entry_write(self, rows, final)

    # -- public API -----------------------------------------------------------------------
    @torch.no_grad()
    def _prefill_native(
        self,
        slots,
        slots_cpu,
        x,
        a,
        b,
        qsl,
        has_initial_state=None,
        conv_metadata=None,
        prefill_query_start_loc=None,
    ):
        """Native prefill: native conv fn + matched-prep boundary pass + segmented fused-core scan.

        Same slot bookkeeping, segment grid and bf16-snapshot/fp32-chain handoff as the eager ``prefill``
        below; only the composition differs -- raw q/k/a/b drive the fused core, and the boundary pass
        consumes ``native_matched_prep`` of the raw buffered values.
        """
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn

        from .gdn_cpr import native_matched_prep
        from .gdn_ops import gdn_native_core_kernel

        N = len(slots_cpu)
        dev = x.device
        C = self.chunk_size
        is_cont = self._continuation_mask(slots_cpu, has_initial_state)
        # One batched claim for all N prompts: one slot-map flush, one LRU host read, one clock
        # scatter. Nothing between here and the end of this forward reads the device slot map
        # (``_continuation_mask`` reads the host mirror and runs first; ``_rows`` is decode-only and
        # decode precedes prefill in a mixed batch), so this is inside the flush's ordering contract.
        rows = self._assign_many(slots_cpu)
        rows_t = _dev_idx(rows, torch.long, dev)

        rows32 = rows_t.to(torch.int32)
        cu = prefill_query_start_loc
        if cu is None:
            cu = _dev_idx(qsl, torch.int32, dev)
        has = _dev_idx(is_cont, torch.bool, dev)
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
        # One launch for the three column slices and the alpha/beta compaction ``gdn_layer_core``
        # deferred here; ``a``/``b`` are not read before this point, so the deferral is a pure move.
        q, k, v, a, b = self._split_qkv_raw(y, a, b)

        fresh_rows = [r for r, c in zip(rows, is_cont) if not c]
        if fresh_rows:
            fr = _dev_idx(fresh_rows, torch.long, dev)
            self.ssm_state[fr] = 0.0
            self._reset_rows(fresh_rows)

        lens = [qsl[i + 1] - qsl[i] for i in range(N)]
        pos0 = self._prefill_pos0(rows, rows_t)
        for i, (p, c) in enumerate(zip(pos0, is_cont)):
            if not c and p != 0:
                raise RuntimeError(f"[isoexec-gdn] fresh row {rows[i]} has pos {p}")

        seq_r = [p % C for p in pos0]
        seq_m = [(seq_r[i] + lens[i]) // C for i in range(N)]
        pass_seqs = [i for i in range(N) if seq_m[i] > 0]
        snaps: dict[int, torch.Tensor] = {}
        finals: dict[int, torch.Tensor] = {}
        if pass_seqs:
            ks, vs, as_, bs_ = [], [], [], []
            cu_l = [0]
            for i in pass_seqs:
                s = qsl[i]
                r = seq_r[i]
                span = seq_m[i] * C - r
                if r:
                    row = rows[i]
                    ks.append(self.k_buf[row, :r])
                    vs.append(self.v_buf[row, :r])
                    as_.append(self.g_buf[row, :r])
                    bs_.append(self.b_buf[row, :r])
                ks.append(k[s : s + span])
                vs.append(v[s : s + span])
                as_.append(a[s : s + span])
                bs_.append(b[s : s + span])
                cu_l.append(cu_l[-1] + seq_m[i] * C)
            kp = torch.cat(ks, dim=0).unsqueeze(0)
            vp = torch.cat(vs, dim=0).unsqueeze(0)
            ap = torch.cat(as_, dim=0)
            bp = torch.cat(bs_, dim=0)
            kn, g_, beta_ = native_matched_prep(kp, ap, bp, self.A_log, self.dt_bias)
            Tp = kp.shape[1]
            cup = _dev_idx(cu_l, torch.int32, dev)
            init = entry_read(self, [rows[i] for i in pass_seqs])
            h, final, _ci = chunk_boundary_states(
                None,
                kn,
                vp,
                g_.reshape(1, Tp, -1),
                beta_.reshape(1, Tp, -1),
                cup,
                C,
                initial_state=init,
                output_final_state=True,
                lens=[cu_l[j + 1] - cu_l[j] for j in range(len(cu_l) - 1)],
            )
            off = 0
            for j, i in enumerate(pass_seqs):
                snaps[i] = h[0, off : off + seq_m[i]]
                finals[i] = final[j].to(torch.float32)
                off += seq_m[i]
            if self._apc:
                ms = [seq_m[i] for i in pass_seqs]
                prev = self._apc_prev_finals(kn, vp, g_.reshape(1, Tp, -1), beta_.reshape(1, Tp, -1), cu_l, ms, init)
                self._apc_capture(x, rows, qsl, pos0, seq_r, seq_m, pass_seqs, finals, prev)

        # (2) Segment grid and state pool. All index construction is host python/numpy; the device
        # sees a handful of batched ops. Pool row order is free, since the kernel reads explicit
        # state_indices:
        #   row 0          null | rows 1..N   first-segment inits (running fp32 states, one gather)
        #   next n_snap    the boundary pass's bf16 snapshots (h) upcast, one cat
        #   next P         bf16(final).fp32 per pass sequence (the new open chunk's entry)
        import numpy as np

        P = len(pass_seqs)
        finals_t = torch.stack([finals[i] for i in pass_seqs], dim=0) if P else None  # [P,...] fp32
        snap_base = 1 + N
        n_snap = int(sum(seq_m[i] for i in pass_seqs))
        final_base = snap_base + n_snap
        pool_parts = [
            torch.zeros(1, *self.ssm_state.shape[1:], dtype=torch.float32, device=dev),
            self.ssm_state[rows_t],
        ]
        snap_off: dict[int, int] = {}
        final_row_of: dict[int, int] = {}
        if P:
            off = 0
            for j, i in enumerate(pass_seqs):
                snap_off[i] = snap_base + off
                off += seq_m[i]
                final_row_of[i] = final_base + j
            pool_parts.append(torch.cat([snaps[i] for i in pass_seqs], dim=0).to(torch.float32))
            pool_parts.append(finals_t.to(torch.bfloat16).to(torch.float32))
        pool = torch.cat(pool_parts, dim=0)

        seg_cu = [0]
        seg_rows: list[int] = []
        seg_last_of_seq: list[int] = []  # POOL row holding each sequence's LAST segment exit
        for i in range(N):
            s, e = qsl[i], qsl[i + 1]
            r = seq_r[i]
            t = s
            first = True
            while t < e:
                seg = min(C - (r if first else 0), e - t) if first else min(C, e - t)
                if first:
                    seg_rows.append(1 + i)
                else:
                    c_rel = (r + (t - s)) // C
                    seg_rows.append(final_row_of[i] if c_rel == seq_m[i] else snap_off[i] + c_rel)
                seg_cu.append(seg_cu[-1] + seg)
                t += seg
                first = False
            seg_last_of_seq.append(seg_rows[-1])
        M = len(seg_rows)
        cu_seg = _dev_idx(seg_cu, torch.int32, dev)
        seg_lens = np.diff(np.asarray(seg_cu, dtype=np.int64))
        Tmax = int(seg_lens.max())
        pad = max(64, 1 << (Tmax - 1).bit_length())
        idx_np = np.zeros((M, pad), dtype=np.int32)
        rows_np = np.asarray(seg_rows, dtype=np.int32)
        idx_np[:, 0] = rows_np
        idx_np[np.arange(M), seg_lens - 1] = rows_np
        idx = torch.from_numpy(idx_np).to(dev, non_blocking=True)[:, :Tmax]
        o = gdn_native_core_kernel(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            a,
            b,
            self.A_log,
            self.dt_bias,
            ssm_state=pool,
            state_indices=idx,
            cu_seqlens=cu_seg,
        )

        # (3) Handoff. Each segment stored its exit at its own pool row, so a sequence's running state
        # is pool[its last segment row]. Sequences ending exactly on a boundary must instead take
        # bf16(final), and seg_last_of_seq does not already point there (their last segment row is a
        # snapshot row whose exit the kernel's store overwrote), so they are selected explicitly.
        r2_np = (np.asarray(seq_r, dtype=np.int64) + np.asarray(lens, dtype=np.int64)) % C
        exits = pool[_dev_idx(seg_last_of_seq, torch.long, dev)]
        if P:
            entry_write(self, [rows[i] for i in pass_seqs], finals_t)
            snap_final = finals_t.to(torch.bfloat16).to(torch.float32)
            boundary = [i for i in pass_seqs if lens[i] > 0 and r2_np[i] == 0]
            if boundary:
                bsel = _dev_idx([pass_seqs.index(i) for i in boundary], torch.long, dev)
                bpos = _dev_idx(boundary, torch.long, dev)
                exits = exits.clone()
                exits[bpos] = snap_final[bsel]
        self.ssm_state[rows_t] = exits
        # Open-chunk buffer tails: flattened (row, col, src) triples, one batched index_put each.
        tail_rows: list[int] = []
        tail_cols: list[int] = []
        tail_src: list[int] = []
        for i in range(N):
            r2 = int(r2_np[i])
            if r2:
                new_tail = min(r2, lens[i])
                base = r2 - new_tail
                e = qsl[i + 1]
                tail_rows.extend([rows[i]] * new_tail)
                tail_cols.extend(range(base, r2))
                tail_src.extend(range(e - new_tail, e))
        if tail_rows:
            tr = _dev_idx(tail_rows, torch.long, dev)
            tc = _dev_idx(tail_cols, torch.long, dev)
            ts = _dev_idx(tail_src, torch.long, dev)
            self.k_buf[tr, tc] = k[ts]
            self.v_buf[tr, tc] = v[ts]
            self.g_buf[tr, tc] = a[ts].to(self.g_buf.dtype)
            self.b_buf[tr, tc] = b[ts].to(self.b_buf.dtype)
        # self.pos is int64 (see __init__); index_put casts to the destination dtype either way.
        self.pos[rows_t] = _dev_idx([pos0[i] + lens[i] for i in range(N)], torch.long, dev)
        for i in range(N):
            self._row_pos[rows[i]] = pos0[i] + lens[i]
            self._entry_pos[rows[i]] = (pos0[i] + lens[i]) // C * C
        return o[0]

    @torch.no_grad()
    def prefill(
        self,
        slots,
        slots_cpu,
        x,
        a,
        b,
        qsl,
        has_initial_state=None,
        conv_metadata=None,
        prefill_query_start_loc=None,
    ):
        """Prefill via the canonical CPR forward; leaves decode-compatible slot state.

        Handles fresh prompts and chunked-prefill continuations (a slot already in the map resumes
        from its running state, entry state, and open-chunk buffer -- a mid-chunk prefill stop and a
        decode mid-chunk are the same state by construction).
        """
        if self._native_core:
            return self._prefill_native(
                slots,
                slots_cpu,
                x,
                a,
                b,
                qsl,
                has_initial_state,
                conv_metadata,
                prefill_query_start_loc,
            )

        N = len(slots_cpu)
        dev = x.device
        C = self.chunk_size
        W = self.conv_weight.shape[-1]
        is_cont = self._continuation_mask(slots_cpu, has_initial_state)
        rows = self._assign_many(slots_cpu)  # batched claim; see _prefill_native's note above
        rows_t = torch.tensor(rows, dtype=torch.long, device=dev)

        # Conv: the native pair is one varlen causal_conv1d_fn launch for all N prompts, resuming
        # continuation rows from their stored window and writing every row's final window back at the
        # block rows in-kernel. Otherwise the eager per-sequence loop resumes the width-W window; each
        # composition is bitwise within itself, since the stored final_state is the raw pre-conv inputs.
        if self._native_conv:
            from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
                causal_conv1d_fn,
            )

            rows32 = rows_t.to(torch.int32)
            cu = prefill_query_start_loc
            if cu is None:
                cu = torch.tensor(qsl, dtype=torch.int32, device=dev)
            has = torch.tensor(is_cont, dtype=torch.bool, device=dev)
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
            convs = None
        else:
            ys, convs = [], []
            for i in range(N):
                s, e = qsl[i], qsl[i + 1]
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

        # Fresh rows: zero the scan state, entry state and position. Buffer columns are overwritten
        # before any resync can read them.
        fresh_rows = [r for r, c in zip(rows, is_cont) if not c]
        if fresh_rows:
            fr = torch.tensor(fresh_rows, dtype=torch.long, device=dev)
            self.ssm_state[fr] = 0.0
            self._reset_rows(fresh_rows)

        lens = [qsl[i + 1] - qsl[i] for i in range(N)]
        pos0 = self._prefill_pos0(rows, rows_t)
        for i, (p, c) in enumerate(zip(pos0, is_cont)):
            if not c and p != 0:
                raise RuntimeError(f"[isoexec-gdn] fresh row {rows[i]} has pos {p}")

        # (1) Boundary pass over [buffered open tail ++ new tokens], complete chunks only. For each
        # sequence r = pos % C tokens are buffered and T_i are new, so m = (r+T_i)//C chunks close in
        # this call. Those m*C tokens are fed from the fp32 entry state; the bf16 snapshots h[1..m-1]
        # feed the segmented scan and the fp32 final becomes the new entry.
        seq_r = [p % C for p in pos0]
        seq_m = [(seq_r[i] + lens[i]) // C for i in range(N)]
        pass_seqs = [i for i in range(N) if seq_m[i] > 0]
        # Per-sequence snapshot tensors ([m_i, HV, V, K] bf16) and fp32 finals.
        snaps: dict[int, torch.Tensor] = {}
        finals: dict[int, torch.Tensor] = {}
        if pass_seqs:
            ks, vs, gs, bs = [], [], [], []
            cu_l = [0]
            for i in pass_seqs:
                s = qsl[i]
                r = seq_r[i]
                span = seq_m[i] * C - r  # new tokens entering the pass
                if r:
                    row = rows[i]
                    ks.append(self.k_buf[row, :r])
                    vs.append(self.v_buf[row, :r])
                    gs.append(self.g_buf[row, :r])
                    bs.append(self.b_buf[row, :r])
                ks.append(k[s : s + span])
                vs.append(v[s : s + span])
                gs.append(g[s : s + span])
                bs.append(beta[s : s + span])
                cu_l.append(cu_l[-1] + seq_m[i] * C)
            kp = torch.cat(ks, dim=0).unsqueeze(0)
            vp = torch.cat(vs, dim=0).unsqueeze(0)
            gp = torch.cat(gs, dim=0).unsqueeze(0).to(torch.float32)
            bp = torch.cat(bs, dim=0).unsqueeze(0)
            cup = torch.tensor(cu_l, dtype=torch.int32, device=dev)
            init = entry_read(self, [rows[i] for i in pass_seqs])
            h, final, _ci = chunk_boundary_states(
                None,
                kp,
                vp,
                gp,
                bp,
                cup,
                C,
                initial_state=init,
                output_final_state=True,
                lens=[cu_l[j + 1] - cu_l[j] for j in range(len(cu_l) - 1)],
            )
            # h is [1, NT_total, HV, V, K] sequence-major in pass order; final is [P, HV, V, K] fp32.
            off = 0
            for j, i in enumerate(pass_seqs):
                snaps[i] = h[0, off : off + seq_m[i]]
                finals[i] = final[j].to(torch.float32)
                off += seq_m[i]
            if self._apc:
                ms = [seq_m[i] for i in pass_seqs]
                prev = self._apc_prev_finals(kp, vp, gp, bp, cu_l, ms, init)
                self._apc_capture(x, rows, qsl, pos0, seq_r, seq_m, pass_seqs, finals, prev)

        # (2) Segmented recurrent scan over the new tokens. The first segment of each sequence
        # continues from the running fp32 state (fresh rows are zero, matching the trainer's h[0]);
        # every later segment starts at a chunk boundary and loads the bf16 snapshot h[c] upcast to
        # fp32, exactly like the trainer pool. The scan runs on a scratch pool (running rows copied in,
        # fp32-exact) so every segment's store is skipped except the last, copied back to the slot row.
        seg_cu = [0]
        seg_init: list[torch.Tensor] = [torch.zeros_like(self.ssm_state[0])]  # scratch row 0 = null
        seg_last_of_seq: list[int] = []  # scratch row holding each sequence's LAST segment
        for i in range(N):
            s, e = qsl[i], qsl[i + 1]
            r = seq_r[i]
            t = s
            first = True
            while t < e:
                seg = min(C - (r if first else 0), e - t) if first else min(C, e - t)
                if first:
                    seg_init.append(self.ssm_state[rows[i]].clone())
                else:
                    c_rel = (r + (t - s)) // C  # chunk index in this call's boundary pass
                    if c_rel == seq_m[i]:
                        # The segment opens the new open chunk, so its entry snapshot is bf16(final):
                        # the h[m] the untruncated batched pass would have stored, and what a decode
                        # resync sets as the running state.
                        seg_init.append(finals[i].to(torch.bfloat16).to(torch.float32))
                    else:
                        seg_init.append(snaps[i][c_rel].to(torch.float32))
                seg_cu.append(seg_cu[-1] + seg)
                t += seg
                first = False
            seg_last_of_seq.append(len(seg_init) - 1)
        pool = torch.stack(seg_init, dim=0)
        M = len(seg_init) - 1
        cu_seg = torch.tensor(seg_cu, dtype=torch.int32, device=dev)
        Tmax = max(seg_cu[j + 1] - seg_cu[j] for j in range(M))
        pad = max(64, 1 << (Tmax - 1).bit_length())
        idx_cpu = torch.zeros(M, pad, dtype=torch.int32)
        for j in range(M):
            idx_cpu[j, 0] = j + 1
            idx_cpu[j, (seg_cu[j + 1] - seg_cu[j]) - 1] = j + 1  # store the exit in place
        idx = idx_cpu.to(dev, non_blocking=True)[:, :Tmax]
        o = gdn_recurrent_kernel(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            g.unsqueeze(0),
            beta.unsqueeze(0),
            ssm_state=pool,
            state_indices=idx,
            cu_seqlens=cu_seg,
        )

        # (3) Hand off decode-compatible slot state.
        for i in range(N):
            row = rows[i]
            r2 = (seq_r[i] + lens[i]) % C
            if seq_m[i] > 0:
                entry_write_one(self, row, finals[i])
            if lens[i] and r2 == 0 and seq_m[i] > 0:
                # The prompt ended on a boundary, so the resync fired as its last act and the running
                # state is the snapshot rather than the scan exit.
                self.ssm_state[row] = finals[i].to(torch.bfloat16).to(torch.float32)
            else:
                self.ssm_state[row] = pool[seg_last_of_seq[i]]
            # Open-chunk tail into the buffer. The tail spans the last r2 tokens, which include
            # carried buffer tokens only when no boundary closed (m == 0); those stay in place and the
            # new tokens append after them.
            s, e = qsl[i], qsl[i + 1]
            if r2:
                new_tail = min(r2, lens[i])  # tokens of the tail that are NEW
                base = r2 - new_tail  # already-buffered tokens that remain (m==0 continuation)
                sl = slice(e - new_tail, e)
                self.k_buf[row, base:r2] = k[sl]
                self.v_buf[row, base:r2] = v[sl]
                self.g_buf[row, base:r2] = g[sl].to(torch.float32)
                self.b_buf[row, base:r2] = beta[sl]
            self.pos[row] = pos0[i] + lens[i]
            self._row_pos[row] = pos0[i] + lens[i]
            # Host mirror for the lazy driver: entry_state[row] now corresponds to the last boundary
            # this prefill crossed (== pos when the prompt ended on one).
            self._entry_pos[row] = (pos0[i] + lens[i]) // C * C

        if W - 1 != self.conv_state.shape[-1]:  # pragma: no cover - shape contract
            raise RuntimeError(f"[isoexec-gdn] conv state width {self.conv_state.shape[-1]} != {W - 1}")
        if convs is not None:  # native conv already wrote the windows in-kernel at the block rows
            self.conv_state[rows_t] = torch.stack(convs, dim=0).to(self.conv_state.dtype)
        return o[0]

    @torch.no_grad()
    def decode(self, slots, x, a, b):
        """One decode step for N requests, plus buffer append and boundary resync where crossed.

        The eager form reads the crossing mask back to the host, so it does not capture into a CUDA
        graph. With ``lazy_resync`` on that branch is skipped entirely -- decode becomes pure device
        work and captures -- and the pre-forward host driver services boundaries between replays.
        """
        N = x.shape[0]
        # One launch for the slot->row chain and both widths; the ATen form is five single-block
        # kernels per layer inside the captured graph.
        rows, rows32 = self._rows_pair(slots)
        C = self.chunk_size

        if self._native_core:
            # Native conv update plus the fused_sigmoid core against our own pool, with the raw k/a/b
            # buffered for the matched-prep boundary pass. Same in-place-``x`` contract as the
            # native-conv branch below.
            from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
                causal_conv1d_update,
            )

            from .gdn_ops import gdn_native_core_kernel

            y = causal_conv1d_update(
                x,
                self.conv_state,
                self.conv_weight,
                self.conv_bias,
                self.activation,
                conv_state_indices=rows32,
            )
            # One launch for the three column slices and the alpha/beta compaction ``gdn_layer_core``
            # deferred here, replacing five per-layer copy launches. Byte movement only.
            q, k, v, a, b = self._split_qkv_raw(y, a, b)
            o = gdn_native_core_kernel(
                q.unsqueeze(1),
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
            o = o[:, 0]
            buf_k, buf_g, buf_b = k, a, b  # raw values; the resync recomputes the prep
        elif self._native_conv:
            # vLLM's decode conv slides the window in place at the block rows. Host-free and
            # shape-static, so it captures; the split-exactness of the fn/update pair is what makes
            # prefill->decode a bitwise continuation. Note causal_conv1d_update also writes its output
            # into ``x`` in place, so callers must not reuse ``x`` after this call.
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
            q, k, v = self._split_qkv(y)
        else:
            from .gdn_ops import gdn_causal_conv_batched

            cs = self.conv_state[rows]  # [N, D, W-1]
            y = gdn_causal_conv_batched(
                x.unsqueeze(1),
                self.conv_weight,
                self.conv_bias,
                initial_state=cs,
                activation=self.activation,
            )
            q, k, v = self._split_qkv(y.reshape(N, -1))
        if not self._native_core:
            g, beta = self._gate_and_beta(a, b)
            o = gdn_recurrent_kernel(
                q.unsqueeze(1),
                k.unsqueeze(1),
                v.unsqueeze(1),
                g.unsqueeze(1),
                beta.unsqueeze(1),
                ssm_state=self.ssm_state,
                state_indices=rows32,
                cu_seqlens=None,
            )
            o = o[:, 0]
            buf_k, buf_g, buf_b = k, g, beta  # post-prep values, as the boundary pass consumes them

        # Append this token to the open-chunk buffer, with the same values the boundary pass consumes.
        # Row 0 absorbs padded/null lanes' writes; its buffer is garbage no live request reads.
        stamp_in_scatter = self._fused_scatter and not self._native
        if stamp_in_scatter:
            # Hoisted above the scatter so the fused kernel can carry the last_used stamp with it;
            # nothing between here and the bottom of this method reads `_clock` or `last_used`.
            self._clock += 1
        if self._fused_scatter:
            # One launch for all four buffer writes, the pos advance and the LRU stamp. Pure data
            # movement, equal to the indexed writes below, and capture-safe.
            from .gdn_cpr_scatter import cpr_buffer_scatter

            cpr_buffer_scatter(
                buf_k.to(self.k_buf.dtype),
                v.to(self.v_buf.dtype),
                buf_g.to(self.g_buf.dtype),
                buf_b.to(self.b_buf.dtype),
                rows,
                self.pos,
                self.k_buf,
                self.v_buf,
                self.g_buf,
                self.b_buf,
                C,
                last_used=self.last_used if stamp_in_scatter else None,
                clock=self._clock if stamp_in_scatter else None,
            )
        else:
            # `live` exists only for these two branches; computing it unconditionally would put a dead
            # single-block compare inside the captured graph.
            live = rows > 0
            col = (self.pos[rows] % C).long()
            self.k_buf[rows, col] = buf_k.to(self.k_buf.dtype)
            self.v_buf[rows, col] = v.to(self.v_buf.dtype)
            self.g_buf[rows, col] = buf_g.to(self.g_buf.dtype)
            self.b_buf[rows, col] = buf_b.to(self.b_buf.dtype)
            # Branch-free and duplicate-safe under capture: pad lanes fold to row 0 and add 0 there,
            # and atomic int adds make duplicate row-0 indices deterministic.
            self.pos.index_put_((rows,), live.to(self.pos.dtype), accumulate=True)

        if not self.lazy_resync:
            # Self-contained boundary resync; host-syncs the crossing mask and is therefore
            # uncapturable. The lazy path skips it: the pre-forward host driver already serviced any
            # boundary before this step's forward and will service the one this token closes before the
            # next. This branch syncs anyway, so recomputing the liveness mask here is free.
            crossed = (rows > 0) & (self.pos[rows] % C == 0)
            if bool(crossed.any().item()):
                rows_b = rows[crossed]
                # This branch already host-syncs, so reading the crossing rows to the host costs
                # nothing new, and the entry state is addressed by host row list either way.
                rows_l = rows_b.tolist()
                # A row can appear at most once per step (one token per request per decode step).
                self._resync(rows_l)
                # Keep the host mirror true so a later switch to the lazy driver cannot double-fire.
                for r, p in zip(rows_l, self.pos[rows_b].tolist()):
                    self._entry_pos[r] = p

        if not self._native_conv:
            self.conv_state[rows] = torch.cat([cs[..., 1:], x.unsqueeze(-1).to(cs.dtype)], dim=-1)
        if not self._native and not stamp_in_scatter:
            self._clock += 1
            self.last_used[rows] = self._clock
        # An offline harness may set `lazy_resync=True` and call `host_resync` only at boundaries;
        # without the installed driver it has no scheduler-authoritative prefill position mirror.
        if not self.lazy_resync or not self._driver_managed:
            self._row_pos.clear()
        return o
