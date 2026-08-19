"""Free expert-weight stacking: one contiguous ``[E, *param_shape]`` buffer per (layer, role), with every
expert's ``.weight`` rebound to a view of it, so the per-forward ``torch.stack`` over experts becomes a
free alias. The bytes and layout are identical to what ``torch.stack`` produces, so this changes no
arithmetic -- it is a memcpy elimination.

Invariants a caller can break:

  * Reads must go through :func:`fused_expert_weights`, which verifies the alias (Parameter object
    identity, ``data_ptr``/shape/stride/dtype per expert, buffer storage size) and returns ``None`` so
    the caller falls back to an exact ``torch.stack``. A broken alias is never auto-re-fused: if an
    optimizer legitimately re-pointed ``param.data`` into its own buffer, re-fusing would sever that
    aliasing. :func:`refuse_module` is the explicit opt-in.
  * Under grad, the read routes through :class:`_StackView` so backward unbinds onto the E leaf
    params; handing back the raw buffer would deliver zero gradient to every expert.
  * :func:`refresh_all_fused` must run eagerly at the weight-sync boundary (via
    :func:`bump_sync_epoch`), outside any forward. CUDA-graph decode replays bake the buffer address
    at capture time and never run the lazy check, so a refresh must copy fresh bytes into the SAME
    buffer; a refresh that would reallocate raises instead of swapping the pointer under a live graph.
  * Verifying the invariant walks all E experts and is too expensive per forward, so it is scheduled:
    parameter mutations are intercepted (``_WatchedParamDict``, ``_DataWatch``) and bump a global
    epoch that triggers the next full scan.
  * Allocated memory is neutral across a fuse, but reserved is not -- per-expert tensors come from the
    caching allocator's small pool and the fused buffers from the large pool. See
    :func:`release_stranded_segments`.

Parameter objects are preserved (only ``.data`` is rebound), so optimizer state keyed on
param identity, ``named_parameters()`` and ``requires_grad`` survive fusing.
"""

from __future__ import annotations

import gc
import logging
import sys
import weakref
from typing import Iterable, Sequence

import torch

logger = logging.getLogger(__name__)

_STATE_ATTR = "_isoexec_fused_weights"

DEFAULT_ROLES: tuple[str, ...] = ("linear_fc1", "linear_fc2")

# Every module holding a live fused state, so a weight sync can eagerly refresh all of them at the sync
# boundary (:func:`refresh_all_fused`) -- the only point a CUDA-graph replay can see fresh bytes.
# WeakSet so a torn-down layer drops out with no unfuse bookkeeping.
_FUSED_MODULES: "weakref.WeakSet" = weakref.WeakSet()

# Mutation epoch: makes "nothing changed" observable in O(1) so the O(E) scan in `_full_check` runs only
# after an actual rebind. The two mutations that can break the alias are intercepted at their only
# entry points -- Parameter object replacement in `mod._parameters` (`_WatchedParamDict`) and
# `param.data = ...` / `set_` / `resize_` / `detach_` on the recorded object (`_DataWatch`) -- and both
# bump `_EPOCH`. `Tensor._version` does NOT bump on a `.data` rebind, so version counters cannot stand
# in for this. The epoch is global, not per-module: a rebind anywhere forces one full scan per layer on
# the next read, which is a fast-path miss and never a correctness hole.
_EPOCH = 0


def _bump_epoch() -> None:
    global _EPOCH
    _EPOCH += 1


class _WatchedParamDict(dict):
    """``mod._parameters`` that reports being mutated. Same dict semantics, plus an epoch bump.

    Installed on each expert's role module at fuse time. Every path that can swap a Parameter OBJECT
    -- ``nn.Module.__setattr__``, ``del mod.weight``, and ``vllm_worker._set_on_module``'s direct
    ``mod._parameters["weight"] = Parameter(...)`` -- goes through one of these methods.
    """

    __slots__ = ()

    def __setitem__(self, k, v):
        _bump_epoch()
        dict.__setitem__(self, k, v)

    def __delitem__(self, k):
        _bump_epoch()
        dict.__delitem__(self, k)

    def pop(self, *a):
        _bump_epoch()
        return dict.pop(self, *a)

    def popitem(self):
        _bump_epoch()
        return dict.popitem(self)

    def clear(self):
        _bump_epoch()
        dict.clear(self)

    def update(self, *a, **kw):
        _bump_epoch()
        dict.update(self, *a, **kw)

    def setdefault(self, k, d=None):
        _bump_epoch()
        return dict.setdefault(self, k, d)


class _DataWatch:
    """Mixin that makes a Parameter's storage rebinds observable.

    Mixed in by reassigning ``__class__`` on the fused params -- the SAME objects, so identity in
    ``_parameters``, ``isinstance(p, nn.Parameter)``, ``requires_grad`` and any subclass attributes
    all survive, arithmetic still yields plain tensors, and :func:`unfuse_module` puts the original
    class back. The getter has to be redefined too: a setter-only property would shadow
    ``Tensor.data``'s getter and make reads raise.
    """

    __slots__ = ()

    @property
    def data(self):  # noqa: D102
        return torch.Tensor.data.__get__(self)

    @data.setter
    def data(self, value):
        _bump_epoch()
        torch.Tensor.data.__set__(self, value)

    def set_(self, *a, **kw):  # noqa: D102
        _bump_epoch()
        return torch.Tensor.set_(self, *a, **kw)

    def resize_(self, *a, **kw):  # noqa: D102
        _bump_epoch()
        return torch.Tensor.resize_(self, *a, **kw)

    def detach_(self, *a, **kw):  # noqa: D102
        _bump_epoch()
        return torch.Tensor.detach_(self, *a, **kw)


# Built per concrete parameter class: Megatron and vLLM put Parameter SUBCLASSES on expert weights, and
# hard-coding ``nn.Parameter`` would leave those unwatched and silently disable fusing.
_WATCHED_CLASSES: dict[type, type] = {}


def _watched_class(cls: type) -> type:
    if issubclass(cls, _DataWatch):
        return cls
    w = _WATCHED_CLASSES.get(cls)
    if w is None:
        w = type(f"_DataWatched{cls.__name__}", (_DataWatch, cls), {"_ix_unwatched": cls})
        _WATCHED_CLASSES[cls] = w
    return w


# Bumped on every violation observed at read time; nonzero means the fused view was invalidated live.
VIOLATION_COUNT = 0

# Re-fuse accounting; dump with :func:`stats`. ALLOC_COUNT should settle at the number of
# (module, role) pairs -- if it keeps climbing, every sync is allocating a fresh large buffer.
ALLOC_COUNT = 0
REUSE_COUNT = 0
REFUSE_COUNT = 0
REFUSE_SKIPPED = 0


_LOG_FIRST_N = 8

# Set when a fuse actually called torch.empty; see release_stranded_segments.
_PENDING_RELEASE = [False]


def release_stranded_segments(force: bool = False) -> float:
    """Give the driver back the memory the first fuse stranded in the caching allocator.

    Per-expert weights come from the allocator's small pool and cannot satisfy the fused buffers'
    large-block requests, so the first fuse leaves a full copy's worth of small-pool segments reserved
    but unusable. Must be called at a sync boundary (via :func:`bump_sync_epoch`), never inside a
    forward. Returns the GiB returned to the driver.
    """
    if not (force or _PENDING_RELEASE[0]) or not torch.cuda.is_available():
        return 0.0
    _PENDING_RELEASE[0] = False
    before = torch.cuda.memory_reserved()
    gc.collect()
    torch.cuda.empty_cache()
    freed = (before - torch.cuda.memory_reserved()) / (1 << 30)
    logger.info("[isoexec-moe-fw] released %.2f GiB of post-fuse allocator fragmentation", freed)
    return freed


def stats() -> dict:
    """Counters for the live topology: how often the alias broke, and what re-fusing cost."""
    return {
        "violations": VIOLATION_COUNT,
        "refuses": REFUSE_COUNT,
        "refuses_rate_limited": REFUSE_SKIPPED,
        "buffer_allocs": ALLOC_COUNT,
        "buffer_reuses": REUSE_COUNT,
        "sync_epoch": SYNC_EPOCH,
    }


class _StackView(torch.autograd.Function):
    """``buf`` in, ``buf`` out (free alias); backward scatters the grad onto the E expert params.

    ``torch.stack``'s autograd behaviour without its forward copy. The params are passed as inputs only
    so autograd records the edges -- their values are never read, because their storage IS ``buf``.
    """

    @staticmethod
    def forward(ctx, buf, *params):  # noqa: D102
        # .view() (not the input object) so autograd owns a distinct output tensor, sharing storage.
        return buf.view(buf.shape)

    @staticmethod
    def backward(ctx, grad_out):  # noqa: D102
        # grad_out is [E, *param_shape]; unbind gives one view per expert, in the order params were passed.
        return (None, *grad_out.unbind(0))


class _RoleBuffer:
    """The fused buffer for one (module, role) plus everything needed to re-verify the invariant."""

    __slots__ = (
        "role",
        "buf",
        "params",
        "owners",
        "probes",
        "shape",
        "stride",
        "dtype",
        "device",
        "buf_ptr",
        "buf_bytes",
        "poisoned",
        "reason",
        "epoch",
        "owner_dicts",
        "pdicts",
        "pdict_ids",
        "classes",
    )

    def __init__(self, role, buf, params, owners):
        self.role = role
        self.buf = buf
        self.params = params  # list[nn.Parameter] -- the OBJECTS recorded at fuse time
        self.owners = owners  # list[(owning_module, attr_name)] -- kept for unfuse/diagnostics
        self.shape = buf[0].shape
        self.stride = buf[0].stride()
        self.dtype = buf.dtype
        self.device = buf.device
        self.buf_ptr = buf.data_ptr()
        self.buf_bytes = buf.untyped_storage().nbytes()
        self.poisoned = False
        self.reason = ""
        # Pre-zipped probe triples: (owning module's _parameters dict, recorded Parameter, expected
        # data_ptr). The hot check walks this rather than the module tree -- at E=256 the zip and the
        # getattr chains cost more than the check itself.
        self.probes = [(mod._parameters, p, buf[i].data_ptr()) for i, ((mod, _), p) in enumerate(zip(owners, params))]
        # Baselines for the O(1) fast path. `owner_dicts`/`pdicts` catch `mod._parameters` being
        # rebound to a fresh dict wholesale, which would leave every probe reading an orphaned dict.
        # `classes` catches a param's `__class__` being reverted, which would unhook the `.data` watcher.
        # `pdicts` holds strong references on purpose: `pdict_ids` compares by id(), which would be
        # unsound if a watched dict could be freed and a fresh one land at the same address.
        self.owner_dicts = [mod.__dict__ for mod, _ in owners]
        self.pdicts = [mod._parameters for mod, _ in owners]
        self.pdict_ids = [id(d) for d in self.pdicts]
        self.classes = [p.__class__ for p in params]
        self.epoch = _EPOCH


def _expert_modules(module, experts_attr: str) -> Sequence:
    experts = getattr(module, experts_attr, None)
    if experts is None:
        raise AttributeError(f"[isoexec-moe-fw] {type(module).__name__} has no '{experts_attr}'")
    return list(experts)


def _owner_of(expert, role: str):
    """(module that holds the Parameter, attr name). ``role`` may be dotted, e.g. 'linear_fc1'."""
    mod = expert
    for part in role.split("."):
        mod = getattr(mod, part)
    return mod, "weight"


def _full_check(rb: _RoleBuffer) -> bool:
    """The alias invariant, evaluated in full. ``True`` iff the fused view can be trusted.

    Order matters: the buffer's own storage is checked first, because if it moved (cumem sleep) the
    per-expert pointer comparisons would compare two addresses that both moved and could agree by
    accident. Covers per expert: object replacement, storage rebind, shape drift, dtype drift; stride
    and device are checked only at fuse time and by ``check_module(strict=True)``.
    """
    buf = rb.buf
    if buf.data_ptr() != rb.buf_ptr or buf.untyped_storage().nbytes() != rb.buf_bytes:
        return False
    shape, dtype = rb.shape, rb.dtype
    for pdict, p, ptr in rb.probes:
        cur = pdict.get("weight")
        if cur is not p or cur.data_ptr() != ptr or cur.shape != shape or cur.dtype is not dtype:
            return False
    return True


def _check(rb: _RoleBuffer) -> bool:
    """Hot-path check: same verdict as :func:`_full_check`, much cheaper in the steady state.

    Verifies that the buffer still owns its storage, that every owning module still holds the
    ``_WatchedParamDict`` installed at fuse time, and that every recorded Parameter is still a
    ``_DataWatch``. With the watchers proven intact, an unchanged ``_EPOCH`` proves no parameter was
    rebound since the last full verification; otherwise the full scan runs and re-baselines.
    """
    buf = rb.buf
    if buf.data_ptr() != rb.buf_ptr or buf.untyped_storage().nbytes() != rb.buf_bytes:
        return False
    if [id(d["_parameters"]) for d in rb.owner_dicts] != rb.pdict_ids:
        return False
    if [p.__class__ for p in rb.params] != rb.classes:
        return False
    if rb.epoch == _EPOCH:
        return True
    if not _full_check(rb):
        return False
    rb.epoch = _EPOCH
    return True


def _explain(rb: _RoleBuffer, *, strict: bool = False) -> str:
    """Why ``_check`` said no (or, under ``strict``, also stride/device). Cold path only."""
    buf = rb.buf
    if buf.data_ptr() != rb.buf_ptr:
        return "fused buffer moved (data_ptr changed)"
    if buf.untyped_storage().nbytes() != rb.buf_bytes:
        return "fused buffer storage resized/freed (cumem sleep?)"
    for i, (d, pd) in enumerate(zip(rb.owner_dicts, rb.pdicts)):
        if d.get("_parameters") is not pd:
            return f"expert {i}: module._parameters REPLACED wholesale (mutation watch lost)"
    for i, (p, cls) in enumerate(zip(rb.params, rb.classes)):
        if p.__class__ is not cls:
            return f"expert {i}: Parameter __class__ changed to {p.__class__.__name__} (.data watch lost)"
    for i, (pdict, p, ptr) in enumerate(rb.probes):
        cur = pdict.get("weight")
        if cur is None:
            return f"expert {i}: Parameter REMOVED from its module"
        if cur is not p:
            # _set_on_module / any `setattr(mod, 'weight', Parameter(...))`
            return f"expert {i}: Parameter OBJECT replaced (weight-sync materialisation?)"
        if cur.data_ptr() != ptr:
            # `param.data = ...` -- optimizer param buffers, checkpoint load, meta materialisation
            return f"expert {i}: .data rebound to foreign storage (ptr {cur.data_ptr()} != {ptr})"
        if cur.shape != rb.shape:
            return f"expert {i}: shape drift ({tuple(cur.shape)} != {tuple(rb.shape)})"
        if cur.dtype is not rb.dtype:
            return f"expert {i}: dtype drift ({cur.dtype} != {rb.dtype})"
        if strict and (cur.stride() != rb.stride or cur.device != rb.device):
            return f"expert {i}: stride/device drift ({cur.stride()}/{cur.device})"
    return ""


def _state(module) -> dict | None:
    return getattr(module, _STATE_ATTR, None)


def _alloc_sleepable(shape, dtype, device):
    """Allocate the fused buffer inside vLLM's sleepable CuMem pool when sleep mode is live.

    A plain ``torch.empty`` lands outside the sleepable pool, so level-1 sleep cannot offload it and
    the expert weights stay resident through the whole training window. ``use_memory_pool`` also gives
    the virtual-address stability vLLM's graph-captured weights rely on. ``CuMemAllocator.instance`` is
    read without ``get_instance()`` on purpose: the getter would create the singleton (and its
    free-callback hooks) in processes that never enabled sleep mode.
    """
    try:
        from vllm.device_allocator.cumem import CuMemAllocator

        alloc = CuMemAllocator.instance
        if alloc is not None:
            with alloc.use_memory_pool(tag="weights"):
                return torch.empty(shape, dtype=dtype, device=device)
    except Exception as e:  # pragma: no cover - vLLM absent or pool unavailable
        logger.info("[isoexec-moe-fw] sleepable-pool alloc unavailable (%s); plain alloc", e)
    return torch.empty(shape, dtype=dtype, device=device)


def fuse_module(
    module,
    roles: Iterable[str] = DEFAULT_ROLES,
    *,
    experts_attr: str = "local_experts",
    force: bool = False,
) -> dict[str, torch.Tensor] | None:
    """Fuse each expert's ``<role>.weight`` into one ``[E, *shape]`` buffer and re-point the params.

    Idempotent: a second call on an already-fused, still-valid module is a no-op returning the
    existing buffers. Returns ``None`` when the module cannot be fused (fewer than 2 experts, ragged
    shapes/dtypes/devices, meta params) -- callers keep ``torch.stack``.

    ``force`` re-fuses even if a previous fuse is poisoned; see :func:`refuse_module` for why that is
    not the default. A forced re-fuse REUSES the buffers of the previous fuse whenever they are still
    intact and the right shape/dtype/device, so re-fusing allocates memory exactly once per module for
    the life of the process (see the note at the allocation site).
    """
    global ALLOC_COUNT, REUSE_COUNT
    experts = _expert_modules(module, experts_attr)
    if len(experts) < 2:
        return None

    state = _state(module)
    if state is not None and not force:
        out = {}
        for role in roles:
            rb = state.get(role)
            if rb is None or rb.poisoned:
                return None
            if not _check(rb):
                rb.poisoned, rb.reason = True, _explain(rb)
                return None
            out[role] = rb.buf
        return out

    old_state = state
    state = {}
    for role in roles:
        owners = [_owner_of(e, role) for e in experts]
        params = []
        for mod, attr in owners:
            p = mod._parameters.get(attr)
            if p is None or p.is_meta or p.device.type == "meta":
                return None
            params.append(p)

        ref = params[0].data
        if any(
            p.data.shape != ref.shape or p.data.dtype is not ref.dtype or p.data.device != ref.device for p in params
        ):
            logger.warning("[isoexec-moe-fw] ragged expert weights for role %s -- not fusing", role)
            return None

        E = len(params)
        # Reuse the existing buffer when there is one, so after the first fuse this path allocates
        # nothing: a re-fuse happens once per weight sync per (layer, role), and allocating a fresh
        # large buffer each time would fragment a GPU shared with the trainer.
        old = old_state.get(role) if old_state else None
        buf = None
        if (
            old is not None
            and old.buf.shape == (E, *ref.shape)
            and old.buf.dtype is ref.dtype
            and old.buf.device == ref.device
            and old.buf.data_ptr() == old.buf_ptr
            and old.buf.untyped_storage().nbytes() == old.buf_bytes
        ):
            buf = old.buf
            REUSE_COUNT += 1
        if buf is None:
            buf = _alloc_sleepable((E, *ref.shape), ref.dtype, ref.device)
            ALLOC_COUNT += 1
            _PENDING_RELEASE[0] = True

        # Copy-then-rebind one expert at a time: rebinding drops the last reference to that expert's
        # old storage, so the transient peak is (fused buffer + one expert), not 2x the expert weights.
        with torch.no_grad():
            for i, p in enumerate(params):
                slot = buf[i]
                # On the reuse path most experts still alias their own slot; skip the self-copy.
                cur = p.data
                if cur.data_ptr() != slot.data_ptr():
                    slot.copy_(cur)
                    p.data = slot
                elif cur.shape != slot.shape or cur.stride() != slot.stride() or cur.dtype is not slot.dtype:
                    # Same address, wrong view: the bytes are ours already, only the metadata is stale.
                    p.data = slot

        # Arm the mutation watch AFTER the rebinds above -- they would each bump the epoch otherwise.
        for (mod, _), p in zip(owners, params):
            d = mod.__dict__.get("_parameters")
            if not isinstance(d, _WatchedParamDict):
                mod.__dict__["_parameters"] = _WatchedParamDict(d)
            p.__class__ = _watched_class(p.__class__)

        state[role] = _RoleBuffer(role, buf, params, owners)

    setattr(module, _STATE_ATTR, state)
    # Register for the sync-boundary eager refresh; the WeakSet dedupes and drops modules on teardown.
    _FUSED_MODULES.add(module)
    return {role: rb.buf for role, rb in state.items()}


def fused_expert_weights(module, role: str) -> torch.Tensor | None:
    """The hot-path read: the ready-made ``[E, *shape]`` stack, or ``None`` if it cannot be trusted.

    ``None`` means fall back to ``torch.stack([e.<role>.weight for e in module.local_experts])``, which
    is always correct. Under grad the result is autograd-connected to the E expert params (see
    :class:`_StackView`), so expert weight gradients land where ``torch.stack`` put them.
    """
    global VIOLATION_COUNT

    state = _state(module)
    if state is None:
        return None
    rb = state.get(role)
    if rb is None or rb.poisoned:
        return None

    if not _check(rb):
        why = _explain(rb)
        rb.poisoned, rb.reason = True, why
        VIOLATION_COUNT += 1
        # Rate-limited: a weight sync breaks the alias in every layer, so this would print per role.
        if VIOLATION_COUNT <= _LOG_FIRST_N:
            print(
                f"[ISOEXEC-MOE-FW] fused expert weights INVALIDATED for role={role}: {why}. "
                f"Falling back to torch.stack for this module (re-fuse is rate-limited to one per "
                f"weight sync -- see refuse_if_synced). violation #{VIOLATION_COUNT}",
                flush=True,
            )
        return None

    if not torch.is_grad_enabled():
        return rb.buf
    if not any(p.requires_grad for p in rb.params):
        return rb.buf
    return _StackView.apply(rb.buf, *rb.params)


def check_module(module, role: str, *, strict: bool = False) -> tuple[bool, str]:
    """Diagnostics: run the validity check WITHOUT poisoning or logging. (ok, reason).

    ``strict`` additionally verifies stride and device, which the hot check omits for cost.
    """
    state = _state(module)
    if state is None:
        return False, "not fused"
    rb = state.get(role)
    if rb is None:
        return False, f"role {role} not fused"
    if rb.poisoned:
        return False, f"poisoned: {rb.reason}"
    if not _check(rb):
        return False, _explain(rb)
    if strict:
        why = _explain(rb, strict=True)
        if why:
            return False, why
    return True, ""


def refuse_module(module, roles: Iterable[str] = DEFAULT_ROLES, *, experts_attr: str = "local_experts"):
    """Explicitly re-fuse after a break the caller has determined is benign.

    Deliberately not automatic: a violation can mean a weight sync materialised a meta param (benign)
    or that the distributed optimizer re-pointed the param into its own buffer (re-fusing there would
    sever the optimizer's aliasing). This must NOT drop the previous state first --
    ``fuse_module(force=True)`` reads it to reuse the existing buffers.
    """
    global REFUSE_COUNT
    REFUSE_COUNT += 1
    return fuse_module(module, roles, experts_attr=experts_attr, force=True)


# Sync epoch: re-fusing is rate-limited to the only event that legitimately causes a break, a completed
# weight sync announcing itself via :func:`bump_sync_epoch`. Without the limit, anything that breaks the
# alias every forward would make every forward pay a full re-fuse. If nothing ever bumps the epoch the
# module fuses once and then falls back to torch.stack permanently -- the safe direction to fail.
SYNC_EPOCH = 0


def refresh_all_fused(roles: Iterable[str] = DEFAULT_ROLES, *, experts_attr: str = "local_experts") -> int:
    """Eagerly re-point every fused module's params into its fixed-address buffer, at the sync boundary.

    A CUDA-graph decode is a replay, so the lazy per-forward refresh never runs and the captured GEMM
    keeps reading the buffer address baked at capture time. This copies fresh bytes through the params
    into that same buffer, outside any forward. Graph safety is a hard invariant: the buffer's
    ``data_ptr`` must never move and no fresh buffer may be allocated across a refresh, so a refresh
    that would reallocate raises. Returns the number of modules refreshed.
    """
    n = 0
    for module in list(_FUSED_MODULES):
        state = _state(module)
        if state is None:
            continue
        pre_ptrs = {role: rb.buf.data_ptr() for role, rb in state.items()}
        pre_alloc = ALLOC_COUNT
        out = fuse_module(module, roles, experts_attr=experts_attr, force=True)
        if out is None:
            # Alias could not be re-established; ``buf`` keeps its last-refreshed bytes, so a captured
            # graph replaying now would read stale weights.
            logger.error(
                "[isoexec-moe-fw] EAGER REFRESH FAILED for %s: fuse_module returned None at the sync "
                "boundary. A captured decode graph may now read STALE expert weights.",
                type(module).__name__,
            )
            continue
        for role, buf in out.items():
            if buf.data_ptr() != pre_ptrs.get(role):
                raise RuntimeError(
                    f"[isoexec-moe-fw] GRAPH-UNSAFE eager refresh: fused buffer for role={role} MOVED "
                    f"({pre_ptrs.get(role)} -> {buf.data_ptr()}) on {type(module).__name__}. A captured "
                    f"decode graph bakes this address; reallocating it under replay reads foreign memory."
                )
        if ALLOC_COUNT != pre_alloc:
            raise RuntimeError(
                f"[isoexec-moe-fw] GRAPH-UNSAFE eager refresh: a fused buffer was REALLOCATED "
                f"(ALLOC_COUNT {pre_alloc} -> {ALLOC_COUNT}) on {type(module).__name__} -- expert "
                f"shape/dtype/device drifted across a sync. The captured graph still points at the OLD "
                f"buffer, so the replay would read stale/foreign memory."
            )
        n += 1
    return n


def bump_sync_epoch() -> int:
    """Announce that a weight sync finished, licensing one re-fuse per fused module.

    Called from the engine's weight-update paths, which are where Parameter objects get replaced.
    """
    global SYNC_EPOCH
    SYNC_EPOCH += 1
    # moe_weight_cache memoizes the stack on "the weights did not change"; a sync boundary is by
    # definition when that is false. sys.modules, not an import: no cache exists if it was never imported.
    _wcache = sys.modules.get(__package__ + ".moe_weight_cache")
    if _wcache is not None:
        _wcache.invalidate_all()
    # Same seam: the router's fp32 cast lives in a fixed-address buffer a captured decode graph reads
    # directly, and invalidates by re-casting IN PLACE here, before the next forward.
    _rcache = sys.modules.get(__package__ + ".moe_router_cast_cache")
    if _rcache is not None:
        _rcache.invalidate_all(reason=f" (sync epoch {SYNC_EPOCH})")
    # The correctness half of the sync boundary: re-point every fused module's params into its
    # fixed-address buffer now, so a replayed CUDA-graph decode never reads stale capture-time bytes.
    # ``refuse_if_synced`` remains the eager-mode (uncaptured) fallback.
    refresh_all_fused()
    # A sync boundary is guaranteed not to be inside a forward, so release fragmentation here.
    release_stranded_segments()
    return SYNC_EPOCH


def refuse_if_synced(
    module,
    roles: Iterable[str] = DEFAULT_ROLES,
    *,
    experts_attr: str = "local_experts",
) -> dict[str, torch.Tensor] | None:
    """:func:`refuse_module`, but at most once per :data:`SYNC_EPOCH` per module.

    ``None`` means "not this time" as well as "cannot fuse" -- both leave the caller on
    ``torch.stack``, which is always correct.
    """
    global REFUSE_SKIPPED
    if module.__dict__.get("_isoexec_fw_refused_at") == SYNC_EPOCH and _state(module) is not None:
        REFUSE_SKIPPED += 1
        return None
    module.__dict__["_isoexec_fw_refused_at"] = SYNC_EPOCH
    return refuse_module(module, roles, experts_attr=experts_attr)


def unfuse_module(module, *, experts_attr: str = "local_experts") -> bool:
    """Give every expert back its own storage (a clone of its slice) and drop the buffer.

    Costs one full copy of the expert weights and transiently 2x their memory; teardown/A-B only.
    """
    state = _state(module)
    if state is None:
        return False
    with torch.no_grad():
        for rb in state.values():
            for (mod, attr), p in zip(rb.owners, rb.params):
                cur = mod._parameters.get(attr)
                if cur is None:
                    continue
                cur.data = cur.data.clone()
                # Disarm the watch so an unfused module is indistinguishable from an untouched one.
                cur.__class__ = getattr(cur.__class__, "_ix_unwatched", cur.__class__)
                d = mod.__dict__.get("_parameters")
                if isinstance(d, _WatchedParamDict):
                    mod.__dict__["_parameters"] = dict(d)
    delattr(module, _STATE_ATTR)
    return True
