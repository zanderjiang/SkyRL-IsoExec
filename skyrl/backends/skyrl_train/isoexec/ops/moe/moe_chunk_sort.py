"""One-gather ``sort_chunks_by_idxs`` for the alltoall dispatcher (``SKYRL_ISOEXEC_MOE_CHUNK_SORT``, OFF).

Megatron's non-fused ``sort_chunks_by_idxs`` splits the input into ``num_splits`` chunks and ``torch.cat``s
them in a new order. That reordering is a bijection on rows, so the identical bytes are reachable as a
single 1-D row gather: for output row ``r`` in output chunk ``j``,
``src(r) = in_starts[sorted_idxs[j]] - out_starts[j] + r``, so
``gather_map = repeat_interleave(base, out_sizes) + arange(n)`` and the whole op is one ``index_select``.
Both forms are pure copies, hence bit-equal by construction, signed zeros and NaN payloads included.

The backward is why this op and not ``permute``: because the map is a bijection the VJP is the inverse
GATHER, never an ``index_add_``. That distinction is real -- ``index_add_`` into a zero buffer would flip
signed zeros (``0.0 + (-0.0) == +0.0``), while two gathers cannot.

Admission is keyed on ``(num_splits, hidden, dtype)`` and earned once at first use on the live operands:
forward bit-equality against the ``cat`` expression, provenance (the map applied to a row-index marker
reproduces the reference row ordering), backward bit-equality, a round trip through the inverse map, and
determinism. The row count is deliberately NOT part of the key: it is data-dependent and effectively unique
per call, so keying on it would run the probe on nearly every call. What each call re-establishes instead
are two host-side integer preconditions on ``num_splits``-long vectors -- ``split_sizes.sum() ==
num_rows``, and ``sorted_idxs`` is a permutation of ``arange(num_splits)``. A failed precondition runs
megatron's ``cat`` for that one call and does not poison the shape.

Scope is only ``sort_chunks_by_idxs``, i.e. only ``MoEAlltoAllTokenDispatcher`` at EP>1. ``permute`` and
``unpermute`` are untouched: their expressions involve accumulation, not only a copy.
"""

from __future__ import annotations

import os

import torch

_ENV_GATE = "SKYRL_ISOEXEC_MOE_CHUNK_SORT"
BANNER = "[ISOEXEC-MOE-CHUNK-SORT]"

# Admission verdicts, keyed by (num_splits, hidden, dtype). True = admitted; str = the reason it was
# rejected. A verdict is a statement about THIS build on THIS device, so it is earned once at first use
# and never re-probed. The key carries NO data-dependent axis -- see the module docstring.
_STATE: dict[tuple, object] = {}

# A bound on the table. Real configs produce a handful of keys (one per dtype the dispatcher runs);
# anything approaching this is a bug, and an unbounded dict on a call this hot is a leak.
_STATE_CAP = 256
_state_full_reported = False

# The `arange(num_splits)` the per-call permutation check compares against, memoised per
# (k, device, dtype). Rebuilding it cost ~2 us of the ~20 us precondition budget.
_ARANGE: dict[tuple, torch.Tensor] = {}

# Engagement census. `served` counts calls that actually ran the gather; `declined` counts calls that fell
# through to megatron's cat for ANY reason.
_served = 0
_declined = 0
_reported = 0
_decline_reason = ""

# Rejection prints are rate-limited: a per-call precondition refusal could otherwise emit ~20k lines
# per rank per forward.
_reject_prints = 0

_installed = False
_orig_sort_chunks = None


def chunk_sort_enabled() -> bool:
    return os.environ.get(_ENV_GATE, "0") == "1"


def chunk_sort_stats() -> tuple[int, int, str]:
    """``(served, declined, last_decline_reason)`` -- for tests, the battery and the live arm."""
    return _served, _declined, _decline_reason


def _maybe_report() -> None:
    """One line at the first call and then at 1/2/4/8/... python calls into the wrapper."""
    global _reported
    total = _served + _declined
    if total < 1 or (total & (total - 1)) != 0 or total == _reported:
        return
    _reported = total
    print(
        f"{BANNER} pid={os.getpid()} served={_served} declined={_declined} "
        f"(python calls into sort_chunks_by_idxs)" + (f" last_decline={_decline_reason}" if _declined else ""),
        flush=True,
    )


def _print_reject(msg: str) -> None:
    """Loud, but bounded: the first four, then powers of two. ~15 lines per rank at 20k calls."""
    global _reject_prints
    _reject_prints += 1
    n = _reject_prints
    if n <= 4 or (n & (n - 1)) == 0:
        print(f"{BANNER} REJECTED[#{n}] {msg} megatron's torch.cat stays in charge (no bits move).", flush=True)


# The map. Pure integer arithmetic on `num_splits` elements -- no float, no D2H, CPU-testable.
def gather_map(split_sizes: torch.Tensor, sorted_idxs: torch.Tensor, num_tokens: int, device) -> torch.Tensor:
    """Row map with ``output[r] = input[gather_map[r]]`` for megatron's chunk reordering.

    ``split_sizes`` / ``sorted_idxs`` may live on either device: megatron's alltoall dispatcher pre-stages
    the per-expert counts to the host when permute fusion is off, so on the live arm they arrive on CPU.
    The offsets are computed wherever they already are and only the two ``num_splits``-long vectors cross
    to the device, so this adds no synchronisation in either layout.
    """
    ss = split_sizes.reshape(-1).to(torch.int64)
    si = sorted_idxs.reshape(-1).to(torch.int64)
    if si.device != ss.device:
        si = si.to(ss.device)
    in_starts = torch.cumsum(ss, 0) - ss
    out_sizes = ss[si]
    out_starts = torch.cumsum(out_sizes, 0) - out_sizes
    base = in_starts[si] - out_starts
    if base.device != device:
        base = base.to(device, non_blocking=True)
        out_sizes = out_sizes.to(device, non_blocking=True)
    g = torch.repeat_interleave(base, out_sizes, output_size=int(num_tokens))
    return g + torch.arange(int(num_tokens), device=device, dtype=torch.int64)


def inverse_map(g: torch.Tensor) -> torch.Tensor:
    """``inv[g[r]] = r``. Defined only because ``g`` is a bijection -- gate (iv) proves it is."""
    inv = torch.empty_like(g)
    inv.index_copy_(0, g, torch.arange(g.numel(), device=g.device, dtype=g.dtype))
    return inv


class _ChunkSortGather(torch.autograd.Function):
    """Forward gather, backward inverse gather. No add anywhere, in either direction."""

    @staticmethod
    def forward(ctx, inp: torch.Tensor, g: torch.Tensor):  # type: ignore[override]
        ctx.save_for_backward(g)
        return inp.index_select(0, g)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        (g,) = ctx.saved_tensors
        return grad_out.index_select(0, inverse_map(g)), None


def _gathered(inp: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    if inp.requires_grad or torch.is_grad_enabled():
        return _ChunkSortGather.apply(inp, g)
    return inp.index_select(0, g)


def sort_chunks_gather(inp, split_sizes, sorted_idxs, probs=None):
    """The replacement expression. Returns ``(output, permuted_probs)`` like megatron's."""
    g = gather_map(split_sizes, sorted_idxs, inp.shape[0], inp.device)
    out = _gathered(inp, g)
    pp = _gathered(probs, g) if probs is not None else None
    return out, pp


def _reference(inp, split_sizes, sorted_idxs, probs=None):
    """Megatron ``moe_utils.sort_chunks_by_idxs``'s non-fused branch, transcribed verbatim."""
    parts = torch.split(inp, split_sizes.reshape(-1).tolist(), dim=0)
    idxs = sorted_idxs.reshape(-1).tolist()
    out = torch.cat([parts[i] for i in idxs], dim=0)
    pp = None
    if probs is not None:
        pparts = torch.split(probs, split_sizes.reshape(-1).tolist(), dim=0)
        pp = torch.cat([pparts[i] for i in idxs], dim=0)
    return out, pp


# Admission
def _bitcmp(a: torch.Tensor, b: torch.Tensor) -> int:
    """Number of differing elements, compared as BITS (so -0.0 != 0.0 and NaN payloads count)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return -1
    n = a.numel()
    if n == 0:
        return 0
    ia = (
        a.detach()
        .contiguous()
        .view(
            torch.int8
            if a.element_size() == 1
            else torch.int16 if a.element_size() == 2 else torch.int32 if a.element_size() == 4 else torch.int64
        )
    )
    ib = b.detach().contiguous().view(ia.dtype)
    return int((ia != ib).sum().item())


def _admit(inp, split_sizes, sorted_idxs, probs) -> tuple[bool, str]:
    n = int(inp.shape[0])
    with torch.no_grad():
        g = gather_map(split_sizes, sorted_idxs, n, inp.device)
        x = inp.detach()
        p = probs.detach() if probs is not None else None

        # (i) forward bits
        ref, refp = _reference(x, split_sizes, sorted_idxs, p)
        got = x.index_select(0, g)
        bad = _bitcmp(ref, got)
        if bad != 0:
            return False, f"forward differs from the cat expression ({bad} elements)"
        if p is not None:
            badp = _bitcmp(refp, p.index_select(0, g))
            if badp != 0:
                return False, f"permuted probs differ ({badp} elements)"

        # (ii) provenance -- exact row ordering, not merely equal payloads
        marker = torch.arange(n, device=inp.device, dtype=torch.int64).unsqueeze(1)
        mref, _ = _reference(marker, split_sizes, sorted_idxs, None)
        if not torch.equal(mref.reshape(-1), g):
            return False, "row provenance differs: the gather map is not megatron's permutation"

        # (iv) round trip -- the map is a bijection
        if not torch.equal(got.index_select(0, inverse_map(g)), x):
            return False, "round trip through the inverse map is not the identity"

        # (v) determinism
        if _bitcmp(got, x.index_select(0, g)) != 0:
            return False, "two identical gathers disagree"

    # (iii) backward bits, autograd through BOTH forms with one shared cotangent.
    #
    # `enable_grad` is LOAD-BEARING. Every MoE forward on this stack runs under `torch.no_grad()`:
    # scoring wraps the whole forward, and training does too, because the recipe pins full recompute so
    # every layer enters through megatron's `CheckpointFunction.forward`. Without this block
    # `torch.autograd.backward` raises "element 0 of tensors does not require grad", the caller caches a
    # PERMANENT rejection for the shape, and by the time the backward recompute re-runs the layer under
    # `enable_grad` the key is already poisoned.
    #
    # It is safe inside a caller's no_grad: the probe operates on detached clones, so the graph it builds
    # is private to these tensors and freed when they go out of scope. It cannot attach anything to the
    # caller's activations.
    if inp.is_floating_point():
        with torch.enable_grad():
            xa = inp.detach().clone().requires_grad_(True)
            xb = inp.detach().clone().requires_grad_(True)
            pa = (
                probs.detach().clone().requires_grad_(True)
                if (probs is not None and probs.is_floating_point())
                else None
            )
            pb = probs.detach().clone().requires_grad_(True) if pa is not None else None
            ra, rap = _reference(xa, split_sizes, sorted_idxs, pa)
            gb, gbp = sort_chunks_gather(xb, split_sizes, sorted_idxs, pb)
            cot = torch.randn_like(ra)
            cotp = torch.randn_like(rap) if rap is not None else None
            outs_a, outs_b = [ra], [gb]
            cots = [cot]
            if rap is not None:
                outs_a.append(rap)
                outs_b.append(gbp)
                cots.append(cotp)
            torch.autograd.backward(outs_a, cots, retain_graph=False)
            torch.autograd.backward(outs_b, cots, retain_graph=False)
        if xa.grad is None or xb.grad is None:
            return False, "backward produced no input gradient"
        badg = _bitcmp(xa.grad, xb.grad)
        if badg != 0:
            return False, f"input gradient differs from the cat expression's VJP ({badg} elements)"
        if pa is not None:
            badgp = _bitcmp(pa.grad, pb.grad)
            if badgp != 0:
                return False, f"probs gradient differs ({badgp} elements)"

    return True, ""


def admission_key(inp, split_sizes) -> tuple:
    """``(num_splits, hidden, dtype)``. Deliberately free of any data-dependent axis.

    ``numel()`` / ``shape`` / ``dtype`` are all tensor METADATA, so building this key reads nothing
    from either device buffer and cannot synchronise.
    """
    return (int(split_sizes.numel()), int(inp.shape[1]), str(inp.dtype))


def preconditions_ok(split_sizes, sorted_idxs, num_rows: int) -> tuple[bool, str]:
    """The two integer facts gate (ii)'s algebra needs from EVERY call, as host arithmetic.

    Gate (ii) proved on live operands that the gather map IS megatron's permutation for this
    ``(num_splits, hidden, dtype)``. The derivation ``src(r) = in_starts[sorted_idxs[j]] - out_starts[j]
    + r`` carries over to any other call at the same shape exactly when the chunk vectors still tile the
    input and still name each chunk once, so those are what a call re-establishes rather than a second
    reference ``cat``:

      1. ``split_sizes.sum() == num_rows`` (and no negative chunk), so the chunks tile the rows;
      2. ``sorted_idxs`` is a permutation of ``arange(num_splits)``, so the map is a bijection.

    On the live layout both vectors are already on the host, so this is host integer work and adds NO
    device sync. In the device layout the reductions do read device memory, which is still not a
    regression: the megatron path it replaces calls ``.tolist()`` on both vectors, i.e. two D2H syncs
    against our one.
    """
    ss = split_sizes if split_sizes.dim() == 1 else split_sizes.reshape(-1)
    si = sorted_idxs if sorted_idxs.dim() == 1 else sorted_idxs.reshape(-1)
    k = ss.numel()
    if k == 0 or si.numel() != k:
        return False, f"split_sizes/sorted_idxs length mismatch ({k} vs {si.numel()})"
    total = int(ss.sum())
    if total != int(num_rows):
        return False, f"split_sizes.sum()={total} does not tile num_rows={int(num_rows)}"
    if int(ss.min()) < 0:
        return False, "split_sizes has a negative chunk"
    ck = (k, si.device, si.dtype)
    ref = _ARANGE.get(ck)
    if ref is None:
        ref = torch.arange(k, device=si.device, dtype=si.dtype)
        if len(_ARANGE) < 64:  # bounded, like _STATE: a handful of (k, device, dtype) per rank
            _ARANGE[ck] = ref
    if not torch.equal(torch.sort(si).values, ref):
        return False, "sorted_idxs is not a permutation of arange(num_splits)"
    return True, ""


def chunk_sort_ready(inp, split_sizes, sorted_idxs, probs) -> bool:
    """Is the gather form admitted for this call? Fail-closed, loud, probed once per SHAPE.

    Two layers: the five-gate probe runs once per ``(num_splits, hidden, dtype)`` on the live operands,
    and every subsequent call pays only ``preconditions_ok`` -- host integers on ``num_splits`` elements.

    A precondition failure does NOT write the shape off. It is a statement about the chunk vectors of THIS
    call, so it refuses this call and leaves the shape's verdict alone; the next well-formed call at the
    same shape is still served. Only the probe's own verdict is cached, and only ever once.
    """
    global _decline_reason, _state_full_reported
    if not chunk_sort_enabled():
        _decline_reason = "flag off"
        return False
    if inp is None or inp.dim() != 2 or inp.numel() == 0:
        _decline_reason = "input is not a non-empty 2-D tensor"
        return False
    if getattr(inp.device, "type", None) != "cuda":
        _decline_reason = "input is not on cuda"
        return False

    # A shape already written off is settled from METADATA alone -- do not pay the preconditions to
    # rediscover it on every one of the ~20k calls.
    key = admission_key(inp, split_sizes)
    state = _STATE.get(key)
    if isinstance(state, str):
        _decline_reason = state
        return False

    ok, why = preconditions_ok(split_sizes, sorted_idxs, inp.shape[0])
    if not ok:
        _decline_reason = why
        _print_reject(f"precondition: {why}. This CALL only -- the shape's verdict is untouched.")
        return False

    if state is True:
        return True
    if torch.cuda.is_current_stream_capturing():
        _decline_reason = "stream capture in progress"
        return False  # never let a captured step be the thing that decides a shape
    if len(_STATE) >= _STATE_CAP:
        _decline_reason = "admission table full"
        if not _state_full_reported:
            _state_full_reported = True
            print(
                f"{BANNER} admission table reached its {_STATE_CAP}-shape cap; further shapes run "
                f"megatron's torch.cat unadmitted. A real config produces a handful of keys, so this "
                f"means the key has picked up a data-dependent axis again.",
                flush=True,
            )
        return False

    try:
        ok, why = _admit(inp, split_sizes, sorted_idxs, probs)
    except Exception as e:  # noqa: BLE001
        ok, why = False, f"admission raised ({type(e).__name__}: {e})"
    if ok:
        _STATE[key] = True
        print(
            f"{BANNER} ADMITTED splits={key[0]} hidden={key[1]} dtype={key[2]} "
            f"(probed at rows={int(inp.shape[0])}): one index_select is bit-equal to the cat "
            f"expression forward AND backward, provenance-exact, bijective and deterministic. The "
            f"row count is NOT part of the key -- per-call it is a precondition, not a shape. Read "
            f"the served= census for engagement, not this line.",
            flush=True,
        )
    else:
        _STATE[key] = why
        _decline_reason = why
        _print_reject(f"splits={key[0]} hidden={key[1]} dtype={key[2]}: {why}. Permanent for this shape.")
    return ok


# Installation
def install_chunk_sort() -> bool:
    """Rebind ``token_dispatcher.sort_chunks_by_idxs``. Idempotent; a no-op unless the flag is on.

    The dispatcher resolves the name through its own module global, so rebinding it there covers
    every call site (``dispatch_postprocess`` and ``combine_preprocess``) without touching megatron's
    source. The wrapper delegates to megatron whenever ``fused=True`` is requested (it never is on
    this stack -- ``force_isoexec_moe_config`` pins ``moe_permute_fusion=False``), whenever the flag is
    off, and whenever the shape has not been admitted. So flag-OFF is byte-for-byte today's path, and
    the module is inert without it.
    """
    global _installed, _orig_sort_chunks
    if _installed:
        return True
    if not chunk_sort_enabled():
        return False
    from megatron.core.transformer.moe import token_dispatcher as _td

    _orig_sort_chunks = _td.sort_chunks_by_idxs

    def sort_chunks_by_idxs(input, split_sizes, sorted_idxs, probs=None, fused=False):
        global _served, _declined, _decline_reason
        if fused:
            return _orig_sort_chunks(input, split_sizes, sorted_idxs, probs=probs, fused=fused)
        try:
            if chunk_sort_ready(input, split_sizes, sorted_idxs, probs):
                out = sort_chunks_gather(input, split_sizes, sorted_idxs, probs)
                _served += 1
                _maybe_report()
                return out
        except Exception as e:  # noqa: BLE001 -- loud, never fatal
            key = admission_key(input, split_sizes)
            _decline_reason = f"{type(e).__name__}: {e}"
            if _STATE.get(key) is True:
                _STATE[key] = f"demoted: {type(e).__name__}: {e}"
                print(
                    f"{BANNER} DEMOTED splits={key[0]} hidden={key[1]} dtype={key[2]}: "
                    f"{type(e).__name__}: {e}. Falling back to torch.cat permanently for this shape.",
                    flush=True,
                )
        _declined += 1
        _maybe_report()
        return _orig_sort_chunks(input, split_sizes, sorted_idxs, probs=probs, fused=fused)

    _td.sort_chunks_by_idxs = sort_chunks_by_idxs
    _installed = True
    print(
        f"{BANNER} INSTALLED impl=index_select (megatron's torch.cat chunk loop replaced by ONE row "
        f"gather; bit-equality admitted per (splits, hidden, dtype) on live operands, forward and "
        f"backward, with the per-call row count as a PRECONDITION and not a key). TransformerEngine "
        f"is NOT installed and is NOT required. THIS LINE IS "
        f"NOT ENGAGEMENT: watch for the served= census that follows.",
        flush=True,
    )
    return True
