"""Order-invariant tensor digests for debug-mode region tracing.

The digest must be computed on-GPU with no device-to-host copy of the tensor (only the final
8-byte scalar crosses), must be bitwise-exact on equality, and must not depend on the reduction
tree that computes it. Float reductions fail the last requirement by construction; integer
addition mod 2**64 is associative and commutative, so bitcasting the tensor to integers and
reducing with wrapping int64 arithmetic gives a digest that is identical under any chunking,
any reduction order, any device. (Verified on this torch: int64 add/mul wrap two's-complement
on CPU and CUDA, and ``sum`` over arbitrary splits is exact mod 2**64.)

Schemes considered:

  (a) plain bitcast + int64 sum -- order-invariant but permutation-blind (commutative sum) and
      low-entropy (a +1/-1 pair of flips cancels). Rejected.
  (b) weighted modular sum -- ``digest = sum_i w_i * x_i mod 2**64`` with ``w_i`` a fixed
      pseudorandom stream indexed by *absolute position*, forced odd. Position-sensitive, one
      fused pass, chunking-invariant because the weights are position-keyed, and because every
      ``w_i`` is odd (hence invertible mod 2**64) ANY single-position difference changes the
      digest deterministically, not probabilistically. Multi-position collisions require
      ``sum w_i d_i == 0 mod 2**64``: ~2**-64 for pseudorandom weights, non-adversarial. CHOSEN.
  (c) chunked/Merkle -- (b) applied per segment of the first non-unit dim, localizing which rows
      diverge, at the cost of more digest values. Available as :func:`segment_digests`, wired
      into the trace by ``SKYRL_ISOEXEC_DEBUG_SEGMENTS`` and read back by ``compare.py``.
  (d) library options -- none in torch or in-tree.

The weight stream is one splitmix64 per GROUP of :data:`_GROUP` consecutive positions,
decorrelated inside the group by xor with ``lane * _LANE`` and forced odd:

    w_i = (splitmix64((i >> 3) ^ salt) ^ ((i & 7) * _LANE)) | 1

One splitmix64 per element measured 868 GB/s on an H100 (3.1x the tensor's read time);
amortizing it over 8 elements reaches 1309-1966 GB/s (1.5-2.1x read time) with the same
guarantees: weights stay position-keyed (so chunking-invariance holds), stay odd (so a
single-position difference is still *deterministically* detected), and stay pairwise distinct (so
a permutation is still detected). Cancellation of a two-position fault inside one group needs
``d*(w_a - w_b) == 0 mod 2**64``; both weights are odd so their difference is even, and for a
16-bit ``d`` that needs ~48 trailing zeros in the difference -- probability ~2**-47, and the lane
constant is odd so consecutive lanes never share a weight. Weights of *different* groups are
independent splitmix64 outputs, as before.

Cost, MEASURED on one idle H100 SM90 (medians; ``read`` is ``copy_`` halved, since a copy moves
the tensor twice). GPU time of the kernel alone:

    shape                       MB   kernel ms    GB/s   read ms   x read   ladder (all rungs)
    [512,2048]    decode        2.1     0.0347      60    0.0097     3.6x   1.04x  (5 rungs)
    [10240,2048]  trainer      41.9     0.0411    1020    0.0197     2.1x   2.44x  (5 rungs)
    [16384,2048]  prefill      67.1     0.0512    1309    0.0277     1.9x   2.81x  (5 rungs)
    [512,151936]  logits bf16 155.6     0.0841    1850    0.0567     1.5x   3.53x  (5 rungs)
    [512,151936]  logits fp32 311.2     0.1583    1966    0.1081     1.5x   3.41x  (8 rungs)
    [16384,128]   router        8.4     0.0358     234    0.0102     3.5x   1.23x  (8 rungs)

Small tensors are launch-latency bound, not bandwidth bound, hence the 3.5x on 2MB. The WHOLE
call, including the ``.item()`` device-to-host sync a caller pays per digest, against the eager
implementation this replaced (same shapes):

    shape                     digest ms   was      x     ladder ms   was      seg256 ms   was
    [512,2048]    decode          0.116   0.292   2.5x       0.146   1.421        0.120   0.559
    [10240,2048]  trainer         0.118   2.453  20.7x       0.208  12.266        0.179  10.223
    [16384,2048]  prefill         0.129   3.873  29.9x       0.256  19.355        0.223  16.321
    [512,151936]  logits bf16     0.166   8.833  53.3x       0.406  44.426        0.177   8.913
    [512,151936]  logits fp32     0.242   8.882  36.7x       0.671  72.177        0.243   8.968
    [16384,128]   router          0.116   0.317   2.7x       0.170   2.469        0.199  16.776

The fixed floor (a 1-element tensor, wall clock) is 98-101us, down from 285us. It is now almost
entirely the synchronous round trip: ~22us of triton launch and ~40us of D2H-after-launch, both
irreducible while each digest reads its own result back. The remaining win available is to defer
the read: the kernels accumulate into a per-device arena, so a whole flush's digests could be
read in ONE D2H, which would take the per-record cost to ~28us. Not done here -- an arena wrap
must not clear a slot whose value has not been read yet, and a silently wrong digest is worse
than a slow one.

A 110-record forward (30 gdn.core + 40 moe.router x 2 outputs, TP=8): 12.8ms decode and 13.0ms
prefill on the trainer, against 38.2ms and 93.7ms for the eager path (and 199.3ms for the
engine's prefill shapes) -- 3-15x.

Peak temp memory is 8 int64 per digest (an arena slot); the eager fallback allocates
``chunk_numel`` int64 elements at a time (default 2**22 -> 32MB).

The k-ladder is computed in ONE pass: the int view is loaded once, the position weight once, and
every rung's mask is applied to that same register, so the whole ladder costs 1.0-3.5x a single
digest instead of 5-8x. ``segment_digests`` likewise issues one kernel for all segments.

Triton is the digest backend on CUDA; CPU tensors, a torch build without triton, and
``SKYRL_ISOEXEC_DEBUG_DIGEST=eager`` take the pure-torch path (0.35-11ms at the shapes above --
correct, just 3-70x slower). The two are required to be BIT-IDENTICAL and
``tests/test_thash_gpu.py`` pins that across dtypes, shapes, chunkings, rungs, segmentations and
devices, all three against an independent pure-Python reference.

Dtype coverage: every dtype in :data:`_DTYPE_TABLE` is digested through a same-size integer view,
which is why fp8, the unsigned widths and complex are cheap to support -- the bits are the bits.
Anything else raises ``TypeError`` and the caller (``trace.py``) turns that into an explicit
``unrecordable`` record rather than dropping the output.

Mantissa-truncation ladder: for a float tensor, masking the mantissa to its top ``k`` bits before
digesting yields a digest of the tensor "rounded" (truncated toward zero in magnitude) to k-bit
precision. Two implementations whose outputs agree to ~2**-k relative error then match at rung k
and diverge at finer rungs, so comparing ladders brackets the divergence magnitude without
exchanging tensors. Rung sets are per dtype (:data:`LADDER_BY_MANTISSA`): fp32/fp64 carry deep
rungs because a 1-ULP fp32 difference is 2**-23 relative and a ladder that stops at 2**-6 would
overstate it by five orders of magnitude. bf16 cannot go past k=6 -- with 7 explicit mantissa
bits its own ULP is 2**-8 relative, so "matches at k=6" is already "within a few ULP" and is the
tightest statement the format can express.

Ladder caveats, stated honestly, because the comparator's wording depends on them:
  * The bound is a MAX over elements: one bad element breaks a whole rung.
  * Truncation is a step function, so two values straddling a truncation boundary diverge at
    every rung even when numerically close. With many elements perturbed (the realistic "two
    kernels round differently everywhere" case) some element straddles every boundary, the whole
    ladder saturates, and the honest verdict is "not bounded by the ladder" -- NOT "large".
  * NaN payloads and -0.0 vs +0.0 are compared bitwise, by design.
  * The digest is position-keyed, so a permutation of the same values (a batch-order or token
    ordering bug) is detected but is indistinguishable from a value change: it breaks every rung
    exactly as a catastrophic numerical fault would. ``segment_digests`` narrows this -- a
    permutation confined to one slab shows up as one differing segment.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

_U64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_MIX1 = 0xBF58476D1CE4E5B9
_MIX2 = 0x94D049BB133111EB
_LANE = 0x9E3779B97F4A7C15  # odd: distinct weight per lane inside a group
_GROUP_SHIFT = 3
_GROUP = 1 << _GROUP_SHIFT

# dtype -> (same-size int view dtype, unsigned mask after upcast to int64, mantissa bits or None)
# Append-only: the enumeration order below feeds _DTYPE_CODE, which is baked into every digest.
_DTYPE_TABLE = {
    torch.bfloat16: (torch.int16, 0xFFFF, 7),
    torch.float16: (torch.int16, 0xFFFF, 10),
    torch.float32: (torch.int32, 0xFFFFFFFF, 23),
    torch.float64: (torch.int64, None, 52),
    torch.int8: (torch.int8, 0xFF, None),
    torch.uint8: (torch.uint8, 0xFF, None),
    torch.int16: (torch.int16, 0xFFFF, None),
    torch.int32: (torch.int32, 0xFFFFFFFF, None),
    torch.int64: (torch.int64, None, None),
    torch.bool: (torch.uint8, 0x1, None),
}

# Widths torch grew later, or that only some builds carry. Same treatment, appended so the
# codes of the dtypes above never move. Complex views land on int64 with twice the elements
# (two components per value); position-keyed weighting handles that with no special case, but
# there is no meaningful mantissa rung, so complex gets no ladder.
for _name, _view, _mask, _mbits in (
    ("float8_e4m3fn", torch.int8, 0xFF, 3),
    ("float8_e5m2", torch.int8, 0xFF, 2),
    ("float8_e4m3fnuz", torch.int8, 0xFF, 3),
    ("float8_e5m2fnuz", torch.int8, 0xFF, 2),
    ("uint16", torch.int16, 0xFFFF, None),
    ("uint32", torch.int32, 0xFFFFFFFF, None),
    ("uint64", torch.int64, None, None),
    ("complex64", torch.int64, None, None),
    ("complex128", torch.int64, None, None),
):
    _dt = getattr(torch, _name, None)
    if _dt is not None and _dt not in _DTYPE_TABLE:
        _DTYPE_TABLE[_dt] = (_view, _mask, _mbits)
del _name, _view, _mask, _mbits, _dt

# mantissa bits -> truncation rungs, finest first. Rungs are ~4 apart in the fine region so a
# non-matching/matching rung pair brackets the relative error within a factor of 16.
LADDER_BY_MANTISSA: Dict[int, Tuple[int, ...]] = {
    2: (1, 0),  # float8_e5m2
    3: (2, 1, 0),  # float8_e4m3
    7: (6, 4, 2, 0),  # bfloat16 -- k=6 is the finest rung the format can express
    10: (9, 7, 5, 3, 1, 0),  # float16
    23: (22, 18, 14, 10, 6, 2, 0),  # float32 -- k=22 resolves a 1-ULP difference
    52: (48, 40, 32, 24, 16, 8, 0),  # float64
}
DEFAULT_LADDER: Tuple[int, ...] = (6, 4, 2, 0)
DEFAULT_CHUNK_NUMEL = 1 << 22
MAX_OUTPUT_DEPTH = 3
_MAX_MASKS_PER_PASS = 8  # mirrors thash_kernels._MAX_MASKS; the deepest ladder (fp64) is 8 wide

ENV_BACKEND = "SKYRL_ISOEXEC_DEBUG_DIGEST"  # "auto" (default) | "eager" | "triton"


def _s64(v: int) -> int:
    """Python int -> signed-int64 representation (two's complement)."""
    v &= _U64
    return v - (1 << 64) if v >= (1 << 63) else v


def _u64(v: int) -> int:
    return v & _U64


def _lsr(x: torch.Tensor, n: int) -> torch.Tensor:
    """Logical right shift for signed int64 tensors (torch >> is arithmetic)."""
    return (x >> n) & ((1 << (64 - n)) - 1)


def _mix64_t(z: torch.Tensor) -> torch.Tensor:
    """splitmix64 finalizer, tensor form. Wrapping int64 arithmetic throughout."""
    z = z ^ _lsr(z, 30)
    z = z * _s64(_MIX1)
    z = z ^ _lsr(z, 27)
    z = z * _s64(_MIX2)
    return z ^ _lsr(z, 31)


def _mix64_i(z: int) -> int:
    """splitmix64 finalizer, python-int form (must match :func:`_mix64_t`)."""
    z = _u64(z)
    z = _u64((z ^ (z >> 30)) * _MIX1)
    z = _u64((z ^ (z >> 27)) * _MIX2)
    return _u64(z ^ (z >> 31))


def mantissa_bits(dtype: torch.dtype) -> Optional[int]:
    """Explicit mantissa bits of a supported float dtype, else None."""
    ent = _DTYPE_TABLE.get(dtype)
    return ent[2] if ent is not None else None


def ladder_for(dtype: torch.dtype) -> Tuple[int, ...]:
    """Truncation rungs to record for ``dtype`` (empty when it has no mantissa to truncate)."""
    mbits = mantissa_bits(dtype)
    if mbits is None:
        return ()
    return LADDER_BY_MANTISSA.get(mbits, DEFAULT_LADDER)


# -- backend selection ----------------------------------------------------------------------

_kernels = None  # None = not probed yet; False = unavailable


def _triton_kernels():
    """The triton backend module, or None. Probed once; any import failure falls back to eager."""
    global _kernels
    if _kernels is None:
        mode = os.environ.get(ENV_BACKEND, "auto").strip().lower()
        if mode == "eager":
            _kernels = False
        else:
            try:
                from . import thash_kernels

                _kernels = thash_kernels
            except Exception:  # noqa: BLE001 -- no triton / no GPU build is a supported config
                if mode == "triton":
                    raise
                _kernels = False
    return _kernels or None


def _reset_backend_for_tests() -> None:
    global _kernels
    _kernels = None


def digest_backend(device: torch.device) -> str:
    """Which implementation would digest a tensor on ``device`` -- ``"triton"`` or ``"eager"``."""
    k = _triton_kernels()
    return "triton" if (k is not None and k.available(device)) else "eager"


def preload(device: Optional[torch.device] = None) -> bool:
    """Compile the digest kernels ahead of the first traced region. False when eager."""
    k = _triton_kernels()
    dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    if k is None or not k.available(dev):
        return False
    k.preload(dev)
    return True


# -- the weighted modular sum ---------------------------------------------------------------


def _int_view(t: torch.Tensor) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
    """(flat same-size-int tensor in logical row-major order, unsigned mask, mantissa bits).

    ``.contiguous()`` fixes the element order to the logical one, so a transposed view and its
    contiguous twin digest identically iff their logical contents match.
    """
    if t.dtype not in _DTYPE_TABLE:
        raise TypeError(f"tensor_digest: unsupported dtype {t.dtype}")
    view_dtype, mask, mbits = _DTYPE_TABLE[t.dtype]
    # Each of these is a real dispatch (~1.5us), and this runs on every recorded output, so only
    # the ones the tensor actually needs are issued.
    flat = t.detach() if t.requires_grad else t
    if not flat.is_contiguous():
        flat = flat.contiguous()
    if flat.dim() != 1:
        flat = flat.reshape(-1)
    return (flat if flat.dtype == view_dtype else flat.view(view_dtype)), mask, mbits


def _and_mask(unsigned_mask: Optional[int], mbits: Optional[int], k: Optional[int]) -> Tuple[int, int]:
    """(signed int64 mask applied after the upcast, mantissa bits kept).

    Truncating in the view dtype and then masking off the sign extension is the same bit pattern
    as masking once with the intersection, which is what lets one kernel argument carry both.
    """
    m = _U64 if unsigned_mask is None else unsigned_mask
    kept = mbits
    if k is not None and mbits is not None and k < mbits:
        m &= _U64 ^ ((1 << (mbits - k)) - 1)
        kept = k
    return _s64(m), (kept if mbits is not None else -1)


def _salt(seed: int) -> int:
    return _s64(_mix64_i(seed) ^ _GOLDEN)


def _weights_t(idx: torch.Tensor, salt: int) -> torch.Tensor:
    """Position weight stream, eager twin of ``thash_kernels._weights``."""
    base = _mix64_t((idx >> _GROUP_SHIFT) ^ salt)
    return (base ^ ((idx & (_GROUP - 1)) * _s64(_LANE))) | 1


def _weighted_sums_eager(iv: torch.Tensor, masks: Sequence[int], salt: int, chunk_numel: int) -> List[int]:
    """``sum_i (x_i & mask_j) * w_i mod 2**64`` per mask, chunked; device-side accumulators."""
    n = iv.numel()
    accs = [torch.zeros((), dtype=torch.int64, device=iv.device) for _ in masks]
    for start in range(0, n, max(1, chunk_numel)):
        chunk = iv[start : min(start + chunk_numel, n)]
        raw = chunk.to(torch.int64)
        idx = torch.arange(start, start + chunk.numel(), dtype=torch.int64, device=iv.device)
        w = _weights_t(idx, salt)
        for j, m in enumerate(masks):
            accs[j] = accs[j] + ((raw & m) * w).sum()
    return [_u64(int(a.item())) for a in accs]


def _weighted_sums(iv: torch.Tensor, masks: Sequence[int], seed: int, chunk_numel: int) -> List[int]:
    """One weighted modular sum per mask, in a single pass over ``iv`` where triton is available."""
    salt = _salt(seed)
    k = _triton_kernels()
    if k is not None and k.available(iv.device) and iv.numel() and len(masks) <= k._MAX_MASKS:
        try:
            return [_u64(v) for v in k.weighted_sums(iv, masks, salt)]
        except Exception:  # noqa: BLE001 -- a kernel that will not launch must not lose the trace
            pass
    return _weighted_sums_eager(iv, masks, salt, chunk_numel)


_DTYPE_CODE = {dt: i + 1 for i, dt in enumerate(_DTYPE_TABLE)}  # stable across processes


def _header(t: torch.Tensor, kept_bits: int, seed: int) -> int:
    h = _mix64_i(seed ^ _GOLDEN)
    h = _mix64_i(h ^ (kept_bits & 0xFF))
    h = _mix64_i(h ^ _mix64_i(_DTYPE_CODE[t.dtype]))
    for d in t.shape:
        h = _mix64_i(h ^ _u64(d + 1))
    return h


def tensor_digest(
    t: torch.Tensor,
    *,
    k: Optional[int] = None,
    seed: int = 0,
    chunk_numel: int = DEFAULT_CHUNK_NUMEL,
) -> int:
    """64-bit order-invariant digest of ``t``'s logical contents (scheme (b) above).

    ``k`` truncates float mantissas to k bits first (None = full precision). Equal tensors (same
    shape, dtype, logical values) digest identically on any device under any chunking; any
    single-element difference changes the digest; a permutation of distinct elements changes it
    with probability ~1 - 2**-64.
    """
    iv, umask, mbits = _int_view(t)
    mask, kept = _and_mask(umask, mbits, k)
    body = _weighted_sums(iv, [mask], seed, chunk_numel)[0]
    return _mix64_i(body ^ _header(t, kept, seed))


def digest_ladder(
    t: torch.Tensor,
    ks: Optional[Sequence[int]] = None,
    *,
    seed: int = 0,
    chunk_numel: int = DEFAULT_CHUNK_NUMEL,
) -> Dict[str, str]:
    """Digests at full precision plus a ladder of mantissa truncations, as ``{"full": hex, "k6": hex, ...}``.

    ``ks`` defaults to :func:`ladder_for` (per dtype). Non-float dtypes get only ``"full"``.
    Rungs >= the dtype's mantissa width are dropped (they would duplicate ``"full"``). Every rung
    is a mask over the SAME loaded value, so the whole ladder is one pass, not one pass per rung.
    """
    iv, umask, mbits = _int_view(t)
    rungs: List[Optional[int]] = [None]
    if mbits is not None:
        rungs += [k for k in (ks if ks is not None else ladder_for(t.dtype)) if 0 <= k < mbits]
    pairs = [_and_mask(umask, mbits, k) for k in rungs]
    limit = _MAX_MASKS_PER_PASS
    out: Dict[str, str] = {}
    for lo in range(0, len(pairs), limit):
        window = pairs[lo : lo + limit]
        bodies = _weighted_sums(iv, [m for m, _ in window], seed, chunk_numel)
        for (_, kept), body, kk in zip(window, bodies, rungs[lo : lo + limit]):
            name = "full" if kk is None else f"k{kk}"
            out[name] = f"{_mix64_i(body ^ _header(t, kept, seed)):016x}"
    return out


def segment_axis(t: torch.Tensor) -> int:
    """Dim :func:`segment_digests` slices along: the first with extent > 1 (0 for a flat/1-elem).

    Slicing dim 0 is useless for the shapes the GDN door actually produces -- the trainer's
    ``[1, T, H, D]`` has one segment covering the whole tensor. Every dim before the first
    non-unit one has extent 1, so the contiguous layout is exactly ``[shape[axis], rest]`` and
    segmenting along ``axis`` needs no reshape. Both sides then slice the same logical axis (T
    for the GDN door), so segment i covers the same slab of tokens on both -- the digests still
    fold the shape, which is what keeps a shape difference its own divergence kind.
    """
    for i, d in enumerate(t.shape):
        if d > 1:
            return i
    return 0


def segment_digests(
    t: torch.Tensor,
    *,
    rows_per_segment: int = 1024,
    k: Optional[int] = None,
    seed: int = 0,
    chunk_numel: int = DEFAULT_CHUNK_NUMEL,
) -> List[str]:
    """Scheme (c): one digest per ``rows_per_segment`` slab of :func:`segment_axis`.

    Weights are keyed by the position WITHIN the segment and the segment is salted by its
    absolute start offset, so segment i is reproducible on its own; each segment is additionally
    finalized with its own header, so segments are directly comparable one-to-one.
    """
    if t.dim() == 0 or t.numel() == 0:
        return [f"{tensor_digest(t, k=k, seed=seed, chunk_numel=chunk_numel):016x}"]
    iv, umask, mbits = _int_view(t)
    mask, kept = _and_mask(umask, mbits, k)
    axis = segment_axis(t)
    rows = t.shape[axis]
    row_numel = iv.numel() // rows  # ints per row of `axis` -- from the view, so complex slabs right
    seg_numel = max(1, rows_per_segment) * row_numel
    nseg = (rows + max(1, rows_per_segment) - 1) // max(1, rows_per_segment)
    head = _header(t, kept, seed)
    kern = _triton_kernels()
    if kern is not None and kern.available(iv.device) and seg_numel:
        try:
            bodies = kern.segment_sums(iv, nseg, seg_numel, mask, _s64(seed))
            return [f"{_mix64_i(_u64(b) ^ _mix64_i(head ^ _u64(si + 1))):016x}" for si, b in enumerate(bodies)]
        except Exception:  # noqa: BLE001 -- fall back rather than lose the segmentation
            pass
    out: List[str] = []
    for si in range(nseg):
        seg = iv[si * seg_numel : min((si + 1) * seg_numel, iv.numel())]
        body = _weighted_sums_eager(seg, [mask], _salt(seed ^ _mix64_i(si * seg_numel + 1)), chunk_numel)[0]
        out.append(f"{_mix64_i(body ^ _mix64_i(head ^ _u64(si + 1))):016x}")
    return out


def iter_tensor_outputs(out, *, max_depth: int = MAX_OUTPUT_DEPTH) -> Iterable[Tuple[str, Optional[torch.Tensor]]]:
    """Enumerate a region call's tensor outputs as ``(path, tensor)``, descending containers.

    Paths are dotted strings ("0", "1.0", "logits") so nested tuples and dict outputs stay
    distinguishable in the record key -- a bare tensor is path "0". Tuples, lists and dicts are
    descended up to ``max_depth``; a container found at the depth limit yields ``(path, None)``
    so the caller records it as unrecordable instead of dropping it silently. Non-tensor leaves
    (ints, strings, None) are skipped -- a call whose outputs contain no tensor at all yields
    nothing, which the caller likewise reports rather than swallows.
    """

    def walk(obj, path: str, depth: int):
        if isinstance(obj, torch.Tensor):
            yield path, obj
            return
        if isinstance(obj, (tuple, list)):
            items = enumerate(obj)
        elif isinstance(obj, dict):
            items = obj.items()
        else:
            return
        if depth >= max_depth:
            yield path, None
            return
        for key, val in items:
            child = f"{path}.{key}" if path else str(key)
            yield from walk(val, child, depth + 1)

    if isinstance(out, torch.Tensor):
        yield "0", out
        return
    yield from walk(out, "", 0)
