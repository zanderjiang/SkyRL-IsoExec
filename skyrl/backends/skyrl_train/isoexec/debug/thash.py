"""Order-invariant tensor digests for debug-mode region tracing.

``digest = sum_i w_i * (x_i & mask) mod 2**64`` over the tensor's integer bitcast, ``w_i`` an odd
position-keyed pseudorandom weight: identical under any chunking, reduction order or device.
Masking float mantissas to their top ``k`` bits gives a ladder that brackets divergence magnitude.
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

# Widths torch grew later, or that only some builds carry; appended so the codes above never move.
# Complex views land on int64 with twice the elements and get no mantissa ladder.
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

# mantissa bits -> truncation rungs, finest first. Rungs are ~4 apart so a matching/non-matching
# pair brackets the relative error within a factor of 16.
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

    ``.contiguous()`` fixes element order to the logical one, so a transposed view and its
    contiguous twin digest identically iff their logical contents match.
    """
    if t.dtype not in _DTYPE_TABLE:
        raise TypeError(f"tensor_digest: unsupported dtype {t.dtype}")
    view_dtype, mask, mbits = _DTYPE_TABLE[t.dtype]
    # Each of these is a real dispatch (~1.5us) on every recorded output, so only issue what's needed.
    flat = t.detach() if t.requires_grad else t
    if not flat.is_contiguous():
        flat = flat.contiguous()
    if flat.dim() != 1:
        flat = flat.reshape(-1)
    return (flat if flat.dtype == view_dtype else flat.view(view_dtype)), mask, mbits


def _and_mask(unsigned_mask: Optional[int], mbits: Optional[int], k: Optional[int]) -> Tuple[int, int]:
    """(signed int64 mask applied after the upcast, mantissa bits kept).

    Truncation and sign-extension masking intersect into one mask, so one kernel argument carries both.
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
    """64-bit order-invariant digest of ``t``'s logical contents.

    ``k`` truncates float mantissas to k bits first (None = full precision). Equal tensors digest
    identically on any device under any chunking; any single-element difference changes the digest.
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

    ``ks`` defaults to :func:`ladder_for` (per dtype); non-float dtypes get only ``"full"``, and
    rungs >= the dtype's mantissa width are dropped. The bound is a max over elements: with many
    elements perturbed every rung breaks and the ladder bounds nothing rather than saying "large".
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

    Every earlier dim has extent 1, so the contiguous layout is exactly ``[shape[axis], rest]``
    and no reshape is needed; both sides slice the same logical axis, so segment i covers the
    same slab on both.
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
    """One digest per ``rows_per_segment`` slab of :func:`segment_axis`.

    Weights are keyed by position within the segment and salted by its absolute start offset, so
    each segment is reproducible on its own and segments compare one-to-one.
    """
    if t.dim() == 0 or t.numel() == 0:
        return [f"{tensor_digest(t, k=k, seed=seed, chunk_numel=chunk_numel):016x}"]
    iv, umask, mbits = _int_view(t)
    mask, kept = _and_mask(umask, mbits, k)
    axis = segment_axis(t)
    rows = t.shape[axis]
    row_numel = iv.numel() // rows  # ints per row of `axis`, from the view so complex slabs are right
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

    Paths are dotted strings ("0", "1.0", "logits"); a bare tensor is path "0". A container found
    at ``max_depth`` yields ``(path, None)`` so the caller records it as unrecordable instead of
    dropping it silently; non-tensor leaves are skipped.
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
