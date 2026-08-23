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
      pseudorandom stream indexed by *absolute position* (splitmix64 of the flat index, forced
      odd). Position-sensitive, one fused pass, chunking-invariant because the weights are
      position-keyed, and because every ``w_i`` is odd (hence invertible mod 2**64) ANY
      single-position difference changes the digest deterministically, not probabilistically.
      Multi-position collisions require ``sum w_i d_i == 0 mod 2**64``: ~2**-64 for
      pseudorandom weights, non-adversarial. CHOSEN.
  (c) chunked/Merkle -- (b) applied per row-segment, localizing which rows diverge, at the cost
      of more digest values. Available as :func:`segment_digests`.
  (d) library options -- none in torch or in-tree.

Cost of (b): per element one int64 upcast, ~6 elementwise int64 ops for the weight, one wrapping
multiply, one sum -- memory-bound, roughly 5-10x the tensor's read time. At [tokens ~1e4,
hidden ~8k] bf16 (160MB) that is on the order of 1ms on an H100 -- fine per region per sampled
forward. Peak temp memory is bounded by ``chunk_numel`` int64 elements (default 2**22 -> 32MB).

Mantissa-truncation ladder: for a float tensor, masking the mantissa to its top ``k`` bits before
digesting yields a digest of the tensor "rounded" (truncated toward zero in magnitude) to k-bit
precision. Two implementations whose outputs agree to ~2**-k relative error then match at rung k
and diverge at finer rungs, so comparing ladders gives an approximate divergence magnitude
without exchanging tensors. Caveats, stated honestly: truncation is a step function, so a pair
of values straddling a truncation boundary can diverge at every rung even when numerically
close; a single bad element breaks a whole rung (the ladder profiles the MAX divergence, not the
mean); NaN payloads and -0.0 vs +0.0 are compared bitwise. In practice the first matching rung
(walking coarser) brackets max relative error in [2**-k_fine, 2**-k_match].
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

_U64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_MIX1 = 0xBF58476D1CE4E5B9
_MIX2 = 0x94D049BB133111EB

# dtype -> (same-size int view dtype, unsigned mask after upcast to int64, mantissa bits or None)
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

DEFAULT_LADDER: Tuple[int, ...] = (6, 4, 2, 0)
DEFAULT_CHUNK_NUMEL = 1 << 22


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


def _flat_int_view(t: torch.Tensor, k: Optional[int]) -> Tuple[torch.Tensor, Optional[int], int]:
    """(flat same-size-int tensor in logical row-major order, unsigned mask, mantissa bits kept).

    ``.contiguous()`` fixes the element order to the logical one, so a transposed view and its
    contiguous twin digest identically iff their logical contents match.
    """
    if t.dtype not in _DTYPE_TABLE:
        raise TypeError(f"tensor_digest: unsupported dtype {t.dtype}")
    view_dtype, mask, mbits = _DTYPE_TABLE[t.dtype]
    flat = t.detach().contiguous().reshape(-1)
    iv = flat if flat.dtype == view_dtype else flat.view(view_dtype)
    kept = mbits
    if k is not None and mbits is not None and k < mbits:
        drop = mbits - k
        iv = iv & _s64(~((1 << drop) - 1)) if view_dtype == torch.int64 else iv & ~((1 << drop) - 1)
        kept = k
    return iv, mask, (kept if mbits is not None else -1)


def _weighted_sum(iv: torch.Tensor, mask: Optional[int], seed: int, chunk_numel: int) -> int:
    """sum_i w_i * x_i mod 2**64 over the flat int tensor, chunked; device-side accumulator."""
    n = iv.numel()
    acc = torch.zeros((), dtype=torch.int64, device=iv.device)
    salt = _s64(_mix64_i(seed) ^ _GOLDEN)
    for start in range(0, n, chunk_numel):
        chunk = iv[start : min(start + chunk_numel, n)]
        x = chunk.to(torch.int64)
        if mask is not None:
            x = x & mask
        idx = torch.arange(start, start + chunk.numel(), dtype=torch.int64, device=iv.device)
        w = _mix64_t(idx ^ salt) | 1  # odd -> invertible mod 2**64
        acc = acc + (x * w).sum()
    return _u64(int(acc.item()))


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
    iv, mask, kept = _flat_int_view(t, k)
    body = _weighted_sum(iv, mask, seed, chunk_numel)
    return _mix64_i(body ^ _header(t, kept, seed))


def digest_ladder(
    t: torch.Tensor,
    ks: Optional[Sequence[int]] = None,
    *,
    seed: int = 0,
    chunk_numel: int = DEFAULT_CHUNK_NUMEL,
) -> Dict[str, str]:
    """Digests at full precision plus a ladder of mantissa truncations, as ``{"full": hex, "k6": hex, ...}``.

    Non-float dtypes get only ``"full"``. Rungs >= the dtype's mantissa width are dropped
    (they would duplicate ``"full"``). Cost is one full pass per rung.
    """
    out = {"full": f"{tensor_digest(t, seed=seed, chunk_numel=chunk_numel):016x}"}
    mbits = mantissa_bits(t.dtype)
    if mbits is None:
        return out
    for k in ks if ks is not None else DEFAULT_LADDER:
        if 0 <= k < mbits:
            out[f"k{k}"] = f"{tensor_digest(t, k=k, seed=seed, chunk_numel=chunk_numel):016x}"
    return out


def segment_digests(
    t: torch.Tensor,
    *,
    rows_per_segment: int = 1024,
    k: Optional[int] = None,
    seed: int = 0,
    chunk_numel: int = DEFAULT_CHUNK_NUMEL,
) -> List[str]:
    """Scheme (c): one digest per ``rows_per_segment`` slab of dim 0, localizing WHICH rows diverge.

    Weights are keyed by absolute flat position, so segment i's digest equals the digest of the
    same rows inside any larger tensor with identical leading layout -- but each segment is
    finalized with its own header, so segments are directly comparable one-to-one.
    """
    if t.dim() == 0:
        return [f"{tensor_digest(t, k=k, seed=seed, chunk_numel=chunk_numel):016x}"]
    iv, mask, kept = _flat_int_view(t, k)
    row_numel = max(1, t[0].numel()) if t.shape[0] else 0
    out: List[str] = []
    for si, r0 in enumerate(range(0, t.shape[0], rows_per_segment)):
        r1 = min(r0 + rows_per_segment, t.shape[0])
        seg = iv[r0 * row_numel : r1 * row_numel]
        body = _weighted_sum(seg, mask, seed ^ _mix64_i(r0 * row_numel + 1), chunk_numel)
        h = _mix64_i(_header(t, kept, seed) ^ _u64(si + 1))
        out.append(f"{_mix64_i(body ^ h):016x}")
    return out


def iter_tensor_outputs(out) -> Iterable[Tuple[int, torch.Tensor]]:
    """Enumerate the tensor outputs of a region call: a bare tensor, or tensors inside a
    tuple/list (one level), indexed by position. Non-tensors are skipped."""
    if isinstance(out, torch.Tensor):
        yield 0, out
        return
    if isinstance(out, (tuple, list)):
        for i, x in enumerate(out):
            if isinstance(x, torch.Tensor):
                yield i, x
