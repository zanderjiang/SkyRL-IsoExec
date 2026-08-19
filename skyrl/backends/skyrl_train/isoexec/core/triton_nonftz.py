"""Non-FTZ Triton primitives: replacements for the libdevice calls that flush subnormals to zero.

Triton links libdevice with ``__CUDA_FTZ`` set, so ``libdevice.div_rn``, ``log1p`` and ``rsqrt``
flush subnormals where ATen does not, and ``libdevice.sqrt`` is not correctly rounded even on
ordinary normal inputs; ``libdevice.exp``, ``fma_rn`` and the arithmetic operators are safe. Op
folders import this as ``from ...core.triton_nonftz import ...``; it imports nothing from the isoexec
tree and stays out of ``core/__init__`` so a bare ``import isoexec.core`` never pulls in triton.
"""

from __future__ import annotations

try:
    import triton  # noqa: F401
    import triton.language as tl
    from triton.language.extra import libdevice

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False


# Smallest positive NORMAL fp32. Below this, values are subnormal and FTZ applies.
SMALLEST_NORMAL_F32 = 1.17549435e-38
# 2**-24: below this, log1p(x) == x after fp32 rounding (the x^2/2 term cannot reach the mantissa).
_LOG1P_LINEAR_BELOW = 5.9604645e-8


if HAVE_TRITON:
    # A jitted kernel can only close over module globals that are ``tl.constexpr`` objects; a plain
    # float raises NameError at compile time inside the caller's kernel. Derived, never re-typed.
    _LOG1P_LINEAR_BELOW_TL = tl.constexpr(_LOG1P_LINEAR_BELOW)

    @triton.jit
    def div_rn(a, b):
        """fp32 ``a / b``, round-to-nearest, subnormal-preserving.

        Replaces ``libdevice.div_rn``, which flushes subnormal quotients, and Triton's ``/``, which
        lowers to an approximate reciprocal.
        """
        return tl.inline_asm_elementwise(
            "div.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1
        )

    @triton.jit
    def sqrt(x):
        """fp32 ``sqrt(x)``, round-to-nearest, subnormal-preserving.

        This fixes more than FTZ: ``libdevice.sqrt`` also disagrees with ATen on ordinary normal
        inputs, so any kernel taking an L2 or RMS norm is exposed on completely ordinary data.
        """
        return tl.inline_asm_elementwise("sqrt.rn.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit
    def log1p(x):
        """fp32 ``log1p(x)``, subnormal-preserving. Replaces ``libdevice.log1p``.

        For ``|x| < 2**-24`` the exact series rounds to ``x`` in fp32, so the subnormal and
        tiny-normal range is answered without reaching the FTZ'd libdevice call. Softplus is then
        ``tl.where(x > 20.0, x, log1p(libdevice.exp(x)))`` using this log1p.
        """
        return tl.where(tl.abs(x) < _LOG1P_LINEAR_BELOW_TL, x, libdevice.log1p(x))

    @triton.jit
    def sigmoid(x):
        """fp32 ``sigmoid`` matching ATen bitwise, including where the result is subnormal.

        The same expression built on ``libdevice.div_rn`` flushes for x below about -87.
        """
        return div_rn(1.0, 1.0 + libdevice.exp(-x))


def rsqrt_is_safe(min_possible_argument: float) -> bool:
    """Is ``libdevice.rsqrt`` bitwise-safe for arguments no smaller than this?

    ``libdevice.rsqrt`` matches ATen on normal inputs but flushes subnormal inputs and returns
    ``inf``, and there is no known exact replacement: ``div_rn(1, sqrt(x))`` double-rounds and
    differs from ATen's single-rounded ``rsqrtf``. So bound the argument instead of replacing the
    op -- for an L2/RMS norm the sum of squares is subnormal only if the whole row is below ~1e-19.
    """
    return min_possible_argument >= SMALLEST_NORMAL_F32
