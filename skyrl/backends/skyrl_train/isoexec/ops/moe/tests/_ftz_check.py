"""Harness helper: catch a flushed subnormal, and refuse to let the check be vacuous.

DROP THIS INTO ANY KERNEL GATE. Two functions:

    assert_no_ftz(name, ref, got)   -- did the kernel zero something ATen kept subnormal?
    ftz_inputs_for(kind)            -- inputs that GUARANTEE subnormals occur, per op family

WHY IT NEEDS TO EXIST. O2 shipped ``libdevice.div_rn`` past a 130-check ``torch.equal`` gate. Every
input in that gate was a well-conditioned ``randn`` matmul, so no subnormal was ever produced, so
the FTZ path was never exercised. The bug surfaced only when an outlier row was added -- one logit
~1e4 above the rest, which is what a real residual stream produces. 7 elements in 262,144.

THE VACUITY RULE, copied from the FFMA pattern in §0: a check that would also pass with the
protection deleted proves nothing. :func:`assert_no_ftz` therefore reports how many of the REFERENCE
values are subnormal, and **fails the harness when that count is zero**, because then the check is
not testing FTZ at all -- it is testing nothing and reporting success.

Run this file directly to reproduce the full libdevice audit behind
``skyrl/backends/skyrl_train/isoexec/core/triton_nonftz.py``.
"""

from __future__ import annotations

import torch

SMALLEST_NORMAL_F32 = 1.17549435e-38


def assert_no_ftz(
    name: str, ref: torch.Tensor, got: torch.Tensor, *, fail_list: list | None = None, expect_flush: bool = False
) -> dict:
    """Compare ``got`` to ``ref``, separating FLUSHED subnormals from ordinary disagreement.

    Returns a dict with ``subnormal`` (how many reference values are subnormal), ``flushed`` (how
    many of those the kernel returned as exactly 0), ``diff`` (total disagreeing elements) and
    ``vacuous`` (True when the reference contains no subnormal at all, i.e. this input cannot
    detect FTZ). Appends to ``fail_list`` on flush OR on vacuity.

    ``expect_flush=True`` inverts the verdict: use it for the NEGATIVE CONTROL that proves this
    helper detects anything at all (run the known-flushing ``libdevice.div_rn`` through it). Then a
    flush is the pass and its absence is the failure.
    """
    assert ref.shape == got.shape, f"{name}: shape {tuple(ref.shape)} vs {tuple(got.shape)}"
    sub = (ref != 0) & (ref.abs() < SMALLEST_NORMAL_F32)
    n_sub = int(sub.sum().item())
    n_flushed = int((sub & (got == 0)).sum().item())
    n_diff = int((ref != got).sum().item())
    vacuous = n_sub == 0
    if vacuous:
        msg = f"VACUOUS: {name} -- reference has no subnormal values, so this input cannot detect FTZ"
        print(f"  FAIL {msg}")
        if fail_list is not None:
            fail_list.append(msg)
    elif expect_flush:
        if n_flushed:
            print(f"  ok   {name}: flushed {n_flushed}/{n_sub} as expected -- the detector works")
        else:
            msg = f"CONTROL VACUOUS: {name} did NOT flush, so this helper detects nothing"
            print(f"  FAIL {msg}")
            if fail_list is not None:
                fail_list.append(msg)
    elif n_flushed:
        msg = f"{name}: FLUSHED {n_flushed}/{n_sub} subnormals to zero (total diff {n_diff}/{ref.numel()})"
        print(f"  FAIL {msg}")
        if fail_list is not None:
            fail_list.append(msg)
    else:
        print(f"  ok   {name}: 0/{n_sub} subnormals flushed, total diff {n_diff}/{ref.numel()}")
    return dict(subnormal=n_sub, flushed=n_flushed, diff=n_diff, vacuous=vacuous)


def ftz_inputs_for(kind: str, n: int = 1 << 16, device: str = "cuda") -> torch.Tensor:
    """Inputs that are GUARANTEED to drive `kind` into the subnormal range.

    ``kind`` in {"exp", "sigmoid", "softplus", "log1p", "sqrt", "rsqrt", "softmax_row"}.
    """
    if kind in ("exp", "sigmoid", "softplus"):
        # exp(x) and sigmoid(x) are subnormal for x below ~-87; softplus routes through log1p(exp(x)).
        return torch.linspace(-120.0, -87.0, n, device=device)
    if kind in ("log1p", "sqrt", "rsqrt"):
        return torch.linspace(1e-45, SMALLEST_NORMAL_F32, n, device=device)
    if kind == "softmax_row":
        # A massive-activation row: one logit far above the rest, so every other exp(x - max)
        # underflows to subnormal. This is the shape that caught O2's bug.
        rows = max(1, n // 256)
        g = torch.Generator(device=device).manual_seed(0)
        x = torch.randn(rows, 256, device=device, generator=g)
        x[:, 0] += 100.0
        return x
    raise ValueError(f"unknown kind {kind!r}")


if __name__ == "__main__":
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    dev = "cuda"
    fails: list = []
    print("=== libdevice FTZ audit (reproduces the table in isoexec/core/triton_nonftz.py) ===")

    @triton.jit
    def _probe(X, O, N, BLOCK: tl.constexpr, OP: tl.constexpr, FIX: tl.constexpr):
        p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = p < N
        x = tl.load(X + p, mask=m)
        if OP == 0:  # exp
            o = libdevice.exp(x)
        elif OP == 1:  # log1p
            o = tl.where(tl.abs(x) < 5.9604645e-8, x, libdevice.log1p(x)) if FIX else libdevice.log1p(x)
        elif OP == 2:  # sqrt
            o = (
                tl.inline_asm_elementwise("sqrt.rn.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1)
                if FIX
                else libdevice.sqrt(x)
            )
        elif OP == 3:  # rsqrt
            o = libdevice.rsqrt(x)
        else:  # sigmoid
            d = 1.0 + libdevice.exp(-x)
            one = tl.full(x.shape, 1.0, tl.float32)
            o = (
                tl.inline_asm_elementwise(
                    "div.rn.f32 $0, $1, $2;", "=r,r,r", [one, d], dtype=tl.float32, is_pure=True, pack=1
                )
                if FIX
                else libdevice.div_rn(one, d)
            )
        tl.store(O + p, o, mask=m)

    cases = [
        ("exp", 0, torch.exp, "exp"),
        ("log1p", 1, torch.log1p, "log1p"),
        ("sqrt", 2, torch.sqrt, "sqrt"),
        ("rsqrt", 3, torch.rsqrt, "rsqrt"),
        ("sigmoid", 4, torch.sigmoid, "sigmoid"),
    ]
    for nm, op, fn, kind in cases:
        x = ftz_inputs_for(kind).contiguous()
        ref = fn(x)
        for fix in (False, True):
            o = torch.empty_like(x)
            _probe[(triton.cdiv(x.numel(), 256),)](x, o, x.numel(), BLOCK=256, OP=op, FIX=fix, enable_fp_fusion=False)
            label = f"{nm:8s} {'FIXED   ' if fix else 'libdevice'}"
            sub = (ref != 0) & (ref.abs() < SMALLEST_NORMAL_F32)
            n_sub = int(sub.sum().item())
            n_fl = int((sub & (o == 0)).sum().item())
            n_d = int((ref != o).sum().item())
            n_inf = int((torch.isinf(o) & ~torch.isinf(ref)).sum().item())
            note = (
                "  <<< FLUSHES OUTPUT"
                if n_fl
                else ("  <<< FLUSHES INPUT (-> inf)" if n_inf else ("  <<< WRONG" if n_d else "  exact"))
            )
            print(f"  {label}  diff {n_d:>6}/{x.numel()}  subnormal_refs={n_sub:>6} flushed={n_fl:>6}{note}")
    print("\nsee isoexec/core/triton_nonftz.py for the recommended form of each")
