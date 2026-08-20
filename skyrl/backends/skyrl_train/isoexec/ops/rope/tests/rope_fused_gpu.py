"""Bitwise gate for O8 -- fused attention RoPE (K5 kernel + K5a cos/sin hoist).

Run: CUDA_VISIBLE_DEVICES=<gpu> uv run --isolated --extra isoexec \
        PYTHONPATH=. python skyrl/backends/skyrl_train/isoexec/ops/rope/tests/rope_fused_gpu.py

WHAT THIS GATES, and why each section exists (every one is a failure this program has already paid
for -- see PERFORM_GAMEPLAN.md §0):

  [1] bitwise vs the eager expression, at EVERY production shape the install touches. `torch.equal`,
      never `allclose`. O6 shipped an installed-but-ungated width (256) because its gate covered two
      of the three widths its installer swapped.
  [2] the reference is the RIGHT one. The stack has TWO `_apply_rotary_pos_emb_bshd` with DIFFERENT
      rounding (stock bf16 3-round vs megatron_patches' fp32 1-round). This section asserts the real
      megatron stock function agrees, and asserts the fp32 patched form DIFFERS -- so "we matched the
      wrong function" cannot pass silently.
  [3] the hoist is not INERT. K2a's first revision hoisted nothing and all 32 bitwise checks passed.
      Counts computes vs calls, with a positive control (fresh freqs each call) that must NOT hoist.
  [4] FFMA contraction. I predicted immunity (all operands bf16-derived, so products are exact,
      and each is rounded to bf16 before the add) and the MEASUREMENT REFUTED IT: fusion-on differs
      from eager in 4,762/16,384, contracting one product into the add and eliding that product's
      explicit down-cast. `enable_fp_fusion=False` is load-bearing, and this section identifies the
      exact contracted form so a future Triton that changes it fails rather than re-blessing.
  [5] FTZ, with a positive control. No divide/exp/log1p/sqrt in the kernel, but "no subnormal was
      ever produced" is how O2 shipped its bug: the input population is driven bf16-subnormal and
      `assert_no_ftz` refuses to pass vacuously.
  [6] the tile is not the contract (no reduction in this kernel) -- proven, not asserted.
  [7] unrepresentative inputs. randn rows are well-conditioned; real residual streams carry
      massive-activation outliers. Heavy-tailed and outlier-injected populations.
  [8] eligibility / fall-through. An unmarked freqs, a grad-enabled call, `rotary_interleaved`, a
      non-contiguous t -- each must take the ORIGINAL path and return the eager answer.
  [9] a 150-step decode loop at ragged positions, and CUDA-graph capture+replay, both bitwise.
      An offline gate that never replays a graph cannot see what production runs.
"""

import os
import sys

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


if not torch.cuda.is_available():  # promoted nightly battery: needs one CUDA device
    print("SKIP: no CUDA device")
    raise SystemExit(0)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 7)))  # repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _ftz_check import assert_no_ftz  # noqa: E402

from skyrl.backends.skyrl_train.isoexec.ops.rope import rope_fused as rf  # noqa: E402
from skyrl.backends.skyrl_train.isoexec.core.contracts import bitwise_equal  # noqa: E402

DEV = "cuda"

# EVERY shape the install actually touches on Qwen3.5-35B-A3B, from its config.json -- not round
# numbers I liked. head_dim=256, partial_rotary_factor=0.25 -> rotary_dim=64. 16 q heads and 2 kv
# heads GLOBAL, so at TP8 a rank sees H=2 (query) and H=1 (key). D=R is the full-rotary case other
# models in this repo hit; D=128/R=64 covers a smaller head_dim.
PROD_HEADS = (2, 1)
PROD_D, PROD_R = 256, 64
SHAPES = [(D, R) for D, R in ((256, 64), (256, 256), (128, 64), (64, 64), (192, 64))]

_fails = []
_checks = 0


_BITS = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}


def check(name, a, b):
    """`torch.equal`, but on BIT PATTERNS for float dtypes.

    Plain `torch.equal` on floats says `-0.0 == +0.0` and `NaN != NaN`. The first would have hidden
    this kernel's `_rotate_half` negation bug entirely (Triton's unary minus does not flip the sign
    of zero); the second turns a NaN-producing bug into a spurious failure. Compare the bits.
    """
    global _checks
    _checks += 1
    ok = a.shape == b.shape and a.dtype == b.dtype
    if ok:
        av, bv = (a.view(_BITS[a.dtype]), b.view(_BITS[b.dtype])) if a.dtype in _BITS else (a, b)
        ok = torch.equal(av, bv)
    if not ok:
        n = (a != b).sum().item() if a.shape == b.shape else -1
        _fails.append(f"{name}: {n}/{a.numel()} differ")
        print(f"  FAIL {name}: {n}/{a.numel()} differ")
    return ok


def expect_differs(name, a, b):
    """A control: these MUST differ, or the thing it protects is not being protected."""
    global _checks
    _checks += 1
    if torch.equal(a, b):
        _fails.append(f"{name}: identical, control is VACUOUS")
        print(f"  FAIL {name}: identical -- this control proves nothing")
        return False
    print(f"  ok   {name}: differs in {(a != b).sum().item()}/{a.numel()} (control has teeth)")
    return True


def expect(name, cond, detail=""):
    global _checks
    _checks += 1
    if not cond:
        _fails.append(f"{name}: {detail}")
        print(f"  FAIL {name}: {detail}")
    return cond


# =================================================================================================
# the reference: megatron's STOCK _apply_rotary_pos_emb_bshd, transcribed. Section [2] checks this
# transcription against the real thing.
# =================================================================================================
def _rotate_half(x):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def eager_stock(t, freqs, mscale=1.0):
    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    cos_ = (torch.cos(freqs) * mscale).to(t.dtype)
    sin_ = (torch.sin(freqs) * mscale).to(t.dtype)
    t = (t * cos_) + (_rotate_half(t) * sin_)
    return torch.cat((t, t_pass), dim=-1)


def eager_fp32_patched(t, freqs, mscale=1.0):
    """megatron_patches' variant -- the WRONG reference for this kernel. Used as a control."""
    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    in_dtype = t.dtype
    cos_ = (torch.cos(freqs) * mscale).to(in_dtype).to(torch.float32)
    sin_ = (torch.sin(freqs) * mscale).to(in_dtype).to(torch.float32)
    tf = t.to(torch.float32)
    out = (tf * cos_) + (_rotate_half(tf) * sin_)
    return torch.cat((out.to(in_dtype), t_pass), dim=-1)


def make_freqs(S, R, seed=0, device=DEV):
    """A real rotary frequency table: theta=1e7 (the model's rope_theta), positions 0..S-1."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    inv = 1.0 / (1.0e7 ** (torch.arange(0, R // 2, dtype=torch.float32) / (R // 2)))
    pos = torch.randperm(1 << 15, generator=g)[:S].float()  # ragged absolute positions, as decode has
    f = torch.outer(pos, inv)
    return torch.cat((f, f), dim=-1).to(device).view(S, 1, 1, R).contiguous()


def fused(t, freqs, mscale=1.0):
    cos_, sin_ = rf._cos_sin_for(freqs, t.dtype, mscale)
    return rf.fused_rope_bshd(t, cos_, sin_)


# =================================================================================================
print("[1] bitwise vs eager stock, every production shape x dtype x token count")
for D, R in SHAPES:
    for dt in (torch.bfloat16, torch.float16):
        for H in PROD_HEADS + (8,):
            for S in (1, 7, 320, 1024, 8192):
                torch.manual_seed(S * 131 + D * 7 + R + H)
                t = torch.randn(S, 1, H, D, dtype=dt, device=DEV)
                fq = make_freqs(S, R, seed=S + D)
                check(f"D={D} R={R} {str(dt)[6:]} H={H} S={S}", fused(t, fq), eager_stock(t, fq))
# batch > 1 (prefill can pack b>1 on the trainer-shaped path; the kernel indexes s = row//(B*H))
for B in (2, 3):
    torch.manual_seed(B)
    t = torch.randn(64, B, 2, PROD_D, dtype=torch.bfloat16, device=DEV)
    fq = make_freqs(64, PROD_R, seed=B)
    check(f"B={B}", fused(t, fq), eager_stock(t, fq))
# STRIDED input. `query`/`key` reach RoPE from a split of the fused qkv projection, and whether
# that split lands contiguous depends on qk_layernorm and the rank's head counts. If the kernel
# required contiguity it would silently fall back on some shard geometries -- correct and worthless.
for nm, mk in (
    ("head-slice", lambda: torch.randn(320, 1, 6, PROD_D, dtype=torch.bfloat16, device=DEV)[:, :, 1:3]),
    ("feature-slice", lambda: torch.randn(320, 1, 2, PROD_D * 2, dtype=torch.bfloat16, device=DEV)[..., :PROD_D]),
    ("seq-slice", lambda: torch.randn(640, 1, 2, PROD_D, dtype=torch.bfloat16, device=DEV)[::2]),
):
    torch.manual_seed(hash(nm) % 10000)
    ts = mk()
    fq = make_freqs(ts.shape[0], PROD_R, seed=11)
    expect(f"strided {nm} is eligible-shaped", ts.stride(-1) == 1 and not ts.is_contiguous(), f"{ts.stride()}")
    check(f"strided {nm}", fused(ts, fq), eager_stock(ts, fq))
# mscale != 1.0 (yarn concentration factor). Must be part of the cache key AND the arithmetic.
for ms in (1.0, 0.7071067811865476, 1.3):
    torch.manual_seed(int(ms * 1000))
    t = torch.randn(320, 1, 2, PROD_D, dtype=torch.bfloat16, device=DEV)
    fq = make_freqs(320, PROD_R, seed=9)
    check(f"mscale={ms}", fused(t, fq, ms), eager_stock(t, fq, ms))

# =================================================================================================
print("\n[2] the REFERENCE is the right one -- stock, not the fp32 patch")
torch.manual_seed(2)
t = torch.randn(512, 1, 2, PROD_D, dtype=torch.bfloat16, device=DEV)
fq = make_freqs(512, PROD_R, seed=2)
try:
    from megatron.core.models.common.embeddings.rope_utils import (
        _apply_rotary_pos_emb_bshd as _meg_bshd,
    )

    check("transcription == real megatron stock", eager_stock(t, fq), _meg_bshd(t, fq))
    check("kernel == real megatron stock", fused(t, fq), _meg_bshd(t, fq))
except Exception as e:  # pragma: no cover
    print(f"  SKIP megatron not importable ({type(e).__name__}: {e}) -- transcription unverified")
    _fails.append("megatron stock reference not verified")
# The control that makes [1] meaningful: the fp32 patched form is a DIFFERENT function. If these
# were equal, matching one would say nothing about matching the other and this whole section is air.
expect_differs("stock vs fp32-patched rope", eager_stock(t, fq), eager_fp32_patched(t, fq))
expect_differs("kernel vs fp32-patched rope", fused(t, fq), eager_fp32_patched(t, fq))

# =================================================================================================
print("\n[3] the HOIST -- and proof it is not inert")
rf.HOIST_STATS.update(calls=0, computes=0)
fq = make_freqs(320, PROD_R, seed=3)
for layer in range(10):  # 10 attention layers x {query, key} = the real per-forward call count
    for H in PROD_HEADS:
        torch.manual_seed(layer * 10 + H)
        t = torch.randn(320, 1, H, PROD_D, dtype=torch.bfloat16, device=DEV)
        check(f"hoisted layer={layer} H={H}", fused(t, fq), eager_stock(t, fq))
expect(
    "hoist: 1 compute per forward",
    rf.HOIST_STATS == {"calls": 20, "computes": 1},
    f"got {rf.HOIST_STATS} -- expected 20 calls, 1 compute",
)
print(f"  ok   {rf.hoist_report()}")
# POSITIVE CONTROL: a fresh freqs object per call must NOT hoist. If this also reported 1 compute,
# the counter would be measuring nothing and section [3] would pass with the hoist deleted.
rf.HOIST_STATS.update(calls=0, computes=0)
for i in range(20):
    fresh = make_freqs(320, PROD_R, seed=3)  # same VALUES, new OBJECT
    fused(torch.randn(320, 1, 2, PROD_D, dtype=torch.bfloat16, device=DEV), fresh)
expect(
    "hoist control: fresh freqs does not hoist",
    rf.HOIST_STATS["computes"] == 20,
    f"got {rf.HOIST_STATS} -- the compute counter has no teeth",
)
# dtype and mscale are part of the key: a second dtype/mscale on the SAME freqs must recompute,
# or a fp16 call would silently reuse a bf16 cos table.
rf.HOIST_STATS.update(calls=0, computes=0)
fq = make_freqs(64, PROD_R, seed=33)
for dt in (torch.bfloat16, torch.bfloat16, torch.float16, torch.bfloat16):
    for ms in (1.0, 1.3):
        fused(torch.randn(64, 1, 2, PROD_D, dtype=dt, device=DEV), fq, ms)
expect(
    "hoist key covers (dtype, mscale)",
    rf.HOIST_STATS["computes"] >= 4,
    f"got {rf.HOIST_STATS} -- a key collision would serve the wrong angles",
)
# and the cache dies with the tensor: no registry, nothing to invalidate.
expect("cache lives on the freqs tensor", hasattr(fq, rf._CACHE_ATTR), "cache is not on the tensor")

# =================================================================================================
print("\n[4] FFMA contraction -- NOT immune: enable_fp_fusion=False is load-bearing")


@triton.jit
def _ffma_probe(T, C, S_, O, n, D: tl.constexpr, R: tl.constexpr, BD: tl.constexpr):
    """`x*c + r*s` kept in fp32 -- the form contraction CAN change. Not the production chain."""
    row = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, BD)
    m = (row < n) & (d < R)
    half: tl.constexpr = R // 2
    x = tl.load(T + row * D + d, mask=m, other=0.0)
    part = tl.where(d < half, d + half, d - half)
    xp = tl.load(T + row * D + part, mask=m, other=0.0)
    r = tl.where(d < half, -xp, xp)
    c = tl.load(C + d, mask=m, other=0.0)
    s = tl.load(S_ + d, mask=m, other=0.0)
    tl.store(O + row * D + d, x * c + r * s, mask=m)


torch.manual_seed(4)
N = 4096
tf = torch.randn(N, PROD_D, dtype=torch.float32, device=DEV)
cf = torch.randn(PROD_R, dtype=torch.float32, device=DEV)
sf = torch.randn(PROD_R, dtype=torch.float32, device=DEV)
outs = {}
for fuse in (False, True):
    o = torch.zeros_like(tf)
    _ffma_probe[(N,)](tf, cf, sf, o, N, D=PROD_D, R=PROD_R, BD=256, num_warps=4, enable_fp_fusion=fuse)
    outs[fuse] = o
# CONTROL: on an fp32 population contraction is observable. If this passed as "identical" the
# check below would be measuring nothing -- which is exactly how O3's FFMA control went vacuous.
expect_differs("FFMA control: fp32 operands, fusion on vs off", outs[False], outs[True])
tfr = tf[:, :PROD_R]
rot = torch.cat((-tfr[:, PROD_R // 2 :], tfr[:, : PROD_R // 2]), dim=-1)
eager_seq = tfr * cf + rot * sf
addcmul = torch.addcmul(rot * sf, tfr, cf)
expect_differs("FFMA control: eager mul+add vs addcmul", eager_seq, addcmul)
check("FFMA control: fused kernel == addcmul", outs[True][:, :PROD_R], addcmul)
# NOW THE PRODUCTION CHAIN -- and it is NOT immune, which is the opposite of what the "bf16
# operands are exact" argument predicts. LLVM contracts one of the two products into the add and
# ELIDES THAT PRODUCT'S EXPLICIT `.to(bf16)`; the sibling's round survives. So the control here is
# the production kernel itself, not a synthetic fp32 stand-in: `enable_fp_fusion=False` is
# load-bearing, and this asserts it.
torch.manual_seed(44)
t = torch.randn(2048, 1, 2, PROD_D, dtype=torch.bfloat16, device=DEV)
fq = make_freqs(2048, PROD_R, seed=44)
cos_, sin_ = rf._cos_sin_for(fq, t.dtype, 1.0)
S, B, H, D = t.shape
got = {}
for fuse in (False, True):
    o = torch.empty_like(t)
    rf._rope_kernel[(triton.cdiv(S * B * H, 8),)](
        t,
        cos_,
        sin_,
        o,
        S * B * H,
        B * H,
        H,
        t.stride(0),
        t.stride(1),
        t.stride(2),
        D=D,
        R=PROD_R,
        BD=256,
        MB=8,
        num_warps=4,
        enable_fp_fusion=fuse,
    )
    got[fuse] = o
expect_differs("production chain: fp_fusion ON vs OFF", got[True], got[False])
check("production chain: fp_fusion OFF == eager", got[False], eager_stock(t, fq))
# and identify WHAT it contracts, so a future Triton that changes this is caught rather than
# silently re-blessed: fuse=True keeps p2's bf16 round and folds x*cos into the add.
xr = t[..., :PROD_R].float()
rot_r = torch.cat((-xr[..., PROD_R // 2 :], xr[..., : PROD_R // 2]), dim=-1)
p2_ref = (rot_r * sin_.float()).to(torch.bfloat16)
contracted = torch.addcmul(p2_ref.float(), xr, cos_.float()).to(torch.bfloat16)
check("fp_fusion ON == bf16(fma(x, cos, bf16(rot*sin)))", got[True][..., :PROD_R], contracted)

# =================================================================================================
print("\n[5] FTZ -- driven subnormal, with a positive control that must flush")


@triton.jit
def _ftz_control(X, O, n, BLOCK: tl.constexpr):
    """`libdevice.div_rn(x, 1.0)` -- a mathematical identity that FTZ turns into a zeroing op."""
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = p < n
    x = tl.load(X + p, mask=m, other=0.0)
    tl.store(O + p, libdevice.div_rn(x, 1.0), mask=m)


# bf16 subnormals live in [1e-41, 1.18e-38); scale a randn population into that window so the
# rope OUTPUT is subnormal too (cos/sin are O(1), so the product stays in the window).
Sf = 1024
torch.manual_seed(5)
t_sub = (torch.randn(Sf, 1, 2, PROD_D, device=DEV) * 3e-39).to(torch.bfloat16)
fq = make_freqs(Sf, PROD_R, seed=5)
ref_sub = eager_stock(t_sub, fq)
got_sub = fused(t_sub, fq)
st = assert_no_ftz("rope on bf16-subnormal activations", ref_sub.float(), got_sub.float(), fail_list=_fails)
_checks += 1
# and the control: the SAME population through a known-FTZ'd op must flush, or the population
# contains nothing detectable and the line above proves nothing.
flat = ref_sub.float().reshape(-1).contiguous()
oc = torch.empty_like(flat)
_ftz_control[(triton.cdiv(flat.numel(), 1024),)](flat, oc, flat.numel(), BLOCK=1024, enable_fp_fusion=False)
assert_no_ftz("FTZ control: libdevice.div_rn(x, 1.0)", flat, oc, fail_list=_fails, expect_flush=True)
_checks += 1
print(f"  (population carried {st['subnormal']} subnormal reference values)")

# =================================================================================================
print("\n[6] the tile is NOT the bitwise contract (no reduction) -- proven, not assumed")
torch.manual_seed(6)
t = torch.randn(4096, 1, 2, PROD_D, dtype=torch.bfloat16, device=DEV)
fq = make_freqs(4096, PROD_R, seed=6)
cos_, sin_ = rf._cos_sin_for(fq, t.dtype, 1.0)
S, B, H, D = t.shape
ref_tile = None
for mb, nw in ((8, 4), (1, 1), (4, 8), (16, 2)):
    o = torch.empty_like(t)
    rf._rope_kernel[(triton.cdiv(S * B * H, mb),)](
        t,
        cos_,
        sin_,
        o,
        S * B * H,
        B * H,
        H,
        t.stride(0),
        t.stride(1),
        t.stride(2),
        D=D,
        R=PROD_R,
        BD=256,
        MB=mb,
        num_warps=nw,
        enable_fp_fusion=False,
    )
    if ref_tile is None:
        ref_tile = o
        check("tile (8,4) == eager", o, eager_stock(t, fq))
    else:
        check(f"tile ({mb},{nw}) == tile (8,4)", o, ref_tile)

# =================================================================================================
print("\n[7] unrepresentative inputs -- outlier and heavy-tailed rows, not randn")
fq = make_freqs(2048, PROD_R, seed=7)
pops = {}
torch.manual_seed(7)
x = torch.randn(2048, 1, 2, PROD_D, device=DEV)
x[:, 0, 0, 0] += 1e4  # massive activation, the shape that caught O2's bug
pops["outlier"] = x.to(torch.bfloat16)
pops["heavy_tail"] = (
    torch.randn(2048, 1, 2, PROD_D, device=DEV) * torch.exp(torch.randn(2048, 1, 2, 1, device=DEV) * 6)
).to(torch.bfloat16)
pops["huge"] = (torch.randn(2048, 1, 2, PROD_D, device=DEV) * 3e38).to(torch.bfloat16)  # inf/nan territory
pops["zeros"] = torch.zeros(2048, 1, 2, PROD_D, dtype=torch.bfloat16, device=DEV)
pops["signed_zero"] = (torch.zeros(2048, 1, 2, PROD_D, device=DEV) * -1.0).to(torch.bfloat16)
for nm, p in pops.items():
    r, g = eager_stock(p, fq), fused(p, fq)
    # BIT PATTERNS, not values. Two reasons: NaN != NaN, and -0.0 == +0.0 under `torch.equal` on
    # floats -- which would have hidden the `_rotate_half` negation bug entirely.
    check(f"pop={nm}", g.view(torch.int16), r.view(torch.int16))


# SIGN OF ZERO, with a positive control. Triton's unary `-x` lowers to `0.0 - x`, which returns
# +0.0 for a +0.0 input where torch's `neg` returns -0.0. The control below is the UNFIXED form:
# it must differ from torch, or the `* -1.0` in the kernel is protecting against nothing.
@triton.jit
def _neg_probe(X, O, n, MODE: tl.constexpr, BLOCK: tl.constexpr):
    p = tl.arange(0, BLOCK)
    m = p < n
    x = tl.load(X + p, mask=m, other=0.0)
    o = (x.to(tl.float32) * -1.0) if MODE else -x
    tl.store(O + p, o.to(X.dtype.element_ty), mask=m)


zsig = torch.tensor([0.0, -0.0, 1.0, -1.0, 3e-39, -3e-39], dtype=torch.bfloat16, device=DEV)
neg_ref = -zsig
for mode, nm in ((1, "x * -1.0 (what the kernel uses)"), (0, "-x (the unfixed form)")):
    o = torch.empty_like(zsig)
    _neg_probe[(1,)](zsig, o, zsig.numel(), MODE=mode, BLOCK=8, enable_fp_fusion=False)
    if mode:
        check(f"neg: {nm} == torch.neg", o.view(torch.int16), neg_ref.view(torch.int16))
    else:
        expect_differs(f"neg control: {nm} != torch.neg", o.view(torch.int16), neg_ref.view(torch.int16))

# =================================================================================================
print("\n[8] eligibility / fall-through -- every guard takes the ORIGINAL path")
rf._ORIG_BSHD = lambda t, freqs, rotary_interleaved=False, mla_rotary_interleaved=False, mscale=1.0: eager_stock(
    t, freqs, mscale
)
torch.manual_seed(8)
t = torch.randn(256, 1, 2, PROD_D, dtype=torch.bfloat16, device=DEV)
fq_marked = rf.mark_engine_rope(make_freqs(256, PROD_R, seed=8))
fq_plain = make_freqs(256, PROD_R, seed=8)
ref = eager_stock(t, fq_marked)
# THE ENGINE RUNS UNDER `torch.inference_mode()` (vLLM's model runner), so grad is disabled there.
# This block must reproduce that or the grad guard rejects everything and the "fused path" check
# below silently measures the FALLBACK -- which returns the same answer and passes. That is exactly
# how a green gate can bless a kernel that never ran.
with torch.no_grad():
    check("marked freqs -> fused path", rf._fused_apply_rotary_pos_emb_bshd(t, fq_marked), ref)
    expect("marked freqs is eligible", rf._eligible(t, fq_marked, False, False), "marked freqs rejected")
    expect("UNMARKED freqs is NOT eligible (no_grad)", not rf._eligible(t, fq_plain, False, False), "unmarked accepted")
    expect("rotary_interleaved is NOT eligible", not rf._eligible(t, fq_marked, True, False), "interleaved accepted")
    expect("mla_interleaved is NOT eligible", not rf._eligible(t, fq_marked, False, True), "mla accepted")
    expect("fp32 t is NOT eligible", not rf._eligible(t.float(), fq_marked, False, False), "fp32 accepted")
    # A non-unit LAST stride is the one layout the kernel cannot address (the feature axis uses an
    # implicit unit stride). It must fall through -- silently rotating the wrong lanes would not
    # fault. A merely non-contiguous t with unit last stride IS handled; section [1] gates it.
    _t_str = torch.randn(256, 1, 2, PROD_D, 2, dtype=torch.bfloat16, device=DEV)[..., 0]
    expect("non-unit last stride is NOT eligible", not rf._eligible(_t_str, fq_marked, False, False), "accepted")
    check(
        "non-unit last stride -> original path",
        rf._fused_apply_rotary_pos_emb_bshd(_t_str, fq_marked),
        eager_stock(_t_str, fq_marked),
    )
    _t_ok = torch.randn(256, 1, 4, PROD_D, dtype=torch.bfloat16, device=DEV)[:, :, :2]
    expect("non-contiguous (unit last stride) IS eligible", rf._eligible(_t_ok, fq_marked, False, False), "rejected")
    # inference_mode specifically -- what vLLM actually uses, and it is not the same object as
    # no_grad even though both disable grad.
    with torch.inference_mode():
        expect("eligible under inference_mode", rf._eligible(t, fq_marked, False, False), "inference_mode rejected")
# UNMARKED: this is the guard that keeps the fusion off the trainer even though the install point
# is a module global. It must be false, and the answer must still be right.
check("unmarked freqs -> original path", rf._fused_apply_rotary_pos_emb_bshd(t, fq_plain), ref)
# grad: a raw Triton call has no grad_fn, so a grad-enabled call must never reach it. This is the
# guard whose absence severed the MoE backward for five live steps under a green forward gate.
expect("grad enabled is NOT eligible", not rf._eligible(t, fq_marked, False, False), "fused path open under grad")
tg = t.clone().requires_grad_(True)
with torch.no_grad():
    expect(
        "requires_grad tensor is NOT eligible", not rf._eligible(tg, fq_marked, False, False), "grad tensor accepted"
    )

# =================================================================================================
print("\n[9] 150 sequential decode steps + CUDA graph replay")
bad = 0
rf.HOIST_STATS.update(calls=0, computes=0)
with torch.no_grad():  # as the engine runs -- otherwise every step takes the fallback and passes
    for step in range(150):
        torch.manual_seed(1000 + step)
        S = 320
        t = torch.randn(S, 1, 2, PROD_D, dtype=torch.bfloat16, device=DEV)
        fq = rf.mark_engine_rope(make_freqs(S, PROD_R, seed=step))  # new positions every step
        got = rf._fused_apply_rotary_pos_emb_bshd(t, fq)
        # bit patterns again: -0.0 == +0.0 would let the negation bug back through
        if not torch.equal(got.view(torch.int16), eager_stock(t, fq).view(torch.int16)):
            bad += 1
expect("150-step decode loop bitwise", bad == 0, f"{bad}/150 steps differed")
expect(
    "150-step loop actually took the FUSED path",
    rf.HOIST_STATS["calls"] == 150,
    f"got {rf.HOIST_STATS} -- the loop measured the fallback, not the kernel",
)

torch.manual_seed(9)
S = 320
t_static = torch.randn(S, 1, 2, PROD_D, dtype=torch.bfloat16, device=DEV)
fq_static = make_freqs(S, PROD_R, seed=99)
ref_graph = eager_stock(t_static, fq_static)
cos_, sin_ = rf._cos_sin_for(fq_static, t_static.dtype, 1.0)
out_static = torch.empty_like(t_static)
g = torch.cuda.CUDAGraph()
torch.cuda.synchronize()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        out_static.copy_(rf.fused_rope_bshd(t_static, cos_, sin_))
torch.cuda.current_stream().wait_stream(s)
with torch.cuda.graph(g):
    out_static.copy_(rf.fused_rope_bshd(t_static, cos_, sin_))
same = 0
for rep in range(20):
    g.replay()
    torch.cuda.synchronize()
    same += int(bitwise_equal(out_static, ref_graph))
expect("CUDA graph: 20/20 replays bitwise", same == 20, f"{same}/20 replays matched")

# determinism: the same input 20x must give the same bits (a racy kernel shows up here, though a
# race across THREADS in an in-place update would not -- this kernel has no in-place update and
# never reads memory another program writes, which is the structural reason O3's bug cannot recur).
torch.manual_seed(10)
t = torch.randn(4096, 1, 2, PROD_D, dtype=torch.bfloat16, device=DEV)
fq = make_freqs(4096, PROD_R, seed=10)
first = fused(t, fq)
expect("determinism 20/20", all(bitwise_equal(fused(t, fq), first) for _ in range(20)), "output varies run to run")

# =================================================================================================
print(f"\n{'=' * 72}")
if _fails:
    print(f"FAILED {len(_fails)} of {_checks} checks:")
    for f in _fails:
        print(f"  - {f}")
    sys.exit(1)
print(f"ALL {_checks} CHECKS PASS (torch.equal)")
