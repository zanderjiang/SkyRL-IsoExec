"""BITWISE gate for O6 -- the fused zero-centred RMSNorm and GDN gated output norm
(`isoexec.gdn_fused_outnorm`).

These kernels are TYPE (a): they must reproduce the eager expressions' exact rounding -- every bf16
down-cast at the same point -- and win only by not going to HBM between steps. A bitwise-equal
replacement may be installed on the engine alone, which is what keeps the trainer untouched and the
zero-KL gate immovable by construction. So the acceptance criterion is `torch.equal`. `allclose`
would tell us nothing: the wrong reduction tile costs 7 elements in 2,097,152.

What is checked:

  1. K2 -- `fused_rms_norm_gamma` vs `F.rms_norm(x, (N,), None, eps) * (1.0 + weight)`, over EVERY
     width the production config builds (2048 hidden, 256 attn head_dim, 128 GDN value head) x
     M in {1, 7, 320, 1280, 8192, 32768}
     (decode, mixed, prefill) -- plus a 16,384-row adversarial pass per width, because a wrong tile
     shows up at a rate of ~3e-6 and a small M cannot see it.
  2. K3 -- `fused_gated_out_norm` vs `_eager_apply_gated_norm`'s expression, with `gate` read from a
     STRIDED 1544-wide in_proj slice exactly as production does, over the same T sweep.
  3. THE TILE IS THE CONTRACT -- the deliberately-wrong tile (one warp too wide) is run and asserted
     to DIFFER. Without this the pinned `_tile_for` could be silently widened for occupancy and
     every other check here would still pass.
  4. SiLU FORM -- asserts ATen fp32 `F.silu` is `x / (1 + exp(-x))` and is NOT `x * sigmoid(x)`,
     which is the form the conv in `gdn_fused_prep` uses. Getting this backwards is a live bug that
     no shape sweep would catch.
  5. FFMA CONTRACTION -- the mandatory hazard check, with LIVE POSITIVE CONTROLS so it cannot pass
     vacuously. Positive control (a): an fp32 gamma makes eager != addcmul in a large fraction of
     elements, proving the comparison has teeth on this shape. Positive control (b): running the
     kernel with `enable_fp_fusion=True` changes its output, proving the flag we set is load-bearing
     here and not a no-op.
  6. K2a -- there is NO cached gamma, so there is nothing to go stale. `1.0 + weight` is formed in
     registers from the live parameter, so a weight write must be picked up with no notification at
     all: asserted for both a REPLACED Parameter object (what vllm_worker._set_on_module does) and an
     in-place write. This replaced a cached-gamma design whose failure mode -- engine silently serves
     last step's norm weights, forward-only gate stays green -- an offline bitwise gate structurally
     cannot see. It had already shipped one such bug here (the staleness stamp never matched, so it
     refreshed every forward and undid the whole hoist) with all 32 checks passing.
  6b. CUDA-GRAPH REPLAY across a weight change. Production decode is graph-replayed, which re-runs
     capture-time kernels against capture-time pointers and skips ALL Python. That is the single
     biggest structural gap between an eager harness and the live engine, and it is where a cached
     gamma would fail while every eager check above passed.
  6c. REPRESENTATIVE INPUTS. A test that passes at max|diff| 0.0 on `randn` and then fails live is a
     test whose inputs are unrepresentative -- this is the diagnosed gap in O3's suite, which passed
     58/58 offline and then moved the live gate to 2.07e-02. `randn` rows are well-conditioned; real
     residual streams carry massive-activation outliers, and an outlier DOMINATES the sum of squares,
     which is exactly the quantity whose fp32 reduction order this kernel must reproduce. Covers
     outlier rows, heavy-tailed rows, and near-denormal rows where eps placement decides the answer.
  6d. BATCH COMPOSITION. Live decode is a ragged 320-sequence batch; this harness runs fixed
     rectangles. A row alone must be bit-identical to that row inside batches of 1..1000.
  7. INSTALL -- `install_engine_fused_norms` reaches nested norms, skips non-norms, produces
     bitwise-identical output, and LEAVES THE CLASS UNTOUCHED. That last one is the whole reason the
     install is per-instance: the trainer imports and constructs the same class, and a class-level
     rebind would hand it an engine-only kernel with no grad_fn. No bitwise check can see that,
     because the kernel is bitwise-correct -- it is the install site that is wrong.

Run: CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=. python skyrl/backends/skyrl_train/isoexec/ops/norms/tests/fused_outnorm_gpu.py
Exit 0 iff every check is exact.
"""

import os
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


if not torch.cuda.is_available():  # promoted nightly battery: needs one CUDA device
    print("SKIP: no CUDA device")
    raise SystemExit(0)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 7)))  # repo root

from skyrl.backends.skyrl_train.isoexec.core import triton_nonftz as _nonftz  # noqa: E402
from skyrl.backends.skyrl_train.isoexec.ops.norms.fused_outnorm import (  # noqa: E402
    _gated_out_norm_kernel,
    _tile_for,
    fused_gated_out_norm,
    fused_rms_norm_gamma,
)

# EVERY norm width Qwen3.5-35B-A3B actually builds, from its config.json -- not a round number I
# liked. 2048 = hidden (main path, unsharded: sequence_parallel is off); 256 = attn head_dim, the
# q_norm/k_norm sites; 128 = linear_value_head_dim, the GDN out-norm. THE TILE IS CHOSEN PER WIDTH,
# so a width that is installed but not gated is a bitwise contract nobody checked. 256 was missing
# from the first version of this gate while `install_engine_fused_norms` happily swapped it.
PROD_WIDTHS = (2048, 256, 128)

DEV = "cuda"
EPS = 1e-6
IN_PROJ = 1544  # the real Qwen3.5-A3B in_proj width `gate` is a last-dim slice of
_fails = []
_checks = 0


def check(name, a, b):
    global _checks
    _checks += 1
    ok = a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
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


def make_gamma(N, seed):
    torch.manual_seed(seed)
    return (1.0 + torch.randn(N, dtype=torch.bfloat16, device=DEV) * 0.05).to(torch.bfloat16)


# =================================================================================================
print("[1] K2 -- fused_rms_norm_gamma vs F.rms_norm(x) * (1.0 + weight)")
for N in PROD_WIDTHS:
    w = torch.randn(N, dtype=torch.bfloat16, device=DEV) * 0.05
    gamma = (1.0 + w).to(torch.bfloat16)
    for M in (1, 7, 320, 1280, 8192, 32768):
        torch.manual_seed(M * 31 + N)
        x = torch.randn(M, N, dtype=torch.bfloat16, device=DEV)
        ref = F.rms_norm(x, (N,), None, EPS) * (1.0 + w)
        check(f"K2 N={N} M={M}", fused_rms_norm_gamma(x, w, EPS), ref)
    # 3-D input, the sbhd shape the main-path norms actually see
    torch.manual_seed(N)
    x3 = torch.randn(320, 1, N, dtype=torch.bfloat16, device=DEV)
    check(f"K2 N={N} sbhd[320,1,{N}]", fused_rms_norm_gamma(x3, w, EPS), F.rms_norm(x3, (N,), None, EPS) * (1.0 + w))
    # adversarial volume: a wrong tile differs at ~3e-6, so small M is blind to it
    torch.manual_seed(99)
    xb = torch.randn(16384, N, dtype=torch.bfloat16, device=DEV)
    check(f"K2 N={N} M=16384 (volume)", fused_rms_norm_gamma(xb, w, EPS), F.rms_norm(xb, (N,), None, EPS) * (1.0 + w))

# =================================================================================================
print("[2] K3 -- fused_gated_out_norm vs _eager_apply_gated_norm, gate read strided")
DV, HV = 128, 4
w3 = torch.randn(DV, dtype=torch.bfloat16, device=DEV) * 0.05  # RAW zero-centred weight


def eager_k3(x, gate):
    """_eager_apply_gated_norm with a ZeroCenteredTorchRMSNorm out_norm and act_fn=F.silu."""
    xd = x.dtype
    x2 = x.reshape(-1, x.shape[-1])
    y = F.rms_norm(x2, (x2.shape[-1],), None, EPS) * (1.0 + w3)
    g2 = gate.reshape(-1, gate.shape[-1])
    return (y * F.silu(g2.float())).to(xd)


for T in (1, 3, 17, 320, 8192):
    torch.manual_seed(T)
    x = torch.randn(T, HV, DV, dtype=torch.bfloat16, device=DEV)
    proj = torch.randn(T, IN_PROJ, dtype=torch.bfloat16, device=DEV)
    gate = proj[:, 300 : 300 + HV * DV].view(T, HV, DV)  # strided exactly like production
    check(f"K3 T={T}", fused_gated_out_norm(x, gate, w3, EPS), eager_k3(x, gate))

torch.manual_seed(4242)
Tb = 16384
xb = torch.randn(Tb, HV, DV, dtype=torch.bfloat16, device=DEV)
pb = torch.randn(Tb, IN_PROJ, dtype=torch.bfloat16, device=DEV)
gb = pb[:, 300 : 300 + HV * DV].view(Tb, HV, DV)
check("K3 T=16384 (volume)", fused_gated_out_norm(xb, gb, w3, EPS), eager_k3(xb, gb))

# =================================================================================================
print("[3] the tile IS the contract -- a wider tile must DIFFER")
mb, nw = _tile_for(DV)
print(f"  pinned tile for N={DV}: rows/program={mb} num_warps={nw} (quack threads_per_row={32 * nw // mb})")
out_wide = torch.empty(Tb * HV, DV, dtype=torch.bfloat16, device=DEV)
_gated_out_norm_kernel[(triton.cdiv(Tb * HV, mb),)](
    xb,
    gb,
    w3,
    out_wide,
    Tb * HV,
    gb.stride(0),
    EPS,
    HV=HV,
    N=DV,
    BN=DV,
    MB=mb,
    num_warps=nw * 2,
    enable_fp_fusion=False,
)
expect_differs("K3 tile widened to 2x warps", out_wide, eager_k3(xb, gb))

# =================================================================================================
print("[4] SiLU form -- ATen fp32 silu is x/(1+exp(-x)), NOT x*sigmoid(x)")
torch.manual_seed(11)
g = torch.randn(1 << 22, dtype=torch.float32, device=DEV) * 4
check("silu == x/(1+exp(-x))", F.silu(g), g / (1 + torch.exp(-g)))
expect_differs("silu vs x*sigmoid(x)", F.silu(g), g * torch.sigmoid(g))

# =================================================================================================
print("[5] FFMA contraction -- with live positive controls")


@triton.jit
def _mac(a_ptr, b_ptr, c_ptr, o_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    a = tl.load(a_ptr + i, mask=m, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + i, mask=m, other=0.0).to(tl.float32)
    c = tl.load(c_ptr + i, mask=m, other=0.0).to(tl.float32)
    tl.store(o_ptr + i, (a + b * c).to(o_ptr.dtype.element_ty), mask=m)


def mac(a, b, c, fuse):
    o = torch.empty_like(a, dtype=torch.float32)
    n = a.numel()
    _mac[(triton.cdiv(n, 1024),)](a, b, c, o, n, BLOCK=1024, enable_fp_fusion=fuse)
    return o


torch.manual_seed(77)
n = 1 << 20
a32 = torch.randn(n, dtype=torch.float32, device=DEV)
b32 = torch.randn(n, dtype=torch.float32, device=DEV)
c32 = torch.randn(n, dtype=torch.float32, device=DEV)
eager = a32 + b32 * c32
addcmul = torch.addcmul(a32, b32, c32)
# Positive control (a): the comparison has teeth on fp32 operands.
expect_differs("CONTROL eager(mul,add) vs addcmul", eager, addcmul)
check("triton enable_fp_fusion=False == eager", mac(a32, b32, c32, False), eager)
# Positive control (b): the flag we set is load-bearing -- turning it on changes the answer.
expect_differs("CONTROL enable_fp_fusion True vs False", mac(a32, b32, c32, True), mac(a32, b32, c32, False))

# Positive control (c): the same test on the K3 kernel itself, via a gamma promoted to fp32-valued
# bf16 -- proving the production kernel's own accumulate is on the protected path.
out_fused = torch.empty(Tb * HV, DV, dtype=torch.bfloat16, device=DEV)
_gated_out_norm_kernel[(triton.cdiv(Tb * HV, mb),)](
    xb,
    gb,
    w3,
    out_fused,
    Tb * HV,
    gb.stride(0),
    EPS,
    HV=HV,
    N=DV,
    BN=DV,
    MB=mb,
    num_warps=nw,
    enable_fp_fusion=True,
)
ref_k3 = eager_k3(xb, gb)
if torch.equal(out_fused, ref_k3):
    print("  note K3 is STRUCTURALLY IMMUNE to contraction here (no `acc + b*c` survives the")
    print("       reduction tree, so fp_fusion has nothing to contract). The generic check above")
    print("       is what carries this hazard; see PERFORM_GAMEPLAN §0 on vacuous FFMA checks.")
else:
    print(f"  ok   K3 with fp_fusion=True differs in {(out_fused != ref_k3).sum().item()} -- flag load-bearing")

# =================================================================================================
print("[6] K2a -- there is no cached gamma, so there is nothing to go stale")
# The spec's K2a caches `1.0 + weight` and invalidates it at the weight-sync boundary. This
# implementation folds the add into the kernel instead, so a weight write is picked up with no
# notification at all. Assert that directly: mutate the weight the way a sync does (REPLACING the
# Parameter object, which is what vllm_worker._set_on_module does) and the very next call must
# already reflect it, with nothing in between.


class _Norm(torch.nn.Module):
    def __init__(self, N):
        super().__init__()
        self.hidden_size = (N,)
        self.eps = EPS
        self.weight = torch.nn.Parameter(torch.randn(N, dtype=torch.bfloat16, device=DEV) * 0.05)


m = _Norm(2048)
torch.manual_seed(3)
xk = torch.randn(320, 2048, dtype=torch.bfloat16, device=DEV)
check("K2a t0 == eager", fused_rms_norm_gamma(xk, m.weight, EPS), F.rms_norm(xk, (2048,), None, EPS) * (1.0 + m.weight))
with torch.no_grad():
    m._parameters["weight"] = torch.nn.Parameter(m.weight.detach() + 0.5)
check(
    "K2a picks up a REPLACED Parameter with no refresh call",
    fused_rms_norm_gamma(xk, m.weight, EPS),
    F.rms_norm(xk, (2048,), None, EPS) * (1.0 + m.weight),
)
with torch.no_grad():
    m.weight.add_(0.25)  # and an IN-PLACE write, the other thing a sync does
check(
    "K2a picks up an IN-PLACE weight write with no refresh call",
    fused_rms_norm_gamma(xk, m.weight, EPS),
    F.rms_norm(xk, (2048,), None, EPS) * (1.0 + m.weight),
)

# =================================================================================================
print("[6b] CUDA-GRAPH REPLAY after a weight change -- the live decode path, which eager tests miss")
# Production decode is graph-REPLAYED. A replay re-runs capture-time kernels against capture-time
# pointers and skips all Python, so anything that depended on host code running between syncs is
# silently dead on the replay path. This is the single biggest structural difference between this
# harness and the live engine, and it is where a cached-gamma design would fail while every eager
# check above passed. Capture, change the weights IN PLACE (a sync writes through the parameter's
# storage), replay, and demand the new weights.
gm = _Norm(2048)
static_x = torch.randn(320, 2048, dtype=torch.bfloat16, device=DEV)
static_out = torch.empty_like(static_x)

_s = torch.cuda.Stream()
_s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(_s):
    for _ in range(3):
        static_out.copy_(fused_rms_norm_gamma(static_x, gm.weight, EPS))
torch.cuda.current_stream().wait_stream(_s)
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    static_out.copy_(fused_rms_norm_gamma(static_x, gm.weight, EPS))
torch.cuda.synchronize()
graph.replay()
torch.cuda.synchronize()
check(
    "graph replay == eager (pre-sync weights)",
    static_out.clone(),
    F.rms_norm(static_x, (2048,), None, EPS) * (1.0 + gm.weight),
)

with torch.no_grad():
    gm.weight.add_(0.75)  # what a weight sync does to the parameter's storage
graph.replay()
torch.cuda.synchronize()
check(
    "graph replay sees POST-SYNC weights with no host code in between",
    static_out.clone(),
    F.rms_norm(static_x, (2048,), None, EPS) * (1.0 + gm.weight),
)

# =================================================================================================
print("[6c] REPRESENTATIVE INPUTS -- randn is not what a residual stream looks like")
# A parity test that passes at max|diff| 0.0 on randn and then fails live is a test whose INPUTS are
# unrepresentative. Real hidden states carry massive-activation outliers (orders of magnitude above
# the rest of the row), and an outlier dominates a sum of squares -- which is exactly the quantity
# whose fp32 reduction ORDER this kernel has to reproduce. randn rows are well-conditioned and hide
# tile mismatches; these do not.
for N in PROD_WIDTHS:
    w = torch.randn(N, dtype=torch.bfloat16, device=DEV) * 0.05
    torch.manual_seed(N)
    M = 8192
    x = torch.randn(M, N, dtype=torch.bfloat16, device=DEV)
    # one massive activation per row, at a random column, 100-10000x the rest
    cols = torch.randint(0, N, (M,), device=DEV)
    mags = (10.0 ** torch.randint(2, 5, (M,), device=DEV).float()).to(torch.bfloat16)
    x[torch.arange(M, device=DEV), cols] = mags
    check(f"outlier rows N={N}", fused_rms_norm_gamma(x, w, EPS), F.rms_norm(x, (N,), None, EPS) * (1.0 + w))
    # heavy-tailed whole rows (Cauchy-ish): wide dynamic range WITHIN the reduction
    xc = (torch.randn(M, N, device=DEV) / torch.randn(M, N, device=DEV).clamp(min=1e-3)).to(torch.bfloat16)
    xc = torch.nan_to_num(xc, nan=0.0, posinf=1e4, neginf=-1e4)
    check(f"heavy-tailed rows N={N}", fused_rms_norm_gamma(xc, w, EPS), F.rms_norm(xc, (N,), None, EPS) * (1.0 + w))
    # near-denormal rows: sum of squares underflows toward eps, so eps placement decides the answer
    xs = (torch.randn(M, N, device=DEV) * 1e-20).to(torch.bfloat16)
    check(
        f"tiny rows (eps dominates) N={N}",
        fused_rms_norm_gamma(xs, w, EPS),
        F.rms_norm(xs, (N,), None, EPS) * (1.0 + w),
    )

# =================================================================================================
print("[6e] FTZ -- subnormal SiLU quotients, the hazard that shipped past O2's 130-check gate")
# Triton links libdevice with __CUDA_FTZ, so `libdevice.div_rn` flushes a SUBNORMAL QUOTIENT to zero
# (64,940/65,536). K3's SiLU is `g / (1 + exp(-g))`: both operands are always normal (the denominator
# is >= 1, the numerator comes from bf16), but for g below about -88 the QUOTIENT is fp32 subnormal
# and that is what gets flushed. ATen keeps it.
#
# NOTE WHY THIS IS A SEPARATE CHECK: [6c] injects outliers into `x`, the norm input. It never touches
# `gate`, so it cannot reach this at all. An "adversarial inputs" section is only adversarial for the
# tensor it perturbs -- that is the same blind spot, one tensor over.
for lo, hi, name in (
    (-120.0, -80.0, "deep subnormal quotients"),
    (-95.0, -85.0, "the FTZ knee"),
    (-40.0, 40.0, "ordinary"),
):
    torch.manual_seed(int(-lo))
    Tg, HVg = 4096, HV
    xg = torch.randn(Tg, HVg, DV, dtype=torch.bfloat16, device=DEV)
    pg = torch.empty(Tg, IN_PROJ, dtype=torch.bfloat16, device=DEV).uniform_(lo, hi)
    gg = pg[:, 300 : 300 + HVg * DV].view(Tg, HVg, DV)
    check(f"K3 gate in [{lo:.0f},{hi:.0f}] ({name})", fused_gated_out_norm(xg, gg, w3, EPS), eager_k3(xg, gg))

# The control: `libdevice.div_rn` FTZ is real, but K3's SiLU is STRUCTURALLY IMMUNE to it, and the
# reason is worth stating because it does NOT generalise to the neighbouring expression.
#
#   sigmoid  1/(1+exp(-g)) : numerator 1.0. At g=-88 the denominator is 1.65e38 and the quotient is
#                            6.1e-39 -- SUBNORMAL. libdevice.div_rn flushes it. EXPOSED.
#   silu     g/(1+exp(-g)) : numerator |g| ~ 88. The same denominator gives 5.3e-37, which is 45x
#                            ABOVE the 1.18e-38 boundary; and by g=-88.7 the fp32 denominator has
#                            OVERFLOWED to inf, so the quotient is exactly -0.0 in both forms. The
#                            quotient can never land in the subnormal window at all. NOT EXPOSED.
#
# So a control built on our own expression proves nothing, and saying "the FTZ check passed" would be
# the vacuous-control failure §0 warns about. Run BOTH forms through the same kernel: silu must be
# identical either way (structural immunity) and sigmoid must DIFFER (the check has teeth, and would
# have fired had K3 been sigmoid-shaped). Measured min|nonzero quotient|: silu 2.6e-37, sigmoid
# 2.9e-39.
import triton.language as _tl  # noqa: E402
from triton.language.extra import libdevice as _ld  # noqa: E402


@triton.jit
def _silu_ftz(g_ptr, o_ptr, n, FORM: _tl.constexpr, FTZ: _tl.constexpr, BLOCK: _tl.constexpr):
    i = _tl.program_id(0) * BLOCK + _tl.arange(0, BLOCK)
    m = i < n
    g = _tl.load(g_ptr + i, mask=m, other=0.0)
    den = 1.0 + _ld.exp(-g)
    num = 1.0 if FORM == 0 else g
    r = _ld.div_rn(num, den) if FTZ == 1 else _nonftz.div_rn(num, den)
    _tl.store(o_ptr + i, r, mask=m)


def _sweep(form, ftz, gs):
    o = torch.empty_like(gs)
    _silu_ftz[(triton.cdiv(gs.numel(), 1024),)](
        gs, o, gs.numel(), FORM=form, FTZ=ftz, BLOCK=1024, enable_fp_fusion=False
    )
    return o


gsweep = torch.empty(1 << 16, dtype=torch.float32, device=DEV).uniform_(-95.0, -87.0)
check("K3's silu form: FTZ and non-FTZ divide agree (structurally immune)", _sweep(1, 1, gsweep), _sweep(1, 0, gsweep))
check("K3's silu form == ATen silu over the same sweep", _sweep(1, 0, gsweep), F.silu(gsweep))
expect_differs("CONTROL sigmoid form DOES flush (so this check has teeth)", _sweep(0, 1, gsweep), _sweep(0, 0, gsweep))
_nz = _sweep(1, 0, gsweep)
_nz = _nz[_nz != 0].abs()
print(
    f"       silu min|nonzero quotient| = {_nz.min().item():.3e} vs subnormal boundary 1.175e-38 "
    f"-> {_nz.min().item() / 1.17549435e-38:.0f}x above it"
)

# =================================================================================================
print("[6d] BATCH COMPOSITION -- a row's output must not depend on its neighbours")
# Live decode is a ragged 320-sequence batch whose composition changes every step; this harness runs
# fixed rectangles. The kernels are row-independent by construction (the reduction is along the
# feature axis only), so assert exactly that rather than trusting it: a row computed alone must be
# bit-identical to the same row inside a batch, and inside a batch of a different size.
for N in PROD_WIDTHS:
    w = torch.randn(N, dtype=torch.bfloat16, device=DEV) * 0.05
    torch.manual_seed(N + 1)
    big = torch.randn(1000, N, dtype=torch.bfloat16, device=DEV)
    full = fused_rms_norm_gamma(big, w, EPS)
    check(f"row-independence N={N}: alone vs in 1000", fused_rms_norm_gamma(big[7:8], w, EPS), full[7:8])
    for sub in (1, 2, 3, 31, 33, 320):
        if not torch.equal(fused_rms_norm_gamma(big[:sub], w, EPS), full[:sub]):
            _fails.append(f"batch-composition N={N} M={sub}: rows changed with batch size")
            print(f"  FAIL batch-composition N={N} M={sub}")
            break
    else:
        print(f"  ok   N={N}: rows unchanged across M in {{1,2,3,31,33,320,1000}} (tile is M-independent)")
    _checks += 1

print("[7] install -- the per-INSTANCE rebind, which is what keeps this off the trainer")
import os  # noqa: E402

os.environ["SKYRL_ISOEXEC_GDN_FUSED_OUTNORM"] = "1"
import skyrl.backends.skyrl_train.isoexec.ops.norms.fused_outnorm as _fo  # noqa: E402
from skyrl.backends.skyrl_train.isoexec.ops.norms.fused_outnorm import (  # noqa: E402
    install_engine_fused_norms,
)
from skyrl.backends.skyrl_train.isoexec.ops.norms.zero_centered_norm import (  # noqa: E402
    ZeroCenteredTorchRMSNorm,
)

_fo._LOGGED = False


class _Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = ZeroCenteredTorchRMSNorm(2048, EPS, params_dtype=torch.bfloat16)
        self.inner = torch.nn.Module()
        self.inner.b = ZeroCenteredTorchRMSNorm(128, EPS, params_dtype=torch.bfloat16)
        self.lin = torch.nn.Linear(8, 8)  # must be skipped


toy = _Toy()
with torch.no_grad():
    toy.a.weight.normal_(0, 0.05)
    toy.inner.b.weight.normal_(0, 0.05)
# Capture eager answers from a PRISTINE instance of the same class before installing anything.
pristine_a = ZeroCenteredTorchRMSNorm(2048, EPS, params_dtype=torch.bfloat16)
pristine_b = ZeroCenteredTorchRMSNorm(128, EPS, params_dtype=torch.bfloat16)
with torch.no_grad():
    pristine_a.weight.copy_(toy.a.weight)
    pristine_b.weight.copy_(toy.inner.b.weight)

torch.manual_seed(21)
xa = torch.randn(320, 1, 2048, dtype=torch.bfloat16, device=DEV)
xb2 = torch.randn(1280, 128, dtype=torch.bfloat16, device=DEV)
ref_a, ref_b = pristine_a(xa), pristine_b(xb2)

n_inst = install_engine_fused_norms(toy)
_checks += 1
if n_inst != 2:
    _fails.append(f"install swapped {n_inst} norms, expected 2 (nested modules must be reached)")
    print(f"  FAIL install swapped {n_inst}/2")
else:
    print("  ok   install swapped 2 norms including the nested one, skipped the Linear")
check("install: swapped forward == eager (N=2048)", toy.a(xa), ref_a)
check("install: swapped forward == eager (N=128)", toy.inner.b(xb2), ref_b)

# THE POINT OF PER-INSTANCE INSTALL: the CLASS must be untouched, or the trainer -- which imports
# and constructs this very class -- inherits an engine-only kernel. That is the install_fused_combine
# failure verbatim, and it is invisible to every bitwise check because the kernel is bitwise-correct.
_checks += 1
if ZeroCenteredTorchRMSNorm.forward is not type(pristine_a).forward or hasattr(pristine_a, "_ix_fused_norm"):
    _fails.append("install leaked onto the CLASS -- the trainer would get the kernel too")
    print("  FAIL install mutated the class, not just the instances")
else:
    print("  ok   the CLASS is untouched: a fresh instance still runs eager (trainer is safe)")
check("install: a fresh instance is still eager", pristine_a(xa), ref_a)

# =================================================================================================
print()
if _fails:
    print(f"FAILED {len(_fails)}/{_checks} checks:")
    for f in _fails:
        print("  - " + f)
    sys.exit(1)
print(f"ALL {_checks}/{_checks} CHECKS EXACT")
