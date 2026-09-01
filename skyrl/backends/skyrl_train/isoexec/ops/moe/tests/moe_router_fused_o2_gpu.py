"""O2 -- fused MoE router (top-k + softmax + dense scatter) and permute sort: BITWISE gate.

Ledger cluster **P8** (the private decode-kernel ledger): 952.0 launches/step,
2.529 ms of a 33.93 ms 35B-A3B TP8 decode step, 23.8 launches per MoE layer. Opportunity **O2**
prices removing 792 of them at **2.13 ms**. `moe_router_o2_kernel` replaces the cluster with three
Triton kernels. This asserts the replacement is `torch.equal` to the eager sequence, not close to it.

The kernel touches `exp`, a division AND an fp32 reduction in one op, so three of the four known
Triton-vs-ATen hazards land here at once and the fourth is checked with live positive controls.

  REF     the reference transcription is FAITHFUL: with vLLM's aten overrides installed,
          `torch.softmax` really is `softmax_batch_invariant`'s five-op decomposition, and the whole
          reference really is megatron's `topk_routing_with_score_function`. Gate the gate first.
  HAZARD  the three numerics hazards, MEASURED on this shape: tl.exp vs libdevice.exp, Triton `/`
          vs libdevice.div_rn, and four candidate fp32 reduction orders for the softmax denominator.
          Prints element-diff counts so "close enough" has a scale.
  FFMA    hazard 4, with TWO LIVE POSITIVE CONTROLS so the check is not vacuous (O3's turned out
          structurally immune, and would have passed with the protection deleted): (a) an fp32
          `acc += b*c` probe whose result flips with `enable_fp_fusion`, (b) an `eager != addcmul`
          control on the same shape. Then: the real kernel is bit-identical under BOTH settings and
          equal to the EAGER form. Matching addcmul would mean NOT matching the trainer.
  ROUTER  dense (routing_probs, routing_map) `torch.equal` vs eager at ragged decode (320x1 token),
          prefill to 8192, E=256 and E non-power-of-2, k in {1,2,4,8,16}, with/without scaling.
  TIES    EXACT ties: an all-equal row, ties straddling the top-k boundary, 128 tied even indices,
          ties at both ends of the index range. Membership is not a rounding detail -- it changes
          which experts run.
  SORT    the counting sort vs `argsort(descending=True, stable=True)` + `remainder` + gather:
          sorted_indices, permuted_probs, tokens_per_expert, incl. empty experts (leading /
          trailing / interior), all-tokens-to-one-expert, E=1, and T*k straddling the tile.
  DEGEN   NaN/inf rows (the class of input that crashed vLLM engine init via an out-of-range expert
          index), T=1, the host-side envelope fallbacks.
  SYNC    determinism, `set_sync_debug_mode("error")`, and CUDA-graph capture + replay.
  BENCH   A/B of the full P8 cluster, eager vs fused, INSIDE A REPLAYED CUDA GRAPH (production
          decode is graph-replayed; an eager-loop timing is a fantasy -- see O3's note).

Run: CUDA_VISIBLE_DEVICES=0 VLLM_BATCH_INVARIANT=1 uv run --isolated --extra isoexec \
       python skyrl/backends/skyrl_train/isoexec/ops/moe/tests/moe_router_fused_o2_gpu.py [ref|hazard|ffma|router|ties|sort|degen|sync|bench|all]
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 7)))  # repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")

import torch  # noqa: E402

if not torch.cuda.is_available():  # promoted nightly battery: needs one CUDA device
    print("SKIP: no CUDA device")
    raise SystemExit(0)
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.language.extra import libdevice  # noqa: E402

from skyrl.backends.skyrl_train.isoexec.ops.moe.moe_router_o2_kernel import (  # noqa: E402
    _router_can_handle,
    fused_permute_index,
    fused_router_dense,
    permute_can_handle,
    ref_permute_index,
    ref_router_dense,
)

dev = "cuda"
E, K = 256, 8
_fail: list[str] = []
_n_checks = 0


def eq(name, a, b):
    """torch.equal, and report exact k/n on failure. NEVER allclose."""
    global _n_checks
    _n_checks += 1
    ok = a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    if ok:
        print(f"  ok   {name}   ({a.numel()}/{a.numel()} elements)")
    else:
        n = a.numel()
        d = int((a != b).sum().item()) if a.shape == b.shape else n
        print(
            f"  FAIL {name}   differ {d}/{n}  (dtype {a.dtype} vs {b.dtype}, shape {tuple(a.shape)} vs {tuple(b.shape)})"
        )
        _fail.append(name)
    return ok


def _install_batch_invariant():
    """Production installs these globally; measure and compare ON them, not on cuBLAS/ATen softmax."""
    from vllm.model_executor.layers import batch_invariant as bi

    lib = torch.library.Library("aten", "IMPL")
    lib.impl("aten::mm", bi.mm_batch_invariant, "CUDA")
    lib.impl("aten::addmm", bi.addmm_batch_invariant, "CUDA")
    lib.impl("aten::softmax", bi.softmax_batch_invariant, "CUDA")
    lib.impl("aten::_softmax", bi.softmax_batch_invariant, "CUDA")
    return lib


_LIB = _install_batch_invariant()


def gating_logits(T, e=E, seed=0):
    """Realistic fp32 router logits: a batch-invariant mm, exactly as the router produces them."""
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(T, 2048, device=dev, dtype=torch.float32, generator=g)
    w = torch.randn(2048, e, device=dev, dtype=torch.float32, generator=g) * 0.02
    return x @ w


# =================================================================================================
def test_ref():
    """Is the reference transcription faithful to what production actually runs?"""
    print("[REF] reference transcription vs the real production ops")
    T = 320
    lg = gating_logits(T)
    scores, _ = torch.topk(lg, K, dim=1, sorted=True)
    # 1. torch.softmax under the vllm override IS the five-op decomposition ref_router_dense spells out
    m = torch.amax(scores, dim=-1, keepdim=True)
    ex = torch.exp(scores - m)
    manual = ex / torch.sum(ex, dim=-1, keepdim=True)
    eq(
        "torch.softmax == softmax_batch_invariant decomposition",
        torch.softmax(scores, dim=-1, dtype=torch.float32),
        manual,
    )

    # 2. ref_router_dense == megatron's topk_routing_with_score_function, deterministic branch
    try:
        from megatron.core.transformer.moe.moe_utils import (
            topk_routing_with_score_function,
        )

        prev = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            mp, mm_ = topk_routing_with_score_function(lg, K, use_pre_softmax=False, score_function="softmax")
        finally:
            torch.use_deterministic_algorithms(prev, warn_only=True)
        rp, rm = ref_router_dense(lg, K)
        eq("ref_router_dense.probs == megatron routing_probs", rp, mp)
        eq("ref_router_dense.map   == megatron routing_map", rm, mm_)
    except ImportError:
        print("  skip megatron cross-check (megatron not importable in this env)")


# =================================================================================================
def test_hazard():
    """The three Triton-vs-ATen hazards this kernel touches, MEASURED. Prints diff counts."""
    print("[HAZARD] exp / division / fp32 reduction order -- measured element diffs vs ATen")
    T = 8192
    lg = gating_logits(T, seed=3)
    scores, _ = torch.topk(lg, K, dim=1, sorted=True)
    z = (scores - torch.amax(scores, dim=-1, keepdim=True)).contiguous()
    aten_exp = torch.exp(z)
    flat = z.reshape(-1).contiguous()
    N = flat.numel()

    @triton.jit
    def _expk(X, O, N, BLOCK: tl.constexpr, LIB: tl.constexpr):
        p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = p < N
        x = tl.load(X + p, mask=m)
        tl.store(O + p, libdevice.exp(x) if LIB else tl.exp(x), mask=m)

    for lib_, nm in ((True, "libdevice.exp"), (False, "tl.exp        ")):
        o = torch.empty_like(flat)
        _expk[(triton.cdiv(N, 1024),)](flat, o, N, BLOCK=1024, LIB=lib_, enable_fp_fusion=False)
        d = int((o != aten_exp.reshape(-1)).sum().item())
        print(f"  {nm} vs ATen exp: {d}/{N} differ" + ("   <- the one we use" if lib_ else ""))
        if lib_ and d:
            _fail.append("libdevice.exp != ATen")
        if not lib_ and d == 0:
            _fail.append("tl.exp control vacuous (expected a large diff)")

    aten_sum = torch.sum(aten_exp, dim=-1, keepdim=True)
    aten_div = aten_exp / aten_sum

    @triton.jit
    def _divk(A, B, O, N, KK: tl.constexpr, BLOCK: tl.constexpr, RN: tl.constexpr):
        p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = p < N
        a = tl.load(A + p, mask=m)
        b = tl.load(B + p // KK, mask=m)
        tl.store(O + p, libdevice.div_rn(a, b) if RN else a / b, mask=m)

    ae = aten_exp.reshape(-1).contiguous()
    for rn, nm in ((True, "libdevice.div_rn"), (False, "triton `/`      ")):
        o = torch.empty_like(ae)
        _divk[(triton.cdiv(N, 1024),)](
            ae, aten_sum.reshape(-1).contiguous(), o, N, KK=K, BLOCK=1024, RN=rn, enable_fp_fusion=False
        )
        d = int((o != aten_div.reshape(-1)).sum().item())
        print(f"  {nm} vs ATen div: {d}/{N} differ" + ("   <- correct on NORMALS only, see below" if rn else ""))
        if rn and d:
            _fail.append("libdevice.div_rn != ATen on normals")
        if not rn and d == 0:
            _fail.append("triton `/` control vacuous (expected a large diff)")

    # HAZARD 2b -- FTZ. Both forms above are wrong, in opposite directions, and neither shows it on
    # well-conditioned data. This is the check the first version of this kernel did not have.
    from skyrl.backends.skyrl_train.isoexec.ops.moe.moe_router_o2_kernel import (
        _div_rn_nonftz,  # noqa: F401
    )

    SMALL = 1.17549435e-38
    zz = torch.linspace(-87.0, -104.0, 1 << 16, device=dev)
    sub = torch.exp(zz)
    subd = (torch.rand_like(sub) * 0.5 + 1.0).contiguous()
    sref = sub / subd
    n_sub = int(((sref > 0) & (sref < SMALL)).sum().item())

    @triton.jit
    def _divk2(A, B, O, N, BLOCK: tl.constexpr, MODE: tl.constexpr):
        p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = p < N
        a = tl.load(A + p, mask=m)
        b = tl.load(B + p, mask=m)
        if MODE == 0:
            o = libdevice.div_rn(a, b)
        elif MODE == 1:
            o = a / b
        else:
            o = tl.inline_asm_elementwise(
                "div.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1
            )
        tl.store(O + p, o, mask=m)

    print(f"  -- subnormal quotients ({n_sub}/{sub.numel()} of these outputs are subnormal) --")
    for mode, nm in ((0, "libdevice.div_rn"), (1, "triton `/`      "), (2, "asm div.rn.f32  ")):
        o = torch.empty_like(sub)
        _divk2[(triton.cdiv(sub.numel(), 256),)](
            sub.contiguous(), subd, o, sub.numel(), BLOCK=256, MODE=mode, enable_fp_fusion=False
        )
        d = int((o != sref).sum().item())
        print(f"  {nm} vs ATen div: {d}/{sub.numel()} differ" + ("   <- the one we use" if mode == 2 else ""))
        if mode == 2 and d:
            _fail.append("asm div.rn.f32 != ATen on subnormals")
        if mode == 0 and d == 0:
            _fail.append("FTZ control vacuous -- libdevice.div_rn did NOT flush, so this check proves nothing")

    # reduction order: which fp32 addition tree is torch.sum over a length-K contiguous last dim?
    ref = aten_sum.squeeze(-1)
    e_ = aten_exp
    cands = {}
    a = e_[:, 0].clone()
    for j in range(1, K):
        a = a + e_[:, j]
    cands["left-to-right     "] = a
    b = e_.clone()
    n = K
    while n > 1:
        b = b[:, 0:n:2] + b[:, 1:n:2]
        n //= 2
    cands["adjacent-pair tree"] = b[:, 0]
    a0, a1, a2, a3 = e_[:, 0] + e_[:, 4], e_[:, 1] + e_[:, 5], e_[:, 2] + e_[:, 6], e_[:, 3] + e_[:, 7]
    cands["4-wide strided    "] = (a0 + a1) + (a2 + a3)
    b = e_.clone()
    n = K
    while n > 1:
        b = b[:, : n // 2] + b[:, n // 2 :]
        n //= 2
    cands["half-fold butterfly"] = b[:, 0]
    for nm, v in cands.items():
        d = int((v != ref).sum().item())
        print(
            f"  {nm} vs torch.sum: {d}/{T} differ"
            + ("   <- the one tl.sum emits, and the one we rely on" if "butterfly" in nm else "")
        )
        if "butterfly" in nm and d:
            _fail.append("half-fold butterfly != torch.sum -- the reduction-order assumption is broken")
    if all(int((v != ref).sum().item()) == 0 for nm, v in cands.items()):
        _fail.append("reduction-order control vacuous (every order agreed)")


# =================================================================================================
def test_ffma():
    """Hazard 4 with LIVE POSITIVE CONTROLS, so a pass cannot be an artefact of immunity.

    The O2 kernel has no `acc += b*c` -- the softmax is exp/sum/div and the scatter is a copy -- so
    it is structurally immune, exactly like O3's conv (0/327,680). A check that passed for THAT
    reason would also pass with `enable_fp_fusion=False` deleted. So: prove the flag reaches the
    compiler and that this harness can SEE contraction, then assert the kernel is invariant to it.
    """
    print("[FFMA] contraction hazard -- positive controls first, then the kernel")
    N = 81920
    g = torch.Generator(device=dev).manual_seed(11)
    a = torch.randn(N, device=dev, dtype=torch.float32, generator=g)
    b = torch.randn(N, device=dev, dtype=torch.float32, generator=g)
    c = torch.randn(N, device=dev, dtype=torch.float32, generator=g)

    # CONTROL (a): eager (separate mul then add) vs torch.addcmul (one fused op) on fp32.
    eager = a + b * c
    fused = torch.addcmul(a, b, c)
    d = int((eager != fused).sum().item())
    print(f"  control (a) eager `a + b*c` vs torch.addcmul: {d}/{N} differ")
    if d == 0:
        _fail.append("FFMA control (a) VACUOUS -- eager and addcmul agreed, the check proves nothing")

    # CONTROL (b): the same expression through Triton, with fp fusion on vs off.
    @triton.jit
    def _fma_probe(A, B, C, O, N, BLOCK: tl.constexpr):
        p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = p < N
        acc = tl.load(A + p, mask=m)
        acc += tl.load(B + p, mask=m) * tl.load(C + p, mask=m)
        tl.store(O + p, acc, mask=m)

    outs = {}
    for ff in (True, False):
        o = torch.empty_like(a)
        _fma_probe[(triton.cdiv(N, 1024),)](a, b, c, o, N, BLOCK=1024, enable_fp_fusion=ff)
        outs[ff] = o
    d = int((outs[True] != outs[False]).sum().item())
    print(f"  control (b) triton acc+=b*c, fp_fusion ON vs OFF: {d}/{N} differ")
    if d == 0:
        _fail.append("FFMA control (b) VACUOUS -- enable_fp_fusion did not reach the compiler")
    eq("control (b) fp_fusion=OFF matches EAGER (not addcmul)", outs[False], eager)
    print(
        f"  control (b) fp_fusion=ON  vs torch.addcmul: {int((outs[True] != fused).sum().item())}/{N} differ (expect 0 -- contraction == addcmul)"
    )

    # THE KERNEL: bit-identical under both settings, and equal to the EAGER reference.
    from skyrl.backends.skyrl_train.isoexec.ops.moe import moe_router_o2_kernel as m2

    lg = gating_logits(4096, seed=5)
    p_off, map_off = fused_router_dense(lg, K)
    orig = m2._router_kernel

    class _FF:
        """Re-launch the same jit'd kernel with enable_fp_fusion=True."""

        def __getitem__(self, grid):
            def run(*args, **kw):
                kw["enable_fp_fusion"] = True
                return orig[grid](*args, **kw)

            return run

    m2._router_kernel = _FF()
    try:
        p_on, map_on = fused_router_dense(lg, K)
    finally:
        m2._router_kernel = orig
    eq("O2 router: fp_fusion ON == fp_fusion OFF (structurally immune)", p_on, p_off)
    rp, rm = ref_router_dense(lg, K)
    eq("O2 router matches the EAGER sequence (NOT addcmul)", p_off, rp)
    eq("O2 router map matches eager", map_off, rm)
    print("  -> verdict: this kernel has no fp32 accumulator fed by a product; it MATCHES EAGER.")


# =================================================================================================
def _router_case(name, lg, k=K, scale=None):
    # Say WHICH path ran. `fused_router_dense` falls back to the eager sequence outside its
    # envelope, and a fallback compares the reference against itself -- a vacuous pass.
    tag = "" if _router_can_handle(lg, k) else "  [FALLBACK -- eager, not the kernel]"
    rp, rm = ref_router_dense(lg, k, scaling_factor=scale)
    fp, fm = fused_router_dense(lg, k, scaling_factor=scale)
    eq(f"{name} probs{tag}", fp, rp)
    eq(f"{name} map{tag}", fm, rm)


def test_router():
    print("[ROUTER] dense routing_probs / routing_map, torch.equal vs eager")
    # ragged decode: 320 sequences x 1 token -- the production decode shape
    _router_case("decode T=320 E=256 k=8", gating_logits(320, seed=1))
    _router_case("decode T=1", gating_logits(1, seed=2))
    _router_case("decode T=7 (partial warp)", gating_logits(7, seed=8))
    # prefill
    for T in (512, 2048, 8192):
        _router_case(f"prefill T={T}", gating_logits(T, seed=T))
    # E non-power-of-2, and E=128 (the ledger's bench shape)
    for e in (128, 192, 255, 100, 8):
        _router_case(f"E={e:<4d} T=320", gating_logits(320, e=e, seed=e), k=min(K, e))
    # k sweep (power-of-two only -- the envelope refuses the rest, see _router_can_handle)
    for k in (1, 2, 4, 8, 16):
        _router_case(f"k={k:<3d} T=320 E=256", gating_logits(320, seed=100 + k), k=k)
    # scaling factor
    _router_case("scaling_factor=2.5", gating_logits(320, seed=9), scale=2.5)


# =================================================================================================
def _tie_rows(T=64, e=E):
    """Adversarial EXACT-tie patterns. Tie-break order decides WHICH EXPERTS RUN."""
    base = gating_logits(T, e=e, seed=42)
    cases = {}
    x = base.clone()
    x[0] = 1.0  # all-equal row
    cases["all-equal row"] = x
    x = base.clone()
    x[1, :] = -5.0
    x[1, 0:12] = 3.0  # 12 tied for k=8 -> straddles the top-k boundary
    cases["12 tied straddling the k=8 boundary"] = x
    x = base.clone()
    x[2, :] = -5.0
    x[2, 0:e:2] = 2.0  # 128 tied even indices
    cases["128 tied even indices"] = x
    x = base.clone()
    x[3, :] = -5.0
    x[3, [0, 1, 2, e - 3, e - 2, e - 1]] = 7.0  # tied at both ends of the index range
    x[3, 50] = 7.0
    x[3, 200] = 7.0
    x[3, 201] = 7.0
    cases["ties at both ends of the index range"] = x
    x = base.clone()
    x[4, :] = 0.0
    x[4, e - 9 :] = 4.0  # 9 tied at the top of the range
    cases["9 tied at the top of the range"] = x
    return cases


def test_ties():
    print("[TIES] EXACT ties -- membership is not a rounding detail")
    for name, lg in _tie_rows().items():
        _router_case(f"tie: {name}", lg)
        # and through the sort, where a different member set would relocate whole rows
        rp, rm = ref_router_dense(lg, K)
        fs = fused_permute_index(rm, rp, K)
        rs = ref_permute_index(rm, rp, K)
        eq(f"tie: {name} -> sorted_indices", fs[0], rs[0])
        eq(f"tie: {name} -> permuted_probs", fs[1], rs[1])


# =================================================================================================
def _sort_case(name, rmap, rprobs, k):
    fs, fp_, fc = fused_permute_index(rmap, rprobs, k)
    rs, rp_, rc = ref_permute_index(rmap, rprobs, k)
    eq(f"{name} sorted_indices", fs, rs)
    eq(f"{name} permuted_probs", fp_, rp_)
    eq(f"{name} tokens_per_expert", fc, rc)


def test_sort():
    print("[SORT] counting sort vs argsort(descending=True, stable=True) + remainder + gather")
    for T in (320, 1, 7, 512, 2048, 8192):
        lg = gating_logits(T, seed=T + 1)
        rp, rm = ref_router_dense(lg, K)
        _sort_case(f"T={T:<5d} E=256 k=8", rm, rp, K)
    for e in (128, 192, 255, 8):
        lg = gating_logits(320, e=e, seed=e)
        k = min(K, e)
        rp, rm = ref_router_dense(lg, k, scaling_factor=None)
        _sort_case(f"E={e:<4d} k={k}", rm, rp, k)
    # E=1: every token to the single expert
    lg = gating_logits(320, e=1, seed=77)
    rp, rm = ref_router_dense(lg, 1)
    _sort_case("E=1 (all tokens -> one expert)", rm, rp, 1)

    # EMPTY EXPERT RUNS: leading / trailing / interior. The counting sort's base offsets and the
    # eager argsort must agree about zero-length segments.
    T = 320
    rp0, rm0 = ref_router_dense(gating_logits(T, seed=5), K)
    for nm, sl in (
        ("leading experts empty", slice(0, 64)),
        ("trailing experts empty", slice(E - 64, E)),
        ("interior experts empty", slice(100, 164)),
    ):
        rm = rm0.clone()
        rp = rp0.clone()
        rm[:, sl] = False
        rp[:, sl] = 0.0
        # re-route the freed slots to expert 0 so every row still has exactly k
        need = K - rm.sum(dim=1)
        for t in torch.nonzero(need).flatten().tolist():
            free = torch.nonzero(~rm[t]).flatten()
            free = free[(free < sl.start) | (free >= sl.stop)][: int(need[t].item())]
            rm[t, free] = True
            rp[t, free] = 0.125
        _sort_case(nm, rm, rp, K)

    # EP>1-STYLE SLICE: fewer than k local experts per token. The kernel must NOT be used here
    # (dispatch_o2_active refuses ep_size>1) because num_out_tokens=T*k then exceeds the routed-row
    # count and the eager argsort pads the tail from the FALSE region. Assert the refusal, and
    # assert that the SEGMENT the kernel does fill still matches -- so if the guard is ever
    # loosened, the reason it exists is visible here rather than as a garbage read in production.
    rp_s, rm_s = ref_router_dense(gating_logits(T, seed=6), K)
    rm_sl, rp_sl = rm_s[:, :64].contiguous(), rp_s[:, :64].contiguous()
    n_routed = int(rm_sl.sum().item())
    fs, fp_, fc = fused_permute_index(rm_sl, rp_sl, K)
    rs, rp2, rc = ref_permute_index(rm_sl, rp_sl, K)
    eq("EP-slice: tokens_per_expert (full)", fc, rc)
    eq("EP-slice: sorted_indices over the ROUTED prefix only", fs[:n_routed], rs[:n_routed])
    print(f"  note: {n_routed}/{T * K} slots are routed here -- the {T * K - n_routed}-slot tail is why")
    print("        dispatch_o2_active requires ep_size==1; the fused path is never taken at EP>1.")

    # ALL TOKENS TO ONE EXPERT (a single maximal segment)
    rm = torch.zeros(T, E, dtype=torch.bool, device=dev)
    rm[:, :K] = True
    rp = torch.zeros(T, E, dtype=torch.float32, device=dev)
    rp[:, :K] = 0.125
    _sort_case("all tokens -> experts 0..7", rm, rp, K)

    # T*k straddling the sort tile (BLOCK_T = 512)
    for T in (511, 512, 513, 1023, 1025):
        lg = gating_logits(T, seed=T)
        rp, rm = ref_router_dense(lg, K)
        _sort_case(f"T={T} (straddles BLOCK_T=512)", rm, rp, K)


# =================================================================================================
def test_dist():
    """UNREPRESENTATIVE INPUTS ARE HOW O3 PASSED OFFLINE AND FAILED LIVE BY 2e-02.

    O6's gap analysis: `randn` rows are well-conditioned, while real residual streams carry
    massive-activation outliers orders of magnitude above the row, and an outlier DOMINATES the
    fp32 reduction these kernels must reproduce bit for bit. So gate on heavy-tailed and
    outlier-injected logits, not just on a well-behaved matmul.

    O2 is *structurally* far less exposed than O3 was -- the only fp32 reduction is a sum of k=8
    values in (0, 1] whose largest term is exactly 1.0 by construction (the max is subtracted
    first), so the summands cannot span orders of magnitude however wild the logits are. That is an
    argument; these are the measurements.
    """
    print("[DIST] adversarial input distributions -- the O3 lesson, applied")
    g = torch.Generator(device=dev).manual_seed(1234)
    T = 1024
    cases = {}
    base = gating_logits(T, seed=1234)
    cases["baseline mm logits"] = base
    x = base.clone()
    x[torch.arange(T), torch.randint(0, E, (T,), device=dev, generator=g)] *= 1e4
    cases["massive-activation outlier (x1e4) per row"] = x
    x = base.clone()
    x[:, 0] = 1e6
    cases["one expert dominating at 1e6"] = x
    cases["heavy-tailed (Cauchy)"] = (
        torch.randn(T, E, device=dev, generator=g) / torch.randn(T, E, device=dev, generator=g).clamp(min=1e-6)
    ).float()
    cases["lognormal x100"] = (torch.randn(T, E, device=dev, generator=g) * 100).exp().float()
    cases["huge magnitude (x1e30)"] = base * 1e30
    cases["denormal scale (x1e-40)"] = base * 1e-40
    cases["all zeros"] = torch.zeros(T, E, device=dev)
    cases["constant 1e30 (softmax underflow)"] = torch.full((T, E), 1e30, device=dev)
    cases["alternating +-1e20"] = (
        torch.where(torch.arange(E, device=dev) % 2 == 0, 1e20, -1e20).float().expand(T, E).contiguous()
    )
    cases["integers (many exact ties)"] = torch.randint(-3, 4, (T, E), device=dev, generator=g).float()
    cases["bf16-quantised logits (dense ties)"] = base.bfloat16().float()
    for name, lg in cases.items():
        rp, rm = ref_router_dense(lg, K)
        fp_, fm = fused_router_dense(lg, K)
        # NaN == NaN is False for torch.equal, so compare bit patterns where NaNs are expected.
        if torch.isnan(rp).any():
            eq(f"dist: {name} probs (bit pattern)", fp_.view(torch.int32), rp.view(torch.int32))
        else:
            eq(f"dist: {name} probs", fp_, rp)
        eq(f"dist: {name} map", fm, rm)
        fs = fused_permute_index(fm, fp_, K)
        rs = ref_permute_index(rm, rp, K)
        eq(f"dist: {name} -> sorted_indices", fs[0], rs[0])


# =================================================================================================
def test_ftz():
    """The FTZ check, through the SHARED helper -- and it must not be vacuous.

    ``ftz_check.assert_no_ftz`` separates a FLUSHED subnormal from ordinary disagreement, and fails
    when the reference contains no subnormal at all -- because then the input cannot detect FTZ and
    a pass proves nothing. That is the same vacuity rule the FFMA controls follow.
    """
    print("[FTZ] flushed-subnormal check on the real kernel, via the shared helper")
    from _ftz_check import assert_no_ftz, ftz_inputs_for

    # The massive-activation router row: this is the input that caught the bug.
    lg = ftz_inputs_for("softmax_row", n=256 * 1024).float()
    rp, rm = ref_router_dense(lg, K)
    fp_, fm = fused_router_dense(lg, K)
    st = assert_no_ftz("O2 router probs (massive-activation rows)", rp, fp_, fail_list=_fail)
    print(f"       ({st['subnormal']} of {rp.numel()} reference probs are subnormal -- the check has teeth)")

    # NEGATIVE CONTROL: the same kernel body with libdevice.div_rn must FLUSH, or the helper is
    # not actually detecting anything and this whole section is decoration.
    @triton.jit
    def _ftz_div(A, B, O, N, BLOCK: tl.constexpr):
        p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = p < N
        tl.store(O + p, libdevice.div_rn(tl.load(A + p, mask=m), tl.load(B + p, mask=m)), mask=m)

    a = torch.exp(torch.linspace(-104.0, -87.0, 1 << 16, device=dev)).contiguous()
    b = (torch.rand_like(a) * 0.5 + 1.0).contiguous()
    o = torch.empty_like(a)
    _ftz_div[(triton.cdiv(a.numel(), 256),)](a, b, o, a.numel(), BLOCK=256, enable_fp_fusion=False)
    assert_no_ftz("control: libdevice.div_rn must flush", a / b, o, fail_list=_fail, expect_flush=True)


# =================================================================================================
def test_steps():
    """A LONG, RAGGED, CHANGING SEQUENCE OF CALLS -- the second hole O3 exposed.

    O3's failure was partly that no test ran its recurrence for more than one step. O2 carries NO
    state: three kernels, all outputs freshly allocated, no module-level buffer, no cache, no slot
    map, nothing that persists between calls. This asserts that rather than claiming it -- 200
    consecutive calls at *changing ragged* shapes, each still bitwise vs eager, and the first shape
    re-checked at the end to prove nothing drifted.
    """
    print("[STEPS] 200 consecutive calls at changing ragged shapes -- statelessness")
    g = torch.Generator(device=dev).manual_seed(7)
    first = gating_logits(320, seed=4242)
    rp0, rm0 = ref_router_dense(first, K)
    bad = 0
    for i in range(200):
        T = int(torch.randint(1, 640, (1,), device=dev, generator=g).item())
        lg = torch.randn(T, E, device=dev, generator=g) * float(
            torch.rand(1, device=dev, generator=g).item() * 10 + 0.1
        )
        rp, rm = ref_router_dense(lg, K)
        fp_, fm = fused_router_dense(lg, K)
        rs, rpp, rc = ref_permute_index(rm, rp, K)
        fs, fpp, fc = fused_permute_index(fm, fp_, K)
        if not (
            torch.equal(fp_, rp)
            and torch.equal(fm, rm)
            and torch.equal(fs, rs)
            and torch.equal(fpp, rpp)
            and torch.equal(fc, rc)
        ):
            bad += 1
            if bad == 1:
                print(f"  FAIL first divergence at step {i}, T={T}")
    if bad:
        _fail.append(f"{bad}/200 steps diverged")
    else:
        print("  ok   200/200 ragged steps bitwise (router + sort + counts)")
    eq("step 0's shape still bitwise after 200 other shapes (no carried state)", fused_router_dense(first, K)[0], rp0)
    eq("...and its map", fused_router_dense(first, K)[1], rm0)


# =================================================================================================
def test_invar():
    """BATCH INVARIANCE -- and the reason no 8-rank test is needed for O2.

    THE TP ARGUMENT, stated so it can be checked rather than assumed. The production engine runs
    EP=1/ETP=TP, so (a) every rank sees all E=256 experts -- the router's expert axis is NOT
    sharded -- and (b) under the no-gather dispatch every rank already holds the full batch, so T is
    the same on every rank. The router's `logits` are the output of a TP-replicated gating GEMM.
    So all 8 ranks feed this kernel IDENTICAL bits and it has no rank-dependent shape, stride or
    layout: there is nothing for TP to vary. (Contrast O3, whose gate ran 4/8 k/v heads while
    production runs 2/4 -- a head-count-dependent kernel wearing TP8 clothing.)

    What that argument REDUCES TO is batch invariance: a token's dense row must be a pure function
    of its own logits, independent of T, of its position, and of who else is in the batch. That is
    directly testable in one process, and it is what is tested here.
    """
    print("[INVAR] a token's dense row is a pure function of its own logits")
    g = torch.Generator(device=dev).manual_seed(31)
    probe = torch.randn(1, E, device=dev, generator=g) * 3.0
    p_alone, m_alone = fused_router_dense(probe, K)
    for T in (1, 2, 7, 64, 320, 1000, 8192):
        for pos in (0, T // 2, T - 1):
            batch = torch.randn(T, E, device=dev, generator=g) * 7.0
            batch[pos] = probe[0]
            p, m = fused_router_dense(batch, K)
            ok = torch.equal(p[pos : pos + 1], p_alone) and torch.equal(m[pos : pos + 1], m_alone)
            if not ok:
                _fail.append(f"batch variance at T={T} pos={pos}")
                print(f"  FAIL probe row differs at T={T} pos={pos}")
    print("  ok   probe row identical across T in {1,2,7,64,320,1000,8192} x 3 positions (21 shapes)")
    print("  -> the kernel cannot depend on T, so it cannot depend on how TP splits a batch.")
    print("     Combined with EP=1 (expert axis unsharded) and replicated gating logits, TP is")
    print("     unable to reach this kernel. No 8-rank test is required; this is the argument.")


# =================================================================================================
def test_degen():
    print("[DEGEN] non-finite rows and the host-side envelope")
    lg = gating_logits(64, seed=13)
    lg[0] = float("nan")
    lg[1, 5] = float("nan")
    lg[2] = float("inf")
    lg[3] = float("-inf")
    fp_, fm = fused_router_dense(lg, K)
    # THE INVARIANT that the engine-init crash was: the scattered map must have exactly k experts
    # per row and never write outside [0, E). A sentinel index would corrupt memory, not just bits.
    cnt = fm.sum(dim=1)
    ok = bool((cnt == K).all().item())
    print(
        f"  {'ok  ' if ok else 'FAIL'} every row selects exactly k={K} in-range experts (min={int(cnt.min())} max={int(cnt.max())})"
    )
    if not ok:
        _fail.append("degenerate rows produced an out-of-range / duplicate expert index")
    # NaN rows produce NaN probs under either ranking, so nothing numerical is at stake; finite rows
    # must still be bitwise.
    fin = torch.isfinite(lg).all(dim=1)
    rp, rm = ref_router_dense(lg, K)
    eq("finite rows still bitwise alongside NaN/inf rows", fp_[fin], rp[fin])
    eq("finite rows map still bitwise", fm[fin], rm[fin])

    # envelope: these must be REFUSED (fall back), not asserted -- an assert here kills a Ray actor
    checks = [
        ("k=3 (non power of two -> reduction order re-associates)", torch.zeros(4, 16, device=dev), 3, False),
        ("bf16 logits", torch.zeros(4, 16, device=dev, dtype=torch.bfloat16), 8, False),
        ("cpu logits", torch.zeros(4, 16), 8, False),
        ("k > E", torch.zeros(4, 4, device=dev), 8, False),
        ("k=64 > _MAX_K", torch.zeros(4, 128, device=dev), 64, False),
        ("E=256 k=8 fp32 cuda", torch.zeros(4, 256, device=dev), 8, True),
    ]
    for nm, t, k, want in checks:
        got = _router_can_handle(t, k)
        print(f"  {'ok  ' if got == want else 'FAIL'} envelope {nm}: can_handle={got} (want {want})")
        if got != want:
            _fail.append(f"envelope {nm}")
    print(
        f"  permute_can_handle(bool [320,256], 8) = {permute_can_handle(torch.zeros(320, 256, dtype=torch.bool, device=dev), 8)}"
    )


# =================================================================================================
def test_sync():
    print("[SYNC] determinism, sync-freedom, CUDA-graph capture + replay")
    lg = gating_logits(320, seed=21)
    a = fused_router_dense(lg, K)
    for i in range(20):
        b = fused_router_dense(lg, K)
        if not (torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])):
            _fail.append(f"router nondeterministic at iteration {i}")
            break
    else:
        print("  ok   router bitwise-identical over 20 repeats")
    s = fused_permute_index(a[1], a[0], K)
    for i in range(20):
        t = fused_permute_index(a[1], a[0], K)
        if not all(torch.equal(x, y) for x, y in zip(s, t)):
            _fail.append(f"sort nondeterministic at iteration {i}")
            break
    else:
        print("  ok   sort bitwise-identical over 20 repeats")

    torch.cuda.set_sync_debug_mode("error")
    try:
        p, m = fused_router_dense(lg, K)
        fused_permute_index(m, p, K)
        print("  ok   no host sync under set_sync_debug_mode('error')")
    except RuntimeError as e:
        _fail.append(f"host sync in the fused path: {e}")
        print(f"  FAIL host sync: {e}")
    finally:
        torch.cuda.set_sync_debug_mode("default")

    # graph capture: production decode is graph-replayed, so this is the mode that matters
    static = lg.clone()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        for _ in range(3):
            pp, mm = fused_router_dense(static, K)
            fused_permute_index(mm, pp, K)
    torch.cuda.current_stream().wait_stream(warm)
    with torch.cuda.graph(g):
        gp, gm = fused_router_dense(static, K)
        gs = fused_permute_index(gm, gp, K)
    static.copy_(lg)
    g.replay()
    torch.cuda.synchronize()
    rp, rm = ref_router_dense(lg, K)
    eq("graph replay: routing_probs", gp, rp)
    eq("graph replay: routing_map", gm, rm)
    rs = ref_permute_index(rm, rp, K)
    eq("graph replay: sorted_indices", gs[0], rs[0])
    eq("graph replay: permuted_probs", gs[1], rs[1])
    # a second replay with different data must track it (nothing baked in at capture)
    lg2 = gating_logits(320, seed=99)
    static.copy_(lg2)
    g.replay()
    torch.cuda.synchronize()
    rp2, rm2 = ref_router_dense(lg2, K)
    eq("graph replay #2 on new data: routing_probs", gp, rp2)


# =================================================================================================
def _bench(fn, iters=50):
    """Median wall time of `fn` inside a REPLAYED CUDA graph (production decode is graph-replayed;
    an eager-loop number is a fantasy -- O3's eager arm read 15.75 ms for a 0.278 ms kernel)."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]


def test_count():
    """CUDA kernel launches per MoE layer, eager vs fused -- the quantity the ledger prices.

    The ledger measures cluster P8 at 23.8 launches/layer (952.0/step over 40 layers) and prices
    O2 at 792 launches removed. This counts the same thing offline, so the live A/B has a
    prediction to land against rather than a hope.
    """
    print("[COUNT] CUDA kernel launches per MoE layer (torch profiler), eager vs fused")
    from torch.profiler import ProfilerActivity, profile

    lg = gating_logits(320, seed=1)

    def eager():
        rp, rm = ref_router_dense(lg, K)
        ref_permute_index(rm, rp, K)

    def fused():
        fp_, fm = fused_router_dense(lg, K)
        fused_permute_index(fm, fp_, K)

    import collections

    def kcount(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        c = collections.Counter()
        for evt in prof.key_averages():
            if evt.device_time_total > 0 and evt.self_device_time_total > 0:
                c[evt.key] += evt.count
        return c

    ce, cf = kcount(eager), kcount(fused)
    ne, nf = sum(ce.values()), sum(cf.values())
    print(f"  eager: {ne:>4} launches/layer -> {ne * 40:>5}/step (ledger P8: 23.8/layer, 952.0/step)")
    for k_, v in ce.most_common():
        print(f"      {v:>3}  {k_[:80]}")
    print(f"  fused: {nf:>4} launches/layer -> {nf * 40:>5}/step")
    for k_, v in cf.most_common():
        print(f"      {v:>3}  {k_[:80]}")
    print(f"  REMOVED: {(ne - nf) * 40} launches/step  (ledger O2 predicts 792)")


def test_bench():
    print("[BENCH] full P8 cluster, eager vs fused, inside a replayed CUDA graph (us per MoE layer)")
    print(f"  {'T':>7} {'eager us':>10} {'fused us':>10} {'speedup':>9}   {'x40 layers eager->fused (ms)':>30}")
    for T in (320, 512, 2048, 8192):
        lg = gating_logits(T, seed=T)

        def eager():
            rp, rm = ref_router_dense(lg, K)
            ref_permute_index(rm, rp, K)

        def fused():
            fp_, fm = fused_router_dense(lg, K)
            fused_permute_index(fm, fp_, K)

        a = _bench(eager)
        b = _bench(fused)
        print(f"  {T:>7} {a:>10.2f} {b:>10.2f} {a/b:>8.2f}x   {a*40/1000:>13.3f} -> {b*40/1000:.3f}")


# =================================================================================================
TESTS = {
    "ref": test_ref,
    "hazard": test_hazard,
    "ffma": test_ffma,
    "router": test_router,
    "ties": test_ties,
    "sort": test_sort,
    "degen": test_degen,
    "dist": test_dist,
    "ftz": test_ftz,
    "steps": test_steps,
    "invar": test_invar,
    "sync": test_sync,
    "count": test_count,
    "bench": test_bench,
}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(TESTS) if which == "all" else [which]
    if which == "all":
        names.remove("bench")
        names.remove("count")
    for n in names:
        print()
        TESTS[n]()
    print()
    if _fail:
        print(f"FAILED ({len(_fail)}): " + "; ".join(_fail))
        sys.exit(1)
    print(f"ALL PASS -- {_n_checks} torch.equal checks")
