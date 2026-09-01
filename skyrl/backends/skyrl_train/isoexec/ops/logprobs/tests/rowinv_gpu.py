"""Single-GPU gates for the row-invariant sampled logprob (``ops/logprobs/rowinv.py``).

What is checked:

  1. ROW-COUNT INVARIANCE (bitwise). The defect this module exists for: ATen's fp32 vocab sum
     changes bits with the launch geometry, so trainer chunks (1024 rows) and engine decode
     batches (1..N rows) disagree. The candidate must be ``torch.equal`` on the int32 words for
     N in {1,2,4,8,16,17,64,128,256,512} against the 1024-row reference, at the production
     V=248320. The incumbent ATen path is run on the same data as a POSITIVE CONTROL: it must
     actually differ across row counts here, or this gate is measuring nothing.
  2. ACCURACY vs an fp64 ground truth. Kahan compensation must make the candidate at least as
     accurate as ATen's fp32 schedule -- asserted on both mean and max |error|, not assumed.
  3. BACKWARD vs torch autograd (allclose, NOT bitwise: grad reaches only the optimizer). Also
     asserts an out-of-vocabulary target row produces logprob 0 and zero gradient.
  4. 3D vs 2D layout: [B, T, V] and the flattened [B*T, V] produce identical bits.
  5. DECLINES: V % G != 0 declines (no silent padding); SKYRL_ISOEXEC=0 declines with served == 0
     (rowinv has no flag of its own -- it is the composed default -- so the master switch is the
     only env state that can still turn it off).
  6. Perf at 1024 rows and 8 rows vs the ATen incumbent (reported, not asserted).
  7. DTYPE-ALTERNATION REGRESSION: alternating bf16 (scoring under
     SKYRL_ISOEXEC_SCORING_LOGITS_BF16) and fp32 (the Float16Module-upcast training forward) on
     the SAME process/group must both SERVE, never raise, and agree bitwise -- the payload dtype
     is per-call eligibility, not an immutable structural fact.
  8. Genuine structural drift (a changed vocabulary partition) must STILL raise, so the dtype fix
     cannot silently disable the safety gate.

Run: CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=. python skyrl/backends/skyrl_train/isoexec/ops/logprobs/tests/rowinv_gpu.py
Exit 0 iff every check passes.
"""

import os
import sys
import time

os.environ["SKYRL_ISOEXEC"] = "1"
os.environ.setdefault("SKYRL_ISOEXEC_PIK_LEAVES", "8")

import torch

if not torch.cuda.is_available():
    print("SKIP: needs a CUDA device")
    raise SystemExit(0)

from skyrl.backends.skyrl_train.isoexec.ops.logprobs import rowinv

DEV = "cuda"
V = 248320
G = int(os.environ["SKYRL_ISOEXEC_PIK_LEAVES"])
ROWS = 1024
NS = (1, 2, 4, 8, 16, 17, 64, 128, 256, 512)

_checks = 0
_fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    global _checks
    _checks += 1
    if ok:
        print(f"  ok   {name}" + (f" ({detail})" if detail else ""))
    else:
        _fails.append(f"{name}: {detail}")
        print(f"  FAIL {name}" + (f" ({detail})" if detail else ""))


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32)


def aten_incumbent(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """The incumbent formulation: amax / sub / exp / ATen fp32 vocab sum / log / gather.

    Invalid (out-of-vocabulary padding) targets are masked to 0 the way the incumbent does.
    """
    valid = (t >= 0) & (t < x.shape[-1])
    safe = t.masked_fill(~valid, 0)
    m = torch.amax(x, dim=-1, keepdim=True)
    s = torch.gather(x, -1, safe.unsqueeze(-1)).squeeze(-1)
    w = (x - m).exp_()
    value = (s - m.squeeze(-1)) - w.sum(-1).log()
    return value.masked_fill(~valid, 0.0)


def call(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor | None:
    return rowinv.rowinv_sampled_logprobs(
        x,
        t,
        vocab_start_index=0,
        vocab_end_index=x.shape[-1],
        group=None,
        src_dtype=torch.bfloat16,
        reference=lambda: aten_incumbent(x.detach().float(), t),
    )


gen = torch.Generator(device=DEV).manual_seed(20260825)
logits = (torch.randn(ROWS, V, device=DEV, dtype=torch.float32, generator=gen) * 3).to(torch.bfloat16).float()
tgen = torch.Generator(device=DEV).manual_seed(1729)
target = torch.randint(0, V, (ROWS,), device=DEV, generator=tgen)
# leaf/boundary adversarial targets
L = V // G
target[:8] = torch.tensor([0, V - 1, L - 1, L, V // 2 - 1, V // 2, 7 * L - 1, 7 * L], device=DEV)

# ================================================================================= 1) invariance
print(f"1) row-count invariance (bitwise), V={V}, G={G}, ref rows={ROWS}:")
ref = call(logits, target)
check("candidate engaged (served > 0)", ref is not None and rowinv.stats()["served"] > 0)
assert ref is not None

aten_ref = aten_incumbent(logits, target)
aten_bad = 0
for n in NS:
    aten_bad += int((bits(aten_incumbent(logits[:n], target[:n])) != bits(aten_ref[:n])).sum())
check(
    "POSITIVE CONTROL: ATen incumbent is NOT row-count invariant on this data",
    aten_bad > 0,
    f"{aten_bad} differing tokens across N={NS}",
)

bad = 0
for n in NS:
    out_n = call(logits[:n].contiguous(), target[:n].contiguous())
    if out_n is None:
        check(f"N={n} admitted", False, "declined: " + rowinv.stats()["decline_reason"])
        continue
    bad += int((bits(out_n) != bits(ref[:n])).sum())
check("candidate is bitwise row-count invariant", bad == 0, f"{bad} differing tokens across N={NS}")

cand_vs_aten = int((bits(ref) != bits(aten_ref)).sum())
print(
    f"  note candidate vs ATen at {ROWS} rows: {cand_vs_aten} differing bits, "
    f"max|d|={float((ref - aten_ref).abs().max()):.2e} (a different fp32 function by design)"
)

# ================================================================================== 2) accuracy
print("2) accuracy vs fp64 ground truth (Kahan must dominate ATen):")
truth = torch.empty(ROWS, device=DEV, dtype=torch.float64)
for s in range(0, ROWS, 64):
    e = min(s + 64, ROWS)
    truth[s:e] = logits[s:e].double().gather(-1, target[s:e].unsqueeze(-1)).squeeze(-1) - torch.logsumexp(
        logits[s:e].double(), dim=-1
    )
err_cand = (ref.double() - truth).abs()
err_aten = (aten_ref.double() - truth).abs()
check(
    "mean |err| candidate <= ATen",
    float(err_cand.mean()) <= float(err_aten.mean()),
    f"candidate={float(err_cand.mean()):.3e} aten={float(err_aten.mean()):.3e}",
)
check(
    "max |err| candidate <= ATen",
    float(err_cand.max()) <= float(err_aten.max()),
    f"candidate={float(err_cand.max()):.3e} aten={float(err_aten.max()):.3e}",
)

# ====================================================================================== 3) perf
print("3) perf curve (median of timed loop, incl. python/admission overhead of the wrapper):")


def _median_ms(fn, warm: int, iters: int) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return samples[len(samples) // 2]


crossover = None
for rows in (1, 8, 16, 32, 64, 128, 256, 512, 1024):
    x_r = logits[:rows].contiguous()
    t_r = target[:rows].contiguous()
    ms_rowinv = _median_ms(lambda: call(x_r, t_r), warm=5, iters=30)
    ms_aten = _median_ms(lambda: aten_incumbent(x_r, t_r), warm=5, iters=30)
    if crossover is None and ms_rowinv <= ms_aten:
        crossover = rows
    print(f"     rows={rows:5d} rowinv={ms_rowinv:7.3f} ms   aten={ms_aten:7.3f} ms   ratio={ms_rowinv / ms_aten:5.2f}")
print(f"     crossover (first N where rowinv <= aten): {crossover}")
print(f"     hot_hits={rowinv.stats()['hot_hits']} (steady-state admissions took the latched route)")

# ================================================================================== 4) backward
print("4) backward vs torch autograd reference (allclose; no bitwise contract):")
rowinv._reset_for_test()  # group=None cache pins shard width; the next phase changes V
V2 = 8192
x = (torch.randn(64, V2, device=DEV, dtype=torch.float32, generator=gen) * 3).to(torch.bfloat16).float()
x.requires_grad_()
t2 = torch.randint(0, V2, (64,), device=DEV, generator=tgen)
t2[0] = -100  # out-of-vocabulary padding row: logprob 0, grad 0
upstream = torch.randn(64, device=DEV, dtype=torch.float32, generator=gen)

out = call(x, t2)
assert out is not None, "backward phase declined: " + rowinv.stats()["decline_reason"]
(out * upstream).sum().backward()

x_ref = x.detach().clone().requires_grad_()
safe_t = t2.masked_fill(t2 < 0, 0)
lp_ref = torch.log_softmax(x_ref, dim=-1).gather(-1, safe_t.unsqueeze(-1)).squeeze(-1)
lp_ref = lp_ref.masked_fill(t2 < 0, 0.0)
(lp_ref * upstream).sum().backward()

gd = (x.grad - x_ref.grad).abs()
denom = x_ref.grad.abs().clamp_min(1e-30)
check(
    "grad allclose(rtol=1e-4, atol=1e-6)",
    torch.allclose(x.grad, x_ref.grad, rtol=1e-4, atol=1e-6),
    f"max|d|={float(gd.max()):.3e} max rel (|g|>1e-8)={float((gd / denom)[x_ref.grad.abs() > 1e-8].max()):.3e}",
)
check("padding row: logprob == 0", bool((out[0] == 0).item()))
check("padding row: grad == 0", bool((x.grad[0] == 0).all().item()))

# ================================================================================ 5) 3D layout
x3 = x.detach()[: 4 * 8].reshape(2, 16, V2).contiguous()
t3 = t2[: 4 * 8].reshape(2, 16).contiguous()
out3 = call(x3, t3)
out2 = call(x3.reshape(-1, V2), t3.reshape(-1))
check(
    "3D [B,T,V] == flattened [B*T,V] (bitwise)",
    out3 is not None and out2 is not None and torch.equal(bits(out3.reshape(-1)), bits(out2)),
)

# ============================================================================ 5) bf16 input path
print("5) low-precision input: transient in-module widen == pre-widened fp32, bit for bit:")
rowinv._reset_for_test()  # section 4 pinned shard width V2 in the group=None cache; V returns here
n5 = 256
x32 = logits[:n5].contiguous()  # fp32, itself an exactly-widened bf16
t5 = target[:n5].contiguous()
ref32 = call(x32, t5)
# Deliberately NO reset between the fp32 and bf16 calls: the payload dtype is per-call
# eligibility, not group structure, so both dtypes must serve on the SAME admission cache.
x16 = x32.to(torch.bfloat16)  # exact narrowing: x32 was built by widening bf16
out16 = call(x16, t5)
check(
    "bf16 input bitwise == pre-widened fp32 input",
    out16 is not None and ref32 is not None and torch.equal(bits(out16), bits(ref32)),
)

# bf16 backward: the ORIGINAL bf16 tensor is what is saved; grad comes back in bf16
rowinv._reset_for_test()
xg = x.detach().to(torch.bfloat16).requires_grad_()
outg = call(xg, t2)
assert outg is not None, "bf16 backward phase declined: " + rowinv.stats()["decline_reason"]
(outg * upstream).sum().backward()
check("bf16 grad dtype == bf16 (matches input)", xg.grad is not None and xg.grad.dtype is torch.bfloat16)
gd16 = (xg.grad.float() - x_ref.grad).abs()
check(
    "bf16 grad allclose(fp32 ref, rtol=2e-2, atol=2e-3)",
    torch.allclose(xg.grad.float(), x_ref.grad, rtol=2e-2, atol=2e-3),
    f"max|d|={float(gd16.max()):.3e} (bf16 rounding of the returned grad dominates)",
)

# ================================================================================== 6) declines
print("6) decline paths:")
rowinv._reset_for_test()
xb = torch.randn(4, 1004, device=DEV, dtype=torch.float32)  # 1004 % 8 == 4
tb = torch.randint(0, 1004, (4,), device=DEV)
res = rowinv.rowinv_sampled_logprobs(
    xb,
    tb,
    vocab_start_index=0,
    vocab_end_index=1004,
    group=None,
    src_dtype=torch.bfloat16,
    reference=lambda: aten_incumbent(xb, tb),
)
st = rowinv.stats()
check(
    "V % G != 0 declines (returns None, no padding)",
    res is None and "not divisible by G" in st["decline_reason"],
    st["decline_reason"],
)

rowinv._reset_for_test()
os.environ["SKYRL_ISOEXEC"] = "0"
res = rowinv.rowinv_sampled_logprobs(
    logits[:4],
    target[:4],
    vocab_start_index=0,
    vocab_end_index=V,
    group=None,
    src_dtype=torch.bfloat16,
    reference=lambda: aten_incumbent(logits[:4], target[:4]),
)
st = rowinv.stats()
check(
    "SKYRL_ISOEXEC=0 declines before any work",
    res is None and st["served"] == 0 and "SKYRL_ISOEXEC" in st["decline_reason"],
    st["decline_reason"],
)
os.environ["SKYRL_ISOEXEC"] = "1"

# ============================================================ 7) dtype alternation regression
print("7) alternating bf16/fp32 on ONE process/group: both serve, no raise, identical bits:")
# The same trainer process scores with bf16 logits (the scoring forward under
# SKYRL_ISOEXEC_SCORING_LOGITS_BF16) and trains with fp32 logits (Float16Module upcasts the
# training forward), so a dtype flip after admission must not read as STRUCTURAL DRIFT. The dtype
# is per-call eligibility: both calls MUST serve and MUST agree bitwise.
rowinv._reset_for_test()
n7 = 64
x32_7 = logits[:n7].contiguous()  # fp32 that is an exactly-widened bf16 (the training forward)
x16_7 = x32_7.to(torch.bfloat16)  # exact narrowing by construction (the scoring forward)
t7 = target[:n7].contiguous()


def call7(x7, src):
    return rowinv.rowinv_sampled_logprobs(
        x7,
        t7,
        vocab_start_index=0,
        vocab_end_index=V,
        group=None,
        src_dtype=src,
        reference=lambda: aten_incumbent(x7.detach().float(), t7),
    )


try:
    out_a = call7(x16_7, torch.bfloat16)  # scoring-shaped call admits the group ...
    out_b = call7(x32_7, torch.float32)  # ... then the fp32 training call arrives (the crash site)
    out_c = call7(x16_7, torch.bfloat16)  # ... and scoring again: alternation, not a one-way switch
    raised: RuntimeError | None = None
except RuntimeError as error:
    out_a = out_b = out_c = None
    raised = error
check("bf16 -> fp32 -> bf16 on one group does not raise", raised is None, repr(raised) if raised else "")
check(
    "both dtypes SERVE (the fp32 training call must not decline)",
    out_a is not None and out_b is not None and out_c is not None,
    rowinv.stats()["decline_reason"],
)
check(
    "bf16 and fp32 calls agree bitwise on the same underlying values",
    out_a is not None
    and out_b is not None
    and out_c is not None
    and torch.equal(bits(out_a), bits(out_b))
    and torch.equal(bits(out_a), bits(out_c)),
)

# =================================================== 8) genuine structural drift must STILL raise
print("8) genuine structural drift still raises (the dtype fix must not weaken the gate):")
# Same group key, changed vocabulary partition (shard width V/2 after admission at V): candidate
# and incumbent issue different collectives, so this rank-local divergence is unsafe and raises.
half = V // 2
xh = logits[:4, :half].contiguous()
th = torch.randint(0, half, (4,), device=DEV, generator=tgen)
drift_raised = False
try:
    rowinv.rowinv_sampled_logprobs(
        xh,
        th,
        vocab_start_index=0,
        vocab_end_index=half,
        group=None,
        src_dtype=torch.bfloat16,
        reference=lambda: aten_incumbent(xh, th),
    )
except RuntimeError as error:
    drift_raised = "STRUCTURAL DRIFT" in str(error)
check("changed vocab partition (shard width) still raises STRUCTURAL DRIFT", drift_raised)

# ============================================================= 9) full-row entry (engine V1 site)
print("9) full-row entry (rowinv_full_logprobs -- the engine's V1 Sampler.compute_logprobs hook):")
# The V1 sampler needs the FULL [N, V] fp32 row (gather_logprobs consumes it for top-k/ranks), so
# the engine serves rowinv through the full-row entry. The claims: (a) gathering the sampled
# column from the full row is bitwise identical to the sampled entry -- i.e. to what the TRAINER's
# rowinv path produces for the same logits; (b) the full row is row-count invariant bitwise;
# (c) bf16 input == pre-widened fp32 input bitwise; (d) SKYRL_ISOEXEC=0 declines before any work.
rowinv._reset_for_test()


def vllm_incumbent(x9: torch.Tensor) -> torch.Tensor:
    return x9.log_softmax(dim=-1, dtype=torch.float32)


def call_full(x9: torch.Tensor):
    return rowinv.rowinv_full_logprobs(x9, src_dtype=x9.dtype, reference=lambda: vllm_incumbent(x9))


n9 = 64
x9 = logits[:n9].contiguous()
t9 = target[:n9].contiguous()
full = call_full(x9)
check("full-row engaged (served > 0)", full is not None and rowinv.stats()["served"] > 0)
assert full is not None
check("full-row output is fp32 [N, V]", full.dtype is torch.float32 and full.shape == x9.shape)

sampled_path = call(x9, t9)
assert sampled_path is not None, "sampled entry declined in section 9: " + rowinv.stats()["decline_reason"]
gathered = torch.gather(full, 1, t9.unsqueeze(1)).squeeze(1)
check(
    "full-row gather at target == sampled entry (bitwise; trainer==engine at world=1)",
    torch.equal(bits(gathered), bits(sampled_path)),
    f"max|d|={float((gathered - sampled_path).abs().max()):.3e}",
)

bad9 = 0
for n in (1, 2, 8, 17, 33):
    fn = call_full(x9[:n].contiguous())
    if fn is None:
        check(f"full-row N={n} admitted", False, "declined: " + rowinv.stats()["decline_reason"])
        continue
    bad9 += int((bits(fn) != bits(full[:n])).sum())
check("full row is bitwise row-count invariant", bad9 == 0, f"{bad9} differing elements")

full16 = call_full(x9.to(torch.bfloat16))
check(
    "full-row bf16 input bitwise == pre-widened fp32 input",
    full16 is not None and torch.equal(bits(full16), bits(full)),
)

d9 = (full - vllm_incumbent(x9)).abs()
print(f"  note full row vs vLLM log_softmax incumbent: max|d|={float(d9.max()):.2e} (different fp32 tree by design)")

rowinv._reset_for_test()
os.environ["SKYRL_ISOEXEC"] = "0"
res9 = call_full(x9)
st9 = rowinv.stats()
check(
    "full-row SKYRL_ISOEXEC=0 declines before any work",
    res9 is None and st9["served"] == 0 and "SKYRL_ISOEXEC" in st9["decline_reason"],
    st9["decline_reason"],
)
os.environ["SKYRL_ISOEXEC"] = "1"

# =================================================================================================
print()
if _fails:
    print(f"FAILED {len(_fails)}/{_checks} checks:")
    for f in _fails:
        print("  - " + f)
    sys.exit(1)
print(f"ALL {_checks}/{_checks} CHECKS PASSED")
