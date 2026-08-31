"""SKYRL_ISOEXEC_MM_FWD_ONLY_BMM: the forward is untouched, the backward is on cuBLAS.

Every forward -- no_grad, grad-enabled, and the checkpoint re-forward on the autograd thread --
must keep vLLM's Triton ``bmm_kernel`` bit for bit; only a real backward falls through to cuBLAS.
Test 0 first proves the two kernels disagree at these shapes, so the rest is non-vacuous.
"""

import os

os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", os.environ.get("ISOEXEC_BMM_TEST_PORT", "12977"))
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
# The A/B runs in process: start from a clean OFF so an inherited env cannot skip the baseline.
os.environ["SKYRL_ISOEXEC_MM_FWD_ONLY_BMM"] = "0"

import torch  # noqa: E402

if not torch.cuda.is_available():  # promoted nightly battery: needs one CUDA device
    print("SKIP: no CUDA device")
    raise SystemExit(0)

DEV = "cuda"
# Shape selection is load-bearing: at most bf16 shapes H100 cuBLAS and Triton share a K-step of 64
# and agree bitwise, which would make the assertions vacuous. These are from the divergent corner.
CASES = [
    (3, 1000, 333, 777, torch.bfloat16),
    (8, 128, 1224, 192, torch.float32),
    (4, 96, 1024, 256, torch.float32),
]

_fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        _fails.append(name)


def _bitcmp(x, y):
    """Bit-pattern mismatch count (torch.equal is blind to signed zero)."""
    vx = x.view(torch.int16) if x.dtype in (torch.bfloat16, torch.float16) else x.view(torch.int32)
    vy = y.view(torch.int16) if y.dtype in (torch.bfloat16, torch.float16) else y.view(torch.int32)
    return int((vx != vy).sum().item())


def _operands(seed=0):
    """Operands whose K-ranges are scaled across exponents, so the order of the K reduction is
    observable in the low bits."""
    torch.manual_seed(seed)
    out = []
    for B, M, K, N, dt in CASES:
        a = torch.randn(B, M, K, device=DEV, dtype=dt)
        b = torch.randn(B, K, N, device=DEV, dtype=dt)
        step = max(1, K // 8)
        for i, s in enumerate(range(0, K, step)):
            b[:, s : s + step, :] *= 10.0 ** (i - 4)
        out.append((a, b))
    return out


# forward captures -- the three contexts that must all stay on the Triton kernel
def capture_fwd_nograd():
    with torch.no_grad():
        return [torch.ops.aten.bmm(a, b).clone() for a, b in _operands()]


def capture_fwd_grad():
    out = []
    for a, b in _operands():
        a = a.clone().requires_grad_(True)
        b = b.clone().requires_grad_(True)
        out.append(torch.ops.aten.bmm(a, b).detach().clone())
    return out


def capture_fwd_recompute():
    """Original forward and checkpoint re-forward, via megatron's ``tensor_parallel.checkpoint``.

    ``seen[0]`` is the original forward; ``seen[1]`` is the re-forward on the autograd thread.
    """
    from megatron.core import tensor_parallel

    per_case = []
    for a0, b0 in _operands():
        seen = []
        a = a0.clone().requires_grad_(True)
        b = b0.clone().requires_grad_(True)

        def body(x, y, _seen=seen):
            z = torch.ops.aten.bmm(x, y)
            _seen.append(z.detach().clone())
            return z

        out = tensor_parallel.checkpoint(body, False, a, b)
        out.float().square().sum().backward()
        torch.cuda.synchronize()
        per_case.append(seen)
    return per_case


# profiler provenance
_TRITON_BMM = "bmm_kernel"


def _is_cublas_gemm(name):
    n = name.lower()
    if _TRITON_BMM in n or "triton" in n:
        return False
    return any(t in n for t in ("gemm", "cutlass", "nvjet", "xmma", "cublas"))


def _kernels(fn):
    """CUDA kernel names launched by ``fn``."""
    from torch.profiler import ProfilerActivity, profile

    fn()  # warm up: autotuning / handle creation must not land in the trace
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [
        e.name
        for e in prof.events()
        if getattr(e, "device_type", None) is not None
        and str(e.device_type).endswith("CUDA")
        and e.self_device_time_total > 0
    ]


def _fwd_bwd_kernels():
    a, b = _operands()[0]
    a = a.clone().requires_grad_(True)
    b = b.clone().requires_grad_(True)
    holder = {}

    def fwd():
        holder["out"] = torch.ops.aten.bmm(a, b)

    fwd_k = _kernels(fwd)
    out = torch.ops.aten.bmm(a, b)
    loss = out.float().square().sum()

    def bwd():
        a.grad = b.grad = None
        loss.backward(retain_graph=True)

    bwd_k = _kernels(bwd)
    return fwd_k, bwd_k


def main():
    assert torch.cuda.is_available(), "needs a GPU"
    from vllm.model_executor.layers import batch_invariant as bi

    bi.enable_batch_invariant_mode()
    print(f"[bmm-scope-test] CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}", flush=True)

    from skyrl.backends.skyrl_train.isoexec.ops.mm import mm_fwd_scope as S

    # ---- 0. non-vacuity: the two kernels must actually disagree
    print("\n[0] non-vacuity: Triton bmm vs cuBLAS bmm at the test shapes")
    diffs = []
    for a, b in _operands():
        diffs.append(_bitcmp(bi.bmm_batch_invariant(a, b), S._cublas_bmm(a, b)))
    check(
        "Triton bmm != cuBLAS bmm bitwise",
        all(d > 0 for d in diffs),
        f"mismatched elements per case: {diffs}",
    )

    # ---- OFF arm (vLLM's unscoped aten::bmm)
    print("\n[1-3] capturing the OFF arm (vLLM's unscoped aten::bmm)")
    assert not S.mm_fwd_only_bmm_enabled()
    off_nograd = capture_fwd_nograd()
    off_grad = capture_fwd_grad()
    off_recompute = capture_fwd_recompute()
    off_fwd_k, off_bwd_k = _fwd_bwd_kernels()
    check(
        "OFF arm backward runs the Triton bmm_kernel (baseline)",
        any(_TRITON_BMM in k for k in off_bwd_k),
        f"{sum(_TRITON_BMM in k for k in off_bwd_k)} bmm_kernel launches in the backward",
    )

    # ---- flip ON
    os.environ["SKYRL_ISOEXEC_MM_FWD_ONLY_BMM"] = "1"
    installed = S.install_bmm_scope()
    print(f"\n[bmm-scope-test] install_bmm_scope() -> {installed}", flush=True)
    check("install_bmm_scope() succeeded (self-check + post-registration verification)", installed)
    if not installed:
        return _report()

    # ---- 1. forward under no_grad
    print("\n[1] forward under no_grad (the engine's prefill/decode context)")
    on_nograd = capture_fwd_nograd()
    bad = [_bitcmp(x, y) for x, y in zip(on_nograd, off_nograd)]
    check("no_grad forward bitwise ON == OFF", all(b == 0 for b in bad), f"mismatches per case: {bad}")

    # ---- 2. forward with grad enabled
    print("\n[2] forward with grad enabled (the trainer's training/scoring forward)")
    on_grad = capture_fwd_grad()
    bad = [_bitcmp(x, y) for x, y in zip(on_grad, off_grad)]
    check("grad-enabled forward bitwise ON == OFF", all(b == 0 for b in bad), f"mismatches per case: {bad}")

    # ---- 3. checkpoint recompute
    print("\n[3] checkpoint RE-forward on the autograd thread (the one path the live gate cannot see)")
    on_recompute = capture_fwd_recompute()
    check(
        "the recompute re-entry hook FIRED",
        S._recompute_fired["n"] > 0,
        "0 means the re-forward ran unmarked -- do not trust anything below it",
    )
    check("checkpoint body ran twice per case (original + re-forward)", all(len(s) == 2 for s in on_recompute))
    bad_self = [_bitcmp(s[0], s[1]) for s in on_recompute if len(s) == 2]
    check(
        "re-forward bitwise == original forward (flag ON)",
        all(b == 0 for b in bad_self),
        f"mismatches per case: {bad_self}",
    )
    bad_ab = [_bitcmp(on[1], off[1]) for on, off in zip(on_recompute, off_recompute) if len(on) == 2 == len(off)]
    check(
        "re-forward bitwise ON == OFF",
        all(b == 0 for b in bad_ab),
        f"mismatches per case: {bad_ab}",
    )

    # ---- 4. backward provenance
    print("\n[4] kernel provenance: forward Triton, backward cuBLAS")
    on_fwd_k, on_bwd_k = _fwd_bwd_kernels()
    check(
        "forward still launches the Triton bmm_kernel",
        any(_TRITON_BMM in k for k in on_fwd_k),
        f"forward kernels: {sorted(set(on_fwd_k))}",
    )
    check(
        "forward launches NO cuBLAS GEMM",
        not any(_is_cublas_gemm(k) for k in on_fwd_k),
        f"forward kernels: {sorted(set(on_fwd_k))}",
    )
    check(
        "backward launches cuBLAS GEMM kernels",
        any(_is_cublas_gemm(k) for k in on_bwd_k),
        f"backward kernels: {sorted(set(on_bwd_k))}",
    )
    check(
        "backward launches NO Triton bmm_kernel",
        not any(_TRITON_BMM in k for k in on_bwd_k),
        f"backward kernels: {sorted(set(on_bwd_k))}",
    )

    # ---- 5. determinism of a full fwd+bwd
    print("\n[5] run-to-run determinism of fwd+bwd")

    def grads():
        gs = []
        for a0, b0 in _operands(seed=3):
            a = a0.clone().requires_grad_(True)
            b = b0.clone().requires_grad_(True)
            torch.ops.aten.bmm(a, b).float().square().sum().backward()
            gs += [a.grad.clone(), b.grad.clone()]
        torch.cuda.synchronize()
        return gs

    g1, g2, g3 = grads(), grads(), grads()
    check(
        "grads bitwise identical over 3 runs",
        all(_bitcmp(x, y) == 0 and _bitcmp(x, z) == 0 for x, y, z in zip(g1, g2, g3)),
    )

    # ---- 6. a custom Function whose backward calls aten::bmm
    print("\n[6] custom autograd Function whose backward calls torch.ops.aten.bmm")

    class _F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w):
            ctx.save_for_backward(x, w)
            return torch.ops.aten.bmm(x, w)

        @staticmethod
        def backward(ctx, g):
            x, w = ctx.saved_tensors
            g = g.contiguous()
            return torch.ops.aten.bmm(g, w.transpose(1, 2)), torch.ops.aten.bmm(x.transpose(1, 2), g)

    a0, b0 = _operands(seed=5)[0]

    def custom_bwd():
        a = a0.clone().requires_grad_(True)
        w = b0.clone().requires_grad_(True)
        _F.apply(a, w).float().square().sum().backward()
        torch.cuda.synchronize()
        return a.grad.clone(), w.grad.clone()

    ka = _kernels(lambda: custom_bwd())
    n_triton = sum(_TRITON_BMM in k for k in ka)
    n_cublas = sum(_is_cublas_gemm(k) for k in ka)
    check(
        "custom-Function: 1 Triton bmm (its forward) + 2 cuBLAS GEMMs (its dgrad+wgrad)",
        n_triton == 1 and n_cublas == 2,
        f"triton={n_triton} cublas={n_cublas}",
    )
    c1, c2 = custom_bwd(), custom_bwd()
    check(
        "custom-Function backward is deterministic run-to-run",
        _bitcmp(c1[0], c2[0]) == 0 and _bitcmp(c1[1], c2[1]) == 0,
    )

    return _report()


def _report():
    print(f"\n{'FAILED: ' + ', '.join(_fails) if _fails else 'ALL CHECKS PASSED'}", flush=True)
    return 1 if _fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
