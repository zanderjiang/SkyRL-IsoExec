"""Certification + measurement battery for the one-gather chunk sort (SKYRL_ISOEXEC_MOE_CHUNK_SORT).

A chunk reordering is a bijection on rows, so it is one ``index_select``; the gates below check
forward/backward bits, row provenance, bijectivity, admission, census and perf against megatron's
``torch.cat`` expression at the shapes the alltoall dispatcher runs.

Usage:
  source /mnt/local_storage/env.sh
  CUDA_VISIBLE_DEVICES=0 uv run --isolated --extra isoexec python \
    skyrl/backends/skyrl_train/isoexec/ops/moe/tests/moe_chunk_sort_battery_gpu.py \
    > /mnt/local_storage/logs/moe_chunk_sort/battery.log 2>&1
"""

import os
import sys
import time

os.environ.setdefault("SKYRL_ISOEXEC_MOE_CHUNK_SORT", "1")

import torch

if not torch.cuda.is_available():  # needs one CUDA device
    print("SKIP: no CUDA device")
    raise SystemExit(0)

from skyrl.backends.skyrl_train.isoexec.ops.moe import moe_chunk_sort as C

DEV = "cuda"

# The live GLM-4.7-Flash DAPO trainer shape: EP=8 x 8 local experts -> 64 splits, h=2048,
# 6874 tokens/microbatch at topk=4 -> 27,496 routed rows.
LIVE_SPLITS = 64
LIVE_HIDDEN = 2048
LIVE_ROWS = 27496

# The Qwen3.5 DAPO arm: EP=32 x 8 local experts -> 256 splits, h=2048, ~16,752 routed rows.
# Gate 7 must admit this shape too, not just the GLM one.
QWEN_SPLITS = 256
QWEN_HIDDEN = 2048
QWEN_ROWS = 16752

PASS = 0
FAIL = 0
SKIP_PERF = os.environ.get("MOE_CS_SKIP_PERF", "0") == "1"


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {name} {detail}", flush=True)
    else:
        FAIL += 1
        print(f"  FAIL  {name} {detail}", flush=True)


def bitne(a, b):
    return C._bitcmp(a, b)


# ------------------------------------------------------------------------------------------------
# Chunk-vector populations: each is a (split_sizes, sorted_idxs) pair on device.
# ------------------------------------------------------------------------------------------------
def layouts(n_splits, total, seed):
    g = torch.Generator().manual_seed(seed)
    out = []

    even = torch.full((n_splits,), total // n_splits, dtype=torch.long)
    even[-1] += total - int(even.sum())
    out.append(("balanced", even))

    skew = torch.full((n_splits,), 1, dtype=torch.long)
    skew[0] = total - (n_splits - 1)
    out.append(("one-huge-expert", skew))

    empt = torch.full((n_splits,), 0, dtype=torch.long)
    live = torch.randperm(n_splits, generator=g)[: max(1, n_splits // 4)]
    per = total // len(live)
    empt[live] = per
    empt[live[-1]] += total - int(empt.sum())
    out.append(("three-quarters-empty", empt))

    rag = torch.randint(0, max(2, 3 * total // n_splits), (n_splits,), generator=g)
    if int(rag.sum()) == 0:
        rag[0] = total
    out.append(("ragged", rag))

    res = []
    for name, ss in out:
        # megatron's reordering is a fixed EP-derived one; a random permutation is harder
        si = torch.randperm(n_splits, generator=g)
        res.append((name, ss.to(DEV), si.to(DEV), int(ss.sum())))
    return res


def populations(n, h, dtype, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    yield "randn", torch.randn(n, h, generator=g, device=DEV, dtype=dtype)
    # an underflow population makes bf16 rounding and signed zeros live: a pure copy must not care
    yield "underflow", (torch.randn(n, h, generator=g, device=DEV, dtype=dtype) * (2.0**-9))


# ================================================================================================
def gate_1_2_forward_and_provenance():
    print("\n[GATE 1+2] forward bits + row provenance vs megatron's cat expression", flush=True)
    for n_splits in (8, 64, 256):
        for name, ss, si, n in layouts(n_splits, LIVE_ROWS, seed=n_splits):
            if n == 0:
                continue
            for dtype in (torch.bfloat16, torch.float32):
                for pname, x in populations(n, 256, dtype, seed=n_splits + 7):
                    p = torch.randn(n, device=DEV, dtype=torch.float32)
                    ref, refp = C._reference(x, ss, si, p)
                    got, gotp = C.sort_chunks_gather(x, ss, si, p)
                    bad = bitne(ref, got)
                    badp = bitne(refp, gotp)
                    check(
                        f"fwd splits={n_splits} {name} {dtype} {pname}",
                        bad == 0 and badp == 0,
                        f"({bad} token elems, {badp} prob elems of {ref.numel()}+{refp.numel()})",
                    )
            marker = torch.arange(n, device=DEV, dtype=torch.int64).unsqueeze(1)
            mref, _ = C._reference(marker, ss, si, None)
            g = C.gather_map(ss, si, n, torch.device(DEV))
            check(f"provenance splits={n_splits} {name}", torch.equal(mref.reshape(-1), g))


def gate_3_backward():
    print("\n[GATE 3] backward bits -- both forms, one shared cotangent", flush=True)
    for n_splits in (8, 64):
        for name, ss, si, n in layouts(n_splits, 8192, seed=100 + n_splits):
            if n == 0:
                continue
            for dtype in (torch.bfloat16, torch.float32):
                base = torch.randn(n, 256, device=DEV, dtype=dtype)
                pbase = torch.randn(n, device=DEV, dtype=torch.float32)
                cot = torch.randn(n, 256, device=DEV, dtype=dtype)
                cotp = torch.randn(n, device=DEV, dtype=torch.float32)

                xa = base.clone().requires_grad_(True)
                pa = pbase.clone().requires_grad_(True)
                ra, rap = C._reference(xa, ss, si, pa)
                torch.autograd.backward([ra, rap], [cot, cotp])

                xb = base.clone().requires_grad_(True)
                pb = pbase.clone().requires_grad_(True)
                rb, rbp = C.sort_chunks_gather(xb, ss, si, pb)
                torch.autograd.backward([rb, rbp], [cot, cotp])

                bad = bitne(xa.grad, xb.grad)
                badp = bitne(pa.grad, pb.grad)
                check(
                    f"bwd splits={n_splits} {name} {dtype}",
                    bad == 0 and badp == 0,
                    f"({bad} dx elems, {badp} dprob elems)",
                )


def gate_4_signed_zeros_and_nans():
    print("\n[GATE 4] signed zeros + NaN payloads survive both directions", flush=True)
    n_splits, total = 16, 4096
    for name, ss, si, n in layouts(n_splits, total, seed=41):
        if n == 0:
            continue
        x = torch.zeros(n, 64, device=DEV, dtype=torch.bfloat16)
        x[::2] = -0.0
        x[1::3] = float("nan")
        ref, _ = C._reference(x, ss, si, None)
        got, _ = C.sort_chunks_gather(x, ss, si, None)
        check(f"fwd -0.0/NaN bits {name}", bitne(ref, got) == 0)

        cot = torch.zeros(n, 64, device=DEV, dtype=torch.bfloat16)
        cot[::2] = -0.0
        xa = torch.randn(n, 64, device=DEV, dtype=torch.bfloat16).requires_grad_(True)
        out, _ = C.sort_chunks_gather(xa, ss, si, None)
        out.backward(cot)
        neg = int(torch.signbit(xa.grad).sum())
        expect = int(torch.signbit(cot).sum())
        check(f"bwd -0.0 preserved {name}", neg == expect, f"({neg} vs {expect} negative zeros)")


def gate_5_bijectivity():
    print("\n[GATE 5] the map is a bijection and the inverse restores the bits", flush=True)
    for n_splits in (8, 64, 256):
        for name, ss, si, n in layouts(n_splits, LIVE_ROWS, seed=200 + n_splits):
            if n == 0:
                continue
            g = C.gather_map(ss, si, n, torch.device(DEV))
            perm = torch.equal(torch.sort(g).values, torch.arange(n, device=DEV, dtype=g.dtype))
            x = torch.randn(n, 128, device=DEV, dtype=torch.bfloat16)
            rt = bitne(x.index_select(0, g).index_select(0, C.inverse_map(g)), x)
            check(f"bijection splits={n_splits} {name}", perm and rt == 0, f"(round-trip diff {rt})")


def gate_6_determinism_and_nonvacuity():
    print("\n[GATE 6] determinism + a control that MUST differ", flush=True)
    n_splits = 64
    for name, ss, si, n in layouts(n_splits, LIVE_ROWS, seed=300):
        if n < n_splits:
            continue
        x = torch.randn(n, 512, device=DEV, dtype=torch.bfloat16)
        a, _ = C.sort_chunks_gather(x, ss, si, None)
        b, _ = C.sort_chunks_gather(x, ss, si, None)
        check(f"determinism {name}", bitne(a, b) == 0)
        # non-vacuity: a rotated chunk order must move bits, else the bit-compare proves nothing
        si2 = torch.roll(si, 1)
        c, _ = C.sort_chunks_gather(x, ss, si2, None)
        moved = bitne(a, c)
        check(f"non-vacuity {name}", moved > 0, f"({moved} elements moved under a rotated order)")


def _live_layout(splits, rows, ep_rows=8):
    ss = torch.full((splits,), rows // splits, dtype=torch.long, device=DEV)
    ss[-1] += rows - int(ss.sum())
    # megatron's `sort_input_by_local_experts` reordering: (ep, local_experts) transposed
    si = torch.arange(splits, device=DEV).reshape(splits // ep_rows, ep_rows).T.reshape(-1)
    return ss, si


def gate_7_production_admission():
    print("\n[GATE 7] the SHIPPED chunk_sort_ready() admits the live DAPO shapes", flush=True)
    for tag, splits, hidden, rows, ep_rows in (
        ("GLM-4.7-Flash", LIVE_SPLITS, LIVE_HIDDEN, LIVE_ROWS, 8),
        ("Qwen3.5", QWEN_SPLITS, QWEN_HIDDEN, QWEN_ROWS, 8),
    ):
        ss, si = _live_layout(splits, rows, ep_rows)
        for dtype in (torch.bfloat16, torch.float32):
            x = torch.randn(rows, hidden, device=DEV, dtype=dtype)
            p = torch.randn(rows, device=DEV, dtype=torch.float32)
            ok = C.chunk_sort_ready(x, ss, si, p)
            check(f"admission {tag} {dtype} splits={splits} h={hidden} rows={rows}", ok)

    # a shape it must refuse rather than crash on
    x = torch.randn(1024, 128, device=DEV, dtype=torch.bfloat16)
    bad_ss = torch.tensor([512, 512], device=DEV)
    bad_si = torch.tensor([0, 0], device=DEV)  # not a permutation
    refused = not C.chunk_sort_ready(x, bad_ss, bad_si, None)
    check("refuses a non-permutation sorted_idxs (fail-closed)", refused)
    # refusing it must not write off the shape, since the key is shape-only
    good_ss = torch.tensor([512, 512], device=DEV)
    good_si = torch.tensor([1, 0], device=DEV)
    y = torch.randn(1024, 128, device=DEV, dtype=torch.bfloat16)
    check(
        "a malformed call does not poison the (2,128,bf16) shape",
        C.chunk_sort_ready(y, good_ss, good_si, None),
    )


def gate_7a_admission_under_no_grad():
    """Regression: admission must not depend on the caller's grad mode. Every MoE forward here runs
    under ``torch.no_grad()``, and a probe that raises there used to cache a permanent rejection."""
    print("\n[GATE 7a] admission is independent of the caller's grad context", flush=True)
    ss, si = _live_layout(QWEN_SPLITS, QWEN_ROWS)
    for dtype in (torch.bfloat16, torch.float32):
        x = torch.randn(QWEN_ROWS, QWEN_HIDDEN, device=DEV, dtype=dtype)
        p = torch.randn(QWEN_ROWS, device=DEV, dtype=torch.float32)
        ok_grad, why_grad = C._admit(x, ss, si, p)
        with torch.no_grad():
            ok_ng, why_ng = C._admit(x, ss, si, p)
        check(f"_admit under enable_grad {dtype}", ok_grad, why_grad)
        check(f"_admit under no_grad {dtype}", ok_ng, why_ng)

        # and the forward bits must not move with the grad context either
        ref, refp = C._reference(x, ss, si, p)
        with torch.no_grad():
            got, gotp = C.sort_chunks_gather(x, ss, si, p)
        check(f"fwd bits under no_grad {dtype}", bitne(ref, got) == 0 and bitne(refp, gotp) == 0)

    # the production predicate end to end: a fresh shape admitted from inside a no_grad
    C._STATE.pop(C.admission_key(torch.empty(1, 512, device=DEV, dtype=torch.bfloat16), ss), None)
    ss2, si2 = _live_layout(QWEN_SPLITS, QWEN_ROWS)
    x2 = torch.randn(QWEN_ROWS, 512, device=DEV, dtype=torch.bfloat16)
    with torch.no_grad():
        ready = C.chunk_sort_ready(x2, ss2, si2, None)
    check("chunk_sort_ready() admits a fresh shape from inside torch.no_grad()", ready)


def gate_7b_key_is_shape_only():
    """Regression: two row counts at the same (splits, hidden, dtype) must hit the same cache entry
    and the second must not re-run the ~2-3 ms probe."""
    print("\n[GATE 7b] the admission key carries no data-dependent axis", flush=True)
    splits, hidden = QWEN_SPLITS, QWEN_HIDDEN
    rows_a, rows_b = QWEN_ROWS, QWEN_ROWS + 1024
    ss_a, si_a = _live_layout(splits, rows_a)
    ss_b, si_b = _live_layout(splits, rows_b)
    xa = torch.randn(rows_a, hidden, device=DEV, dtype=torch.bfloat16)
    xb = torch.randn(rows_b, hidden, device=DEV, dtype=torch.bfloat16)

    ka, kb = C.admission_key(xa, ss_a), C.admission_key(xb, ss_b)
    check(f"same key across row counts {rows_a} vs {rows_b}", ka == kb, f"({ka} vs {kb})")
    check("the key has exactly three axes (splits, hidden, dtype)", len(ka) == 3, f"({ka})")

    C._STATE.pop(ka, None)
    n0 = len(C._STATE)
    ok_a = C.chunk_sort_ready(xa, ss_a, si_a, None)
    check(f"first call at rows={rows_a} admits", ok_a)
    check("first call added exactly one entry", len(C._STATE) == n0 + 1, f"({len(C._STATE)} entries)")

    t0 = time.perf_counter()
    ok_b = C.chunk_sort_ready(xb, ss_b, si_b, None)
    torch.cuda.synchronize()
    dt_us = (time.perf_counter() - t0) * 1e6
    check(f"second call at rows={rows_b} is served from the cache", ok_b)
    check("no new cache entry for a new row count", len(C._STATE) == n0 + 1, f"({len(C._STATE)} entries)")
    # the probe costs ~2-3 ms; the precondition path is host integers on `splits` elements
    check("the cached path did not re-probe", dt_us < 500.0, f"({dt_us:.1f} us for the second call)")


def gate_7c_wrapper_census():
    """Drives the installed wrapper (not the raw helper) and asserts the served counter moves."""
    print("\n[GATE 7c] the installed wrapper keeps a served/declined census", flush=True)
    try:
        installed = C.install_chunk_sort()
        from megatron.core.transformer.moe import token_dispatcher as td
    except Exception as e:  # noqa: BLE001
        check("install_chunk_sort()", False, f"({type(e).__name__}: {e})")
        return
    check("install_chunk_sort() rebound the dispatcher's module global", installed)

    ss, si = _live_layout(QWEN_SPLITS, QWEN_ROWS)
    x = torch.randn(QWEN_ROWS, QWEN_HIDDEN, device=DEV, dtype=torch.bfloat16)
    p = torch.randn(QWEN_ROWS, device=DEV, dtype=torch.float32)
    s0, d0, _ = C.chunk_sort_stats()
    for _ in range(8):
        out, outp = td.sort_chunks_by_idxs(x, ss, si, probs=p)
    s1, d1, reason = C.chunk_sort_stats()
    check("served count advanced through the wrapper", s1 - s0 == 8, f"(served {s0} -> {s1}, declined {d0} -> {d1})")

    ref, refp = C._reference(x, ss, si, p)
    check("the wrapper's output is bit-equal to megatron's cat", bitne(ref, out) == 0 and bitne(refp, outp) == 0)

    # a malformed call must decline, not crash, and must not block the next good call
    td.sort_chunks_by_idxs(x, ss, torch.zeros_like(si), probs=p)
    s2, d2, reason = C.chunk_sort_stats()
    check("a malformed call declines", d2 == d1 + 1 and s2 == s1, f"(declined {d1} -> {d2}, reason={reason!r})")
    td.sort_chunks_by_idxs(x, ss, si, probs=p)
    s3, _, _ = C.chunk_sort_stats()
    check("the shape is still served after a malformed call", s3 == s2 + 1)


def _bench(fn, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us


def gate_8_perf():
    print("\n[GATE 8] gather vs megatron's cat at the live trainer shape", flush=True)
    if SKIP_PERF:
        print("  SKIPPED (MOE_CS_SKIP_PERF=1) -- timings on a contended GPU measure contention", flush=True)
        return
    ss = torch.full((LIVE_SPLITS,), LIVE_ROWS // LIVE_SPLITS, dtype=torch.long, device=DEV)
    ss[-1] += LIVE_ROWS - int(ss.sum())
    si = torch.arange(LIVE_SPLITS, device=DEV).reshape(8, 8).T.reshape(-1)
    ss_cpu = ss.cpu()  # megatron pre-stages this to the host; measure both layouts
    x = torch.randn(LIVE_ROWS, LIVE_HIDDEN, device=DEV, dtype=torch.bfloat16)
    p = torch.randn(LIVE_ROWS, device=DEV, dtype=torch.float32)

    print(f"  shape: rows={LIVE_ROWS} hidden={LIVE_HIDDEN} splits={LIVE_SPLITS} bf16", flush=True)
    for tag, s, i in (("split_sizes on GPU", ss, si), ("split_sizes on CPU", ss_cpu, si.cpu())):
        ref = _bench(lambda s=s, i=i: C._reference(x, s, i, p))
        got = _bench(lambda s=s, i=i: C.sort_chunks_gather(x, s, i, p))
        print(f"  FWD  {tag:22s}  cat {ref:8.1f} us -> gather {got:8.1f} us   {ref / got:5.2f}x", flush=True)

    def fb(fn):
        xa = x.detach().clone().requires_grad_(True)
        out, _ = fn(xa, ss, si, p)
        out.backward(torch.ones_like(out))

    ref = _bench(lambda: fb(C._reference), iters=20)
    got = _bench(lambda: fb(C.sort_chunks_gather), iters=20)
    print(f"  FWD+BWD                        cat {ref:8.1f} us -> gather {got:8.1f} us   {ref / got:5.2f}x", flush=True)

    # the per-call tax once the probe no longer re-runs; charged on every call, so it has to stay
    # small against the saving above
    for tag, s, i, rows in (
        ("splits=64 host", ss.cpu(), si.cpu(), LIVE_ROWS),
        ("splits=64 device", ss, si, LIVE_ROWS),
    ):
        us = _bench(lambda s=s, i=i, rows=rows: C.preconditions_ok(s, i, rows), iters=200)
        print(f"  PRECONDITIONS {tag:18s} {us:8.2f} us/call", flush=True)
    qss, qsi = _live_layout(QWEN_SPLITS, QWEN_ROWS)
    for tag, s, i in (("splits=256 host", qss.cpu(), qsi.cpu()), ("splits=256 device", qss, qsi)):
        us = _bench(lambda s=s, i=i: C.preconditions_ok(s, i, QWEN_ROWS), iters=200)
        print(f"  PRECONDITIONS {tag:18s} {us:8.2f} us/call", flush=True)
    print(
        "  Per MoE layer the alltoall dispatcher calls this TWICE; the arm has 46 MoE layers.\n"
        "  Halo trace attribution for the dispatch-side pair alone: 127.1 us/layer.",
        flush=True,
    )


def main():
    if not torch.cuda.is_available():
        print("no CUDA -- this battery is GPU-only")
        return 2
    print(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__}", flush=True)
    print(f"SKYRL_ISOEXEC_MOE_CHUNK_SORT={os.environ.get('SKYRL_ISOEXEC_MOE_CHUNK_SORT')}", flush=True)
    gate_1_2_forward_and_provenance()
    gate_3_backward()
    gate_4_signed_zeros_and_nans()
    gate_5_bijectivity()
    gate_6_determinism_and_nonvacuity()
    gate_7_production_admission()
    gate_7a_admission_under_no_grad()
    gate_7b_key_is_shape_only()
    gate_7c_wrapper_census()
    gate_8_perf()
    served, declined, reason = C.chunk_sort_stats()
    print(f"\ncensus: served={served} declined={declined} last_decline={reason!r}", flush=True)
    print(f"admission table: {len(C._STATE)} shape(s) -- {sorted(C._STATE.keys())}", flush=True)
    print(f"\n=== {PASS} PASS / {FAIL} FAIL ===", flush=True)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
