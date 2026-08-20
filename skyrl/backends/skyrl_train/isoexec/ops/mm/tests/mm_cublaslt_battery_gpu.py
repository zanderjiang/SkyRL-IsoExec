"""Offline certification battery for the pinned-cuBLASLt dense-GEMM provider (wave-5).

Runs against the ACTUAL landed-shape code (imports mm_cublaslt from this scratchpad, and the repo's
mm_tiles for the composition test).  Every gate prints PASS/FAIL; a final count gates the landing.

GATES (per shape x per bucket unless noted):
  1. M-invariance + cross-bucket : row r of a decode-M (<1024) call == the same row embedded in an
     M=8192 (trainer-bucket) call, bit-for-bit, at M in {1,2,17,64,320,512,1023}, over 3 input
     populations INCLUDING a live signed-zero population (proves the int16 bit-compare is non-vacuous).
  2. determinism : two identical calls are bit-equal (decode + trainer bucket).
  3. graph capture+replay == eager, bit-for-bit (decode M=320 + trainer M=8192).
  4. bias != None routed to the Triton fallthrough (installed wrapper == stock bit-for-bit).
  5. non-contiguous / non-production layout routed to the fallthrough (== stock bit-for-bit).
  6. install-order composition with mm_tiles (BOTH orders): idempotent, and every shape UNLISTED by
     cuBLASLt is bit-equal to stock Triton; report which order lets cuBLASLt win (load-bearing order).

Usage:  CUDA_VISIBLE_DEVICES=4 uv run --isolated --extra isoexec python battery.py
"""

import os
import sys

os.environ.setdefault("SKYRL_ISOEXEC_MM_CUBLASLT", "1")

import torch


if not torch.cuda.is_available():  # promoted nightly battery: needs one CUDA device
    print("SKIP: no CUDA device")
    raise SystemExit(0)

from skyrl.backends.skyrl_train.isoexec.ops.mm import mm_cublaslt as MC

DEV = "cuda"
SHAPES = sorted(MC._LEGACY_CUBLASLT_SHAPES)  # 5 (K,N) bf16 keys (the grandfathered Qwen table)
_P = 0
_F = 0
_FAILS = []


def _check(name, ok, detail=""):
    global _P, _F
    if ok:
        _P += 1
        print(f"  PASS  {name}  {detail}")
    else:
        _F += 1
        _FAILS.append(name)
        print(f"  FAIL  {name}  {detail}")


def bitcmp(a, b):
    va = a.view(torch.int16) if a.dtype == torch.bfloat16 else a.view(torch.int32)
    vb = b.view(torch.int16) if b.dtype == torch.bfloat16 else b.view(torch.int32)
    return int((va != vb).sum().item())


def neg_zeros(t):
    """Count -0.0 bit patterns (0x8000 in bf16) -- proves a population is signed-zero-live."""
    return int((t.view(torch.int16) == torch.tensor(-32768, dtype=torch.int16, device=t.device)).sum().item())


def populations(M, K):
    """Two magnitude populations (the third, signed-zero-live, is built with its own tiny W in the
    invariance gate because underflow to -0.0 requires BOTH operands tiny).  Returns {name: x}."""
    torch.manual_seed(1234 + M + K)
    return {
        "normal": torch.randn(M, K, device=DEV, dtype=torch.bfloat16) * 0.05,
        "big": torch.randn(M, K, device=DEV, dtype=torch.bfloat16) * 1.0,
    }


def signz_x(M, K):
    """Tiny mixed-sign activations that, against a matching tiny W, underflow products so ~half the
    output dot-products round to +/-0.0 with random sign -> a LIVE signed-zero population."""
    torch.manual_seed(4321 + M + K)
    return torch.randn(M, K, device=DEV, dtype=torch.bfloat16) * (2.0 ** -72)


def ext():
    e = MC._ext()
    e.init()
    return e


def raw_mm(e, x, W):
    """x:[M,K] @ W.t() via the driver (W is [N,K])."""
    M, K = x.shape
    N = W.shape[0]
    out = torch.empty((M, N), device=DEV, dtype=torch.bfloat16)
    e.mm(x.contiguous(), W, out)
    return out


# ------------------------------------------------------------------------------------------------
def gate_invariance_crossbucket(e):
    print("\n[1] M-invariance + cross-bucket (decode-M rows == same rows in M=8192 trainer batch)")
    # comparator sanity: the int16-view bit-compare MUST distinguish +0.0 from -0.0 (else the whole
    # battery is vacuous on signed zero -- mm_tiles hazard 6).
    pz = torch.zeros(1, dtype=torch.bfloat16, device=DEV)
    nz = torch.tensor([-0.0], dtype=torch.bfloat16, device=DEV)
    _check("bit-comparator distinguishes +0.0 from -0.0", bitcmp(pz, nz) == 1)

    Ms = [1, 2, 17, 64, 320, 512, 1023]
    signz_live = 0
    for (K, N) in SHAPES:
        torch.manual_seed(7 + K + N)
        W_norm = torch.randn(N, K, device=DEV, dtype=torch.bfloat16) * 0.05
        W_signz = torch.randn(N, K, device=DEV, dtype=torch.bfloat16) * (2.0 ** -72)
        worst = 0
        checked = 0
        for popname in ("normal", "big", "signz"):
            for M in Ms:
                if popname == "signz":
                    x = signz_x(M, K); W = W_signz
                    x8 = signz_x(8192, K); x8[:M] = x
                else:
                    x = populations(M, K)[popname]; W = W_norm
                    x8 = torch.randn(8192, K, device=DEV, dtype=torch.bfloat16) * 0.05
                    x8[:M] = x
                o_small = raw_mm(e, x, W)          # M<1024 -> decode bucket
                o_big = raw_mm(e, x8, W)           # M=8192 -> trainer bucket
                if popname == "signz":
                    signz_live += neg_zeros(o_small)
                worst = max(worst, bitcmp(o_small, o_big[:M]))
                checked += o_small.numel()
        _check(f"invariance+crossbucket K={K} N={N}", worst == 0,
               f"{checked:,} elems (3 pops incl signed-zero), worst_mismatch={worst}")
    _check("signed-zero population is LIVE (non-vacuous bit-compare)", signz_live > 0,
           f"{signz_live} negative-zero outputs observed across the signed-zero reference")


def gate_determinism(e):
    print("\n[2] determinism (two identical calls bit-equal, both buckets)")
    for (K, N) in SHAPES:
        torch.manual_seed(11 + K + N)
        W = torch.randn(N, K, device=DEV, dtype=torch.bfloat16) * 0.05
        bad = 0
        for M in (320, 8192):
            x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16) * 0.05
            bad = max(bad, bitcmp(raw_mm(e, x, W), raw_mm(e, x, W)))
        _check(f"determinism K={K} N={N}", bad == 0, f"worst_mismatch={bad}")


def gate_graph(e):
    print("\n[3] CUDA-graph capture+replay == eager (bit-for-bit)")
    for (K, N) in SHAPES:
        torch.manual_seed(13 + K + N)
        W = torch.randn(N, K, device=DEV, dtype=torch.bfloat16) * 0.05
        bad = 0
        note = "ok"
        for M in (320, 8192):
            x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16) * 0.05
            eager = torch.empty((M, N), device=DEV, dtype=torch.bfloat16)
            e.mm(x, W, eager)
            torch.cuda.synchronize()
            cg = torch.empty((M, N), device=DEV, dtype=torch.bfloat16)
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                e.mm(x, W, cg)                      # warm (algo already pinned; no alloc)
            torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(g):
                    e.mm(x, W, cg)
            except Exception as ex:  # noqa: BLE001
                note = "CAPTURE_FAIL:" + type(ex).__name__
                bad = -1
                break
            cg.zero_(); g.replay(); torch.cuda.synchronize()
            bad = max(bad, bitcmp(cg, eager))
        _check(f"graph==eager K={K} N={N}", bad == 0, note if bad else "ok")


# ---- installed-wrapper routing/composition gates -----------------------------------------------
def _snapshot(bi):
    return (bi.matmul_persistent,
            getattr(bi, "_isoexec_stock_matmul_persistent", None))


def _restore(bi, snap):
    bi.matmul_persistent = snap[0]
    if snap[1] is None:
        if hasattr(bi, "_isoexec_stock_matmul_persistent"):
            del bi._isoexec_stock_matmul_persistent
    else:
        bi._isoexec_stock_matmul_persistent = snap[1]


def gate_routing():
    print("\n[4/5] bias + odd-layout routed to Triton fallthrough (installed wrapper == stock)")
    from vllm.model_executor.layers import batch_invariant as bi
    e = ext()
    snap = _snapshot(bi)
    stock = bi.matmul_persistent
    MC._LOG_ONCE = True   # silence duplicate install banner
    MC._DISABLED = False
    # install cuBLASLt over pristine stock
    ok = MC.install_mm_cublaslt()
    wrapped = bi.matmul_persistent
    try:
        _check("install returned True", ok)
        K, N = 2048, 1024  # a cuBLASLt table shape
        M = 320
        x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16) * 0.05
        W = torch.randn(N, K, device=DEV, dtype=torch.bfloat16) * 0.05
        b = W.t()  # production layout [K,N] stride (1,K)

        # sanity: production call is ROUTED to cuBLASLt (wrapper output == raw ext.mm output).
        # (It need NOT differ from Triton -- cuBLASLt and Triton coincide bit-for-bit on some
        # well-conditioned shapes; the signature move shows up on other shapes/data.  Routing, not
        # divergence, is the property.)
        o_wrap = wrapped(x, b)
        o_raw = raw_mm(e, x, W)
        _check("production shape ROUTED to cuBLASLt (wrapper == raw ext.mm)",
               bitcmp(o_wrap, o_raw) == 0)

        # (4) bias != None -> fallthrough == stock(bias)
        bias = torch.randn(N, device=DEV, dtype=torch.bfloat16) * 0.05
        _check("bias!=None routed to fallthrough",
               bitcmp(wrapped(x, b, bias=bias), stock(x, b, bias=bias)) == 0)

        # (5a) non-contiguous b (not the (1,K) production stride) -> fallthrough == stock
        Wpad = torch.randn(N, K + 8, device=DEV, dtype=torch.bfloat16) * 0.05
        b_odd = Wpad[:, :K].t()  # [K,N] but stride (1, K+8) != (1,K)
        _check("odd-layout b routed to fallthrough",
               bitcmp(wrapped(x, b_odd), stock(x, b_odd)) == 0,
               f"b.stride={tuple(b_odd.stride())}")

        # (5b) non-contiguous a rows -> still correct (wrapper contiguous()-es or falls through == cuBLASLt/stock consistent)
        x_odd = torch.randn(M, K * 2, device=DEV, dtype=torch.bfloat16)[:, ::2]  # stride (2K,2)
        _check("non-contiguous a handled (wrapper == stock on odd a-stride)",
               bitcmp(wrapped(x_odd, b_odd), stock(x_odd, b_odd)) == 0,
               f"a.stride={tuple(x_odd.stride())}")

        # unlisted shape (not in cuBLASLt table) -> fallthrough == stock
        Ku, Nu = 2048, 999
        xu = torch.randn(M, Ku, device=DEV, dtype=torch.bfloat16) * 0.05
        Wu = torch.randn(Nu, Ku, device=DEV, dtype=torch.bfloat16) * 0.05
        _check("unlisted shape routed to fallthrough",
               bitcmp(wrapped(xu, Wu.t()), stock(xu, Wu.t())) == 0)

        # fp32 excluded even if (K,N) matched -> fallthrough (dtype guard)
        xf = torch.randn(M, K, device=DEV, dtype=torch.float32)
        Wf = torch.randn(N, K, device=DEV, dtype=torch.float32)
        _check("fp32 routed to fallthrough (bf16-only guard)",
               bitcmp(wrapped(xf, Wf.t()), stock(xf, Wf.t())) == 0)
    finally:
        _restore(bi, snap)


class _CountingExt:
    """Proxy over the real ext that counts mm() calls -- so the composition test can tell whether
    cuBLASLt actually RAN for a shape, independent of any numeric coincidence with Triton (on H100
    the two are bit-identical for these shapes, so 'who won' cannot be read off the output)."""
    def __init__(self, real):
        self._real = real
        self.mm_calls = 0

    def init(self):
        return self._real.init()

    def probe(self, K, N, bucket):
        return self._real.probe(K, N, bucket)

    def pinned_count(self):
        return self._real.pinned_count()

    def mm(self, x, w, out):
        self.mm_calls += 1
        return self._real.mm(x, w, out)


def gate_composition():
    print("\n[6] install-order composition with mm_tiles (both orders; who-ran via call counter)")
    os.environ["SKYRL_ISOEXEC_MM_TILES"] = "1"
    from vllm.model_executor.layers import batch_invariant as bi
    from skyrl.backends.skyrl_train.isoexec.ops.mm.mm_tiles import install_mm_tiles
    import skyrl.backends.skyrl_train.isoexec.ops.mm.mm_tiles as MT

    snap = _snapshot(bi)
    stock = snap[0]
    real_ext = ext()
    counting = _CountingExt(real_ext)
    # force install_mm_cublaslt to close over the counting proxy
    orig_ext_fn = MC._ext
    MC._ext = lambda: counting

    def reset():
        _restore(bi, snap)
        MT._TILE_LOG_ONCE = True
        MC._LOG_ONCE = True
        MC._DISABLED = False
        MC._WRAPPER = None

    Kg, Ng = 2048, 1     # gate scalar: mm_tiles yes, cublaslt no
    Ku, Nu = 2048, 999   # neither table
    Kc, Nc = 2048, 1024  # cublaslt (and mm_tiles) shape
    M = 320
    xg = torch.randn(M, Kg, device=DEV, dtype=torch.bfloat16) * 0.05
    Wg = torch.randn(Ng, Kg, device=DEV, dtype=torch.bfloat16) * 0.05
    xu = torch.randn(M, Ku, device=DEV, dtype=torch.bfloat16) * 0.05
    Wu = torch.randn(Nu, Ku, device=DEV, dtype=torch.bfloat16) * 0.05
    xc = torch.randn(M, Kc, device=DEV, dtype=torch.bfloat16) * 0.05
    Wc = torch.randn(Nc, Kc, device=DEV, dtype=torch.bfloat16) * 0.05
    stock_g = stock(xg, Wg.t())
    stock_u = stock(xu, Wu.t())

    try:
        for order in (("mm_tiles", "cublaslt"), ("cublaslt", "mm_tiles")):
            reset()
            for who in order:
                install_mm_tiles() if who == "mm_tiles" else MC.install_mm_cublaslt()
            w = bi.matmul_persistent

            # unlisted-by-cublaslt shapes must be BIT-EQUAL TO STOCK TRITON (mm_tiles neutral)
            _check(f"[{'>'.join(order)}] unlisted (K2048,N999) == stock Triton",
                   bitcmp(w(xu, Wu.t()), stock_u) == 0)
            _check(f"[{'>'.join(order)}] gate N=1 (mm_tiles-only) == stock Triton",
                   bitcmp(w(xg, Wg.t()), stock_g) == 0)

            # who ran the shared cuBLASLt shape? (call counter, not output compare)
            counting.mm_calls = 0
            got = w(xc, Wc.t())
            ran_cublaslt = counting.mm_calls == 1
            # correctness regardless of who ran: bit-equal to at least stock (Triton==cuBLASLt here)
            _check(f"[{'>'.join(order)}] shared shape correct (== stock Triton)",
                   bitcmp(got, stock(xc, Wc.t())) == 0,
                   f"cuBLASLt {'RAN (wins)' if ran_cublaslt else 'shadowed by mm_tiles'}")
            if order == ("mm_tiles", "cublaslt"):
                _check("[load-bearing order] cuBLASLt installed LAST -> cuBLASLt WINS the shared shape",
                       ran_cublaslt, "this is the engine install order (mm_tiles L90, cublaslt after)")
            else:
                # documents WHY order matters: cublaslt-then-mm_tiles shadows cuBLASLt.
                _check("[reverse order] cublaslt-then-mm_tiles shadows cuBLASLt (documented hazard)",
                       not ran_cublaslt, "engine MUST install cuBLASLt LAST; manifest handshake is the backstop")

            # idempotency: re-call install in the SAME order -> chain does NOT grow (one mm/call)
            for who in order:
                install_mm_tiles() if who == "mm_tiles" else MC.install_mm_cublaslt()
            counting.mm_calls = 0
            w2 = bi.matmul_persistent
            _ = w2(xc, Wc.t())
            _check(f"[{'>'.join(order)}] idempotent re-install (<=1 cuBLASLt call/matmul)",
                   counting.mm_calls <= 1, f"mm_calls={counting.mm_calls}")
        reset()
    finally:
        MC._ext = orig_ext_fn
        os.environ.pop("SKYRL_ISOEXEC_MM_TILES", None)


def main():
    print("=" * 90)
    print("WAVE-5 pinned-cuBLASLt provider -- OFFLINE CERTIFICATION BATTERY")
    print("device:", torch.cuda.get_device_name(0))
    print("=" * 90)
    e = ext()
    gate_invariance_crossbucket(e)
    gate_determinism(e)
    gate_graph(e)
    gate_routing()
    gate_composition()
    print("\n" + "=" * 90)
    print(f"RESULT: {_P} PASS / {_F} FAIL")
    if _FAILS:
        print("FAILED GATES:")
        for f in _FAILS:
            print("   -", f)
    print("=" * 90)
    sys.exit(1 if _F else 0)


if __name__ == "__main__":
    main()
