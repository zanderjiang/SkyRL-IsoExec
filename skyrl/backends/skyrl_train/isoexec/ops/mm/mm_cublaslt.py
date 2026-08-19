"""Pinned-cuBLASLt dense-GEMM provider for the batch-invariant matmul seam.

A signature-moving replacement for vLLM's Triton ``matmul_persistent`` on the bf16 dense GEMM shapes
where a pinned, non-split-K cuBLASLt kernel is both faster and legal. Unlike ``mm_tiles``, which
re-tiles the same Triton kernel, this is a different kernel and carries no general promise of
bit-equality -- so the shapes derived from the model census are admitted only after measuring
bit-equality against Triton per shape, which makes census extension bitwise-neutral. The five
grandfathered legacy shapes predate that gate and are admitted on the soundness gates alone.

The provider's own contract: M-invariant (the key excludes M), cross-bucket identical, deterministic
(pinned algo, no split-K), graph-safe (workspace allocated at init), installed in BOTH runtimes or
not at all, and fail-safe -- any doubt routes to the Triton fallthrough. The fp32 router and the N=1
gate scalar are excluded structurally: legal cuBLASLt is far slower there, and the fp32 router's
non-invariance under stock cuBLAS is the reason the batch-invariant override exists at all.

Flag: ``SKYRL_ISOEXEC_MM_CUBLASLT=1`` (default off, ``("both",)`` scope). Composes with
``SKYRL_ISOEXEC_MM_TILES`` in either install order; both wrap ``bi.matmul_persistent``.
"""

from __future__ import annotations

import functools
import glob
import os
import pathlib

import torch

_HERE = pathlib.Path(__file__).parent
_LOG_ONCE = False
_DISABLED = False  # set permanently True if the init-time self-check fails -> pure fallthrough.
_WRAPPER = None  # stable identity of OUR installed wrapper (for true idempotency).

# The legacy (grandfathered) table: five distinct (K, N) keys from Qwen3-Next, where (512, 2048)
# covers two sites. This is NOT a census -- several keys are dead on the models actually run (some
# widths omit a gate; the row-parallel ones never reach aten::mm when pik is installed) -- and a dead
# key costs one dict miss that never happens, so they stay. These five are admitted on the three
# soundness gates alone; every other shape must additionally prove bit-equality against Triton.
_LEGACY_CUBLASLT_SHAPES: set[tuple[int, int]] = {
    (2048, 1032),  # GDN in_proj_qkvzba   (N=1032)
    (512, 2048),  # GDN out_proj + attn o_proj  (two sites, one shape)
    (2048, 1024),  # attn qkv             (N=1024)
    (2048, 128),  # shared-expert fc1    (N=128)
    (64, 2048),  # shared-expert fc2    (K=64)
}

# The parallelisms a census is instantiated over. Wider than any one recipe on purpose, and taken as
# a union rather than per-process: the trainer and engine run at different TP, and a per-process
# table would give the two runtimes different tables.
_CENSUS_TPS = (1, 2, 4, 8)
_CENSUS_ETPS = (1, 2, 4, 8)

# Bring-up memory budget for one shape's self-check probe, applied per (K, N) so a shape over budget
# is dropped with its own measured size in the reason. A wide vocabulary GEMM can want hundreds of
# MiB per worker at small TP; 192 MiB admits it at the parallelisms this recipe actually runs.
# Deliberately NOT an environment variable: an unregistered SKYRL_ISOEXEC_* read is not forwarded to
# the Ray actors, so a launcher setting it would move the budget in the driver only.
_PROBE_BUDGET_BYTES = 192 * (1 << 20)


def _probe_bytes(K: int, N: int) -> int:
    """Peak output bytes one self-check probe allocates for ``(K, N)``.

    The cross-bucket gate is the high-water mark: two outputs plus the trainer-bucket input live at
    once, at 2 B/element. A pure function of the shape, so it is a CPU test rather than a surprise.
    """
    m_tr = _trainer_probe_m(N)
    return 2 * ((320 * N) + (m_tr * N) + (m_tr * K))


# The live shape set: legacy union the census shapes that passed every gate. Empty until install, so
# an un-installed provider can never claim a shape.
_ADMITTED: set[tuple[int, int]] = set()
_CENSUS_SITES: dict[tuple[int, int], str] = {}  # (K, N) -> site name, for the banner
_DROPPED: list[str] = []  # human-readable reasons, for the banner


@functools.lru_cache(maxsize=1)
def _ext():
    """JIT-load the C++ driver."""
    from torch.utils.cpp_extension import load

    inc: list[str] = []
    lib: list[str] = []
    rpath: list[str] = []
    for base in (
        pathlib.Path(torch.__file__).parent.parent / "nvidia" / "cu13",
        pathlib.Path(torch.__file__).parent.parent / "nvidia" / "cublas",
        pathlib.Path("/usr/local/cuda"),
    ):
        if (base / "include" / "cublasLt.h").exists():
            inc.append(str(base / "include"))
        for sub in ("lib", "lib64"):
            if list((base / sub).glob("libcublasLt.so*")):
                lib.append(str(base / sub))
                rpath.append(str(base / sub))

    # The wheel ships versioned libs (libcublasLt.so.13) but -lcublasLt needs an unversioned soname,
    # so stage symlinks in a shim dir and rpath the real dirs for load time.
    shim = pathlib.Path(_build_dir()) / "libshim"
    shim.mkdir(parents=True, exist_ok=True)
    ldflags: list[str] = []
    for nm in ("cublasLt", "cublas"):
        hits: list[str] = []
        for d in lib:
            hits += sorted(glob.glob(str(pathlib.Path(d) / f"lib{nm}.so*")))
        if hits:
            dst = shim / f"lib{nm}.so"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(hits[0])
    ldflags.append(f"-L{shim}")
    ldflags += [f"-Wl,-rpath,{p}" for p in rpath]
    ldflags += ["-lcublasLt", "-lcublas"]

    return load(
        name="isoexec_mm_cublaslt",
        sources=[str(_HERE / "csrc" / "cublaslt_pinned.cpp")],
        extra_include_paths=inc,
        extra_ldflags=ldflags,
        extra_cflags=["-O3"],
        build_directory=_build_dir(),
        verbose=bool(os.environ.get("ISOEXEC_MM_CUBLASLT_VERBOSE")),
    )


def _build_dir() -> str:
    d = (
        pathlib.Path(
            os.environ.get("ISOEXEC_MM_CUBLASLT_CACHE", pathlib.Path.home() / ".cache" / "isoexec_mm_cublaslt")
        )
        / "ext"
    )
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def mm_cublaslt_enabled() -> bool:
    return os.environ.get("SKYRL_ISOEXEC_MM_CUBLASLT", "0") == "1"


def _bitcmp(a: torch.Tensor, b: torch.Tensor) -> int:
    """Bit-pattern mismatch count; ``torch.equal`` is blind to signed zero."""
    va = a.view(torch.int16) if a.dtype == torch.bfloat16 else a.view(torch.int32)
    vb = b.view(torch.int16) if b.dtype == torch.bfloat16 else b.view(torch.int32)
    return int((va != vb).sum().item())


def _trainer_probe_m(N: int) -> int:
    """An M in the trainer bucket (>= 1024) whose ``[M, N]`` output is bounded in memory.

    The check needs a trainer-bucket M, not a large one, and a fixed large M would allocate hundreds
    of MiB per rank at bring-up for the widest census shapes.
    """
    return max(1024, min(8192, (1 << 23) // max(N, 1)))


def _biteq_vs_triton(K: int, N: int) -> tuple[bool, int]:
    """Is the pinned cuBLASLt provider bit-equal to the Triton kernel it is displacing?

    The admission gate for every derived (census) shape, measured per shape on the machine that will
    run it: where the two kernels happen to agree bit for bit, swapping one in cannot move the gate.
    Two populations, both buckets. Returns ``(bit_equal, mismatching_elements)``.
    """
    from vllm.model_executor.layers import batch_invariant as bi

    stock = getattr(bi, "_isoexec_stock_matmul_persistent", None) or bi.matmul_persistent
    dev = "cuda"
    bad = 0
    for M in (320, _trainer_probe_m(N)):
        for scale, kind in ((0.05, "randn"), (2.0**-9, "underflow")):
            # The underflow population drives products into the subnormal/zero region, where two
            # kernels that disagree at all will disagree first.
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16) * scale
            W = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * scale
            b = W.t()
            ref = stock(x, b, bias=None)
            got = _call(ext_handle(), x, b)
            bad += _bitcmp(ref, got)
            del ref, got, x, W, b
    return bad == 0, bad


_EXT = None


def ext_handle():
    return _EXT


def _self_check_shape(ext, K: int, N: int, *, require_biteq: bool) -> tuple[bool, str]:
    """The three soundness gates for one shape, plus bit-equality when the shape is derived.

      (i)   M-invariance : row 0 of a ``[17, K]`` input == the same row computed as ``[1, K]``.
      (ii)  cross-bucket : decode-pin output == trainer-pin output on identical rows.
      (iii) determinism  : two calls on the same input are bit-equal.
      (iv)  bit-equality vs Triton -- derived shapes only.

    Also confirms via ``probe()`` that each pinned algo is splitk==1 / reduction NONE. Returns
    ``(admitted, reason)``; a rejected shape falls through to Triton and is named in the banner.
    """
    dev = "cuda"
    for bucket in (0, 1):
        _cfg, _tile, _stages, splitk, redsc, found, _x = ext.probe(K, N, bucket)
        if found <= 0 or splitk != 1 or redsc != 0:
            return False, f"bucket{bucket} not a legal pin (found={found} splitK={splitk} red={redsc})"

    W = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05
    b = W.t()  # production layout: [K, N], stride (1, K)

    x1 = torch.randn(1, K, device=dev, dtype=torch.bfloat16) * 0.05
    x17 = torch.randn(17, K, device=dev, dtype=torch.bfloat16) * 0.05
    x17[0] = x1[0]
    if _bitcmp(_call(ext, x1, b)[0], _call(ext, x17, b)[0]) != 0:
        return False, "M-invariance ([1,K] vs row0 of [17,K])"

    m_tr = _trainer_probe_m(N)
    x320 = torch.randn(320, K, device=dev, dtype=torch.bfloat16) * 0.05
    o_dec = _call(ext, x320, b)
    x_tr = torch.randn(m_tr, K, device=dev, dtype=torch.bfloat16) * 0.05
    x_tr[:320] = x320
    o_tr = _call(ext, x_tr, b)
    if _bitcmp(o_dec, o_tr[:320]) != 0:
        return False, f"cross-bucket (decode-pin@320 == trainer-pin@{m_tr} on identical rows)"

    if _bitcmp(_call(ext, x320, b), _call(ext, x320, b)) != 0:
        return False, "determinism (two identical calls)"

    if require_biteq:
        ok, bad = _biteq_vs_triton(K, N)
        if not ok:
            return False, f"NOT bit-equal to Triton ({bad} elements differ) -- derived shapes must be"
    return True, ""


def _self_check(ext) -> bool:
    """All-or-nothing check over the grandfathered shapes, kept for the test batteries; production
    installs go through ``_admit_shapes``."""
    torch.manual_seed(0)
    for K, N in sorted(_LEGACY_CUBLASLT_SHAPES):
        ok, why = _self_check_shape(ext, K, N, require_biteq=False)
        if not ok:
            print(
                f"[ISOEXEC-MM-CUBLASLT] SELF-CHECK FAIL (K={K},N={N}): {why}. "
                "Disabling cuBLASLt provider (Triton fallthrough).",
                flush=True,
            )
            return False
    return True


def _admit_shapes(ext) -> tuple[set, dict, list]:
    """Build the live shape set: grandfathered legacy union derived-and-bit-equal.

    Returns ``(admitted, census_site_names, dropped_reasons)``. Never raises: a census that cannot be
    resolved yields the legacy table plus a banner saying so.
    """
    from . import mm_shapes

    torch.manual_seed(0)
    admitted: set = set()
    dropped: list = []

    for K, N in sorted(_LEGACY_CUBLASLT_SHAPES):
        ok, why = _self_check_shape(ext, K, N, require_biteq=False)
        if ok:
            admitted.add((K, N))
        else:
            dropped.append(f"(legacy {K},{N}): {why}")

    sites: dict = {}
    ctx = mm_shapes.live_context()
    if ctx is not None:
        try:
            census = mm_shapes.census_union(ctx["model_path"], _CENSUS_TPS, _CENSUS_ETPS)
        except Exception as e:  # noqa: BLE001
            census = {}
            dropped.append(f"census unavailable: {type(e).__name__}: {e}")
        for (K, N, dtype), name in sorted(census.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            if dtype is not torch.bfloat16:
                continue  # fp32 router / lm_head stay on Triton, by shape, structurally
            if (K, N) in admitted:
                sites[(K, N)] = name
                continue
            nbytes = _probe_bytes(K, N)
            if nbytes > _PROBE_BUDGET_BYTES:
                dropped.append(
                    f"({K},{N}) {name}: self-check probe needs {nbytes / (1 << 20):.0f} MiB > the "
                    f"{_PROBE_BUDGET_BYTES / (1 << 20):.0f} MiB budget "
                    f"(SKYRL_ISOEXEC_MM_CUBLASLT_PROBE_BUDGET_MB) -- NOT a self-check failure, a "
                    f"refusal to spend bring-up memory on a parallelism this process is not running"
                )
                continue
            try:
                ok, why = _self_check_shape(ext, K, N, require_biteq=True)
            except Exception as e:  # noqa: BLE001
                ok, why = False, f"self-check raised {type(e).__name__}: {e}"
            if ok:
                admitted.add((K, N))
                sites[(K, N)] = name
            else:
                dropped.append(f"({K},{N}) {name}: {why}")
    return admitted, sites, dropped


def _call(ext, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a:[M,K] @ b:[K,N] via the driver (b is W.t(); pass w=b.t()=[N,K] as the driver expects)."""
    M, K = a.shape
    N = b.shape[1]
    out = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    ext.mm(a.contiguous(), b.t(), out)
    return out


def install_mm_cublaslt() -> bool:
    """Point ``bi.matmul_persistent`` at the pinned-cuBLASLt provider for the admitted shapes.

    Wraps whatever ``matmul_persistent`` currently is as the fallthrough, so it composes with
    ``install_mm_tiles`` in either order and is idempotent. Off by default.

    Split-brain guard: this provider is not bit-equal to Triton, so an engine running cuBLASLt on a
    shape the trainer runs on Triton would break the gate. One registered ``("both",)``-scope flag is
    forwarded to every actor, both runtimes call this under it, and the cross-process op manifest
    catches a one-sided install as a loud mismatch.
    """
    global _LOG_ONCE, _DISABLED, _WRAPPER, _EXT, _ADMITTED, _CENSUS_SITES, _DROPPED

    if not mm_cublaslt_enabled():
        return False
    if _DISABLED:
        return False

    from vllm.model_executor.layers import batch_invariant as bi

    # Idempotency by stable identity: a no-op when our exact wrapper is already the outermost
    # binding, but re-asserts on top if something has since wrapped over us. Only ever wraps the
    # current top, so the chain cannot grow unboundedly.
    if _WRAPPER is not None and bi.matmul_persistent is _WRAPPER:
        return True

    # Build and init the driver (handle + workspace) before any graph capture or hot path.
    try:
        ext = _ext()
        ext.init()
    except Exception as e:  # noqa: BLE001
        print(f"[ISOEXEC-MM-CUBLASLT] build/init failed ({type(e).__name__}: {e}) -- Triton fallthrough.", flush=True)
        _DISABLED = True
        return False

    _EXT = ext

    # Init-time self-check, per shape. Pins both buckets for every candidate so no pinning happens
    # during graph capture later. A shape that fails any gate falls through to Triton; only an empty
    # result disables the provider.
    try:
        admitted, sites, dropped = _admit_shapes(ext)
    except Exception as e:  # noqa: BLE001
        print(f"[ISOEXEC-MM-CUBLASLT] self-check raised ({type(e).__name__}: {e}) -- Triton fallthrough.", flush=True)
        admitted, sites, dropped = set(), {}, [f"self-check raised {type(e).__name__}: {e}"]
    for reason in dropped:
        print(f"[ISOEXEC-MM-CUBLASLT] SHAPE DROPPED (fail-closed, Triton fallthrough) {reason}", flush=True)
    if not admitted:
        print(
            "[ISOEXEC-MM-CUBLASLT] no shape passed the self-check -- provider disabled " "(pure Triton fallthrough).",
            flush=True,
        )
        _DISABLED = True
        return False
    _ADMITTED, _CENSUS_SITES, _DROPPED = admitted, sites, dropped

    # The fallthrough is whatever matmul_persistent is right now (stock, or mm_tiles' launcher).
    stock = getattr(bi, "_isoexec_stock_matmul_persistent", bi.matmul_persistent)
    prev = bi.matmul_persistent  # <- the fallthrough (order-independent w.r.t. install_mm_tiles)

    from . import mm_shapes

    admitted_local = _ADMITTED  # bind once: the closure must not re-read a module global per call

    def matmul_persistent_cublaslt(a, b, bias=None):
        # Take cuBLASLt only for the exact production case; everything else keeps `prev`.
        if a.dim() == 2 and b.dim() == 2:
            key = (a.shape[1], b.shape[1])
            take = (
                bias is None
                and a.dtype == torch.bfloat16
                and b.dtype == torch.bfloat16
                and key in admitted_local
                and a.stride(1) == 1  # a row-major [M,K]
                and b.stride() == (1, a.shape[1])  # b == W.t(): [K,N] stride (1, K) production layout
            )
            if take:
                try:
                    M, K = a.shape
                    N = b.shape[1]
                    out = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
                    ac = a if a.is_contiguous() else a.contiguous()
                    ext.mm(ac, b.t(), out)  # w = b.t() = [N,K] K-contiguous
                    return out
                except Exception:  # noqa: BLE001 -- any driver refusal (e.g. split-K pin) -> fallthrough
                    return prev(a, b, bias=bias)
        return prev(a, b, bias=bias)

    matmul_persistent_cublaslt._isoexec_cublaslt = True
    bi._isoexec_stock_matmul_persistent = stock
    bi.matmul_persistent = matmul_persistent_cublaslt
    _WRAPPER = matmul_persistent_cublaslt

    if not _LOG_ONCE:
        _LOG_ONCE = True
        n_legacy = len(_ADMITTED & _LEGACY_CUBLASLT_SHAPES)
        n_derived = len(_ADMITTED) - n_legacy
        print(
            f"[ISOEXEC-MM-CUBLASLT] pinned non-split-K cuBLASLt provider installed for "
            f"{len(_ADMITTED)} bf16 shapes ({n_legacy} legacy/grandfathered + {n_derived} derived "
            f"from the model census): M-buckets probe@512 (decode, M<1024) / probe@8192 (trainer). "
            f"Self-check PASSED per shape (M-invariance + cross-bucket + determinism; derived shapes "
            f"additionally bit-equal to Triton, which is what makes census extension "
            f"bitwise-neutral). {len(_DROPPED)} shape(s) dropped fail-closed. "
            f"Flag SKYRL_ISOEXEC_MM_CUBLASLT=1. The fp32 router and the N=1 gate scalar stay on "
            f"Triton STRUCTURALLY (measured 13-23x worse / no legal algo).",
            flush=True,
        )
        if _CENSUS_SITES:
            print(
                "[ISOEXEC-MM-CUBLASLT] derived shapes: "
                + "  ".join(f"({K},{N}){nm}" for (K, N), nm in sorted(_CENSUS_SITES.items())),
                flush=True,
            )
        mm_shapes.report_install_coverage(
            "MM_CUBLASLT",
            lambda key: (key[0], key[1]) in _ADMITTED and key[2] is torch.bfloat16,
            flag="SKYRL_ISOEXEC_MM_CUBLASLT",
        )
    return True
