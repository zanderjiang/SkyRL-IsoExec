"""The dense-GEMM shape census, and the install-time coverage report built on it.

``mm_cublaslt`` and ``mm_tiles`` are shape tables, so they can install, pass their own self-checks,
and still cover nothing the live model runs -- the self-checks probe the table, not the model. Each
``models/<name>.py`` declares a ``gemm_census(tp, etp)``, ``census_for()`` instantiates it at the
run's actual parallelism, and ``report_install_coverage`` prints the intersection before the first
forward. A census records shapes only; tile constants and provider choices live in the tables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

import torch

# dtype spelling used by the per-model censuses, so a models/*.py never has to import torch.
_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
_DTYPE_NAMES = {v: k for k, v in _DTYPES.items()}


@dataclass(frozen=True)
class GemmSite:
    """One dense 2-D GEMM site, instantiated at a concrete ``(tp, etp)``.

    name:      dotted site id, e.g. ``mla.q_up``; used in the banner so a miss is legible.
    K, N:      the live per-rank shape of the ``mm`` that ``F.linear`` decomposes to, already
               divided by whatever axis shards the site.
    axis:      "tp" | "etp" | "none" -- which parallelism divided this site.
    calls:     how many times EACH TOKEN passes through this shape in one forward, normally the
               number of layers hosting the site. Per-token, not per-launch: a site whose launches
               partition the tokens rather than repeat them counts once.
    table_candidate:
               False for a site deliberately left out of the tables. Still censused, so the banner
               distinguishes "we decided not to" from "nobody looked".
    """

    name: str
    K: int
    N: int
    dtype: torch.dtype
    axis: str = "none"
    calls: int = 1
    table_candidate: bool = True

    @property
    def key(self) -> tuple:
        return (self.K, self.N, self.dtype)

    @property
    def work(self) -> int:
        """``calls * K * N`` -- MACs per forward at M=1, i.e. FLOPs with the batch factored out.

        Used to rank misses, since ranking by call count alone puts the widest (and least frequently
        called) sites last. It is a FLOP proxy, not a time proxy: at decode M these GEMMs are
        occupancy- and launch-bound, so it must never be used to predict a saving.
        """
        return int(self.calls) * int(self.K) * int(self.N)

    def __str__(self) -> str:
        return f"({self.K},{self.N},{_DTYPE_NAMES.get(self.dtype, self.dtype)}) {self.name}"


def dtype_of(name: str) -> torch.dtype:
    """``"bf16"`` -> ``torch.bfloat16``.  Used by the per-model censuses."""
    return _DTYPES[name]


_LIVE: Optional[dict] = None  # explicitly-registered context, when a runtime knows better than env


def set_live_context(*, model_path: str, tp: int, etp: int, side: str = "?") -> None:
    """Register the model/parallelism this process is actually running.

    Runtimes that know these before the providers install should call this; everything else falls
    back to the environment. Last-writer-wins and never raises -- this feeds diagnostics only.
    """
    global _LIVE
    _LIVE = {"model_path": model_path, "tp": int(tp), "etp": int(etp), "side": str(side)}


def _int_env(name: str) -> Optional[int]:
    try:
        v = os.environ.get(name)
        return int(v) if v else None
    except ValueError:
        return None


def live_context() -> Optional[dict]:
    """``{model_path, tp, etp, side}`` for this process, or None if it cannot be determined.

    Order: an explicit ``set_live_context``, then the environment, then the live parallel state of
    whichever runtime is initialised. Every step is best-effort and None is a legitimate answer.
    """
    if _LIVE is not None:
        return dict(_LIVE)

    model_path = os.environ.get("SKYRL_ISOEXEC_MODEL_PATH") or ""
    if not model_path:
        return None

    side, tp, etp = "?", None, None

    # vLLM is asked FIRST because an engine worker runs a megatron GPTModel inside vLLM and so
    # initialises a megatron parallel state too; only vLLM's state discriminates the two sides.
    try:
        from vllm.distributed import parallel_state as vps

        if vps.model_parallel_is_initialized():
            tp = vps.get_tensor_model_parallel_world_size()
            side = "ENGINE"
    except Exception:
        pass

    if tp is None:
        try:
            from megatron.core import parallel_state as mpu

            if mpu.model_parallel_is_initialized():
                tp = mpu.get_tensor_model_parallel_world_size()
                try:
                    etp = mpu.get_expert_tensor_parallel_world_size()
                except Exception:
                    etp = None
                side = "TRAINER"
        except Exception:
            pass

    if side == "ENGINE" or tp is None:
        tp = tp or _int_env("SKYRL_ISOEXEC_ENGINE_TP")
        # At EP=1 every rank tensor-shards every expert, so engine ETP == engine TP.
        if etp is None and _int_env("SKYRL_ISOEXEC_ENGINE_EP") == 1:
            etp = tp
        if side == "?" and tp is not None:
            side = "ENGINE"

    if tp is None:
        return None
    return {"model_path": model_path, "tp": int(tp), "etp": int(etp or tp), "side": side}


def census_for(model_path: str, tp: int, etp: int) -> tuple:
    """The dense forward GEMM census for a model at a concrete parallelism.

    Returns ``()`` when the model resolves but declares no census; callers report that case
    distinctly from a covered model.
    """
    from ...models.resolve import resolve_model_module

    mod, _how = resolve_model_module(model_path)
    fn: Optional[Callable] = getattr(mod, "gemm_census", None)
    if fn is None:
        return ()
    return tuple(fn(tp=int(tp), etp=int(etp)))


def live_census() -> tuple:
    """``(sites, context)`` for this process; ``((), None)`` when nothing can be resolved."""
    ctx = live_context()
    if ctx is None:
        return (), None
    try:
        return census_for(ctx["model_path"], ctx["tp"], ctx["etp"]), ctx
    except Exception:
        return (), ctx


def census_union(model_path: str, tps, etps) -> dict:
    """``{(K, N, dtype): site_name}`` over a cross-product of parallelisms.

    This is how a shape table is populated: lookup is by key, so listing every shape the supported
    TPs can produce costs nothing at a TP that does not produce it. Combinations whose division is
    not exact are dropped, as are sites marked ``table_candidate=False``.
    """
    ref = {s.name: s for s in census_for(model_path, 1, 1)}
    out: dict = {}
    for tp in tps:
        for etp in etps:
            try:
                sites = census_for(model_path, tp, etp)
            except Exception:
                continue
            for s in sites:
                if not s.table_candidate:
                    continue
                base = ref.get(s.name)
                if base is not None:
                    div = tp if s.axis == "tp" else (etp if s.axis == "etp" else 1)
                    if base.K % div or base.N % div:
                        continue  # this parallelism does not divide this site exactly
                out.setdefault(s.key, f"{s.name}@tp{tp}/etp{etp}")
    return out


def report_install_coverage(table_name: str, covered: Callable[[tuple], bool], *, flag: str) -> None:
    """Print which LIVE-MODEL shapes ``table_name`` covers, and which it does not.

    ``covered(key)`` answers membership for a ``(K, N, dtype)`` key.  Never raises.
    """
    try:
        sites, ctx = live_census()
        if ctx is None:
            print(
                f"[ISOEXEC-MM-COVERAGE] {table_name}: could not resolve the live model/parallelism "
                f"(set SKYRL_ISOEXEC_MODEL_PATH in the launcher, or call mm_shapes.set_live_context). "
                f"Coverage will be reported from OBSERVED traffic instead -- see the "
                f"[ISOEXEC-MM-COVERAGE] OBSERVED banner after the first forward.",
                flush=True,
            )
            return
        if not sites:
            print(
                f"[ISOEXEC-MM-COVERAGE] {table_name}: model {ctx['model_path']!r} declares NO "
                f"gemm_census, so install-time coverage is UNKNOWN -- a green self-check on this "
                f"table proves only that the table is internally sound, NOT that it covers this "
                f"model. Add gemm_census() to its models/*.py. Falling back to OBSERVED traffic.",
                flush=True,
            )
            return

        hit = [s for s in sites if covered(s.key)]
        miss = [s for s in sites if not covered(s.key)]
        hit_calls = sum(s.calls for s in hit)
        all_calls = sum(s.calls for s in sites) or 1
        hit_work = sum(s.work for s in hit)
        all_work = sum(s.work for s in sites) or 1
        head = (
            f"[ISOEXEC-MM-COVERAGE] {table_name} vs the LIVE census "
            f"(model={ctx['model_path']} side={ctx['side']} tp={ctx['tp']} etp={ctx['etp']}): "
            f"{len(hit)}/{len(sites)} sites covered, {100.0 * hit_calls / all_calls:.1f}% of "
            f"forward GEMM calls, {100.0 * hit_work / all_work:.1f}% of forward GEMM work "
            f"(calls*K*N; a FLOP proxy, NOT a time proxy -- these GEMMs are occupancy-bound at "
            f"decode M)."
        )
        print(head, flush=True)
        if hit:
            print("[ISOEXEC-MM-COVERAGE]   COVERED  : " + "  ".join(str(s) for s in hit), flush=True)
        if miss:
            # Ranked by work, not calls: the widest site is often the least frequently called.
            miss = sorted(miss, key=lambda s: (-s.work, -s.calls))
            print(
                "[ISOEXEC-MM-COVERAGE]   UNCOVERED: "
                + "  ".join(f"{s} x{s.calls} work={100.0 * s.work / all_work:.1f}%" for s in miss)
                + "  <- these run vLLM's Triton matmul_persistent (2.6-4x slower than cuBLAS/GEMM),"
                + "  ranked by WORK share.",
                flush=True,
            )
        if not hit:
            print(
                f"[ISOEXEC-MM-COVERAGE]   *** {table_name} covers ZERO live shapes on this model. "
                f"{flag} is ON and is doing NOTHING. This is the silent-green failure mode: the "
                f"table's own self-check passes because it probes the TABLE, not the MODEL. Either "
                f"populate the census shapes above (and measure them) or turn {flag} off so the log "
                f"stops claiming a win that is not happening. ***",
                flush=True,
            )
    except Exception as e:  # noqa: BLE001 -- diagnostics must never take down a run
        print(f"[ISOEXEC-MM-COVERAGE] {table_name}: coverage report failed ({type(e).__name__}: {e})", flush=True)
