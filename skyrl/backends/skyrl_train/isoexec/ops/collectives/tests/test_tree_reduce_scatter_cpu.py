"""CPU-side guarantees for ``pik.allreduce.tree_reduce_scatter`` (trainer sequence parallelism).

The GPU halves -- real payload classes, the triton tree kernel, and the full SP layer forward --
live in the private repo's nightly ``trainer_sp_battery.py``. What CAN be pinned without a GPU is
the TRANSPORT/SLICING ALGEBRA, which is exactly the part Megatron's convention constrains:

  1. RS == AR-then-slice, bitwise, on BOTH all-reduce transports (one-shot gather and two-shot
     all_to_all), at payloads on each side of ONESHOT_MAX_BYTES. tree_reduce_scatter uses the
     two-shot structure UNCONDITIONALLY -- size must never change the expression tree -- so
     this is the check that the one-shot path a small SP-off payload takes is still the same
     expression the RS evaluates.
  2. The slice is Megatron's sequence-parallel convention: FIRST dim, equal chunks, RANK order
     (``_reduce_scatter_along_first_dim`` / ``_split_along_first_dim``).
  3. Divisibility is refused, not mis-sliced: first dim not divisible by world -> AssertionError.
  4. bf16-leaf wire semantics mirror tree_all_reduce: bf16 partial -> fp32 adds -> bf16 root.

The triton ``_tree_reduce`` kernel cannot run on CPU; it is stubbed with a torch reference that
evaluates the same balanced tree (pik.plan.combine_order over the stacked rank axis) in BOTH the
AR and RS paths, so what this file proves is the algebra AROUND the tree, on real gloo
collectives (gloo's all_to_all_single/all_gather move bytes exactly like NCCL's: rank order).

Run (CPU only):
    uv run --isolated --extra dev python -m pytest \
        skyrl/backends/skyrl_train/isoexec/ops/collectives/tests/test_tree_reduce_scatter_cpu.py -q
"""

from __future__ import annotations

import importlib
import os
import pathlib
import sys
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[7]))  # repo root

_WORLD = 4
_N_COLS = 8


def _load_pik_cpu():
    """Import the vendored pik's allreduce+plan WITHOUT importing the pik package (pik.gemm's
    triton autotune decorators query the GPU at import). A stub package whose __path__ points at
    the real directory lets the relative `.codegen` import resolve; codegen generates lazily."""
    if "_pik_cpu.allreduce" in sys.modules:
        return sys.modules["_pik_cpu.allreduce"], sys.modules["_pik_cpu.plan"]
    pikdir = _HERE.parents[1] / "pik"
    pkg = types.ModuleType("_pik_cpu")
    pkg.__path__ = [str(pikdir)]
    sys.modules["_pik_cpu"] = pkg
    ar = importlib.import_module("_pik_cpu.allreduce")
    plan = importlib.import_module("_pik_cpu.plan")
    return ar, plan


def _reference_tree(stacked: torch.Tensor, combine_order) -> torch.Tensor:
    """The balanced tree over the stacked rank axis, elementwise -- the CPU twin of the triton
    tree_reduce_kernel (same combine_order, fp32 adds)."""
    c = stacked.shape[0]
    if c == 1:
        return stacked[0].clone()
    slots = [stacked[i].clone() for i in range(c)]
    for dst, lhs, rhs in combine_order(c):
        slots[dst] = slots[lhs] + slots[rhs]
    return slots[0]


def _adversarial(shape, seed) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * torch.exp(torch.randn(*shape, generator=g) * 12)).float()


def _worker(rank: int, results):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29537")
    dist.init_process_group("gloo", rank=rank, world_size=_WORLD)
    try:
        ar, plan = _load_pik_cpu()

        # Stub the triton kernel with the torch twin -- in BOTH paths, so AR vs RS compares
        # transports and slicing, with identical tree arithmetic.
        def _tree_reduce_cpu(stacked, out):
            out.copy_(_reference_tree(stacked, plan.combine_order))
            return out

        ar._tree_reduce = _tree_reduce_cpu

        # gloo insists output chunks have the INPUT's shape; NCCL (production) treats the output
        # as flat. Shim to the flat view so pik's one-shot [world, M, N] gather buffer works here.
        _orig_ag = dist.all_gather_into_tensor

        def _ag_flat(out, inp, group=None, **kw):
            world = dist.get_world_size(group)
            inp = inp.contiguous()
            return _orig_ag(out.view(world * inp.shape[0], *inp.shape[1:]), inp, group=group, **kw)

        ar.dist.all_gather_into_tensor = _ag_flat

        failures = []

        def check(name, cond):
            if not cond:
                failures.append(name)

        for rows, oneshot_mb, seed in (
            (8, 1 << 30, 0),  # tiny payload, AR takes ONE-SHOT (forced huge crossover)
            (8, 0, 1),  # tiny payload, AR forced TWO-SHOT
            (64, 1 << 30, 2),
            (64, 0, 3),
        ):
            ar.ONESHOT_MAX_BYTES = oneshot_mb
            # every rank builds every rank's partial deterministically; keeps its own
            partials = [_adversarial((rows, _N_COLS), 1000 * seed + r) for r in range(_WORLD)]
            mine = partials[rank].clone()

            full = ar.tree_all_reduce(mine.clone(), group=None, backend="nccl")
            sliced = ar.tree_reduce_scatter(mine.clone(), group=None)

            # (2) Megatron convention: first dim, equal chunks, rank order
            lo = rank * (rows // _WORLD)
            hi = lo + rows // _WORLD
            check(
                f"rs==ar_slice rows={rows} oneshot={oneshot_mb} ",
                torch.equal(sliced, full[lo:hi]),
            )

            # (1) all-gather of the RS slices reproduces AR bitwise
            gathered = torch.empty_like(full)
            dist.all_gather_into_tensor(gathered, sliced.contiguous())
            check(f"ag(rs)==ar rows={rows} oneshot={oneshot_mb}", torch.equal(gathered, full))

            # cross-check against the single-process reference tree
            ref = _reference_tree(torch.stack(partials), plan.combine_order)
            check(f"ar==ref rows={rows} oneshot={oneshot_mb}", torch.equal(full, ref))

        # (4) bf16 leaf wire: fp32 adds, bf16 root, equal to AR's bf16 root sliced
        parts_bf = [_adversarial((8, _N_COLS), 77 + r).to(torch.bfloat16) for r in range(_WORLD)]
        full_bf = ar.tree_all_reduce(parts_bf[rank].clone(), group=None, backend="nccl")
        sliced_bf = ar.tree_reduce_scatter(parts_bf[rank].clone(), group=None)
        check("bf16 dtype", sliced_bf.dtype == torch.bfloat16)
        check("bf16 rs==ar_slice", torch.equal(sliced_bf, full_bf[2 * rank : 2 * rank + 2]))
        # decoupled root: bf16 wire + fp32 root
        full_f = ar.tree_all_reduce(parts_bf[rank].clone(), group=None, backend="nccl", root_dtype=torch.float32)
        sliced_f = ar.tree_reduce_scatter(parts_bf[rank].clone(), group=None, root_dtype=torch.float32)
        check("bf16->fp32 root dtype", sliced_f.dtype == torch.float32)
        check("bf16->fp32 rs==ar_slice", torch.equal(sliced_f, full_f[2 * rank : 2 * rank + 2]))

        # (3) refusal: first dim 6 not divisible by world 4
        try:
            ar.tree_reduce_scatter(torch.randn(6, _N_COLS), group=None)
            failures.append("divisibility not refused")
        except AssertionError:
            pass

        results[rank] = failures
    finally:
        dist.destroy_process_group()


def test_tree_reduce_scatter_algebra_gloo():
    ctx = mp.get_context("spawn")
    with ctx.Manager() as mgr:
        results = mgr.dict()
        procs = [ctx.Process(target=_worker, args=(r, results)) for r in range(_WORLD)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(180)
        for r in range(_WORLD):
            assert not procs[r].exitcode, f"rank {r} exited {procs[r].exitcode}"
            assert r in results, f"rank {r} produced no result"
            assert results[r] == [], f"rank {r} failures: {results[r]}"


if __name__ == "__main__":
    test_tree_reduce_scatter_algebra_gloo()
    print("OK")
