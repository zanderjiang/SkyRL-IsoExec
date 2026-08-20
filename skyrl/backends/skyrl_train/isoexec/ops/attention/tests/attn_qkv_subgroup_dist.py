"""Bitwise + admission + perf gate for the QKV subgroup all-gather (OPEN_WORK 2.4b item c).

WHAT IS BEING PROVEN. ``ops/attention/qkv_subgroup_gather.py`` replaces megatron's fused-QKV
all-gather over the WHOLE tensor-parallel group with an all-gather over the ``num_query_groups``
subgroup whose shards are the only columns the rank keeps. The claim is not "close", it is "the same
bytes in the same order", and it rests on two facts about code, not about arithmetic:

  * ``ColumnParallelLinear`` shards the output dim into contiguous rank-ordered chunks, so rank r
    owns full columns ``[r*L, (r+1)*L)``;
  * megatron's ``_gather_along_last_dim`` (``tensor_parallel/mappings.py:80-96``) lays the shards
    down in RANK ORDER, so ``out[..., r*L + j] == rank r's in[..., j]``.

Hence ``mixed_qkv[..., idx*size : (idx+1)*size]`` with ``size = full // ng == (world//ng) * L`` is
exactly ranks ``[idx*rpg, (idx+1)*rpg)``'s shards, concatenated in rank order -- which is what an
all-gather over that subgroup produces. The REFERENCE below is megatron's own function, imported,
not a re-derivation of it, so a megatron that changes the convention breaks this gate instead of
sliding past it.

ROWS
  1  admission predicates (pure, no mesh): the refusal cases must refuse and say why, the live
     geometry must admit, and the "kept window is a whole number of rank shards" invariant must
     hold across a brute-force sweep of admissible geometries
  2  bitwise vs megatron's full gather + slice, ``torch.equal`` on the uint8 view, at
     B in {1, 8, 64, 256, 1024} -- and independently against the un-sharded source tensor
  3  positive controls (a gate that cannot fail is not a gate): the wrong subgroup index and a
     reversed rank order must both DIFFER
  4  CUDA-graph capture + replay: the subgroup gather runs INSIDE vLLM's decode graphs, so it must
     track a CHANGED input across replays; a frozen capture would pass every static check above and
     be silently wrong every decode step after the first
  5  perf: us/call, subgroup vs full, at the live Qwen3.5-35B-A3B decode geometry, plus what the
     new communicator costs in MiB/rank. This row is a DECISION input, not decoration: it is what
     refuted the design note's proposed ``max_ctas=4`` (at four channels the subgroup gather is
     SLOWER than the 8-rank gather it replaces from B=256 up) and it is what says there is no win
     at all below B~64. See the ladder in the op's docstring.

LIVE GEOMETRY (Qwen3.5-35B-A3B, engine TP=8): head_dim 256, 16 q heads, 2 kv heads,
attn_output_gate -> fused qkv width 256*(2*16 + 2*2) = 9216; shard L = 1152 (four and a HALF heads,
which is why megatron's branch exists); num_query_groups 2 -> kept window 4608 = 18 heads.

LAST RUN 2026-08-12, 8xH100, ALL PASS (rows 1-4 green; row 5 below). Communicator cost 326
MiB/rank at max_ctas=16. Graph-replay us/call, median of 3 -- full vs subgroup:
B=1 21.7/17.0, B=64 25.8/22.5, B=256 38.7/38.9, B=512 56.5/56.9, B=1024 90.4/74.9,
B=2048 158.7/116.4. Back-to-back replays PIPELINE, so this row reads as throughput and
understates the latency win; the isolated-collective ladder in the op's docstring -- taken against
the launcher's actual `NCCL_MAX_NCHANNELS=8` baseline -- is the number to size the arm with
(71.1 -> 58.9 us at B=256, 151.8 -> 76.6 at B=1024).

RUN (needs all 8 GPUs, ~4 min; check `nvidia-smi --query-compute-apps=pid --format=csv` first):
    /mnt/local_storage/venvs/skyrl-isoexec/bin/torchrun \
        --nproc_per_node=8 --master_port=29573 \
        skyrl/backends/skyrl_train/isoexec/ops/attention/tests/attn_qkv_subgroup_dist.py \
        > /mnt/local_storage/logs/qkv_subgroup/attn_qkv_subgroup_test.log 2>&1
"""

import os
import sys

import torch
import torch.distributed as dist

if int(os.environ.get("WORLD_SIZE", "1")) < 2 or not torch.cuda.is_available():
    # multi-rank battery: launch under torchrun with world >= 2 (CI); skip gracefully elsewhere
    print("SKIP: needs torchrun with WORLD_SIZE >= 2 and CUDA devices")
    raise SystemExit(0)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 7)))  # repo root

from megatron.core.tensor_parallel.mappings import _gather_along_last_dim  # noqa: E402

from skyrl.backends.skyrl_train.isoexec.ops.attention.qkv_subgroup_gather import (  # noqa: E402
    SUBGROUP_MAX_CTAS,
    SubgroupPlan,
    admission_refusal,
    build_qkv_subgroups,
    subgroup_gather_last_dim,
    warm_subgroup,
)

# The live engine geometry.
FULL_WIDTH = 9216
NUM_QUERY_GROUPS = 2
BATCHES = (1, 8, 64, 256, 1024)

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    if not ok:
        FAILS += 1
    if dist.get_rank() == 0:
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def bits_equal(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def megatron_full_then_slice(local, group, world, ngroups, idx):
    """Exactly what megatron does today (attention.py:1580-1588), using megatron's own function."""
    full = _gather_along_last_dim(local, group)
    size = full.size(-1) // ngroups
    return full[..., idx * size : (idx + 1) * size]


# ------------------------------------------------------------------------------------------------
# Row 1 -- admission predicates. Pure: this row would run on a laptop.
# ------------------------------------------------------------------------------------------------


def row1_admission():
    cases = [
        # (kwargs, must_refuse, label)
        (dict(world=8, num_query_groups=2, local_width=1152), False, "live geometry admits"),
        (dict(world=8, num_query_groups=8), True, "ng == world: megatron takes no gather at all"),
        (dict(world=8, num_query_groups=16), True, "ng > world"),
        (dict(world=8, num_query_groups=1), True, "ng == 1: subgroup would be the whole tp group"),
        (dict(world=8, num_query_groups=3), True, "ng does not divide world"),
        (dict(world=8, num_query_groups=None), True, "num_query_groups is None"),
        (dict(world=1, num_query_groups=1), True, "world 1: nothing to gather"),
        (dict(world=4, num_query_groups=2, local_width=1152), False, "trainer TP=4 geometry admits"),
    ]
    for kw, must_refuse, label in cases:
        r = admission_refusal(**kw)
        ok = (r is not None) if must_refuse else (r is None)
        check(f"row1 {label}", ok, f"-> {r!r}")

    # The load-bearing invariant, brute-forced over every admissible geometry in range: the kept
    # window is always a WHOLE number of rank shards, so the slice bounds can never cut a shard.
    bad = []
    for world in range(2, 33):
        for ng in range(1, world + 1):
            for local in (1, 7, 64, 1152, 4097):
                if admission_refusal(world=world, num_query_groups=ng, local_width=local) is not None:
                    continue
                size = (local * world) // ng
                if size != (world // ng) * local:
                    bad.append((world, ng, local))
    check("row1 kept window is a whole number of rank shards (brute force)", not bad, str(bad[:4]))


# ------------------------------------------------------------------------------------------------
# Rows 2-5 -- live, 8 ranks.
# ------------------------------------------------------------------------------------------------


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.cuda.current_device()

    if rank == 0:
        print(
            f"world={world} full_width={FULL_WIDTH} num_query_groups={NUM_QUERY_GROUPS} "
            f"shard={FULL_WIDTH // world}",
            flush=True,
        )

    row1_admission()

    if world % NUM_QUERY_GROUPS or FULL_WIDTH % world:
        check("world/width divisible", False, f"world={world}")
        dist.destroy_process_group()
        return

    group = dist.group.WORLD
    rpg = world // NUM_QUERY_GROUPS
    local_w = FULL_WIDTH // world
    idx = rank // rpg
    size = FULL_WIDTH // NUM_QUERY_GROUPS

    # The communicator is NOT free -- report what it costs, because that is half the ship/no-ship
    # decision (this project has refused a channel widening on exactly this trade before).
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    dist.barrier()
    free_before = torch.cuda.mem_get_info()[0] / 2**20
    sub_pg = build_qkv_subgroups(group, NUM_QUERY_GROUPS)
    warm_subgroup(sub_pg)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    dist.barrier()
    comm_mib = free_before - torch.cuda.mem_get_info()[0] / 2**20
    if rank == 0:
        print(
            f"subgroup communicator cost: {comm_mib:.0f} MiB/rank at max_ctas={SUBGROUP_MAX_CTAS}",
            flush=True,
        )
    sub_ranks = tuple(dist.get_process_group_ranks(sub_pg))
    check(
        "row1 subgroup membership == the slice's shard range",
        sub_ranks == tuple(range(idx * rpg, (idx + 1) * rpg)),
        f"rank {rank}: subgroup {idx} = {sub_ranks}",
    )

    plan = SubgroupPlan(
        world=world,
        num_query_groups=NUM_QUERY_GROUPS,
        ranks_per_group=rpg,
        group_index=idx,
        tp_group=group,
        sub_pg=sub_pg,
        tp_ranks=tuple(range(world)),
        sub_ranks=sub_ranks,
    )

    # ---- Row 2: bitwise, over the batch sweep -------------------------------------------------
    for B in BATCHES:
        # The SAME source on every rank, then sharded column-parallel style: rank r owns
        # [r*L, (r+1)*L). This is the layout ColumnParallelLinear produces.
        g = torch.Generator(device="cuda").manual_seed(20260812 + B)
        src = torch.randn(1, B, FULL_WIDTH, generator=g, device=dev, dtype=torch.bfloat16)
        local = src[..., rank * local_w : (rank + 1) * local_w].contiguous()

        ref = megatron_full_then_slice(local, group, world, NUM_QUERY_GROUPS, idx)
        got_full = subgroup_gather_last_dim(local, plan)
        got = got_full[..., idx * size : (idx + 1) * size]

        check(f"row2 B={B} bitwise vs megatron gather+slice", bits_equal(ref, got))
        check(
            f"row2 B={B} bitwise vs the un-sharded source window",
            bits_equal(src[..., idx * size : (idx + 1) * size], got),
        )
        check(
            f"row2 B={B} shape/stride identical to the stock path",
            got.shape == ref.shape and got.stride() == ref.stride(),
            f"{tuple(got.shape)} {got.stride()}",
        )

    # ---- Row 3: positive controls -------------------------------------------------------------
    g = torch.Generator(device="cuda").manual_seed(7)
    src = torch.randn(1, 64, FULL_WIDTH, generator=g, device=dev, dtype=torch.bfloat16)
    local = src[..., rank * local_w : (rank + 1) * local_w].contiguous()
    ref = megatron_full_then_slice(local, group, world, NUM_QUERY_GROUPS, idx)

    wrong_idx = SubgroupPlan(**{**plan.__dict__, "group_index": (idx + 1) % NUM_QUERY_GROUPS})
    wrong = subgroup_gather_last_dim(local, wrong_idx)[..., idx * size : (idx + 1) * size]
    check("row3 control: wrong subgroup index must DIFFER", not bits_equal(ref, wrong))

    rev_members = list(reversed(range(idx * rpg, (idx + 1) * rpg)))
    # sort_ranks=False is load-bearing: torch.distributed.new_group SORTS its rank list by default,
    # so without it this "reversed" group is bit-for-bit the forward group and the control silently
    # cannot fail. (It did, the first time this harness ran.)
    all_rev = [
        dist.new_group(ranks=list(reversed(range(g0 * rpg, (g0 + 1) * rpg))), sort_ranks=False)
        for g0 in range(NUM_QUERY_GROUPS)
    ]
    rev_plan = SubgroupPlan(**{**plan.__dict__, "sub_pg": all_rev[idx]})
    rev = subgroup_gather_last_dim(local, rev_plan)[..., idx * size : (idx + 1) * size]
    check(
        "row3 control: reversed rank order must DIFFER",
        not bits_equal(ref, rev),
        f"members={rev_members}",
    )

    # ---- Row 4: CUDA graph capture + replay ---------------------------------------------------
    try:
        static_in = local.clone()
        # Warm on a side stream first -- capture of an uninitialised allocator/comm is the hazard
        # this whole design is about.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                subgroup_gather_last_dim(static_in, plan)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = subgroup_gather_last_dim(static_in, plan)

        ok_replays = True
        for seed in (11, 12, 13):
            g2 = torch.Generator(device="cuda").manual_seed(seed)
            new_src = torch.randn(1, 64, FULL_WIDTH, generator=g2, device=dev, dtype=torch.bfloat16)
            static_in.copy_(new_src[..., rank * local_w : (rank + 1) * local_w])
            graph.replay()
            torch.cuda.synchronize()
            want = new_src[..., idx * size : (idx + 1) * size]
            ok_replays &= bits_equal(want, static_out[..., idx * size : (idx + 1) * size])
        check("row4 cudagraph replay tracks a CHANGED input", ok_replays)
        # Tear the graph down together with everything its capture pool points at, before row 5
        # captures more. A live output tensor pins the pool; a dead input un-pins storage the
        # graph still references.
        del graph, static_out, static_in
    except Exception as e:
        check("row4 cudagraph capture + replay", False, f"{type(e).__name__}: {e}")

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    dist.barrier()

    # ---- Row 5: perf, INSIDE A CUDA GRAPH ------------------------------------------------------
    # This is measured as a graph replay on purpose. An eager Python loop over either path is
    # CPU-LAUNCH-BOUND at these payloads -- ~137 us/call of dispatch and allocator against a ~50 us
    # collective -- so it measures Python, and the subgroup path (one extra empty + one extra copy)
    # loses in it by construction while the transport it replaces is twice as fast. Live, this
    # gather runs inside vLLM's decode graphs, where launch cost is zero and the collective is the
    # whole cost. Timing replays is the honest analogue AND the configuration production runs.
    # Reps are interleaved with the median taken across them: run-to-run skew on this node is worth
    # tens of microseconds, so a single back-to-back A-then-B comparison is not a result.
    def capture(fn, x):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(5):
                fn(x)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        gph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gph):
            fn(x)
        torch.cuda.synchronize()
        return gph

    def bench_graph(gph, iters=300):
        for _ in range(30):
            gph.replay()
        torch.cuda.synchronize()
        dist.barrier()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            gph.replay()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters * 1e3

    # ONE B at a time, and the static input is held alive for exactly as long as its graphs are.
    # Doing otherwise is not a style preference: capturing all six B's first lets each iteration's
    # `x` fall out of scope while a captured graph still points at its storage, the allocator hands
    # that block to the next iteration, and the replays segfault in cuGraphLaunch. (They did.)
    PERF_BS = (1, 64, 256, 512, 1024, 2048)
    perf = {}
    for B in PERF_BS:
        g3 = torch.Generator(device="cuda").manual_seed(B)
        src = torch.randn(1, B, FULL_WIDTH, generator=g3, device=dev, dtype=torch.bfloat16)
        x = src[..., rank * local_w : (rank + 1) * local_w].contiguous()
        try:
            gs = {
                "full": capture(lambda t: megatron_full_then_slice(t, group, world, NUM_QUERY_GROUPS, idx), x),
                "sub": capture(lambda t: subgroup_gather_last_dim(t, plan), x),
            }
            for _rep in range(3):
                for tag in ("full", "sub"):
                    perf.setdefault((tag, B), []).append(bench_graph(gs[tag]))
                    dist.barrier()
        except Exception as e:
            check(f"row5 B={B} capture/replay for timing", False, f"{type(e).__name__}: {e}")
        finally:
            gs = None
            del src, x
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            dist.barrier()

    if rank == 0:
        print(
            f"\n{'B':>6} {'full AG+slice':>15} {'subgroup AG':>13} {'speedup':>9}   "
            f"(us per graph replay, median of 3; max_ctas={SUBGROUP_MAX_CTAS})",
            flush=True,
        )
        for B in PERF_BS:
            if ("full", B) not in perf or ("sub", B) not in perf:
                continue
            a = sorted(perf[("full", B)])[1]
            b = sorted(perf[("sub", B)])[1]
            print(f"{B:>6} {a:>15.1f} {b:>13.1f} {a / b:>8.2f}x", flush=True)
        print(
            "NOTE: below B~64 both paths sit on the same NCCL latency floor and there is no "
            "win; the live 73 us decode measurement sits at production payloads, where there "
            "is. The subgroup path also allocates one full-width buffer per captured shape "
            "that it never fully writes -- graph-pool memory, counted against the prize.",
            flush=True,
        )

    dist.barrier()
    if rank == 0:
        print(f"\n{'ALL PASS' if FAILS == 0 else str(FAILS) + ' FAILED'}", flush=True)
    dist.destroy_process_group()
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
