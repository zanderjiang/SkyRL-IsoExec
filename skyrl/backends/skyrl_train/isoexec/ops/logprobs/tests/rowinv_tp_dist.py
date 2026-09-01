"""Bitwise TP-invariance gate for ``ops/logprobs/rowinv.py``.

Evaluates the candidate at TP degrees {1, 2, 4, 8} over one seeded full ``[rows, V]`` tensor and
asserts bit equality across degrees and across ranks, with the rank-partial anti-pattern as a live
negative control.

CI: torchrun --standalone --nproc-per-node=8 <thisfile>   (>= 2 ranks; degrees are capped at world)
"""

import os

os.environ["SKYRL_ISOEXEC"] = "1"
os.environ.setdefault("SKYRL_ISOEXEC_PIK_LEAVES", "8")

import torch

if int(os.environ.get("WORLD_SIZE", "1")) < 2 or not torch.cuda.is_available():
    # multi-rank battery: launch under torchrun with world >= 2 (CI); skip gracefully elsewhere
    print("SKIP: needs torchrun with WORLD_SIZE >= 2 and CUDA devices")
    raise SystemExit(0)

import torch.distributed as dist

from skyrl.backends.skyrl_train.isoexec.ops.logprobs import rowinv

ROWS = 256
V = 248320


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32)


def aten_incumbent_full(full: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    m = torch.amax(full, dim=-1, keepdim=True)
    s = torch.gather(full, -1, target.unsqueeze(-1)).squeeze(-1)
    w = (full - m).exp_()
    return (s - m.squeeze(-1)) - w.sum(-1).log()


def make_reference(shard, target, group, world):
    """The incumbent for this degree: gather the full fp32 row, then the ATen formulation."""

    def reference() -> torch.Tensor:
        if world == 1:
            return aten_incumbent_full(shard.clone(), target)
        parts = [torch.empty_like(shard) for _ in range(world)]
        dist.all_gather(parts, shard, group=group)
        return aten_incumbent_full(torch.cat(parts, dim=1), target)

    return reference


def rank_partial_antipattern(shard, group, world) -> torch.Tensor:
    """The refuted scheme (negative control only): one exp-sum partial per rank, summed in order.

    Its rounding points sit at rank boundaries, which move with the TP degree.
    """
    row_max = torch.amax(shard, dim=-1)
    if world > 1:
        dist.all_reduce(row_max, op=dist.ReduceOp.MAX, group=group)
    partial = torch.exp(shard - row_max.unsqueeze(1)).sum(dim=-1)  # one partial per whole shard
    if world > 1:
        parts = [torch.empty_like(partial) for _ in range(world)]
        dist.all_gather(parts, partial, group=group)
    else:
        parts = [partial]
    acc = parts[0]
    for p in parts[1:]:  # sequential rank-order combine: the moving-boundary rounding
        acc = acc + p
    return row_max + torch.log(acc)


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    g_leaves = int(os.environ["SKYRL_ISOEXEC_PIK_LEAVES"])
    degrees = [c for c in (1, 2, 4, 8) if c <= world and g_leaves % c == 0 and V % c == 0]

    # identical full logits on every rank: CUDA Philox is seed-, not device-index-, dependent
    gen = torch.Generator(device=device).manual_seed(20260825)
    full = (torch.randn(ROWS, V, device=device, dtype=torch.float32, generator=gen) * 3).to(torch.bfloat16).float()
    tgen = torch.Generator(device=device).manual_seed(1729)
    target = torch.randint(0, V, (ROWS,), device=device, generator=tgen)
    leaf = V // g_leaves
    adversarial = [0, V - 1, leaf - 1, leaf, V // 2 - 1, V // 2]
    for c in degrees:
        if c > 1:
            adversarial.extend((V // c - 1, V // c))
    target[: len(adversarial)] = torch.tensor(adversarial, device=device, dtype=target.dtype)

    groups = {}
    results = {}  # rank0 only: degree -> output bits
    for degree in degrees:
        group = dist.new_group(list(range(degree))) if degree > 1 else None  # collective: all ranks call
        groups[degree] = group
        if rank < degree:
            shard_vocab = V // degree
            start = rank * shard_vocab
            shard = full[:, start : start + shard_vocab].contiguous()
            out = rowinv.rowinv_sampled_logprobs(
                shard,
                target,
                vocab_start_index=start,
                vocab_end_index=start + shard_vocab,
                group=group,
                src_dtype=torch.bfloat16,
                reference=make_reference(shard, target, group, degree),
            )
            if out is None:
                raise AssertionError(
                    f"rank {rank}: candidate declined at TP={degree}: {rowinv.stats()['decline_reason']}"
                )
            if degree > 1:  # every rank of the subgroup must hold identical bits
                out_bits = bits(out)
                gathered = [torch.empty_like(out_bits) for _ in range(degree)]
                dist.all_gather(gathered, out_bits, group=group)
                if not all(torch.equal(out_bits, peer) for peer in gathered):
                    raise AssertionError(f"rank {rank}: TP={degree} ranks disagree among themselves")
            if rank == 0:
                results[degree] = bits(out).clone()
        dist.barrier()

    # ---- bf16 shards (the trainer's native dtype) reproduce the fp32-widened bits exactly ----
    # Deliberately the SAME subgroup the fp32 shards were admitted on: the payload dtype is
    # per-call eligibility, not group structure, so it must serve a bf16 call too.
    bf16_degree = degrees[-1]
    bf16_group = groups[bf16_degree]
    bf16_bits = None
    if rank < bf16_degree:
        shard_vocab = V // bf16_degree
        start = rank * shard_vocab
        shard16 = full[:, start : start + shard_vocab].to(torch.bfloat16).contiguous()  # exact narrowing

        def reference16(shard16=shard16, group=bf16_group, world=bf16_degree) -> torch.Tensor:
            if world == 1:
                return aten_incumbent_full(shard16.float(), target)
            parts = [torch.empty_like(shard16) for _ in range(world)]
            dist.all_gather(parts, shard16, group=group)
            return aten_incumbent_full(torch.cat(parts, dim=1).float(), target)

        out16 = rowinv.rowinv_sampled_logprobs(
            shard16,
            target,
            vocab_start_index=start,
            vocab_end_index=start + shard_vocab,
            group=bf16_group,
            src_dtype=torch.bfloat16,
            reference=reference16,
        )
        if out16 is None:
            raise AssertionError(f"rank {rank}: bf16-shard call declined: {rowinv.stats()['decline_reason']}")
        if rank == 0:
            bf16_bits = bits(out16).clone()
    dist.barrier()
    if rank == 0 and not torch.equal(bf16_bits, results[bf16_degree]):
        raise AssertionError(
            f"bf16 shards at TP={bf16_degree} are not bit-identical to fp32 shards: "
            f"{int((bf16_bits != results[bf16_degree]).sum().item())} differing tokens"
        )

    served = rowinv.stats()["served"]
    if served < 1:
        raise AssertionError(f"rank {rank}: served={served}, the candidate never engaged")

    # ---- the invariance verdict, plus divergence from the incumbent for context (rank 0) -------
    if rank == 0:
        ref_degree = degrees[0]
        mismatch = {d: int((results[d] != results[ref_degree]).sum().item()) for d in degrees if d != ref_degree}
        if any(mismatch.values()):
            raise AssertionError(f"TP-invariance FAILED: differing tokens vs TP={ref_degree}: {mismatch}")
        incumbent = aten_incumbent_full(full.clone(), target)
        vs_aten = int((results[ref_degree] != bits(incumbent)).sum().item())
    dist.barrier()

    # ---- negative control: rank-partial rounding must NOT be TP-invariant ----------------------
    control_degrees = [d for d in degrees if d > 1][-2:]
    control_diff = None
    if len(control_degrees) == 2:
        control = {}
        for degree in control_degrees:
            if rank < degree:
                shard_vocab = V // degree
                shard = full[:, rank * shard_vocab : (rank + 1) * shard_vocab].contiguous()
                lse = rank_partial_antipattern(shard, groups[degree], degree)
                if rank == 0:
                    control[degree] = bits(lse).clone()
            dist.barrier()
        if rank == 0:
            a, b = control_degrees
            control_diff = int((control[a] != control[b]).sum().item())
            if control_diff == 0:
                raise AssertionError(
                    f"rank-partial control came out identical at TP={a} vs TP={b} -- the negative "
                    "control is vacuous, tighten the data"
                )

    if rank == 0:
        print(
            "ROWINV_TP "
            f"world={world} rows={ROWS} vocab={V} G={g_leaves} degrees={degrees} "
            f"bit_equal_across_tp=True vs_aten_differing_tokens={vs_aten} "
            f"rank_partial_control_degrees={control_degrees} rank_partial_control_diff_rows={control_diff} "
            f"bf16_shards_bit_equal_at_tp{bf16_degree}=True "
            f"served={served} probes={rowinv.stats()['probes']}",
            flush=True,
        )

    for group in list(groups.values()) + [bf16_group]:
        if group is not None:
            rowinv.release_group(group)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
