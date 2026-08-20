"""Multi-rank, no-model harness for the distributed sampled-logprob seam.

Gates the ``logprobs.log_softmax`` impls ``aten_reference`` vs ``aten_reference_fused_exp``
(bitwise, torch.equal on the fp32 words) and the ``logprobs.lm_head_slice:sampled_rows`` extract
path, over the live TP gather wire. Promoted from the private repo's nightly logprob_extract_4rank.py.
CI: torchrun --standalone --nproc-per-node=4 <thisfile>.
"""

import argparse
import os
import statistics

import torch
import torch.distributed as dist

if int(os.environ.get("WORLD_SIZE", "1")) < 2 or not torch.cuda.is_available():
    # multi-rank battery: launch under torchrun with world >= 2 (CI); skip gracefully elsewhere
    print("SKIP: needs torchrun with WORLD_SIZE >= 2 and CUDA devices")
    raise SystemExit(0)
import triton
import triton.language as tl
from triton.language.extra import libdevice

from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
    ChunkedDistributedLogprob,
    _ix_gather_full_vocab,
)
from skyrl.backends.skyrl_train.isoexec.ops.logprobs import exact_sampled as exact


@triton.jit
def _fused_exp_inplace_kernel(full_ptr, row_max_ptr, stride, vocab, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < vocab
    value = tl.load(full_ptr + row * stride + offsets, mask=mask, other=0.0).to(tl.float32)
    row_max = tl.load(row_max_ptr + row)
    value = libdevice.exp(value - row_max)
    tl.store(full_ptr + row * stride + offsets, value, mask=mask)


@torch.no_grad()
def _candidate_chunk(
    logits: torch.Tensor,
    target: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    world = dist.get_world_size(group)
    full = _ix_gather_full_vocab(logits.float().contiguous(), group=group, world=world, src_dtype=logits.dtype)
    valid_target = (target >= 0) & (target < full.shape[-1])
    safe_target = target.masked_fill(~valid_target, 0)
    selected = torch.gather(full, -1, safe_target.unsqueeze(-1)).squeeze(-1)
    row_max = torch.amax(full, dim=-1, keepdim=True)
    flat = full.reshape(-1, full.shape[-1])
    grid = (flat.shape[0], triton.cdiv(flat.shape[1], 2048))
    _fused_exp_inplace_kernel[grid](flat, row_max, flat.stride(0), flat.shape[1], BLOCK=2048)
    lse = full.sum(-1, keepdim=True).float().log()
    result = (selected - row_max.squeeze(-1)) - lse.squeeze(-1)
    result.masked_fill_(~valid_target, 0.0)
    return result


@torch.no_grad()
def _candidate(
    logits: torch.Tensor,
    target: torch.Tensor,
    group: dist.ProcessGroup,
    chunk_size: int,
) -> torch.Tensor:
    chunks = []
    for start in range(0, logits.shape[1], chunk_size):
        end = min(logits.shape[1], start + chunk_size)
        chunks.append(_candidate_chunk(logits[:, start:end], target[:, start:end], group))
    return torch.cat(chunks, dim=1)


@torch.no_grad()
def _distributed_sum_control(
    logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start: int,
    vocab_end: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Positive control: change the full-row ATen sum tree to per-shard sums."""
    fp32 = logits.float()
    row_max = torch.amax(fp32, dim=-1, keepdim=True)
    dist.all_reduce(row_max, op=dist.ReduceOp.MAX, group=group)
    shifted = fp32 - row_max
    denominator = shifted.exp().sum(-1, keepdim=True).float()
    dist.all_reduce(denominator, op=dist.ReduceOp.SUM, group=group)
    mask = (target < vocab_start) | (target >= vocab_end)
    local_target = (target - vocab_start).masked_fill(mask, 0)
    selected = torch.gather(shifted, -1, local_target.unsqueeze(-1)).squeeze(-1)
    selected.sub_(denominator.log_().squeeze(-1)).masked_fill_(mask, 0.0)
    dist.all_reduce(selected, op=dist.ReduceOp.SUM, group=group)
    return selected


def _max_across_ranks(value: float, device: torch.device) -> float:
    result = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return float(result.item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--vocab", type=int, default=248320)
    parser.add_argument("--actual-vocab", type=int)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--pattern", choices=("random", "ties", "nonfinite"), default="random")
    parser.add_argument("--source-lever", action="store_true")
    parser.add_argument("--fixed-source-mode", choices=("incumbent", "candidate"))
    parser.add_argument("--measure-periodic-probe", action="store_true")
    args = parser.parse_args()
    actual_vocab = args.vocab if args.actual_vocab is None else args.actual_vocab

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    if args.vocab % world:
        raise ValueError("The baseline harness currently requires an even vocabulary partition")
    shard_vocab = args.vocab // world
    start = rank * shard_vocab
    end = start + shard_vocab

    generator = torch.Generator(device=device)
    generator.manual_seed(20260815 + rank)
    if args.pattern in ("random", "nonfinite"):
        logits = torch.randn((1, args.rows, shard_vocab), device=device, dtype=torch.bfloat16, generator=generator)
    else:
        cols = torch.arange(shard_vocab, device=device, dtype=torch.int64) + start
        logits = ((cols.remainder(17) - 8).to(torch.float32) / 4).to(torch.bfloat16)
        logits = logits.view(1, 1, -1).expand(1, args.rows, -1).contiguous()
    if args.pattern == "nonfinite" and rank == 0:
        logits[0, 0, 0] = torch.inf
    target_generator = torch.Generator(device=device)
    target_generator.manual_seed(1729)
    target = torch.randint(0, actual_vocab, (1, args.rows), device=device, generator=target_generator)
    boundaries = [0, actual_vocab - 1, 0, actual_vocab - 1]
    for partition in range(1, world):
        boundary = partition * shard_vocab
        if boundary < actual_vocab:
            boundaries.extend((boundary - 1, boundary))
    adversarial = torch.tensor(
        boundaries,
        device=device,
        dtype=target.dtype,
    )
    target.view(-1)[: min(target.numel(), adversarial.numel())] = adversarial[: target.numel()]

    def current() -> torch.Tensor:
        if args.source_lever:
            os.environ["SKYRL_ISOEXEC_EXACT_SAMPLED_LOGPROBS"] = "0"
        return ChunkedDistributedLogprob.apply(logits, target, start, end, args.chunk_size, dist.group.WORLD, True)

    def candidate() -> torch.Tensor:
        if args.source_lever:
            os.environ["SKYRL_ISOEXEC_EXACT_SAMPLED_LOGPROBS"] = "1"
            return ChunkedDistributedLogprob.apply(logits, target, start, end, args.chunk_size, dist.group.WORLD, True)
        return _candidate(logits, target, dist.group.WORLD, args.chunk_size)

    if args.fixed_source_mode is not None:
        os.environ[exact.ENV] = "1" if args.fixed_source_mode == "candidate" else "0"

        def operation() -> torch.Tensor:
            return ChunkedDistributedLogprob.apply(logits, target, start, end, args.chunk_size, dist.group.WORLD, True)

        for _ in range(args.warmup):
            operation()
        torch.cuda.synchronize()
        dist.barrier()

        torch.cuda.reset_peak_memory_stats(device)
        output = operation()
        torch.cuda.synchronize()
        peak = _max_across_ranks(float(torch.cuda.max_memory_allocated(device)), device)

        latencies = []
        for _ in range(args.iterations):
            begin = torch.cuda.Event(enable_timing=True)
            finish = torch.cuda.Event(enable_timing=True)
            dist.barrier()
            begin.record()
            output = operation()
            finish.record()
            finish.synchronize()
            latencies.append(_max_across_ranks(begin.elapsed_time(finish), device))

        periodic_probe_ms = None
        if args.measure_periodic_probe:
            if args.fixed_source_mode != "candidate":
                raise ValueError("--measure-periodic-probe requires --fixed-source-mode=candidate")
            probe_latencies = []
            for _ in range(3):
                exact._S["served"] = exact.PROBE_EVERY
                begin = torch.cuda.Event(enable_timing=True)
                finish = torch.cuda.Event(enable_timing=True)
                dist.barrier()
                begin.record()
                output = operation()
                finish.record()
                finish.synchronize()
                probe_latencies.append(_max_across_ranks(begin.elapsed_time(finish), device))
            periodic_probe_ms = statistics.median(probe_latencies)

        output_bits = output.contiguous().view(torch.int32)
        gathered_bits = [torch.empty_like(output_bits) for _ in range(world)]
        dist.all_gather(gathered_bits, output_bits)
        rank_equal = all(torch.equal(output_bits, peer) for peer in gathered_bits)
        if rank == 0:
            median_ms = statistics.median(latencies)
            probe_fields = ""
            if periodic_probe_ms is not None:
                incremental_ms = periodic_probe_ms - median_ms
                probe_fields = (
                    f" periodic_probe_ms={periodic_probe_ms:.3f}"
                    f" periodic_incremental_ms={incremental_ms:.3f}"
                    f" amortized_probe_ms_per_chunk={incremental_ms / exact.PROBE_EVERY:.6f}"
                )
            print(
                "LOGP_FIXED_SOURCE "
                f"mode={args.fixed_source_mode} world={world} rows={args.rows} vocab={args.vocab} "
                f"shard_vocab={shard_vocab} actual_vocab={actual_vocab} chunk_size={args.chunk_size} "
                f"TOKENS/device={args.rows} full_vocab_elements/device={args.rows * args.vocab} "
                f"dtype={logits.dtype} pattern={args.pattern} rank_equal={rank_equal} "
                f"latency_ms_median={median_ms:.3f} latency_ms_min={min(latencies):.3f} "
                f"latency_ms_max={max(latencies):.3f} peak_alloc_gib={peak / 2**30:.3f} "
                f"served={exact.stats()['served']} declined={exact.stats()['declined']} "
                f"probes={exact.stats()['probes']}" + probe_fields,
                flush=True,
            )
        dist.destroy_process_group()
        return

    for _ in range(args.warmup):
        current()
        candidate()
    torch.cuda.synchronize()
    dist.barrier()

    reference = current()
    proposed = candidate()
    bit_equal = torch.equal(reference.view(torch.int32), proposed.view(torch.int32))
    control = _distributed_sum_control(logits, target, start, end, dist.group.WORLD)
    control_mismatches = int((reference.view(torch.int32) != control.view(torch.int32)).sum().item())
    equal_vote = torch.tensor(int(bit_equal), dtype=torch.int32, device=device)
    dist.all_reduce(equal_vote, op=dist.ReduceOp.MIN)
    if not equal_vote.item():
        mismatch = (reference.view(torch.int32) != proposed.view(torch.int32)).sum().item()
        max_diff = (reference - proposed).abs().max().item()
        raise AssertionError(f"candidate mismatch on rank {rank}: mismatches={mismatch} max_diff={max_diff}")

    del reference, proposed, control
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    peak_output = current()
    torch.cuda.synchronize()
    current_peak = _max_across_ranks(float(torch.cuda.max_memory_allocated(device)), device)
    del peak_output
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    peak_output = candidate()
    torch.cuda.synchronize()
    candidate_peak = _max_across_ranks(float(torch.cuda.max_memory_allocated(device)), device)
    del peak_output

    current_latencies = []
    candidate_latencies = []
    output = None
    for iteration in range(args.iterations):
        ordered = (("current", current), ("candidate", candidate))
        if iteration % 2:
            ordered = tuple(reversed(ordered))
        for name, operation in ordered:
            begin = torch.cuda.Event(enable_timing=True)
            finish = torch.cuda.Event(enable_timing=True)
            dist.barrier()
            begin.record()
            output = operation()
            finish.record()
            finish.synchronize()
            elapsed = _max_across_ranks(begin.elapsed_time(finish), device)
            (current_latencies if name == "current" else candidate_latencies).append(elapsed)

    assert output is not None
    output_bits = output.contiguous().view(torch.int32)
    gathered_bits = [torch.empty_like(output_bits) for _ in range(world)]
    dist.all_gather(gathered_bits, output_bits)
    rank_equal = all(torch.equal(output_bits, peer) for peer in gathered_bits)
    if rank == 0:
        tokens_per_device = args.rows
        gathered_vocab_elements_per_device = args.rows * args.vocab
        print(
            "LOGP_COMPARE "
            f"world={world} rows={args.rows} vocab={args.vocab} shard_vocab={shard_vocab} "
            f"actual_vocab={actual_vocab} "
            f"chunk_size={args.chunk_size} TOKENS/device={tokens_per_device} "
            f"full_vocab_elements/device={gathered_vocab_elements_per_device} "
            f"dtype={logits.dtype} pattern={args.pattern} bit_equal={bool(equal_vote.item())} rank_equal={rank_equal} "
            f"sum_tree_control_mismatches={control_mismatches} "
            f"source_lever={args.source_lever} "
            f"current_ms_median={statistics.median(current_latencies):.3f} "
            f"current_ms_min={min(current_latencies):.3f} current_ms_max={max(current_latencies):.3f} "
            f"candidate_ms_median={statistics.median(candidate_latencies):.3f} "
            f"candidate_ms_min={min(candidate_latencies):.3f} candidate_ms_max={max(candidate_latencies):.3f} "
            f"ratio={statistics.median(candidate_latencies) / statistics.median(current_latencies):.3f} "
            f"current_peak_alloc_gib={current_peak / 2**30:.3f} "
            f"candidate_peak_alloc_gib={candidate_peak / 2**30:.3f}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("SKYRL_ZERO_KL", "1")
    os.environ.setdefault("SKYRL_ISOEXEC_LOGPROB_GATHER_BF16_WIRE", "1")
    main()
