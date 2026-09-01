#!/usr/bin/env python3
"""Bounded, model-free GPU qualification for exact-vocab TP transport.

Checks the gathered FP32 bit pattern from ``gather_rank_ordered_fp32`` against all-gather+cat and
the historical gather-to-owner P2P transport, timing all three. Run each dtype as its own torchrun
invocation so a process-group lifecycle carries one wire contract; each rank writes one JSON file::

    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
      skyrl/backends/skyrl_train/isoexec/ops/logprobs/tests/exact_vocab_transport_dist.py \
      --dtype bf16 --rows 17,257,1024 --shard-widths 257,62080 \
      --warmup 3 --repeats 10 --output-prefix /tmp/exact_vocab_transport_bf16
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import pathlib
import statistics
import time

import torch
import torch.distributed as dist

if int(os.environ.get("WORLD_SIZE", "1")) < 2 or not torch.cuda.is_available():
    # multi-rank battery: launch under torchrun with world >= 2 (CI); skip gracefully elsewhere
    print("SKIP: needs torchrun with WORLD_SIZE >= 2 and CUDA devices")
    raise SystemExit(0)
import yaml

from skyrl.backends.skyrl_train.isoexec.ops.logprobs import (
    exact_vocab_transport as transport,
)

_CONFIG_FIELDS = frozenset(
    {
        "dtype",
        "rows",
        "shard_widths",
        "warmup",
        "repeats",
        "output_prefix",
        "transport_init",
        "prewarm_payload_mib",
        "post_lazy_zero_tolerance_mib",
        "expected_tp_world",
        "nccl_max_nchannels",
        "tp_max_ctas",
    }
)


def _csv_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _load_config(path: str | None) -> dict:
    if not path:
        return {}
    document = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("battery YAML must contain one mapping")
    unknown = set(document) - _CONFIG_FIELDS
    if unknown:
        raise ValueError(f"unknown battery YAML fields: {sorted(unknown)!r}")
    result = dict(document)
    for name in ("rows", "shard_widths"):
        if name in result:
            values = result[name]
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(value, int) or value < 1 for value in values)
            ):
                raise ValueError(f"{name} must be a non-empty list of positive integers")
            result[name] = tuple(values)
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, _ = pre.parse_known_args(argv)
    defaults = _load_config(known.config)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--rows", type=_csv_ints, default=(17, 257, 1024))
    parser.add_argument("--shard-widths", type=_csv_ints, default=(257, 62080))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output-prefix")
    parser.add_argument("--transport-init", choices=("first-use", "prewarmed"), default="first-use")
    parser.add_argument("--prewarm-payload-mib", type=float, default=8.0)
    parser.add_argument("--post-lazy-zero-tolerance-mib", type=float, default=16.0)
    parser.add_argument("--expected-tp-world", type=int, default=0)
    parser.add_argument("--nccl-max-nchannels", type=int)
    parser.add_argument("--tp-max-ctas", type=int)
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if not args.dtype or not args.output_prefix:
        parser.error("--dtype and --output-prefix are required (directly or through --config)")
    if args.warmup < 1 or args.repeats < 1:
        parser.error("--warmup and --repeats must be positive")
    if args.prewarm_payload_mib <= 0 or args.post_lazy_zero_tolerance_mib < 0:
        parser.error("prewarm payload must be positive and post-lazy tolerance non-negative")
    if args.expected_tp_world < 0:
        parser.error("--expected-tp-world must be non-negative")
    if any(value is not None and value < 1 for value in (args.nccl_max_nchannels, args.tp_max_ctas)):
        parser.error("--nccl-max-nchannels and --tp-max-ctas must be positive")
    return args


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype is torch.float32:
        return tensor.contiguous().view(torch.int32)
    if tensor.dtype in (torch.bfloat16, torch.float16):
        return tensor.contiguous().view(torch.int16)
    raise TypeError(tensor.dtype)


def _wire(rows: int, width: int, rank: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1701 + 1009 * rank + 17 * rows + width)
    value = torch.randn((1, rows, width), generator=generator, dtype=torch.float32)
    value.mul_(2.0).add_(rank * 0.25)
    flat = value.view(-1)
    if flat.numel() >= 4:
        flat[0] = 0.0
        flat[1] = -0.0
        flat[2] = float(rank)
        flat[3] = -float(rank)
    return value.to(dtype=dtype, device=device)


@torch.no_grad()
def _incumbent(wire: torch.Tensor, world: int, group) -> torch.Tensor:
    gathered = [torch.empty_like(wire) for _ in range(world)]
    dist.all_gather(gathered, wire, group=group)
    return torch.cat(gathered, dim=-1).float()


@torch.no_grad()
def _owner_p2p(wire: torch.Tensor, world: int, group, owner: int = 0) -> torch.Tensor | None:
    """Historical gather-to-owner byte transport, reconstructed from its contract.

    P2P only moves bytes; the FP32 widening and final layout match the other references exactly.
    """

    rank = int(dist.get_rank(group=group))
    global_rank = getattr(dist, "get_global_rank", None)
    if group is None or group is dist.group.WORLD:
        members = tuple(range(world))
    elif global_rank is not None:
        members = tuple(int(global_rank(group, peer)) for peer in range(world))
    else:
        raise RuntimeError("owner-P2P comparator requires get_global_rank for TP subgroups")
    flat = wire.contiguous().view(-1)
    if rank == owner:
        received = torch.empty(world * flat.numel(), dtype=wire.dtype, device=wire.device)
        received.view(world, -1)[owner].copy_(flat)
        ops = [
            dist.P2POp(dist.irecv, received.view(world, -1)[peer], members[peer], group=group)
            for peer in range(world)
            if peer != owner
        ]
    else:
        received = None
        ops = [dist.P2POp(dist.isend, flat, members[owner], group=group)]
    requests = dist.batch_isend_irecv(ops)
    for request in requests:
        request.wait()
    if rank != owner:
        return None
    rank_major = received.view(world, *wire.shape)
    shard_vocab = wire.shape[-1]
    full = torch.empty((*wire.shape[:-1], world * shard_vocab), dtype=torch.float32, device=wire.device)
    for source in range(world):
        full[..., source * shard_vocab : (source + 1) * shard_vocab].copy_(rank_major[source])
    return full


def _timed(fn, repeats: int) -> tuple[float, object]:
    samples = []
    result = None
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples), result


def _nontorch_row(before: transport.MemorySnapshot, after: transport.MemorySnapshot) -> dict:
    return {
        "before": dataclasses.asdict(before),
        "after": dataclasses.asdict(after),
        "incremental_nontorch_mib": after.absolute_nontorch_mib - before.absolute_nontorch_mib,
    }


@torch.no_grad()
def _prewarm_tp_a2a(world: int, device: torch.device, payload_mib: float, group) -> dict:
    """Mirror production's balanced BF16 TP A2A prewarm on the same PG."""

    payload_bytes = math.ceil(payload_mib * 1024**2)
    elements_per_peer = math.ceil(payload_bytes / (world * torch.empty((), dtype=torch.bfloat16).element_size()))
    source = torch.zeros(world * elements_per_peer, dtype=torch.bfloat16, device=device)
    destination = torch.empty_like(source)
    before = transport._memory_snapshot(device)
    dist.all_to_all_single(destination, source, group=group)
    after = transport._memory_snapshot(device)
    del source, destination
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return {
        "payload_bytes": payload_bytes,
        "elements_per_peer": elements_per_peer,
        **_nontorch_row(before, after),
    }


def _selected(wire: torch.Tensor, world: int, group) -> tuple[str, torch.Tensor | None]:
    if transport.use_owner_gather(wire.numel(), wire.element_size(), world):
        return "owner_a2a", transport.gather_rank_ordered_fp32(wire, group=group, world=world)
    return "canonical_all_gather", _incumbent(wire, world, group)


def _make_tp_group(world: int, max_ctas: int | None):
    if max_ctas is None:
        return dist.group.WORLD
    options = dist.ProcessGroupNCCL.Options()
    options.config.max_ctas = max_ctas
    options.config.min_ctas = max_ctas
    return dist.new_group(ranks=list(range(world)), pg_options=options)


def _readback_max_ctas(group) -> int | None:
    candidates = (group, getattr(group, "_get_backend", lambda _device: None)(torch.device("cuda")))
    for candidate in candidates:
        config = getattr(getattr(candidate, "options", None), "config", None)
        value = getattr(config, "max_ctas", None)
        if value is not None:
            return int(value)
    return None


def main() -> None:
    args = _parse_args()
    if args.nccl_max_nchannels is not None:
        configured = os.environ.get("NCCL_MAX_NCHANNELS")
        requested = str(args.nccl_max_nchannels)
        if configured not in (None, requested):
            raise RuntimeError(f"NCCL_MAX_NCHANNELS={configured} conflicts with --nccl-max-nchannels={requested}")
        os.environ["NCCL_MAX_NCHANNELS"] = requested

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    tp_group = _make_tp_group(world, args.tp_max_ctas)
    tp_max_ctas_readback = _readback_max_ctas(tp_group)
    if args.tp_max_ctas is not None and tp_max_ctas_readback != args.tp_max_ctas:
        raise RuntimeError(f"TP ProcessGroupNCCL max_ctas readback={tp_max_ctas_readback} requested={args.tp_max_ctas}")
    dtype = _dtype(args.dtype)
    rows_out = []
    all_exact = True
    prewarm_memory = None
    try:
        if args.expected_tp_world and world != args.expected_tp_world:
            raise RuntimeError(f"expected TP world {args.expected_tp_world}, torchrun built {world}")
        # Initialize the ordinary TP communicator before transport's first-lazy memory baseline,
        # so the measured delta isolates the A2A transport from generic NCCL setup.
        warm = torch.ones(1, dtype=torch.float32, device=device)
        dist.all_reduce(warm, group=tp_group)
        torch.cuda.synchronize()
        if args.transport_init == "prewarmed":
            prewarm_memory = _prewarm_tp_a2a(world, device, args.prewarm_payload_mib, tp_group)
        for rows in args.rows:
            for width in args.shard_widths:
                wire = _wire(rows, width, rank, dtype, device)
                for _ in range(args.warmup):
                    candidate = transport.gather_rank_ordered_fp32(wire, group=tp_group, world=world)
                    if rank == 0:
                        del candidate
                candidate_ms, candidate = _timed(
                    lambda wire=wire: transport.gather_rank_ordered_fp32(wire, group=tp_group, world=world),
                    args.repeats,
                )
                p2p_before = transport._memory_snapshot(device)
                p2p_ms, p2p = _timed(lambda wire=wire: _owner_p2p(wire, world, tp_group), args.repeats)
                p2p_after = transport._memory_snapshot(device)
                incumbent_ms, incumbent = _timed(lambda wire=wire: _incumbent(wire, world, tp_group), args.repeats)
                selected_ms, selected_pair = _timed(lambda wire=wire: _selected(wire, world, tp_group), args.repeats)
                selected_owner, selected = selected_pair
                local_exact = True
                local_selected_exact = True
                mismatches = 0
                selected_mismatches = 0
                if rank == 0:
                    candidate_bits = _bits(candidate)
                    incumbent_bits = _bits(incumbent)
                    p2p_bits = _bits(p2p)
                    mismatches = int((candidate_bits != incumbent_bits).sum().item())
                    p2p_mismatches = int((p2p_bits != incumbent_bits).sum().item())
                    local_exact = mismatches == 0 and p2p_mismatches == 0
                    selected_bits = _bits(selected)
                    selected_mismatches = int((selected_bits != incumbent_bits).sum().item())
                    local_selected_exact = selected_mismatches == 0
                verdict = torch.tensor(int(local_exact), dtype=torch.int32, device=device)
                dist.all_reduce(verdict, op=dist.ReduceOp.MIN, group=tp_group)
                exact = bool(verdict.item())
                selected_verdict = torch.tensor(int(local_selected_exact), dtype=torch.int32, device=device)
                dist.all_reduce(selected_verdict, op=dist.ReduceOp.MIN, group=tp_group)
                selected_exact = bool(selected_verdict.item())
                all_exact &= exact and selected_exact
                full_bytes = transport.full_wire_bytes(wire.numel(), wire.element_size(), world)
                selected_candidate = transport.use_owner_gather(wire.numel(), wire.element_size(), world)
                rows_out.append(
                    {
                        "rows": rows,
                        "shard_width": width,
                        "wire_bytes_per_rank": wire.numel() * wire.element_size(),
                        "full_wire_bytes": full_bytes,
                        "canonical_threshold_bytes": transport.MIN_FULL_WIRE_BYTES,
                        "selected_owner": selected_owner,
                        "selected_candidate": selected_candidate,
                        "canonical_fallback": not selected_candidate,
                        "candidate_median_ms": candidate_ms,
                        "owner_p2p_median_ms": p2p_ms,
                        "incumbent_median_ms": incumbent_ms,
                        "selected_median_ms": selected_ms,
                        "speedup": incumbent_ms / candidate_ms,
                        "speedup_vs_owner_p2p": p2p_ms / candidate_ms,
                        "exact": exact,
                        "selected_exact": selected_exact,
                        "owner_mismatches": mismatches,
                        "owner_p2p_mismatches": p2p_mismatches if rank == 0 else 0,
                        "owner_p2p_memory": _nontorch_row(p2p_before, p2p_after),
                        "selected_owner_mismatches": selected_mismatches,
                        "measured_candidate_profitable": incumbent_ms > candidate_ms,
                        "measured_candidate_profitable_vs_owner_p2p": p2p_ms > candidate_ms,
                        "dispatch_matches_measured_sign": selected_candidate == (incumbent_ms > candidate_ms),
                    }
                )
                del candidate, p2p, incumbent, selected, wire
        census = transport.stats()
        group_row = census["groups"][0]
        post_lazy_delta = group_row["incremental_nontorch_mib"]
        local_post_lazy_zero = post_lazy_delta is not None and abs(post_lazy_delta) <= args.post_lazy_zero_tolerance_mib
        zero_vote = torch.tensor(int(local_post_lazy_zero), dtype=torch.int32, device=device)
        dist.all_reduce(zero_vote, op=dist.ReduceOp.MIN, group=tp_group)
        all_post_lazy_zero = bool(zero_vote.item())
        result = {
            "schema": "isoexec.exact_vocab_transport_battery.v3",
            "rank": rank,
            "world": world,
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "dtype": args.dtype,
            "transport_init": args.transport_init,
            "nccl_max_nchannels": os.environ.get("NCCL_MAX_NCHANNELS"),
            "tp_max_ctas_requested": args.tp_max_ctas,
            "tp_max_ctas_readback": tp_max_ctas_readback,
            "expected_tp_world": args.expected_tp_world,
            "canonical_threshold_bytes": transport.MIN_FULL_WIRE_BYTES,
            "prewarm_memory": prewarm_memory,
            "post_lazy_zero_expected": args.transport_init == "prewarmed",
            "post_lazy_zero_tolerance_mib": args.post_lazy_zero_tolerance_mib,
            "local_post_lazy_zero": local_post_lazy_zero,
            "all_post_lazy_zero": all_post_lazy_zero,
            "all_exact": all_exact,
            "cases": rows_out,
            "transport": census,
        }
        path = pathlib.Path(f"{args.output_prefix}.rank{rank}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        if rank == 0:
            print("RESULT " + json.dumps(result, sort_keys=True), flush=True)
        if not all_exact:
            raise SystemExit(2)
        if args.transport_init == "prewarmed" and not all_post_lazy_zero:
            raise SystemExit(3)
    finally:
        transport.reset_for_teardown()
        if tp_group is not dist.group.WORLD:
            dist.destroy_process_group(tp_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
