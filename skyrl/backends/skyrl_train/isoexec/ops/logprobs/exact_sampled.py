"""Exact sampled-logprob extraction for the IsoExec full-vocabulary gather.

The full vocabulary is still gathered and reduced by ATen: its fp32 ``sum`` tree is the
forward contract.  This execution twin gathers the sampled raw logit before the gathered
buffer is overwritten, fuses ``sub`` + ``exp`` into one pass, and never materializes the
full local-vocabulary logprob tensor.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import torch
import torch.distributed as dist

from ..collectives.logprob_gather_wire import gather_full_vocab
from .exact_vocab_pipeline import (
    maybe_exact_vocab_pipeline,
)
from .exact_vocab_pipeline import (
    release_group as _release_pipeline_group,
)
from .exact_vocab_pipeline import (
    reset_for_teardown as _reset_pipeline_for_teardown,
)
from .exact_vocab_pipeline import (
    stats as _pipeline_stats,
)

ENV = "SKYRL_ISOEXEC_EXACT_SAMPLED_LOGPROBS"
BANNER = "[ISOEXEC-EXACT-SAMPLED-LOGPROBS]"
PROBE_EVERY = 512
MIN_ROWS = 17

_KERNEL = None
_KERNEL_ERROR = ""
_GROUP_ADMISSION = {}
_S = {
    "calls": 0,
    "served": 0,
    "declined": 0,
    "decline_reason": "",
    "agreed": None,
    "probes": 0,
    "tokens": 0,
    "full_pass_bytes_saved": 0,
    "local_logprob_bytes_saved": 0,
    "local_logprob_elements_not_materialized": 0,
    "sampled_allreduces_saved": 0,
    "reported": 0,
    "bannered": False,
}


def enabled() -> bool:
    return os.environ.get(ENV, "0") == "1"


def stats() -> dict:
    """Return a copy of the engagement census; only ``served > 0`` proves ownership."""
    result = dict(_S)
    result["pipeline"] = _pipeline_stats()
    return result


def _reset_for_test() -> None:
    global _KERNEL, _KERNEL_ERROR
    _KERNEL = None
    _KERNEL_ERROR = ""
    _GROUP_ADMISSION.clear()
    _reset_pipeline_for_teardown()
    _S.update(
        {
            "calls": 0,
            "served": 0,
            "declined": 0,
            "decline_reason": "",
            "agreed": None,
            "probes": 0,
            "tokens": 0,
            "full_pass_bytes_saved": 0,
            "local_logprob_bytes_saved": 0,
            "local_logprob_elements_not_materialized": 0,
            "sampled_allreduces_saved": 0,
            "reported": 0,
            "bannered": False,
        }
    )


def release_group(group) -> None:
    """Release admission and transport state before destroying one TP group."""

    _GROUP_ADMISSION.pop(id(group), None)
    _release_pipeline_group(group)


def reset_for_teardown() -> None:
    """Release all process-group-bound state before distributed reinitialization."""

    _GROUP_ADMISSION.clear()
    _reset_pipeline_for_teardown()
    _S.update(
        {
            "calls": 0,
            "served": 0,
            "declined": 0,
            "decline_reason": "",
            "agreed": None,
            "probes": 0,
            "tokens": 0,
            "full_pass_bytes_saved": 0,
            "local_logprob_bytes_saved": 0,
            "local_logprob_elements_not_materialized": 0,
            "sampled_allreduces_saved": 0,
            "reported": 0,
            "bannered": False,
        }
    )


def _load_kernel():
    global _KERNEL, _KERNEL_ERROR
    if _KERNEL is not None or _KERNEL_ERROR:
        return _KERNEL
    try:
        import triton
        import triton.language as tl
        from triton.language.extra import libdevice

        @triton.jit
        def fused_exp_inplace(full_ptr, row_max_ptr, stride, vocab, BLOCK: tl.constexpr):
            row = tl.program_id(0)
            block = tl.program_id(1)
            offsets = block * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < vocab
            value = tl.load(full_ptr + row * stride + offsets, mask=mask, other=0.0).to(tl.float32)
            row_max = tl.load(row_max_ptr + row)
            value = libdevice.exp(value - row_max)
            tl.store(full_ptr + row * stride + offsets, value, mask=mask)

        _KERNEL = (triton, fused_exp_inplace)
    except Exception as error:  # noqa: BLE001 -- unavailable means the incumbent remains owner
        _KERNEL_ERROR = repr(error)
    return _KERNEL


def _banner_once(on: bool) -> None:
    if _S["bannered"]:
        return
    _S["bannered"] = True
    if on:
        print(
            f"{BANNER} ON ({ENV}=1, default OFF): scoring-only execution twin gathers the sampled "
            "raw logit before overwriting the required full-vocabulary buffer, fuses fp32 sub+exp, "
            "keeps ATen amax/sum/log unchanged, and returns only `(sampled-max)-lse`. Unsupported "
            "layouts and a failed first-call bit probe fall back to the incumbent. Judge engagement "
            "only by served>0 in the census.",
            flush=True,
        )
    else:
        print(
            f"{BANNER} OFF ({ENV}={os.environ.get(ENV, '0')}, default 0): the incumbent materializes "
            "local-vocabulary logprobs and all-reduces the selected value.",
            flush=True,
        )


def _report() -> None:
    calls = _S["calls"]
    if calls < 1 or (calls & (calls - 1)) or calls == _S["reported"]:
        return
    _S["reported"] = calls
    print(
        f"{BANNER} CENSUS pid={os.getpid()} calls={calls} served={_S['served']} "
        f"declined={_S['declined']} tokens={_S['tokens']} probes={_S['probes']} "
        f"full_pass_saved={_S['full_pass_bytes_saved'] / 1e9:.2f} GB "
        f"local_logprob_saved={_S['local_logprob_bytes_saved'] / 1e9:.2f} GB "
        f"sampled_allreduces_saved={_S['sampled_allreduces_saved']}"
        + (f" last_decline={_S['decline_reason']}" if _S["decline_reason"] else ""),
        flush=True,
    )


def _decline(reason: str) -> None:
    _S["declined"] += 1
    _S["decline_reason"] = reason
    _report()


def _admit(
    logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start_index: int,
    vocab_end_index: int,
    group,
    src_dtype: torch.dtype,
) -> tuple[int | None, str]:
    _S["calls"] += 1
    configured = os.environ.get(ENV)
    on = configured == "1"
    _banner_once(on)
    # An unset flag declines before any rank query or rendezvous, so this lever is invisible to
    # callers that never configured it. An explicit value -- including an explicit "0" -- stays part
    # of the TP-unanimous contract, so an asymmetric environment cannot split the later collectives.
    if configured is None:
        reason = f"{ENV} is unset (default 0)"
        _decline(reason)
        return None, reason
    if not dist.is_available() or not dist.is_initialized():
        reason = "torch.distributed is not initialized"
        _decline(reason)
        return None, reason
    world = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    shard_vocab = logits.shape[-1] if logits.ndim else 0

    reason = ""
    if not on:
        reason = f"{ENV}=0"
    elif os.environ.get("SKYRL_ISOEXEC") != "1":
        reason = "SKYRL_ISOEXEC!=1"
    elif not logits.is_cuda or logits.dtype is not torch.float32:
        reason = f"requires CUDA fp32 widened logits, got device={logits.device} dtype={logits.dtype}"
    elif src_dtype not in (torch.bfloat16, torch.float16):
        reason = f"source dtype {src_dtype} is not bf16/fp16"
    elif logits.ndim != 3 or target.shape != logits.shape[:-1] or logits.stride(-1) != 1:
        reason = f"unsupported logits/target layout logits={tuple(logits.shape)} target={tuple(target.shape)}"
    elif target.dtype not in (torch.int32, torch.int64):
        reason = f"target dtype {target.dtype} is not int32/int64"
    elif target.numel() < MIN_ROWS:
        reason = f"rows={target.numel()} < {MIN_ROWS}: launch-bound incumbent is retained"
    elif world <= 1:
        reason = f"world={world}: distributed gather branch is not the owner"
    elif vocab_start_index != rank * shard_vocab or vocab_end_index != vocab_start_index + shard_vocab:
        reason = (
            f"non-contiguous/equal partition start={vocab_start_index} end={vocab_end_index} "
            f"rank={rank} shard={shard_vocab}"
        )
    elif torch.cuda.get_device_capability(logits.device) != (9, 0):
        reason = f"unvalidated CUDA capability {torch.cuda.get_device_capability(logits.device)}"
    elif _load_kernel() is None:
        reason = f"Triton/libdevice kernel unavailable: {_KERNEL_ERROR}"

    contract = (
        on,
        os.environ.get("SKYRL_ISOEXEC"),
        logits.device.type,
        logits.device.index,
        logits.dtype,
        src_dtype,
        target.dtype,
        world,
        shard_vocab,
        vocab_start_index,
        vocab_end_index,
        torch.cuda.get_device_capability(logits.device) if logits.is_cuda else None,
        _KERNEL is not None,
        _KERNEL_ERROR,
    )
    signature = (
        tuple(logits.shape),
        tuple(logits.stride()),
        tuple(target.shape),
        reason,
    )
    group_key = id(group)
    group_cache = _GROUP_ADMISSION.get(group_key)
    if group_cache is not None:
        if contract != group_cache["contract"]:
            raise RuntimeError(
                f"{BANNER} STRUCTURAL DRIFT after the TP-unanimous admission vote: "
                f"first={group_cache['contract']!r} now={contract!r}. ENV, kernel availability, "
                "capability, layout and vocabulary partition are immutable for this process; "
                "refusing before entering either collective sequence."
            )
        if group_cache["latched_off"]:
            cached_reason = reason or "cached TP peer refusal or end-to-end probe failure"
            _decline(cached_reason)
            return None, cached_reason
        cached = group_cache["verdicts"].get(signature)
        if cached is not None and not cached:
            cached_reason = reason or "cached TP peer refusal or end-to-end probe failure"
            _decline(cached_reason)
            return None, cached_reason
        if cached:
            return world, ""
    else:
        # Retain the group object so a destroy/reinitialize cycle cannot reuse its Python id and
        # inherit a prior group's admission or probe verdict.
        group_cache = {
            "group": group,
            "contract": contract,
            "verdicts": {},
            "latched_off": False,
            "agreed": None,
            "served": 0,
        }
        _GROUP_ADMISSION[group_key] = group_cache

    # Load-bearing: candidate and incumbent issue different collectives after this point, so the
    # first signature per TP group must be unanimous. MIN makes any one rank's refusal everybody's.
    vote = torch.tensor(int(not reason), dtype=torch.int32, device=logits.device)
    dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=group)
    admitted = bool(vote.item())
    group_cache["verdicts"][signature] = admitted
    if not admitted:
        if not reason:
            reason = "a TP peer declined the candidate signature"
        _decline(reason)
        return None, reason
    return world, ""


def _run_fused_exp(full: torch.Tensor, row_max: torch.Tensor) -> None:
    triton, kernel = _KERNEL
    flat = full.reshape(-1, full.shape[-1])
    grid = (flat.shape[0], triton.cdiv(flat.shape[1], 2048))
    kernel[grid](flat, row_max, flat.stride(0), flat.shape[1], BLOCK=2048)


def _probe_final(candidate: torch.Tensor, reference: torch.Tensor, group) -> tuple[bool, str]:
    """Bit-compare every final sampled output against the incumbent and vote unanimously."""
    finite = bool(torch.isfinite(candidate).all().item()) and bool(torch.isfinite(reference).all().item())
    same = torch.equal(candidate.contiguous().view(torch.int32), reference.contiguous().view(torch.int32))
    ok = torch.tensor(int(finite and same), dtype=torch.int32, device=candidate.device)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
    _S["probes"] += 1
    if not finite:
        return False, "non-finite final sampled output"
    if not same:
        mismatches = int(
            (candidate.contiguous().view(torch.int32) != reference.contiguous().view(torch.int32)).sum().item()
        )
        max_diff = float((candidate.float() - reference.float()).abs().max().item())
        return (
            False,
            f"final sampled output differs from the incumbent: mismatches={mismatches} max_diff={max_diff:.3e}",
        )
    return bool(ok.item()), "peer probe failed"


@torch.no_grad()
def maybe_exact_sampled_logprobs(
    logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start_index: int,
    vocab_end_index: int,
    group,
    src_dtype: torch.dtype,
    reference: Callable[[], torch.Tensor],
) -> torch.Tensor | None:
    """Return exact sampled logprobs, or ``None`` to retain the incumbent owner."""
    world, _ = _admit(logits, target, vocab_start_index, vocab_end_index, group, src_dtype)
    if world is None:
        return None

    def finalize(full: torch.Tensor, final_target: torch.Tensor) -> torch.Tensor:
        valid = (final_target >= 0) & (final_target < full.shape[-1])
        safe = final_target.masked_fill(~valid, 0)
        sampled = torch.gather(full, -1, safe.unsqueeze(-1)).squeeze(-1)
        row_max = torch.amax(full, dim=-1, keepdim=True)
        _run_fused_exp(full, row_max)
        lse = full.sum(-1, keepdim=True).float().log()
        value = (sampled - row_max.squeeze(-1)) - lse.squeeze(-1)
        value.masked_fill_(~valid, 0.0)
        return value

    result = maybe_exact_vocab_pipeline(
        logits,
        target,
        group=group,
        world=world,
        src_dtype=src_dtype,
        finalize=finalize,
    )
    if result is None:
        full = gather_full_vocab(logits.contiguous(), group=group, world=world, src_dtype=src_dtype)
        result = finalize(full, target)

    group_cache = _GROUP_ADMISSION[id(group)]
    should_probe = group_cache["agreed"] is None or (
        group_cache["served"] > 0 and group_cache["served"] % PROBE_EVERY == 0
    )
    if should_probe:
        # The reference callback runs the incumbent end to end, second full gather included.
        incumbent = reference()
        ok, reason = _probe_final(result, incumbent, group)
        if not ok:
            group_cache["agreed"] = False
            _S["agreed"] = False
            group_cache["latched_off"] = True
            _decline(f"end-to-end probe failed: {reason}")
            return incumbent
        group_cache["agreed"] = True
        _S["agreed"] = True

    full_numel = logits.numel() * world
    local_numel = logits.numel()
    _S["served"] += 1
    group_cache["served"] += 1
    _S["tokens"] += target.numel()
    _S["full_pass_bytes_saved"] += full_numel * 8
    _S["local_logprob_bytes_saved"] += local_numel * 16
    _S["local_logprob_elements_not_materialized"] += local_numel
    _S["sampled_allreduces_saved"] += 1
    _report()
    return result
