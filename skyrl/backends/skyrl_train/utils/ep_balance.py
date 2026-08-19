"""Length-balanced microbatch scheduling -- the EP-skew barrier-tax fix (SKYRL_ISOEXEC_TRAINER_EP_BALANCE).

THE MECHANISM IT KILLS (probe B, GLM-4.7-Flash, 2026-08-08). At EP=8 every MoE layer's alltoall
dispatch is preceded by a tp_ep-group allgather of ``num_local_tokens_per_expert`` (megatron
``MoEAlltoAllTokenDispatcher.preprocess``, token_dispatcher.py:545) whose CPU values are host-synced
before the A2A can launch -- a rendezvous of ALL ranks, 46 layers x every microbatch. The gathered
payload is 64 int64s; the measured 28.6 s (81.4% of ALL trainer kernel time in the probe-B window)
is not bandwidth, it is WAITING: with ``remove_microbatch_padding=false`` each microbatch is trimmed
to its OWN max real length (``remove_left_padding``), microbatch composition is raw generation
order, and DP shards are contiguous slices -- so at microbatch index i the 8 ranks run wildly
different token counts and every layer serializes on the slowest. Identical 46-call sequences
measured 2.2 ms on one microbatch and 15,326 ms on another; same shapes, same bytes.

WHY SCHEDULING AND NOT THE COLLECTIVE. The barrier cannot be removed: the A2A itself is a per-layer
EP-group rendezvous, so deferring/async-ing the metadata allgather only moves the wait into the A2A
kernels (and between the allgather and its host sync there is less than one layer of independent
work to hide it behind -- ``cuda_sync_point = before_ep_alltoall``). Padding every rank to a uniform
length converts the wait into pad compute at the SAME wall cost (wall per microbatch is already
max-over-ranks). The only real fix is to equalize what each rank computes between barriers -- i.e.
schedule by length. Simulated on the production geometry (64 prompts x 5 samples, DP=8, mbs=2,
GRPO group-correlated lognormal lengths): generation-order tax 3.96 s/minibatch over the no-skew
floor; within-rank sort 1.09 s; cross-rank stripe + sort 0.14 s -- and sorting also shrinks the
floor itself (similar lengths share a pad-to-max microbatch).

THREE SITES, ONE FLAG:
  * ``stripe_minibatch_rows_for_dp`` (trainer, before ``stage_data``): within each mini-batch,
    sort rows by real length (desc, stable) and deal them round-robin into DP-rank-contiguous
    stripes, so every rank's shard carries a near-identical length profile. Mini-batch MEMBERSHIP
    is untouched -- only which rank trains which of that mini-batch's samples.
  * ``RayPPOTrainer._execute_forward_pass`` (policy scoring): stage the identical stripe, then
    invert the permutation on per-sample outputs before comparing them with rollout logprobs. This
    fixes the old scoring/training bin skew without changing any consumer-visible row order.
  * ``sort_shard_rows_by_length`` (worker, before microbatch chunking): sort the rank's shard by
    real length (desc, stable) so the i-th microbatch pairs like lengths and rank-to-rank
    microbatch cost profiles align. Idempotent after the trainer stripe (stripes arrive sorted),
    and still recovers ~72% of the tax alone if the stripe declined.

WHY THE ISOEXEC GATE CANNOT MOVE (the legality argument, in full):
  1. Per-token forward values are BATCH-INVARIANT -- that is the pinned IsoExec property the whole
     stack exists to enforce (batch-invariant aten overrides, SequentialMLP per-expert F.linear,
     fixed-order EP combine, num_splits=1 attention). Regrouping which sequences share a rank or a
     microbatch therefore changes NO real token's logprob at a given theta. The gate
     (policy_kl == 0.0, rollout_train_logprobs_abs_diff at the cross-runtime floor) compares
     trainer vs engine per token at the SAME theta each step; it is preserved by construction,
     and it re-verifies that construction live at every step.
  2. What DOES change is the fp32 gradient-accumulation ORDER: different samples share a
     microbatch (local main_grad accumulation reassociates) and different samples share a DP rank
     (the grad allreduce forms different partial sums). Theta therefore evolves differently from
     an unbalanced run by ROUNDING REASSOCIATION only -- the same legality class as changing
     ``micro_train_batch_size_per_gpu`` or DP size, neither of which the gate pins. The gate never
     compares two trainer runs; it compares trainer to engine at each theta, and both sides see
     every theta through the same weight sync.
  3. Nothing downstream keys on training-step row order: ``_execute_training_step`` consumes only
     all-reduced metrics (per-sample ``loss_fn_outputs`` of ``forward_backward`` are not read),
     and the worker's DYN_RECOMPUTE ledger keys on the UNTRIMMED shard seq_len x mbs, which a
     row permutation cannot move. Scoring's positional outputs are inversely permuted back to the
     original rollout order before they leave ``_execute_forward_pass``.

Pure functions + one env read; no torch.distributed, no megatron imports -- CPU-testable.
"""

import os
from typing import List, Optional, Tuple

import torch

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch

# Registered + forwarded on the TRAIN channel (isoexec.core.flags) -- read in the trainer driver
# AND inside the policy Ray actor, so without the allowlist entry the worker half would be the
# silent-no-op trap (91eee5ba class).
EP_BALANCE_ENV = "SKYRL_ISOEXEC_TRAINER_EP_BALANCE"


def ep_balance_enabled() -> bool:
    """Default OFF."""
    return os.environ.get(EP_BALANCE_ENV, "0") == "1"


def real_sequence_lengths(data: TrainingInputBatch) -> Optional[torch.Tensor]:
    """Per-row real token counts from the attention mask, on CPU (int64). None if unavailable."""
    attention_mask = data.get("attention_mask")
    if attention_mask is None or not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
        return None
    return attention_mask.sum(dim=1).to("cpu", torch.long)


def length_descending_order(lengths: torch.Tensor) -> torch.Tensor:
    """Stable argsort, longest first. Stable => deterministic tie-break by original index."""
    return torch.argsort(lengths.cpu(), descending=True, stable=True)


def permute_batch_rows(data: TrainingInputBatch, perm: torch.Tensor) -> TrainingInputBatch:
    """Row-permuted copy of ``data`` (metadata carried by reference, like ``TensorBatch.chunk``).

    ``perm`` stays on CPU: torch advanced indexing accepts CPU index tensors on CUDA storage, and
    ``TensorList.__getitem__`` takes a tensor index directly.
    """
    permuted = {}
    for key, value in data.items():
        permuted[key] = None if value is None else value[perm]
    out = data.__class__(permuted)
    out.metadata = data.metadata
    return out


def microbatch_maxlen_sum(lengths: torch.Tensor, micro_batch_size: int) -> int:
    """sum_i max(lengths[i*mbs:(i+1)*mbs]) -- the padded-cost proxy the per-layer barriers see.

    ``remove_left_padding`` trims each microbatch to its own max real length, so a microbatch's
    compute is ~ mbs * max(len); this sum is what within-rank scheduling minimizes and what the
    [ISOEXEC-EPBAL] log line reports.
    """
    total = 0
    n = lengths.numel()
    for i in range(0, n, micro_batch_size):
        total += int(lengths[i : i + micro_batch_size].max().item())
    return total


def sort_shard_rows_by_length(
    data: TrainingInputBatch, micro_batch_size: int
) -> Tuple[TrainingInputBatch, Optional[dict]]:
    """Worker site: sort this rank's shard by real length, longest first.

    Returns ``(data, None)`` unchanged when it declines (no 2D attention_mask, or a shard of at
    most one microbatch, where order cannot matter).
    """
    lengths = real_sequence_lengths(data)
    if lengths is None or lengths.numel() <= micro_batch_size:
        return data, None
    order = length_descending_order(lengths)
    if torch.equal(order, torch.arange(lengths.numel())):
        # Already sorted (e.g. the trainer stripe ran): report, don't copy.
        before = microbatch_maxlen_sum(lengths, micro_batch_size)
        return data, {"applied": False, "mb_maxlen_sum_before": before, "mb_maxlen_sum_after": before}
    stats = {
        "applied": True,
        "mb_maxlen_sum_before": microbatch_maxlen_sum(lengths, micro_batch_size),
        "mb_maxlen_sum_after": microbatch_maxlen_sum(lengths[order], micro_batch_size),
    }
    return permute_batch_rows(data, order), stats


def stripe_order_for_dp(lengths: torch.Tensor, dp_size: int) -> Optional[torch.Tensor]:
    """Permutation laying out rows as DP-rank-contiguous, length-striped blocks.

    Rows sorted by length (desc, stable) are dealt round-robin: sorted rank k -> DP rank
    ``k % dp_size``. The permuted layout is rank 0's stripe, then rank 1's, ..., matching
    ``MeshDispatch``'s contiguous ``chunk``. Each rank's stripe is itself descending, so the
    worker's microbatch chunking pairs like lengths with no further sort. None when the row count
    is not divisible by ``dp_size`` (stage_chunks would pad AFTER us and shift the stripe
    boundaries off the chunk boundaries -- decline rather than mis-stripe).
    """
    n = lengths.numel()
    if dp_size <= 1 or n % dp_size != 0:
        return None
    order = length_descending_order(lengths)
    return torch.cat([order[r::dp_size] for r in range(dp_size)])


def stripe_minibatch_rows_for_dp(
    data: TrainingInputBatch,
    mini_batch_boundaries: List[Tuple[int, int]],
    dp_size: int,
) -> Tuple[TrainingInputBatch, Optional[dict]]:
    """Trainer site: within each mini-batch range, stripe rows across DP ranks by length.

    Mini-batch MEMBERSHIP is preserved (permutations never cross a boundary), so every optimizer
    step still consumes exactly the samples it would have -- only the rank assignment and the
    accumulation grouping move. Returns ``(data, None)`` when every range declines.

    Stats report the per-rank REAL-token spread (max rank total - min rank total, summed over
    ranges) before/after -- the coarse skew the stripe removes; the worker line reports the
    finer per-microbatch profile.
    """
    perm, stats = stripe_minibatch_permutation_for_dp(data, mini_batch_boundaries, dp_size)
    if perm is None:
        return data, None
    return permute_batch_rows(data, perm), stats


def stripe_minibatch_permutation_for_dp(
    data: TrainingInputBatch,
    mini_batch_boundaries: List[Tuple[int, int]],
    dp_size: int,
) -> Tuple[Optional[torch.Tensor], Optional[dict]]:
    """Build the trainer stripe without applying it.

    Scoring uses this form so it can stage the same balanced DP shards as training, then invert
    the row permutation on the per-sample outputs before comparing them with rollout logprobs.
    Returning the exact permutation from the shared implementation prevents the scoring and
    training schedules from drifting apart.
    """
    lengths = real_sequence_lengths(data)
    if lengths is None or dp_size <= 1:
        return None, None

    def rank_token_spread(ls: torch.Tensor) -> int:
        per_rank = ls.reshape(dp_size, -1).sum(dim=1)
        return int((per_rank.max() - per_rank.min()).item())

    perm = torch.arange(lengths.numel())
    applied_ranges = 0
    spread_before = 0
    spread_after = 0
    for start, end in mini_batch_boundaries:
        sub = lengths[start:end]
        sub_order = stripe_order_for_dp(sub, dp_size)
        if sub_order is None:
            continue
        perm[start:end] = sub_order + start
        applied_ranges += 1
        spread_before += rank_token_spread(sub)
        spread_after += rank_token_spread(sub[sub_order])
    if applied_ranges == 0:
        return None, None
    stats = {
        "applied_ranges": applied_ranges,
        "total_ranges": len(mini_batch_boundaries),
        "rank_token_spread_before": spread_before,
        "rank_token_spread_after": spread_after,
    }
    return perm, stats


def inverse_permutation(perm: torch.Tensor) -> torch.Tensor:
    """Return indices that restore values emitted in ``perm`` order to their original rows."""
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(perm.numel(), dtype=perm.dtype, device=perm.device)
    return inverse
