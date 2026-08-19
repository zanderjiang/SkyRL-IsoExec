"""CPU tests for the EP-skew balance scheduler (SKYRL_ISOEXEC_TRAINER_EP_BALANCE).

The mechanism under test is pure scheduling: sort a rank's shard by real length before microbatch
chunking (worker site) and stripe rows across DP ranks within each mini-batch (trainer site). The
bitwise-legality argument lives in ``skyrl/backends/skyrl_train/utils/ep_balance.py``; what a CPU
test CAN pin is everything mechanical: rows move as intact units, mini-batch membership is
preserved, stripes align with ``MeshDispatch``'s contiguous chunking, declines are safe no-ops,
the permutation is deterministic, and the skew objective (sum over microbatch indices of the
max-over-ranks padded cost -- what the per-layer EP barriers serialize on) actually shrinks.

uv run --isolated --extra dev pytest tests/backends/skyrl_train/workers/test_ep_balance.py
"""

import torch

from skyrl.backends.skyrl_train.training_batch import TensorList, TrainingInputBatch
from skyrl.backends.skyrl_train.utils.ep_balance import (
    EP_BALANCE_ENV,
    ep_balance_enabled,
    length_descending_order,
    microbatch_maxlen_sum,
    permute_batch_rows,
    real_sequence_lengths,
    sort_shard_rows_by_length,
    stripe_minibatch_rows_for_dp,
    stripe_order_for_dp,
)


def _make_batch(lengths, seq_len=None, with_tensorlist=False):
    """Batch whose row i has ``lengths[i]`` real tokens (left-padded like production sequences)
    and whose sequences encode the row identity, so tests can verify rows moved as units."""
    n = len(lengths)
    seq_len = seq_len or max(lengths)
    attention_mask = torch.zeros((n, seq_len), dtype=torch.long)
    sequences = torch.zeros((n, seq_len), dtype=torch.long)
    for i, ln in enumerate(lengths):
        attention_mask[i, seq_len - ln :] = 1
        sequences[i, :] = i  # row identity
    data = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "advantages": torch.arange(n, dtype=torch.float32).unsqueeze(1).repeat(1, 4),
            "maybe_none": None,
        }
    )
    if with_tensorlist:
        data["var_field"] = TensorList([torch.full((i + 1,), i) for i in range(n)])
    data.metadata = {"response_length": 4}
    return data


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv(EP_BALANCE_ENV, raising=False)
    assert not ep_balance_enabled()
    monkeypatch.setenv(EP_BALANCE_ENV, "1")
    assert ep_balance_enabled()
    monkeypatch.setenv(EP_BALANCE_ENV, "0")
    assert not ep_balance_enabled()


def test_real_lengths_and_stable_order():
    data = _make_batch([5, 9, 5, 12, 1])
    lengths = real_sequence_lengths(data)
    assert lengths.tolist() == [5, 9, 5, 12, 1]
    order = length_descending_order(lengths)
    # stable: the two 5s keep original relative order (index 0 before index 2)
    assert order.tolist() == [3, 1, 0, 2, 4]


def test_permute_moves_rows_as_units():
    data = _make_batch([5, 9, 5, 12], with_tensorlist=True)
    perm = torch.tensor([3, 1, 0, 2])
    out = permute_batch_rows(data, perm)
    for i, src in enumerate(perm.tolist()):
        assert out["sequences"][i, 0].item() == src
        assert int(out["attention_mask"][i].sum()) == int(data["attention_mask"][src].sum())
        assert out["advantages"][i, 0].item() == float(src)
        assert out["var_field"].tensors[i].tolist() == data["var_field"].tensors[src].tolist()
    assert out["maybe_none"] is None
    assert out.metadata is data.metadata
    # original untouched
    assert data["sequences"][0, 0].item() == 0


def test_worker_sort_reduces_barrier_objective_and_is_idempotent():
    # generation-order pairing puts a long with a short in every microbatch (mbs=2)
    lengths = [400, 30, 900, 25, 1500, 40, 700, 35]
    data = _make_batch(lengths)
    sorted_data, stats = sort_shard_rows_by_length(data, micro_batch_size=2)
    assert stats["applied"] is True
    got = real_sequence_lengths(sorted_data).tolist()
    assert got == sorted(lengths, reverse=True)
    # the barrier objective: sum of per-microbatch max lengths
    assert stats["mb_maxlen_sum_before"] == 400 + 900 + 1500 + 700
    assert stats["mb_maxlen_sum_before"] == microbatch_maxlen_sum(real_sequence_lengths(data), 2)
    assert stats["mb_maxlen_sum_after"] == 1500 + 700 + 40 + 30
    assert stats["mb_maxlen_sum_after"] < stats["mb_maxlen_sum_before"]
    # untrimmed shard width (the DYN_RECOMPUTE ledger input) is untouched
    assert sorted_data["sequences"].shape[1] == data["sequences"].shape[1]
    # idempotent: a second call reports already-sorted and returns the SAME object (no copy)
    again, stats2 = sort_shard_rows_by_length(sorted_data, micro_batch_size=2)
    assert again is sorted_data
    assert stats2["applied"] is False
    assert stats2["mb_maxlen_sum_before"] == stats2["mb_maxlen_sum_after"]


def test_worker_sort_declines_on_tiny_or_maskless_shards():
    data = _make_batch([7, 3])
    out, stats = sort_shard_rows_by_length(data, micro_batch_size=2)  # one microbatch: order moot
    assert out is data and stats is None
    data2 = _make_batch([7, 3, 9, 1])
    del data2["attention_mask"]
    out2, stats2 = sort_shard_rows_by_length(data2, micro_batch_size=2)
    assert out2 is data2 and stats2 is None


def test_stripe_order_alignment_with_contiguous_chunking():
    # 16 rows, DP=4 -> MeshDispatch gives rank r rows [4r, 4r+4). After striping, rank r's block
    # must hold sorted ranks {r, r+4, r+8, r+12}, each block descending (so the worker's
    # contiguous pairing needs no further sort).
    torch.manual_seed(0)
    lengths = torch.randint(10, 1500, (16,))
    perm = stripe_order_for_dp(lengths, dp_size=4)
    striped = lengths[perm]
    sorted_desc = torch.sort(lengths, descending=True).values
    for r in range(4):
        block = striped[4 * r : 4 * (r + 1)]
        expected = sorted_desc[r::4]
        assert torch.equal(block, expected)
        assert torch.equal(block, torch.sort(block, descending=True).values)
    # non-divisible declines
    assert stripe_order_for_dp(lengths[:15], dp_size=4) is None
    assert stripe_order_for_dp(lengths, dp_size=1) is None


def test_trainer_stripe_preserves_minibatch_membership():
    torch.manual_seed(1)
    n, dp = 32, 4
    lengths = torch.randint(20, 1400, (n,)).tolist()
    data = _make_batch(lengths)
    boundaries = [(0, 16), (16, 32)]
    out, stats = stripe_minibatch_rows_for_dp(data, boundaries, dp_size=dp)
    assert stats["applied_ranges"] == 2
    # membership: the set of row ids inside each boundary range is unchanged
    for start, end in boundaries:
        assert set(out["sequences"][start:end, 0].tolist()) == set(range(start, end))
    # the coarse skew objective shrinks (or at worst ties)
    assert stats["rank_token_spread_after"] <= stats["rank_token_spread_before"]

    # per-rank chunks inside each mini-batch now carry near-identical profiles: the i-th
    # microbatch's max-over-ranks cost must not exceed the unstriped one, summed over indices
    def barrier_objective(batch, start, end, mbs=2):
        ls = real_sequence_lengths(batch)[start:end].reshape(dp, -1)
        per_mb = ls.reshape(dp, -1, mbs).max(dim=2).values  # [dp, n_mb]
        return int(per_mb.max(dim=0).values.sum())

    for start, end in boundaries:
        assert barrier_objective(out, start, end) <= barrier_objective(data, start, end)


def test_trainer_stripe_declines_cleanly():
    data = _make_batch([10, 20, 30, 40, 50])
    # 5 rows, dp 4 -> not divisible -> decline whole range
    out, stats = stripe_minibatch_rows_for_dp(data, [(0, 5)], dp_size=4)
    assert out is data and stats is None
    # dp_size 1 -> decline
    out2, stats2 = stripe_minibatch_rows_for_dp(data, [(0, 5)], dp_size=1)
    assert out2 is data and stats2 is None


def test_determinism():
    torch.manual_seed(2)
    lengths = torch.randint(10, 1500, (32,)).tolist()
    a = stripe_minibatch_rows_for_dp(_make_batch(lengths), [(0, 32)], dp_size=8)[0]
    b = stripe_minibatch_rows_for_dp(_make_batch(lengths), [(0, 32)], dp_size=8)[0]
    assert torch.equal(a["sequences"], b["sequences"])


def test_skew_tax_shrinks_on_production_geometry():
    """End-to-end objective check on the shipped GLM geometry: 320 seqs, DP=8, mbs=2.

    Wall proxy per mini-batch = sum over microbatch indices of the max-over-ranks padded cost
    (every MoE layer is an all-rank barrier, so the slowest rank at index i paces everyone).
    """
    torch.manual_seed(3)
    dp, mbs, n = 8, 2, 320
    # GRPO group correlation: 64 prompts x 5 samples sharing a difficulty scale
    lengths = []
    for _ in range(64):
        base = float(torch.randint(60, 300, (1,)))
        scale = float(torch.empty(1).log_normal_(5.2, 0.55).clamp(max=1024))
        for _ in range(5):
            lengths.append(int(min(base + scale * float(torch.rand(1) + 0.5), 1536)))
    data = _make_batch(lengths)

    def wall(batch):
        ls = real_sequence_lengths(batch).reshape(dp, -1)  # contiguous chunks, like MeshDispatch
        per_mb = ls.reshape(dp, -1, mbs).max(dim=2).values
        return int(per_mb.max(dim=0).values.sum()), int(per_mb.sum(dim=1).max())

    wall_before, _ = wall(data)
    striped, stats = stripe_minibatch_rows_for_dp(data, [(0, n)], dp_size=dp)
    wall_after, slowest_rank_after = wall(striped)
    assert stats is not None
    assert wall_after < wall_before
    # after striping, waiting is nearly gone: the barrier-paced wall is within 2% of the
    # slowest rank's own compute (no rank waits long for another)
    assert wall_after <= int(1.02 * slowest_rank_after)
