"""The FORWARD-only token budget: `trainer.max_tokens_per_microbatch_forward`.

WHY IT EXISTS. The scoring (old-logprob) forward and the training step share one packer and, until
now, one budget. They do not share a memory profile: scoring runs under `torch.no_grad` and keeps no
activations, so a bin size the training step cannot afford is free there, while the cost that
dominates the phase (per-microbatch host launch + EP rendezvous) falls as `1 / bins`. The forward
budget lets the scoring phase pack larger WITHOUT touching the training grouping.

WHAT THIS FILE PINS, and what it deliberately does not:

  * the resolution rule (`resolve_forward_token_budget`) -- one function, read by the FSDP/base
    worker and by the Megatron worker, so the two backends cannot drift;
  * that the forward path uses the SHARED packer with a different capacity, not a second packing
    implementation (a second packer is the failure mode this whole design avoids);
  * the plumbing half of the bitwise obligation: bin membership changes with the budget, but the
    per-sample payload that comes back does NOT -- `torch.equal` on a round trip at two different
    budgets, against the unpacked reference.

  * NOT the kernel half. Whether the *model* returns bit-identical logprobs for a sequence scored
    alone vs scored alongside others is a property of the forward stack (MoE capacity pinned off,
    `_fixed_order_combine` per-token, M-invariant expert GEMMs, THD multi-segment attention, GDN
    cross-sequence invariance), and is provable only on GPU. That is the live acceptance: the arm's
    `policy/rollout_train_logprobs_abs_diff_mean` must stay in the low e-07 order it holds at the
    train budget. This file proves that IF the forward is grouping-invariant, the budget cannot
    corrupt the result on the way in or out.
"""

from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.workers.worker_utils import (
    SampleBasedBatchIterator,
    TokenBasedBatchIterator,
    get_microbatch_iterator,
    resolve_forward_token_budget,
)
from skyrl.train.config.config import TrainerConfig
from skyrl.train.dataset.bin_packing import make_seq_packer


def _make_batch(seq_lens, num_actions=4) -> TrainingInputBatch:
    """A batch whose rows are individually identifiable, so a reorder bug cannot hide."""
    batch_size = len(seq_lens)
    max_seq_len = max(seq_lens)
    sequences = torch.zeros((batch_size, max_seq_len), dtype=int)
    attention_mask = torch.zeros((batch_size, max_seq_len), dtype=int)
    for i, seq_len in enumerate(seq_lens):
        # Row i is filled with (i + 1): the payload identifies its own original index.
        sequences[i, :seq_len] = i + 1
        attention_mask[i, :seq_len] = 1

    data = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "action_log_probs": 0.4 * torch.ones((batch_size, num_actions)),
            "base_action_log_probs": 0.3 * torch.ones((batch_size, num_actions)),
            "values": 0.5 * torch.ones((batch_size, num_actions)),
            "returns": 0.5 * torch.ones((batch_size, num_actions)),
            "advantages": 0.6 * torch.ones((batch_size, num_actions)),
            "loss_mask": torch.ones((batch_size, num_actions), dtype=int),
            "response_mask": torch.ones((batch_size, num_actions), dtype=int),
        }
    )
    data.metadata = {"response_length": num_actions}
    return data


class TestResolveForwardTokenBudget:
    """One rule, and it must be inert unless the operator opts in."""

    def test_unset_inherits_the_training_budget(self):
        cfg = SimpleNamespace(max_tokens_per_microbatch=20480, max_tokens_per_microbatch_forward=-1)
        assert resolve_forward_token_budget(cfg) == 20480

    def test_set_overrides_the_training_budget(self):
        cfg = SimpleNamespace(max_tokens_per_microbatch=20480, max_tokens_per_microbatch_forward=81920)
        assert resolve_forward_token_budget(cfg) == 81920

    def test_packing_disabled_stays_disabled(self):
        cfg = SimpleNamespace(max_tokens_per_microbatch=-1, max_tokens_per_microbatch_forward=-1)
        assert resolve_forward_token_budget(cfg) == -1

    def test_config_without_the_field_resolves_to_the_training_budget(self):
        """Resumed runs and hand-built test configs predate the field; they must not raise."""
        cfg = SimpleNamespace(max_tokens_per_microbatch=16384)
        assert resolve_forward_token_budget(cfg) == 16384

    def test_none_is_treated_as_unset(self):
        cfg = SimpleNamespace(max_tokens_per_microbatch=16384, max_tokens_per_microbatch_forward=None)
        assert resolve_forward_token_budget(cfg) == 16384

    def test_default_config_is_inert(self):
        """A stock TrainerConfig must resolve to exactly its training budget -- shipping this
        knob changes no existing run."""
        cfg = TrainerConfig()
        assert cfg.max_tokens_per_microbatch_forward == -1
        assert resolve_forward_token_budget(cfg) == cfg.max_tokens_per_microbatch


class TestConfigValidation:
    def test_zero_is_refused(self):
        with pytest.raises(ValueError, match="max_tokens_per_microbatch_forward"):
            TrainerConfig(max_tokens_per_microbatch=20480, max_tokens_per_microbatch_forward=0)

    def test_negative_other_than_minus_one_is_refused(self):
        with pytest.raises(ValueError, match="max_tokens_per_microbatch_forward"):
            TrainerConfig(max_tokens_per_microbatch=20480, max_tokens_per_microbatch_forward=-2)

    def test_forward_budget_without_training_packing_is_refused(self):
        """De-coupling the scoring grouping from a sample-count training grouping is a
        configuration nobody has measured; refuse it rather than surprise the gate."""
        with pytest.raises(ValueError, match="requires token-based micro-batching"):
            TrainerConfig(max_tokens_per_microbatch=-1, max_tokens_per_microbatch_forward=81920)

    def test_matched_budgets_are_accepted(self):
        cfg = TrainerConfig(max_tokens_per_microbatch=20480, max_tokens_per_microbatch_forward=81920)
        assert resolve_forward_token_budget(cfg) == 81920


class TestSharedPacker:
    """The forward path must reach the SAME packer the train path reaches, with a different
    capacity -- never a second implementation."""

    def test_forward_bins_are_the_shared_balanced_packer_at_the_forward_capacity(self):
        seq_lens = [8, 3, 5, 6, 2, 7, 4, 9]
        batch = _make_batch(seq_lens)
        budget = 12

        iterator = get_microbatch_iterator(batch, micro_batch_size=1, max_tokens_per_microbatch=budget)
        assert isinstance(iterator, TokenBasedBatchIterator)

        expected = make_seq_packer("balanced", bin_capacity=budget).pack(seq_lens)
        assert iterator._microbatches == expected

    def test_a_larger_forward_budget_packs_strictly_fewer_bins(self):
        """The whole point of the knob: fewer, larger microbatches in the no_grad phase."""
        seq_lens = [8, 3, 5, 6, 2, 7, 4, 9]
        batch = _make_batch(seq_lens)

        train_bins = len(TokenBasedBatchIterator(batch, max_tokens_per_microbatch=12)._microbatches)
        forward_bins = len(TokenBasedBatchIterator(batch, max_tokens_per_microbatch=48)._microbatches)
        assert forward_bins < train_bins

    def test_disabled_budget_still_yields_the_sample_based_iterator(self):
        batch = _make_batch([8, 3, 5, 6])
        cfg = SimpleNamespace(max_tokens_per_microbatch=-1, max_tokens_per_microbatch_forward=-1)
        iterator = get_microbatch_iterator(
            batch, micro_batch_size=1, max_tokens_per_microbatch=resolve_forward_token_budget(cfg)
        )
        assert isinstance(iterator, SampleBasedBatchIterator)


class TestGroupingCannotMovePayload:
    """The plumbing half of the bitwise obligation, at every budget including 'unpacked'.

    A grouping-invariant forward is simulated by a per-sample function of that sample's own row
    only. If binning or the reorder ever mixed rows, or returned them out of order, these
    `torch.equal` assertions fail -- which is exactly the failure the gate could not distinguish
    from a kernel regression.
    """

    @staticmethod
    def _score(microbatch: TrainingInputBatch) -> TrainingInputBatch:
        """A stand-in forward: per-row, order-independent, no cross-sample state."""
        rows = microbatch["sequences"].to(torch.float64)
        per_sample = rows.sum(dim=1, keepdim=True) * 1.5 - rows.max(dim=1, keepdim=True).values
        out = TrainingInputBatch({"output": per_sample})
        out.metadata = microbatch.metadata
        return out

    def _run(self, batch: TrainingInputBatch, budget: int) -> torch.Tensor:
        iterator = get_microbatch_iterator(batch, micro_batch_size=1, max_tokens_per_microbatch=budget)
        outputs = [self._score(mb) for mb in iterator]
        return iterator.reorder_and_combine_batches(outputs)["output"]

    @pytest.mark.parametrize("budget", [-1, 9, 12, 20, 48, 4096])
    def test_every_budget_returns_bit_identical_per_sample_results(self, budget):
        seq_lens = [8, 3, 5, 6, 2, 7, 4, 9]
        batch = _make_batch(seq_lens)

        # Reference: one sample per microbatch, the historical (unpacked) grouping.
        reference = self._run(batch, budget=-1)
        packed = self._run(batch, budget=budget)

        assert packed.shape == reference.shape
        assert torch.equal(packed, reference), (
            f"budget={budget} changed per-sample results. Binning or the reorder moved a payload; "
            f"this is not a kernel difference."
        )

    def test_the_bins_really_did_change(self):
        """Guards the test above from passing vacuously: at these budgets the groupings differ."""
        seq_lens = [8, 3, 5, 6, 2, 7, 4, 9]
        batch = _make_batch(seq_lens)
        small = TokenBasedBatchIterator(batch, max_tokens_per_microbatch=9)._microbatches
        large = TokenBasedBatchIterator(batch, max_tokens_per_microbatch=48)._microbatches
        assert small != large
        assert max(len(b) for b in large) > 1

    def test_result_is_addressed_by_original_index(self):
        """The returned row i must be the score of input row i, not of whichever row landed
        first in some bin."""
        seq_lens = [8, 3, 5, 6, 2, 7, 4, 9]
        batch = _make_batch(seq_lens)
        packed = self._run(batch, budget=20)

        for i, seq_len in enumerate(seq_lens):
            row_value = float(i + 1)
            expected = row_value * seq_len * 1.5 - row_value
            assert packed[i].item() == pytest.approx(expected), f"row {i} came back as another sample's score"
