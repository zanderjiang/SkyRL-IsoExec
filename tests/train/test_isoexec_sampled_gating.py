"""The scoring-forward skip and the banner that advertises it must read ONE predicate.

Arm 6kv38f6e printed "[ISOEXEC-SAMPLED-GATING] INSTALLED ... every 10 steps" for its whole life
while `policy/scoring_forward_skipped` was 0.0 on every step: the banner's veto list had been
hand-copied from `_isoexec_sampled_gating_skip` and did not know about
`off_policy_correction.tis_ratio_type`, which the matched arm had pinned. ~944 s/step, found by a
wandb config diff. These tests exist so the next veto added to the skip cannot drift out of the
banner: both call `_isoexec_sampled_gating_static_vetoes`, and the skip must refuse for exactly the
configs that produce a non-empty list.

CPU only, no Ray: the two methods are exercised against a stub `self`, which is also the point --
they must not need a live trainer to decide anything.

Run: uv run --isolated --extra dev python -m pytest tests/train/test_isoexec_sampled_gating.py -q
"""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from skyrl.train.trainer import RayPPOTrainer


def _cfg(**over):
    base = {
        "trainer": {
            "algorithm": {
                "use_kl_in_reward": False,
                "off_policy_correction": {
                    "tis_ratio_type": None,
                    "sequence_mask_metric": None,
                    "outlier_token_is_threshold_low": None,
                    "outlier_token_is_threshold_high": None,
                    "token_mask_is_threshold_low": None,
                    "token_mask_is_threshold_high": None,
                },
            },
            "critic": {"model": {"path": None}},
        }
    }
    cfg = OmegaConf.create(base)
    for dotted, value in over.items():
        OmegaConf.update(cfg, dotted.replace("__", "."), value, merge=True)
    return cfg


def _stub(cfg, *, interval=10, step=5, ref_model=None):
    """A `self` carrying only what the two methods may legitimately read. The veto method is bound
    from the REAL class, so the skip below genuinely calls it rather than a test double."""
    ns = SimpleNamespace(
        cfg=cfg,
        has_critic=bool(cfg.trainer.critic.model.path),
        _isoexec_scoring_audit_interval=interval,
        ref_model=ref_model,
        global_step=step,
    )
    ns._isoexec_sampled_gating_static_vetoes = lambda: RayPPOTrainer._isoexec_sampled_gating_static_vetoes(ns)
    return ns


class _Batch:
    def __init__(self, rollout_logprobs):
        self._v = rollout_logprobs

    def get(self, key, default=None):
        return self._v if key == "rollout_logprobs" else default


_VETO_CASES = [
    ("clean", {}, 0),
    ("kl_in_reward", {"trainer__algorithm__use_kl_in_reward": True}, 1),
    ("tis", {"trainer__algorithm__off_policy_correction__tis_ratio_type": "token"}, 1),
    ("critic", {"trainer__critic__model__path": "/some/critic"}, 1),
]


@pytest.mark.parametrize("name, over, n_vetoes", _VETO_CASES)
def test_banner_vetoes_and_skip_agree(name, over, n_vetoes, monkeypatch):
    """One predicate, two readers: a non-empty veto list <=> the skip can never fire."""
    monkeypatch.setenv("SKYRL_ISOEXEC", "1")
    self_ = _stub(_cfg(**over))
    vetoes = RayPPOTrainer._isoexec_sampled_gating_static_vetoes(self_)
    assert len(vetoes) == n_vetoes, vetoes
    skipped = RayPPOTrainer._isoexec_sampled_gating_skip(self_, _Batch(object()))
    assert skipped is (n_vetoes == 0), (name, vetoes)


def test_isoexec_off_is_a_veto_the_banner_reports(monkeypatch):
    """The env guard is static too, so the banner must own it -- it was invisible before."""
    monkeypatch.delenv("SKYRL_ISOEXEC", raising=False)
    self_ = _stub(_cfg())
    vetoes = RayPPOTrainer._isoexec_sampled_gating_static_vetoes(self_)
    assert any("SKYRL_ISOEXEC" in v for v in vetoes)
    assert RayPPOTrainer._isoexec_sampled_gating_skip(self_, _Batch(object())) is False


@pytest.mark.parametrize(
    "field, value",
    [("interval", 1), ("step", 1), ("ref_model", object()), ("rollout", None)],
)
def test_dynamic_guards_still_refuse_without_a_static_veto(field, value, monkeypatch):
    """The non-static guards (interval, step 1, a ref model, missing rollout logprobs) are NOT in
    the banner's list by design -- they are per-step facts. They must still refuse."""
    monkeypatch.setenv("SKYRL_ISOEXEC", "1")
    kw = {"interval": 10, "step": 5, "ref_model": None}
    batch = _Batch(object())
    if field == "rollout":
        batch = _Batch(None)
    else:
        kw[field] = value
    self_ = _stub(_cfg(), **kw)
    assert RayPPOTrainer._isoexec_sampled_gating_static_vetoes(self_) == []
    assert RayPPOTrainer._isoexec_sampled_gating_skip(self_, batch) is False
