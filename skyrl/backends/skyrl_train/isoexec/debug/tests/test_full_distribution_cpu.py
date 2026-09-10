"""CPU tests for the opt-in full raw-logprob distribution diagnostic."""

from __future__ import annotations

import contextlib
import os
import pathlib
import shutil
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[6]))

from skyrl.backends.skyrl_train.isoexec.debug import (  # noqa: E402
    compare,
    full_distribution,
    trace,
)


@contextlib.contextmanager
def _trace_env(
    path: str,
    side: str,
    *,
    sample: str = "1",
    arm: bool = True,
    weight_version: int | None = 0,
):
    names = (
        trace.ENV_TRACE,
        trace.ENV_SIDE,
        trace.ENV_SAMPLE,
        full_distribution.ENV,
        "SKYRL_ISOEXEC",
        "RANK",
        "LOCAL_RANK",
    )
    saved = {name: os.environ.get(name) for name in names}
    try:
        os.environ.update(
            {
                trace.ENV_TRACE: path,
                trace.ENV_SIDE: side,
                full_distribution.ENV: "1",
                "SKYRL_ISOEXEC": "1",
                "RANK": "0",
                trace.ENV_SAMPLE: sample,
            }
        )
        os.environ.pop("LOCAL_RANK", None)
        trace._reset_for_tests()
        full_distribution._reset_for_tests()
        tracer = trace.get_tracer()
        if arm:
            tracer.regions_hooked.add(full_distribution.REGION)
        tracer.write_manifest()
        if weight_version is not None:
            full_distribution.set_weight_version(weight_version)
        yield
        trace.flush()
    finally:
        trace._reset_for_tests()
        full_distribution._reset_for_tests()
        for name, value in saved.items():
            os.environ.pop(name, None)
            if value is not None:
                os.environ[name] = value


def _record(path: str, side: str, rows: torch.Tensor, histories, *, weight_version: int = 0, step=None):
    with _trace_env(path, side, weight_version=weight_version):
        if step is not None:
            trace.set_step(step)
        keys = [full_distribution.history_key(history) for history in histories]
        full_distribution._record_rows(rows, keys, case="trainer_score" if side == "trainer" else "engine")


def _compare(trainer_dir: str, engine_dir: str):
    return compare.compare(
        compare.load_dir(trainer_dir),
        compare.load_dir(engine_dir),
        man_a=compare.load_manifests(trainer_dir),
        man_b=compare.load_manifests(engine_dir),
    )


def test_non_sampled_vocab_difference_is_detected():
    root = tempfile.mkdtemp(prefix="isoexec-full-dist-")
    try:
        trainer_dir, engine_dir = os.path.join(root, "trainer"), os.path.join(root, "engine")
        trainer = torch.log(torch.tensor([[0.5, 0.25, 0.25]], dtype=torch.float32))
        engine = torch.log(torch.tensor([[0.5, 0.49, 0.01]], dtype=torch.float32))
        _record(trainer_dir, "trainer", trainer, [[11, 12]])
        _record(engine_dir, "engine", engine, [[11, 12]])
        report = _compare(trainer_dir, engine_dir)
        assert len(compare.load_dir(trainer_dir)[0]["digest"].split(":")) == 2
        assert trainer[0, 0] == engine[0, 0]  # a sampled-token-only check would pass
        assert report["status"] == "divergent"
        assert report["first_divergence"]["region"] == full_distribution.REGION
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_history_identity_survives_batch_permutation_and_duplicates():
    root = tempfile.mkdtemp(prefix="isoexec-full-dist-")
    try:
        trainer_dir, engine_dir = os.path.join(root, "trainer"), os.path.join(root, "engine")
        a = torch.tensor([[-0.1, -2.0], [-3.0, -0.2], [-0.1, -2.0]], dtype=torch.float32)
        histories_a = [[1, 2], [7, 8], [1, 2]]
        b = a[[1, 2, 0]]
        histories_b = [histories_a[i] for i in (1, 2, 0)]
        _record(trainer_dir, "trainer", a, histories_a)
        _record(engine_dir, "engine", b, histories_b)
        report = _compare(trainer_dir, engine_dir)
        assert report["status"] == "clean"
        assert report["regions"][full_distribution.REGION]["matched"] == 3
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_identical_duplicate_history_is_compared_as_a_multiset():
    root = tempfile.mkdtemp(prefix="isoexec-full-dist-")
    try:
        trainer_dir, engine_dir = os.path.join(root, "trainer"), os.path.join(root, "engine")
        rows = torch.tensor([[-0.1, -2.0], [-0.1, -2.0]], dtype=torch.float32)
        histories = [[1, 2], [1, 2]]
        _record(trainer_dir, "trainer", rows, histories)
        _record(engine_dir, "engine", rows.flip(0), histories)
        assert _compare(trainer_dir, engine_dir)["status"] == "clean"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_duplicate_history_with_multiple_values_is_never_clean():
    root = tempfile.mkdtemp(prefix="isoexec-full-dist-")
    try:
        trainer_dir, engine_dir = os.path.join(root, "trainer"), os.path.join(root, "engine")
        rows = torch.tensor([[-0.1, -2.0], [-3.0, -0.2]], dtype=torch.float32)
        histories = [[1, 2], [1, 2]]
        _record(trainer_dir, "trainer", rows, histories)
        _record(engine_dir, "engine", rows.flip(0), histories)
        report = _compare(trainer_dir, engine_dir)
        assert report["status"] == "divergent"
        assert report["first_divergence"]["kind"] == "within-side-variation"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_weight_version_not_local_step_is_the_full_distribution_identity():
    root = tempfile.mkdtemp(prefix="isoexec-full-dist-")
    try:
        trainer_dir, engine_dir = os.path.join(root, "trainer"), os.path.join(root, "engine")
        row = torch.tensor([[-0.1, -2.0]], dtype=torch.float32)
        _record(trainer_dir, "trainer", row, [[1]], weight_version=3, step=9)
        _record(engine_dir, "engine", row, [[1]], weight_version=3, step=1)
        assert _compare(trainer_dir, engine_dir)["status"] == "clean"

        shutil.rmtree(engine_dir)
        _record(engine_dir, "engine", row, [[1]], weight_version=4, step=9)
        report = _compare(trainer_dir, engine_dir)
        assert report["status"] == "inconclusive"
        assert report["full_distribution_coverage_gap"] is True
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_capture_requires_an_active_completed_weight_sync():
    root = tempfile.mkdtemp(prefix="isoexec-full-dist-")
    try:
        with _trace_env(root, "trainer", weight_version=None):
            with pytest.raises(RuntimeError, match="no active completed weight-sync transaction"):
                full_distribution._record_rows(torch.zeros(1, 4), ["history"], case="trainer_score")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_receiver_activates_weight_version_only_after_finish():
    from skyrl.backends.skyrl_train.inference_servers.layerwise_reload import (
        LayerwiseReloadWorkerMixin,
    )

    root = tempfile.mkdtemp(prefix="isoexec-full-dist-")
    try:
        with _trace_env(root, "engine", weight_version=0):
            worker = LayerwiseReloadWorkerMixin()
            worker.skyrl_start_weight_update(is_checkpoint_format=False, full_distribution_version=1)
            with pytest.raises(RuntimeError, match="no active completed weight-sync transaction"):
                full_distribution._record_rows(torch.zeros(1, 4), ["history"], case="engine")
            worker.skyrl_finish_weight_update()
            assert full_distribution._record_rows(torch.zeros(1, 4), ["history"], case="engine") == 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_missing_rows_on_either_side_are_inconclusive():
    root = tempfile.mkdtemp(prefix="isoexec-full-dist-")
    try:
        trainer_dir, engine_dir = os.path.join(root, "trainer"), os.path.join(root, "engine")
        row = torch.tensor([[-0.1, -2.0]], dtype=torch.float32)
        _record(trainer_dir, "trainer", row, [[1]])
        _record(engine_dir, "engine", torch.cat((row, row)), [[1], [9]])
        report = _compare(trainer_dir, engine_dir)
        assert report["status"] == "inconclusive"

        shutil.rmtree(engine_dir)
        _record(engine_dir, "engine", row, [[9]])
        report = _compare(trainer_dir, engine_dir)
        assert report["status"] == "inconclusive"
        assert report["first_divergence"]["side_absent"] == "engine"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_numeric_mismatch_is_not_hidden_by_a_missing_history():
    root = tempfile.mkdtemp(prefix="isoexec-full-dist-")
    try:
        trainer_dir, engine_dir = os.path.join(root, "trainer"), os.path.join(root, "engine")
        trainer = torch.tensor([[-0.1, -2.0], [-3.0, -0.2]], dtype=torch.float32)
        engine = torch.tensor([[-0.2, -1.8]], dtype=torch.float32)
        _record(trainer_dir, "trainer", trainer, [[1], [2]])
        _record(engine_dir, "engine", engine, [[1]])
        report = _compare(trainer_dir, engine_dir)
        assert report["status"] == "divergent"
        assert report["first_divergence"]["kind"] == "value"
        assert "FIRST DIVERGENCE" in compare.render_text(report)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_armed_but_unobserved_is_inconclusive():
    root = tempfile.mkdtemp(prefix="isoexec-full-dist-")
    try:
        for side in ("trainer", "engine"):
            path = os.path.join(root, side)
            with _trace_env(path, side):
                pass
        report = _compare(os.path.join(root, "trainer"), os.path.join(root, "engine"))
        assert report["status"] == "inconclusive"
        assert "no active trainer history" in report["required_observation"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_one_sided_arming_is_inconclusive():
    root = tempfile.mkdtemp(prefix="isoexec-full-dist-")
    try:
        trainer_dir, engine_dir = os.path.join(root, "trainer"), os.path.join(root, "engine")
        _record(trainer_dir, "trainer", torch.tensor([[-0.1, -2.0]]), [[1]])
        with _trace_env(engine_dir, "engine", arm=False):
            pass
        report = _compare(trainer_dir, engine_dir)
        assert report["status"] == "inconclusive"
        assert report["full_distribution_arming_mismatch"] is True
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_region_filter_can_exclude_full_distribution_requirement():
    records = [
        {
            "v": compare.FORMAT_VERSION,
            "region": "norms.rms",
            "case": "x",
            "side": "trainer",
            "rank": 0,
            "layer": None,
            "step": 1,
            "call": 1,
            "out": "0",
            "shape": [1],
            "dtype": "float32",
            "digest": "same",
        }
    ]
    manifest = [{"side": "trainer", "regions_hooked": [full_distribution.REGION]}]
    report = compare.compare(
        records,
        [{**records[0], "side": "engine"}],
        regions={"norms.rms"},
        man_a=manifest,
        man_b=[{"side": "engine", "regions_hooked": [full_distribution.REGION]}],
    )
    assert report["status"] == "clean"


def test_armed_side_without_rows_is_inconclusive():
    root = tempfile.mkdtemp(prefix="isoexec-full-dist-")
    try:
        trainer_dir, engine_dir = os.path.join(root, "trainer"), os.path.join(root, "engine")
        _record(trainer_dir, "trainer", torch.tensor([[-0.1, -2.0]]), [[1]])
        with _trace_env(engine_dir, "engine"):
            pass
        report = _compare(trainer_dir, engine_dir)
        assert report["status"] == "inconclusive"
        assert report["full_distribution_side_unobserved"] is True
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_unrecordable_full_row_is_never_clean():
    base = {
        "v": compare.FORMAT_VERSION,
        "region": full_distribution.REGION,
        "case": "trainer_score",
        "rank": 0,
        "layer": None,
        "step": 1,
        "call": 1,
        "out": "history:x",
        "shape": [4],
        "dtype": "float32",
    }
    good = {**base, "out": "history:good", "digest": "a"}
    bad = {**base, "unrecordable": "unsupported dtype"}
    manifest = [{"side": "trainer", "regions_hooked": [full_distribution.REGION]}]
    report = compare.compare(
        [{**good, "side": "trainer"}, {**bad, "side": "trainer"}],
        [{**good, "side": "engine"}, {**bad, "side": "engine"}],
        man_a=manifest,
        man_b=[{"side": "engine", "regions_hooked": [full_distribution.REGION]}],
    )
    assert report["status"] == "inconclusive"
    assert report["full_distribution_coverage_gap"] is True


def test_full_distribution_requires_unsampled_trace():
    root = tempfile.mkdtemp(prefix="isoexec-full-dist-")
    try:
        with _trace_env(root, "trainer", sample="2"):
            with pytest.raises(RuntimeError, match="DEBUG_SAMPLE=1"):
                full_distribution.arm("trainer")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_trainer_action_rows_use_history_and_loss_mask_not_batch_position():
    logits = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    sequences = torch.tensor([[0, 10, 11, 12, 13], [20, 21, 22, 23, 24]])
    attention = torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    loss_mask = torch.tensor([[1, 0], [0, 1]])
    action_logits, indices, keys = full_distribution.trainer_action_rows(
        logits, sequences, attention, loss_mask, num_actions=2
    )
    assert torch.equal(action_logits.reshape(-1, 3)[indices], torch.stack((logits[0, 2], logits[1, 3])))
    assert keys == [
        full_distribution.history_key([10, 11]),
        full_distribution.history_key([20, 21, 22, 23]),
    ]


def test_distribution_rows_require_full_fp32_vectors():
    with pytest.raises(RuntimeError, match="fp32"):
        full_distribution._record_rows(torch.ones(1, 4, dtype=torch.float16), ["a"], case="engine")
    with pytest.raises(RuntimeError, match=r"\[V\]"):
        full_distribution._record_rows(torch.ones(4, dtype=torch.float32), ["a"], case="engine")


def test_trainer_capture_bounds_full_distribution_chunks(monkeypatch):
    from skyrl.backends.skyrl_train.isoexec.ops.logprobs import rowinv

    seen = []

    def fake_full_logprobs(logits):
        seen.append(logits.shape[0])
        return logits.log_softmax(dim=-1, dtype=torch.float32).detach()

    monkeypatch.setattr(full_distribution, "_full_logprobs_for_trace", fake_full_logprobs)
    monkeypatch.setattr(rowinv, "stats", lambda: {"served": 1, "declined": 0, "agreed": True})
    logits = torch.randn(2, 5, 7)
    sequences = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    attention = torch.ones(2, 5, dtype=torch.int64)
    loss_mask = torch.ones(2, 3, dtype=torch.int64)

    root = tempfile.mkdtemp(prefix="isoexec-full-dist-")
    try:
        with _trace_env(root, "trainer"):
            recorded = full_distribution.record_trainer_action_distributions(
                logits=logits,
                sequences=sequences,
                attention_mask=attention,
                loss_mask=loss_mask,
                num_actions=3,
                temperature=1.0,
                packed=False,
                tp_size=1,
                pp_size=1,
                cp_size=1,
                dp_size=1,
                rowinv_before={"served": 0, "declined": 0, "agreed": None},
            )
        assert recorded == 6
        assert seen == [4, 2]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_rowinv_diagnostic_rejects_cpu_without_touching_census(monkeypatch):
    from skyrl.backends.skyrl_train.isoexec.ops.logprobs import rowinv

    monkeypatch.setenv("SKYRL_ISOEXEC", "1")
    before = rowinv.stats()
    with pytest.raises(RuntimeError, match="CUDA"):
        rowinv.diagnostic_full_logprobs(torch.zeros(1, 8))
    assert rowinv.stats() == before


def _fake_runner(*, tp=1, async_scheduling=False):
    input_batch = SimpleNamespace(
        num_reqs=2,
        temperature_cpu=np.array([1.0, 1.0], dtype=np.float32),
        num_tokens_no_spec=np.array([3, 2], dtype=np.int32),
        token_ids_cpu=np.array([[1, 2, 3, 0], [7, 8, 0, 0]], dtype=np.int32),
        is_token_ids=np.ones((2, 4), dtype=bool),
        req_ids=["a", "b"],
        request_lora_mapping=np.zeros(2, dtype=np.int64),
        sampling_metadata=SimpleNamespace(max_num_logprobs=1, logprob_token_ids=None),
    )
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=tp, pipeline_parallel_size=1, data_parallel_size=1)
        ),
        use_async_scheduling=async_scheduling,
        sampler=SimpleNamespace(logprobs_mode="raw_logprobs"),
        input_batch=input_batch,
        discard_request_mask=SimpleNamespace(np=np.array([False, True])),
        requests={"a": SimpleNamespace(mm_features=None), "b": SimpleNamespace(mm_features=None)},
    )


def test_engine_history_keys_exclude_partial_prefill_and_fail_closed():
    full_distribution._ENGINE_HASH_CACHE.clear()
    logits = torch.zeros(2, 4)
    runner = _fake_runner()
    keys = full_distribution.engine_history_keys(runner, logits, None, grammar_active=False)
    assert keys == (full_distribution.history_key([1, 2, 3]), None)
    runner.requests["a"] = SimpleNamespace(mm_features=None)
    runner.input_batch.token_ids_cpu[0, :3] = [4, 5, 6]
    keys = full_distribution.engine_history_keys(runner, logits, None, grammar_active=False)
    assert keys[0] == full_distribution.history_key([4, 5, 6])
    with pytest.raises(RuntimeError, match="speculative"):
        full_distribution.engine_history_keys(runner, logits, object(), grammar_active=False)
    with pytest.raises(RuntimeError, match="grammar"):
        full_distribution.engine_history_keys(runner, logits, None, grammar_active=True)
    with pytest.raises(RuntimeError, match="async"):
        full_distribution.engine_history_keys(_fake_runner(async_scheduling=True), logits, None, grammar_active=False)
    with pytest.raises(RuntimeError, match="TP=PP=DP=1"):
        full_distribution.engine_history_keys(_fake_runner(tp=2), logits, None, grammar_active=False)

    runner = _fake_runner()
    runner.input_batch.request_lora_mapping[0] = 1
    with pytest.raises(RuntimeError, match="LoRA"):
        full_distribution.engine_history_keys(runner, logits, None, grammar_active=False)

    runner = _fake_runner()
    runner.input_batch.sampling_metadata.max_num_logprobs = None
    with pytest.raises(RuntimeError, match="logprobs"):
        full_distribution.engine_history_keys(runner, logits, None, grammar_active=False)

    runner = _fake_runner()
    runner.requests["a"].mm_features = {"image": object()}
    with pytest.raises(RuntimeError, match="text-only"):
        full_distribution.engine_history_keys(runner, logits, None, grammar_active=False)


def test_engine_hooks_record_actual_compute_output_and_reset_context():
    class Sampler:
        logprobs_mode = "raw_logprobs"

        @staticmethod
        def compute_logprobs(logits):
            return logits.log_softmax(dim=-1, dtype=torch.float32)

    class Runner:
        def __init__(self):
            state = _fake_runner()
            self.__dict__.update(state.__dict__)
            self.sampler = Sampler()
            self.logits = torch.tensor([[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]])

        def _sample(self, logits, spec_decode_metadata):
            return self.sampler.compute_logprobs(logits)

        def _update_streaming_request(self, req_id, new_req_data):
            return new_req_data

        def sample_tokens(self, grammar_output):
            return self._sample(self.logits, None)

    root = tempfile.mkdtemp(prefix="isoexec-full-dist-")
    try:
        full_distribution._ENGINE_HASH_CACHE.clear()
        with _trace_env(root, "engine"):
            assert full_distribution._wrap_engine_classes(Runner, Sampler) == 4
            assert full_distribution._wrap_engine_classes(Runner, Sampler) == 0
            runner = Runner()
            output = runner.sample_tokens(None)
            assert torch.equal(output, runner.logits.log_softmax(dim=-1, dtype=torch.float32))
            assert full_distribution._ENGINE_HISTORY_KEYS.get() is None
            assert full_distribution._ENGINE_GRAMMAR_ACTIVE.get() is False
            assert (id(runner), "a") in full_distribution._ENGINE_HASH_CACHE
            runner._update_streaming_request("a", object())
            assert (id(runner), "a") not in full_distribution._ENGINE_HASH_CACHE
        records = compare.load_dir(root)
        assert len([record for record in records if record["region"] == full_distribution.REGION]) == 1

        with _trace_env(root, "engine"):
            runner = Runner()
            with pytest.raises(RuntimeError, match="grammar"):
                runner.sample_tokens(object())
            assert full_distribution._ENGINE_HISTORY_KEYS.get() is None
            assert full_distribution._ENGINE_GRAMMAR_ACTIVE.get() is False
    finally:
        shutil.rmtree(root, ignore_errors=True)
