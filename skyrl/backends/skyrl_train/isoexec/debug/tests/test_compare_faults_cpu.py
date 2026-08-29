"""Fault-matrix guarantees for the comparator, on the two-sided pipeline in ``pipeline_fixture``.

One test per guarantee: causal (not alphabetical) first divergence and contamination marking,
shape/dtype divergences
that keep their step and print their shapes, absent records as divergences instead of shifted
alignment, rank-aware grouping and structural rank mismatch, honest sampling coverage, k-ladder
brackets that do not saturate into a magnitude claim, surfaced unrecordable outputs, segment row
localization, refusal of old trace formats, and a torch-free CLI.

Run (CPU only):
    python skyrl/backends/skyrl_train/isoexec/debug/tests/test_compare_faults_cpu.py
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[6]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pipeline_fixture as pf  # noqa: E402

from skyrl.backends.skyrl_train.isoexec.debug import compare, thash  # noqa: E402

_DEBUG_DIR = pathlib.Path(__file__).resolve().parents[1]


def _tmp():
    return tempfile.mkdtemp(prefix="isoexec-debug-fault-")


def _report(da, db, **kw):
    return compare.compare(
        compare.load_dir(da),
        compare.load_dir(db),
        man_a=compare.load_manifests(da),
        man_b=compare.load_manifests(db),
        **kw,
    )


def _rec(region, *, digest, ts, step=None, call=1, shape=(4, 4), dtype="bfloat16", rank=0, **extra):
    r = {
        "v": compare.FORMAT_VERSION,
        "region": region,
        "case": "x",
        "side": "?",
        "rank": rank,
        "rank_src": "env:RANK",
        "layer": 0,
        "step": step,
        "call": call,
        "out": "0",
        "shape": list(shape),
        "dtype": dtype,
        "digest": digest,
        "ts": ts,
    }
    r.update(extra)
    return r


def _write(d, name, recs):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


# -- causal ordering -------------------------------------------------------------------


def test_first_divergence_is_causal_not_alphabetical():
    """A fault in the pipeline-FIRST region must not be reported as collectives.row_parallel_ar,
    the alphabetically first contaminated region."""
    base = _tmp()
    try:
        da, db = pf.run_pair(base, fault={"kind": "add", "region": "norms.rms", "layer": 1, "step": 1, "delta": 1.0})
        rep = _report(da, db)
        fd = rep["first_divergence"]
        assert fd["region"] == "norms.rms", fd["region"]
        assert (fd["layer"], fd["step"], fd["rank"]) == (1, 1, 0)
        assert fd["contaminated"] is False
        text = compare.render_text(rep)
        assert "FIRST DIVERGENCE: region=norms.rms rank=0 layer=1" in text
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_downstream_regions_are_marked_contaminated():
    base = _tmp()
    try:
        da, db = pf.run_pair(base, fault={"kind": "add", "region": "norms.rms", "layer": 1, "step": 1, "delta": 1.0})
        rep = _report(da, db)
        later = [d for d in rep["divergences"] if d["contaminated"]]
        assert later, "downstream contamination must be visible"
        assert {d["region"] for d in later} >= {"mm.matmul", "gdn.core", "collectives.row_parallel_ar"}
        assert all(d["region"] != "norms.rms" or d["step"] != 1 for d in later[:1])
        text = compare.render_text(rep)
        assert "contaminated (after first divergence)" in text
        # the per-region table names the cause once and marks the consequences
        origin_rows = [ln for ln in text.splitlines() if "[ORIGIN]" in ln]
        assert len(origin_rows) == 1 and "layer=1 step=1" in origin_rows[0]
        assert text.count("[contaminated: after the first divergence]") == 5
    finally:
        shutil.rmtree(base, ignore_errors=True)


# -- shape/dtype divergences -----------------------------------------------------------


def test_shape_divergence_keeps_step_and_prints_both_shapes():
    base = _tmp()
    try:
        da, db = pf.run_pair(base, fault={"kind": "pad_row", "region": "gdn.core", "layer": 1, "step": 1})
        rep = _report(da, db)
        fd = rep["first_divergence"]
        assert fd["region"] == "gdn.core" and fd["kind"] == "shape/dtype"
        assert (fd["layer"], fd["step"]) == (1, 1)
        assert fd["case_a"] == "trainer_score" and fd["case_b"] == "engine"
        assert fd["a"]["shape"] == [64, 32] and fd["b"]["shape"] == [65, 32]
        assert "[64, 32] bfloat16 vs [65, 32] bfloat16" in compare.render_text(rep)
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_shape_divergence_does_not_outrank_an_earlier_value_divergence():
    """kind='shape/dtype' used to omit step, so _order_key read 0 -- the global minimum."""
    base = _tmp()
    try:
        a, b = os.path.join(base, "a"), os.path.join(base, "b")
        early = [_rec("aaa.region", digest="d1", ts=100.0, step=1)]
        late = [_rec("zzz.region", digest="d2", ts=200.0, step=9)]
        _write(a, "t.jsonl", early + late)
        _write(b, "e.jsonl", [_rec("aaa.region", digest="XX", ts=100.0, step=1), dict(late[0], shape=[4, 5])])
        rep = compare.compare(compare.load_dir(a), compare.load_dir(b))
        assert rep["first_divergence"]["region"] == "aaa.region"
        assert rep["first_divergence"]["kind"] == "value"
    finally:
        shutil.rmtree(base, ignore_errors=True)


# -- absent records --------------------------------------------------------------------


def test_absent_region_is_a_divergence_with_the_cuda_graph_hint():
    """The CUDA-graph decode case: a region never traced on the engine used to render as
    'NO DIVERGENCE ... exit 0' with a bare count."""
    base = _tmp()
    try:
        da, db = pf.run_pair(base, fault={"kind": "drop_region", "region": "collectives.row_parallel_ar"})
        rep = _report(da, db)
        assert rep["status"] == "divergent"
        fd = rep["first_divergence"]
        assert fd["kind"] == "absent" and fd["region"] == "collectives.row_parallel_ar"
        assert fd["absent_in"] == "B" and fd["side_absent"] == "engine"
        text = compare.render_text(rep)
        assert "enforce_eager" in text and "engine-side records are missing" in text
        assert "NO DIVERGENCE" not in text
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_missing_record_does_not_fabricate_value_divergences():
    """One record missing mid-stream used to shift every later pair, producing value
    divergences (with magnitude verdicts) for bits that were never compared."""
    base = _tmp()
    try:
        da, db = pf.run_pair(
            base,
            steps=4,
            fault={"kind": "drop_record", "region": "collectives.row_parallel_ar", "layer": 0, "step": 1},
        )
        rep = _report(da, db)
        kinds = [d["kind"] for d in rep["divergences"]]
        assert kinds == ["absent"], kinds
        fd = rep["first_divergence"]
        assert (fd["region"], fd["layer"], fd["step"]) == ("collectives.row_parallel_ar", 0, 1)
        assert fd["scope"] == "record" and fd["aligned_by"] == "step"
        assert rep["regions"]["collectives.row_parallel_ar"]["matched"] == 7  # 4 steps x 2 layers - 1
    finally:
        shutil.rmtree(base, ignore_errors=True)


# -- rank awareness --------------------------------------------------------------------


def test_two_ranks_per_side_align_by_rank():
    """Records from several processes in one directory used to align by pid sort order."""
    base = _tmp()
    try:
        da, db = pf.run_pair(base, ranks_a=(0, 1), ranks_b=(0, 1), steps=2)
        rep = _report(da, db)
        assert rep["status"] == "clean"
        assert rep["sides"]["a"]["ranks"] == [0, 1] and rep["sides"]["b"]["ranks"] == [0, 1]
        keys = compare._group(compare.load_dir(da), None, None, None)
        assert {k[3] for k in keys} == {0, 1}
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_rank_count_mismatch_is_structural_not_element_mismatches():
    base = _tmp()
    try:
        da, db = pf.run_pair(base, ranks_a=(0, 1), ranks_b=(0,), steps=2)
        rep = _report(da, db)
        fd = rep["first_divergence"]
        assert fd["kind"] == "rank_mismatch"
        assert fd["only_in_a"] == [1] and fd["only_in_b"] == []
        assert not [d for d in rep["divergences"] if d["kind"] == "value"]
        text = compare.render_text(rep)
        assert "STRUCTURAL DIVERGENCE: the two sides traced different rank sets." in text
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_first_divergence_names_the_rank():
    base = _tmp()
    try:
        da, db = pf.run_pair(
            base,
            ranks_a=(0, 1),
            ranks_b=(0, 1),
            steps=2,
            fault={"kind": "ulp", "region": "gdn.core", "layer": 1, "step": 1},
        )
        rep = _report(da, db)
        assert rep["first_divergence"]["region"] == "gdn.core"
        assert rep["first_divergence"]["rank"] in (0, 1)
        assert len(rep["origins"]) == 2  # the fault fires on both ranks; both are reported
        assert "rank=" in compare.render_text(rep).split("FIRST DIVERGENCE")[1]
    finally:
        shutil.rmtree(base, ignore_errors=True)


# -- sampling --------------------------------------------------------------------


def test_side_disjoint_sampling_is_inconclusive_not_divergent():
    """set_step wired on the trainer only + SAMPLE=2 selects disjoint records on the two sides;
    the comparator used to report the resulting misalignment as a numerical divergence."""
    base = _tmp()
    try:
        da, db = pf.run_pair(
            base,
            steps=4,
            set_step_b=False,
            env_a={"SKYRL_ISOEXEC_DEBUG_SAMPLE": "2"},
            env_b={"SKYRL_ISOEXEC_DEBUG_SAMPLE": "2"},
        )
        rep = _report(da, db)
        assert rep["status"] == "inconclusive"
        assert "set_step is not wired" in rep["disjoint_sampling"]
        text = compare.render_text(rep)
        assert "COMPARISON INCONCLUSIVE" in text and "NO DIVERGENCE" not in text
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_sampled_out_steps_are_reported_not_silently_clean():
    """A fault on a step neither side recorded is invisible; the report must say so."""
    base = _tmp()
    try:
        env = {"SKYRL_ISOEXEC_DEBUG_SAMPLE": "2"}
        da, db = pf.run_pair(
            base,
            steps=3,
            env_a=env,
            env_b=env,
            fault={"kind": "add", "region": "gdn.core", "layer": 0, "step": 1, "delta": 1.0},
        )
        rep = _report(da, db)
        assert rep["status"] == "clean"
        assert rep["sides"]["a"]["steps_not_observed"] == [1]
        assert rep["sides"]["b"]["steps_not_observed"] == [1]
        text = compare.render_text(rep)
        assert "not observed (sampled out at 1/2)" in text
        assert "invisible here, not absent" in text
    finally:
        shutil.rmtree(base, ignore_errors=True)


# -- k-ladder magnitude ----------------------------------------------------------------


def _verdict(a, b):
    la, lb = thash.digest_ladder(a), thash.digest_ladder(b)
    assert la.pop("full") != lb.pop("full")
    return compare._ladder_verdict({"ladder": la}, {"ladder": lb})


def test_ladder_bracket_resolves_one_ulp_fp32():
    """DEFAULT_LADDER stopped at k=6, so a 1e-07 difference was reported as '~2e-02'."""
    x = torch.randn(128, 64, dtype=torch.float32)
    y = x.clone()
    y.view(-1).view(torch.int32)[100] ^= 1
    true_rel = float((y - x).abs().max() / x.reshape(-1)[100].abs())
    agree, mag = _verdict(x, y)
    assert agree == 22 and mag.startswith("< 2^-22")
    assert true_rel < 2.0**-22 and true_rel > 2.0**-25
    assert "2e-02" not in mag


def test_ladder_bracket_contains_a_1e_3_perturbation():
    x = torch.randn(128, 64, dtype=torch.float32)
    x.view(-1)[100] = 1.0
    y = x.clone()
    y.view(-1)[100] = 1.001
    agree, mag = _verdict(x, y)
    assert mag.startswith("between ~2^-10 and 2^-6"), mag
    assert 2.0**-10 <= 1e-3 <= 2.0**-6  # the true relative error lies inside the bracket


def test_ladder_bracket_for_one_ulp_bf16_is_the_finest_rung():
    x = (torch.rand(128, 64, dtype=torch.float32) + 1.0).bfloat16()
    y = x.clone()
    y.view(-1).view(torch.int16)[100] ^= 1
    agree, mag = _verdict(x, y)
    assert agree == 6 and mag.startswith("< 2^-6")
    assert "finest rung recorded" in mag
    assert float((y.float() - x.float()).abs().max() / x.float().reshape(-1)[100]) < 2.0**-6


def test_reduction_order_magnitude_is_not_overstated():
    """The measured reduction-order fault (true max rel err 8.8e-03) used to render as
    'exponent-level (>~1 relative)'."""
    x = (torch.randn(1024, 512, generator=torch.Generator().manual_seed(2)) * 0.7).bfloat16()
    shards = [x * torch.tensor(c, dtype=torch.bfloat16) for c in (0.1, 0.2, 0.30000001, 0.4)]
    fwd, rev = shards[0].clone(), shards[::-1][0].clone()
    for s in shards[1:]:
        fwd = fwd + s
    for s in shards[::-1][1:]:
        rev = rev + s
    true_rel = float(((fwd.float() - rev.float()).abs() / fwd.float().abs().clamp_min(1e-30)).max())
    assert 5e-3 < true_rel < 2e-2, true_rel
    agree, mag = _verdict(fwd, rev)
    assert agree == 0
    assert true_rel <= 2.0**-agree  # the reported upper end is a real bound
    assert ">~1 relative" not in mag and "exponent-level" not in mag
    assert "the upper end is a bound" in mag


def test_fully_saturated_ladder_claims_no_magnitude_at_all():
    """When even k=0 differs the ladder bounds nothing, and the verdict must say exactly that."""
    x = (torch.randn(256, 64) * 0.7).bfloat16()
    agree, mag = _verdict(x, -x)
    assert agree is None
    assert mag.startswith("not bounded by the ladder")
    assert "NOT a claim of large error" in mag
    assert "relative" not in mag


def test_verdict_without_a_ladder_claims_no_magnitude():
    base = _tmp()
    try:
        da, db = pf.run_pair(base, fault={"kind": "ulp", "region": "gdn.core", "layer": 0, "step": 1})
        rep = _report(da, db)
        mag = rep["first_divergence"]["magnitude"]
        assert mag.startswith("unknown (no shared k-ladder")
        assert "SKYRL_ISOEXEC_DEBUG_LADDER=1" in mag
        assert "relative" not in mag
    finally:
        shutil.rmtree(base, ignore_errors=True)


# -- segment localization -------------------------------------------------------------


def test_segments_localize_rows_and_flag_whole_tensor_differences():
    """segment_digests was exported and unit-tested but unreachable from debug mode."""
    base = _tmp()
    env = {"SKYRL_ISOEXEC_DEBUG_SEGMENTS": "16"}
    try:
        da, db = pf.run_pair(
            base,
            env_a=env,
            env_b=env,
            fault={"kind": "add", "region": "gdn.core", "layer": 0, "step": 1, "delta": 1.0, "where": "first"},
        )
        rep = _report(da, db)
        fd = rep["first_divergence"]
        assert fd["region"] == "gdn.core"
        assert fd["segments"].startswith("1 of 4 row segments of dim 0 differ (first: segment 0, rows 0..15)")
        assert "segments: 1 of 4" in compare.render_text(rep)
    finally:
        shutil.rmtree(base, ignore_errors=True)

    base = _tmp()
    try:
        da, db = pf.run_pair(
            base,
            env_a=env,
            env_b=env,
            fault={"kind": "reduce_order", "region": "collectives.row_parallel_ar", "layer": 0, "step": 1},
        )
        rep = _report(da, db)
        assert "whole-tensor difference" in rep["first_divergence"]["segments"]
    finally:
        shutil.rmtree(base, ignore_errors=True)


# -- unrecordable outputs --------------------------------------------------------------


def test_unrecordable_outputs_are_surfaced():
    base = _tmp()
    try:
        a, b = os.path.join(base, "a"), os.path.join(base, "b")
        both = _rec("r", digest="d", ts=1.0, step=0)
        both.pop("digest")
        both["unrecordable"] = "TypeError: unsupported dtype torch.float8_e8m0fnu"
        one = _rec("s", digest="d", ts=2.0, step=0)
        one_bad = dict(one)
        one_bad.pop("digest")
        one_bad["unrecordable"] = "no tensor outputs (got dict)"
        _write(a, "t.jsonl", [both, one])
        _write(b, "e.jsonl", [dict(both), one_bad])
        rep = compare.compare(compare.load_dir(a), compare.load_dir(b))
        assert len(rep["unrecordable"]) == 2
        # both sides equally undigestable is not evidence of divergence; one side is
        fd = rep["first_divergence"]
        assert fd["kind"] == "unrecordable" and fd["region"] == "s"
        text = compare.render_text(rep)
        assert "could not be digested" in text and "no tensor outputs (got dict)" in text
    finally:
        shutil.rmtree(base, ignore_errors=True)


# -- format version and CLI ----------------------------------------------------------------


def test_old_format_traces_are_refused():
    base = _tmp()
    try:
        a = os.path.join(base, "a")
        old = _rec("r", digest="d", ts=1.0)
        old["v"] = 1
        _write(a, "t.jsonl", [old])
        try:
            compare.load_dir(a)
        except SystemExit as e:
            assert f"requires v{compare.FORMAT_VERSION}" in str(e)
        else:
            raise AssertionError("a v1 trace must be refused, not silently mis-read")
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_cli_runs_without_torch():
    """The offline comparator must work on a machine with no torch / CUDA / TransformerEngine."""
    base = _tmp()
    try:
        da, db = pf.run_pair(base, steps=1, layers=1, fault={"kind": "ulp", "region": "gdn.core"})
        poison = os.path.join(base, "poison")
        os.makedirs(poison, exist_ok=True)
        with open(os.path.join(poison, "torch.py"), "w") as f:
            f.write("raise ImportError('torch must not be imported by the offline comparator')\n")
        env = dict(os.environ, PYTHONPATH=poison)
        by_path = subprocess.run(
            [sys.executable, str(_DEBUG_DIR / "compare.py"), da, db], env=env, capture_output=True, text=True
        )
        assert by_path.returncode == 2, by_path.stderr
        assert "FIRST DIVERGENCE" in by_path.stdout
        env_m = dict(os.environ, PYTHONPATH=os.pathsep.join([poison, str(_DEBUG_DIR.parent)]))
        for argv in (["-m", "debug.compare"], ["-m", "debug"]):
            out = subprocess.run([sys.executable, *argv, da, db], env=env_m, capture_output=True, text=True)
            assert out.returncode == 2, (argv, out.stderr)
            assert "FIRST DIVERGENCE" in out.stdout
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _run():
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for n in names:
        globals()[n]()
        print("PASS", n)
    print(f"\n{len(names)} passed")


if __name__ == "__main__":
    _run()
