"""CPU tests for the offline trace comparator (``debug/compare.py``).

Covers first-divergence location, k-ladder magnitude, call->layer folding, case filtering,
shape mismatches, JSON caps, exit codes, and an end-to-end run through the real capture layer.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[6]))

from skyrl.backends.skyrl_train.isoexec.debug import compare, trace  # noqa: E402


def _write(d, name, recs):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def _rec(region, call, digest, *, case="x", layer=None, out="0", step=None, ladder=None, rank=0):
    r = {
        "v": compare.FORMAT_VERSION,
        "region": region,
        "case": case,
        "side": "?",
        "rank": rank,
        "rank_src": "env:RANK",
        "layer": layer,
        "step": step,
        "call": call,
        "out": out,
        "shape": [4, 4],
        "dtype": "bfloat16",
        "digest": digest,
    }
    if ladder is not None:
        r["ladder"] = ladder
    return r


def _synthetic_dirs(base):
    """Two traces over 12 layers x 2 regions; gdn.core agrees everywhere, moe.router diverges
    from layer 7 on, with a ladder that agrees at k<=2."""
    a, b = os.path.join(base, "a"), os.path.join(base, "b")
    ra, rb = [], []
    lad_ok = {"k6": "aa", "k4": "bb", "k2": "cc", "k0": "dd"}
    lad_bad = {"k6": "XX", "k4": "YY", "k2": "cc", "k0": "dd"}
    call = 0
    for layer in range(12):
        call += 1
        ra.append(_rec("gdn.core", call, f"g{layer:02d}"))
        rb.append(_rec("gdn.core", call, f"g{layer:02d}"))
        ra.append(_rec("moe.router", call, f"m{layer:02d}", ladder=lad_ok))
        diverged = layer >= 7
        rb.append(
            _rec(
                "moe.router",
                call,
                f"M{layer:02d}" if diverged else f"m{layer:02d}",
                ladder=lad_bad if diverged else lad_ok,
            )
        )
    _write(a, "trainer-h-1.jsonl", ra)
    _write(b, "engine-h-1.jsonl", rb)
    return a, b


def test_first_divergence_region_and_layer():
    base = tempfile.mkdtemp(prefix="isoexec-debug-cmp-")
    try:
        a, b = _synthetic_dirs(base)
        rep = compare.compare(compare.load_dir(a), compare.load_dir(b), layers=12)
        assert rep["regions"]["gdn.core"]["mismatched"] == 0
        assert rep["regions"]["gdn.core"]["matched"] == 12
        mr = rep["regions"]["moe.router"]
        assert (mr["compared"], mr["matched"], mr["mismatched"]) == (12, 7, 5)
        fd = rep["first_divergence"]
        assert fd["region"] == "moe.router" and fd["layer"] == 7
        assert fd["agree_k"] == 2
        assert "1e-01" in fd["magnitude"] or "2e-01" in fd["magnitude"]  # 2**-2 = 0.25
        text = compare.render_text(rep)
        assert "FIRST DIVERGENCE: region=moe.router rank=0 layer=7" in text
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_layer_fold_without_layers_flag():
    base = tempfile.mkdtemp(prefix="isoexec-debug-cmp-")
    try:
        a, b = _synthetic_dirs(base)
        rep = compare.compare(compare.load_dir(a), compare.load_dir(b))  # no --layers
        fd = rep["first_divergence"]
        assert fd["region"] == "moe.router" and fd["layer"] is None and fd["call_a"] == 8
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_case_filtering_and_unaligned():
    base = tempfile.mkdtemp(prefix="isoexec-debug-cmp-")
    try:
        a, b = os.path.join(base, "a"), os.path.join(base, "b")
        ra = [_rec("r", 1, "d1", case="trainer_score"), _rec("r", 2, "zz", case="trainer_fwd")]
        rb = [_rec("r", 1, "d1", case="engine_prefill"), _rec("r", 2, "d2", case="engine_prefill")]
        _write(a, "t.jsonl", ra)
        _write(b, "e.jsonl", rb)
        rep = compare.compare(compare.load_dir(a), compare.load_dir(b), case_a="trainer_score", case_b="engine_prefill")
        s = rep["regions"]["r"]
        assert (s["compared"], s["matched"], s["mismatched"], s["unaligned"]) == (1, 1, 0, 1)
        fd = rep["first_divergence"]
        assert fd["kind"] == "absent" and fd["absent_in"] == "A" and fd["call"] == 2
        assert rep["status"] == "divergent"
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_shape_mismatch_reported():
    base = tempfile.mkdtemp(prefix="isoexec-debug-cmp-")
    try:
        a, b = os.path.join(base, "a"), os.path.join(base, "b")
        rb0 = _rec("r", 1, "d1")
        rb0["shape"] = [8, 2]
        _write(a, "t.jsonl", [_rec("r", 1, "d1")])
        _write(b, "e.jsonl", [rb0])
        rep = compare.compare(compare.load_dir(a), compare.load_dir(b))
        fd = rep["first_divergence"]
        assert fd["kind"] == "shape/dtype" and "n/a" in fd["magnitude"]
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_missing_ladder_message():
    base = tempfile.mkdtemp(prefix="isoexec-debug-cmp-")
    try:
        a, b = os.path.join(base, "a"), os.path.join(base, "b")
        _write(a, "t.jsonl", [_rec("r", 1, "d1")])
        _write(b, "e.jsonl", [_rec("r", 1, "d2")])
        rep = compare.compare(compare.load_dir(a), compare.load_dir(b))
        assert "SKYRL_ISOEXEC_DEBUG_LADDER" in rep["first_divergence"]["magnitude"]
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_cli_exit_codes_and_json():
    base = tempfile.mkdtemp(prefix="isoexec-debug-cmp-")
    try:
        a, b = _synthetic_dirs(base)
        jout = os.path.join(base, "rep.json")
        rc = compare.main([a, b, "--layers", "12", "--json", jout])
        assert rc == 2
        rep = json.load(open(jout))
        assert rep["first_divergence"]["region"] == "moe.router"
        rc = compare.main([a, a])  # identical -> clean
        assert rc == 0
        rc = compare.main([a, b, "--regions", "gdn.core"])  # diverging region filtered out
        assert rc == 0
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_end_to_end_with_real_capture():
    base = tempfile.mkdtemp(prefix="isoexec-debug-e2e-")
    da, db = os.path.join(base, "trainer"), os.path.join(base, "engine")
    saved = {k: os.environ.get(k) for k in (trace.ENV_TRACE, trace.ENV_SIDE, trace.ENV_LADDER)}
    try:
        torch.manual_seed(7)
        layers = 4
        # values in [1, 1.4): the ~2**-6 relative bump below never crosses an exponent boundary
        outs = [(torch.rand(64, 32) * 0.4 + 1.0).bfloat16() for _ in range(layers)]
        outs_b = [o.clone() for o in outs]
        outs_b[2] = (outs_b[2].float() * (1.0 + 2.0**-6)).bfloat16()

        def run_side(d, side, tensors):
            os.environ[trace.ENV_TRACE] = d
            os.environ[trace.ENV_SIDE] = side
            os.environ[trace.ENV_LADDER] = "1"
            trace._reset_for_tests()
            src = {"i": -1}

            def region():
                src["i"] += 1
                return tensors[src["i"]]

            w = trace.wrap_region("moe.router", region)
            with torch.no_grad():
                for _ in range(layers):
                    w()
            trace.flush()

        run_side(da, "trainer", outs)
        run_side(db, "engine", outs_b)
        rep = compare.compare(
            compare.load_dir(da),
            compare.load_dir(db),
            case_a="trainer_score",
            case_b="engine",
            layers=layers,
        )
        s = rep["regions"]["moe.router"]
        assert (s["compared"], s["matched"], s["mismatched"]) == (4, 3, 1)
        fd = rep["first_divergence"]
        assert fd["layer"] == 2
        # the bump breaks the fine rungs but k0 (sign+exponent) agrees -> a bracket, not saturation
        assert fd["agree_k"] == 0 and fd["magnitude"].startswith("between ~2^-2 and 2^-0")
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        trace._reset_for_tests()
        shutil.rmtree(base, ignore_errors=True)


def _fully_divergent(n_per_region=300, regions=("gdn.core", "moe.router")):
    a, b = [], []
    for region in regions:
        for i in range(n_per_region):
            a.append(_rec(region, i + 1, f"{i:016x}", layer=i % 8))
            b.append(_rec(region, i + 1, f"{i + 1:016x}", layer=i % 8))
    return compare.compare(a, b)


def test_json_cap_trims_per_region_but_keeps_counts():
    """Capping trims per-region mismatch lists while leaving counts and the verdict intact."""
    rep = _fully_divergent()
    full = len(rep["divergences"])
    assert full == 600
    capped = compare.cap_report(rep, 25)
    assert len(capped["divergences"]) == 50  # 25 per region, two regions
    assert {d["region"] for d in capped["divergences"]} == {"gdn.core", "moe.router"}
    # the verdict is never trimmed
    assert capped["status"] == rep["status"]
    assert capped["first_divergence"] == rep["first_divergence"]
    assert capped["regions"] == rep["regions"]
    assert capped["regions"]["gdn.core"]["mismatched"] == 300
    assert capped["truncated"]["divergences_total"] == 600
    assert capped["truncated"]["divergences_dropped"] == {"gdn.core": 275, "moe.router": 275}
    assert rep["divergences"][:1] == capped["divergences"][:1]  # earliest kept, causal order


def test_json_cap_is_a_noop_when_small_or_disabled():
    rep = _fully_divergent(n_per_region=3)
    assert compare.cap_report(rep, 0) is rep
    assert compare.cap_report(rep, -1) is rep
    capped = compare.cap_report(rep, 100)
    assert len(capped["divergences"]) == len(rep["divergences"])
    assert capped["truncated"] is None


def test_json_cap_shrinks_the_serialized_file(tmp_path=None):
    import tempfile

    rep = _fully_divergent()
    d = tempfile.mkdtemp(prefix="ix-cap-")
    try:
        big = os.path.join(d, "big.json")
        small = os.path.join(d, "small.json")
        with open(big, "w") as f:
            json.dump(rep, f, indent=2)
        with open(small, "w") as f:
            json.dump(compare.cap_report(rep, 10), f, indent=2)
        assert os.path.getsize(small) < os.path.getsize(big) / 10
        back = json.load(open(small))
        assert back["status"] == "divergent" and back["first_divergence"]["region"] == "gdn.core"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_cli_json_cap_flag():
    d = tempfile.mkdtemp(prefix="ix-cli-")
    try:
        da, db = os.path.join(d, "a"), os.path.join(d, "b")
        _write(da, "trainer-r0-h-1.jsonl", [_rec("gdn.core", i + 1, f"{i:016x}") for i in range(40)])
        _write(db, "engine-r0-h-1.jsonl", [_rec("gdn.core", i + 1, f"{i + 1:016x}") for i in range(40)])
        out = os.path.join(d, "r.json")
        assert compare.main([da, db, "--json", out, "--json-max-per-region", "5"]) == 2
        rep = json.load(open(out))
        assert len(rep["divergences"]) == 5
        assert rep["regions"]["gdn.core"]["mismatched"] == 40
        assert rep["truncated"]["divergences_dropped"]["gdn.core"] == 35
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _run():
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f()
            print("PASS", n)
    print(f"\n{sum(1 for n in globals() if n.startswith('test_'))} passed")


if __name__ == "__main__":
    _run()
