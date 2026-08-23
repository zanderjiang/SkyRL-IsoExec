"""CPU guarantees for the offline trace comparator (``debug/compare.py``).

Covers: correct first-divergence (region and layer) on synthetic traces built to diverge at a
known point, magnitude extraction from the k-ladder, call->layer folding via --layers, case
filtering, shape-mismatch reporting, exit codes, and an end-to-end run where traces are produced
by the real capture layer on real tensors.

Run (CPU only):
    python skyrl/backends/skyrl_train/isoexec/debug/tests/test_compare_cpu.py
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

from skyrl.backends.skyrl_train.isoexec.debug import compare, thash, trace  # noqa: E402


def _write(d, name, recs):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def _rec(region, call, digest, *, case="x", layer=None, out=0, step=None, ladder=None):
    r = {
        "v": 1,
        "region": region,
        "case": case,
        "side": "?",
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
        assert "FIRST DIVERGENCE: region=moe.router layer=7" in text
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
        rep = compare.compare(
            compare.load_dir(a), compare.load_dir(b), case_a="trainer_score", case_b="engine_prefill"
        )
        s = rep["regions"]["r"]
        assert (s["compared"], s["matched"], s["mismatched"], s["unaligned"]) == (1, 1, 0, 1)
        assert rep["first_divergence"] is None
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
        # engine reproduces trainer exactly except layer 2, off by ~2**-6 relative
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
        # ~2**-6 bump breaks the fine rungs; sign+exponent (k0) still agrees
        assert fd["agree_k"] == 0 and "k<=0" in fd["magnitude"]
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        trace._reset_for_tests()
        shutil.rmtree(base, ignore_errors=True)


def _run():
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f()
            print("PASS", n)
    print(f"\n{sum(1 for n in globals() if n.startswith('test_'))} passed")


if __name__ == "__main__":
    _run()
