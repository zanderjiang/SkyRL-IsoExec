"""CPU tests for the debug-mode capture layer (``debug/trace.py`` + ``debug/install.py``).

Covers zero-overhead-off, record/JSONL round-trip, sampling, re-entrancy, ladder capture, case
inference, the moe.router and gdn.core installers, layer context, rank stamping and the manifest.
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import shutil
import sys
import tempfile
import types
from types import SimpleNamespace

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[6]))

from skyrl.backends.skyrl_train.isoexec.debug import install, thash, trace  # noqa: E402

_ENVS = (
    trace.ENV_TRACE,
    trace.ENV_SIDE,
    trace.ENV_SAMPLE,
    trace.ENV_LADDER,
    trace.ENV_SEGMENTS,
    trace.ENV_RING,
    "RANK",
    "LOCAL_RANK",
)


class _env:
    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        self.saved = {k: os.environ.get(k) for k in _ENVS}
        for k in _ENVS:
            os.environ.pop(k, None)
        for k, v in self.kw.items():
            os.environ[k] = v
        trace._reset_for_tests()
        install._wrapper_cache.clear()
        return self

    def __exit__(self, *a):
        for k, v in self.saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        trace._reset_for_tests()
        install._wrapper_cache.clear()


def _tmpdir():
    return tempfile.mkdtemp(prefix="isoexec-debug-trace-")


def _read(d):
    recs = []
    for fp in sorted(pathlib.Path(d).glob("*.jsonl")):
        recs += [json.loads(l) for l in fp.read_text().splitlines() if l.strip()]
    return recs


def test_zero_overhead_off():
    with _env():  # env unset

        def f(x):
            return x + 1

        assert trace.wrap_region("r", f) is f
        assert trace.get_tracer() is None
        assert install.install_debug_hooks() == 0
        assert install.install_gdn_layer_hooks(SimpleNamespace(decoder=SimpleNamespace(layers=[]))) == 0


def test_record_roundtrip_and_digest():
    d = _tmpdir()
    try:
        with _env(**{trace.ENV_TRACE: d, trace.ENV_SIDE: "trainer"}):
            t1 = torch.randn(4, 8, dtype=torch.bfloat16)
            t2 = torch.randn(3, dtype=torch.float32)

            def region(x):
                return (t1, "not-a-tensor", t2)

            w = trace.wrap_region("moe.router", region)
            assert w is not region and getattr(w, "_isoexec_debug_region") == "moe.router"
            with torch.no_grad():
                w(0)
            trace.flush()
            recs = _read(d)
            assert len(recs) == 2
            assert [r["out"] for r in recs] == ["0", "2"]
            assert all(r["v"] == trace.FORMAT_VERSION for r in recs)
            r0 = recs[0]
            assert r0["region"] == "moe.router" and r0["side"] == "trainer"
            assert r0["case"] == "trainer_score"  # no_grad on the trainer side
            assert r0["shape"] == [4, 8] and r0["dtype"] == "bfloat16" and r0["call"] == 1
            assert r0["digest"] == f"{thash.tensor_digest(t1):016x}"
            assert recs[1]["digest"] == f"{thash.tensor_digest(t2):016x}"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_case_follows_grad_mode_and_ladder_env():
    d = _tmpdir()
    try:
        with _env(**{trace.ENV_TRACE: d, trace.ENV_SIDE: "trainer", trace.ENV_LADDER: "1"}):
            w = trace.wrap_region("r", lambda: torch.randn(5, dtype=torch.bfloat16))
            with torch.enable_grad():
                w()
            with torch.no_grad():
                w()
            trace.flush()
            recs = _read(d)
            assert [r["case"] for r in recs] == ["trainer_fwd", "trainer_score"]
            assert set(recs[0]["ladder"]) == {"k6", "k4", "k2", "k0"}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_sampling():
    d = _tmpdir()
    try:
        with _env(**{trace.ENV_TRACE: d, trace.ENV_SAMPLE: "3"}):
            w = trace.wrap_region("r", lambda: torch.zeros(2))
            for _ in range(7):
                w()
            trace.flush()
            recs = _read(d)
            assert [r["call"] for r in recs] == [1, 4, 7]
        # step-keyed sampling: every Nth forward, all calls within a sampled step recorded
        with _env(**{trace.ENV_TRACE: d, trace.ENV_SAMPLE: "2"}):
            w = trace.wrap_region("r2", lambda: torch.zeros(2))
            for step in range(4):
                trace.set_step(step)
                w()
                w()
            trace.flush()
            recs = [r for r in _read(d) if r["region"] == "r2"]
            assert [r["step"] for r in recs] == [0, 0, 2, 2]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_reentrancy_collapses_nested_regions():
    d = _tmpdir()
    try:
        with _env(**{trace.ENV_TRACE: d}):
            inner = trace.wrap_region("gdn.core", lambda: torch.ones(3))

            def outer_fn():
                return inner()

            outer = trace.wrap_region("gdn.core", outer_fn)
            outer()
            trace.flush()
            recs = _read(d)
            assert len(recs) == 1 and recs[0]["call"] == 1
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_wrap_is_idempotent_and_shared():
    d = _tmpdir()
    try:
        with _env(**{trace.ENV_TRACE: d}):
            f = lambda: torch.zeros(1)  # noqa: E731
            w1 = install._shared_wrap("r", f)
            w2 = install._shared_wrap("r", f)
            assert w1 is w2
            assert trace.wrap_region("r", w1) is w1
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _fake_megatron(topk_fn):
    mods = {}
    names = [
        "megatron",
        "megatron.core",
        "megatron.core.transformer",
        "megatron.core.transformer.moe",
        "megatron.core.transformer.moe.moe_utils",
        "megatron.core.transformer.moe.router",
        "megatron.core.transformer.moe.token_dispatcher",
    ]
    for n in names:
        mods[n] = types.ModuleType(n)
    for parent, child in (
        ("megatron", "core"),
        ("megatron.core", "transformer"),
        ("megatron.core.transformer", "moe"),
    ):
        setattr(mods[parent], child, mods[f"{parent}.{child}"])
    for leaf in ("moe_utils", "router", "token_dispatcher"):
        full = f"megatron.core.transformer.moe.{leaf}"
        setattr(mods["megatron.core.transformer.moe"], leaf, mods[full])
        mods[full].topk_routing_with_score_function = topk_fn
    return mods


def test_moe_router_install_on_faked_namespaces():
    d = _tmpdir()
    saved = {k: sys.modules.get(k) for k in list(sys.modules) if k.startswith("megatron")}
    try:
        with _env(**{trace.ENV_TRACE: d, trace.ENV_SIDE: "engine"}):
            probs, rmap = torch.rand(6, 4), torch.rand(6, 4) > 0.5

            def topk(logits, k):
                return probs, rmap

            for k in list(sys.modules):
                if k.startswith("megatron"):
                    del sys.modules[k]
            sys.modules.update(_fake_megatron(topk))
            assert install._install_moe_router() == 3
            assert install._install_moe_router() == 0  # idempotent
            from megatron.core.transformer.moe import (
                moe_utils,
                router,
                token_dispatcher,
            )

            assert (
                moe_utils.topk_routing_with_score_function
                is router.topk_routing_with_score_function
                is token_dispatcher.topk_routing_with_score_function
            )
            out = router.topk_routing_with_score_function(torch.rand(6, 4), 2)
            assert torch.equal(out[0], probs)
            trace.flush()
            recs = _read(d)
            assert {r["region"] for r in recs} == {"moe.router"}
            assert len(recs) == 2 and recs[0]["case"] == "engine"
    finally:
        for k in list(sys.modules):
            if k.startswith("megatron"):
                del sys.modules[k]
        sys.modules.update({k: v for k, v in saved.items() if v is not None})
        shutil.rmtree(d, ignore_errors=True)


_GDN_OPS_DOORS = sorted({a for r, m, a, _i, _k in install.DOORS if r == "gdn.core" and m.endswith(".gdn_ops")})


def test_gdn_core_install_on_real_gdn_ops():
    d = _tmpdir()
    from skyrl.backends.skyrl_train.isoexec.ops.gdn import gdn_ops

    originals = {n: getattr(gdn_ops, n) for n in _GDN_OPS_DOORS if hasattr(gdn_ops, n)}
    try:
        with _env(**{trace.ENV_TRACE: d, trace.ENV_SIDE: "trainer"}):
            n = install._install_gdn_core()
            assert n >= len(originals) > 0
            for name, orig in originals.items():
                cur = getattr(gdn_ops, name)
                assert getattr(cur, "_isoexec_debug_region") == "gdn.core"
                assert cur._isoexec_debug_inner is orig
                assert cur.__wrapped__ is orig  # transparent to installers that unwrap
            assert install._install_gdn_core() == 0  # idempotent
    finally:
        for name, orig in originals.items():
            setattr(gdn_ops, name, orig)
        shutil.rmtree(d, ignore_errors=True)


class _FakeMD:
    def __init__(self, nd, npf):
        self.num_decodes, self.num_prefills = nd, npf


@contextlib.contextmanager
def _fake_forward_context(md):
    """Stand in for vllm.forward_context so the honest decode/prefill signal can be tested."""
    mods = {}
    for name in ("vllm", "vllm.forward_context"):
        mods[name] = sys.modules.get(name)
    pkg = types.ModuleType("vllm")
    fc = types.ModuleType("vllm.forward_context")
    fc.get_forward_context = lambda: SimpleNamespace(attn_metadata=md)
    pkg.forward_context = fc
    sys.modules["vllm"], sys.modules["vllm.forward_context"] = pkg, fc
    try:
        yield
    finally:
        for name, old in mods.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def test_engine_case_from_forward_context():
    """The case label must come from the batch, not from a token-count heuristic."""
    d = _tmpdir()
    try:
        with _env(**{trace.ENV_TRACE: d, trace.ENV_SIDE: "engine"}):
            trace.get_tracer()
            for md, want in (
                ({"model.layers.0.gdn": _FakeMD(8, 0)}, "engine_decode"),
                ({"model.layers.0.gdn": _FakeMD(0, 3)}, "engine_prefill"),
                (_FakeMD(4, 2), "engine_mixed"),
            ):
                with _fake_forward_context(md):
                    assert install._case((), {}, None) == want
                    # forward_context wins over the structural cu_seqlens fallback
                    assert install._case((), {"cu_seqlens": torch.tensor([0, 1, 2, 3])}, None) == want
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_engine_case_structural_fallback_without_forward_context():
    d = _tmpdir()
    try:
        with _env(**{trace.ENV_TRACE: d, trace.ENV_SIDE: "engine"}):
            trace.get_tracer()
            assert install._case((), {"cu_seqlens": torch.tensor([0, 17, 40])}, None) == "engine_prefill"
            assert install._case((), {"cu_seqlens": torch.tensor([0, 1, 2, 3])}, None) == "engine_prefill"
            assert install._case((), {"cu_seqlens": None, "ssm_state": object()}, None) == "engine_decode"
            assert install._case((), {}, None) == "engine"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_layer_context_hooks_label_the_kernel_door():
    """Layer-context hooks stamp the layer index onto the kernel door's own record."""
    d = _tmpdir()
    try:
        with _env(**{trace.ENV_TRACE: d, trace.ENV_SIDE: "engine"}):
            out = torch.randn(7, 16, dtype=torch.bfloat16)
            door = trace.wrap_region("gdn.core", lambda: out)
            layers = []
            for ln in (1, 2):
                gdn = SimpleNamespace(_isoexec_state=object())
                lay = SimpleNamespace(self_attention=gdn, layer_number=ln)
                lay.forward = lambda *a, **k: door()
                layers.append(lay)
            gpt = SimpleNamespace(decoder=SimpleNamespace(layers=layers))
            assert install.install_layer_context_hooks(gpt) == 2
            assert install.install_layer_context_hooks(gpt) == 0  # idempotent
            layers[0].forward()
            layers[1].forward()
            trace.flush()
            recs = _read(d)
            assert [r["layer"] for r in recs] == [0, 1]
            assert [r["layer_src"] for r in recs] == ["module", "module"]
            assert all(r["region"] == "gdn.core" for r in recs)
            assert all(r["shape"] == [7, 16] for r in recs)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_call_order_layer_when_no_module_context():
    """Trainer fallback: a per-layer region gets a per-step call ordinal, and says so."""
    d = _tmpdir()
    try:
        with _env(**{trace.ENV_TRACE: d, trace.ENV_SIDE: "trainer"}):
            door = trace.wrap_region("gdn.core", lambda: torch.zeros(2))
            other = trace.wrap_region("collectives.tree_all_reduce", lambda: torch.zeros(2))
            trace.set_step(0)
            door(), door(), other()
            trace.set_step(1)
            door()
            trace.flush()
            recs = _read(d)
            gdn = [r for r in recs if r["region"] == "gdn.core"]
            assert [r["layer"] for r in gdn] == [0, 1, 0]
            assert {r["layer_src"] for r in gdn} == {"call_order"}
            rest = [r for r in recs if r["region"] != "gdn.core"]
            assert [(r["layer"], r["layer_src"]) for r in rest] == [(None, None)]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_ring_flushes_when_full():
    d = _tmpdir()
    try:
        with _env(**{trace.ENV_TRACE: d, trace.ENV_RING: "2"}):
            w = trace.wrap_region("r", lambda: torch.zeros(1))
            for _ in range(5):
                w()
            assert len(_read(d)) >= 4  # flushed without an explicit flush() call
            trace.flush()
            assert len(_read(d)) == 5
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _manifest(d):
    fps = sorted(pathlib.Path(d).glob("manifest-*.json"))
    assert len(fps) == 1, fps
    return json.loads(fps[0].read_text())


def test_rank_is_stamped_and_sourced():
    """Records and the manifest carry the rank and its source, falling back to pid."""
    d = _tmpdir()
    try:
        with _env(**{trace.ENV_TRACE: d, trace.ENV_SIDE: "engine", "RANK": "3"}):
            trace.wrap_region("r", lambda: torch.zeros(2))()
            trace.flush()
            recs = _read(d)
            assert [(r["rank"], r["rank_src"]) for r in recs] == [(3, "env:RANK")]
            man = _manifest(d)
            assert (man["rank"], man["rank_src"]) == (3, "env:RANK")
        with _env(**{trace.ENV_TRACE: d, trace.ENV_SIDE: "engine"}):  # no RANK anywhere
            trace.wrap_region("r2", lambda: torch.zeros(2))()
            trace.flush()
            r = [x for x in _read(d) if x["region"] == "r2"][0]
            assert r["rank"] == os.getpid() and r["rank_src"] == "pid"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_manifest_records_sampling_coverage():
    """The manifest records which steps were seen and recorded, so 'clean' and 'not observed'
    stay distinguishable."""
    d = _tmpdir()
    try:
        with _env(**{trace.ENV_TRACE: d, trace.ENV_SIDE: "trainer", trace.ENV_SAMPLE: "2"}):
            w = trace.wrap_region("r", lambda: torch.zeros(2))
            for step in range(5):
                trace.set_step(step)
                w()
            trace.flush()
            man = _manifest(d)
            assert man["v"] == trace.FORMAT_VERSION and man["sample"] == 2
            assert man["step_signal"] is True
            assert man["steps_seen"] == [0, 1, 2, 3, 4]
            assert man["steps_recorded"] == [0, 2, 4]
            assert man["records"] == 3 and man["rank_src"] == "pid"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_unrecordable_outputs_are_recorded_not_dropped():
    """Dict / None / nested / unsupported-dtype outputs are recorded, never dropped silently."""
    d = _tmpdir()
    try:
        with _env(**{trace.ENV_TRACE: d, trace.ENV_SIDE: "trainer"}):
            t = torch.zeros(3)
            trace.wrap_region("nested", lambda: (t, (t, t)))()
            trace.wrap_region("dictout", lambda: {"h": t})()
            trace.wrap_region("noneout", lambda: None)()
            trace.wrap_region("scalarout", lambda: 7)()
            trace.flush()
            by = {}
            for r in _read(d):
                by.setdefault(r["region"], []).append(r)
            assert [r["out"] for r in by["nested"]] == ["0", "1.0", "1.1"]
            assert [r["out"] for r in by["dictout"]] == ["h"]
            for region in ("noneout", "scalarout"):
                (r,) = by[region]
                assert "digest" not in r and "no tensor outputs" in r["unrecordable"]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_unsupported_dtype_becomes_unrecordable_record():
    d = _tmpdir()
    dtype = next(
        (
            getattr(torch, n)
            for n in ("float8_e8m0fnu", "bits8")
            if getattr(torch, n, None) is not None and getattr(torch, n) not in thash._DTYPE_TABLE
        ),
        None,
    )
    if dtype is None:
        return
    try:
        with _env(**{trace.ENV_TRACE: d, trace.ENV_SIDE: "trainer"}):
            trace.wrap_region("odd", lambda: torch.zeros(4).to(dtype))()
            trace.flush()
            (r,) = _read(d)
            assert "digest" not in r and "unsupported dtype" in r["unrecordable"]
            assert r["dtype"] == str(dtype).replace("torch.", "")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_segment_digests_wired_behind_env():
    """Segment digests are emitted only when the SEGMENTS env var is set."""
    d = _tmpdir()
    try:
        with _env(**{trace.ENV_TRACE: d, trace.ENV_SEGMENTS: "4"}):
            t = torch.randn(16, 8, dtype=torch.float32)
            trace.wrap_region("r", lambda: t)()
            trace.flush()
            (r,) = _read(d)
            assert r["seg_rows"] == 4
            assert r["segments"] == thash.segment_digests(t, rows_per_segment=4)
        with _env(**{trace.ENV_TRACE: d}):  # off by default
            trace.wrap_region("r2", lambda: torch.zeros(4, 2))()
            trace.flush()
            r = [x for x in _read(d) if x["region"] == "r2"][0]
            assert "segments" not in r
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_capture_safety_skips_and_counts():
    """A hook reached under CUDA-graph capture must never issue the digest's D2H copy."""
    d = _tmpdir()
    real = trace._capturing
    try:
        with _env(**{trace.ENV_TRACE: d, trace.ENV_SIDE: "engine"}):
            calls = []
            w = trace.wrap_region("gdn.core", lambda: (calls.append(1), torch.zeros(4))[1])
            trace._capturing = lambda: True
            out = w()
            assert out.shape == (4,)  # the wrapped call still runs; only the record is skipped
            trace.flush()
            assert _read(d) == []
            assert _manifest(d)["capture_skipped"] == 1
            trace._capturing = lambda: False
            w()
            trace.flush()
            assert len(_read(d)) == 1
            assert _manifest(d)["capture_skipped"] == 1
            assert len(calls) == 2
    finally:
        trace._capturing = real
        shutil.rmtree(d, ignore_errors=True)


def test_nested_different_regions_both_record():
    """Nested wraps of different regions both record; only same-region re-entry collapses."""
    d = _tmpdir()
    try:
        with _env(**{trace.ENV_TRACE: d}):
            inner = trace.wrap_region("norms.rms", lambda: torch.ones(3))
            outer = trace.wrap_region("gdn.core", lambda: (inner(), torch.zeros(2))[1])
            outer()
            trace.flush()
            assert {r["region"] for r in _read(d)} == {"gdn.core", "norms.rms"}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_region_allow_list():
    d = _tmpdir()
    try:
        with _env(**{trace.ENV_TRACE: d}):
            tr = trace.get_tracer()
            assert tr.wants("gdn.core") and not tr.wants("mm")  # mm is high volume, off by default
        with _env(**{trace.ENV_TRACE: d, trace.ENV_REGIONS: "+mm"}):
            assert trace.get_tracer().wants("mm") and trace.get_tracer().wants("gdn.core")
        with _env(**{trace.ENV_TRACE: d, trace.ENV_REGIONS: "gdn.core"}):
            tr = trace.get_tracer()
            assert tr.wants("gdn.core") and not tr.wants("moe.router")
        with _env(**{trace.ENV_TRACE: d, trace.ENV_REGIONS: "all"}):
            assert trace.get_tracer().wants("mm")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_segment_axis_is_recorded():
    d = _tmpdir()
    try:
        with _env(**{trace.ENV_TRACE: d, trace.ENV_SEGMENTS: "8"}):
            trace.wrap_region("gdn.core", lambda: torch.randn(1, 32, 2, 4, dtype=torch.bfloat16))()
            trace.flush()
            rec = _read(d)[0]
            assert rec["seg_axis"] == 1 and rec["seg_rows"] == 8 and len(rec["segments"]) == 4
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_wrapper_is_transparent_to_installer_probes():
    """Several installers read _isoexec_* markers off the live binding to pick a code path."""
    d = _tmpdir()
    try:
        with _env(**{trace.ENV_TRACE: d}):

            def door():
                return torch.zeros(1)

            door._isoexec_accepts_out_dtype = True
            door._isoexec_tiled = "abc"
            w = trace.wrap_region("moe.combine", door)
            assert w is not door
            assert w._isoexec_accepts_out_dtype is True and w._isoexec_tiled == "abc"
            assert w.__wrapped__ is door
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_out_fn_digests_a_void_doors_buffer():
    d = _tmpdir()
    try:
        with _env(**{trace.ENV_TRACE: d}):
            buf = torch.arange(6, dtype=torch.float32)

            def void_door(*, C):
                C.add_(1)
                return None

            w = trace.wrap_region("moe.epilogue", void_door, out_fn=install._out_kwarg("C"))
            w(C=buf)
            trace.flush()
            rec = _read(d)[0]
            assert rec["shape"] == [6] and "unrecordable" not in rec
            assert rec["digest"] == f"{thash.tensor_digest(buf):016x}"
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
