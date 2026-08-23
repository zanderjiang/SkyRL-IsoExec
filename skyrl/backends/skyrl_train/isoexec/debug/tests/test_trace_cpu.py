"""CPU guarantees for the debug-mode capture layer (``debug/trace.py`` + ``debug/install.py``).

Covers: zero-overhead-off (identity passthrough, zero installs), record content and JSONL
round-trip, sampling, re-entrancy collapse, ladder capture, case inference, the moe.router
installer against a faked megatron namespace triple, the gdn.core installer against the real
``gdn_ops`` module, and layer-indexed engine GDN hooks.

Run (CPU only):
    python skyrl/backends/skyrl_train/isoexec/debug/tests/test_trace_cpu.py
"""

from __future__ import annotations

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

_ENVS = (trace.ENV_TRACE, trace.ENV_SIDE, trace.ENV_SAMPLE, trace.ENV_LADDER, trace.ENV_RING)


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
            assert [r["out"] for r in recs] == [0, 2]
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
            from megatron.core.transformer.moe import moe_utils, router, token_dispatcher

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


def test_gdn_core_install_on_real_gdn_ops():
    d = _tmpdir()
    from skyrl.backends.skyrl_train.isoexec.ops.gdn import gdn_ops

    originals = {n: getattr(gdn_ops, n) for n in install._GDN_TARGETS if hasattr(gdn_ops, n)}
    try:
        with _env(**{trace.ENV_TRACE: d, trace.ENV_SIDE: "trainer"}):
            n = install._install_gdn_core()
            assert n >= len(originals) > 0
            for name, orig in originals.items():
                cur = getattr(gdn_ops, name)
                assert getattr(cur, "_isoexec_debug_region") == "gdn.core"
                assert cur._isoexec_debug_inner is orig
            assert install._install_gdn_core() == 0  # idempotent
    finally:
        for name, orig in originals.items():
            setattr(gdn_ops, name, orig)
        shutil.rmtree(d, ignore_errors=True)


def test_gdn_case_inference():
    d = _tmpdir()
    try:
        with _env(**{trace.ENV_TRACE: d, trace.ENV_SIDE: "engine"}):
            trace.get_tracer()
            decode = {"cu_seqlens": torch.tensor([0, 1, 2, 3])}
            prefill = {"cu_seqlens": torch.tensor([0, 17, 40])}
            assert install._gdn_case((), decode, None) == "engine_decode"
            assert install._gdn_case((), prefill, None) == "engine_prefill"
            assert install._gdn_case((), {}, None) == "engine"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_gdn_layer_hooks_record_layer_index():
    d = _tmpdir()
    try:
        with _env(**{trace.ENV_TRACE: d, trace.ENV_SIDE: "engine"}):
            out = torch.randn(7, 16, dtype=torch.bfloat16)
            layers = []
            for ln in (1, 2):
                gdn = SimpleNamespace(_isoexec_state=object(), forward=lambda *a, **k: out)
                layers.append(SimpleNamespace(self_attention=gdn, layer_number=ln))
            layers.append(SimpleNamespace(self_attention=None))  # non-GDN layer skipped
            gpt = SimpleNamespace(decoder=SimpleNamespace(layers=layers))
            assert install.install_gdn_layer_hooks(gpt) == 2
            assert install.install_gdn_layer_hooks(gpt) == 0  # idempotent
            layers[0].self_attention.forward()
            layers[1].self_attention.forward()
            trace.flush()
            recs = _read(d)
            assert [r["layer"] for r in recs] == [0, 1]
            assert all(r["region"] == "gdn.core" for r in recs)
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


def _run():
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f()
            print("PASS", n)
    print(f"\n{sum(1 for n in globals() if n.startswith('test_'))} passed")


if __name__ == "__main__":
    _run()
