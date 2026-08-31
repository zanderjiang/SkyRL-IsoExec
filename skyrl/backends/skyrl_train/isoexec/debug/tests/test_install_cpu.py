"""The hook table covers the registry, and every door in it resolves or is explained.

Pins the partition: every registry region is either hooked with named doors or listed in
``NOT_HOOKED`` with a reason. Also covers door installation shape and layer-context hooks.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[6]))

from skyrl.backends.skyrl_train.isoexec.debug import install, trace  # noqa: E402


@contextlib.contextmanager
def _env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    os.environ.update(kv)
    trace._reset_for_tests()
    install._wrapper_cache.clear()
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        trace._reset_for_tests()
        install._wrapper_cache.clear()


def _registry_regions():
    from skyrl.backends.skyrl_train.isoexec.core.registry_build import build_registry

    return set(build_registry().ops)


def test_every_registry_region_is_hooked_or_explained():
    regions = _registry_regions()
    cov = install.coverage()
    assert set(cov) == regions, set(cov) ^ regions
    hooked = {r for r, e in cov.items() if e["status"] in ("hooked", "partial")}
    not_hooked = {r for r, e in cov.items() if e["status"] == "not_hooked"}
    assert hooked | not_hooked == regions
    assert len(hooked) >= 20, sorted(regions - hooked)
    for region in not_hooked:
        assert len(cov[region]["note"] or "") > 40, region  # a reason, not a shrug
    for region in hooked:
        assert cov[region]["doors"], region


def test_partial_regions_declare_why():
    cov = install.coverage()
    for region, entry in cov.items():
        if entry["status"] == "partial":
            assert entry["note"], region


def test_doors_are_unique_and_well_formed():
    seen = set()
    for region, mod, attr, import_ok, kw in install.DOORS:
        key = (mod, attr)
        assert key not in seen, key  # two regions must never claim the same binding
        seen.add(key)
        assert "." in mod and attr
        assert isinstance(import_ok, bool) and isinstance(kw, dict)
        if mod.startswith("vllm."):
            assert not import_ok, f"{mod}: importing vLLM into a trainer process is not allowed"


def test_doors_resolve_where_their_module_is_present():
    """Import-only arming check: a door that resolves must produce a wrappable callable."""
    unresolved = [(r, m, a) for r, m, a, ok in install.smoke_install_report() if not ok]
    resolved = [(r, m, a) for r, m, a, ok in install.smoke_install_report() if ok]
    assert len(resolved) >= 25, unresolved
    for _r, mod, attr in unresolved:
        # only third-party / other-side doors may be missing in this process
        assert mod.startswith("vllm.") or ".runtimes.vllm." in mod or mod.startswith("megatron.") or ".ops." in mod


def test_install_arms_many_regions_and_is_idempotent():
    d = tempfile.mkdtemp(prefix="ix-inst-")
    try:
        with _env(**{trace.ENV_TRACE: d, trace.ENV_SIDE: "trainer"}):
            n = install.install_debug_hooks()
            assert n >= 20
            tr = trace.get_tracer()
            assert len(tr.regions_hooked) >= 15
            assert "mm" not in tr.regions_hooked  # high volume, opt-in only
            assert install.install_debug_hooks() == 0  # every door already wrapped
    finally:
        _uninstall_everything()
        shutil.rmtree(d, ignore_errors=True)


def _uninstall_everything():
    """Restore every door this test module may have wrapped, so the suite stays order-free."""
    for region, mod, attr, import_ok, _kw in install.DOORS:
        res = install._resolve(mod, attr, import_ok)
        if res is None:
            continue
        holder, name, cur = res
        raw = cur.__func__ if isinstance(cur, (staticmethod, classmethod)) else cur
        inner = getattr(raw, "_isoexec_debug_inner", None)
        if inner is not None:
            with contextlib.suppress(Exception):
                setattr(holder, name, inner)


def test_layer_context_needs_no_recording_of_its_own():
    """Layer-context hooks set context only and emit no trace records of their own."""
    d = tempfile.mkdtemp(prefix="ix-lay-")
    try:
        with _env(**{trace.ENV_TRACE: d, trace.ENV_SIDE: "engine"}):
            from types import SimpleNamespace

            layers = [SimpleNamespace(layer_number=i + 1, forward=lambda *a, **k: torch.zeros(2)) for i in range(3)]
            gpt = SimpleNamespace(decoder=SimpleNamespace(layers=layers))
            assert install.install_layer_context_hooks(gpt) == 3
            for lay in layers:
                lay.forward()
            trace.flush()
            files = list(pathlib.Path(d).glob("*.jsonl"))
            assert not files or not files[0].read_text().strip()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_install_is_inert_without_the_env():
    trace._reset_for_tests()
    assert install.install_debug_hooks() == 0
    assert install.install_layer_context_hooks(None) == 0


class _StaticDoor:
    """A real staticmethod door, the shape vLLM's ``Sampler.compute_logprobs`` has."""

    @staticmethod
    def compute(x):
        return x + 1


def test_staticmethod_door_stays_a_staticmethod():
    """Wrapping a staticmethod door keeps it a staticmethod, so it is not rebound as a method."""
    d = tempfile.mkdtemp(prefix="ix-static-")
    try:
        with _env(**{trace.ENV_TRACE: d, trace.ENV_SIDE: "trainer"}):
            holder, name, cur = install._resolve(__name__, "_StaticDoor.compute", True)
            assert holder is _StaticDoor and isinstance(cur, staticmethod)
            assert install._install_door("norms.rms", __name__, "_StaticDoor.compute", True, {})
            assert isinstance(_StaticDoor.__dict__["compute"], staticmethod)
            # live, and still takes only its own argument -- through the instance too
            assert _StaticDoor.compute(torch.zeros(2)).tolist() == [1.0, 1.0]
            assert _StaticDoor().compute(torch.zeros(2)).tolist() == [1.0, 1.0]
            trace.flush()
            recs = "".join(p.read_text() for p in pathlib.Path(d).glob("*.jsonl"))
            assert recs.count('"region":"norms.rms"') == 2
    finally:
        _StaticDoor.compute = staticmethod(lambda x: x + 1)
        shutil.rmtree(d, ignore_errors=True)


def test_layer_context_walks_a_virtual_pipeline_chunk_list():
    """A list of pipeline chunks (no ``.decoder``) is walked flat; ``layer_number`` is global
    across chunks, so the resulting indices are contiguous."""
    from types import SimpleNamespace

    d = tempfile.mkdtemp(prefix="ix-vp-")
    try:
        with _env(**{trace.ENV_TRACE: d, trace.ENV_SIDE: "trainer"}):

            def _chunk(numbers):
                layers = [SimpleNamespace(layer_number=n, forward=lambda *a, **k: None) for n in numbers]
                return SimpleNamespace(decoder=SimpleNamespace(layers=layers))

            # wrapped as the megatron worker holds them: DDP(Float16Module(GPTModel))
            def _wrapped(numbers):
                return SimpleNamespace(module=SimpleNamespace(module=_chunk(numbers)))

            chunks = [_wrapped([1, 2]), _chunk([3, 4])]
            assert install.install_layer_context_hooks(chunks) == 4
            got = [lay.forward._isoexec_debug_layer_ctx for c in chunks for lay in install._decoder_layers(c)]
            assert got == [0, 1, 2, 3]
            # ...and the single-module and no-model forms still behave.
            assert install.install_layer_context_hooks(_chunk([1])) == 1
            assert install.install_layer_context_hooks(object()) == 0
    finally:
        shutil.rmtree(d, ignore_errors=True)
