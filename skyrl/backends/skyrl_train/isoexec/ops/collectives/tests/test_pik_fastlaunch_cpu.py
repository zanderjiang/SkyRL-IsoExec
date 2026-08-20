"""CPU guarantees for pik's memoized Triton launcher (``pik/fastlaunch.py``).

WHAT CAN BE PINNED WITHOUT A GPU. The lever removes no arithmetic -- it removes Triton's per-call
RE-DERIVATION of a constant. So what has to hold is a state machine and a key algebra, and both are
CPU-testable in full:

  1. FLAG OFF IS THE UNTOUCHED CALL. ``launch(kern, grid, *a, **kw)`` is exactly
     ``kern[grid](*a, **kw)``: same object, same args, same kwargs, once.
  2. THE KEY IS A SUPERSET OF TRITON'S SPECIALIZATION INPUTS. Two calls collide only if the kernel
     identity, every non-tensor argument, the launch kwargs, and every tensor's (type, dtype,
     16-byte alignment) agree. Change any one and the key must move -- otherwise a shape class
     could inherit another's compiled kernel, which is the ONLY way this lever could move a bit.
  3. ADMISSION LAUNCHES ONCE, ON THE ORIGINAL PATH. The first call at a class issues through
     ``kern[grid]`` and nothing else; the pin is built from what THAT call resolved.
  4. THE FAST PATH REPLAYS THE RESOLVED ARGUMENT VECTOR with only the tensor slots re-bound, and
     ``served`` counts it. Engagement is a count, never a banner.
  5. FAIL-CLOSED. A pin that cannot be built, or that raises once built, is a loud PERMANENT
     per-class rejection back to the untouched call -- never an exception into the model.

Triton's own resolution (``binder`` -> ``compute_cache_key`` -> ``kernel_cache``) needs a live CUDA
driver, so it is stubbed here. Live-operand behavior requires a separate GPU qualification on the
target Triton runtime.

Run (CPU only):
    uv run --isolated --extra dev python -m pytest \
        skyrl/backends/skyrl_train/isoexec/ops/collectives/tests/test_pik_fastlaunch_cpu.py -q
"""

from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys

import pytest
import torch

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[7]))  # repo root

# pik is only importable under its canonical TOP-LEVEL name (codegen emits `from pik.gemm import
# ...` into generated kernel modules), and importing the package initialises an arch profile off a
# live GPU. ``fastlaunch`` deliberately has NO intra-pik imports -- only ``os`` and ``torch`` -- so
# a CPU test loads the file directly and exercises the module exactly as the runtime would.
_SRC = _HERE.parents[1] / "pik" / "fastlaunch.py"


def _fastlaunch_module():
    spec = importlib.util.spec_from_file_location("pik_fastlaunch_under_test", _SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fl(monkeypatch):
    """A fresh module with the flag ON and CUDA-capture probing neutralised."""
    monkeypatch.setenv("SKYRL_ISOEXEC_PIK_FASTLAUNCH", "1")
    mod = _fastlaunch_module()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return mod


# ----------------------------------------------------------------------------------------------
# a fake Triton surface: exactly the attributes fastlaunch touches, and nothing else
# ----------------------------------------------------------------------------------------------
class FakeCompiled:
    """Stands in for ``triton.compiler.CompiledKernel``: named, and callable through ``[grid]``."""

    def __init__(self, name="fake_kernel"):
        self.name = name
        self.launches: list = []

    def __getitem__(self, grid):
        def runner(*args, stream=None):
            self.launches.append((tuple(grid), list(args)))

        return runner


class FakeJIT:
    """Stands in for ``triton.runtime.jit.JITFunction``."""

    def __init__(self, argnames, compiled):
        self.argnames = argnames
        self.compiled = compiled
        self.pre_run_hooks: list = []
        self.device_caches = {0: ({}, {}, None, None, None)}
        self.calls = 0

    def add_pre_run_hook(self, hook):
        self.pre_run_hooks.append(hook)

    def __getitem__(self, grid):
        def call(*args, **kwargs):
            self.calls += 1
            for hook in self.pre_run_hooks:
                hook(*args, **kwargs)
            self.compiled[tuple(grid) if not callable(grid) else (1,)](*args)

        return call


def _install_resolver(mod, jit, compiled, argv_of):
    """Replace Triton's derivation with the fake one, recording how often it is asked."""
    seen = {"n": 0}

    def _resolve(_jit, cap_args, cap_kwargs, grid):
        seen["n"] += 1
        argv = argv_of(cap_args, cap_kwargs)
        g = grid(dict(zip(jit.argnames, argv))) if callable(grid) else grid
        return compiled, argv, tuple(g)

    mod._resolve = _resolve
    return seen


# ----------------------------------------------------------------------------------------------
# 1. flag off is the untouched call
# ----------------------------------------------------------------------------------------------
def test_flag_off_is_the_untouched_bracket_call(monkeypatch):
    monkeypatch.delenv("SKYRL_ISOEXEC_PIK_FASTLAUNCH", raising=False)
    mod = _fastlaunch_module()

    seen = []

    class K:
        def __getitem__(self, grid):
            return lambda *a, **kw: seen.append((grid, a, kw))

    t = torch.zeros(4)
    mod.launch(K(), (7,), t, 3, BLOCK=64)
    assert seen == [((7,), (t, 3), {"BLOCK": 64})]
    assert mod.fastlaunch_counts()["served"] == 0
    assert mod.fastlaunch_counts()["admitted"] == 0


# ----------------------------------------------------------------------------------------------
# 2. the key algebra -- the ONLY way this lever could ever move a bit
# ----------------------------------------------------------------------------------------------
def test_key_separates_every_specialization_input(fl):
    k = object()
    a = torch.zeros(64, dtype=torch.float32)
    base = fl._key(k, (a, 1024, 8), {"BLOCK": 1024, "num_warps": 4})

    assert fl._key(k, (a, 1024, 8), {"BLOCK": 1024, "num_warps": 4}) == base  # stable
    assert fl._key(object(), (a, 1024, 8), {"BLOCK": 1024, "num_warps": 4}) != base  # kernel
    assert fl._key(k, (a, 1025, 8), {"BLOCK": 1024, "num_warps": 4}) != base  # a scalar
    assert fl._key(k, (a, 1024, 8), {"BLOCK": 512, "num_warps": 4}) != base  # a kwarg
    assert fl._key(k, (a, 1024, 8), {"BLOCK": 1024, "num_warps": 8}) != base  # num_warps
    assert fl._key(k, (a.to(torch.float64), 1024, 8), {"BLOCK": 1024, "num_warps": 4}) != base  # dtype

    # ALIGNMENT: Triton specializes a pointer on divisibility by 16. A misaligned view must not
    # inherit an aligned view's pin -- this is the case that would silently run the wrong kernel.
    buf = torch.zeros(64, dtype=torch.float32)
    aligned = buf[0:16]
    while buf.data_ptr() % 16 != 0:  # pragma: no cover -- torch's allocator aligns, belt and braces
        buf = torch.zeros(64, dtype=torch.float32)
    misaligned = buf[1:17]
    assert (aligned.data_ptr() & 15) != (misaligned.data_ptr() & 15)
    assert fl._key(k, (aligned,), {}) != fl._key(k, (misaligned,), {})


def test_key_is_insensitive_to_the_tensor_object_itself(fl):
    """Two distinct tensors with the same dtype and alignment ARE one shape class -- that is the
    whole point (the pin re-binds tensor slots per call). Anything finer would never serve."""
    k = object()
    a = torch.zeros(64)
    b = torch.zeros(64)
    assert a is not b
    assert fl._key(k, (a, 5), {}) == fl._key(k, (b, 5), {})


# ----------------------------------------------------------------------------------------------
# 3./4. admission launches once on the original path; the fast path replays and counts
# ----------------------------------------------------------------------------------------------
def test_admission_then_serve(fl):
    compiled = FakeCompiled()
    jit = FakeJIT(["p", "o", "n", "BLOCK"], compiled)
    fl._underlying_jit = lambda kern: jit
    seen = _install_resolver(fl, jit, compiled, lambda a, kw: [a[0], a[1], a[2], kw["BLOCK"]])

    p0, o0 = torch.zeros(8), torch.zeros(8)
    fl.launch(jit, (3,), p0, o0, 8, BLOCK=1024)
    assert jit.calls == 1, "admission must issue exactly one launch, on the ORIGINAL path"
    assert seen["n"] == 1
    assert fl.fastlaunch_counts()["admitted"] == 1
    assert fl.fastlaunch_counts()["served"] == 0
    assert compiled.launches[-1][1][:2] == [p0, o0]

    p1, o1 = torch.zeros(8), torch.zeros(8)
    fl.launch(jit, (3,), p1, o1, 8, BLOCK=1024)
    assert jit.calls == 1, "a pinned call must NOT re-enter Triton's derivation path"
    assert seen["n"] == 1, "the resolution is memoized, not repeated"
    c = fl.fastlaunch_counts()
    assert c["served"] == 1 and c["pins"] == 1
    grid, argv = compiled.launches[-1]
    assert grid == (3,)
    assert argv[0] is p1 and argv[1] is o1, "tensor slots re-bound"
    assert argv[2:] == [8, 1024], "every non-tensor argument replayed verbatim"

    # a different shape class admits separately and does not disturb the first
    fl.launch(jit, (4,), torch.zeros(16), torch.zeros(16), 16, BLOCK=1024)
    assert fl.fastlaunch_counts()["admitted"] == 2


def test_callable_grid_is_resolved_once_and_replayed(fl):
    compiled = FakeCompiled()
    jit = FakeJIT(["p", "o", "n", "BLOCK"], compiled)
    fl._underlying_jit = lambda kern: jit
    _install_resolver(fl, jit, compiled, lambda a, kw: [a[0], a[1], a[2], kw["BLOCK"]])

    grid = lambda META: (META["n"] // META["BLOCK"],)  # noqa: E731
    fl.launch(jit, grid, torch.zeros(8), torch.zeros(8), 4096, BLOCK=1024)
    fl.launch(jit, grid, torch.zeros(8), torch.zeros(8), 4096, BLOCK=1024)
    assert compiled.launches[-1][0] == (4,)
    assert fl.fastlaunch_counts()["served"] == 1


# ----------------------------------------------------------------------------------------------
# 5. fail-closed, both halves
# ----------------------------------------------------------------------------------------------
def test_unpinnable_class_is_a_permanent_loud_rejection(fl, capsys):
    compiled = FakeCompiled()
    jit = FakeJIT(["p", "o"], compiled)
    fl._underlying_jit = lambda kern: jit
    # a bound tensor that is NOT one of the caller's positional args cannot be re-bound per call
    ghost = torch.zeros(8)
    _install_resolver(fl, jit, compiled, lambda a, kw: [a[0], ghost])

    fl.launch(jit, (1,), torch.zeros(8), 5)
    assert jit.calls == 1, "the launch it owed still happened, on the original path"
    assert "REFUSED" in capsys.readouterr().out
    assert fl.fastlaunch_counts()["rejected"] == 1

    fl.launch(jit, (1,), torch.zeros(8), 5)
    assert jit.calls == 2, "a rejected class stays on Triton's own path forever"
    assert fl.fastlaunch_counts()["fallback"] == 1
    assert fl.fastlaunch_counts()["served"] == 0


def test_fast_path_error_demotes_instead_of_raising(fl, capsys):
    compiled = FakeCompiled()
    jit = FakeJIT(["p", "o"], compiled)
    fl._underlying_jit = lambda kern: jit
    _install_resolver(fl, jit, compiled, lambda a, kw: [a[0], a[1]])

    fl.launch(jit, (1,), torch.zeros(8), torch.zeros(8))
    pin = next(v for v in fl._PINS.values() if isinstance(v, fl._Pin))

    def _boom(*a, **kw):
        raise RuntimeError("launcher exploded")

    pin.runner = _boom
    fl.launch(jit, (1,), torch.zeros(8), torch.zeros(8))  # must NOT raise
    assert "demoted after a fast-path error" in capsys.readouterr().out
    assert jit.calls == 2, "the demoted call fell back to the untouched path and still launched"


def test_a_real_kernel_error_is_not_swallowed(fl):
    class Angry:
        def __getitem__(self, grid):
            def call(*a, **kw):
                raise RuntimeError("illegal memory access")

            return call

        pre_run_hooks: list = []

        def add_pre_run_hook(self, hook):
            self.pre_run_hooks.append(hook)

    kern = Angry()
    fl._underlying_jit = lambda k: kern
    with pytest.raises(RuntimeError, match="illegal memory access"):
        fl.launch(kern, (1,), torch.zeros(8))


# ----------------------------------------------------------------------------------------------
# retention: a pin must not keep the admission-time OPERANDS alive
# ----------------------------------------------------------------------------------------------
def test_pin_retains_no_tensor_from_the_admitting_call(fl):
    """A pin is a template for the SCALAR half of the argument vector -- nothing more.

    This is a memory contract, not a speed one. ``_PINS`` has no size cap and no eviction, so any
    tensor a pin stores is retained for the life of the PROCESS. Two of pik's GEMM schedules hand
    the layer input activation into the launch, and the ``no_grad`` around it suppresses new graph
    nodes without clearing an existing ``grad_fn`` -- so a stored operand can drag a whole
    micro-batch's autograd graph across the scoring -> policy_train boundary and never release it.
    That is the 2026-08-15 production OOM.

    Nothing reads the stored tensors: every tensor position is covered by ``slots`` (a tensor that
    cannot be mapped REFUSES the pin) and is refilled from the live call before use. So the
    contract is simply that none survive.
    """
    compiled = FakeCompiled()
    jit = FakeJIT(["p", "o", "n", "BLOCK"], compiled)
    fl._underlying_jit = lambda kern: jit
    _install_resolver(fl, jit, compiled, lambda a, kw: [a[0], a[1], a[2], kw["BLOCK"]])

    p0, o0 = torch.zeros(8), torch.zeros(8)
    fl.launch(jit, (3,), p0, o0, 8, BLOCK=1024)
    pin = next(v for v in fl._PINS.values() if isinstance(v, fl._Pin))

    assert not any(
        isinstance(v, torch.Tensor) for v in pin.argv
    ), "a pin must hold no tensor from the admitting call -- _PINS is never evicted"
    # the scalar half IS still the memo, and the tensor positions are exactly the mapped slots
    assert pin.argv[2:] == [8, 1024], "every non-tensor argument must still be replayed verbatim"
    assert {i for i, _ in pin.slots} == {0, 1}
    assert all(pin.argv[i] is None for i, _ in pin.slots)

    # and the serve is unaffected: both tensor slots come from the LIVE call
    p1, o1 = torch.zeros(8), torch.zeros(8)
    fl.launch(jit, (3,), p1, o1, 8, BLOCK=1024)
    grid, argv = compiled.launches[-1]
    assert grid == (3,) and argv[0] is p1 and argv[1] is o1
    assert argv[2:] == [8, 1024]
    assert fl.fastlaunch_counts()["served"] == 1


def test_pin_does_not_keep_the_admitting_operand_alive(fl):
    """The retention contract, observed the only way that cannot be faked: through a weakref."""
    import gc
    import weakref

    compiled = FakeCompiled()
    jit = FakeJIT(["p", "o", "n", "BLOCK"], compiled)
    fl._underlying_jit = lambda kern: jit
    _install_resolver(fl, jit, compiled, lambda a, kw: [a[0], a[1], a[2], kw["BLOCK"]])

    p0, o0 = torch.zeros(8), torch.zeros(8)
    ref = weakref.ref(p0)
    fl.launch(jit, (3,), p0, o0, 8, BLOCK=1024)

    # FakeCompiled records every launch's argv, so drop that too -- the subject is _PINS.
    compiled.launches.clear()
    del p0, o0
    gc.collect()
    assert ref() is None, "the admitting operand is still reachable -- _PINS is retaining it"
    assert fl.fastlaunch_counts()["pins"] == 1, "and the pin itself must survive"


# ----------------------------------------------------------------------------------------------
# the flag is registered, default OFF, and reaches both actor channels
# ----------------------------------------------------------------------------------------------
def test_flags_registered_default_off_and_forwarded():
    from skyrl.backends.skyrl_train.isoexec.core.flags import ENGINE, FLAGS, TRAIN, actor_forwarding_tuple

    cat = {f.name: f for f in FLAGS}
    for name in ("SKYRL_ISOEXEC_PIK_FASTLAUNCH",):
        assert name in cat, f"{name} must be in the flag catalog"
        assert cat[name].default == "0", f"{name} must default OFF"
        assert name in actor_forwarding_tuple(TRAIN)
        assert name in actor_forwarding_tuple(ENGINE)
