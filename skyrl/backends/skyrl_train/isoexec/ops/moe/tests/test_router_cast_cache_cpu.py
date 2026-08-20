"""CPU tests for the cached router fp32 weight cast (``moe_router_cast_cache``).

WHAT THIS GATES. ``RouterGatingLinearFunction.forward`` (megatron_core 0.19.0,
``moe_utils.py:1269-1276``) re-casts the ``[num_experts, hidden]`` bf16 router weight to fp32 on
every forward -- 4.24 us x 40 MoE layers = 169.8 us of every decode step, positionally attributed
in the 2026-08-13 traces_rowsfuse_1 graph -- although the weight changes only at a weight sync.

A CACHED COPY IS A CORRECTNESS RISK, NOT A PERFORMANCE ONE, and that is what this file is for. The
whole of the standing in-repo refusal (``moe_preamble_o12.py:167-181``, ``fused_outnorm.py`` K2a)
reduces to one sentence: a missed refresh serves LAST sync's router weights, silently, past a
forward-only gate that cannot see it. So the tests below are weighted accordingly:

  * THE MANDATORY ONE, :func:`test_new_weights_change_the_gating_output`: cache, write new weights,
    run the sync-seam invalidation, and assert the gating output MOVED. A cache that passed
    everything else and failed this one would be the exact bug the refusal predicted.
  * the buffer is refreshed IN PLACE and never reallocated (a captured decode graph holds its
    address), and a shape/device change RAISES instead of reallocating;
  * the cached path reproduces megatron's own formula bitwise;
  * every decline predicate fires, and grad-enabled always declines (the trainer's cast is
    grad-carrying; a detached cached copy would zero the router weight's gradient -- O4).

The install itself (``install_engine_router_cast_cache``) needs megatron's ``TopKRouter`` and a CUDA
weight, so it is covered by the GPU battery and the arm's ``[ISOEXEC-MOE-ROUTERCAST]`` banner; what
is exercised here is all of the logic that can be wrong.

Run: uv run --isolated --extra dev python -m pytest \
       skyrl/backends/skyrl_train/isoexec/ops/moe/tests/test_router_cast_cache_cpu.py -q
"""

import types

import pytest

torch = pytest.importorskip("torch")

from skyrl.backends.skyrl_train.isoexec.ops.moe import (  # noqa: E402
    moe_router_cast_cache as rc,
)


def _cast_cache_stats():
    # inlined: the public module keeps the state but not this test accessor
    return {
        "entries": len(rc._ENTRIES),
        "served": rc._served,
        "declined": rc._declined,
        "refreshed": rc._refreshed,
        "last_decline": rc._decline_reason,
    }


E, H, T = 16, 32, 5


class _FakeRouter:
    """A ``TopKRouter``'s shape, minus megatron: the four attributes the cached path reads."""

    def __init__(self, weight):
        self.weight = weight
        self.bias = None
        self.config = types.SimpleNamespace(moe_router_dtype="fp32")
        self.gating_calls = 0

    def _ix_castcache_orig_gating(self, inp):
        """VERBATIM megatron: router.py:104-107 -> moe_utils.py:1266-1279, the bias-None branch."""
        self.gating_calls += 1
        inp_shape = inp.shape
        flat = inp.view(-1, inp_shape[-1])
        out = torch.mm(flat.to(torch.float32), self.weight.to(torch.float32).t())
        return out.view(*inp_shape[:-1], -1)

    gating = rc._cached_cast_gating


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    rc.drop_all()
    monkeypatch.setenv(rc._ENV, "1")
    yield
    rc.drop_all()


def _router(seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(E, H, generator=g).to(torch.bfloat16)
    return _FakeRouter(w)


def _inp(seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(T, 1, H, generator=g).to(torch.bfloat16)


# ================================================================================================
# THE MANDATORY TEST: a weight sync must move the output
# ================================================================================================
def test_new_weights_change_the_gating_output():
    """Cache, sync new weights, assert the cast output CHANGED -- and equals the fresh eager answer."""
    r = _router()
    x = _inp()
    first = r.gating(x)
    assert _cast_cache_stats()["entries"] == 1

    g = torch.Generator().manual_seed(99)
    new_w = torch.randn(E, H, generator=g).to(torch.bfloat16)
    assert not torch.equal(new_w, r.weight)
    with torch.no_grad():
        r.weight.copy_(new_w)  # exactly what the engine's reapply does: an in-place H2D copy

    rc.invalidate_all()  # the moe_fused_weights.bump_sync_epoch seam
    second = r.gating(x)

    assert not torch.equal(first, second), "the cache served last sync's router weights"
    assert torch.equal(second, r._ix_castcache_orig_gating(x)), "post-sync output is not the eager answer"


def test_without_the_refresh_the_cache_is_stale_the_test_would_catch_it():
    """The negative control for the test above: skipping the seam DOES serve stale weights.

    Stated explicitly so nobody 'fixes' the mandatory test by making it pass vacuously -- if the
    cached path were secretly re-casting every call, this assertion would fail and the mandatory
    test above would be proving nothing.
    """
    r = _router()
    x = _inp()
    first = r.gating(x)
    with torch.no_grad():
        r.weight.copy_(torch.randn(E, H, generator=torch.Generator().manual_seed(7)).to(torch.bfloat16))
    assert torch.equal(r.gating(x), first)  # no invalidate_all() -> deliberately stale


def test_refresh_is_in_place_and_never_reallocates():
    """A captured decode graph holds the buffer's address; a new allocation would strand it."""
    r = _router()
    r.gating(_inp())
    entry = rc._ENTRIES[id(r)]
    ptr, buf = entry.buf.data_ptr(), entry.buf
    for _ in range(3):
        with torch.no_grad():
            r.weight.add_(1.0)
        rc.invalidate_all()
        assert rc._ENTRIES[id(r)].buf is buf and entry.buf.data_ptr() == ptr


def test_refresh_raises_on_a_shape_change():
    r = _router()
    r.gating(_inp())
    r.weight = torch.randn(E + 1, H).to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="GRAPH-UNSAFE"):
        rc.invalidate_all()


# ================================================================================================
# bitwise: the cached path is megatron's own formula
# ================================================================================================
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_cached_gating_is_bitwise_equal_to_megatron(seed):
    r = _router(seed)
    x = _inp(seed + 10)
    got = r.gating(x)
    ref = r._ix_castcache_orig_gating(x)
    assert got.shape == ref.shape == (T, 1, E)
    assert torch.equal(got.view(torch.int32), ref.view(torch.int32))


def test_the_cached_buffer_is_exactly_the_cast():
    r = _router(4)
    r.gating(_inp())
    assert torch.equal(rc._ENTRIES[id(r)].buf, r.weight.to(torch.float32))
    assert rc._ENTRIES[id(r)].buf.is_contiguous()


# ================================================================================================
# declines
# ================================================================================================
def test_flag_off_runs_megatron(monkeypatch):
    monkeypatch.setenv(rc._ENV, "0")
    r = _router()
    r.gating(_inp())
    assert r.gating_calls == 1
    assert _cast_cache_stats()["last_decline"] == "flag off"


def test_grad_enabled_always_declines():
    """The trainer's cast is grad-carrying; a detached cached copy would zero grad_weight (O4)."""
    r = _router()
    r.weight = r.weight.float().requires_grad_(True).to(torch.bfloat16).detach().requires_grad_(True)
    x = _inp().float().requires_grad_(True)
    r.gating(x)
    assert r.gating_calls == 1
    assert _cast_cache_stats()["last_decline"] == "grad enabled"


def test_a_bias_declines():
    r = _router()
    r.bias = torch.zeros(E)
    r.gating(_inp())
    assert r.gating_calls == 1


def test_non_fp32_router_dtype_declines():
    r = _router()
    r.config.moe_router_dtype = "fp64"
    r.gating(_inp())
    assert r.gating_calls == 1


def test_router_supported_declines_on_a_cpu_weight():
    """Every clause is a decline reason; the positive case needs CUDA and lives in the battery."""
    assert rc.router_supported(_router()) is False


def test_drop_all_releases_and_a_later_call_rebuilds():
    r = _router()
    r.gating(_inp())
    assert rc.drop_all() == 1
    assert _cast_cache_stats()["entries"] == 0
    r.gating(_inp())
    assert _cast_cache_stats()["entries"] == 1


def test_sync_seam_looks_us_up_by_sys_modules_not_by_import():
    """``bump_sync_epoch`` uses ``sys.modules.get(pkg + '.moe_router_cast_cache')``; that key must hit."""
    import sys

    assert sys.modules.get("skyrl.backends.skyrl_train.isoexec.ops.moe.moe_router_cast_cache") is rc
    assert rc.__package__ == "skyrl.backends.skyrl_train.isoexec.ops.moe"
