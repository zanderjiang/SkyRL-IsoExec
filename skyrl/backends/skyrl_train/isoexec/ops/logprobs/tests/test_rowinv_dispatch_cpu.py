"""The ONE dispatch point for the rowinv logprob, and what it owes when rowinv cannot serve.

``_ix_logprobs_apply`` sits in the two ``from_parallel_logits_to_logprobs*`` wrappers, OUTSIDE any
autograd.Function. That placement is the load-bearing fact this file pins: an in-Function hook can
serve only ``inference_only=True`` (a Function applied inside another Function's forward cannot
carry its backward out), which would improve the reported scoring gate while the grad-bearing
training forward -- the objective actually optimized -- kept the old ATen schedule, and would split
scoring and training onto different functions. Asserted here, all CPU:

  * AN UNIMPORTABLE ROWINV IS BYTE-FOR-BYTE TODAY'S PATH: rowinv is the composed default, so
    the only remaining "not serving" state is the import shim. With it, the dispatch returns
    bitwise-identical tensors to calling the incumbent Functions directly, in both chunk regimes
    and both grad modes, and rowinv is never consulted (a consulting dispatch trips a planted
    bomb). The contract still names rowinv, so this state is not silent: the census stays
    served=0 and the engagement boundary refuses at the next weight sync;
  * BOTH GRAD MODES ARE SERVED: the dispatch consults rowinv for ``inference_only=True`` AND
    ``False``, and a served result carries gradient back to the input -- the exact thing the
    in-Function shape could not do;
  * THE CHUNK WALK IS THE IDENTITY ON A ROW-INDEPENDENT FUNCTION: chunked and unchunked dispatch
    agree bitwise. The stand-in used here is row-independent by construction (per-element ops and
    a fixed column loop); that the REAL kernel's function is row-independent is pinned separately
    -- the expression by ``test_rowinv_leaftree_cpu.py``, the kernel by ``rowinv_gpu.py`` -- so
    this test asserts exactly the slice the dispatch owns: slicing (native dtype through --
    rowinv widens in-kernel, exactly), target alignment, concatenation;
  * NO MID-TENSOR FUNCTION MIX: a decline on any chunk, a graph-free chunk under grad (the shape
    of rowinv's probe-failure fallback), or a census latch after the walk each abandon rowinv for
    the WHOLE call and the incumbent serves, bit-for-bit;
  * ONE DISPATCH POINT, structurally: an AST scan asserts both wrappers route through
    ``_ix_logprobs_apply`` and neither Function's forward references rowinv -- a future edit that
    reintroduces a second dispatch has to delete this test to land.

TP here is world=1 over a single-rank gloo group: the dispatch's TP behaviour is rowinv's own
(vote/decline unanimity lives inside rowinv and is gated by ``rowinv_tp_dist.py``); what the
dispatch owes TP is only that it calls or abandons identically on every rank, which follows from
its inputs (rowinv's verdicts) being TP-unanimous.

Run (CPU only):
    uv run --extra dev pytest skyrl/backends/skyrl_train/isoexec/ops/logprobs/tests/test_rowinv_dispatch_cpu.py -q
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
import tempfile
import textwrap
import types

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist  # noqa: E402

# -- import model_utils, stubbing megatron.core.parallel_state when megatron is absent -----------
_STUBBED = []
if importlib.util.find_spec("megatron") is None:
    for name in ("megatron", "megatron.core", "megatron.core.parallel_state"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
            _STUBBED.append(name)
    sys.modules["megatron"].core = sys.modules["megatron.core"]
    sys.modules["megatron.core"].parallel_state = sys.modules["megatron.core.parallel_state"]

from skyrl.backends.skyrl_train.distributed.megatron import (  # noqa: E402
    model_utils as M,
)

B, S, K = 2, 13, 32  # world=1: the shard IS the full vocabulary


@pytest.fixture(scope="module", autouse=True)
def _single_rank_group():
    """A world=1 gloo group so the incumbent Functions' all_reduce calls are real, then teardown."""
    owned = not dist.is_initialized()
    if owned:
        store = dist.FileStore(tempfile.mktemp(prefix="isoexec-rowinv-dispatch-"), 1)
        dist.init_process_group("gloo", store=store, rank=0, world_size=1)
    yield dist.group.WORLD
    if owned:
        dist.destroy_process_group()
    for name in _STUBBED:
        sys.modules.pop(name, None)


@pytest.fixture()
def _rowinv_unavailable(monkeypatch):
    """The import shim's state: the module is absent, so the dispatch may not consult it."""
    monkeypatch.setattr(M, "_ix_rowinv_available", lambda: False)


def _data(requires_grad: bool = False):
    g = torch.Generator().manual_seed(3)
    logits = torch.randn(B, S, K, generator=g, dtype=torch.float32).to(torch.bfloat16)
    logits.requires_grad_(requires_grad)
    target = torch.randint(0, K, (B, S), generator=g)
    return logits, target


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int32) if t.dtype == torch.float32 else t.view(torch.int16)


def _rowwise_standin(logits_chunk: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """A row-independent sampled-logprob stand-in: per-element ops plus a FIXED column loop.

    Runs in whatever dtype it is handed (the dispatch now passes the native bf16 chunk through).

    Row-independent by construction -- no op here can see how many rows sit alongside -- which is
    the property the real kernel earns via one program per (row, leaf). Differentiable, so the
    grad tests below exercise the dispatch's backward path end to end.
    """
    m = logits_chunk.max(dim=-1).values
    e = torch.exp(logits_chunk - m.unsqueeze(-1))
    s = torch.zeros_like(m)
    for j in range(e.shape[-1]):  # fixed order, invariant to batch and to the seq-dim chunk walk
        s = s + e[..., j]
    sampled = logits_chunk.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return (sampled - m) - torch.log(s)


def _fake_rowinv(monkeypatch, fn):
    """Install ``fn`` as the rowinv entrypoint on an importable rowinv."""
    monkeypatch.setattr(M, "_ix_rowinv_available", lambda: True)
    monkeypatch.setattr(M, "_ix_rowinv_sampled_logprobs", fn)
    monkeypatch.setattr(M, "_ix_rowinv_stats", lambda: {})


# =================================================================================================
# rowinv unimportable: byte-for-byte today's Functions, rowinv never consulted
# =================================================================================================
@pytest.mark.parametrize("inference_only", [True, False])
@pytest.mark.parametrize("chunk_size", [None, 5])
def test_unavailable_rowinv_is_byte_for_byte_the_incumbent(
    _rowinv_unavailable, monkeypatch, inference_only, chunk_size
):
    def bomb(*a, **k):  # noqa: ARG001
        raise AssertionError("rowinv consulted while the module is unimportable")

    monkeypatch.setattr(M, "_ix_rowinv_sampled_logprobs", bomb)

    logits, target = _data()
    via = M._ix_logprobs_apply(logits, target, 0, K, dist.group.WORLD, inference_only, chunk_size)
    if chunk_size is not None and chunk_size < S:
        direct = M.ChunkedDistributedLogprob.apply(
            logits, target.clone(), 0, K, chunk_size, dist.group.WORLD, inference_only
        ).contiguous()
    else:
        direct = M.DistributedLogprob.apply(logits, target.clone(), 0, K, dist.group.WORLD, inference_only).contiguous()
    assert torch.equal(_bits(via.float()), _bits(direct.float()))


# =================================================================================================
# both grad modes consulted and served; backward carries out of the dispatch
# =================================================================================================
def test_dispatch_consults_rowinv_for_both_grad_modes(monkeypatch):
    seen = []

    def declining(logits_chunk, target, *, vocab_start_index, vocab_end_index, group, src_dtype, reference):
        assert logits_chunk.dtype is torch.bfloat16 and logits_chunk.is_contiguous()  # native dtype through
        assert src_dtype is torch.bfloat16
        assert (vocab_start_index, vocab_end_index) == (0, K)
        assert target.shape == logits_chunk.shape[:-1]
        seen.append((tuple(logits_chunk.shape), torch.is_grad_enabled()))
        return None

    _fake_rowinv(monkeypatch, declining)
    logits, target = _data(requires_grad=True)

    out_score = M._ix_logprobs_apply(logits, target, 0, K, dist.group.WORLD, True, None)
    out_train = M._ix_logprobs_apply(logits, target, 0, K, dist.group.WORLD, False, None)
    assert len(seen) == 2, "rowinv must be consulted for inference_only=True AND False"
    # inference_only mirrors the incumbent contract: no backward is offered there, so the rowinv
    # attempt runs with grad off; the training forward keeps grad on.
    assert seen[0][1] is False and seen[1][1] is True
    # both declined -> both served by the incumbent, which still owes the training forward a
    # graph. (No requires_grad claim on out_score: the incumbent Function keeps a grad_fn under
    # inference_only=True too -- it skips only the saving -- and the dispatch mirrors it exactly.)
    assert out_train.requires_grad
    direct = M.DistributedLogprob.apply(logits, target.clone(), 0, K, dist.group.WORLD, True).contiguous()
    assert torch.equal(_bits(out_score.detach().float()), _bits(direct.detach().float()))


def test_grad_flows_through_a_served_dispatch(monkeypatch):
    def served(logits_chunk, target, *, reference, **kw):  # noqa: ARG001
        return _rowwise_standin(logits_chunk, target)

    _fake_rowinv(monkeypatch, served)
    for chunk_size in (None, 5):
        logits, target = _data(requires_grad=True)
        out = M._ix_logprobs_apply(logits, target, 0, K, dist.group.WORLD, False, chunk_size)
        assert out.requires_grad, "a served grad-mode dispatch must carry a graph"
        out.sum().backward()
        assert logits.grad is not None and bool(torch.isfinite(logits.grad).all())
        assert bool((logits.grad != 0).any()), "gradient must actually reach the input"
        # scoring mode: same function, no graph -- the incumbent's inference_only contract
        got = M._ix_logprobs_apply(logits.detach(), target, 0, K, dist.group.WORLD, True, chunk_size)
        assert not got.requires_grad


# =================================================================================================
# the chunk walk is the identity on a row-independent function
# =================================================================================================
def test_chunked_walk_equals_unchunked_bitwise(monkeypatch):
    def served(logits_chunk, target, *, reference, **kw):  # noqa: ARG001
        return _rowwise_standin(logits_chunk, target)

    _fake_rowinv(monkeypatch, served)
    logits, target = _data()
    whole = M._ix_logprobs_apply(logits, target, 0, K, dist.group.WORLD, True, None)
    for chunk_size in (4, 5, 13, 64):
        walked = M._ix_logprobs_apply(logits, target, 0, K, dist.group.WORLD, True, chunk_size)
        assert torch.equal(_bits(whole), _bits(walked)), f"chunk_size={chunk_size} changed the bits"
    # and the walk is exactly the stand-in applied whole: slicing/cat added nothing (the chunk
    # reaches the callee in its native dtype; nothing widens in the dispatch any more)
    direct = _rowwise_standin(logits, target)
    assert torch.equal(_bits(whole), _bits(direct))


# =================================================================================================
# fallback discipline: never a mid-tensor mix of functions
# =================================================================================================
def test_mid_walk_decline_abandons_rowinv_for_the_whole_call(monkeypatch):
    calls = {"n": 0}

    def first_serves_then_declines(logits_chunk, target, *, reference, **kw):  # noqa: ARG001
        calls["n"] += 1
        return _rowwise_standin(logits_chunk, target) if calls["n"] == 1 else None

    _fake_rowinv(monkeypatch, first_serves_then_declines)
    logits, target = _data()
    out = M._ix_logprobs_apply(logits, target, 0, K, dist.group.WORLD, True, 5)
    incumbent = M.ChunkedDistributedLogprob.apply(logits, target.clone(), 0, K, 5, dist.group.WORLD, True).contiguous()
    assert calls["n"] == 2, "the walk must stop consulting after the first decline"
    assert torch.equal(_bits(out.float()), _bits(incumbent.float())), "a partial walk must not leak rowinv chunks"


def test_graph_free_chunk_under_grad_falls_back_whole(monkeypatch):
    # The shape of rowinv's probe-failure fallback: it returns the REFERENCE's tensor, which
    # carries no graph. Under grad that would be a silent zero-gradient chunk; the dispatch must
    # treat it as a decline and let the incumbent serve the whole call, graph included.
    def graph_free(logits_chunk, target, *, reference, **kw):  # noqa: ARG001
        with torch.no_grad():
            return _rowwise_standin(logits_chunk, target)

    _fake_rowinv(monkeypatch, graph_free)
    logits, target = _data(requires_grad=True)
    out = M._ix_logprobs_apply(logits, target, 0, K, dist.group.WORLD, False, None)
    assert out.requires_grad, "the fallback must restore the incumbent's graph"
    direct = M.DistributedLogprob.apply(logits, target.clone(), 0, K, dist.group.WORLD, False).contiguous()
    assert torch.equal(_bits(out.detach().float()), _bits(direct.detach().float()))


def test_census_latch_after_the_walk_falls_back_whole(monkeypatch):
    # A probe failure on the LAST chunk returns incumbent bits for that chunk and latches the
    # census (stats()["agreed"] is False) without any later chunk to decline: the only mixed-bits
    # escape hatch, closed by the post-walk latch check.
    def served(logits_chunk, target, *, reference, **kw):  # noqa: ARG001
        return _rowwise_standin(logits_chunk, target)

    _fake_rowinv(monkeypatch, served)
    monkeypatch.setattr(M, "_ix_rowinv_stats", lambda: {"agreed": False})
    logits, target = _data()
    out = M._ix_logprobs_apply(logits, target, 0, K, dist.group.WORLD, True, 5)
    incumbent = M.ChunkedDistributedLogprob.apply(logits, target.clone(), 0, K, 5, dist.group.WORLD, True).contiguous()
    assert torch.equal(_bits(out.float()), _bits(incumbent.float()))


# =================================================================================================
# structure: one dispatch point, and it stays that way
# =================================================================================================
def test_single_dispatch_point_ast():
    """Both wrappers route through _ix_logprobs_apply; neither Function's forward knows rowinv.

    A future edit that adds a second dispatch -- a direct ``.apply`` back in a wrapper, or a
    rowinv hook back inside a Function -- has to delete this test to land.
    """

    def calls_in(fn) -> list:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Name):
                    names.append(f.id)
                elif isinstance(f, ast.Attribute):
                    base = f.value.id if isinstance(f.value, ast.Name) else ""
                    names.append(f"{base}.{f.attr}")
        return names

    for wrapper in (M.from_parallel_logits_to_logprobs, M.from_parallel_logits_to_logprobs_packed_sequences):
        names = calls_in(wrapper)
        assert names.count("_ix_logprobs_apply") == 1, f"{wrapper.__name__} must dispatch exactly once"
        assert (
            "DistributedLogprob.apply" not in names and "ChunkedDistributedLogprob.apply" not in names
        ), f"{wrapper.__name__} must not bypass the dispatch point"

    for fwd in (M.DistributedLogprob.forward, M.ChunkedDistributedLogprob.forward):
        src = inspect.getsource(fwd)
        assert "_ix_rowinv" not in src, "no in-Function rowinv hook: it can only serve inference_only=True"

    # and the dispatch point itself consults rowinv before either Function
    src = inspect.getsource(M._ix_logprobs_apply)
    assert "_ix_rowinv_available" in src and "_ix_try_rowinv_logprobs" in src
