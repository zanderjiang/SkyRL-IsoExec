"""CPU tests for the packed-sequence host-read memo (``packed_meta_cache``).

The memo must be a host-work change only: every memoized read returns exactly what the expression
it replaces returns, the key (``id(cu) + cu._version``, pinned by a strong reference) invalidates on
any in-place write, and the patched megatron bodies stay equivalent to the upstream ones.
"""

import sys
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")

from skyrl.backends.skyrl_train.isoexec.ops.gdn import (  # noqa: E402
    packed_meta_cache as pmc,  # noqa: E402
)


def _reset_packed_meta_census():
    # inlined: the public module keeps the counters but not this test accessor
    pmc._STATS.update(served=0, built=0, declined=0)


BATCHES = [
    [96, 64, 137, 201],  # the census harness's packed row: ragged, none chunk-aligned
    [64],  # single sequence
    [1, 1, 1],  # degenerate
    [512, 512],  # uniform
    [7, 129, 3, 64, 1000],  # very ragged
]


def _cu(lens, dtype=torch.int32):
    return torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=dtype)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("SKYRL_ISOEXEC_PACKED_META_CACHE", raising=False)
    monkeypatch.delenv("SKYRL_ISOEXEC_GDN_VALIDATE_ONCE", raising=False)
    pmc._CACHE.clear()
    _reset_packed_meta_census()
    yield
    pmc._CACHE.clear()
    _reset_packed_meta_census()


# ---------------------------------------------------------------------------------------------
# 1. exactness -- the memo returns what the expression it replaces returns
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("lens", BATCHES)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_cu_list_equals_tolist(lens, dtype):
    cu = _cu(lens, dtype)
    assert pmc.cu_list(cu) == cu.tolist()


@pytest.mark.parametrize("lens", BATCHES)
def test_cu_last_equals_the_select_read(lens):
    cu = _cu(lens)
    assert pmc.cu_last(cu) == int(cu[-1].cpu().item())


@pytest.mark.parametrize("lens", BATCHES)
@pytest.mark.parametrize("div", [1, 2, 4])
def test_seq_lens_equals_the_device_expression(lens, div):
    """The host-side arithmetic must agree with `((cu[1:] - cu[:-1]) // div).tolist()` elementwise."""
    cu = _cu([n * div for n in lens])
    want = ((cu[1:] - cu[:-1]) // div).tolist()
    assert pmc.seq_lens(cu, div) == want


def test_seq_lens_rejects_a_nonsense_divisor():
    cu = _cu([4, 4])
    with pytest.raises(ValueError, match="div must be >= 1"):
        pmc.seq_lens(cu, 0)


@pytest.mark.parametrize("lens", BATCHES)
def test_fla_stable_clone_is_a_real_equal_clone(lens):
    cu = _cu(lens)
    clone = pmc.fla_stable_clone(cu)
    assert clone is not cu
    assert torch.equal(clone, cu)
    assert clone.dtype == cu.dtype and clone.device == cu.device


# ---------------------------------------------------------------------------------------------
# 2. the point of the thing: one read per forward, not one per layer
# ---------------------------------------------------------------------------------------------
def test_thirty_layers_pay_one_read_each_and_get_identical_objects():
    """A 30-layer forward: 1 build + 29 hits per derived key, the same object every time."""
    cu = _cu([96, 64, 137, 201])
    firsts = (pmc.cu_list(cu), pmc.cu_last(cu), pmc.seq_lens(cu), pmc.fla_stable_clone(cu))
    for _ in range(29):
        assert pmc.cu_list(cu) is firsts[0]
        assert pmc.cu_last(cu) == firsts[1]
        assert pmc.seq_lens(cu) is firsts[2]
        assert pmc.fla_stable_clone(cu) is firsts[3]
    c = pmc.packed_meta_census()
    # The +2 on the first round is `cu_last` and `seq_lens` consuming the already-built "list":
    # one device read backs all four derived values.
    assert c["built"] == 4, c
    assert c["served"] == 29 * 4 + 2, c
    assert c["declined"] == 0, c


def test_causal_conv_metadata_is_built_once_per_packed_forward(monkeypatch):
    """The conv metadata is built once per packed forward, not once per layer."""
    module_name = "vllm.v1.attention.backends.utils"
    fake_module = ModuleType(module_name)
    builds = []

    def fake_compute(cu_cpu, *, device):
        builds.append((cu_cpu.clone(), device))
        return {8: {"tot": 3}}, torch.tensor([1]), torch.tensor([2])

    fake_module.compute_causal_conv1d_metadata = fake_compute
    monkeypatch.setitem(sys.modules, module_name, fake_module)
    cu = _cu([96, 64, 137, 201])

    first = pmc.causal_conv1d_metadata(cu)
    for _ in range(29):
        assert pmc.causal_conv1d_metadata(cu) is first

    assert len(builds) == 1
    assert torch.equal(builds[0][0], cu)
    assert first.nums_dict == {8: {"tot": 3}}
    assert torch.equal(first.batch_ptr, torch.tensor([1]))
    assert torch.equal(first.token_chunk_offset_ptr, torch.tensor([2]))
    assert torch.equal(first.cache_indices, torch.tensor([1, 2, 3, 4], dtype=torch.int32))
    assert torch.equal(first.has_initial_state, torch.zeros(4, dtype=torch.bool))
    assert pmc.causal_conv1d_metadata(cu).cache_indices is first.cache_indices
    assert pmc.causal_conv1d_metadata(cu).has_initial_state is first.has_initial_state


def test_causal_conv_metadata_accepts_scheduler_host_offsets_without_readback(monkeypatch):
    module_name = "vllm.v1.attention.backends.utils"
    fake_module = ModuleType(module_name)
    seen = []

    def fake_compute(cu_cpu, *, device):
        seen.append(cu_cpu.tolist())
        return {}, torch.tensor([], dtype=torch.int32), torch.tensor([], dtype=torch.int32)

    fake_module.compute_causal_conv1d_metadata = fake_compute
    monkeypatch.setitem(sys.modules, module_name, fake_module)
    monkeypatch.setattr(pmc, "cu_list", lambda _cu: (_ for _ in ()).throw(AssertionError("D2H")))
    cu = _cu([3, 5, 7])
    assert pmc.causal_conv1d_metadata(cu, cu_host=[0, 3, 8, 15]) is not None
    assert seen == [[0, 3, 8, 15]]


@pytest.mark.parametrize("bad", [[], [1, 4], [0, 4]])
def test_causal_conv_scheduler_offsets_decline_invalid_shape(bad):
    assert pmc.causal_conv1d_metadata(_cu([4, 5]), cu_host=bad) is None


def test_causal_conv_metadata_cache_off_restores_vllm_internal_build(monkeypatch):
    monkeypatch.setenv("SKYRL_ISOEXEC_PACKED_META_CACHE", "0")
    assert pmc.causal_conv1d_metadata(_cu([4, 9])) is None


def test_the_stable_clone_is_what_makes_an_identity_keyed_cache_hit():
    """A fresh clone per layer is a guaranteed miss in FLA's identity-keyed cache."""
    cu = _cu([96, 64, 137, 201])
    # hold the references: CPython recycles the id of a clone dropped on the same line, which
    # would make the control look like a hit.
    fresh = [cu.clone() for _ in range(8)]
    stable = [pmc.fla_stable_clone(cu) for _ in range(8)]
    assert len({id(t) for t in stable}) == 1, "the memo must hand every layer the SAME object"
    assert len({id(t) for t in fresh}) == 8, "sanity: fresh clones really are distinct objects"


# ---------------------------------------------------------------------------------------------
# 3. key soundness -- an in-place write through ANY alias invalidates
# ---------------------------------------------------------------------------------------------
def test_in_place_write_invalidates():
    cu = _cu([64, 64])
    before = pmc.cu_list(cu)
    cu[1] = 32  # bumps _version
    after = pmc.cu_list(cu)
    assert after != before
    assert after == cu.tolist()


def test_in_place_write_through_a_VIEW_invalidates():
    """torch's version counter is shared across views and bases, so a view write invalidates."""
    cu = _cu([64, 64, 64])
    view = cu[:]
    before = pmc.cu_list(cu)
    view[2] = 99
    assert pmc.cu_list(cu) != before
    assert pmc.cu_list(cu) == cu.tolist()


def test_a_different_tensor_with_equal_values_is_not_served_the_first_entry():
    a = _cu([64, 64])
    b = _cu([64, 64])
    assert torch.equal(a, b)
    ca, cb = pmc.fla_stable_clone(a), pmc.fla_stable_clone(b)
    assert ca is not cb, "the key must be identity, never value"
    assert torch.equal(ca, cb)


def test_the_entry_holds_a_STRONG_reference_so_the_id_cannot_be_recycled():
    cu = _cu([64, 64])
    pmc.cu_list(cu)
    key = (id(cu), cu._version, str(cu.device), str(cu.dtype))
    assert key in pmc._CACHE
    assert pmc._CACHE[key][0] is cu, "the entry must pin the tensor it is keyed on"


def test_inference_tensor_declines_versioned_memo_without_raising(monkeypatch):
    """vLLM builds query-start offsets under inference_mode; those tensors have no ``_version``."""

    with torch.inference_mode():
        cu = _cu([3, 5, 7])
    assert torch.is_inference(cu)
    assert pmc.cu_list(cu) == [0, 3, 8, 15]
    assert pmc.is_memoized(cu) is False
    assert pmc.packed_meta_census() == {"served": 0, "built": 0, "declined": 1}

    module_name = "vllm.v1.attention.backends.utils"
    fake_module = ModuleType(module_name)
    seen = []

    def fake_compute(cu_cpu, *, device):
        seen.append(cu_cpu.tolist())
        return {}, torch.tensor([], dtype=torch.int32), torch.tensor([], dtype=torch.int32)

    fake_module.compute_causal_conv1d_metadata = fake_compute
    monkeypatch.setitem(sys.modules, module_name, fake_module)
    got = pmc.causal_conv1d_metadata(cu, cu_host=[0, 3, 8, 15], include_launch_args=False)
    assert got is not None and seen == [[0, 3, 8, 15]]


# ---------------------------------------------------------------------------------------------
# 4. declines -- and a decline is still a CORRECT answer
# ---------------------------------------------------------------------------------------------
def test_a_large_tensor_is_declined_but_still_answered_correctly():
    cu = torch.arange(0, pmc._MAX_NUMEL + 2, dtype=torch.int32)
    assert pmc.cu_list(cu) == cu.tolist()
    assert pmc.packed_meta_census()["declined"] > 0
    assert pmc.packed_meta_census()["built"] == 0


def test_the_flag_off_restores_the_per_call_read_and_the_values_are_unchanged(monkeypatch):
    monkeypatch.setenv("SKYRL_ISOEXEC_PACKED_META_CACHE", "0")
    cu = _cu([96, 64, 137])
    assert not pmc.packed_meta_cache_enabled()
    assert pmc.cu_list(cu) == cu.tolist()
    assert pmc.seq_lens(cu, 2) == ((cu[1:] - cu[:-1]) // 2).tolist()
    a, b = pmc.fla_stable_clone(cu), pmc.fla_stable_clone(cu)
    assert a is not b, "flag off must restore the fresh clone per call"
    assert torch.equal(a, b)
    c = pmc.packed_meta_census()
    assert c["served"] == 0 and c["built"] == 0 and c["declined"] > 0


@pytest.mark.parametrize("val", ["0", "false", "no", "FALSE", ""])
def test_the_off_spellings(monkeypatch, val):
    monkeypatch.setenv("SKYRL_ISOEXEC_PACKED_META_CACHE", val)
    assert not pmc.packed_meta_cache_enabled()


def test_default_is_ON():
    assert pmc.packed_meta_cache_enabled()


def test_the_cache_is_bounded():
    for i in range(pmc._CACHE_MAX * 3):
        pmc.cu_list(_cu([64, 64 + i]))
    assert len(pmc._CACHE) <= pmc._CACHE_MAX


# ---------------------------------------------------------------------------------------------
# 5. the patched megatron bodies ARE the upstream bodies
# ---------------------------------------------------------------------------------------------
def _upstream_unpack(x, cu_seqlens, dim=1):
    """megatron/core/ssm/gated_delta_net.py:716-726, verbatim."""
    unpacked_x = []
    cu_seqlens_list = cu_seqlens.tolist()
    num_seqs = len(cu_seqlens_list) - 1
    for i in range(num_seqs):
        idx_start = cu_seqlens_list[i]
        idx_end = cu_seqlens_list[i + 1]
        chunked_index = [slice(None)] * dim + [slice(idx_start, idx_end)]
        unpacked_x.append(x[tuple(chunked_index)])
    return unpacked_x


@pytest.mark.parametrize("lens", BATCHES)
@pytest.mark.parametrize("dim", [0, 1])
def test_unpack_sequence_memo_matches_upstream_slice_for_slice(lens, dim):
    from skyrl.backends.skyrl_train.isoexec.runtimes.megatron import (
        gdn_fla_shim as shim,
    )

    cu = _cu(lens)
    total = int(cu[-1])
    x = torch.randn(3, total) if dim == 1 else torch.randn(total, 3)
    got = shim._unpack_sequence_memo(x, cu, dim=dim)
    want = _upstream_unpack(x, cu, dim=dim)
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert torch.equal(g, w)


class _Dummy:
    """Just enough of GatedDeltaNet for the unbound method."""


@pytest.mark.parametrize("validate_once", [False, True])
def test_resolve_cu_seqlens_returns_an_IS_IDENTICAL_tensor(monkeypatch, validate_once):
    from skyrl.backends.skyrl_train.isoexec.runtimes.megatron import (
        gdn_fla_shim as shim,
    )

    monkeypatch.setenv("SKYRL_ISOEXEC_GDN_VALIDATE_ONCE", "1" if validate_once else "0")
    cu = _cu([96, 64, 136])
    out = shim._resolve_cu_seqlens_memo(_Dummy(), None, cu, int(cu[-1]), "cu_seqlens_q", cp_size=1)
    assert out is cu, "the function computes nothing; it must hand back its own input"

    padded = _cu([128, 128])
    out2 = shim._resolve_cu_seqlens_memo(_Dummy(), padded, cu, int(padded[-1]), "cu_seqlens_q", cp_size=1)
    assert out2 is padded, "padded wins when present, exactly as upstream"


@pytest.mark.parametrize("validate_once", [False, True])
def test_resolve_cu_seqlens_still_raises_on_a_total_mismatch(monkeypatch, validate_once):
    from skyrl.backends.skyrl_train.isoexec.runtimes.megatron import (
        gdn_fla_shim as shim,
    )

    monkeypatch.setenv("SKYRL_ISOEXEC_GDN_VALIDATE_ONCE", "1" if validate_once else "0")
    cu = _cu([96, 64])
    with pytest.raises(ValueError, match="does not match total_sequence_length"):
        shim._resolve_cu_seqlens_memo(_Dummy(), None, cu, 999, "cu_seqlens_q", cp_size=1)


@pytest.mark.parametrize("validate_once", [False, True])
def test_cp1_resolve_never_inspects_sequence_lengths(monkeypatch, validate_once):
    """Divisibility by one is true without indexing or launching work on ``cu``."""
    from skyrl.backends.skyrl_train.isoexec.ops.gdn import packed_meta_cache
    from skyrl.backends.skyrl_train.isoexec.runtimes.megatron import (
        gdn_fla_shim as shim,
    )

    class NoDeviceValueReads:
        _version = 0

        def __getitem__(self, _key):
            raise AssertionError("cp_size=1 must not inspect sequence lengths")

    cu = NoDeviceValueReads()
    monkeypatch.setenv("SKYRL_ISOEXEC_GDN_VALIDATE_ONCE", "1" if validate_once else "0")
    monkeypatch.setattr(packed_meta_cache, "cu_last", lambda candidate: 123 if candidate is cu else None)

    out = shim._resolve_cu_seqlens_memo(_Dummy(), None, cu, 123, "cu_seqlens_q", cp_size=1)

    assert out is cu


def test_cp1_guard_structurally_owns_every_divisibility_operation():
    """Keep both host/device modulo paths unreachable until the CP=1 tautology is discharged."""
    import ast
    import inspect
    import textwrap

    from skyrl.backends.skyrl_train.isoexec.runtimes.megatron import (
        gdn_fla_shim as shim,
    )

    tree = ast.parse(textwrap.dedent(inspect.getsource(shim._resolve_cu_seqlens_memo)))
    function = tree.body[0]
    guards = [
        statement
        for statement in function.body
        if isinstance(statement, ast.If) and ast.unparse(statement.test) == "cp_size != 1"
    ]
    assert len(guards) == 1

    guarded = {id(node) for statement in guards[0].body for node in ast.walk(statement)}
    modulo = [node for node in ast.walk(function) if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod)]
    any_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "any"
    ]
    assert len(modulo) == 2
    assert len(any_calls) == 1
    assert all(id(node) in guarded for node in modulo + any_calls)


@pytest.mark.parametrize("validate_once", [False, True])
def test_cp_greater_than_one_still_returns_the_identical_valid_tensor(monkeypatch, validate_once):
    from skyrl.backends.skyrl_train.isoexec.runtimes.megatron import (
        gdn_fla_shim as shim,
    )

    monkeypatch.setenv("SKYRL_ISOEXEC_GDN_VALIDATE_ONCE", "1" if validate_once else "0")
    cu = _cu([96, 64, 136])

    out = shim._resolve_cu_seqlens_memo(_Dummy(), None, cu, int(cu[-1]), "cu_seqlens_q", cp_size=2)

    assert out is cu


@pytest.mark.parametrize("validate_once", [False, True])
def test_resolve_cu_seqlens_still_raises_on_a_cp_divisibility_violation(monkeypatch, validate_once):
    """The memo must not blind the check it memoizes -- both branches refuse the same input."""
    from skyrl.backends.skyrl_train.isoexec.runtimes.megatron import (
        gdn_fla_shim as shim,
    )

    monkeypatch.setenv("SKYRL_ISOEXEC_GDN_VALIDATE_ONCE", "1" if validate_once else "0")
    cu = _cu([96, 65])  # 65 is not divisible by 2
    with pytest.raises(ValueError, match="divisible by cp_size"):
        shim._resolve_cu_seqlens_memo(_Dummy(), None, cu, int(cu[-1]), "cu_seqlens_q", cp_size=2)


def test_validate_once_agrees_with_the_device_check_on_every_batch(monkeypatch):
    """Host-side `% cp_size` and the device `(lens % cp).any()` must accept/refuse identically."""
    from skyrl.backends.skyrl_train.isoexec.runtimes.megatron import (
        gdn_fla_shim as shim,
    )

    for lens in BATCHES + [[3, 6, 9], [2, 4, 7]]:
        for cp in (1, 2, 3):
            cu = _cu(lens)
            verdicts = []
            for once in ("0", "1"):
                monkeypatch.setenv("SKYRL_ISOEXEC_GDN_VALIDATE_ONCE", once)
                pmc._CACHE.clear()
                try:
                    shim._resolve_cu_seqlens_memo(_Dummy(), None, cu, int(cu[-1]), "q", cp_size=cp)
                    verdicts.append("ok")
                except ValueError:
                    verdicts.append("raise")
            assert verdicts[0] == verdicts[1], f"lens={lens} cp={cp} disagreed: {verdicts}"


def test_validate_once_default_is_OFF():
    from skyrl.backends.skyrl_train.isoexec.runtimes.megatron import (
        gdn_fla_shim as shim,
    )

    assert not shim._validate_once_enabled()


# ---------------------------------------------------------------------------------------------
# 6. the rope thd half -- bitwise, not tolerant: it is the same arithmetic
# ---------------------------------------------------------------------------------------------
class _CPGroup:
    def __init__(self, size=1, rank=0):
        self._s, self._r = size, rank

    def size(self):
        return self._s

    def rank(self):
        return self._r


@pytest.mark.parametrize("lens", [[96, 64, 137, 201], [64], [7, 129, 3]])
@pytest.mark.parametrize("memo", ["0", "1"])
def test_rope_thd_memo_is_bitwise_equal_to_upstream(monkeypatch, lens, memo):
    """Both branches must produce byte-identical output -- same splits, same freqs, same kernel."""
    pytest.importorskip("megatron.core.models.common.embeddings.rope_utils")
    from megatron.core.models.common.embeddings import rope_utils as ru

    from skyrl.backends.skyrl_train.isoexec.runtimes.megatron import (
        gdn_fla_shim as shim,
    )

    monkeypatch.setenv("SKYRL_ISOEXEC_PACKED_META_CACHE", memo)
    cu = _cu(lens)
    total, d, h = int(cu[-1]), 16, 2
    torch.manual_seed(0)
    t = torch.randn(total, h, d)
    freqs = torch.randn(max(lens) + 8, 1, 1, d)
    kw = dict(rotary_interleaved=False, mscale=1.0, cp_group=_CPGroup())

    got = shim._apply_rotary_pos_emb_thd_memo(t, cu, freqs, **kw)
    assert got.shape == (total, h, d)
    assert torch.isfinite(got).all()

    # the memoized branch and the per-call-read branch must agree BITWISE, in both directions
    monkeypatch.setenv("SKYRL_ISOEXEC_PACKED_META_CACHE", "0" if memo == "1" else "1")
    pmc._CACHE.clear()
    other = shim._apply_rotary_pos_emb_thd_memo(t, cu, freqs, **kw)
    assert torch.equal(got, other), "the memo and the per-call read must be BITWISE equal"

    # and against megatron's own thd rope, which is what we are standing in for
    if getattr(ru._apply_rotary_pos_emb_thd, "__name__", "") != "_apply_rotary_pos_emb_thd_memo":
        assert torch.equal(got, ru._apply_rotary_pos_emb_thd(t, cu, freqs, **kw))


# ---------------------------------------------------------------------------------------------
# 7. the divided-cu hoist: `_unpack_sequence(x, cu // cp_size)`
# ---------------------------------------------------------------------------------------------
@pytest.fixture
def shim():
    from skyrl.backends.skyrl_train.isoexec.runtimes.megatron import gdn_fla_shim as s

    s._LAST_RESOLVED.update(cu=None, cp=1, ver=-1)
    s._DIV_PROOF.clear()
    for k in s._DIV_STATS:
        s._DIV_STATS[k] = 0
    yield s
    s._LAST_RESOLVED.update(cu=None, cp=1, ver=-1)
    s._DIV_PROOF.clear()


def _resolve(shim, cu, cp=1):
    """What `GatedDeltaNet.forward` does before it calls `_unpack_sequence`."""
    return shim._resolve_cu_seqlens_memo(_Dummy(), None, cu, int(cu[-1]), "cu_seqlens_q", cp_size=cp)


@pytest.mark.parametrize("lens", [[96, 64, 137, 201], [64], [7, 129, 3]])
def test_divided_cu_is_proven_once_then_served(shim, lens):
    """30 fresh `cu // 1` tensors cost one real read; the other 29 are served off the proof."""
    cu = _cu(lens)
    _resolve(shim, cu, cp=1)
    want = cu.tolist()
    for _ in range(30):
        d = cu // 1  # exactly what the call site builds, a fresh tensor every time
        assert shim._divided_cu_list(d, 0) == want
    c = shim.divided_cu_census()
    assert c["proved"] == 1, c
    assert c["served"] == 29 - c["reverified"], c
    assert c["disproved"] == 0 and c["unmatched"] == 0, c


@pytest.mark.parametrize("cp", [1, 2, 4])
def test_divided_cu_is_correct_for_every_cp_size(shim, cp):
    cu = _cu([96 * cp, 64 * cp, 136 * cp])
    _resolve(shim, cu, cp=cp)
    d = cu // cp
    assert shim._divided_cu_list(d, 0) == d.tolist()


def test_a_LIE_is_caught_and_the_site_is_never_fast_pathed_again(shim):
    """A tensor that is not `cu // cp` is refused, and stays refused on every later call."""
    cu = _cu([64, 64, 64])
    _resolve(shim, cu, cp=1)
    liar = cu * 2  # same shape/dtype/device, freshly derived (so not refused), different values
    assert shim._divided_cu_list(liar, 0) == liar.tolist(), "must return the TRUTH, not the guess"
    assert shim.divided_cu_census()["disproved"] == 1
    # and every later call at that key falls through to the real read
    for _ in range(5):
        assert shim._divided_cu_list(liar, 0) == liar.tolist()
    assert shim.divided_cu_census()["served"] == 0


def test_reverification_catches_a_caller_that_diverges_midway(shim):
    """Changing the values under a served proof is caught by the periodic re-verification."""
    cu = _cu([64, 64, 64])
    _resolve(shim, cu, cp=1)
    good = cu // 1
    # drive exactly _DIV_REVERIFY calls so the NEXT one is the re-verification
    for _ in range(shim._DIV_REVERIFY):
        assert shim._divided_cu_list(good, 0) == cu.tolist()
    assert shim.divided_cu_census()["disproved"] == 0
    assert shim.divided_cu_census()["reverified"] == 0
    bad = cu * 2  # freshly derived, same shape, different values
    got = shim._divided_cu_list(bad, 0)  # the re-verification call
    assert got == bad.tolist(), "re-proof must return the observed truth"
    assert shim.divided_cu_census()["disproved"] == 1


def test_fails_closed_with_no_resolved_cu(shim):
    cu = _cu([64, 64])
    assert shim._LAST_RESOLVED["cu"] is None
    assert shim._divided_cu_list(cu // 1, 0) == cu.tolist()
    assert shim.divided_cu_census()["proved"] == 0


def test_fails_closed_on_a_shape_or_dtype_mismatch(shim):
    cu = _cu([64, 64, 64])
    _resolve(shim, cu, cp=1)
    assert shim._divided_cu_list(_cu([64, 64]), 0) == [0, 64, 128]
    assert shim._divided_cu_list(_cu([64, 64, 64], torch.int64), 0) == cu.tolist()
    assert shim.divided_cu_census()["unmatched"] == 2
    assert shim.divided_cu_census()["proved"] == 0


def test_fails_closed_with_the_flag_off(shim, monkeypatch):
    cu = _cu([64, 64])
    _resolve(shim, cu, cp=1)
    monkeypatch.setenv("SKYRL_ISOEXEC_PACKED_META_CACHE", "0")
    assert shim._divided_cu_list(cu // 1, 0) == cu.tolist()
    assert shim.divided_cu_census()["proved"] == 0


def test_a_new_forward_reproves_rather_than_reusing_the_old_proof(shim):
    cu_a = _cu([64, 64])
    _resolve(shim, cu_a, cp=1)
    shim._divided_cu_list(cu_a // 1, 0)
    cu_b = _cu([32, 96])  # a different forward, different values, same shape
    _resolve(shim, cu_b, cp=1)
    assert shim._divided_cu_list(cu_b // 1, 0) == cu_b.tolist()
    assert shim.divided_cu_census()["proved"] == 2, "each forward buys its own proof"


def test_unpack_sequence_still_matches_upstream_through_the_divided_path(shim):
    """Identical slices, whichever path served the boundaries."""
    cu = _cu([96, 64, 137, 201])
    _resolve(shim, cu, cp=1)
    x = torch.randn(int(cu[-1]), 3)
    for _ in range(4):
        got = shim._unpack_sequence_memo(x, cu // 1, dim=0)
        want = _upstream_unpack(x, cu // 1, dim=0)
        assert len(got) == len(want)
        for g, w in zip(got, want):
            assert torch.equal(g, w)


def test_the_divided_probe_does_not_evict_the_real_entry(shim):
    """A probe that inserted would push cu out of the bounded LRU and undo the memo."""
    cu = _cu([96, 64, 137])
    _resolve(shim, cu, cp=1)
    pmc.cu_list(cu)
    before = len(pmc._CACHE)
    for _ in range(pmc._CACHE_MAX * 4):
        shim._unpack_sequence_memo(torch.randn(int(cu[-1]), 2), cu // 1, dim=0)
    assert len(pmc._CACHE) == before, "probing must not insert entries"
    assert pmc.is_memoized(cu), "the real cu_seqlens entry must survive"


def test_is_memoized_never_creates_an_entry():
    cu = _cu([64, 64])
    assert not pmc.is_memoized(cu)
    assert len(pmc._CACHE) == 0, "a peek must not insert"
    pmc.cu_list(cu)
    assert pmc.is_memoized(cu)
    assert not pmc.is_memoized(cu, "nosuchkey")


def test_rope_thd_memo_splits_exactly_as_upstream_would(monkeypatch):
    """`torch.split` must receive the identical size list, or every downstream kernel shifts."""
    cu = _cu([96, 64, 137, 201])
    for cp in (1, 2, 4):
        cu_cp = _cu([n * cp for n in [96, 64, 136, 200]])
        want = ((cu_cp[1:] - cu_cp[:-1]) // cp).tolist()
        pmc._CACHE.clear()
        assert pmc.seq_lens(cu_cp, cp) == want
    assert pmc.cu_last(cu) == int(cu[-1])


# ---------------------------------------------------------------------------------------------
# 8. the ledger must not be keyed on a recycled address: without a strong reference CPython reuses
#    `id(cu)`, and a padding microbatch's offsets can be served to a real one of the same shape.
# ---------------------------------------------------------------------------------------------
_PADDING_MB = [4] * 13  # 13 dummy rows, one valid token each, padded to align_size = tp_size = 4
_REAL_MB = [1484] * 12 + [19308 - 1484 * 12]  # same segment COUNT, 19,308 tokens


def _microbatch(shim, seglens, nlayers=30):
    """One GDN forward's worth of packed-meta traffic: resolve (q, kv) then unpack, per layer."""
    cu = _cu(seglens)
    total = int(cu[-1])
    x = torch.zeros(total, 2)
    wrong = 0
    for _ in range(nlayers):
        cu_q = shim._resolve_cu_seqlens_memo(_Dummy(), cu, None, total, "cu_seqlens_q", cp_size=1)
        shim._resolve_cu_seqlens_memo(_Dummy(), cu, None, total, "cu_seqlens_kv", cp_size=1)
        parts = shim._unpack_sequence_memo(x, cu_q // 1, dim=0)
        if sum(p.shape[0] for p in parts) != total:
            wrong += 1
    return wrong, id(cu), total


def test_the_ledger_never_serves_another_microbatchs_cu(shim):
    """500 alternating padding/real microbatches produce zero wrong slicings."""
    seen, wrong_microbatches = {}, 0
    for step in range(500):
        seglens = _PADDING_MB if step % 7 == 3 else _REAL_MB
        wrong, cu_id, total = _microbatch(shim, seglens)
        wrong_microbatches += bool(wrong)
        seen.setdefault(cu_id, set()).add(total)

    recycled = [i for i, totals in seen.items() if len(totals) > 1]
    if not recycled:
        pytest.skip("this interpreter did not recycle a cu_seqlens address; nothing was exercised")
    assert wrong_microbatches == 0, (
        f"{wrong_microbatches}/500 microbatches were sliced with another microbatch's cu_seqlens; "
        f"{len(recycled)} of {len(seen)} addresses were reused across different totals"
    )
    census = shim.divided_cu_census()
    assert census["stale"] == 0, census
    assert census["served"] > 0, f"the hoist went inert instead of getting fixed: {census}"


def test_generations_are_unique_so_ledger_keys_cannot_alias(shim):
    """Distinct microbatches never share a proof key."""
    keys = []
    for step in range(200):
        shim._DIV_PROOF.clear()
        _microbatch(shim, _PADDING_MB if step % 3 == 0 else _REAL_MB, nlayers=2)
        keys.extend(shim._DIV_PROOF.keys())
    assert len(keys) == len(set(keys)), "a proof key was reused by a different microbatch"


def test_div_hoist_kill_switch_restores_the_plain_read(shim, monkeypatch):
    monkeypatch.setenv("SKYRL_ISOEXEC_PACKED_META_DIV_HOIST", "0")
    cu = _cu(_REAL_MB)
    _resolve(shim, cu, cp=1)
    for _ in range(30):
        assert shim._divided_cu_list(cu // 1, 0) == cu.tolist()
    assert shim.divided_cu_census()["served"] == 0, "the kill switch must not serve from the ledger"


def test_packed_meta_cache_off_restores_the_plain_read(shim, monkeypatch):
    """With the cache flag off, nothing in the ledger can fire at all."""
    monkeypatch.setenv("SKYRL_ISOEXEC_PACKED_META_CACHE", "0")
    for step in range(60):
        wrong, _, _ = _microbatch(shim, _PADDING_MB if step % 5 == 0 else _REAL_MB, nlayers=4)
        assert wrong == 0
    census = shim.divided_cu_census()
    assert census["served"] == 0 and census["proved"] == 0, census


def test_an_argument_that_is_not_a_freshly_derived_divisor_is_refused(shim):
    """Only a freshly derived (version-0, not-`cu`) tensor may enter the ledger, because the
    backward's `recompute_norm_out` can arrive after `_LAST_RESOLVED` moved on."""
    cu_now = _cu([64, 64, 64])
    _resolve(shim, cu_now, cp=1)
    for _ in range(3):
        assert shim._divided_cu_list(cu_now // 1, 0) == cu_now.tolist()  # site A: admitted

    assert shim._divided_cu_list(cu_now, 0) == cu_now.tolist()  # `d is cu`
    old = _cu([16, 16, 16])
    old[0] = 0  # a real cu_seqlens is built by an in-place cumsum write -> _version >= 1
    assert shim._divided_cu_list(old, 0) == old.tolist(), "must not be served the current cu's list"
    assert shim.divided_cu_census()["unmatched"] == 2
