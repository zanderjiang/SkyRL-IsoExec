"""CPU proof obligations for pik's fused barrier+reduce collective (one launch per site).

Two halves are decidable without a GPU: ``codegen._ar_core`` (the only float-touching lines) is
byte-identical in both templates, and AOT ``triton.compile`` against an explicit sm_90 target
lets the fused kernel's arithmetic PTX and barrier lowering be compared to the unfused one.
"""

from __future__ import annotations

import importlib
import re

import pytest

pytest.importorskip("triton")

AR = importlib.import_module("skyrl.backends.skyrl_train.isoexec.ops.collectives.pik.allreduce")


def _reset_fused_state():
    AR._FUSED_ADMIT.clear()
    AR._FUSED_GROUP.clear()
    for k in AR._FUSED_COUNTS:
        AR._FUSED_COUNTS[k] = 0
    AR._FUSED_FIRST_REJECT = {}


def _reset_root_cast_state():
    AR._ROOT_CAST_ADMIT.clear()
    for k in AR._ROOT_CAST_COUNTS:
        AR._ROOT_CAST_COUNTS[k] = 0


CG = importlib.import_module("skyrl.backends.skyrl_train.isoexec.ops.collectives.pik.codegen")

WORLDS = (2, 4, 8)
WIRES = ((False, False), (True, True), (True, False))  # (in_bf16, out_bf16); the 3 legal forms


def _variants():
    for c in WORLDS:
        for push in (False, True):
            for in_bf16, out_bf16 in WIRES:
                yield c, push, in_bf16, out_bf16


# 1. the arithmetic is the same string in both templates
def test_core_is_byte_identical_in_both_templates():
    """The only float-touching lines are generated once and pasted into both kernels."""
    for c, push, bi, bo in _variants():
        core = CG._ar_core(c, push, bi, bo)
        assert core in CG._ar_tmpl(c, push, bi, bo), (c, push, bi, bo, "unfused")
        assert core in CG._fused_ar_tmpl(c, push, bi, bo), (c, push, bi, bo, "fused")


def test_core_is_the_only_arithmetic():
    """Nothing outside ``_ar_core`` adds, multiplies or converts a float.

    The barrier prologue/epilogue must stay integer flag traffic only.
    """
    for c, push, bi, bo in _variants():
        core = CG._ar_core(c, push, bi, bo)
        outside = CG._fused_ar_tmpl(c, push, bi, bo).replace(core, "")
        for banned in ("float", "other=0.0", "tl.dot"):
            assert banned not in outside, (c, push, bi, bo, banned)
        # the only dtype the barrier ever names is the flag/epoch type
        assert set(re.findall(r"tl\.(int\d+|float\d+|bfloat\d+)", outside)) <= {"int32"}


def test_unfused_template_unchanged_by_the_refactor():
    """Golden literal for the reference path: extracting ``_ar_core`` must not move a byte."""
    assert CG._ar_tmpl(2, False, False, False) == (
        "\nimport triton\nimport triton.language as tl\n\n@triton.jit\n"
        "def kernel(in0, in1, out0, base, n, BLOCK: tl.constexpr):\n"
        "    offs = base + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n"
        "    mask = offs < base + n\n\n"
        "    t0 = tl.load(in0 + offs, mask=mask, other=0.0)\n"
        "    t1 = tl.load(in1 + offs, mask=mask, other=0.0)\n"
        "    t0 = t0 + t1\n\n"
        "    tl.store(out0 + offs, t0, mask=mask)\n"
    )


# 2. the barrier's shape, in source
def test_publishes_to_every_peer_exactly_once_per_region():
    """A barrier that skips a peer is a barrier that does not synchronise with it."""
    for c, push, bi, bo in _variants():
        src = CG._fused_ar_tmpl(c, push, bi, bo)
        regions = 2 if push else 1
        assert src.count('sem="release", scope="sys"') == c * regions
        assert src.count('sem="acquire", scope="sys"') == regions
        for i in range(c):
            assert src.count(f"tl.atomic_xchg(fl{i} + my_slot") == regions
        assert f"fl{c}" not in src  # no phantom peer


def test_one_shot_has_no_trailing_barrier():
    """One-shot writes only its own destination, so nothing of its can be read too early.

    The trailing barrier exists only for the push, where peers write into my output buffer.
    """
    for c, bi, bo in ((w, x, y) for w in WORLDS for x, y in WIRES):
        one = CG._fused_ar_tmpl(c, False, bi, bo)
        two = CG._fused_ar_tmpl(c, True, bi, bo)
        assert "seq1" not in one and "wbase1 + lane" not in one
        assert "seq1" in two and "wbase1 + lane" in two


def test_epoch_bump_is_barriered():
    """``tl.debug_barrier()`` must sit between the epoch load and the epoch store.

    Every thread reads the epoch, one writes it back; without the barrier a fast warp bumps
    the counter before a slow warp read it and the collective deadlocks.
    """
    for c, push, bi, bo in _variants():
        src = CG._fused_ar_tmpl(c, push, bi, bo)
        for r in range(2 if push else 1):
            i_ld = src.index(f"seq{r} = tl.load(ep_ptr")
            i_st = src.index("tl.store(ep_ptr", i_ld)
            assert "tl.debug_barrier()" in src[i_ld:i_st], (c, push, r)


def test_wait_precedes_every_peer_read():
    """No peer byte may be read before every peer has arrived."""
    for c, push, bi, bo in _variants():
        src = CG._fused_ar_tmpl(c, push, bi, bo)
        i_wait_done = src.index("ready0 = tl.min(")
        i_first_read = src.index("tl.load(in0 + offs")
        assert i_wait_done < i_first_read
        assert "tl.debug_barrier()" in src[i_wait_done:i_first_read]


def test_push_precedes_the_trailing_release():
    """A release that runs before the data it is releasing is a silent wrong answer."""
    for c, bi, bo in ((w, x, y) for w in WORLDS for x, y in WIRES):
        src = CG._fused_ar_tmpl(c, True, bi, bo)
        i_last_store = src.rindex("tl.store(out")
        i_trail = src.index("seq1 = tl.load(ep_ptr")
        assert i_last_store < i_trail
        assert "tl.debug_barrier()" in src[i_last_store:i_trail]


# 3. the flag-array layout
def test_flag_slots_never_collide():
    """Every (region, source rank, block) owns a distinct int32."""
    for world in WORLDS:
        for nb in (1, 7, 1024):
            seen = set()
            for region in (0, 1):
                for rank in range(world):
                    for blk in range(nb):
                        seen.add(AR.flag_slot(region, world, rank, nb) + blk)
            assert len(seen) == 2 * world * nb
            assert max(seen) == 2 * world * nb - 1  # and the allocation is exactly big enough


def test_wait_base_matches_publish_slot():
    """Rank r spins on lane r of its own array; peers publish into exactly that element."""
    for world in WORLDS:
        nb = 64
        for region in (0, 1):
            for src_rank in range(world):
                published = AR.flag_slot(region, world, src_rank, nb)
                waited = AR.flag_wait_base(region, world, nb) + src_rank * nb
                assert published == waited, (world, region, src_rank)


# 4. the branch rule is shared, so fusion cannot change which expression runs
def test_shot_rule_is_the_historical_expression():
    """Fusion is a launch-count change: it must not move the one-shot/two-shot boundary."""
    budget = AR.P2P_ONESHOT_MAX_BYTES
    for world in WORLDS:
        for elt in (2, 4):
            for n in (1, 1024, 4096, 65536, 262144, 1 << 20, 1 << 22):
                assert AR._p2p_shot(n, elt, world) == (n * elt * (world - 1) > budget)


def test_shot_rule_has_no_model_or_arch_constant():
    """The only number in the rule is the env budget -- no shape, no arch, no model."""
    import inspect

    src = inspect.getsource(AR._p2p_shot)
    body = src.split('"""')[-1]
    assert "P2P_ONESHOT_MAX_BYTES" in body
    for hardcoded in ("2048", "4096", "glm", "GLM", "47", "sm90", "SM90"):
        assert hardcoded not in body


# 5. AOT: real sm_90 PTX, on a box with no GPU
def _compile(c, push, in_bf16, out_bf16, fused):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    k = (CG.fused_p2p_allreduce_kernel if fused else CG.p2p_allreduce_kernel)(c, push, in_bf16, out_bf16)
    fn = k
    while not hasattr(fn, "arg_names"):
        fn = fn.fn
    ptr_in = "*bf16" if in_bf16 else "*fp32"
    ptr_out = "*bf16" if out_bf16 else "*fp32"
    sig, consts = {}, {}
    for nm in fn.arg_names:
        if nm.startswith("in"):
            sig[nm] = ptr_in
        elif nm.startswith("out"):
            sig[nm] = ptr_out
        elif nm.startswith("fl") or nm in ("self_fl", "ep_ptr"):
            sig[nm] = "*i32"
        elif nm == "BLOCK":
            sig[nm] = "constexpr"
            consts[nm] = 1024
        elif nm == "WORLD":
            sig[nm] = "constexpr"
            consts[nm] = c
        else:
            sig[nm] = "i32"
    return triton.compile(ASTSource(fn=fn, signature=sig, constexprs=consts), target=GPUTarget("cuda", 90, 32)).asm[
        "ptx"
    ]


def _fp_ops(ptx: str) -> list[str]:
    """Every floating-point instruction, in order, opcode only."""
    ops = []
    for line in ptx.splitlines():
        tok = line.strip().split()
        if not tok:
            continue
        op = tok[1] if tok[0].startswith("@") and len(tok) > 1 else tok[0]
        if op.split(".")[0] in ("add", "mul", "fma", "sub", "cvt", "max", "min") and (
            ".f32" in op or ".bf16" in op or ".f16" in op
        ):
            ops.append(op.rstrip(";"))
    return ops


@pytest.mark.parametrize("c,push,in_bf16,out_bf16", list(_variants()))
def test_ptx_arithmetic_is_identical_to_the_unfused_kernel(c, push, in_bf16, out_bf16):
    """Same sm_90 float instruction sequence: values, association and rounding all unchanged."""
    pytest.importorskip("triton.backends.compiler")
    ref = _fp_ops(_compile(c, push, in_bf16, out_bf16, fused=False))
    got = _fp_ops(_compile(c, push, in_bf16, out_bf16, fused=True))
    assert ref == got, f"c={c} push={push} bf16={in_bf16}/{out_bf16}\nref={ref}\ngot={got}"
    assert ref, "the reduction emitted no float instructions at all -- the compare is vacuous"


@pytest.mark.parametrize("c,push", [(w, p) for w in WORLDS for p in (False, True)])
def test_ptx_barrier_lowers_to_system_scope_release_acquire(c, push):
    """The memory ordering must survive the compiler: a relaxed store would let a peer observe
    the flag before the data. One release per peer per region, plus an acquire to spin."""
    ptx = _compile(c, push, False, False, fused=True)
    regions = 2 if push else 1
    assert ptx.count("atom.global.sys.release.") == c * regions, ptx.count("atom.global.sys.release.")
    assert ptx.count("atom.global.sys.acquire.") == regions
    assert "bar.sync" in ptx
    # the reference kernel has none of it: its barriers are separate launches
    assert "atom.global.sys.release." not in _compile(c, push, False, False, fused=False)


@pytest.mark.parametrize("c", WORLDS)
def test_ptx_spin_is_a_real_loop(c):
    """A spin the compiler turned into a straight-line read is not a barrier: a backward
    branch must be present, with the acquire atomic inside it."""
    import re

    lines = _compile(c, False, False, False, fused=True).splitlines()
    label_at = {m.group(1): i for i, ln in enumerate(lines) if (m := re.match(r"^(\$L__BB\d+_\d+):", ln.strip()))}
    backward = [
        (i, m.group(1))
        for i, ln in enumerate(lines)
        if (m := re.search(r"bra\s+(\$L__BB\d+_\d+)", ln)) and label_at.get(m.group(1), 1 << 30) < i
    ]
    assert backward, "the spin compiled to straight-line code -- it is not a barrier"
    # the acquire atomic must be inside that loop, not hoisted above it
    start, end = label_at[backward[0][1]], backward[0][0]
    assert any(
        "atom.global.sys.acquire." in ln for ln in lines[start:end]
    ), "the peer flag is read once and then spun on a stale register"


def test_ptx_epoch_store_follows_a_barrier():
    """The race the source test guards, verified in the instruction stream that will run."""
    ptx = _compile(4, True, False, False, fused=True).splitlines()
    ld = next(i for i, ln in enumerate(ptx) if "ld.global.b32" in ln)
    st = next(i for i, ln in enumerate(ptx) if i > ld and "st.global.b32" in ln)
    assert any("bar.sync" in ln for ln in ptx[ld:st]), "epoch bump is unsynchronised"


# 6. dispatch, admission and the loud fallback -- no CUDA, all stubs
class _FakeReduceOp:
    MIN = "min"


class _FakeDist:
    """Enough torch.distributed for the dispatcher, including the agreement collectives.

    ``peer`` is what the other ranks contribute to every ``all_reduce(MIN)``: 1 agrees, 0 vetoes.
    """

    ReduceOp = _FakeReduceOp

    def __init__(self, world=4, rank=0, peer=1):
        self.world, self.rank, self.peer = world, rank, peer
        self.all_reduces = 0

    def is_initialized(self):
        return True

    def get_world_size(self, group=None):
        return self.world

    def get_rank(self, group=None):
        return self.rank

    def get_process_group_ranks(self, group=None):
        return list(range(self.world))

    def all_reduce(self, t, op=None, group=None):
        self.all_reduces += 1
        t.fill_(min(int(t.item()), self.peer))
        return None


@pytest.fixture
def wired(monkeypatch):
    """Dispatcher under test with both transports stubbed and every verdict forgotten."""
    import torch

    _reset_fused_state()
    monkeypatch.setattr(AR, "dist", _FakeDist())
    calls = {"unfused": 0, "fused": 0}

    def unfused(partial, group, out, root_dtype=None):
        calls["unfused"] += 1
        return partial * 2

    def fused(partial, group, out, root_dtype=None):
        calls["fused"] += 1
        return partial * 2

    monkeypatch.setattr(AR, "_p2p_unfused", unfused)
    monkeypatch.setattr(AR, "_p2p_fused", fused)
    monkeypatch.setattr(AR, "_capturing", lambda: False)
    yield calls, torch
    _reset_fused_state()
    AR.set_fused_enabled(False)


def test_default_is_off_and_the_reference_path_is_untouched(wired):
    calls, torch = wired
    AR.set_fused_enabled(False)
    x = torch.arange(64, dtype=torch.float32)
    for _ in range(5):
        AR._tree_all_reduce_p2p(x, None, None, None)
    assert calls == {"unfused": 5, "fused": 0}
    assert AR.fused_counts()["calls"] == 0


def test_flag_default_from_env_is_off():
    """The module-level default must be OFF."""
    import os

    assert AR._FUSED_ENV == "SKYRL_ISOEXEC_PIK_FUSED_BARRIER"
    assert AR._env_on(AR._FUSED_ENV) is (
        os.environ.get(AR._FUSED_ENV, "0").strip().lower() not in ("", "0", "false", "no", "off")
    )
    assert AR._env_on("SKYRL_ISOEXEC_DEFINITELY_UNSET_FLAG_XYZ") is False


def test_admission_runs_both_paths_once_then_only_the_fused_one(wired):
    calls, torch = wired
    AR.set_fused_enabled(True)
    x = torch.arange(64, dtype=torch.float32)
    AR._tree_all_reduce_p2p(x, None, None, None)
    # admission: one reference + one fused, then the call itself on the fused path
    assert calls["unfused"] == 1 and calls["fused"] == 2
    for _ in range(4):
        AR._tree_all_reduce_p2p(x, None, None, None)
    assert calls["unfused"] == 1 and calls["fused"] == 6
    assert AR.fused_counts()["admitted"] == 1


def test_admission_is_per_shape_and_per_dtype(wired):
    calls, torch = wired
    AR.set_fused_enabled(True)
    for t in (
        torch.arange(64, dtype=torch.float32),
        torch.arange(128, dtype=torch.float32),
        torch.arange(64, dtype=torch.bfloat16),
    ):
        AR._tree_all_reduce_p2p(t, None, None, None)
    assert AR.fused_counts()["admitted"] == 3, "each generated variant is its own proof"


def test_a_differing_bit_rejects_loudly_and_permanently(monkeypatch, capsys):
    """Positive control: a one-ULP difference must be caught by the bit compare, refused for
    that shape permanently, and logged."""
    import torch

    _reset_fused_state()
    monkeypatch.setattr(AR, "dist", _FakeDist())
    monkeypatch.setattr(AR, "_capturing", lambda: False)
    monkeypatch.setattr(AR, "_p2p_unfused", lambda p, g, o, r=None: p * 2)
    # one ULP out: allclose would pass, a bit compare must not
    monkeypatch.setattr(AR, "_p2p_fused", lambda p, g, o, r=None: (p * 2).view(torch.int32).add(1).view(torch.float32))
    AR.set_fused_enabled(True)
    x = torch.arange(1, 65, dtype=torch.float32)
    out = AR._tree_all_reduce_p2p(x, None, None, None)
    assert torch.equal(out, x * 2), "a rejected shape must still return the REFERENCE answer"
    cnt = AR.fused_counts()
    assert cnt["rejected"] == 1 and cnt["first_reject"]["reason"] == "BIT PATTERNS DIFFER"
    assert "WARNING" in capsys.readouterr().out
    for _ in range(3):
        AR._tree_all_reduce_p2p(x, None, None, None)
    assert AR.fused_counts()["rejected"] == 1, "refused once, remembered, never re-tried"
    AR.set_fused_enabled(False)
    _reset_fused_state()


def test_a_raising_fused_kernel_falls_back_instead_of_propagating(monkeypatch, capsys):
    import torch

    _reset_fused_state()
    monkeypatch.setattr(AR, "dist", _FakeDist())
    monkeypatch.setattr(AR, "_capturing", lambda: False)
    monkeypatch.setattr(AR, "_p2p_unfused", lambda p, g, o, r=None: p * 2)

    def boom(*a, **k):
        raise RuntimeError("no symmetric memory")

    monkeypatch.setattr(AR, "_p2p_fused", boom)
    AR.set_fused_enabled(True)
    x = torch.arange(64, dtype=torch.float32)
    assert torch.equal(AR._tree_all_reduce_p2p(x, None, None, None), x * 2)
    assert AR.fused_counts()["errors"] == 1
    assert "no symmetric memory" in capsys.readouterr().out
    AR.set_fused_enabled(False)
    _reset_fused_state()


def test_graph_capture_never_admits_and_never_records(monkeypatch):
    """A compare needs a host sync, which capture forbids: capture keeps the reference path
    and must record no verdict, or the shape would be locked out permanently."""
    import torch

    _reset_fused_state()
    monkeypatch.setattr(AR, "dist", _FakeDist())
    monkeypatch.setattr(AR, "_p2p_unfused", lambda p, g, o, r=None: p * 2)
    monkeypatch.setattr(AR, "_p2p_fused", lambda p, g, o, r=None: p * 2)
    monkeypatch.setattr(AR, "_capturing", lambda: True)
    AR.set_fused_enabled(True)
    x = torch.arange(64, dtype=torch.float32)
    AR._tree_all_reduce_p2p(x, None, None, None)
    assert AR.fused_counts()["capture_skips"] == 1
    assert AR.fused_counts()["shapes"] == {}, "no verdict may be recorded under capture"
    monkeypatch.setattr(AR, "_capturing", lambda: False)
    AR._tree_all_reduce_p2p(x, None, None, None)
    assert AR.fused_counts()["admitted"] == 1, "and the shape gets its chance eagerly later"
    AR.set_fused_enabled(False)
    _reset_fused_state()


def test_non_power_of_two_world_is_refused(monkeypatch, capsys):
    """A balanced tree needs a power-of-two leaf count; world=6 must fall back, not raise."""
    import torch

    _reset_fused_state()
    monkeypatch.setattr(AR, "dist", _FakeDist(world=6))
    monkeypatch.setattr(AR, "_capturing", lambda: False)
    monkeypatch.setattr(AR, "_p2p_unfused", lambda p, g, o, r=None: p * 2)
    monkeypatch.setattr(AR, "_p2p_fused", lambda p, g, o, r=None: p * 3)
    AR.set_fused_enabled(True)
    x = torch.arange(64, dtype=torch.float32)
    assert torch.equal(AR._tree_all_reduce_p2p(x, None, None, None), x * 2)
    assert "power of two" in capsys.readouterr().out
    AR.set_fused_enabled(False)
    _reset_fused_state()


def test_bits_is_a_bit_compare_not_a_float_compare():
    """-0.0 == 0.0 and NaN != NaN under torch.equal. Neither may pass an admission."""
    import torch

    z = torch.tensor([0.0, 1.0])
    nz = torch.tensor([-0.0, 1.0])
    assert torch.equal(z, nz) and not torch.equal(AR._bits(z), AR._bits(nz))
    n1 = torch.tensor([float("nan")])
    n2 = n1.clone()
    assert not torch.equal(n1, n2) and torch.equal(AR._bits(n1), AR._bits(n2))
    b = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    assert AR._bits(b).dtype == torch.int16


# 7. rank invariance -- the decision must be the group's, never a rank's own
# The two launch structures issue different rendezvous, so a rank that picks one while a peer
# picks the other hangs. Every input to the choice must be group-wide or made so by a collective.
def test_enablement_is_settled_by_a_collective_before_the_flag_is_read(monkeypatch, capsys):
    """A rank with the flag ON and a peer with it OFF must fall back, not deadlock.

    The handshake runs on every rank whatever its own flag says, before ``_FUSED_ON`` is read.
    """
    import torch

    _reset_fused_state()
    fake = _FakeDist(peer=0)  # a peer has the flag OFF
    monkeypatch.setattr(AR, "dist", fake)
    monkeypatch.setattr(AR, "_capturing", lambda: False)
    monkeypatch.setattr(AR, "_p2p_unfused", lambda p, g, o, r=None: p * 2)
    monkeypatch.setattr(AR, "_p2p_fused", lambda p, g, o, r=None: p * 3)
    AR.set_fused_enabled(True)
    x = torch.arange(64, dtype=torch.float32)
    assert torch.equal(AR._tree_all_reduce_p2p(x, None, None, None), x * 2)
    assert fake.all_reduces == 1, "enablement was decided without asking the group"
    assert "peer disagrees" in capsys.readouterr().out
    for _ in range(3):
        AR._tree_all_reduce_p2p(x, None, None, None)
    assert fake.all_reduces == 1, "the handshake is once per group, not per call"
    AR.set_fused_enabled(False)
    _reset_fused_state()


def test_a_rank_that_agrees_still_asks(monkeypatch, capsys):
    """An agreeing rank still runs the group handshake."""
    import torch

    _reset_fused_state()
    fake = _FakeDist(peer=1)
    monkeypatch.setattr(AR, "dist", fake)
    monkeypatch.setattr(AR, "_capturing", lambda: False)
    monkeypatch.setattr(AR, "_p2p_unfused", lambda p, g, o, r=None: p * 2)
    monkeypatch.setattr(AR, "_p2p_fused", lambda p, g, o, r=None: p * 2)
    AR.set_fused_enabled(True)
    x = torch.arange(64, dtype=torch.float32)
    AR._tree_all_reduce_p2p(x, None, None, None)
    out = capsys.readouterr().out
    assert "ENABLED and agreed by all" in out
    assert fake.all_reduces >= 2, "enablement handshake + the per-shape admission agreement"
    AR.set_fused_enabled(False)
    _reset_fused_state()


def test_a_peer_veto_on_the_bit_compare_pulls_the_whole_group_back(monkeypatch, capsys):
    """This rank's own compare passes and the shape is still refused: the group's answer is
    the AND of every rank's, and acting on a local verdict is the deadlock."""
    import torch

    _reset_fused_state()

    class _VetoAfterEnable(_FakeDist):
        """Agrees to enablement, then vetoes the shape."""

        def all_reduce(self, t, op=None, group=None):
            self.all_reduces += 1
            if self.all_reduces > 1:
                t.fill_(0)
            return None

    fake = _VetoAfterEnable()
    monkeypatch.setattr(AR, "dist", fake)
    monkeypatch.setattr(AR, "_capturing", lambda: False)
    monkeypatch.setattr(AR, "_p2p_unfused", lambda p, g, o, r=None: p * 2)
    monkeypatch.setattr(AR, "_p2p_fused", lambda p, g, o, r=None: p * 2)  # locally IDENTICAL
    AR.set_fused_enabled(True)
    x = torch.arange(64, dtype=torch.float32)
    assert torch.equal(AR._tree_all_reduce_p2p(x, None, None, None), x * 2)
    cnt = AR.fused_counts()
    assert cnt["peer_vetoes"] == 1 and cnt["admitted"] == 0 and cnt["rejected"] == 1
    assert "a PEER rank refused it" in capsys.readouterr().out
    AR.set_fused_enabled(False)
    _reset_fused_state()


def test_barrier_count_depends_only_on_rank_invariant_facts():
    """One-shot fuses one barrier, two-shot two: a count that differed between ranks would
    hang, so the predicate must depend only on (numel, element size, world)."""
    import inspect

    sig = inspect.signature(AR._p2p_shot)
    assert list(sig.parameters) == [
        "n",
        "elt",
        "world",
    ], "the branch predicate grew an argument -- if it is rank-local, that is a deadlock"
    for world in WORLDS:
        for n in (1024, 65536, 1 << 20):
            for elt in (2, 4):
                answers = {AR._p2p_shot(n, elt, world) for _ in range(8)}
                assert len(answers) == 1, "the branch predicate is not a pure function"


def test_flag_stride_is_a_pure_function_of_the_payload():
    """NB is baked into every flag index, so ranks must agree on it without negotiating: it
    and its growth rule must derive only from numel and BLOCK, never from anything local."""
    import inspect

    src = inspect.getsource(AR._flag_pool)
    assert "get_rank" not in src and "device_count" not in src
    assert "max(nb, 2 * pool.nb if pool is not None else 0, 1024)" in src


# 8. absorbing the root's fp32 -> bf16 round into the reduce kernel
# Unlike the fusion, this MOVES A ROUNDING POINT: separate flag, and the reference is the round
# done afterwards (the expression being replaced).
@pytest.fixture
def rounded(monkeypatch):
    import torch

    _reset_root_cast_state()
    monkeypatch.setattr(AR, "dist", _FakeDist())
    monkeypatch.setattr(AR, "_capturing", lambda: False)
    calls = {"fp32_root": 0, "bf16_root": 0}

    def fake_ar(partial, group=None, out=None, backend=None, root_dtype=None):
        if root_dtype == torch.bfloat16:
            calls["bf16_root"] += 1
            return partial.to(torch.bfloat16)
        calls["fp32_root"] += 1
        return partial

    monkeypatch.setattr(AR, "tree_all_reduce", fake_ar)
    yield calls, torch
    AR.set_root_cast_enabled(False)
    _reset_root_cast_state()


def test_root_cast_default_is_off(rounded):
    """The rounding-point move must be opt-in."""
    import os

    calls, torch = rounded
    assert AR._ROOT_CAST_ENV == "SKYRL_ISOEXEC_PIK_FUSED_ROOT_CAST"
    assert AR._env_on(AR._ROOT_CAST_ENV) is (
        os.environ.get(AR._ROOT_CAST_ENV, "0").strip().lower() not in ("", "0", "false", "no", "off")
    )
    AR.set_root_cast_enabled(False)
    x = torch.randn(256, dtype=torch.float32)
    out = AR.tree_all_reduce_rounded(x, None, torch.bfloat16)
    assert out.dtype == torch.bfloat16
    assert calls == {"fp32_root": 1, "bf16_root": 0}, "the round must still happen afterwards"


def test_root_cast_admits_then_absorbs(rounded):
    calls, torch = rounded
    AR.set_root_cast_enabled(True)
    x = torch.randn(256, dtype=torch.float32)
    AR.tree_all_reduce_rounded(x, None, torch.bfloat16)
    # admission runs the reference and the candidate, then the call takes the absorbed path
    assert calls["fp32_root"] == 1 and calls["bf16_root"] == 2
    for _ in range(3):
        AR.tree_all_reduce_rounded(x, None, torch.bfloat16)
    assert calls["fp32_root"] == 1 and calls["bf16_root"] == 5
    assert AR.root_cast_counts()["admitted"] == 1


def test_root_cast_reference_is_the_expression_being_replaced(monkeypatch, capsys):
    """The reference must be reduce-then-round, not reduce-with-bf16-root: a one-ULP stub has
    to be refused loudly, with the original expression still returned."""
    import torch

    _reset_root_cast_state()
    monkeypatch.setattr(AR, "dist", _FakeDist())
    monkeypatch.setattr(AR, "_capturing", lambda: False)

    def fake_ar(partial, group=None, out=None, backend=None, root_dtype=None):
        if root_dtype == torch.bfloat16:
            wrong = partial.to(torch.bfloat16).view(torch.int16) + 1
            return wrong.view(torch.bfloat16)
        return partial

    monkeypatch.setattr(AR, "tree_all_reduce", fake_ar)
    AR.set_root_cast_enabled(True)
    x = torch.randn(256, dtype=torch.float32)
    out = AR.tree_all_reduce_rounded(x, None, torch.bfloat16)
    assert torch.equal(AR._bits(out), AR._bits(x.to(torch.bfloat16)))
    assert AR.root_cast_counts()["rejected"] == 1
    assert "REFUSED" in capsys.readouterr().out
    AR.set_root_cast_enabled(False)
    _reset_root_cast_state()


def test_root_cast_peer_veto_and_capture(monkeypatch):
    """Same group-safety properties as the fusion gate: a peer can veto, capture cannot admit."""
    import torch

    _reset_root_cast_state()
    monkeypatch.setattr(AR, "dist", _FakeDist(peer=0))
    monkeypatch.setattr(AR, "_capturing", lambda: False)
    monkeypatch.setattr(
        AR,
        "tree_all_reduce",
        lambda p, group=None, out=None, backend=None, root_dtype=None: (
            p.to(torch.bfloat16) if root_dtype == torch.bfloat16 else p
        ),
    )
    AR.set_root_cast_enabled(True)
    x = torch.randn(64, dtype=torch.float32)
    AR.tree_all_reduce_rounded(x, None, torch.bfloat16)
    assert AR.root_cast_counts()["rejected"] == 1, "a peer veto must refuse the shape"

    _reset_root_cast_state()
    monkeypatch.setattr(AR, "dist", _FakeDist(peer=1))
    monkeypatch.setattr(AR, "_capturing", lambda: True)
    AR.tree_all_reduce_rounded(x, None, torch.bfloat16)
    assert AR.root_cast_counts()["capture_skips"] == 1
    assert AR.root_cast_counts()["shapes"] == {}
    AR.set_root_cast_enabled(False)
    _reset_root_cast_state()


def test_narrowing_root_is_now_a_legal_request():
    """The assert that used to forbid fp32-partial + bf16-root has to admit it, and no more."""
    import torch

    x = torch.randn(8, dtype=torch.float32)
    assert AR.tree_all_reduce(x, root_dtype=torch.bfloat16).dtype == torch.bfloat16
    assert AR.tree_all_reduce(x, root_dtype=torch.float32).dtype == torch.float32
    with pytest.raises(AssertionError):
        AR.tree_all_reduce(x, root_dtype=torch.float64)


def test_the_two_row_parallel_twins_carry_the_same_guard():
    """linear.row_parallel_linear (trainer) and vllm_patch._row_forward (engine) must not drift.

    The absorption is only legal with no bias, fp32 leaves and a bf16 output.
    """
    import inspect

    from skyrl.backends.skyrl_train.isoexec.ops.collectives.pik import linear
    from skyrl.backends.skyrl_train.isoexec.ops.collectives.pik.integrations import (
        vllm_patch,
    )

    for src in (inspect.getsource(linear.row_parallel_linear), inspect.getsource(vllm_patch._row_forward)):
        lines = src.splitlines()
        call = next(
            i for i, ln in enumerate(lines) if "tree_all_reduce_rounded(" in ln and not ln.strip().startswith("#")
        )
        guard = next(
            ln for ln in reversed(lines[:call]) if ln.strip().startswith("if ") and not ln.strip().startswith("#")
        )
        assert "bias is None" in guard, guard
        assert "bf16_leaves" in guard, guard
        assert "bfloat16" in guard, guard
