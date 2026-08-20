"""CPU proof obligations for pik's FUSED barrier+reduce collective (one launch per site).

WHAT IS BEING PROVEN HERE, AND WHY IT CAN BE PROVEN WITHOUT A GPU.

The fused kernel's whole claim is "same expression, fewer launches". That claim decomposes
into two halves, and BOTH halves are decidable on a CPU:

  1. THE SOURCE. ``codegen._ar_core`` -- offsets, the balanced tree, the stores, the only
     lines in either kernel that touch a float -- is emitted byte-for-byte identically into
     the unfused template and the fused one. Not "equivalent". The same string. A
     containment assertion over every generated variant is therefore a complete argument
     that the two kernels compute the same expression, and it fails the instant someone
     edits one template without the other.

  2. THE MACHINE CODE. Triton compiles ahead of time: ``triton.compile`` against an explicit
     ``GPUTarget("cuda", 90, 32)`` produces real sm_90 PTX on a box with no CUDA device at
     all. So the tests below compile all 36 variants (world 2/4/8 x one-shot/two-shot x
     {fp32, bf16-wire+bf16-root, bf16-wire+fp32-root} x fused/unfused) and assert that the
     ARITHMETIC INSTRUCTION SEQUENCE of the fused kernel is identical to the unfused one --
     and separately that the barrier lowered to the memory-ordering primitives it must:
     ``atom.global.sys.release.*`` to publish, ``atom.global.sys.acquire.*`` to spin.

     This half is not ceremony. Reading the emitted PTX is what found the epoch-bump race
     (test_epoch_bump_is_barriered): Triton lowers ``tl.load(ep_ptr + pid)`` to an
     UNPREDICATED load -- every thread needs the epoch for its spin comparison -- and
     ``tl.store`` to a ``@tid==0``-predicated store. Warps run independently, so warp 0
     could bump the counter before warp 3 had read it; warp 3 would then wait for an epoch
     no peer ever publishes and the collective would hang, intermittently, under load. No
     amount of running it on 8 idle GPUs would reliably show that.

WHAT IS NOT PROVEN HERE. That the barrier actually synchronises 8 real GPUs, that the
release/acquire pair is sufficient on live NVLink, and that any of it is faster. Those need
the GPU battery (the private repo's nightly ``pik_fused_collective_test.py``) and the decode
re-trace. This file is the part that must never be allowed to regress silently.
"""

from __future__ import annotations

import importlib
import re

import pytest

pytest.importorskip("triton")

AR = importlib.import_module("skyrl.backends.skyrl_train.isoexec.ops.collectives.pik.allreduce")


def _reset_fused_state():
    # inlined: the public module keeps the state but not this test hook
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


# ---------------------------------------------------------------------------------------
# 1. the arithmetic is the SAME STRING in both templates
# ---------------------------------------------------------------------------------------
def test_core_is_byte_identical_in_both_templates():
    """The only float-touching lines are generated once and pasted into both kernels."""
    for c, push, bi, bo in _variants():
        core = CG._ar_core(c, push, bi, bo)
        assert core in CG._ar_tmpl(c, push, bi, bo), (c, push, bi, bo, "unfused")
        assert core in CG._fused_ar_tmpl(c, push, bi, bo), (c, push, bi, bo, "fused")


def test_core_is_the_only_arithmetic():
    """Nothing outside ``_ar_core`` adds, multiplies or converts a float.

    The barrier prologue/epilogue is integer flag traffic only. If a future edit sneaks a
    float op into it, the fused kernel stops being bitwise-equal by construction and this
    catches it in the source, before anyone has to bisect a golden hash.
    """
    for c, push, bi, bo in _variants():
        core = CG._ar_core(c, push, bi, bo)
        outside = CG._fused_ar_tmpl(c, push, bi, bo).replace(core, "")
        for banned in ("float", "other=0.0", "tl.dot"):
            assert banned not in outside, (c, push, bi, bo, banned)
        # the only dtype the barrier ever names is the flag/epoch type
        assert set(re.findall(r"tl\.(int\d+|float\d+|bfloat\d+)", outside)) <= {"int32"}


def test_unfused_template_unchanged_by_the_refactor():
    """A golden for the reference path: extracting ``_ar_core`` must not have moved a byte.

    The unfused kernel is what every frozen bit in this stack was proven against. If the
    refactor that made the fused variant possible changed its source at all, the generated
    module changes name-for-content in the pik cache and the claim "nothing about the
    default path moved" is false. Pinned here as a literal.
    """
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


# ---------------------------------------------------------------------------------------
# 2. the barrier's SHAPE, in source
# ---------------------------------------------------------------------------------------
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
    """One-shot writes only its OWN destination, so nothing of its can be read too early.

    The trailing barrier exists for the push: peers write into MY output buffer and I must
    not return until they have. One-shot's single barrier is the same saving ``_SymPool``'s
    double-buffering already bought the unfused path -- fusion must not silently give it back.
    """
    for c, bi, bo in ((w, x, y) for w in WORLDS for x, y in WIRES):
        one = CG._fused_ar_tmpl(c, False, bi, bo)
        two = CG._fused_ar_tmpl(c, True, bi, bo)
        assert "seq1" not in one and "wbase1 + lane" not in one
        assert "seq1" in two and "wbase1 + lane" in two


def test_epoch_bump_is_barriered():
    """``tl.debug_barrier()`` MUST sit between the epoch load and the epoch store.

    Every thread reads the epoch (it needs ``seq`` to compare against peers' flags); one
    thread writes it back. Without the barrier a fast warp bumps the counter before a slow
    warp has read it, the slow warp waits for ``seq+1``, and the collective deadlocks.
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


# ---------------------------------------------------------------------------------------
# 3. the flag-array layout
# ---------------------------------------------------------------------------------------
def test_flag_slots_never_collide():
    """Every (region, source rank, block) owns a distinct int32. Nothing else is a barrier."""
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


# ---------------------------------------------------------------------------------------
# 4. the branch rule is SHARED, so fusion cannot change which expression runs
# ---------------------------------------------------------------------------------------
def test_shot_rule_is_the_historical_expression():
    """``_p2p_shot`` must reproduce the inline wave6c test exactly, at every size.

    Fusion is a launch-count change. If it also moved the one-shot/two-shot boundary it
    would be changing which of two (bitwise-equal, but differently scheduled) kernels runs,
    and any perf number afterwards would be measuring two things at once.
    """
    budget = AR.P2P_ONESHOT_MAX_BYTES
    for world in WORLDS:
        for elt in (2, 4):
            for n in (1, 1024, 4096, 65536, 262144, 1 << 20, 1 << 22):
                assert AR._p2p_shot(n, elt, world) == (n * elt * (world - 1) > budget)


def test_shot_rule_has_no_model_or_arch_constant():
    """The only number in the rule is the documented env budget -- no shape, no arch, no model."""
    import inspect

    src = inspect.getsource(AR._p2p_shot)
    body = src.split('"""')[-1]
    assert "P2P_ONESHOT_MAX_BYTES" in body
    for hardcoded in ("2048", "4096", "glm", "GLM", "47", "sm90", "SM90"):
        assert hardcoded not in body


# ---------------------------------------------------------------------------------------
# 5. AOT: real sm_90 PTX, on a box with no GPU
# ---------------------------------------------------------------------------------------
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
    """The strongest form of "the arithmetic did not change": same sm_90 instructions.

    Bit-identity of a float reduction is decided by (a) which values are added, (b) in what
    association, and (c) with what rounding. All three are visible in the PTX opcode stream,
    and the fused kernel's is asserted equal to the reference's here -- before a single GPU
    is booked.
    """
    pytest.importorskip("triton.backends.compiler")
    ref = _fp_ops(_compile(c, push, in_bf16, out_bf16, fused=False))
    got = _fp_ops(_compile(c, push, in_bf16, out_bf16, fused=True))
    assert ref == got, f"c={c} push={push} bf16={in_bf16}/{out_bf16}\nref={ref}\ngot={got}"
    assert ref, "the reduction emitted no float instructions at all -- the compare is vacuous"


@pytest.mark.parametrize("c,push", [(w, p) for w in WORLDS for p in (False, True)])
def test_ptx_barrier_lowers_to_system_scope_release_acquire(c, push):
    """The memory ordering must survive the compiler, not just the source.

    A publish that lowers to a plain relaxed store lets a peer observe the flag before the
    data -- a silent wrong value, never a crash. So the requirement is checked where it is
    actually decided: ``atom.global.sys.release.*`` to arrive, ``atom.global.sys.acquire.*``
    to spin, one release per peer per region.
    """
    ptx = _compile(c, push, False, False, fused=True)
    regions = 2 if push else 1
    assert ptx.count("atom.global.sys.release.") == c * regions, ptx.count("atom.global.sys.release.")
    assert ptx.count("atom.global.sys.acquire.") == regions
    assert "bar.sync" in ptx
    # and the reference kernel has none of it -- its barriers are separate launches
    assert "atom.global.sys.release." not in _compile(c, push, False, False, fused=False)


@pytest.mark.parametrize("c", WORLDS)
def test_ptx_spin_is_a_real_loop(c):
    """A "barrier" the compiler turned into a straight-line read is not a barrier.

    Triton's reduction broadcasts through shared memory, so the loop body contains
    ``bar.sync``; that is only legal because the exit condition is UNIFORM across the CTA
    (every thread reduces over the same WORLD flags). A backward branch must be present.
    """
    import re

    lines = _compile(c, False, False, False, fused=True).splitlines()
    label_at = {m.group(1): i for i, ln in enumerate(lines) if (m := re.match(r"^(\$L__BB\d+_\d+):", ln.strip()))}
    backward = [
        (i, m.group(1))
        for i, ln in enumerate(lines)
        if (m := re.search(r"bra\s+(\$L__BB\d+_\d+)", ln)) and label_at.get(m.group(1), 1 << 30) < i
    ]
    assert backward, "the spin compiled to straight-line code -- it is not a barrier"
    # and the acquire atomic is INSIDE that loop, not hoisted above it
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


# ---------------------------------------------------------------------------------------
# 6. dispatch, admission and the loud fallback -- no CUDA, all stubs
# ---------------------------------------------------------------------------------------
class _FakeReduceOp:
    MIN = "min"


class _FakeDist:
    """Enough torch.distributed for the dispatcher, including the AGREEMENT collectives.

    ``peer`` is what the (fictional) other ranks contribute to every ``all_reduce(MIN)``: 1
    means they agree with this rank, 0 means at least one of them vetoes. That is the only
    lever the rank-invariance tests need -- the deadlock being guarded against is exactly a
    rank acting on its own verdict when a peer's differs.
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
    """Nobody gets this by accident: the module-level default must be OFF."""
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
    """The positive control of the admission machinery itself.

    A fused kernel that reassociated the tree is still allclose and still symmetric. It must
    be caught by the BIT PATTERN compare, refused for that shape forever, and it must say so
    in the log -- a silent fallback is the failure mode this stack has been bitten by before.
    """
    import torch

    _reset_fused_state()
    monkeypatch.setattr(AR, "dist", _FakeDist())
    monkeypatch.setattr(AR, "_capturing", lambda: False)
    monkeypatch.setattr(AR, "_p2p_unfused", lambda p, g, o, r=None: p * 2)
    # one ULP out: allclose would pass this, a bit compare must not
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
    """A compare needs a host sync; a capture forbids one. So capture keeps the reference
    path -- and must NOT record a verdict, or the shape would be locked out of the fused
    path forever because of when it happened to be seen first."""
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
    """The tree is codegen'd per leaf count and a balanced tree needs a power of two.

    world=6 must fall back, not raise out of the collective and not produce a lopsided tree.
    """
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


# ---------------------------------------------------------------------------------------
# 7. RANK INVARIANCE -- the decision must be the group's, never a rank's own
# ---------------------------------------------------------------------------------------
# The two launch structures issue different rendezvous: the reference's barriers are their own
# symm-mem kernels, the fused kernel's are inside it on its own flag pool. A rank that picks
# one while a peer picks the other does not run slowly, it HANGS -- and inside a captured
# decode graph a hang has no error, no output, and holds the node. So every input to the
# choice is either group-wide by construction or made group-wide by a collective, and these
# are the tests for the two that are not free.
def test_enablement_is_settled_by_a_collective_before_the_flag_is_read(monkeypatch, capsys):
    """A rank with the flag ON and a peer with it OFF must fall back, not deadlock.

    The handshake has to run on EVERY rank whatever its own flag says -- a rank that skipped
    it because its own flag was off would leave the others blocked in the collective. So the
    dispatcher calls it before it tests ``_FUSED_ON``, and this asserts both halves: the
    veto is honoured, and the collective happened at all.
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
    """The banner is the evidence (and the collective is what makes it true for the group)."""
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
    """This rank's own compare PASSES and the shape is still refused.

    That is the point: an admission verdict is a rank-local computation, so acting on it
    locally is the deadlock. The group's answer is the AND of every rank's, and a rank whose
    own answer was yes must follow the group to the reference path and say why.
    """
    import torch

    _reset_fused_state()

    class _VetoAfterEnable(_FakeDist):
        """Agrees to enablement, then vetoes the shape -- the interesting ordering."""

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
    """The number of synchronisation points must be a function of group-wide facts alone.

    One-shot fuses ONE barrier into the kernel, two-shot TWO. If that count could differ
    between ranks the collective would hang, so the predicate that decides it is checked here
    to depend on nothing else: same (numel, element size, world) in, same answer out, for any
    rank, any device, any call order.
    """
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
    """NB is baked into every flag index, so ranks must agree on it without negotiating.

    It is derived from the grid, and the grid from numel and BLOCK -- all group-wide. The
    growth rule is geometric on the same rank-invariant sequence, so the pools grow in
    lockstep; this pins the rule so it cannot become dependent on anything local.
    """
    import inspect

    src = inspect.getsource(AR._flag_pool)
    assert "get_rank" not in src and "device_count" not in src
    assert "max(nb, 2 * pool.nb if pool is not None else 0, 1024)" in src


# ---------------------------------------------------------------------------------------
# 8. absorbing the ROOT's fp32 -> bf16 round into the reduce kernel
# ---------------------------------------------------------------------------------------
# Production runs PIK_LEAF_DTYPE=fp32 (a CONTRACT constant, not a per-model choice) against a
# bf16 residual stream, so every row-parallel site ends with `full.to(bf16)` as its own
# elementwise kernel over all of [M, N]. The reduce kernel already stores those elements.
#
# Unlike the fusion, this MOVES A ROUNDING POINT, so it is a different claim with a different
# proof obligation -- hence a different flag and a different reference (the round done
# afterwards, which is the expression being replaced).
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
    """Nobody moves a rounding point by accident."""
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
    # admission runs the reference (fp32 root + .to) and the candidate (bf16 root), then the
    # call itself takes the absorbed path
    assert calls["fp32_root"] == 1 and calls["bf16_root"] == 2
    for _ in range(3):
        AR.tree_all_reduce_rounded(x, None, torch.bfloat16)
    assert calls["fp32_root"] == 1 and calls["bf16_root"] == 5
    assert AR.root_cast_counts()["admitted"] == 1


def test_root_cast_reference_is_the_expression_being_replaced(monkeypatch, capsys):
    """The reference must be reduce-then-round, not reduce-with-bf16-root.

    Comparing the new path to itself would pass unconditionally. Here the in-kernel round is
    stubbed one ULP off -- allclose would not notice, the bit compare must -- and the shape has
    to be refused, loudly, with the ORIGINAL expression still returned.
    """
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
    """Same two group-safety properties as the fusion gate: a peer can veto, capture cannot admit."""
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
    """linear.row_parallel_linear and vllm_patch._row_forward must not drift.

    They are the same expression written twice -- one for the trainer, one for the engine --
    and the whole zero-KL claim is that they agree bit for bit. The absorption is only legal
    with NO BIAS, fp32 leaves and a bf16 output; if one twin grows the optimisation and the
    other does not, or one drops a condition, the two stop being the same function.
    """
    import inspect

    from skyrl.backends.skyrl_train.isoexec.ops.collectives.pik import linear
    from skyrl.backends.skyrl_train.isoexec.ops.collectives.pik.integrations import (
        vllm_patch,
    )

    for src in (inspect.getsource(linear.row_parallel_linear), inspect.getsource(vllm_patch._row_forward)):
        lines = src.splitlines()
        call = next(
            i for i, ln in enumerate(lines)
            if "tree_all_reduce_rounded(" in ln and not ln.strip().startswith("#")
        )
        guard = next(
            ln for ln in reversed(lines[:call])
            if ln.strip().startswith("if ") and not ln.strip().startswith("#")
        )
        assert "bias is None" in guard, guard
        assert "bf16_leaves" in guard, guard
        assert "bfloat16" in guard, guard
