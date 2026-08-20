"""CPU proof obligations for the FUSED owner-computes MoE combine (one launch per site).

WHAT IS BEING PROVEN, AND WHY IT NEEDS NO GPU. The WAVE-10 claim is "same expression, three
launches down to one". The owner combine's exchange is

    reference : h_in.barrier() ; owner-push kernel ; h_out.barrier()      3 launches
    fused     : owner-push kernel with both barriers folded in            1 launch

and the claim decomposes into two halves, both decidable on a CPU:

  1. THE SOURCE. ``_oc_core`` -- the offsets, the balanced tree over peers, the ascending-expert
     k-sum, the single bf16 round and the stores, i.e. the only lines in either kernel that touch a
     float -- is emitted byte-for-byte identically into the unfused template and the fused one. Not
     "equivalent": the same string. A containment assertion over every generated variant is a
     complete argument that the two kernels compute the same expression, and it fails the instant
     someone edits one template without the other.

  2. THE MACHINE CODE. Triton compiles ahead of time, so ``triton.compile`` against an explicit
     ``GPUTarget("cuda", 90, 32)`` produces real sm_90 PTX on a box with no CUDA device. The tests
     below compile every (world x k x wire) variant in both launch structures and assert the
     ARITHMETIC INSTRUCTION SEQUENCE is identical -- and separately that the barrier lowered to the
     memory-ordering primitives it must (``atom.global.sys.release.*`` to publish,
     ``atom.global.sys.acquire.*`` to spin), because a publish that lowers to a relaxed store is a
     silent wrong value, never a crash.

  3. THE FLAT FLAG INDEX, which is the ONE thing not inherited from pik's 1-D fused all-reduce.
     This kernel's grid is 2-D; two blocks sharing a flag slot would race on the local epoch and
     publish one arrival for two blocks. The slot must be ``program_id(0)*NHB + program_id(1)``.

WHAT IS NOT PROVEN HERE. That the barrier synchronises 8 real GPUs, that release/acquire is
sufficient on live NVLink, or that any of it is faster. Those need the GPU battery
(the private repo's nightly ``pik_fused_collective_test.py`` --phases owner) and a decode re-trace. This
file is the part that must never be allowed to regress silently.
"""

from __future__ import annotations

import importlib
import re
import sys

import pytest
import torch

pytest.importorskip("triton")

# Importing the PIK package normally asks torch for the live device capability while constructing
# its autotune table. This file compiles an explicit SM90 target and promises not to touch a GPU, so
# supply that same architecture fact during import instead of initializing the CUDA driver. Restore
# torch immediately: this is collection-time isolation, not a process-wide test double.
_real_get_device_capability = torch.cuda.get_device_capability
torch.cuda.get_device_capability = lambda _device=None: (9, 0)
try:
    OC = importlib.import_module("skyrl.backends.skyrl_train.isoexec.ops.moe.moe_pik_combine_owner")
    CG = importlib.import_module("skyrl.backends.skyrl_train.isoexec.ops.collectives.pik.codegen")
finally:
    torch.cuda.get_device_capability = _real_get_device_capability


def _reset_fused_owner_state():
    # inlined: the public module keeps the state but not this test hook
    OC._FUSED_ADMIT.clear()
    OC._FUSED_GROUP.clear()
    for key in OC._FUSED_COUNTS:
        OC._FUSED_COUNTS[key] = 0
    OC._FUSED_FIRST_REJECT = {}


WORLDS = (2, 4, 8)
KS = (1, 2, 8)
WIRES = (False, True)  # in_bf16: fp32 wire and bf16 wire


def test_runtime_uses_one_canonical_pik_module_identity():
    """Owner combine must share PIK's registries/pools, not execute a dotted duplicate."""
    canonical = OC._ensure_canonical_pik()
    dotted = "skyrl.backends.skyrl_train.isoexec.ops.collectives.pik"
    assert sys.modules["pik"] is canonical
    assert sys.modules[dotted] is canonical
    assert sys.modules["pik.allreduce"] is sys.modules[f"{dotted}.allreduce"]


def _variants():
    for c in WORLDS:
        for k in KS:
            for bf in WIRES:
                yield c, k, bf


# ---------------------------------------------------------------------------------------
# 1. the arithmetic is the SAME STRING in both templates
# ---------------------------------------------------------------------------------------
def test_core_is_byte_identical_in_both_templates():
    """The only float-touching lines are generated once and pasted into both kernels."""
    for c, k, bf in _variants():
        core = OC._oc_core(c, k, bf)
        assert core in OC._push_src(c, k, bf), (c, k, bf, "unfused")
        assert core in OC._fused_push_src(c, k, bf), (c, k, bf, "fused")


def test_core_is_the_only_arithmetic():
    """Nothing outside ``_oc_core`` adds, multiplies, rounds or converts a float.

    The barrier prologue/epilogue is integer flag traffic only. If a future edit sneaks a float op
    into it, the fused kernel stops being bitwise-equal by construction, and this catches it in the
    source rather than in a bisect of a golden hash.
    """
    for c, k, bf in _variants():
        outside = OC._fused_push_src(c, k, bf).replace(OC._oc_core(c, k, bf), "")
        for banned in ("float32", "bfloat16", "other=0.0", "tl.dot", ".to("):
            assert banned not in outside, (c, k, bf, banned)
        # the only dtype the barrier ever names is the flag/epoch type
        assert set(re.findall(r"tl\.(int\d+|float\d+|bfloat\d+)", outside)) <= {"int32"}


def test_unfused_push_source_unchanged_by_the_refactor():
    """A golden for the reference path: extracting ``_oc_core`` must not have moved a byte.

    The unfused push kernel is what the WAVE-8 bitwise batteries (test_owner_combine_offline.py,
    test_owner_combine_dist.py) proved. If the refactor that made the fused variant possible changed
    its source at all, the generated module changes name-for-content in the owner_combine cache and
    the claim "nothing about the default path moved" is false. Pinned here as a literal.
    """
    assert OC._push_src(2, 2, False) == (
        "\n"
        "import triton\n"
        "import triton.language as tl\n"
        "\n"
        "@triton.jit\n"
        "def kernel(in0, in1, rows_ptr, out0, out1, out_base, T, H, K: tl.constexpr, BLOCK: tl.constexpr):\n"
        "    t = tl.program_id(0)\n"
        "    hb = tl.program_id(1)\n"
        "    hoff = hb * BLOCK + tl.arange(0, BLOCK)\n"
        "    hmask = hoff < H\n"
        "    acc = tl.zeros((BLOCK,), dtype=tl.float32)\n"
        "    for j in tl.static_range(K):\n"
        "        row = tl.load(rows_ptr + t * K + j)\n"
        "        base = row * H + hoff\n"
        "        t0 = tl.load(in0 + base, mask=hmask, other=0.0)\n"
        "        t1 = tl.load(in1 + base, mask=hmask, other=0.0)\n"
        "        t0 = t0 + t1\n"
        "        acc = acc + t0\n"
        "    res = acc.to(tl.bfloat16)\n"
        "    goff = (out_base + t) * H + hoff\n"
        "    tl.store(out0 + goff, res, mask=hmask)\n"
        "    tl.store(out1 + goff, res, mask=hmask)\n"
    )


def test_the_barrier_is_piks_own_emitter_not_a_second_copy():
    """One cross-rank handshake in the stack, not two.

    The barrier whose epoch-bump race was found by reading PTX is hard enough to get right once.
    This module must not carry its own transcription of it: the emitted prologue/epilogue have to be
    exactly what ``pik.codegen._fused_barrier`` produces, character for character.
    """
    for c, k, bf in _variants():
        src = OC._fused_push_src(c, k, bf)
        assert CG._fused_barrier(0, c, "0", "pid") in src, (c, "leading")
        assert CG._fused_barrier(1, c, "wbase1", "NB + pid") in src, (c, "trailing")


# ---------------------------------------------------------------------------------------
# 2. the barrier's SHAPE, in source
# ---------------------------------------------------------------------------------------
def test_publishes_to_every_peer_exactly_once_per_region():
    """A barrier that skips a peer is a barrier that does not synchronise with it."""
    for c, k, bf in _variants():
        src = OC._fused_push_src(c, k, bf)
        assert src.count('sem="release", scope="sys"') == c * 2  # two regions, always
        assert src.count('sem="acquire", scope="sys"') == 2
        for i in range(c):
            assert src.count(f"tl.atomic_xchg(fl{i} + my_slot") == 2
        assert f"fl{c}" not in src  # no phantom peer


def test_wait_precedes_every_peer_read_and_push_precedes_the_release():
    """Ordering, read off the source: arrive+wait, then read peers; store, then arrive+wait."""
    for c, k, bf in _variants():
        src = OC._fused_push_src(c, k, bf)
        lead_wait = src.index("ready0 = tl.min")
        first_peer_read = src.index("tl.load(in0 + base")
        first_store = src.index("tl.store(out0 + goff")
        trail_pub = src.index("+ my_slot1 +")
        assert lead_wait < first_peer_read, (c, "a peer partial is read before every peer arrived")
        assert first_store < trail_pub, (c, "the trailing release precedes this block's pushes")
        # and the push is fenced from the release
        assert "tl.debug_barrier()   # this block's pushes precede its release" in src


def test_flag_slot_is_the_FLAT_2d_block_id():
    """The one thing not inherited from pik's 1-D kernel.

    ``program_id(0)`` alone would give every H-block of a token the SAME flag slot: they would race
    on the local epoch counter and publish one arrival for NHB blocks, so a peer could pass the
    barrier before all of that token's blocks had staged. The slot must be the flat block id.
    """
    for c, k, bf in _variants():
        src = OC._fused_push_src(c, k, bf)
        assert "pid = tl.program_id(0) * NHB + tl.program_id(1)" in src
        assert "NHB: tl.constexpr" in src
        # the flag/epoch expressions use `pid`, never a bare program_id
        for expr in ("ep_ptr + pid", "ep_ptr + NB + pid", "+ my_slot0 + pid", "+ my_slot1 + pid"):
            assert expr in src, (c, expr)


# ---------------------------------------------------------------------------------------
# 3. AOT: real sm_90 PTX, on a box with no GPU
# ---------------------------------------------------------------------------------------
def _compile(c, k, in_bf16, fused):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    fn = OC._owner_push_fused_kernel(c, k, in_bf16) if fused else OC._owner_push_kernel(c, k, in_bf16)
    while not hasattr(fn, "arg_names"):
        fn = fn.fn
    ptr_in = "*bf16" if in_bf16 else "*fp32"
    sig, consts = {}, {}
    for nm in fn.arg_names:
        if nm.startswith("in"):
            sig[nm] = ptr_in
        elif nm.startswith("out") and nm != "out_base":
            sig[nm] = "*bf16"
        elif nm == "rows_ptr":
            sig[nm] = "*i32"
        elif nm.startswith("fl") or nm in ("self_fl", "ep_ptr"):
            sig[nm] = "*i32"
        elif nm in ("K", "BLOCK", "NHB", "WORLD"):
            sig[nm] = "constexpr"
            consts[nm] = {"K": k, "BLOCK": 1024, "NHB": 2, "WORLD": c}[nm]
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


@pytest.mark.parametrize("c,k,in_bf16", list(_variants()))
def test_ptx_float_instruction_multiset_is_identical(c, k, in_bf16):
    """The machine code performs the SAME float operations, in the same quantities.

    WHY A MULTISET AND NOT THE SEQUENCE, which is what pik's own fused-barrier test asserts. The
    linear opcode order is a property of the SCHEDULER, not of the arithmetic. At (c=8, k=8,
    bf16-wire) the fused kernel carries extra live registers for the barrier, and ptxas responds by
    interleaving the (independent, and exactly-lossless) ``cvt.f32.bf16`` peer upcasts differently
    against the ``add.f32``s -- 1032 instructions, identical multiset {512 upcasts, 512 adds, 8
    RNE rounds}, different order. Asserting the order there would fail a kernel that is provably
    fine, which is worse than useless: it trains people to relax the assertion.

    So the ORDER-SENSITIVE obligation is discharged where order actually means something -- the
    TTIR data-flow DAG, below, which is emitted before any scheduling and which pins the
    association, the operands and the rounding. This test keeps the complementary machine-code
    obligation: ptxas must not have CONTRACTED adds into fmas, dropped or added a round, changed a
    rounding mode, or emitted a different number of conversions in one kernel and not the other.
    All of those move the multiset.
    """
    pytest.importorskip("triton.backends.compiler")
    from collections import Counter

    ref = Counter(_fp_ops(_compile(c, k, in_bf16, fused=False)))
    got = Counter(_fp_ops(_compile(c, k, in_bf16, fused=True)))
    assert ref == got, f"c={c} k={k} bf16={in_bf16}\nref={ref}\ngot={got}"
    assert ref, "the combine emitted no float instructions at all -- the compare is vacuous"
    # ...and the multiset is the one the expression predicts, so "equal" cannot mean "equally
    # wrong". BLOCK=1024 over num_warps=4 (128 threads) is 8 elements per thread, and per element a
    # token costs (c-1) tree adds + 1 accumulate = c adds and c peer loads, over k experts.
    per_thread = 1024 // (4 * 32)
    assert ref["cvt.rn.bf16.f32"] == per_thread, "exactly one RNE round per element: the round IS the contract"
    assert ref["add.f32"] == per_thread * k * c, ref
    if in_bf16:
        assert ref["cvt.f32.bf16"] == per_thread * k * c, "one exact widening upcast per peer load"
    assert not any(o.startswith("fma") for o in ref), "an add was contracted into an fma -- a different expression"


# ---------------------------------------------------------------------------------------
# 3b. the TTIR data-flow DAG -- the order-sensitive proof, taken before the scheduler
# ---------------------------------------------------------------------------------------
_SSA = re.compile(r"%[A-Za-z0-9_]+")


def _ttir(c, k, in_bf16, fused, src=None, tag=None):
    """Triton IR straight out of the frontend: SSA, in emission order, before any scheduling."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    if src is None:
        fn = OC._owner_push_fused_kernel(c, k, in_bf16) if fused else OC._owner_push_kernel(c, k, in_bf16)
    else:
        fn = OC._load(tag, src)
    while not hasattr(fn, "arg_names"):
        fn = fn.fn
    ptr_in = "*bf16" if in_bf16 else "*fp32"
    sig, consts = {}, {}
    for nm in fn.arg_names:
        if nm.startswith("in"):
            sig[nm] = ptr_in
        elif nm.startswith("out") and nm != "out_base":
            sig[nm] = "*bf16"
        elif nm == "rows_ptr" or nm.startswith("fl") or nm in ("self_fl", "ep_ptr"):
            sig[nm] = "*i32"
        elif nm in ("K", "BLOCK", "NHB", "WORLD"):
            sig[nm] = "constexpr"
            consts[nm] = {"K": k, "BLOCK": 1024, "NHB": 2, "WORLD": c}[nm]
        else:
            sig[nm] = "i32"
    return triton.compile(ASTSource(fn=fn, signature=sig, constexprs=consts), target=GPUTarget("cuda", 90, 32)).asm[
        "ttir"
    ]


def _stored_value_dags(ttir: str) -> list[tuple[str, str]]:
    """(address expression, stored-value expression) for every float store, fully canonicalised.

    Each value is expanded RECURSIVELY through its whole definition chain down to kernel arguments,
    with SSA numbering erased. So the returned string encodes exactly the three things bit-identity
    of a float reduction depends on -- WHICH values are combined (the chain bottoms out at
    ``ARG<%in3>``, a named kernel parameter, so a peer permutation shows up), in WHAT association
    (the nesting), and with WHAT rounding (``arith.truncf``) -- plus, for free, the ADDRESS each
    value was loaded from and stored to. SSA names differ between the two kernels; canonical text
    does not.
    """
    defs, stores = {}, []
    for raw in ttir.splitlines():
        ln = re.sub(r"\s*loc\(#loc\d*\)", "", raw).strip()
        if ln.startswith("tt.store"):
            ops = _SSA.findall(ln.split(" : ")[0])
            if len(ops) >= 2:
                stores.append((ops[0], ops[1]))
            continue
        m = re.match(r"^(%[A-Za-z0-9_]+) = (.*)$", ln)
        if m:
            defs[m.group(1)] = m.group(2)

    memo: dict[str, str] = {}

    def canon(v: str) -> str:
        if v in memo:
            return memo[v]
        if v not in defs:
            memo[v] = f"ARG<{v}>"  # a kernel parameter: named, so peer identity is preserved
            return memo[v]
        memo[v] = "CYC"
        rhs, parts, last = defs[v], [], 0
        for m in _SSA.finditer(rhs):
            parts.append(rhs[last : m.start()])
            parts.append(canon(m.group(0)))
            last = m.end()
        parts.append(rhs[last:])
        memo[v] = "".join(parts)
        return memo[v]

    out = []
    for ptr, val in stores:
        cv = canon(val)
        if "arith.truncf" in cv:  # the float stores; the flag/epoch stores are integer
            out.append((canon(ptr), cv))
    return out


@pytest.mark.parametrize("c,k,in_bf16", list(_variants()))
def test_ttir_dataflow_dag_is_identical_to_the_unfused_kernel(c, k, in_bf16):
    """THE order-sensitive proof: same expression DAG, address for address, add for add.

    This is what the PTX opcode SEQUENCE was a (fragile) proxy for. Taken at TTIR, before any
    scheduling pass has run, so a reordering of independent instructions cannot perturb it -- while
    a reassociated tree, a permuted peer, a moved rounding point, a changed offset or an extra add
    all do. ``test_ttir_dag_control_*`` below establish that it is not vacuous.
    """
    pytest.importorskip("triton.backends.compiler")
    ref = _stored_value_dags(_ttir(c, k, in_bf16, fused=False))
    got = _stored_value_dags(_ttir(c, k, in_bf16, fused=True))
    assert len(ref) == c, f"expected one push per peer, got {len(ref)}"
    assert ref == got, f"c={c} k={k} bf16={in_bf16}: the fused kernel evaluates a different expression"


def test_ttir_dag_control_a_permuted_tree_differs():
    """Swapping in1 with in2 turns ((L0+L1)+(L2+L3)) into ((L0+L2)+(L1+L3)).

    It must be a re-ASSOCIATION, not a mere re-ordering: IEEE addition is exactly commutative, so
    swapping the two operands of a single add is bitwise-identical and would make a useless control.
    That is also why this needs c >= 4. If this control does not fail, no PASS above means anything.
    """
    pytest.importorskip("triton.backends.compiler")
    c, k, bf = 4, 2, False
    src = OC._fused_push_src(c, k, bf)
    src = src.replace("tl.load(in1 + base", "tl.load(in__A + base")
    src = src.replace("tl.load(in2 + base", "tl.load(in1 + base")
    src = src.replace("tl.load(in__A + base", "tl.load(in2 + base")
    ref = _stored_value_dags(_ttir(c, k, bf, fused=False))
    bad = _stored_value_dags(_ttir(c, k, bf, fused=True, src=src, tag="_owner_ctl_permuted"))
    assert ref != bad, "a re-associated tree produced an identical DAG -- the compare is vacuous"


def test_ttir_dag_control_an_extra_add_differs():
    """A single extra fp32 add before the round is invisible to allclose and must not be here."""
    pytest.importorskip("triton.backends.compiler")
    c, k, bf = 4, 2, False
    src = OC._fused_push_src(c, k, bf).replace("res = acc.to(tl.bfloat16)", "res = (acc + 0.0).to(tl.bfloat16)")
    ref = _stored_value_dags(_ttir(c, k, bf, fused=False))
    bad = _stored_value_dags(_ttir(c, k, bf, fused=True, src=src, tag="_owner_ctl_extra_add"))
    assert ref != bad, "an inserted add produced an identical DAG -- the compare is vacuous"


@pytest.mark.parametrize("c", WORLDS)
def test_ptx_barrier_lowers_to_system_scope_release_acquire(c):
    """The memory ordering must survive the compiler, not just the source."""
    ptx = _compile(c, 2, False, fused=True)
    assert ptx.count("atom.global.sys.release.") == c * 2, ptx.count("atom.global.sys.release.")
    assert ptx.count("atom.global.sys.acquire.") == 2
    assert "bar.sync" in ptx
    # and the reference kernel has none of it -- its barriers are separate launches
    assert "atom.global.sys.release." not in _compile(c, 2, False, fused=False)


@pytest.mark.parametrize("c", WORLDS)
def test_ptx_spin_is_a_real_loop(c):
    """A "barrier" the compiler turned into a straight-line read is not a barrier."""
    lines = _compile(c, 2, False, fused=True).splitlines()
    label_at = {m.group(1): i for i, ln in enumerate(lines) if (m := re.match(r"^(\$L__BB\d+_\d+):", ln.strip()))}
    backward = [
        (i, m.group(1))
        for i, ln in enumerate(lines)
        if (m := re.search(r"bra\s+(\$L__BB\d+_\d+)", ln)) and label_at.get(m.group(1), 1 << 30) < i
    ]
    assert backward, "the spin compiled to straight-line code -- it is not a barrier"
    start, end = label_at[backward[0][1]], backward[0][0]
    assert any(
        "atom.global.sys.acquire." in ln for ln in lines[start:end]
    ), "the peer flag is read once and then spun on a stale register"


def test_ptx_epoch_store_follows_a_barrier():
    """The epoch-bump race, verified in the instruction stream that will run.

    Triton lowers the epoch LOAD unpredicated (every thread needs ``seq`` for its spin compare) and
    the STORE ``@tid==0``-predicated. Warps run independently, so without a ``bar.sync`` between
    them warp 0 can bump the counter before warp 3 reads it; warp 3 then waits for an epoch no peer
    will publish and the collective hangs, intermittently, under load.
    """
    ptx = _compile(4, 2, False, fused=True).splitlines()
    ld = next(i for i, ln in enumerate(ptx) if "ld.global.b32" in ln)
    st = next(i for i, ln in enumerate(ptx) if i > ld and "st.global.b32" in ln)
    assert any("bar.sync" in ln for ln in ptx[ld:st]), "epoch bump is unsynchronised"


# ---------------------------------------------------------------------------------------
# 4. the flag, the default, and the registry
# ---------------------------------------------------------------------------------------
def test_flag_name_and_default_are_off():
    assert OC._FUSED_ENV == "SKYRL_ISOEXEC_PIK_FUSED_OWNER_COMBINE"
    for off in ("", "0", "false", "no", "off", "OFF"):
        assert not OC._env_on(OC._FUSED_ENV, off)
    for on in ("1", "true", "yes", "on"):
        assert OC._env_on(OC._FUSED_ENV, on)


def test_flag_default_from_env_is_off(monkeypatch):
    monkeypatch.delenv("SKYRL_ISOEXEC_PIK_FUSED_OWNER_COMBINE", raising=False)
    assert OC._env_on(OC._FUSED_ENV) is False


def test_admission_key_includes_everything_that_changes_the_generated_kernel_or_its_grid():
    """A verdict recorded under too coarse a key is a verdict applied to a kernel it never proved.

    ``(world, k, wire)`` chooses the generated VARIANT; ``(s, NHB)`` chooses the GRID and therefore
    the flag stride. All five are in the key.
    """
    import inspect

    src = inspect.getsource(OC._owner_push_exchange)
    assert "key = (ex.world, ex.s, triton.cdiv(ex.H, ex.block), ex.k," in src
    assert "ex.wire_bf16" in src


def test_flag_pool_registry_is_keyed_on_the_GROUPS_RANKS_not_just_its_size():
    """pik's own registry keys on (device, world_size), which would hand a tp_ep group a pool
    rendezvoused against a same-sized TP group's peers. This one keys on the rank tuple."""
    import inspect

    src = inspect.getsource(OC._oc_flag_pool)
    assert "_group_key(group)" in src
    assert "_OC_FLAGS" in src and "_OC_FLAG_KEEP" in src


def test_counters_start_clean_and_report_enablement():
    _reset_fused_owner_state()
    c = OC.fused_owner_counts()
    assert c["calls"] == 0 and c["admitted"] == 0 and c["rejected"] == 0
    assert c["first_reject"] is None
    assert "enabled" in c and "shapes" in c


def test_rejection_is_loud_and_remembered_once(capsys):
    _reset_fused_owner_state()
    key = (8, 64, 2, 8, "bf16")
    OC._oc_reject(key, "a made-up reason", "detail")
    OC._oc_reject((8, 32, 2, 8, "bf16"), "a second reason", "")
    out = capsys.readouterr().out
    assert "REFUSED" in out and "a made-up reason" in out
    assert "a second reason" not in out, "only the FIRST rejection is printed, but both are counted"
    c = OC.fused_owner_counts()
    assert c["rejected"] == 2
    assert c["first_reject"]["reason"] == "a made-up reason"
    assert OC._FUSED_ADMIT[key] is False
    _reset_fused_owner_state()


def test_the_grid_gate_defaults_to_the_measured_crossover(monkeypatch):
    """Fusion is bit-neutral but NOT time-neutral, and the gate is the measured crossover.

    The in-kernel handshake publishes to every peer ONCE PER BLOCK, so it costs ~2.0 us/block (flat
    in payload, measured on 8xH100 by holding one payload fixed and sweeping BLOCK 8x) against a
    device-wide symm-mem barrier pair's 10-35 us TOTAL. Break-even is grid ~ 5-17 blocks; the
    default is the conservative centre. Pinned here so nobody "tidies" it into a bigger number
    without a measurement.
    """
    monkeypatch.delenv(OC._MAX_BLOCKS_ENV, raising=False)
    assert OC._MAX_BLOCKS_ENV == "SKYRL_ISOEXEC_PIK_FUSED_OWNER_MAX_BLOCKS"
    assert OC._max_fused_blocks() == 8
    monkeypatch.setenv(OC._MAX_BLOCKS_ENV, "64")
    assert OC._max_fused_blocks() == 64
    monkeypatch.setenv(OC._MAX_BLOCKS_ENV, "not-a-number")
    assert OC._max_fused_blocks() == 8, "a malformed override must fall back, never crash a decode"


def test_the_grid_gate_refuses_the_production_decode_geometry(monkeypatch):
    """The shipped geometry is 128 blocks, so ON must be a LOUD no-op rather than a 268 us tax.

    T=512 (max_num_seqs) over world=8 owns s=64 tokens, H=2048 at BLOCK=1024 is NHB=2, so the grid
    is 128 blocks -- 16x the crossover. The point of the gate is that setting the flag on a
    production arm cannot regress it, and that the arm's own log says the fusion did not engage
    instead of leaving a reader to infer it from a barrier count.
    """
    monkeypatch.delenv(OC._MAX_BLOCKS_ENV, raising=False)
    s, nhb = -(-512 // 8), 2048 // 1024
    assert s * nhb == 128
    assert s * nhb > OC._max_fused_blocks()


def test_served_tick_prints_early_then_rarely(capsys):
    _reset_fused_owner_state()
    for _ in range(3):
        OC._served_tick((8, 64, 2, 8, "bf16"))
    out = capsys.readouterr().out
    assert out.count("[ISOEXEC-MOE-FUSED-OWNER]") == 2, out
    assert "served=" in out
    assert OC.fused_owner_counts()["calls"] == 3
    _reset_fused_owner_state()
