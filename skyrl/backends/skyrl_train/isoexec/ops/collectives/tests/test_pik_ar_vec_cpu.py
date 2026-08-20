"""The vec element-to-thread layout must not move a single element's value DAG.

``codegen._ar_core(vec=N)`` reshapes the SAME BLOCK of elements as (BLOCK//N, N) so each thread
owns N contiguous elements and Triton can emit 128-bit accesses (measured -2.2 us on the
production two-shot push; see the SKYRL_ISOEXEC_PIK_AR_VEC flags entry). That is a claim about
LAYOUT ONLY: which thread computes an element moves; the element set, each element's load
sources, the tree association and the single rounding must not.

WHAT IS PINNED HERE, at TTIR -- before any scheduling pass, the same altitude as the
owner-combine DAG proof (test_owner_combine_fused_cpu.py) and for the same reason: machine-code
order is scheduler noise, TTIR data flow is the expression. From each compiled kernel we take
every float store's value expression, canonicalised through its whole SSA chain, and reduce it
to its OPERATION SKELETON: the prefix sequence of {arith.addf, arith.truncf, ARG<in_i>} tokens.
Binary ops in prefix order identify the tree uniquely, so skeleton equality pins association,
peer operand order and rounding -- while the addresses (which legitimately differ between
layouts: that is the whole point) are excluded. The controls below establish the skeleton is
not vacuous: a permuted tree and an inserted add must both move it.

The live half of the obligation -- int bit-pattern equality on real operands, per shape --
is ``allreduce._vec_admit`` and the GPU battery; this file is the structural half.

Run (CPU only, needs a Triton with the NVIDIA backend importable):
    /mnt/local_storage/venvs/skyrl-isoexec/bin/python -m pytest <thisfile> -q
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_COLLECTIVES_DIR = os.path.dirname(_HERE)


def _load_by_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_BOOT = _load_by_path("_ix_pik_bootstrap_arvec", os.path.join(_COLLECTIVES_DIR, "pik_bootstrap.py"))
_BOOT.ensure_pik()
import pik.codegen as CG  # type: ignore  # noqa: E402

_SSA = re.compile(r"%[A-Za-z0-9_]+")
_TOK = re.compile(r"arith\.addf|arith\.truncf|arith\.extf|ARG<%in\d+>")


def _ttir_of_src(src: str, tag: str, c: int, push: bool, out_bf16: bool, in_bf16: bool = False):
    triton = pytest.importorskip("triton")
    pytest.importorskip("triton.backends.compiler")
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    fn = CG._load(tag, src)
    while not hasattr(fn, "arg_names"):
        fn = fn.fn
    sig, consts = {}, {}
    for nm in fn.arg_names:
        if nm.startswith("in"):
            sig[nm] = "*bf16" if in_bf16 else "*fp32"
        elif nm.startswith("out"):
            sig[nm] = "*bf16" if out_bf16 else "*fp32"
        elif nm == "BLOCK":
            sig[nm] = "constexpr"
            consts[nm] = 1024
        else:
            sig[nm] = "i32"
    return triton.compile(ASTSource(fn=fn, signature=sig, constexprs=consts), target=GPUTarget("cuda", 90, 32)).asm[
        "ttir"
    ]


def _stored_value_exprs(ttir: str) -> list[str]:
    """Every float store's value, canonicalised through its whole SSA chain (owner-test method)."""
    defs, stores = {}, []
    for raw in ttir.splitlines():
        ln = re.sub(r"\s*loc\(#loc\d*\)", "", raw).strip()
        if ln.startswith("tt.store"):
            ops = _SSA.findall(ln.split(" : ")[0])
            if len(ops) >= 2:
                stores.append(ops[1])
            continue
        m = re.match(r"^(%[A-Za-z0-9_]+) = (.*)$", ln)
        if m:
            defs[m.group(1)] = m.group(2)

    memo: dict[str, str] = {}

    def canon(v: str) -> str:
        if v in memo:
            return memo[v]
        if v not in defs:
            memo[v] = f"ARG<{v}>"
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

    return [canon(v) for v in stores]


def _skeletons(
    c: int,
    push: bool,
    out_bf16: bool,
    vec: int,
    src: str | None = None,
    tag: str | None = None,
    in_bf16: bool = False,
):
    """The op skeleton of every FLOAT store: prefix {addf, truncf, extf, ARG<in_i>} token sequence."""
    if src is None:
        src = CG._ar_tmpl(c, push, in_bf16, out_bf16, vec)
        tag = f"_arvec_test_c{c}_{int(push)}_{int(out_bf16)}{'_bi' if in_bf16 else ''}_v{vec}"
    exprs = _stored_value_exprs(_ttir_of_src(src, tag, c, push, out_bf16, in_bf16))
    skels = []
    for e in exprs:
        toks = _TOK.findall(e)
        if any(t.startswith("arith.") for t in toks):  # float stores only (flag stores have none)
            skels.append(tuple(toks))
    return skels


WORLD = 8


@pytest.mark.parametrize("push,out_bf16", [(True, True), (True, False), (False, True), (False, False)])
def test_vec4_value_dag_skeleton_equals_vec1(push, out_bf16):
    ref = _skeletons(WORLD, push, out_bf16, vec=1)
    got = _skeletons(WORLD, push, out_bf16, vec=4)
    n_stores = WORLD if push else 1
    assert len(ref) == n_stores, f"expected {n_stores} float stores in the reference, got {len(ref)}"
    assert ref == got, "the vec layout changed a stored value's expression skeleton"
    # ...and the skeleton is the one the tree predicts, so equal cannot mean equally wrong:
    # c-1 adds over the 8 peers in ascending order, one truncf iff the root narrows.
    args = [t for t in ref[0] if t.startswith("ARG")]
    assert args == [f"ARG<%in{i}>" for i in range(WORLD)], args
    assert sum(1 for t in ref[0] if t == "arith.addf") == WORLD - 1
    assert sum(1 for t in ref[0] if t == "arith.truncf") == (1 if out_bf16 else 0)


def test_control_a_permuted_tree_moves_the_skeleton():
    """Swapping in1 with in2 re-ASSOCIATES ((L0+L1)+(L2+L3)) into ((L0+L2)+(L1+L3)); c>=4."""
    src = CG._ar_tmpl(WORLD, True, False, True, 4)
    src = src.replace("tl.load(in1 + offs", "tl.load(in__A + offs")
    src = src.replace("tl.load(in2 + offs", "tl.load(in1 + offs")
    src = src.replace("tl.load(in__A + offs", "tl.load(in2 + offs")
    ref = _skeletons(WORLD, True, True, vec=1)
    bad = _skeletons(WORLD, True, True, vec=4, src=src, tag="_arvec_ctl_perm")
    assert ref != bad, "a permuted peer tree produced an identical skeleton -- the compare is vacuous"


def test_control_an_extra_add_moves_the_skeleton():
    src = CG._ar_tmpl(WORLD, True, False, True, 4)
    src = src.replace("t0.to(tl.bfloat16)", "(t0 + 0.0).to(tl.bfloat16)")
    ref = _skeletons(WORLD, True, True, vec=1)
    bad = _skeletons(WORLD, True, True, vec=4, src=src, tag="_arvec_ctl_add")
    assert ref != bad, "an inserted add produced an identical skeleton -- the compare is vacuous"


def test_vec1_emission_is_byte_identical_to_the_historical_form():
    """vec=1 must be the SAME STRING the pre-vec generator emitted: the default path cannot move."""
    core = CG._ar_core(WORLD, True, False, True)
    assert "offs = base + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n" in core
    assert "[:, None]" not in core, "vec=1 must keep the historical 1-D arange"
    assert CG._ar_core(WORLD, True, False, True) == CG._ar_core(WORLD, True, False, True, 1)


def test_fused_templates_still_embed_the_default_core():
    """The barrier-fused kernels stay on the vec=1 core: fusion proofs must not silently drift."""
    core = CG._ar_core(WORLD, True, False, True, 1)
    assert core in CG._fused_ar_tmpl(WORLD, True, False, True)
    assert core in CG._dw_fused_ar_tmpl(WORLD, True, False, True)


# ---------------------------------------------------------------------------------------------
# bf16 WIRE forms (in_bf16=True): the leaf-dtype contract's engine-side kernels.
#
# Under a bf16-leaf plan at TP == G, each rank's partial IS a leaf and crosses the wire in bf16
# (ReductionPlan.leaf_dtype; ../LEAF_DTYPE.md). The reduce kernel must then evaluate EXACTLY
#     tree( extf(leaf_0) ... extf(leaf_{c-1}) )  in fp32,  truncf'd once iff the root narrows
# -- one exact upcast per peer, adds in fp32, at most one rounding. These tests pin that value
# DAG at TTIR for the bf16-wire forms exactly as the fp32-wire tests above do, including that
# the vec layout does not move it. (The live half is allreduce._vec_admit per shape, unchanged:
# its admission key already carries the wire dtype string.)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("push,out_bf16", [(True, True), (True, False), (False, True), (False, False)])
def test_bf16_wire_vec4_value_dag_skeleton_equals_vec1(push, out_bf16):
    ref = _skeletons(WORLD, push, out_bf16, vec=1, in_bf16=True)
    got = _skeletons(WORLD, push, out_bf16, vec=4, in_bf16=True)
    n_stores = WORLD if push else 1
    assert len(ref) == n_stores, f"expected {n_stores} float stores in the reference, got {len(ref)}"
    assert ref == got, "the vec layout changed a stored value's expression skeleton (bf16 wire)"
    # The skeleton the bf16-leaf contract predicts, so equal cannot mean equally wrong:
    # every peer leaf is upcast EXACTLY ONCE (extf, the exact bf16->fp32 embedding), the c-1
    # adds run over the 8 peers in ascending order IN FP32, and the root is rounded at most
    # once (truncf iff out_bf16) -- adds in fp32, single rounding point, no internal-node round.
    args = [t for t in ref[0] if t.startswith("ARG")]
    assert args == [f"ARG<%in{i}>" for i in range(WORLD)], args
    assert sum(1 for t in ref[0] if t == "arith.extf") == WORLD
    assert sum(1 for t in ref[0] if t == "arith.addf") == WORLD - 1
    assert sum(1 for t in ref[0] if t == "arith.truncf") == (1 if out_bf16 else 0)


def test_bf16_wire_skeleton_differs_from_fp32_wire_only_by_the_upcasts():
    """Same tree, same order, same rounding -- deleting the extf tokens recovers the fp32-wire
    skeleton exactly. This is the 'only the peer-load dtype narrows' claim as a machine check."""
    for push, out_bf16 in ((True, True), (True, False), (False, True), (False, False)):
        wide = _skeletons(WORLD, push, out_bf16, vec=1, in_bf16=False)
        narrow = _skeletons(WORLD, push, out_bf16, vec=1, in_bf16=True)
        stripped = [tuple(t for t in s if t != "arith.extf") for s in narrow]
        assert stripped == wide, (push, out_bf16)


def test_bf16_wire_control_a_permuted_tree_moves_the_skeleton():
    """The bf16-wire compare must not be vacuous either: a permuted peer tree must move it."""
    src = CG._ar_tmpl(WORLD, True, True, True, 4)
    src = src.replace("tl.load(in1 + offs", "tl.load(in__A + offs")
    src = src.replace("tl.load(in2 + offs", "tl.load(in1 + offs")
    src = src.replace("tl.load(in__A + offs", "tl.load(in2 + offs")
    ref = _skeletons(WORLD, True, True, vec=1, in_bf16=True)
    bad = _skeletons(WORLD, True, True, vec=4, src=src, tag="_arvec_ctl_perm_bi", in_bf16=True)
    assert ref != bad, "a permuted peer tree produced an identical skeleton -- the compare is vacuous"
