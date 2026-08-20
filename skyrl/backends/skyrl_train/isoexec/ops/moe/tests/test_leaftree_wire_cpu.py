"""CPU proof obligations for the WAVE-11 leaftree WIRE STORE (owner-combine wire staging).

THE CLAIM. With ``WIRE_STORE=1`` and ``N_LEAVES=1`` the fc2 leaf-tree kernel stores, into the bf16
symmetric staging buffer, EXACTLY the bytes the production path produced by storing fp32 and letting
the owner combine's stage ``copy_`` RNE-round them: ``stored_wire == truncf(stored_ref)``, where
``stored_ref`` is the reference kernel's fp32 store. That truncf IS the dtype-converting ``copy_``
(both are RNE). So proving the store-DAG relation at TTIR -- before any scheduling pass -- plus the
mathematical identity that RNE(fp32-promotion-of-bf16) is the bf16 value itself, discharges the
whole "no stored float's value moves" obligation on the CPU.

WHAT IS NOT PROVEN HERE: that the symmetric buffer plumbing stages the right buffer phase, or that
live operands agree -- that is the per-geometry live admission
(``moe_fused_experts._leaftree_wire_dispatch``, int16 bit compare, host-synced, eager-only) and the
GPU battery. This file is the part that must never regress silently, with controls that MUST fail
so a PASS cannot be vacuous.
"""

from __future__ import annotations

import pathlib
import re
import textwrap

import pytest

pytest.importorskip("triton")

import triton
import triton.language as tl

from skyrl.backends.skyrl_train.isoexec.ops.moe import (
    moe_fused_leaftree as LT,
)

_SSA = re.compile(r"%[A-Za-z0-9_]+")


# ---------------------------------------------------------------------------------------
# compile helpers: real TTIR at sm_90, on a box with no GPU
# ---------------------------------------------------------------------------------------
def _kernel_fn(src: str | None, tag: str):
    """The production kernel, or a DOCTORED copy of its source (for the controls).

    The doctored copy is written to a real file and imported, the same trick
    ``moe_pik_combine_owner._load`` uses, because Triton's frontend re-reads the source."""
    if src is None:
        return LT.fused_moe_leaftree_kernel
    import importlib.util
    import os
    import sys

    cache = pathlib.Path(os.environ.get("PIK_CACHE", pathlib.Path.home() / ".cache" / "pik")) / "leaftree_wire_tests"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{tag}.py"
    if not path.exists() or path.read_text() != src:
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(src)
        tmp.replace(path)
    spec = importlib.util.spec_from_file_location(f"leaftree_wire_tests.{tag}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.fused_moe_leaftree_kernel


def _base_src() -> str:
    """The production kernel's own source, as a standalone module."""
    import inspect

    fn = LT.fused_moe_leaftree_kernel
    while not hasattr(fn, "__wrapped__") and hasattr(fn, "fn"):
        fn = fn.fn
    body = textwrap.dedent(inspect.getsource(fn))
    if body.startswith("@triton.jit"):
        body = body[len("@triton.jit") :].lstrip("\n")
    return "import triton\nimport triton.language as tl\n\n@triton.jit\n" + body


def _ttir(wire: bool, n_leaves: int = 1, src: str | None = None, tag: str | None = None) -> str:
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    fn = _kernel_fn(src, tag or "prod")
    while not hasattr(fn, "arg_names"):
        fn = fn.fn
    sig, consts = {}, {}
    for nm in fn.arg_names:
        if nm in ("a_ptr", "b_ptr"):
            sig[nm] = "*bf16"
        elif nm == "c_ptr":
            sig[nm] = "*bf16" if wire else "*fp32"
        elif nm in ("sorted_token_ids_ptr", "expert_ids_ptr", "num_tokens_post_padded_ptr"):
            sig[nm] = "*i32"
        elif nm in (
            "BLOCK_SIZE_M",
            "BLOCK_SIZE_N",
            "BLOCK_SIZE_K",
            "GROUP_SIZE_M",
            "N_LEAVES",
            "compute_type",
            "WIRE_STORE",
        ):
            sig[nm] = "constexpr"
            consts[nm] = {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 1,
                "N_LEAVES": n_leaves,
                "compute_type": tl.bfloat16,
                "WIRE_STORE": wire,
            }[nm]
        else:
            sig[nm] = "i32"
    return triton.compile(ASTSource(fn=fn, signature=sig, constexprs=consts), target=GPUTarget("cuda", 90, 32)).asm[
        "ttir"
    ]


def _stored_float_dags(ttir: str) -> list[str]:
    """The fully-inlined value DAG of every float store, canonicalised (SSA names erased).

    Same construction as test_owner_combine_fused_cpu._stored_value_dags: WHICH values are
    combined, in WHAT association, with WHAT rounding -- everything bit-identity depends on."""
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

    out = []
    for val in stores:
        cv = canon(val)
        if "truncf" in cv or "tt.dot" in cv:  # float stores; index/flag stores are integer
            out.append(cv)
    return out


def _strip_types(expr: str) -> str:
    """Erase tensor-type annotations so fp32 and bf16 instantiations of the SAME chain compare."""
    return re.sub(r"tensor<[^>]*>", "T", expr)


def _wire_matches_truncf_of_ref(ref_val: str, wire_val: str) -> bool:
    """The claim, as a relation on canonical DAG text: the wire store is the RNE round of the
    reference store -- either literally ``truncf(ref)`` (round-trip kept) or, if the frontend
    folded the exact round-trip, the reference's own inner bf16 value (``ref == extf(wire)``)."""
    r, w = _strip_types(ref_val), _strip_types(wire_val)
    if re.fullmatch(r"arith\.truncf\s+" + re.escape(r) + r"\s*:.*", w):
        return True
    if re.fullmatch(r"arith\.extf\s+" + re.escape(w) + r"\s*:.*", r):
        return True
    return False


# ---------------------------------------------------------------------------------------
# 1. the store-DAG relation, at the production tile config
# ---------------------------------------------------------------------------------------
def test_wire_store_is_the_rne_round_of_the_reference_store():
    """stored_wire == truncf(stored_ref): the kernel's wire store IS the stage copy_'s round."""
    ref = _stored_float_dags(_ttir(wire=False))
    got = _stored_float_dags(_ttir(wire=True))
    assert len(ref) == 1 and len(got) == 1, (len(ref), len(got))
    assert _wire_matches_truncf_of_ref(ref[0], got[0]), f"\nref={ref[0]}\n\nwire={got[0]}"


def test_reference_store_shape_is_pinned():
    """The WIRE_STORE constexpr must not have perturbed the production (fp32) expression: still
    exactly one float store, still ``extf(truncf(acc))`` at one leaf.

    The accumulator is a loop-carried block argument in TTIR (the runtime K-loop), so the DAG
    bottoms out at ``ARG<%acc>`` -- the chain INSIDE the loop is pinned separately by
    ``test_wire_constexpr_touches_only_the_store`` (source containment: WIRE_STORE cannot reach
    the accumulator) and by the live per-geometry bit admission."""
    ref = _stored_float_dags(_ttir(wire=False))
    assert len(ref) == 1
    r = _strip_types(ref[0])
    assert re.fullmatch(r"arith\.extf arith\.truncf ARG<%[A-Za-z0-9_#]+>[^:]*: T to T : T to T", r), r
    # exactly ONE rounding point in the whole chain: the leaf round. A second would be a new value.
    assert r.count("arith.truncf") == 1, r


def test_wire_constexpr_touches_only_the_store():
    """Source containment: ``WIRE_STORE`` appears in the kernel body ONLY inside the final store
    branch, so it cannot perturb the accumulator chain the store DAGs bottom out at."""
    src = _base_src()
    tail = src[src.index("offs_cn = pid_n") :]
    head = src[: src.index("offs_cn = pid_n")]
    head_body = head[head.index("def fused_moe_leaftree_kernel") :]
    # the signature declares it; the body before the store must never read it
    sig_end = head_body.index("):")
    assert "WIRE_STORE" in head_body[:sig_end]
    assert "WIRE_STORE" not in head_body[sig_end:], "WIRE_STORE leaked upstream of the store"
    assert tail.count("if WIRE_STORE:") == 1


def test_wire_store_has_no_second_rounding_point():
    """truncf(extf(truncf(x))) == truncf(x) is an identity ONLY because the middle type is wider.
    The wire DAG must contain no rounding through any OTHER type -- one bf16 leaf round, plus at
    most the folded round-trip itself."""
    got = _stored_float_dags(_ttir(wire=True))[0]
    g = _strip_types(got)
    assert g.count("arith.truncf") <= 2, g
    assert "f16" not in got.replace("bf16", ""), "a half-precision cast appeared in the wire chain"


# ---------------------------------------------------------------------------------------
# 2. controls -- each MUST fail the comparator, or a PASS above is vacuous
# ---------------------------------------------------------------------------------------
def _doctored(replacement: str, tag: str) -> str:
    src = _base_src()
    needle = "tl.store(c_ptrs, result.to(compute_type), mask=c_mask)"
    assert needle in src, "the wire store line moved; update the controls"
    return src.replace(needle, replacement)


def test_control_an_extra_add_differs():
    """A single extra fp32 add before the round is invisible to allclose and must be caught."""
    src = _doctored("tl.store(c_ptrs, (result + 0.0).to(compute_type), mask=c_mask)", "extra_add")
    ref = _stored_float_dags(_ttir(wire=False))
    bad = _stored_float_dags(_ttir(wire=True, src=src, tag="ctl_extra_add"))
    assert not _wire_matches_truncf_of_ref(ref[0], bad[0]), "an inserted add passed -- the compare is vacuous"


def test_control_a_double_round_differs():
    """Rounding through fp16 on the way to bf16 changes stored values and must be caught."""
    src = _doctored(
        "tl.store(c_ptrs, result.to(tl.float16).to(tl.float32).to(compute_type), mask=c_mask)",
        "double_round",
    )
    ref = _stored_float_dags(_ttir(wire=False))
    bad = _stored_float_dags(_ttir(wire=True, src=src, tag="ctl_double_round"))
    assert not _wire_matches_truncf_of_ref(ref[0], bad[0]), "a double round passed -- the compare is vacuous"


def test_the_frontend_folds_the_exact_round_trip():
    """DOCUMENTED BEHAVIOUR, not an obligation: Triton's frontend folds
    ``truncf(extf(truncf(acc)))`` to ``truncf(acc)`` -- the two ARE the same value (RNE round-trip
    through the wider type is the identity), which is the entire wave-11 claim stated by the
    compiler itself. Pinned so a Triton upgrade that stops folding flips us onto the
    ``truncf(ref)`` branch of the comparator knowingly rather than silently."""
    got = _strip_types(_stored_float_dags(_ttir(wire=True))[0])
    ref = _strip_types(_stored_float_dags(_ttir(wire=False))[0])
    assert got.startswith("arith.truncf"), got
    assert ref == f"arith.extf {got} : T to T", (ref, got)


# ---------------------------------------------------------------------------------------
# 3. wrapper guards
# ---------------------------------------------------------------------------------------
def test_invoke_refuses_wire_store_at_multi_leaf():
    """A multi-leaf result is an fp32 internal tree node; rounding it on the wire moves bits.
    The wrapper must refuse BEFORE any launch."""
    import torch

    A = torch.zeros(4, 8, dtype=torch.bfloat16)
    B = torch.zeros(2, 8, 8, dtype=torch.bfloat16)
    C = torch.zeros(4, 1, 8, dtype=torch.bfloat16)
    sti = torch.zeros(4, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="wire store"):
        LT.invoke_fused_moe_leaftree_kernel(
            A=A,
            B=B,
            C=C,
            sorted_token_ids=sti,
            expert_ids=torch.zeros(1, dtype=torch.int32),
            num_tokens_post_padded=torch.tensor([4], dtype=torch.int32),
            n_leaves=2,
            config={"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1},
            compute_type=tl.bfloat16,
            wire_store=True,
        )
