"""CPU proof for the fc2 in-GEMM leaf tree's TILE GEOMETRY -- value-DAG identity and the spill ledger.

WHAT IS BEING PROVEN, AND WHY IT NEEDS NO GPU.

``bmm_kernel_indexed_leaftree`` is on the gate-critical forward, so retuning it is only admissible
if the per-element floating-point expression is untouched. Triton compiles ahead of time, so
``triton.compile`` against an explicit ``GPUTarget("cuda", 90, 32)`` produces real sm_90 PTX on a
box with no CUDA device, and ``ptxas -v`` reports the register/spill accounting that motivates the
change. Both halves are therefore decidable here:

  1. THE VALUE DAG. For every candidate geometry the kernel must emit, PER OUTPUT ELEMENT, the same
     number of sequential ``wgmma`` k-steps at the same k-extent (the K-accumulation), the same
     number of ``cvt.rn.bf16.f32`` (the eight per-leaf RNE rounds that ARE the contract), and the
     same number of ``add.f32`` (the seven balanced-tree folds). And it must contain no cross-lane
     float reduction -- ``redux.sync`` at all, or a ``shfl.sync`` on anything but ``.b32`` integer
     grid boilerplate -- because a warp-shuffle tree IS an association order. That is exactly the
     ARITHMETIC-vs-SCHEDULE line the private repo's autotune ledger draws for ``moe.expert_bmm``;
     this file is the evidence that classification cites.

  2. THE SPILL LEDGER -- the reason to touch this kernel at all. The leaf tree holds FOUR live
     ``[BLOCK_M, BLOCK_N]`` tiles (``accumulator`` + the pending partials ``s0``/``s1``/``s2``),
     while the geometry it inherits from ``bmm_launch_config`` was tuned by vLLM for a kernel with
     ONE. At the production shape that costs thousands of bytes of per-thread spill; the numbers
     below are a REGRESSION LEDGER, not a target, so a future edit that quietly makes the live
     geometry worse (or makes a candidate stop being an improvement) fails here rather than in a
     trace six weeks later.

WHAT IS NOT PROVEN HERE. That any geometry is faster (the private repo's nightly
``moe_fc2_leaftree_geometry_bench.py``, which asserts bitwise BEFORE it times), and that the live
operands are bit-identical (the module's own fail-closed first-use self-check does that on the GPU,
at whatever geometry is actually configured).
"""

from __future__ import annotations

import collections
import importlib
import os
import re
import shutil
import subprocess
import tempfile

import pytest

pytest.importorskip("triton")

import triton  # noqa: E402
import triton.language as tl  # noqa: E402,F401  (kernel decorators need it importable)
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

FC2 = importlib.import_module("skyrl.backends.skyrl_train.isoexec.ops.moe.moe_fc2_ingemm")
IBMM = importlib.import_module("skyrl.backends.skyrl_train.isoexec.ops.moe.moe_indexed_bmm")

TARGET = GPUTarget("cuda", 90, 32)

# Production trainer shape (Qwen3.5-35B-A3B, TP=4 / EP=8 / ETP=1): the rank owns all G=8 leaves of
# the f=768 moe_intermediate, so K_LEAF=96 against the pinned BLOCK_SIZE_K=64 -- every leaf carries
# a 32-lane masked tail, which is the shape the geometry has to survive.
N_LEAVES = 8
K_LEAF = 96
BLOCK_K = 64
KBLOCKS = N_LEAVES * ((K_LEAF + BLOCK_K - 1) // BLOCK_K)  # 16
K16_PER_BLOCK = BLOCK_K // 16  # Hopper bf16 wgmma is k16-only
WGMMA_PER_ELEMENT = KBLOCKS * K16_PER_BLOCK  # 64 sequential k-steps per output element

_LEAF_SIG = {
    "a_ptr": "*bf16",
    "b_ptr": "*bf16",
    "idx_ptr": "*i64",
    "c_ptr": "*fp32",
    "B": "i32",
    "M": "i32",
    "N": "i32",
    "stride_ab": "i32",
    "stride_am": "i32",
    "stride_ak": "i32",
    "stride_bb": "i32",
    "stride_bk": "i32",
    "stride_bn": "i32",
    "stride_cb": "i32",
    "stride_cm": "i32",
    "stride_cn": "i32",
}


def _compile(block_m: int, block_n: int, num_warps: int, num_stages: int = 3):
    consts = {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": BLOCK_K,
        "A_LARGE": False,
        "B_LARGE": False,
        "C_LARGE": False,
        "N_LEAVES": N_LEAVES,
        "K_LEAF": K_LEAF,
        "HAS_IDX": True,
    }
    sig = dict(_LEAF_SIG)
    sig.update({k: "constexpr" for k in consts})
    src = ASTSource(fn=FC2.bmm_kernel_indexed_leaftree, signature=sig, constexprs=consts)
    return triton.compile(src, target=TARGET, options={"num_warps": num_warps, "num_stages": num_stages})


# Geometries under test. The first is what production runs today (inherited from
# ``bmm_launch_config``); the rest are the candidates the bench will time.
LIVE = (128, 128, 8)
CANDIDATES = ((128, 64, 8), (128, 32, 8))
ALL_GEOM = (LIVE,) + CANDIDATES


def _ptx(block_m, block_n, num_warps):
    return _compile(block_m, block_n, num_warps).asm["ptx"]


def _counts(ptx: str) -> dict:
    wg = collections.Counter(re.findall(r"wgmma\.mma_async\.sync\.aligned\.m(\d+)n(\d+)k(\d+)", ptx))
    return {
        "wgmma": wg,
        "wgmma_total": sum(wg.values()),
        "cvt_bf16": len(re.findall(r"cvt\.rn\.bf16\.f32", ptx)),
        "add_f32": len(re.findall(r"\badd\.f32\b", ptx)),
        "redux": len(re.findall(r"redux\.sync", ptx)),
        "shfl": re.findall(r"shfl\.sync\.\w+\.(\w+)", ptx),
    }


# ---------------------------------------------------------------------------------------
# 1. the value DAG is the same at every candidate geometry
# ---------------------------------------------------------------------------------------
def test_k_accumulation_is_identical_at_every_geometry():
    """Same count of sequential wgmma k-steps, same k-extent -- only the N extent may move.

    The K-accumulation is the ONE thing BLOCK_M/BLOCK_N are not allowed to touch. A geometry that
    made Triton split K across warps, or that changed the wgmma k-extent, would change the order in
    which an output element's k-terms reach its fp32 accumulator and would be a manifest event, not
    a retune.
    """
    for bm, bn, w in ALL_GEOM:
        c = _counts(_ptx(bm, bn, w))
        kexts = {k for (_m, _n, k) in c["wgmma"]}
        assert kexts == {"16"}, (bm, bn, w, kexts, "wgmma k-extent moved")
        assert c["wgmma_total"] == WGMMA_PER_ELEMENT, (bm, bn, w, c["wgmma_total"], WGMMA_PER_ELEMENT)


def test_rounding_and_fold_counts_scale_exactly_with_tile_elements():
    """8 leaf RNE rounds and 7 fp32 tree adds PER OUTPUT ELEMENT, at every geometry.

    ``cvt.rn.bf16.f32`` is where the contract lives: the buffer path's leaf is a bf16 TENSOR, so the
    in-register tree must round each leaf exactly once. ``add.f32`` is the balanced fold. Both are
    per-element quantities, so both must come out as (elements per thread) x (8, 7) -- if a geometry
    ever dropped a round or grew a fold, this is where it shows.
    """
    for bm, bn, w in ALL_GEOM:
        c = _counts(_ptx(bm, bn, w))
        elems_per_thread = bm * bn // (w * 32)
        assert c["cvt_bf16"] == N_LEAVES * elems_per_thread, (bm, bn, w, c["cvt_bf16"])
        assert c["add_f32"] == (N_LEAVES - 1) * elems_per_thread, (bm, bn, w, c["add_f32"])


def test_no_cross_lane_float_reduction_at_any_geometry():
    """No ``redux.sync``, and every ``shfl.sync`` is a ``.b32`` integer broadcast.

    A warp-shuffle reduction tree IS an association order (``autotune_ledger``'s own words), so its
    presence would demote num_warps and the tile blocks from SCHEDULE to ARITHMETIC and this whole
    change with them.
    """
    for bm, bn, w in ALL_GEOM:
        c = _counts(_ptx(bm, bn, w))
        assert c["redux"] == 0, (bm, bn, w, "redux.sync present: a cross-lane fold appeared")
        assert set(c["shfl"]) <= {"b32"}, (bm, bn, w, c["shfl"], "non-b32 shuffle: possible float fold")


# ---------------------------------------------------------------------------------------
# 2. the spill ledger -- the motivation, pinned so it cannot silently rot
# ---------------------------------------------------------------------------------------
def _ptxas() -> str | None:
    for cand in (
        os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "ptxas"),
        shutil.which("ptxas") or "",
    ):
        if cand and os.path.exists(cand):
            return cand
    return None


def _resource(ptx: str) -> tuple[int, int, int]:
    """(registers, spill-store bytes, spill-load bytes) from ``ptxas -v``."""
    exe = _ptxas()
    assert exe is not None
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "k.ptx")
        with open(p, "w") as fh:
            fh.write(ptx)
        # The PTX declares .target sm_90a (wgmma is an sm_90a feature), so ptxas must be told so.
        r = subprocess.run(
            [exe, "-v", "-arch=sm_90a", "-o", os.path.join(d, "k.cubin"), p],
            capture_output=True,
            text=True,
            check=True,
        )
    out = r.stderr + r.stdout
    regs = int(re.search(r"Used (\d+) registers", out).group(1))
    ss = re.search(r"(\d+) bytes spill stores", out)
    sl = re.search(r"(\d+) bytes spill loads", out)
    return regs, int(ss.group(1)) if ss else 0, int(sl.group(1)) if sl else 0


@pytest.mark.skipif(_ptxas() is None, reason="ptxas not available")
def test_live_geometry_spills_and_the_candidates_do_not():
    """The measured ledger this change exists for -- a regression guard, not a target.

    Recorded 2026-08-14 on triton 3.7.1 / sm_90a. Bounds are loose (the point is the ORDER of
    magnitude, which is what makes the live geometry pathological) so a toolchain bump does not
    fail the suite spuriously, but a REVERSAL fails immediately.
    """
    live_regs, live_ss, live_sl = _resource(_ptx(*LIVE))
    assert live_ss + live_sl > 4000, (live_regs, live_ss, live_sl, "live geometry stopped spilling -- re-price")
    for bm, bn, w in CANDIDATES:
        _, ss, sl = _resource(_ptx(bm, bn, w))
        assert ss + sl < (live_ss + live_sl) / 4, (bm, bn, w, ss, sl, live_ss + live_sl)


@pytest.mark.skipif(_ptxas() is None, reason="ptxas not available")
def test_fc1_at_the_same_geometry_does_not_spill():
    """The fc1/fc2 asymmetry, made explicit: the SAME [128,128] tile is fine with ONE accumulator.

    This is the whole structural diagnosis in one assertion. ``bmm_kernel_indexed`` and
    ``bmm_kernel_indexed_leaftree`` are the same op family on the same hardware at the same tile;
    the only difference is that the leaf tree must hold log2(8)+1 = 4 live fp32 tiles because its
    reduction axis is the one the ENGINE shards over ETP. fc1's reduction axis is the hidden size,
    which no parallelism partitions, so it needs no tree and no extra liveness.
    """
    sig = {
        "a_ptr": "*bf16",
        "b_ptr": "*bf16",
        "idx_ptr": "*i64",
        "c_ptr": "*bf16",
        "B": "i32",
        "M": "i32",
        "N": "i32",
        "K": "i32",
        "stride_ab": "i32",
        "stride_am": "i32",
        "stride_ak": "i32",
        "stride_bb": "i32",
        "stride_bk": "i32",
        "stride_bn": "i32",
        "stride_cb": "i32",
        "stride_cm": "i32",
        "stride_cn": "i32",
    }
    consts = {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": BLOCK_K,
        "A_LARGE": False,
        "B_LARGE": False,
        "C_LARGE": False,
    }
    sig.update({k: "constexpr" for k in consts})
    src = ASTSource(fn=IBMM.bmm_kernel_indexed, signature=sig, constexprs=consts)
    fc1 = triton.compile(src, target=TARGET, options={"num_warps": 8, "num_stages": 3})
    _, ss, sl = _resource(fc1.asm["ptx"])
    assert ss + sl < 1024, (ss, sl, "fc1 started spilling -- the asymmetry argument needs re-deriving")


# ---------------------------------------------------------------------------------------
# 3. the plumbing: default is byte-for-byte today's config, and BLOCK_K cannot be moved
# ---------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clean_geometry_env(monkeypatch):
    for env in FC2._GEOM_ENV.values():
        monkeypatch.delenv(env, raising=False)


def _needs_inherited_config():
    """The INHERITED half of the config is read off vLLM's ``batch_invariant`` at call time.

    Everything about the DERIVATION is pure and runs anywhere; only the baseline it narrows from
    needs vLLM present. Skipping is right here -- reporting these as failures on a box without the
    engine installed buries real regressions in noise -- and the checks still run wherever the
    engine is available, which is every environment that can actually launch this kernel.
    """
    pytest.importorskip("vllm", reason="the inherited config is read off vLLM's batch_invariant")


def test_unset_environment_is_exactly_the_inherited_config(monkeypatch):
    import torch

    _needs_inherited_config()

    for dtype in (torch.bfloat16, torch.float32):
        assert FC2.leaftree_launch_config(dtype) == IBMM.bmm_launch_config(dtype)
        assert "(inherited)" in FC2.leaftree_geometry(dtype)


def test_overrides_apply_and_are_named_in_the_banner_string(monkeypatch):
    import torch

    _needs_inherited_config()

    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_FC2_INGEMM_BLOCK_N", "64")
    cfg = FC2.leaftree_launch_config(torch.bfloat16)
    assert cfg["BLOCK_SIZE_N"] == 64
    assert cfg["BLOCK_SIZE_K"] == IBMM.bmm_launch_config(torch.bfloat16)["BLOCK_SIZE_K"]
    geom = FC2.leaftree_geometry(torch.bfloat16)
    assert "[128,64,64]" in geom and "OVERRIDDEN" in geom


def test_block_k_is_not_exposed_as_a_knob():
    """BLOCK_SIZE_K is the ARITHMETIC knob; there must be no environment path to it.

    It sets the K-tiling and hence the accumulation order, and for the ragged production leaf
    (K_LEAF=96 vs BLOCK_K=64) it additionally decides how many zero lanes enter the tail dot --
    which is observable on a padded all-zero row, where an accumulator can legitimately hold -0.0
    and ``acc + 0.0 != acc``.
    """
    assert "BLOCK_SIZE_K" not in FC2._GEOM_ENV
    assert not any("BLOCK_K" in env for env in FC2._GEOM_ENV.values())


@pytest.mark.parametrize("bad", ["0", "-4", "96", "notanint"])
def test_non_power_of_two_or_garbage_overrides_refuse(monkeypatch, bad):
    import torch

    _needs_inherited_config()

    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_FC2_INGEMM_BLOCK_N", bad)
    with pytest.raises(RuntimeError):
        FC2.leaftree_launch_config(torch.bfloat16)


# ---------------------------------------------------------------------------------------
# 4. the DERIVATION: the tile as a rule with no model dimension in it
# ---------------------------------------------------------------------------------------
#: The measured sm_90a frontier: (n_leaves, BLOCK_N) -> did ptxas spill. Twelve points, one
#: threshold. Pinned here so the rule is checked against data rather than against itself.
MEASURED_SPILL_GRID = {
    (2, 128): True,
    (2, 64): False,
    (2, 32): False,
    (4, 128): True,
    (4, 64): True,
    (4, 32): False,
    (8, 128): True,
    (8, 64): True,
    (8, 32): False,
    (8, 16): False,
}


def test_derivation_reproduces_the_measured_sm90_frontier():
    """The rule must retrodict every measured point, or it is a fit rather than a frontier."""
    assert FC2.live_accumulator_tiles(8) == 4 and FC2.live_accumulator_tiles(2) == 2
    for (n_leaves, block_n), spilled in MEASURED_SPILL_GRID.items():
        over = FC2.accumulator_regs(128, block_n, 8, n_leaves) > FC2._ACCUM_REG_BUDGET
        assert over is spilled, (n_leaves, block_n, spilled, FC2.accumulator_regs(128, block_n, 8, n_leaves))
    # ...and it must land on the widest spill-free tile at each leaf count, not merely a safe one.
    for n_leaves in (2, 4, 8):
        chosen = FC2.derived_block_n(128, 128, 8, n_leaves)
        assert MEASURED_SPILL_GRID.get((n_leaves, chosen)) is False, (n_leaves, chosen)
        assert MEASURED_SPILL_GRID.get((n_leaves, chosen * 2), True) is True, (n_leaves, chosen)


def test_derivation_contains_no_model_dimension_and_tracks_leaf_count():
    """Generality, stated as a property: the only inputs are kernel/tree facts.

    Hidden size, FFN width, expert count and tile capacity cannot reach ``derived_block_n`` -- its
    whole signature is (BLOCK_M, BLOCK_N, num_warps, n_leaves). So the same rule serves Qwen3.5 and
    GLM-4.7 MLA, and a shallower tree is correctly allowed a wider tile rather than inheriting
    Qwen's number.
    """
    import inspect

    assert list(inspect.signature(FC2.derived_block_n).parameters) == [
        "block_m",
        "block_n",
        "num_warps",
        "n_leaves",
    ]
    widths = {leaves: FC2.derived_block_n(128, 128, 8, leaves) for leaves in (2, 4, 8)}
    assert widths[2] >= widths[4] >= widths[8], widths
    assert widths[2] > widths[8], widths
    assert all(w >= FC2._MIN_BLOCK_N and (w & (w - 1)) == 0 for w in widths.values()), widths


def test_auto_selects_the_derived_tile_and_says_so(monkeypatch):
    import torch

    _needs_inherited_config()
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_FC2_INGEMM_BLOCK_N", "auto")
    inherited = IBMM.bmm_launch_config(torch.bfloat16)
    cfg = FC2.leaftree_launch_config(torch.bfloat16, 8)
    assert cfg["BLOCK_SIZE_N"] == 32
    assert cfg["BLOCK_SIZE_K"] == inherited["BLOCK_SIZE_K"]  # the arithmetic axis never moves
    assert "(derived)" in FC2.leaftree_geometry(torch.bfloat16, 8)
    # No call in hand (the banner) derives against the WORST supported liveness, so it can only
    # ever pick a narrower tile than the truth -- never a spilling one.
    assert FC2.leaftree_launch_config(torch.bfloat16)["BLOCK_SIZE_N"] == 32


def _compile_leaves(block_m, block_n, num_warps, n_leaves, k_leaf, num_stages=3):
    consts = {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": BLOCK_K,
        "A_LARGE": False,
        "B_LARGE": False,
        "C_LARGE": False,
        "N_LEAVES": n_leaves,
        "K_LEAF": k_leaf,
        "HAS_IDX": True,
    }
    sig = dict(_LEAF_SIG)
    sig.update({k: "constexpr" for k in consts})
    src = ASTSource(fn=FC2.bmm_kernel_indexed_leaftree, signature=sig, constexprs=consts)
    return triton.compile(src, target=TARGET, options={"num_warps": num_warps, "num_stages": num_stages})


@pytest.mark.skipif(_ptxas() is None, reason="ptxas not available")
@pytest.mark.parametrize("n_leaves", [2, 4, 8])
def test_derived_geometry_does_not_spill_at_any_supported_leaf_count(n_leaves):
    """THE GENERALITY MECHANISM, and the reason this needs no per-model census.

    The one constant the rule cannot derive (``_NON_ACCUM_REGS``) is a property of this kernel on
    this architecture and this Triton -- so it is VERIFIED here rather than trusted, offline,
    against ptxas, across the whole supported leaf grid. A model whose pik tree is 2 or 4 leaves
    wide gets its own answer from the same rule and that answer is checked here too. If a toolchain
    bump moves the kernel's non-accumulator pressure, this fails and the constant is re-measured;
    nothing has to be re-tuned per model.
    """
    block_m, num_warps = 128, 8
    block_n = FC2.derived_block_n(block_m, 128, num_warps, n_leaves)
    ptx = _compile_leaves(block_m, block_n, num_warps, n_leaves, K_LEAF).asm["ptx"]
    regs, ss, sl = _resource(ptx)
    assert ss + sl == 0, (n_leaves, block_n, regs, ss, sl, "derived geometry spills -- re-measure _NON_ACCUM_REGS")
    assert regs <= FC2._PTXAS_REG_BUDGET, (n_leaves, block_n, regs)
