"""CPU gate for the QKV subgroup all-gather's ADMISSION predicates.

The op's byte-identity argument (``ops/attention/qkv_subgroup_gather.py``) rests entirely on one
structural claim: the window megatron keeps, ``[idx*size, (idx+1)*size)`` with
``size = full // num_query_groups``, is exactly the shards of ranks ``[idx*rpg, (idx+1)*rpg)``. The
live 8-GPU gate (``attn_qkv_subgroup_dist.py`` in this directory) proves that on real
operands at the real geometry. This file proves the GUARD -- that every geometry where the claim
does not hold is refused, and refused with a reason -- and it does so on CPU, in CI, without a mesh.

The failure mode this is aimed at: a model whose ``num_query_groups`` does not divide the tp world
would take megatron's gather branch, produce a slice that CUTS A RANK SHARD, and the subgroup
gather would return different bytes. That must be a refusal with a printed reason, never a fast
path -- and it must stay a refusal after someone edits the predicate list.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 7)))  # repo root

import torch  # noqa: F401  -- torch-first, as in production (engine workers load torch before isoexec)

from skyrl.backends.skyrl_train.isoexec.ops.attention.qkv_subgroup_gather import (  # noqa: E402
    admission_refusal,
    qkv_subgroup_ag_enabled,
)


def test_live_engine_geometry_admits():
    # Qwen3.5-35B-A3B, engine TP=8: head_dim 256, 16 q heads, 2 kv heads, output gate ->
    # fused qkv width 256*(2*16 + 2*2) = 9216, shard 1152, kept window 4608 = 4 whole shards.
    assert admission_refusal(world=8, num_query_groups=2, local_width=1152) is None


def test_trainer_geometry_admits_on_the_predicates_alone():
    # The trainer at TP=4 has the same over-gather and passes the PURE predicates; it is kept out
    # by the install-side "tp group must be the whole default world" rule, not by these. If this
    # ever starts refusing, the trainer arm's scoping rationale needs rewriting too.
    assert admission_refusal(world=4, num_query_groups=2, local_width=1152) is None


def test_refusals_are_reasoned():
    cases = [
        dict(world=8, num_query_groups=8),  # megatron takes no gather at all
        dict(world=8, num_query_groups=16),  # ditto, harder
        dict(world=8, num_query_groups=1),  # subgroup would BE the tp group: no saving
        dict(world=8, num_query_groups=3),  # does not divide: the window would cut a shard
        dict(world=8, num_query_groups=6),  # ditto
        dict(world=8, num_query_groups=0),  # nonsense
        dict(world=8, num_query_groups=None),  # megatron asserts on this itself
        dict(world=1, num_query_groups=1),  # nothing to gather
    ]
    for kw in cases:
        r = admission_refusal(**kw)
        assert r is not None, kw
        assert isinstance(r, str) and len(r) > 20, (kw, r)  # a reason, not a bare False


def test_kept_window_is_always_a_whole_number_of_rank_shards():
    """THE invariant. Every admissible geometry must satisfy it; if one does not, the guard is
    letting through a case where the subgroup gather is not the slice."""
    checked = 0
    for world in range(1, 65):
        for ng in range(0, world + 2):
            for local in (1, 3, 64, 1152, 4097):
                if admission_refusal(world=world, num_query_groups=ng, local_width=local):
                    continue
                rpg = world // ng
                full = local * world
                size = full // ng
                assert size == rpg * local, (world, ng, local)
                # ...and the window's bounds land on shard boundaries for EVERY group index.
                for idx in range(ng):
                    assert (idx * size) % local == 0, (world, ng, local, idx)
                    assert ((idx + 1) * size) % local == 0, (world, ng, local, idx)
                checked += 1
    assert checked > 100, checked


def test_flag_defaults_on_and_zero_is_the_escape_hatch():
    """DEFAULT FLIPPED ON 2026-08-13 -- so the thing to pin is that `=0` still works.

    Rule 6 said a flag defaulting OFF means the OFF path is production, and that was the state
    while the win was unmeasured. The one-session duration A/B (B=512, n=249 decode steps per
    arm: 20.019 -> 19.761 ms/step median) is what moved the default; the OFF path remains the
    reference the 33/33 bit gate proved against, so `=0` must keep selecting it exactly.
    """
    import os

    old = os.environ.pop("SKYRL_ISOEXEC_ATTN_QKV_SUBGROUP_AG", None)
    try:
        assert qkv_subgroup_ag_enabled() is True
        os.environ["SKYRL_ISOEXEC_ATTN_QKV_SUBGROUP_AG"] = "0"
        assert qkv_subgroup_ag_enabled() is False
        os.environ["SKYRL_ISOEXEC_ATTN_QKV_SUBGROUP_AG"] = "1"
        assert qkv_subgroup_ag_enabled() is True
    finally:
        os.environ.pop("SKYRL_ISOEXEC_ATTN_QKV_SUBGROUP_AG", None)
        if old is not None:
            os.environ["SKYRL_ISOEXEC_ATTN_QKV_SUBGROUP_AG"] = old


def test_grad_enabled_takes_the_stock_autograd_gather():
    """The subgroup path records NO autograd node (it calls the functional gather, not the
    ``_AllGatherFromTensorParallelRegion.apply`` wrapper whose backward is a reduce-scatter). Under
    the engine's inference_mode that is exactly right; anywhere grad is live it would delete a
    gradient behind a green bitwise gate. The shim must fall back, and say so.

    Driven on CPU by arming the shim's own globals -- no mesh, no CUDA, no megatron model."""
    import torch

    from skyrl.backends.skyrl_train.isoexec.ops.attention import (
        qkv_subgroup_gather as _qkv,
    )

    group = object()
    plan = _qkv.SubgroupPlan(
        world=8,
        num_query_groups=2,
        ranks_per_group=4,
        group_index=0,
        sub_pg=object(),
        ok_groups={id(group): group},
    )
    calls = []
    saved = (_qkv._ORIG_AG, _qkv._ARMED, dict(_qkv._COUNTS), set(_qkv._REFUSED))
    try:
        _qkv._ORIG_AG = lambda x, g: calls.append((x, g)) or x
        x = torch.zeros(2, 1, 8)

        _qkv._ARMED = plan
        with torch.enable_grad():
            _qkv._shim_all_gather_last_dim(x, group=group)
        assert len(calls) == 1, "grad enabled must fall through to megatron's own autograd gather"
        assert any("grad is enabled" in r for r in _qkv._REFUSED), sorted(_qkv._REFUSED)

        # ...and requires_grad on the operand alone is refused too, even under no_grad.
        _qkv._ARMED = plan
        with torch.no_grad():
            _qkv._shim_all_gather_last_dim(x.detach().requires_grad_(True), group=group)
        assert len(calls) == 2, calls
    finally:
        _qkv._ORIG_AG, _qkv._ARMED = saved[0], saved[1]
        _qkv._COUNTS.clear()
        _qkv._COUNTS.update(saved[2])
        _qkv._REFUSED.clear()
        _qkv._REFUSED.update(saved[3])


def test_install_off_mesh_is_silent_when_off_and_reasoned_when_on():
    """Two properties of the installer that only a call can establish, and neither needs a mesh.

    OFF: production runs this path on every engine build, so it must be a no-op that prints
    NOTHING -- an installer that banners on the default path is how banners stop being read.
    ON without a mesh: it must refuse with the reason, not raise and not half-install. The
    restructure that AND-s the verdict across ranks (so a split flag cannot hang `new_group`) runs
    a collective before the flag test, and this is the case that proves it degrades off-mesh."""
    import os

    from skyrl.backends.skyrl_train.isoexec.ops.attention import (
        qkv_subgroup_gather as _qkv,
    )

    old = os.environ.pop("SKYRL_ISOEXEC_ATTN_QKV_SUBGROUP_AG", None)
    saved = set(_qkv._REFUSED)
    try:
        # OFF must now be spelled `=0`: the flag defaults ON since 2026-08-13.
        os.environ["SKYRL_ISOEXEC_ATTN_QKV_SUBGROUP_AG"] = "0"
        _qkv._REFUSED.clear()
        assert _qkv.install_engine_qkv_subgroup_ag(model=None) is False
        assert _qkv._REFUSED == set(), f"the OFF path must be silent, printed: {_qkv._REFUSED}"

        os.environ["SKYRL_ISOEXEC_ATTN_QKV_SUBGROUP_AG"] = "1"
        assert _qkv.install_engine_qkv_subgroup_ag(model=None) is False
        assert _qkv._REFUSED, "the ON path off-mesh must say why it declined"
        assert _qkv.qkv_subgroup_status()["installed"] is False
    finally:
        os.environ.pop("SKYRL_ISOEXEC_ATTN_QKV_SUBGROUP_AG", None)
        if old is not None:
            os.environ["SKYRL_ISOEXEC_ATTN_QKV_SUBGROUP_AG"] = old
        _qkv._REFUSED.clear()
        _qkv._REFUSED.update(saved)


def _run():
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f()
            print("PASS", n)
    print("\nall passed")


if __name__ == "__main__":
    _run()
