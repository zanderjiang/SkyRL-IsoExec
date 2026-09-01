"""CPU gate for the QKV subgroup all-gather's admission predicates.

Byte-identity holds only when megatron's kept window ``[idx*size, (idx+1)*size)`` is exactly whole
rank shards; every geometry where it is not (e.g. ``num_query_groups`` not dividing the tp world,
which would cut a shard) must be refused with a printed reason, never taken as a fast path.
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
    # The trainer at TP=4 passes the pure predicates; it is kept out by the install-side
    # "tp group must be the whole default world" rule, not by these.
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
    """Every admissible geometry keeps a window that is a whole number of rank shards."""
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
    """The flag defaults ON, and `=0` still selects the reference OFF path exactly."""
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
    """The subgroup path records no autograd node, so with grad live the shim must fall back."""

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

        # requires_grad on the operand alone is refused too, even under no_grad.
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
    """OFF is a silent no-op; ON without a mesh refuses with a reason instead of raising."""
    import os

    from skyrl.backends.skyrl_train.isoexec.ops.attention import (
        qkv_subgroup_gather as _qkv,
    )

    old = os.environ.pop("SKYRL_ISOEXEC_ATTN_QKV_SUBGROUP_AG", None)
    saved = set(_qkv._REFUSED)
    try:
        # OFF must be spelled `=0`: the flag defaults ON.
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
