"""CPU proofs for the WAVE-11 wire-stage SINK's refusal logic -- the plumbing, not the arithmetic.

The arithmetic half of ``SKYRL_ISOEXEC_MOE_OC_WIRE_STAGE`` is proven elsewhere
(``test_leaftree_wire_cpu.py`` at TTIR level, ``owner_combine_anatomy.py --phases bits`` on live
operands). What this file pins is the half that FAILED battery 2 on 2026-08-13: the sink's
production entry point returned ``None`` at world 8, 4 AND 2 -- deterministically, with every bit
check green -- because ``None`` was doing double duty as BOTH "no owner combine has recorded a
group yet" AND the perfectly ordinary group value ``torch.distributed``'s default process group
carries. A combine that runs on the default group therefore recorded "unrecorded", and the sink
refused forever: the lever would have shipped bit-identical and completely inert, which is the
PIK_FUSED_BARRIER failure mode our engagement standard exists to catch.

Two obligations, both cheap and both CPU-only:

  1. RECORDING THE DEFAULT GROUP IS RECORDING. After a combine on group ``None`` the sink must get
     PAST the group gate -- it may still decline for a later, DIFFERENT reason (on a CPU box it
     always will: there is no symmetric memory), but "group_not_recorded" must be gone.
  2. EVERY REFUSAL NAMES ITSELF. ``wire_stage_counts()['refused_by_reason']`` must say WHICH of the
     rank-local paths declined. Battery 2 could prove the mechanism and still not say why the
     plumbing never engaged, because one undifferentiated ``refusals`` counter cannot.
"""

from __future__ import annotations

import pytest

from skyrl.backends.skyrl_train.isoexec.ops.moe import moe_pik_combine_owner as OC


@pytest.fixture
def sink_state(monkeypatch):
    """Both flags ON, counters and the recorded group reset to a fresh process's state."""
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_OC_WIRE_STAGE", "1")
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_PIK_OWNER_COMBINE", "1")
    saved_group = dict(OC._WIRE_GROUP)
    saved_counts = dict(OC._WIRE_COUNTS)
    saved_reasons = dict(OC._WIRE_REFUSALS)
    OC._WIRE_GROUP.update({"group": OC._WIRE_UNRECORDED, "device": None})
    OC._WIRE_COUNTS.update({k: 0 for k in OC._WIRE_COUNTS})
    OC._WIRE_REFUSALS.clear()
    yield
    OC._WIRE_GROUP.update(saved_group)
    OC._WIRE_COUNTS.update(saved_counts)
    OC._WIRE_REFUSALS.clear()
    OC._WIRE_REFUSALS.update(saved_reasons)


def _reasons() -> dict:
    return OC.wire_stage_counts()["refused_by_reason"]


def test_fresh_process_refuses_by_name_not_silently(sink_state):
    """Before any combine, the sink declines -- and says so under a name a log can be grepped for."""
    assert OC.owner_wire_sink(4096, 2048, "cpu") is None
    assert _reasons() == {"group_not_recorded": 1}
    assert OC.wire_stage_counts()["grants"] == 0


def test_recording_the_DEFAULT_group_counts_as_recorded(sink_state):
    """THE BATTERY-2 BUG, pinned. ``None`` is torch's default process group, not a sentinel.

    A combine that ran on it has recorded a group, so the sink must be past the group gate. On a CPU
    box the next gate (symmetric-memory P2P) then declines -- which is the point: the refusal has
    MOVED to a different, correctly named reason. Before the fix this asserted None forever.
    """
    OC._WIRE_GROUP.update({"group": None, "device": "cpu"})  # what _owner_combine records
    assert OC.owner_wire_sink(4096, 2048, "cpu") is None  # no symm-mem on CPU: still a refusal
    assert "group_not_recorded" not in _reasons()
    assert _reasons() == {"no_symm_p2p": 1}


def test_device_mismatch_is_its_own_reason(sink_state):
    OC._WIRE_GROUP.update({"group": None, "device": "cpu"})
    assert OC.owner_wire_sink(4096, 2048, "cuda:3") is None
    assert _reasons() == {"device_mismatch": 1}


def test_flags_off_is_not_a_refusal(sink_state, monkeypatch):
    """The lever being OFF is not the sink declining -- it must not pollute the refusal counters,
    or 'why did it not engage' becomes unreadable in exactly the arm that cares."""
    monkeypatch.setenv("SKYRL_ISOEXEC_MOE_OC_WIRE_STAGE", "0")
    OC._WIRE_GROUP.update({"group": None, "device": "cpu"})
    assert OC.owner_wire_sink(4096, 2048, "cpu") is None
    assert _reasons() == {}
    assert OC.wire_stage_counts()["refusals"] == 0


def test_reason_names_are_all_documented(sink_state):
    """Every name a refusal can carry must have prose attached: the name is what a future owner
    greps for, the prose is what tells them whether it is expected."""
    for name in (
        "group_not_recorded",
        "device_mismatch",
        "no_symm_p2p",
        "bf16_wire_off",
        "world_ne_leaves",
        "pool_growth_under_capture",
        "exception",
    ):
        assert name in OC._WIRE_REASONS and OC._WIRE_REASONS[name]


def test_repeated_refusals_accumulate_under_one_name(sink_state):
    """Counted every time, printed once -- a per-call print would drown a 40-layer decode step."""
    for _ in range(3):
        OC.owner_wire_sink(4096, 2048, "cpu")
    assert _reasons() == {"group_not_recorded": 3}
    assert OC.wire_stage_counts()["refusals"] == 3
