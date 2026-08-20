"""CPU gates for the all-reduce branch POLICY (pik/ar_branch.py).

The branch choice cannot move a bit -- both branches evaluate the identical tree -- so nothing
here is about numerics. It is about the two ways this selector can go wrong:

  A. it silently changes what production runs. ``legacy`` is the default and must reproduce the
     shipped byte threshold EXACTLY, at every payload, world and dtype.
  B. it makes two ranks disagree. The branches issue DIFFERENT barrier counts (1 vs 2), and a
     symmetric-memory barrier is a rendezvous, so disagreement is a HANG, not a slowdown. The
     predicate must therefore be a pure function of facts every rank shares.

Also gated: an unmeasured architecture must FALL BACK, never extrapolate.
"""

import importlib

import pytest

ar_branch = importlib.import_module(
    "skyrl.backends.skyrl_train.isoexec.ops.collectives.pik.ar_branch"
)

BUDGET = 2 << 20  # the shipped SKYRL_ISOEXEC_PIK_ONESHOT_MB default, in bytes-read


def legacy(n, elt, world, budget=BUDGET):
    return n * elt * (world - 1) > budget


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    ar_branch.reset()
    monkeypatch.delenv("SKYRL_ISOEXEC_PIK_AR_CROSSOVER", raising=False)
    yield
    ar_branch.reset()


# ---------------------------------------------------------------------------- A. default inertia
@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("elt", [2, 4])
@pytest.mark.parametrize("n", [1024, 65536, 262144, 524288, 1048576, 4194304])
def test_default_is_bit_for_bit_the_shipped_threshold(n, elt, world):
    """The default mode must be INERT: same branch as the constant, at every shape."""
    assert ar_branch.two_shot(n, elt, world, BUDGET) == legacy(n, elt, world)


def test_default_mode_is_legacy():
    assert ar_branch.mode() == "legacy"


def test_unknown_mode_is_a_loud_error(monkeypatch):
    monkeypatch.setenv("SKYRL_ISOEXEC_PIK_AR_CROSSOVER", "sometimes")
    with pytest.raises(ValueError, match="must be one of"):
        ar_branch.mode()


# ------------------------------------------------------------------- B. the rank-agreement gate
def test_predicate_depends_only_on_rank_invariant_facts():
    """THE anti-hang property, asserted on the signature itself.

    If someone ever adds a rank-, device- or time-dependent argument to this predicate, two ranks
    can pick different branches and park on barriers no peer will reach. The signature IS the
    contract, so the test reads the signature."""
    ar_branch.self_check(world=4, legacy_budget=BUDGET)


def test_predicate_is_deterministic_across_repeated_calls():
    seen = {ar_branch.two_shot(1 << 18, 4, 4, BUDGET) for _ in range(50)}
    assert len(seen) == 1


# ------------------------------------------------------------- C. arch table: fall back, never guess
def test_arch_mode_falls_back_when_the_device_is_not_in_the_table(monkeypatch):
    """An unmeasured GPU must inherit NOTHING. It gets the legacy rule and a warning."""
    monkeypatch.setenv("SKYRL_ISOEXEC_PIK_AR_CROSSOVER", "arch")
    monkeypatch.setattr(ar_branch, "_arch", lambda: "sm999")
    for n in (1 << 14, 1 << 20, 1 << 22):
        assert ar_branch.two_shot(n, 4, 4, BUDGET) == legacy(n, 4, 4)
    assert ar_branch.status()["counts"]["legacy"] == 3
    assert ar_branch.status()["counts"]["arch"] == 0


def test_arch_mode_uses_a_measured_entry_and_keys_on_payload_not_bytes_read(monkeypatch):
    """The arch table is keyed by (arch, world) and compares PAYLOAD.

    The shipped constant folded a `world` factor into a budget that was already world-dependent,
    which is how one number came to mean two things. Here world is part of the key, so the value
    is a plain payload crossover."""
    monkeypatch.setenv("SKYRL_ISOEXEC_PIK_AR_CROSSOVER", "arch")
    monkeypatch.setattr(ar_branch, "_arch", lambda: "sm90")
    monkeypatch.setitem(ar_branch.ARCH_CROSSOVER, ("sm90", 4), 1 << 20)  # one-shot up to 1 MiB
    assert ar_branch.two_shot((1 << 20) // 4, 4, 4, BUDGET) is False  # exactly 1 MiB -> one-shot
    assert ar_branch.two_shot((1 << 21) // 4, 4, 4, BUDGET) is True  # 2 MiB -> two-shot
    assert ar_branch.status()["counts"]["arch"] == 2


def test_arch_entry_for_one_world_does_not_leak_to_another(monkeypatch):
    monkeypatch.setenv("SKYRL_ISOEXEC_PIK_AR_CROSSOVER", "arch")
    monkeypatch.setattr(ar_branch, "_arch", lambda: "sm90")
    monkeypatch.setitem(ar_branch.ARCH_CROSSOVER, ("sm90", 4), 1 << 30)
    n = 1 << 20
    assert ar_branch.two_shot(n, 4, 4, BUDGET) is False  # world=4 measured: one-shot
    assert ar_branch.two_shot(n, 4, 8, BUDGET) == legacy(n, 4, 8)  # world=8 unmeasured -> legacy


def test_shipped_arch_table_is_empty_until_it_is_measured():
    """A table of guesses is worse than no table. It ships empty and the harness fills it."""
    assert ar_branch.ARCH_CROSSOVER == {}, (
        "ARCH_CROSSOVER must only ever contain rows emitted by "
        "the private repo's nightly pik_oneshot_crossover.py --emit-arch-table"
    )


# ------------------------------------------------------------------------- D. calibrate mode
def test_calibrate_mode_falls_back_and_warns_for_an_uncalibrated_shape(monkeypatch):
    monkeypatch.setenv("SKYRL_ISOEXEC_PIK_AR_CROSSOVER", "calibrate")
    n = 1 << 20
    assert ar_branch.two_shot(n, 4, 4, BUDGET) == legacy(n, 4, 4)
    assert ar_branch.status()["counts"]["legacy"] == 1


def test_calibrate_mode_honours_a_calibrated_verdict(monkeypatch):
    monkeypatch.setenv("SKYRL_ISOEXEC_PIK_AR_CROSSOVER", "calibrate")
    n = 1 << 20  # legacy at world=4 would say two-shot (12 MiB read)
    assert legacy(n, 4, 4) is True
    monkeypatch.setitem(ar_branch._CALIBRATED, (4, 4, ar_branch._bucket(n)), {"one_shot": True})
    assert ar_branch.two_shot(n, 4, 4, BUDGET) is False
    assert ar_branch.status()["counts"]["calibrated"] == 1


def test_bucketing_is_power_of_two_and_shared_by_neighbouring_batch_sizes():
    b = ar_branch._bucket
    assert b(96 * 2048) == b(128 * 2048) == 1 << 18
    assert b(1) == 1 and b(2) == 2 and b(3) == 4
    assert b(1 << 20) == 1 << 20  # exact powers are their own bucket, not the next one up


def test_force_override_is_off_by_default_and_restores(monkeypatch):
    """calibrate() pins the branch to time the real path; the pin must never survive the call."""
    assert ar_branch._FORCE[0] is None
    ar_branch._FORCE[0] = 0
    try:
        assert ar_branch.two_shot(1 << 30, 4, 8, BUDGET) is False  # forced one-shot
    finally:
        ar_branch._FORCE[0] = None
    assert ar_branch.two_shot(1 << 30, 4, 8, BUDGET) is True  # policy again


def test_status_reports_the_mode_and_the_tally(monkeypatch):
    monkeypatch.setenv("SKYRL_ISOEXEC_PIK_AR_CROSSOVER", "legacy")
    ar_branch.two_shot(1 << 18, 4, 4, BUDGET)
    s = ar_branch.status()
    assert s["mode"] == "legacy"
    assert s["counts"]["decisions"] == 1 and s["counts"]["legacy"] == 1
