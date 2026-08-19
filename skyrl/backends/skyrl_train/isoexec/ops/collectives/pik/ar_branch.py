"""One-shot vs two-shot branch selection for the P2P tree all-reduce.

Both branches evaluate the identical reduction tree over the identical operands in rank order and differ only
in launch structure (one-shot: leading barrier plus one kernel; two-shot: leading barrier, kernel, trailing
barrier), so the choice is a pure performance knob. ``SKYRL_ISOEXEC_PIK_AR_CROSSOVER`` picks ``legacy`` (the
shipped byte threshold; the default), ``arch`` (a measured per-(arch, world) table that falls back loudly when
an entry is missing), or ``calibrate`` (time both branches on this device at warmup). Rank agreement is a
correctness requirement: the branches issue different numbers of symmetric-memory barriers, so ranks that
disagree hang rather than run slowly -- every policy here is a function of world size, dtype, and element count
only, and ``calibrate`` broadcasts rank 0's verdict instead of trusting per-rank timings.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

MODES = ("legacy", "arch", "calibrate")


def mode() -> str:
    m = os.environ.get("SKYRL_ISOEXEC_PIK_AR_CROSSOVER", "legacy").strip().lower()
    if m not in MODES:
        raise ValueError(f"SKYRL_ISOEXEC_PIK_AR_CROSSOVER must be one of {MODES}, got {m!r}")
    return m


# (arch, world) -> largest PAYLOAD in bytes for which one-shot won, with both branches' barriers inside the
# timed region. Payload, not bytes-read: bytes-read folds a `world` factor into a key that is already
# world-keyed. Regenerate with examples/isoexec/nightly/pik_oneshot_crossover.py --emit-arch-table rather than
# hand-editing; an absent (arch, world) falls back to `legacy` and says so once, so no device silently
# inherits another architecture's crossover.
ARCH_CROSSOVER: dict[tuple[str, int], int] = {}

_CALIBRATED: dict = {}
_NOTED: dict = {}
_STATE = {"decisions": 0, "legacy": 0, "arch": 0, "calibrated": 0}
# Branch override used only by calibrate(). None = policy decides; 0 = one-shot; 1 = two-shot.
_FORCE: list = [None]


def _arch() -> str:
    try:
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        return f"sm{p.major}{p.minor}"
    except Exception:  # noqa: BLE001
        return "unknown"


def _bucket(n: int) -> int:
    """Power-of-two bucket of the element count, so a serving engine does not calibrate once per batch size.

    Both branches produce the same bits, so a mis-bucketed choice costs time and never correctness, and the
    bucket is identical on every rank because ``n`` is.
    """
    return 1 << max(0, (n - 1).bit_length())


def _note(reason: str, msg: str) -> None:
    if reason in _NOTED:
        return
    _NOTED[reason] = True
    print(msg, flush=True)
    logger.warning(msg)


def status() -> dict:
    """What this process decided, and how."""
    return {
        "mode": mode(),
        "arch": _arch(),
        "counts": dict(_STATE),
        "calibrated": {str(k): v for k, v in _CALIBRATED.items()},
        "arch_table_entries": sorted(f"{a}/w{w}" for a, w in ARCH_CROSSOVER),
    }


def reset() -> None:
    """Test hook: forget every calibrated verdict and every once-only warning."""
    _CALIBRATED.clear()
    _NOTED.clear()
    for k in _STATE:
        _STATE[k] = 0


def two_shot(n: int, elt: int, world: int, legacy_budget: int) -> bool:
    """True -> two-shot. A pure performance choice; both branches return the same bits.

    ``legacy_budget`` is ``allreduce.P2P_ONESHOT_MAX_BYTES``, passed in rather than imported so
    this module never has to reach back into the one it is called from (and so a test can drive
    the policy without touching module globals)."""
    if _FORCE[0] is not None:
        # Calibration only: pin the branch so the timed arm is the real code path, not a twin.
        return bool(_FORCE[0])
    _STATE["decisions"] += 1
    m = mode()

    if m == "calibrate":
        hit = _CALIBRATED.get((world, elt, _bucket(n)))
        if hit is not None:
            _STATE["calibrated"] += 1
            return not hit["one_shot"]
        _note(
            "uncalibrated",
            f"[ISOEXEC-PIK] all-reduce crossover mode=calibrate but shape (world={world} "
            f"elt={elt} numel~{_bucket(n)}) was never calibrated -- falling back to the legacy "
            f"byte threshold for it. Call pik.ar_branch.calibrate() at warmup, BEFORE graph "
            f"capture. Bits are unaffected either way.",
        )

    elif m == "arch":
        cross = ARCH_CROSSOVER.get((_arch(), world))
        if cross is not None:
            _STATE["arch"] += 1
            return n * elt > cross
        _note(
            f"noarch:{_arch()}:{world}",
            f"[ISOEXEC-PIK] all-reduce crossover mode=arch but there is NO MEASURED ENTRY for "
            f"({_arch()}, world={world}) -- falling back to the legacy byte threshold. This is "
            f"deliberate: an unmeasured device must not inherit another architecture's crossover. "
            f"Regenerate with examples/isoexec/nightly/pik_oneshot_crossover.py --emit-arch-table.",
        )

    _STATE["legacy"] += 1
    return n * elt * (world - 1) > legacy_budget


def calibrate(shapes, group=None, iters: int = 20, warmup: int = 5, verbose: bool = True) -> dict:
    """Time both branches on this device and keep the winner per shape bucket.

    ``shapes`` is an iterable of ``(numel, torch.dtype)``. Call this at warmup: it synchronises (illegal under
    CUDA-graph capture) and issues collectives, so every rank must call it with the same list in the same
    order. The timed region includes each branch's own barriers, which is what the byte threshold ignores.
    """
    from . import allreduce as ar

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "pik.ar_branch.calibrate() cannot run inside CUDA-graph capture (it synchronises). "
            "Call it during warmup, before capture."
        )
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    dev = torch.cuda.current_device()
    out = {}

    for n, dtype in shapes:
        elt = torch.empty(0, dtype=dtype).element_size()
        key = (world, elt, _bucket(n))
        if key in _CALIBRATED:
            continue
        if n % world != 0:
            # two-shot needs an even split; one-shot is the only legal branch, nothing to time.
            _CALIBRATED[key] = {"one_shot": True, "reason": "numel not divisible by world"}
            continue

        # Drive the shipped unfused path with the branch pinned, so each arm is exactly the production code.
        def _arm(force, n=n, dtype=dtype):
            def go():
                _FORCE[0] = force
                try:
                    staged = ar.sym_partial((n,), dev, group, dtype=dtype)
                    return ar._p2p_unfused(staged, group, None, None)
                finally:
                    _FORCE[0] = None

            return go

        try:
            one_us = _time(_arm(0), iters, warmup)
            two_us = _time(_arm(1), iters, warmup)
        finally:
            _FORCE[0] = None

        # Rank agreement: a per-rank verdict hangs, since the branches issue different barrier counts.
        d = torch.tensor([1 if one_us < two_us else 0], device=dev, dtype=torch.int32)
        dist.broadcast(d, src=_src(group), group=group)
        one_shot = bool(d.item())

        _CALIBRATED[key] = {
            "one_shot": one_shot,
            "one_shot_us": one_us,
            "two_shot_us": two_us,
            "payload_bytes": n * elt,
        }
        out[key] = _CALIBRATED[key]
        if verbose and rank == 0:
            print(
                f"[ISOEXEC-PIK] all-reduce branch CALIBRATED world={world} numel~{_bucket(n)} "
                f"{dtype} ({n*elt/2**20:.3f} MiB): one-shot {one_us:.2f} us vs two-shot "
                f"{two_us:.2f} us (barriers INCLUDED) -> "
                f"{'ONE-SHOT' if one_shot else 'TWO-SHOT'}",
                flush=True,
            )
    return out


def _src(group) -> int:
    if group is None:
        return 0
    try:
        return dist.get_global_rank(group, 0)
    except Exception:  # noqa: BLE001
        return 0


def _time(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / iters


def self_check(world: int, legacy_budget: int) -> None:
    """Assert the predicate depends only on rank-invariant facts.

    This is the anti-hang check, and the reason ``two_shot`` takes ``(n, elt, world)`` and nothing else.
    """
    import inspect

    params = list(inspect.signature(two_shot).parameters)
    assert params == ["n", "elt", "world", "legacy_budget"], (
        f"two_shot's signature is the rank-agreement contract; it may only take facts every rank "
        f"shares. Got {params}."
    )
    for n in (1024, 262144, 1048576):
        for elt in (2, 4):
            a = two_shot(n, elt, world, legacy_budget)
            b = two_shot(n, elt, world, legacy_budget)
            assert a is b, "two_shot must be deterministic for a fixed (n, elt, world)"
