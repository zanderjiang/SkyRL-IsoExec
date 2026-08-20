"""Exact identities for the process-wide NCCL channel composition.

An uncapped communicator and one capped at eight channels are different implementations, so the normalization
lives here and model admission, trainer census, and engine census cannot give them the same name. This module
is deliberately CUDA- and torch-free: it describes process environment state, it does not create communicators.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

NCCL_ALGO = "NCCL_ALGO"
NCCL_MIN_NCHANNELS = "NCCL_MIN_NCHANNELS"
NCCL_MAX_NCHANNELS = "NCCL_MAX_NCHANNELS"
NCCL_EFFECTIVE_KEYS = (NCCL_ALGO, NCCL_MIN_NCHANNELS, NCCL_MAX_NCHANNELS)

PINNED = "pinned"
CAP8 = "cap8"
ENGINE_CAP8 = "engine_cap8"
ENGINE_CAP16 = "engine_cap16"
UNPINNED = "unpinned"

PINNED_CONSTANTS = {
    NCCL_ALGO: "allreduce:tree",
    NCCL_MIN_NCHANNELS: "1",
    NCCL_MAX_NCHANNELS: "1",
}
CAP8_CONSTANTS = {
    NCCL_ALGO: None,
    NCCL_MIN_NCHANNELS: None,
    NCCL_MAX_NCHANNELS: "8",
}
ENGINE_CAP8_CONSTANTS = {
    NCCL_ALGO: "allreduce:tree",
    NCCL_MIN_NCHANNELS: None,
    NCCL_MAX_NCHANNELS: "8",
}
ENGINE_CAP16_CONSTANTS = {
    NCCL_ALGO: "allreduce:tree",
    NCCL_MIN_NCHANNELS: None,
    NCCL_MAX_NCHANNELS: "16",
}
UNPINNED_CONSTANTS = {
    NCCL_ALGO: None,
    NCCL_MIN_NCHANNELS: None,
    NCCL_MAX_NCHANNELS: None,
}

_CONSTANTS_BY_IMPL = {
    PINNED: PINNED_CONSTANTS,
    CAP8: CAP8_CONSTANTS,
    ENGINE_CAP8: ENGINE_CAP8_CONSTANTS,
    ENGINE_CAP16: ENGINE_CAP16_CONSTANTS,
    UNPINNED: UNPINNED_CONSTANTS,
}


def constants_for_impl(impl_id: str) -> dict[str, str | None]:
    """Return a fresh exact ALGO/MIN/MAX declaration for a known identity."""
    try:
        return dict(_CONSTANTS_BY_IMPL[impl_id])
    except KeyError as exc:
        raise ValueError(f"unknown NCCL composition identity {impl_id!r}") from exc


def effective_constants(env: Mapping[str, str] | None = None) -> dict[str, str | None]:
    """Read the three values NCCL itself sees, normalizing absent/blank values to ``None``."""
    src = os.environ if env is None else env
    return {key: (str(src.get(key, "")).strip() or None) for key in NCCL_EFFECTIVE_KEYS}


def identity_for_constants(constants: Mapping[str, str | None]) -> str:
    """Name an exact supported tuple; reject every partially pinned or differently capped tuple."""
    normalized = {key: constants.get(key) for key in NCCL_EFFECTIVE_KEYS}
    for impl_id, expected in _CONSTANTS_BY_IMPL.items():
        if normalized == expected:
            return impl_id
    rendered = ", ".join(f"{key}={normalized[key]!r}" for key in NCCL_EFFECTIVE_KEYS)
    raise ValueError(f"unsupported effective NCCL channel composition: {rendered}")


def effective_identity(
    env: Mapping[str, str] | None = None,
) -> tuple[str, dict[str, str | None]]:
    """Return ``(identity, exact constants)`` for the live process, refusing unknown mixtures."""
    constants = effective_constants(env)
    return identity_for_constants(constants), constants


def requested_trainer_identity(env: Mapping[str, str] | None = None) -> str:
    """Resolve launcher intent before Ray materializes it as effective ``NCCL_*`` variables."""
    src = os.environ if env is None else env
    pin = str(src.get("SKYRL_ISOEXEC_NCCL_PIN", "1")).strip()
    if pin == "1":
        return PINNED
    if pin != "0":
        raise ValueError(f"SKYRL_ISOEXEC_NCCL_PIN must be '0' or '1', got {pin!r}")
    cap = str(src.get("SKYRL_ISOEXEC_NCCL_MAX_NCHANNELS", "")).strip() or None
    return identity_for_constants({NCCL_ALGO: None, NCCL_MIN_NCHANNELS: None, NCCL_MAX_NCHANNELS: cap})


def requested_engine_identity(env: Mapping[str, str] | None = None) -> str:
    """Resolve the neutralized engine channel policy (whose cap defaults to eight).

    vLLM deliberately retains ``NCCL_ALGO=allreduce:tree`` while removing the one-channel floor.
    This differs from the trainer's ``cap8`` tuple and therefore has a distinct identity.
    """
    src = os.environ if env is None else env
    unpin = str(src.get("SKYRL_ISOEXEC_ENGINE_NCCL_UNPIN", "0")).strip()
    if unpin != "1":
        # Runtime census still reads effective values and reports ``pinned`` if vLLM re-pins.
        return UNPINNED
    cap = str(src.get("SKYRL_ISOEXEC_ENGINE_NCCL_MAX_NCHANNELS", "8")).strip() or None
    return identity_for_constants(
        {
            NCCL_ALGO: "allreduce:tree",
            NCCL_MIN_NCHANNELS: None,
            NCCL_MAX_NCHANNELS: cap,
        }
    )


def census(impl_id: str, constants: Mapping[str, str | None]) -> str:
    """Stable, grep-friendly identity plus the exact values represented by it."""
    return (
        f"impl={impl_id} ALGO={constants.get(NCCL_ALGO)!r} "
        f"MIN={constants.get(NCCL_MIN_NCHANNELS)!r} "
        f"MAX={constants.get(NCCL_MAX_NCHANNELS)!r}"
    )


def assert_contract_matches(view, sites, impl_id: str, constants: Mapping[str, str | None]) -> None:
    """Fail closed when an active runtime's exact tuple differs from its contract entries.

    ``view`` is the contract's ``{(op, site) -> {impl_id, pinned_constants, ...}}`` projection
    (``core.process_contract.cached_contract_view``).
    """
    if view is None:
        raise RuntimeError("active NCCL composition cannot be admitted because the process contract is unavailable")
    actual = dict(constants)
    problems = []
    for site in sites:
        entry = view.get(("collectives.nccl_pin", site))
        if entry is None:
            problems.append(f"{site}: contract has no collectives.nccl_pin entry")
        elif entry["impl_id"] != impl_id or entry["pinned_constants"] != actual:
            problems.append(
                f"{site}: contract={entry['impl_id']!r}/{entry['pinned_constants']!r} " f"runtime={impl_id!r}/{actual!r}"
            )
    if problems:
        raise RuntimeError(
            "[ISOEXEC-NCCL-MANIFEST] effective ALGO/MIN/MAX disagrees with the frozen contract; "
            "refusing before forward:\n  " + "\n  ".join(problems)
        )
