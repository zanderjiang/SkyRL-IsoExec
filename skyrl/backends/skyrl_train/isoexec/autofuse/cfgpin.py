"""Build the compiled-region inductor-config pin once, instead of rebuilding it on every call.

The pin itself is still entered per call: a dynamo recompile must regenerate under the same
numerics rather than the ambient config, so the patch cannot simply be deleted. The fast path uses
torch's ``ConfigModule._make_closure_patcher`` and is installed only after a build-time proof that
it sets and reverts the whole live config identically to ``icfg.patch``; anything else falls back.
Gated by ``SKYRL_ISOEXEC_AUTOFUSE_FAST_CFG`` (default off).
"""

from __future__ import annotations

import os
from typing import Any

_COUNTS = {
    "served": 0,  # calls entered through the fast patcher
    "built_fast": 0,  # pins whose fast patcher was proven equivalent and installed
    "built_slow": 0,  # pins that fell back to icfg.patch
    "refused": 0,  # fast patchers refused by the equivalence proof
}
_REFUSALS: list[str] = []


def fast_cfg_enabled() -> bool:
    return os.environ.get("SKYRL_ISOEXEC_AUTOFUSE_FAST_CFG", "0") == "1"


def cfgpin_counts() -> dict:
    out = dict(_COUNTS)
    out["enabled"] = fast_cfg_enabled()
    out["refusals"] = list(_REFUSALS)
    return out


def _reset_for_tests() -> None:
    for k in _COUNTS:
        _COUNTS[k] = 0
    _REFUSALS.clear()


def _slow_pin(cfg: dict[str, Any]):
    """A fresh ``icfg.patch`` context per call."""

    def enter():
        import torch._inductor.config as icfg

        ctx = icfg.patch(cfg)
        ctx.__enter__()
        return lambda: ctx.__exit__(None, None, None)

    return enter


def _prove_equivalent(cfg: dict[str, Any], fast_enter) -> str | None:
    """Enter both patchers on the live config and compare the whole resulting config.

    Returns None on proof, or the reason to refuse. The compare covers the whole config, not just the
    keys in ``cfg``: an alias or derived key set by one path only would change codegen invisibly.
    """
    import torch._inductor.config as icfg

    before = icfg.get_config_copy()

    revert = fast_enter()
    fast_state = icfg.get_config_copy()
    revert()
    after_fast = icfg.get_config_copy()

    ctx = icfg.patch(cfg)
    ctx.__enter__()
    slow_state = icfg.get_config_copy()
    ctx.__exit__(None, None, None)
    after_slow = icfg.get_config_copy()

    if fast_state != slow_state:
        diff = sorted(
            k for k in set(fast_state) | set(slow_state) if fast_state.get(k, object()) != slow_state.get(k, object())
        )
        return f"config differs from icfg.patch on {diff[:8]}"
    if after_fast != before or after_slow != before:
        return "one of the patchers did not restore the prior config exactly"
    return None


def make_config_pin(cfg: dict[str, Any]):
    """Return ``enter() -> revert()`` for ``cfg``, built once; falls back to ``icfg.patch``.

    Call this off the hot path; the returned ``enter`` is what a compiled region calls per invocation.
    """
    cfg = dict(cfg)
    if not fast_cfg_enabled():
        _COUNTS["built_slow"] += 1
        return _slow_pin(cfg)

    try:
        import torch._inductor.config as icfg

        maker = getattr(icfg, "_make_closure_patcher", None)
        if maker is None:
            raise RuntimeError("this torch has no ConfigModule._make_closure_patcher")
        fast_enter = maker(**cfg)
        reason = _prove_equivalent(cfg, fast_enter)
        if reason is not None:
            raise RuntimeError(reason)
    except Exception as exc:  # noqa: BLE001 -- a host-time pin must never break a compile
        msg = f"{type(exc).__name__}: {exc}"
        _REFUSALS.append(msg)
        _COUNTS["refused"] += 1
        _COUNTS["built_slow"] += 1
        print(
            f"[ISOEXEC-AUTOFUSE] fast inductor-config pin REFUSED ({msg}); the per-call "
            f"icfg.patch stays in charge for this artifact (correct, just not memoized).",
            flush=True,
        )
        return _slow_pin(cfg)

    def enter():
        _COUNTS["served"] += 1
        return fast_enter()

    _COUNTS["built_fast"] += 1
    print(
        f"[ISOEXEC-AUTOFUSE] fast inductor-config pin BUILT for {len(cfg)} keys: torch's own "
        f"ConfigModule._make_closure_patcher, PROVEN on the live config object to install and "
        f"revert byte-identically to icfg.patch (whole-config compare, not just these keys). "
        f"The pin is still entered on every call -- a dynamo recompile still regenerates under "
        f"the admitted numerics -- it is just no longer rebuilt on every call. Read "
        f"cfgpin_counts()['served'].",
        flush=True,
    )
    return enter
