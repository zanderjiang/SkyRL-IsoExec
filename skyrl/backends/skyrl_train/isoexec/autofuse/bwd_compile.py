"""Compile backward-only regions: tune once, pin the decision, fail closed to eager."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import threading
from dataclasses import dataclass

BANNER = "[ISOEXEC-BWD-COMPILE]"

#: Bumped when the admission bar changes; folded into the ledger key so verdicts recorded under a
#: weaker battery resolve to eager instead of being honoured.
GATE_VERSION = 1

#: Sites whose target is reachable only from an ``autograd.Function.backward`` staticmethod. This
#: table is the scope proof; ``tests/test_bwd_compile_cpu.py`` asserts each entry against the
#: target's call sites, so adding a site without extending that test fails.
_BACKWARD_ONLY_SITES: dict[str, str] = {
    "gdn.conv_vjp.dz_chunk": "skyrl/backends/skyrl_train/isoexec/ops/gdn/gdn_ops.py",
    "gdn.conv_vjp.act_chunk": "skyrl/backends/skyrl_train/isoexec/ops/gdn/gdn_ops.py",
    "gdn.conv_vjp.dxdw_chunk": "skyrl/backends/skyrl_train/isoexec/ops/gdn/gdn_ops.py",
    "moe.fastbwd.epilogue_hs_chunk": "skyrl/backends/skyrl_train/isoexec/ops/moe/moe_backward_kernel.py",
    "moe.fastbwd.epilogue_vjp_chunk": "skyrl/backends/skyrl_train/isoexec/ops/moe/moe_backward_kernel.py",
}

_COUNT_TEMPLATE: dict[str, int] = {
    "served": 0,  # calls that ran a compiled artifact
    "eager": 0,  # calls that ran the eager fallback
    "no_entry": 0,  # distinct keys with no ledger verdict
    "drift": 0,  # distinct keys whose graph digest moved
    "error": 0,  # distinct keys whose compile raised
    "unengaged": 0,  # distinct keys where dynamo produced no compiled frame
    "compiled_keys": 0,  # distinct keys serving a compiled artifact
    "reported": 0,
}
_COUNTS: dict[str, int] = dict(_COUNT_TEMPLATE)
# Per-site counters keep a partially admitted family debuggable: a hot site must not hide a
# zero-served one.
_SITE_COUNTS: dict[str, dict[str, int]] = {}
_CACHE: dict[tuple, object] = {}  # (site, shape_key) -> artifact or None (None == pinned eager)
_LOCK = threading.RLock()
_BANNERED: set[str] = set()


def _site_census(site: str) -> dict[str, int]:
    with _LOCK:
        return _SITE_COUNTS.setdefault(site, dict(_COUNT_TEMPLATE))


def _increment(site: str, bucket: str, amount: int = 1) -> None:
    with _LOCK:
        _COUNTS[bucket] += amount
        _site_census(site)[bucket] += amount


def bwd_compile_enabled() -> bool:
    """Read at call time; the default (off) keeps behaviour byte-identical."""
    return os.environ.get("SKYRL_ISOEXEC_BWD_COMPILE", "0") == "1"


def bwd_compile_role() -> str:
    r = os.environ.get("SKYRL_ISOEXEC_BWD_COMPILE_ROLE", "reader").lower()
    return r if r in ("reader", "writer") else "reader"


def ledger_path() -> pathlib.Path:
    env = os.environ.get("SKYRL_ISOEXEC_BWD_COMPILE_LEDGER", "")
    if env:
        return pathlib.Path(env)
    return pathlib.Path.home() / ".cache" / "skyrl-isoexec" / "bwd_compile.json"


@dataclass(frozen=True)
class BackwardCompileSelectionIdentity:
    """Immutable identity payload for composition integration, owned by the release builder.

    This module never registers or mutates the process manifest. Ledger path and reader/writer role
    are deployment details and excluded; everything arithmetic-bearing is included.
    """

    schema_version: int
    enabled: bool
    gate_version: int
    arch: str
    toolchain: str
    config_digest: str
    ledger_digest: str
    admitted_sites: tuple[str, ...]


def backward_compile_selection_identity() -> BackwardCompileSelectionIdentity:
    """Return the declarative identity payload; never writes or registers anything."""
    enabled = bwd_compile_enabled()
    digest = "disabled"
    sites: tuple[str, ...] = ()
    if enabled:
        try:
            data = json.loads(ledger_path().read_text())
            if not isinstance(data, dict):
                raise TypeError("ledger root is not an object")
            suffix = f"|{_arch_tag()}|{_torch_fp()}|{_config_digest()}|gate{GATE_VERSION}"
            applicable = {key: entry for key, entry in data.items() if isinstance(key, str) and key.endswith(suffix)}
            canonical = json.dumps(applicable, sort_keys=True, separators=(",", ":")).encode()
            digest = hashlib.sha256(canonical).hexdigest()
            sites = tuple(
                sorted(
                    {
                        entry.get("site")
                        for entry in applicable.values()
                        if isinstance(entry, dict) and entry.get("site") in _BACKWARD_ONLY_SITES
                    }
                )
            )
        except FileNotFoundError:
            digest = "absent"
        except Exception:  # noqa: BLE001 -- invalid ledgers fail closed to eager, but remain identifiable
            try:
                digest = f"invalid:{hashlib.sha256(ledger_path().read_bytes()).hexdigest()}"
            except Exception:  # noqa: BLE001
                digest = "invalid:unreadable"
    return BackwardCompileSelectionIdentity(
        schema_version=1,
        enabled=enabled,
        gate_version=GATE_VERSION,
        arch=_arch_tag(),
        toolchain=_torch_fp(),
        config_digest=_config_digest(),
        ledger_digest=digest,
        admitted_sites=sites,
    )


def _arch_tag() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            return f"sm{cap[0]}{cap[1]}"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def _torch_fp() -> str:
    """Toolchain identity: torch, CUDA and triton versions."""
    parts = []
    try:
        import torch

        parts.append(torch.__version__)
        parts.append(getattr(torch.version, "cuda", None) or "cpu")
    except Exception:  # noqa: BLE001
        parts.append("no-torch")
    try:
        import triton

        parts.append(f"triton-{triton.__version__}")
    except Exception:  # noqa: BLE001
        parts.append("triton-none")
    return "|".join(parts)


# Drift probe: a digest of the region's aten-level graph under the pinned config.
def _config_digest() -> str:
    from .region_gate import REGION_INDUCTOR_CONFIG

    blob = json.dumps(REGION_INDUCTOR_CONFIG, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def graph_digest(fn, example_args) -> str:
    """SHA-256 (16 hex) of the region's aten graph, traced on fake tensors -- no device work.

    Callers catch raises and demote the site to eager: a region that cannot be fingerprinted must not
    be served compiled.
    """
    from .region_gate import _trace_aten

    gm = _trace_aten(fn, example_args)
    src = gm.code
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def entry_key(site: str, shape_key: str) -> str:
    return "|".join((site, shape_key, _arch_tag(), _torch_fp(), _config_digest(), f"gate{GATE_VERSION}"))


# The ledger; read-mostly, only the offline battery writes.
def _load() -> dict:
    try:
        data = json.loads(ledger_path().read_text())
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 -- a corrupt ledger must fail CLOSED, never re-tune
        return {}


def record(site: str, shape_key: str, digest: str, *, oracle: dict | None = None) -> None:
    """Writer-only: admit (site, shape) with the graph digest the battery verified.

    Refuses in a reader process. The only writer is the offline battery; a verdict written from a live
    step would make the pinned artifact a function of that step's shapes.
    """
    if bwd_compile_role() != "writer":
        raise RuntimeError(
            f"{BANNER} record() called with role={bwd_compile_role()!r}. Verdicts are written only "
            "by the offline battery (SKYRL_ISOEXEC_BWD_COMPILE_ROLE=writer); a production step that "
            "could write its own verdict is exactly the run-to-run gradient drift this ledger exists "
            "to remove."
        )
    with _LOCK:
        data = _load()
        data[entry_key(site, shape_key)] = {
            "site": site,
            "shape_key": shape_key,
            "graph_digest": digest,
            "oracle": oracle or {},
        }
        p = ledger_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
        tmp.replace(p)  # atomic: ranks may write concurrently


def verdict(site: str, shape_key: str) -> dict | None:
    e = _load().get(entry_key(site, shape_key))
    return e if isinstance(e, dict) else None


def _banner_once(msg: str) -> None:
    if msg in _BANNERED:
        return
    _BANNERED.add(msg)
    print(msg, flush=True)


def _resolve(site: str, fn, args):
    """Return a compiled artifact for (site, shape), or None meaning pinned eager. Never raises."""
    from .region_gate import shape_key_of

    shape_key = shape_key_of(args)
    ck = (site, shape_key)
    with _LOCK:
        if ck in _CACHE:
            return _CACHE[ck]

    def pin_eager(reason: str, bucket: str):
        with _LOCK:
            _CACHE[ck] = None
            _increment(site, bucket)
        _banner_once(f"{BANNER} pid={os.getpid()} site={site} shape={shape_key} -> EAGER ({reason})")
        return None

    v = verdict(site, shape_key)
    if v is None:
        return pin_eager("no ledger verdict for this (shape, arch, toolchain, config, gate)", "no_entry")

    try:
        live = graph_digest(fn, args)
    except Exception as e:  # noqa: BLE001
        return pin_eager(f"graph digest failed: {type(e).__name__}: {e}", "error")
    if live != v.get("graph_digest"):
        return pin_eager(f"DRIFT: graph digest {live} != admitted {v.get('graph_digest')}", "drift")

    try:
        from .region_gate import compile_pinned

        art = compile_pinned(fn, list(args), site=site)
    except Exception as e:  # noqa: BLE001
        return pin_eager(f"compile raised {type(e).__name__}: {e}", "error")
    if not getattr(art, "engaged", False):
        return pin_eager("dynamo produced no compiled frame (artifact is eager)", "unengaged")

    with _LOCK:
        _CACHE[ck] = art
        _increment(site, "compiled_keys")
    _banner_once(
        f"{BANNER} pid={os.getpid()} site={site} shape={shape_key} -> COMPILED "
        f"(ledger ADMITTED; graph_digest={live}; oracle={v.get('oracle', {})})"
    )
    return art


def call_region(site: str, fn, *args):
    """Run ``fn(*args)`` through its admitted compiled artifact, or eagerly. Fails closed.

    ``site`` must be in :data:`_BACKWARD_ONLY_SITES`; the backward-only scope proof is that table plus
    the CPU test on the target's call sites, not a runtime predicate.
    """
    if not bwd_compile_enabled():
        _increment(site, "eager")
        return fn(*args)
    if site not in _BACKWARD_ONLY_SITES:
        # A site whose backward-only scope has not been asserted must never be compiled.
        raise KeyError(
            f"{BANNER} site {site!r} is not in _BACKWARD_ONLY_SITES. Register it and extend "
            "tests/test_bwd_compile_cpu.py's call-site assertion before compiling it."
        )
    art = _resolve(site, fn, args)
    if art is None:
        _increment(site, "eager")
        return fn(*args)
    try:
        out = art(*args)
    except Exception as e:  # noqa: BLE001 -- a runtime failure demotes rather than propagating
        from .region_gate import shape_key_of

        with _LOCK:
            _CACHE[(site, shape_key_of(args))] = None
            _increment(site, "error")
        _banner_once(f"{BANNER} pid={os.getpid()} site={site} DEMOTED (runtime {type(e).__name__}: {e})")
        _increment(site, "eager")
        return fn(*args)
    _increment(site, "served")
    _report(site)
    return out


def _report(site: str) -> None:
    served = _COUNTS["served"]
    if served >= 1 and (served & (served - 1)) == 0 and served != _COUNTS["reported"]:
        _COUNTS["reported"] = served
        print(
            f"{BANNER} pid={os.getpid()} served={served} eager={_COUNTS['eager']} "
            f"compiled_keys={_COUNTS['compiled_keys']} no_entry={_COUNTS['no_entry']} "
            f"drift={_COUNTS['drift']} error={_COUNTS['error']} unengaged={_COUNTS['unengaged']}",
            flush=True,
        )
    site_counts = _site_census(site)
    site_served = site_counts["served"]
    if site_served < 1 or (site_served & (site_served - 1)) != 0 or site_served == site_counts["reported"]:
        return
    site_counts["reported"] = site_served
    print(
        f"{BANNER} pid={os.getpid()} site={site} served={site_served} eager={site_counts['eager']} "
        f"compiled_keys={site_counts['compiled_keys']} no_entry={site_counts['no_entry']} "
        f"drift={site_counts['drift']} error={site_counts['error']} unengaged={site_counts['unengaged']}",
        flush=True,
    )


def install_banner() -> str:
    """One line at install; not evidence of engagement."""
    if not bwd_compile_enabled():
        return (
            f"{BANNER} pid={os.getpid()} INERT: SKYRL_ISOEXEC_BWD_COMPILE is not set "
            f"({len(_BACKWARD_ONLY_SITES)} backward-only sites untouched)"
        )
    return (
        f"{BANNER} pid={os.getpid()} install: ledger={ledger_path()} role={bwd_compile_role()} "
        f"arch={_arch_tag()} cfg={_config_digest()} gate={GATE_VERSION} "
        f"sites={len(_BACKWARD_ONLY_SITES)} -- this is an INSTALL line, not engagement; "
        f"read served= from {BANNER}"
    )
