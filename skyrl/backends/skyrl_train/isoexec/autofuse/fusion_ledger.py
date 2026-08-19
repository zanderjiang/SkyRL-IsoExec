"""The fusion decision ledger: verdicts are produced offline by the gate, and installs only read.

Neither runtime decides anything locally, so the two sides cannot split-brain; a site absent from
the ledger is eager on both. Entry keys carry (site, region signature, shape key, arch, torch
fingerprint, config fingerprint), so a changed compiler, arch or pinned config cannot serve an old
verdict. Missing, stale or corrupt entries resolve to eager; asymmetry is refused by the manifest
handshake, via the ``{site: decision}`` pin returned by ``manifest_pin``.

Ledger location: ``SKYRL_ISOEXEC_AUTOFUSE_LEDGER`` or ``~/.cache/skyrl-isoexec/autofuse.json``.
Writes are read-modify-write to a tmp file plus ``os.replace``.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
from typing import Any

from .region_gate import ADMITTED, RegionVerdict

BANNER = "[ISOEXEC-AUTOFUSE]"


def ledger_path() -> pathlib.Path:
    env = os.environ.get("SKYRL_ISOEXEC_AUTOFUSE_LEDGER", "")
    if env:
        return pathlib.Path(env)
    return pathlib.Path.home() / ".cache" / "skyrl-isoexec" / "autofuse.json"


def autofuse_enabled() -> bool:
    """Default on. The flag only licenses the ledger to be consulted; with no matching entry every
    site still resolves eager, so an absent, stale or foreign ledger is bit-identical to off."""
    return os.environ.get("SKYRL_ISOEXEC_AUTOFUSE", "1") == "1"


def _entry_key(site: str, region_sig: str, shape_key: str, arch: str, torch_fp: str, config_fp: str) -> str:
    raw = "|".join((site, region_sig, shape_key, arch, torch_fp, config_fp))
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


class FusionLedger:
    """Read/write access to the verdict store. Instances are cheap; state is the file."""

    def __init__(self, path: pathlib.Path | None = None):
        self.path = path or ledger_path()

    def load(self) -> dict[str, dict]:
        try:
            data = json.loads(self.path.read_text())
        except Exception:  # noqa: BLE001 -- missing/corrupt ledger = empty ledger = all eager
            return {}
        entries = data.get("entries")
        return entries if isinstance(entries, dict) else {}

    def lookup(
        self, *, site: str, region_sig: str, shape_key: str, arch: str, torch_fp: str, config_fp: str
    ) -> dict | None:
        return self.load().get(_entry_key(site, region_sig, shape_key, arch, torch_fp, config_fp))

    def digest(self) -> str:
        """Canonical content hash; 'ABSENT' for a missing or empty ledger, so that state is hashable
        and still agrees across sides."""
        entries = self.load()
        if not entries:
            return "ABSENT"
        blob = json.dumps(entries, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def fingerprint_census(self, *, arch: str, torch_fp: str, config_fp: str) -> dict:
        """How many of this ledger's entries can ever resolve in this process.

        Fingerprint misses are silent and resolve to eager, so the installer reports this count to
        distinguish an engaged ledger from an inert one.
        """
        entries = self.load()
        matching = 0
        admitted = 0
        for e in entries.values():
            if not isinstance(e, dict):
                continue
            if e.get("arch") == arch and e.get("torch_fp") == torch_fp and e.get("config_fp") == config_fp:
                matching += 1
                if e.get("verdict") == ADMITTED:
                    admitted += 1
        return {
            "path": str(self.path),
            "exists": self.path.exists(),
            "total": len(entries),
            "matching": matching,
            "admitted": admitted,
            "arch": arch,
            "torch_fp": torch_fp,
            "config_fp": config_fp,
        }

    # Write path: harness only, never called from an install path.

    def record(self, verdict: RegionVerdict, *, sweep: dict | None = None) -> None:
        entries = self.load()
        key = _entry_key(
            verdict.site,
            verdict.region_sig,
            verdict.shape_key,
            verdict.arch,
            verdict.torch_fp,
            verdict.config_fp,
        )
        entries[key] = verdict.to_dict()
        payload: dict[str, Any] = {"entries": entries}
        if sweep is not None:
            payload["transcendental_sweep"] = sweep
        else:
            payload["transcendental_sweep"] = self.sweep_result()
        self._write(payload)

    def record_sweep(self, sweep: dict) -> None:
        payload = {"entries": self.load(), "transcendental_sweep": sweep}
        self._write(payload)

    def sweep_result(self) -> dict | None:
        try:
            data = json.loads(self.path.read_text())
            s = data.get("transcendental_sweep")
            return s if isinstance(s, dict) else None
        except Exception:  # noqa: BLE001
            return None

    def admitted_transcendentals(self, *, arch: str, torch_fp: str) -> frozenset[str]:
        """The swept-and-admitted primitive set, valid only for the exact (arch, torch) it was measured
        on; anything else returns the empty set, leaving transcendentals banned."""
        s = self.sweep_result()
        if not s or s.get("arch") != arch or s.get("torch_fp") != torch_fp:
            return frozenset()
        return frozenset(s.get("admitted", ()))

    def _write(self, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, indent=1, sort_keys=True))
        tmp.replace(self.path)


# Install-time resolution: the only api a runtime may call.


def resolve_decision(
    *,
    site: str,
    region_sig: str,
    shape_key: str,
    arch: str,
    torch_fp: str,
    config_fp: str,
    ledger: FusionLedger | None = None,
) -> tuple[str, str]:
    """Resolve one (site, shape) to ``("compiled" | "eager", reason)``.

    "compiled" only when the master flag is on and the ledger holds an ADMITTED verdict under the
    exact same (region_sig, shape_key, arch, torch_fp, config_fp); everything else is eager.
    """
    if not autofuse_enabled():
        return "eager", "SKYRL_ISOEXEC_AUTOFUSE is not set"
    ledger = ledger or FusionLedger()
    entry = ledger.lookup(
        site=site,
        region_sig=region_sig,
        shape_key=shape_key,
        arch=arch,
        torch_fp=torch_fp,
        config_fp=config_fp,
    )
    if entry is None:
        return "eager", "no ledger entry for this (site, region, shape, arch, torch, config)"
    if entry.get("verdict") != ADMITTED:
        return "eager", f"ledger verdict is {entry.get('verdict')}: {entry.get('reason')}"
    return "compiled", "ADMITTED in ledger"


def manifest_pin(sites: dict[str, tuple[str, str, str]], *, ledger: FusionLedger | None = None) -> dict:
    """The canonical dict a manifest entry pins, so asymmetric resolution refuses cross-side.

    ``sites`` maps site -> (region_sig, shape_key, config_fp). Fold the result into
    ``pinned_constants`` of the autofuse manifest entry; the ledger digest is included so even a
    decision-neutral content change is visible.
    """
    ledger = ledger or FusionLedger()
    import_torch_fp = None
    try:
        from .region_gate import arch_tag, torch_fingerprint

        import_torch_fp = torch_fingerprint()
        arch = arch_tag()
    except Exception:  # noqa: BLE001
        arch = "unknown"
    decisions = {}
    for site, (region_sig, shape_key, config_fp) in sorted(sites.items()):
        decision, _ = resolve_decision(
            site=site,
            region_sig=region_sig,
            shape_key=shape_key,
            arch=arch,
            torch_fp=import_torch_fp or "unknown",
            config_fp=config_fp,
            ledger=ledger,
        )
        decisions[site] = decision
    return {"autofuse_ledger_digest": ledger.digest(), "autofuse_decisions": decisions}


def install_banner(decisions: dict[str, tuple[str, str]]) -> str:
    """One line per site with decision and reason, so "refused" and "never looked" read differently."""
    lines = [f"{BANNER} ledger={ledger_path()} digest={FusionLedger().digest()}"]
    for site, (decision, reason) in sorted(decisions.items()):
        lines.append(f"{BANNER}   {site:<40} -> {decision.upper():<8} ({reason})")
    return "\n".join(lines)
