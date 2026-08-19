"""The composition manifest: one declared object, resolved once, delivered to both runtimes.

A manifest keys on ``(op, site)`` and every installed key carries an explicit entry -- absence means
"this op has no such site", never "default". Each entry is classified per (op, site): FUNCTION
attributes can move bits and are hashed, DEPLOYMENT attributes are proven bitwise-neutral, so they
are logged and cross-checked but excluded from the hash and refused without a neutrality proof. The
hash keys gate signatures, folds in the arch, and is handshaked at weight sync so a mismatched
composition refuses to run.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from .arch import ARCH, is_accelerator_arch
from .registry import Registry

# The env var carrying the manifest hash for the delivery cross-check.
MANIFEST_HASH_ENV = "ISOEXEC_MANIFEST_HASH"

# Per-entry classification vocabulary.
FUNCTION = "function"
DEPLOYMENT = "deployment"
_CLASSES = frozenset({FUNCTION, DEPLOYMENT})

Key = Tuple[str, str]  # (op, site)


class CompositionError(ValueError):
    """Raised on any manifest-rule violation: unknown class, deployment without neutrality
    proof, mutation after freeze, or an installed/named key mismatch."""


class FrozenManifestError(CompositionError):
    """Raised on mutation after ``freeze()``."""


class PinValidationError(CompositionError):
    """Raised when a manifest's pinned constants disagree with the selected impl's declared
    ``machine_assertable`` rounding schedule (``Manifest.validate_pins``)."""


@dataclass(frozen=True)
class ManifestEntry:
    """One resolved (op, site) selection; identity is impl@version x arch.

    ``pinned_constants`` (autotune index, boundary dtypes, block sizes, leaf counts) move bits, so
    they are normalized into the hash payload. ``neutrality_proof`` is mandatory for a DEPLOYMENT
    entry: it points at the gate run that proved this entry bitwise-neutral.
    """

    impl_id: str
    version: int
    pinned_constants: Dict[str, object] = field(default_factory=dict)
    classification: str = FUNCTION
    neutrality_proof: Optional[str] = None

    def __post_init__(self) -> None:
        if self.classification not in _CLASSES:
            raise CompositionError(f"unknown classification {self.classification!r}; must be one of {sorted(_CLASSES)}")
        if self.classification == DEPLOYMENT and not self.neutrality_proof:
            raise CompositionError(
                "deployment classification requires a recorded neutrality_proof (a run id / gate "
                "result pointer). A neutrality proof licenses exactly the entry it was measured "
                "on; refusing to classify without one."
            )
        if self.classification == FUNCTION and self.neutrality_proof:
            # A function-half entry is hashed regardless, so carrying a proof is a classification bug.
            raise CompositionError(
                "a function-half entry must not carry a neutrality_proof; either it is proven "
                "neutral (classify it deployment) or it is not (leave it function)."
            )

    def _hash_payload(self) -> dict:
        """The canonical, order-stable dict folded into the function-half hash."""
        return {
            "impl_id": self.impl_id,
            "version": self.version,
            "pinned_constants": _canonical(self.pinned_constants),
        }


def _canonical(obj):
    """Recursively sort dict keys so the JSON serialization, and therefore the hash, is stable
    across insertion order."""
    if isinstance(obj, dict):
        return {k: _canonical(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(x) for x in obj]
    return obj


@dataclass
class ResolvedFingerprint:
    """What the adapter records was actually installed per (op, site): module, qualname, pinned
    constants. Stronger than hashing the config -- it catches "the flag arrived but the install did
    not happen"."""

    installed: Dict[Key, dict] = field(default_factory=dict)

    def keys(self) -> frozenset:
        return frozenset(self.installed.keys())


class Manifest:
    """The single declared composition both runtimes carry. Built once, frozen, delivered.

    Entries are keyed on (op, site). ``arch`` is a top-level field folded into the hash. The
    function half keys gate signatures; the deployment half is logged but not hashed.
    """

    def __init__(self, arch: str = ARCH, model: Optional[str] = None) -> None:
        self.arch = arch
        self.model = model
        self._entries: Dict[Key, ManifestEntry] = {}
        self._frozen = False
        self._freeze_count = 0

    def set_entry(self, op: str, site: str, entry: ManifestEntry) -> None:
        if self._frozen:
            raise FrozenManifestError("cannot mutate a frozen manifest")
        key = (op, site)
        if key in self._entries:
            raise CompositionError(
                f"duplicate manifest entry for {key}; composition construction is append-only and "
                "must resolve precedence before insertion (silent overwrite is forbidden)"
            )
        self._entries[key] = entry

    def entries(self) -> Dict[Key, ManifestEntry]:
        return dict(self._entries)

    def keys(self) -> frozenset:
        return frozenset(self._entries.keys())

    def function_entries(self) -> Dict[Key, ManifestEntry]:
        return {k: e for k, e in self._entries.items() if e.classification == FUNCTION}

    def deployment_entries(self) -> Dict[Key, ManifestEntry]:
        return {k: e for k, e in self._entries.items() if e.classification == DEPLOYMENT}

    def freeze(self) -> "Manifest":
        """Resolve-once boundary; after freeze, ``set_entry`` raises."""
        if self._frozen:
            raise FrozenManifestError("manifest freeze is a one-shot boundary; already frozen")
        self._frozen = True
        self._freeze_count += 1
        return self

    @property
    def frozen(self) -> bool:
        return self._frozen

    def hash(self, allow_non_accelerator_arch: bool = False) -> str:
        """Stable sha256 over the sorted function-half entries, their pinned constants and the arch.

        Deployment entries are excluded. Any change to a function-half impl, version or constant, or
        to the arch, rotates the key and requires a freshly admitted gate signature. Hashing a
        manifest whose arch is the non-accelerator sentinel is refused here: it would key every
        signature away from the real accelerator table while the file/env cross-check still passed,
        because a worker recomputes the same sentinel hash from the file.
        """
        if not allow_non_accelerator_arch and not is_accelerator_arch(self.arch):
            raise CompositionError(
                f"refusing to hash a manifest whose arch is {self.arch!r}: that is the "
                f"non-accelerator sentinel (core/arch.NON_ACCELERATOR_ARCH), not a real "
                f"accelerator key. Folding it into the manifest hash silently mis-keys every "
                f"gate signature away from the real accelerator table (e.g. sm90), and the "
                f"file/env hash cross-check would still PASS -- the exact split-brain this "
                f"manifest exists to prevent. This usually means the manifest was "
                f"built/frozen in a process with no visible CUDA device (a launcher/driver); "
                f"build it where the accelerator is visible, or pass an explicit arch. "
                f"CPU-only tests that intentionally hash a sentinel manifest must pass "
                f"allow_non_accelerator_arch=True."
            )
        payload = {
            "arch": self.arch,
            "function": {
                f"{op}::{site}": e._hash_payload() for (op, site), e in sorted(self.function_entries().items())
            },
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def log_deployment(self) -> dict:
        """The deployment half, for logging and cross-side consistency checks. Not hashed."""
        return {
            f"{op}::{site}": {
                "impl_id": e.impl_id,
                "version": e.version,
                "neutrality_proof": e.neutrality_proof,
            }
            for (op, site), e in sorted(self.deployment_entries().items())
        }

    def validate_against_installed(self, fingerprint) -> None:
        """Reject any installed (op, site) the manifest does not name, and any manifest key that is
        not installed. ``fingerprint`` is a ResolvedFingerprint, a Registry, or a set of keys."""
        if isinstance(fingerprint, ResolvedFingerprint):
            installed = fingerprint.keys()
        elif isinstance(fingerprint, Registry):
            installed = fingerprint.installed_keys()
        else:
            installed = frozenset(fingerprint)
        named = self.keys()
        unnamed = installed - named  # installed but the manifest is silent
        missing = named - installed  # named but not installed: the flag arrived and nothing installed
        if unnamed or missing:
            raise CompositionError(
                "manifest does not match the installed composition:\n"
                f"  installed but UNNAMED by the manifest: {sorted(unnamed)}\n"
                f"  named by the manifest but NOT installed: {sorted(missing)}\n"
                "Every installed (op, site) must carry an explicit entry; absence means "
                "'no such site', never 'default'."
            )

    def validate_pins(self, registry: Registry) -> None:
        """Refuse any pinned constant the selected impl does not declare, or contradicts.

        Every other check here compares the manifest against itself or against the other runtime,
        so none of them can see a manifest that is internally consistent, delivered intact,
        identical on both sides -- and names the wrong function. An unchecked pin is exactly that:
        it hashes, it matches across runtimes, and it constrains nothing, so the gate goes green on
        the wrong model. The impl's schedule is the authority, since it is a claim about the code
        while the profile is a claim about the model. A failure means the profile or the
        declaration is wrong, never that the pin should be dropped or the schedule loosened.

        Raises ``PinValidationError`` naming every offending pin at once.
        """
        problems = []
        who = f"model {self.model!r}" if self.model else "manifest"
        for (op, site), e in sorted(self._entries.items()):
            if not registry.has_op(op):
                if e.pinned_constants:
                    problems.append(f"({op}, {site}): op is not registered, so its pins cannot be checked")
                continue
            impl = registry.get_op(op).impls.get(e.impl_id)
            if impl is None:
                if e.pinned_constants:
                    problems.append(f"({op}, {site}): impl {e.impl_id!r} is not registered on the op")
                continue
            for key in sorted(e.pinned_constants, key=str):
                value = e.pinned_constants[key]
                reason = impl.rounding.check_pin(key, value)
                if reason:
                    problems.append(f"({op}, {site}) impl {e.impl_id}@v{impl.version}: pin {key}={value!r} -- {reason}")
        if problems:
            raise PinValidationError(
                f"{who}: pinned constants disagree with the selected impls' declared rounding "
                f"schedules:\n  " + "\n  ".join(problems) + "\n\n"
                "A pinned constant is a claim about the FUNCTION the manifest names. An unchecked "
                "claim hashes identically on both runtimes, so the IsoExec gate stays GREEN while "
                "the composition names the wrong function -- the failure mode this check exists "
                "for. Fix the profile (it describes a model this impl does not implement) or fix "
                "the impl's machine_assertable schedule (it is stale); do not delete the pin and "
                "do not weaken the schedule to make this pass."
            )

    def assert_layer_uniformity(self, registry: Optional[Registry] = None) -> None:
        """Freeze-time assert point for the constraint that an op resolves identically at every
        layer, since the manifest has no layer dimension. If per-layer variation becomes real, the
        key grows a layer scope here."""
        if registry is not None:
            for op_name, _site in self.keys():
                if registry.has_op(op_name) and not registry.get_op(op_name).per_layer_uniform:
                    raise CompositionError(
                        f"op {op_name!r} declares per_layer_uniform=False but the manifest has no "
                        f"layer dimension (Section 7); a layer-scoped key is required before this "
                        f"op can be installed."
                    )
        return None

    def to_dict(self) -> dict:
        return {
            "arch": self.arch,
            "model": self.model,
            "frozen": self._frozen,
            "hash": self.hash(),
            "entries": {
                f"{op}::{site}": {
                    "impl_id": e.impl_id,
                    "version": e.version,
                    "pinned_constants": _canonical(e.pinned_constants),
                    "classification": e.classification,
                    "neutrality_proof": e.neutrality_proof,
                }
                for (op, site), e in sorted(self._entries.items())
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Manifest":
        m = cls(arch=data.get("arch", ARCH), model=data.get("model"))
        for compound, e in data.get("entries", {}).items():
            op, site = compound.split("::", 1)
            m.set_entry(
                op,
                site,
                ManifestEntry(
                    impl_id=e["impl_id"],
                    version=e["version"],
                    pinned_constants=e.get("pinned_constants", {}) or {},
                    classification=e.get("classification", FUNCTION),
                    neutrality_proof=e.get("neutrality_proof"),
                ),
            )
        if data.get("frozen"):
            m.freeze()
        # Cross-check the serialized hash if one was recorded.
        recorded = data.get("hash")
        if recorded is not None and recorded != m.hash():
            raise CompositionError(f"serialized manifest hash {recorded!r} does not match recomputed {m.hash()!r}")
        return m

    def write_manifest_file(self, path: str) -> str:
        """Write the frozen manifest to the file both sides read and return its hash, which the
        caller should also export as ISOEXEC_MANIFEST_HASH."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)
        return self.hash()


def load_manifest(path: str, expected_hash_env: str = MANIFEST_HASH_ENV) -> Manifest:
    """Read a delivered manifest file and cross-check its hash against the env var.

    A dropped or edited delivery then fails loudly. The file's self-consistency is always checked;
    when the env var is set it must additionally equal the file's hash.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    manifest = Manifest.from_dict(data)
    env_hash = os.environ.get(expected_hash_env)
    if env_hash is not None and env_hash != manifest.hash():
        raise CompositionError(
            f"manifest delivery cross-check FAILED: {expected_hash_env}={env_hash!r} but the "
            f"file at {path!r} hashes to {manifest.hash()!r}. A mismatched composition refuses "
            f"to run rather than producing a silent divergence."
        )
    return manifest


def build_manifest(
    registry: Registry,
    selections: dict,
    arch: str = ARCH,
    model: Optional[str] = None,
    validate_pins: bool = True,
) -> Manifest:
    """Build a manifest from a registry and explicit selections, validating both.

    ``selections`` maps (op, site) to a ManifestEntry or a dict coerced to one; every key must name
    a site the registry's op declares. ``validate_pins=True`` is the derivation-time gate on pins
    and the only supported production setting -- turning it off re-opens the hole where a manifest
    hashes and cross-matches while naming the wrong function. Completeness against what was
    actually installed is a separate, adapter-time check (``validate_against_installed``).
    """
    m = Manifest(arch=arch, model=model)
    for key, val in selections.items():
        op, site = key
        if not registry.has_op(op):
            raise CompositionError(f"selection names unknown op {op!r} (not in registry)")
        op_spec = registry.get_op(op)
        if site not in op_spec.sites:
            raise CompositionError(
                f"selection names site {site!r} for op {op!r}, but that op declares only "
                f"{sorted(op_spec.sites)} (absence means 'no such site')"
            )
        entry = val if isinstance(val, ManifestEntry) else ManifestEntry(**val)
        if entry.impl_id not in op_spec.impls:
            raise CompositionError(
                f"selection for ({op!r}, {site!r}) names impl {entry.impl_id!r} not registered "
                f"on the op (known: {sorted(op_spec.impls)})"
            )
        m.set_entry(op, site, entry)
    if validate_pins:
        m.validate_pins(registry)
    return m
