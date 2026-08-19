"""The per-(model, function-manifest-hash, arch) gate-signature table.

A gate signature is the step-1 bitwise fingerprint of a composition. ``admit`` refuses a signature
that carries no proof reference: a signature is admissible only with a passing site-parity proof,
never because a live number moved. The key includes the function-half manifest hash, which folds in
arch, so any bit-moving change rotates the key and requires a freshly admitted signature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Sentinel hash for the seeded reference, whose composition predates the manifest builder.
LEGACY_HASH = "LEGACY-PRE-MANIFEST"

Key = Tuple[str, str, str]  # (model, function_manifest_hash, arch)


class SignatureError(ValueError):
    """Raised on an inadmissible signature (no proof) or a duplicate/conflicting admission."""


@dataclass(frozen=True)
class SignatureRecord:
    """One admitted gate signature.

    ``signature`` is policy/rollout_train_logprobs_abs_diff_mean from pre-update scoring and
    ``entropy`` the step-1 policy_entropy recorded alongside it. ``proof_ref`` is mandatory: it
    points at the passing site-parity proof or soak that admitted this signature.
    """

    model: str
    function_manifest_hash: str
    arch: str
    signature: float
    entropy: float
    proof_ref: str
    band: Optional[Tuple[float, float]] = None
    grad_norm: Optional[Tuple[float, float]] = None
    notes: str = ""

    @property
    def key(self) -> Key:
        return (self.model, self.function_manifest_hash, self.arch)


class SignatureTable:
    """The admissible-signature registry; ``admit`` enforces the proof rule."""

    def __init__(self) -> None:
        self._records: Dict[Key, SignatureRecord] = {}

    def admit(
        self,
        model: str,
        function_manifest_hash: str,
        arch: str,
        signature: float,
        entropy: float,
        proof_ref: str,
        band: Optional[Tuple[float, float]] = None,
        grad_norm: Optional[Tuple[float, float]] = None,
        notes: str = "",
    ) -> SignatureRecord:
        """Admit a signature; refused without a proof reference.

        Re-admitting the same (model, hash, arch) with a different signature is a conflict and
        raises -- a rotated bit-moving attribute must rotate the hash, not overwrite a signature.
        Identical re-admission is idempotent.
        """
        if not proof_ref or not str(proof_ref).strip():
            raise SignatureError(
                f"refusing to admit a signature for ({model!r}, {function_manifest_hash!r}, "
                f"{arch!r}) without a proof reference. A signature is admissible ONLY with a "
                f"passing site-parity proof, never because a live number moved."
            )
        rec = SignatureRecord(
            model=model,
            function_manifest_hash=function_manifest_hash,
            arch=arch,
            signature=signature,
            entropy=entropy,
            proof_ref=proof_ref,
            band=band,
            grad_norm=grad_norm,
            notes=notes,
        )
        existing = self._records.get(rec.key)
        if existing is not None and existing.signature != rec.signature:
            raise SignatureError(
                f"conflicting signature for {rec.key}: existing {existing.signature!r} vs new "
                f"{rec.signature!r}. A rotated bit-moving attribute must rotate the HASH (a new "
                f"key), not overwrite a signature."
            )
        self._records[rec.key] = rec
        return rec

    def lookup(self, model: str, function_manifest_hash: str, arch: str) -> Optional[SignatureRecord]:
        return self._records.get((model, function_manifest_hash, arch))

    def require(self, model: str, function_manifest_hash: str, arch: str) -> SignatureRecord:
        rec = self.lookup(model, function_manifest_hash, arch)
        if rec is None:
            raise SignatureError(
                f"no admitted signature for ({model!r}, {function_manifest_hash!r}, {arch!r}); "
                f"a composition with no reference signature cannot be gated."
            )
        return rec

    def for_model(self, model: str) -> List[SignatureRecord]:
        return [r for r in self._records.values() if r.model == model]

    def records(self) -> List[SignatureRecord]:
        return list(self._records.values())


# Seeded reference: Qwen3.5-35B-A3B gsm8k, trainer TP2/EP8/ETP1 vs engine TP8, arch sm90.
SIGNATURES = SignatureTable()
SIGNATURES.admit(
    model="qwen3.5-35b-a3b-gsm8k-pik",
    function_manifest_hash=LEGACY_HASH,
    arch="sm90",
    signature=6.559800453942444e-07,
    entropy=0.29838278889656067,
    proof_ref="90-step soak + 12 bit-identical reproductions, 2026-07-21..22",
    band=(4.1e-07, 6.8e-07),
    grad_norm=(0.372, 0.374),
    notes="Phase-0 freeze; LEGACY hash = launcher env/flag set at the branch point "
    "(pre-manifest-builder). Replace with a real Manifest.hash() in Phase 2.",
)
