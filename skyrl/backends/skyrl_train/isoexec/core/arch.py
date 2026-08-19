"""HardwareTarget: architecture as a first-class dimension of a composition.

Kernel bit patterns are arch-specific, so an implementation's identity is ``impl@version x arch``:
``arch`` is folded into the manifest hash, every recorded signature is arch-scoped, and the
weight-sync handshake refuses trainer-arch != engine-arch. Detection is CPU-safe and falls back to
the ``NON_ACCELERATOR_ARCH`` sentinel, which ``Manifest.hash`` rejects rather than mis-key on.
"""

from __future__ import annotations

from dataclasses import dataclass

# Reserved arch tag for hosts where no accelerator was detected. It never collides with a real
# accelerator key, and the manifest-hash boundary rejects it instead of accepting it as a key.
NON_ACCELERATOR_ARCH: str = "cpu"


def is_accelerator_arch(arch: str) -> bool:
    """True iff ``arch`` names a real accelerator, i.e. non-empty and not the sentinel.

    Manifest hashing requires this: keying a bitwise gate table on the non-accelerator sentinel is
    a silent mis-key away from the real table.
    """
    return bool(arch) and arch != NON_ACCELERATOR_ARCH


def detect_arch() -> str:
    """Return the current CUDA compute arch as a short tag.

    (9, 0) -> "sm90"; (10, x) -> "sm100"; (M, N) -> "sm{M}{N}"; no torch or no visible CUDA
    device -> NON_ACCELERATOR_ARCH. Once CUDA reports itself available a failure to read the device
    capability propagates instead of collapsing to the sentinel, which would otherwise be baked
    into a manifest hash.
    """
    try:
        import torch  # deferred: core stays importable in an environment without torch
    except Exception:
        return NON_ACCELERATOR_ARCH
    try:
        cuda_available = torch.cuda.is_available()
    except Exception:
        return NON_ACCELERATOR_ARCH
    if not cuda_available:
        return NON_ACCELERATOR_ARCH
    # CUDA is available, so a capability-read failure is a detection bug: surface it, do not mis-key.
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) == (9, 0):
        return "sm90"
    if major == 10:
        return "sm100"
    return f"sm{major}{minor}"


# Resolved once at import; the manifest folds this constant into its hash.
ARCH: str = detect_arch()


class ArchMismatchError(RuntimeError):
    """Raised when trainer-arch != engine-arch: a heterogeneous deployment refuses to run."""


@dataclass(frozen=True)
class HardwareTarget:
    """The arch dimension of a composition; two targets are equal iff their arch tags are equal."""

    arch: str

    @classmethod
    def current(cls) -> "HardwareTarget":
        return cls(arch=ARCH)

    def supports(self, supported_archs) -> bool:
        return self.arch in set(supported_archs)

    def assert_homogeneous(self, other_arch: str) -> None:
        """Assert the other runtime runs the same arch. A mismatch is refuse-to-run, never a warning."""
        if self.arch != other_arch:
            raise ArchMismatchError(
                f"arch handshake failed: this side is {self.arch!r}, other side is "
                f"{other_arch!r}. IsoExec requires trainer-arch == engine-arch; no cross-arch "
                f"bitwise claim is ever made."
            )


def assert_homogeneous(arch_a: str, arch_b: str) -> None:
    """Free-function form of the handshake assert, for call sites holding two bare tags."""
    if arch_a != arch_b:
        raise ArchMismatchError(f"arch handshake failed: {arch_a!r} != {arch_b!r}.")
