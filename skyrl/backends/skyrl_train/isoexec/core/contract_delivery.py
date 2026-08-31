"""Deliver the ExecutionContract as the artifact both runtimes read.

Canonical-bytes file plus a hash carried in ``ISOEXEC_CONTRACT_HASH`` for cross-check. Every
installed (op, site) must carry an explicit entry; absence means "no such site", never "default".
"""

from __future__ import annotations

import logging
import os

from ..contract import (
    ExecutionContract,
    compute_identities,
    from_canonical_json,
    to_canonical_json,
)
from .fingerprint import ResolvedFingerprint
from .registry import Registry

CONTRACT_HASH_ENV = "ISOEXEC_CONTRACT_HASH"


class ContractDeliveryError(ValueError):
    pass


def write_contract_file(contract: ExecutionContract, path: str) -> str:
    """Write the frozen contract atomically and return the numerical_policy hash, which the caller
    should also export as ISOEXEC_CONTRACT_HASH."""
    data = to_canonical_json(contract)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)
    return contract.identities.numerical_policy


def load_contract(path: str, expected_hash_env: str = CONTRACT_HASH_ENV) -> ExecutionContract:
    """Read a delivered contract and cross-check it against the env var.

    File self-consistency is always enforced; debug mode demotes only the env cross-check.
    """
    from . import enforce

    with open(path, "rb") as fh:
        contract = from_canonical_json(fh.read())
    if compute_identities(contract) != contract.identities:
        raise ContractDeliveryError(
            f"serialized contract identities at {path!r} do not match the recomputed identities"
        )
    env_hash = os.environ.get(expected_hash_env)
    if env_hash is not None and env_hash != contract.identities.numerical_policy:
        msg = (
            f"contract delivery cross-check FAILED: {expected_hash_env}={env_hash!r} but the "
            f"file at {path!r} carries numerical_policy "
            f"{contract.identities.numerical_policy!r}. A mismatched composition refuses to "
            f"run rather than producing a silent divergence."
        )
        enforce.report(f"{enforce.BUILD_VALID}:contract", enforce.INSTALL, enforce.VIOLATION, msg)
        if not enforce.demoted():
            raise ContractDeliveryError(msg)
        logging.getLogger(__name__).error(
            "[ISOEXEC-DEBUG] DEMOTED (violation still recorded): %s",
            msg,
        )
    return contract


def _primary_op(registry: Registry, entry) -> str:
    # The region member whose OpSpec registers this entry's impl; subsumed ops never install.
    owners = [op for op in entry.region if registry.has_op(op) and entry.impl.id in registry.get_op(op).impls]
    if len(owners) != 1:
        raise ContractDeliveryError(
            f"entry {entry.region} impl {entry.impl.id!r}: expected exactly one owning op, " f"got {owners}"
        )
    return owners[0]


def expected_installed_keys(contract: ExecutionContract, registry: Registry) -> frozenset:
    """The (op, site) keys the ResolvedFingerprint must record for this contract."""
    return frozenset((_primary_op(registry, e), case) for e in contract.composition for case in e.cases)


def validate_contract_against_installed(contract: ExecutionContract, registry: Registry, fingerprint) -> None:
    """Reject any installed (op, site) the contract does not name, and any contract-named key that
    is not installed."""
    if isinstance(fingerprint, ResolvedFingerprint):
        installed = fingerprint.keys()
    elif isinstance(fingerprint, Registry):
        installed = fingerprint.installed_keys()
    else:
        installed = frozenset(fingerprint)
    named = expected_installed_keys(contract, registry)
    unnamed = installed - named
    missing = named - installed
    if unnamed or missing:
        raise ContractDeliveryError(
            "contract does not match the installed composition:\n"
            f"  installed but UNNAMED by the contract: {sorted(unnamed)}\n"
            f"  named by the contract but NOT installed: {sorted(missing)}\n"
            "Every installed (op, site) must carry an explicit entry; absence means "
            "'no such site', never 'default'."
        )
