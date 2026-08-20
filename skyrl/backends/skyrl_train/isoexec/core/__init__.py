"""Construction-time scaffolding for the isoexec op abstraction.

Registers ops and impls (``registry``), builds the per-(op, site) selections into the hashed
ExecutionContract (``contract_build``) delivered by ``contract_delivery``, and carries the
supporting arch target, flag table, test contracts and gate-signature table.
"""

from __future__ import annotations

from .arch import (
    ARCH,
    ArchMismatchError,
    HardwareTarget,
    assert_homogeneous,
    detect_arch,
)
from .contract_build import (
    DEPLOYMENT as ENTRY_DEPLOYMENT,
)
from .contract_build import (
    FUNCTION as ENTRY_FUNCTION,
)
from .contract_build import (
    ContractBuildError,
    PinValidationError,
    build_execution_contract,
    validate_pins,
)
from .contract_delivery import (
    CONTRACT_HASH_ENV,
    ContractDeliveryError,
    expected_installed_keys,
    load_contract,
    validate_contract_against_installed,
    write_contract_file,
)
from .contracts import (
    ConnectivityError,
    GateResult,
    VacuousTestError,
    assert_grad_count,
    assert_grad_set_equality,
    assert_hazard_exercised,
    bitwise_equal,
    check_hazard_coverage,
    collect_grad_set,
)
from .fingerprint import ResolvedFingerprint
from .flags import (
    FLAGS,
    Flag,
    actor_forwarding_list,
    by_disposition,
    latent_split_brain,
)
from .registry import (
    HAZARDS,
    PER_MODEL,
    ImplSpec,
    OneOf,
    OpSpec,
    Registry,
    RegistryError,
    RoundingSchedule,
    StateInvalidation,
)
from .signatures import (
    LEGACY_HASH,
    SIGNATURES,
    SignatureError,
    SignatureRecord,
    SignatureTable,
)

__all__ = [
    # arch
    "ARCH",
    "ArchMismatchError",
    "HardwareTarget",
    "assert_homogeneous",
    "detect_arch",
    # contract build
    "ResolvedFingerprint",
    "ContractBuildError",
    "PinValidationError",
    "ENTRY_FUNCTION",
    "ENTRY_DEPLOYMENT",
    "build_execution_contract",
    "validate_pins",
    # contract delivery
    "CONTRACT_HASH_ENV",
    "ContractDeliveryError",
    "expected_installed_keys",
    "load_contract",
    "validate_contract_against_installed",
    "write_contract_file",
    # registry
    "Registry",
    "OpSpec",
    "ImplSpec",
    "RoundingSchedule",
    "PER_MODEL",
    "OneOf",
    "StateInvalidation",
    "RegistryError",
    "HAZARDS",
    # flags
    "FLAGS",
    "Flag",
    "actor_forwarding_list",
    "latent_split_brain",
    "by_disposition",
    # contracts
    "bitwise_equal",
    "assert_hazard_exercised",
    "check_hazard_coverage",
    "assert_grad_set_equality",
    "assert_grad_count",
    "collect_grad_set",
    "GateResult",
    "VacuousTestError",
    "ConnectivityError",
    # signatures
    "SIGNATURES",
    "SignatureTable",
    "SignatureRecord",
    "SignatureError",
    "LEGACY_HASH",
]
