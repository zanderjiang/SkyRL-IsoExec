"""SkyRL-IsoExec: bitwise-identical token logprobs between the vLLM rollout engine and the
Megatron trainer, so the importance ratio is exactly 1.

The numerical work is two kernel patches on the Megatron side (RoPE -> fp32, RMSNorm -> the vLLM
C++ kernel) plus batch invariance on both engines. Wire it by calling ``apply_vllm_isoexec_env()``
in the vLLM worker before engine creation and ``apply_megatron_isoexec_patches()`` in the Megatron
worker after model build; supported models are Qwen3 dense, Megatron MoE, and hybrid GDN (Qwen3.5).

Importing this package runs the installers below at module top, before anything can import
``megatron.bridge``; that ordering is load-bearing. The public re-exports are lazy (PEP 562) so that
importing a leaf op does not drag in megatron or vllm.
"""

# Install the no-TransformerEngine import guard FIRST -- before anything that may import
# megatron.bridge, whose eager model-zoo import has unguarded `import transformer_engine`. No-op
# unless real TE is absent.
from .runtimes.megatron.no_te_guard import install_no_te_guard

install_no_te_guard()

# Bind this process to its own GPU before any installer below can touch CUDA: in a vLLM worker
# subprocess the current device is still 0 at this point, and the fla shim imports code that probes
# the device in its module body, leaving a phantom CUDA context on GPU 0. No-op outside a vLLM worker.
from .runtimes.vllm.worker_device_pin import pin_worker_cuda_device  # noqa: E402

pin_worker_cuda_device()

# GatedDeltaNet (Qwen3.5 hybrid): register the `fla` facade BEFORE anything imports
# megatron.core.ssm.gated_delta_net, which binds `chunk_gated_delta_rule` at import time. No-op
# unless SKYRL_ISOEXEC_GDN=1.
from .runtimes.megatron.gdn_fla_shim import install_fla_shim  # noqa: E402

install_fla_shim()

# Autofuse contract-handshake pin, registered at PACKAGE import because every process that computes
# the handshake hash must carry it -- including the engine actor, which never wires sites. The pin
# is a pure function of (flag, fusion-ledger digest), so a split-brain flag delivery or a diverged
# ledger refuses at weight sync. The default literal is spelled out to match
# fusion_ledger.autofuse_enabled's reading of "unset".
import os as _os  # noqa: E402

if _os.environ.get("SKYRL_ISOEXEC_AUTOFUSE", "1") == "1":
    from .autofuse.sites import (
        autofuse_pin_digest as _autofuse_pin_digest,  # noqa: E402
    )
    from .core.process_contract import (
        register_contract_extension as _register_ext,  # noqa: E402
    )

    _register_ext("autofuse", _autofuse_pin_digest)

# Install-attestation digest, folded into the handshake composite for every process (core/enforce).
# Both sides of a correct pair digest to the literal CLEAN, so the composite stays symmetric by
# construction; a side that declared identically but installed differently carries its INSTALL-phase
# violations in the digest and the pair refuses at weight sync -- the failure class hashing alone
# can never see (both sides equally wrong is still caught when only one side's install deviates).
from .core.enforce import (
    install_attestation_digest as _install_attestation_digest,  # noqa: E402
)
from .core.process_contract import (
    register_contract_extension as _register_attest_ext,  # noqa: E402
)

_register_attest_ext("install_attestation", _install_attestation_digest)

# Lazy public API: name -> (submodule, attr). Importing the submodule is deferred to first access,
# so `import isoexec.ops.*` does not pull in megatron/vllm-heavy adapters.
_LAZY = {
    # runtimes.megatron.megatron_patches (imports megatron)
    "apply_megatron_isoexec_patches": (".runtimes.megatron.megatron_patches", "apply_megatron_isoexec_patches"),
    "revert_megatron_isoexec_patches": (".runtimes.megatron.megatron_patches", "revert_megatron_isoexec_patches"),
    "enable_megatron_batch_invariant": (".runtimes.megatron.megatron_patches", "enable_megatron_batch_invariant"),
    "apply_rope_fp32_patch": (".runtimes.megatron.megatron_patches", "apply_rope_fp32_patch"),
    "apply_vops_rmsnorm_patch": (".runtimes.megatron.megatron_patches", "apply_vops_rmsnorm_patch"),
    "scoring_mode": (".runtimes.megatron.megatron_patches", "scoring_mode"),
    "isoexec_patch_status": (".runtimes.megatron.megatron_patches", "isoexec_patch_status"),
    # ops.moe.moe_batch_invariant (imports megatron.core specs)
    "enable_moe_deterministic_ops": (".ops.moe.moe_batch_invariant", "enable_moe_deterministic_ops"),
    "force_isoexec_moe_config": (".ops.moe.moe_batch_invariant", "force_isoexec_moe_config"),
    "make_isoexec_local_layer_spec": (".ops.moe.moe_batch_invariant", "make_isoexec_local_layer_spec"),
    "prepare_isoexec_moe": (".ops.moe.moe_batch_invariant", "prepare_isoexec_moe"),
    "provider_is_moe": (".ops.moe.moe_batch_invariant", "provider_is_moe"),
    "revert_moe_deterministic_ops": (".ops.moe.moe_batch_invariant", "revert_moe_deterministic_ops"),
    # ops.collectives.pik_tp_invariant
    "apply_pik_tp_invariant": (".ops.collectives.pik_tp_invariant", "apply_pik_tp_invariant"),
    "revert_pik_tp_invariant": (".ops.collectives.pik_tp_invariant", "revert_pik_tp_invariant"),
    "pik_enabled": (".ops.collectives.pik_tp_invariant", "pik_enabled"),
    "pik_status": (".ops.collectives.pik_tp_invariant", "pik_status"),
    "pik_transport_status": (".ops.collectives.pik_tp_invariant", "pik_transport_status"),
    "pik_get_plan": (".ops.collectives.pik_tp_invariant", "get_plan"),
    # runtimes.vllm.vllm_patches (imports vllm)
    "ISOEXEC_VLLM_ENV": (".runtimes.vllm.vllm_patches", "ISOEXEC_VLLM_ENV"),
    "apply_vllm_isoexec_env": (".runtimes.vllm.vllm_patches", "apply_vllm_isoexec_env"),
    "isoexec_engine_arg_overrides": (".runtimes.vllm.vllm_patches", "isoexec_engine_arg_overrides"),
    "isoexec_sampling_constraints": (".runtimes.vllm.vllm_patches", "isoexec_sampling_constraints"),
}


def __getattr__(name):  # PEP 562 module-level lazy attribute
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    mod = importlib.import_module(target[0], __name__)
    val = getattr(mod, target[1])
    globals()[name] = val  # cache so subsequent access is a plain attribute lookup
    return val


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY.keys()))


__all__ = ["install_fla_shim", "install_no_te_guard", *_LAZY.keys()]
