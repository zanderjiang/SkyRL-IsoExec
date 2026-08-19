"""The vLLM wrapper class, chosen by capability tuple rather than by an env-var branch.

vLLM resolves a model's ``_ModelInfo`` (``is_hybrid``, ``supports_mrope``, ``supports_pp``, ...) from
the CLASS before any instance exists, and caches it keyed by module+class name -- so one class can
never answer ``is_hybrid`` differently on different runs, and there must be one class per capability
tuple. The shipped tuples keep their exact existing class names: those names are written into
``hf_overrides={"architectures": [...]}`` and imported by launchers and harnesses, and renaming one
would invalidate every cached ``_ModelInfo``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# One value every process (driver, EngineCore, each Ray worker) reads to resolve the SAME profile,
# and therefore the same class name -- rather than several per-capability flags that can disagree.
MODEL_PATH_ENV = "SKYRL_ISOEXEC_MODEL_PATH"

_MODULE = "skyrl.backends.skyrl_train.isoexec.runtimes.vllm.gptmodel_vllm"


@dataclass(frozen=True)
class VLLMCapabilities:
    """The protocol answers vLLM reads off the class; the key for choosing a wrapper class."""

    is_hybrid: bool
    supports_mrope: bool


# Shipped tuples keep the class names they already have -- do not rename (see module docstring).
_CANONICAL_NAMES = {
    VLLMCapabilities(is_hybrid=True, supports_mrope=True): "MegatronGPTModelHybridForCausalLM",
}

# Base class each tuple's wrapper derives from, by name (resolved lazily -- importing
# gptmodel_vllm pulls in megatron and vLLM).
_CANONICAL_BASES = {
    "MegatronGPTModelHybridForCausalLM": "GPTModelVLLMHybridWrapper",
}


def capabilities_for_profile(profile) -> VLLMCapabilities:
    """Derive the tuple from a ``ModelProfile``: the only place architecture becomes protocol."""
    return VLLMCapabilities(is_hybrid=True, supports_mrope=True)


def name_for(caps: VLLMCapabilities) -> str:
    """The registered vLLM architecture name for a tuple: canonical if shipped, else generated.

    The generated form is a pure function of the tuple. A name that varied run-to-run would let one
    run's cached ``_ModelInfo`` decide another run's protocol set.
    """
    if caps in _CANONICAL_NAMES:
        return _CANONICAL_NAMES[caps]
    return f"MegatronGPTModel_h{int(caps.is_hybrid)}r{int(caps.supports_mrope)}ForCausalLM"


def capabilities_from_env() -> VLLMCapabilities:
    """Fallback tuple for a process that has no readable model path."""
    return VLLMCapabilities(is_hybrid=True, supports_mrope=True)


def resolve_capabilities(model_path: str | None = None) -> VLLMCapabilities:
    """The tuple for this process: profile first, env fallback second.

    A profile/env disagreement is logged at ERROR rather than raised -- this runs during worker
    bring-up, where a raise turns a diagnosable mismatch into an opaque spawn failure; the manifest
    hash handshake is what refuses the run.
    """
    path = model_path or os.environ.get(MODEL_PATH_ENV)
    if not path:
        return capabilities_from_env()
    try:
        from ...models.resolve import resolve_profile

        caps = capabilities_for_profile(resolve_profile(path))
    except Exception as e:
        logger.warning("[isoexec-vllm] no profile for %r (%s); falling back to legacy flags", path, e)
        return capabilities_from_env()

    legacy = capabilities_from_env()
    if legacy != caps:
        logger.error(
            "[isoexec-vllm] CAPABILITY DISAGREEMENT for %r: profile says %s, legacy flags say %s. "
            "The launcher's flags and the model's manifest describe different models; the manifest "
            "hash handshake will refuse the run. Fix the launcher flags or the profile.",
            path,
            caps,
            legacy,
        )
    return caps


def resolve_model_name(model_path: str | None = None) -> str:
    """The vLLM architecture name this process should register / override to."""
    return name_for(resolve_capabilities(model_path))


def import_path_for(caps: VLLMCapabilities) -> str:
    """vLLM's string registration form (``module:ClassName``), which is what survives across worker
    processes -- each worker lazily imports the class rather than unpickling it."""
    name = name_for(caps)
    base = _CANONICAL_BASES.get(name)
    if base:
        return f"{_MODULE}:{base}"
    return f"{__name__}:{name}"


def synthesize(caps: VLLMCapabilities):
    """Build (and cache in this module's namespace) the wrapper class for a non-canonical tuple.

    The class is installed into ``globals()`` under its generated name so that ``module:ClassName``
    string registration resolves it from a freshly spawned worker; a class living only in a closure
    is invisible to every process but the one that made it.
    """
    name = name_for(caps)
    existing = globals().get(name)
    if existing is not None:
        return existing

    from .gptmodel_vllm import GPTModelVLLMWrapper

    attrs = {
        "supports_mrope": caps.supports_mrope,
        "__doc__": (
            f"Generated IsoExec vLLM wrapper for capabilities {caps}. One class per capability "
            f"tuple (never per model): vLLM caches _ModelInfo per class name, so the protocol "
            f"answers must be class-level constants."
        ),
    }
    if caps.is_hybrid:
        attrs["is_hybrid"] = True

        @classmethod
        def _copy_func(cls):
            from vllm.model_executor.layers.mamba.mamba_utils import (
                MambaStateCopyFuncCalculator,
            )

            return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

        attrs["get_mamba_state_copy_func"] = _copy_func

    cls = type(name, (GPTModelVLLMWrapper,), attrs)
    cls.__module__ = __name__
    cls.__qualname__ = name
    globals()[name] = cls
    logger.info("[isoexec-vllm] synthesized wrapper class %s for %s", name, caps)
    return cls


def needs_hybrid_config_pass(caps: VLLMCapabilities) -> bool:
    """Whether vLLM's ``HybridAttentionMambaModelConfig`` pass must be re-mapped onto our
    architecture name -- vLLM selects that pass by architecture name, and ``hf_overrides`` has just
    replaced the name with ours."""
    return caps.is_hybrid
