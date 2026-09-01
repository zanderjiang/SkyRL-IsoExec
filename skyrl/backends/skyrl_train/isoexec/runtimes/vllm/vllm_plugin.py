"""vLLM general plugin that installs the IsoExec engine stack in every vLLM process.

Each TP rank is its own process, so everything must be installed per worker. vLLM may load plugins
several times per process, so every install here is idempotent and gated on ``SKYRL_ISOEXEC=1``.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_installed = False


def _install_shared_owner_memory_profile_scope(worker_cls=None) -> None:
    """Scope shared-owner pool provisioning around vLLM's ``profile_run``.

    Pools are provisioned inside ``profile_run`` so they count against the KV budget, but the scope
    ends before ``profile_cudagraph_memory`` so graph capture is sized for the candidate.
    """
    if worker_cls is None:
        from vllm.v1.worker.gpu_worker import Worker as worker_cls

    original = worker_cls.determine_available_memory
    if getattr(original, "_isoexec_shared_owner_profile_scope", False):
        return

    def scoped(self, *args, **kwargs):
        from ...ops.moe.moe_pik_combine_owner import (
            shared_owner_memory_profile_scope,
        )

        if getattr(getattr(self, "cache_config", None), "kv_cache_memory_bytes", None):
            raise RuntimeError(
                "SKYRL_ISOEXEC_MOE_SHARED_OWNER_FUSION requires vLLM's measured KV sizing; "
                "explicit kv_cache_memory_bytes bypasses persistent-pool accounting, so the "
                "memory-safe admission contract refuses this configuration"
            )
        runner = self.model_runner
        original_profile_run = runner.profile_run

        def profile_run_scoped(*profile_args, **profile_kwargs):
            with shared_owner_memory_profile_scope():
                return original_profile_run(*profile_args, **profile_kwargs)

        runner.profile_run = profile_run_scoped
        try:
            return original(self, *args, **kwargs)
        finally:
            runner.profile_run = original_profile_run

    scoped._isoexec_shared_owner_profile_scope = True
    scoped._isoexec_shared_owner_profile_original = original
    worker_cls.determine_available_memory = scoped
    logger.info(
        "[ISOEXEC-MOE] installed shared-owner vLLM memory-profile scope on %s",
        worker_cls.__name__,
    )


def register() -> None:
    """Entry point. Idempotent; normal non-IsoExec jobs are untouched."""
    global _installed

    if _installed:
        return
    _isoexec = os.environ.get("SKYRL_ISOEXEC") == "1"
    if not _isoexec:
        return
    _installed = True

    # MUST be first and MUST be here: vLLM re-pins NCCL_MIN/MAX_NCHANNELS=1 inside each worker before
    # its communicator exists. Default OFF; gated on SKYRL_ISOEXEC_ENGINE_NCCL_UNPIN=1.
    from .vllm_patches import neutralize_vllm_nccl_channel_pin

    neutralize_vllm_nccl_channel_pin()

    if os.environ.get("SKYRL_ISOEXEC_MOE_SHARED_OWNER_FUSION", "0") not in (
        "",
        "0",
        "false",
        "no",
        "off",
    ):
        _install_shared_owner_memory_profile_scope()

    from .gptmodel_vllm import VLLM_MODEL_NAME, register_gptmodel_to_vllm

    register_gptmodel_to_vllm()

    if os.environ.get("SKYRL_ISOEXEC_LOCAL_SPEC") == "1":
        # `attention_backend="CUSTOM"` is resolved per worker, so register per worker.
        from ...ops.attention import varlen_backend

        varlen_backend.register_varlen_custom_backend()
        from .vllm_patches import (
            patch_vllm_sampler_logprobs_rowinv,
            patch_vllm_sampler_temperature,
        )

        # Sampler.compute_logprobs is the V1 runner's actual producer of both sampled and prompt
        # logprobs.
        patch_vllm_sampler_logprobs_rowinv()
        # The V1 sampler's unconditional full-vocab temperature divide is the identity at 1.0.
        patch_vllm_sampler_temperature()

    if os.environ.get("SKYRL_ISOEXEC_MOE_DETERMINISTIC") == "1":
        # SM90: vLLM's batch-invariant mode leaves cuBLAS M-variant. Same override the trainer uses.
        from ...ops.moe.moe_batch_invariant import _install_moe_matmul_invariance

        _install_moe_matmul_invariance()

        # Re-tile that same kernel for the skinny decode GEMMs. Output tiling only (BLOCK_SIZE_K
        # pinned, no split-K), so bitwise-neutral. MUST come after the install above: both
        # re-register the same two aten ops and the last registration wins.
        from ...ops.mm.mm_tiles import install_mm_tiles

        install_mm_tiles()

    if os.environ.get("SKYRL_ISOEXEC_GDN") == "1":
        # The `fla` facade must exist before megatron.core.ssm.gated_delta_net is imported; the
        # package __init__ already does this, repeated here to survive unusual import orders.
        from ...ops.gdn.gdn_batch_invariant import (
            pin_fla_autotune_configs,
            pin_gdn_rmsnorm_rows_per_block,
        )
        from ..megatron.gdn_fla_shim import install_fla_shim
        from .gdn_engine_patch import lift_gdn_batch_invariance_veto

        install_fla_shim()
        pin_fla_autotune_configs()
        pin_gdn_rmsnorm_rows_per_block()
        lift_gdn_batch_invariance_veto()

    if os.environ.get("SKYRL_ISOEXEC_SLEEP_SKIP_WEIGHTS_BACKUP") == "1":
        # Skip the D2H backup of byte ranges the next weight sync overwrites before any read; back
        # up only the residual non-synced pool state.
        from .sleep_skip_backup import install_sleep_skip_weights_backup

        install_sleep_skip_weights_backup()

    # Inventory check of the vLLM surface this adapter patches. Warns by default; raises only under
    # SKYRL_ISOEXEC_COMPAT_STRICT=1.
    try:
        from .compat import check_vllm_compat

        problems = check_vllm_compat()
        if not problems:
            logger.info("[ISOEXEC-COMPAT] vLLM surface OK")
        else:
            print(f"[ISOEXEC-COMPAT] {len(problems)} vLLM surface problem(s); see logs", flush=True)
    except Exception as e:  # pragma: no cover - never let the compat check crash a run
        logger.warning("[ISOEXEC-COMPAT] compat check skipped (%s)", e)

    # Bitwise auto-fusion sites; inert unless SKYRL_ISOEXEC_AUTOFUSE=1. Decisions come from the
    # fusion ledger, never from here; a wiring failure demotes to eager per site.
    try:
        from ...autofuse.sites import (
            install_autofuse_sites,
            selected_autofuse_requires_exact_install,
        )

        install_autofuse_sites("engine")
    except Exception as e:  # pragma: no cover - fail-to-eager, never fail the engine
        if "selected_autofuse_requires_exact_install" in locals() and selected_autofuse_requires_exact_install():
            raise RuntimeError("selected AUTOFUSE ledger has admitted artifacts but engine installation failed") from e
        logger.error("[ISOEXEC-AUTOFUSE] engine install skipped on error: %s", e)

    print(
        f"[ISOEXEC-PLUGIN] installed in pid {os.getpid()} (arch={VLLM_MODEL_NAME}, "
        f"gdn={os.environ.get('SKYRL_ISOEXEC_GDN') == '1'})",
        flush=True,
    )
