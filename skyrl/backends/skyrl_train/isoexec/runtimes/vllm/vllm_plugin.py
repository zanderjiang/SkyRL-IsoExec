"""vLLM general plugin that installs the IsoExec engine stack in every vLLM process.

vLLM puts each TP rank in its own process, so model registration, the attention backend, the sampler
patches and the GDN pins have to be installed per worker rather than only in the engine actor. vLLM
may load plugins several times per process, so everything here is idempotent, and the whole stack is
gated on ``SKYRL_ISOEXEC=1``.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_installed = False


def _install_shared_owner_memory_profile_scope(worker_cls=None) -> None:
    """Scope shared-owner pool provisioning around vLLM's ``profile_run``.

    Persistent candidate pools are provisioned inside ``profile_run`` so they count against the KV
    budget, but the scope ends before ``profile_cudagraph_memory`` so graph capture is sized for the
    candidate rather than for a temporary exact reference graph. ``worker_cls`` is an injection seam
    for tests; production resolves vLLM's concrete GPU worker.
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

    # MUST be first, and MUST be here rather than in setup_envvars_for_vllm: vLLM's
    # init_batch_invariance re-pins NCCL_MIN/MAX_NCHANNELS=1 inside each worker before that worker's
    # communicator exists, and a patch installed in the engine actor never runs in the worker
    # subprocesses. Default OFF; gated on SKYRL_ISOEXEC_ENGINE_NCCL_UNPIN=1 (SKYRL_ISOEXEC_NCCL_PIN is
    # the trainer's flag and does not gate this).
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
        # `attention_backend="CUSTOM"` is resolved per worker, so the registration must exist per
        # worker.
        from ...ops.attention import varlen_backend

        varlen_backend.register_varlen_custom_backend()
        # vLLM's fused-Triton sampler logprob kernel never calls aten log_softmax, and the sampler
        # runs in the worker.
        from .vllm_patches import (
            patch_vllm_logprobs_batch_invariant,
            patch_vllm_sampler_temperature,
        )

        patch_vllm_logprobs_batch_invariant()
        # The V1 sampler's unconditional full-vocab temperature divide is the identity at 1.0.
        patch_vllm_sampler_temperature()

    if os.environ.get("SKYRL_ISOEXEC_MOE_DETERMINISTIC") == "1":
        # SM90: vLLM's batch-invariant mode leaves cuBLAS M-variant. Same override the trainer uses.
        from ...ops.moe.moe_batch_invariant import _install_moe_matmul_invariance

        _install_moe_matmul_invariance()

        # Re-tile that same kernel for the skinny decode GEMMs (shared-expert fc1, the
        # shared_expert_gate linear, the fp32 router). Output tiling only -- BLOCK_SIZE_K pinned, no
        # split-K -- so it is bitwise-neutral. MUST come after the install above: both re-register
        # the same two aten ops and the last registration wins. Default OFF.
        from ...ops.mm.mm_tiles import install_mm_tiles

        install_mm_tiles()

    if os.environ.get("SKYRL_ISOEXEC_GDN") == "1":
        # The `fla` facade must exist before megatron.core.ssm.gated_delta_net is imported. The
        # isoexec package __init__ already does this on import; called again in case of an unusual
        # import order.
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
        # glm_sleep_skip (2026-08-09): THE sleep lever. Skip the D2H backup of the byte ranges the
        # Skip the D2H backup of the byte ranges the next weight sync provably overwrites before
        # any read; back up only the residual non-synced pool state. Fail-to-stock: see the module
        # docstring.
        from .sleep_skip_backup import install_sleep_skip_weights_backup

        install_sleep_skip_weights_backup()

    # Import-time inventory check of the vLLM surface this adapter patches and vendors. WARN by
    # default -- logs each problem and continues; raises only under SKYRL_ISOEXEC_COMPAT_STRICT=1.
    # Wrapped so a compat-check bug can never take down a run.
    try:
        from .compat import check_vllm_compat

        problems = check_vllm_compat()
        if not problems:
            logger.info("[ISOEXEC-COMPAT] vLLM surface OK")
        else:
            print(f"[ISOEXEC-COMPAT] {len(problems)} vLLM surface problem(s); see logs", flush=True)
    except Exception as e:  # pragma: no cover - never let the compat check crash a run
        logger.warning("[ISOEXEC-COMPAT] compat check skipped (%s)", e)

    # Bitwise auto-fusion sites. Runs in every plugin process; the TP workers are where the megatron
    # modules live, so that is where sites actually wire. Inert unless SKYRL_ISOEXEC_AUTOFUSE=1
    # (default 0); decisions are consumed from the fusion ledger, never made here. Wrapped: a wiring
    # failure demotes to eager per site and must never take down the plugin.
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
