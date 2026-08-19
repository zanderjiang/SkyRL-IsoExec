"""Megatron-side monkey patches that make the trainer's forward bitwise-identical to the vLLM engine.

Only two ops diverge between Megatron's and vLLM's batch-invariant kernels -- RoPE (Megatron's
unfused rotate-half multiply-add runs in bf16 where vLLM's kernel runs fp32) and RMSNorm (a different
reduction, 1 ULP apart); GEMM and log_softmax already match once ``batch_invariant_mode`` is enabled.
Nothing in the installed megatron-core / vLLM packages is edited: every patch here is a monkey patch,
idempotent and reversible. Extension points for MoE / hybrid / MLA are marked ``# EXTEND:``.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_RMSNORM_ORIG_FORWARD = None  # type: ignore[var-annotated]
_ROPE_ORIG_BSHD = None  # type: ignore[var-annotated]
_APPLY_NORM_ORIG = None  # type: ignore[var-annotated]
_BIK_ENABLED = False
_VOPS_RMSNORM_INSTALLED = False
_ROPE_FP32_INSTALLED = False
_TE_FUSED_NORM_INSTALLED = False
_TE_GRAD_GEMM_ROUTE_INSTALLED = False
_TE_BIK_GENERAL_GEMM = None  # type: ignore[var-annotated]
_TE_GEMM_ROUTE_COUNTS = {"forward_bik": 0, "gradient_te": 0}
_FLASH_ENV_SET = False
_SCORING_ACTIVE = False  # set True around the parity/scoring forward (see scoring_mode())


# (1) RoPE: force the rotate-half multiply-add to fp32 (matches vLLM's CUDA kernel).
def _patched_apply_rotary_pos_emb_bshd(
    t: torch.Tensor,
    freqs: torch.Tensor,
    rotary_interleaved: bool = False,
    mla_rotary_interleaved: bool = False,
    mscale: float = 1.0,
    multi_latent_attention: Optional[bool] = None,
) -> torch.Tensor:
    """fp32 drop-in for megatron rope_utils._apply_rotary_pos_emb_bshd.

    Identical to upstream EXCEPT the cos/sin multiply-add is done in fp32 with a single
    final round back to the input dtype. cos/sin are still rounded to the input dtype
    first, mirroring vLLM's bf16 cos/sin cache. The ``thd`` path routes through this
    function upstream, so patching this single global covers both bshd and thd.
    """
    from megatron.core.models.common.embeddings.rope_utils import _rotate_half

    if multi_latent_attention is not None:  # preserve upstream deprecation shim
        mla_rotary_interleaved = multi_latent_attention

    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    if mla_rotary_interleaved:
        x1 = t[..., 0::2]
        x2 = t[..., 1::2]
        t = torch.cat((x1, x2), dim=-1)

    in_dtype = t.dtype
    # Round cos/sin to the input dtype (vLLM stores a bf16 cos/sin cache), then upcast
    # to fp32 so the rotate-half multiply-add accumulates in fp32 like vLLM's kernel.
    cos_ = (torch.cos(freqs) * mscale).to(in_dtype).to(torch.float32)
    sin_ = (torch.sin(freqs) * mscale).to(in_dtype).to(torch.float32)
    t_fp32 = t.to(torch.float32)
    out = (t_fp32 * cos_) + (_rotate_half(t_fp32, rotary_interleaved) * sin_)
    t = out.to(in_dtype)
    return torch.cat((t, t_pass), dim=-1)


def apply_rope_fp32_patch() -> None:
    """Patch Megatron's unfused RoPE to use fp32 arithmetic. Autograd-safe; harmless to
    leave installed (only changes bf16 behavior; fp32 inputs are unaffected)."""
    global _ROPE_ORIG_BSHD, _ROPE_FP32_INSTALLED
    if _ROPE_FP32_INSTALLED:
        return
    from megatron.core.models.common.embeddings import rope_utils

    # NOTE: requires apply_rope_fusion=False (the SkyRL RL path already sets this). The
    # fused TE kernel path (fused_apply_rotary_pos_emb*) is not intercepted here.
    _ROPE_ORIG_BSHD = rope_utils._apply_rotary_pos_emb_bshd
    rope_utils._apply_rotary_pos_emb_bshd = _patched_apply_rotary_pos_emb_bshd
    _ROPE_FP32_INSTALLED = True
    logger.info("[isoexec] installed fp32 RoPE patch (bshd+thd)")


def is_rope_fp32_installed() -> bool:
    """True iff :func:`apply_rope_fp32_patch` is currently installed in this process.

    Public accessor so the engine adapter can pass this state down into the ops-layer
    ``install_engine_fused_rope`` guard rather than the op importing this runtime module. The
    fused-rope kernel matches stock bf16 rope, not this fp32 patch, so it must refuse to install over
    the patched global.
    """
    return _ROPE_FP32_INSTALLED


def revert_rope_fp32_patch() -> None:
    global _ROPE_ORIG_BSHD, _ROPE_FP32_INSTALLED
    if not _ROPE_FP32_INSTALLED:
        return
    from megatron.core.models.common.embeddings import rope_utils

    rope_utils._apply_rotary_pos_emb_bshd = _ROPE_ORIG_BSHD
    _ROPE_FP32_INSTALLED = False


# (2) RMSNorm: route the BIK norm forward through vLLM's C++ kernel (bitwise match), keeping the
#     original fp32 backward so training gradients are unchanged.
#
# Megatron norms are all applied "no-residual" (the residual add happens separately in the layer), so
# each site must emit whatever kernel vLLM uses there:
#   * main decoder norms (input/pre-mlp/final, over hidden_size): vLLM's C++ fused_add_rms_norm,
#     reproduced exactly by calling it with a ZERO residual on the already-added stream r (r+0==r in
#     bf16, same kernel).
#   * q_norm/k_norm (over head_dim): vLLM's no-residual Triton kernel.
# EXTEND: head-vs-hidden is detected by weight size (<= _HEAD_NORM_MAX); set it for unusual head dims.
_HEAD_NORM_MAX = 1024


def _patched_rmsnorm_forward(ctx, x, weight, eps, zero_centered_gamma):
    """Forward replacement for BatchInvariantRMSNormFn.forward.

    Output bits come from vLLM's own kernels (so trainer norm == rollout norm). rsigma is
    still computed in fp32 and saved for the *unchanged* upstream backward, so this is safe
    to leave installed during the training fwd/bwd as well.
    """
    from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik

    # Fall back to the original implementation for cases the vLLM kernels can't represent
    # (zero-centered gamma, non-CUDA, unsupported dtype).
    if zero_centered_gamma or (not x.is_cuda) or x.dtype not in (torch.bfloat16, torch.float16):
        return _RMSNORM_ORIG_FORWARD(ctx, x, weight, eps, zero_centered_gamma)

    import vllm._custom_ops as vops

    x_fp32 = x.float()
    ms = bik.mean_dim(x_fp32 * x_fp32, dim=-1, keepdim=True)
    rsigma = torch.rsqrt(ms + eps)  # for backward only

    shp = x.shape
    x2 = x.reshape(-1, shp[-1]).contiguous()
    w = weight.contiguous()
    if weight.numel() <= _HEAD_NORM_MAX:
        # q_norm / k_norm: vLLM uses the no-residual Triton batch-invariant kernel.
        from vllm.model_executor.layers.batch_invariant import (
            rms_norm as _vllm_triton_rms,
        )

        out = _vllm_triton_rms(x2, w, eps=float(eps)).reshape(shp)
    else:
        # main norms: reproduce vLLM's fused_add_rms_norm exactly via a zero residual.
        out2 = x2.clone()
        zero = torch.zeros_like(out2)
        vops.fused_add_rms_norm(out2, zero, w, float(eps))
        out = out2.reshape(shp)

    ctx.eps = eps
    ctx.zero_centered_gamma = zero_centered_gamma
    ctx.rsigma = rsigma
    ctx.save_for_backward(x, weight, rsigma)
    return out


def apply_vops_rmsnorm_patch() -> None:
    """Override BatchInvariantRMSNormFn.forward to emit vLLM C++ norm bits.

    Both Megatron's functional norm wrapper and its TE RMSNorm.forward patch route through
    BatchInvariantRMSNormFn.apply, so this single override covers all sites. Requires
    batch_invariant_mode to have installed the TE RMSNorm patches first (see
    enable_megatron_batch_invariant)."""
    global _RMSNORM_ORIG_FORWARD, _VOPS_RMSNORM_INSTALLED
    if _VOPS_RMSNORM_INSTALLED:
        return
    from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik

    # Accessing a staticmethod via the class yields the plain underlying function.
    _RMSNORM_ORIG_FORWARD = bik.BatchInvariantRMSNormFn.forward
    bik.BatchInvariantRMSNormFn.forward = staticmethod(_patched_rmsnorm_forward)
    _VOPS_RMSNORM_INSTALLED = True
    logger.info("[isoexec] routed BatchInvariantRMSNormFn.forward -> vLLM vops.rms_norm")


def revert_vops_rmsnorm_patch() -> None:
    global _VOPS_RMSNORM_INSTALLED
    if not _VOPS_RMSNORM_INSTALLED:
        return
    from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik

    bik.BatchInvariantRMSNormFn.forward = staticmethod(_RMSNORM_ORIG_FORWARD)
    _VOPS_RMSNORM_INSTALLED = False


# (2b) TE *fused* LayerNormLinear / LayerNormMLP main norms. These compute their RMSNorm via
# `transformer_engine.pytorch.module._common.apply_normalization`, which is NOT the standalone-RMSNorm
# path Megatron's BIK patches, so the residual-stream main norms (input_layernorm, pre_mlp_layernorm)
# bypass interception entirely. This routes the fused main norm through vLLM's C++ kernel for the
# scoring/parity forward only; training keeps native TE for speed and backward correctness.
def _patched_apply_normalization(
    inputmat,
    ln_out,
    ln_weight,
    ln_bias,
    eps,
    output_quantizer,
    output_dtype,
    normalization,
    fwd_ln_sm_margin,
    zero_centered_gamma,
):
    import torch as _t

    # Activate for the trainer's scoring forward (scoring_mode, runs grad-enabled inside
    # forward_backward_func) AND for any pure-inference forward (no_grad) -- which is how the
    # vLLM generator runs GPTModel. Training fwd/bwd (grad enabled, no scoring_mode) stays on TE.
    use_vops = (
        (_SCORING_ACTIVE or not _t.is_grad_enabled())
        and normalization == "RMSNorm"
        and not zero_centered_gamma
        and ln_bias is None
        and output_quantizer is None  # not fp8
        and inputmat.is_cuda
        and inputmat.dtype in (_t.bfloat16, _t.float16)
    )
    if not use_vops:
        return _APPLY_NORM_ORIG(
            inputmat,
            ln_out,
            ln_weight,
            ln_bias,
            eps,
            output_quantizer,
            output_dtype,
            normalization,
            fwd_ln_sm_margin,
            zero_centered_gamma,
        )
    import vllm._custom_ops as vops

    shp = inputmat.shape
    x2 = inputmat.reshape(-1, shp[-1]).contiguous()
    w = ln_weight.contiguous()
    out2 = x2.clone()
    zero = _t.zeros_like(out2)
    vops.fused_add_rms_norm(out2, zero, w, float(eps))  # bitwise == vLLM main norm
    out = out2.reshape(shp).to(ln_out.dtype if hasattr(ln_out, "dtype") else inputmat.dtype)
    xf = inputmat.float()
    rsigma = _t.rsqrt((xf * xf).mean(-1, keepdim=True) + eps)  # unused under no_grad
    return out, None, rsigma


def apply_te_fused_norm_patch() -> None:
    """Intercept the TE fused-layer norm (apply_normalization) for the scoring forward.

    NOTE: `layernorm_linear` / `layernorm_mlp` do `from ._common import apply_normalization`
    at import time, binding a *local* name — so we must patch those module references, not
    just `_common.apply_normalization`."""
    global _APPLY_NORM_ORIG, _TE_FUSED_NORM_INSTALLED
    if _TE_FUSED_NORM_INSTALLED:
        return
    try:
        import transformer_engine.pytorch.module._common as _cm
    except Exception as e:  # TE not installed
        logger.warning("[isoexec] TE not available, skipping fused-norm patch: %s", e)
        return
    _APPLY_NORM_ORIG = _cm.apply_normalization
    _cm.apply_normalization = _patched_apply_normalization
    patched_mods = ["_common"]
    for modname in ("layernorm_linear", "layernorm_mlp"):
        try:
            mod = __import__(f"transformer_engine.pytorch.module.{modname}", fromlist=[modname])
            if getattr(mod, "apply_normalization", None) is _APPLY_NORM_ORIG:
                mod.apply_normalization = _patched_apply_normalization
                patched_mods.append(modname)
        except Exception as e:
            logger.warning("[isoexec] could not patch %s.apply_normalization: %s", modname, e)
    _TE_FUSED_NORM_INSTALLED = True
    logger.info("[isoexec] patched TE apply_normalization in %s (fused main norms -> vLLM)", patched_mods)


@contextlib.contextmanager
def scoring_mode():
    """Activate vLLM-bitwise fused-norm interception for the enclosed parity/scoring forward.

    Megatron's forward_backward_func(forward_only=True) runs with grad enabled, so we can't
    key off torch.is_grad_enabled(); wrap the scoring forward in this context instead. Leaves
    the training fwd/bwd on native TE fused norms (fast, correct backward)."""
    global _SCORING_ACTIVE
    prev = _SCORING_ACTIVE
    _SCORING_ACTIVE = True
    try:
        yield
    finally:
        _SCORING_ACTIVE = prev


def force_flash_attention_env() -> None:
    """Force TE to use the flash_attn backend (bitwise == vLLM flash) over cuDNN fused attn.
    Must be set before TE selects its attention backend (first attention forward)."""
    global _FLASH_ENV_SET
    import os

    os.environ["NVTE_FLASH_ATTN"] = "1"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    _FLASH_ENV_SET = True
    logger.info("[isoexec] set NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0 (TE flash == vLLM flash)")


# (3) Enable Megatron batch-invariant mode (GEMM + log_softmax + TE patches).
def enable_megatron_batch_invariant(skip_aten_registration: bool = False) -> None:
    """Enable Megatron's batch-invariant kernels (already bitwise-identical to vLLM for
    GEMM + log_softmax). Also installs the TE gemm/rmsnorm patches that our vops override
    then rides on.

    skip_aten_registration: when GPTModel runs INSIDE vLLM (VLLM_BATCH_INVARIANT=1), vLLM has
    already registered the aten mm/addmm/_log_softmax/mean overrides; re-registering them errors
    ("already a kernel registered ..."). In that case we only set the mode flag + apply the TE
    monkey-patches (TE GEMM/RMSNorm are NOT aten ops, so vLLM doesn't cover them)."""
    global _BIK_ENABLED
    if _BIK_ENABLED:
        return
    from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik

    if skip_aten_registration:
        bik._batch_invariant_MODE = True  # so is_batch_invariant_mode_enabled() == True
        bik._te_patch_for_batch_invariant()  # TE general_gemm + RMSNorm monkey-patches only
        logger.info("[isoexec] enabled Megatron batch-invariant (TE patches only; aten via vLLM)")
    else:
        bik.enable_batch_invariant_mode()
        logger.info("[isoexec] enabled Megatron batch_invariant_mode (GEMM+log_softmax+TE)")
    _BIK_ENABLED = True


def _te_gemm_is_gradient(args, kwargs) -> bool:
    """Read TE 2.17's ``grad`` routing bit from either supported call form."""
    if "grad" in kwargs:
        return bool(kwargs["grad"])
    # Pinned TE 2.17.1 ABI: grad is the fourteenth positional argument.  The install-time
    # signature check below fails closed if that ABI moves.
    return len(args) > 13 and bool(args[13])


def _report_te_gemm_route(route: str) -> None:
    count = _TE_GEMM_ROUTE_COUNTS[route]
    if count == 1 or count & (count - 1) == 0:
        print(f"[ISOEXEC-TE-GEMM] served_{route}={count}", flush=True)


def _te_general_gemm_forward_bik_backward_te(*args, **kwargs):
    """Keep the gate-critical forward on BIK and return gradient GEMMs to native TE.

    TE's ``grad`` flag describes a dgrad/wgrad GEMM; it is not PyTorch grad mode.  Both the
    original checkpoint forward and its recompute therefore take the exact same ``grad=False``
    BIK owner.  Only backward-only GEMMs take the captured native TE implementation.
    """
    from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik

    if _te_gemm_is_gradient(args, kwargs):
        _TE_GEMM_ROUTE_COUNTS["gradient_te"] += 1
        _report_te_gemm_route("gradient_te")
        return bik._TE_GENERAL_GEMM_ORIG(*args, **kwargs)
    _TE_GEMM_ROUTE_COUNTS["forward_bik"] += 1
    _report_te_gemm_route("forward_bik")
    return _TE_BIK_GENERAL_GEMM(*args, **kwargs)


def apply_te_backward_gemm_route() -> None:
    """Route only TE gradient GEMMs to native TE after MCore installs its global BIK patch."""
    global _TE_BIK_GENERAL_GEMM, _TE_GRAD_GEMM_ROUTE_INSTALLED
    if _TE_GRAD_GEMM_ROUTE_INSTALLED:
        return

    import inspect

    from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik

    native = getattr(bik, "_TE_GENERAL_GEMM_ORIG", None)
    current = getattr(bik, "_te_general_gemm_patched", None)
    if native is None or current is None:
        raise RuntimeError("[ISOEXEC-TE-GEMM] BIK/native TE general_gemm owners were not captured")
    parameters = list(inspect.signature(native).parameters)
    if len(parameters) <= 13 or parameters[13] != "grad":
        raise RuntimeError(
            "[ISOEXEC-TE-GEMM] unsupported TE general_gemm ABI: expected positional grad at 13, " f"found {parameters}"
        )

    _TE_BIK_GENERAL_GEMM = current
    route = _te_general_gemm_forward_bik_backward_te
    import megatron.core.extensions.transformer_engine as meg_te
    import transformer_engine.pytorch.cpp_extensions as te_cpp
    import transformer_engine.pytorch.module.layernorm_linear as te_layernorm_linear
    import transformer_engine.pytorch.module.linear as te_linear

    te_cpp.general_gemm = route
    te_linear.general_gemm = route
    te_layernorm_linear.general_gemm = route
    meg_te.general_gemm = route
    _TE_GRAD_GEMM_ROUTE_INSTALLED = True
    print(
        "[ISOEXEC-TE-GEMM] installed forward=BIK gradient=TE-native (served counts pending)",
        flush=True,
    )


# Orchestrator
def apply_megatron_isoexec_patches(enable_bik: bool = True, skip_aten_registration: bool = False) -> None:
    """Install all Megatron-side IsoExec patches. Idempotent.

    Order matters: enable batch invariance first (installs the TE RMSNorm patch), then
    override the RMSNorm forward to vops, then patch RoPE.
    """
    force_flash_attention_env()  # TE flash == vLLM flash (set before first attention fwd)
    if enable_bik:
        enable_megatron_batch_invariant(skip_aten_registration=skip_aten_registration)
        # Keep the established production arm unchanged. Selective TE explicitly admits this
        # backward-only owner split and validates/counters it as part of that composition.
        import os

        if os.environ.get("SKYRL_ISOEXEC_SELECTIVE_TE", "0") == "1":
            apply_te_backward_gemm_route()
    apply_vops_rmsnorm_patch()  # standalone TENorm (q/k + final)
    apply_te_fused_norm_patch()  # fused LayerNormLinear/MLP main norms (scoring forward)
    apply_rope_fp32_patch()
    # EXTEND: for MoE add batch-invariant grouped-GEMM / router-softmax parity here.
    # EXTEND: for MLA patch the interleaved RoPE path + mla_rotary_interleaved handling.
    _check_megatron_compat_at_startup()


def _check_megatron_compat_at_startup() -> None:
    """Verify the rebound megatron/bridge/TE surface at adapter startup (see compat.py).

    Gated on SKYRL_ISOEXEC=1 and fully wrapped: a broken or absent compat check must never crash the
    trainer. WARN-only unless SKYRL_ISOEXEC_COMPAT_STRICT=1 (honoured inside check_megatron_compat)."""
    import os

    if os.environ.get("SKYRL_ISOEXEC") != "1":
        return
    try:
        from .compat import check_megatron_compat

        check_megatron_compat()  # logs "[ISOEXEC-COMPAT] megatron surface OK (N)" or the problems
    except Exception as e:  # a strict-mode RuntimeError is intentional; anything else is swallowed
        if os.environ.get("SKYRL_ISOEXEC_COMPAT_STRICT") == "1":
            raise
        logger.warning("[ISOEXEC-COMPAT] megatron compat check skipped (%s)", e)


def revert_megatron_isoexec_patches() -> None:
    revert_rope_fp32_patch()
    revert_vops_rmsnorm_patch()
    # batch_invariant_mode left enabled intentionally (cheap, and used by training too).


def isoexec_patch_status() -> dict:
    return {
        "batch_invariant_enabled": _BIK_ENABLED,
        "vops_rmsnorm_installed": _VOPS_RMSNORM_INSTALLED,
        "te_fused_norm_installed": _TE_FUSED_NORM_INSTALLED,
        "te_grad_gemm_route_installed": _TE_GRAD_GEMM_ROUTE_INSTALLED,
        "te_gemm_route_counts": dict(_TE_GEMM_ROUTE_COUNTS),
        "rope_fp32_installed": _ROPE_FP32_INSTALLED,
        "flash_env_set": _FLASH_ENV_SET,
    }
