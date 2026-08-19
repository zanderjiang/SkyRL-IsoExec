"""Zero-centered-gamma RMSNorm for the local spec, which has no TransformerEngine.

Models whose checkpoint stores gamma centred on zero normalise with ``rms(x) * (1 + w)``. Megatron
expresses that as ``config.layernorm_zero_centered_gamma`` and implements it only in ``TENorm``;
without TransformerEngine the stack falls back to ``WrappedTorchNorm``, which asserts the flag off
and so fails while building the first transformer layer.

The replacement delegates normalisation to ``F.rms_norm`` with no weight -- the same aten op the
validated dense path runs, so it inherits that op's batch invariance -- and applies ``(1 + w)``
afterwards, which is elementwise and invariant by construction. Both runtimes install and call it.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_installed = False


class ZeroCenteredTorchRMSNorm(torch.nn.Module):
    """``y = rms_norm(x) * (1 + weight)``. ``weight`` initialised to 0, as the checkpoint expects."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, params_dtype=None):
        super().__init__()
        self.hidden_size = (hidden_size,)
        self.eps = eps
        self.weight = torch.nn.Parameter(
            torch.zeros(hidden_size, dtype=params_dtype, device=torch.cuda.current_device())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.hidden_size, None, self.eps) * (1.0 + self.weight)

    def extra_repr(self) -> str:
        return f"{self.hidden_size[0]}, eps={self.eps}, zero_centered_gamma=True"


def install_zero_centered_torch_norm() -> bool:
    """Teach Megatron's ``WrappedTorchNorm`` about ``layernorm_zero_centered_gamma``. Idempotent.

    Must run before any transformer layer is built, i.e. alongside the other isoexec import-time
    hooks. Models that do not set the flag get Megatron's original behaviour untouched.
    """
    global _installed

    if _installed:
        return True
    try:
        from megatron.core.transformer import torch_norm
    except Exception as e:  # pragma: no cover - megatron absent
        logger.info("[isoexec] megatron.core.transformer.torch_norm unavailable (%s)", e)
        return False

    orig_new = torch_norm.WrappedTorchNorm.__new__

    def _new(
        cls,
        config,
        hidden_size,
        eps=1e-5,
        persist_layer_norm=False,
        zero_centered_gamma=False,
        normalization="LayerNorm",
    ):
        if not getattr(config, "layernorm_zero_centered_gamma", False):
            return orig_new(
                cls,
                config,
                hidden_size,
                eps,
                persist_layer_norm,
                zero_centered_gamma,
                normalization,
            )
        if config.normalization != "RMSNorm":
            raise NotImplementedError(
                f"[isoexec] zero-centered gamma is only implemented for RMSNorm, got {config.normalization}"
            )
        # Per-token RMSNorm is layout-independent, so megatron's sequence-parallel refusal is only
        # about grad bookkeeping: marking the weight `sequence_parallel` makes finalize_model_grads
        # sum its per-slice grads across TP. The same gate lives in sp_norm_lift.py for the plain
        # path, since either patch may install first and this branch never delegates to the other.
        sp = bool(getattr(config, "sequence_parallel", False))
        if sp:
            from .sp_norm_lift import trainer_sp_lift_enabled

            assert trainer_sp_lift_enabled(), "sequence parallel not supported by torch LayerNorm"
        norm = ZeroCenteredTorchRMSNorm(hidden_size, eps, params_dtype=config.params_dtype)
        if sp:
            setattr(norm.weight, "sequence_parallel", True)
        return norm

    torch_norm.WrappedTorchNorm.__new__ = _new
    _installed = True
    print(
        "[ISOEXEC] WrappedTorchNorm now supports layernorm_zero_centered_gamma "
        "(rms_norm(x) * (1 + w)) -- required by Qwen3.5 / Qwen3-Next",
        flush=True,
    )
    return True
