"""Lift the no-TE norm's sequence-parallel refusal, where it is safe to.

Megatron's ``WrappedTorchNorm`` asserts ``not config.sequence_parallel``. That is a capability
statement about TE's fused norms, not an arithmetic one: RMSNorm and LayerNorm are per-token, so
under SP every token's full hidden vector still lives on one rank and no within-token reduction is
split.

What torch norms genuinely lack under SP is the gradient marking. Each TP rank computes norm-weight
grads from its sequence slice only, and megatron sums grads across TP only for params carrying
``param.sequence_parallel``. So the lift has two inseparable halves: construct the torch norm anyway
(bypassing the assert via a config proxy), and mark its weight and bias exactly as
``FusedLayerNorm.__init__`` does. The GatedDeltaNet ``out_norm`` is the one exemption and is
un-marked again by :func:`install_gdn_out_norm_sp_exemption` -- read that docstring first.

Trainer-only by construction: gated on ``SKYRL_ISOEXEC_TRAINER_SP=1`` and ``config.sequence_parallel``,
and the engine builds with sequence parallelism off. Composes with the zero-centered norm patch in
either install order; both wrap ``WrappedTorchNorm.__new__`` and delegate to whatever they captured.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_installed = False


def trainer_sp_lift_enabled() -> bool:
    """Same flag that gates the pik reduce-scatter path and the megatron_worker force-off."""
    return os.environ.get("SKYRL_ISOEXEC_TRAINER_SP", "0") == "1"


class _SPMaskedConfig:
    """Read-only view of a TransformerConfig whose ``sequence_parallel`` reads False.

    Only used to carry the original config past the ``WrappedTorchNorm`` assert; the constructed
    module keeps no reference to it.
    """

    __slots__ = ("_cfg",)

    def __init__(self, cfg):
        object.__setattr__(self, "_cfg", cfg)

    def __getattr__(self, name):
        if name == "sequence_parallel":
            return False
        return getattr(object.__getattribute__(self, "_cfg"), name)


def install_trainer_sp_norm_lift() -> bool:
    """Teach ``WrappedTorchNorm`` to build under sequence parallelism, with grads marked.

    Must run before any transformer layer is built. No-op unless the flag is set. Idempotent.
    """
    global _installed

    if _installed:
        return True
    if not trainer_sp_lift_enabled():
        return False
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
        if not getattr(config, "sequence_parallel", False):
            return orig_new(
                cls,
                config,
                hidden_size,
                eps,
                persist_layer_norm,
                zero_centered_gamma,
                normalization,
            )
        # Build the norm as if SP were off (per-token math is layout-independent), then mark its
        # params so finalize_model_grads sums their grads across TP -- each rank saw only its
        # sequence slice, so the sum is the full-batch grad. The forward bits are untouched.
        mod = orig_new(
            cls,
            _SPMaskedConfig(config),
            hidden_size,
            eps,
            persist_layer_norm,
            zero_centered_gamma,
            normalization,
        )
        for pname in ("weight", "bias"):
            p = getattr(mod, pname, None)
            if p is not None:
                setattr(p, "sequence_parallel", True)
        return mod

    torch_norm.WrappedTorchNorm.__new__ = _new
    _installed = True
    print(
        "[ISOEXEC-TRAINER] WrappedTorchNorm SP refusal LIFTED (SKYRL_ISOEXEC_TRAINER_SP=1): "
        "per-token torch RMSNorm/LayerNorm built under sequence_parallel, weights marked "
        "`sequence_parallel` so finalize_model_grads sums their grads across TP",
        flush=True,
    )
    install_gdn_out_norm_sp_exemption()
    return True


_gdn_exempt_installed = False


def install_gdn_out_norm_sp_exemption() -> bool:
    """Un-mark the GatedDeltaNet ``out_norm`` -- the one norm the blanket marking gets wrong.

    The marking rule assumes the input is the sequence shard. The GDN layer un-shards its own input:
    ``in_proj`` is a sequence-parallel ``ColumnParallelLinear``, so it all-gathers the sequence and
    everything downstream inside ``GatedDeltaNet.forward``, ``out_norm`` included, runs on the full
    sequence. The sequence re-shards only at the row-parallel ``out_proj``.

    So ``out_norm.weight`` already sees under SP exactly what it sees with SP off: all tokens, this
    rank's own value heads. Marking it ``sequence_parallel`` would sum different head-groups'
    gradients into a replicated parameter, making SP=1 a different update than SP=0.

    Installed from :func:`install_trainer_sp_norm_lift`, so gated on the same flag. Idempotent, and
    a no-op when megatron has no GDN.
    """
    global _gdn_exempt_installed

    if _gdn_exempt_installed:
        return True
    if not trainer_sp_lift_enabled():
        return False
    try:
        from megatron.core.ssm.gated_delta_net import GatedDeltaNet
    except Exception as e:  # pragma: no cover - no GDN in this megatron
        logger.info(
            "[isoexec] GatedDeltaNet unavailable, out_norm SP exemption not needed (%s)",
            e,
        )
        return False
    if getattr(GatedDeltaNet, "_isoexec_out_norm_sp_exempt", False):
        _gdn_exempt_installed = True
        return True

    orig_init = GatedDeltaNet.__init__

    def _init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        out_norm = getattr(self, "out_norm", None)
        if out_norm is None or not hasattr(out_norm, "parameters"):
            return
        for p in out_norm.parameters(recurse=True):
            if getattr(p, "sequence_parallel", False):
                p.sequence_parallel = False
                if not getattr(GatedDeltaNet, "_isoexec_out_norm_logged", False):
                    GatedDeltaNet._isoexec_out_norm_logged = True
                    print(
                        "[ISOEXEC-TRAINER] GDN out_norm EXEMPTED from the SP grad marking: in_proj "
                        "all-gathers the sequence, so out_norm already sees every token (over this "
                        "rank's own value heads). Summing its grad across TP would make SP=1 a "
                        "different update than SP=0.",
                        flush=True,
                    )

    GatedDeltaNet.__init__ = _init
    GatedDeltaNet._isoexec_out_norm_sp_exempt = True
    _gdn_exempt_installed = True
    return True
