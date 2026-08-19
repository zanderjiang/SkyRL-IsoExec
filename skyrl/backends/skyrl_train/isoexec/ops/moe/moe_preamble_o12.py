"""Engine MoE preamble kernels and the shared+routed owner handoff.

The O12 path fuses the shared-expert SwiGLU and gate-scale elementwise chains while preserving their bf16
rounding boundaries. Installation is instance-scoped to engine modules so a colocated trainer keeps its
autograd-owned forwards; unsupported dtype, layout, activation, gate, or collective conditions delegate to
the captured Megatron forward.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

from ...core.triton_nonftz import div_rn as _nonftz_div_rn
from ...core.triton_nonftz import sigmoid as _nonftz_sigmoid

_ENV = "SKYRL_ISOEXEC_MOE_PREAMBLE_O12"

# Pinned tiles, never autotuned.
_GLU_BLOCK_T = 32
# BLOCK_F=64 because the production shared FFN is 512/TP8 = 64 wide; a wider tile masks off half of
# every tile and costs more than eager. A model with a wider shared expert should re-tune this -- the
# kernel is pure elementwise, so it is bitwise-safe -- but the value must stay a PINNED per-process
# constant, never derived from a token count.
_GLU_BLOCK_F = 64
_GLU_WARPS = 4
_GATE_BLOCK_T = 16
_GATE_BLOCK_H = 256
_GATE_WARPS = 4


def preamble_o12_enabled() -> bool:
    """Flag read, default OFF."""
    return os.environ.get(_ENV, "0") == "1"


if HAVE_TRITON:

    @triton.jit
    def _glu_kernel(
        x_ptr,  # [T, 2F] bf16, contiguous
        out_ptr,  # [T, F] bf16, contiguous
        T,
        F,
        OFFSET,  # fp32 scalar, config.glu_linear_offset
        BLOCK_T: tl.constexpr,
        BLOCK_F: tl.constexpr,
        # GATE-ONLY TOGGLES; production always runs (0, 0). They exist so the FTZ controls run through
        # THIS kernel rather than a synthetic stand-in. DIV_FTZ=1 selects the flushing libdevice divide;
        # DIV_FORM=1 selects the SIGMOID-shaped numerator, which is the form that can actually go
        # subnormal.
        DIV_FTZ: tl.constexpr = False,
        DIV_FORM: tl.constexpr = 0,
    ):
        pid_t = tl.program_id(0).to(tl.int64)
        pid_f = tl.program_id(1).to(tl.int64)
        rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)[:, None]
        cols = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)[None, :]
        mask = (rows < T) & (cols < F)

        two_f = 2 * F
        g = tl.load(x_ptr + rows * two_f + cols, mask=mask, other=0.0).to(tl.float32)
        lin = tl.load(x_ptr + rows * two_f + F + cols, mask=mask, other=0.0).to(tl.float32)

        # 1. F.silu on bf16: fp32 x / (1 + exp(-x)), then round. `g * -1.0`, never `-g`.
        num = g if DIV_FORM == 0 else 1.0
        den = 1.0 + libdevice.exp(g * -1.0)
        if DIV_FTZ:
            s = libdevice.div_rn(num, den).to(tl.bfloat16)
        else:
            s = _nonftz_div_rn(num, den).to(tl.bfloat16)
        # 2. x_linear + offset: fp32 add, then round. NOT a no-op at 0.0: (-0.0)+0.0 == +0.0.
        lb = (lin + OFFSET).to(tl.bfloat16)
        # 3. the product: both bf16 operands promote to fp32, multiply, round.
        h = (s.to(tl.float32) * lb.to(tl.float32)).to(tl.bfloat16)

        tl.store(out_ptr + rows * F + cols, h, mask=mask)


def fused_shared_glu(
    x: torch.Tensor, offset: float, *, _div_ftz: bool = False, _div_form: int = 0, _fp_fusion: bool = False
) -> torch.Tensor:
    """``silu(x[..., :F]) * (x[..., F:] + offset)`` in one launch, bitwise-equal to the eager chain.

    ``x`` is ``linear_fc1``'s contiguous ``[..., 2F]`` bf16 output. The underscored keywords are
    GATE-ONLY controls (see ``_glu_kernel``); production never passes them.
    """
    assert x.is_contiguous(), "the chunk halves are read by stride; a non-contiguous fc1 output must fall back"
    assert x.dtype == torch.bfloat16
    lead = x.shape[:-1]
    two_f = x.shape[-1]
    assert two_f % 2 == 0
    f = two_f // 2
    x2 = x.view(-1, two_f)
    t = x2.shape[0]
    out = torch.empty((t, f), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(t, _GLU_BLOCK_T), triton.cdiv(f, _GLU_BLOCK_F))
    _glu_kernel[grid](
        x2,
        out,
        t,
        f,
        float(offset),
        BLOCK_T=_GLU_BLOCK_T,
        BLOCK_F=_GLU_BLOCK_F,
        DIV_FTZ=bool(_div_ftz),
        DIV_FORM=int(_div_form),
        num_warps=_GLU_WARPS,
        enable_fp_fusion=bool(_fp_fusion),
    )
    return out.view(*lead, f)


if HAVE_TRITON:

    @triton.jit
    def _gate_kernel(
        out_ptr,  # [T, H] bf16 -- the shared expert output
        logit_ptr,  # [T] bf16 -- the pre-sigmoid gate logit, one per token
        dst_ptr,  # [T, H] bf16
        T,
        H,
        BLOCK_T: tl.constexpr,
        BLOCK_H: tl.constexpr,
        # GATE-ONLY. DIV_FTZ=1 swaps in the flushing libdevice divide; this site's numerator IS 1.0, so
        # it is genuinely exposed and the control must show a flush.
        DIV_FTZ: tl.constexpr = False,
    ):
        pid_t = tl.program_id(0).to(tl.int64)
        pid_h = tl.program_id(1).to(tl.int64)
        rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)[:, None]
        cols = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)[None, :]
        rmask = rows < T
        mask = rmask & (cols < H)

        # ATen's bf16 sigmoid: promote, 1/(1+exp(-x)) -- the SIGMOID shape, so the quotient can be
        # subnormal and the non-FTZ divide is mandatory here (unlike O12-A's silu shape).
        lg = tl.load(logit_ptr + rows, mask=rmask, other=0.0).to(tl.float32)
        if DIV_FTZ:
            gs = libdevice.div_rn(1.0, 1.0 + libdevice.exp(lg * -1.0)).to(tl.bfloat16)
        else:
            gs = _nonftz_sigmoid(lg).to(tl.bfloat16)  # eager rounds BEFORE the multiply

        o = tl.load(out_ptr + rows * H + cols, mask=mask, other=0.0).to(tl.float32)
        r = (o * gs.to(tl.float32)).to(tl.bfloat16)
        tl.store(dst_ptr + rows * H + cols, r, mask=mask)


def fused_gate_scale(
    output: torch.Tensor, logits: torch.Tensor, *, _div_ftz: bool = False, _fp_fusion: bool = False
) -> torch.Tensor:
    """``output * torch.sigmoid(logits)`` in one launch, bitwise-equal to the eager pair.

    ``output`` is ``[..., H]`` bf16 contiguous; ``logits`` is ``[..., 1]`` bf16 with the same
    leading shape (``F.linear(hidden, gate_weight)``).
    """
    assert output.is_contiguous() and output.dtype == torch.bfloat16
    assert logits.dtype == torch.bfloat16
    lead = output.shape[:-1]
    h = output.shape[-1]
    o2 = output.view(-1, h)
    lg = logits.reshape(-1)
    assert lg.numel() == o2.shape[0] and lg.is_contiguous()
    t = o2.shape[0]
    dst = torch.empty_like(o2)
    grid = (triton.cdiv(t, _GATE_BLOCK_T), triton.cdiv(h, _GATE_BLOCK_H))
    _gate_kernel[grid](
        o2,
        lg,
        dst,
        t,
        h,
        BLOCK_T=_GATE_BLOCK_T,
        BLOCK_H=_GATE_BLOCK_H,
        DIV_FTZ=bool(_div_ftz),
        num_warps=_GATE_WARPS,
        enable_fp_fusion=bool(_fp_fusion),
    )
    return dst.view(*lead, h)


def shared_expert_supported(mod) -> bool:
    """Can this ``SharedExpertMLP`` instance take the fused path? Anything else falls back."""
    import torch.nn.functional as F

    cfg = mod.config
    return bool(
        HAVE_TRITON
        and getattr(cfg, "gated_linear_unit", False)
        and getattr(mod, "activation_func", None) is F.silu
        and getattr(cfg, "activation_func_clamp_value", None) is None
        and not getattr(cfg, "use_te_activation_func", False)
        and not getattr(cfg, "bias_activation_fusion", False)
        and not getattr(cfg, "add_bias_linear", False)
        and not getattr(cfg, "moe_shared_expert_overlap", False)
        and not getattr(cfg, "fp8", None)
        and not getattr(cfg, "fp4", None)
        and not getattr(mod, "shared_experts_recompute", False)
    )


def _fused_shared_expert_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Replaces ``SharedExpertMLP.forward`` (which is ``MLP.forward`` + the gate) on ENGINE
    instances only. Falls back to the captured original for any shape it was not gated for."""
    inter, bias_parallel = self.linear_fc1(hidden_states)
    if bias_parallel is not None or not inter.is_contiguous() or inter.dtype != torch.bfloat16:
        return self._ix_o12_orig_forward(hidden_states)

    h = fused_shared_glu(inter, self.config.glu_linear_offset)
    output, _ = self.linear_fc2(h)

    if self.use_shared_expert_gate:
        import torch.nn.functional as F

        logits = F.linear(hidden_states, self.gate_weight)
        if output.is_contiguous() and output.dtype == torch.bfloat16 and logits.dtype == torch.bfloat16:
            output = fused_gate_scale(output, logits)
        else:  # pragma: no cover
            output = output * torch.nn.functional.sigmoid(logits)
    return output


def _shared_owner_expert_forward(self, hidden_states: torch.Tensor):
    """Engine-instance seam that hands shared FC2 leaves to the exact owner composition.

    Returning ``None`` is intentional: ``MoELayer.shared_experts_compute`` forwards it to
    ``postprocess``, which therefore performs no separate final add.  The dispatcher has already
    consumed the payload and included the exact shared tree/gate/add in its owner output.  Any
    unsupported local condition calls the captured original and returns a normal shared tensor, so
    the unmodified postprocess remains the fail-closed path.
    """
    inter, bias_parallel = self.linear_fc1(hidden_states)
    if bias_parallel is not None or not inter.is_contiguous() or inter.dtype != torch.bfloat16:
        return self._ix_shared_owner_orig_forward(hidden_states)

    if preamble_o12_enabled():
        h = fused_shared_glu(inter, self.config.glu_linear_offset)
    else:
        x_glu, x_linear = torch.chunk(inter, 2, dim=-1)
        h = self.config.activation_func(x_glu) * (x_linear + self.config.glu_linear_offset)

    from .moe_pik_combine_owner import shared_fc2_subtree_partial

    got = shared_fc2_subtree_partial(self.linear_fc2, h)
    if got is None:
        return self._ix_shared_owner_orig_forward(hidden_states)
    partial, shared_bf16 = got

    if not self.use_shared_expert_gate:
        return self._ix_shared_owner_orig_forward(hidden_states)
    import torch.nn.functional as F

    logits = F.linear(hidden_states, self.gate_weight)
    gate_score = torch.sigmoid(logits)
    if gate_score.dtype != torch.bfloat16 or not gate_score.is_contiguous():
        return self._ix_shared_owner_orig_forward(hidden_states)

    T = hidden_states.numel() // hidden_states.shape[-1]
    dispatcher = self._ix_shared_owner_dispatcher
    if getattr(dispatcher, "_isoexec_shared_owner_payload", None) is not None:
        raise RuntimeError(
            "shared-owner payload was not consumed before the next SharedExpertMLP forward; "
            "refusing to overwrite live FC2 leaves"
        )
    dispatcher._isoexec_shared_owner_payload = (
        partial.reshape(T, -1),
        gate_score.reshape(T),
        shared_bf16,
    )
    return None


def install_engine_moe_preamble(gpt_modules) -> int:
    """Rebind the ENGINE GPTModel's supported shared-expert forwards.

    INSTANCE-level, so the trainer -- which constructs the identical classes and, under
    ``VLLM_ENABLE_V1_MULTIPROCESSING=0``, shares this process -- is untouched by construction.
    Returns the number of instances swapped, 0 if the flag is off.
    """
    on = preamble_o12_enabled()
    try:
        from megatron.core.transformer.moe.moe_layer import MoELayer
        from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
    except Exception:  # pragma: no cover
        return 0

    from .moe_batch_invariant import _nogather_active
    from .moe_pik_combine_owner import (
        shared_owner_fusion_enabled,
        shared_owner_group_enabled,
    )

    shared_owner_requested = shared_owner_fusion_enabled()
    n_shared_owner = 0
    # Install the paired seam from the owning MoELayer, where shared expert and dispatcher are both
    # explicit instance references.  Never class-patch SharedExpertMLP: colocated trainer modules
    # use the same class and a forward-only equality gate cannot detect a severed backward.
    for layer in gpt_modules.modules():
        if not isinstance(layer, MoELayer):
            continue
        shared = getattr(layer, "shared_experts", None)
        dispatcher = getattr(layer, "token_dispatcher", None)
        if (
            shared is None
            or dispatcher is None
            or not _nogather_active(dispatcher)
            or not shared_expert_supported(shared)
            or hasattr(shared, "_ix_shared_owner_orig_forward")
        ):
            continue
        # Every eligible rank enters this vote even when its local env says OFF.  Only the agreed
        # result may change a launch structure; a one-sided shared-expert rebind would deadlock.
        if not shared_owner_group_enabled(getattr(dispatcher, "tp_ep_group", None), shared.gate_weight.device):
            continue
        shared._ix_shared_owner_orig_forward = shared.forward
        shared._ix_shared_owner_dispatcher = dispatcher
        dispatcher._isoexec_shared_owner_payload = None
        shared.forward = _shared_owner_expert_forward.__get__(shared, type(shared))
        n_shared_owner += 1

    n_se = 0
    skipped_se = 0
    for m in gpt_modules.modules():
        # The `hasattr` guards make this idempotent. Without them a second call would capture the
        # ALREADY-FUSED bound method as the "original", and every fallback would recurse forever.
        if isinstance(m, SharedExpertMLP):
            if hasattr(m, "_ix_shared_owner_orig_forward"):
                # The shared-owner wrapper performs O12-A itself when O12 is on; stacking the old
                # wrapper over it would run linear_fc2's full reduction and defeat the fusion.
                skipped_se += 1
            elif on and not hasattr(m, "_ix_o12_orig_forward") and shared_expert_supported(m):
                m._ix_o12_orig_forward = m.forward
                m.forward = _fused_shared_expert_forward.__get__(m, type(m))
                n_se += 1
            else:
                skipped_se += 1
    # Always print the resolved setting so a launch can verify that the actor received it.
    print(
        f"[ISOEXEC-MOE] O12 preamble fused on {n_se} shared expert(s) "
        f"(skipped {skipped_se}); {_ENV}="
        f"{'ON' if on else 'OFF'}. "
        f"Shared+routed owner instance seam={'ON on ' + str(n_shared_owner) + ' layer(s)' if n_shared_owner else 'OFF'} "
        f"(local request={'ON' if shared_owner_requested else 'OFF'}; group-agreed), "
        "(SKYRL_ISOEXEC_MOE_SHARED_OWNER_FUSION, default OFF; engagement requires its ADMITTED + served lines). "
        "Router gating keeps the existing batch-invariant GEMM.",
        flush=True,
    )
    return n_se
