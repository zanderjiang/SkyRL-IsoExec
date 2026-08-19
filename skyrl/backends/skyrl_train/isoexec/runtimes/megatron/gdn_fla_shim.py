"""Make Megatron's GatedDeltaNet execute the engine's GDN ops, by faking the ``fla`` package.

``megatron.core.ssm.gated_delta_net`` imports ``chunk_gated_delta_rule``, ``l2norm`` and
``causal_conv1d`` from ``flash-linear-attention`` and raises without them, and its
``deterministic_mode`` fallback cannot run the packed (thd) path. This registers a facade ``fla``
package in ``sys.modules`` forwarding those three symbols to ``isoexec.ops.gdn``, so the trainer runs
literally the same functions as the rollout engine. Two extra rebinds serve the same goal: the
``jit_fuser``-compiled ``_compute_g_and_beta`` / ``_prepare_qkv_for_gated_delta_rule`` become eager
(a compiled ``exp``/``softplus`` is libdevice's, which disagrees with ATen's in the last ulp), and
``causal_conv1d`` slices on ``cu_seqlens`` so it never convolves across a packed-sequence boundary.

Megatron binds these symbols at import time, so :func:`install_fla_shim` MUST run before anything
imports megatron.bridge / megatron.core. It is gated on ``SKYRL_ISOEXEC_GDN=1`` and is idempotent.
"""

from __future__ import annotations

import importlib.machinery
import logging
import os
import sys
import types

logger = logging.getLogger(__name__)

_installed = False


def gdn_enabled() -> bool:
    return os.environ.get("SKYRL_ISOEXEC_GDN") == "1"


def _eager_prep_enabled() -> bool:
    return os.environ.get("SKYRL_ISOEXEC_GDN_EAGER_PREP", "1") == "1"


def _scoring_fused_outnorm_enabled() -> bool:
    """Default-OFF: use the byte-exact fused GDN output norm in eval/no-grad scoring only."""
    return os.environ.get("SKYRL_ISOEXEC_GDN_SCORING_FUSED_OUTNORM", "0") == "1"


_SCORING_FUSED_OUTNORM_COUNTS = {"served": 0}
_SCORING_FUSED_OUTNORM_MAX_WIDTH = 16384  # fused_outnorm's single-CTA/quack-equivalent limit


def _scoring_gate_view(gate):
    """Return the zero-copy ``[tokens, heads, width]`` gate view accepted by fused outnorm.

    The GDN gate is a last-dimension slice of ``in_proj``. Its head/width axes are dense but its
    token stride includes the other projected fields, so it is intentionally not contiguous.
    ``Tensor.view`` is used instead of ``reshape``: an unsupported leading layout returns ``None``
    rather than silently allocating the very copy this path exists to remove.
    """
    if gate.ndim < 3:
        return None
    heads, width = gate.shape[-2:]
    if gate.stride(-1) != 1 or gate.stride(-2) != width:
        return None
    try:
        return gate.view(-1, heads, width)
    except RuntimeError:
        return None


def _scoring_fused_outnorm_static_contract(self, gate) -> bool:
    """Admission facts known both during QKV prep and at the outnorm call site."""
    import torch

    if not (
        _scoring_fused_outnorm_enabled()
        and not self.training
        and not torch.is_grad_enabled()
        and gate.device.type == "cuda"
        and gate.dtype == torch.bfloat16
        and gate.ndim >= 3
        and getattr(self, "activation", None) in ("silu", "swish")
    ):
        return False
    from ...ops.norms.zero_centered_norm import ZeroCenteredTorchRMSNorm

    out_norm = getattr(self, "out_norm", None)
    if not isinstance(out_norm, ZeroCenteredTorchRMSNorm):
        return False
    weight = getattr(out_norm, "weight", None)
    width = gate.shape[-1]
    return bool(
        gate.numel() > 0
        and 0 < width <= _SCORING_FUSED_OUTNORM_MAX_WIDTH
        and weight is not None
        and tuple(weight.shape) == (width,)
        and weight.device == gate.device
        and weight.dtype == gate.dtype
        and weight.is_contiguous()
    )


def _l2norm_subsumed_by_core() -> bool:
    """True iff the selected GDN core does l2norm in-kernel, so every standalone l2norm site must
    be an identity passthrough -- otherwise the forward double-normalizes.

    Single source for that decision. Three sites must consult it; keep the list current if a fourth
    appears:
        1. ``_shim_chunk_gated_delta_rule`` -- native branch takes q/k raw
        2. ``_shim_l2norm``                 -- returns ``x`` unchanged when subsumed
        3. ``_eager_prepare_qkv``           -- skips ``gdn_l2norm`` when subsumed (negated)
    """
    from ...ops.gdn.gdn_ops import (
        chunk_synced_mode,
        gdn_native_kernels_enabled,
        recurrent_mode,
    )

    # chunk_synced-native also runs the fused core within chunks, so l2norm is in-kernel there too.
    return gdn_native_kernels_enabled() and (recurrent_mode() or chunk_synced_mode())


def _shim_chunk_gated_delta_rule(
    query,
    key,
    value,
    g=None,
    beta=None,
    scale=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
    **_ignored,
):
    """``fla.ops.gated_delta_rule.chunk_gated_delta_rule`` -> :func:`gdn_ops.gdn_core`.

    Megatron calls this positionally for q/k/v and by keyword for the rest. It never asks for
    in-kernel L2 norm (it normalises in ``_prepare_qkv_for_gated_delta_rule``), and never passes a
    custom ``scale``; both are rejected rather than silently ignored, because either one would make
    the trainer and the engine compute different things.

    The name is deliberately inaccurate in one mode: ``gdn_core`` dispatches on
    ``SKYRL_ISOEXEC_GDN_KERNEL``, so under ``recurrent`` this call executes the recurrent scan, which
    is what the engine's decode runs. Megatron binds this symbol at import and offers no other door.
    """
    from ...ops.gdn.gdn_ops import gdn_core, gdn_l2norm, gdn_native_core

    if scale is not None:
        raise NotImplementedError("isoexec GDN shim: custom `scale` would diverge from the engine")

    if _l2norm_subsumed_by_core():
        # Native composition: l2norm and gating happen in-kernel to match the engine bitwise, so
        # q/k arrive raw (_shim_l2norm is an identity here) and g/beta are the raw (alpha, b) with
        # (A_log, dt_bias) stashed by _compute_g_and_beta for the kernel to use.
        if initial_state is not None or output_final_state:
            raise NotImplementedError("isoexec GDN shim (native): stateless training entry only")
        A_log, dt_bias = _pop_gating_stash()
        from ...ops.gdn.gdn_ops import chunk_synced_mode, gdn_native_chunk_synced

        if chunk_synced_mode():
            # Native chunk-synced: fused kernel within chunks, boundary states on the matched prep.
            return gdn_native_chunk_synced(query, key, value, g, beta, A_log, dt_bias, cu_seqlens=cu_seqlens), None
        return gdn_native_core(query, key, value, g, beta, A_log, dt_bias, cu_seqlens=cu_seqlens), None

    if use_qk_l2norm_in_kernel:
        # The engine normalises outside the kernel too, so route through the same l2norm rather than
        # the kernel's internal one.
        query, key = gdn_l2norm(query), gdn_l2norm(key)
    return gdn_core(
        query,
        key,
        value,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )


def _no_recurrent(*_a, **_k):
    """`fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule` -- present only to fail loudly.

    Nothing may reach the recurrent kernel through this door in either mode. In ``chunk`` mode it is
    the kernel chunk-consistent decode exists to replace; in ``recurrent`` mode the trainer runs it
    through ``gdn_ops.gdn_recurrent``, which supplies the state pool and slot index map the bare FLA
    entry point does not. A caller landing here has bypassed gdn_ops, so it fails.
    """
    raise NotImplementedError(
        "[isoexec-gdn] call the fused recurrent delta-rule kernel through isoexec.gdn_ops "
        "(gdn_recurrent / gdn_recurrent_kernel), not through the fla facade."
    )


def _shim_l2norm(x, dim: int = -1, eps: float = 1e-6, **_ignored):
    """``fla.modules.l2norm.l2norm`` -> :func:`gdn_ops.gdn_l2norm` (row-local, last dim).

    Identity under the native composition: the fused kernel normalises in-kernel and its
    rsqrt-multiply is not bitwise-equal to ``l2norm_fwd``, so normalising here too would diverge.
    """
    from ...ops.gdn.gdn_ops import gdn_l2norm

    if _l2norm_subsumed_by_core():
        return x

    if dim not in (-1, x.ndim - 1):
        raise NotImplementedError(f"isoexec GDN shim: l2norm only over the last dim (got dim={dim})")
    return gdn_l2norm(x.contiguous())


# (A_log, dt_bias) handoff between the pass-through _compute_g_and_beta and the shim's kernel call.
# One GDN forward computes the gating then immediately calls the core, layer by layer, so the stash
# holds at most one entry; a mismatch means a Megatron code path we have not audited -- fail loud.
_GATING_STASH: list = []


def _pop_gating_stash():
    if len(_GATING_STASH) != 1:
        raise RuntimeError(
            f"[isoexec-gdn] native gating stash holds {len(_GATING_STASH)} entries, expected 1. "
            "GatedDeltaNet._compute_g_and_beta pass-through and the core call are out of step -- "
            "an unaudited Megatron path is running between them."
        )
    return _GATING_STASH.pop()


def _patch_megatron_checkpoint_fp8_default() -> bool:
    """Fix ``CheckpointWithoutOutput.__init__``'s fp8 default so no-TE selective recompute works.

    Upstream sets ``self.fp8 = fp8 is not None`` with a default of ``fp8=False`` -- truthy, so the
    forward takes the TransformerEngine-only branch (``FP8GlobalStateManager``) and NameErrors in a
    no-TE environment. The intended semantics (see multi_latent_attention, which passes
    ``fp8=quantization_or_None``) is "fp8 iff a quantization config was handed in"; normalise
    ``False`` to mean "no fp8" as well.
    """
    try:
        from megatron.core.tensor_parallel.random import CheckpointWithoutOutput
    except Exception:
        return False

    if getattr(CheckpointWithoutOutput.__init__, "__isoexec_fp8_fix__", False):
        return True
    orig = CheckpointWithoutOutput.__init__

    def __init__(self, fp8=None):
        orig(self, fp8=fp8)
        self.fp8 = fp8 is not None and fp8 is not False

    __init__.__isoexec_fp8_fix__ = True
    CheckpointWithoutOutput.__init__ = __init__
    return True


def _patch_megatron_gdn_native_gating() -> bool:
    """Rebind ``GatedDeltaNet._compute_g_and_beta`` for the native composition (call-time gated).

    Under the native composition the wrapper passes (alpha, b) through untouched and stashes
    (A_log, dt_bias) so the fused kernel computes the gating itself, exactly as the engine does;
    Megatron's own bf16 ``exp(A_log)`` differs in the last ulp. Any other mode falls through to the
    original (jit_fuser-compiled) method.
    """
    try:
        from megatron.core.ssm.gated_delta_net import GatedDeltaNet
    except Exception:
        return False

    if getattr(GatedDeltaNet._compute_g_and_beta, "__isoexec_native_gating__", False):
        return True
    orig = GatedDeltaNet._compute_g_and_beta

    def _compute_g_and_beta(self, A_log_local_cp, dt_bias_local_cp, alpha, beta):
        from ...ops.gdn.gdn_ops import (
            chunk_synced_mode,
            gdn_native_kernels_enabled,
            recurrent_mode,
        )

        if gdn_native_kernels_enabled() and (recurrent_mode() or chunk_synced_mode()):
            _GATING_STASH.append((A_log_local_cp, dt_bias_local_cp))
            return alpha, beta
        return orig(self, A_log_local_cp, dt_bias_local_cp, alpha, beta)

    _compute_g_and_beta.__isoexec_native_gating__ = True
    GatedDeltaNet._compute_g_and_beta = _compute_g_and_beta
    return True


def _shim_causal_conv1d(
    x,
    weight,
    bias=None,
    activation=None,
    initial_state=None,
    output_final_state: bool = False,
    cu_seqlens=None,
    **_ignored,
):
    """``fla.modules.convolution.causal_conv1d`` -> :func:`gdn_ops.gdn_causal_conv`, per sequence.

    Args mirror FLA: ``x`` is ``[B, T, D]``, ``weight`` is ``[D, W]``. Returns ``(y, final_state)``,
    with ``final_state`` ``None`` unless requested.

    With ``cu_seqlens`` (packed/thd, ``B == 1``) each sequence is convolved on its own: a width-4
    causal conv that ran across a packed boundary would leak the previous sequence's last 3 tokens
    into the next one's first 3 outputs. FLA does the same; we do it explicitly.
    """
    import torch

    from ...ops.gdn.gdn_ops import (
        chunk_synced_mode,
        gdn_causal_conv,
        gdn_native_conv,
        gdn_native_conv_enabled,
        gdn_native_kernels_enabled,
        recurrent_mode,
    )

    if x.ndim != 3:
        raise ValueError(f"isoexec GDN shim: causal_conv1d expects x=[B, T, D], got {tuple(x.shape)}")
    if weight.ndim != 2:
        raise ValueError(f"isoexec GDN shim: causal_conv1d expects weight=[D, W], got {tuple(weight.shape)}")

    if (gdn_native_kernels_enabled() and (recurrent_mode() or chunk_synced_mode())) or (
        gdn_native_conv_enabled() and chunk_synced_mode()
    ):
        # One varlen launch for the whole packed batch, bitwise-equal to the engine's prefill conv;
        # its fp32 bias-in-accumulator rounding differs from the eager conv below, so the trainer has
        # to switch with the engine. B>1 flattens to varlen with uniform lengths.
        if initial_state is not None or output_final_state:
            raise NotImplementedError("isoexec GDN shim (native): conv is stateless in training")
        if cu_seqlens is not None:
            if x.shape[0] != 1:
                raise ValueError("isoexec GDN shim: packed causal_conv1d requires batch == 1")
            y = gdn_native_conv(x[0], weight, bias, cu_seqlens=cu_seqlens.to(torch.int32), activation=activation)
            return y.unsqueeze(0), None
        B, T, D = x.shape
        cu = torch.arange(0, B * T + 1, T, dtype=torch.int32, device=x.device)
        y = gdn_native_conv(x.reshape(B * T, D), weight, bias, cu_seqlens=cu, activation=activation)
        return y.reshape(B, T, D), None

    def _one(seq, state):
        return gdn_causal_conv(seq, weight, bias, initial_state=state, activation=activation, return_final_state=True)

    if cu_seqlens is not None:
        if x.shape[0] != 1:
            raise ValueError("isoexec GDN shim: packed causal_conv1d requires batch == 1")
        # Memoized: the underlying `.tolist()` is a D2H sync, and the value changes once per forward.
        from ...ops.gdn.packed_meta_cache import cu_list

        bounds = cu_list(cu_seqlens)
        if bounds[-1] != x.shape[1]:
            raise ValueError(f"cu_seqlens[-1]={bounds[-1]} != T={x.shape[1]}")
        if initial_state is not None:
            raise NotImplementedError("isoexec GDN shim: initial_state with cu_seqlens is unsupported")
        ys, states = [], []
        for s, e in zip(bounds[:-1], bounds[1:]):
            y, st = _one(x[0, s:e], None)
            ys.append(y)
            states.append(st)
        y = torch.cat(ys, dim=0).unsqueeze(0)
        final_state = torch.stack(states, dim=0) if output_final_state else None
        return y, final_state

    ys, states = [], []
    for i in range(x.shape[0]):
        y, st = _one(x[i], None if initial_state is None else initial_state[i])
        ys.append(y)
        states.append(st)
    y = torch.stack(ys, dim=0)
    final_state = torch.stack(states, dim=0) if output_final_state else None
    return y, final_state


def _eager_compute_g_and_beta(self, A_log_local_cp, dt_bias_local_cp, alpha, beta):
    """Same expression as ``GatedDeltaNet._compute_g_and_beta``, minus ``@jit_fuser``.

    Kept character-for-character on purpose (``A_log.exp()`` in the parameter dtype, softplus in
    fp32): ``isoexec.gdn_ops.gdn_gate_and_beta`` implements this identical expression, so trainer and
    engine round the same way.
    """
    import torch.nn.functional as F

    g = -A_log_local_cp.exp() * F.softplus(alpha.float() + dt_bias_local_cp)
    beta = beta.sigmoid()
    return g, beta


def _eager_prepare_qkv(self, qkv, gate, beta, alpha, batch, seq_len):
    """Eager copy of ``GatedDeltaNet._prepare_qkv_for_gated_delta_rule`` (no ``@jit_fuser``)."""
    import torch

    from ...ops.gdn.gdn_ops import gdn_l2norm

    query_key, value = torch.split(
        qkv,
        [2 * self.qk_dim_local_tp // self.cp_size, self.v_dim_local_tp // self.cp_size],
        dim=-1,
    )
    query_key = query_key.reshape(batch, seq_len, -1, self.key_head_dim)
    value = value.reshape(batch, seq_len, -1, self.value_head_dim)
    if self.use_qk_l2norm:
        # Negated use of the shared predicate: this is the site that does the standalone
        # normalisation, so it runs only when the core does not already subsume it.
        if not _l2norm_subsumed_by_core():
            query_key = gdn_l2norm(query_key.contiguous())
    split_size = self.qk_dim_local_tp // self.key_head_dim // self.cp_size
    query, key = torch.split(query_key, [split_size, split_size], dim=2)
    if self.num_value_heads // self.num_key_heads > 1:
        repeat_factor = self.num_value_heads // self.num_key_heads
        query = query.repeat_interleave(repeat_factor, dim=2)
        key = key.repeat_interleave(repeat_factor, dim=2)
    # The scoring fused-outnorm accepts the token-strided gate view produced by ``split``; every
    # other path (training, grad, unsupported norm/dtype/device/layout) keeps the contiguous copy.
    gate_out = (
        gate
        if (_scoring_fused_outnorm_static_contract(self, gate) and _scoring_gate_view(gate) is not None)
        else gate.contiguous()
    )
    return (
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        gate_out,
        beta.contiguous(),
        alpha.contiguous(),
    )


def _eager_apply_gated_norm(self, x, gate):
    """Eager copy of ``GatedDeltaNet._apply_gated_norm`` (no ``@jit_fuser``).

    ``jit_fuser`` is ``torch.compile`` here, and inductor shape-specializes the fused norm+silu+mul
    kernel, so decode and prefill get different kernels whose fp32 reductions round differently on
    the same row -- drift that compounds across layers and leaks across ranks through out_proj's
    row-parallel allreduce. Eager aten is shape-invariant.
    """
    # Scoring-only fast path: the Triton kernel is byte-exact here but has no backward, so both eval
    # mode and disabled autograd are required before it is used.
    gate3 = _scoring_gate_view(gate)
    if (
        _scoring_fused_outnorm_static_contract(self, gate)
        and gate3 is not None
        and x.shape == gate.shape
        and x.device == gate.device
        and x.dtype == gate.dtype
        and x.is_contiguous()
    ):
        from ...ops.norms.fused_outnorm import fused_gated_out_norm

        heads, width = x.shape[-2:]
        result = fused_gated_out_norm(
            x.view(-1, heads, width),
            gate3,
            self.out_norm.weight,
            self.out_norm.eps,
        )
        _SCORING_FUSED_OUTNORM_COUNTS["served"] += 1
        return result

    x_dtype = x.dtype
    x = x.reshape(-1, x.shape[-1])
    y = self.out_norm(x)
    gate = gate.reshape(-1, gate.shape[-1])
    y = y * self.act_fn(gate.float())
    y = y.to(x_dtype)
    return y


def _patch_megatron_gdn_eager() -> bool:
    """Rebind the three ``jit_fuser``-decorated helpers to eager equivalents. Returns True if done."""
    try:
        from megatron.core.ssm import gated_delta_net as mg
    except Exception as e:  # pragma: no cover - megatron absent (engine-only process)
        logger.info("[isoexec-gdn] megatron.core.ssm.gated_delta_net unavailable (%s)", e)
        return False
    mg.GatedDeltaNet._compute_g_and_beta = _eager_compute_g_and_beta
    mg.GatedDeltaNet._prepare_qkv_for_gated_delta_rule = _eager_prepare_qkv
    mg.GatedDeltaNet._apply_gated_norm = _eager_apply_gated_norm
    print(
        "[ISOEXEC-GDN] megatron GatedDeltaNet: g/beta + qkv-prep + gated-norm run EAGER " "(no torch.compile)",
        flush=True,
    )
    return True


def _validate_once_enabled() -> bool:
    """``SKYRL_ISOEXEC_GDN_VALIDATE_ONCE`` (default OFF): run the cp-divisibility check once/forward.

    Off by default because, unlike the memoized host reads, collapsing this check deletes kernel
    launches on every layer after the first -- they only write temporaries the host discards, but the
    launch sequence changes, so enabling it needs its own verification.
    """
    return os.environ.get("SKYRL_ISOEXEC_GDN_VALIDATE_ONCE", "0").lower() not in ("0", "false", "no", "")


def _resolve_cu_seqlens_memo(self, cu_seqlens_padded, cu_seqlens_actual, total_seq_len, name, cp_size: int = 1):
    """``GatedDeltaNet._resolve_cu_seqlens`` with its host reads memoized per forward.

    Upstream host-syncs on ``cu_seqlens[-1]`` and runs a divisibility check per GDN layer, twice
    over, to validate a tensor that changes once per microbatch. The function computes nothing -- it
    returns one of its inputs unchanged, everything else is validation -- so the memo cannot change
    the value the model sees. The D2H is removed unconditionally; at CP>1 the divisibility launches
    are removed only under ``SKYRL_ISOEXEC_GDN_VALIDATE_ONCE``, and at CP=1 the check is the identity
    ``n % 1 == 0`` and is skipped outright.
    """
    from ...ops.gdn.packed_meta_cache import cu_last, seq_lens

    cu_seqlens = cu_seqlens_padded if cu_seqlens_padded is not None else cu_seqlens_actual

    total_cu = cu_last(cu_seqlens)
    if total_cu != total_seq_len:
        raise ValueError(
            f"GDN: {name}[-1]={total_cu} does not match total_sequence_length={total_seq_len}. "
            f"({cu_seqlens_padded=}, {cu_seqlens_actual=})."
        )

    # Record the resolved (cu, cp_size) for this forward so `_unpack_sequence_memo` can recognise
    # the `cu_seqlens_q // self.cp_size` its call site builds; the object is stored, not a copy.
    # The monotone generation is the ledger key: `id(cu)` aliases, because CPython recycles tensor
    # object addresses and nothing holds the old cu alive.
    if (
        cu_seqlens is not _LAST_RESOLVED["cu"]
        or cu_seqlens._version != _LAST_RESOLVED["ver"]
        or int(cp_size) != _LAST_RESOLVED["cp"]
    ):
        _LAST_RESOLVED["gen"] += 1
        _LAST_RESOLVED["cu"] = cu_seqlens
        _LAST_RESOLVED["ver"] = cu_seqlens._version
        _LAST_RESOLVED["cp"] = int(cp_size)

    # At cp_size == 1 the upstream device check is a tautology; skipping it removes only that.
    if cp_size != 1:
        if _validate_once_enabled():
            # Host-side, off the same memoized list read: exact for non-negative integer offsets.
            bad = [n for n in seq_lens(cu_seqlens) if n % cp_size != 0]
            if bad:
                raise ValueError(
                    f"All per-sequence lengths in cu_seqlens must be divisible by cp_size={cp_size}, "
                    f"but got lengths: {seq_lens(cu_seqlens)}"
                )
        else:
            seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            if (seq_lengths % cp_size != 0).any():
                raise ValueError(
                    f"All per-sequence lengths in cu_seqlens must be divisible by cp_size={cp_size}, "
                    f"but got lengths: {seq_lengths.tolist()}"
                )

    return cu_seqlens


# The (cu_seqlens, cp_size) most recently resolved, plus the monotone generation identifying it.
# One slot: every GDN layer of a forward resolves the same object. ``cu`` is held strongly on purpose.
_LAST_RESOLVED: dict = {"cu": None, "cp": 1, "ver": -1, "gen": 0}

# Divided-list proof ledger: key -> the proven list, or None once disproven. Keyed on the generation,
# never on `id(cu)`, so an entry cannot be served to a different microbatch's cu_seqlens.
_DIV_PROOF: dict = {}
_DIV_PROOF_MAX = 16
_DIV_REVERIFY = 64  # re-prove every Nth serve, so the ledger polices itself
_DIV_STATS = {"proved": 0, "served": 0, "reverified": 0, "disproved": 0, "unmatched": 0, "stale": 0}
_MISSING = object()


def divided_cu_census() -> dict:
    """``{proved, served, reverified, disproved, unmatched, stale}``.

    ``stale`` must stay 0: it counts serves where the ledger disagreed with the list derived from the
    currently-resolved ``cu``. Non-zero is a defect; the serve is refused and the real read is paid.
    """
    return dict(_DIV_STATS)


def _div_hoist_enabled() -> bool:
    """``SKYRL_ISOEXEC_PACKED_META_DIV_HOIST`` (default ON): the cp-division hoist specifically.

    A kill switch for :func:`_divided_cu_list` alone, leaving the rest of
    ``SKYRL_ISOEXEC_PACKED_META_CACHE`` in place. Off restores the plain per-layer ``.tolist()``.
    """
    return os.environ.get("SKYRL_ISOEXEC_PACKED_META_DIV_HOIST", "1").lower() not in ("0", "false", "no", "")


def _divided_cu_list(d, dim: int):
    """The host list of ``d``, served from a per-forward proof that ``d == cu // cp_size``.

    ``GatedDeltaNet.forward`` passes ``cu_seqlens_q // self.cp_size``, a fresh tensor allocated at
    the call site on every GDN layer, so an identity-keyed memo can never hit it. Instead the first
    call of each forward pays the real ``.tolist()`` and compares it against the value derived on the
    host from the memoized ``cu``; only an exact match records the proof, and the proof is re-taken
    every ``_DIV_REVERIFY`` serves. The ledger key is the monotone generation, never ``id(cu)``:
    nothing holds the old ``cu`` alive and CPython recycles tensor addresses, so an address key can
    serve one microbatch's offsets to another. Every unexpected condition -- no resolved cu, mismatched
    shape/dtype/device, an argument that is not freshly derived, a failed proof or tripwire, either
    flag off -- falls through to the plain per-call read.
    """
    from ...ops.gdn.packed_meta_cache import cu_list, packed_meta_cache_enabled

    cu = _LAST_RESOLVED.get("cu")
    cp = int(_LAST_RESOLVED.get("cp") or 1)
    if not packed_meta_cache_enabled() or not _div_hoist_enabled() or cu is None or cp < 1:
        return d.tolist()
    # Structural precondition: `cu // cp` has cu's shape, dtype and device and is a fresh tensor;
    # anything else is a different caller this ledger has no claim on.
    if d is cu or d._version != 0:
        _DIV_STATS["unmatched"] += 1
        return d.tolist()
    if d.shape != cu.shape or d.dtype != cu.dtype or d.device != cu.device:
        _DIV_STATS["unmatched"] += 1
        return d.tolist()

    key = (_LAST_RESOLVED["gen"], cp, int(dim), tuple(d.shape))
    expected = [v // cp for v in cu_list(cu)]
    proof = _DIV_PROOF.get(key, _MISSING)

    if proof is _MISSING:
        actual = d.tolist()  # the one read per forward that BUYS the proof
        if actual == expected:
            _DIV_PROOF[key] = [expected, 0]
            _DIV_STATS["proved"] += 1
        else:
            _DIV_PROOF[key] = None
            _DIV_STATS["disproved"] += 1
        while len(_DIV_PROOF) > _DIV_PROOF_MAX:
            _DIV_PROOF.pop(next(iter(_DIV_PROOF)))
        return actual

    if proof is None:  # disproven for this forward -- never fast-path it again
        return d.tolist()

    proven, n = proof
    proof[1] = n + 1
    if proof[1] % _DIV_REVERIFY == 0:
        actual = d.tolist()  # periodic re-proof: the ledger polices itself
        _DIV_STATS["reverified"] += 1
        if actual != proven:
            _DIV_PROOF[key] = None
            _DIV_STATS["disproved"] += 1
        return actual
    if proven != expected:
        # Unreachable while the generation key holds; kept as a tripwire.
        _DIV_PROOF[key] = None
        _DIV_STATS["stale"] += 1
        return d.tolist()
    _DIV_STATS["served"] += 1
    return expected


def _unpack_sequence_memo(x, cu_seqlens, dim=1):
    """``gated_delta_net._unpack_sequence`` with its ``cu_seqlens.tolist()`` memoized.

    Otherwise byte-for-byte the upstream body: the same slices in the same order. ``.tolist()``
    issues no kernel, so this removes a pageable D2H and nothing else.

    One call site passes ``cu_seqlens_q`` directly and the identity memo hits; the other passes
    ``cu_seqlens_q // self.cp_size``, which no identity memo can hit, and is served by
    :func:`_divided_cu_list`. The ``is``-check comes first because the direct site also runs from the
    recompute hook, by which time the packed-meta LRU may have evicted this ``cu``.
    """
    from ...ops.gdn.packed_meta_cache import cu_list, is_memoized

    if cu_seqlens is _LAST_RESOLVED["cu"] or is_memoized(cu_seqlens):
        cu_seqlens_list = cu_list(cu_seqlens)  # a plain identity hit
    else:
        cu_seqlens_list = _divided_cu_list(cu_seqlens, dim)
    unpacked_x = []
    for i in range(len(cu_seqlens_list) - 1):
        chunked_index = [slice(None)] * dim + [slice(cu_seqlens_list[i], cu_seqlens_list[i + 1])]
        unpacked_x.append(x[tuple(chunked_index)])
    return unpacked_x


def _apply_rotary_pos_emb_thd_memo(
    t,
    cu_seqlens,
    freqs,
    rotary_interleaved: bool = False,
    mla_rotary_interleaved: bool = False,
    mscale: float = 1.0,
    cp_group=None,
    multi_latent_attention=None,
):
    """``rope_utils._apply_rotary_pos_emb_thd`` with its two host reads memoized.

    Upstream syncs twice per attention layer on the same per-forward ``cu_seqlens``. Everything else
    here is upstream verbatim; ``seqlens`` is computed on the host from one memoized list read and is
    elementwise equal to the device expression (exact integer arithmetic on monotone non-negative
    offsets), so ``torch.split`` receives identical sizes and every downstream kernel sees identical
    arguments.
    """
    import torch

    from ...ops.gdn.packed_meta_cache import (
        cu_last,
        cu_list,
        packed_meta_cache_enabled,
        seq_lens,
    )

    if multi_latent_attention is not None:
        mla_rotary_interleaved = multi_latent_attention
    if cp_group is None:
        raise ValueError("cp_group must be provided for THD format RoPE")
    cp_size = cp_group.size()
    cp_rank = cp_group.rank()
    # Flag off must restore the upstream expression exactly, including the `sub`/`floor_divide`.
    memo = packed_meta_cache_enabled()
    if memo:
        seqlens = seq_lens(cu_seqlens, cp_size)
        total = cu_last(cu_seqlens)
    else:
        seqlens = ((cu_seqlens[1:] - cu_seqlens[:-1]) // cp_size).tolist()
        total = cu_seqlens[-1]

    from megatron.core.models.common.embeddings.rope_utils import (
        _apply_rotary_pos_emb_bshd,
        _get_thd_freqs_on_this_cp_rank,
    )

    sequence_splits = torch.split(t, seqlens)
    if freqs.dim() >= 1 and freqs.size(0) == total:
        # Exact mapping with offsets: upstream's `cu_seqlens[i].item()` is a sync per sequence.
        offsets = cu_list(cu_seqlens) if memo else [cu_seqlens[i].item() for i in range(len(sequence_splits))]
        freq_slices = [
            _get_thd_freqs_on_this_cp_rank(cp_rank, cp_size, x, freqs, offsets[i])
            for i, x in enumerate(sequence_splits)
        ]
    else:
        # CASE 2: traditional mapping without offsets
        freq_slices = [_get_thd_freqs_on_this_cp_rank(cp_rank, cp_size, x, freqs) for x in sequence_splits]

    freqs_packed = torch.cat(freq_slices, dim=0)
    return _apply_rotary_pos_emb_bshd(
        t.unsqueeze(1),
        freqs_packed,
        rotary_interleaved=rotary_interleaved,
        mla_rotary_interleaved=mla_rotary_interleaved,
        mscale=mscale,
    ).squeeze(1)


def _patch_megatron_packed_meta() -> bool:
    """Install the memoized packed-sequence host reads on megatron's GatedDeltaNet.

    Call-time gated on ``SKYRL_ISOEXEC_PACKED_META_CACHE`` (the memo declines and falls through to a
    plain read when the flag is off), so installing unconditionally is free and the flag stays a
    runtime decision rather than an import-time one.
    """
    try:
        from megatron.core.ssm import gated_delta_net as mg
    except Exception as e:  # pragma: no cover - megatron absent (engine-only process)
        logger.info("[isoexec-gdn] megatron.core.ssm.gated_delta_net unavailable (%s)", e)
        return False
    mg.GatedDeltaNet._resolve_cu_seqlens = _resolve_cu_seqlens_memo
    mg._unpack_sequence = _unpack_sequence_memo
    # `apply_rotary_pos_emb` reaches `_apply_rotary_pos_emb_thd` by module-global lookup, so
    # rebinding the module attribute is enough.
    try:
        from megatron.core.models.common.embeddings import rope_utils

        rope_utils._apply_rotary_pos_emb_thd = _apply_rotary_pos_emb_thd_memo
    except Exception as e:  # noqa: BLE001 -- the GDN half is worth landing without the rope half
        logger.info("[isoexec-gdn] rope_utils packed-meta patch skipped (%s)", e)
    return True


def _module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    # A module with `__spec__ = None` makes `importlib.util.find_spec(name)` raise ValueError, and
    # `transformers.utils.is_flash_linear_attention_available()` calls exactly that on `fla` while
    # importing Qwen3-Next. Give the facade a real (loader-less) spec.
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


def install_fla_shim(*, force: bool = False) -> bool:
    """Register the ``fla`` facade so ``megatron.core.ssm.gated_delta_net`` imports our ops.

    No-op (returning False) unless ``SKYRL_ISOEXEC_GDN=1`` or ``force``. Idempotent. Refuses to
    shadow a real ``flash-linear-attention`` install, because then the two runtimes would silently
    stop sharing a kernel.
    """
    global _installed

    if _installed:
        return True
    if not (force or gdn_enabled()):
        return False

    if "fla" in sys.modules and not getattr(sys.modules["fla"], "__isoexec_shim__", False):
        raise RuntimeError(
            "[isoexec-gdn] a real `fla` package is already imported. Uninstall flash-linear-attention: "
            "IsoExec requires the trainer and the engine to share ONE chunk-kernel implementation "
            "(and one autotune decision)."
        )

    # __version__ "0.0.0": `transformers.is_flash_linear_attention_available()` requires >= 0.2.2 and
    # falls back to the module's __version__ when there is no distribution metadata. Declaring an old
    # version makes HF answer "no FLA" and use its own torch reference, which is what we want -- its
    # Qwen3-Next modeling would otherwise import `fused_recurrent_gated_delta_rule` from this facade.
    # Megatron imports our symbols by name and never consults that check.
    fla = _module("fla", __isoexec_shim__=True, __version__="0.0.0", __path__=[])
    ops = _module("fla.ops", __path__=[])
    gdr = _module(
        "fla.ops.gated_delta_rule",
        chunk_gated_delta_rule=_shim_chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule=_no_recurrent,
    )
    modules = _module("fla.modules", __path__=[])
    conv = _module("fla.modules.convolution", causal_conv1d=_shim_causal_conv1d)
    l2 = _module("fla.modules.l2norm", l2norm=_shim_l2norm)

    fla.ops, fla.modules = ops, modules
    ops.gated_delta_rule = gdr
    modules.convolution, modules.l2norm = conv, l2
    # `from fla.ops.gated_delta_rule import chunk_gated_delta_rule` also works via `fla.ops`
    ops.chunk_gated_delta_rule = _shim_chunk_gated_delta_rule
    modules.causal_conv1d, modules.l2norm_fn = _shim_causal_conv1d, _shim_l2norm

    _installed = True

    from ...ops.gdn.gdn_ops import gdn_kernel_mode

    print(
        f"[ISOEXEC-GDN] installed `fla` shim -> isoexec.gdn_ops "
        f"(delta-rule kernel: {gdn_kernel_mode()} / l2norm / causal_conv1d)",
        flush=True,
    )

    if _eager_prep_enabled():
        _patch_megatron_gdn_eager()

    # Pass-through gating wrapper for the native composition; call-time gated, so installing it
    # unconditionally is free.
    _patch_megatron_gdn_native_gating()

    # the packed-sequence host-read memo (call-time gated on SKYRL_ISOEXEC_PACKED_META_CACHE)
    _patch_megatron_packed_meta()

    # Report which l2norm regime this process installs under, so a double-normalize is visible at
    # startup. Reflects the env at install time; the predicate itself is call-time gated.
    print(
        f"[ISOEXEC-GDN] l2norm subsumption: {'identity' if _l2norm_subsumed_by_core() else 'normalizing'} "
        "(all standalone l2norm sites route through _l2norm_subsumed_by_core())",
        flush=True,
    )

    # Upstream megatron bug that only fires without TransformerEngine; see the patch's docstring.
    _patch_megatron_checkpoint_fp8_default()

    # Qwen3.5 normalises with rms(x) * (1 + w); the no-TE torch norm asserts against that flag and
    # would abort while building the first layer, in the trainer and in the in-vLLM GPTModel alike.
    from ...ops.norms.zero_centered_norm import install_zero_centered_torch_norm

    install_zero_centered_torch_norm()

    # Memoize only the two device-derived halves of torch's `_native` RMSNorm admission predicate;
    # the boolean it returns, and so which kernel runs, cannot move. Default off, fail-soft.
    try:
        from ...ops.norms.native_rmsnorm_memo import install_native_rmsnorm_memo

        install_native_rmsnorm_memo()
    except Exception as e:  # pragma: no cover - a host-time memo must never break an install
        logger.info("[isoexec-norm] native RMSNorm admission memo not installed (%s)", e)

    # Pin the Triton autotune configs now if vLLM is importable in this process; `gdn_chunk` pins
    # again (idempotently) before its first launch, so a deferred pin is safe, never skipped.
    try:
        from ...ops.gdn.gdn_batch_invariant import pin_fla_autotune_configs

        pin_fla_autotune_configs()
    except Exception as e:  # pragma: no cover - vLLM not importable yet
        logger.info("[isoexec-gdn] deferring autotune pin to first gdn_chunk call (%s)", e)
    return True
