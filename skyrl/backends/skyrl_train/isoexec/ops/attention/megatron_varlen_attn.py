"""Grad-capable trainer attention that is bitwise-identical to the rollout engine's kernel.

The engine computes attention with ``varlen_attn(..., num_splits=1, window_size=(-1, 0))``, so
Megatron's ``SelfAttention.core_attention`` must call the same function with the same arguments;
SDPA and ``flash_attn`` are different kernels and leave large per-token logprob outliers.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


def isoexec_local_head_counts(cfg, tp: int) -> tuple[int, int]:
    """Per-rank (q_heads, kv_heads) that Megatron's SelfAttention hands core_attention at TP=tp.

    Mirrors Megatron's own derivation: when num_query_groups < TP it replicates kv heads (pure data
    movement, so bitwise-safe) and each rank runs num_attention_heads/TP q heads over one kv group.
    """
    q, g = cfg.num_attention_heads, cfg.num_query_groups
    if q % tp:
        raise ValueError(f"[isoexec] num_attention_heads ({q}) must divide TP={tp}")
    if g >= tp:
        if g % tp:
            raise ValueError(f"[isoexec] num_query_groups ({g}) must divide TP={tp}")
        return q // tp, g // tp
    if tp % g:
        raise ValueError(
            f"[isoexec] TP={tp} must be a multiple of num_query_groups ({g}) " "for megatron's kv-replication path"
        )
    return q // tp, 1


class TorchVarlenCoreAttn(nn.Module):
    """Drop-in for ``SelfAttention.core_attention`` using torch ``varlen_attn`` == the engine kernel."""

    def __init__(self, *, num_heads, num_kv_heads, head_dim, scale):
        super().__init__()
        import os as _os

        import torch.nn.attention.varlen as _V  # noqa: N814

        self._varlen_attn = _V.varlen_attn
        # Non-paged varlen_attn does not reliably honor num_splits=1 across runtime contexts: its FA3
        # split-K heuristic can vary with GPU occupancy, giving a 1-ULP bf16 diff. varlen_attn_out (the
        # engine's kernel) does honor it.
        self._use_out = _os.environ.get("SKYRL_ISOEXEC_VARLEN_OUT", "0") == "1"
        self._varlen_attn_out = getattr(_V, "varlen_attn_out", None)
        if self._use_out and self._varlen_attn_out is None:
            self._use_out = False
        # The paged path (block_table) forces num_splits=1 to be honored context-invariantly.
        # block_size must be divisible by 256 (FA3 constraint).
        self._paged = _os.environ.get("SKYRL_ISOEXEC_VARLEN_PAGED", "0") == "1" and self._varlen_attn_out is not None
        self._page_bs = 256
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale
        self.enable_gqa = num_heads > num_kv_heads

    def forward(
        self,
        query,
        key,
        value,
        attention_mask=None,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
        **extra_kwargs,
    ):
        # `**extra_kwargs` is for MLA, whose `_run_core_attention` forwards its caller's kwargs
        # straight through. None of them change the causal varlen math, so accept and ignore.
        # Two layouts reach this kernel:
        #   sbhd  q [sq, b, np, hn] -- the unpacked micro-forward, b == 1.
        #   thd   q [t,  np, hn]    -- sample packing; Megatron folds the batch dim away and passes
        #                             the sequence boundaries in `packed_seq_params`.
        if query.dim() == 4:
            sq, b = query.shape[0], query.shape[1]
            packed = False
        elif query.dim() == 3:
            sq, b = query.shape[0], 1
            packed = True
        else:
            raise ValueError(f"unexpected core_attention query rank {query.dim()}")
        if not packed and b > 1:
            # sbhd [sq, b, np, hn] -> varlen [b*sq, np, hn], the b equal-length sequences laid out
            # contiguously with cu_seqlens = [0, sq, 2sq, ...]. num_splits=1 makes each row's
            # reduction independent of how many sequences share the launch, so b>1 stays bitwise.
            q = query.permute(1, 0, 2, 3).reshape(b * sq, self.num_heads, self.head_dim).contiguous()
            k = key.permute(1, 0, 2, 3).reshape(b * sq, self.num_kv_heads, self.head_dim).contiguous()
            v = value.permute(1, 0, 2, 3).reshape(b * sq, self.num_kv_heads, self.head_dim).contiguous()
            cu = torch.arange(0, (b + 1) * sq, sq, device=q.device, dtype=torch.int32)
            if self._paged:
                raise NotImplementedError("[isoexec] the PAGED varlen recipe assumes b == 1")
            out = (
                self._varlen_attn(
                    q,
                    k,
                    v,
                    cu,
                    cu,
                    sq,
                    sq,
                    scale=self.scale,
                    num_splits=1,
                    enable_gqa=self.enable_gqa,
                    window_size=(-1, 0),
                )
                if not self._use_out
                else self._varlen_attn_out(
                    torch.empty_like(q),
                    q,
                    k,
                    v,
                    cu,
                    cu,
                    sq,
                    sq,
                    scale=self.scale,
                    num_splits=1,
                    enable_gqa=self.enable_gqa,
                    window_size=(-1, 0),
                )
            )
            if isinstance(out, tuple):
                out = out[0]
            hp = self.num_heads * self.head_dim
            # [b*sq, np, hn] -> [b, sq, hp] -> sbhd [sq, b, hp]
            return out.reshape(b, sq, hp).permute(1, 0, 2).contiguous()

        q = query.reshape(sq, self.num_heads, self.head_dim)
        k = key.reshape(sq, self.num_kv_heads, self.head_dim)
        v = value.reshape(sq, self.num_kv_heads, self.head_dim)

        # Without the packed boundaries a packed row would attend across sequence boundaries, while
        # the rollout engine attends per sequence.
        max_q = max_k = sq
        if packed_seq_params is not None and getattr(packed_seq_params, "qkv_format", None) == "thd":
            cu = packed_seq_params.cu_seqlens_q.to(device=q.device, dtype=torch.int32)
            cu_kv = packed_seq_params.cu_seqlens_kv.to(device=q.device, dtype=torch.int32)
            if not torch.equal(cu, cu_kv):
                raise NotImplementedError("[isoexec] cu_seqlens_q != cu_seqlens_kv")
            max_q = int(packed_seq_params.max_seqlen_q)
            max_k = int(packed_seq_params.max_seqlen_kv)
        elif packed:
            raise ValueError("[isoexec] thd core_attention input without PackedSeqParams(qkv_format='thd')")
        else:
            cu = torch.tensor([0, sq], device=q.device, dtype=torch.int32)
        if self._paged:
            if packed:
                raise NotImplementedError("[isoexec] the PAGED varlen recipe does not handle thd packing")
            # Pack K/V into [num_blocks, 256, kv_heads, head_dim] + block_table so FA3 honors
            # num_splits=1. The pad is non-inplace to keep autograd intact.
            _bs = self._page_bs
            nb = (sq + _bs - 1) // _bs
            pad = nb * _bs - sq
            if pad:
                k = torch.cat([k, k.new_zeros(pad, self.num_kv_heads, self.head_dim)], dim=0)
                v = torch.cat([v, v.new_zeros(pad, self.num_kv_heads, self.head_dim)], dim=0)
            kc = k.view(nb, _bs, self.num_kv_heads, self.head_dim)
            vc = v.view(nb, _bs, self.num_kv_heads, self.head_dim)
            bt = torch.arange(nb, device=q.device, dtype=torch.int32).unsqueeze(0)
            su = torch.tensor([sq], device=q.device, dtype=torch.int32)
            _o = torch.empty_like(q)
            # FA3 rejects cu_seqlens_k together with a block_table; seqused_k carries the KV lengths
            # instead. The engine's backend does exactly the same.
            from torch.nn.attention import current_flash_attention_impl

            cu_k = None if current_flash_attention_impl() == "FA3" else cu
            out = self._varlen_attn_out(
                _o,
                q,
                kc,
                vc,
                cu,
                cu_k,
                sq,
                sq,
                scale=self.scale,
                num_splits=1,
                enable_gqa=self.enable_gqa,
                window_size=(-1, 0),
                block_table=bt,
                seqused_k=su,
            )
        elif self._use_out:
            # The engine's exact kernel; honors num_splits=1 context-invariantly.
            _o = torch.empty_like(q)
            out = self._varlen_attn_out(
                _o,
                q,
                k,
                v,
                cu,
                cu,
                max_q,
                max_k,
                scale=self.scale,
                num_splits=1,
                enable_gqa=self.enable_gqa,
                window_size=(-1, 0),
            )
        else:
            out = self._varlen_attn(
                q,
                k,
                v,
                cu,
                cu,
                max_q,
                max_k,
                scale=self.scale,
                num_splits=1,  # single KV-reduction split -> bitwise == engine prefill/decode
                enable_gqa=self.enable_gqa,
                window_size=(-1, 0),  # unlimited left, zero right == causal (the engine's recipe)
            )
        if isinstance(out, tuple):
            out = out[0]
        hp = self.num_heads * self.head_dim
        # thd keeps the folded layout Megatron handed us; sbhd gets its batch dim back.
        return out.reshape(sq, hp) if packed else out.reshape(sq, b, hp)


def enable_trainer_batch_invariant():
    """Enable the same vLLM batch-invariant aten ops the engine runs, so the trainer's non-attention
    ops are bitwise-identical to the rollout.

    Uses vLLM's implementation rather than megatron-core's so both sides share the exact same
    kernels. Idempotent.
    """
    try:
        from vllm.model_executor.layers.batch_invariant import (
            enable_batch_invariant_mode,
        )
    except Exception as e:  # pragma: no cover
        logger.warning("[isoexec] vLLM batch_invariant unavailable, trainer non-attn not batch-invariant: %s", e)
        return False
    enable_batch_invariant_mode()

    # The scoped aten::bmm must be installed HERE, not at the mm site: vLLM always registers
    # aten::bmm and `enable_batch_invariant_mode` runs after the mm-site install, so a bmm registered
    # there would be silently clobbered by the line above. install_bmm_scope is idempotent and
    # no-ops when its flag is off, so calling it unconditionally is safe.
    from ..mm.mm_fwd_scope import install_bmm_scope

    install_bmm_scope()

    print(
        "[ISOEXEC-TRAINER] enabled vLLM batch-invariant aten ops (mm/addmm/linear/log_softmax/mean) "
        "== engine -> bitwise non-attention",
        flush=True,
    )
    return True


def activate_trainer_flash_attention_impl():
    """Activate torch's FA3 flash-attention impl in the trainer runtime, as the engine does.

    The engine's varlen backend activates FA3 on build; the trainer process builds no vLLM engine, so
    torch's flash impl would stay unset and the same ``varlen_attn`` call would dispatch to a
    different kernel. That mismatch is worth ~1 ULP per layer in bf16 and shows up as a residual.
    """
    try:
        from torch.nn.attention import (
            activate_flash_attention_impl,
            current_flash_attention_impl,
        )
    except Exception as e:  # pragma: no cover - old torch without the flash-impl API
        logger.warning("[isoexec] torch flash-attn impl API unavailable, trainer attn may not match engine: %s", e)
        return None
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        logger.warning("[isoexec] FA3 requires SM 9.0+; trainer attention will NOT be bitwise == engine")
        return current_flash_attention_impl()
    if current_flash_attention_impl() != "FA3":
        activate_flash_attention_impl("FA3")
    impl = current_flash_attention_impl()
    print(
        f"[ISOEXEC-TRAINER] activated flash-attn impl = {impl} (== engine varlen_backend FA3) "
        f"-> trainer varlen_attn bitwise == engine",
        flush=True,
    )
    return impl


def swap_trainer_core_attention_varlen(gpt_modules):
    """Replace each decoder layer's core_attention with the torch-varlen kernel (== rollout engine)."""
    # Match the engine's flash-attn dispatch before any trainer forward.
    activate_trainer_flash_attention_impl()

    modules = gpt_modules if isinstance(gpt_modules, (list, tuple)) else [gpt_modules]
    n = 0
    for m in modules:
        inner = m
        for _ in range(4):  # unwrap DDP(Float16Module(GPTModel)) -> GPTModel (the one with .decoder)
            if hasattr(inner, "decoder"):
                break
            inner = getattr(inner, "module", inner)
        if not hasattr(inner, "decoder"):
            continue
        cfg = inner.config
        head_dim = getattr(cfg, "kv_channels", cfg.hidden_size // cfg.num_attention_heads)
        # Megatron shards attention heads across TP, so core_attention sees per-rank head counts,
        # including the kv-replication case (num_query_groups < TP -> 1 kv head per rank).
        tp = getattr(cfg, "tensor_model_parallel_size", 1) or 1
        local_q, local_kv = isoexec_local_head_counts(cfg, tp)
        for layer in inner.decoder.layers:
            sa = getattr(layer, "self_attention", None)
            if sa is None or not hasattr(sa, "core_attention"):
                continue  # GatedDeltaNet layers have no core_attention
            scale = head_dim**-0.5
            sa.core_attention = TorchVarlenCoreAttn(
                num_heads=local_q, num_kv_heads=local_kv, head_dim=head_dim, scale=scale
            )
            n += 1
    import os as _os2

    _mode = (
        "PAGED varlen_attn_out"
        if _os2.environ.get("SKYRL_ISOEXEC_VARLEN_PAGED") == "1"
        else (
            "varlen_attn_out (non-paged)"
            if _os2.environ.get("SKYRL_ISOEXEC_VARLEN_OUT") == "1"
            else "varlen_attn (non-paged)"
        )
    )
    logger.info("[isoexec] swapped TRAINER core_attention -> %s on %d layers", _mode, n)
    print(
        f"[ISOEXEC-TRAINER] swapped core_attention -> {_mode} num_splits=1 window=(-1,0) on {n} layers "
        f"(VARLEN_PAGED={_os2.environ.get('SKYRL_ISOEXEC_VARLEN_PAGED')!r} "
        f"VARLEN_OUT={_os2.environ.get('SKYRL_ISOEXEC_VARLEN_OUT')!r})",
        flush=True,
    )
    return n
