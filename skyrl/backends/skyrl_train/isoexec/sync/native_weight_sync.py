"""Native (no-HF) weight sync for the unified-GPTModel IsoExec route: when the generator runs the
same Megatron ``GPTModel`` as the trainer, both sides hold an identical state_dict, so the sync is a
direct native-layout tensor copy instead of an HF export/repack round-trip.

At matched TP no reshard is needed: the CUDA-IPC transport keys handles by physical GPU UUID, so the
routing is correct exactly because colocated placement puts trainer TP rank r and engine TP rank r on
the same GPU. Resharding (``reshard=True``) is only for a mismatched engine TP/EP.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Iterator, Tuple

import torch

logger = logging.getLogger(__name__)


# Buffers that are model state on the logprob path and must ride the sync alongside
# ``named_parameters()``. An explicit allowlist rather than ``include_buffers=True``, because most
# buffers are per-forward inference scratch (rope caches, per-expert token counters) that the engine
# rebuilds itself. Matched on a dotted-name suffix, so layer-index and wrapper-prefix independent.
SYNCED_BUFFER_SUFFIXES = (
    "mlp.router.expert_bias",  # GLM-4.7-Flash / DeepSeek noaux_tc routing bias (fp32, replicated)
)


def is_synced_buffer(name: str) -> bool:
    """True for a buffer that must ride the weight sync (see ``SYNCED_BUFFER_SUFFIXES``)."""
    return any(name == s or name.endswith("." + s) for s in SYNCED_BUFFER_SUFFIXES)


def _to_full(t: torch.Tensor) -> torch.Tensor:
    """Materialize a (possibly DTensor) parameter to a plain local tensor."""
    if hasattr(t, "full_tensor"):
        try:
            return t.full_tensor()
        except Exception:
            return t
    return t


# Mismatched-TP resharding (trainer TP > engine TP). Weights are laid out per-rank at TP>1, so a
# smaller-TP engine needs each TP-sharded param gathered across the trainer TP group, respecting
# Megatron's partition convention (dim + stride) so the result is bitwise the layout a TP=1 build
# would hold.


def _megatron_tp():
    """(trainer_tp_size, tp_group). (1, None) if model-parallel is not initialized."""
    try:
        from megatron.core import parallel_state as mpu

        if not mpu.model_parallel_is_initialized():
            return 1, None
        return mpu.get_tensor_model_parallel_world_size(), mpu.get_tensor_model_parallel_group()
    except Exception:
        return 1, None


def engine_tp_target() -> int | None:
    """Engine TP the trainer must reshard TO, from ``SKYRL_ISOEXEC_ENGINE_TP`` (set by the launcher).
    None -> no resharding (matched TP / feature off)."""
    v = os.environ.get("SKYRL_ISOEXEC_ENGINE_TP")
    return int(v) if v else None


# Mismatched-EP remap+gather (trainer EP > engine EP). Every EP rank names its experts
# `local_experts.0..num_local-1`, so the same NAME means a DIFFERENT global expert on each rank
# (global = ep_rank*num_local + local). A smaller-EP engine needs all experts under their global
# names, so expert params are all-gathered across the EP group and re-emitted -- gathering a LIST of
# distinct tensors, not reassembling shards as the TP path does.
_EXPERT_RE = re.compile(r"\.local_experts\.(\d+)\.")


def _megatron_ep():
    """(ep_size, ep_rank, ep_group). (1, 0, None) if EP is not initialized."""
    try:
        from megatron.core import parallel_state as mpu

        if not mpu.model_parallel_is_initialized():
            return 1, 0, None
        return (
            mpu.get_expert_model_parallel_world_size(),
            mpu.get_expert_model_parallel_rank(),
            mpu.get_expert_model_parallel_group(),
        )
    except Exception:
        return 1, 0, None


def engine_ep_target() -> int | None:
    v = os.environ.get("SKYRL_ISOEXEC_ENGINE_EP")
    return int(v) if v else None


def _needs_ep_remap(trainer_ep: int) -> bool:
    ee = engine_ep_target()
    return ee is not None and trainer_ep > 1 and ee != trainer_ep


def _expert_local_idx(name: str):
    m = _EXPERT_RE.search(name)
    return int(m.group(1)) if m else None


def _global_expert_name(name: str, local_idx: int, ep_rank: int, num_local: int) -> str:
    g = ep_rank * num_local + local_idx
    return _EXPERT_RE.sub(f".local_experts.{g}.", name, count=1)


def _needs_reshard(trainer_tp: int) -> bool:
    et = engine_tp_target()
    return et is not None and trainer_tp > 1 and et != trainer_tp


def _is_vocab_param(name: str) -> bool:
    return name.endswith("word_embeddings.weight") or name.endswith("output_layer.weight")


def _vocab_padded(vocab_size: int, tp: int, make_div_by: int) -> int:
    """Megatron pads the vocab to a multiple of (make_vocab_size_divisible_by * TP)."""
    m = make_div_by * tp
    return ((vocab_size + m - 1) // m) * m


def _gather_tp_to_full(shard: torch.Tensor, dim: int, stride: int, tp_group) -> torch.Tensor:
    """All-gather a TP shard across the group and rebuild the full tensor along ``dim``.

    Reconstruction matches Megatron's own sharding (``_initialize_affine_weight_*``): the full
    dim is split into ``world*stride`` equal pieces and rank r owns pieces ``[r, r+world, ...]``
    (strided). So master piece k came from rank ``k % world``'s ``(k // world)``-th sub-piece.
    stride==1 (the common case) collapses to a plain concat in rank order.
    """
    import torch.distributed as dist

    world = dist.get_world_size(tp_group)
    if world == 1:
        return shard
    shard = shard.contiguous()
    gathered = [torch.empty_like(shard) for _ in range(world)]
    dist.all_gather(gathered, shard, group=tp_group)
    if stride == 1:
        return torch.cat(gathered, dim=dim)
    subs = [torch.chunk(gathered[r], stride, dim=dim) for r in range(world)]  # [world][stride]
    pieces = [None] * (world * stride)
    for r in range(world):
        for j in range(stride):
            pieces[r + j * world] = subs[r][j]
    return torch.cat(pieces, dim=dim)


def _split_full_to_shard(
    full: torch.Tensor, dim: int, stride: int, engine_tp: int, engine_tp_rank: int
) -> torch.Tensor:
    """Inverse of _gather_tp_to_full: take engine-TP-rank ``engine_tp_rank``'s shard of ``full`` along
    ``dim``, matching Megatron's split-with-stride convention (rank r owns full-pieces [r, r+W, ...]).
    engine_tp==1 -> the whole tensor. Used to reshard a trainer weight to a SMALLER (but >1) engine TP,
    so a 35B engine that cannot hold the full model still receives its correct per-rank shard."""
    if engine_tp == 1:
        return full
    pieces = torch.chunk(full, engine_tp * stride, dim=dim)  # W*stride pieces
    mine = [pieces[engine_tp_rank + j * engine_tp] for j in range(stride)]
    return torch.cat(mine, dim=dim).contiguous()


def _gdn_config_dims(cfg):
    """(qk_dim, v_dim, num_value_heads) for a GatedDeltaNet config, or None if not GDN.

    Read from the Megatron TransformerConfig (the GDN module itself does
    ``self.qk_dim = config.linear_key_head_dim * config.linear_num_key_heads`` etc.)."""
    if cfg is None:
        return None
    nk = getattr(cfg, "linear_num_key_heads", None)
    nv = getattr(cfg, "linear_num_value_heads", None)
    kd = getattr(cfg, "linear_key_head_dim", None)
    vd = getattr(cfg, "linear_value_head_dim", None) or kd
    if not (nk and nv and kd):
        return None
    return nk * kd, nv * vd, nv


def _gdn_segments_full(name: str, dims):
    """Full-tensor segment sizes along partition_dim for a fused GDN param, or None.

    GatedDeltaNet ``in_proj`` and ``conv1d`` concatenate several INDEPENDENTLY-TP-sharded segments
    along dim 0 -- in_proj is qkvzba = [qk, qk, v, v, nvh, nvh]; conv1d is qkv = [qk, qk, v] (see
    ``GatedDeltaNet.forward``'s ``split_sections``). So rank r's shard is
    ``[seg0/tp, seg1/tp, ...]`` and a plain gather (concat rank0 ++ rank1) INTERLEAVES the segments,
    producing a full-width tensor of the right shape but scrambled rows -> engine gibberish. These
    must be gathered/split PER SEGMENT. (out_proj is row-parallel single-segment; A_log/dt_bias are
    per-head single-segment -- all fine with the plain path.)"""
    if dims is None:
        return None
    qk, v, nvh = dims
    if name.endswith("self_attention.in_proj.weight") or name.endswith("self_attention.in_proj.bias"):
        return [qk, qk, v, v, nvh, nvh]
    if name.endswith("self_attention.conv1d.weight") or name.endswith("self_attention.conv1d.bias"):
        return [qk, qk, v]
    return None


def _gather_segmented_to_full(shard: torch.Tensor, seg_full: list, dim: int, tp_group) -> torch.Tensor:
    """All-gather a segment-fused shard and rebuild the full tensor, reconstructing EACH segment
    across ranks separately (the segments were sharded independently, then concatenated per rank)."""
    import torch.distributed as dist

    world = dist.get_world_size(tp_group)
    if world == 1:
        return shard
    shard = shard.contiguous()
    gathered = [torch.empty_like(shard) for _ in range(world)]
    dist.all_gather(gathered, shard, group=tp_group)
    seg_local = [s // world for s in seg_full]
    per_rank = [torch.split(g, seg_local, dim=dim) for g in gathered]  # [world][nseg]
    out = [torch.cat([per_rank[r][si] for r in range(world)], dim=dim) for si in range(len(seg_full))]
    return torch.cat(out, dim=dim)


def _split_full_segmented(
    full: torch.Tensor, seg_full: list, dim: int, engine_tp: int, engine_tp_rank: int
) -> torch.Tensor:
    """Inverse of _gather_segmented_to_full: take engine-rank ``engine_tp_rank``'s contiguous slice of
    EACH segment and concat -- exactly the layout the engine's own GDN build shards to."""
    if engine_tp == 1:
        return full
    out = []
    for s in torch.split(full, seg_full, dim=dim):
        local = s.shape[dim] // engine_tp
        out.append(s.narrow(dim, engine_tp_rank * local, local))
    return torch.cat(out, dim=dim).contiguous()


def _reshard_to_full(
    name: str,
    p: torch.Tensor,
    tp_group,
    trainer_tp: int,
    engine_tp: int,
    vocab_size: int,
    make_div_by: int,
    engine_tp_rank: int = 0,
    seg_full: list = None,
) -> torch.Tensor:
    """Reshard one param from the trainer's TP layout to the engine's, or pass it through if replicated.

    ``p`` MUST be the original Parameter, not a ``.detach()``'d copy: Megatron's TP attributes
    (tensor_model_parallel / partition_dim / partition_stride) live on the Parameter and are lost by
    ``.detach()``, which would make every param look replicated and silently skip the gather."""
    is_parallel = bool(getattr(p, "tensor_model_parallel", False))
    if not is_parallel:
        return p.detach()  # norms / router / biases are replicated -- already full on every rank
    dim = int(getattr(p, "partition_dim", 0))
    stride = int(getattr(p, "partition_stride", 1))
    if seg_full is not None:
        # Fused GDN param: gather and re-split PER SEGMENT so independently sharded rows do not interleave.
        full = _gather_segmented_to_full(p.detach(), seg_full, dim, tp_group)
        return _split_full_segmented(full, seg_full, dim, engine_tp, engine_tp_rank)
    full = _gather_tp_to_full(p.detach(), dim, stride, tp_group)
    if _is_vocab_param(name) and dim == 0:
        # vocab padding differs with TP; slice the trainer's (larger) padding down to what the
        # engine (engine_tp) pads to, so the shape matches the engine's own output_layer/embedding.
        keep = _vocab_padded(vocab_size, engine_tp, make_div_by)
        if full.shape[0] > keep:
            full = full[:keep].contiguous()
    # Re-split to the engine's TP shard. `engine_tp_rank = global_rank % engine_tp` assumes each
    # engine's TP ranks sit on contiguous GPUs in rank order; other placements would send wrong shards.
    return _split_full_to_shard(full, dim, stride, engine_tp, engine_tp_rank)


def native_full_shape(
    name: str, p: torch.Tensor, trainer_tp: int, engine_tp: int, vocab_size: int, make_div_by: int
) -> list[int]:
    """Analytic shape of a param after resharding to the engine's TP layout, with no collective (so it
    is safe on rank 0 alone). Must mirror the shapes ``_reshard_to_full`` actually produces."""
    shape = list(p.shape)
    if not bool(getattr(p, "tensor_model_parallel", False)):
        return shape
    dim = int(getattr(p, "partition_dim", 0))
    shape[dim] = shape[dim] * trainer_tp
    if _is_vocab_param(name) and dim == 0:
        shape[0] = min(shape[0], _vocab_padded(vocab_size, engine_tp, make_div_by))
    shape[dim] = shape[dim] // engine_tp  # engine holds only its 1/engine_tp shard (sender-split)
    return shape


def _model_vocab_cfg(inner) -> tuple[int, int]:
    """(vocab_size, make_vocab_size_divisible_by) from a GPTModel, with safe defaults."""
    cfg = getattr(inner, "config", None)
    vocab = getattr(inner, "vocab_size", None) or getattr(cfg, "vocab_size", 0) or 0
    mdiv = getattr(cfg, "make_vocab_size_divisible_by", 128) or 128
    return int(vocab), int(mdiv)


def extract_native_weights(
    actor_module,
    *,
    dtype: torch.dtype = torch.bfloat16,
    include_buffers: bool = False,
    reshard: bool = False,
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Yield ``(native_name, tensor)`` from a Megatron GPTModel in native layout, cast to ``dtype``.

    ``actor_module`` may be a single module or a list of PP/vpp chunks. ``reshard=True`` performs an
    all-gather collective, so it must run on every trainer TP rank -- call it only from the transport
    path, never from rank-0-only metadata (use ``native_resharded_metadata`` there).
    """
    modules = actor_module if isinstance(actor_module, (list, tuple)) else [actor_module]
    seen = set()
    trainer_tp, tp_group = _megatron_tp()
    do_reshard = reshard and _needs_reshard(trainer_tp)
    engine_tp = engine_tp_target() or 1
    # Which engine-TP-rank's shard THIS trainer rank must produce: colocated placement counts ranks
    # over CUDA_VISIBLE_DEVICES in order, so the engine worker on this GPU is global_rank % engine_tp.
    import torch.distributed as _distm

    _grank = _distm.get_rank() if _distm.is_initialized() else 0
    engine_tp_rank = _grank % engine_tp if engine_tp > 1 else 0
    for m in modules:
        # Fully unwrap DDP/Float16Module/etc to the bare GPTModel so names carry no `module.` prefix.
        inner = m
        for _ in range(4):
            if hasattr(inner, "module"):
                inner = inner.module
            else:
                break
        vocab, mdiv = _model_vocab_cfg(inner) if do_reshard else (0, 128)
        gdn_dims = _gdn_config_dims(getattr(inner, "config", None)) if do_reshard else None
        ep_size, ep_rank, ep_group = _megatron_ep() if reshard else (1, 0, None)
        do_ep_remap = reshard and _needs_ep_remap(ep_size)
        # Is an expert param WHOLE on this rank (ETP=1)? Only then can the EP branch pre-slice it
        # straight to the engine's shard; at ETP>1 it is itself a shard needing a gather-then-resplit.
        trainer_tp_expert_is_whole = False
        if do_ep_remap:
            try:
                from megatron.core import parallel_state as _mpu_e

                trainer_tp_expert_is_whole = (
                    _mpu_e.model_parallel_is_initialized() and _mpu_e.get_expert_tensor_parallel_world_size() == 1
                )
            except Exception:
                trainer_tp_expert_is_whole = False
        num_local_experts = None
        if do_ep_remap:
            cfg = getattr(inner, "config", None)
            total_e = getattr(cfg, "num_moe_experts", None) or 0
            num_local_experts = (total_e // ep_size) if total_e else None
        for name, p in inner.named_parameters():
            # EP remap+gather: emit every EP rank's copy of this local expert under its global name.
            if do_ep_remap and num_local_experts and _expert_local_idx(name) is not None:
                import torch.distributed as _dist

                li = _expert_local_idx(name)
                gk = _global_expert_name(name, li, 0, num_local_experts)  # canonical dedup key (src rank 0)
                if gk in seen:
                    continue
                shard = p.detach().contiguous()
                bucket = [torch.empty_like(shard) for _ in range(ep_size)]
                _dist.all_gather(bucket, shard, group=ep_group)
                # Pre-slice to the engine's shard so the transport moves 1/engine_tp of the bytes the
                # receiver would otherwise slice away. Same arithmetic, and the same rank mapping every
                # dense param ships under. Only valid when the expert is not itself trainer-TP-sharded.
                _exp_pre_slice = (
                    engine_tp > 1 and trainer_tp_expert_is_whole and bool(getattr(p, "tensor_model_parallel", False))
                )
                _edim = int(getattr(p, "partition_dim", 0))
                _estride = int(getattr(p, "partition_stride", 1))
                for s in range(ep_size):
                    gname = _global_expert_name(name, li, s, num_local_experts)
                    seen.add(gname)
                    t = _to_full(bucket[s])
                    if _exp_pre_slice and t.shape[_edim] % (engine_tp * _estride) == 0:
                        t = _split_full_to_shard(t, _edim, _estride, engine_tp, engine_tp_rank)
                    t = t.to(dtype)
                    yield gname, t
                continue
            if name in seen:
                continue
            seen.add(name)
            # Reshard from the ORIGINAL param `p`: TP attributes live on it and .detach() strips them.
            if do_reshard:
                seg_full = _gdn_segments_full(name, gdn_dims)
                src = _reshard_to_full(
                    name, p, tp_group, trainer_tp, engine_tp, vocab, mdiv, engine_tp_rank, seg_full=seg_full
                )
            else:
                src = p.detach()
            t = _to_full(src).to(dtype)
            yield name, t
        for name, b in inner.named_buffers():
            if name in seen or b is None:
                continue
            allowed = is_synced_buffer(name)
            if not (include_buffers or allowed):
                continue
            seen.add(name)
            # Native dtype for an allowlisted buffer, bf16 only for the bulk `include_buffers` dump:
            # bf16-rounding a routing bias can flip a top-k boundary and pick a different expert.
            yield name, _to_full(b.detach()) if allowed else _to_full(b.detach()).to(dtype)


def native_resharded_metadata(actor_module, *, dtype: torch.dtype = torch.bfloat16):
    """Yield ``(name, dtype_name, full_shape)`` for the resharded (engine-TP) layout, analytically.

    No collective and no tensor materialization, so it is safe on rank 0 alone. Shapes must match what
    ``extract_native_weights(reshard=True)`` actually sends."""
    dtype_name = str(dtype).split(".")[-1]
    trainer_tp, _ = _megatron_tp()
    do_reshard = _needs_reshard(trainer_tp)
    engine_tp = engine_tp_target() or 1
    ep_size, ep_rank, _ = _megatron_ep()
    do_ep_remap = _needs_ep_remap(ep_size)
    modules = actor_module if isinstance(actor_module, (list, tuple)) else [actor_module]
    seen = set()
    # Mirror extract_native_weights' expert pre-slice: the metadata must advertise the shape actually
    # shipped, or the receiver's expected shapes and the sender's stream disagree.
    expert_is_whole = False
    if do_ep_remap:
        try:
            from megatron.core import parallel_state as _mpu_e

            expert_is_whole = (
                _mpu_e.model_parallel_is_initialized() and _mpu_e.get_expert_tensor_parallel_world_size() == 1
            )
        except Exception:
            expert_is_whole = False
    for m in modules:
        inner = m
        for _ in range(4):
            inner = inner.module if hasattr(inner, "module") else inner
            if not hasattr(inner, "module"):
                break
        vocab, mdiv = _model_vocab_cfg(inner) if do_reshard else (0, 128)
        num_local_experts = None
        if do_ep_remap:
            total_e = getattr(getattr(inner, "config", None), "num_moe_experts", None) or 0
            num_local_experts = (total_e // ep_size) if total_e else None
        for name, p in inner.named_parameters():
            # EP remap: emit one metadata entry per GLOBAL expert (all ep_size sources), same shape.
            if do_ep_remap and num_local_experts and _expert_local_idx(name) is not None:
                li = _expert_local_idx(name)
                gk = _global_expert_name(name, li, 0, num_local_experts)
                if gk in seen:
                    continue
                eshape = list(p.shape)
                _edim = int(getattr(p, "partition_dim", 0))
                _estride = int(getattr(p, "partition_stride", 1))
                if (
                    engine_tp > 1
                    and expert_is_whole
                    and bool(getattr(p, "tensor_model_parallel", False))
                    and eshape[_edim] % (engine_tp * _estride) == 0
                ):
                    eshape[_edim] //= engine_tp
                for s in range(ep_size):
                    gname = _global_expert_name(name, li, s, num_local_experts)
                    seen.add(gname)
                    yield gname, dtype_name, list(eshape)
                continue
            if name in seen:
                continue
            seen.add(name)
            shape = native_full_shape(name, p, trainer_tp, engine_tp, vocab, mdiv) if do_reshard else list(p.shape)
            yield name, dtype_name, shape
        # Allowlisted buffers ride the same transport and must be advertised here too, at their NATIVE
        # dtype rather than the transport dtype.
        for name, b in inner.named_buffers():
            if name in seen or b is None or not is_synced_buffer(name):
                continue
            seen.add(name)
            yield name, str(b.dtype).split(".")[-1], list(b.shape)


def load_native_weights(target_module, weights_iter, *, strict: bool = True) -> set[str]:
    """Copy ``(native_name, tensor)`` pairs into ``target_module`` in place, returning the loaded names.

    The target is a GPTModel built with an identical spec, so names and shapes match 1:1 and this is a
    straight ``copy_`` with no repack.
    """
    modules = target_module if isinstance(target_module, (list, tuple)) else [target_module]
    dst, dst_bufs = {}, {}
    for m in modules:
        inner = m.module if hasattr(m, "module") else m
        dst.update(dict(inner.named_parameters()))
        dst_bufs.update(dict(inner.named_buffers()))
    loaded = set()
    for name, tensor in weights_iter:
        dest = dst.get(name, dst_bufs.get(name))
        if dest is None:
            if strict:
                raise KeyError(f"[isoexec] native weight name not in target model: {name}")
            continue
        d = _to_full(dest)
        if tuple(d.shape) != tuple(tensor.shape):
            raise ValueError(f"[isoexec] shape mismatch for {name}: {tuple(d.shape)} vs {tuple(tensor.shape)}")
        with torch.no_grad():
            dest.copy_(tensor.to(dest.dtype))
        loaded.add(name)
    missing = set(dst) - loaded
    if strict and missing:
        raise KeyError(f"[isoexec] {len(missing)} target params not synced, e.g. {sorted(missing)[:3]}")
    logger.info("[isoexec] native weight sync: copied %d tensors (no HF conversion)", len(loaded))
    return loaded
