"""Debug-mode hook installation: wrap installed region impls with digest capture.

Reaches the regions the way the adapters do -- by rebinding the module attribute or class
attribute that holds the *currently installed* impl -- so whatever kernel each side actually runs
(IsoExec's or native) is what gets traced. Nothing here edits adapter or worker files; the call
sites named in ``INTEGRATION.md`` invoke :func:`install_debug_hooks` after their normal isoexec
install, which is why every door below is the post-install binding.

Everything is a no-op unless ``SKYRL_ISOEXEC_DEBUG_TRACE`` is set. Idempotent; safe to call more
than once and safe to call when megatron / vLLM / the GDN ops are not importable in this process
-- a door whose module is absent is skipped and reported, never raised.

:data:`DOORS` is the table: one row per (region, door), where a door is the exact attribute that
gets rebound. :data:`NOT_HOOKED` records, for every registry region with no door, WHY. Between
them the two cover all 22 registry regions; :func:`coverage` returns that as data and
:func:`install_debug_hooks` prints it.

Two rules make a multi-region table safe to install:

  * The door must be the SAME mathematical point on both sides, or the comparator is comparing
    different quantities and calls a clean run divergent. ``gdn.core`` is the cautionary case:
    the engine used to be hooked at ``GatedDeltaNet.forward`` (post-``out_proj``, ``[T,1,2048]``)
    while the trainer was hooked at the kernel door (``[1,T,H,D]``) -- never comparable. Both
    sides now hook the kernel door, and the layer index comes from
    :func:`install_layer_context_hooks`, which publishes the running layer's index around the
    whole decoder-layer forward instead of recording a second, different tensor.
  * The wrapper must be transparent. ``trace.wrap_region`` copies ``_isoexec_*`` capability
    markers and sets ``__wrapped__``; doors whose installer asserts on the *identity* of the live
    binding (``rope.rope``) are listed in :data:`NOT_HOOKED` rather than wrapped.
"""

from __future__ import annotations

import importlib
import sys
from typing import Callable, Dict, List, Optional, Tuple

import torch

from . import trace

_wrapper_cache: Dict[Tuple[str, int], Callable] = {}

_IX = "skyrl.backends.skyrl_train.isoexec"


def _shared_wrap(region: str, fn: Callable, **kw) -> Callable:
    """One wrapper per (region, underlying function), so multi-namespace rebinding shares one."""
    if getattr(fn, trace._WRAP_ATTR, None) is not None:
        return fn
    key = (region, id(fn))
    w = _wrapper_cache.get(key)
    if w is None or getattr(w, "_isoexec_debug_inner", None) is not fn:
        w = trace.wrap_region(region, fn, **kw)
        _wrapper_cache[key] = w
    return w


# -- case labels ------------------------------------------------------------------------------


def _engine_batch_case() -> Optional[str]:
    """decode / prefill / mixed from vLLM's forward context -- the batch composition itself.

    ``attn_metadata`` is a dict keyed by layer prefix; every entry describes the same batch, so
    any one of them answers the question. This is the only honest signal at the kernel door: the
    door's own arguments cannot tell "no ``cu_seqlens`` because this is decode" from "no
    ``cu_seqlens`` because this caller does not pass it".
    """
    try:
        from vllm.forward_context import get_forward_context

        md = get_forward_context().attn_metadata
    except Exception:  # noqa: BLE001 -- no vLLM, or called outside a forward context
        return None
    if isinstance(md, dict):
        md = next(iter(md.values()), None)
    nd, npf = getattr(md, "num_decodes", None), getattr(md, "num_prefills", None)
    if nd is None and npf is None:
        return None
    if nd and npf:
        return "engine_mixed"
    if nd:
        return "engine_decode"
    if npf:
        return "engine_prefill"
    return None


def _case(args, kwargs, out) -> str:
    """Region case label. Trainer: fwd vs score (grad mode). Engine: the batch composition."""
    tr = trace.get_tracer()
    if tr is None:
        return "unknown"
    if tr.side != "engine":
        return tr.default_case()
    ctx = _engine_batch_case()
    if ctx is not None:
        return ctx
    # Structural fallback: the varlen kernels take a real cu_seqlens, the state kernels are
    # called with an explicit cu_seqlens=None alongside ssm_state. Weaker than the forward
    # context (it reads the call site, not the batch), so it never overrides it.
    cu = kwargs.get("cu_seqlens")
    if isinstance(cu, torch.Tensor) and cu.numel() >= 2:
        return "engine_prefill"
    if "cu_seqlens" in kwargs and cu is None and "ssm_state" in kwargs:
        return "engine_decode"
    return "engine"


_gdn_case = _case  # kept: the historical name for the gdn.core case function


def _out_kwarg(name: str) -> Callable:
    """Digest the buffer a void door filled, since it returns None."""

    def pick(args, kwargs, out):
        return out if out is not None else kwargs.get(name)

    return pick


def _first(args, kwargs, out):
    """Doors that return ``(tensor, bias|None)``: the bias is a Parameter, not a region output."""
    return out[0] if isinstance(out, tuple) and out else out


# -- the door table ---------------------------------------------------------------------------
#
# (region, module path, attribute, import_ok, kwargs for wrap_region)
# ``attribute`` may be "Class.method"; ``import_ok`` False means "wrap it only if the module is
# already imported" -- used for vLLM, which must never be imported into a trainer process.

DOORS: Tuple[Tuple[str, str, str, bool, dict], ...] = (
    # -- attention -------------------------------------------------------------------------
    ("attention.varlen", f"{_IX}.ops.attention.megatron_varlen_attn", "TorchVarlenCoreAttn.forward", True, {}),
    ("attention.varlen", f"{_IX}.runtimes.vllm.gptmodel_vllm", "MegatronCoreAttnToVLLM.forward", False, {}),
    # -- collectives -----------------------------------------------------------------------
    (
        "collectives.row_parallel",
        "megatron.core.tensor_parallel.layers",
        "RowParallelLinear.forward",
        True,
        {"out_fn": _first},
    ),
    ("collectives.tree_all_reduce", f"{_IX}.ops.collectives.pik.allreduce", "tree_all_reduce", True, {}),
    ("collectives.tree_all_reduce", f"{_IX}.ops.collectives.pik.linear", "tree_all_reduce", True, {}),
    # -- gdn -------------------------------------------------------------------------------
    ("gdn.conv", f"{_IX}.ops.gdn.gdn_ops", "gdn_causal_conv", True, {}),
    ("gdn.conv", f"{_IX}.ops.gdn.gdn_ops", "gdn_causal_conv_batched", True, {}),
    ("gdn.conv", f"{_IX}.ops.gdn.gdn_ops", "gdn_native_conv", True, {}),
    ("gdn.conv", f"{_IX}.ops.gdn.gdn_recurrent_state", "gdn_causal_conv", True, {}),
    ("gdn.conv", f"{_IX}.ops.gdn.gdn_cpr_state", "gdn_causal_conv", True, {}),
    ("gdn.core", f"{_IX}.ops.gdn.gdn_ops", "gdn_core", True, {}),
    ("gdn.core", f"{_IX}.ops.gdn.gdn_ops", "gdn_native_core", True, {}),
    ("gdn.core", f"{_IX}.ops.gdn.gdn_ops", "gdn_native_cpr", True, {}),
    ("gdn.core", f"{_IX}.ops.gdn.gdn_ops", "gdn_recurrent_kernel", True, {}),
    ("gdn.core", f"{_IX}.ops.gdn.gdn_ops", "gdn_native_core_kernel", True, {}),
    ("gdn.core", f"{_IX}.ops.gdn.gdn_recurrent_state", "gdn_core", True, {}),
    ("gdn.core", f"{_IX}.ops.gdn.gdn_recurrent_state", "gdn_recurrent_kernel", True, {}),
    ("gdn.core", f"{_IX}.ops.gdn.gdn_recurrent_state", "gdn_native_core_kernel", True, {}),
    ("gdn.core", f"{_IX}.ops.gdn.gdn_cpr", "gdn_native_cpr", True, {}),
    ("gdn.core", f"{_IX}.ops.gdn.gdn_cpr_state", "gdn_native_cpr", True, {}),
    ("gdn.gating", f"{_IX}.ops.gdn.gdn_ops", "gdn_gate_and_beta", True, {}),
    ("gdn.gating", f"{_IX}.ops.gdn.gdn_recurrent_state", "gdn_gate_and_beta", True, {}),
    ("gdn.gating", f"{_IX}.ops.gdn.gdn_cpr", "native_matched_prep", True, {}),
    ("gdn.gating", "megatron.core.ssm.gated_delta_net", "GatedDeltaNet._compute_g_and_beta", True, {}),
    ("gdn.l2norm", f"{_IX}.ops.gdn.gdn_ops", "gdn_l2norm", True, {}),
    ("gdn.l2norm", f"{_IX}.ops.gdn.gdn_recurrent_state", "gdn_l2norm", True, {}),
    ("gdn.state", f"{_IX}.ops.gdn.gdn_cpr", "chunk_boundary_states", True, {}),
    ("gdn.state", f"{_IX}.ops.gdn.gdn_cpr_state", "entry_read", True, {}),
    # -- logprobs --------------------------------------------------------------------------
    (
        "logprobs.lm_head_slice",
        f"{_IX}.runtimes.vllm.gptmodel_vllm",
        "GPTModelVLLMWrapper.compute_logits",
        False,
        {},
    ),
    (
        "logprobs.log_softmax",
        "skyrl.backends.skyrl_train.distributed.megatron.model_utils",
        "_ix_logprobs_apply",
        True,
        {},
    ),
    ("logprobs.log_softmax", "vllm.v1.sample.sampler", "Sampler.compute_logprobs", False, {}),
    # -- mm (allow-listed off by default: fires on every projection of every layer) ---------
    ("mm", "vllm.model_executor.layers.batch_invariant", "matmul_persistent", False, {}),
    # -- moe -------------------------------------------------------------------------------
    ("moe.blockmap", f"{_IX}.ops.moe.moe_fused_experts", "_block_map", True, {}),
    ("moe.combine", "megatron.core.transformer.moe.moe_utils", "unpermute", True, {}),
    ("moe.combine", "megatron.core.transformer.moe.token_dispatcher", "unpermute", True, {}),
    (
        "moe.dispatch",
        "megatron.core.transformer.moe.token_dispatcher",
        "MoEAllGatherTokenDispatcher.dispatch_postprocess",
        True,
        {},
    ),
    (
        "moe.epilogue",
        f"{_IX}.ops.moe.moe_epilogue_kernel",
        "invoke_fused_moe_fc1_glu_kernel",
        True,
        {"out_fn": _out_kwarg("C")},
    ),
    ("moe.experts", "megatron.core.transformer.moe.experts", "SequentialMLP.forward", True, {"out_fn": _first}),
    ("moe.router", "megatron.core.transformer.moe.moe_utils", "topk_routing_with_score_function", True, {}),
    ("moe.router", "megatron.core.transformer.moe.router", "topk_routing_with_score_function", True, {}),
    ("moe.router", "megatron.core.transformer.moe.token_dispatcher", "topk_routing_with_score_function", True, {}),
    ("moe.router", "megatron.core.transformer.moe.router", "TopKRouter.routing", True, {}),
    ("moe.weights", f"{_IX}.ops.moe.moe_fused_weights", "fused_expert_weights", True, {}),
    # -- norms -----------------------------------------------------------------------------
    ("norms.gated_out", "megatron.core.ssm.gated_delta_net", "GatedDeltaNet._apply_gated_norm", True, {}),
    ("rope.rope", "megatron.core.models.common.embeddings.rope_utils", "_apply_rotary_pos_emb_bshd", True, {}),
    ("norms.gated_out", f"{_IX}.ops.norms.fused_outnorm", "fused_gated_out_norm", True, {}),
    ("norms.rms", f"{_IX}.ops.norms.zero_centered_norm", "ZeroCenteredTorchRMSNorm.forward", True, {}),
    ("norms.rms", f"{_IX}.ops.norms.fused_outnorm", "fused_rms_norm_gamma", True, {}),
)

# Region -> why there is no door. Read together with DOORS this covers all 22 registry regions.
NOT_HOOKED: Dict[str, str] = {
    "collectives.nccl_pin": (
        "no callable: all five impl ids are an environment pin (NCCL_ALGO / MIN_NCHANNELS / "
        "MAX_NCHANNELS, ops/collectives/nccl_identity.py) read by NCCL itself. The only nearby "
        "rebinds are side-effect-only c10d collectives that return None, so there is nothing "
        "whose output could be digested."
    ),
}

# Regions whose door only partially covers the op, recorded so coverage never overstates itself.
PARTIAL: Dict[str, str] = {
    "gdn.state": (
        "read side only (gdn_cpr.chunk_boundary_states, gdn_cpr_state.entry_read). The state is "
        "mutated IN PLACE in the private pool / vLLM mamba kv-cache; every write door returns "
        "None or a row count, so there is no value to digest at the write."
    ),
    "gdn.gating": "subsumed by the native fused core when gdn.core=native_fused_sigmoid (registry subsumption).",
    "gdn.l2norm": "subsumed by the native fused core when gdn.core=native_fused_sigmoid (registry subsumption).",
    "norms.rms": (
        "engine shadows the class method with a per-instance m.forward (ops/norms/fused_outnorm.py "
        "install_engine_fused_norms), so the class-attribute door sees trainer calls plus the "
        "engine's fused_rms_norm_gamma door, not the engine's per-instance bindings."
    ),
    "logprobs.lm_head_slice": "engine only, and a pure passthrough unless SPLIT_LM_HEAD is on.",
    "mm": "hooked but OFF by default -- see trace.HIGH_VOLUME_REGIONS; enable with SKYRL_ISOEXEC_DEBUG_REGIONS=+mm.",
}


def _resolve(module_path: str, attr: str, import_ok: bool):
    """(holder, name, current value) for a module or ``Class.method`` attribute, or None."""
    mod = sys.modules.get(module_path)
    if mod is None:
        if not import_ok:
            return None
        try:
            mod = importlib.import_module(module_path)
        except Exception:  # noqa: BLE001 -- an absent runtime is a skipped door, not a failure
            return None
    holder, name = mod, attr
    if "." in attr:
        cls_name, name = attr.split(".", 1)
        holder = getattr(mod, cls_name, None)
        if holder is None:
            return None
    if isinstance(holder, type):
        # THROUGH __dict__, not getattr: getattr runs the descriptor protocol, so a staticmethod
        # comes back as a plain function and _install_door re-binds it as an ordinary method --
        # vLLM's `Sampler.compute_logprobs` then gets `self` as its first positional argument and
        # the engine dies at the first sample. The MRO walk keeps inherited doors
        # resolvable; setattr still lands on `holder`.
        cur = next((k.__dict__[name] for k in holder.__mro__ if name in k.__dict__), None)
    else:
        cur = getattr(holder, name, None)
    return (holder, name, cur) if cur is not None else None


def _install_door(region: str, module_path: str, attr: str, import_ok: bool, kw: dict) -> bool:
    res = _resolve(module_path, attr, import_ok)
    if res is None:
        return False
    holder, name, cur = res
    raw = cur.__func__ if isinstance(cur, (staticmethod, classmethod)) else cur
    if not callable(raw) or getattr(raw, trace._WRAP_ATTR, None) is not None:
        return False
    wrapped = _shared_wrap(region, raw, case_fn=_case, **kw)
    if isinstance(cur, staticmethod):
        wrapped = staticmethod(wrapped)
    elif isinstance(cur, classmethod):
        wrapped = classmethod(wrapped)
    try:
        setattr(holder, name, wrapped)
    except Exception:  # noqa: BLE001 -- a read-only binding is a skipped door, not a failure
        return False
    return True


def coverage() -> Dict[str, dict]:
    """Region -> {status, doors, note}: what this table covers and what it deliberately does not."""
    out: Dict[str, dict] = {}
    for region, mod, attr, _imp, _kw in DOORS:
        ent = out.setdefault(region, {"status": "hooked", "doors": [], "note": PARTIAL.get(region)})
        ent["doors"].append(f"{mod}:{attr}")
        if region in PARTIAL:
            ent["status"] = "partial"
    for region, why in NOT_HOOKED.items():
        out[region] = {"status": "not_hooked", "doors": [], "note": why}
    return out


def _decoder_layers(obj) -> List:
    """Every decoder layer reachable from ``obj``: a module, or an iterable of them.

    The megatron worker's ``model_fn`` returns a LIST of GPTModel chunks (virtual pipeline), which
    has no ``.decoder`` -- the trainer got zero layer-context hooks and fell back to
    ``layer_src="call_order"``, which does not align with the engine's module indices. Each
    chunk's layers carry megatron's global ``layer_number``, so a flat walk is correct.
    """
    if obj is None:
        return []
    # Unwrap the .module chain first: a chunk usually arrives inside DistributedDataParallel and
    # Float16Module, neither of which forwards `.decoder`.
    inner, depth = obj, 0
    while inner is not None and depth < 8:
        layers = getattr(getattr(inner, "decoder", None), "layers", None)
        if layers is not None:
            return list(layers)
        inner, depth = getattr(inner, "module", None), depth + 1
    if isinstance(obj, torch.nn.Module):
        return []
    try:
        chunks = list(obj)
    except TypeError:  # not a module and not iterable: nothing to walk
        return []
    out: List = []
    for chunk in chunks:
        out.extend(_decoder_layers(chunk))
    return out


def install_layer_context_hooks(gpt_modules) -> int:
    """Publish the running decoder layer's index to every region door inside its forward.

    ``gpt_modules`` may be one GPTModel or the list of virtual-pipeline chunks the megatron worker
    holds; see :func:`_decoder_layers`.

    Call site: right after the isoexec install on each side -- the engine's ``swap_gdn_core(...)``
    and, once the trainer passes its model to :func:`install_debug_hooks`, the trainer's model
    build. This wraps ``layer.forward`` with a thread-local ``layer_context`` only; it records
    nothing itself, which is the point: the layer index reaches the kernel door instead of a
    second, differently shaped tensor being recorded next to it.

    ``layer_number`` is megatron's 1-based index INCLUDING the pipeline offset, so ``- 1`` is a
    global layer id and trainer (PP>1) and engine (PP=1) records align.
    """
    if not trace.enabled():
        return 0
    n = 0
    for i, layer in enumerate(_decoder_layers(gpt_modules)):
        cur = getattr(layer, "forward", None)
        if cur is None or getattr(cur, "_isoexec_debug_layer_ctx", None) is not None:
            continue
        num = getattr(layer, "layer_number", None)
        idx = num - 1 if isinstance(num, int) else i

        def wrapped(*args, _fn=cur, _idx=idx, **kwargs):
            with trace.layer_context(_idx):
                return _fn(*args, **kwargs)

        wrapped._isoexec_debug_layer_ctx = idx
        wrapped.__name__ = getattr(cur, "__name__", "forward")
        wrapped.__wrapped__ = cur
        try:
            layer.forward = wrapped
        except Exception:  # noqa: BLE001 -- a frozen module is a skipped layer, not a failure
            continue
        n += 1
    return n


# Historical name; the engine call site in runtimes/vllm/gptmodel_vllm.py still calls this. It no
# longer records post-out_proj tensors -- that door was not comparable with the trainer's.
install_gdn_layer_hooks = install_layer_context_hooks


def install_debug_hooks(model=None) -> int:
    """Install every region door reachable in this process. Returns bindings wrapped (0 = off).

    ``model`` (optional) is this process's GPTModel; passing it installs the layer-index context
    on both sides. Without it the trainer falls back to ``layer_src="call_order"``.
    """
    if not trace.enabled():
        return 0
    tr = trace.get_tracer()
    n = 0
    hooked: Dict[str, int] = {}
    for region, mod, attr, import_ok, kw in DOORS:
        if not tr.wants(region):
            continue
        if _install_door(region, mod, attr, import_ok, kw):
            n += 1
            hooked[region] = hooked.get(region, 0) + 1
    tr.regions_hooked.update(hooked)
    nlayer = install_layer_context_hooks(model) if model is not None else 0
    from . import thash

    backend = "unknown"
    try:
        thash.preload()  # compile now, not inside the first traced forward
        backend = thash.digest_backend(_probe_device())
    except Exception as e:  # noqa: BLE001 -- JIT on first use is a slowdown, not a failure
        backend = f"unprobed ({type(e).__name__})"
    cov = coverage()
    off = sorted(r for r in cov if r not in hooked)
    print(
        f"[ISOEXEC-DEBUG] tracing ON: {n} binding(s) over {len(hooked)} region(s) wrapped, "
        f"{nlayer} layer-context hook(s), side={tr.side}, rank={tr.rank} (via {tr.rank_src}), "
        f"sample=1/{tr.sample}, ladder={'on' if tr.ladder else 'off'}, "
        f"segments={tr.segment_rows or 'off'}, digest={backend} "
        f"-> {tr.path}\n[ISOEXEC-DEBUG] regions live: {sorted(hooked)}\n"
        f"[ISOEXEC-DEBUG] regions with no door here: {off}",
        flush=True,
    )
    return n


def _probe_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _install_moe_router() -> int:
    """Back-compat shim for the pre-table API; installs just the moe.router doors."""
    return sum(_install_door(r, m, a, i, k) for r, m, a, i, k in DOORS if r == "moe.router")


def _install_gdn_core() -> int:
    """Back-compat shim for the pre-table API; installs just the gdn.core doors."""
    return sum(_install_door(r, m, a, i, k) for r, m, a, i, k in DOORS if r == "gdn.core")


def smoke_install_report() -> List[Tuple[str, str, str, bool]]:
    """(region, module, attr, resolved) for every door -- the import-only arming check."""
    return [(r, m, a, _resolve(m, a, i) is not None) for r, m, a, i, _k in DOORS]
