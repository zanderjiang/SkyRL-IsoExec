"""Integrate the canonical IsoExec GDN state machine with vLLM.

Routes GDN model calls through the selected IsoExec core and supplies scheduler-owned lifecycle
metadata. vLLM-visible Mamba pages are slot identities for private state, not state storage, and
engine args that would advance that state on their own terms are refused at initialization.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

_patched = False


# Engagement count for both native-vLLM and Megatron-in-vLLM GDN call paths.
CALL_COUNT = 0


# vLLM features that would read/advance ssm_state outside chunk-consistent decode's control.
INCOMPATIBLE_ENGINE_ARGS = (
    "enable_prefix_caching",
    "enable_chunked_prefill",
    "speculative_config",
)


def gdn_engine_patch_enabled() -> bool:
    return os.environ.get("SKYRL_ISOEXEC_GDN") == "1"


def assert_engine_args_compatible(kwargs: dict) -> None:
    """Raise if any engine arg is incompatible with chunk-consistent GDN decode.

    Loud on purpose: each of these silently reinterprets ``ssm_state`` and would produce a
    plausible-looking, non-bitwise rollout.
    """
    from ...ops.gdn.gdn_ops import cpr_mode, recurrent_mode
    from ...ops.gdn.gdn_recurrent_state import chunked_prefill_enabled

    # Chunked prefill is bitwise-safe under recurrent/cpr GDN with the resume flag on, since a
    # continuation chunk resumes from the carried state.
    incompatible = INCOMPATIBLE_ENGINE_ARGS
    if (recurrent_mode() or cpr_mode()) and chunked_prefill_enabled():
        incompatible = tuple(k for k in incompatible if k != "enable_chunked_prefill")
    # The native composition resumes at any token boundary from vLLM's own state blocks, so chunked
    # prefill and prefix caching are both sound there; speculative decode stays incompatible.
    from ...ops.gdn.gdn_ops import gdn_native_kernels_enabled

    if recurrent_mode() and gdn_native_kernels_enabled():
        incompatible = tuple(k for k in incompatible if k not in ("enable_chunked_prefill", "enable_prefix_caching"))
    bad = [k for k in incompatible if kwargs.get(k)]
    # CUDA graphs are compatible: chunk decode is shape-static and sync-free (fixed cu grid,
    # persistent index buffers, masked chunk roll), and recurrent decode is static by construction.
    if cpr_mode() and not kwargs.get("enforce_eager", False):
        # Graphs under cpr ride on the lazy resync driver, so decode() stays pure device work and
        # the boundary resync runs host-driven between replays. Installed here (idempotent) because
        # this assert runs before the engine is built.
        try:
            install_cpr_lazy_driver()
        except Exception as e:  # noqa: BLE001 - converted to the loud config error below
            bad.append(f"enforce_eager=False (CUDA graphs) under cpr: lazy resync driver failed ({e})")
    elif (
        not kwargs.get("enforce_eager", False)
        and not recurrent_mode()
        and os.environ.get("SKYRL_ISOEXEC_GDN_PADDED_DECODE") != "1"
    ):
        bad.append("enforce_eager=False (CUDA graphs) without SKYRL_ISOEXEC_GDN_PADDED_DECODE=1")
    if bad:
        raise ValueError(
            f"[isoexec-gdn] SKYRL_ISOEXEC_GDN=1 but the engine was configured with {bad}. "
            "Chunk-consistent GDN decode redefines ssm_state[slot] as the state at the last CHUNK "
            "BOUNDARY; these features read or advance it on their own terms and would break bitwise "
            "IsoExec. They are safe for the softmax layers (proven, 4.6x rollout) -- supporting them "
            "for GDN is a follow-up. Unset them or unset SKYRL_ISOEXEC_GDN."
        )


def lift_gdn_batch_invariance_veto() -> None:
    """Let ``VLLM_BATCH_INVARIANT=1`` coexist with GDN. Idempotent.

    vLLM's veto on GDN_ATTN would also block the softmax layers. The backend getter is cached, so
    this must run before the backend is first resolved (KV-cache spec collection).
    """
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend

    GDNAttentionBackend.supports_batch_invariance = classmethod(lambda cls: True)


def _get_layer_state(self):
    """Lazily build this layer's GDN core -- CprGDN or RecurrentGDN, per the kernel mode.

    Weights are re-read every call rather than captured: weight sync rebinds ``.data``, and a stale
    reference would silently roll back the policy update.
    """
    from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE

    from ...ops.gdn.gdn_cpr_state import build_cpr_gdn
    from ...ops.gdn.gdn_ops import cpr_mode, recurrent_mode
    from ...ops.gdn.gdn_recurrent_state import build_recurrent_gdn, native_state_enabled

    conv_weight = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
    conv_bias = getattr(self.conv1d, "bias", None)

    cc = getattr(self, "_isoexec_gdn", None)
    # Native state: re-check the kv_cache binding every call, since a core built against the
    # discarded profiling cache overflows on real block ids.
    if cc is not None and getattr(cc, "_native", False):
        kv = self.kv_cache
        key = (kv[1].data_ptr(), tuple(kv[1].shape)) if len(kv) > 1 and kv[1].numel() else None
        if key != getattr(self, "_isoexec_gdn_kv_key", None):
            logger.info("[isoexec-gdn] %s: NATIVE kv_cache rebound, rebuilding state core", self.prefix)
            cc = self._isoexec_gdn = None
    # The bound mamba page tensor is sized by the actual block-id space, so measure the slot map
    # from it rather than from an env default. Guarded: a harness may bind no kv_cache.
    _kvh = getattr(self, "kv_cache", None)
    slot_map_hint = 0
    try:
        if _kvh is not None and len(_kvh) > 1 and _kvh[1].dim() > 0:
            slot_map_hint = int(_kvh[1].shape[0])
    except Exception:  # pragma: no cover - profiling dummies with exotic shapes
        slot_map_hint = 0

    if cc is None and cpr_mode():
        # Private pool sized by the scheduler's concurrency cap, NOT by kv_cache rows: an open-chunk
        # buffer per slot at thousands of slots is tens of GiB per layer.
        cc = build_cpr_gdn(
            max_num_seqs=getattr(self, "_isoexec_max_num_seqs", None) or self.kv_cache[1].shape[0],
            slot_map_hint=slot_map_hint,
            chunk_size=FLA_CHUNK_SIZE,
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            num_k_heads=self.num_k_heads // self.tp_size,
            head_k_dim=self.head_k_dim,
            num_v_heads=self.num_v_heads // self.tp_size,
            head_v_dim=self.head_v_dim,
            activation=self.activation,
            dtype=conv_weight.dtype,
            device=conv_weight.device,
        )
        # Decode stays pure device work; boundaries are serviced pre-forward by the builder wrap.
        cc.lazy_resync = True
        self._isoexec_gdn = cc
        logger.info(
            "[isoexec-gdn] %s: CPR decode+prefill (LAZY resync every %d), capacity=%d",
            self.prefix,
            FLA_CHUNK_SIZE,
            cc.capacity,
        )
    elif cc is None and recurrent_mode():
        ssm_state = conv_state = None
        if native_state_enabled():
            # State lives in vLLM's own mamba kv_cache blocks. kv_cache is (conv, ssm); orient conv
            # to (num_blocks, D, W-1) via vLLM's SD/DS flag, as the native _forward_core does.
            from vllm.model_executor.layers.mamba.mamba_utils import (
                is_conv_state_dim_first,
            )

            ssm_state = self.kv_cache[1]
            conv_state = self.kv_cache[0] if is_conv_state_dim_first() else self.kv_cache[0].transpose(-1, -2)
            self._isoexec_gdn_kv_key = (ssm_state.data_ptr(), tuple(ssm_state.shape))
            logger.info(
                "[isoexec-gdn] %s: NATIVE state ssm=%s %s conv oriented=%s (layout=%s)",
                self.prefix,
                tuple(ssm_state.shape),
                ssm_state.dtype,
                tuple(conv_state.shape),
                "DS" if is_conv_state_dim_first() else "SD",
            )
        cc = build_recurrent_gdn(
            max_num_seqs=getattr(self, "_isoexec_max_num_seqs", None) or self.kv_cache[1].shape[0],
            slot_map_hint=slot_map_hint,
            ssm_state=ssm_state,
            conv_state=conv_state,
            conv_weight=conv_weight,
            conv_bias=conv_bias,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            num_k_heads=self.num_k_heads // self.tp_size,
            head_k_dim=self.head_k_dim,
            num_v_heads=self.num_v_heads // self.tp_size,
            head_v_dim=self.head_v_dim,
            activation=self.activation,
            dtype=conv_weight.dtype,
            device=conv_weight.device,
        )
        self._isoexec_gdn = cc
        logger.info("[isoexec-gdn] %s: recurrent decode+prefill, capacity=%d", self.prefix, cc.capacity)
    cc.conv_weight, cc.conv_bias = conv_weight, conv_bias
    cc.A_log, cc.dt_bias = self.A_log, self.dt_bias
    return cc


# Per-forward host-side step data (decode/prefill slot ids and offsets), computed once and shared by
# every GDN layer: each `.tolist()` on engine metadata is a GPU sync, and paying it per layer
# serializes decode. Fed by the lazy driver from scheduler-owned host arrays; falls back to a device
# read on direct recurrent calls and the first forward.
_DRIVER_HOST_STEP = None
_DRIVER_HOST_STEP_STATS = {"served": 0, "fallback": 0}


def _build_driver_host_step(req_ids, scheduled, nct, slots) -> dict | None:
    """Reproduce GDN metadata's decode/prefill split from scheduler-owned host arrays.

    Admission is stricter than vLLM's builder: every row must schedule >= 1 token and the batch must
    be decode(1-token) then prefill(>1-token). Anything else returns ``None``.
    """

    if scheduled is None:
        return None
    # Recognise the common all-decode step early. The ``all`` keeps the strict contract: a synthetic
    # {0, 2} dictionary must not be mistaken for decode merely because its total is two.
    n = len(req_ids)
    if len(scheduled) == n and all(int(q) == 1 for q in scheduled.values()):
        dec = slots[:n]
        dec_slots = dec.tolist() if hasattr(dec, "tolist") else [int(v) for v in dec]
        return {
            "key": (n, 0, n),
            "dec_slots": dec_slots,
            "pre_slots": [],
            "pre_has_init": [],
            "qsl": [0],
        }
    qlens = [int(scheduled.get(rid, 0)) for rid in req_ids]
    if any(q <= 0 for q in qlens):
        return None
    n_dec = next((i for i, q in enumerate(qlens) if q != 1), len(qlens))
    if any(q == 1 for q in qlens[n_dec:]):
        return None
    pre_lens = qlens[n_dec:]
    qsl = [0]
    for q in pre_lens:
        qsl.append(qsl[-1] + q)
    return {
        "key": (n_dec, len(qlens) - n_dec, sum(qlens)),
        "dec_slots": [int(v) for v in slots[:n_dec]],
        "pre_slots": [int(v) for v in slots[n_dec:]],
        "pre_has_init": [bool(v > 0) for v in nct[n_dec:]],
        "qsl": qsl,
    }


def _driver_prefill_slot_positions(step: dict | None, nct) -> list[tuple[int, int]] | None:
    """Scheduler positions for only the positively-identified prefill suffix.

    ``None`` means the split was not proven and callers must invalidate their position mirror; an
    empty list is a proven pure-decode step.
    """

    if step is None:
        return None
    n_dec, n_pre, _ = step["key"]
    pre_slots = step["pre_slots"]
    if len(pre_slots) != n_pre or len(nct) < n_dec + n_pre:
        return None
    return [(int(pre_slots[j]), int(nct[n_dec + j])) for j in range(n_pre)]


def _step_cpu(md, n_dec: int, n_pre: int) -> dict:
    from vllm.forward_context import get_forward_context

    ctx = get_forward_context()
    step = getattr(ctx, "_isoexec_gdn_step_cpu", None)
    if step is not None and step.get("key") == (n_dec, n_pre, md.num_actual_tokens):
        # Identity guard: this cache is shared across layers, which is sound only if every layer got
        # the SAME metadata object. A mismatch means multiple KV-cache groups with no alias active.
        if step.get("md") is not md and _group_alias_mode() != "unsafe":
            raise RuntimeError(
                "[isoexec-gdn] two GDN layers received DIFFERENT metadata objects in one forward: "
                "vLLM has split the mamba layers across multiple KV-cache groups (each with its "
                "own block table / state-slot ids) and no canonical-group metadata alias is "
                "active. Host bookkeeping would track one group's ids while each layer's decode "
                "gathers its own group's -- the unmapped ids fold onto null row 0 and every "
                "request shares one state row. cpr runs "
                "install the alias via resolve_gdn_groups (lazy driver); other modes must not run "
                "this geometry."
            )
        return step

    key = (n_dec, n_pre, md.num_actual_tokens)
    step = {"key": key, "md": md}
    fed = _DRIVER_HOST_STEP
    use_fed = fed is not None and fed.get("key") == key
    if use_fed:
        _DRIVER_HOST_STEP_STATS["served"] += 1
    else:
        _DRIVER_HOST_STEP_STATS["fallback"] += 1
    if n_dec:
        step["dec_slots"] = fed["dec_slots"] if use_fed else md.non_spec_state_indices_tensor[:n_dec].tolist()
    if n_pre:
        # One D2H for the whole step: every layer's `_continuation_mask` needs this same mask, and
        # the check below reads this host copy rather than paying its own sync.
        hi = md.prefill_has_initial_state
        step["pre_has_init"] = (
            fed["pre_has_init"] if use_fed else (None if hi is None else [bool(v) for v in hi.tolist()])
        )
        # A prefill normally starts at position 0, so an initial state means prefix caching or
        # chunked prefill snuck on -- except for a resume-flagged chunked-prefill continuation.
        if step["pre_has_init"] is not None and any(step["pre_has_init"]):
            from ...ops.gdn.gdn_ops import (
                cpr_mode,
                gdn_native_kernels_enabled,
                recurrent_mode,
            )
            from ...ops.gdn.gdn_recurrent_state import chunked_prefill_enabled

            # Under native kernels an initial state is routine (both continuations and prefix-cache
            # hits resume from vLLM's state blocks); the raise below guards the other compositions.
            resumable = (recurrent_mode() and (chunked_prefill_enabled() or gdn_native_kernels_enabled())) or (
                cpr_mode() and chunked_prefill_enabled()
            )
            if not resumable:
                raise RuntimeError(
                    "[isoexec-gdn] a prefill carries an initial state (prefix caching or chunked "
                    "prefill is on). ssm_state is a chunk-boundary state under this patch and cannot "
                    "be resumed."
                )
        pre_slots = fed["pre_slots"] if use_fed else md.prefill_state_indices.tolist()
        if n_dec and set(pre_slots) & set(step["dec_slots"]):
            raise RuntimeError("[isoexec-gdn] a slot is being prefilled and decoded in one batch")
        step["pre_slots"] = pre_slots
        # offsets INTO the packed prefill region. The fed values are the cumulative scheduler query
        # lengths; fallback reads the tensor the metadata builder uploaded from those same values.
        step["qsl"] = fed["qsl"] if use_fed else md.prefill_query_start_loc.tolist()
        # Build causal-conv's sequence metadata once for the whole forward; every GDN layer sees the
        # same metadata object. None here leaves causal_conv1d_fn's own internal builder in charge.
        from ...ops.gdn.packed_meta_cache import causal_conv1d_metadata

        step["conv_metadata"] = causal_conv1d_metadata(
            md.prefill_query_start_loc,
            cu_host=step["qsl"] if use_fed else None,
            include_launch_args=False,
        )
    try:
        ctx._isoexec_gdn_step_cpu = step
    except Exception:
        pass  # uncacheable context: every layer recomputes (correct, just not fast)
    return step


@torch.no_grad()
def recurrent_core(rg, md, mixed_qkv, a, b):
    """Run one GDN layer's core for a vLLM batch: ``GDNAttentionMetadata`` -> RecurrentGDN.

    ``mixed_qkv`` is PRE-conv, ``a`` the gate input and ``b`` the beta input; returns ``[T, Hv, Dv]``.
    A pure-decode step touches no host data, so it captures into a CUDA graph directly: ``_step_cpu``
    syncs and is consulted only for batches with a prefill, which are never captured.
    """
    global CALL_COUNT
    CALL_COUNT += 1
    if md.spec_sequence_masks is not None:
        raise RuntimeError(
            "[isoexec-gdn] speculative decoding advances ssm_state several tokens at a time with the "
            "spec kernels. Disable spec decode for bitwise IsoExec."
        )

    n_dec, n_pre = md.num_decodes, md.num_prefills
    if n_dec == 0 and n_pre == 0:
        return None
    if md.num_decode_tokens != n_dec:
        raise RuntimeError(f"[isoexec-gdn] expected 1 token per decode, got {md.num_decode_tokens} for {n_dec}")

    T = md.num_actual_tokens
    x, a, b = mixed_qkv[:T], a[:T], b[:T]

    # Pure decode returns the core's own output tensor: ``decode`` already allocates it fresh,
    # contiguous and at the right shape, so staging it through ``out`` would be a pure D2D copy.
    if n_pre == 0:
        return rg.decode(md.non_spec_state_indices_tensor[:n_dec], x, a, b)

    out = torch.empty(T, rg.num_v_heads, rg.head_v_dim, dtype=x.dtype, device=x.device)

    # Non-spec token order is decode-first, then prefill (see GDNAttentionMetadata builder).
    if n_dec:
        out[:n_dec] = rg.decode(md.non_spec_state_indices_tensor[:n_dec], x[:n_dec], a[:n_dec], b[:n_dec])
    if n_pre:
        step = _step_cpu(md, n_dec, n_pre)  # also runs the initial-state / slot-overlap checks
        out[n_dec:] = rg.prefill(
            md.prefill_state_indices,
            step["pre_slots"],
            x[n_dec:],
            a[n_dec:],
            b[n_dec:],
            step["qsl"],
            has_initial_state=step["pre_has_init"],
            conv_metadata=step.get("conv_metadata"),
            prefill_query_start_loc=md.prefill_query_start_loc,
        )
    return out


def gdn_layer_core(state, md, mixed_qkv, a, b):
    """Run one GDN layer's core, whichever mode built ``state``.

    Single owner of the alpha/beta compaction decision: the native core materialises them inside its
    fused q/k/v split, every other mode gets them compacted here.
    """
    from ...ops.gdn.gdn_fused_split import defer_ab

    if not defer_ab(state):
        a, b = a.contiguous(), b.contiguous()
    return recurrent_core(state, md, mixed_qkv, a, b)


def gdn_metadata(prefix: str):
    """This layer's ``GDNAttentionMetadata``, or None during a V1 profiling run.

    Every layer resolves to the FIRST GDN layer's metadata object so the stack shares one slot-id
    space. The alias must live here, not in a driver hook: CUDA-graph capture bakes in the address
    of the buffer read here. With a single mamba group it is identical to a per-prefix fetch.
    """
    from vllm.forward_context import get_forward_context
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    if not _GDN_CANON_PREFIX:
        _GDN_CANON_PREFIX.append(prefix)  # first GDN fetch of the process == first GDN layer
    md = get_forward_context().attn_metadata
    if md is None:
        return None
    assert isinstance(md, dict)
    md = md[_GDN_CANON_PREFIX[0]] if gdn_alias_active() else md[prefix]
    assert isinstance(md, GDNAttentionMetadata)
    return md


@torch.no_grad()
def _isoexec_forward_core(self, mixed_qkv, b, a, core_attn_out):
    """Drop-in ``QwenGatedDeltaNetAttention._forward_core``: one code path, both phases."""
    md = gdn_metadata(self.prefix)  # CALL_COUNT is bumped by recurrent_core, below
    if md is None:
        # V1 profiling run: vLLM warms its own chunk kernel here, which we never call, and a single
        # pinned autotune config means there is nothing to benchmark. Nothing to warm.
        return

    out = gdn_layer_core(_get_layer_state(self), md, mixed_qkv, a, b)
    if out is not None:
        core_attn_out[: out.shape[0]] = out


# Mamba-group geometry: where the GDN state slot id lives in the block table.
_MAMBA_GROUP: list[int] = []  # resolved lazily from kv_cache_config; [-1] = no mamba group
_ALIGN_BS: list[int] = []  # mamba block size under align mode; [0] = fixed column 0


def _mamba_group_index(runner) -> int:
    if not _MAMBA_GROUP:
        gi = -1
        try:
            from vllm.v1.kv_cache_interface import MambaSpec

            for j, grp in enumerate(runner.kv_cache_config.kv_cache_groups):
                if isinstance(grp.kv_cache_spec, MambaSpec):
                    gi = j
                    break
        except Exception:  # pragma: no cover - config shape drift fails loud below
            pass
        _MAMBA_GROUP.append(gi)
    gi = _MAMBA_GROUP[0]
    if gi < 0:
        raise RuntimeError("[isoexec-gdn] lazy driver: no MambaSpec kv-cache group found")
    return gi


def _align_block_size(runner) -> int:
    """Mamba block size when vLLM is in ALIGN cache mode, else 0 (fixed column 0).

    Align mode gathers the block covering the LAST token of the step, so a request's slot id rotates
    every ``block_size`` tokens; reading column 0 there would hand the pool a stale slot.
    """
    if not _ALIGN_BS:
        bs = 0
        try:
            if runner.vllm_config.cache_config.mamba_cache_mode == "align":
                gi = _mamba_group_index(runner)
                bs = int(runner.kv_cache_config.kv_cache_groups[gi].kv_cache_spec.block_size)
        except Exception:  # pragma: no cover - older vLLM without mamba_cache_mode
            bs = 0
        _ALIGN_BS.append(bs)
    return _ALIGN_BS[0]


# Multi-group guard: one slot-id space for all GDN layers. vLLM can split the mamba layers across
# KV-cache groups, each with its own block table, at which point unmapped state ids fold onto null
# row 0 and every request silently shares one state row. Under cpr the pages carry no state, so all
# layers can safely read the canonical (first) GDN layer's ids. Not applied under NATIVE_STATE,
# where each layer's state really does live in its own group's blocks.
_GDN_CANON_PREFIX: list = []  # [first GDN layer's prefix], self-registered at the first fetch
_gdn_groups_resolved = False
_slot_space_checked = False  # one-shot: vLLM's num_blocks fits the slot map (checked in _drive)


def _group_alias_mode() -> str:
    """SKYRL_ISOEXEC_GDN_GROUP_ALIAS: '1' (default, alias to the canonical mamba group's metadata),
    '0' (refuse a multi-group geometry), 'unsafe' (DIAGNOSTIC ONLY: per-layer, guards down)."""
    return os.environ.get("SKYRL_ISOEXEC_GDN_GROUP_ALIAS", "1").strip().lower() or "1"


def gdn_alias_active() -> bool:
    """True iff GDN metadata fetches canonicalize to the first GDN layer's metadata object.

    Default ON; OFF under NATIVE_STATE and in the '0'/'unsafe' diagnostic modes.
    """
    if _group_alias_mode() != "1":
        return False
    from ...ops.gdn.gdn_recurrent_state import native_state_enabled

    return not native_state_enabled()


def resolve_gdn_groups(runner) -> None:
    """Validate vLLM's mamba KV-cache grouping ONCE, at the driver's first step, and banner it.

    The fail-closed half of the alias (which itself lives in ``gdn_metadata``): it proves the
    canonical GDN layer belongs to the same MambaSpec group ``_slot_ids`` reads block ids from, and
    raises rather than logs if not. The banner goes to fd 1; vLLM filters INFO out of subprocesses.
    """
    global _gdn_groups_resolved
    if _gdn_groups_resolved:
        return
    _gdn_groups_resolved = True
    from vllm.v1.kv_cache_interface import MambaSpec

    groups = [
        (j, g) for j, g in enumerate(runner.kv_cache_config.kv_cache_groups) if isinstance(g.kv_cache_spec, MambaSpec)
    ]
    if len(groups) <= 1:
        return  # one slot-id space by construction; the alias is the identity
    sizes = [len(g.layer_names) for _, g in groups]
    mode = _group_alias_mode()
    if mode == "0":
        raise RuntimeError(
            f"[isoexec-gdn] vLLM split the mamba layers across {len(groups)} KV-cache groups "
            f"(sizes {sizes}) and SKYRL_ISOEXEC_GDN_GROUP_ALIAS=0 refuses to run that geometry. "
            "Each group carries its own block table, so without the canonical-group metadata "
            "alias the GDN layers disagree on state slot ids and decode folds the unmapped ones "
            "onto null row 0 -- silent corruption. Unset the flag (or set "
            "1) to run with the alias."
        )
    gi = _mamba_group_index(runner)
    gi_names = list(runner.kv_cache_config.kv_cache_groups[gi].layer_names)
    if mode == "unsafe":
        try:
            os.write(
                1,
                f"[ISOEXEC-GDN-GROUPS] pid={os.getpid()} {len(groups)} mamba KV-cache groups "
                f"(sizes {sizes}) and GROUP_ALIAS=unsafe: per-layer metadata, NO guards. This is "
                "the v9 corruption repro arm; every rollout from this engine is garbage.\n".encode(),
            )
        except Exception:  # pragma: no cover - fd 1 closed
            pass
        return
    canon = _GDN_CANON_PREFIX[0] if _GDN_CANON_PREFIX else gi_names[0]
    if _GDN_CANON_PREFIX and canon not in gi_names:
        raise RuntimeError(
            f"[isoexec-gdn] the canonical GDN layer {canon!r} (first metadata fetch of the "
            f"process) is NOT in the driver's slot-id source group {gi} ({gi_names[:3]}...). The "
            "metadata alias and _slot_ids would read DIFFERENT block tables -- the exact "
            "disagreement the alias exists to remove. vLLM's grouping/order changed; fix "
            "_mamba_group_index / the canonical registration together."
        )
    try:
        os.write(
            1,
            f"[ISOEXEC-GDN-GROUPS] pid={os.getpid()} {len(groups)} mamba KV-cache groups "
            f"(sizes {sizes}); slot-id source = group {gi}; metadata for ALL GDN layers "
            f"canonicalized to '{canon}' since the first forward (capture included), so the "
            "stack keeps one slot-id space (the pages carry no state -- only slot ids).\n".encode(),
        )
    except Exception:  # pragma: no cover - fd 1 closed
        pass


def _slot_ids(runner, sched_out, n: int, nct):
    """This step's GDN state slot id per request row, matching what the metadata builder will use."""
    import numpy as np

    gi = _mamba_group_index(runner)
    bt = runner.input_batch.block_table.block_tables[gi].block_table.np
    bs = _align_block_size(runner)
    if bs <= 0:
        return bt[:n, 0]
    ns = np.zeros(n, dtype=np.int64)
    nst = getattr(sched_out, "num_scheduled_tokens", None) if sched_out is not None else None
    if nst:
        ids = runner.input_batch.req_ids
        for i in range(n):
            ns[i] = int(nst.get(ids[i], 0))
    seq = np.asarray(nct[:n], dtype=np.int64) + ns
    col = np.clip((seq - 1) // bs, 0, bt.shape[1] - 1)
    return bt[np.arange(n), col]


_DRIVER_SLOT_BY_REQ: dict[str, int] = {}


def _remember_slots_before_layers(req_ids, slots) -> None:
    """Remember the first prefill's slot names before the private GDN pools exist.

    The driver runs before the first forward, when ``CPR_LAYERS`` is still empty; retaining these
    names lets the next chunk rebind a row when Mamba ALIGN rotates to a new block.
    """

    for rid, slot in zip(req_ids, slots, strict=True):
        _DRIVER_SLOT_BY_REQ.setdefault(str(rid), int(slot))


def _sync_slot_lifecycle(runner, sched_out, n: int, nct, slots, layers) -> None:
    """Maintain request->slot ownership during chunked prefill.

    Mamba ALIGN may rotate a live request's block id between chunks, and the private CPR state must
    follow that rename without moving the row. Unscheduled requests keep their rows; finished and
    preempted ones are released. Host bookkeeping only, before the forward and the device-map flush.
    """

    live = getattr(runner, "requests", None)
    resumed = getattr(getattr(sched_out, "scheduled_cached_reqs", None), "resumed_req_ids", None) or ()
    finished = getattr(sched_out, "finished_req_ids", None) or ()
    preempted = getattr(sched_out, "preempted_req_ids", None) or ()

    # Steady-decode fast path: under mamba cache mode ``none`` column zero is a lifetime-stable slot
    # name, so with no lifecycle transition the full release/rebind pass is an identity.
    scheduled = getattr(sched_out, "num_scheduled_tokens", None) if sched_out is not None else None
    total_scheduled = getattr(sched_out, "total_num_scheduled_tokens", None) if sched_out is not None else None
    pure_decode = (
        n > 0
        and scheduled is not None
        and len(scheduled) == n
        and (int(total_scheduled) == n if total_scheduled is not None else all(int(q) == 1 for q in scheduled.values()))
    )
    nct_positive = bool((nct[:n] > 0).all()) if hasattr(nct, "dtype") else all(int(v) > 0 for v in nct[:n])
    if (
        pure_decode
        and not resumed
        and not finished
        and not preempted
        and live is not None
        and len(_DRIVER_SLOT_BY_REQ) == len(live)
        and _align_block_size(runner) <= 0
        and nct_positive
    ):
        # When the scheduler names lifecycle deltas explicitly, empty transition sets plus equal
        # live/owned counts prove it. Otherwise compare pairs, catching same-cardinality replacement.
        new_reqs = getattr(sched_out, "scheduled_new_reqs", None)
        if new_reqs is not None and not new_reqs:
            return
        if new_reqs is None:
            req_ids = runner.input_batch.req_ids
            for i in range(n):
                if _DRIVER_SLOT_BY_REQ.get(str(req_ids[i])) != int(slots[i]):
                    break
            else:
                return

    ids = [str(rid) for rid in runner.input_batch.req_ids[:n]]
    live_ids = set(ids) if live is None else {str(rid) for rid in live}
    resumed_ids = {str(rid) for rid in resumed}
    finished_ids = {str(rid) for rid in finished}
    preempted_ids = {str(rid) for rid in preempted}

    def release(rid: str) -> None:
        old = _DRIVER_SLOT_BY_REQ.pop(rid, None)
        if old is not None:
            for ly in layers:
                ly.release_slot(old)

    # Explicit scheduler transitions precede liveness inference and rebinding: that ordering covers
    # preemption (the runner keeps the request object) and the finished+new-same-id edge.
    for rid in finished_ids | preempted_ids:
        release(rid)
    for rid in list(_DRIVER_SLOT_BY_REQ):
        if rid not in live_ids:
            release(rid)
    for rid in resumed_ids:
        release(rid)

    for i, rid in enumerate(ids):
        slot = int(slots[i])
        old = _DRIVER_SLOT_BY_REQ.get(rid)
        # A reset to zero without a resumed_req_ids surface is still a fresh prefill.
        if old is not None and int(nct[i]) == 0 and rid not in resumed_ids:
            release(rid)
            old = None
        if old is not None and old != slot:
            for ly in layers:
                ly.rebind_slot(old, slot)
        _DRIVER_SLOT_BY_REQ[rid] = slot


_lazy_driver_installed = False


def install_cpr_lazy_driver() -> bool:
    """Wrap ``GDNAttentionMetadataBuilder.build`` with the CPR LAZY-resync driver.

    The builder runs on the host once per step, outside any CUDA graph, so it is the one place a
    boundary resync can run while decode() stays capturable. A missed resync would move logprobs
    silently, so the driver never fails soft -- any exception aborts the step.
    """
    global _lazy_driver_installed
    if _lazy_driver_installed:
        return True

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    def _drive(runner, sched_out=None) -> None:
        """Fire pending boundary resyncs from the scheduler's host state, with zero device syncs.

        ``CommonAttentionMetadata.seq_lens_cpu`` must not be used here: its getter is an implicit
        D2H sync.
        """
        import numpy as np

        global _DRIVER_HOST_STEP
        _DRIVER_HOST_STEP = None  # fail closed on every early return below; this step must re-arm it

        from ...ops.gdn.gdn_cpr_state import (
            CPR_LAYERS,
        )

        # Before the layer-existence check: the KV-group geometry must be resolved before the first
        # forward builds any layer state, and on step 1 the layers do not exist yet.
        resolve_gdn_groups(runner)

        ib = runner.input_batch
        n = ib.num_reqs
        nct = ib.num_computed_tokens_cpu[:n]  # tokens consumed BEFORE this step, host numpy
        # A zero-row output can still carry a request's only finished/preempted transition, so it
        # must still reach lifecycle release and the device-map flush below.
        slots = (
            _slot_ids(runner, sched_out, n, nct) if n else np.empty(0, dtype=np.int64)
        )  # honours mamba align mode's rotating column
        if not CPR_LAYERS or not CPR_LAYERS[0].lazy_resync:
            # The first prefill reaches the driver before the private GDN pools exist; remember the
            # slot names its forward will assign so the next chunk can rebind.
            if n:
                _remember_slots_before_layers(ib.req_ids[:n], slots)
            return
        C = CPR_LAYERS[0].chunk_size

        # Latch scheduler lifecycle ownership before adoption can call `_assign`: from here on a
        # mapped row is never an LRU victim.
        if not CPR_LAYERS[0]._driver_managed:
            for ly in CPR_LAYERS:
                ly._driver_managed = True

        _sync_slot_lifecycle(runner, sched_out, n, nct, slots, CPR_LAYERS)

        # Position mirror over the prefill suffix only. If the split cannot be validated, clear each
        # mirror so a continuation falls back to the exact D2H read.
        from ...ops.gdn.gdn_recurrent_state import _HOST_BOOKKEEPING_STATS

        if n:
            nst = getattr(sched_out, "num_scheduled_tokens", None) if sched_out is not None else None
            _DRIVER_HOST_STEP = _build_driver_host_step(ib.req_ids[:n], nst, nct, slots)
            slot_pos = _driver_prefill_slot_positions(_DRIVER_HOST_STEP, nct)
            if slot_pos is None:
                for ly in CPR_LAYERS:
                    ly.note_slot_positions(None)
                _HOST_BOOKKEEPING_STATS["driver_position_layer_calls"] += len(CPR_LAYERS)
            elif slot_pos:
                for ly in CPR_LAYERS:
                    ly.note_slot_positions(slot_pos)
                nlayers = len(CPR_LAYERS)
                _HOST_BOOKKEEPING_STATS["driver_position_layer_calls"] += nlayers
                _HOST_BOOKKEEPING_STATS["driver_position_rows"] += nlayers * len(slot_pos)

        _HOST_BOOKKEEPING_STATS["driver_steps"] += 1

        # Fail-closed slot audit: on the device a padding lane and a live-but-unmapped slot both
        # fold onto row 0, but every id in `slots` belongs to a live request, so a miss is real.
        l0 = CPR_LAYERS[0]
        if not l0._native:
            global _slot_space_checked
            if not _slot_space_checked:
                _slot_space_checked = True
                nb = int(getattr(runner.kv_cache_config, "num_blocks", 0) or 0)
                if nb > l0.slot2row.numel():
                    raise RuntimeError(
                        f"[isoexec-gdn] vLLM minted {nb} KV blocks but the GDN slot map holds only "
                        f"{l0.slot2row.numel()} entries -- block ids past the map would fold onto "
                        "null row 0 at decode, silently. The map cannot be grown (a captured CUDA "
                        "graph holds its address): raise SKYRL_ISOEXEC_GDN_SLOT_MAP_SIZE above the "
                        "block count, or fix the construction-time slot_map_hint "
                        "(gdn_engine_patch._get_layer_state reads kv_cache[1].shape[0])."
                    )
            # Every request past its first scheduled token must hold a mapped row before the forward
            # reads the device map; a miss is unrecoverable and would otherwise be silent.
            for i in range(n):
                if nct[i] > 0:
                    l0.require_row(int(slots[i]))

        # Publish the step's slot-map edits while still on the host and outside capture. Rebinds
        # above only touched the numpy mirror, so this must sit above every return path below.
        from ...ops.gdn.gdn_recurrent_state import flush_slot_maps

        flush_slot_maps(CPR_LAYERS)

        if n == 0:
            return

        hits = np.nonzero((nct > 0) & (nct % C == 0))[0]
        if hits.size == 0:
            return
        # The batched resync dedups per (row, pos), so prefill-serviced boundaries are no-ops.
        pairs = [(int(slots[i]), int(nct[i])) for i in hits]
        from ...ops.gdn.gdn_cpr_state import layer_batched_resync

        layer_batched_resync(CPR_LAYERS, pairs)

    # Hook after _update_states (input_batch reflects THIS step's scheduling) and before the
    # forward; hooking execute_model instead would give the driver last step's positions.
    orig_prepare = GPUModelRunner._prepare_inputs

    def _prepare_inputs(self, *args, **kw):
        # args[0] is the SchedulerOutput: the driver needs this step's per-request scheduled token
        # counts to reproduce the metadata builder's align-mode block-table column.
        _drive(self, args[0] if args else kw.get("scheduler_output"))
        return orig_prepare(self, *args, **kw)

    GPUModelRunner._prepare_inputs = _prepare_inputs
    _lazy_driver_installed = True
    logger.info(
        "[isoexec-gdn] CPR LAZY resync driver installed on GPUModelRunner._prepare_inputs "
        "(host-state only, no device syncs)"
    )
    return True


def install_gdn_engine_patch(*, force: bool = False) -> bool:
    """Rebind ``QwenGatedDeltaNetAttention._forward_core``. Idempotent; no-op unless SKYRL_ISOEXEC_GDN=1."""
    global _patched

    if _patched:
        return True
    if not (force or gdn_engine_patch_enabled()):
        return False

    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )

    from ...ops.gdn.gdn_batch_invariant import (
        pin_fla_autotune_configs,
        pin_gdn_rmsnorm_rows_per_block,
    )

    pin_fla_autotune_configs()
    from ...ops.gdn.gdn_ops import cpr_mode as _cpr_mode

    if _cpr_mode():
        # The lazy driver removes the per-step host sync even for eager engines, and is what makes
        # decode capturable under graphs.
        install_cpr_lazy_driver()
    # RMSNormGated sizes its Triton tile from the row count, so its fp32 reduction order differs
    # between decode and prefill; chunk-consistent decode is pointless while that stands.
    pin_gdn_rmsnorm_rows_per_block()

    # `get_current_vllm_config()` is a contextvar that is only set while the model is being built,
    # so capture max_num_seqs there and read it back at first forward.
    _orig_init = QwenGatedDeltaNetAttention.__init__

    def _init(self, config, vllm_config, prefix="", gqa_interleaved_layout=False):
        _orig_init(self, config, vllm_config, prefix, gqa_interleaved_layout)
        self._isoexec_max_num_seqs = vllm_config.scheduler_config.max_num_seqs

    QwenGatedDeltaNetAttention.__init__ = _init
    QwenGatedDeltaNetAttention._forward_core = _isoexec_forward_core

    lift_gdn_batch_invariance_veto()
    # vLLM's packed recurrent decode fast path is a different kernel from the one both modes use.
    # `_forward_core` never consults the flag, but leaving it True misleads.
    QwenGatedDeltaNetAttention.enable_packed_recurrent_decode = False
    _patched = True

    from ...ops.gdn.gdn_ops import gdn_kernel_mode

    mode = gdn_kernel_mode()
    _desc = {
        "recurrent": "recurrent prefill+decode",
        "cpr": "CPR prefill + recurrent decode w/ boundary resync",
    }.get(mode, "chunk-consistent decode")
    print(
        f"[ISOEXEC-GDN] engine: QwenGatedDeltaNetAttention._forward_core -> {mode} ({_desc})",
        flush=True,
    )
    return True
