"""The production autofuse sites: shared region functions plus the install-time wiring.

The region functions live here, not in the offline probe: a ledger verdict is keyed by the region's
traced signature, so the function the probe admitted and the function the runtime installs must be
the same code. Editing one invalidates its ledger entries by construction, which is the correct
failure -- re-run the probe.

``install_autofuse_sites(side)`` runs on both sides. With the flag off it prints one inert line and
registers no manifest extension, so the handshake hash is unchanged. With it on, each wired site's
target is wrapped by a per-shape dispatcher that resolves the ledger, compiles, and re-proves
bit-equality against eager on the live operands once per shape; any mismatch or unseen shape stays
eager for that shape, permanently, with a banner. The manifest extension registered at install is
what makes the weight-sync handshake refuse when the two sides resolved different ledgers.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

BANNER = "[ISOEXEC-AUTOFUSE]"


def chain_router_tail_glue(probs_fp32, idx_long, x_bf16):
    """Router chain pointwise tail: gather selected probs, scale the token, cast for the expert GEMM."""
    import torch

    sel = torch.gather(probs_fp32, 1, idx_long)
    scale = sel.to(torch.bfloat16)
    return x_bf16 * scale[:, :1], scale


def chain_rope_glue(q_bf16, cos_fp32, sin_fp32):
    """Rotate-half RoPE, fp32-single-round variant. Not wired: this transcribes the fp32-RoPE chain,
    not stock ``rope_utils._apply_rotary_pos_emb_bshd``; ``chain_rope_bshd_tail`` is that one."""
    import torch

    d = q_bf16.shape[-1] // 2
    q32 = q_bf16.to(torch.float32)
    x1, x2 = q32[..., :d], q32[..., d:]
    rot = torch.cat([-x2, x1], dim=-1)
    return (q32 * cos_fp32 + rot * sin_fp32).to(torch.bfloat16)


def chain_mlp_bda_residual_add(mlp_out_bf16, residual_bf16):
    """The mlp_bda residual add, transcribed so its refusal is citable. Never wired: it mutates the
    MLP output, which the classifier protects; see the ``qwen.mlp_bda_residual_add`` spec below."""
    return mlp_out_bf16.add_(residual_bf16)


def chain_rope_bshd_tail(t, t_pass, cos_, sin_):
    """Verbatim transcription of stock rope's glue (non-interleaved branch), with cos_/sin_ as inputs.

    The trig is transcendental (CONDITIONAL class) and stays eager in the adapter, as production
    computes it. Three-round bf16 chain: bf16(t*cos_) + bf16(rot*sin_), rounded again, then the
    t_pass cat.
    """
    import torch

    x1, x2 = torch.chunk(t, 2, dim=-1)
    rot = torch.cat((-x2, x1), dim=-1)
    out = (t * cos_) + (rot * sin_)
    return torch.cat((out, t_pass), dim=-1)


@dataclass(frozen=True)
class SiteSpec:
    """One production site: ``target`` names the module attribute the dispatcher wraps
    ('pkg.module:attr'), and ``adapt``/``emit`` map the target's calling convention onto the region
    function's canonical arguments and back.

    ``pair=True`` selects the PAIR_EQUALITY class: resolution keys on the shape class, both sides run
    the shared symbolic artifact, grad-carrying calls route through PairRegionFn, and a failed wire
    on a side in ``sides`` raises at install rather than degrading to eager. ``installer`` overrides
    the default attribute-wrap for targets that are not a flat module global.

    ``unadmitted_fallback="target"`` is for a region that subsumes an already-fused target: an
    unknown shape, capture-time first sighting or demoted artifact must return to that fused
    implementation rather than to the slower eager region.
    """

    site: str
    region_fn: Callable
    target: str | None
    sides: tuple[str, ...]
    adapt: Callable | None = None  # (*target_args, **kw) -> region args tuple, or None => eager
    emit: Callable | None = None  # (region_out, *target_args, **kw) -> target return shape
    pair: bool = False
    trainer_target: str | None = None  # pair sites: the trainer-side seam (may differ in form)
    installer: Callable | None = None  # (spec, side) -> (state, detail); overrides default wiring
    notes: str = ""
    unadmitted_fallback: Literal["region", "target"] = "region"


def chain_o12a_shared_glu(x, offset: float = 0.0):
    """Verbatim megatron ``MLP.forward`` glu closure: chunk, silu, add, mul, three ATen roundings.

    ``silu`` is CONDITIONAL, so admission requires the ledger sweep to have admitted it on the running
    (arch, torch). ``offset`` is baked per admission, since it is part of the shape key.
    """
    import torch
    import torch.nn.functional as F

    x_glu, x_linear = torch.chunk(x, 2, dim=-1)
    return F.silu(x_glu) * (x_linear + offset)


def chain_moe_combine_tail(permuted, rows):
    """Verbatim ``_fixed_order_combine`` tail given the row map: k index_selects and k-1
    out-of-place adds in ascending-expert order.

    The row map is a counting sort -- a different op class -- and is built eagerly by the adapter,
    never inside the region.
    """
    out = permuted.index_select(0, rows[:, 0].contiguous())
    for j in range(1, rows.shape[1]):
        out = out + permuted.index_select(0, rows[:, j].contiguous())
    return out


def _flatten_leading(t):
    """[.., C] -> ([N, C] view, original shape). The canonical region shape is 2D, so one ledger grid
    serves every leading-dim layout."""
    shape = t.shape
    return t.reshape(-1, shape[-1]), shape


def _o12a_adapt(x, offset=0.0):
    """Map ``fused_shared_glu(x, offset)`` onto the region, flattening leading dims to the canonical
    2D shape. Gating mirrors the manual kernel's; ineligible calls run the wrapped original."""
    import torch

    if not isinstance(x, torch.Tensor) or x.dim() < 2 or x.shape[-1] % 2:
        return None
    if x.dtype != torch.bfloat16 or not x.is_contiguous():
        return None
    if torch.is_grad_enabled() and x.requires_grad:
        return None
    flat, _ = _flatten_leading(x)
    return (flat, float(offset))


def _o12a_emit(out, x, offset=0.0):
    """Restore the target's leading-dim layout; without it, target-equivalence would compare a 2D
    region output against a higher-rank target output and demote every shape."""
    return out.reshape(*x.shape[:-1], out.shape[-1])


def _combine_adapt(permuted_tokens, sorted_indices, restore_shape, *, permuted_probs=None, rows=None, validate=False):
    """Map ``fused_fixed_order_combine``'s calling convention onto the verbatim tail.

    The probs branch stays with the original, ``validate`` calls keep the original's raise semantics,
    and a grad-carrying input refuses, since a raw compiled forward would sever the backward. The row
    map is built eagerly by the module's own ``build_combine_rows`` and widened to int64.
    """
    import torch

    if permuted_probs is not None or validate:
        return None
    if not isinstance(permuted_tokens, torch.Tensor) or permuted_tokens.dim() != 2:
        return None
    if torch.is_grad_enabled() and permuted_tokens.requires_grad:
        return None
    num_tokens = int(restore_shape[0])
    if rows is None:
        from ..ops.moe.moe_combine_kernel import build_combine_rows

        rows = build_combine_rows(sorted_indices, num_tokens)
        if rows is None:
            return None
    if rows.shape[0] != num_tokens:
        return None
    return (permuted_tokens, rows.long().contiguous())


def chain_gate_scale(output, logits):
    """Verbatim shared-expert gate, identical in form on both sides.

    It contains sigmoid, so EAGER_SPEC refuses it; it is admissible under PAIR_EQUALITY because both
    sides run this one artifact.
    """
    import torch

    return output * torch.sigmoid(logits)


def _gate_scale_adapt(output, logits):
    """Both-sides seam adapter, flattening leading dims to the canonical 2D class.

    Both seams route through this one function, so trainer and engine present the same shape class by
    construction. Grad-carrying inputs are not refused; the pair dispatcher routes them through
    PairRegionFn.
    """
    import torch

    if not isinstance(output, torch.Tensor) or not isinstance(logits, torch.Tensor):
        return None
    if output.dim() < 2 or logits.dim() < 2 or logits.shape[-1] != 1:
        return None
    flat_o, _ = _flatten_leading(output)
    flat_l, _ = _flatten_leading(logits)
    if flat_o.shape[0] != flat_l.shape[0]:
        return None
    return (flat_o, flat_l)


def _gate_scale_emit(out, output, logits):
    """Region out is the flattened [N, C]; the target returns the caller's layout."""
    return out.reshape(output.shape)


def _install_shexp_gate_class_seam(spec: "SiteSpec", side: str) -> tuple[str, str]:
    """Both-sides class seam for moe.gate_scale: the gate is inline in ``SharedExpertMLP.forward``,
    so the wire is a verbatim class-forward transcription with the gate routed through the dispatcher.

    It installs on trainer and engine alike, because a one-side-only class seam would be exactly the
    eligibility asymmetry the pair class must not have. Only the gate goes through PairRegionFn;
    fc1/fc2/gate-linear keep megatron autograd. Any structural surprise falls through to the original
    method, loudly, once per instance.
    """
    import importlib

    mod = importlib.import_module("megatron.core.transformer.moe.shared_experts")
    cls = mod.SharedExpertMLP
    if getattr(cls, "_ix_pair_gate_wired", False):
        return ("wired", "megatron...shared_experts:SharedExpertMLP.forward (already)")
    orig_forward = cls.forward
    dispatcher = SiteDispatcher(spec, chain_gate_scale)

    def forward(self, hidden_states):
        import torch

        if not getattr(self, "use_shared_expert_gate", False):
            return orig_forward(self, hidden_states)
        try:
            from megatron.core.transformer.mlp import MLP

            output, _ = MLP.forward(self, hidden_states)
            logits = torch.nn.functional.linear(hidden_states, self.gate_weight)
            # Through the shared adapter, like the engine's module-global seam: the trainer's live
            # tensors are [s, b, C] and the pair class is keyed on the flattened [N, C] form.
            pair_args = _gate_scale_adapt(output, logits)
            if pair_args is None:
                return output * torch.sigmoid(logits)  # verbatim stock gate
            out = dispatcher.call_region_grad(*pair_args)
            return out.reshape(output.shape)
        except Exception as e:  # noqa: BLE001 -- fall to the original, loudly, once
            if not getattr(self, "_ix_pair_gate_warned", False):
                self._ix_pair_gate_warned = True
                print(
                    f"{BANNER} pid={os.getpid()} site={spec.site} trainer wrapper fell back to "
                    f"the original forward ({type(e).__name__}: {e})",
                    flush=True,
                )
            return orig_forward(self, hidden_states)

    cls.forward = forward
    cls._ix_pair_gate_wired = True
    return ("wired", "megatron.core.transformer.moe.shared_experts:SharedExpertMLP.forward")


def _rope_adapt(
    t, freqs, rotary_interleaved=False, mla_rotary_interleaved=False, mscale=1.0, multi_latent_attention=None
):
    """Map stock ``_apply_rotary_pos_emb_bshd``'s calling convention onto the verbatim region.

    Eligible on the non-interleaved branch, inference only; anything else returns None and the
    original runs untouched.
    """
    import torch

    if rotary_interleaved or mla_rotary_interleaved or multi_latent_attention:
        return None
    if torch.is_grad_enabled() and (t.requires_grad or freqs.requires_grad):
        return None
    if t.dim() != 4 or freqs.dim() != 4:
        return None
    rot_dim = freqs.shape[-1]
    if rot_dim > t.shape[-1] or rot_dim % 2:
        return None
    t_head, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    # the trig, verbatim from production -- eager, outside the region
    cos_ = (torch.cos(freqs) * mscale).to(t.dtype)
    sin_ = (torch.sin(freqs) * mscale).to(t.dtype)
    return (t_head, t_pass, cos_, sin_)


def _sites() -> dict[str, SiteSpec]:
    """Built lazily so importing this module never imports megatron/vllm."""
    return {s.site: s for s in _SITE_LIST}


# A site with adapt=None is declared but not wired: the banner says so and the notes say why. Wiring
# demands a verbatim region transcription of the target's glue.
_SITE_LIST: list[SiteSpec] = [
    SiteSpec(
        site="qwen.rope_bshd_tail",
        region_fn=chain_rope_bshd_tail,
        target="megatron.core.models.common.embeddings.rope_utils:_apply_rotary_pos_emb_bshd",
        sides=("trainer", "engine"),
        adapt=None,  # not wired -- see notes
        notes="Not wired: the compiled region is slower than the FUSED_ROPE kernel and signed-zero wrong -- inductor "
        "lowers "
        "aten.neg to `-x`, which differs from the manual `x * -1.0` on negative-zero lanes. The gate's hazard "
        "populations refuse this region mechanically; the adapter is kept for a re-probe, and FUSED_ROPE keeps "
        "precedence.",
    ),
    SiteSpec(
        site="moe.o12a_shared_glu",
        region_fn=chain_o12a_shared_glu,
        target="skyrl.backends.skyrl_train.isoexec.ops.moe.moe_preamble_o12:fused_shared_glu",
        sides=("engine",),
        adapt=_o12a_adapt,
        emit=_o12a_emit,
        unadmitted_fallback="target",
        notes="Subsumable: the compiled region is bit-identical to the eager chain per (shape, hazard) population. "
        "ENGINE-ONLY, and the asymmetry is licensed rather than hidden -- the manual op is engine-only by instance "
        "rebind, the trainer runs megatron's own glu closure untouched, and an unadmitted, demoted or capture-first "
        "shape returns to the shape-polymorphic manual Triton target (one launch), never the three-launch eager "
        "region.",
    ),
    SiteSpec(
        site="moe.gate_scale",
        region_fn=chain_gate_scale,
        target="skyrl.backends.skyrl_train.isoexec.ops.moe.moe_preamble_o12:fused_gate_scale",
        sides=("trainer", "engine"),
        adapt=_gate_scale_adapt,
        emit=_gate_scale_emit,
        pair=True,
        trainer_target="megatron.core.transformer.moe.shared_experts:SharedExpertMLP.forward",
        installer=_install_shexp_gate_class_seam,
        notes="The pair-equality mechanism: `output * sigmoid(logits)` (the shared-expert gate) has an identical form "
        "on "
        "both sides. An eager spec was refused because a fused sigmoid does not match ATen elementwise; under the "
        "pair contract both sides run one shared symbolic artifact, so sigmoid-fused == sigmoid-fused by "
        "construction. Trainer gradients route through PairRegionFn (shared forward under no_grad, independent eager "
        "VJP, no bitwise contract). Wired at two seams: the class forward and the fused_gate_scale module global.",
    ),
    SiteSpec(
        site="moe.combine",
        region_fn=chain_moe_combine_tail,
        target="skyrl.backends.skyrl_train.isoexec.ops.moe.moe_combine_kernel:fused_fixed_order_combine",
        sides=("engine",),
        adapt=_combine_adapt,
        notes="Subsumable and bit-identical including hazard populations. ENGINE-ONLY, and mandatory here: the trainer "
        "deliberately gets moe_combine_backward.differentiable_unpermute because a raw kernel output has no grad_fn "
        "and would sever the backward, so wiring both sides would be a correctness bug rather than an optimization.",
    ),
    SiteSpec(
        site="qwen.rope_glue",
        region_fn=chain_rope_glue,
        target=None,
        sides=("trainer", "engine"),
        notes="Not wired, permanently: it transcribes the fp32-RoPE variant (one fp32 round), which is not the shipped "
        "local-spec path.",
    ),
    SiteSpec(
        site="moe.router_tail_glue",
        region_fn=chain_router_tail_glue,
        target=None,
        sides=("trainer", "engine"),
        notes="Not wired: the probed ops are not adjacent in production -- the gather, the cast and the multiply are "
        "separated by the topk and the denominator. That seam is already owned by install_router_chain, so "
        "subsumption goes through it or not at all.",
    ),
    SiteSpec(
        site="qwen.mlp_bda_residual_add",
        region_fn=chain_mlp_bda_residual_add,
        target=None,
        sides=("trainer", "engine"),
        notes="Declared, not wired. Megatron's current eval/no-grad path uses in-place `dropout_(x); "
        "x.add_(residual)`, and "
        "the mutation is protected by the compiler. Three independent reasons it stays manual: (1) there is no single "
        "seam to wrap -- the add runs in `TransformerLayer._forward_post_mlp` on layer L while its consumer "
        "`input_layernorm` runs in `_forward_attention` on the layer-L+1 instance, reached only after "
        "`TransformerLayer.forward` has returned; a class-(a) SiteSpec wraps one module attribute, and crossing that "
        "boundary needs a deferred-add package that survives a layer return through pipeline p2p, activation "
        "checkpointing and CUDA-graph capture. (2) The region would be refused anyway: its second half is RMSNorm, a "
        "float reduction permanently banned by safe_ops and by the pair contract. (3) The add alone is a single "
        "fusable op and comes back DECLINED_SHORT under any min_fusable the gate uses. The honest owner is the manual "
        "twin: extend fused_add_rmsnorm's within-layer seam across the layer boundary, with the package materialised "
        "before `forward` returns.",
    ),
]


def _register_site(spec: SiteSpec) -> SiteSpec:
    _SITE_LIST.append(spec)
    return spec


class SiteDispatcher:
    """Wraps one production callable. Per shape: resolve the ledger once, compile once, self-check
    once, then serve; anything else uses its declared fail-safe fallback forever."""

    def __init__(self, spec: SiteSpec, eager_target: Callable):
        self.spec = spec
        self.eager = eager_target
        self._by_shape: dict[str, tuple[str, Callable | None]] = {}  # key -> (state, compiled)
        # Target-equivalence verdicts per shape. The ledger proves region-compiled == region-eager;
        # this proves region == the wrapped target on live operands. Without a True here, the
        # original target runs.
        self._target_ok: dict[str, bool] = {}
        # Hit-rate accounting: decode M is quantized to the cudagraph capture ladder, prefill M is
        # arbitrary. The rate line separates eager-region misses from fused-target fallbacks.
        self._n_compiled = 0
        self._n_eager = 0
        self._n_target_fallback = 0
        # (site, shape_key) index of the ledger, built lazily at first resolution: answers the common
        # prefill miss without a trace. Safe to cache because a ledger is only written offline.
        self._known_shapes: set | None = None

    def _count(self, compiled: bool, *, target_fallback: bool = False) -> None:
        if compiled:
            self._n_compiled += 1
        elif target_fallback:
            self._n_target_fallback += 1
        else:
            self._n_eager += 1
        total = self._n_compiled + self._n_eager + self._n_target_fallback
        if total % 4096 == 0:
            shapes = len(self._by_shape)
            hits = sum(
                1
                for key, (state, _) in self._by_shape.items()
                if state == "compiled" and self._target_ok.get(key) is not False
            )
            print(
                f"{BANNER} pid={os.getpid()} site={self.spec.site} hit-rate: "
                f"compiled={self._n_compiled} eager={self._n_eager} "
                f"target_fallback={self._n_target_fallback} "
                f"({hits}/{shapes} shapes compiled)",
                flush=True,
            )

    def _unadmitted_label(self) -> str:
        return "TARGET-FALLBACK" if self.spec.unadmitted_fallback == "target" else "EAGER"

    def _resolve(self, region_args) -> tuple[str, Callable | None]:

        from .fusion_ledger import FusionLedger, resolve_decision
        from .region_gate import (
            REGION_INDUCTOR_CONFIG,
            _flatten_outputs,
            _trace_aten,
            arch_tag,
            bit_equal,
            config_fingerprint,
            region_signature,
            shape_key_of,
            torch_fingerprint,
        )

        key = shape_key_of(region_args)
        pid = os.getpid()
        site = self.spec.site
        try:
            # Cheap pre-check before the make_fx trace: most live shapes have no ledger entry, and a
            # trace per novel shape would tax exactly the slowest steps.
            if self._known_shapes is None:
                from .fusion_ledger import FusionLedger as _FL

                self._known_shapes = {(e.get("site"), e.get("shape_key")) for e in _FL().load().values()}
            if (site, key) not in self._known_shapes:
                print(
                    f"{BANNER} pid={pid} site={site} shape={key} -> {self._unadmitted_label()} "
                    f"(no ledger entry for this shape; trace skipped)",
                    flush=True,
                )
                return ("eager", None)

            import torch._inductor.config as icfg

            with icfg.patch(dict(REGION_INDUCTOR_CONFIG)):
                gm = _trace_aten(self.spec.region_fn, region_args)
            sig = region_signature(gm)
            from .decomp import active_table

            table = active_table()
            decision, reason = resolve_decision(
                site=site,
                region_sig=sig,
                shape_key=key,
                arch=arch_tag(),
                torch_fp=torch_fingerprint(),
                config_fp=config_fingerprint(decomps=table),
                ledger=FusionLedger(),
            )
            if decision != "compiled":
                print(
                    f"{BANNER} pid={pid} site={site} shape={key} -> " f"{self._unadmitted_label()} ({reason})",
                    flush=True,
                )
                return ("eager", None)

            # One fresh pinned artifact per shape key: the dynamo wrapper is never shared across
            # shapes, every invocation runs under the admitted config patch, and a foreign shape
            # raises instead of recompiling.
            from .region_gate import compile_pinned

            compiled = compile_pinned(self.spec.region_fn, region_args, site=site, config=dict(REGION_INDUCTOR_CONFIG))
            if not getattr(compiled, "engaged", True):
                # Dynamo produced no compiled frame, so the artifact is eager and the self-check
                # below would compare eager against eager and report "0 diff". Refuse instead.
                print(
                    f"{BANNER} pid={pid} site={site} shape={key} -> {self._unadmitted_label()} "
                    f"(dynamo produced no compiled frame for this shape; artifact is eager, "
                    f"so the self-check would be vacuous -- see region_gate.PINNED_DYNAMO_CONFIG)",
                    flush=True,
                )
                return ("eager", None)
            got = _flatten_outputs(compiled(*region_args))
            ref = _flatten_outputs(self.spec.region_fn(*region_args))
            ndiff = 0
            for g, r in zip(got, ref):
                _, n = bit_equal(g, r)
                ndiff += n
            if len(got) != len(ref) or ndiff:
                print(
                    f"{BANNER} pid={pid} site={site} shape={key} DEMOTED (install-time artifact "
                    f"self-check FAILED: {ndiff} differing elements against eager on live "
                    f"operands -- the ledger admitted this shape but THIS process's compile did "
                    f"not reproduce the artifact)",
                    flush=True,
                )
                return ("eager", None)
            print(
                f"{BANNER} pid={pid} site={site} shape={key} -> COMPILED " f"(ledger ADMITTED; self-check 0 diff)",
                flush=True,
            )
            return ("compiled", compiled)
        except Exception as e:  # noqa: BLE001 -- resolution must never break a forward
            print(
                f"{BANNER} pid={pid} site={site} shape={key} DEMOTED " f"(resolution error: {type(e).__name__}: {e})",
                flush=True,
            )
            return ("eager", None)

    def _resolve_pair(self, region_args) -> tuple[str, Callable | None]:
        """Pair-class resolution: keyed on the shape class, artifact is the shared symbolic one, and
        the self-check is determinism only -- the pair contract is artifact == artifact across sides
        and arrangements, proven at admission by ``pair_admit``, so no eager comparison applies."""
        from .fusion_ledger import FusionLedger, resolve_decision
        from .pair import compile_pair, shape_class_of
        from .region_gate import (
            REGION_INDUCTOR_CONFIG,
            _flatten_outputs,
            arch_tag,
            bit_equal,
            config_fingerprint,
            region_signature,
            torch_fingerprint,
        )

        key = shape_class_of(region_args)
        pid = os.getpid()
        site = self.spec.site
        try:
            if self._known_shapes is None:
                from .fusion_ledger import FusionLedger as _FL

                self._known_shapes = {(e.get("site"), e.get("shape_key")) for e in _FL().load().values()}
            if (site, key) not in self._known_shapes:
                print(
                    f"{BANNER} pid={pid} site={site} class={key} -> EAGER (no PAIR ledger entry)",
                    flush=True,
                )
                return ("eager", None)
            import torch._inductor.config as icfg

            from .pair import trace_region_symbolic

            with icfg.patch(dict(REGION_INDUCTOR_CONFIG)):
                gm = trace_region_symbolic(self.spec.region_fn, region_args)
            decision, reason = resolve_decision(
                site=site,
                region_sig=region_signature(gm),
                shape_key=key,
                arch=arch_tag(),
                torch_fp=torch_fingerprint(),
                config_fp=config_fingerprint(),
                ledger=FusionLedger(),
            )
            if decision != "compiled":
                print(f"{BANNER} pid={pid} site={site} class={key} -> EAGER ({reason})", flush=True)
                return ("eager", None)
            artifact = compile_pair(self.spec.region_fn, region_args, site=site, config=dict(REGION_INDUCTOR_CONFIG))
            # Bake dynamo's 0/1 specialization here, on the eager path: the engine's first M=1 call
            # arrives inside cudagraph capture, where a dynamo recompile would demote the class.
            import torch as _torch

            m1_args = [
                a[:1] if isinstance(a, _torch.Tensor) and a.dim() >= 1 and a.shape[0] > 1 else a for a in region_args
            ]
            artifact(*m1_args)
            r1 = _flatten_outputs(artifact(*region_args))
            r2 = _flatten_outputs(artifact(*region_args))
            for a, b in zip(r1, r2):
                ok, n = bit_equal(a, b)
                if not ok:
                    print(
                        f"{BANNER} pid={pid} site={site} class={key} DEMOTED " f"(pair artifact nondeterministic: {n})",
                        flush=True,
                    )
                    return ("eager", None)
            print(
                f"{BANNER} pid={pid} site={site} class={key} -> COMPILED "
                f"(PAIR ledger ADMITTED; shared symbolic artifact, determinism 0 diff)",
                flush=True,
            )
            return ("compiled", artifact)
        except Exception as e:  # noqa: BLE001
            print(
                f"{BANNER} pid={pid} site={site} class={key} DEMOTED "
                f"(pair resolution error: {type(e).__name__}: {e})",
                flush=True,
            )
            return ("eager", None)

    def call_region_grad(self, *region_args):
        """Pair-class entry: the shared artifact for inference calls, PairRegionFn for grad-carrying
        calls, eager region on any fallback.

        The eager fallback is pair-safe because resolution is a pure function of the shared ledger and
        the shape class, so both sides fall back together or not at all.
        """
        import torch

        from .pair import PairRegionFn, shape_class_of

        key = shape_class_of(region_args)
        state = self._by_shape.get(key)
        if state is None:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                return self.spec.region_fn(*region_args)
            state = self._resolve_pair(region_args)
            self._by_shape[key] = state
        verdict, artifact = state
        if verdict == "compiled" and artifact is not None:
            try:
                needs_grad = torch.is_grad_enabled() and any(
                    isinstance(a, torch.Tensor) and a.requires_grad for a in region_args
                )
                out = (
                    PairRegionFn.apply(artifact, self.spec.region_fn, *region_args)
                    if needs_grad
                    else artifact(*region_args)
                )
                self._count(True)
                return out
            except Exception as e:  # noqa: BLE001
                self._by_shape[key] = ("eager", None)
                print(
                    f"{BANNER} pid={os.getpid()} site={self.spec.site} class={key} DEMOTED "
                    f"(pair runtime error: {type(e).__name__}: {e})",
                    flush=True,
                )
        self._count(False)
        return self.spec.region_fn(*region_args)

    def _try_compiled_region(self, region_args) -> tuple[bool, object | None]:
        """Return ``(served, output)`` without choosing or counting a fallback.

        Fallback selection stays outside this method because only the target-convention entry has the
        original target arguments needed to serve the manual fused implementation.
        """
        import torch

        from .region_gate import shape_key_of

        key = shape_key_of(region_args)
        state = self._by_shape.get(key)
        if state is None:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                # Resolution compares on the host, which capture forbids. Not recorded: the next
                # eager call at this shape resolves.
                return False, None
            state = self._resolve(region_args)
            self._by_shape[key] = state
        verdict, compiled = state
        if verdict == "compiled" and compiled is not None:
            try:
                return True, compiled(*region_args)
            except Exception as e:  # noqa: BLE001 -- demote, never break the forward
                self._by_shape[key] = ("eager", None)
                print(
                    f"{BANNER} pid={os.getpid()} site={self.spec.site} shape={key} DEMOTED "
                    f"(runtime error after admission: {type(e).__name__}: {e})",
                    flush=True,
                )
        return False, None

    def call_region(self, *region_args):
        """Run the region: compiled where a resolved shape allows, eager everywhere else."""
        import torch

        from .region_gate import shape_key_of

        key = shape_key_of(region_args)
        capture_first_sighting = (
            key not in self._by_shape and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
        served, out = self._try_compiled_region(region_args)
        if served:
            self._count(True)
            return out
        # An unresolved first sighting under capture is served eagerly but not counted, because the
        # next non-capture call resolves it.
        if not capture_first_sighting:
            self._count(False)
        return self.spec.region_fn(*region_args)

    def _call_with_target_fallback(self, region_args, args, kwargs, key, tstate):
        """Serve an admitted artifact, otherwise the original already-fused target.

        No eager region executes on the fallback path, so an unseen shape remains one fused target
        launch rather than the eager launches the target had replaced.
        """
        import torch

        from .region_gate import _flatten_outputs, bit_equal

        if tstate is False:
            self._count(False, target_fallback=True)
            return self.eager(*args, **kwargs)

        # Target equivalence needs a host-visible comparison, impossible inside capture; do not
        # invoke an unverified compiled artifact there.
        if tstate is None and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            self._count(False, target_fallback=True)
            return self.eager(*args, **kwargs)

        served, region_out = self._try_compiled_region(region_args)
        if not served:
            self._count(False, target_fallback=True)
            return self.eager(*args, **kwargs)
        got = self.spec.emit(region_out, *args, **kwargs) if self.spec.emit is not None else region_out
        if tstate is True:
            self._count(True)
            return got

        # First admitted, adaptable call at this shape: prove the region really subsumes the wrapped
        # fused target. Returning ref is free because it is bit-identical.
        ref = self.eager(*args, **kwargs)
        ndiff = 0
        ref_flat, got_flat = _flatten_outputs(ref), _flatten_outputs(got)
        if len(ref_flat) != len(got_flat):
            ndiff = -1
        else:
            for r, g in zip(ref_flat, got_flat):
                _, n = bit_equal(r, g)
                ndiff += n
        if ndiff:
            self._target_ok[key] = False
            self._count(False, target_fallback=True)
            print(
                f"{BANNER} pid={os.getpid()} site={self.spec.site} shape={key} DEMOTED "
                f"(target-equivalence FAILED: region differs from the wrapped target in "
                f"{ndiff} elements on live operands -- the region is not a verbatim "
                f"transcription of this target at this shape)",
                flush=True,
            )
            return ref
        self._target_ok[key] = True
        self._count(True)
        return ref

    def __call__(self, *args, **kwargs):
        """Target-convention entry: adapt -> target-equivalence (once per shape) -> region.

        Unadaptable calls run the original untouched. The first adaptable call at each shape runs both
        the original target and the region path on the live operands and demands bit equality.
        """
        import torch

        from .region_gate import _flatten_outputs, bit_equal, shape_key_of

        if self.spec.adapt is None:
            return self.eager(*args, **kwargs)
        if self.spec.pair:
            # PAIR sites skip target-equivalence: a pair region containing sigmoid legitimately
            # differs from an eager or manual target. The proof lives in the ledger's record.
            try:
                pair_args = self.spec.adapt(*args, **kwargs)
            except Exception:  # noqa: BLE001
                pair_args = None
            if pair_args is None:
                return self.eager(*args, **kwargs)
            out = self.call_region_grad(*pair_args)
            return self.spec.emit(out, *args, **kwargs) if self.spec.emit is not None else out
        try:
            region_args = self.spec.adapt(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 -- an adapter bug must cost fusion, not a forward
            print(
                f"{BANNER} pid={os.getpid()} site={self.spec.site} adapter error "
                f"({type(e).__name__}: {e}); serving the original",
                flush=True,
            )
            region_args = None
        if region_args is None:
            return self.eager(*args, **kwargs)

        key = shape_key_of(region_args)
        tstate = self._target_ok.get(key)
        if self.spec.unadmitted_fallback == "target":
            return self._call_with_target_fallback(region_args, args, kwargs, key, tstate)
        if tstate is False:
            return self.eager(*args, **kwargs)

        def _region_out():
            out = self.call_region(*region_args)
            return self.spec.emit(out, *args, **kwargs) if self.spec.emit is not None else out

        if tstate is True:
            return _region_out()

        # first adaptable call at this shape
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return self.eager(*args, **kwargs)  # verify on the next eager call instead
        ref = self.eager(*args, **kwargs)
        try:
            got = _region_out()
        except Exception as e:  # noqa: BLE001
            self._target_ok[key] = False
            print(
                f"{BANNER} pid={os.getpid()} site={self.spec.site} shape={key} DEMOTED "
                f"(region raised during target-equivalence: {type(e).__name__}: {e})",
                flush=True,
            )
            return ref
        ndiff = 0
        ref_flat, got_flat = _flatten_outputs(ref), _flatten_outputs(got)
        if len(ref_flat) != len(got_flat):
            ndiff = -1
        else:
            for r, g in zip(ref_flat, got_flat):
                _, n = bit_equal(r, g)
                ndiff += n
        if ndiff:
            self._target_ok[key] = False
            print(
                f"{BANNER} pid={os.getpid()} site={self.spec.site} shape={key} DEMOTED "
                f"(target-equivalence FAILED: region differs from the wrapped target in "
                f"{ndiff} elements on live operands -- the region is not a verbatim "
                f"transcription of this target at this shape)",
                flush=True,
            )
            return ref
        self._target_ok[key] = True
        return ref  # bit-equal to got; returning the reference costs nothing


def autofuse_enabled() -> bool:
    # Default on; see fusion_ledger.autofuse_enabled for why that is not an engagement claim.
    return os.environ.get("SKYRL_ISOEXEC_AUTOFUSE", "1") == "1"


def autofuse_pin_digest() -> str:
    """The handshake extension digest: flag plus ledger digest, identical across sides by construction
    when both read the same ledger."""
    from .fusion_ledger import FusionLedger

    return f"enabled={autofuse_enabled()}|ledger={FusionLedger().digest()}"


def selected_autofuse_requires_exact_install() -> bool:
    """Whether this process has any matching admitted artifact; a foreign or absent ledger is inert."""

    if not autofuse_enabled():
        return False
    from .fusion_ledger import FusionLedger
    from .region_gate import arch_tag, config_fingerprint, torch_fingerprint

    census = FusionLedger().fingerprint_census(
        arch=arch_tag(),
        torch_fp=torch_fingerprint(),
        config_fp=config_fingerprint(),
    )
    return census["admitted"] > 0


_INSTALLED: dict[str, tuple[str, str]] | None = None


def install_autofuse_sites(side: str) -> dict[str, tuple[str, str]]:
    """Wire every site with a resolvable target; idempotent per process.

    Returns ``{site: (state, detail)}`` where state is 'wired', 'not-wired' or 'inert'.
    """
    global _INSTALLED
    if _INSTALLED is not None:
        return _INSTALLED
    pid = os.getpid()
    specs = _sites()
    if not autofuse_enabled():
        print(
            f"{BANNER} pid={pid} side={side} INERT: SKYRL_ISOEXEC_AUTOFUSE is not set "
            f"({len(specs)} sites untouched)",
            flush=True,
        )
        _INSTALLED = {s: ("inert", "flag off") for s in specs}
        return _INSTALLED

    from ..core.process_manifest import register_manifest_extension
    from .fusion_ledger import FusionLedger, ledger_path

    register_manifest_extension("autofuse", autofuse_pin_digest)
    print(
        f"{BANNER} pid={pid} side={side} install: ledger={ledger_path()} "
        f"digest={FusionLedger().digest()} sites={len(specs)}",
        flush=True,
    )
    # Ledger census, printed before any site is wired: "the flag is on" and "this ledger can serve
    # this process" are different claims. A miss is eager on both sides, so this reports, never refuses.
    try:
        from .region_gate import arch_tag, config_fingerprint, torch_fingerprint

        census = FusionLedger().fingerprint_census(
            arch=arch_tag(), torch_fp=torch_fingerprint(), config_fp=config_fingerprint()
        )
        if census["matching"] == 0:
            state = "NO LEDGER" if not census["exists"] or census["total"] == 0 else "STALE/FOREIGN LEDGER"
            print(
                f"{BANNER} pid={pid} side={side} {state}: {census['total']} entr(ies) at "
                f"{census['path']}, 0 of them match this process "
                f"(arch={census['arch']} torch={census['torch_fp']} config={census['config_fp']}). "
                f"EVERY SITE WILL RESOLVE EAGER -- autofusion is INERT here, bit-identical to "
                f"SKYRL_ISOEXEC_AUTOFUSE=0. Point SKYRL_ISOEXEC_AUTOFUSE_LEDGER at a ledger probed "
                f"on this arch/torch/gate version, or re-run examples/isoexec/nightly/autofuse_probe.py.",
                flush=True,
            )
        else:
            print(
                f"{BANNER} pid={pid} side={side} ledger census: {census['matching']}/"
                f"{census['total']} entries match this process, {census['admitted']} ADMITTED "
                f"(arch={census['arch']} torch={census['torch_fp']} config={census['config_fp']}).",
                flush=True,
            )
    except Exception as e:  # noqa: BLE001 -- a census failure must never block the install
        print(f"{BANNER} pid={pid} side={side} ledger census unavailable ({type(e).__name__}: {e})", flush=True)

    results: dict[str, tuple[str, str]] = {}
    for site, spec in sorted(specs.items()):
        if spec.pair and side in spec.sides:
            # PAIR sites: absence on a side is a trainer/engine mismatch, so a failed wire raises
            # instead of degrading to eager. Both seams must land: the class seam on both sides, and
            # engine-side the module-global target.
            try:
                state = spec.installer(spec, side) if spec.installer is not None else None
                if side == "engine" and spec.target is not None:
                    import importlib

                    mod_name, _, attr = spec.target.partition(":")
                    mod = importlib.import_module(mod_name)
                    original = getattr(mod, attr)
                    if not isinstance(original, SiteDispatcher):
                        setattr(mod, attr, SiteDispatcher(spec, original))
                results[site] = state or ("wired", spec.target or "installer")
                print(
                    f"{BANNER} pid={pid} site={site} WIRED (PAIR) -> {results[site][1]}"
                    + (f" + {spec.target}" if side == "engine" and spec.target else ""),
                    flush=True,
                )
            except Exception as e:
                raise RuntimeError(
                    f"{BANNER} pid={pid} site={site} PAIR wiring FAILED on side {side!r} "
                    f"({type(e).__name__}: {e}). A pair region missing on one side IS a "
                    f"trainer/engine mismatch; refusing to run rather than diverge."
                ) from e
            continue
        if side not in spec.sides:
            results[site] = ("not-wired", f"site does not run on side {side!r}")
            print(f"{BANNER} pid={pid} site={site} NOT WIRED (not a {side} site)", flush=True)
            continue
        if spec.target is None:
            results[site] = ("not-wired", "no verbatim target (see spec notes)")
            print(
                f"{BANNER} pid={pid} site={site} NOT WIRED (no verbatim target; " f"{spec.notes.split('.')[0]})",
                flush=True,
            )
            continue
        if spec.adapt is None:
            results[site] = ("not-wired", f"target {spec.target} declared but no adapter")
            print(
                f"{BANNER} pid={pid} site={site} NOT WIRED (target declared, adapter missing)",
                flush=True,
            )
            continue
        try:
            import importlib

            mod_name, _, attr = spec.target.partition(":")
            mod = importlib.import_module(mod_name)
            original = getattr(mod, attr)
            if isinstance(original, SiteDispatcher):  # already wired (idempotence across callers)
                results[site] = ("wired", spec.target)
                continue
            setattr(mod, attr, SiteDispatcher(spec, original))
            results[site] = ("wired", spec.target)
            print(f"{BANNER} pid={pid} site={site} WIRED -> {spec.target}", flush=True)
        except Exception as e:  # noqa: BLE001 -- an unwirable site is eager, loudly
            results[site] = ("not-wired", f"{type(e).__name__}: {e}")
            print(
                f"{BANNER} pid={pid} site={site} NOT WIRED ({type(e).__name__}: {e})",
                flush=True,
            )
    _INSTALLED = results
    return results
