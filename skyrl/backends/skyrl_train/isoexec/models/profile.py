"""``ModelProfile`` -- the structural facts about an architecture, with no impl choices in it.

``policy.py`` holds the model-independent derivation from these facts to ``(op, site) -> impl``, so a
``models/<name>.py`` file is a profile plus an exception list. A profile is either DECLARED in such a
file or read off a live provider by ``profile_from_megatron_config``; the two are reconciled at
startup by ``assert_profile_matches_config``, which refuses to run on a disagreement -- a declared
fact that has rotted is wrong identically on both runtimes, so no bitwise gate would catch it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional, Tuple

from ..contract import STATE_EVENTS

# Attention families.
ATTN_GQA = "gqa"
_ATTENTIONS = frozenset({ATTN_GQA})

# Router score functions.
SCORE_SOFTMAX = "softmax"
SCORE_SIGMOID = "sigmoid"


class ProfileError(ValueError):
    """Raised on an ill-formed profile, or when a declared profile disagrees with the live config."""


@dataclass(frozen=True)
class TrainerNcclAdmission:
    """Capability admission for one exact trainer NCCL composition.

    The derivation matches the requested effective tuple and checks the named premise contract. A
    profile with no matching admission stays pinned, or refuses if a caller requests that tuple.
    """

    algo: Optional[str]
    min_channels: Optional[str]
    max_channels: Optional[str]
    premise_contract: str
    evidence: str

    def __post_init__(self) -> None:
        if not self.premise_contract:
            raise ProfileError("trainer NCCL admission requires a centralized premise_contract")
        if not self.evidence:
            raise ProfileError("trainer NCCL admission requires evidence")

    def effective_constants(self) -> dict[str, str | None]:
        return {
            "NCCL_ALGO": self.algo,
            "NCCL_MIN_NCHANNELS": self.min_channels,
            "NCCL_MAX_NCHANNELS": self.max_channels,
        }


@dataclass(frozen=True)
class TopologyAxisFact:
    """One parallelism axis's proven envelope, declared with the gate that proved it.

    ``kind="invariant"`` states the composition is bitwise-identical at every degree in ``domain``
    and requires ``proof`` -- a repo-relative path to the colocated gate that measured it.
    ``kind="pinned"`` states exactly one supported ``degree`` (with its ``collective_plan``).
    These derive the contract's ``TopologyClaim``s verbatim; an undeclarable axis is simply absent,
    and absence means "no claim", never "any value".
    """

    axis: str
    kind: str
    degree: Optional[int] = None
    collective_plan: Optional[str] = None
    domain: Tuple[int, ...] = ()
    proof: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in ("pinned", "invariant"):
            raise ProfileError(f"topology axis {self.axis!r}: unknown kind {self.kind!r}")
        if self.proof is not None and not self.proof.strip():
            raise ProfileError(f"topology axis {self.axis!r}: empty proof ref names no gate")
        if self.kind == "pinned":
            if self.degree is None or self.collective_plan is None:
                raise ProfileError(f"topology axis {self.axis!r}: pinned requires degree and collective_plan")
            if self.degree < 1:
                raise ProfileError(
                    f"topology axis {self.axis!r}: pinned degree {self.degree} is not a deployable "
                    "degree (>= 1); a runtime reporting it would MATCH the claim"
                )
        elif not self.domain or not self.proof:
            raise ProfileError(
                f"topology axis {self.axis!r}: invariant requires a non-empty proven domain AND a "
                "proof ref (the colocated gate that measured it); an unproven domain is not declarable"
            )
        elif len(set(self.domain)) < 2:
            raise ProfileError(
                f"topology axis {self.axis!r}: invariance over the single degree "
                f"{sorted(set(self.domain))} is a tautology, not an invariance claim; pin the axis instead"
            )


@dataclass(frozen=True)
class StateFact:
    """One engine-side state pool's lifecycle fact: what invalidates it and the EXISTING hook that
    implements the invalidation. Declaration only -- ``ref`` names code that already runs
    (``"path/inside/isoexec.py::symbol"``); it never introduces hook machinery of its own."""

    state_id: str
    invalidated_by: Tuple[str, ...]
    replay_safe: bool
    ref: str

    def __post_init__(self) -> None:
        if not self.state_id or not self.invalidated_by or not self.ref:
            raise ProfileError(
                f"state fact {self.state_id!r}: requires a non-empty invalidated_by and a ref "
                "naming the existing hook (a claim with no implementing hook is prose)"
            )
        unknown = sorted(set(self.invalidated_by) - STATE_EVENTS)
        if unknown:
            raise ProfileError(
                f"state fact {self.state_id!r}: unknown lifecycle event(s) {unknown}; the "
                f"vocabulary is {sorted(STATE_EVENTS)}. An event no boundary observes cannot "
                "invalidate anything, so declaring it states nothing checkable."
            )


@dataclass(frozen=True)
class ToleranceFact:
    """An accepted numeric envelope for one case pair. Bounds are decimal STRINGS (threshold text,
    hashed as declared), never parsed floats."""

    case_pair: Tuple[str, str]
    bounds: Tuple[Tuple[str, str], ...]
    attributed_to: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.case_pair) != 2 or not self.bounds:
            raise ProfileError(f"tolerance fact {self.case_pair!r}: requires a 2-case pair and bounds")
        for k, v in self.bounds:
            try:
                parsed = float(v)
            except (TypeError, ValueError):
                raise ProfileError(f"tolerance fact {self.case_pair!r}: bound {k}={v!r} is not a decimal string")
            # "nan"/"inf" parse fine and are exactly the values that make the gate's comparison
            # against the bound vacuous, so float() alone is not the check.
            if not math.isfinite(parsed):
                raise ProfileError(
                    f"tolerance fact {self.case_pair!r}: bound {k}={v!r} is not a finite threshold; "
                    "a gate comparing against it admits everything"
                )


@dataclass(frozen=True)
class RouterProfile:
    """The MoE router's shape: decides which ``moe.router`` impl is eligible and the pins that ride along.

    ``fused_o2`` requires ``score_function == "softmax"`` and no expert bias and declines otherwise,
    so ``policy.router_is_fusable`` consults that up front rather than letting the manifest name an
    impl that will decline at install time.
    """

    score_function: str = SCORE_SOFTMAX
    expert_bias: bool = False
    topk: Optional[int] = None
    num_experts: Optional[int] = None
    scaling_factor: Optional[float] = None
    group_topk: Optional[int] = None
    eps: Optional[float] = None

    def __post_init__(self) -> None:
        if self.score_function not in (SCORE_SOFTMAX, SCORE_SIGMOID, "sqrtsoftplus"):
            raise ProfileError(f"unknown router score_function {self.score_function!r}")


@dataclass(frozen=True)
class ModelProfile:
    """Structural facts about one architecture. No impl ids and no site policy -- those live in
    ``policy.py``.

    architectures:      HF ``config.architectures`` class names this profile claims -- the primary
                        dispatch key. ``name_patterns`` is the fallback; see ``resolve.py``.
    has_gdn:            GatedDeltaNet hybrid (megatron ``experimental_attention_variant``).
    has_moe:            any sparse layer (megatron ``num_moe_experts``); requires ``router``.
    zero_centered_norms: gamma applied as ``* (1 + w)``. A function difference, not a constant, so
                        ``norms.rms`` has two impls; the engine's fused twin exists only for this form.
    tensor_parallel:    TP>1 anywhere, i.e. the pik collectives install. At TP=1 they carry no entry.
    gdn_kernel:         which GDN core kernel is pinned. Decides two ops: the ``gdn.core`` kernel pin
                        and, since cpr owns its own state pool, the engine's ``gdn.state``.
    gdn_chunk_size:     C, the chunk-boundary period; only reaches the manifest under
                        ``gdn_kernel="cpr"``, where trainer and engine agree only at equal C.
    trainer_nccl_admissions: effective NCCL tuples this profile admits on trainer sites. Empty means
                        a non-pinned trainer manifest must refuse before NCCL initialization.
    """

    model: str
    architectures: Tuple[str, ...] = ()
    name_patterns: Tuple[str, ...] = ()
    # Disambiguator for an architecture shared by models with different IsoExec compositions.
    # Signature ``(raw_hf_config_dict) -> bool``; consulted only when a config dict is readable.
    arch_discriminator: Optional[object] = field(default=None, compare=False)
    attention: str = ATTN_GQA
    has_gdn: bool = False
    has_moe: bool = False
    zero_centered_norms: bool = True
    tensor_parallel: bool = True
    router: Optional[RouterProfile] = None
    gdn_kernel: str = "recurrent"
    gdn_chunk_size: int = 64
    ssm_cache_dtype: str = "float32"
    pik_leaves: int = 8
    # Both values are TP-invariant but their bits differ: the manifest pins whichever is resolved,
    # and install refuses an env/manifest split.
    pik_leaf_dtype: str = "bf16"
    # Admission evidence, not an architecture discriminator. Excluded from profile equality so
    # reconciliation cannot mistake a composition capability for structure.
    trainer_nccl_admissions: Tuple[TrainerNcclAdmission, ...] = field(default=(), compare=False)
    # Proven parallelism envelopes (TopologyAxisFact per axis) -> contract TopologyClaims. Proof
    # evidence like the NCCL admissions, so excluded from live-config structural equality.
    topology: Tuple[TopologyAxisFact, ...] = field(default=(), compare=False)
    # Lifecycle facts (StateFact) -> contract StateClaims; each names its existing hook.
    states: Tuple[StateFact, ...] = field(default=(), compare=False)
    # Accepted numeric envelopes (ToleranceFact) -> contract ToleranceClaims.
    tolerances: Tuple[ToleranceFact, ...] = field(default=(), compare=False)
    notes: str = ""
    # Provenance: "declared" (a models/*.py file) or "config" (read off a live provider).
    source: str = "declared"

    def __post_init__(self) -> None:
        if self.attention not in _ATTENTIONS:
            raise ProfileError(f"unknown attention family {self.attention!r}; expected one of {sorted(_ATTENTIONS)}")
        if self.has_moe and self.router is None:
            raise ProfileError(
                f"{self.model!r}: has_moe=True requires a RouterProfile (which router is a FUNCTION difference)"
            )
        if not self.has_moe and self.router is not None:
            raise ProfileError(f"{self.model!r}: router declared but has_moe=False")
        admitted = [tuple(item.effective_constants().items()) for item in self.trainer_nccl_admissions]
        if len(admitted) != len(set(admitted)):
            raise ProfileError(f"{self.model!r}: duplicate trainer NCCL admission for one effective tuple")
        axes = [t.axis for t in self.topology]
        if len(axes) != len(set(axes)):
            raise ProfileError(f"{self.model!r}: duplicate topology axis declaration")

    def op_families(self) -> frozenset:
        """Which op families this architecture needs, for comparison against the registry."""
        fams = {"mm", "attention", "norms", "logprobs"}
        fams.add("rope")
        if self.has_gdn:
            fams.add("gdn")
        if self.has_moe:
            fams.add("moe")
        if self.tensor_parallel:
            fams.add("collectives")
        return frozenset(fams)

    def with_overrides(self, **kw) -> "ModelProfile":
        """A copy with fields replaced -- for a recipe that pins e.g. ``tensor_parallel=False``."""
        return replace(self, **kw)


# The config field each profile fact is read from. Kept as data so ``assert_profile_matches_config``
# and the onboarding probe agree by construction, and so a megatron rename shows up in one place.
_CONFIG_FIELDS = {
    "attention": "multi_latent_attention",
    "has_gdn": "experimental_attention_variant",
    "has_moe": "num_moe_experts",
    "zero_centered_norms": "layernorm_zero_centered_gamma",
    "tensor_parallel": "tensor_model_parallel_size",
}


def profile_from_megatron_config(cfg, *, model: str, architectures: Tuple[str, ...] = ()) -> ModelProfile:
    """Read a ``ModelProfile`` off a live ``TransformerConfig`` / ``GPTModelProvider``.

    Reads only fields megatron itself declares (``_CONFIG_FIELDS`` plus the ``moe_router_*`` block);
    anything not readable here is a fact the profile must declare.
    """
    g = lambda name, default=None: getattr(cfg, name, default)  # noqa: E731

    has_moe = bool(g("num_moe_experts", None))

    router = None
    if has_moe:
        router = RouterProfile(
            score_function=g("moe_router_score_function", SCORE_SOFTMAX) or SCORE_SOFTMAX,
            expert_bias=bool(g("moe_router_enable_expert_bias", False)),
            topk=g("moe_router_topk", None),
            num_experts=g("num_moe_experts", None),
            scaling_factor=g("moe_router_topk_scaling_factor", None),
            group_topk=g("moe_router_group_topk", None),
        )

    return ModelProfile(
        model=model,
        architectures=tuple(architectures),
        attention=ATTN_GQA,
        has_gdn=g("experimental_attention_variant", None) == "gated_delta_net",
        has_moe=has_moe,
        zero_centered_norms=bool(g("layernorm_zero_centered_gamma", False)),
        tensor_parallel=int(g("tensor_model_parallel_size", 1) or 1) > 1,
        router=router,
        source="config",
    )


# Facts whose megatron field is authoritative and therefore worth cross-checking a declared profile
# against. Router pins (topk / scaling_factor / eps) are deliberately excluded: a recipe may force
# provider fields, so the declared pin is the post-forcing truth and a difference there is intended.
_CROSS_CHECKED = ("attention", "has_gdn", "has_moe", "zero_centered_norms")


def assert_profile_matches_config(profile: ModelProfile, cfg, *, strict: bool = True) -> list:
    """Reconcile a declared profile against the config actually built. Returns the disagreements, and
    raises on any when ``strict``.

    Runtime adapters call this once at startup with the provider they just built. The failure mode it
    catches is not bitwise: both runtimes read the same provider, so a stale declared fact is wrong
    identically on both sides and the IsoExec gate stays green.
    """
    live = profile_from_megatron_config(cfg, model=profile.model, architectures=profile.architectures)
    bad = []
    for fact in _CROSS_CHECKED:
        declared, actual = getattr(profile, fact), getattr(live, fact)
        if declared != actual:
            bad.append(
                f"{fact}: profile declares {declared!r} but the built config says {actual!r} "
                f"(megatron field {_CONFIG_FIELDS.get(fact, '?')!r})"
            )
    if profile.has_moe and live.router is not None:
        for f in ("score_function", "expert_bias"):
            d, a = getattr(profile.router, f), getattr(live.router, f)
            if d != a:
                bad.append(f"router.{f}: profile declares {d!r} but the built config says {a!r}")
    if bad and strict:
        raise ProfileError(
            f"declared ModelProfile for {profile.model!r} disagrees with the config actually built:\n  "
            + "\n  ".join(bad)
            + "\n\nThis is a CORRECTNESS hazard, not a bitwise one: both runtimes read the same "
            "provider, so a wrong fact is wrong identically on both sides and the IsoExec gate stays "
            "GREEN while the model is a different model. Fix the profile or the provider forcing; "
            "do not silence this."
        )
    return bad
