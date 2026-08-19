"""Machine-checkable inventory of the megatron internals this adapter rebinds or source-patches.

The adapter does not fork megatron; it rebinds roughly two dozen megatron-core / megatron-bridge / TE
internals at import time, all of them undocumented. When those move the failure mode is silent -- a
rebind lands on a symbol that no longer means what the adapter assumes, or a bridge mapping matches
nothing and the model loads random weights. ``check_megatron_compat`` checks existence and light
signatures at import time; it defaults to WARN and raises only under ``SKYRL_ISOEXEC_COMPAT_STRICT=1``,
every probe is wrapped, and an unimportable ``optional`` parent module is tolerated so the check is
safe to run on a partial environment.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Version strings attached to every entry -- the surface these were validated against.
MCORE_VER = "megatron-core 0.19.0+71e418ea7 (git 71e418ea7d7b3a6c9a53238c543c3e0b43e11026)"
MBRIDGE_VER = "megatron-bridge 0.5.0 (git 91a15142a4b4442a8d46ab539d1b923bd08570d0)"
TE_VER = "transformer-engine[pytorch]==2.11.0 (absent on the no-TE nightly stack)"


@dataclass(frozen=True)
class CompatEntry:
    """One point where the megatron-side adapter rebinds or source-patches an external symbol.

    ``attr`` is a dotted attribute chain within ``module`` (``""`` means the module itself);
    ``expect_params`` is an exact parameter-name tuple and ``expect_params_subset`` a looser
    must-be-present set; ``optional`` downgrades an unimportable parent module from a problem to a
    note; ``files`` lists source files that must exist under ``module``'s root, checked without
    importing it.
    """

    module: str
    attr: str
    patched_in: str
    version: str
    expect_params: Optional[Tuple[str, ...]] = None
    expect_params_subset: Optional[Tuple[str, ...]] = None
    optional: bool = False
    files: Optional[Tuple[str, ...]] = None
    note: str = ""

    @property
    def fq(self) -> str:
        return f"{self.module}.{self.attr}" if self.attr else self.module


MEGATRON_COMPAT: List[CompatEntry] = [
    # megatron_patches.py -- RoPE fp32, RMSNorm/vops, batch-invariant, TE fused norm
    CompatEntry(
        module="megatron.core.models.common.embeddings.rope_utils",
        attr="_apply_rotary_pos_emb_bshd",
        patched_in="megatron_patches.py (apply_rope_fp32_patch rebinds this global)",
        version=MCORE_VER,
        expect_params=(
            "t",
            "freqs",
            "rotary_interleaved",
            "mla_rotary_interleaved",
            "mscale",
            "multi_latent_attention",
        ),
    ),
    CompatEntry(
        module="megatron.core.models.common.embeddings.rope_utils",
        attr="_rotate_half",
        patched_in="megatron_patches.py (imported inside the fp32 RoPE replacement)",
        version=MCORE_VER,
    ),
    CompatEntry(
        module="megatron.core.transformer.custom_layers.batch_invariant_kernels",
        attr="BatchInvariantRMSNormFn",
        patched_in="megatron_patches.py (class whose .forward is overridden)",
        version=MCORE_VER,
    ),
    CompatEntry(
        module="megatron.core.transformer.custom_layers.batch_invariant_kernels",
        attr="BatchInvariantRMSNormFn.forward",
        patched_in="megatron_patches.py (apply_vops_rmsnorm_patch -> vLLM C++ norm)",
        version=MCORE_VER,
        expect_params=("ctx", "x", "weight", "eps", "zero_centered_gamma"),
    ),
    CompatEntry(
        module="megatron.core.transformer.custom_layers.batch_invariant_kernels",
        attr="enable_batch_invariant_mode",
        patched_in="megatron_patches.py (enable_megatron_batch_invariant)",
        version=MCORE_VER,
    ),
    CompatEntry(
        module="megatron.core.transformer.custom_layers.batch_invariant_kernels",
        attr="_batch_invariant_MODE",
        patched_in="megatron_patches.py (set True under skip_aten_registration)",
        version=MCORE_VER,
    ),
    CompatEntry(
        module="megatron.core.transformer.custom_layers.batch_invariant_kernels",
        attr="_te_patch_for_batch_invariant",
        patched_in="megatron_patches.py (TE general_gemm + RMSNorm monkey-patches)",
        version=MCORE_VER,
    ),
    CompatEntry(
        module="megatron.core.transformer.custom_layers.batch_invariant_kernels",
        attr="mean_dim",
        patched_in="megatron_patches.py (used in _patched_rmsnorm_forward for the fp32 rsigma)",
        version=MCORE_VER,
    ),
    CompatEntry(
        module="megatron.core.transformer.custom_layers.batch_invariant_kernels",
        attr="is_batch_invariant_mode_enabled",
        patched_in="megatron_patches.py (the mode-query the flag drives)",
        version=MCORE_VER,
    ),
    # TE fused-layer normalization, patched in all three modules. Optional: TE is deliberately
    # absent on the no-TE stack and the patch self-skips.
    CompatEntry(
        module="transformer_engine.pytorch.module._common",
        attr="apply_normalization",
        patched_in="megatron_patches.py (apply_te_fused_norm_patch)",
        version=TE_VER,
        optional=True,
        note="TE absent on the no-TE nightly stack; the patch warns + skips.",
    ),
    CompatEntry(
        module="transformer_engine.pytorch.module.layernorm_linear",
        attr="apply_normalization",
        patched_in="megatron_patches.py (rebinds the local name bound at TE import time)",
        version=TE_VER,
        optional=True,
        note="TE absent on the no-TE nightly stack; the patch warns + skips.",
    ),
    CompatEntry(
        module="transformer_engine.pytorch.module.layernorm_mlp",
        attr="apply_normalization",
        patched_in="megatron_patches.py (rebinds the local name bound at TE import time)",
        version=TE_VER,
        optional=True,
        note="TE absent on the no-TE nightly stack; the patch warns + skips.",
    ),
    # gdn_fla_shim.py -- CheckpointWithoutOutput fp8 default + GatedDeltaNet rebinds
    CompatEntry(
        module="megatron.core.tensor_parallel.random",
        attr="CheckpointWithoutOutput.__init__",
        patched_in="gdn_fla_shim.py:217,223-230 (_patch_megatron_checkpoint_fp8_default)",
        version=MCORE_VER,
        expect_params=("self", "fp8"),
    ),
    CompatEntry(
        module="megatron.core.ssm.gated_delta_net",
        attr="GatedDeltaNet._compute_g_and_beta",
        patched_in="gdn_fla_shim.py (native gating) + :420 (eager rebind)",
        version=MCORE_VER,
        expect_params=("self", "A_log_local_cp", "dt_bias_local_cp", "alpha", "beta"),
    ),
    CompatEntry(
        module="megatron.core.ssm.gated_delta_net",
        attr="GatedDeltaNet._prepare_qkv_for_gated_delta_rule",
        patched_in="gdn_fla_shim.py (_patch_megatron_gdn_eager -> _eager_prepare_qkv)",
        version=MCORE_VER,
        expect_params=("self", "qkv", "gate", "beta", "alpha", "batch", "seq_len"),
    ),
    CompatEntry(
        module="megatron.core.ssm.gated_delta_net",
        attr="GatedDeltaNet._apply_gated_norm",
        patched_in="gdn_fla_shim.py (_patch_megatron_gdn_eager -> _eager_apply_gated_norm)",
        version=MCORE_VER,
        expect_params=("self", "x", "gate"),
    ),
    CompatEntry(
        module="megatron.core.ssm.gated_delta_net",
        attr="GatedDeltaNetSubmodules",
        patched_in="gdn_hybrid_spec.py:45,52 (assembled into the local GDN layer spec)",
        version=MCORE_VER,
    ),
    # ops/norms/zero_centered_norm.py -- WrappedTorchNorm.__new__ (installed by the shim)
    CompatEntry(
        module="megatron.core.transformer.torch_norm",
        attr="WrappedTorchNorm.__new__",
        patched_in="ops/norms/zero_centered_norm.py:72,84 (install_zero_centered_torch_norm) + "
        "ops/norms/sp_norm_lift.py (install_trainer_sp_norm_lift, SKYRL_ISOEXEC_TRAINER_SP=1: bypasses "
        "the `assert not config.sequence_parallel` via a config proxy and marks the built norm's "
        "params `sequence_parallel` for finalize_model_grads' TP grad sum; composes in either order)",
        version=MCORE_VER,
        expect_params=(
            "cls",
            "config",
            "hidden_size",
            "eps",
            "persist_layer_norm",
            "zero_centered_gamma",
            "normalization",
        ),
    ),
    # gdn_hybrid_spec.py -- structural spec-builder deps the hybrid assembly consumes
    CompatEntry(
        module="megatron.core.models.backends",
        attr="LocalSpecProvider.column_parallel_linear",
        patched_in="gdn_hybrid_spec.py:56,82 (LocalSpecProvider().column_parallel_linear())",
        version=MCORE_VER,
        expect_params=("self",),
    ),
    CompatEntry(
        module="megatron.core.models.backends",
        attr="LocalSpecProvider.row_parallel_linear",
        patched_in="gdn_hybrid_spec.py (LocalSpecProvider().row_parallel_linear())",
        version=MCORE_VER,
        expect_params=("self",),
    ),
    CompatEntry(
        module="megatron.core.models.backends",
        attr="LocalSpecProvider.layer_norm",
        patched_in="gdn_hybrid_spec.py:57,101 (LocalSpecProvider().layer_norm(rms_norm=, for_qk=))",
        version=MCORE_VER,
        expect_params_subset=("rms_norm", "for_qk"),
    ),
    CompatEntry(
        module="megatron.core.models.gpt.experimental_attention_variant_module_specs",
        attr="get_linear_attention_pattern",
        patched_in="gdn_hybrid_spec.py:73,84 (linear-attention layer pattern)",
        version=MCORE_VER,
        expect_params=("config",),
    ),
    CompatEntry(
        module="megatron.core.models.gpt.gpt_layer_specs",
        attr="get_gpt_layer_local_spec",
        patched_in="gdn_hybrid_spec.py (base dense/attention layer spec)",
        version=MCORE_VER,
        expect_params_subset=("num_experts", "moe_grouped_gemm", "qk_layernorm", "normalization"),
    ),
    CompatEntry(
        module="megatron.core.transformer.transformer_block",
        attr="TransformerBlockSubmodules",
        patched_in="gdn_hybrid_spec.py:77,100 (the block the hybrid spec returns)",
        version=MCORE_VER,
    ),
    CompatEntry(
        module="megatron.core.transformer.spec_utils",
        attr="ModuleSpec",
        patched_in="gdn_hybrid_spec.py:46,50 (each layer/submodule spec node)",
        version=MCORE_VER,
    ),
    # gdn_hybrid_spec.py -- megatron-bridge mapping rebinds. Optional: the bridge parent is
    # importable only after the no-TE guard has run, or in a full environment.
    CompatEntry(
        module="megatron.bridge.models.conversion.param_mapping",
        attr="ChunkedMapping.get_shard_idx",
        patched_in="gdn_hybrid_spec.py:154,167-171 (patch_chunked_mapping_index_device, base + subclasses)",
        version=MBRIDGE_VER,
        expect_params=("self", "config", "local_tp"),
        optional=True,
        note="ChunkedMapping subclasses GDNConv1dMapping/MambaConv1dMapping each shadow get_shard_idx.",
    ),
    CompatEntry(
        module="megatron.bridge.models.qwen.qwen35_bridge",
        attr="Qwen35Bridge._get_dense_lm_mappings",
        patched_in="gdn_hybrid_spec.py (patch_qwen35_bridge_for_local_spec)",
        version=MBRIDGE_VER,
        optional=True,
    ),
    CompatEntry(
        module="megatron.bridge.models.qwen.qwen35_bridge",
        attr="Qwen35MoEBridge._get_moe_lm_mappings",
        patched_in="gdn_hybrid_spec.py (patch_qwen35_bridge_for_local_spec)",
        version=MBRIDGE_VER,
        optional=True,
    ),
    # no_te_guard.py -- three megatron-bridge source files patched on disk (unguarded TE imports)
    CompatEntry(
        module="megatron.bridge",
        attr="",
        patched_in="no_te_guard.py (install_no_te_guard source-patches these files)",
        version=MBRIDGE_VER,
        optional=True,
        files=("peft/lora_layers.py", "peft/lora.py", "diffusion/models/wan/utils.py"),
        note="The needle line each file must contain is validated at patch time by install_no_te_guard.",
    ),
]


def _resolve_entry(e: CompatEntry) -> Tuple[List[str], List[str]]:
    """Probe one entry, returning ``(problems, notes)``.

    A problem is a hard existence failure or a signature drift, which ``strict`` raises on; a note is
    a tolerated absence of an ``optional`` parent module.
    """
    problems: List[str] = []
    notes: List[str] = []

    # (a) source-file entry: check the files exist under the package root WITHOUT importing it.
    if e.files is not None:
        try:
            spec = importlib.util.find_spec(e.module)
        except Exception as ex:  # a broken/partial install
            spec = None
            notes.append(f"{e.module}: find_spec failed ({type(ex).__name__}: {ex})")
        if spec is None or not spec.origin:
            msg = f"{e.module}: package root not found (source guard cannot locate {e.files})"
            (notes if e.optional else problems).append(msg + f"  [{e.patched_in}]")
            return problems, notes
        root = os.path.dirname(spec.origin)
        for rel in e.files:
            if not os.path.exists(os.path.join(root, rel)):
                problems.append(f"{e.module}: source-guard target missing: {rel}  [{e.patched_in}]")
        return problems, notes

    # (b) attribute entry: import the parent module (lazily), then walk the getattr chain.
    try:
        mod = importlib.import_module(e.module)
    except Exception as ex:
        msg = f"{e.fq}: parent module '{e.module}' not importable ({type(ex).__name__}: {ex})  [{e.patched_in}]"
        if e.optional:
            notes.append(msg)
        else:
            problems.append(msg)
        return problems, notes

    obj = mod
    for part in filter(None, e.attr.split(".")):
        if not hasattr(obj, part):
            problems.append(
                f"{e.fq}: missing attribute '{part}' on {getattr(obj, '__name__', obj)!r}  [{e.patched_in}]"
            )
            return problems, notes
        obj = getattr(obj, part)

    # (c) light signature checks (only when a signature is introspectable).
    if e.expect_params is not None or e.expect_params_subset is not None:
        try:
            got = tuple(inspect.signature(obj).parameters.keys())
        except (ValueError, TypeError):
            got = None  # builtins / C-level / unintrospectable: existence is all we can assert
        if got is not None:
            if e.expect_params is not None and got != e.expect_params:
                problems.append(
                    f"{e.fq}: signature drift: params {got} != expected {e.expect_params}  [{e.patched_in}]"
                )
            if e.expect_params_subset is not None:
                missing = [p for p in e.expect_params_subset if p not in got]
                if missing:
                    problems.append(
                        f"{e.fq}: signature drift: missing required param(s) {tuple(missing)} (have {got})  [{e.patched_in}]"
                    )
    return problems, notes


def check_megatron_compat(strict: Optional[bool] = None) -> List[str]:
    """Verify the megatron-side patch surface and return the list of problem strings.

    Defaults to WARN. Raises ``RuntimeError`` only when ``strict`` is True, which defaults to
    ``SKYRL_ISOEXEC_COMPAT_STRICT=1``; a bug inside a probe becomes a problem entry, never a crash.
    """
    if strict is None:
        strict = os.environ.get("SKYRL_ISOEXEC_COMPAT_STRICT") == "1"

    problems: List[str] = []
    notes: List[str] = []
    for e in MEGATRON_COMPAT:
        try:
            p, n = _resolve_entry(e)
        except Exception as ex:  # a bug in the check itself must never crash the adapter
            p, n = [f"{e.fq}: compat-check internal error ({type(ex).__name__}: {ex})  [{e.patched_in}]"], []
        problems.extend(p)
        notes.extend(n)

    n_total = len(MEGATRON_COMPAT)
    for note in notes:
        logger.info("[ISOEXEC-COMPAT] (tolerated) %s", note)
    if problems:
        logger.warning("[ISOEXEC-COMPAT] megatron surface: %d problem(s) across %d entries:", len(problems), n_total)
        for p in problems:
            logger.warning("[ISOEXEC-COMPAT]   - %s", p)
        if strict:
            raise RuntimeError(
                "[ISOEXEC-COMPAT] megatron compatibility surface changed (SKYRL_ISOEXEC_COMPAT_STRICT=1):\n  "
                + "\n  ".join(problems)
            )
    else:
        logger.info(
            "[ISOEXEC-COMPAT] megatron surface OK (%d symbols verified%s)",
            n_total,
            f", {len(notes)} tolerated-absent" if notes else "",
        )
    return problems
