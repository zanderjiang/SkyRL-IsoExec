"""Guard that stops torch.compile from silently bypassing the batch-invariant matmul override.

Unless ``SKYRL_ISOEXEC_COMPILE=1`` is set, compilation is forced to ``CompilationMode.NONE`` for
compile-eligible models. When it is set, the required inductor configuration must be in force and a
probe must positively show a compiled GEMM matching eager, or the engine refuses to start.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

COMPILE_OPT_IN_ENV = "SKYRL_ISOEXEC_COMPILE"

REQUIRED_INDUCTOR_CONFIG: dict[str, Any] = {
    "fallback_by_default": True,
    "selective_decompose": True,
    "use_pre_grad_passes": False,
    "use_joint_graph_passes": False,
    "use_post_grad_passes": False,
    "emulate_precision_casts": True,
    "eager_numerics.division_rounding": True,
}

_REQUIRED_REASON: dict[str, str] = {
    "fallback_by_default": "unannotated ops must fall back through their own overload, or "
    "inductor lowers aten::mm itself and bypasses the batch-invariant override",
    "selective_decompose": "decompositions rewrite norms/softmax into a different arithmetic "
    "sequence, which is a different rounding schedule",
    "use_pre_grad_passes": "pre-grad passes rewrite math (matmul splitting/merging)",
    "use_joint_graph_passes": "joint-graph passes rewrite math",
    "use_post_grad_passes": "post-grad passes rewrite math (pad_mm, decompose_mem_bound_mm)",
    "emulate_precision_casts": "without it a fused bf16 chain carries fp32 intermediates where "
    "eager rounds after every op; also sets enable_fp_fusion=False and the toolkit libdevice",
    "eager_numerics.division_rounding": "Triton lowers / to the approximate div.full; eager uses "
    "div_rn. NOT implied by emulate_precision_casts",
}


def compile_opt_in() -> bool:
    return os.environ.get(COMPILE_OPT_IN_ENV) == "1"


def model_is_compile_eligible(model_cls: type | None) -> bool | None:
    if model_cls is None:
        return None
    try:
        from vllm.compilation.decorators import TorchCompileWithNoGuardsWrapper
    except Exception:
        return None
    try:
        return any(b is TorchCompileWithNoGuardsWrapper for b in model_cls.__mro__)
    except Exception:
        return None


def _resolve_registered_model_cls() -> type | None:
    try:
        import importlib

        from .model_classes import import_path_for, resolve_capabilities

        module_name, _, cls_name = import_path_for(resolve_capabilities()).partition(":")
        return getattr(importlib.import_module(module_name), cls_name, None)
    except Exception:
        return None


_COMPILING_MODES = (1, 2, 3)


def _requested_mode(kwargs: dict) -> int | None:
    cc = kwargs.get("compilation_config")
    if isinstance(cc, dict) and "mode" in cc:
        mode = cc["mode"]
        return int(mode) if isinstance(mode, int) else getattr(mode, "value", None)
    if cc is not None and hasattr(cc, "mode"):
        mode = getattr(cc, "mode")
        return int(mode) if isinstance(mode, int) else getattr(mode, "value", None)
    return None


def _would_compile(kwargs: dict) -> bool:
    if kwargs.get("enforce_eager"):
        return False
    mode = _requested_mode(kwargs)
    if mode is None:
        return True
    return mode in _COMPILING_MODES


def _get_inductor_setting(cfg, dotted: str):
    obj = cfg
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def check_required_inductor_config(required: dict[str, Any] | None = None) -> list[str]:
    required = REQUIRED_INDUCTOR_CONFIG if required is None else required
    try:
        import torch._inductor.config as cfg
    except Exception as e:
        return [f"torch._inductor.config is not importable ({e})"]

    violations: list[str] = []
    for key, want in required.items():
        try:
            got = _get_inductor_setting(cfg, key)
        except AttributeError:
            violations.append(
                f"{key}: ABSENT from this torch build (required {want!r}) -- {_REQUIRED_REASON.get(key, '')}"
            )
            continue
        if got != want:
            violations.append(f"{key}={got!r}, required {want!r} -- {_REQUIRED_REASON.get(key, '')}")
    return violations


def _bi_overrides_installed() -> bool | None:
    try:
        from ...ops.moe import moe_batch_invariant as ix_bi
    except Exception:
        ix_bi = None
    if ix_bi is not None and getattr(ix_bi, "_matmul_invariance_lib", None) is not None:
        return True
    try:
        from vllm.model_executor.layers import batch_invariant as v_bi
    except Exception:
        return None if ix_bi is None else False
    for attr in ("_batch_invariant_MODE", "_batch_invariant_LIB"):
        val = getattr(v_bi, attr, None)
        if val:
            return True
    return False


def verify_batch_invariant_survives_compilation(*, required: dict[str, Any] | None = None) -> dict:
    report: dict[str, Any] = {
        "verdict": "inconclusive",
        "reason": None,
        "overrides_installed": None,
        "config_violations": None,
    }
    try:
        return _verify_batch_invariant_survives_compilation(report, required)
    except Exception as e:
        report["verdict"] = "inconclusive"
        report["reason"] = f"probe error: {type(e).__name__}: {e}"
        return report


def _verify_batch_invariant_survives_compilation(report: dict, required) -> dict:
    violations = check_required_inductor_config(required)
    report["config_violations"] = violations
    if violations:
        report["reason"] = "required inductor configuration not in force"
        return report

    installed = _bi_overrides_installed()
    report["overrides_installed"] = installed
    if installed is not True:

        report["reason"] = (
            "batch-invariant matmul override is not installed in this process, so a compiled-vs-"
            "eager comparison would pass vacuously"
        )
        return report

    try:
        import torch
    except Exception as e:
        report["reason"] = f"torch unavailable ({e})"
        return report
    if not torch.cuda.is_available():
        report["reason"] = "no CUDA device: the override is registered for the CUDA key only"
        return report

    try:

        m, k, n = 64, 4096, 128
        gen = torch.Generator(device="cuda").manual_seed(0)
        a = torch.randn(m, k, generator=gen, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, generator=gen, device="cuda", dtype=torch.bfloat16)

        def _f(x, y):
            return x @ y

        eager = _f(a, b)
        import torch._inductor.config as icfg

        with icfg.patch({k_: v for k_, v in (required or REQUIRED_INDUCTOR_CONFIG).items()}):
            compiled = torch.compile(_f, dynamic=False)(a, b)
        if torch.equal(eager, compiled):
            report["verdict"] = "ok"
        else:
            diff = int((eager != compiled).sum().item())
            report["verdict"] = "FAILED"
            report["reason"] = (
                f"compiled GEMM differs from eager in {diff}/{eager.numel()} elements -- inductor "
                "lowered past the dispatcher and the batch-invariant matmul was BYPASSED"
            )
    except Exception as e:
        report["reason"] = f"probe error: {type(e).__name__}: {e}"
    return report


BANNER = "[ISOEXEC-COMPILE-GUARD]"


class _Unset:
    pass


UNSET = _Unset()


class CompileGuardRefusal(RuntimeError):
    pass


def assert_compilation_admissible(kwargs: dict, *, model_cls: Any = UNSET) -> dict:
    opted_in = compile_opt_in()
    if isinstance(model_cls, _Unset):
        model_cls = _resolve_registered_model_cls()
    eligible = model_is_compile_eligible(model_cls)
    would_compile = _would_compile(kwargs)
    mode = _requested_mode(kwargs)

    decision: dict[str, Any] = {
        "opted_in": opted_in,
        "model_eligible": eligible,
        "requested_mode": mode,
        "would_compile": would_compile,
        "outcome": None,
        "violations": [],
    }

    treat_as_eligible = eligible is not False

    if not opted_in:
        if would_compile and treat_as_eligible:
            cc = dict(kwargs.get("compilation_config") or {})
            cc["mode"] = 0
            kwargs["compilation_config"] = cc
            decision["outcome"] = "forced-none"
            why = (
                "this model class is compile-eligible"
                if eligible is True
                else f"compile-eligibility could not be determined ({eligible!r}), which is treated as eligible"
            )
            logger.warning(
                f"{BANNER} FORCING CompilationMode.NONE (was mode={mode!r}): {why} and "
                f"{COMPILE_OPT_IN_ENV} is not set. torch.compile lowers past the dispatcher and "
                f"would bypass the batch-invariant matmul override; see compile_guard.py's "
                f"docstring. cudagraph_mode is left untouched."
            )
        else:
            decision["outcome"] = "inert"

            logger.warning(
                f"{BANNER} inert: nothing would compile "
                f"(model_eligible={eligible!r}, would_compile={would_compile}, mode={mode!r}). "
                f"Engine config left exactly as-is."
            )
        return decision

    violations = check_required_inductor_config()
    decision["violations"] = violations
    if violations:
        decision["outcome"] = "refused"
        raise CompileGuardRefusal(
            f"{BANNER} REFUSING to enable compilation: {COMPILE_OPT_IN_ENV}=1 but the required "
            f"dispatcher-preserving / eager-numerics configuration is not in force.\n  "
            + "\n  ".join(violations)
            + "\nSee compile_guard.REQUIRED_INDUCTOR_CONFIG. This is a bitwise hazard: a compiled "
            "region containing a GEMM would silently run cuBLAS instead of the batch-invariant "
            "kernel (live 35B A/B when this happened once: gate 6.9e-07 -> 8.8e-03)."
        )

    pin = verify_batch_invariant_survives_compilation()
    decision["pin_report"] = pin
    if pin.get("verdict") != "ok":
        decision["outcome"] = "refused"
        raise CompileGuardRefusal(
            f"{BANNER} REFUSING to enable compilation: could not POSITIVELY verify that the "
            f"batch-invariant matmul survives compilation (verdict={pin.get('verdict')!r}, "
            f"reason={pin.get('reason')!r}). An unverified pin is treated as a bypassed pin."
        )

    decision["outcome"] = "admitted"
    logger.warning(
        f"{BANNER} ADMITTED: {COMPILE_OPT_IN_ENV}=1, required inductor configuration verified, "
        f"and the batch-invariant matmul verified to survive compilation. mode={mode!r}"
    )
    return decision
