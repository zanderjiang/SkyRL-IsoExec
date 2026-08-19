"""Keep the scoring forward's logits in the model's own dtype (``SKYRL_ISOEXEC_SCORING_LOGITS_BF16``).

Megatron's ``Float16Module.forward`` upcasts the last pipeline stage's output to fp32; for the
scoring forward that output is the whole ``[1, T, V/TP]`` logits tensor, so this lever passes
Megatron's ``fp32_output=False`` keyword and lets the chunked logprob head widen per chunk instead --
the same bits, only later. Admission is purely structural (env, sampling temperature, and the module
chain's precision flags) and never reads a tensor value, because a rank that disagreed with its peers
would post a different dtype into the downstream vocabulary all-gather.
"""

from __future__ import annotations

import inspect
import os
from typing import Any, Optional

import torch

ENV = "SKYRL_ISOEXEC_SCORING_LOGITS_BF16"
BANNER = "[ISOEXEC-SCORING-LOGITS-BF16]"

#: model precisions an fp32 widening is exact from, i.e. the ones this lever may leave in place.
#: fp32 is deliberately absent: with an fp32 model there is no upcast to skip.
_MODEL_DTYPES = (torch.bfloat16, torch.float16)

_S: dict = {
    "calls": 0,  # scoring forwards that consulted this module at all
    "served": 0,  # scoring microbatches that actually ran with fp32_output=False
    "declined": 0,
    "decline_reason": "",
    "out_dtypes": (),  # observed output dtypes on served calls -- proof the kwarg did something
    "reported": 0,
    "bannered": False,
}


def enabled() -> bool:
    """Default OFF. :func:`admit` and the caller add two further layers, so the env alone changes nothing."""
    return os.environ.get(ENV, "0") == "1"


def stats() -> dict:
    """A copy of the census counters, for tests and reporting."""
    return dict(_S)


def _reset_for_test() -> None:
    _S.update(
        {
            "calls": 0,
            "served": 0,
            "declined": 0,
            "decline_reason": "",
            "out_dtypes": (),
            "reported": 0,
            "bannered": False,
        }
    )


def _decline(reason: str) -> bool:
    _S["declined"] += 1
    _S["decline_reason"] = reason
    return False


def find_float16_module(model: Any, max_depth: int = 8) -> Optional[Any]:
    """The ``Float16Module`` in ``model``'s wrapper chain, or None.

    Megatron stacks the wrappers as ``DistributedDataParallel(Float16Module(GPTModel))`` and each
    wrapper forwards its keywords, so walking the chain -- rather than testing the top object --
    works for the DDP-wrapped policy module, the unwrapped ref module and a virtual-pipeline chunk
    alike.
    """
    try:
        from megatron.core.transformer.module import Float16Module
    except Exception:  # noqa: BLE001 -- no megatron, no Float16Module, no lever
        return None

    cur = model
    for _ in range(max_depth):
        if cur is None:
            return None
        if isinstance(cur, Float16Module):
            return cur
        cur = getattr(cur, "module", None)
    return None


def _accepts_fp32_output(f16: Any) -> bool:
    """Does this Megatron's ``Float16Module.forward`` take the ``fp32_output`` keyword?

    Checked by signature rather than by try/except around the live call, because the verdict has to
    be identical on every rank before any collective is posted.
    """
    try:
        params = inspect.signature(type(f16).forward).parameters
    except (TypeError, ValueError):  # C-implemented or otherwise unintrospectable
        return False
    p = params.get("fp32_output")
    return p is not None and p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)


def admitted_output_dtype(model_chunks: Any) -> Optional[torch.dtype]:
    """The single model precision every chunk would return with ``fp32_output=False``, else None.

    ``None`` covers both an unsupported chain and mixed chunk precisions; the returned dtype is part
    of the scoring wrapper's TP-unanimous pre-forward contract.
    """
    chunks = model_chunks if isinstance(model_chunks, (list, tuple)) else [model_chunks]
    dtypes = set()
    for chunk in chunks:
        f16 = find_float16_module(chunk)
        cfg = getattr(f16, "config", None) if f16 is not None else None
        if getattr(cfg, "bf16", False):
            dtypes.add(torch.bfloat16)
        elif getattr(cfg, "fp16", False):
            dtypes.add(torch.float16)
        else:
            return None
    return next(iter(dtypes)) if len(dtypes) == 1 else None


def admit(model_chunks: Any, temperature: float) -> bool:
    """Structural verdict: may the scoring forward keep its logits in model precision?

    Every chunk of ``model_chunks`` must qualify, and ``temperature`` must be exactly 1.0 -- anything
    else means the scoring ``collection_func`` runs ``logits.div_(temperature)`` in place before the
    logprob head, which is not dtype-neutral. Reads no tensor values, so every rank agrees.
    """
    _S["calls"] += 1
    if not enabled():
        return _decline(f"{ENV}=0 (default): the fp32 upcast stands")
    if os.environ.get("SKYRL_ISOEXEC") != "1":
        return _decline("SKYRL_ISOEXEC != 1: this lever exists to feed the IsoExec gather branch")
    if float(temperature) != 1.0:
        return _decline(
            f"algorithm.temperature={temperature!r} != 1.0: collection_func runs logits.div_(temperature) "
            "in place before the logprob head, and that arithmetic is not dtype-neutral"
        )

    chunks = model_chunks if isinstance(model_chunks, (list, tuple)) else [model_chunks]
    if not chunks:
        return _decline("no model chunks")
    for chunk in chunks:
        f16 = find_float16_module(chunk)
        if f16 is None:
            return _decline("no Float16Module in the wrapper chain: there is no fp32 upcast to skip")
        if not _accepts_fp32_output(f16):
            return _decline(
                "this megatron-core's Float16Module.forward has no `fp32_output` keyword; the "
                "upcast is unconditional and cannot be skipped without patching it"
            )
        cfg = getattr(f16, "config", None)
        if not (getattr(cfg, "bf16", False) or getattr(cfg, "fp16", False)):
            return _decline("Float16Module config declares neither bf16 nor fp16")
    if admitted_output_dtype(chunks) is None:
        return _decline("model chunks do not declare one unanimous bf16/fp16 output dtype")
    return True


def forward_kwargs(admitted: bool) -> dict:
    """``{"fp32_output": False}`` when admitted, else ``{}``."""
    return {"fp32_output": False} if admitted else {}


def _report() -> None:
    served = _S["served"]
    # Power-of-two census cadence: the first served call is evidence and long runs stay quiet.
    if served < 1 or (served & (served - 1)) != 0 or served == _S["reported"]:
        return
    print(
        f"{BANNER} CENSUS pid={os.getpid()} calls={_S['calls']} served={served} "
        f"declined={_S['declined']} out_dtypes={_S['out_dtypes']} "
        f"last_decline={_S['decline_reason']!r}",
        flush=True,
    )
    _S["reported"] = served
    _S["bannered"] = True
