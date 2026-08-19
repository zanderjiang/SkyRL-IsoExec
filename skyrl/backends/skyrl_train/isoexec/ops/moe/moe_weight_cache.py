"""Memoize the stacked expert-weight tensors across forwards (``SKYRL_ISOEXEC_MOE_WEIGHT_CACHE``).

``torch.stack`` over the local experts' weights is a pure function of parameters that change only at an
optimizer step or a weight sync, so the result is memoized per ``(module, role)`` and re-validated on every
read against ``(id(p), p.data_ptr(), p._version)``. Those probes cannot see a write through ``param.data``
or through a DDP/optimizer param buffer the parameters are views into, so the cache is additionally keyed
on a global epoch that every weight-mutation site must bump via :func:`invalidate_all`.

Under grad the buffer is returned through ``moe_fused_weights._StackView``; returning the raw detached
buffer would deliver zero gradient to the expert parameters. When the aliasing path in
``moe_fused_weights`` is live it is preferred -- it is memory-neutral, whereas this cache holds a full
extra copy of the local expert weights per layer.
"""

from __future__ import annotations

import logging
import os

import torch

from .moe_fused_weights import _StackView, fused_expert_weights

logger = logging.getLogger(__name__)

_ENV_GATE = "SKYRL_ISOEXEC_MOE_WEIGHT_CACHE"
_STATE_ATTR = "_isoexec_moe_wstack"

# Bumped by the weight-mutation boundaries (see the module docstring). An entry recorded under an
# older epoch is rebuilt unconditionally, whatever the per-parameter probes say.
_EPOCH = 0

BUILD_COUNT = 0  # stacks actually materialised
HIT_COUNT = 0  # forwards served from the cache
INVALIDATE_COUNT = 0  # rebuilds forced by a probe/epoch mismatch


def weight_cache_enabled() -> bool:
    """Default ON. Read per call so an A/B can flip it in-process."""
    return os.environ.get(_ENV_GATE, "1") == "1"


def invalidate_all() -> int:
    """Announce that expert weights may have changed anywhere; returns the new epoch.

    Cheap and unconditional. Call it from any site that writes model weights through a path the
    per-parameter probes cannot see (``param.data``, or a DDP/optimizer param buffer the params are views
    into). Over-calling costs one re-stack per layer.
    """
    global _EPOCH
    _EPOCH += 1
    return _EPOCH


def stats() -> dict:
    return {"builds": BUILD_COUNT, "hits": HIT_COUNT, "invalidations": INVALIDATE_COUNT, "epoch": _EPOCH}


class _Entry:
    """One (module, role) memo: the stacked buffer plus everything needed to re-validate it."""

    __slots__ = ("params", "probes", "buf", "epoch")

    def __init__(self, params, buf):
        self.params = params  # STRONG refs: makes the id() probe below sound (no id recycling)
        self.probes = [(id(p), p.data_ptr(), p._version) for p in params]
        self.buf = buf
        self.epoch = _EPOCH


def _params(module, role: str):
    """The E expert parameters for ``role``, in expert order -- the exact list ``torch.stack`` took."""
    return [getattr(e, role).weight for e in module.local_experts]


def stacked_expert_weights(module, role: str) -> torch.Tensor:
    """The ``[E, *shape]`` stack of ``module.local_experts[i].<role>.weight``, memoized.

    Always returns a correct tensor: on any doubt it re-stacks, which is the uncached expression.
    """
    global BUILD_COUNT, HIT_COUNT, INVALIDATE_COUNT

    params = _params(module, role)

    # The aliasing path, when live, is strictly better than caching a copy: the buffer IS the parameters'
    # storage, so it is memory-neutral and cannot go stale. It returns None whenever the module is not
    # fused or the alias cannot be trusted, and we fall through.
    fused = fused_expert_weights(module, role)
    if fused is not None:
        return fused

    state = module.__dict__.get(_STATE_ATTR)
    if state is None:
        state = module.__dict__[_STATE_ATTR] = {}
    entry = state.get(role)

    probes = [(id(p), p.data_ptr(), p._version) for p in params]
    if entry is not None and entry.epoch == _EPOCH and entry.probes == probes:
        HIT_COUNT += 1
        buf = entry.buf
    else:
        if entry is not None:
            INVALIDATE_COUNT += 1
        with torch.no_grad():
            buf = torch.stack(params).detach()
        BUILD_COUNT += 1
        state[role] = _Entry(params, buf)

    if torch.is_grad_enabled() and any(p.requires_grad for p in params):
        # Re-attach autograd to THIS forward: free alias out, grad unbound back onto the E params.
        return _StackView.apply(buf, *params)
    return buf


def drop(module) -> bool:
    """Forget every cached stack for ``module`` (frees the copy). For teardown / A-B only."""
    return module.__dict__.pop(_STATE_ATTR, None) is not None
