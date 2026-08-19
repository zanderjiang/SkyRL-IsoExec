"""Load FLA's fused Triton GDN backward without disturbing the IsoExec forward shim.

The loader temporarily removes the synthetic ``fla`` modules used by the trainer forward, imports the real
FLA package from ``SKYRL_ISOEXEC_FLA_SOURCE``, keeps the resolved callable, then restores the shim. Only
backward is replaced; the reference VJP is used when ``SKYRL_ISOEXEC_GDN_FLA_BACKWARD`` is unset.
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)


def _default_fla_source() -> str:
    """Repo-local checkout placed by ``build_isoexec_env.sh``.

    FLA is loaded from source rather than installed: transformers gates
    ``from fla.modules import FusedRMSNormGated`` on the distribution being present, and if it is,
    that import resolves to this package's ``fla`` shim and fails.
    """
    from pathlib import Path

    candidate = Path(__file__).resolve().parents[6] / "third_party" / "flash-linear-attention"
    return str(candidate) if (candidate / "fla").is_dir() else ""


#: Path to a flash-linear-attention source checkout. ``SKYRL_ISOEXEC_FLA_SOURCE`` overrides;
#: otherwise the repo-local ``third_party/`` checkout is used when present.
_FLA_SOURCE = os.environ.get("SKYRL_ISOEXEC_FLA_SOURCE", "").strip() or _default_fla_source()

_fla_chunk = None  # memoized real FLA chunk_gated_delta_rule (fwd+bwd)


def fla_backward_enabled() -> bool:
    """Read at call time; default off, in which case the reference VJP is used."""
    return os.environ.get("SKYRL_ISOEXEC_GDN_FLA_BACKWARD") == "1"


def get_fla_chunk_backward():
    """Return the real FLA ``chunk_gated_delta_rule`` (with its fused backward), loaded once.

    Temp-swaps the ``fla`` shim out of ``sys.modules`` for the import, then restores it. A tiny warmup
    forces every kernel-module import while the real ``fla`` is live, so the captured function never needs
    ``sys.modules['fla']`` again.
    """
    global _fla_chunk
    if _fla_chunk is not None:
        return _fla_chunk

    import importlib

    # FLA reads this at call time to pick its backend; force Triton. Set globally rather than
    # restored, because FLA may read it on every call and not just at import.
    os.environ["FLA_DISABLE_BACKEND_DISPATCH"] = "1"

    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "fla" or k.startswith("fla.")}
    added_path = bool(_FLA_SOURCE) and _FLA_SOURCE not in sys.path
    if added_path:
        sys.path.insert(0, _FLA_SOURCE)
    try:
        try:
            real = importlib.import_module("fla.ops.gated_delta_rule")
        except ModuleNotFoundError as exc:
            # Reached only with the lever ON, and only on the first backward -- far from the cause.
            # Name the remedy here rather than surfacing a bare ModuleNotFoundError from a Ray actor.
            raise ModuleNotFoundError(
                "SKYRL_ISOEXEC_GDN_FLA_BACKWARD=1 needs a flash-linear-attention checkout. Run "
                "examples/isoexec/build_isoexec_env.sh to place one at the validated commit, or "
                "point SKYRL_ISOEXEC_FLA_SOURCE at an existing one. Set "
                "SKYRL_ISOEXEC_GDN_FLA_BACKWARD=0 to fall back to the reference VJP."
            ) from exc
        chunk = real.chunk_gated_delta_rule
        _warmup(chunk)  # force deferred imports while the real fla is still in sys.modules
        _fla_chunk = chunk
        ver = getattr(importlib.import_module("fla"), "__version__", "?")
        logger.info(
            "[isoexec-gdn] loaded FLA fused backward (v%s) from %s", ver, _FLA_SOURCE or "the installed environment"
        )
    finally:
        # Drop the real fla.* we imported and put the shim back, so forward routing is unchanged.
        for k in [k for k in sys.modules if k == "fla" or k.startswith("fla.")]:
            del sys.modules[k]
        sys.modules.update(saved)
        if added_path:
            try:
                sys.path.remove(_FLA_SOURCE)
            except ValueError:
                pass
    return _fla_chunk


def _warmup(chunk) -> None:
    """One tiny fwd+bwd so every FLA kernel module is imported before the shim is restored.

    Runs under ``enable_grad`` because this is called from inside a backward pass, where autograd is
    otherwise disabled and the warmup forward would record no graph.
    """
    import torch

    if not torch.cuda.is_available():
        return
    dev, dt = "cuda", torch.bfloat16
    with torch.enable_grad():
        q = torch.nn.functional.normalize(torch.randn(1, 8, 1, 128, device=dev), dim=-1).to(dt).requires_grad_(True)
        k = torch.nn.functional.normalize(torch.randn(1, 8, 1, 128, device=dev), dim=-1).to(dt).requires_grad_(True)
        v = torch.randn(1, 8, 1, 128, device=dev, dtype=dt, requires_grad=True)
        g = torch.nn.functional.logsigmoid(torch.randn(1, 8, 1, device=dev) + 4.0).to(dt).requires_grad_(True)
        beta = torch.rand(1, 8, 1, device=dev, dtype=dt, requires_grad=True)
        o, _ = chunk(q, k, v, g, beta, scale=None, use_qk_l2norm_in_kernel=False, use_beta_sigmoid_in_kernel=False)
        torch.autograd.grad(o, [q, k, v, g, beta], torch.ones_like(o))


def fla_chunk_vjp(q, k, v, g, beta, do, initial_state, cu_seqlens, chunk_size):
    """Gradient of the chunk gated-delta-rule w.r.t. (q, k, v, g, beta) via FLA's fused Triton backward.

    Follows the pinned forward's conventions: q/k are already L2-normed and GQA-expanded, beta is
    already sigmoid'd, and ``scale=None`` means ``K**-0.5``. ``initial_state`` is treated as a
    constant. Returns grads cast back to each input's dtype.
    """
    import torch

    chunk = get_fla_chunk_backward()
    with torch.enable_grad():
        leaves = [t.detach().requires_grad_(True) for t in (q, k, v, g, beta)]
        o, _ = chunk(
            leaves[0],
            leaves[1],
            leaves[2],
            leaves[3],
            leaves[4],
            scale=None,
            initial_state=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=False,
            use_beta_sigmoid_in_kernel=False,
        )
        grads = torch.autograd.grad(o, leaves, do.to(o.dtype))
    return [gr.to(t.dtype) for gr, t in zip(grads, (q, k, v, g, beta))]
