"""Scope the global batch-invariant mm/addmm/bmm override to the FORWARD only.

vLLM registers its Triton persistent matmul on ``aten::mm``/``aten::addmm``/``aten::bmm`` at the
CUDA dispatch key with no forward/backward gate, so the trainer's dgrad and wgrad -- which have no
invariance requirement -- also pay the Triton kernel. The IsoExec gate is a forward metric, so every
forward stays on Triton bit for bit and the backward falls through to cuBLAS.

POLARITY. A backward that stays on Triton is merely slow; a forward that reaches cuBLAS breaks the
gate silently. So Triton is the default and cuBLAS is taken only on a positively identified backward
kernel: every probe failure, unknown, or unhooked path degrades to Triton.

THE DISCRIMINATOR is ``torch._C._current_graph_task_id()`` -- per-thread autograd state, ``-1``
outside a backward graph task. Neither grad mode nor a forward-thread flag works: both runtimes'
forwards must stay on Triton whatever their grad mode, and the backward runs on a different thread,
so the verdict has to be reached on the autograd thread itself.

THE RECOMPUTE SUBTLETY is the one thing that can move bits. A ``recompute_granularity=full``
re-forward runs inside the graph task and so looks exactly like a backward, yet its activations must
match the original forward bitwise. Grad mode cannot rescue it either: megatron's parallel linears
do their GEMM inside a ``torch.autograd.Function.forward``, which PyTorch runs with grad DISABLED
even during a recompute. So the recompute is marked explicitly by wrapping the checkpoint
``run_function`` with a thread-local depth counter raised only when the wrapper is invoked inside a
graph task. Fail-closed: if that hook cannot be installed, the scoping is not installed either.

THE cuBLAS FALLTHROUGH is the ``.out`` overload. Calling ``torch.mm`` from inside an ``aten::mm``
CUDA kernel recurses, and dispatch-key exclusion has no fallback for ``aten::mm.out``; but
``lib.impl("aten::mm", ...)`` registers the ``default`` overload only, so ``aten::mm.out`` still
reaches ATen's native cuBLAS kernel. ``CUBLAS_WORKSPACE_CONFIG`` is already pinned process-wide, so
the fallthrough is non-split-K and deterministic.

INSTALL ORDER for the bmm half is the opposite of mm's: vLLM's ``enable_batch_invariant_mode``
registers ``aten::bmm`` unconditionally and runs LATER than the mm install site, so a scoped bmm
registered there would be silently clobbered. ``install_bmm_scope()`` is therefore callable from
every plausible site and refuses until vLLM's mode is already on. Its registration is then proven,
not trusted, by ``_verify_bmm_scope``. vLLM's python ``torch.bmm`` attribute rebind is deliberately
left alone: it bypasses the dispatcher and records no autograd, so nothing reaching it is a backward.

Flags: ``SKYRL_ISOEXEC_MM_FWD_ONLY=1`` (mm/addmm) and ``SKYRL_ISOEXEC_MM_FWD_ONLY_BMM=1`` (bmm), both
default off in code and independent of each other. Both are backward-only, so bitwise-neutral on
both runtimes by construction; ``0`` restores the unscoped global override exactly.
"""

from __future__ import annotations

import os
import threading

import torch

# Per-thread recompute depth. A fresh thread defaults to 0, which with the graph-task probe means
# "forward -> Triton".
_tls = threading.local()

_LOG_ONCE = False
_RECOMPUTE_HOOK_INSTALLED = False
_recompute_fired = {"n": 0}  # diagnostic: proves the re-entry hook actually ran on a live step

# Captured once. If torch ever drops this probe the scoping is disabled rather than guessed at.
_current_graph_task_id = getattr(torch._C, "_current_graph_task_id", None)


def mm_fwd_only_enabled() -> bool:
    """``SKYRL_ISOEXEC_MM_FWD_ONLY``. Default OFF, i.e. the unscoped global override."""
    return os.environ.get("SKYRL_ISOEXEC_MM_FWD_ONLY", "0") == "1"


def mm_fwd_only_bmm_enabled() -> bool:
    """``SKYRL_ISOEXEC_MM_FWD_ONLY_BMM``. Default OFF, i.e. vLLM's unscoped ``aten::bmm`` override.

    Independent of ``SKYRL_ISOEXEC_MM_FWD_ONLY``: the two halves register on different Libraries at
    different moments."""
    return os.environ.get("SKYRL_ISOEXEC_MM_FWD_ONLY_BMM", "0") == "1"


def _in_autograd_graph_task() -> bool:
    """True iff a backward graph task is executing on this thread; any doubt reports False, which
    routes to Triton."""
    try:
        return _current_graph_task_id() != -1
    except Exception:  # noqa: BLE001
        return False


def use_batch_invariant() -> bool:
    """True -> the Triton batch-invariant kernel. Every disjunct pushes toward Triton; cuBLAS
    requires all of them to be false, i.e. a positively identified backward kernel."""
    if not _in_autograd_graph_task():
        return True  # every gate-feeding forward on both runtimes
    if getattr(_tls, "recompute", 0) > 0:
        return True  # checkpoint re-forward: activations must match the original forward bitwise
    if torch.is_grad_enabled():
        return True  # redundant insurance (recompute bodies, create_graph double-backward)
    return False  # a real backward kernel -- unconstrained, take cuBLAS


# The cuBLAS fallthrough: the ``.out`` overloads are separate dispatcher entries from the ``default``
# ones overridden here, so they reach ATen's native cuBLAS kernel without recursing.
def _cublas_mm(a, b):
    out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
    return torch.ops.aten.mm.out(a, b, out=out)


def _cublas_addmm(bias, a, b, *, beta=1, alpha=1):
    out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
    return torch.ops.aten.addmm.out(bias, a, b, beta=beta, alpha=alpha, out=out)


def _cublas_bmm(a, b):
    out = torch.empty((a.shape[0], a.shape[1], b.shape[2]), device=a.device, dtype=a.dtype)
    return torch.ops.aten.bmm.out(a, b, out=out)


def make_scoped_impls():
    """Return ``(mm_impl, addmm_impl)`` -- the functions to register on aten::mm / aten::addmm.

    The Triton branch calls vLLM's own wrappers exactly as the unscoped install does, so it resolves
    ``bi.matmul_persistent`` through module globals at call time and still picks up ``mm_tiles`` /
    ``mm_cublaslt``; the forward composition is untouched by construction.
    """
    from vllm.model_executor.layers import batch_invariant as bi

    def mm_impl(a, b):
        if use_batch_invariant():
            return bi.mm_batch_invariant(a, b)
        try:
            return _cublas_mm(a, b)
        except Exception:  # noqa: BLE001  -- any odd layout falls back to the safe kernel
            return bi.mm_batch_invariant(a, b)

    def addmm_impl(bias, a, b, **kwargs):
        if use_batch_invariant():
            return bi.addmm_batch_invariant(bias, a, b, **kwargs)
        try:
            return _cublas_addmm(bias, a, b, **kwargs)
        except Exception:  # noqa: BLE001
            return bi.addmm_batch_invariant(bias, a, b, **kwargs)

    mm_impl._isoexec_fwd_scoped = True
    addmm_impl._isoexec_fwd_scoped = True
    return mm_impl, addmm_impl


# Registration probe. Armed only inside ``_verify_bmm_scope``; one dict lookup per bmm otherwise.
_bmm_probe = {"armed": False, "n": 0}


def make_scoped_bmm_impl():
    """Return the function to register on ``aten::bmm``; its Triton branch calls the same callable
    ``enable_batch_invariant_mode`` registered, resolved through module globals at call time."""
    from vllm.model_executor.layers import batch_invariant as bi

    def bmm_impl(a, b):
        if _bmm_probe["armed"]:
            _bmm_probe["n"] += 1
        if use_batch_invariant():
            return bi.bmm_batch_invariant(a, b)
        try:
            return _cublas_bmm(a, b)
        except Exception:  # noqa: BLE001  -- any odd layout falls back to the safe kernel
            return bi.bmm_batch_invariant(a, b)

    bmm_impl._isoexec_fwd_scoped = True
    return bmm_impl


def _wrap_run_function(fn):
    """Mark this callable's recompute invocations, i.e. the ones that happen inside a graph task.

    The same wrapped object is called twice: on the main thread for the original forward (already
    Triton by default) and on the autograd thread for the re-forward. Marking the callable rather
    than ``CheckpointFunction.backward`` as a whole is what keeps the real backward on cuBLAS."""

    def _isoexec_recompute_scoped(*args, **kwargs):
        if not _in_autograd_graph_task():
            return fn(*args, **kwargs)  # the original forward; already Triton by default
        _tls.recompute = getattr(_tls, "recompute", 0) + 1
        if _recompute_fired["n"] == 0:
            _recompute_fired["n"] = 1
            print(
                "[ISOEXEC-MM-FWDONLY] checkpoint recompute re-entry FIRED -- the re-forward is "
                "pinned back onto the Triton batch-invariant matmul (activations stay bitwise "
                "equal to the original forward); the backward around it keeps cuBLAS.",
                flush=True,
            )
        try:
            return fn(*args, **kwargs)
        finally:
            _tls.recompute -= 1

    _isoexec_recompute_scoped._isoexec_wrapped = fn
    return _isoexec_recompute_scoped


def install_recompute_reentry() -> bool:
    """Wrap every megatron checkpoint entry so its re-forward re-enters the Triton kernel.

    Both recompute implementations (full and selective) are covered by wrapping the callable they
    are handed. Megatron's callers resolve ``tensor_parallel.checkpoint`` as a module attribute at
    call time, so rebinding it reaches all of them; the package re-export is patched too because it
    bound the name at import. Idempotent, and returns False on any failure -- after which the caller
    must NOT install the scoping."""
    global _RECOMPUTE_HOOK_INSTALLED
    if _RECOMPUTE_HOOK_INSTALLED:
        return True
    try:
        from megatron.core import tensor_parallel as _tp
        from megatron.core.tensor_parallel import random as _rand
    except Exception as e:  # noqa: BLE001
        print(
            f"[ISOEXEC-MM-FWDONLY] megatron checkpoint module not importable ({type(e).__name__}: {e}) "
            f"-- forward-only mm scoping NOT installed (fail-closed).",
            flush=True,
        )
        return False

    try:
        orig_checkpoint = _rand.checkpoint

        def checkpoint(function, distribute_saved_activations, *args):
            return orig_checkpoint(_wrap_run_function(function), distribute_saved_activations, *args)

        checkpoint._isoexec_fwd_scoped = True
        _rand.checkpoint = checkpoint
        patched = ["tensor_parallel.random.checkpoint"]
        # The package re-export is what every megatron caller actually resolves.
        if getattr(_tp, "checkpoint", None) is orig_checkpoint:
            _tp.checkpoint = checkpoint
            patched.append("tensor_parallel.checkpoint")

        # Selective recompute: mark the callable before CheckpointWithoutOutput stores it.
        cwo = getattr(_rand, "CheckpointWithoutOutput", None)
        if cwo is not None and not getattr(cwo, "_isoexec_fwd_scoped", False):
            _orig_cwo = cwo.checkpoint

            def cwo_checkpoint(self, run_function, *args):
                return _orig_cwo(self, _wrap_run_function(run_function), *args)

            cwo.checkpoint = cwo_checkpoint
            cwo._isoexec_fwd_scoped = True
            patched.append("CheckpointWithoutOutput.checkpoint")
    except Exception as e:  # noqa: BLE001
        print(
            f"[ISOEXEC-MM-FWDONLY] recompute hook install FAILED ({type(e).__name__}: {e}) "
            f"-- forward-only mm scoping NOT installed (fail-closed).",
            flush=True,
        )
        return False

    _RECOMPUTE_HOOK_INSTALLED = True
    print(f"[ISOEXEC-MM-FWDONLY] recompute re-entry hook installed on: {', '.join(patched)}", flush=True)
    return True


def _self_check() -> bool:
    """Prove in-process at install time that the ``.out`` fallthrough is recursion-free and bitwise
    equal to unoverridden cuBLAS, and that the predicate says Triton outside a graph task.

    Must run BEFORE the registration exists, so the reference values are genuinely un-overridden; a
    torch build that routed ``.out`` back through the ``default`` overload recurses here and fails
    loudly instead of silently corrupting the backward."""
    try:
        if not use_batch_invariant():
            return False  # module scope is a forward; must be Triton
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if dev == "cpu":
            return True  # nothing to certify off-GPU; the aten override is CUDA-only anyway
        a = torch.randn(37, 65, device=dev, dtype=torch.bfloat16)
        b = torch.randn(65, 43, device=dev, dtype=torch.bfloat16)
        bias = torch.randn(43, device=dev, dtype=torch.bfloat16)
        ref_mm, ref_addmm = torch.mm(a, b), torch.addmm(bias, a, b)
        if not torch.equal(_cublas_mm(a, b), ref_mm):
            return False
        return bool(torch.equal(_cublas_addmm(bias, a, b), ref_addmm))
    except Exception:  # noqa: BLE001
        return False


# The bmm half, registered on its own Library at its own moment -- see the module docstring on
# install order for why it cannot ride along with mm/addmm.
_bmm_lib = None
_BMM_INSTALLED = False
_BMM_REFUSED_LOGGED = False


def _self_check_bmm() -> bool:
    """Certify the ``aten::bmm.out`` fallthrough before anything is registered on it.

    Unlike mm there is no un-overridden ``torch.bmm`` to compare against -- vLLM has already claimed
    ``aten::bmm``, which is the install precondition -- so this only certifies that ``.out`` computes
    a bmm and does not recurse. The bitwise half runs after registration in ``_verify_bmm_scope``.
    """
    try:
        if not use_batch_invariant():
            return False  # module scope is a forward; must be Triton
        if not torch.cuda.is_available():
            return True  # nothing to certify off-GPU; the aten override is CUDA-only anyway
        from vllm.model_executor.layers import batch_invariant as bi

        dev = "cuda"
        a = torch.randn(5, 37, 65, device=dev, dtype=torch.bfloat16)
        b = torch.randn(5, 65, 43, device=dev, dtype=torch.bfloat16)
        got = _cublas_bmm(a, b)
        if got.shape != (5, 37, 43):
            return False
        if not torch.equal(got, _cublas_bmm(a, b)):
            return False  # cuBLAS must be run-to-run stable (CUBLAS_WORKSPACE_CONFIG is pinned)
        ref = bi.bmm_batch_invariant(a, b)
        return bool(torch.allclose(got.float(), ref.float(), rtol=2e-2, atol=2e-2))
    except Exception:  # noqa: BLE001
        return False


def _verify_bmm_scope() -> bool:
    """Prove in-process, after registration, that the scoping is what the dispatcher reaches.

    (a) ours is the live impl -- vLLM registers ``aten::bmm`` too and the loser says nothing, so
        entries are counted through a real dispatch;
    (b) the forward is bitwise ``bmm_batch_invariant``, which is the gate metric;
    (c) the backward is recursion-free (exactly one entry per backward GEMM);
    (d) two identical fwd+bwd give bitwise-identical grads.
    """
    try:
        if not torch.cuda.is_available():
            return True
        from vllm.model_executor.layers import batch_invariant as bi

        dev = "cuda"
        torch.manual_seed(0)
        a0 = torch.randn(5, 37, 65, device=dev, dtype=torch.bfloat16)
        b0 = torch.randn(5, 65, 43, device=dev, dtype=torch.bfloat16)

        _bmm_probe["armed"], _bmm_probe["n"] = True, 0
        try:
            # (a) + (b): one entry, and the forward is the Triton kernel bit for bit.
            a = a0.clone().requires_grad_(True)
            b = b0.clone().requires_grad_(True)
            out = torch.ops.aten.bmm(a, b)
            if _bmm_probe["n"] != 1:
                return False
            if not torch.equal(out, bi.bmm_batch_invariant(a0, b0)):
                return False
            # (c): the backward's dgrad+wgrad -> 2 more entries, no recursion.
            _bmm_probe["n"] = 0
            out.float().square().sum().backward()
            torch.cuda.synchronize()
            if _bmm_probe["n"] != 2:
                return False
        finally:
            _bmm_probe["armed"] = False

        # (d): determinism of the whole fwd+bwd.
        def _grads():
            x = a0.clone().requires_grad_(True)
            y = b0.clone().requires_grad_(True)
            torch.ops.aten.bmm(x, y).float().square().sum().backward()
            return x.grad, y.grad

        g1a, g1b = _grads()
        g2a, g2b = _grads()
        return bool(torch.equal(g1a, g2a) and torch.equal(g1b, g2b))
    except Exception:  # noqa: BLE001
        return False


def install_bmm_scope() -> bool:
    """Register the forward-scoped ``aten::bmm``. Idempotent, order-robust, fail-closed.

    Call it from every site that could plausibly be after vLLM's ``enable_batch_invariant_mode``; it
    no-ops and stays re-callable until that is actually true.
    """
    global _bmm_lib, _BMM_INSTALLED, _BMM_REFUSED_LOGGED

    if _BMM_INSTALLED:
        return True
    if not mm_fwd_only_bmm_enabled():
        return False
    try:
        from vllm.model_executor.layers import batch_invariant as bi
    except Exception:  # noqa: BLE001
        return False
    if not getattr(bi, "_batch_invariant_MODE", False):
        # Too early: enable_batch_invariant_mode has not run, and when it does it would register
        # aten::bmm over the top of ours. Say nothing and let a later call site do the work.
        return False
    if not _self_check_bmm():
        if not _BMM_REFUSED_LOGGED:
            _BMM_REFUSED_LOGGED = True
            print(
                "[ISOEXEC-MM-FWDONLY] bmm .out fallthrough self-check FAILED -- keeping vLLM's "
                "UNSCOPED aten::bmm (forward bits are never at risk; no backward win).",
                flush=True,
            )
        return False
    if not install_recompute_reentry():
        return False  # Fail-closed: no recompute re-entry means no scoping.

    lib = torch.library.Library("aten", "IMPL")
    try:
        lib.impl("aten::bmm", make_scoped_bmm_impl(), "CUDA")
    except Exception as e:  # noqa: BLE001
        print(f"[ISOEXEC-MM-FWDONLY] aten::bmm registration FAILED ({type(e).__name__}: {e})", flush=True)
        lib._destroy()
        return False
    if not _verify_bmm_scope():
        print(
            "[ISOEXEC-MM-FWDONLY] bmm scope post-registration verification FAILED -- de-registering, "
            "vLLM's unscoped aten::bmm stands.",
            flush=True,
        )
        lib._destroy()
        return False

    _bmm_lib = lib
    _BMM_INSTALLED = True
    print(
        "[ISOEXEC-MM-FWDONLY] aten::bmm override SCOPED TO THE FORWARD "
        "(SKYRL_ISOEXEC_MM_FWD_ONLY_BMM=1): forwards (trainer training + scoring, engine prefill + "
        "decode, and checkpoint RE-forwards) keep vLLM's Triton bmm_kernel bit for bit; BmmBackward0 "
        "and the MoE expert-kernel wgrad bmms fall through to non-split-K cuBLAS via the aten .out "
        "overload. Same thread-local discriminator and the SAME recompute re-entry hook as the "
        "mm/addmm half. vLLM's python torch.bmm rebind is deliberately NOT touched (it records no "
        "autograd, so nothing that reaches it is a backward).",
        flush=True,
    )
    return True


def bmm_scope_installed() -> bool:
    """Diagnostic: True once the scoped ``aten::bmm`` is the live dispatcher impl."""
    return _BMM_INSTALLED


def install(lib) -> bool:
    """Register the forward-scoped mm/addmm on ``lib``; False means the caller should install the
    plain unscoped override instead.

    Also makes an opportunistic attempt at the bmm half, which no-ops unless vLLM's mode is already
    on. That attempt is independent of the mm flag and of the mm result."""
    global _LOG_ONCE

    install_bmm_scope()

    if not mm_fwd_only_enabled():
        return False
    # Order matters: the self-check must run while aten::mm still resolves to stock cuBLAS, so its
    # reference values are genuinely un-overridden.
    if not _self_check():
        print(
            "[ISOEXEC-MM-FWDONLY] install self-check FAILED -- keeping the UNSCOPED global override "
            "(forward bits are never at risk; this run simply does not get the backward win).",
            flush=True,
        )
        return False
    if not install_recompute_reentry():
        return False

    mm_impl, addmm_impl = make_scoped_impls()
    lib.impl("aten::mm", mm_impl, "CUDA")
    lib.impl("aten::addmm", addmm_impl, "CUDA")

    if not _LOG_ONCE:
        _LOG_ONCE = True
        print(
            "[ISOEXEC-MM-FWDONLY] mm/addmm override SCOPED TO THE FORWARD "
            "(SKYRL_ISOEXEC_MM_FWD_ONLY=1): forwards (trainer training + scoring, engine prefill + "
            "decode, and checkpoint RE-forwards) keep the Triton batch-invariant kernel bit for "
            "bit; backward dgrad/wgrad fall through to non-split-K cuBLAS via the aten .out "
            "overload (~3.0-3.4x on the trainer's GEMM shapes). The gate is a forward metric and "
            "must not move. Default-Triton polarity: every probe failure and every unmarked path "
            "degrades to Triton, never to cuBLAS.",
            flush=True,
        )
    return True
