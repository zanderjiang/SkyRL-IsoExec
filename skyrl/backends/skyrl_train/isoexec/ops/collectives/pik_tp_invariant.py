"""Tensor-parallel-invariant row-parallel linear, via the vendored ``pik`` package.

K is cut into ``G`` fixed leaves summed by a fixed balanced tree, so the expression tree is identical
for every TP size dividing G. Gated by ``SKYRL_ISOEXEC_PIK``; idempotent and reversible.
"""

from __future__ import annotations

import logging
import os

import torch

from .pik_bootstrap import ensure_pik

logger = logging.getLogger(__name__)

_PATCHED = False
_ORIG_ROW_FORWARD = None
_PLAN = None
_FELL_BACK: set = set()  # (reason, K, tp) tuples we've already warned about, to avoid log spam


def pik_enabled() -> bool:
    return os.environ.get("SKYRL_ISOEXEC_PIK", "0") == "1"


def trainer_sp_enabled() -> bool:
    """``SKYRL_ISOEXEC_TRAINER_SP`` (default off): Megatron sequence parallelism in the trainer.

    Must gate both halves: SP with the native reduce-scatter breaks TP-invariance.
    """
    return os.environ.get("SKYRL_ISOEXEC_TRAINER_SP", "0") == "1"


def _num_leaves() -> int:
    g = int(os.environ.get("SKYRL_ISOEXEC_PIK_LEAVES", "8"))
    if g < 1 or (g & (g - 1)) != 0:
        raise ValueError(
            f"SKYRL_ISOEXEC_PIK_LEAVES must be a power of two, got {g}. It is a CONTRACT CONSTANT: "
            "set it to the largest TP either the trainer or the engine will ever run (and no larger "
            "-- every extra leaf is pure cost), and set it identically on both sides."
        )
    return g


def _leaf_dtype() -> torch.dtype:
    # The manifest pins the resolved value, so trainer and engine cannot split on it.
    s = os.environ.get("SKYRL_ISOEXEC_PIK_LEAF_DTYPE", "bf16").lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32", "f32"):
        return torch.float32
    raise ValueError(f"SKYRL_ISOEXEC_PIK_LEAF_DTYPE must be bf16 or fp32, got {s!r}")


def _disable_pik_p2p() -> None:
    """Force pik's tree all-reduce onto NCCL, disabling the P2P symmetric-memory path.

    Colocated trainer/engine ranks share GPUs and ``symm_mem.rendezvous`` aborts. Bitwise neutral:
    NCCL moves bytes, pik still owns the tree.
    """
    import pik.allreduce as _ar  # type: ignore
    import pik.linear as _lin  # type: ignore

    _false = lambda *a, **k: False  # noqa: E731
    _ar.p2p_available = _false
    _lin.p2p_available = _false  # linear.py bound its own reference at import
    try:
        from pik.integrations import vllm_patch as _vp  # type: ignore

        _vp.p2p_available = _false
    except Exception:
        pass
    print(
        "[ISOEXEC-PIK] disabled pik P2P symmetric-memory all-reduce -> NCCL transport "
        "(colocated trainer/engine share GPUs; pik still owns the tree arithmetic)",
        flush=True,
    )


# Which transport actually ran: a silent P2P->NCCL degradation costs latency with no other symptom.
# impl_id stays `pik_tree` either way, since both transports evaluate the same tree.
def pik_transport_status() -> dict:
    """Counts of which transport pik's tree all-reduce ran on in THIS process.

    ``state == "none"`` means no all-reduce has run yet (TP=1, or pre-forward), not that P2P is off.
    """
    try:
        ensure_pik()
        import pik.allreduce as _ar  # type: ignore

        return _ar.transport_counts()
    except Exception as e:  # noqa: BLE001 -- a status read must never raise
        return {"state": "unknown", "error": repr(e)}


def _record_transport_fingerprint(side: str, state: str, impl_fn) -> None:
    """Record the resolved transport against ``collectives.tree_all_reduce``. Fail-soft."""
    try:
        from ...core.fingerprint import (
            ENGINE_SITES,
            TRAINER_SITES,
            log_fingerprint_once,
            record_installs,
        )
        from ...core.process_contract import cached_contract_view

        if side == "ENGINE":
            record_installs("collectives.tree_all_reduce", ENGINE_SITES, "pik_tree", impl_fn)
        elif side == "TRAINER":
            record_installs("collectives.tree_all_reduce", TRAINER_SITES, "pik_tree", impl_fn)
        else:
            return  # a harness ("TEST"/"?") has no site vocabulary
        log_fingerprint_once(cached_contract_view(), tag=f"{side.lower()}_first_collective")
    except Exception as e:  # pragma: no cover - never fatal
        logger.warning(f"[ISOEXEC-FINGERPRINT] transport record skipped: {e}")


def _install_transport_hook(side: str) -> None:
    """Make the FIRST resolved transport (and any later degradation) a logged, recorded fact."""
    ensure_pik()
    import pik.allreduce as _ar  # type: ignore

    def _on_transport(state: str, impl_fn) -> None:
        counts = _ar.transport_counts()
        print(
            f"[ISOEXEC-{side}] pik tree all-reduce TRANSPORT RESOLVED: {state} via "
            f"{getattr(impl_fn, '__module__', '?')}.{getattr(impl_fn, '__qualname__', impl_fn)} "
            f"(p2p={counts['p2p_calls']} nccl={counts['nccl_calls']} fallbacks={counts['p2p_fallbacks']})",
            flush=True,
        )
        _record_transport_fingerprint(side, state, impl_fn)

    _ar.register_transport_hook(_on_transport)


def get_plan():
    """The single shared ReductionPlan. Both trainer and engine MUST build the same one."""
    global _PLAN
    if _PLAN is None:
        ensure_pik()
        from pik import ReductionPlan  # type: ignore

        _PLAN = ReductionPlan(num_leaves=_num_leaves(), leaf_dtype=_leaf_dtype())
    return _PLAN


def _assert_plan_matches_manifest(side: str, plan) -> None:
    """Refuse the install when the plan the env built is not the composition the contract names.

    Skipped with no built contract (benches, unit harnesses).
    """
    try:
        from ...core.process_contract import cached_contract_view

        view = cached_contract_view()
    except Exception:  # noqa: BLE001 -- a status read must never break the install
        return
    if view is None:
        return
    pinned = None
    for (op, _site), entry in view.items():
        if op == "collectives.tree_all_reduce":
            pinned = entry.get("pinned_constants") or {}
            break
    if not pinned:
        return
    want_leaves = pinned.get("leaves")
    want_dtype = pinned.get("leaf_dtype")
    have_dtype = "bf16" if plan.bf16_leaves else "fp32"
    problems = []
    if want_leaves is not None and int(want_leaves) != int(plan.num_leaves):
        problems.append(f"leaves: manifest pins {want_leaves}, env built G={plan.num_leaves}")
    if want_dtype is not None and str(want_dtype) != have_dtype:
        problems.append(f"leaf_dtype: manifest pins {want_dtype!r}, env built {have_dtype!r}")
    # Report the pin check per site; fail-safe, verdict unchanged.
    try:
        from ...core import enforce
        from ...core.fingerprint import ENGINE_SITES, TRAINER_SITES

        for site in {"ENGINE": ENGINE_SITES, "TRAINER": TRAINER_SITES}.get(side, ()):
            enforce.report(
                f"fingerprint:collectives.tree_all_reduce:{site}",
                enforce.INSTALL,
                enforce.OK if not problems else enforce.VIOLATION,
                "; ".join(problems) or f"plan leaves={plan.num_leaves} leaf_dtype={have_dtype} match the pins",
            )
    except Exception as _e:  # pragma: no cover - never fatal
        logger.warning(f"[ISOEXEC-ENFORCE] pik plan report skipped: {_e}")
    if not problems:
        return
    msg = (
        f"[ISOEXEC-{side}] pik plan/manifest SPLIT: {'; '.join(problems)}. The reduction tree the "
        "env vars built is not the composition this process's manifest names (and hashes), so the "
        "weight-sync handshake would agree about the wrong function. Set "
        "SKYRL_ISOEXEC_PIK_LEAVES / SKYRL_ISOEXEC_PIK_LEAF_DTYPE to the pinned values, or change "
        "the model profile's pik_leaves / pik_leaf_dtype so the manifest names what you run "
        "(a different composition: new hash, new gate signature, new proof obligation)."
    )
    # Go through the demotion helper so a debug run still reaches its trace.
    from ...core.enforce import refuse

    refuse(msg)


class _PikRowParallel(torch.autograd.Function):
    """Forward is pik's leaf-tree GEMM plus tree all-reduce; backward reduces over N and M, never K.

    Backward therefore carries no invariance requirement and runs stock cuBLAS.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, tp_size, tp_rank, k_full, group, out_dtype, accum_main_grad, sequence_parallel):
        ensure_pik()
        from pik.linear import row_parallel_linear as _row_fwd  # type: ignore
        from pik.linear import row_parallel_linear_rs as _row_fwd_rs  # type: ignore

        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        ctx.accum_main_grad = accum_main_grad
        ctx.sequence_parallel = sequence_parallel
        ctx.group = group
        with torch.no_grad():
            if sequence_parallel:
                # Same leaf-tree GEMM, tree_reduce_scatter combine -> this rank's sequence slice.
                # Bias is None by contract; the caller adds it on the slice.
                assert bias is None, "sequence-parallel pik row-parallel adds bias outside the op"
                y = _row_fwd_rs(
                    x,
                    weight,
                    plan=get_plan(),
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                    k_full=k_full,
                    group=group,
                    out_dtype=out_dtype,
                )
            else:
                y = _row_fwd(
                    x,
                    weight,
                    bias,
                    plan=get_plan(),
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                    k_full=k_full,
                    group=group,
                    out_dtype=out_dtype,
                )
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        dy = dy.contiguous()
        if ctx.sequence_parallel:
            # Backward of a reduce-scatter is an all-gather along the same dim: transport only, so a
            # plain NCCL all-gather suffices (backward owes correctness, not invariance).
            import torch.distributed as dist

            world = dist.get_world_size(ctx.group)
            if world > 1:
                full = torch.empty((dy.shape[0] * world, *dy.shape[1:]), device=dy.device, dtype=dy.dtype)
                dist.all_gather_into_tensor(full, dy, group=ctx.group)
                dy = full
        n = dy.shape[-1]
        dy2 = dy.reshape(-1, n)
        x2 = x.reshape(-1, x.shape[-1])

        # Row-parallel does not shard N, so dY is already full on every rank: no comm, no K-reduction.
        dx = (dy2 @ weight).reshape(x.shape).to(x.dtype)
        dw = (dy2.t() @ x2).to(weight.dtype)
        db = dy2.sum(0).to(weight.dtype) if ctx.has_bias else None

        wgrad = dw
        if ctx.accum_main_grad and hasattr(weight, "main_grad") and weight.main_grad is not None:
            # Under grad-accum fusion the optimizer reads main_grad, not .grad: accumulate there.
            weight.main_grad.add_(dw.to(weight.main_grad.dtype))
            wgrad = None
        return dx, wgrad, db, None, None, None, None, None, None, None


def _pik_supported(self, k_full: int, tp_size: int) -> bool:
    """Whether this (layer, K, TP) is expressible under the plan; an inexpressible one warns once."""
    plan = get_plan()
    g = plan.num_leaves
    reason = None
    if tp_size > g:
        reason = f"tp_size={tp_size} > G={g}"
    elif g % tp_size != 0:
        reason = f"tp_size={tp_size} does not divide G={g}"
    elif k_full % g != 0:
        reason = f"K={k_full} not divisible by G={g}"
    if reason is None:
        return True
    key = (reason,)
    if key not in _FELL_BACK:
        _FELL_BACK.add(key)
        print(
            f"[ISOEXEC-PIK] WARNING: falling back to NATIVE row-parallel for a layer ({reason}). "
            f"This layer is NOT TP-invariant across a trainer/engine TP mismatch. Choose "
            f"SKYRL_ISOEXEC_PIK_LEAVES so every row-parallel K is divisible by it and every TP "
            f"divides it (power of two).",
            flush=True,
        )
    return False


def _pik_row_forward(self, input_: torch.Tensor):
    """Drop-in for Megatron ``RowParallelLinear.forward`` using pik's TP-invariant reduction tree."""
    group = getattr(self, "tp_group", None)
    try:
        tp_size = torch.distributed.get_world_size(group) if group is not None else 1
    except Exception:
        tp_size = 1
    tp_rank = torch.distributed.get_rank(group) if (group is not None and tp_size > 1) else 0

    k_full = int(getattr(self, "input_size"))
    is_expert = getattr(self, "is_expert", False)
    # Expert fc2 needs pik only at ETP>1; at ETP=1 it is already a whole non-split-K GEMM, and pik
    # would crash on the 0-token experts the sequential MoE loop produces (cuBLASLt rejects M=0).
    sp = bool(getattr(self, "sequence_parallel", False)) and tp_size > 1
    if sp and not trainer_sp_enabled():
        # SP layers go native unless the trainer-SP flag is set.
        return _ORIG_ROW_FORWARD(self, input_)
    if (
        (is_expert and tp_size == 1)
        or getattr(self, "explicit_expert_comm", False)
        or (input_.numel() == 0 or input_.shape[0] == 0)  # empty (0-token) GEMM -> native
        or not _pik_supported(self, k_full, tp_size)
    ):
        if sp and not getattr(self, "explicit_expert_comm", False):
            # Native under SP swaps the pik tree for an NCCL reduce-scatter: neither TP-invariant nor
            # bitwise vs SP-off. explicit_expert_comm is exempt (that path performs no reduce).
            key = ("sp_native_fallback", k_full, tp_size)
            if key not in _FELL_BACK:
                _FELL_BACK.add(key)
                print(
                    f"[ISOEXEC-PIK] WARNING: SEQUENCE-PARALLEL layer (K={k_full}, tp={tp_size}) "
                    f"fell back to the NATIVE NCCL reduce-scatter -- this layer's forward is no "
                    f"longer bitwise vs SP-off or across TP sizes. Fix the plan (leaves/K "
                    f"divisibility) or run this recipe with SKYRL_ISOEXEC_TRAINER_SP=0.",
                    flush=True,
                )
        return _ORIG_ROW_FORWARD(self, input_)

    if self.input_is_parallel:
        x = input_
    else:
        from megatron.core.tensor_parallel.mappings import (
            scatter_to_tensor_model_parallel_region,
        )

        assert not self.sequence_parallel
        x = scatter_to_tensor_model_parallel_region(input_, group=group)

    out_dtype = input_.dtype
    accum_main_grad = bool(getattr(self, "gradient_accumulation_fusion", False))

    if sp:
        # Mirror native SP: bias is added on the scattered slice, outside the op.
        output = _PikRowParallel.apply(
            x, self.weight, None, tp_size, tp_rank, k_full, group, out_dtype, accum_main_grad, True
        )
        if not self.skip_bias_add:
            output = (output + self.bias) if self.bias is not None else output
            return output, None
        return output, self.bias

    bias = None if self.skip_bias_add else self.bias
    output = _PikRowParallel.apply(
        x, self.weight, bias, tp_size, tp_rank, k_full, group, out_dtype, accum_main_grad, False
    )

    output_bias = self.bias if self.skip_bias_add else None
    return output, output_bias


def apply_pik_tp_invariant(side: str = "?", selfcheck: bool = False) -> bool:
    """Patch Megatron's RowParallelLinear onto the pik TP-invariant reduction tree. Idempotent.

    ``side`` is only for the banner. Returns True if active; no-op unless ``SKYRL_ISOEXEC_PIK=1``.
    """
    global _PATCHED, _ORIG_ROW_FORWARD
    if not pik_enabled():
        return False
    if _PATCHED:
        return True

    ensure_pik()
    # P2P symm-mem is disabled on the TRAINER (colocated ranks share GPUs, rendezvous aborts) and kept
    # on the ENGINE (1 rank/GPU). Bitwise neutral; SKYRL_ISOEXEC_PIK_P2P=0 disables it on both sides.
    _p2p_engine_ok = side == "ENGINE" and os.environ.get("SKYRL_ISOEXEC_PIK_P2P", "1") == "1"
    if not _p2p_engine_ok:
        _disable_pik_p2p()
    else:
        print(
            "[ISOEXEC-ENGINE] pik P2P symmetric-memory all-reduce KEPT (engine TP group is 1 rank/GPU, "
            "so the colocation overlap that forces NCCL on the trainer does not apply; bitwise-equal). "
            "This states an INTENTION -- wait for the TRANSPORT RESOLVED line for what actually ran.",
            flush=True,
        )
    # The banners state an intention; this records what actually ran, at the first all-reduce.
    _install_transport_hook(side)
    if selfcheck or os.environ.get("SKYRL_ISOEXEC_PIK_SELFCHECK") == "1":
        from .pik_bootstrap import pik_arch_selfcheck

        if not pik_arch_selfcheck():
            raise RuntimeError(
                "[isoexec-pik] arch self-check FAILED on this GPU: a GEMM tiling knob moves the bits "
                "of a non-split-K K-reduction, so pik's TP-invariance premise does not hold here. "
                "Refusing to enable (would silently break IsoExec). See pik/arch.py."
            )

    from megatron.core.tensor_parallel.layers import RowParallelLinear

    _ORIG_ROW_FORWARD = RowParallelLinear.forward
    RowParallelLinear.forward = _pik_row_forward
    _PATCHED = True

    plan = get_plan()
    _assert_plan_matches_manifest(side, plan)
    print(
        f"[ISOEXEC-{side}] pik TP-INVARIANT row-parallel installed: {plan.contract()}. "
        f"Row-parallel (o_proj/down_proj) K-reduction now follows a fixed G={plan.num_leaves} leaf "
        f"tree -> bitwise-identical across ANY trainer/engine TP that divides G. Column-parallel "
        f"untouched (already TP-invariant).",
        flush=True,
    )
    return True


def revert_pik_tp_invariant() -> None:
    global _PATCHED
    if not _PATCHED:
        return
    from megatron.core.tensor_parallel.layers import RowParallelLinear

    RowParallelLinear.forward = _ORIG_ROW_FORWARD
    _PATCHED = False


def pik_status() -> dict:
    out = {
        "pik_enabled": pik_enabled(),
        "row_parallel_patched": _PATCHED,
        "contract": get_plan().contract() if pik_enabled() else None,
        # Which transport the tree all-reduce actually ran on.
        "transport": pik_transport_status() if _PATCHED else {"state": "none"},
    }
    # Launch-structure decisions: what ran, not what was exported.
    try:
        from .pik_bootstrap import ensure_pik

        ensure_pik()
        import pik.allreduce as _ar  # type: ignore
        import pik.ar_branch as _abr  # type: ignore

        out["fused_barrier"] = _ar.fused_counts()
        out["root_cast"] = _ar.root_cast_counts()
        out["ar_crossover"] = _abr.status()
    except Exception as e:  # noqa: BLE001 -- status must never break a caller
        out["launch_structure"] = f"unavailable: {type(e).__name__}: {e}"
    # The MoE owner combine rendezvouses itself, so its counters live outside pik.
    try:
        from ..moe.moe_pik_combine_owner import fused_owner_counts

        out["fused_owner_combine"] = fused_owner_counts()
    except Exception as e:  # noqa: BLE001 -- status must never break a caller
        out["fused_owner_combine"] = f"unavailable: {type(e).__name__}: {e}"
    # The owner combine's producer-side staging counters.
    try:
        from ..moe.moe_pik_combine_owner import wire_stage_counts

        out["moe_wire_stage"] = wire_stage_counts()
    except Exception as e:  # noqa: BLE001 -- status must never break a caller
        out["moe_wire_stage"] = f"unavailable: {type(e).__name__}: {e}"
    return out
