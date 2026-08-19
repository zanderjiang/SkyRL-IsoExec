"""BF16 wire for the BACKWARD half of the MoE expert combine all-to-all (trainer, EP>1).

The FORWARD payload of ``MoEAlltoAllTokenDispatcher.token_combine`` is the fp32 pik leaf-tree fc2 root -- a
sum of bf16 leaves, not bf16-representable -- and narrowing it would round before the top-k sum, so the
forward is left byte for byte alone. The BACKWARD payload is different: it is the VJP of the single bf16
round in ``combine_postprocess``, i.e. an exact upcast of bf16 values, permuted on the way here by a
scatter that does no arithmetic. Narrow -> exchange -> widen is therefore the identity bit for bit, and
that claim is re-checked on live operands with an int32 bit compare (never a float compare, which would
call ``-0.0`` equal to ``+0.0``).

A collective's wire dtype must be agreed by every rank, which makes three rules load-bearing: the
direction predicate is STRUCTURAL (which autograd method is running) and never reads a tensor value; the
first-call losslessness verdict is MIN-reduced across the EP group so one dissenting rank disables the
path everywhere; and every subsequent sampled check RAISES instead of falling back, because a rank-local
fallback puts fp32 on the wire while its peers put bf16 there -- a mismatched collective, not a safe state.

``SKYRL_ISOEXEC_MOE_A2A_BF16_WIRE`` is ``off`` / ``on`` / ``probe``; probe mode censuses both directions and
changes no wire. Install rebinds one method and points ``token_dispatcher.all_to_all`` at the wrapper only
for the duration of that call -- a permanent rebind would also capture the DISPATCH collectives, which must
stay untouched.
"""

from __future__ import annotations

import os

import torch

BANNER = "[ISOEXEC-MOE-A2A-WIRE]"
_ENV = "SKYRL_ISOEXEC_MOE_A2A_BF16_WIRE"

MODE_OFF = "off"
MODE_ON = "on"
MODE_PROBE = "probe"

#: how often the admitted backward re-proves losslessness on live operands (call 1 is the first-call
#: vote; then every N-th admitted call). Each probe costs one collective plus one sync.
PROBE_EVERY = 1024

_S = {
    "mode": None,  # resolved once at install
    "installed": False,
    "bwd_calls": 0,  # every backward that reached this module
    "served": 0,  # backward exchanges actually shipped as bf16
    "declined": 0,  # backward exchanges left fp32
    "bytes_saved": 0,  # wire bytes not sent (send side + recv side)
    "agreed": None,  # None = the EP-group vote has not happened yet
    "decline_reason": "",
    "probes": 0,
    "reported": 0,
}

# device -> int64[4] = [fwd_seen, fwd_pass, bwd_seen, bwd_pass]  (census mode only)
_CENSUS: dict = {}
_census_calls = 0
_census_reported = 0


def wire_mode() -> str:
    """``"off"`` / ``"on"`` / ``"probe"``. Default OFF."""
    v = os.environ.get(_ENV, "0").strip().lower()
    if v == "probe":
        return MODE_PROBE
    if v in ("1", "true", "yes", "on"):
        return MODE_ON
    return MODE_OFF


def roundtrip_is_bitwise(x: torch.Tensor) -> torch.Tensor:
    """0-dim bool tensor ON DEVICE (no host sync): is ``x`` exactly bf16-representable?

    Compared as INTEGERS, deliberately: a float compare would call ``-0.0`` equal to ``+0.0`` and two
    NaNs unequal, while the question here is whether the bf16 round trip returns the same 32 bits.
    Consequences, all intended: ``-0.0`` and +-inf survive and are admitted; NO NaN survives, because
    torch's fp32->bf16 emits the canonical NaN rather than truncating, so every NaN is refused (which on
    the sampled probe stops the run); fp32 subnormals below bf16's grid flush to zero and are refused,
    which the backward payload cannot contain since it is an exact upcast of bf16 values.
    """
    xc = x.contiguous()
    r = xc.to(torch.bfloat16).to(xc.dtype)
    bits = torch.int32 if xc.dtype == torch.float32 else torch.int16
    return torch.eq(xc.view(bits), r.view(bits)).all()


def _world(group) -> int:
    try:
        return int(group.size())
    except Exception:  # noqa: BLE001 - a stub group in a CPU test, or a torn-down PG
        return 1


def _raw_a2a(group, x, output_split_sizes, input_split_sizes, use_nccl_stream):
    """megatron's own ``_AllToAll`` forward, executed with autograd off.

    Delegating rather than copying the body means this follows whatever megatron's collective does
    (contiguity, the world==1 bypass, the nccl-stream variant) across a megatron bump.

    NO DOUBLE BACKWARD: megatron's ``_AllToAll.backward`` re-enters ``_AllToAll.apply`` and is itself
    differentiable, this one is not. A second-order pass through a narrowed wire would need its own
    losslessness argument, so it is refused by construction rather than half-supported.
    """
    from megatron.core.tensor_parallel.mappings import _AllToAll

    with torch.no_grad():
        return _AllToAll.apply(group, x, output_split_sizes, input_split_sizes, use_nccl_stream)


def _census_note(x: torch.Tensor, *, forward: bool) -> None:
    """Accumulate (seen, pass) on DEVICE. No host sync on the hot path."""
    global _census_calls
    t = _CENSUS.get(x.device)
    if t is None:
        t = torch.zeros(4, dtype=torch.int64, device=x.device)
        _CENSUS[x.device] = t
    i = 0 if forward else 2
    t[i] += 1
    t[i + 1] += roundtrip_is_bitwise(x).to(torch.int64)
    _census_calls += 1
    _census_report()


def _census_report(force: bool = False) -> None:
    """One line at 1/2/4/8/... observed payloads. This is the ONLY host sync census mode adds."""
    global _census_reported
    n = _census_calls
    if not force:
        if n < 1 or (n & (n - 1)) != 0 or n == _census_reported:
            return
    _census_reported = n
    tot = [0, 0, 0, 0]
    for t in _CENSUS.values():
        vals = t.tolist()
        tot = [a + b for a, b in zip(tot, vals)]
    fs, fp, bs, bp = tot
    print(
        f"{BANNER} CENSUS pid={os.getpid()} forward {fp}/{fs} bf16-representable | "
        f"backward {bp}/{bs} bf16-representable  "
        f"(expected on real data: forward 0/N -- the fp32 leaf-tree root is a sum of 8 bf16 leaves; "
        f"backward N/N -- an exact upcast permuted by a scatter with no arithmetic)",
        flush=True,
    )


def _agree_lossless(g: torch.Tensor, group) -> bool:
    """The EP-group-wide verdict, taken exactly ONCE, on the first admitted backward.

    MIN over the group means a single dissenting rank disables the bf16 wire on EVERY rank, which is the
    only verdict that keeps the collective symmetric. Every rank reaches this line at the same structural
    position, so the reduction itself cannot desynchronize.
    """
    ok = roundtrip_is_bitwise(g).to(torch.int32).reshape(1)
    try:
        import torch.distributed as dist

        live = dist.is_available() and dist.is_initialized()
    except Exception:  # noqa: BLE001
        live = False
    if live:
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
    # else: unreachable in production -- an all-to-all needs an initialized process group, so a
    # non-initialized dist means there is no wire and therefore no cross-rank hazard (CPU tests).
    return bool(ok.item())


def _drift_probe_or_raise(g: torch.Tensor, group) -> None:
    """Re-prove losslessness on live operands, COLLECTIVELY, and stop the world on failure.

    Read the module docstring before "fixing" this to fall back: falling back here is the failure
    mode. One rank sending fp32 while its peers send bf16 is a mismatched all-to-all.
    """
    ok = roundtrip_is_bitwise(g).to(torch.int32).reshape(1)
    try:
        import torch.distributed as dist

        live = dist.is_available() and dist.is_initialized()
    except Exception:  # noqa: BLE001
        live = False
    if live:
        # MIN so that a failure on ANY rank raises on ALL of them. A rank-local raise would leave
        # the survivors deadlocked on the next collective instead of dying with this message.
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
    _S["probes"] += 1
    if bool(ok.item()):
        return
    nan = int(torch.isnan(g).sum().item())
    raise RuntimeError(
        f"{BANNER} DRIFT: the MoE combine backward gradient is no longer exactly "
        f"bf16-representable at admitted call {_S['served']} (shape={tuple(g.shape)} "
        f"dtype={g.dtype} nan_elements={nan}). The bf16 backward wire is lossless ONLY because the "
        f"VJP of the bf16 round in combine_postprocess is an exact upcast and everything between "
        f"there and this collective is a pure permutation. Something now does arithmetic on that "
        f"path. REFUSING TO FALL BACK: a rank-local fallback would put fp32 on the wire while the "
        f"other EP ranks put bf16 there, which is a mismatched collective, not a safe state. "
        f"Set {_ENV}=0 to run."
    )


class CombineAllToAll(torch.autograd.Function):
    """Forward: megatron's ``_AllToAll``, unchanged. Backward: the same exchange on a bf16 wire.

    The asymmetry IS the design. The direction is decided by which method is running -- a purely
    structural fact, identical on every rank -- never by anything read out of a tensor.
    """

    @staticmethod
    def forward(ctx, group, input_, output_split_sizes, input_split_sizes, use_nccl_stream=False):
        """BYTE-IDENTICAL to ``megatron.core.tensor_parallel.mappings._AllToAll.forward``.

        Nothing below narrows anything. The census call observes the payload and returns; it is the
        only statement here that the flag-off path does not also execute.
        """
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.use_nccl_stream = use_nccl_stream
        ctx.wire_dtype = input_.dtype
        if _S["mode"] == MODE_PROBE and input_.dtype in (torch.float32, torch.bfloat16):
            _census_note(input_, forward=True)
        return _raw_a2a(group, input_, output_split_sizes, input_split_sizes, use_nccl_stream)

    @staticmethod
    def backward(ctx, *grad_output):
        return (None, _combine_backward(ctx, grad_output[0]), None, None, None)


def _combine_backward(ctx, g: torch.Tensor) -> torch.Tensor:
    """The reverse exchange -- narrowed to bf16 when admitted, verbatim megatron when not."""
    _S["bwd_calls"] += 1
    mode = _S["mode"]

    if mode == MODE_PROBE and g.dtype in (torch.float32, torch.bfloat16):
        _census_note(g, forward=False)

    # Admission is STRUCTURAL ONLY: payload dtype and group size, both identical on every rank of the EP
    # group by construction, so every rank admits or declines the same call. Nothing here reads a tensor
    # value -- a value-dependent admission is exactly how an all-to-all desynchronizes.
    if mode != MODE_ON:
        return _decline(ctx, g, "" if mode == MODE_OFF else "probe mode: census only, wires unchanged")
    if g.dtype != torch.float32:
        return _decline(ctx, g, f"grad dtype {g.dtype} (only fp32 payloads are narrowed)")
    if _world(ctx.group) <= 1:
        return _decline(ctx, g, "EP group world<=1 (megatron bypasses the collective)")

    if _S["agreed"] is None:
        # Call 1: the drift probe AND the group-wide vote, in one collective.
        _S["agreed"] = _agree_lossless(g, ctx.group)
        _S["probes"] += 1
        print(
            f"{BANNER} first-call vote: the MoE combine backward gradient is "
            f"{'EXACTLY bf16-representable on every EP rank -- bf16 wire ENGAGED' if _S['agreed'] else 'NOT bf16-representable on at least one EP rank -- bf16 wire PERMANENTLY DISABLED on every rank (MIN-reduced, so this verdict is unanimous by construction)'} "
            f"(shape={tuple(g.shape)} world={_world(ctx.group)})",
            flush=True,
        )
    if not _S["agreed"]:
        return _decline(ctx, g, "first-call EP-group vote said the gradient is not bf16-representable")

    if _S["served"] > 0 and _S["served"] % PROBE_EVERY == 0:
        _drift_probe_or_raise(g, ctx.group)

    gb = g.contiguous().to(torch.bfloat16)
    ob = _raw_a2a(ctx.group, gb, ctx.input_split_sizes, ctx.output_split_sizes, ctx.use_nccl_stream)
    out = ob.to(ctx.wire_dtype)
    _S["served"] += 1
    _S["bytes_saved"] += (gb.numel() + ob.numel()) * 2
    _report()
    return out


def _decline(ctx, g: torch.Tensor, reason: str) -> torch.Tensor:
    if reason:
        _S["decline_reason"] = reason
    _S["declined"] += 1
    out = _raw_a2a(ctx.group, g, ctx.input_split_sizes, ctx.output_split_sizes, ctx.use_nccl_stream)
    _report()
    return out


def _report() -> None:
    """One line at 1/2/4/8/... backward calls, so an INERT install is visible without a profiler."""
    n = _S["bwd_calls"]
    if n < 1 or (n & (n - 1)) != 0 or n == _S["reported"]:
        return
    _S["reported"] = n
    print(
        f"{BANNER} pid={os.getpid()} mode={_S['mode']} served={_S['served']} "
        f"declined={_S['declined']} bytes_saved={_S['bytes_saved'] / 1e6:.1f} MB "
        f"probes={_S['probes']} agreed={_S['agreed']}"
        + (f" last_decline={_S['decline_reason']}" if _S["decline_reason"] else ""),
        flush=True,
    )


def wire_stats() -> dict:
    """A copy of the counters -- for tests, the nightly battery and the phase report."""
    return dict(_S)


def _reset_for_test() -> None:
    _S.update(
        {
            "bwd_calls": 0,
            "served": 0,
            "declined": 0,
            "bytes_saved": 0,
            "agreed": None,
            "decline_reason": "",
            "probes": 0,
            "reported": 0,
        }
    )
    _CENSUS.clear()
    global _census_calls, _census_reported
    _census_calls = 0
    _census_reported = 0


def wire_all_to_all(group, input_, output_split_sizes_=None, input_split_sizes=None, use_nccl_stream=False):
    """Signature-compatible stand-in for ``megatron.core.tensor_parallel.all_to_all``."""
    assert group is not None, "group should not be None"
    return CombineAllToAll.apply(group, input_, output_split_sizes_, input_split_sizes, use_nccl_stream)


def install_moe_a2a_bf16_wire() -> bool:
    """Rebind ONE method: ``MoEAlltoAllTokenDispatcher.token_combine``. Idempotent.

    Prints a banner whether the flag is on or off, because "the lever was never installed" and "the
    lever was installed and declined" are different findings and a silent install cannot tell them
    apart.
    """
    mode = wire_mode()
    _S["mode"] = mode
    if mode == MODE_OFF:
        print(
            f"{BANNER} OFF ({_ENV}=0): the MoE combine all-to-all carries fp32 in BOTH directions "
            f"(~45 MB/rank/call at the 35B trainer shape). =1 ships the BACKWARD leg as bf16 "
            f"(bitwise lossless: the backward payload is an exact upcast of bf16 values permuted by "
            f"a scatter that does no arithmetic); =probe censuses both directions and changes nothing.",
            flush=True,
        )
        return False
    try:
        from megatron.core.transformer.moe import token_dispatcher as _td
        from megatron.core.transformer.moe.token_dispatcher import (
            MoEAlltoAllTokenDispatcher as A,
        )
    except Exception as e:  # pragma: no cover - megatron layout drift
        print(f"{BANNER} unavailable ({type(e).__name__}: {e}) -- combine all-to-all left as-is", flush=True)
        return False
    if getattr(A, "_isoexec_a2a_wire_patched", False):
        return True

    _orig_token_combine = A.token_combine

    def token_combine(self, *args, **kwargs):
        """Point ``token_dispatcher.all_to_all`` at our wrapper for the duration of THIS call.

        The same intervention as ``moe_batch_invariant._make_sorted_topk_routing``'s ``torch.topk`` swap:
        it does not fork megatron's body, and the forward is single-threaded. A PERMANENT rebind of the
        module-level name would also capture the DISPATCH collectives, whose payload is not covered by
        the losslessness argument in this module's docstring.
        """
        real = _td.all_to_all
        _td.all_to_all = wire_all_to_all
        try:
            return _orig_token_combine(self, *args, **kwargs)
        finally:
            _td.all_to_all = real

    A.token_combine = token_combine
    A._isoexec_a2a_wire_patched = True
    _S["installed"] = True
    if mode == MODE_PROBE:
        print(
            f"{BANNER} CENSUS MODE ({_ENV}=probe): every combine all-to-all payload is tested for "
            f"bf16 representability in BOTH directions and NO wire changes. Expect forward 0/N "
            f"(the fp32 leaf-tree root is a sum of 8 bf16 leaves) and backward N/N (an exact upcast "
            f"permuted by a scatter with no arithmetic). That split is the proof; a forward N/N or a "
            f"backward 0/N REFUTES the design and the =1 arm must not be run.",
            flush=True,
        )
    else:
        print(
            f"{BANNER} INSTALLED ({_ENV}=1) on MoEAlltoAllTokenDispatcher.token_combine: FORWARD "
            f"unchanged (fp32 -- narrowing it is the canary-v10 defect), BACKWARD exchanged as bf16 "
            f"and widened back to fp32. Bitwise by construction; verified by an EP-group-wide vote on "
            f"call 1 and an int32 bit compare every {PROBE_EVERY} admitted calls, which RAISES rather "
            f"than falls back (a unilateral fallback is a mismatched collective, not a safe state).",
            flush=True,
        )
    return True
