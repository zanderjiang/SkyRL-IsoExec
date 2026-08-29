# Utils ported from NeMo-Aligner by way of NeMo-RL
# https://github.com/NVIDIA-NeMo/RL/blob/9301d36cbf847212430b84a27cfe6990f773b7cf/nemo_rl/distributed/model_utils.py#L4
# The original copyright is reproduced below:

#  Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import threading
from typing import Any, Optional

import megatron.core.parallel_state as mpu
import torch
import torch.distributed as dist

# =============================================================================================
# ISOEXEC -- the log-softmax BACKWARD owes the generator nothing
# =============================================================================================
# WHAT THIS CHANGES, IN ONE SENTENCE. ``ChunkedDistributedLogprob.backward`` recomputes the
# log-softmax that its forward chose not to save, and under ``SKYRL_ISOEXEC=1`` that recompute takes
# the GATHER branch of ``_compute_distributed_log_softmax`` below: an ``all_gather`` of the whole
# vocabulary (``world`` x ``[B, chunk, V/TP]`` fp32), a ``cat`` into one ``[B, chunk, V]`` fp32
# buffer -- 1.02 GB at the live 35B shape (1024 rows x V=248,320) -- and then amax / sub_ / exp_ /
# sum / log over it. This lever sends the BACKWARD, and only the backward, down the standard
# megatron/NeMo formulation instead: ``all_reduce(MAX)`` then ``all_reduce(SUM)`` on
# ``[B, chunk, 1]``. Per admitted chunk that removes ``(world-1) x 254 MB`` of wire traffic
# (762 MB at TP=4) and the entire 1.02 GB materialization.
#
# WHY THE FORWARD MUST GATHER AND THE BACKWARD MUST NOT -- the whole argument, and the reason this
# block exists at all:
#
#   * FORWARD. Its output is the trainer's scored logprob, and the IsoExec gate compares that
#     against the generator's, elementwise (``policy/rollout_train_logprobs_abs_diff_mean``, emitted
#     from ``skyrl/train/trainer.py``; the train-side signature lives in
#     ``isoexec/core/signatures.py``). The generator computes lse as ONE aten sum over the GATHERED
#     full vocabulary row; the distributed formulation computes ``log(allreduce_sum(per-shard aten
#     sums))``, whose fp32 summation order differs -- measured, ~60% of rows differ in the last ulp.
#     The gather branch exists precisely to reproduce the generator's summation order. IT IS THE
#     BITWISE CONTRACT AND NOTHING HERE TOUCHES IT.
#
#   * BACKWARD. Its output is ``grad_input``. It reaches the optimizer and nothing else: no gate
#     reads it, no signature hashes it, and the generator never computes a counterpart for it to be
#     bitwise-equal TO. There is therefore no summation order to reproduce. Both formulations are
#     max-shifted (numerically stable) log-softmaxes of the same logits and differ only by fp32
#     reassociation -- ~1e-7 relative, the same magnitude as the gradient's own accumulation noise
#     and far below the bf16 grad-buffer round every value takes a few ops later.
#
# THE CLAIM IS ASYMMETRIC AND THE TESTS ASSERT IT ASYMMETRICALLY. The forward is asserted
# BIT-IDENTICAL with this flag on and off (int32 bit compare, not ``allclose``); the backward is
# asserted only ``allclose``. That is not a weaker test of the same thing, it is the correct test of
# a different thing -- see ``tests/backends/skyrl_train/distributed/test_logprob_bwd_reduced_comm.py``.
# Anyone "tightening" the backward assertion to ``equal`` has misread which of the two owes a
# contract, and will be reverting a 30-40 s/step win to satisfy a contract that does not exist.
#
# COLLECTIVE SAFETY -- and why the usual IsoExec "fall back on a failed self-check" convention is
# INVERTED here, for a different reason than ``ops/moe/moe_a2a_wire.py`` inverts it.
# This is not a collective-DTYPE change, so the a2a module's specific hazard (send/recv byte counts
# disagreeing) does not apply. But it IS a collective-STRUCTURE change: the gather branch issues one
# ``all_gather``, the reduced-comm branch issues two ``all_reduce``s. A rank that took one branch
# while its peers took the other would be in a mismatched collective just the same. Two consequences:
#
#   1. ADMISSION IS PURELY STRUCTURAL. It reads the env var (forwarded to every actor by the TRAIN
#      channel), whether the executing method is ``backward``, and the group size. Nothing reads a
#      tensor VALUE. Every TP rank therefore takes the same branch by construction -- TP ranks are
#      data-replicated and run the identical chunk loop, so even the admitted-call COUNTER advances
#      in lockstep, which is what lets the sampled probe below fire on all ranks at once.
#   2. THE PROBE RAISES; IT DOES NOT FALL BACK. A rank-local fallback here is the failure mode, not
#      the safe state. The verdict is MIN-reduced over the group so that a failure on any rank
#      raises on all of them, rather than one rank dying and the rest deadlocking on the next
#      collective.
#
# The probe is cheap on purpose: it gathers a fixed 8-row SLICE of the chunk (~2 MB, not 1.02 GB) on
# call 1 and every 512th admitted call, so re-proving agreement on live operands can never be the
# thing that OOMs the step this lever exists to make smaller.
#: The gather branch's WIRE lever lives in its own module (isoexec/ops/collectives) so this ported
#: upstream file carries one call, not a second census. Imported lazily-but-once at module import so
#: a missing isoexec tree degrades to today's inlined behaviour instead of breaking the import.
try:
    from skyrl.backends.skyrl_train.isoexec.ops.collectives.logprob_gather_wire import (
        gather_full_vocab as _ix_gather_full_vocab,
    )
except Exception as _ix_wire_import_error:  # noqa: BLE001 -- refusal still joins the TP vote

    _IX_WIRE_IMPORT_REFUSAL_PRINTED = False
    _IX_WIRE_IMPORT_REFUSAL_GROUPS = {}
    _IX_WIRE_IMPORT_ERROR = repr(_ix_wire_import_error)

    def _ix_gather_full_vocab(shard, *, group, world, src_dtype=None):  # type: ignore[misc]
        """Incumbent gather plus the candidate's first structural vote.

        A rank-local import failure cannot simply fall back: peers with a working module would
        otherwise enter the candidate's agreement collectives and bf16 gather.  Eligible=0 in the
        same fixed-size MIN/MAX vote makes every peer choose fp32, after which the verdict is
        latched and the hot path is the incumbent again.
        """
        global _IX_WIRE_IMPORT_REFUSAL_PRINTED
        if not _IX_WIRE_IMPORT_REFUSAL_PRINTED:
            _IX_WIRE_IMPORT_REFUSAL_PRINTED = True
            print(
                "[ISOEXEC-LOGPROB-GATHER-WIRE] REFUSED: module import failed; this rank will vote "
                f"eligible=0 before the first gather ({_IX_WIRE_IMPORT_ERROR})",
                flush=True,
            )
        if os.environ.get("SKYRL_ISOEXEC_LOGPROB_GATHER_BF16_WIRE") is None:
            fallback_shard = shard if shard.dtype is torch.float32 else shard.to(torch.float32)
            gathered = [torch.empty_like(fallback_shard) for _ in range(world)]
            torch.distributed.all_gather(gathered, fallback_shard, group=group)
            full = torch.cat(gathered, dim=-1)
            del gathered, fallback_shard
            return full
        if (
            id(group) not in _IX_WIRE_IMPORT_REFUSAL_GROUPS
            and world > 1
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            dtype_code = {None: 0, torch.bfloat16: 1, torch.float16: 2, torch.float32: 3}
            contract = (
                0,
                dtype_code.get(src_dtype, -1),
                dtype_code.get(shard.dtype, -1),
                int(world),
            )
            low = torch.tensor(contract, dtype=torch.int64, device=shard.device)
            high = low.clone()
            torch.distributed.all_reduce(low, op=torch.distributed.ReduceOp.MIN, group=group)
            torch.distributed.all_reduce(high, op=torch.distributed.ReduceOp.MAX, group=group)
            # Retain the group object so a later process-group lifecycle cannot reuse its Python id
            # and silently skip the mandatory refusal vote.
            _IX_WIRE_IMPORT_REFUSAL_GROUPS[id(group)] = group
        fallback_shard = shard if shard.dtype is torch.float32 else shard.to(torch.float32)
        gathered = [torch.empty_like(fallback_shard) for _ in range(world)]
        torch.distributed.all_gather(gathered, fallback_shard, group=group)
        full = torch.cat(gathered, dim=-1)
        del gathered, fallback_shard
        return full


try:
    from skyrl.backends.skyrl_train.isoexec.ops.logprobs.exact_sampled import (
        maybe_exact_sampled_logprobs as _ix_maybe_exact_sampled_logprobs,
    )
except ImportError as _ix_exact_sampled_import_error:
    _IX_EXACT_SAMPLED_IMPORT_REFUSAL_PRINTED = False
    _IX_EXACT_SAMPLED_IMPORT_CACHE = {}
    _IX_EXACT_SAMPLED_IMPORT_ERROR = repr(_ix_exact_sampled_import_error)

    def _ix_maybe_exact_sampled_logprobs(logits, target, start, end, group, src_dtype, reference):  # type: ignore[misc]
        global _IX_EXACT_SAMPLED_IMPORT_REFUSAL_PRINTED
        contract = (
            os.environ.get("SKYRL_ISOEXEC_EXACT_SAMPLED_LOGPROBS", "0"),
            logits.device,
            logits.dtype,
            src_dtype,
            target.dtype,
            start,
            end,
        )
        signature = (
            tuple(logits.shape),
            tuple(logits.stride()),
            tuple(target.shape),
        )
        group_key = id(group)
        cached = _IX_EXACT_SAMPLED_IMPORT_CACHE.get(group_key)
        if cached is None:
            # Match the available implementation's one-time pre-branch vote. If only one rank
            # failed import, MIN makes every peer decline before collective sequences can diverge.
            vote = torch.zeros(1, dtype=torch.int32, device=logits.device)
            torch.distributed.all_reduce(vote, op=torch.distributed.ReduceOp.MIN, group=group)
            _IX_EXACT_SAMPLED_IMPORT_CACHE[group_key] = {"contract": contract, "signatures": {signature}}
        elif cached["contract"] != contract:
            raise RuntimeError(
                "[ISOEXEC-EXACT-SAMPLED-LOGPROBS] STRUCTURAL DRIFT after import refusal: "
                f"first={cached['contract']!r} now={contract!r}"
            )
        elif signature not in cached["signatures"]:
            vote = torch.zeros(1, dtype=torch.int32, device=logits.device)
            torch.distributed.all_reduce(vote, op=torch.distributed.ReduceOp.MIN, group=group)
            cached["signatures"].add(signature)
        if (
            os.environ.get("SKYRL_ISOEXEC_EXACT_SAMPLED_LOGPROBS") == "1"
            and not _IX_EXACT_SAMPLED_IMPORT_REFUSAL_PRINTED
        ):
            _IX_EXACT_SAMPLED_IMPORT_REFUSAL_PRINTED = True
            print(
                "[ISOEXEC-EXACT-SAMPLED-LOGPROBS] REFUSED: optional implementation import failed; "
                f"all TP ranks retain the incumbent. error={_IX_EXACT_SAMPLED_IMPORT_ERROR}",
                flush=True,
            )
        refusal = None
        return refusal


#: The row-count- and TP-invariant leaf-tree sampled logprob -- the composed default at every
#: logprob site, no flag. Same lazy-import-with-refusal-shim shape as the two levers above, so a
#: missing isoexec tree degrades to today's path instead of breaking this module's import. That
#: degradation is now VISIBLE rather than expected: the contract names rowinv, so a rank that
#: imports the shim serves the incumbent and its census stays served=0, which
#: ``enforce.rowinv_engagement_boundary`` refuses at the next weight sync. Unlike the
#: exact-sampled shim, this one mirrors no pre-branch vote: rowinv.py owns its own admission
#: protocol and this file cannot reproduce a protocol it did not import. What the shim CAN say is
#: narrower and still true -- with the module absent, every call declines locally before any
#: rowinv collective is entered. An import failure is rank-symmetric in practice (every TP rank
#: runs the same tree and env), and a rank-ASYMMETRIC failure is outside this shim's protection;
#: it surfaces as the peers' admission collectives timing out, not as a silent split.
#:
#: The consumer is ``_ix_logprobs_apply`` -- the ONE dispatch point, sitting in the two
#: ``from_parallel_logits_to_logprobs*`` wrappers OUTSIDE any autograd.Function, so rowinv's own
#: Function can carry its backward out and BOTH the scoring forward (inference_only=True) and the
#: grad-bearing training forward (inference_only=False) are served by the same function.
try:
    from skyrl.backends.skyrl_train.isoexec.ops.logprobs.rowinv import (
        rowinv_sampled_logprobs as _ix_rowinv_sampled_logprobs,
    )
    from skyrl.backends.skyrl_train.isoexec.ops.logprobs.rowinv import (
        stats as _ix_rowinv_stats,
    )

    def _ix_rowinv_available() -> bool:
        return True

except ImportError as _ix_rowinv_import_error:
    _IX_ROWINV_IMPORT_REFUSAL_PRINTED = False
    _IX_ROWINV_IMPORT_ERROR = repr(_ix_rowinv_import_error)

    def _ix_rowinv_available() -> bool:  # type: ignore[misc]
        global _IX_ROWINV_IMPORT_REFUSAL_PRINTED
        if not _IX_ROWINV_IMPORT_REFUSAL_PRINTED:
            _IX_ROWINV_IMPORT_REFUSAL_PRINTED = True
            print(
                "[ISOEXEC-ROWINV-LOGPROB] REFUSED: module import failed; this rank keeps the "
                f"incumbent logprob path. error={_IX_ROWINV_IMPORT_ERROR}",
                flush=True,
            )
        return False

    def _ix_rowinv_sampled_logprobs(*args, **kwargs):  # type: ignore[misc]  # pragma: no cover
        return None  # unreachable while _ix_rowinv_available() is False; kept for signature parity

    def _ix_rowinv_stats() -> dict:  # type: ignore[misc]  # pragma: no cover
        return {}


_IX_BWD_ENV = "SKYRL_ISOEXEC_LOGPROB_BWD_REDUCED_COMM"
_IX_BWD_BANNER = "[ISOEXEC-LOGPROB-BWD-REDUCED-COMM]"

#: call 1, then every N-th admitted chunk, re-proves agreement with the gather formulation.
_IX_BWD_PROBE_EVERY = 512
#: rows of the chunk the probe gathers. Fixed (structural), tiny, and sufficient: a formulation bug
#: moves every row, not row 9.
_IX_BWD_PROBE_ROWS = 8
#: tolerance on the log-softmax output. NOT a bitwise claim -- the backward does not owe one.
#: SIZED TO SEPARATE TWO REGIMES, not to be tight.
#:
#: THE LEGAL DIFFERENCE IS RELATIVE, SO THE TOLERANCE MUST BE TOO. The two paths take the SAME
#: global max (max is exact, so ``x - logits_max`` is bitwise identical either way) and differ ONLY
#: in lse. Write ``t = x - max``; both branches then return ``fl(t - lse)`` for their own lse, so the
#: whole difference is ``|fl(t - lse_gather) - fl(t - lse_reduced)|``, which is a difference of two
#: fp32 grid points and is therefore ZERO OR A MULTIPLE OF ONE ULP OF THE OUTPUT. Measured on CPU
#: over 300 (V, TP, scale, seed) configurations: max k = 1.000 ulp, i.e. the difference is at most
#: one ulp of ``max|out|`` and never more. That is a SCALE-FREE RELATIVE bound (~1.2e-7 relative),
#: not an absolute one.
#:
#: A FLAT ABSOLUTE ATOL IS THEREFORE A LATENT RUN-KILLER. One ulp crosses 1e-3 at |out| = 8388.6
#: (ulp jumps to 1.95e-3 at the 2^14 binade), so past that magnitude a perfectly LEGAL reassociation
#: exceeds a flat 1e-3 and this probe -- which refuses to fall back, by design -- kills the run.
#: |out| is ``|x - max - lse|``, i.e. the logit RANGE, and ``logits.div_(temperature)`` at a small
#: temperature or a diverging step reaches it. Not hypothetical: a CPU seed scan found a natural,
#: unperturbed, entirely legal case (V=8192, TP=4, logits spanning 24000) whose gap is exactly one
#: ulp = 1.953e-3 and which made the flat-atol probe raise -- see
#: ``test_probe_does_not_raise_on_a_legal_large_magnitude_difference``.
#:
#: THE SHAPE OF THE FIX. ``tol(|ref|) = _IX_BWD_PROBE_ATOL * max(1, |ref| / _IX_BWD_PROBE_REL_FROM)``:
#: flat below ``REL_FROM`` and proportional above it. Concretely:
#:   * |out| <= 512 -- every production shape measured (|out|max is 41 at scale-4 logits, 204 at
#:     scale-20, 425 at scale-40, all at V=248,320) -- the tolerance is UNCHANGED at 1e-3, which is
#:     ~65x above the observed noise and ~1000x below an O(1) formulation bug. This fix does not
#:     loosen anything in the regime the lever actually runs in.
#:   * |out| > 512 the tolerance is 16.4-32.8 ulp of |out|: 16x above the measured legal maximum
#:     (k=1.000), 8x above the k<=2 bound the adversarial suite asserts, and still 21x below an O(1)
#:     bug at the 24000-magnitude case above.
#: A real formulation bug scales with the operand range too (taking a per-shard max instead of
#: ``all_reduce(MAX)`` is wrong by the cross-shard max gap), so relative is the right shape for the
#: detector as well as for the noise. The one regime this gives up is an O(1) error at |out| > ~5e5,
#: where one ulp is itself O(1) and fp32 has no signal left to detect it with.
#:
#: KEEP THE RELATIVE TERM KEYED OFF ``_IX_BWD_PROBE_ATOL`` (a multiplier, never an added term):
#: setting ``_IX_BWD_PROBE_ATOL = 0`` must still make every difference a violation, which is how
#: ``test_drift_probe_actually_raises`` proves the probe is not inert.
_IX_BWD_PROBE_ATOL = 1e-3
#: |ref| below which the tolerance stays flat at ``_IX_BWD_PROBE_ATOL``. Above it the tolerance grows
#: in proportion, because the legal difference does.
_IX_BWD_PROBE_REL_FROM = 512.0

_IX_BWD = {
    "fwd_gather": 0,  # forward chunks that took the gather branch (must keep rising -- untouched)
    "bwd_calls": 0,  # backward chunks that reached the IsoExec branch at all
    "served": 0,  # backward chunks actually run on the reduced-comm formulation
    "declined": 0,  # backward chunks left on the gather branch
    "decline_reason": "",
    "wire_bytes_saved": 0,  # all_gather payload not sent
    "buffer_bytes_saved": 0,  # the [B, chunk, V] fp32 cat never materialized
    "probes": 0,
    "max_probe_diff": 0.0,
    "reported": 0,
    "bannered": False,
}

#: Re-entrancy guard for ``_ix_bwd_probe`` (which calls this module's own entry point twice on a
#: slice). THREAD-LOCAL, not a key in ``_IX_BWD``: production runs one rank per process so a plain
#: flag would do, but autograd is free to execute a Function's backward on a thread other than the
#: caller's, and the CPU test harness simulates the TP group with threads inside ONE process. A
#: shared flag there lets rank 0's probe suppress rank 1's, which desynchronizes the collectives and
#: hangs -- the exact failure this lever's structural admission exists to prevent, reproduced inside
#: the test. Everything else in ``_IX_BWD`` is a counter whose worst case is a lost increment.
_IX_BWD_TLS = threading.local()


def _ix_bwd_probing() -> bool:
    return getattr(_IX_BWD_TLS, "probing", False)


def _ix_bwd_probe_tol(ref_mag: float) -> float:
    """Drift tolerance at a reference output magnitude. See ``_IX_BWD_PROBE_ATOL`` for the derivation.

    Flat at ``_IX_BWD_PROBE_ATOL`` up to ``_IX_BWD_PROBE_REL_FROM``, proportional to ``ref_mag``
    above it. A non-finite magnitude falls back to the flat term rather than to ``inf``: an ``inf``
    tolerance would make the probe unconditionally blind, and a non-finite reference is itself a
    finding the caller must see (the diff against it is inf or nan, and both fail ``diff <= tol``).
    """
    if not math.isfinite(ref_mag):
        return _IX_BWD_PROBE_ATOL
    return _IX_BWD_PROBE_ATOL * max(1.0, abs(ref_mag) / _IX_BWD_PROBE_REL_FROM)


def ix_bwd_reduced_comm_enabled() -> bool:
    """``SKYRL_ISOEXEC_LOGPROB_BWD_REDUCED_COMM``. Default OFF -- this lever fails closed.

    Fail-closed has three independent layers, any one of which keeps today's behaviour:
    this default, the ``for_backward=False`` default on ``_compute_distributed_log_softmax``
    (so every pre-existing caller is unedited and unaffected), and the structural guards in
    ``_ix_bwd_admit``.
    """
    return os.environ.get(_IX_BWD_ENV, "0") == "1"


def _ix_bwd_banner_once(enabled: bool, world: int) -> None:
    """Print ON *and* OFF. "the lever was never reached" and "the lever was reached and declined"
    are different findings, and a silent path cannot tell them apart.

    THE BANNER IS NOT ENGAGEMENT. It fires on the first backward chunk that reaches the IsoExec
    branch, which happens whether or not anything is served. Only ``served>0`` in the census line
    below is evidence that the reduced-comm path ran.
    """
    if _IX_BWD["bannered"]:
        return
    _IX_BWD["bannered"] = True
    if enabled:
        print(
            f"{_IX_BWD_BANNER} ON ({_IX_BWD_ENV}=1, world={world}): the log-softmax BACKWARD "
            f"recompute drops the full-vocab all_gather + cat and uses all_reduce(MAX) then "
            f"all_reduce(SUM) on [B, chunk, 1]. The FORWARD still gathers -- it owes the generator "
            f"a bitwise lse and this changes none of its bytes. Backward moves by ~1e-7 relative "
            f"(fp32 reassociation) and reaches only the optimizer. Agreement is re-proved on live "
            f"operands at call 1 and every {_IX_BWD_PROBE_EVERY} admitted calls, MIN-reduced over "
            f"the TP group, and RAISES rather than falling back (a rank-local fallback would put "
            f"one rank in an all_gather while its peers are in an all_reduce). Judge engagement by "
            f"served>0 below, never by this line.",
            flush=True,
        )
    else:
        print(
            f"{_IX_BWD_BANNER} OFF ({_IX_BWD_ENV}=0, world={world}): the log-softmax BACKWARD "
            f"recompute keeps the forward's gather branch -- an all_gather of {world}x[B, chunk, "
            f"V/TP] fp32 and a [B, chunk, V] fp32 cat (1.02 GB at the live 35B shape) per chunk, "
            f"for an lse the backward has no bitwise contract on. =1 replaces it with "
            f"all_reduce(MAX)+all_reduce(SUM) on [B, chunk, 1]; the forward is untouched either way.",
            flush=True,
        )


def _ix_bwd_report() -> None:
    """One census line at 1/2/4/8/... backward chunks, so an INERT lever is visible without a
    profiler. ``fwd_gather`` keeps climbing on purpose: it is the running proof that the forward is
    still on the gather branch while the backward is not."""
    n = _IX_BWD["bwd_calls"]
    if n < 1 or (n & (n - 1)) != 0 or n == _IX_BWD["reported"]:
        return
    _IX_BWD["reported"] = n
    print(
        f"{_IX_BWD_BANNER} CENSUS pid={os.getpid()} bwd_calls={n} served={_IX_BWD['served']} "
        f"declined={_IX_BWD['declined']} fwd_gather={_IX_BWD['fwd_gather']} "
        f"wire_saved={_IX_BWD['wire_bytes_saved'] / 1e9:.2f} GB "
        f"buffer_saved={_IX_BWD['buffer_bytes_saved'] / 1e9:.2f} GB "
        f"probes={_IX_BWD['probes']} max_probe_diff={_IX_BWD['max_probe_diff']:.3e}"
        + (f" last_decline={_IX_BWD['decline_reason']}" if _IX_BWD["decline_reason"] else ""),
        flush=True,
    )


def _ix_bwd_probe(
    vocab_parallel_logits: torch.Tensor,
    group: torch.distributed.ProcessGroup,
    src_dtype: Optional[torch.dtype] = None,
) -> None:
    """Re-prove on LIVE operands that the two formulations agree, COLLECTIVELY, and stop the world
    on failure.

    Runs both formulations on a fixed ``_IX_BWD_PROBE_ROWS``-row slice by re-entering
    ``_compute_distributed_log_softmax`` itself -- so the reference really is the shipped gather
    branch, not a paraphrase of it that could drift away from it. The re-entrancy guard keeps the
    two probe calls out of the census.

    Read the block comment above before "fixing" this to fall back on failure: the two branches
    issue structurally different collectives, so a rank-local fallback is a mismatched collective,
    not a safe state.

    ``src_dtype`` is forwarded so ``ref`` really is the shipped gather branch under whatever wire
    dtype the run is using. It cannot change the verdict -- the wire lever is bit-neutral by
    construction -- but a reference that silently ran a different code path than production would
    make this probe attest to something nobody ships.
    """
    rows = min(_IX_BWD_PROBE_ROWS, int(vocab_parallel_logits.shape[1]))
    sl = vocab_parallel_logits[:, :rows, :].contiguous()
    _IX_BWD_TLS.probing = True
    try:
        got = _compute_distributed_log_softmax(sl, group=group, for_backward=True, src_dtype=src_dtype)
        ref = _compute_distributed_log_softmax(sl, group=group, for_backward=False, src_dtype=src_dtype)
    finally:
        _IX_BWD_TLS.probing = False
    diff = (ref.float() - got.float()).abs().max()
    # RELATIVE, because the legal difference is (one ulp of |out|, i.e. ~1.2e-7 relative). The
    # magnitude is this rank's own -- a value read, but only of the VERDICT, never of the branch
    # decision, so it cannot split the group; and the MIN-reduce below still makes any rank's
    # failure everybody's.
    tol = _ix_bwd_probe_tol(float(ref.float().abs().max().item()))
    ok = (diff <= tol).to(torch.int32).reshape(1)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        # MIN so a failure on ANY rank raises on ALL of them. A rank-local raise would leave the
        # survivors deadlocked on the next collective instead of dying with this message.
        torch.distributed.all_reduce(ok, op=torch.distributed.ReduceOp.MIN, group=group)
    _IX_BWD["probes"] += 1
    _IX_BWD["max_probe_diff"] = max(_IX_BWD["max_probe_diff"], float(diff.item()))
    if bool(ok.item()):
        return
    raise RuntimeError(
        f"{_IX_BWD_BANNER} DRIFT: the reduced-comm backward log-softmax no longer agrees with the "
        f"gather formulation at served chunk {_IX_BWD['served']} (max |diff| "
        f"{float(diff.item()):.3e} > tol {tol:.3e}, shape={tuple(sl.shape)}). The tolerance is "
        f"{_IX_BWD_PROBE_ATOL:.0e} flat up to |out|={_IX_BWD_PROBE_REL_FROM:.0f} and proportional "
        f"above it, because the legal difference is one ulp of |out|. The two are "
        f"supposed to differ only by fp32 reassociation of the same max-shifted log-softmax; a "
        f"difference this large means one of them is no longer that. REFUSING TO FALL BACK: the two "
        f"branches issue different collectives (all_gather vs all_reduce x2), so a rank-local "
        f"fallback would desynchronize the TP group rather than protect it. Set {_IX_BWD_ENV}=0 to run."
    )


def _ix_bwd_admit(
    vocab_parallel_logits: torch.Tensor,
    group: torch.distributed.ProcessGroup,
    world: int,
    src_dtype: Optional[torch.dtype] = None,
) -> bool:
    """Structural admission for the reduced-comm backward. True = take it.

    Every term is structural (env var, which method is executing, group size). Nothing here reads a
    tensor value, so every TP rank returns the same answer on the same chunk.

    ADMISSION IS NOT ENGAGEMENT, and this function deliberately does NOT touch ``served``. It
    returns a VERDICT; whether the reduced-comm formulation then runs is the caller's business, and
    the caller is the only place that knows. Counting here would make ``served`` a count of
    admissions -- and a lever whose final ``return True`` had been flipped to ``return False`` would
    keep reporting engagement it was not delivering, which is exactly the mutation
    ``test_served_counts_executions_not_admissions`` exists to kill. ``_ix_bwd_record_served``
    below is called from the far side of the two all_reduces instead.
    """
    if _ix_bwd_probing():  # re-entered by _ix_bwd_probe; do not census, do not recurse
        return True
    _IX_BWD["bwd_calls"] += 1
    enabled = ix_bwd_reduced_comm_enabled()
    _ix_bwd_banner_once(enabled, world)
    if not enabled:
        _IX_BWD["declined"] += 1
        _IX_BWD["decline_reason"] = f"{_IX_BWD_ENV}=0 (default): backward stays on the gather branch"
        _ix_bwd_report()
        return False
    served = _IX_BWD["served"]
    if served == 0 or served % _IX_BWD_PROBE_EVERY == 0:
        _ix_bwd_probe(vocab_parallel_logits, group, src_dtype)
    return True


def _ix_bwd_record_served(shard_numel: int, shard_element_size: int, world: int) -> None:
    """Census for a chunk the reduced-comm formulation ACTUALLY RAN.

    Called from ``_compute_distributed_log_softmax`` after both all_reduces have executed, never
    from ``_ix_bwd_admit``. ``served`` is advertised as the only engagement evidence for this lever
    (the banner fires whether or not anything is served), so it has to count executions or the one
    acceptance signal the live arm is judged by is unearned.

    The byte figures are passed in rather than re-read off the tensor because by this point the
    caller has rebound its local name to the max-shifted copy; the numbers must describe the SHARD
    that would have been gathered.
    """
    if _ix_bwd_probing():  # the probe runs both formulations on a slice; neither is a served chunk
        return
    shard_bytes = shard_numel * shard_element_size
    _IX_BWD["served"] += 1
    _IX_BWD["wire_bytes_saved"] += (world - 1) * shard_bytes
    _IX_BWD["buffer_bytes_saved"] += world * shard_bytes
    _ix_bwd_report()


def ix_bwd_stats() -> dict:
    """A copy of the census counters -- for tests, the nightly battery and the phase report.

    ``served`` is the ONLY engagement evidence. A banner without it means the lever was reached and
    declined.
    """
    return dict(_IX_BWD)


def _ix_bwd_reset_for_test() -> None:
    _IX_BWD.update(
        {
            "fwd_gather": 0,
            "bwd_calls": 0,
            "served": 0,
            "declined": 0,
            "decline_reason": "",
            "wire_bytes_saved": 0,
            "buffer_bytes_saved": 0,
            "probes": 0,
            "max_probe_diff": 0.0,
            "reported": 0,
            "bannered": False,
        }
    )
    _IX_BWD_TLS.probing = False


@torch.no_grad()
def _compute_distributed_log_softmax(
    vocab_parallel_logits: torch.Tensor,
    group: torch.distributed.ProcessGroup,
    for_backward: bool = False,
    src_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Compute a stable distributed log softmax across tensor parallel workers.

    Taken from: https://github.com/NVIDIA/NeMo-Aligner/blob/9faab404f21994a7eb1d6ed5890b76152b941636/nemo_aligner/utils/distributed.py#L265

    Args:
        vocab_parallel_logits (torch.Tensor): Logits tensor with shape [batch_size, seq_length, vocab_size//TP]
            where TP is the tensor parallel size.
        group (torch.distributed.ProcessGroup): Process group for the all-reduce operations.
        for_backward (bool): True only when this is the BACKWARD's recompute, whose output reaches
            the optimizer and nothing else. Under ``SKYRL_ISOEXEC_LOGPROB_BWD_REDUCED_COMM=1`` it
            takes the standard all_reduce formulation instead of the IsoExec gather branch. Defaults
            to False so every pre-existing caller keeps today's exact behaviour with no edit -- see
            the ISOEXEC block comment at the top of this module for why the asymmetry is legitimate.
        src_dtype (torch.dtype): the dtype ``vocab_parallel_logits`` had BEFORE the caller widened it
            to fp32, or None when the caller cannot make that statement. Read ONLY by
            ``SKYRL_ISOEXEC_LOGPROB_GATHER_BF16_WIRE`` (default OFF) to decide whether the gather's
            wire may be narrowed back to it -- a decision that is bit-neutral precisely because the
            widening was exact. None is the fail-closed default: no declaration, no narrowing.

    Returns:
        torch.Tensor: Log softmax output with the same shape as input, but values represent
            log probabilities normalized across the full vocabulary dimension.
    """
    # SkyRL-IsoExec: at TP>1 the distributed reduction below computes lse as
    # log(allreduce_sum(per-shard aten sums)), whose fp32 summation order differs from the
    # generator's single-row aten sum over the GATHERED full vocab (gptmodel_vllm sets
    # parallel_output=False; the sampler patch computes x - amax(x) - log(sum(exp(x))) on the
    # full row). Measured: ~60% of rows differ in the last ulp (|lse diff| up to ~3e-7) -- small
    # but not bitwise. Under IsoExec, gather the shards (pure data movement, bitwise) and use the
    # generator's exact single-row formula; the local shard's logprobs are then elementwise
    # identical to the corresponding slice of the generator's full-row logprobs.
    ix_bwd_census: Optional[tuple[int, int, int]] = None
    if os.environ.get("SKYRL_ISOEXEC") == "1" and torch.distributed.get_world_size(group=group) > 1:
        world = torch.distributed.get_world_size(group=group)
        # The BACKWARD has no bitwise contract to this branch's summation order (see the ISOEXEC
        # block comment at the top of this module). When the lever admits it, fall through to the
        # standard all_reduce(MAX)/all_reduce(SUM) formulation below and skip the gather entirely.
        # Admission is structural, so every TP rank falls through together or none does.
        if not (for_backward and _ix_bwd_admit(vocab_parallel_logits, group, world, src_dtype)):
            if not for_backward and not _ix_bwd_probing():
                _IX_BWD["fwd_gather"] += 1
            # -- FORWARD BITWISE CONTRACT: the statements below are untouched by this lever. --
            shard = vocab_parallel_logits.contiguous()
            # The CONTIGUOUS [..., V] layout is load-bearing: aten's sum reduces in a shape-dependent
            # order, so only this layout reproduces the generator's single-row lse bitwise. Do the
            # rest IN PLACE on `full` (never on vocab_parallel_logits, which autograd saved): the
            # naive version held ~4 full-vocab fp32 copies and OOMed at micro_batch>1 (32 GiB at
            # b=4, s=1536, V=248320). Peak is now 2 copies, and the caller's `chunk_size` bounds
            # even that. Chunking the SEQ dim is exact -- each token's softmax is its own row.
            #
            # SKYRL_ISOEXEC_LOGPROB_GATHER_BF16_WIRE (default OFF, i.e. `world` empty_like + one
            # all_gather + one cat, exactly as before) can ship this shard as the low-precision dtype
            # it was widened FROM and widen it back on arrival. `full` is bit-identical either way --
            # narrow->exchange->widen is the identity on a widened bf16, not an approximation of it.
            # `src_dtype=None` (a caller that cannot declare the pre-widening dtype) keeps fp32.
            full = _ix_gather_full_vocab(shard, group=group, world=world, src_dtype=src_dtype)
            logits_max = torch.amax(full, dim=-1, keepdim=True)
            full.sub_(logits_max).exp_()
            lse = full.sum(-1, keepdim=True).float().log()
            del full
            return (vocab_parallel_logits - logits_max) - lse.to(vocab_parallel_logits.dtype)
        # Admitted. Sizes are captured HERE, while the name still refers to the shard; the census
        # itself is deferred to the far side of the two all_reduces, so `served` counts chunks the
        # reduced-comm formulation actually ran rather than chunks it was cleared to run.
        ix_bwd_census = (vocab_parallel_logits.numel(), vocab_parallel_logits.element_size(), world)

    logits_max = torch.amax(vocab_parallel_logits, dim=-1, keepdim=True)
    torch.distributed.all_reduce(
        logits_max,
        op=torch.distributed.ReduceOp.MAX,
        group=group,
    )

    # Subtract the maximum value.
    vocab_parallel_logits = vocab_parallel_logits - logits_max

    sum_exp_logits = vocab_parallel_logits.exp().sum(-1, keepdim=True).float()

    torch.distributed.all_reduce(
        sum_exp_logits,
        op=torch.distributed.ReduceOp.SUM,
        group=group,
    )

    out = vocab_parallel_logits - sum_exp_logits.log_().to(vocab_parallel_logits.dtype)

    if ix_bwd_census is not None:
        # Both all_reduces are behind us and the result exists: this chunk was SERVED.
        _ix_bwd_record_served(*ix_bwd_census)

    return out


class DistributedLogprob(torch.autograd.Function):
    """Custom autograd function for computing log probabilities in a distributed setting.

    Taken from https://github.com/NVIDIA/NeMo-Aligner/blob/9faab404f21994a7eb1d6ed5890b76152b941636/nemo_aligner/utils/distributed.py#L286
    """

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]  Always ignore torch.autograd.Function.forward's type since it's always more specific than the base class
        ctx: Any,
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        vocab_start_index: int,
        vocab_end_index: int,
        group: torch.distributed.ProcessGroup,
        inference_only: bool = False,
    ) -> torch.Tensor:
        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target - vocab_start_index
        masked_target[target_mask] = 0

        # Captured BEFORE the rebinding below: the gather wire's identity argument is about the dtype
        # this tensor was widened FROM, and after the next line that information is gone.
        src_dtype = vocab_parallel_logits.dtype
        vocab_parallel_logits = vocab_parallel_logits.to(dtype=torch.float32)

        if inference_only:

            def incumbent_sampled_logprobs() -> torch.Tensor:
                incumbent = _compute_distributed_log_softmax(
                    vocab_parallel_logits,
                    group=group,
                    src_dtype=src_dtype,
                )
                incumbent = torch.gather(incumbent, -1, masked_target.unsqueeze(-1)).squeeze(-1)
                incumbent[target_mask] = 0.0
                torch.distributed.all_reduce(incumbent, op=torch.distributed.ReduceOp.SUM, group=group)
                return incumbent

            exact_sampled = _ix_maybe_exact_sampled_logprobs(
                vocab_parallel_logits,
                target,
                vocab_start_index,
                vocab_end_index,
                group,
                src_dtype,
                incumbent_sampled_logprobs,
            )
            if exact_sampled is not None:
                return exact_sampled

        log_probs = _compute_distributed_log_softmax(vocab_parallel_logits, group=group, src_dtype=src_dtype)
        softmax_output = log_probs.exp()

        log_probs = torch.gather(log_probs, -1, masked_target.unsqueeze(-1)).squeeze(-1)
        log_probs[target_mask] = 0.0

        torch.distributed.all_reduce(
            log_probs,
            op=torch.distributed.ReduceOp.SUM,
            group=group,
        )

        if not inference_only:
            # only save for backward when we have inference only=False
            ctx.save_for_backward(softmax_output, target_mask, masked_target)

        return log_probs

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None, None, None]:
        # NOTE (IsoExec, SKYRL_ISOEXEC_LOGPROB_BWD_REDUCED_COMM): unlike ChunkedDistributedLogprob,
        # this backward does NOT recompute the log-softmax -- its forward computed the full
        # [B, S, V/TP] fp32 softmax at :487, SAVED it at :500, and this reads it straight off ctx.
        # There is therefore no _compute_distributed_log_softmax call here to pass for_backward=True
        # to, and no all_gather for that lever to remove. The two Functions trade memory against
        # recompute in opposite directions, which is exactly why from_parallel_logits_to_logprobs
        # picks between them on chunk_size (:728-746, and :837-855 in the packed variant).
        grad_output = grad_outputs[0]
        softmax, target_mask, masked_target = ctx.saved_tensors

        if softmax.ndim == 3:
            B, S, V = softmax.shape

            # skip `torch.nn.functional.one_hot`
            row = torch.arange(B, device=softmax.device).view(-1, 1).expand(-1, S).reshape(-1)
            col = torch.arange(S, device=softmax.device).expand(B, -1).reshape(-1)
            flat_idx = (row * S + col) * V

            flat_chosen = flat_idx.masked_select(~target_mask.reshape(-1)) + masked_target.masked_select(~target_mask)

            # `neg` is zero-copy
            grad_input = softmax.neg()
            grad_input = grad_input.mul_(grad_output.unsqueeze(-1))

            grad_output_selected = grad_output.masked_select(~target_mask)
            grad_input.view(-1).scatter_add_(0, flat_chosen, grad_output_selected)
        else:
            V = softmax.size(-1)
            is_chosen = (~target_mask).unsqueeze(-1) * torch.nn.functional.one_hot(masked_target, num_classes=V)
            grad_input = is_chosen.float().sub_(softmax)
            grad_input.mul_(grad_output.unsqueeze(-1))

        # if you add an argument to the forward method, then you must add a corresponding None here
        return grad_input, None, None, None, None, None, None


class ChunkedDistributedLogprob(torch.autograd.Function):
    """Custom autograd function for computing log probabilities in a distributed setting.

    The log probabilities computation is chunked in the sequence dimension
    to mitigate GPU OOM (especially during backward pass).
    In addition, logits casting from float16 or bfloat16 -> float32 is performed
    inside the chunk loop to avoid materializing a whole float32 logits tensor.

    Adapted from https://github.com/NVIDIA/NeMo-Aligner/blob/9faab404f21994a7eb1d6ed5890b76152b941636/nemo_aligner/utils/distributed.py#L286
    """

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]  Always ignore torch.autograd.Function.forward's type since it's always more specific than the base class
        ctx: Any,
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        vocab_start_index: int,
        vocab_end_index: int,
        chunk_size: int,
        tp_group: torch.distributed.ProcessGroup,
        inference_only: bool = False,
    ) -> torch.Tensor:
        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target - vocab_start_index
        masked_target[target_mask] = 0

        seq_size = int(vocab_parallel_logits.shape[1])
        num_chunks = (seq_size + chunk_size - 1) // chunk_size
        all_log_probs = []

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(seq_size, (chunk_idx + 1) * chunk_size)

            logits = vocab_parallel_logits[:, chunk_start:chunk_end, :]
            logits = logits.to(dtype=torch.float32)

            if inference_only:
                chunk_target = target[:, chunk_start:chunk_end]
                chunk_target_mask = target_mask[:, chunk_start:chunk_end]
                chunk_masked_target = masked_target[:, chunk_start:chunk_end]

                def incumbent_sampled_logprobs(
                    logits=logits,
                    chunk_masked_target=chunk_masked_target,
                    chunk_target_mask=chunk_target_mask,
                ) -> torch.Tensor:
                    incumbent = _compute_distributed_log_softmax(
                        logits,
                        group=tp_group,
                        src_dtype=vocab_parallel_logits.dtype,
                    )
                    incumbent = torch.gather(incumbent, -1, chunk_masked_target.unsqueeze(-1)).squeeze(-1)
                    incumbent[chunk_target_mask] = 0.0
                    torch.distributed.all_reduce(incumbent, op=torch.distributed.ReduceOp.SUM, group=tp_group)
                    return incumbent

                exact_sampled = _ix_maybe_exact_sampled_logprobs(
                    logits,
                    chunk_target,
                    vocab_start_index,
                    vocab_end_index,
                    tp_group,
                    vocab_parallel_logits.dtype,
                    incumbent_sampled_logprobs,
                )
                if exact_sampled is not None:
                    all_log_probs.append(exact_sampled)
                    continue

            log_probs = _compute_distributed_log_softmax(
                logits,
                group=tp_group,
                # `vocab_parallel_logits` is the un-rebound Function argument, so this is the dtype
                # the model produced -- the slice above and the `.to(float32)` are a view and an
                # exact widening, neither of which touches a mantissa bit.
                src_dtype=vocab_parallel_logits.dtype,
            )

            log_probs = torch.gather(log_probs, -1, masked_target[:, chunk_start:chunk_end].unsqueeze(-1)).squeeze(-1)
            log_probs[target_mask[:, chunk_start:chunk_end]] = 0.0

            torch.distributed.all_reduce(
                log_probs,
                op=torch.distributed.ReduceOp.SUM,
                group=tp_group,
            )

            all_log_probs.append(log_probs)

        log_probs = torch.cat(all_log_probs, dim=1)

        if not inference_only:
            # only save for backward when we have inference only=False
            ctx.save_for_backward(vocab_parallel_logits, target_mask, masked_target)
            ctx.chunk_size = chunk_size
            ctx.tp_group = tp_group

        return log_probs

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None, None, None]:
        grad_output = grad_outputs[0]
        vocab_parallel_logits, target_mask, masked_target = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        tp_group = ctx.tp_group

        partition_vocab_size = int(vocab_parallel_logits.shape[-1])
        seq_size = int(vocab_parallel_logits.shape[1])
        num_chunks = (seq_size + chunk_size - 1) // chunk_size

        all_grad_input = []

        batch_size = int(vocab_parallel_logits.shape[0])

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(seq_size, (chunk_idx + 1) * chunk_size)
            chunk_len = chunk_end - chunk_start

            logits = vocab_parallel_logits[:, chunk_start:chunk_end, :]
            logits = logits.to(dtype=torch.float32)

            # for_backward=True: this recompute's result reaches the optimizer and nothing else, so
            # it does not owe the generator the gather branch's summation order. Gated by
            # SKYRL_ISOEXEC_LOGPROB_BWD_REDUCED_COMM (default 0 = today's gather); the forward's call
            # at :583 deliberately does NOT pass it. See the ISOEXEC block comment at the top.
            softmax_output = _compute_distributed_log_softmax(
                logits,
                group=tp_group,
                for_backward=True,
                # ctx-saved raw logits: same un-widened dtype the forward declared.
                src_dtype=vocab_parallel_logits.dtype,
            )
            softmax_output = softmax_output.exp()

            # Memory-efficient scatter-add fast path (ported from DistributedLogprob.backward).
            # Materializing one_hot(masked_target, num_classes=partition_vocab_size) would
            # allocate a [B, chunk_len, partition_vocab_size] int64 tensor (~8x the size of
            # softmax_output in float32), which causes OOM for large vocabularies. Instead,
            # compute -softmax * grad_output in place and add grad_output at the chosen-token
            # positions via scatter_add_.
            chunk_target_mask = target_mask[:, chunk_start:chunk_end]
            chunk_masked_target = masked_target[:, chunk_start:chunk_end]
            chunk_grad_output = grad_output[:, chunk_start:chunk_end]

            row = torch.arange(batch_size, device=softmax_output.device).view(-1, 1).expand(-1, chunk_len).reshape(-1)
            col = torch.arange(chunk_len, device=softmax_output.device).expand(batch_size, -1).reshape(-1)
            # Flat offset to the start of each [b, s, :] row in the chunk's flattened tensor.
            flat_idx = (row * chunk_len + col) * partition_vocab_size

            valid_mask = ~chunk_target_mask
            flat_chosen = flat_idx.masked_select(valid_mask.reshape(-1)) + chunk_masked_target.masked_select(valid_mask)

            # `neg` is zero-copy; the subsequent mul_ writes in place.
            grad_input = softmax_output.neg_()
            grad_input.mul_(chunk_grad_output.unsqueeze(-1))

            grad_output_selected = chunk_grad_output.masked_select(valid_mask)
            grad_input.view(-1).scatter_add_(0, flat_chosen, grad_output_selected)

            all_grad_input.append(grad_input)

        grad_input = torch.cat(all_grad_input, dim=1)

        # if you add an argument to the forward method, then you must add a corresponding None here
        return grad_input, None, None, None, None, None, None


def _ix_try_rowinv_logprobs(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start_index: int,
    vocab_end_index: int,
    group: torch.distributed.ProcessGroup,
    inference_only: bool,
    chunk_size: Optional[int],
) -> Optional[torch.Tensor]:
    """Serve the whole call on rowinv, or return None so the caller keeps the incumbent whole.

    The seq dim is walked in the same chunks the incumbent would use, each chunk handed through
    in its NATIVE dtype: rowinv widens TRANSIENTLY inside its own forward (the exact widen keeps
    the bits identical to pre-widening; the fp32 copy dies at call exit instead of being retained
    for backward), and this dispatch materializes no fp32 copy of the shard at all.
    rowinv is row-count invariant BY CONSTRUCTION -- its leaf boundaries and combine tree are
    functions of G alone, never of rows -- so the chunked and unchunked walks are the same
    function (chunk-walk asserted bitwise in ops/logprobs/tests/test_rowinv_dispatch_cpu.py; the
    scheme's row independence in test_rowinv_leaftree_cpu.py and rowinv_gpu.py).

    Grad note: under the grad path rowinv saves each NATIVE-dtype chunk for its rank-local
    backward -- a view of the live logits when the slice is contiguous, so nothing beyond the
    activation the caller already holds -- plus per-row scalars. This matches the incumbent
    ChunkedDistributedLogprob's save-raw-and-recompute memory order instead of the +100% a
    retained widened fp32 shard would cost at live shapes.

    Fallback discipline: any chunk DECLINE (None) abandons rowinv for the WHOLE call -- one
    function per returned tensor, never a mid-tensor mix. A failed first-call probe inside rowinv
    returns the reference's own chunk (incumbent bits, but graph-free), so the grad path treats a
    graph-free chunk as a decline too, and a tail-chunk probe failure is caught by the census
    latch (``stats()["agreed"] is False``) after the loop. Declines and verdict votes are
    TP-unanimous inside rowinv, so every rank abandons together or none does.
    """
    src_dtype = vocab_parallel_logits.dtype
    seq_len = int(vocab_parallel_logits.shape[1])
    step = seq_len if chunk_size is None else int(chunk_size)
    # inference_only mirrors the incumbent contract exactly: the Functions save nothing under it
    # and offer no backward, so rowinv must not quietly grow one (nor retain fp32 chunks for a
    # backward nobody may take).
    grad_on = torch.is_grad_enabled() and not inference_only
    outs = []
    with torch.set_grad_enabled(grad_on):
        # torch.split, NOT per-chunk basic indexing. The views are byte-identical to
        # `vocab_parallel_logits[:, cs:ce, :]` (same storage, shape, stride -- same hot key, same
        # kernels, same collectives in the same order), so the FORWARD bits cannot move. What
        # changes is the BACKWARD's grad ROUTING, where this dispatch has latitude (grad reaches
        # only the optimizer; allclose-gated): basic indexing gives every chunk its own
        # SliceBackward, whose grad is a FULL-size [.., seq, V/TP] zero-fill with the chunk's grad
        # copied in, and AccumulateGrad then sums num_chunks of those full tensors -- measured at
        # 141 ms per 20480x62080 fp32 microbatch at TP=4 against 12.6 ms of actual rowinv backward
        # math (~20 x 25 GB of zero+add traffic). split's ONE backward node cats all chunk grads
        # in a single pass (~5 GB), and the summed grad values are unchanged: each element gets
        # exactly one chunk's contribution either way.
        for chunk, target_chunk in zip(
            torch.split(vocab_parallel_logits, step, dim=1),
            torch.split(target, step, dim=1),
        ):
            if not chunk.is_contiguous():
                chunk = chunk.contiguous()

            def reference(chunk=chunk, target_chunk=target_chunk) -> torch.Tensor:
                # The true incumbent for this chunk, end to end: the exact statement sequence of
                # the Functions' inference path (widen, gather branch, aten-order lse, mask,
                # all_reduce). Probe-path only, so the fp32 copy it widens is transient.
                # rowinv's first-call probe compares against this, and returns it on failure.
                target_mask = (target_chunk < vocab_start_index) | (target_chunk >= vocab_end_index)
                masked_target = (target_chunk - vocab_start_index).masked_fill(target_mask, 0)
                ref = _compute_distributed_log_softmax(chunk.to(dtype=torch.float32), group=group, src_dtype=src_dtype)
                ref = torch.gather(ref, -1, masked_target.unsqueeze(-1)).squeeze(-1)
                ref = ref.masked_fill(target_mask, 0.0)
                torch.distributed.all_reduce(ref, op=torch.distributed.ReduceOp.SUM, group=group)
                return ref

            got = _ix_rowinv_sampled_logprobs(
                chunk,
                target_chunk,
                vocab_start_index=vocab_start_index,
                vocab_end_index=vocab_end_index,
                group=group,
                src_dtype=src_dtype,
                reference=reference,
            )
            if got is None:
                return None
            if grad_on and vocab_parallel_logits.requires_grad and not got.requires_grad:
                # Probe-failure fallback under grad: rowinv handed back the reference's tensor,
                # which carries no graph. A silent zero-gradient chunk is worse than the fallback
                # being slow; refuse it and let the incumbent Function serve the whole call.
                return None
            outs.append(got)
    if _ix_rowinv_stats().get("agreed") is False:
        # A probe failure latched rowinv off. Chunks before it carry rowinv bits and the failing
        # chunk carries incumbent bits -- never return that mix; the incumbent serves whole.
        return None
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=1)


def _ix_logprobs_apply(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start_index: int,
    vocab_end_index: int,
    group: torch.distributed.ProcessGroup,
    inference_only: bool,
    chunk_size: Optional[int],
) -> torch.Tensor:
    """The ONE dispatch point between the rowinv leaf-tree logprob and the incumbent Functions.

    Sits OUTSIDE any autograd.Function on purpose: ``rowinv_sampled_logprobs`` is itself
    autograd-capable, but a Function applied inside another Function's forward cannot carry its
    backward out, so an in-Function hook could serve only ``inference_only=True`` -- improving the
    scoring gate while the optimized objective kept the old schedule, and splitting scoring from
    training onto different functions. Dispatching here serves BOTH, which is what the contract's
    ``trainer_fwd`` + ``trainer_score`` entries declare when the flag is on.

    Flag off (the default), this is byte-for-byte today's selection: the same ``.apply(...)`` on
    the same chunk-size branch, and rowinv is never consulted.
    """
    seq_len_local = int(vocab_parallel_logits.shape[1])
    # Only use the chunked path when chunking actually splits the sequence into multiple chunks.
    # When chunk_size >= seq_len the whole sequence is one chunk, but ChunkedDistributedLogprob
    # still saves the raw vocab_parallel_logits and recomputes softmax in backward (~3x peak
    # memory vs DistributedLogprob's ~2x), so chunking actively hurts in that regime.
    use_chunked = chunk_size is not None and chunk_size < seq_len_local

    if _ix_rowinv_available():
        got = _ix_try_rowinv_logprobs(
            vocab_parallel_logits,
            target,
            vocab_start_index,
            vocab_end_index,
            group,
            inference_only,
            chunk_size if use_chunked else None,
        )
        if got is not None:
            return got.contiguous()

    if use_chunked:
        return ChunkedDistributedLogprob.apply(  # type: ignore
            vocab_parallel_logits,
            target,
            vocab_start_index,
            vocab_end_index,
            chunk_size,
            group,
            inference_only,
        ).contiguous()
    return DistributedLogprob.apply(  # type: ignore
        vocab_parallel_logits,
        target,
        vocab_start_index,
        vocab_end_index,
        group,
        inference_only,
    ).contiguous()


def from_parallel_logits_to_logprobs(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start_index: int,
    vocab_end_index: int,
    tp_group: torch.distributed.ProcessGroup,
    inference_only: bool = False,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Get log probabilities from TP+CP sharded vocab logits.

    Args:
        vocab_parallel_logits (torch.Tensor): Logits tensor with shape [batch_size, seq_len // CP, vocab_size // TP]
            where TP is the tensor parallel size.
        target (torch.Tensor): Target token indices with shape [batch_size, seq_len].
            NOTE: Must be the unmodified targets as this function will shift them internally.
        vocab_start_index (int): Starting vocabulary index for this worker's partition.
        vocab_end_index (int): Ending vocabulary index for this worker's partition.
        tp_group (torch.distributed.ProcessGroup): Process group for distributed communication.
        inference_only (bool, optional): If True, tensors won't be saved for backward pass. Defaults to False.
        cp_group (torch.distributed.ProcessGroup, optional): Context parallelism process group. Defaults to None.
        chunk_size (int, optional): Sequence dimension chunk size for computing the log probabilities.

    Returns:
        torch.Tensor: Log probabilities tensor with shape [batch_size, seq_len-1].
            The sequence dimension is reduced by 1 due to the target shifting.

    Taken from: https://github.com/NVIDIA/NeMo-Aligner/blob/9faab404f21994a7eb1d6ed5890b76152b941636/nemo_aligner/utils/distributed.py#L354
    """
    target = target.roll(shifts=-1, dims=-1)
    cp_size = 1 if cp_group is None else torch.distributed.get_world_size(cp_group)
    pad_len = 0
    # if cp_size > 1:
    # Pad the targets to local size * cp_size
    pad_len = vocab_parallel_logits.shape[1] * cp_size - target.shape[1]
    if pad_len > 0:
        target = torch.nn.functional.pad(target, (0, pad_len), value=0)

    # Shard the targets by context parallelism
    cp_rank = torch.distributed.get_rank(cp_group)
    target = _get_tokens_on_this_cp_rank(target, cp_rank, cp_size, seq_dim=1)

    # The chunk-size regime choice and the rowinv dispatch both live in _ix_logprobs_apply --
    # the ONE dispatch point, outside any autograd.Function so the scoring AND the grad-bearing
    # forward run the same function.
    logprobs: torch.Tensor = _ix_logprobs_apply(
        vocab_parallel_logits,
        target,
        vocab_start_index,
        vocab_end_index,
        tp_group,
        inference_only,
        chunk_size,
    )

    if cp_size > 1:
        # we need to gather the logits by context parallelism
        logprobs = allgather_cp_sharded_tensor(logprobs, cp_group, seq_dim=1)  # , unpadded_seqlen=target.shape[1])

    if pad_len > 0:
        logprobs = logprobs[:, :-pad_len]

    return logprobs[:, :-1]


def from_parallel_logits_to_logprobs_packed_sequences(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    unpacked_seqlen: int,
    vocab_start_index: int,
    vocab_end_index: int,
    group: torch.distributed.ProcessGroup,
    inference_only: bool = False,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    chunk_size: Optional[int] = None,
    attention_mask: Optional[torch.Tensor] = None,
    sub_seq_lengths: Optional[list[list[int]]] = None,
) -> torch.Tensor:
    """Get log probabilities from TP sharded vocab logits for packed sequences.

    Args:
        vocab_parallel_logits (torch.Tensor): Packed logits tensor with shape [1, T // CP, vocab_size//TP]
            where T is the total number of tokens across all packed sequences.
        target (torch.Tensor): Packed target token indices with shape [1, T].
            NOTE: Must be the unmodified targets as this function will shift them internally.
        cu_seqlens (torch.Tensor): Cumulative sequence lengths tensor with shape [batch_size + 1].
            cu_seqlens[i] indicates the start position of sequence i in the packed format.
        unpacked_seqlen (int): The length of the unpacked sequence tensor.
        vocab_start_index (int): Starting vocabulary index for this worker's partition.
        vocab_end_index (int): Ending vocabulary index for this worker's partition.
        group (torch.distributed.ProcessGroup): Process group for distributed communication.
        inference_only (bool, optional): If True, tensors won't be saved for backward pass. Defaults to False.
        cp_group (torch.distributed.ProcessGroup, optional): Context parallelism process group. Defaults to None.
        chunk_size (int, optional): Sequence dimension chunk size for computing the log probabilities.
        attention_mask (torch.Tensor, optional): Original unpacked attention mask with shape [batch_size, unpacked_seqlen].
            When provided, packed log probabilities are scattered back to their original padded sequence positions.
        sub_seq_lengths (list[list[int]], optional): Per-row sub-sequence lengths for controller-side sequence packing.
            When provided, ``cu_seqlens_padded`` is interpreted as one entry per sub-sequence, and output values are
            scattered back to the row offsets used by ``PackedDataCollator``.

    Returns:
        torch.Tensor: Unpacked log probabilities tensor with shape [batch_size, unpacked_seqlen-1].
            The total length is reduced by batch_size due to target shifting (one token per sequence).
    """
    # This packed logprob path has been verified by Megatron GSM8K E2E runs covering no-CP, CP ring, and CP a2a.
    # Remove batch dimension to work with [T, vocab_size] and [T]
    vocab_parallel_logits = vocab_parallel_logits.squeeze(0)
    target = target.squeeze(0)

    batch_size = len(sub_seq_lengths) if sub_seq_lengths is not None else cu_seqlens_padded.shape[0] - 1
    cp_size = 1 if cp_group is None else torch.distributed.get_world_size(cp_group)
    cp_rank = 0 if cp_group is None else torch.distributed.get_rank(cp_group)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=target.device, dtype=torch.bool)

    cu_seqlens_padded, _, seq_indices, seq_offsets, seq_lens_padded = _packed_sequence_indices(
        cu_seqlens_padded, target.shape[0], target.device
    )

    next_offsets = torch.remainder(seq_offsets + 1, seq_lens_padded[seq_indices])
    rolled_targets_full = target[cu_seqlens_padded[seq_indices] + next_offsets]
    if cp_size > 1:
        cp_rank_for_token, local_indices = _packed_cp_rank_and_local_indices(
            cu_seqlens_padded, seq_indices, seq_offsets, seq_lens_padded, cp_size
        )
        rolled_targets = torch.empty(target.shape[0] // cp_size, dtype=target.dtype, device=target.device)
        current_rank_mask = cp_rank_for_token == cp_rank
        rolled_targets[local_indices[current_rank_mask]] = rolled_targets_full[current_rank_mask]
    else:
        rolled_targets = rolled_targets_full

    # Add batch dimension back for DistributedLogprob
    rolled_targets = rolled_targets.unsqueeze(0)
    vocab_parallel_logits = vocab_parallel_logits.unsqueeze(0)

    # Apply distributed log probability computation. The chunk-size regime choice and the rowinv
    # dispatch both live in _ix_logprobs_apply -- the ONE dispatch point, outside any
    # autograd.Function so the scoring AND the grad-bearing forward run the same function.
    probs: torch.Tensor = _ix_logprobs_apply(
        vocab_parallel_logits,
        rolled_targets,
        vocab_start_index,
        vocab_end_index,
        group,
        inference_only,
        chunk_size,
    )

    # Remove batch dimension for filtering
    probs = probs.squeeze(0)

    # Ensure probs is 1D after squeezing
    if probs.dim() != 1:
        raise ValueError(
            f"Expected probs to be 1D after squeezing, but got shape {probs.shape}. "
            f"Original shape before squeeze: {probs.unsqueeze(0).shape}"
        )

    if cp_size > 1:
        probs = allgather_cp_sharded_packed_tensor(probs, cu_seqlens_padded, cp_group)

    out_logprobs = torch.zeros((batch_size, unpacked_seqlen - 1), dtype=probs.dtype, device=probs.device)
    _, _, seq_indices, seq_offsets, seq_lens_padded = _packed_sequence_indices(
        cu_seqlens_padded, probs.shape[0], probs.device
    )

    if sub_seq_lengths is not None:
        row_indices, row_offsets, seq_lens = _packed_subseq_row_indices_offsets_and_lens(
            cu_seqlens_padded, sub_seq_lengths, probs.device
        )
        valid_counts = torch.clamp(seq_lens - 1, min=0)
        packed_mask = seq_offsets < valid_counts[seq_indices]
        output_cols = row_offsets[seq_indices[packed_mask]] + seq_offsets[packed_mask]
        output_rows = row_indices[seq_indices[packed_mask]]
        output_in_bounds = output_cols < unpacked_seqlen - 1
        out_logprobs[output_rows[output_in_bounds], output_cols[output_in_bounds]] = probs[packed_mask][
            output_in_bounds
        ]
        return out_logprobs

    if attention_mask is not None:
        seq_lens = attention_mask.sum(dim=1, dtype=torch.long)
        token_ordinals = attention_mask.to(torch.long).cumsum(dim=1)
        output_mask = attention_mask[:, :-1] & (token_ordinals[:, :-1] < seq_lens.unsqueeze(1))
        valid_counts = torch.clamp(seq_lens - 1, min=0)
        packed_mask = seq_offsets < valid_counts[seq_indices]
        out_logprobs[output_mask] = probs[packed_mask]
        return out_logprobs

    valid_counts = torch.clamp(seq_lens_padded - 1, min=0)
    packed_mask = (seq_offsets < valid_counts[seq_indices]) & (seq_offsets < unpacked_seqlen - 1)
    out_logprobs[seq_indices[packed_mask], seq_offsets[packed_mask]] = probs[packed_mask]

    return out_logprobs


def _packed_subseq_row_indices_offsets_and_lens(
    cu_seqlens_padded: torch.Tensor, sub_seq_lengths: list[list[int]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-packed-segment row metadata for controller-side sequence packing."""
    cu_seqlens_cpu = cu_seqlens_padded.detach().cpu().tolist()
    padded_lens = [cu_seqlens_cpu[i + 1] - cu_seqlens_cpu[i] for i in range(len(cu_seqlens_cpu) - 1)]

    row_indices: list[int] = []
    row_offsets: list[int] = []
    seq_lens: list[int] = []
    seg_idx = 0
    for row_idx, row_lens in enumerate(sub_seq_lengths):
        row_offset = 0
        for seq_len in row_lens:
            if seg_idx >= len(padded_lens):
                raise ValueError("sub_seq_lengths contains more sub-sequences than cu_seqlens_padded")
            row_indices.append(row_idx)
            row_offsets.append(row_offset)
            seq_lens.append(int(seq_len))
            row_offset += padded_lens[seg_idx]
            seg_idx += 1

    if seg_idx != len(padded_lens):
        raise ValueError(
            f"sub_seq_lengths describes {seg_idx} sub-sequences, but cu_seqlens_padded describes {len(padded_lens)}"
        )

    return (
        torch.tensor(row_indices, dtype=torch.long, device=device),
        torch.tensor(row_offsets, dtype=torch.long, device=device),
        torch.tensor(seq_lens, dtype=torch.long, device=device),
    )


def _packed_sequence_indices(
    cu_seqlens_padded: torch.Tensor, total_tokens: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cu_seqlens_padded = cu_seqlens_padded.to(device=device, dtype=torch.long)
    token_indices = torch.arange(total_tokens, device=device)
    seq_indices = torch.searchsorted(cu_seqlens_padded[1:], token_indices, right=True)
    seq_offsets = token_indices - cu_seqlens_padded[seq_indices]
    seq_lens_padded = cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
    return cu_seqlens_padded, token_indices, seq_indices, seq_offsets, seq_lens_padded


def _packed_cp_rank_and_local_indices(
    cu_seqlens_padded: torch.Tensor,
    seq_indices: torch.Tensor,
    seq_offsets: torch.Tensor,
    seq_lens_padded: torch.Tensor,
    cp_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cp_size == 1:
        return torch.zeros_like(seq_indices), cu_seqlens_padded[seq_indices] + seq_offsets

    seq_lens_for_token = seq_lens_padded[seq_indices]
    chunk_size = torch.div(seq_lens_for_token, 2 * cp_size, rounding_mode="floor")
    chunk_indices = torch.div(seq_offsets, chunk_size, rounding_mode="floor")
    rank_for_token = torch.where(chunk_indices < cp_size, chunk_indices, 2 * cp_size - chunk_indices - 1)
    within_chunk_offsets = seq_offsets - chunk_indices * chunk_size
    within_rank_offsets = torch.where(
        chunk_indices < cp_size,
        within_chunk_offsets,
        chunk_size + within_chunk_offsets,
    )
    local_starts = torch.div(cu_seqlens_padded[:-1], cp_size, rounding_mode="floor")
    local_indices = local_starts[seq_indices] + within_rank_offsets
    return rank_for_token, local_indices


def _get_tokens_on_this_cp_rank(
    input_ids: torch.Tensor,
    cp_rank: int,
    cp_size: int,
    seq_dim: int = 1,
) -> torch.Tensor:
    """Get tokens on this context parallelism rank.

    Assumes that input_ids are already padded to a multiple of cp_size * 2 or cp_size == 1.

    Args:
        input_ids: Input token IDs [seq_length, ]
        cp_rank: Context parallelism rank
        cp_size: Context parallelism size

    Returns:
        Tokens on this context parallelism rank [1, seq_length // cp_size]
    """
    if cp_size == 1:
        return input_ids

    # load balance for causal attention
    shard_size = input_ids.shape[seq_dim] // (cp_size * 2)
    shard_inds = (cp_rank, (cp_size * 2) - cp_rank - 1)

    # Create slices for each dimension
    slices = [slice(None)] * input_ids.dim()
    ids_chunks = []

    for ind in shard_inds:
        slices[seq_dim] = slice(ind * shard_size, (ind + 1) * shard_size)
        ids_chunks.append(input_ids[slices])

    ids = torch.cat(ids_chunks, dim=seq_dim)
    return ids


def allgather_cp_sharded_tensor(tensor, cp_group, seq_dim=1):  # , unpadded_seqlen=None):
    return AllGatherCPTensor.apply(tensor, cp_group, seq_dim)  # , unpadded_seqlen)


def allgather_cp_sharded_packed_tensor(tensor, cu_seqlens_padded, cp_group):
    return AllGatherPackedCPTensor.apply(tensor, cu_seqlens_padded, cp_group)


def vocab_parallel_entropy_packed_sequences(
    vocab_parallel_logits: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    unpacked_seqlen: int,
    num_actions: int,
    attention_mask: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    cp_group: Optional[torch.distributed.ProcessGroup],
    sub_seq_lengths: Optional[list[list[int]]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute action-token entropy directly on TP+CP sharded packed logits.

    Returns:
        A tuple of (global entropy metric, local entropy term for loss). The
        local term is normalized by the global action-token count. Megatron's
        schedule already applies the CP loss scale for two-output loss funcs.
    """
    entropy_tokens = vocab_parallel_entropy(vocab_parallel_logits).squeeze(0)
    device = entropy_tokens.device
    dtype = entropy_tokens.dtype

    attention_mask = attention_mask.to(device=device, dtype=torch.bool)
    cu_seqlens_padded = cu_seqlens_padded.to(device=device, dtype=torch.long)
    batch_size = attention_mask.shape[0]

    action_weights = torch.zeros((batch_size, unpacked_seqlen - 1), dtype=dtype, device=device)
    if loss_mask is None:
        action_weights[:, -num_actions:] = 1.0
    else:
        action_weights[:, -num_actions:] = loss_mask.to(device=device, dtype=dtype)

    packed_weights = torch.zeros((int(cu_seqlens_padded[-1].item()),), dtype=dtype, device=device)
    if sub_seq_lengths is not None:
        _, _, seq_indices, seq_offsets, _ = _packed_sequence_indices(cu_seqlens_padded, packed_weights.shape[0], device)
        row_indices, row_offsets, seq_lens = _packed_subseq_row_indices_offsets_and_lens(
            cu_seqlens_padded, sub_seq_lengths, device
        )
        valid_counts = torch.clamp(seq_lens - 1, min=0)
        packed_mask = seq_offsets < valid_counts[seq_indices]
        output_cols = row_offsets[seq_indices[packed_mask]] + seq_offsets[packed_mask]
        output_rows = row_indices[seq_indices[packed_mask]]
        output_in_bounds = output_cols < action_weights.shape[1]
        packed_weights[torch.arange(packed_weights.shape[0], device=device)[packed_mask][output_in_bounds]] = (
            action_weights[output_rows[output_in_bounds], output_cols[output_in_bounds]]
        )
    else:
        seq_lens = attention_mask.sum(dim=1, dtype=torch.long)
        token_ordinals = attention_mask.to(torch.long).cumsum(dim=1)
        output_mask = attention_mask[:, :-1] & (token_ordinals[:, :-1] < seq_lens.unsqueeze(1))

        token_offsets = token_ordinals - 1
        packed_indices = cu_seqlens_padded[:-1].unsqueeze(1) + token_offsets
        packed_weights[packed_indices[:, :-1][output_mask]] = action_weights[output_mask]

    cp_size = 1 if cp_group is None else torch.distributed.get_world_size(cp_group)
    if cp_size > 1:
        cp_rank = torch.distributed.get_rank(cp_group)
        _, _, seq_indices, seq_offsets, seq_lens_padded = _packed_sequence_indices(
            cu_seqlens_padded, packed_weights.shape[0], device
        )
        cp_rank_for_token, local_indices = _packed_cp_rank_and_local_indices(
            cu_seqlens_padded, seq_indices, seq_offsets, seq_lens_padded, cp_size
        )
        local_weights = torch.zeros_like(entropy_tokens)
        current_rank_mask = cp_rank_for_token == cp_rank
        local_weights[local_indices[current_rank_mask]] = packed_weights[current_rank_mask]
    else:
        local_weights = packed_weights

    local_entropy_sum = (entropy_tokens * local_weights).sum()
    local_count = local_weights.sum()
    global_count = local_count.detach().clone()
    global_entropy_sum = local_entropy_sum.detach().clone()
    if cp_size > 1:
        torch.distributed.all_reduce(global_count, group=cp_group)
        torch.distributed.all_reduce(global_entropy_sum, group=cp_group)
    global_count = global_count.clamp(min=1.0)

    entropy = global_entropy_sum / global_count
    entropy_for_loss = local_entropy_sum / global_count
    return entropy, entropy_for_loss


class AllGatherPackedCPTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, cu_seqlens_padded: torch.Tensor, cp_group: torch.distributed.ProcessGroup):
        cp_size = torch.distributed.get_world_size(cp_group)
        cp_rank_chunks = [torch.empty_like(tensor) for _ in range(cp_size)]
        torch.distributed.all_gather(tensor_list=cp_rank_chunks, tensor=tensor, group=cp_group)

        total_tokens = tensor.shape[0] * cp_size
        cu_seqlens_padded, _, seq_indices, seq_offsets, seq_lens_padded = _packed_sequence_indices(
            cu_seqlens_padded, total_tokens, tensor.device
        )
        cp_rank_for_token, local_indices = _packed_cp_rank_and_local_indices(
            cu_seqlens_padded, seq_indices, seq_offsets, seq_lens_padded, cp_size
        )

        gathered = torch.stack(cp_rank_chunks, dim=0)
        output = gathered[cp_rank_for_token, local_indices]

        ctx.cp_group = cp_group
        ctx.save_for_backward(cu_seqlens_padded)
        ctx.local_tokens = tensor.shape[0]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        cp_size = torch.distributed.get_world_size(ctx.cp_group)
        cp_rank = torch.distributed.get_rank(ctx.cp_group)
        (cu_seqlens_padded,) = ctx.saved_tensors

        cu_seqlens_padded, _, seq_indices, seq_offsets, seq_lens_padded = _packed_sequence_indices(
            cu_seqlens_padded, grad_output.shape[0], grad_output.device
        )
        cp_rank_for_token, local_indices = _packed_cp_rank_and_local_indices(
            cu_seqlens_padded, seq_indices, seq_offsets, seq_lens_padded, cp_size
        )

        local_rank_mask = cp_rank_for_token == cp_rank
        grad_input = torch.zeros(ctx.local_tokens, dtype=grad_output.dtype, device=grad_output.device)
        grad_input[local_indices[local_rank_mask]] = grad_output[local_rank_mask]
        return grad_input, None, None


class AllGatherCPTensor(torch.autograd.Function):
    def forward(
        ctx, tensor, cp_group: torch.distributed.ProcessGroup, seq_dim=1
    ):  # , unpadded_seqlen: Optional[int] = None):
        cp_size = torch.distributed.get_world_size(cp_group)
        cp_rank_chunks = []
        for _ in range(cp_size):
            cp_rank_chunks.append(torch.empty_like(tensor))

        torch.distributed.all_gather(tensor_list=cp_rank_chunks, tensor=tensor, group=cp_group)

        # undo the CP load balancing chunking
        tensor_chunks = []
        for logit_chunk in cp_rank_chunks:
            tensor_chunks.extend(torch.chunk(logit_chunk, chunks=2, dim=seq_dim))

        chunk_indices = []
        for cp_rank in range(cp_size):
            chunk_indices.append(cp_rank)
            chunk_indices.append(2 * cp_size - cp_rank - 1)

        chunks_and_indices = list(zip(tensor_chunks, chunk_indices))
        chunks_and_indices = sorted(chunks_and_indices, key=lambda tup: tup[1])
        ret_tensor = [chunk for chunk, _ in chunks_and_indices]
        ret_tensor = torch.cat(ret_tensor, dim=seq_dim)

        ctx.seq_dim = seq_dim
        ctx.cp_group = cp_group
        # ctx.unpadded_seqlen = unpadded_seqlen

        return ret_tensor

    def backward(ctx, grad_output):
        cp_size = torch.distributed.get_world_size(ctx.cp_group)
        cp_rank = torch.distributed.get_rank(ctx.cp_group)
        torch.distributed.all_reduce(grad_output, group=ctx.cp_group)

        # chunk the seqdim in 2*cp chunks, and select with a CP load balanced indexing
        seq_dim = ctx.seq_dim
        # if ctx.unpadded_seqlen is not None:
        # # Zero out grad_output along the seq_dim after unpadded_seqlen
        # slicer = [slice(None)] * grad_output.dim()
        # slicer[seq_dim] = slice(ctx.unpadded_seqlen, None)
        #     grad_output[tuple(slicer)] = 0

        grad_output = grad_output.view(
            *grad_output.shape[0:seq_dim],
            2 * cp_size,
            grad_output.shape[seq_dim] // (2 * cp_size),
            *grad_output.shape[(seq_dim + 1) :],
        )

        index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True).cuda(
            non_blocking=True
        )

        grad_input = grad_output.index_select(seq_dim, index)
        grad_input = grad_input.view(*grad_input.shape[0:seq_dim], -1, *grad_input.shape[(seq_dim + 2) :])

        return grad_input, None, None  # , None


# Below ported from https://github.com/volcengine/verl/blob/main/verl/utils/megatron/tensor_parallel.py#L109
class _VocabParallelEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
        @torch.compile(dynamic=True)
        def mul_reduce(a, b):
            return (a * b).sum(dim=-1, keepdim=True)

        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group())
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
        normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(normalized_sum_exp_logits, group=mpu.get_tensor_model_parallel_group())
        softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
        sum_softmax_times_logits = mul_reduce(softmax_logits, vocab_parallel_logits)
        dist.all_reduce(sum_softmax_times_logits, group=mpu.get_tensor_model_parallel_group())
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        return entropy.squeeze(dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        # grad = softmax * (sum_softmax_times_logits - vocab_parallel_logits) * grad_output
        # NOTE: do NOT mutate vocab_parallel_logits in-place. The same logits tensor may also
        # be saved for backward by ChunkedDistributedLogprob; even the "sub_ then add_" restore
        # pattern bumps the storage version counter and trips that Function's version check.
        softmax_logits.mul_(sum_softmax_times_logits - vocab_parallel_logits)
        softmax_logits.mul_(grad_output.unsqueeze(dim=-1))
        return softmax_logits


def vocab_parallel_entropy(vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy when the logits are sharded in tp ranks

    Args:
        vocab_parallel_logits: (total_nnz, vocab_size // tp_size)

    Returns: (total_nnz,)

    """
    return _VocabParallelEntropy.apply(vocab_parallel_logits)
