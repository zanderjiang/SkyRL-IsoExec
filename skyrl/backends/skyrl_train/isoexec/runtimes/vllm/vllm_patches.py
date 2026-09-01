"""vLLM monkey-patches for bitwise-matched rollouts.

Covers the batch-invariance env pins, flash-attention ``num_splits=1``, the two logprob sites and
the temperature-divide skip. Every patch is idempotent and leaves vLLM unchanged when it cannot
install.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

ISOEXEC_VLLM_ENV = {
    "VLLM_BATCH_INVARIANT": "1",
    "VLLM_USE_AOT_COMPILE": "0",
    "NCCL_ALGO": "allreduce:tree",
    "NCCL_MIN_NCHANNELS": "1",
    "NCCL_MAX_NCHANNELS": "1",
}


def apply_vllm_isoexec_env(env: dict | None = None) -> dict:
    target = os.environ if env is None else env
    to_set = dict(ISOEXEC_VLLM_ENV)

    if os.environ.get("SKYRL_ISOEXEC_NCCL_PIN", "1") != "1":
        for k in ("NCCL_ALGO", "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS"):
            to_set.pop(k, None)
            target.pop(k, None)
        logger.info("[isoexec] SKYRL_ISOEXEC_NCCL_PIN=0 -> NCCL channel/algo pins NOT applied (A/B mode)")

    for k, v in to_set.items():
        target[k] = v
    logger.info("[isoexec] set vLLM batch-invariant env: %s", to_set)
    return dict(to_set)


def neutralize_vllm_nccl_channel_pin() -> bool:
    """Wrap vLLM's ``override_envs_for_invariance`` so its NCCL channel pin is relaxed afterwards.

    Must be installed from the general plugin, before ``init_batch_invariance`` runs in the worker.
    """
    if os.environ.get("SKYRL_ISOEXEC_ENGINE_NCCL_UNPIN", "0") != "1":
        return False
    try:
        import vllm.model_executor.layers.batch_invariant as _bi
    except Exception as e:
        logger.warning("[isoexec] cannot neutralize the NCCL channel pin: %s", e)
        return False
    if getattr(_bi, "_isoexec_channel_pin_neutralized", False):
        return True
    _orig = _bi.override_envs_for_invariance

    def _wrapped(*args, **kwargs):
        r = _orig(*args, **kwargs)

        os.environ.pop("NCCL_MIN_NCHANNELS", None)
        _cap = os.environ.get("SKYRL_ISOEXEC_ENGINE_NCCL_MAX_NCHANNELS", "8").strip()
        if _cap:
            os.environ["NCCL_MAX_NCHANNELS"] = _cap
        else:
            os.environ.pop("NCCL_MAX_NCHANNELS", None)

        print(
            "[ISOEXEC-NCCL] engine worker post-override env: "
            f"NCCL_ALGO={os.environ.get('NCCL_ALGO')} "
            f"NCCL_MIN_NCHANNELS={os.environ.get('NCCL_MIN_NCHANNELS')} "
            f"NCCL_MAX_NCHANNELS={os.environ.get('NCCL_MAX_NCHANNELS')}",
            flush=True,
        )
        return r

    _wrapped._isoexec_orig = _orig
    _bi.override_envs_for_invariance = _wrapped
    _bi._isoexec_channel_pin_neutralized = True
    logger.info("[isoexec] ENGINE NCCL channel pin neutralized (SKYRL_ISOEXEC_ENGINE_NCCL_UNPIN=1)")
    print("[ISOEXEC-NCCL] channel pin NEUTRALIZED -- vLLM's re-pin is wrapped", flush=True)
    return True


def apply_flash_num_splits_patch() -> bool:
    try:
        import vllm.v1.attention.backends.flash_attn as _fa
    except Exception as e:
        logger.warning("[isoexec] flash num_splits patch: cannot import flash_attn backend: %s", e)
        return False
    if getattr(_fa, "_isoexec_num_splits_patched", False):
        return True
    _orig = _fa.flash_attn_varlen_func

    def _wrapped(*args, **kwargs):
        if os.environ.get("VLLM_BATCH_INVARIANT") == "1" and "num_splits" not in kwargs:
            kwargs["num_splits"] = 1
        return _orig(*args, **kwargs)

    _wrapped._isoexec_orig = _orig
    _fa.flash_attn_varlen_func = _wrapped
    _fa._isoexec_num_splits_patched = True
    logger.info("[isoexec] patched flash_attn_varlen_func -> num_splits=1 (main decode path, batch-invariant)")
    print("[ISOEXEC-ATTN] flash_attn_varlen_func pinned to num_splits=1 (decode==prefill fix)", flush=True)
    return True


def isoexec_engine_arg_overrides() -> dict:
    return {
        "enforce_eager": True,
        "enable_prefix_caching": False,
    }


def patch_vllm_logprobs_batch_invariant() -> bool:
    """Rebind the **Model Runner V2** token-logprob kernel to the trainer's aten formulation.

    INERT on the V1 runner, which every production composition resolves; kept only so a future flip
    to V2 does not silently reopen the divergence. The live V1 site is
    :func:`patch_vllm_sampler_logprobs_rowinv`; never attest an engine logprob impl off this return.
    """
    try:
        import torch
        import vllm.v1.worker.gpu.sample.logprob as _lp
    except Exception as e:
        logger.warning("[isoexec] cannot patch vLLM logprob kernel: %s", e)
        return False
    if getattr(_lp, "_isoexec_logprob_patched", False):
        return True

    # Lazy import, fail-to-incumbent: a missing rowinv module leaves the aten-order path in charge.
    try:
        from skyrl.backends.skyrl_train.isoexec.ops.logprobs.rowinv import (
            rowinv_sampled_logprobs as _rowinv_sampled_logprobs,
        )

        _rowinv_available = True
    except ImportError as _rowinv_import_error:
        _err = repr(_rowinv_import_error)
        _rowinv_available = False
        logger.error("[isoexec] rowinv import failed; engine keeps the incumbent: %s", _err)
        print(
            f"[ISOEXEC-ROWINV-LOGPROB] REFUSED: module import failed; engine keeps the incumbent ({_err})",
            flush=True,
        )

        def _rowinv_sampled_logprobs(*args, **kwargs):  # type: ignore[misc]  # pragma: no cover
            return None

    def _reference_compute_token_logprobs(logits, token_ids):

        token_ids = token_ids.to(torch.int64)
        x = logits.to(torch.float32)
        x = x - torch.amax(x, dim=-1, keepdim=True)
        lse = x.exp().sum(-1, keepdim=True).float().log()
        logprobs = x - lse
        return logprobs.gather(-1, token_ids)

    def _batch_invariant_compute_token_logprobs(logits, token_ids):
        # rowinv first, incumbent on DECLINE (None). The engine receives an already-gathered full
        # [N, V] row, so it passes group=None and computes all G leaves locally while the trainer
        # hands in its TP shard; leaf boundaries depend only on G, so the bits agree.
        if _rowinv_available:
            got = _rowinv_sampled_logprobs(
                logits,
                token_ids,
                vocab_start_index=0,
                vocab_end_index=logits.shape[-1],
                group=None,
                # The dtype the row arrives in, before the incumbent's fp32 widening.
                src_dtype=logits.dtype,
                reference=lambda: _reference_compute_token_logprobs(logits, token_ids),
            )
            if got is not None:
                return got
        return _reference_compute_token_logprobs(logits, token_ids)

    _lp.compute_token_logprobs = _batch_invariant_compute_token_logprobs
    _lp._isoexec_logprob_patched = True
    logger.info("[isoexec] patched V2-runner compute_token_logprobs (INERT on the V1 runner)")
    print(
        "[ISOEXEC-ENGINE] patched vLLM V2-runner compute_token_logprobs -> aten log_softmax. "
        "NOTE: this module belongs to Model Runner V2 and is NEVER CALLED by the V1 GPUModelRunner "
        "that production resolves (use_v2_model_runner=False for this arch); kept for a future V2 "
        "flip only. The V1 logprob producer is Sampler.compute_logprobs -- see "
        "patch_vllm_sampler_logprobs_rowinv.",
        flush=True,
    )
    return True


#: Census of the V1 ``Sampler.compute_logprobs`` hook: where calls go, not whether rowinv engaged.
#: Engagement truth stays with ``ops/logprobs/rowinv.py::stats()['served']``.
_CL_STATE = {"calls": 0, "rowinv_owned": 0, "incumbent": 0, "reported": 0}


def sampler_logprobs_census() -> dict:
    return dict(_CL_STATE)


def _cl_report() -> None:
    n = _CL_STATE["calls"]
    if n < 1 or (n & (n - 1)) != 0 or n == _CL_STATE["reported"]:
        return
    _CL_STATE["reported"] = n
    print(
        f"[ISOEXEC-SAMPLER-LOGPROBS] pid={os.getpid()} V1 Sampler.compute_logprobs hook: "
        f"calls={n} rowinv_owned={_CL_STATE['rowinv_owned']} incumbent={_CL_STATE['incumbent']}",
        flush=True,
    )


def sampler_logprobs_hook_state() -> dict:
    """Live evidence for the engine attestation, read off the ``Sampler`` class itself.

    Never attest off "a patch function returned True": the V2 patch installs but never executes.
    """
    state = {"v1_hook_installed": False, "v1_hook_calls": 0, "rowinv_available": False, "error": ""}
    try:
        from vllm.v1.sample.sampler import Sampler

        raw = Sampler.__dict__.get("compute_logprobs")
        if isinstance(raw, staticmethod):
            raw = raw.__func__
        state["v1_hook_installed"] = bool(getattr(raw, "_isoexec_rowinv_full_row_hook", False))
        state["v1_hook_calls"] = int(_CL_STATE["calls"])
    except Exception as e:  # noqa: BLE001 -- no vllm means no evidence, and the record says so
        state["error"] = repr(e)
    try:
        from skyrl.backends.skyrl_train.isoexec.ops.logprobs import rowinv  # noqa: F401

        state["rowinv_available"] = True
    except Exception as e:  # noqa: BLE001
        state["error"] = (state["error"] + "; " if state["error"] else "") + repr(e)
    return state


def patch_vllm_sampler_logprobs_rowinv() -> bool:
    """Hook the V1 runner's actual logprob producer: ``Sampler.compute_logprobs``.

    This single site produces both sampled-token and prompt logprobs. Unlike the V2 site's
    gather-only contract it must return the FULL fp32 row (``gather_logprobs`` needs it for top-k
    and ranks), so it routes through ``rowinv.rowinv_full_logprobs``. Installed unconditionally; a
    failed import or a rowinv DECLINE falls back to the incumbent with bits unchanged.
    """
    try:
        from vllm.v1.sample.sampler import Sampler
    except Exception as e:
        logger.warning("[isoexec] cannot hook V1 Sampler.compute_logprobs: %s", e)
        return False
    raw = Sampler.__dict__.get("compute_logprobs")
    if isinstance(raw, staticmethod):
        raw = raw.__func__
    if raw is None:
        logger.warning("[isoexec] V1 Sampler has no compute_logprobs; hook not installed")
        return False
    if getattr(raw, "_isoexec_rowinv_full_row_hook", False):
        return True
    try:
        from skyrl.backends.skyrl_train.isoexec.ops.logprobs.rowinv import (
            rowinv_full_logprobs as _rowinv_full_logprobs,
        )
    except ImportError as e:
        logger.error("[isoexec] rowinv import failed; V1 sampler hook not installed: %r", e)
        print(
            f"[ISOEXEC-ROWINV-LOGPROB] V1 sampler hook REFUSED: rowinv import failed ({e!r}); "
            "the engine keeps the incumbent and the census stays served=0, which the engagement "
            "boundary refuses",
            flush=True,
        )
        return False

    _orig = raw

    def _isoexec_compute_logprobs(logits):
        _CL_STATE["calls"] += 1
        got = _rowinv_full_logprobs(
            logits,
            # The dtype the row arrives in, before any fp32 widening.
            src_dtype=logits.dtype,
            reference=lambda: _orig(logits),
        )
        if got is not None:
            _CL_STATE["rowinv_owned"] += 1
            _cl_report()
            return got
        _CL_STATE["incumbent"] += 1
        _cl_report()
        return _orig(logits)

    _isoexec_compute_logprobs._isoexec_rowinv_full_row_hook = True
    _isoexec_compute_logprobs._isoexec_orig = _orig
    Sampler.compute_logprobs = staticmethod(_isoexec_compute_logprobs)
    logger.info("[isoexec] hooked V1 Sampler.compute_logprobs -> rowinv full-row leaf tree")
    print(
        "[ISOEXEC-ROWINV-LOGPROB] V1 Sampler.compute_logprobs HOOKED (full-row leaf-tree "
        "denominator; sampled and prompt logprobs both flow through this site). A hook installed "
        "is not a hook engaged: judge by served>0 in the rowinv census.",
        flush=True,
    )
    return True


_TEMP_STATE = {"all_one": False, "n": -1, "seen": 0, "skipped": 0, "reported": 0}


def _ix_temp_note(all_one: bool, n: int) -> None:
    _TEMP_STATE["all_one"] = bool(all_one)
    _TEMP_STATE["n"] = int(n)


def patch_vllm_sampler_temperature() -> bool:
    """Make ``Sampler.apply_temperature`` the identity when every request temperature is exactly 1.0.

    The host-side mirror decides when to skip; the device tensor is re-checked periodically and a
    disagreement raises rather than silently sampling from untempered logits.
    """
    if os.environ.get("SKYRL_ISOEXEC_SAMPLER_TEMP_SKIP", "1").lower() in ("", "0", "false", "no"):
        print("[ISOEXEC-SAMPLER] temperature-divide skip OFF (SKYRL_ISOEXEC_SAMPLER_TEMP_SKIP)", flush=True)
        return False
    try:
        import numpy as np
        import torch
        from vllm.v1.sample.sampler import Sampler
        from vllm.v1.worker.gpu_input_batch import InputBatch
    except Exception as e:
        logger.warning("[isoexec] cannot patch vLLM sampler temperature: %s", e)
        return False
    if getattr(Sampler, "_isoexec_temp_patched", False):
        return True

    _orig_meta = InputBatch._make_sampling_metadata
    _orig_apply = Sampler.apply_temperature

    def _make_sampling_metadata(self):
        md = _orig_meta(self)
        try:
            n = self.num_reqs
            cpu = self.temperature_cpu[:n]
            _ix_temp_note(bool(n > 0 and np.all(cpu == 1.0)), n)
        except Exception:
            _ix_temp_note(False, -1)
        return md

    def _apply_temperature(logits, temp, all_random):
        st = _TEMP_STATE
        st["seen"] += 1
        skip = st["all_one"] and temp is not None and temp.numel() == st["n"]
        if skip and (st["skipped"] == 0 or st["seen"] % 1024 == 0):

            if not bool(torch.all(temp == 1.0)):
                raise RuntimeError(
                    "[isoexec-sampler] temperature drift: the host mirror said every temperature is "
                    f"1.0 for {st['n']} requests and the device tensor disagrees. Refusing to skip "
                    "the temperature divide rather than silently sample from untempered logits."
                )
        if skip:
            st["skipped"] += 1
            _ix_temp_report()
            return logits
        _ix_temp_report()
        return _orig_apply(logits, temp, all_random)

    InputBatch._make_sampling_metadata = _make_sampling_metadata
    Sampler.apply_temperature = staticmethod(_apply_temperature)
    Sampler._isoexec_temp_patched = True
    logger.info("[isoexec] patched vLLM V1 Sampler.apply_temperature -> identity at temperature 1.0")
    print(
        "[ISOEXEC-SAMPLER] temperature-divide skip INSTALLED "
        "(V1 Sampler.apply_temperature is the identity when every temperature is exactly 1.0; "
        "x/1.0 is bit-preserving, so logits, sampled tokens and logprobs are unchanged)",
        flush=True,
    )
    return True


def _ix_temp_report() -> None:
    st = _TEMP_STATE
    n = st["seen"]
    if n < 1 or (n & (n - 1)) != 0 or n == st["reported"]:
        return
    st["reported"] = n
    print(
        f"[ISOEXEC-SAMPLER] pid={os.getpid()} temperature-divide skipped={st['skipped']}/{n} calls "
        f"(last batch: all_one={st['all_one']} n={st['n']})",
        flush=True,
    )


def isoexec_sampling_constraints() -> dict:
    return {"temperature": 1.0, "logprobs_mode": "raw_logprobs"}
