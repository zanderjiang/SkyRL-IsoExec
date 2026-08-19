"""vLLM monkey-patches for bitwise-matched rollouts.

Covers the batch-invariance env pins, flash-attention ``num_splits=1``, an aten-equivalent logprob
kernel, and skipping the sampler's temperature divide at temperature 1.0. Every patch is idempotent
and fails to stock: if the vLLM surface it targets is missing it logs and leaves vLLM unchanged.
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

    Must be installed before ``init_batch_invariance`` runs in the worker, i.e. from the general
    plugin rather than from the engine actor. Gated on ``SKYRL_ISOEXEC_ENGINE_NCCL_UNPIN=1``.
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
    """Replace vLLM's fused-Triton logprob kernel with one that matches the trainer bitwise.

    An optional Triton fast path is bitwise self-checked against the aten reference on first use and
    permanently disabled if it ever disagrees.
    """
    try:
        import torch
        import vllm.v1.worker.gpu.sample.logprob as _lp
    except Exception as e:
        logger.warning("[isoexec] cannot patch vLLM logprob kernel: %s", e)
        return False
    if getattr(_lp, "_isoexec_logprob_patched", False):
        return True

    def _reference_compute_token_logprobs(logits, token_ids):

        token_ids = token_ids.to(torch.int64)
        x = logits.to(torch.float32)
        x = x - torch.amax(x, dim=-1, keepdim=True)
        lse = x.exp().sum(-1, keepdim=True).float().log()
        logprobs = x - lse
        return logprobs.gather(-1, token_ids)

    _fastpath_enabled = os.environ.get("SKYRL_ISOEXEC_LOGPROBS_FASTPATH", "1") == "1"
    _fastpath = {"kernel": None, "verified": False, "disabled": not _fastpath_enabled}
    try:
        import triton
        import triton.language as tl
        from triton.language.extra import libdevice

        @triton.jit
        def _isoexec_fused_exp_kernel(e_ptr, logits_ptr, amax_ptr, stride, vocab, BLOCK: tl.constexpr):
            row = tl.program_id(0)
            col_blk = tl.program_id(1)
            mx = tl.load(amax_ptr + row)
            off = col_blk * BLOCK + tl.arange(0, BLOCK)
            m = off < vocab
            v = tl.load(logits_ptr + row * stride + off, mask=m, other=0.0).to(tl.float32)
            e = libdevice.exp(v - mx)
            tl.store(e_ptr + row * stride + off, e, mask=m)

        _fastpath["kernel"] = _isoexec_fused_exp_kernel
    except Exception as _te:
        _fastpath["disabled"] = True
        logger.warning("[isoexec] logprobs fast path unavailable (no triton/libdevice): %s", _te)

    def _fast_compute_token_logprobs(logits, token_ids):
        token_ids = token_ids.to(torch.int64)
        logits = logits.to(torch.float32)
        amax = torch.amax(logits, dim=-1, keepdim=True)
        e = torch.empty_like(logits)
        grid = (logits.shape[0], triton.cdiv(logits.shape[1], 2048))
        _fastpath["kernel"][grid](e, logits, amax, logits.stride(0), logits.shape[1], BLOCK=2048)
        lse = e.sum(-1, keepdim=True).float().log()
        return logits.gather(-1, token_ids) - amax - lse

    def _batch_invariant_compute_token_logprobs(logits, token_ids):
        use_fast = (
            not _fastpath["disabled"]
            and _fastpath["kernel"] is not None
            and logits.is_cuda
            and logits.dim() == 2
            and logits.shape[0] > 16
            and logits.stride(1) == 1
        )
        if not use_fast:
            return _reference_compute_token_logprobs(logits, token_ids)
        if not _fastpath["verified"]:

            if bool(torch.isfinite(logits).all()):
                ref = _reference_compute_token_logprobs(logits, token_ids)
                fast = _fast_compute_token_logprobs(logits, token_ids)
                if torch.equal(ref, fast):
                    _fastpath["verified"] = True
                    print("[ISOEXEC-ENGINE] logprobs fast path self-check PASS (bitwise)", flush=True)
                    return fast
                _fastpath["disabled"] = True
                logger.error(
                    "[isoexec] logprobs fast path FAILED bitwise self-check "
                    "(libdevice.exp != aten.exp on this stack?) -- permanently using reference"
                )
                print("[ISOEXEC-ENGINE] logprobs fast path self-check FAIL -> reference", flush=True)
                return ref
            return _reference_compute_token_logprobs(logits, token_ids)
        return _fast_compute_token_logprobs(logits, token_ids)

    _lp.compute_token_logprobs = _batch_invariant_compute_token_logprobs
    _lp._isoexec_logprob_patched = True
    logger.info("[isoexec] patched vLLM compute_token_logprobs -> aten log_softmax (== trainer)")
    print(
        "[ISOEXEC-ENGINE] patched vLLM compute_token_logprobs -> aten log_softmax (== trainer) "
        "for bitwise rollout logprobs",
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
