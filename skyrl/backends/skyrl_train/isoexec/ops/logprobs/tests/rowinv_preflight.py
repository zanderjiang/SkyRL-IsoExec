"""Pre-flight engagement smoke for the rowinv logprob: minutes, not hours into a run.

The failure this exists to catch BEFORE a launch: both processes compose rowinv, the contract
hashes match, the handshake passes -- and only one side actually EXECUTES it (trainer
served=4096/4096 against engine served=0 on every worker). Composed is not executed;
``stats()['served'] > 0`` is the ONLY evidence of engagement, and this script judges nothing else.
With rowinv the unconditional default, a structural decline is the only way one side falls off it,
and it is silent until a boundary.

WHAT IT PROVES (single GPU, one process, ~a minute):

  * the rowinv kernel compiles, admits, and SERVES on this node under this env/stack, for BOTH
    call shapes the two runtimes present: the trainer's grad-bearing chunk ([B, chunk, V], plus a
    no-grad scoring call) and the engine's sampled-rows decode shape ([N, V], no_grad) -- every
    real dispatch funnels into the one entry point exercised here;
  * per side, ``served`` moved and the script prints ``served / declined / last_decline`` so a
    decline is diagnosable from the census, not from a banner;
  * the enforcement wiring is LIVE, both directions: a serving census passes
    ``enforce.rowinv_engagement_boundary(side, require=True)``, and the negative control (census
    reset to zero) must make the same boundary REFUSE -- a preflight whose refusal path is inert
    proves nothing.

WHAT IT CANNOT PROVE (the honest floor): that the production trainer wrapper
(distributed/megatron/model_utils.py) and the vLLM hook (runtimes/vllm/vllm_patches.py) actually
route into rowinv inside REAL Megatron and vLLM worker processes -- with the flag forwarded by
both actor channels, real TP groups voting admission, and vLLM's own logprob path patched. That is
a full engine+trainer bring-up by definition (two conflicting runtime environments); the in-run
boundaries cover it instead: the trainer refuses at the first post-step weight sync
(megatron_worker.broadcast_to_inference_engines) and every engine worker at its once-per-sync
reapply seam (vllm_worker.isoexec_reapply_cached_weights), i.e. within ONE training step of a
one-sided composition, not hours.

Run (repo root, one idle GPU):
    CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=. python \
        skyrl/backends/skyrl_train/isoexec/ops/logprobs/tests/rowinv_preflight.py

Exit 0 iff every check passes AND the negative control refuses. No CUDA exits nonzero: a preflight
that cannot observe engagement must not bless a launch.
"""

# ruff: noqa: E402 -- the env pins and the CUDA gate must precede the isoexec imports by design.

import os
import time

os.environ["SKYRL_ISOEXEC"] = "1"
os.environ.setdefault("SKYRL_ISOEXEC_PIK_LEAVES", "8")
os.environ.pop("SKYRL_ISOEXEC_MANIFEST_STRICT", None)  # the negative control must run strict
os.environ.pop("SKYRL_ISOEXEC_DEBUG_TRACE", None)

import torch

BLOCKED = "PREFLIGHT-ROWINV: BLOCKED"
READY = "PREFLIGHT-ROWINV: READY"

if not torch.cuda.is_available():
    print(f"{BLOCKED} (no CUDA device: engagement cannot be observed, so nothing is proven)")
    raise SystemExit(3)

from skyrl.backends.skyrl_train.isoexec.core import enforce
from skyrl.backends.skyrl_train.isoexec.core import process_contract as pc
from skyrl.backends.skyrl_train.isoexec.core.registry_build import build_registry
from skyrl.backends.skyrl_train.isoexec.models import qwen3_5
from skyrl.backends.skyrl_train.isoexec.ops.logprobs import rowinv

DEV = "cuda"
V = 248320  # production vocabulary; V % G == 0 for G=8
G = int(os.environ["SKYRL_ISOEXEC_PIK_LEAVES"])
_failures = []


def check(name: str, ok: bool, detail: str = "") -> None:
    line = f"  {'ok  ' if ok else 'FAIL'} {name}" + (f" ({detail})" if detail else "")
    print(line, flush=True)
    if not ok:
        _failures.append(f"{name}: {detail}")


def aten_incumbent(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    valid = (t >= 0) & (t < x.shape[-1])
    safe = t.masked_fill(~valid, 0)
    m = torch.amax(x, dim=-1, keepdim=True)
    s = torch.gather(x, -1, safe.unsqueeze(-1)).squeeze(-1)
    w = (x - m).exp_()
    value = (s - m.squeeze(-1)) - w.sum(-1).log()
    return value.masked_fill(~valid, 0.0)


def call(x: torch.Tensor, t: torch.Tensor):
    return rowinv.rowinv_sampled_logprobs(
        x,
        t,
        vocab_start_index=0,
        vocab_end_index=x.shape[-1],
        group=None,
        src_dtype=torch.bfloat16,
        reference=lambda: aten_incumbent(x.detach().float(), t),
    )


t0 = time.time()
print("[rowinv-preflight] pinning the contract into this process", flush=True)
reg = build_registry(strict=True)
contract = qwen3_5.build(reg, arch="sm90", profile=qwen3_5.PROFILE)
pc._CONTRACT, pc._VIEW = contract, pc.build_contract_view(contract, reg)
for side in ("trainer", "engine"):
    cases = enforce.rowinv_selected_cases(side)
    check(f"{side} contract selects rowinv_leaftree", bool(cases), f"cases={list(cases)}")

gen = torch.Generator(device=DEV).manual_seed(20260826)


def exercise(side: str) -> dict:
    """Run the side's call shape through the one dispatch entry; return the census DELTA."""
    before = rowinv.stats()
    if side == "trainer":
        # Grad-bearing training chunk + no-grad scoring call, the two trainer grad modes.
        x = (torch.randn(2, 64, V, device=DEV, dtype=torch.float32, generator=gen) * 3).requires_grad_(True)
        t = torch.randint(0, V, (2, 64), device=DEV, generator=gen)
        out = call(x, t)
        if out is not None:
            out.sum().backward()
        with torch.no_grad():
            xs = torch.randn(1, 32, V, device=DEV, dtype=torch.float32, generator=gen)
            ts = torch.randint(0, V, (1, 32), device=DEV, generator=gen)
            call(xs, ts)
    else:
        # Engine decode shape: a handful of sampled rows, already-gathered full vocab, no grad.
        with torch.no_grad():
            for n in (1, 8):
                x = torch.randn(n, V, device=DEV, dtype=torch.float32, generator=gen)
                t = torch.randint(0, V, (n,), device=DEV, generator=gen)
                call(x, t)
    after = rowinv.stats()
    return {
        "served": after["served"] - before["served"],
        "declined": after["declined"] - before["declined"],
        "last_decline": after["decline_reason"],
    }


for side in ("trainer", "engine"):
    d = exercise(side)
    print(
        f"[rowinv-preflight] side={side} served={d['served']} declined={d['declined']} "
        f"last_decline={d['last_decline']!r}",
        flush=True,
    )
    check(
        f"{side} census moved (served > 0 is the ONLY evidence of engagement)",
        d["served"] > 0,
        f"served={d['served']} declined={d['declined']} last_decline={d['last_decline']!r}",
    )
    try:
        ok = enforce.rowinv_engagement_boundary(side, require=True)
        check(f"{side} engagement boundary passes on a serving census", ok is True)
    except RuntimeError as e:
        check(f"{side} engagement boundary passes on a serving census", False, str(e)[:200])

# NEGATIVE CONTROL: a zero census on a selecting side must REFUSE, or this preflight is inert.
enforce._reset_for_tests()
pc._CONTRACT, pc._VIEW = contract, pc.build_contract_view(contract, reg)  # keep rowinv selected
rowinv._reset_for_test()
try:
    enforce.rowinv_engagement_boundary("engine", require=True)
    check("negative control: zero census REFUSES", False, "boundary returned instead of raising")
except RuntimeError as e:
    check("negative control: zero census REFUSES", "NEVER SERVED" in str(e), str(e)[:160])

print(f"[rowinv-preflight] done in {time.time() - t0:.1f}s", flush=True)
if _failures:
    print(f"{BLOCKED} ({len(_failures)} failure(s)):")
    for f in _failures:
        print(f"  - {f}")
    raise SystemExit(1)
print(READY + " (single-process proof only -- see the docstring for what a full bring-up still owes)")
