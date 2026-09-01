"""Pre-flight engagement smoke for the rowinv logprob: single GPU, one process, ~a minute.

Checks that rowinv compiles, admits and SERVES both runtimes' call shapes on this node, and that
the engagement boundary refuses on a zero census. Exit 0 iff every check passes.

Run (repo root, one idle GPU):
    CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=. python \
        skyrl/backends/skyrl_train/isoexec/ops/logprobs/tests/rowinv_preflight.py
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
    """Run the side's call shape through the one dispatch entry; return the census delta."""
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

# Negative control: a zero census on a selecting side must refuse, or this preflight is inert.
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
