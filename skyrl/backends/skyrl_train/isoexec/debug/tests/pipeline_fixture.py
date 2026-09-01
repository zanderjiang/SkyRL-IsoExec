"""Two-sided synthetic pipeline for comparator tests.

Runs the six-region causal chain in ``REGION_ORDER`` (close to reverse-alphabetical) through the
real :func:`trace.wrap_region` once per side, with at most one fault injected on the engine side.
Faults mutate region output in place, so contamination propagates downstream.
"""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Optional, Sequence

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[6]))

from skyrl.backends.skyrl_train.isoexec.debug import trace  # noqa: E402

REGION_ORDER = (
    "norms.rms",
    "mm.matmul",
    "gdn.core",
    "moe.router",
    "moe.combine",
    "collectives.row_parallel_ar",
)
_ENV_KEYS = (
    trace.ENV_TRACE,
    trace.ENV_SIDE,
    trace.ENV_SAMPLE,
    trace.ENV_LADDER,
    trace.ENV_SEGMENTS,
    trace.ENV_RING,
    "RANK",
    "LOCAL_RANK",
)


class _Pipeline:
    """One side, one rank. Weights are seed-fixed so both sides compute identically."""

    def __init__(self, side: str, fault: Optional[dict], tokens: int, hidden: int, experts: int, rank: int):
        self.side, self.fault, self.rank = side, fault if side == "engine" else None, rank
        g = torch.Generator().manual_seed(1234)
        self.w = {n: (torch.randn(hidden, generator=g) * 0.05).to(torch.bfloat16) for n in REGION_ORDER}
        self.wr = (torch.randn(hidden, experts, generator=g) * 0.02).to(torch.float32)
        self.tokens = tokens
        self.reg = {n: self._wrap(n) for n in REGION_ORDER}

    # -- fault plumbing --------------------------------------------------------------------
    def _hits(self, region: str, layer: int, step: int, kind: Optional[str] = None) -> bool:
        f = self.fault
        return bool(
            f
            and f["region"] == region
            and f.get("layer", layer) == layer
            and f.get("step", step) == step
            and (kind is None or f["kind"] == kind)
        )

    def _apply(self, region: str, layer: int, step: int, out):
        f = self.fault
        if not self._hits(region, layer, step):
            return out
        kind = f["kind"]
        t = out
        if kind in ("reduce_order", "fp32_cast", "drop_record", "drop_region"):
            return out  # handled at the call site
        if kind == "ulp":
            flat = t.reshape(-1)
            iv = flat.view(torch.int16 if t.dtype == torch.bfloat16 else torch.int32)
            iv[self._pos(flat, f)] ^= 1
        elif kind == "add":
            flat = t.reshape(-1)
            p = self._pos(flat, f)
            flat[p] = (flat[p].float() + f["delta"]).to(t.dtype)
        elif kind == "nan":
            t.reshape(-1)[self._pos(t.reshape(-1), f)] = float("nan")
        elif kind == "neg_zero":
            t.reshape(-1)[7] = -0.0
        elif kind == "ftz":
            t.reshape(-1)[8] = 0.0
        elif kind == "dtype":
            return t.float()
        elif kind == "permute_rows":
            idx = torch.arange(t.shape[0])
            idx[0], idx[1] = 1, 0
            return t[idx].contiguous()
        elif kind == "pad_row":
            return torch.cat([t, t[:1] * 0], dim=0).contiguous()
        else:
            raise ValueError(kind)
        return out

    @staticmethod
    def _pos(flat, f) -> int:
        return {"first": 0, "last": flat.numel() - 1, "middle": flat.numel() // 2}[f.get("where", "middle")]

    # -- regions ---------------------------------------------------------------------------
    def _impl(self, region: str):
        def norms_rms(x, layer, step):
            v = x.float()
            y = (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w[region].float()).to(torch.bfloat16)
            fl = y.reshape(-1)
            fl[7] = 0.0  # a +0.0 and a subnormal, so neg_zero / ftz faults have a baseline
            fl[8] = torch.tensor([1], dtype=torch.int16).view(torch.bfloat16)[0]
            return y

        def mm_matmul(x, layer, step):
            if self._hits(region, layer, step, "fp32_cast"):
                return (x.float() * self.w[region].float() + x.float().roll(1, -1) * 0.5).to(torch.bfloat16)
            half = torch.tensor(0.5, dtype=torch.bfloat16)
            return (x * self.w[region] + x.roll(1, -1) * half).to(torch.bfloat16)

        def gdn_core(x, layer, step):
            return (x * torch.sigmoid(x.float()).to(torch.bfloat16) + self.w[region]).to(torch.bfloat16)

        def moe_router(x, layer, step):
            return torch.softmax(x.float() @ self.wr, dim=-1)

        def moe_combine(x, probs, layer, step):
            scale = probs.max(-1).values.unsqueeze(-1).to(torch.bfloat16)
            return (x * scale + self.w[region]).to(torch.bfloat16)

        def collectives(x, layer, step):
            shards = [x * torch.tensor(c, dtype=torch.bfloat16) for c in (0.1, 0.2, 0.30000001, 0.4)]
            if self._hits(region, layer, step, "reduce_order"):
                shards = shards[::-1]
            acc = shards[0]
            for s in shards[1:]:
                acc = acc + s
            return (acc + self.w[region]).to(torch.bfloat16)

        return {
            "norms.rms": norms_rms,
            "mm.matmul": mm_matmul,
            "gdn.core": gdn_core,
            "moe.router": moe_router,
            "moe.combine": moe_combine,
            "collectives.row_parallel_ar": collectives,
        }[region]

    def _wrap(self, region: str):
        fn = self._impl(region)

        def impl(*a, layer=0, step=0):
            return self._apply(region, layer, step, fn(*a, layer=layer, step=step))

        impl.__name__ = region.replace(".", "_")
        return trace.wrap_region(region, impl, layer_fn=lambda a, k: k.get("layer"))

    def _call(self, region: str, *a, layer: int, step: int):
        """Route through the traced wrapper unless this call is the one that never fires."""
        f = self.fault
        if f and f["kind"] in ("drop_record", "drop_region") and f["region"] == region:
            dropped = f["kind"] == "drop_region" or (f.get("layer", layer) == layer and f.get("step", step) == step)
            if dropped:
                return self._impl(region)(*a, layer=layer, step=step)
        return self.reg[region](*a, layer=layer, step=step)

    def run(self, steps: int, layers: int, hidden: int, use_set_step: bool = True) -> None:
        base = (torch.randn(self.tokens, hidden, generator=torch.Generator().manual_seed(99)) * 0.7).to(torch.bfloat16)
        for step in range(steps):
            if use_set_step:
                trace.set_step(step)
            h = (base.float() + 0.01 * step + 0.5 * self.rank).to(torch.bfloat16)
            for layer in range(layers):
                for region in REGION_ORDER:
                    if region == "moe.router":
                        p = self._call(region, h, layer=layer, step=step)
                    elif region == "moe.combine":
                        h = self._call(region, h, p, layer=layer, step=step)
                    else:
                        h = self._call(region, h, layer=layer, step=step)
                if h.shape[0] != self.tokens:  # pad_row fault: renormalize the chain length
                    h = h[: self.tokens]
        trace.flush()


def run_side(
    out_dir: str,
    side: str,
    *,
    fault: Optional[dict] = None,
    ranks: Sequence[int] = (0,),
    steps: int = 3,
    layers: int = 2,
    tokens: int = 64,
    hidden: int = 32,
    experts: int = 8,
    env: Optional[dict] = None,
    use_set_step: bool = True,
) -> str:
    """Write one side's trace (one file per rank) into ``out_dir``; returns the directory."""
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    try:
        for rank in ranks:
            for k in _ENV_KEYS:
                os.environ.pop(k, None)
            os.environ.update(env or {})
            os.environ[trace.ENV_TRACE] = out_dir
            os.environ[trace.ENV_SIDE] = side
            os.environ["RANK"] = str(rank)
            trace._reset_for_tests()
            with torch.no_grad():
                _Pipeline(side, fault, tokens, hidden, experts, rank).run(steps, layers, hidden, use_set_step)
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        trace._reset_for_tests()
    return out_dir


def run_pair(
    base: str,
    *,
    fault: Optional[dict] = None,
    ranks_a=(0,),
    ranks_b=(0,),
    env_a=None,
    env_b=None,
    set_step_a: bool = True,
    set_step_b: bool = True,
    **kw,
):
    """Trainer (clean) and engine (faulted) traces under ``base``; returns (dir_a, dir_b).

    ``set_step_b=False`` wires set_step on the trainer only, so with SAMPLE>1 the sides sample by
    different keys.
    """
    da, db = os.path.join(base, "trainer"), os.path.join(base, "engine")
    run_side(da, "trainer", fault=None, ranks=ranks_a, env=env_a, use_set_step=set_step_a, **kw)
    run_side(db, "engine", fault=fault, ranks=ranks_b, env=env_b, use_set_step=set_step_b, **kw)
    return da, db
