"""Checks the rollout-only fused MoE forward is bitwise the production batched-bmm forward, and
that both are batch-invariant, at the real Qwen3.5-35B-A3B ETP=8 shard dims.

Run: CUDA_VISIBLE_DEVICES=<gpu> VLLM_BATCH_INVARIANT=1 uv run --isolated --extra isoexec \
       python skyrl/backends/skyrl_train/isoexec/ops/moe/tests/moe_fused_vs_bmm_gpu.py
"""

import os
import sys
from types import SimpleNamespace

import torch

if not torch.cuda.is_available():  # needs one CUDA device
    print("SKIP: no CUDA device")
    raise SystemExit(0)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 7)))  # repo root

# Qwen3.5-35B-A3B at ETP=8: 256 experts per rank, hidden 2048, moe_intermediate 512 sharded to 64
E, H, INTER = 256, 2048, 64
DTYPE = torch.bfloat16
dev = "cuda"


def _fake_sequential_mlp(w1_param, w2_param):
    """A stand-in ``SequentialMLP`` exposing only what ``_batched_experts_forward`` reads."""
    cfg = SimpleNamespace(
        activation_func=torch.nn.functional.silu,
        activation_func_clamp_value=None,
        glu_linear_offset=0.0,
        fp8=False,
        bias_activation_fusion=False,
        gated_linear_unit=True,
        add_bias_linear=False,
    )
    experts = [
        SimpleNamespace(
            linear_fc1=SimpleNamespace(weight=w1_param[e]),  # [2f, h]
            linear_fc2=SimpleNamespace(weight=w2_param[e]),  # [h, f]
        )
        for e in range(w1_param.shape[0])
    ]
    return SimpleNamespace(config=cfg, num_local_experts=w1_param.shape[0], local_experts=experts)


def _make(counts, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    T = int(counts.sum())
    x = torch.randn(T, H, dtype=DTYPE, device=dev, generator=g) * 0.5
    probs = torch.rand(T, dtype=torch.float32, device=dev, generator=g)  # router probs are fp32
    return x, probs


def main():
    if os.environ.get("VLLM_BATCH_INVARIANT") != "1":
        print("!! VLLM_BATCH_INVARIANT is not 1 -- both paths are batch-VARIANT without it", flush=True)
        return 1

    # batch-invariant mode, as the engine and trainer both install: without it torch.bmm's
    # accumulation order is token-count-dependent
    from vllm.model_executor.layers.batch_invariant import enable_batch_invariant_mode

    enable_batch_invariant_mode()

    from skyrl.backends.skyrl_train.isoexec.ops.moe.moe_batched_experts import (
        _batched_experts_forward,
    )
    from skyrl.backends.skyrl_train.isoexec.ops.moe.moe_fused_experts import (
        _fused_forward,
    )

    torch.manual_seed(0)
    # megatron param layout: linear_fc1.weight [2f, h], linear_fc2.weight [h, f]
    w1_param = (torch.randn(E, 2 * INTER, H, dtype=DTYPE, device=dev) * 0.02).contiguous()
    w2_param = (torch.randn(E, H, INTER, dtype=DTYPE, device=dev) * 0.02).contiguous()
    self_mlp = _fake_sequential_mlp(w1_param, w2_param)

    def production(x, probs, counts):
        out, _ = _batched_experts_forward(self_mlp, x, counts, probs)
        return out

    def fused(x, probs, counts):
        # what the SequentialMLP hook builds: stack the params, no transpose
        w1 = torch.stack([e.linear_fc1.weight for e in self_mlp.local_experts])  # [E, 2f, h]
        w2 = torch.stack([e.linear_fc2.weight for e in self_mlp.local_experts])  # [E, h, f]
        return _fused_forward(x, w1, w2, probs, counts, E)

    ok = 1

    def report(name, a, b):
        nonlocal ok
        exact = int((a == b).all().item())
        diff = (a.float() - b.float()).abs().max().item()
        same = int((a == b).sum().item())
        print(f"  {name:<46} exact={same}/{a.numel()}  max|diff|={diff:.3e}  {'OK' if exact else 'FAIL'}")
        ok &= exact

    print(
        f"[fused-vs-bmm] E={E} H={H} inter={INTER}  VLLM_BATCH_INVARIANT=" f"{os.environ.get('VLLM_BATCH_INVARIANT')}\n"
    )

    # 1. bitwise, over a range of routings (balanced, skewed, sparse)
    print("1. fused forward == production bmm forward, BITWISE:")
    routings = {
        "balanced T=1024": torch.full((E,), 4, device=dev),
        "balanced T=4096": torch.full((E,), 16, device=dev),
        "skewed (expert 0 heavy)": torch.tensor([300] + [2] * (E - 1), device=dev),
        "sparse (8 experts hot)": torch.tensor([8] * 8 + [0] * (E - 8), device=dev),
    }
    for name, counts in routings.items():
        counts = counts.long()
        x, probs = _make(counts, seed=int(counts.sum()))
        report(name, fused(x, probs, counts), production(x, probs, counts))

    # 2. batch invariance: hold expert 0's rows fixed, vary the other experts' loads, check the
    #    probe rows do not move on either path
    print("\n2. batch invariance (expert-0 rows fixed, other experts' load 0 -> 4000):")
    x0, p0 = _make(torch.tensor([64] + [0] * (E - 1), device=dev).long(), seed=99)
    x0, p0 = x0[:64], p0[:64]
    ref_f = ref_p = None
    for other in (0, 128, 1000, 4000):
        g = torch.Generator(device=dev).manual_seed(other + 5)
        counts = torch.zeros(E, dtype=torch.long, device=dev)
        counts[0] = 64
        if other:
            rest = torch.randint(0, 2 * other // (E - 1) + 1, (E - 1,), device=dev, generator=g)
            rest = rest * other // max(1, int(rest.sum()))
            rest[0] += other - int(rest.sum())
            counts[1:] = rest.clamp(min=0)
        T = int(counts.sum())
        xr = torch.randn(T, H, dtype=DTYPE, device=dev, generator=g) * 0.5
        pr = torch.rand(T, dtype=torch.float32, device=dev, generator=g)
        xr[:64], pr[:64] = x0, p0
        rows_f = fused(xr, pr, counts)[:64]
        rows_p = production(xr, pr, counts)[:64]
        if ref_f is None:
            ref_f, ref_p = rows_f, rows_p
            print(f"  probe alone (T={T})                          reference (fused & bmm)")
            continue
        report(f"fused: probe with {other:5d} other tokens", ref_f, rows_f)
        report(f"bmm:   probe with {other:5d} other tokens", ref_p, rows_p)

    print(
        "\n"
        + (
            "PASS: fused forward is BITWISE the production bmm forward, both batch-invariant "
            "-> rollout-only fused MoE preserves zero-KL"
            if ok
            else "FAIL: fused != bmm bitwise -- align _fused_forward's epilogue to the bmm path"
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
