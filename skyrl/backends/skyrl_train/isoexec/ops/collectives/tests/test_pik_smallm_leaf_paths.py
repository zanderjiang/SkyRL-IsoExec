"""CPU-side guarantees for the pik small-M leaf paths (batched leaves / fused leaf-tree).

Pins the fold algebra -- the sequential, batched and fused schedules must be the same expression
tree, bit for bit -- and the flag wiring. ``pik/plan.py`` is loaded by file path because
importing the ``pik`` package pulls Triton autotune decorators that query the GPU.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import torch

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[7]))  # repo root

from skyrl.backends.skyrl_train.isoexec.core import flags  # noqa: E402


def _plan_module():
    if "_pik_plan_cpu" in sys.modules:
        return sys.modules["_pik_plan_cpu"]
    path = _HERE.parents[1] / "pik" / "plan.py"
    spec = importlib.util.spec_from_file_location("_pik_plan_cpu", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_pik_plan_cpu"] = mod  # dataclasses resolves cls.__module__ through sys.modules
    spec.loader.exec_module(mod)
    return mod


def _adversarial_leaves(m: int, n: int = 4096, seed: int = 0) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(n, generator=g) * torch.exp(torch.randn(n, generator=g) * 12)).float() for _ in range(m)]


def test_leaf_fold_equals_pair_fold_bitwise():
    """tree(m raw leaves) == pair-sums + tree(m/2) == binary-counter walk, bit for bit."""
    plan = _plan_module()
    for m in (2, 4, 8):
        for seed in range(8):
            leaves = _adversarial_leaves(m, seed=seed)

            # schedule A: sequential fp32 -- beta=1 pair sums, then tree over pairs
            pairs = [leaves[2 * j] + leaves[2 * j + 1] for j in range(m // 2)]
            if m == 2:
                root_a = pairs[0]
            else:
                slots = list(pairs)
                for dst, lhs, rhs in plan.combine_order(m // 2):
                    slots[dst] = slots[lhs] + slots[rhs]
                root_a = slots[0]

            # schedule B: batched -- raw leaves, tree over leaves
            slots = list(leaves)
            for dst, lhs, rhs in plan.combine_order(m):
                slots[dst] = slots[lhs] + slots[rhs]
            root_b = slots[0]

            # schedule C: fused kernel's in-register fold -- binary-counter carries
            stack: dict[int, torch.Tensor] = {}
            for j, leaf in enumerate(leaves):
                acc, lvl, tz = leaf, 0, j
                while tz & 1:
                    acc = stack[lvl] + acc  # left operand carries the LOWER leaf indices
                    tz >>= 1
                    lvl += 1
                stack[lvl] = acc
            root_c = stack[(m).bit_length() - 1]

            assert torch.equal(root_a, root_b), f"pair-fold != leaf-fold at m={m} seed={seed}"
            assert torch.equal(root_b, root_c), f"leaf-fold != counter-fold at m={m} seed={seed}"


def test_bf16_leaf_rounding_point_is_the_leaf():
    """Under a bf16-leaf plan both paths round each LEAF once and add in fp32 -- same tree."""
    plan = _plan_module()
    for m in (2, 4, 8):
        leaves = _adversarial_leaves(m, seed=99)
        rounded = [leaf.to(torch.bfloat16).float() for leaf in leaves]
        slots = list(rounded)
        for dst, lhs, rhs in plan.combine_order(m):
            slots[dst] = slots[lhs] + slots[rhs]
        root_tree = slots[0]

        stack: dict[int, torch.Tensor] = {}
        for j, leaf in enumerate(leaves):
            acc = leaf.to(torch.bfloat16).float()  # round the LEAF only, then fp32 adds
            lvl, tz = 0, j
            while tz & 1:
                acc = stack[lvl] + acc
                tz >>= 1
                lvl += 1
            stack[lvl] = acc
        assert torch.equal(root_tree, stack[(m).bit_length() - 1]), f"bf16-leaf fold at m={m}"


def test_flags_registered_and_forwarded_on_both_channels():
    expected = {
        "SKYRL_ISOEXEC_PIK_BATCHED_LEAVES": "0",
        "SKYRL_ISOEXEC_PIK_BATCHED_LEAVES_MAX_M": "256",
    }
    train = set(flags.actor_forwarding_tuple(flags.TRAIN))
    engine = set(flags.actor_forwarding_tuple(flags.ENGINE))
    for name, default in expected.items():
        f = flags.get(name)
        assert f.default == default, f"{name} default drifted: {f.default!r} != {default!r}"
        assert f.disposition == flags.DEPLOYMENT, f"{name} must stay DEPLOYMENT (bitwise by admission)"
        assert name in train, f"{name} missing from the TRAIN forwarding loop (silent no-op trap)"
        assert name in engine, f"{name} missing from the ENGINE forwarding loop (silent no-op trap)"


def _run():
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f()
            print("PASS", n)
    print("\n3/3 passed")


if __name__ == "__main__":
    _run()
