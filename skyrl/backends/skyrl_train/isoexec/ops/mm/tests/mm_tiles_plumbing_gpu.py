"""Plumbing gate for mm_tiles: does the flag actually reach the kernel, and is it bitwise-neutral?

§0: seven flags in this program were silently unforwarded, producing A/Bs that compared a baseline
against itself. An offline unit test of the flag passes happily while the live run ignores it.
So this asserts the OVERRIDE TOOK, by counting kernel launches through the tiled launcher.
"""

import os

import torch


if not torch.cuda.is_available():  # promoted nightly battery: needs one CUDA device
    print("SKIP: no CUDA device")
    raise SystemExit(0)

os.environ["SKYRL_ISOEXEC_MM_TILES"] = "1"
from vllm.model_executor.layers import batch_invariant as bi

from skyrl.backends.skyrl_train.isoexec.ops.mm import mm_tiles
from skyrl.backends.skyrl_train.isoexec.ops.moe.moe_batch_invariant import (
    _install_moe_matmul_invariance,
)

# a deliberately NON-default table so a no-op install is detectable
mm_tiles._TILE_TABLE = {
    (2048, 128, torch.bfloat16): dict(
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, num_stages=3, num_warps=4
    ),
    (2048, 1, torch.bfloat16): dict(
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=16, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, num_stages=3, num_warps=4
    ),
    (2048, 256, torch.float32): dict(
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=8, num_stages=3, num_warps=4
    ),
}
_install_moe_matmul_invariance()


# reference BEFORE the tile install, through the real aten::mm dispatch
def bits(t):
    return t.view(torch.int16 if t.dtype == torch.bfloat16 else torch.int32)


cases = [
    (320, 2048, 128, torch.bfloat16),
    (320, 2048, 1, torch.bfloat16),
    (320, 2048, 256, torch.float32),
    (8192, 2048, 128, torch.bfloat16),
    (64, 2048, 256, torch.float32),
    (320, 2048, 777, torch.bfloat16),
]  # unlisted shape -> must fall through to stock
refs = {}
for M, K, N, dt in cases:
    g = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float32, generator=g).to(dt)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32, generator=g).to(dt)
    refs[(M, K, N, dt)] = (a, b, torch.mm(a, b).clone())

assert mm_tiles.install_mm_tiles() is True, "install_mm_tiles returned False with flag=1 and a table"
assert getattr(bi.matmul_persistent, "_isoexec_tiled", False), "OVERRIDE DID NOT TAKE"
assert mm_tiles.install_mm_tiles() is True and getattr(bi.matmul_persistent, "_isoexec_tiled", False), "not idempotent"

# count how many launches actually go through the tiled branch
hits = {"tiled": 0, "fallthrough": 0}
tiled = bi.matmul_persistent


def counting(a, b, bias=None):
    key = (a.shape[1], b.shape[1], a.dtype)
    hits["tiled" if key in mm_tiles._TILE_TABLE else "fallthrough"] += 1
    return tiled(a, b, bias=bias) if bias is not None else tiled(a, b)


bi.matmul_persistent = counting

bad = 0
for (M, K, N, dt), (a, b, ref) in refs.items():
    got = torch.mm(a, b)
    d = int((bits(ref) != bits(got)).sum())
    bad += d
    print(f"  [{M},{K}]@[{K},{N}] {str(dt).split('.')[-1]:<9} bitdiff={d}")
print(f"\n  dispatch: {hits['tiled']} launches took the TILED path, {hits['fallthrough']} fell through to stock")
assert hits["tiled"] == 5, f"expected 5 tiled launches, got {hits['tiled']} -- the override is not reaching the kernel"
assert hits["fallthrough"] == 1, "the unlisted shape did not fall through to stock"
assert bad == 0, f"{bad} bits differ -- re-tiling changed the result"

# NEGATIVE CONTROL: with the flag off, install must decline
mm_tiles._TILE_LOG_ONCE = False
os.environ["SKYRL_ISOEXEC_MM_TILES"] = "0"
assert mm_tiles.install_mm_tiles() is False, "install ran with the flag OFF -- default-off is broken"
print("  flag-off negative control: install declined  OK")
print("\nPLUMBING GATE PASS")
