"""Architecture profiles: Hopper (sm90) and Blackwell (sm100).

The library rests on the claim that inside a non-split-K GEMM the fp32 accumulation order along K is sequential
over MMA k-steps and unaffected by tile sizes, warps, stages, or grid order. That claim is about the tensor-core
instruction, which is architecture-specific (Hopper wgmma k-step 16, Blackwell tcgen05 k-step 32), so bits
differ across architectures: the trainer and the rollout engine must run on the same GPU architecture, and the
arch tag is part of the contract. The kernels themselves are arch-neutral -- only the autotune space and the
verification status recorded here differ.
"""

from __future__ import annotations

import functools

import torch

# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
_HOPPER_CONFIGS = [
    (128, 256, 64, 8, 3),
    (128, 128, 64, 8, 4),
    (128, 128, 64, 4, 4),
    (256, 128, 64, 8, 3),
    (128, 64, 64, 4, 4),
    (64, 128, 64, 4, 4),
    (64, 64, 64, 4, 4),
    (64, 256, 64, 8, 3),
    (32, 128, 64, 4, 4),
    (16, 128, 128, 4, 4),
    (16, 64, 128, 4, 4),
]

_BLACKWELL_CONFIGS = [
    (128, 256, 64, 8, 3),
    (128, 256, 64, 8, 4),
    (128, 128, 64, 8, 4),
    (128, 128, 128, 8, 3),
    (256, 128, 64, 8, 3),
    (128, 64, 64, 4, 4),
    (64, 128, 64, 4, 4),
    (64, 64, 64, 4, 4),
    (64, 256, 64, 8, 3),
    (32, 128, 128, 4, 4),
    (32, 256, 64, 4, 4),
    (16, 128, 256, 4, 4),
    (16, 256, 128, 4, 4),
    (16, 64, 256, 4, 4),
]


class ArchProfile:
    def __init__(self, name, sm, mma, k_step, configs, verified):
        self.name = name
        self.sm = sm
        self.mma = mma
        self.k_step = k_step
        self.configs = configs
        self.verified = verified  # has the arch self-check been run and passed here?

    def __repr__(self):
        v = "VERIFIED" if self.verified else "UNVERIFIED -- run `python -m pik.arch --verify`"
        return f"<ArchProfile {self.name} sm{self.sm} {self.mma} k={self.k_step} [{v}]>"


# verified=True: `python -m pik.arch --verify` has passed here; re-run it if the GPU or toolchain changes.
HOPPER = ArchProfile("hopper", 90, "wgmma.mma_async", 16, _HOPPER_CONFIGS, verified=True)
BLACKWELL = ArchProfile("blackwell", 100, "tcgen05.mma", 32, _BLACKWELL_CONFIGS, verified=True)


@functools.lru_cache(maxsize=None)
def current(device=None) -> ArchProfile:
    major, _ = torch.cuda.get_device_capability(device)
    if major >= 10:
        return BLACKWELL
    if major == 9:
        return HOPPER
    raise RuntimeError(
        f"pik supports Hopper (sm90) and Blackwell (sm100+); got sm{major}0. "
        "The library will probably work -- the design is arch-neutral -- but the "
        "premise it rests on has not been verified there. Run "
        "`python -m pik.arch --verify` and, if it passes, add a profile here."
    )


def arch_tag(device=None) -> str:
    """Goes in the contract: a checkpoint made deterministic on one arch is not deterministic against another."""
    a = current(device)
    return f"{a.name}/sm{a.sm}/{a.mma}/k{a.k_step}"


def _verify(device=None) -> bool:
    """Check on this GPU that no tiling knob moves the bits of a non-split-K K-reduction."""
    import itertools

    import triton
    import triton.language as tl

    @triton.jit
    def gemm(A, B, C, M, N, K, sam, sak, sbn, sbk, scm, scn, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid = tl.program_id(0)
        npn = tl.cdiv(N, BN)
        pid_m, pid_n = pid // npn, pid % npn
        om = (pid_m * BM + tl.arange(0, BM)) % M
        on = (pid_n * BN + tl.arange(0, BN)) % N
        ok = tl.arange(0, BK)
        ap = A + om[:, None] * sam + ok[None, :] * sak
        bp = B + on[:, None] * sbn + ok[None, :] * sbk
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(K // BK):
            acc = tl.dot(tl.load(ap), tl.trans(tl.load(bp)), acc)
            ap += BK * sak
            bp += BK * sbk
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        tl.store(C + rm[:, None] * scm + rn[None, :] * scn, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))

    dev = device or "cuda"
    torch.manual_seed(0)
    M, K, N = 512, 4096, 512
    # Wide exponent spread, so any reordering shows up loudly.
    a = (torch.randn(M, K, device=dev) * torch.exp(torch.randn(M, K, device=dev) * 4)).bfloat16()
    b = (torch.randn(N, K, device=dev) * torch.exp(torch.randn(N, K, device=dev) * 4)).bfloat16()

    sigs, n = set(), 0
    for BM, BN, BK, w, s in itertools.product([64, 128], [64, 128], [32, 64, 128], [4, 8], [2, 3]):
        c = torch.empty(M, N, device=dev, dtype=torch.float32)
        try:
            gemm[(triton.cdiv(M, BM) * triton.cdiv(N, BN),)](
                a,
                b,
                c,
                M,
                N,
                K,
                a.stride(0),
                1,
                b.stride(0),
                1,
                c.stride(0),
                1,
                BM=BM,
                BN=BN,
                BK=BK,
                num_warps=w,
                num_stages=s,
            )
        except Exception:
            continue
        sigs.add(hash(c.view(torch.int32).cpu().numpy().tobytes()))
        n += 1

    a_ = current(device)
    ok = len(sigs) == 1
    print(f"arch     : {a_}")
    print(f"configs  : {n}")
    print(f"distinct : {len(sigs)}  (must be 1)")
    print()
    if ok:
        print("PASS -- no tiling knob moves the bits on this GPU. The premise holds:")
        print("        pin only the reduction plan, and tune everything else freely.")
    else:
        print("FAIL -- tiling DOES perturb the K-reduction on this GPU.")
        print("        Do not trust pik here until this is understood. The autotuner")
        print("        would silently change results.")
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if _verify() else 1)
