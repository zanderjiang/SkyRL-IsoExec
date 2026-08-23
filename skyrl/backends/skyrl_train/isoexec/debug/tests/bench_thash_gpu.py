"""GPU microbenchmark for the tensor digest. SKIP-GATED: runs only with
SKYRL_ISOEXEC_DEBUG_TEST_GPU=1 on an idle GPU node -- it allocates GPU memory.

    SKYRL_ISOEXEC_DEBUG_TEST_GPU=1 python .../debug/tests/bench_thash_gpu.py

Reports digest time and effective bandwidth at the production scale [tokens ~1e4, hidden 4-8k]
bf16, for the plain digest, a 4-rung ladder, and 1024-row segment digests.
"""

from __future__ import annotations

import os
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[6]))

from skyrl.backends.skyrl_train.isoexec.debug import thash  # noqa: E402


def main() -> int:
    if os.environ.get("SKYRL_ISOEXEC_DEBUG_TEST_GPU") != "1":
        print("skipped: set SKYRL_ISOEXEC_DEBUG_TEST_GPU=1 on an idle GPU node")
        return 0
    if not torch.cuda.is_available():
        print("skipped: no CUDA")
        return 0
    dev = torch.device("cuda")
    for tokens, hidden in ((10_000, 4096), (10_000, 8192)):
        t = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=dev)
        nbytes = t.numel() * t.element_size()
        for label, fn in (
            ("digest", lambda: thash.tensor_digest(t)),
            ("ladder4", lambda: thash.digest_ladder(t)),
            ("segments", lambda: thash.segment_digests(t, rows_per_segment=1024)),
        ):
            fn()  # warmup
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            iters = 10
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / iters
            print(
                f"[{tokens}x{hidden} bf16 {nbytes / 1e6:.0f}MB] {label:9s} "
                f"{dt * 1e3:8.3f} ms  ({nbytes / dt / 1e9:7.1f} GB/s effective)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
