# norms/tests

Claim -> test:
- `norms.rms:fused` `bitwise_equal_to: eager_zero_centered` -> `fused_outnorm_gpu.py` (1 GPU; torch.equal incl. subnormal/signed-zero probes)
- `norms.gated_out:fused` `bitwise_equal_to: eager` -> `fused_outnorm_gpu.py`
- fused residual-add + zero-centered RMSNorm (forward bits + backward) -> `fused_add_rmsnorm_gpu.py` (1 GPU)
- native RMSNorm memoization envelope -> `test_native_rmsnorm_memo_cpu.py` (CPU, pytest)
