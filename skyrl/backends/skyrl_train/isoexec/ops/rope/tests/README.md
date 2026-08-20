# rope/tests

Claim -> test:
- `rope.rope:fused` `bitwise_equal_to: eager` -> `rope_fused_gpu.py` (1 GPU; torch.equal at every
  installed shape, plus the declared `signed_zero` / `subnormals` / `non_contiguous` hazards and a
  CUDA-graph decode replay). `_ftz_check.py` is a vendored helper that fails a vacuous FTZ check.
