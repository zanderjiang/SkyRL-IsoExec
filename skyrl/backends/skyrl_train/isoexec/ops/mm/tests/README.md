# mm/tests

No `bitwise_equal_to` claims; both impls are certified against invariance contracts (1 GPU, run as files):
- `mm:cublaslt_pinned` (M-invariance, cross-bucket bit-identity, subnormals/signed-zero probes) -> `mm_cublaslt_battery_gpu.py`
- `mm:triton_batch_invariant` tile plumbing (flag reaches the kernel, bitwise-neutral) -> `mm_tiles_plumbing_gpu.py`
- forward-only scoping for bmm (forward bits untouched, backward on cuBLAS) -> `mm_fwd_only_bmm_gpu.py`
