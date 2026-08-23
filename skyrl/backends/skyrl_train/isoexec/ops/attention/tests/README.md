# attention/tests

Claim -> test:
- `attention.varlen:vllm_flash_ns1` `bitwise_equal_to: varlen_custom` -> `varlen_vllm_flash_ns1_gpu.py` (1 GPU; covers the `non_contiguous` hazard)
- QKV subgroup all-gather admission predicates (fail-closed envelope) -> `test_qkv_subgroup_admission.py` (CPU, pytest)
- QKV subgroup all-gather live bitwise proof -> `attn_qkv_subgroup_dist.py` (torchrun world>=2; skips elsewhere)

`attention.qwen35_context_layout:qwen35_context_layout_sm90a` was deregistered from the public
contract: its module/cubin and GPU gate remain in the private repo, so the public tree hashed an
entry nothing could install or check. `*_gpu.py`/`*_dist.py` are run as files, not collected by pytest.
