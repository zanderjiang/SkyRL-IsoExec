# moe/tests

Claim -> test:
- `moe.dispatch:chunk_sort_gather` `bitwise_equal_to: megatron_cat_chunk_loop` (external baseline) -> `test_chunk_sort_bitwise.py` (CPU) + `moe_chunk_sort_battery_gpu.py` (1 GPU admission battery)
- `moe.experts:fused` `bitwise_equal_to: batched_bmm` -> `moe_fused_vs_bmm_gpu.py` (1 GPU, VLLM_BATCH_INVARIANT=1)
- `moe.router:fused_o2` `bitwise_equal_to: deterministic` -> `moe_router_fused_o2_gpu.py` (1 GPU, VLLM_BATCH_INVARIANT=1)
- `moe.experts:grouped_cublaslt` `bitwise_equal_to: batched_bmm` -> NO TEST HERE: the impl module and its battery remain in the private repo.

CPU envelopes (pytest): permute/dispatch `test_fused_permute_cpu.py`, `test_dense_scatter_install_cpu.py`;
combine `test_combine_fold_cpu.py`, `test_leaftree_wire_cpu.py`, `test_oc_*.py`, `test_owner_*.py`;
router `test_router_chain_cpu.py`, `test_router_cast_cache_cpu.py`; experts/backward
`test_batched_experts_branches_cpu.py`, `test_fastbwd_storage_cpu.py`, `test_invariant_tax_engagement_cpu.py`,
`test_fc2_ingemm_geometry_cpu.py`; wires `test_a2a_wire_cpu.py`, `test_flat_stage_bitwise.py`.
