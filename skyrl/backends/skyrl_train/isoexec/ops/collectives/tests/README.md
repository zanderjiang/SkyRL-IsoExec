# collectives/tests

No `bitwise_equal_to` claims in this family; these pin the pik tree all-reduce / row-parallel
schedule and its fail-closed envelopes (CPU, pytest):
- `collectives.tree_all_reduce:pik_tree` fold order -> `test_tree_reduce_scatter_cpu.py`, `test_bf16_leaf_scheme_cpu.py`, `test_pik_smallm_leaf_paths.py`
- AR vectorization / branch / oneshot crossover -> `test_pik_ar_vec_cpu.py`, `test_pik_ar_branch.py`
- autotune M-bucketing (config cannot flap) -> `test_autotune_m_bucket_cpu.py`
- launcher/codegen envelopes -> `test_pik_fastlaunch_cpu.py`, `test_pik_fused_barrier_cpu.py`, `test_pik_sym_pool_key.py`, `test_pik_cublas_elf_contract.py`
- transport observability + plan-vs-contract coherence -> `test_pik_transport_observability.py`, `test_plan_manifest_coherence_cpu.py`

`collectives.nccl_pin` impl selection is proven in `core/tests/test_manifest_qwen35.py`.
Multi-rank bitwise batteries stay in the private nightly tree.
