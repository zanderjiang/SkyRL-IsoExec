# gdn/tests

Claim -> test:
- `gdn.conv:causal_conv1d_update` `equivalence_proof` (split-exactness, fn/update composition) -> `gdn_native_kernel_parity_test_gpu.py` (1 GPU; also gates `causal_conv1d_fn`, `gdn.core:native_fused_sigmoid`, `gdn.l2norm:l2norm_fwd`)

CPU envelopes (pytest): state pools `test_state_pool_cpu.py`/`test_assign_many_cpu.py`;
chunk-synced core `test_cs_*_cpu.py`, `test_meta_abstract_contracts_cpu.py`; native conv backward
`test_native_conv_backward_cpu.py`; bv64 core `test_native_core_bv64_cpu.py`; fused split/prep gates
`test_fused_split_cpu.py`, `test_matched_prep_gate_cpu.py`; meta caches `test_packed_meta_cache_cpu.py`,
`test_seq_meta_cache_cpu.py`; scoring fastpaths `test_megatron_scoring_fastpaths_cpu.py`.

`_op_census.py` is a vendored test helper. Multi-GPU / engine-level GDN batteries stay private.
