# Debug-mode integration (changes owned by files this package must not touch)

Debug mode is gated by `SKYRL_ISOEXEC_DEBUG_TRACE` (trace dir). Everything below is the
integration this package specifies but does not implement, because the owning files are under a
concurrent change.

1. **Flags** (`core/flags.py`): register `SKYRL_ISOEXEC_DEBUG_TRACE`, `SKYRL_ISOEXEC_DEBUG_SIDE`,
   `SKYRL_ISOEXEC_DEBUG_SAMPLE`, `SKYRL_ISOEXEC_DEBUG_LADDER`, `SKYRL_ISOEXEC_DEBUG_RING` as
   `DIAGNOSTIC`, sides `("both",)`, forwarded by `(TRAIN, ENGINE)` — without this the envs never
   reach the Ray actors (the audited forwarding trap). `_SIDE` should be stamped per-process by
   the launcher (`trainer` in the Megatron worker path, `engine` in the vLLM path), not by hand.

2. **Hook installation**: trainer — in `workers/megatron/megatron_worker.py`, right after
   `apply_megatron_isoexec_patches()` / model build, call
   `isoexec.debug.install_debug_hooks()`. Engine — in `runtimes/vllm/gptmodel_vllm.py`, after
   `swap_gdn_core(...)`, call `install_debug_hooks()` **and** `install_gdn_layer_hooks(self.gpt)`
   (layer-indexed GDN records). Optionally call `isoexec.debug.set_step(n)` per training step /
   scoring pass for step-keyed alignment and true every-Nth-forward sampling.

3. **Non-enforcement semantics**: when `SKYRL_ISOEXEC_DEBUG_TRACE` is set, the contract
   handshake goes warn-only and the `numerical_policy` comparison becomes informational:
   `core/process_contract.py::assert_contract_agreement` and `assert_init_info_contract` treat a
   mismatch as a logged warning (same effect as `SKYRL_ISOEXEC_MANIFEST_STRICT=0`), and
   `core/contract_delivery.py`'s env-hash check likewise must not refuse. Users may then run any
   kernel mix on either side; the trace, not the gate, reports the disagreement.

4. **Execution config for debug runs**: run the engine eager (no CUDA-graph decode) — replayed
   graphs execute no Python, so captured decode steps would produce no records.
