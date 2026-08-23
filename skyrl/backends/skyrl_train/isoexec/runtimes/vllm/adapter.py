"""Engine-side ContractAdapter: runtime facts read off the vLLM/mpu deployment exactly as the
gptmodel_vllm call site computed them; ``install()`` wraps the existing install sequence."""

from __future__ import annotations

from ...core.adapter import ContractAdapter


class VLLMContractAdapter(ContractAdapter):
    def __init__(self, model_path, *, vllm_config, mp, tp_size, install_fn):
        super().__init__("engine", model_path)
        self._vllm_config = vllm_config
        self._mp = mp
        self._tp_size = int(tp_size)
        self._install_fn = install_fn

    def runtime_facts(self) -> dict:
        # SP/CP: this adapter forces mp.sequence_parallel=False and initializes mpu with TP only,
        # so both are structurally their read values, not guesses.
        from megatron.core import parallel_state as mpu

        from ...core.arch import ARCH

        parallel = getattr(self._vllm_config, "parallel_config", None)
        return {
            "TP": self._tp_size,
            "PP": int(getattr(parallel, "pipeline_parallel_size", 1) or 1),
            "SP": 1 if getattr(self._mp, "sequence_parallel", False) else 0,
            "CP": mpu.get_context_parallel_world_size() if mpu.model_parallel_is_initialized() else 1,
            "world": int(getattr(parallel, "world_size", 0) or 0),
            "arch": ARCH,
        }

    def install(self) -> None:
        self._install_fn()

    def on_weight_sync(self, peer_hash_or_stamp=None) -> bool:
        """Receiver half of the handshake: ``peer_hash_or_stamp`` is the trainer-stamped init_info."""
        from ...core.enforce import weight_sync_boundary
        from ...core.process_contract import assert_init_info_contract

        assert_init_info_contract(peer_hash_or_stamp, other_side="trainer")
        return weight_sync_boundary(self.side)
