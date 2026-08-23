"""Trainer-side ContractAdapter: runtime facts from megatron_config + the trainer SP env exactly as
the megatron_worker call site computed them; ``install()`` wraps the existing init_model install
path. Contract build stays fail-soft (the pre-adapter trainer behavior); claim violations and the
INSTALL close refuse as before."""

from __future__ import annotations

import os

from ...core.adapter import ContractAdapter


class MegatronContractAdapter(ContractAdapter):
    build_failsoft = True

    def __init__(self, model_path, *, megatron_config, install_fn, world_size=None):
        super().__init__("trainer", model_path)
        self._megatron_config = megatron_config
        self._install_fn = install_fn
        self._world_size = world_size

    def runtime_facts(self) -> dict:
        from ...core.arch import ARCH

        m = self._megatron_config
        return {
            "TP": int(m.tensor_model_parallel_size),
            "PP": int(getattr(m, "pipeline_model_parallel_size", 1) or 1),
            "CP": int(getattr(m, "context_parallel_size", 1) or 1),
            "SP": 1 if os.environ.get("SKYRL_ISOEXEC_TRAINER_SP", "0") == "1" else 0,
            "world": self._world_size,
            "arch": ARCH,
        }

    def install(self) -> None:
        self._install_fn()

    def on_weight_sync(self, peer_hash_or_stamp=None) -> bool:
        """Sender half: ``peer_hash_or_stamp`` is the composite this trainer stamped on init_info
        (None when no local contract exists). The stamp IS the trainer's half of the handshake."""
        from ...core import enforce

        if peer_hash_or_stamp is not None:
            enforce.report(
                "handshake:numerical_policy",
                enforce.WEIGHT_SYNC,
                enforce.OK,
                f"stamped composite={peer_hash_or_stamp}",
            )
        else:
            enforce.report(
                "handshake:numerical_policy",
                enforce.WEIGHT_SYNC,
                enforce.SKIPPED,
                "no local contract; stamped nothing",
            )
        return enforce.weight_sync_boundary(self.side)
