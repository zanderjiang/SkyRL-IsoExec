# FSDP Backend

## Overview

Default backend (`trainer.strategy=fsdp`). Uses PyTorch FSDP2 for distributed training.

- **FSDPConfig** in `skyrl/train/config/config.py`.
- **FSDPStrategy** in `skyrl/backends/skyrl_train/distributed/fsdp_strategy.py`.
- **FSDPWeightExtractor** for extracting weights from sharded parameters (in `skyrl/backends/skyrl_train/workers/fsdp/fsdp_worker.py`).

## CPU Offload

- `trainer.fsdp_config.cpu_offload=true` offloads optimizer states to CPU.
- Also available for reference model: `ref.fsdp_config.cpu_offload=true`.
- Useful when GPU memory is low but adds overhead.
- NOT to be confused with `offload_after_step`: This is for colocated training where training state is offloaded to CPU after a training step is complete, so that the inference workers can be loaded on the same GPUs.

## Sharding

**STALE 2026-08-12:** `FULL_SHARD` / `NO_SHARD` are FSDP**1** `ShardingStrategy` names and appear nowhere in this tree. The backend is **FSDP2** (`distributed/fsdp_strategy.py` uses `fully_shard` / `apply_fsdp2`), where the equivalent knob is `reshard_after_forward` (`config.py:139`, default `True`). There is no `NO_SHARD` world_size=1 fallback path.
- `fsdp_size`: Controls sharding group size. `-1` = auto (full world). For Hybrid Sharded Data Parallelism (HSDP), use `fsdp_size=<num_gpus_per_node>`
