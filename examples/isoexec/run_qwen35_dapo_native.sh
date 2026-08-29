#!/usr/bin/env bash
# Native baseline: Qwen3.5-35B-A3B-Base DAPO on AIME, 1 node of 8xH100. The comparator arm for
# run_qwen35_dapo_isoexec.sh — identical config except no IsoExec composition and prefix caching on.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
local_root=${ISOEXEC_LOCAL_ROOT:-${HOME}/isoexec}   # venvs, wheel cache, logs; override per site
cd "${repo}"
# Optional site preamble (cluster module loads, proxies, shared caches). Absent on most machines.
if [[ -f "${local_root}/env.sh" ]]; then source "${local_root}/env.sh"; fi

MODEL="${ISOEXEC_MODEL:-Qwen/Qwen3.5-35B-A3B-Base}"
TRAIN_FILE="${ISOEXEC_TRAIN_FILE:-${HOME}/data/dapo/dapo-math-17k-cleaned.parquet}"
TEST_FILE="${ISOEXEC_TEST_FILE:-${HOME}/data/dapo/aime-2024-cleaned.parquet}"
RUN_NAME="${WANDB_RUN_NAME:-native_dapo_qwen35b_$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${local_root}/logs/${RUN_NAME}"
mkdir -p "${LOG_DIR}"
export HF_HOME="${HF_HOME:-${local_root}/hf}"
export SKYRL_LOG_FILE="${LOG_DIR}/infra.log"
unset SKYRL_ISOEXEC  # this arm must carry no IsoExec composition

uv run --isolated --extra megatron -m examples.train.algorithms.dapo.main_dapo \
  data.train_data="['${TRAIN_FILE}']" \
  data.val_data="['${TEST_FILE}']" \
  environment.env_class=aime \
  generator.apply_overlong_filtering=true \
  generator.batched=true \
  generator.n_samples_per_prompt=16 \
  generator.eval_n_samples_per_prompt=32 \
  generator.max_input_length=2048 \
  generator.sampling_params.max_generate_length=8192 \
  generator.eval_sampling_params.max_generate_length=8192 \
  generator.eval_sampling_params.temperature=1.0 \
  generator.eval_sampling_params.top_p=0.7 \
  generator.inference_engine.tensor_parallel_size=8 \
  generator.inference_engine.language_model_only=true \
  generator.inference_engine.engine_init_kwargs.gdn_prefill_backend=triton \
  generator.inference_engine.gpu_memory_utilization=0.70 \
  generator.inference_engine.max_num_batched_tokens=10240 \
  generator.inference_engine.max_num_seqs=512 \
  generator.inference_engine.override_existing_update_group=disable \
  trainer.strategy=megatron \
  trainer.policy.model.path="${MODEL}" \
  trainer.ref.model.path="${MODEL}" \
  trainer.policy.language_model_only=true \
  trainer.policy.megatron_config.tensor_model_parallel_size=4 \
  trainer.policy.megatron_config.expert_model_parallel_size=8 \
  trainer.policy.megatron_config.expert_tensor_parallel_size=1 \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=true \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=1.0 \
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=true \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=true \
  trainer.policy.optimizer_config.num_warmup_steps=40 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.placement.policy_num_gpus_per_node=8 \
  trainer.algorithm.policy_loss_type=dual_clip \
  trainer.algorithm.eps_clip_high=0.28 \
  trainer.algorithm.clip_ratio_c=10 \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.overlong_buffer_len=4096 \
  trainer.algorithm.off_policy_correction.tis_ratio_type=token \
  trainer.train_batch_size=128 \
  trainer.policy_mini_batch_size=32 \
  trainer.max_tokens_per_microbatch=20480 \
  trainer.max_prompt_length=2048 \
  trainer.epochs=20 \
  trainer.eval_interval=1000 \
  trainer.eval_before_train=false \
  trainer.ckpt_interval=-1 \
  trainer.max_ckpts_to_keep=3 \
  trainer.hf_save_interval=300 \
  trainer.resume_mode=null \
  trainer.project_name="${WANDB_PROJECT:-qwen3.5-35b-dapo}" \
  trainer.run_name="${RUN_NAME}" \
  trainer.ckpt_path="${local_root}/ckpts/${RUN_NAME}" \
  trainer.export_path="${local_root}/exports/${RUN_NAME}" \
  "$@"
