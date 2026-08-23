#!/usr/bin/env bash
# IsoExec DAPO: Qwen3.5-35B-A3B-Base on AIME, 1 node of 8xH100 (trainer TP4/EP8/ETP1, engine TP8).
# Only deviations from code/config defaults are set here. Env vars: read-site defaults that differ
# from the qualified production composition (flags.py is the census). Hydra: keys that differ from
# skyrl config defaults. Build the runtime first: examples/isoexec/build_isoexec_env.sh
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
local_root=${ISOEXEC_LOCAL_ROOT:-${HOME}/isoexec}   # venvs, wheel cache, logs; override per site
cd "${repo}"
# Optional site preamble (cluster module loads, proxies, shared caches). Absent on most machines.
if [[ -f "${local_root}/env.sh" ]]; then source "${local_root}/env.sh"; fi

MODEL="${ISOEXEC_MODEL:-Qwen/Qwen3.5-35B-A3B-Base}"
TRAIN_FILE="${ISOEXEC_TRAIN_FILE:-${HOME}/data/dapo/dapo-math-17k-cleaned.parquet}"
TEST_FILE="${ISOEXEC_TEST_FILE:-${HOME}/data/dapo/aime-2024-cleaned.parquet}"
RUN_NAME="${WANDB_RUN_NAME:-isoexec_dapo_qwen35b_$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${local_root}/logs/${RUN_NAME}"
mkdir -p "${LOG_DIR}"

# pinned CUDA-13 runtime + FlashInfer JIT toolchain
venv=${ISOEXEC_RUNTIME_VENV:-${local_root}/venvs/skyrl-isoexec-cu130}
jit=${ISOEXEC_JIT_VENV:-${local_root}/venvs/skyrl-cuda130-jit}/lib/python3.12/site-packages/nvidia/cu13
site="${venv}/lib/python3.12/site-packages"
[[ -x "${venv}/bin/python" && -x "${jit}/bin/nvcc" ]] || {
  echo "REFUSAL: pinned runtime/JIT venv missing; run examples/isoexec/build_isoexec_env.sh" >&2; exit 1; }
export SKYRL_RAY_PY_EXECUTABLE="${venv}/bin/python" RAY_ADDRESS=auto RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
unset RAY_RUNTIME_ENV_HOOK CUDA_VERSION NCCL_VERSION
export SKYRL_PYTHONPATH_EXPORT=1 SKYRL_LD_LIBRARY_PATH_EXPORT=1
export PYTHONPATH="${repo}/examples/isoexec/nightly/_torchvision_stub:${repo}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_HOME="${jit}" SKYRL_FLASHINFER_JIT_CUDA_HOME="${jit}" CUDA_LIB_PATH="${jit}/lib64"
export FLASHINFER_WORKSPACE_BASE="${local_root}/cache/flashinfer-te2171" PIK_CACHE="${local_root}/cache/pik-te2171-cu13"
export CUDNN_PATH="${site}/nvidia/cudnn" SKYRL_TE_CUDA_RUNTIME_LIB="${site}/nvidia/cu13/lib"
export LD_LIBRARY_PATH="${site}/torch/lib:${site}/nvidia/cu13/lib:${site}/nvidia/nccl/lib:${site}/nvidia/cudnn/lib:${site}/nvidia/nvshmem/lib"
mkdir -p "${FLASHINFER_WORKSPACE_BASE}" "${PIK_CACHE}"
export HF_HOME="${HF_HOME:-${local_root}/hf}"
export SKYRL_LOG_FILE="${LOG_DIR}/infra.log" ISOEXEC_CONTRACT_PATH="${LOG_DIR}/contract.json"

# the qualified IsoExec composition: every value differs from its read-site default
export SKYRL_ISOEXEC=1 SKYRL_ISOEXEC_LOCAL_SPEC=1 SKYRL_ISOEXEC_TE_PRIMITIVES=1
export SKYRL_ISOEXEC_MODEL_PATH="${MODEL}" SKYRL_ISOEXEC_ENGINE_TP=8 SKYRL_ISOEXEC_ENGINE_EP=1
export SKYRL_ISOEXEC_ENGINE_LOAD_WEIGHTS=0 SKYRL_ISOEXEC_MAX_MODEL_LEN=10240 SKYRL_ISOEXEC_MAX_BATCHED_TOKENS=10240
export SKYRL_ISOEXEC_ENABLE_CUDAGRAPH=1 SKYRL_ISOEXEC_SYNC_BUCKET_MB=128 SKYRL_ISOEXEC_SLEEP_SKIP_WEIGHTS_BACKUP=1
export SKYRL_ISOEXEC_TRAINER_SP=1 SKYRL_ISOEXEC_TRAINER_EP_BALANCE=1 SKYRL_ISOEXEC_SPLIT_LM_HEAD=1
export SKYRL_ISOEXEC_PIK=1 SKYRL_ISOEXEC_PIK_FASTLAUNCH=1 SKYRL_ISOEXEC_NCCL_PREWARM=1
export SKYRL_ISOEXEC_NCCL_MAX_NCHANNELS=1 SKYRL_ISOEXEC_NCCL_CHANNEL_PLAN=tp:24,ep:32 SKYRL_ISOEXEC_NCCL_CHANNEL_BUDGET_GIB=8
export SKYRL_ISOEXEC_ENGINE_NCCL_UNPIN=1 SKYRL_ISOEXEC_ENGINE_NCCL_MAX_NCHANNELS=16
export SKYRL_ISOEXEC_MM_CUBLASLT=1 SKYRL_ISOEXEC_MM_TILES=1 SKYRL_ISOEXEC_MM_TILES_DECODE_BUCKET=1
export SKYRL_ISOEXEC_MM_FWD_ONLY=1 SKYRL_ISOEXEC_MM_FWD_ONLY_BMM=1
export SKYRL_ISOEXEC_FUSED_ADD_NORM=1 SKYRL_ISOEXEC_FUSED_ROPE=1 SKYRL_ISOEXEC_NATIVE_NORM_MEMO=1
export SKYRL_ISOEXEC_GDN=1 SKYRL_ISOEXEC_GDN_KERNEL=chunk_synced SKYRL_ISOEXEC_GDN_NATIVE_KERNELS=1
export SKYRL_ISOEXEC_GDN_CHUNKED_PREFILL=1 SKYRL_ISOEXEC_GDN_CS_MIN_PAGES=1 SKYRL_ISOEXEC_GDN_CS_SLEEP=1
export SKYRL_ISOEXEC_GDN_FLA_BACKWARD=1 SKYRL_ISOEXEC_GDN_ANALYTIC_CONV_BWD=1 SKYRL_ISOEXEC_GDN_VALIDATE_ONCE=1
export SKYRL_ISOEXEC_GDN_FUSED_OUTNORM=1 SKYRL_ISOEXEC_GDN_SCORING_FUSED_OUTNORM=1
export SKYRL_ISOEXEC_MOE_ROUTER_O2=1 SKYRL_ISOEXEC_MOE_PREAMBLE_O12=1 SKYRL_ISOEXEC_MOE_SHARED_OWNER_FUSION=1
export SKYRL_ISOEXEC_MOE_PIK_FC2=1 SKYRL_ISOEXEC_MOE_PIK_OWNER_COMBINE=1 SKYRL_ISOEXEC_MOE_FC2_INGEMM=1
export SKYRL_ISOEXEC_MOE_FUSED_COMBINE=1 SKYRL_ISOEXEC_MOE_FUSED_COMBINE_TRAINER=1 SKYRL_ISOEXEC_MOE_FUSED_LEAFCOMBINE=1
export SKYRL_ISOEXEC_MOE_FUSED_BLOCKMAP=1 SKYRL_ISOEXEC_MOE_FUSED_EPILOGUE=1 SKYRL_ISOEXEC_MOE_INDEXED_BMM=1
export SKYRL_ISOEXEC_MOE_FLAT_STAGE=1 SKYRL_ISOEXEC_MOE_COMBINE_SORT=1 SKYRL_ISOEXEC_MOE_CHUNK_SORT=1
export SKYRL_ISOEXEC_MOE_COMBINE_FOLD_ROUND=1 SKYRL_ISOEXEC_MOE_A2A_BF16_WIRE=1
export SKYRL_ISOEXEC_EXACT_VOCAB_PIPELINE=1 SKYRL_ISOEXEC_EXACT_SAMPLED_LOGPROBS=1
export SKYRL_ISOEXEC_SCORING_LOGITS_BF16=1 SKYRL_ISOEXEC_LOGPROB_GATHER_BF16_WIRE=1 SKYRL_ISOEXEC_LOGPROB_BWD_REDUCED_COMM=1
export SKYRL_ISOEXEC_AUTOFUSE_FAST_CFG=1 SKYRL_ISOEXEC_AUTOFUSE_LEDGER="${local_root}/isoexec/autofuse_v10.json"
export SKYRL_ISOEXEC_BWD_COMPILE=1 SKYRL_ISOEXEC_BWD_COMPILE_LEDGER="${local_root}/logs/bwd_compile_combined_prod_20260816/combined_ledger.json"
export VLLM_BATCH_INVARIANT=1 VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export _SKYRL_USE_NEW_INFERENCE=0

uv run --isolated --no-project -- "${venv}/bin/python" -m examples.train.algorithms.dapo.main_dapo \
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
  generator.inference_engine.enable_prefix_caching=false \
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
