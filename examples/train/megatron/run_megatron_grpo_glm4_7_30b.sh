set -x

# Colocated GRPO training+generation for GLM-4.7-Flash on GSM8K with Megatron.
# GLM-4.7-Flash (zai-org/GLM-4.7-Flash) is a DeepSeek-V3 architecture clone
# with MLA + MoE (64 routed experts, 4 active per token, ~3B active parameters).
#
# Runs on 1 node of 8 GPUs (TP=1 EP=8 for Megatron, 2x TP=4 vLLM engines).
# GLM-4.7-Flash has 20 attention heads, so vLLM TP must divide 20 (use TP=4).
#
# Setup:
#   1. Install deps:
#        uv sync --extra megatron
#   2. GLM-4.7-Flash needs transformers>=5.0.0 (for Glm4MoeLiteConfig).
#      If not yet available via uv sync, install manually:
#        uv pip install "transformers>=5.0.0"
#   3. Prepare data:
#        uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
#   4. Run:
#        export WANDB_API_KEY=<your_key_here>  # or set LOGGER=console below
#        bash examples/train/megatron/run_megatron_grpo_glm4_7_30b.sh

# This script doubles as the NATIVE control arm for an IsoExec comparison -- same model, same task,
# stock kernels. The task-side knobs below are env-overridable for exactly that reason: a comparison
# is only worth reading if both arms see the same data, batch sizes, lengths and lr, and shipped
# defaults differ between scripts (this one ships train_batch 128 / eval every 5 / wd 0.1). The
# STACK-side settings -- moe_grouped_gemm, flash_attn, async_engine, remove_microbatch_padding --
# are deliberately NOT parameterized: they are what "native" MEANS here, and matching them to the
# IsoExec arm would defeat the comparison.
MODEL_NAME="zai-org/GLM-4.7-Flash"
DATA_DIR=${DATA_DIR:-"$HOME/data/gsm8k"}
CKPT_DIR=${CKPT_DIR:-"$HOME/ckpts/glm4_7_30b_a3b_grpo_megatron"}
LOGGER="${LOGGER:-wandb}"  # change to "console" to print to stdout
PROJECT_NAME="${PROJECT_NAME:-glm4_7_30b_grpo}"
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
POLICY_MINI_BATCH_SIZE=${POLICY_MINI_BATCH_SIZE:-64}
EVAL_INTERVAL=${EVAL_INTERVAL:-5}
CKPT_INTERVAL=${CKPT_INTERVAL:-10}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.5}

INFERENCE_BACKEND="vllm"

NUM_NODES=1
NUM_GPUS=8

# Megatron parallelism: TP=1, EP=8 fits 64 MoE experts across 8 GPUs (8 experts/GPU)
MEGATRON_TP=1
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=8
MEGATRON_ETP=1

# vLLM inference: 2 engines x TP=4 = 8 GPUs (20 heads / 4 = 5 heads per GPU)
NUM_INFERENCE_ENGINES=2
INFERENCE_ENGINE_TP=4
INFERENCE_ENGINE_MAX_MODEL_LEN=${INFERENCE_ENGINE_MAX_MODEL_LEN:-2048}

RUN_NAME="${RUN_NAME:-glm4_7_30b_a3b_grpo_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}_ep${MEGATRON_EP}_etp${MEGATRON_ETP}}"

# GLM-4.7-Flash supports flash attention (v_head_dim == qk_head_dim + qk_rope_head_dim == 256).
# Most other MLA models (DeepSeek-V3, Moonlight) do NOT support flash attention due to
# mismatched Q/V head dimensions. Use flash_attn=false for those models.
FLASH_ATTN=true

# MoE routing flags (DeepSeek-V3 style: sigmoid scoring with expert bias)
MOE_TOKEN_DISPATCHER="alltoall"
MOE_ROUTER_LB="none"
MOE_GROUPED_GEMM=true
MOE_ROUTER_SCORE_FN="sigmoid"
MOE_ROUTER_EXPERT_BIAS=true

# CPU optimizer offload to fit in 80GB GPUs
OPTIMIZER_CPU_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1.0

ENFORCE_EAGER=false

uv run --isolated --extra megatron -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TP \
  generator.inference_engine.enforce_eager=$ENFORCE_EAGER \
  generator.inference_engine.engine_init_kwargs.max_model_len=$INFERENCE_ENGINE_MAX_MODEL_LEN \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.policy.megatron_config.moe_token_dispatcher_type=$MOE_TOKEN_DISPATCHER \
  trainer.policy.megatron_config.moe_router_load_balancing_type=$MOE_ROUTER_LB \
  trainer.policy.megatron_config.moe_grouped_gemm=$MOE_GROUPED_GEMM \
  trainer.policy.megatron_config.moe_router_score_function=$MOE_ROUTER_SCORE_FN \
  trainer.policy.megatron_config.moe_router_enable_expert_bias=$MOE_ROUTER_EXPERT_BIAS \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=$OPTIMIZER_CPU_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION \
  trainer.policy.megatron_config.empty_cuda_cache=true \
  trainer.remove_microbatch_padding=true \
  trainer.flash_attn=$FLASH_ATTN \
  trainer.epochs=20 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=false \
  trainer.eval_interval=$EVAL_INTERVAL \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$POLICY_MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=$CKPT_INTERVAL \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.weight_decay=$WEIGHT_DECAY \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.algorithm.use_kl_loss=false \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.inference_engine.gpu_memory_utilization=$GPU_MEM_UTIL \
  trainer.logger="$LOGGER" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.run_name="$RUN_NAME" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPT_DIR" \
  $@
