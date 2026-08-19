set -x

# ==========================================================================================
# NATIVE (non-IsoExec) DAPO baseline for zai-org/GLM-4.7-Flash (30B-A3B, MLA + sigmoid MoE)
# on the DAPO-math-17k / AIME-2024 task spec.
# ==========================================================================================
# THIS IS THE BASELINE ARM of a 10-step A/B acceptance test: the native stack, against which an
# IsoExec arm on the same task spec is measured.
#
# The contract of the pair, and the reason every knob below is where it is:
#
#   TASK SIDE  -- byte-identical to the IsoExec arm. Same two parquet files, same env_class,
#     same global batch / mini batch / n_samples, same lengths, same DAPO clip knobs, same
#     optimizer schedule, same eval cadence. A difference in reward or entropy between the two
#     arms must be attributable to the STACK, never to the recipe.
#
#   STACK SIDE -- MAXIMALLY FAVORABLE TO NATIVE. This file exists to produce an HONEST
#     baseline, not a handicapped one. So: TransformerEngine layer spec, TE/cutlass grouped
#     GEMM, packed varlen (THD) microbatches, fused RoPE, fused masked softmax, vLLM's own
#     Glm4MoeLite model class on its own FLASH_ATTN_MLA backend with ABSORBED decode, prefix
#     caching / chunked prefill / full CUDA graphs at vLLM's native defaults, sleep level 2,
#     and an engine memory budget derived UPWARD from the native lifecycle (see the KV BUDGET
#     block at the foot of this file) rather than inherited from the IsoExec arm's ledger.
#
#   NOT ONE SKYRL_ISOEXEC / SKYRL_ISOEXEC_* FLAG IS EXPORTED HERE, and that is the whole point.
#     Their absence is what selects the native paths:
#       * megatron_worker.py:721  -- no SKYRL_ISOEXEC_LOCAL_SPEC => megatron-bridge's DEFAULT
#         dense layer spec, which is hard-wired to TransformerEngine. Native gets TE.
#       * megatron_worker.py:697  -- no SKYRL_ISOEXEC => apply_rope_fusion, variable_seq_lengths
#         (varlen/THD), masked_softmax_fusion and gradient_accumulation_fusion all stay at their
#         native values instead of being forced off for bitwise reasons.
#       * ray_wrapped_inference_engine.py:350 -- no SKYRL_ISOEXEC and no LoRA => sleep_level=2.
#         The engine DISCARDS weights and frees the KV pool between steps, so the trainer gets
#         the whole GPU during policy_train. (The IsoExec arm is pinned to level 1 and pays a
#         6.7-9.3 s/worker D2H weights backup every step.)
#       * vLLM loads its OWN Glm4MoeLite, not a megatron GPTModel inside a vLLM worker, so it
#         picks FLASH_ATTN_MLA with absorbed (latent-KV) decode by itself. Nothing below tells
#         it which attention backend or MLA form to use -- that is deliberate.
#
# PROVENANCE OF THE STACK SIDE. Every native number quoted in this file is measured, from a
# 50-step native GLM-4.7-Flash GSM8K control arm (106.4 s/step). That arm ran trainer
# TP=1 / EP=8 / ETP=1 with engine 2 x TP4, and this file keeps that parallelism.
#
# CAVEAT ON "TRUE NATIVE": none. TransformerEngine 2.11.0 IS available on this node -- it is a
# hard dependency of the `megatron` extra and is present in the built env. It is the
# `isoexec` extra that deliberately omits it (pyproject.toml:158-161, "NO transformer-engine /
# flash-attn / mamba-ssm / causal-conv1d"). So `uv run --isolated --extra megatron` below gets
# the genuine TE path, and the "no-TE" line in the gap analysis's confound table is a property
# of the IsoExec arm only. Nothing in this baseline is degraded for lack of a dependency.
#
# Prepare data -- SHARE the files with the IsoExec arm, do NOT regenerate:
#   bash examples/train/algorithms/dapo/prepare_dapo_data.sh
# Launch (10-step acceptance arm):
#   WANDB_API_KEY=<key> bash examples/train/megatron/run_megatron_dapo_glm4.7_flash.sh

DATA_DIR="${DATA_DIR:-$HOME/data}"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE="$DATA_DIR/aime-2024-cleaned.parquet"
LOGGER="${LOGGER:-wandb}"          # LOGGER=console prints step metrics to stdout
MODEL_NAME="${GLM47_FLASH_PATH:-zai-org/GLM-4.7-Flash}"

INFERENCE_BACKEND="vllm"

NUM_NODES=1
NUM_GPUS=8

# ----- parallelism: the native GSM8K control arm's, unchanged --------------------------------
# 20 attention heads => any TP must divide 20; TP=8 does not exist on this model, which is why
# the engine runs 2 x TP4 rather than 1 x TP8.
#
# TRAINER TP=1 is what the native arm measured and it is also what the IsoExec arm now runs, so
# the two arms are matched on this axis. It is genuinely native's best here:
#   * global batch 128 prompts x 16 samples = 2048 sequences/step; mini batch 32 => 512
#     sequences per optimizer update, 4 updates/step. At TP=1, DP=8 => 64 microbatches per rank
#     per update (256/step at micro_train_batch=1). TP=2 halves DP and therefore DOUBLES both
#     the microbatch count AND the tokens each rank must push through -- at 10240 tokens per
#     microbatch the loop is compute bound, not launch bound, so TP=2 is close to a 2x loss.
#   * it fits: see the TRAINER MEMORY block at the foot of this file (~68 GiB peak reserved of
#     79.2, derived from the native arm's own measured static + activation slope).
# If the trainer OOMs, the levers in order are micro_forward_batch_size_per_gpu 4 -> 1, then
# MEGATRON_TP=2 -- NOT cutting the global batch, which is the task spec.
MEGATRON_TP=${MEGATRON_TP:-1}
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=${MEGATRON_EP:-8}     # 64 routed experts / 8 GPUs = 8 experts per rank
MEGATRON_ETP=${MEGATRON_ETP:-1}

NUM_INFERENCE_ENGINES=2
INFERENCE_ENGINE_TP=4

# ----- native stack knobs. DELIBERATELY NOT parameterized: they are what "native" MEANS -------
# GLM-4.7-Flash supports flash attention in the trainer (v_head_dim == qk_head_dim +
# qk_rope_head_dim == 256); most other MLA models do not. megatron_worker.py:690 maps
# trainer.flash_attn=true -> provider.attention_backend="flash".
FLASH_ATTN=true
# TE / cutlass grouped GEMM for the expert MLPs. This is the single biggest native-only MoE
# win and the exact thing the IsoExec arm must give up (it runs SequentialMLP so each expert is
# a plain F.linear and therefore batch-invariant). Native keeps it.
MOE_GROUPED_GEMM=true
MOE_TOKEN_DISPATCHER="alltoall"   # required at EP>1
# DeepSeek-V3 style routing: sigmoid scoring with a `noaux_tc` expert bias, and no aux loss.
MOE_ROUTER_LB="none"
MOE_ROUTER_SCORE_FN="sigmoid"
MOE_ROUTER_EXPERT_BIAS=true
# Packed varlen (THD) microbatches -- no right-padding carried to the batch width. The IsoExec
# arm must run remove_microbatch_padding=false (padded BSHD) to match the engine's per-sequence
# paged attention. Native does not have that constraint.
REMOVE_MICROBATCH_PADDING=true
# CUDA graphs ON in the engine. vLLM captures a full decode-size set plus piecewise mixed
# graphs; the native GSM8K arm measured "Graph capturing finished in 46 secs, took 1.05 GiB".
ENFORCE_EAGER=false

# NOT SET, and each omission is a native default we are deliberately taking:
#   generator.inference_engine.distributed_executor_backend  -> "ray" (config.py:611). The
#     native control arm used it; the DAPO recipes' "mp" is a IsoExec/GDN-era workaround.
#   generator.inference_engine.async_engine                  -> true (config.py:578). Native
#     runs the async engine, which is both faster (overlapped scheduling) and the reason the
#     native arm emits `Running: N reqs, Waiting: M reqs, GPU KV cache usage / Prefix cache hit
#     rate` scheduler lines at all. The IsoExec arm sets it false and consequently logs no
#     scheduler stats -- read those lines here, they are the wave-count evidence.
#   _SKYRL_USE_NEW_INFERENCE                                 -> 1 (env_vars.py:96). The native
#     control arm ran the new inference-server path end to end for 50 steps.
#   trainer.policy.megatron_config.moe_router_dtype          -> "fp32" (config.py:210).
#   trainer.logprobs_chunk_size                              -> 1024 (config.py:803). The
#     IsoExec arm cuts it to 256; native keeps the larger chunk (fewer logits launches).
#   trainer.policy.megatron_config.transformer_config_kwargs -> SkyRL's own default
#     {recompute_granularity: full, recompute_modules: [core_attn], recompute_method: uniform,
#     recompute_num_layers: 1} (config.py:180-186). NOTE this is ALREADY the same recompute
#     setting the IsoExec arm passes explicitly, so the two arms are matched here and native is
#     NOT being handicapped -- and the native arm's measured memory numbers below were taken
#     WITH it. To probe a no-recompute native arm, append
#     `trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity=null`.
#   trainer.policy.language_model_only / generator...language_model_only -> false. GLM-4.7-Flash
#     is text-only; there is no VL bridge to force past (that flag is a Qwen3-VL concern).
#   engine attention backend / MLA form / prefix caching / chunked prefill -> vLLM's own choice.
#     On this checkpoint it resolves to FLASH_ATTN_MLA with ABSORBED decode over the 576-wide
#     latent (52.99 KiB/token/rank measured) and APC on, which is exactly the configuration the
#     IsoExec arm had to spend a whole program (OPTION ABS) to reach. Do not pin any of it.

# ----- optimizer offload: all four kwargs, which is the native-favorable set ------------------
# SkyRL's defaults are all off/0.0 (config.py:171-176). optimizer_cpu_offload +
# offload_fraction=1.0 move the fp32 master weights and Adam state to host RAM, which is what
# makes a 30B model fit at TP=1 at all; overlap_cpu_optimizer_d2h_h2d then hides the D2H/H2D
# behind compute and use_precision_aware_optimizer avoids a redundant fp32 GPU copy. The proven
# native Qwen3.5-35B DAPO recipe sets all four; so does the IsoExec arm. Matched, and fastest.
OPTIMIZER_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1.0

# ==========================================================================================
# TASK SPEC -- BYTE-IDENTICAL TO run_megatron_glm4.7_flash_aime_dapo_isoexec.sh. DO NOT TUNE.
# ==========================================================================================
#   dataset      dapo-math-17k-cleaned (train) / aime-2024-cleaned (val), env_class=aime
#   batching     train_batch 128 / mini 32 (4 optimizer updates per step) / n_samples 16
#   lengths      2K prompt / 8K response          eval  interval 5, 32 samples/prompt
#   DAPO knobs   dual_clip 0.2/0.28, c=10.0, token_mean, overlong filtering + 4K buffer
#   optimizer    lr 1e-6, warmup 40, wd 0.1, clip 1.0, 20 epochs
# The micro batch sizes are the ONE task-adjacent pair that is allowed to differ, because they
# are a memory/occupancy decision rather than a statistical one -- see MICRO SIZING below.
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
CLIP_RATIO_C=10.0
LOSS_REDUCTION="token_mean"
APPLY_OVERLONG_FILTERING=true
OVERLONG_BUFFER_LEN=$((1024 * 4))
OVERLONG_BUFFER_PENALTY_FACTOR=1.0
TEMPERATURE=1.0
TOP_P=1.0
EVAL_TOP_P=0.7

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-$((1024 * 2))}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-$((1024 * 8))}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
POLICY_MINI_BATCH_SIZE=${POLICY_MINI_BATCH_SIZE:-32}
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))   # 10240

# TIS stays OFF (tis_ratio_type=null), matching the IsoExec arm. This is the one place where
# "favor native" and "match the task spec" could be argued to pull apart -- token-level TIS is
# an off-policy correction for exactly the train/rollout logprob mismatch that native HAS and
# IsoExec does not, so enabling it here would arguably help native's LEARNING. It is not
# enabled, because it changes the objective and the objective is task side. If a TIS-on native
# arm is wanted it is a THIRD arm, not this one:
#   ... trainer.algorithm.off_policy_correction.tis_ratio_type=token \
#       trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=2.0

# ----- MICRO SIZING: native gets the larger scoring microbatch, which is a real native win ----
# micro_forward_batch_size_per_gpu=4 is the native GSM8K arm's value and the gap analysis calls
# it out directly: "4x fewer scoring launches on native -- part of why its scoring is 7.3 s
# against our 33.8-43.2 s on audit steps". The scoring pass is no-grad, so its activations are
# freed layer by layer and the peak is O(a few layers), not O(4 x 10240 tokens): the transient
# is the per-layer expert intermediate, 4 x 10240 tok x 4 experts x 3072 x 2 B ~ 1.0 GiB, live
# for one layer. Against ~11 GiB of measured headroom (TRAINER MEMORY block) that is affordable.
# TRAINING microbatch stays at 1 -- a single 10240-token sequence is already ~9x the GSM8K
# microbatch, so there is no occupancy floor left to raise and mbs=2 would only double an
# activation peak that is the binding trainer constraint. 1 is also what both DAPO recipes run.
MICRO_FORWARD_BSZ=${MICRO_FORWARD_BSZ:-4}
MICRO_TRAIN_BSZ=${MICRO_TRAIN_BSZ:-1}

# ----- engine memory: derived UPWARD from the native lifecycle, not inherited -----------------
# Full arithmetic in the KV BUDGET block at the foot of this file. Summary:
#   measured anchor (native GSM8K arm, util 0.5): 40.0 GiB budget - 21.03 GiB KV => 18.97 GiB
#     of engine non-KV (~15.0 weights at TP4 + ~4.0 profiling/non-torch/graphs), and
#     21.03 GiB / 416,176 tok = 52.99 KiB/token/rank for the ABSORBED MLA latent.
#   native lifecycle: during generate the policy is on CPU (trainer.py:970-973); during training
#     the engine is asleep at LEVEL 2 so its weights and KV are gone; the KV pool is (re)woken
#     only AFTER _finish_weight_sync puts the policy back on CPU (worker_dispatch.py:600-608).
#     So the engine never co-resides with the trainer's activation peak, and 0.5 -- inherited
#     from a GSM8K script -- badly under-budgets native at DAPO lengths.
# ***** 0.75 IS REFUTED BY MEASUREMENT. THE ARM THAT RAN, AND STILL RUNS, IS 0.62. *****
# The 0.75 derivation below is preserved because its ERROR is the point, and it is the same
# error the IsoExec arm made independently: it assumed a "~4 GiB trainer residual" during
# generate. Measured, that residual is 16.22 GiB at step 2.
#   THE AUTOPSY (2026-08-09 17:27:40, infra-260809_165240.log:4730). The 0.75 arm died at step
#   2's post-train wake: engine worker 62.86 GiB (= 60 budget + 2.86 non-torch overshoot -- NCCL
#   plus 341 MiB of CUDA-graph pools that engine profiling never sees) + trainer residual 16.22
#   = 79.1 of 79.18, and a 142 MiB sampler alloc OOMed.
#   WHY 16.22 AND NOT 4. Measured by phase on the relaunched 0.62 arm: the trainer PID holds 2.1
#   GiB at STEP-1 generate (before any training phase has run) and 16.22 at step 2. The ~14 GiB
#   is BORN DURING THE FIRST TRAINING PHASE and survives the offload: lazily-created NCCL
#   communicators (native runs uncapped and un-prewarmed), cuBLAS/TE workspaces, and
#   hybrid-optimizer GPU staging. Stock offload moves the megatron BUFFERS; it cannot move any of
#   that. This is precisely the failure mode a step-1 check cannot see -- do not validate a util
#   on step 1.
#   => the real generate-phase ceiling is util <= (79.2 - 16.22 - 1.5 contexts - 2.9 engine
#      overshoot)/80 ~= 0.73, and 0.62 is the measured-safe rung below it.
# DERIVED-FROM: mla_form=absorbed(vLLM's own) trainer_tp=1 engine_tp=4 hbm_gib=79.2
#   max_seqs=256 max_batched=10240 workload=dapo-2k+8k trainer_residual_gib=16.22(MEASURED)
#   evidence=a 10-step native run of this file (measured pool 635,936 tokens/rank). The IsoExec
#   comparator runs the same 0.62 for the same reason, so the pair is matched on this axis.
#   Raising toward ~0.73 is a DELIBERATE experiment: it needs a step-2-or-later reading of the
#   trainer residual, not a step-1 reading of "Available KV cache memory".
#
# THE SUPERSEDED 0.75 DERIVATION, kept so nobody re-derives it:
#   0.75 x 80 = 60.0 GiB budget - 5.0 non-KV => 55.0 GiB KV => ~1,089,000 tokens/rank
#     => ~106 concurrent at the full 10240 ceiling, ~147 at the observed ~7.4k mean sequence.
#   "The generate-phase ceiling is util <= ~0.92 (79.2 usable - ~4 GiB trainer residual - ~1.5
#     GiB of two CUDA contexts per GPU); 0.75 leaves ~13.7 GiB of deliberate margin because the
#     ledger is DERIVED at DAPO shapes, not measured there."  <-- the ~4 GiB is the wrong term.
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.62}
# 256 is ~2x the KV-sustained concurrency above (106 full-length / 147 typical), which is the
# point: the SLOT ceiling must never be the binding constraint -- vLLM admits by KV
# availability, and a ceiling below what KV sustains is exactly the mistake that cost the
# IsoExec arm ~2,400 s of a ~3,450 s generate. The schema default of 1024 (config.py:599) would
# be ~9x and buys nothing but scheduler/sampler bookkeeping over unreachable slots.
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
# 10240 == max_model_len, chosen so that (a) a full-length prefill is never forced to chunk and
# the setting is valid whether or not vLLM enables chunked prefill for this model, and (b) five
# whole 2048-token prompts pack into one prefill step (the schema default 8192 packs four).
# Native vLLM computes logits only at sampled positions during profiling, so this does NOT carry
# the full-vocab fp32 profiling cost that forced the IsoExec arm down to 4096.
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-10240}

# ----- the acceptance cap ---------------------------------------------------------------------
# trainer.max_training_steps is a real config key (config.py:752); trainer.py:233-234 clamps
# total_training_steps to it and trainer.py:535-538 stops the loop once global_step exceeds it.
# It is safe for the A/B: the LR schedule is driven by total_training_steps (trainer.py:761-762)
# but num_warmup_steps=40 is ABSOLUTE, so a capped arm that never leaves warmup has an LR
# trajectory identical to an uncapped run's first N steps. Eval fires at step 5 and step 10
# either way (interval 5).
#
# THE CAP CANNOT BE 10. trainer.py:761-762 sets the scheduler's lr_decay_steps to
# `total_training_steps * policy_steps_per_train_batch`, and this launcher's batch sizes give
# 128/32 x 1 = 4 optimizer updates per step. A cap of 10 therefore yields lr_decay_steps=40
# against num_warmup_steps=40, and megatron/core/optimizer_param_scheduler.py:160 asserts
# `lr_warmup_steps < lr_decay_steps` -- 40 < 40 is false, so the run dies at optimizer
# construction before step 1. The cap must satisfy cap * 4 > 40, i.e. cap >= 11. Eleven still
# never leaves warmup (44 updates, warmup 40 -> the last 4 updates are the first off-warmup
# ones, and steps 1-10 are unaffected), so the 10-step comparison window is unchanged.
#
# **THE SAME CAP MUST BE GIVEN TO THE ISOEXEC ARM**, which does not default to one -- append
# `trainer.max_training_steps=11` to its launch line. Set MAX_TRAINING_STEPS=null here for an
# uncapped production run.
MAX_TRAINING_STEPS=${MAX_TRAINING_STEPS:-11}

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

uv run --isolated --extra megatron -m examples.train.algorithms.dapo.main_dapo \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$TEST_FILE']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.policy_loss_type="dual_clip" \
  trainer.algorithm.eps_clip_low=$CLIP_RATIO_LOW \
  trainer.algorithm.eps_clip_high=$CLIP_RATIO_HIGH \
  trainer.algorithm.clip_ratio_c=$CLIP_RATIO_C \
  trainer.algorithm.loss_reduction=$LOSS_REDUCTION \
  trainer.algorithm.overlong_buffer_len=$OVERLONG_BUFFER_LEN \
  trainer.algorithm.overlong_buffer_penalty_factor=$OVERLONG_BUFFER_PENALTY_FACTOR \
  generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.off_policy_correction.tis_ratio_type=null \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TP \
  generator.inference_engine.engine_init_kwargs.max_model_len=$MAX_MODEL_LEN \
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
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION \
  trainer.policy.megatron_config.empty_cuda_cache=true \
  trainer.remove_microbatch_padding=$REMOVE_MICROBATCH_PADDING \
  trainer.flash_attn=$FLASH_ATTN \
  generator.inference_engine.enforce_eager=$ENFORCE_EAGER \
  generator.sampling_params.temperature=$TEMPERATURE \
  generator.sampling_params.top_p=$TOP_P \
  generator.eval_sampling_params.temperature=$TEMPERATURE \
  generator.eval_sampling_params.top_p=$EVAL_TOP_P \
  generator.eval_sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  trainer.epochs=20 \
  trainer.max_training_steps=$MAX_TRAINING_STEPS \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$POLICY_MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=$MICRO_FORWARD_BSZ \
  trainer.micro_train_batch_size_per_gpu=$MICRO_TRAIN_BSZ \
  trainer.ckpt_interval=${CKPT_INTERVAL:-10} \
  trainer.max_ckpts_to_keep=2 \
  trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.num_warmup_steps=40 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.batched=true \
  environment.env_class=aime \
  generator.n_samples_per_prompt=16 \
  generator.eval_n_samples_per_prompt=32 \
  generator.inference_engine.max_num_seqs=$MAX_NUM_SEQS \
  generator.inference_engine.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
  generator.inference_engine.gpu_memory_utilization=$GPU_MEM_UTIL \
  trainer.logger="$LOGGER" \
  trainer.project_name="${WANDB_PROJECT:-glm47_flash_dapo_aime}" \
  trainer.run_name="${WANDB_RUN_NAME:-native_dapo_glm47_flash_baseline}" \
  trainer.resume_mode=null \
  trainer.ckpt_path="${CKPT_DIR:-$HOME/ckpts/native_dapo_aime_glm47_flash}" \
  "$@"

# ==========================================================================================
# KV BUDGET -- the full derivation behind GPU_MEM_UTIL / MAX_NUM_SEQS / MAX_NUM_BATCHED_TOKENS
# ==========================================================================================
# MEASURED ANCHOR. Native GLM-4.7-Flash GSM8K control arm, 2026-08-07, same node, same model,
# same trainer TP=1/EP=8/ETP=1, same engine 2 x TP4, gpu_memory_utilization=0.5:
#     Available KV cache memory        21.03 GiB / rank
#     GPU KV cache size               416,176 tokens
#     => per-token KV                  21.03 GiB / 416,176 = 52.99 KiB / token / rank
#     => engine non-KV                 0.50 x 80 - 21.03 = 18.97 GiB
#                                      ( ~15.0 GiB weight shard  [30B bf16 / TP4]
#                                      + ~4.0  GiB profiling peak + non-torch + CUDA graphs )
# 52.99 KiB/token is the ABSORBED MLA latent: (512 kv_lora + 64 rope) x 2 B x 47 layers,
# TP-replicated because an MQA latent cannot shard. vLLM chooses this by itself on its own
# Glm4MoeLite / FLASH_ATTN_MLA path -- it is 4.44x cheaper than materialized heads and it is
# native's for free. Three independent derivations agree with the live measurement.
#
# WHY NATIVE MAY SPEND MUCH MORE THAN 0.5 -- the co-residency ledger, phase by phase, per GPU
# (79.2 GiB usable of a nominal 80):
#   A. engine init / profiling  policy built then offloaded (colocate_all), empty_cuda_cache=true.
#                               Engine effectively alone => util x 80 is all available.
#   B. generate                 policy weights AND optimizer on CPU (trainer.py:970-973 states
#                               the contract). Engine holds util x 80. Trainer residual is NCCL
#                               comms + persistent buffers, ~4 GiB. Two CUDA contexts per GPU
#                               (trainer process + vLLM worker), ~1.5 GiB.
#                               => util x 80 + 4 + 1.5 <= 79.2  =>  util <= ~0.92.  <-- BINDING
#   C. weight sync              worker_dispatch.save_weights_for_sampler ordering is
#                               wake(weights) -> broadcast -> _finish_weight_sync (policy leaves
#                               the GPU) -> wake(kv_cache). The KV pool is therefore allocated
#                               only AFTER the trainer's weights are gone; the pre-KV peak is
#                               engine weights 15.0 + trainer bf16 weights ~11.4 + NCCL staging.
#   D. training                 sleep LEVEL 2 (ray_wrapped_inference_engine.py:350 -- level 1 is
#                               forced only under SKYRL_ISOEXEC or LoRA). Engine weights AND KV
#                               are released; the trainer has the whole GPU. Independent of util.
#
# SHIPPED 0.62, and what it MEASURED (this is the arm every native GLM-4.7-Flash DAPO number in
# the reports comes from -- see the GPU_MEM_UTIL block for why 0.75 was refused):
#     engine budget            0.62 x 80                        = 49.6 GiB
#   - engine non-KV            15.0 weights + ~2.5 profiling/graphs/non-torch
#     = KV pool / rank                                          ~ 32.2 GiB
#     MEASURED pool                                               635,936 tokens/rank
#                                                                 (infra-260809_174952.log)
#     / 10240 tokens per full-length sequence                   ~ 62 concurrent per engine
#     at the observed ~7.4k mean sequence (5.4k response)       ~ 86 concurrent per engine
#   Requests: 128 prompts x 16 samples = 2048 / 2 engines = 1024 per engine
#     => ~12 scheduling waves at the full-length worst case, ~12 at the observed mean, BEFORE
#        APC credit for the 16 siblings that share each 2048-token prompt.
#   For scale: still 1.5x the KV pool the native GSM8K arm ran at 0.5. And the matched IsoExec
#   arm measures 606,096 tokens at the SAME three settings -- 1.049x, i.e. pool parity, which is
#   the fact that lets the two arms' generate times be compared at all.
#
# THE SUPERSEDED 0.75 ARITHMETIC, kept so nobody re-derives it:
#     engine budget            0.75 x 80                        = 60.0 GiB
#   - engine non-KV            15.0 weights + ~5.0 profiling/graphs/non-torch
#     = KV pool / rank                                          ~ 55.0 GiB
#     / 52.99 KiB per token                                     ~ 1,089,000 tokens
#   -- arithmetically fine, and it OOMed at step 2 anyway, because the binding term was never the
#   engine's budget: it was the trainer residual the ledger priced at ~4 GiB and measured at 16.2.
#
# MAX_NUM_SEQS = 256 is ~2x the 106-147 the pool sustains, so it is never the governor.
# MAX_NUM_BATCHED_TOKENS = 10240 = max_model_len, so a full-length prefill never chunks.
#
# ==========================================================================================
# TRAINER MEMORY -- why TP=1 with micro_train_batch=1 is expected to fit at 2K+8K
# ==========================================================================================
# From the same native GSM8K arm (per-step trainer telemetry, steps 1-50), at padded_seqlen
# ~1160 and WITH SkyRL's default recompute (full / [core_attn] / uniform / 1):
#     static_alloc        45.86 GiB   (bf16 weights + fp32 grad buffers; sequence-independent)
#     peak_alloc          48.10 GiB
#     peak_reserved       50.69 GiB   (=> a ~2.6 GiB alloc -> reserved fragmentation gap)
#     activation slope    (48.10 - 45.86) / 1160 tok = 1.98 MiB / token
# Extrapolated to this recipe's 10240-token microbatch:
#     45.86 + 1.98 MiB x 10240 / 1024 = 45.86 + 19.8  = 65.7 GiB alloc
#     + 2.6 GiB fragmentation                          ~ 68.3 GiB reserved   of 79.2
#     => ~11 GiB headroom, and the engine is asleep at level 2 while this happens.
# THIS IS AN EXTRAPOLATION, NOT A MEASUREMENT. The 1.98 MiB/token slope is taken from a step
# whose peak may have been set by the mbs=4 SCORING pass rather than the mbs=1 backward, so
# treat it as a lower bound. WATCH isoexec_train_peak_reserved_gib_max AT EVERY STEP, not just
# step 1 -- the IsoExec arm's own history has a config that survived every 1-step validation and
# then OOMed at production step 4. If it climbs past ~72 GiB, the levers in order are:
#   1. MICRO_FORWARD_BSZ=1     (costs scoring launches, no memory risk)
#   2. MEGATRON_TP=2           (halves per-rank activations and dense weights; roughly doubles
#                               the per-rank microbatch count -- a real throughput loss, so only
#                               if 1 is not enough)
#   3. transformer_config_kwargs.recompute_modules=[core_attn,mlp]  (more recompute FLOPs)
# NEVER cut TRAIN_BATCH_SIZE / POLICY_MINI_BATCH_SIZE / n_samples / the lengths: those are the
# task spec, and changing them voids the comparison this arm exists to make.
