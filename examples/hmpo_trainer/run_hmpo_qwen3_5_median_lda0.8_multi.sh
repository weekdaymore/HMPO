#!/bin/bash
# ============================================================================
# HMPO (Hybrid Median-length Policy Optimization for Chain-of-Thought Compression) — Qwen3.5 GRPO training
#   reward_manager = hmpo
#   length-reward baseline b = median length of correct rollouts within a group
#   ablation switches: overlong_buffer_cfg.multi (mul/add), .lambda (cosine bias)
#
# Before running, edit MODEL_PATH (and optionally the data paths) in the USER CONFIG section below.
# Multi-node: this script forms a cluster via RANK / MASTER_ADDR / NODE_NUM env vars,
#           RANK=0 starts the ray head and submits the job; other ranks join as workers.
# ============================================================================


set -xo pipefail

export NCCL_NVLS_ENABLE=0
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN

# script dir -> examples/hmpo_trainer; repo root = two levels up
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"   # verl repo root
echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "WORK_ROOT=${WORK_ROOT}"
cd "${WORK_ROOT}"

GPUS_PER_NODE=8
NODE_RANK=${RANK}

# 9b 2nodes; 35b 4nodes; 122b 4nodes;
NODE_NUM=${NODE_NUM:-1}
NNODES=${NODE_NUM}
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo " - WORLD_SIZE:${NODE_NUM}"

# ---- install dependencies ----
pip install math-verify
pip install timm
pip install numpy==1.26.4
python3 -m pip install IPython
pip install transformers==5.3.0

# mbridge (Megatron <-> HuggingFace bridge, includes Qwen3.5 support)
cd "${WORK_ROOT}/third_party/mbridge"
pip install -e .
cd "${WORK_ROOT}"

export PYTHONPATH="${WORK_ROOT}:${PYTHONPATH}"
export REDIS_PORT=6379
export RAY_num_server_call_thread=1
export RAY_DEDUP_LOGS=0

########################### Quick Config ###########################

# dense model parallelism
TP=${TP:-2}
PP=${PP:-8}
CP=${CP:-1}
EP=${EP:-1}
ETP=${ETP:-1}
GEN_TP=${GEN_TP:-2}

ALL_OFFLOAD=${ALL_OFFLOAD:-True}


project_name='hmpo_qwen3_5_9b'
# project_name='hmpo_qwen3_5_35b'
# project_name='hmpo_qwen3_5_122b'
exp_name='median_lda08_multi'

adv_estimator=grpo
rollout_name="vllm"
use_kl_in_reward=True
kl_coef=0.001
use_kl_loss=True
kl_loss_coef=0.001
kl_loss_type=low_var_kl

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 4))
max_response_length=$((1024 * 24))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 8))
overlong_penalty_factor=1.0
overlong_buffer_lambda=0.8
clip_reward_high=1
loss_agg_mode="token-mean"

train_prompt_bsz=64
n_resp_per_prompt=10
train_prompt_mini_bsz=16

actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))

RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}


######################### USER CONFIG (edit me) #########################

MODEL_PATH=${MODEL_PATH:-"<YOUR_QWEN3.5_9B_MODEL_PATH>"}
# MODEL_PATH=${MODEL_PATH:-"<YOUR_QWEN3.5_35B_A3B_MODEL_PATH>"}
# MODEL_PATH=${MODEL_PATH:-"<YOUR_QWEN3.5_122B_A10B_MODEL_PATH>"}


# train/val data (defaults to the parquet files under ./dataset/)
DATASET_DIR="${SCRIPT_DIR}/dataset"
TRAIN_FILE=${TRAIN_FILE:-"${DATASET_DIR}/DeepMath-103K-v1_with_token_length_5_9.parquet"}
TEST_FILE=${TEST_FILE:-"${DATASET_DIR}/aime-2024.parquet"}
####################################################################

timestamp=$(date +"%Y-%m-%d-%H")
export OUTPUT_PATH="${WORK_ROOT}/hmpo_output/${project_name}/${exp_name}"
export CKPT_OUTPUT_PATH="${OUTPUT_PATH}/checkpoint/${project_name}/${exp_name}"
export TENSORBOARD_DIR="${OUTPUT_PATH}/tensorboard/${project_name}/${exp_name}"


LOG_DIR=${OUTPUT_PATH}/${timestamp}
export JOBLOG=${LOG_DIR}/training.log
RAY_WORKING_DIR="${OUTPUT_PATH}/ray_working_dir/"

mkdir -p ${OUTPUT_PATH} ${TENSORBOARD_DIR} ${RAY_WORKING_DIR} ${LOG_DIR} ${CKPT_OUTPUT_PATH}

# put triton / vllm cache on local disk (never on shared FS: concurrent multi-node access causes stale file handle)
export TRITON_CACHE_DIR="/dev/shm/triton_hmpo/"
export VLLM_CACHE_ROOT="/dev/shm/vllmca_hmpo/"

if [ "${RANK}" = "0" ]; then
    ray start --head --port=${REDIS_PORT} --include-dashboard=true --disable-usage-stats \
        2>&1 | tee -a ray_master_${RANK}.log
    sleep 30s
else
    sleep 10s
    ray start --address="${MASTER_ADDR}:${REDIS_PORT}" --block \
        2>&1 | tee -a ray_worker_${RANK}.log
fi

if [ "${RANK}" = "0" ]; then
  export RAY_ADDRESS='http://127.0.0.1:8265'
  # ---- assemble training args (grouped by module) ----
  ARGS=()

  # data
  ARGS+=(data.train_files="${TRAIN_FILE}")
  ARGS+=(data.val_files="${TEST_FILE}")
  ARGS+=(data.truncation='error')
  ARGS+=(data.filter_overlong_prompts=True)
  ARGS+=(data.max_prompt_length=${max_prompt_length})
  ARGS+=(data.max_response_length=${max_response_length})
  ARGS+=(data.train_batch_size=${train_prompt_bsz})
  ARGS+=(data.shuffle=False)

  # algorithm (advantage & KL)
  ARGS+=(algorithm.adv_estimator=${adv_estimator})
  ARGS+=(algorithm.use_kl_in_reward=${use_kl_in_reward})
  ARGS+=(algorithm.kl_ctrl.kl_coef=${kl_coef})

  # model
  ARGS+=(actor_rollout_ref.model.path="${MODEL_PATH}")
  ARGS+=(actor_rollout_ref.model.trust_remote_code=True)
  ARGS+=(actor_rollout_ref.model.use_remove_padding=True)
  ARGS+=(++actor_rollout_ref.model.enable_gradient_checkpointing=True)
  ARGS+=(actor_rollout_ref.model.mtp.enable=False)
  ARGS+=(actor_rollout_ref.model.mtp.enable_train=False)

  # actor: optimization & PPO
  ARGS+=(actor_rollout_ref.actor.optim.lr=5e-7)
  ARGS+=(actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz})
  ARGS+=(actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1)
  ARGS+=(actor_rollout_ref.actor.use_kl_loss=${use_kl_loss})
  ARGS+=(actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef})
  ARGS+=(actor_rollout_ref.actor.kl_loss_type=${kl_loss_type})
  ARGS+=(actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low})
  ARGS+=(actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high})
  ARGS+=(actor_rollout_ref.actor.clip_ratio_c=10.0)
  ARGS+=(actor_rollout_ref.actor.entropy_coeff=0)
  ARGS+=(actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode})
  ARGS+=(actor_rollout_ref.actor.use_dynamic_bsz=False)
  ARGS+=(actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len})

  # actor: Megatron parallelism
  ARGS+=(actor_rollout_ref.actor.megatron.use_mbridge=True)
  ARGS+=(actor_rollout_ref.actor.megatron.vanilla_mbridge=True)
  ARGS+=(actor_rollout_ref.actor.megatron.use_remove_padding=False)
  ARGS+=(actor_rollout_ref.actor.megatron.dtype=bfloat16)
  ARGS+=(actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP})
  ARGS+=(actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP})
  ARGS+=(actor_rollout_ref.actor.megatron.context_parallel_size=${CP})
  ARGS+=(actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP})
  ARGS+=(actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP})
  ARGS+=(actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD})
  ARGS+=(actor_rollout_ref.actor.megatron.grad_offload=${ALL_OFFLOAD})
  ARGS+=(actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD})

  # rollout (vLLM generation)
  ARGS+=(actor_rollout_ref.rollout.name=${rollout_name})
  ARGS+=(actor_rollout_ref.rollout.mode=async)
  ARGS+=(actor_rollout_ref.rollout.dtype=bfloat16)
  ARGS+=(actor_rollout_ref.rollout.gpu_memory_utilization=0.5)
  ARGS+=(actor_rollout_ref.rollout.enforce_eager=True)
  ARGS+=(actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP})
  ARGS+=(actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1)
  ARGS+=(actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False)
  ARGS+=(actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len})
  ARGS+=(actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096)

  # reference policy
  ARGS+=(actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False)
  ARGS+=(actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1)
  ARGS+=(actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len})
  ARGS+=(actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TP})
  ARGS+=(actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${PP})
  ARGS+=(actor_rollout_ref.ref.megatron.context_parallel_size=${CP})
  ARGS+=(actor_rollout_ref.ref.megatron.expert_model_parallel_size=${EP})
  ARGS+=(actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP})
  ARGS+=(actor_rollout_ref.ref.megatron.param_offload=True)

  # checkpoint
  ARGS+=(actor_rollout_ref.actor.checkpoint.save_contents=["model","extra","hf_model"])

  # reward: HMPO reward manager
  ARGS+=(reward_model.reward_manager=hmpo)
  ARGS+=(+reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer})
  ARGS+=(+reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len})
  ARGS+=(+reward_model.reward_kwargs.overlong_buffer_cfg.lambda=${overlong_buffer_lambda})
  ARGS+=(+reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor})
  ARGS+=(+reward_model.reward_kwargs.overlong_buffer_cfg.clip_reward_high=${clip_reward_high})
  ARGS+=(+reward_model.reward_kwargs.overlong_buffer_cfg.multi=True)
  ARGS+=(+reward_model.reward_kwargs.overlong_buffer_cfg.log=False)
  ARGS+=(+reward_model.reward_kwargs.max_resp_len=${max_response_length})
  ARGS+=(++reward_model.enable=True)
  ARGS+=(++reward_model.enable_resource_pool=False)

  # trainer
  ARGS+=(trainer.critic_warmup=0)
  ARGS+=(trainer.logger='["console","tensorboard"]')
  ARGS+=(trainer.project_name="${project_name}")
  ARGS+=(trainer.experiment_name="${exp_name}")
  ARGS+=(trainer.n_gpus_per_node=8)
  ARGS+=(trainer.nnodes="${NNODES}")
  ARGS+=(trainer.val_before_train=True)
  ARGS+=(actor_rollout_ref.rollout.val_kwargs.n=4)
  ARGS+=(actor_rollout_ref.rollout.val_kwargs.do_sample=True)
  ARGS+=(+trainer.multisample_val=True)
  ARGS+=(trainer.test_freq=4)
  ARGS+=(trainer.save_freq=4)
  ARGS+=(trainer.total_epochs=1)
  ARGS+=(trainer.rollout_data_dir="${LOG_DIR}")
  ARGS+=(trainer.default_local_dir="${CKPT_OUTPUT_PATH}")
  ARGS+=(trainer.resume_mode=disable)

  ray job submit --address ${RAY_ADDRESS} -- python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    "${ARGS[@]}" 2>&1 | tee ${JOBLOG}
fi
