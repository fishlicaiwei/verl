set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
num_gpus=8

# === Centralized Data Routing ===
# Define the task for this run. The data_router.py script will provide paths based on this.
#available tasks: IF  chat  code  knowledge  math  safe  writting  agent
TASK_NAME='math'
echo "Running data router for task: ${TASK_NAME}"

# === Configuration Variables ===
PROJECT_NAME="llm-as-judge_${TASK_NAME}_grpo_qwen2-5_7b"
# EXPERIMENT_NAME='hot_dsr1_ct_vr_exp1'
EXPERIMENT_NAME='hot_dsr1_or_prm_dm124'

# === Model Paths ===
#MODEL_PATH="/workspace/models/Qwen2.5-7B-Base"
# 替换你下载下来的模型，因为是持续训练，不需要完整ckpt，hf拉模型权重即可
#MODEL_PATH="/workspace/models/checkpoints/full_sft_qwen2-5_7b_openr1_3k_context8k"
MODEL_PATH="/workspace/models/r1_process_rl/llm-as-judge_math_grpo_qwen2-5_7b/hot_dsr1_think_exp1"
CHECKPOINT_PATH="/workspace/models/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"

# Call the Python script to get the data paths.
# It prints two lines: 1. train path, 2. val paths string.
DATA_PATHS_OUTPUT=$(python3 -m verl.ours.data_router --task ${TASK_NAME})

# Check if the script ran successfully
if [ $? -ne 0 ]; then
    echo "Error: data_router.py failed. Aborting."
    exit 1
fi

# Read the two lines of output into shell variables
TRAIN_DATASET_PATH=$(echo "${DATA_PATHS_OUTPUT}" | sed -n '1p')
VAL_DATASET_PATH=$(echo "${DATA_PATHS_OUTPUT}" | sed -n '2p')

echo "Using training data: ${TRAIN_DATASET_PATH}"
echo "Using validation data: ${VAL_DATASET_PATH}"

# Ensure the checkpoint directory exists
mkdir -p $CHECKPOINT_PATH

# === Main Training Command ===
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATASET_PATH \
    data.val_files=$VAL_DATASET_PATH \
    data.train_batch_size=480 \
    data.val_batch_size=400 \
    data.max_prompt_length=1024 \
    data.max_response_length=9216 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=160 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.rollout.max_model_len=10240 \
    actor_rollout_ref.actor.use_kl_loss=True \
    +actor_rollout_ref.actor.use_independent_advantage=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=10 \
    actor_rollout_ref.ref.fsdp_config.param_offload=TRUE \
    +actor_rollout_ref.rollout.trace.backend=weave \
    algorithm.kl_ctrl.kl_coef=0.001 \
    reward_model.reward_manager=dsr1-rule \
    +reward_model.reward_manager_type=think-MATH \
    +reward_model.val_reward_manager=skyworkqwen \
    +reward_model.val_reward_manager_type=outcome \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${CHECKPOINT_PATH} \
    trainer.n_gpus_per_node=$num_gpus \
    trainer.nnodes=1 \
    +trainer.val_before_train=False \
    +trainer.val_only=False \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.test_freq=-1 \
    trainer.save_freq=2 \
    trainer.total_epochs=1 $@
    # trainer.total_training_steps=5 $@

#/workspace/verl/verl/ours/run_qwen-7b_llmasjudge_grpo.sh