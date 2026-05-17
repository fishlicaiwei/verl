set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
FILE="/workspace/verl/verl/ours/llm_reward.py"
FUNCTION_NAME="llm_as_judge_reward"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/workspace/data/gsm8k/train.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/workspace/models/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    reward_model.reward_manager=llmasjudge \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='llm-as-judge_grpo_qwen2-5_7b' \
    trainer.experiment_name='qwen25_7b_function_rm_test1' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    +trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_training_steps=5 $@
    # custom_reward_function.path=$FILE\
    # custom_reward_function.name=$FUNCTION_NAME\
    # trainer.total_epochs=15 $@