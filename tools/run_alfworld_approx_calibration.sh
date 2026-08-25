#!/usr/bin/env bash
set -euo pipefail

REPO=/home/agentopsd/src/agentopsd-repair
cd "$REPO"
export ALFWORLD_DATA=/home/agentopsd/data/alfworld
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

# One-update calibration before any longer approximation. The reduced batch,
# group, context and horizon are explicit deviations from the paper's 8-GPU
# ALFWorld script and must remain labelled as engineering evidence.
exec /home/agentopsd/bin/micromamba run -n agentopsd python -m verl.trainer.main_opsd \
  algorithm.adv_estimator=grpo \
  data.train_files=/home/agentopsd/data/alfworld_approx_4x8/train.parquet \
  data.val_files=/home/agentopsd/data/alfworld_approx_4x8/val.parquet \
  data.train_batch_size=4 \
  data.val_batch_size=8 \
  data.max_prompt_length=1024 \
  data.max_response_length=32 \
  data.filter_overlong_prompts=False \
  data.truncation=error \
  data.return_raw_chat=True \
  data.shuffle=False \
  +data.dataloader_num_workers=0 \
  actor_rollout_ref.model.path=/home/agentopsd/models/Qwen2.5-3B-Instruct \
  actor_rollout_ref.model.use_shm=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=False \
  +actor_rollout_ref.model.override_config.attn_implementation=eager \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048 \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.use_torch_compile=false \
  +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.rollout.name=hf \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.response_length=32 \
  actor_rollout_ref.rollout.do_sample=false \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
  actor_rollout_ref.rollout.max_model_len=1056 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=false \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.use_invalid_action_penalty=true \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  algorithm.use_kl_in_reward=false \
  +algorithm.opsd.v0_prior=0.5 \
  +algorithm.opsd.belief_mult=true \
  +algorithm.opsd.mult_lambda=0.5 \
  +algorithm.opsd.granularity=turn \
  +algorithm.opsd.signed=true \
  +algorithm.opsd.skills_dir=/home/agentopsd/src/agentopsd-repair/skills/alfworld \
  +algorithm.opsd.skill_all=false \
  env.env_name=alfworld/AlfredTWEnv \
  env.seed=0 \
  env.max_steps=4 \
  env.history_length=2 \
  env.rollout.n=2 \
  env.resources_per_worker.num_cpus=0.1 \
  env.alfworld.eval_dataset=eval_in_distribution \
  trainer.critic_warmup=0 \
  trainer.logger=[console] \
  trainer.project_name=agentopsd_alfworld_approx \
  trainer.experiment_name=qwen25_3b_1gpu_calibration \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  trainer.val_before_train=false \
  trainer.resume_mode=disable \
  trainer.ray_wait_register_center_timeout=600 \
  ray_init.num_cpus=8 \
  hydra.run.dir=/home/agentopsd/runs/alfworld_approx_calibration/hydra \
  hydra.output_subdir=null
