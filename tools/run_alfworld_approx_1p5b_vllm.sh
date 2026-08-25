#!/usr/bin/env bash
set -euo pipefail

# Single-card AgentOPSD approximation using Qwen2.5-1.5B and the validated
# FSDP2/vLLM path.  This is intentionally a reduced engineering run, not a
# paper-scale reproduction: one GPU, four training rows, two environment
# rollouts per row, short context/horizon, and five updates.
REPO=/home/agentopsd/src/agentopsd-repair
cd "$REPO"
export ALFWORLD_DATA=/home/agentopsd/data/alfworld
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

exec /home/agentopsd/bin/micromamba run -n agentopsd python -m verl.trainer.main_opsd \
  algorithm.adv_estimator=grpo \
  data.train_files=/home/agentopsd/data/alfworld_approx_4x8/train.parquet \
  data.val_files=/home/agentopsd/data/alfworld_approx_4x8/val.parquet \
  data.train_batch_size=4 \
  data.val_batch_size=8 \
  data.max_prompt_length=1152 \
  data.max_response_length=128 \
  data.filter_overlong_prompts=False \
  data.truncation=error \
  data.return_raw_chat=True \
  data.shuffle=False \
  +data.dataloader_num_workers=0 \
  actor_rollout_ref.model.path=/home/agentopsd/models/Qwen2.5-1.5B-Instruct \
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
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.response_length=128 \
  actor_rollout_ref.rollout.do_sample=true \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
  actor_rollout_ref.rollout.max_model_len=1280 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.enable_chunked_prefill=false \
  actor_rollout_ref.rollout.enforce_eager=false \
  actor_rollout_ref.rollout.free_cache_engine=false \
  actor_rollout_ref.rollout.val_kwargs.do_sample=true \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
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
  trainer.experiment_name=qwen25_1p5b_1gpu_vllm_5updates \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_epochs=5 \
  trainer.val_before_train=false \
  trainer.resume_mode=disable \
  trainer.ray_wait_register_center_timeout=600 \
  trainer.rollout_data_dir=/home/agentopsd/runs/alfworld_approx_1p5b_vllm_5updates/rollouts \
  ray_init.num_cpus=8 \
  hydra.run.dir=/home/agentopsd/runs/alfworld_approx_1p5b_vllm_5updates/hydra \
  hydra.output_subdir=null
