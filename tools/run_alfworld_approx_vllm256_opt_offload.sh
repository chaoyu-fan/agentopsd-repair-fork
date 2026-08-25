#!/usr/bin/env bash
set -euo pipefail

# Corrected FSDP2/vLLM probe with optimizer state on CPU.  This is a
# resource-saving approximation; it is kept separate from the default run so
# that the paper-shaped configuration is never silently changed.
sed \
  -e 's@/home/agentopsd/data/alfworld_approx_4x8/train.parquet@/home/agentopsd/data/alfworld_approx_1x1/train.parquet@' \
  -e 's@/home/agentopsd/data/alfworld_approx_4x8/val.parquet@/home/agentopsd/data/alfworld_approx_1x1/val.parquet@' \
  -e 's/data.train_batch_size=4/data.train_batch_size=1/' \
  -e 's/data.val_batch_size=8/data.val_batch_size=1/' \
  -e 's/data.max_response_length=32/data.max_response_length=256/' \
  -e 's/actor_rollout_ref.actor.ppo_mini_batch_size=4/actor_rollout_ref.actor.ppo_mini_batch_size=1/' \
  -e 's/actor_rollout_ref.rollout.response_length=32/actor_rollout_ref.rollout.response_length=256/' \
  -e 's/actor_rollout_ref.rollout.max_model_len=1056/actor_rollout_ref.rollout.max_model_len=1280/' \
  -e 's/env.max_steps=4/env.max_steps=1/' \
  -e 's/env.rollout.n=2/env.rollout.n=1/' \
  -e 's/actor_rollout_ref.rollout.name=hf/actor_rollout_ref.rollout.name=vllm/' \
  -e 's/qwen25_3b_1gpu_calibration/qwen25_3b_1gpu_vllm256_opt_offload/' \
  -e 's@/home/agentopsd/runs/alfworld_approx_calibration/hydra@/home/agentopsd/runs/alfworld_approx_vllm256_opt_offload/hydra@' \
  -e 's/trainer.logger=\[console\]/trainer.logger=[console] actor_rollout_ref.rollout.gpu_memory_utilization=0.4 actor_rollout_ref.actor.fsdp_config.optimizer_offload=true/' \
  -e 's@hydra.run.dir=/home/agentopsd/runs/alfworld_approx_calibration/hydra@hydra.run.dir=/home/agentopsd/runs/alfworld_approx_vllm256_opt_offload/hydra@' \
  /mnt/c/Users/1/Documents/ChatGPT/agentOPSD/tools/run_alfworld_approx_calibration.sh | bash
