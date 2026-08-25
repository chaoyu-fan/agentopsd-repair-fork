#!/usr/bin/env bash
set -euo pipefail

# Faithful engine check: vLLM avoids the HF rollout's Qwen position_ids
# degeneration, but may exceed a single 32-GiB card once colocated with FSDP.
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
  -e 's/qwen25_3b_1gpu_calibration/qwen25_3b_1gpu_vllm256_dump/' \
  -e 's@/home/agentopsd/runs/alfworld_approx_calibration/hydra@/home/agentopsd/runs/alfworld_approx_vllm256_dump/hydra@' \
  -e 's/trainer.logger=\[console\]/trainer.logger=[console] trainer.rollout_data_dir=\/home\/agentopsd\/runs\/alfworld_approx_vllm256_dump\/rollouts/' \
  /mnt/c/Users/1/Documents/ChatGPT/agentOPSD/tools/run_alfworld_approx_calibration.sh | bash
