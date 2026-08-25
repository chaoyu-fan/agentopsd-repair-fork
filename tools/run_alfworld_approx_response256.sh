#!/usr/bin/env bash
set -euo pipefail

# One-row parser/resource check using the minimum response budget observed to
# contain both the model's closing </think> and a valid <action> block.
sed \
  -e 's/data.train_batch_size=4/data.train_batch_size=1/' \
  -e 's/data.val_batch_size=8/data.val_batch_size=1/' \
  -e 's/data.max_response_length=32/data.max_response_length=256/' \
  -e 's/actor_rollout_ref.actor.ppo_mini_batch_size=4/actor_rollout_ref.actor.ppo_mini_batch_size=1/' \
  -e 's/actor_rollout_ref.rollout.response_length=32/actor_rollout_ref.rollout.response_length=256/' \
  -e 's/actor_rollout_ref.rollout.max_model_len=1056/actor_rollout_ref.rollout.max_model_len=1280/' \
  -e 's/env.max_steps=4/env.max_steps=1/' \
  -e 's/env.rollout.n=2/env.rollout.n=1/' \
  -e 's/actor_rollout_ref.rollout.do_sample=false/actor_rollout_ref.rollout.do_sample=false/' \
  -e 's/qwen25_3b_1gpu_calibration/qwen25_3b_1gpu_response256/' \
  -e 's@/home/agentopsd/runs/alfworld_approx_calibration/hydra@/home/agentopsd/runs/alfworld_approx_response256/hydra@' \
  /mnt/c/Users/1/Documents/ChatGPT/agentOPSD/tools/run_alfworld_approx_calibration.sh | bash
