#!/usr/bin/env bash
set -euo pipefail

# Parser-viability check: keep one environment and one turn while restoring a
# response budget long enough for Qwen2.5-3B to emit both <think> and <action>.
sed \
  -e 's/data.train_batch_size=4/data.train_batch_size=1/' \
  -e 's/data.val_batch_size=8/data.val_batch_size=1/' \
  -e 's/data.max_response_length=32/data.max_response_length=128/' \
  -e 's/actor_rollout_ref.actor.ppo_mini_batch_size=4/actor_rollout_ref.actor.ppo_mini_batch_size=1/' \
  -e 's/actor_rollout_ref.rollout.response_length=32/actor_rollout_ref.rollout.response_length=128/' \
  -e 's/actor_rollout_ref.rollout.max_model_len=1056/actor_rollout_ref.rollout.max_model_len=1152/' \
  -e 's/env.max_steps=4/env.max_steps=1/' \
  -e 's/env.rollout.n=2/env.rollout.n=1/' \
  -e 's/actor_rollout_ref.rollout.do_sample=false/actor_rollout_ref.rollout.do_sample=true/' \
  -e 's/qwen25_3b_1gpu_calibration/qwen25_3b_1gpu_response128/' \
  -e 's@/home/agentopsd/runs/alfworld_approx_calibration/hydra@/home/agentopsd/runs/alfworld_approx_response128/hydra@' \
  /mnt/c/Users/1/Documents/ChatGPT/agentOPSD/tools/run_alfworld_approx_calibration.sh | bash
