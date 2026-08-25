#!/usr/bin/env bash
set -euo pipefail

# Batch-diversity follow-up: 16 placeholder rows (the environment supplies
# the real ALFWorld tasks), two rollouts per row, eight-step horizon, three
# updates.  Still single-card and 1.5B, so this remains an approximation.
sed \
  -e 's@alfworld_approx_4x8@alfworld_approx_16x8@g' \
  -e 's@alfworld_approx_1p5b_vllm_5updates@alfworld_approx_1p5b_vllm_16x8_3updates@g' \
  -e 's/qwen25_1p5b_1gpu_vllm_5updates/qwen25_1p5b_1gpu_vllm_16x8_3updates/g' \
  -e 's/data.train_batch_size=4/data.train_batch_size=16/' \
  -e 's/data.val_batch_size=8/data.val_batch_size=16/' \
  -e 's/actor_rollout_ref.actor.ppo_mini_batch_size=4/actor_rollout_ref.actor.ppo_mini_batch_size=16/' \
  -e 's/data.max_prompt_length=1152/data.max_prompt_length=2048/' \
  -e 's/actor_rollout_ref.rollout.max_num_batched_tokens=2048/actor_rollout_ref.rollout.max_num_batched_tokens=4096/' \
  -e 's/actor_rollout_ref.rollout.max_model_len=1280/actor_rollout_ref.rollout.max_model_len=2304/' \
  -e 's/env.max_steps=4/env.max_steps=8/' \
  -e 's/trainer.total_epochs=5/trainer.total_epochs=3/' \
  /mnt/c/Users/1/Documents/ChatGPT/agentOPSD/tools/run_alfworld_approx_1p5b_vllm.sh | bash
