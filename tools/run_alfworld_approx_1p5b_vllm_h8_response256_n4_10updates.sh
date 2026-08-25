#!/usr/bin/env bash
set -euo pipefail

# Same safe n=4 single-card approximation as the five-update follow-up, with
# ten updates to measure whether sparse positive reward persists.
sed \
  -e 's@alfworld_approx_1p5b_vllm_5updates@alfworld_approx_1p5b_vllm_h8_5updates@g' \
  -e 's/qwen25_1p5b_1gpu_vllm_5updates/qwen25_1p5b_1gpu_vllm_h8_5updates/g' \
  -e 's/data.max_prompt_length=1152/data.max_prompt_length=2048/' \
  -e 's/actor_rollout_ref.rollout.max_num_batched_tokens=2048/actor_rollout_ref.rollout.max_num_batched_tokens=4096/' \
  -e 's/actor_rollout_ref.rollout.max_model_len=1280/actor_rollout_ref.rollout.max_model_len=2304/' \
  -e 's/env.max_steps=4/env.max_steps=8/' \
  /mnt/c/Users/1/Documents/ChatGPT/agentOPSD/tools/run_alfworld_approx_1p5b_vllm.sh |
sed \
  -e 's/data.max_response_length=128/data.max_response_length=256/' \
  -e 's/actor_rollout_ref.rollout.response_length=128/actor_rollout_ref.rollout.response_length=256/' \
  -e 's/actor_rollout_ref.rollout.max_model_len=2304/actor_rollout_ref.rollout.max_model_len=2560/' \
  -e 's/env.rollout.n=2/env.rollout.n=4/' \
  -e 's/trainer.total_epochs=5/trainer.total_epochs=10/' \
  -e 's/qwen25_1p5b_1gpu_vllm_h8_5updates/qwen25_1p5b_1gpu_vllm_h8_response256_n4_10updates/g' \
  -e 's@/home/agentopsd/runs/alfworld_approx_1p5b_vllm_h8_5updates@/home/agentopsd/runs/alfworld_approx_1p5b_vllm_h8_response256_n4_10updates@g' |
bash
