#!/usr/bin/env bash
set -euo pipefail

# Eight-step horizon follow-up to the five-update 1.5B smoke.  It keeps the
# same single-card FSDP2/vLLM setup while allowing enough interaction turns
# for a sparse ALFWorld success signal.  Still an engineering approximation.
sed \
  -e 's@alfworld_approx_1p5b_vllm_5updates@alfworld_approx_1p5b_vllm_h8_5updates@g' \
  -e 's/qwen25_1p5b_1gpu_vllm_5updates/qwen25_1p5b_1gpu_vllm_h8_5updates/g' \
  -e 's/data.max_prompt_length=1152/data.max_prompt_length=2048/' \
  -e 's/actor_rollout_ref.rollout.max_num_batched_tokens=2048/actor_rollout_ref.rollout.max_num_batched_tokens=4096/' \
  -e 's/actor_rollout_ref.rollout.max_model_len=1280/actor_rollout_ref.rollout.max_model_len=2304/' \
  -e 's/env.max_steps=4/env.max_steps=8/' \
  /mnt/c/Users/1/Documents/ChatGPT/agentOPSD/tools/run_alfworld_approx_1p5b_vllm.sh | bash
