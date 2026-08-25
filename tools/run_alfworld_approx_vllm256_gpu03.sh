#!/usr/bin/env bash
set -euo pipefail

# Lower-memory corrected FSDP2/vLLM probe.  The shorter token budgets are
# explicit engineering deviations used only to test whether a 32-GiB card can
# sustain the corrected rollout path.
sed \
  -e 's@vllm256_dump@vllm256_gpu03@g' \
  -e 's@qwen25_3b_1gpu_vllm256_dump@qwen25_3b_1gpu_vllm256_gpu03@g' \
  -e 's/actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048/actor_rollout_ref.actor.ppo_max_token_len_per_gpu=1024/' \
  -e 's/actor_rollout_ref.rollout.max_num_batched_tokens=2048/actor_rollout_ref.rollout.max_num_batched_tokens=1536/' \
  -e 's/actor_rollout_ref.rollout.max_model_len=1280/actor_rollout_ref.rollout.max_model_len=1056/' \
  -e 's/trainer.logger=\[console\]/trainer.logger=[console] actor_rollout_ref.rollout.gpu_memory_utilization=0.3/' \
  /mnt/c/Users/1/Documents/ChatGPT/agentOPSD/tools/run_alfworld_approx_vllm256_dump.sh | bash
