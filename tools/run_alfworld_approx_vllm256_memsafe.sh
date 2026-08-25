#!/usr/bin/env bash
set -euo pipefail

# Same corrected FSDP2/vLLM smoke as vllm256_dump, with a deliberately lower
# vLLM memory fraction.  The default 0.5 setting crossed the physical 32-GiB
# card during the first corrected run; this is a resource-safety approximation.
sed \
  -e 's@vllm256_dump@vllm256_memsafe@g' \
  -e 's@qwen25_3b_1gpu_vllm256_dump@qwen25_3b_1gpu_vllm256_memsafe@g' \
  -e 's/trainer.logger=\[console\]/trainer.logger=[console] actor_rollout_ref.rollout.gpu_memory_utilization=0.2/' \
  /mnt/c/Users/1/Documents/ChatGPT/agentOPSD/tools/run_alfworld_approx_vllm256_dump.sh | bash
