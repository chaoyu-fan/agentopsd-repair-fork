#!/usr/bin/env bash
set -euo pipefail

# Corrected FSDP2/vLLM smoke with the smallest tested cache fraction that may
# still initialize a 3B engine alongside the FSDP model.  This remains a
# single-card approximation and must be rechecked against nvidia-smi.
sed \
  -e 's@vllm256_dump@vllm256_gpu04@g' \
  -e 's@qwen25_3b_1gpu_vllm256_dump@qwen25_3b_1gpu_vllm256_gpu04@g' \
  -e 's/trainer.logger=\[console\]/trainer.logger=[console] actor_rollout_ref.rollout.gpu_memory_utilization=0.4/' \
  /mnt/c/Users/1/Documents/ChatGPT/agentOPSD/tools/run_alfworld_approx_vllm256_dump.sh | bash
