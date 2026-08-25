#!/usr/bin/env bash
set -euo pipefail

# Reuse the calibrated command without changing its frozen provenance. The
# paper's training default samples HF rollouts; this wrapper changes only that
# flag and the experiment/Hydra output names.
sed \
  -e 's/actor_rollout_ref.rollout.do_sample=false/actor_rollout_ref.rollout.do_sample=true/' \
  -e 's/qwen25_3b_1gpu_calibration/qwen25_3b_1gpu_sampled/' \
  -e 's@/home/agentopsd/runs/alfworld_approx_calibration/hydra@/home/agentopsd/runs/alfworld_approx_sampled/hydra@' \
  /mnt/c/Users/1/Documents/ChatGPT/agentOPSD/tools/run_alfworld_approx_calibration.sh | bash
