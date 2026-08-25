#!/usr/bin/env bash
set -euo pipefail
sed 's/max_new_tokens=64/max_new_tokens=256/' \
  /mnt/c/Users/1/Documents/ChatGPT/agentOPSD/tools/alfworld_generation_probe.py | \
  /home/agentopsd/bin/micromamba run -n agentopsd python -
