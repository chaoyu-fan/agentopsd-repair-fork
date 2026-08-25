"""Inspect one local Qwen ALFWorld completion with the repository prompt."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO = Path("/home/agentopsd/src/agentopsd-repair")
sys.path.insert(0, str(REPO))
os.environ.setdefault("ALFWORLD_DATA", "/home/agentopsd/data/alfworld")

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment  # noqa: E402
from agent_system.environments.prompts.alfworld import ALFWORLD_TEMPLATE_NO_HIS  # noqa: E402


def main() -> None:
    cfg_path = REPO / "agent_system/environments/env_package/alfworld/configs/config_tw.yaml"
    config = yaml.safe_load(cfg_path.read_text())
    env = get_environment(config["env"]["type"])(config, train_eval="train").init_env(batch_size=1)
    env.seed(0)
    observations, infos = env.reset()
    obs = observations[0]
    admissible = "\n ".join(f"'{x}'" for x in infos["admissible_commands"][0] if x != "help")
    prompt = ALFWORLD_TEMPLATE_NO_HIS.format(
        current_observation=obs,
        admissible_actions=admissible,
    )

    model_path = "/home/agentopsd/models/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    ).cuda().eval()
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    ).cuda()
    with torch.inference_mode():
        output = model.generate(
            inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(output[0, inputs.shape[1] :], skip_special_tokens=False)
    print("PROMPT_TOKENS", inputs.shape[1])
    print("OBSERVATION", obs)
    print("ADMISSIBLE", infos["admissible_commands"][0])
    print("COMPLETION", repr(completion))


if __name__ == "__main__":
    main()
