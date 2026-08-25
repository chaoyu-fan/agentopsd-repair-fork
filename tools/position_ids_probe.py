"""Compare Qwen generation with and without verl's explicit position_ids."""

from __future__ import annotations

import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    record = json.loads(
        open("/home/agentopsd/runs/alfworld_approx_response256_dump/rollouts/1.jsonl", encoding="utf-8").readline()
    )
    text = record["input"]
    tokenizer = AutoTokenizer.from_pretrained("/home/agentopsd/models/Qwen2.5-3B-Instruct", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        "/home/agentopsd/models/Qwen2.5-3B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    ).cuda().eval()
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs.input_ids.cuda()
    attention_mask = inputs.attention_mask.cuda()
    position_ids = torch.clamp(torch.cumsum(attention_mask, dim=-1) - 1, min=0)
    with torch.inference_mode():
        for label, kwargs in [("implicit", {}), ("explicit", {"position_ids": position_ids})]:
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                **kwargs,
            )
            completion = tokenizer.decode(out[0, input_ids.shape[1] :], skip_special_tokens=False)
            print(label, repr(completion))


if __name__ == "__main__":
    main()
