# ALFWorld single-card approximation results

These runs are engineering approximations of AgentOPSD on one RTX 5090. They
must not be reported as reproductions of the paper's 8-GPU, 150-update setup.

| Run | Model/engine | Horizon | Updates | Valid-action ratio | Reward / success | Physical GPU peak |
|---|---|---:|---:|---:|---:|---:|
| `alfworld_approx_1p5b_vllm_5updates` | Qwen2.5-1.5B, FSDP2 + vLLM | 4 | 5 | 0.562–0.969 | 0 / 0 | 22,514 MiB; 9,674 MiB free |
| `alfworld_approx_1p5b_vllm_h8_5updates` | Qwen2.5-1.5B, FSDP2 + vLLM | 8 | 5 | 0.734–0.922 | 0 / 0 | 22,562 MiB; 9,626 MiB free |
| `alfworld_approx_1p5b_vllm_h8_response256_10updates` | Qwen2.5-1.5B, FSDP2 + vLLM | 8 | 10 | 0.844–0.984 | one positive episode at update 8 (score 10; 1/8 rollout episodes); other updates 0 | logical 20.078 GiB allocated; 26.106 GiB reserved; physical sampler not attached |
| `alfworld_approx_1p5b_vllm_h8_response256_n4_5updates` | Qwen2.5-1.5B, FSDP2 + vLLM | 8 | 5 | 0.828–0.969 | update 5: success 0.062 (1/16 episodes), reward mean 0.625; `pick_and_place` 0.25 | 23,485 MiB; 8,703 MiB free |
| `alfworld_approx_1p5b_vllm_h8_response256_n4_10updates` | Qwen2.5-1.5B, FSDP2 + vLLM | 8 | 10 | 0.828–0.977 | 0 / 0 across 1,280 transition records | 23,485 MiB; 8,703 MiB free |

All four runs completed their requested updates without GPU OOM. The first
two used response length 128 and remained at zero reward. Increasing the
response budget to 256 produced a first positive episode at update 8; using
four rollouts per row produced a measured 1/16 episode success at update 5.
These are stochastic engineering signals, not a paper benchmark: the runs use
four placeholder rows, one GPU, short horizons, and at most 16 episodes per
update, while the paper uses substantially larger data and training budgets.
The n=4 runs' physical GPU sampler reached 23,485 MiB used (8,703 MiB free);
their PyTorch high-water mark was 20.084 GiB allocated and 26.106 GiB reserved.
The five-update n=4 positive episode was not reproduced by the ten-update
rerun (all 1,280 transition records had score at most zero), so it is best
treated as a sparse stochastic reachability signal rather than evidence of a
stable learned policy.

Raw evidence is retained in WSL ext4:

The structured summary and model hash for the two response-256 follow-ups are
also recorded in [alfworld_response256_followups.json](alfworld_response256_followups.json).

- `/home/agentopsd/runs/alfworld_approx_1p5b_vllm_5updates/`
- `/home/agentopsd/runs/alfworld_approx_1p5b_vllm_h8_5updates/`
- `/home/agentopsd/runs/alfworld_approx_1p5b_vllm_h8_response256_10updates/`
- `/home/agentopsd/runs/alfworld_approx_1p5b_vllm_h8_response256_n4_5updates/`
- `/home/agentopsd/runs/alfworld_approx_1p5b_vllm_h8_response256_n4_10updates/`

Re-run wrappers are [run_alfworld_approx_1p5b_vllm.sh](../tools/run_alfworld_approx_1p5b_vllm.sh), [run_alfworld_approx_1p5b_vllm_h8.sh](../tools/run_alfworld_approx_1p5b_vllm_h8.sh), [run_alfworld_approx_1p5b_vllm_h8_response256_10updates.sh](../tools/run_alfworld_approx_1p5b_vllm_h8_response256_10updates.sh), [run_alfworld_approx_1p5b_vllm_h8_response256_n4_5updates.sh](../tools/run_alfworld_approx_1p5b_vllm_h8_response256_n4_5updates.sh), and [run_alfworld_approx_1p5b_vllm_h8_response256_n4_10updates.sh](../tools/run_alfworld_approx_1p5b_vllm_h8_response256_n4_10updates.sh).
