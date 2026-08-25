# AgentOPSD reproduction plan (approved execution)

Status: approved to proceed through model/environment and single-card functional gates. Those gates are now complete for ALFWorld, WebShop, and a deliberately tiny CPU-only Search retriever path. The corrected FSDP2/vLLM path is functional on one RTX 5090, but its measured physical-memory headroom is only 50 MiB at the tested 3B setting; no long paper experiment is being presented as a reproduction.

## Objective

Reproduce the released AgentOPSD implementation and paper protocol as faithfully as possible on one RTX 5090 (32 GB VRAM), while reporting any deviation from the paper's multi-GPU setup explicitly. Existing `fsm-code-v1` pilot artifacts are engineering evidence only and are excluded from paper claims.

## Final freeze index (2026-08-25)

- [RTX 5090 reproduction matrix](docs/agentopsd/RTX5090_REPRODUCTION_MATRIX_CN.md)
- [RTX 5090 failure ledger](docs/agentopsd/RTX5090_FAILURE_LEDGER_CN.md)
- [RTX 5090 deviation statement](docs/agentopsd/RTX5090_DEVIATIONS_CN.md)
- [RTX 5090 evidence freeze](docs/agentopsd/RTX5090_EVIDENCE_FREEZE_CN.md)

The current package is a single-card approximation. It contains no paper-scale learning
curve and no final checkpoint for independent evaluation.

## Fixed provenance

- Repair fork: `b6831f63f166760876cfb727c00d7b3ac2cee225` (`fork-engineering-pilot-20260820`)
- Upstream reference: `0c478b2d7cdc201d9b1f076ec5b3dec7e88a161b`
- Runtime: WSL2 Ubuntu 22.04, Python 3.12, CUDA Toolkit 12.8, PyTorch 2.8.0+cu128
- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB VRAM
- Formal data/runs: WSL ext4 under `/home/agentopsd/{data,runs,models}`

## Gates and phases

1. **Environment gate (complete):** source hashes, WSL/GPU preflight, PyTorch CUDA smoke, editable project install, vLLM 0.11.0, and flash-attn 2.7.4.post1. Flash-attn was compiled for sm120 only on this GPU.
2. **Core software gate (complete with exceptions recorded):** AgentOPSD/import/protocol tests, CPU utilities, Ray CPU tests, and flash-attn forward/backward smoke. Optional sandbox tests still need `pyext`; one generic scheduler test differs under Torch 2.8 and is not silently patched.
3. **Model/environment gate (complete):** Qwen2.5-3B-Instruct was downloaded through `hf-mirror.com` and verified by SHA-256; ALFWorld is installed and its direct `AlfredTWEnv` plus AgentOPSD Ray wrapper reset/step pass. The three core text archives are persistently retained under `/home/agentopsd/runs/alfworld_archives/`, each passed `unzip -t`, and their hashes are recorded in the smoke manifest. The optional MaskRCNN checkpoint is not required for the current text-only AlfredTWEnv path and is not being treated as installed. WebShop has a separate Python 3.10 environment with the 1k product subset, generated 100/1k Lucene indexes, and a text reset/search smoke. Search-R1 train/test parquet files were downloaded and processed; a 20-document BM25 index and local retriever service pass the Search environment search/information/answer smoke. The WebShop subset and Search tiny index are interface gates only, not benchmark data.
4. **Single-card functional gate (ALFWorld complete):** the local 3B model passed Transformers load/generation and vLLM generation. HF rollout is not claimable for Qwen2 because verl's explicit position IDs caused degenerate repetition. A minimal FSDP2/vLLM compatibility patch was applied on branch `codex/fsdp2-vllm-compat`; the corrected one-row trainer smoke completed one update with `episode/valid_action_ratio=1.0`. At `gpu_memory_utilization=0.4`, direct `nvidia-smi` sampling reached 32,138 MiB used of 32,607 MiB (50 MiB free), so this is an engineering smoke/resource boundary, not a safe long-run setting or a paper result. Lowering the vLLM fraction to 0.2 or 0.3 failed KV-cache initialization; optimizer offload did not reduce the measured high-water mark.
5. **Paper-protocol experiments:** run fixed-seed AgentOPSD and listed baselines on each reachable benchmark. Preserve raw configs, dataset/model hashes, logs, checkpoints, failure artifacts, and GPU-memory/time telemetry. Start with the smallest faithful slice, then scale only if the 5090 remains within safe VRAM and runtime limits.
6. **Evaluation and audit:** use the repository evaluation paths, compare against paper tables/curves, report confidence intervals where multiple seeds are feasible, and label all single-card, reduced-batch, LoRA/quantized, or shortened runs as approximations.
7. **Final handoff:** produce a reproducibility matrix (paper setting / executed setting / deviation / evidence), results tables, failed-run ledger, and exact rerun commands.

## Resource policy on one 5090

- Paper scripts request 8 GPUs (ALFWorld), 2 GPUs (WebShop), and 4 GPUs (Search); those settings cannot be claimed literally on this host.
- First attempt: full-precision 3B single-card inference/training smoke with gradient checkpointing, micro-batching, and no concurrent retriever. 7B is a later feasibility branch only after measured VRAM usage.
- If a paper configuration cannot fit, use the least-distorting change in this order: smaller micro-batch, gradient accumulation, CPU/offload, then parameter-efficient or quantized training. Every change is recorded and prevents a “paper-level” label.
- No W&B credentials are required for local evidence; logging defaults to local console/files unless explicitly configured.
- The ALFWorld smoke reached 29.154 GiB allocated and 31.852 GiB reserved on a 32,607 MiB card. Any benchmark run must use a fresh idle-GPU check and conservative sequence/rollout limits; concurrent model or retriever jobs are prohibited.

## Current gate review (2026-08-22)

| Benchmark/path | Executed setting | Result | Claim status |
|---|---|---|---|
| ALFWorld | Qwen2.5-3B, FSDP2+vLLM, one-row/one-update smoke | valid action ratio 1.0; reward 0; about 50 MiB free at peak | engineering compatibility only |
| WebShop | Python 3.10, 1k products, local 100/1k Lucene index, text reset + `search[shoes]` | reset/step passed | environment interface only |
| Search | full train/test parquet preprocessing; 20-document CPU BM25 index/service; one search + answer | reward 1.0 | environment interface only |
| ALFWorld | Qwen2.5-1.5B, FSDP2+vLLM, 4 rows × 2 rollouts, 4-step horizon, 5 updates | valid action 0.562–0.969; reward/success 0; 22,514 MiB physical peak | stable single-card approximation; not a paper result |
| ALFWorld | Qwen2.5-1.5B, same setup with 8-step horizon, 5 updates | valid action 0.734–0.922; reward/success 0; 22,562 MiB physical peak | stable horizon follow-up; not a paper result |
| ALFWorld | Qwen2.5-1.5B, 4 rows × 2 rollouts, 8-step horizon, response 256, 10 updates | valid action 0.844–0.984; one positive episode at update 8 (score 10; 1/8 episodes); 20.078 GiB allocated / 26.106 GiB reserved | positive stochastic signal; not a paper result |
| ALFWorld | Qwen2.5-1.5B, 4 rows × 4 rollouts, 8-step horizon, response 256, 5 updates | valid action 0.828–0.969; update-5 success 0.062 (1/16 episodes), reward mean 0.625, pick-and-place 0.25; 23,485 MiB physical peak | stable rollout-count follow-up; not a paper result |
| ALFWorld | Qwen2.5-1.5B, 4 rows × 4 rollouts, 8-step horizon, response 256, 10 updates | valid action 0.828–0.977; zero positive score across 1,280 transition records; 23,485 MiB physical peak | negative reproducibility follow-up; not a paper result |

The 1.5B reduced branch is now proven stable for short multi-update runs and can occasionally reach a positive ALFWorld episode, but the ten-update n=4 rerun did not reproduce that episode. This is a sparse stochastic reachability signal, not evidence of a stable learned policy. Full WebShop data, the Wikipedia-18 dense index, paper GPU counts, and 150-update jobs remain unexecuted; any further training should be justified by a changed data/horizon budget rather than presented as a paper result.

## Stop/approval points (resolved for this run)

Before phase 5, review and confirm:

- whether to prioritize ALFWorld first (recommended because it is the paper's clearest end-to-end path), or Search/WebShop;
- whether local-only logging is acceptable (recommended) or a W&B key will be supplied later;
- whether reduced single-card approximations should be run after the faithful 3B smoke, even when they cannot reproduce the paper's GPU count.

User approved the plan on 2026-08-21. Priority is ALFWorld first, then WebShop and Search; local-only logging is acceptable; reduced single-card approximations may be attempted only after the faithful 3B smoke and must remain labeled approximations. The 150-update benchmark jobs remain gated on successful functional smoke and a fresh resource check.

## Evidence ledger (2026-08-22)

- Model: `/home/agentopsd/runs/model_sha256.txt`, `/home/agentopsd/runs/model_transformers_smoke.log`, `/home/agentopsd/runs/model_vllm_smoke.log`.
- ALFWorld archives: `/home/agentopsd/runs/alfworld_archives/sha256.txt` and `unzip_test.txt`; SHA-256 values are also copied into `/home/agentopsd/runs/opsd_functional_smoke/manifest.json`.
- Functional trainer smoke: `/home/agentopsd/runs/opsd_functional_smoke/manifest.json` and `terminal.log`.
- ALFWorld single-card calibration: `/home/agentopsd/runs/alfworld_approx_calibration/manifest.json` and `terminal_retry2.log`; one reduced update completed in 111.985 s with 29.153 GiB allocated and no OOM, but zero reward and zero valid-action ratio.
- 16-row scheduling probes: `/home/agentopsd/runs/alfworld_approx_1p5b_vllm_16x8_3updates/`, `/home/agentopsd/runs/alfworld_approx_1p5b_vllm_16x8_safe_1update/`, `/home/agentopsd/runs/alfworld_approx_1p5b_vllm_16x8_safe_1update_v2/`, and `/home/agentopsd/runs/alfworld_approx_1p5b_vllm_16x8_safe_1update_v3/`; the first hit Ray host-memory pressure and the CPU-reservation variants stalled before model registration. They remain failure evidence, not benchmark results.
- Known compatibility failures retained in the terminal log: FSDP1 with CPU parameter offload hit a compute-device assertion; FSDP1 without offload hit CUDA illegal memory access in `clip_grad_norm_`; a Torch 2.8 scheduler test differs from its historical expectation and was not patched.
- FSDP2/vLLM pre-patch failure: `/home/agentopsd/runs/compatibility/fsdp2_vllm_fail_manifest.json` records the FSDP1-only `_fsdp_wrapped_module` AttributeError. The minimal compatibility change is isolated to `verl/workers/sharding_manager/fsdp_vllm.py` on branch `codex/fsdp2-vllm-compat`.
- Corrected vLLM smoke/resource evidence: `/home/agentopsd/runs/alfworld_approx_vllm256_gpu04/manifest.json`, `terminal_resource_check.log`, `nvidia_smi_resource_check.csv`, and `rollouts/1.jsonl`. The run is valid-action functional but has effectively zero memory headroom; no 150-update or paper-shaped job may start at this setting.
- WebShop environment/data evidence: `/home/agentopsd/runs/webshop_data_sha256.txt` records the three downloaded subset files; `/home/agentopsd/runs/webshop_text_smoke.json` records a deterministic 1k-product reset and `search[shoes]` step. Generated Lucene resources/indexes are local working assets under the WebShop package and are intentionally not committed.
- Search data/preprocessing evidence: `/home/agentopsd/runs/search_r1_raw_sha256.txt` records the Hugging Face train/test parquet hashes; `/home/agentopsd/data/searchR1_processed_direct/{train,test}.parquet` are the processed full files, with `{train,test}_tiny4.parquet` deterministic slices. `/home/agentopsd/data/searchR1_tiny/` contains a 20-document BM25 corpus/index. `/home/agentopsd/runs/search_tiny_env_smoke.json` records the local retriever and Search environment `<search>` → `<information>` → `<answer>` path with reward 1.0. This is a CPU interface smoke and must not be compared with the paper's full Wikipedia-18 dense-retrieval setup.
- FSDP2/vLLM compatibility patch: commit `7b77b4b31306083994b275e5f7125c2a88aaae36` on branch `codex/fsdp2-vllm-compat` changes only `verl/workers/sharding_manager/fsdp_vllm.py`; it is covered by the corrected ALFWorld vLLM smoke above. The generated WebShop resources remain untracked and are not part of this commit.
- Resource probes at vLLM cache fractions 0.2 and 0.3 failed with `No available memory for the cache blocks`; optimizer CPU offload at 0.4 completed but retained the same 36.354-GiB PyTorch high-water mark and was slower. These are retained as negative engineering evidence, not silently discarded.
- Qwen2.5-1.5B-Instruct: `/home/agentopsd/models/Qwen2.5-1.5B-Instruct/model.safetensors` downloaded via ModelScope after a slow HF-mirror partial; SHA-256 is `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`.
- 1.5B five-update reduced run: `/home/agentopsd/runs/alfworld_approx_1p5b_vllm_5updates/terminal.log`, `nvidia_smi.csv`, and `rollouts/1.jsonl`–`5.jsonl`; FSDP2+vLLM completed all 5 updates, valid-action ratios were `0.969, 0.719, 0.562, 0.844, 0.750`, reward/success remained zero, and physical peak was 22,514 MiB (9,674 MiB free).
- 1.5B eight-step follow-up: `/home/agentopsd/runs/alfworld_approx_1p5b_vllm_h8_5updates/terminal.log`, `nvidia_smi.csv`, and `rollouts/1.jsonl`–`5.jsonl`; all 5 updates completed, valid-action ratios were `0.797, 0.734, 0.828, 0.922, 0.734`, reward/success remained zero, and physical peak was 22,562 MiB (9,626 MiB free).
- 1.5B response-256 ten-update follow-up: `/home/agentopsd/runs/alfworld_approx_1p5b_vllm_h8_response256_10updates/rollouts/1.jsonl`–`10.jsonl`; valid-action ratios were `0.844, 0.906, 1.000, 0.891, 0.938, 0.953, 0.984, 0.937, 0.906, 0.984`. Update 8 contains one successful episode represented by seven `score=10` transition records; other updates contain no positive score. PyTorch high-water marks were 20.078 GiB allocated and 26.106 GiB reserved; no physical sampler was attached.
- 1.5B response-256 four-rollout follow-up: `/home/agentopsd/runs/alfworld_approx_1p5b_vllm_h8_response256_n4_5updates/rollouts/1.jsonl`–`5.jsonl` and `nvidia_smi.csv`; valid-action ratios were `0.828, 0.930, 0.969, 0.953, 0.937`. Update 5 reached success `0.062` (1/16 episodes), reward mean `0.625`, and pick-and-place success `0.25`; physical peak was 23,485 MiB (8,703 MiB free), with 20.084 GiB allocated and 26.106 GiB reserved.
- 1.5B response-256 four-rollout ten-update rerun: `/home/agentopsd/runs/alfworld_approx_1p5b_vllm_h8_response256_n4_10updates/terminal.log`, `nvidia_smi.csv`, and `rollouts/1.jsonl`–`10.jsonl`; valid-action ratios were `0.828, 0.930, 0.953, 0.961, 0.938, 0.953, 0.945, 0.969, 0.953, 0.977`, and all 1,280 transition records had score at most zero. Physical peak was 23,485 MiB (8,703 MiB free); PyTorch high-water marks were 20.084 GiB allocated and 26.106 GiB reserved.

The calibration indicates that a full paper-shaped 16×8 ALFWorld update at 2048/512 context would exceed the available single-card safety margin. The next ALFWorld run must therefore remain a reduced approximation (or use a separately justified memory-saving branch), with no claim of strict paper reproduction.

The structured summary and model hash for the response-256 follow-ups are also recorded in
`evidence/alfworld_response256_followups.json`. Re-run wrappers are under `tools/` and
write raw outputs to `/home/agentopsd/runs/<experiment>/`.
