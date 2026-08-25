# RTX 5090 单卡复现矩阵

冻结日期：2026-08-25
状态：**single-card approximation（单卡近似）**；不是论文规模复现，也不能与论文表格直接横向比较。

## 结论先行

在一张 RTX 5090（32,607 MiB）上，修复后的 FSDP2/vLLM 路径可以稳定完成短程
ALFWorld 训练循环。Qwen2.5-1.5B、8 步、response length 256 的缩减配置偶尔产生
正奖励：一次是 1/8 episode，一次是 1/16 episode；但 10-update 的四 rollout
复跑在 1,280 条 transition 中没有正奖励。因此目前只能报告“稀疏随机可达性信号”，
不能报告已经学到稳定策略。

论文级 AgentOPSD 复现仍未完成：论文脚本使用 3B/7B、8/2/4 张 GPU、完整数据和
150 个训练周期/步数，本机只完成了环境/接口 gate、3B 一步资源边界 smoke，以及
1.5B 的短程 ALFWorld 近似。

## 固定来源与运行时

| 项目 | 冻结值 |
|---|---|
| repair fork 基线 | `b6831f63f166760876cfb727c00d7b3ac2cee225` |
| 上游参考 | `0c478b2d7cdc201d9b1f076ec5b3dec7e88a161b` |
| FSDP2/vLLM 兼容修复 | 本地分支 `codex/fsdp2-vllm-compat`，commit `7b77b4b31306083994b275e5f7125c2a88aaae36`；只改 FSDP2-safe wrapped-module fallback |
| 操作系统/运行时 | WSL2 Ubuntu 22.04；Python 3.12；CUDA 12.8；PyTorch 2.8.0+cu128；vLLM 0.11.0；flash-attn 2.7.4.post1 |
| GPU | NVIDIA GeForce RTX 5090；32,607 MiB VRAM |
| 1.5B 模型 | Qwen2.5-1.5B-Instruct；SHA-256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee` |
| 记录原则 | 原始 rollout、terminal log、`nvidia-smi` 采样和 manifest 保留在 WSL ext4 `/home/agentopsd/runs`；本仓库只提交可审计摘要、hash 和重跑脚本 |

## 论文设置与实际执行

论文列取自 `examples/agentopsd_trainer/run_*.sh`；“执行结果”只表示本机实际完成的
gate 或近似实验。

| 路径 | 论文/目标设置 | 实际执行设置 | 观测结果 | 证据 | 可声明范围 |
|---|---|---|---|---|---|
| ALFWorld 兼容 gate | Qwen2.5-3B；8 GPU；`max_steps=50`；`n=8`；`max_response_length=512`；150 epochs | Qwen2.5-3B；1 GPU；FSDP2+vLLM；1 train row/1 val row；1 update；`max_steps=1`；response 256；vLLM fraction 0.4 | valid-action 1.0；reward/success 0；物理峰值 32,138 MiB，最低余量 50 MiB | `evidence/alfworld_vllm_gpu04_manifest.json`；WSL `/home/agentopsd/runs/alfworld_approx_vllm256_gpu04/` | 工程兼容性与资源边界，不是训练结果 |
| ALFWorld 短程 A | 同上 | Qwen2.5-1.5B；1 GPU；4 rows × 2 rollouts；4 步；response 128；5 updates | valid-action `0.562–0.969`；reward/success 0；22,514 MiB 峰值 | `evidence/alfworld_single_card_results.md`；WSL `alfworld_approx_1p5b_vllm_5updates/` | 稳定单卡近似，不是论文结果 |
| ALFWorld 短程 B | 同上 | Qwen2.5-1.5B；4 rows × 2 rollouts；8 步；response 128；5 updates | valid-action `0.734–0.922`；reward/success 0；22,562 MiB 峰值 | 同上；WSL `alfworld_approx_1p5b_vllm_h8_5updates/` | horizon follow-up |
| ALFWorld response-256 | 同上 | Qwen2.5-1.5B；4 rows × 2 rollouts；8 步；response 256；10 updates | valid-action `0.844–0.984`；update 8 出现 1/8 成功 episode（score 10） | `evidence/alfworld_response256_followups.json`；WSL `alfworld_approx_1p5b_vllm_h8_response256_10updates/` | 稀疏随机正信号；未证明可复现 |
| ALFWorld n=4 | 同上 | Qwen2.5-1.5B；4 rows × 4 rollouts；8 步；response 256；5 updates | valid-action `0.828–0.969`；update 5 成功 `1/16=0.062`；reward mean 0.625；`pick_and_place=0.25`；23,485 MiB 峰值 | `evidence/alfworld_response256_followups.json`；WSL `alfworld_approx_1p5b_vllm_h8_response256_n4_5updates/` | 稳定完成短程循环，但只是一批次内偶发成功 |
| ALFWorld n=4 复跑 | 同上 | 与上一行相同；10 updates | valid-action `0.828–0.977`；1,280 条 transition 全部 score ≤ 0；23,485 MiB 峰值 | `evidence/alfworld_response256_followups.json`；WSL `alfworld_approx_1p5b_vllm_h8_response256_n4_10updates/` | 负向复现证据；不能宣称学到策略 |
| WebShop 接口 gate | Qwen2.5-3B；2 GPU；完整 WebShop；`max_steps=15`；150 epochs | 独立 Python 3.10 环境；1k 商品子集；本地 100/1k Lucene index；text reset + `search[shoes]` | reset/step 通过 | 计划中的 WebShop smoke manifest（WSL `/home/agentopsd/runs/webshop_text_smoke.json`） | 仅环境接口 gate；非完整 benchmark |
| Search 接口 gate | Qwen2.5-3B；4 GPU；Wikipedia-18 dense retriever；128/512 batch；150 steps | 完整 Search-R1 parquet 已预处理；20 文档 CPU BM25 index/service；`<search>` → `<information>` → `<answer>` | 环境 smoke reward 1.0 | WSL `/home/agentopsd/runs/search_tiny_env_smoke.json` | CPU 接口 gate；非 Wikipedia-18 结果 |

## 结果解释

- 1.5B 分支的 5/10 update 都能跑完请求的 update 数，说明单卡训练循环、环境
  reset/step、rollout 写盘和 FSDP2/vLLM 兼容修复是可用的。
- response length 从 128 提到 256 后才观察到成功 episode，但成功只在一次 10-update
  n=2 run 和一次 5-update n=4 run 中出现；相同 n=4 配置的 10-update 复跑没有任何
  正奖励。这是稀疏且不稳定的随机信号。
- 3B smoke 的 50 MiB 物理余量是停止条件。不能据此启动论文形状的 16×8 或 150-update
  作业；16-row 探针已经在 Ray 主机内存/调度阶段失败。

## 重跑入口

在当前 Windows 工作区的 WSL 中：

```bash
cd /mnt/c/Users/1/Documents/ChatGPT/agentOPSD
bash tools/run_alfworld_approx_1p5b_vllm_h8_response256_n4_5updates.sh
bash tools/run_alfworld_approx_1p5b_vllm_h8_response256_n4_10updates.sh
```

所有 wrapper 都显式标注为 approximation，并把 rollout、Hydra 输出和 telemetry
写入 `/home/agentopsd/runs/<experiment>/`。若要做独立评估，必须先把训练脚本的
`trainer.save_freq=-1` 改为明确的 checkpoint 策略，并单独记录新 seed；当前冻结包
没有可供独立 evaluation 的最终 checkpoint。
