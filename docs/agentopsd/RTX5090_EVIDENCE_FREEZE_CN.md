# RTX 5090 证据冻结说明

冻结时间：2026-08-25
冻结对象：当前工作区中可公开、可审计、可重跑的 AgentOPSD 单卡近似证据。

## 冻结包内容

- `REPRODUCTION_PLAN.md`：执行计划、环境 gate、资源边界和完整 evidence ledger。
- `evidence/alfworld_single_card_results.md`：ALFWorld 五组单卡结果摘要。
- `evidence/alfworld_response256_followups.json`：response-256 follow-up 的结构化指标。
- `evidence/alfworld_vllm_gpu04_manifest.json`：3B FSDP2/vLLM 资源边界 manifest。
- `evidence/fsdp2_vllm_fail_manifest.json`：兼容修复前的失败 manifest。
- `tools/`：placeholder fixture、诊断 probe 和所有已执行 approximation wrapper。
- 本目录的 `RTX5090_REPRODUCTION_MATRIX_CN.md`、`RTX5090_FAILURE_LEDGER_CN.md`、
  `RTX5090_DEVIATIONS_CN.md`：最终审计入口。

## 原始数据位置

完整 rollout、terminal log、Hydra 输出和 `nvidia-smi` CSV 没有复制进 GitHub：它们
位于 WSL ext4 `/home/agentopsd/runs/`，单次运行可能包含大量 JSONL，且依赖本机路径。
摘要中的每一项都给出对应 run directory；这些目录应作为原始证据保留，不要覆盖或
合并不同实验目录。

模型和环境资产也不提交：

- Qwen2.5-1.5B/3B 权重
- ALFWorld archives
- WebShop 生成的 `resources*`/Lucene index
- Search Wikipedia-18 dense index
- 本地 cache 和 parquet 下载文件

它们的 hash、获取路径和是否完整已记录在 `REPRODUCTION_PLAN.md` 与 manifest 中。

## 校验与重跑

在提交后，使用下面命令校验冻结包（`SHA256SUMS.txt` 是本目录清单，不包含清单自身）：

```bash
cd /mnt/c/Users/1/Documents/ChatGPT/agentOPSD
sha256sum --check --strict evidence/SHA256SUMS.txt
```

在当前机器重跑一个近似：

```bash
cd /mnt/c/Users/1/Documents/ChatGPT/agentOPSD
bash tools/run_alfworld_approx_1p5b_vllm_h8_response256_n4_5updates.sh
```

重跑前必须确认 GPU 空闲；新结果写入新的 `/home/agentopsd/runs/<id>/`，不能复用
冻结目录。任何新 seed 都应新增 manifest 和 failure entry，而不是修改本冻结摘要。

## 冻结后的结论

这次交付完成的是“可审计的单卡近似证据包”，不是论文级复现。继续工作时应先
建立 checkpoint + 独立 evaluation + 多 seed 方案，再决定是否值得消耗更多单卡时间。
