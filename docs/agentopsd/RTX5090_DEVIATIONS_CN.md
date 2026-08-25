# 与 AgentOPSD 论文协议的偏差说明（冻结版）

这份说明定义“为什么当前结果不能叫论文复现”。所有偏差都是有意显式记录的，
没有在脚本里静默替换后再用论文配置命名。

| 维度 | 论文/仓库协议 | RTX 5090 实际执行 | 影响 |
|---|---|---|---|
| GPU 数量 | ALFWorld 8；WebShop 2；Search 4 | 全部单 GPU；3B 仅做 1-row/1-update smoke | 并行度、显存分配、吞吐和优化噪声不同 |
| 模型规模 | 主要脚本为 Qwen2.5-3B，另有 7B | 可持续短训使用 Qwen2.5-1.5B；3B 只到资源边界 | 参数量和能力不同，不能比较绝对成功率 |
| ALFWorld 数据 | 官方 text preprocessing，训练/验证批量 16/128 | 4 个 placeholder parquet row；环境 reset 提供实际任务 | 不是论文离线数据规模，样本覆盖极小 |
| WebShop 数据 | 完整 WebShop 资源和商品索引 | 1k 商品子集；本地生成 100/1k Lucene index | 只能做 reset/search 接口 gate |
| Search 数据/检索 | Wikipedia-18 dense retriever；4 GPU 服务 | 20 文档 CPU BM25 index；本地 service | 检索分布和服务性能完全不同 |
| ALFWorld horizon | `env.max_steps=50` | 4 或 8；3B smoke 为 1 | 很多任务无法在短 horizon 完成 |
| WebShop/Search horizon | WebShop 15；Search 4 | WebShop interface 只走 reset + 一次 search；Search tiny smoke | 非完整训练轨迹 |
| response/context | ALFWorld response 512、prompt 2048；WebShop/Search response 512、prompt 4096 | 1.5B 分支 response 128/256，prompt 1152/2048；3B smoke response 256 | 动作闭合概率和显存占用改变 |
| rollout group | 论文脚本 `env.rollout.n=8` | 2 或 4 | advantage/group statistics 方差不同 |
| 训练长度 | ALFWorld/WebShop 150 epochs；Search 150 steps | 1-step smoke；5 或 10 updates | 没有论文学习曲线或最终策略 |
| batch / mini-batch | 论文 batch 16/128（Search 128/512），mini-batch 64/256 | 4 rows；mini-batch 4；micro-batch 1 | 梯度估计和数据覆盖显著缩小 |
| rollout engine | 目标路径 vLLM；论文 cache fraction 0.5/0.6 | 修复后使用 vLLM；1.5B fraction 0.35；3B 可运行 probe 为 0.4 | cache 预算和并行设置改变 |
| sharding/代码 | 官方脚本使用仓库默认路径 | 本机需要 FSDP2/vLLM wrapped-module fallback；HF Qwen rollout 另有 position-id 退化 | 这是兼容性工程修复，不应隐藏为算法改动 |
| precision/offload | 论文脚本 BF16、默认不做 actor param/optimizer offload | 1.5B 仍 BF16；尝试过 optimizer/param offload；未降低 3B 峰值 | 负向资源探针不改变主结论 |
| logging | 论文默认 console + W&B | 本机 `WANDB_DISABLED=1`，只保存本地 logs/JSONL/CSV | 没有 W&B 曲线，但原始本地证据可审计 |
| checkpoint/eval | 论文脚本设置测试频率，理论上可在训练后评测 | approximation wrapper `save_freq=-1`、`test_freq=-1` | 没有最终 checkpoint，不能做独立 evaluation |
| 统计报告 | 论文表格使用完整 benchmark 和多次实验 | 当前只有短程单次/少数 follow-up，没有 CI 或显著性 | 结果只能作工程可行性信号 |

## 不允许的表述

- “复现了 AgentOPSD 论文结果”
- “达到论文 ALFWorld/WebShop/Search 成功率”
- “单卡结果证明 AgentOPSD 优于 GRPO”
- “一次 1/8 或 1/16 成功代表模型学会了策略”

## 允许的表述

> 在 RTX 5090 上，我们完成了 AgentOPSD repair fork 的环境 gate、FSDP2/vLLM
> 单卡兼容性验证，以及 Qwen2.5-1.5B 的短程 ALFWorld single-card approximation。
> 该近似偶尔产生成功 episode，但在相同 n=4 配置的 10-update 复跑中未重现，
> 因而不能作为论文复现或稳定策略学习的证据。
