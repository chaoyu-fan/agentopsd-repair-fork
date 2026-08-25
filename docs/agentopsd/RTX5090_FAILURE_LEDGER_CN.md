# RTX 5090 失败记录（冻结版）

冻结日期：2026-08-25。失败项保留用于解释停止条件和复现边界；没有把失败运行
从摘要中删除，也没有把失败改写成“未运行”。原始日志仍在 WSL ext4 的对应目录。

| ID | 阶段 | 现象 | 根因/判断 | 处理与当前状态 | 原始证据 |
|---|---|---|---|---|---|
| F-01 | FSDP2/vLLM 3B smoke | `AttributeError: FSDPQwen2ForCausalLM object has no attribute _fsdp_wrapped_module` | sharding manager 使用了 FSDP1-only wrapped-module 属性 | 兼容分支用 FSDP2-safe `getattr(..., self.module)` fallback；修复后 smoke 通过；没有改算法或数据 | `evidence/fsdp2_vllm_fail_manifest.json`；`/home/agentopsd/runs/compatibility/fsdp2_vllm_fail_manifest.json` |
| F-02 | HF rollout | Qwen2 生成出现重复/退化续写 | verl 显式 `position_ids` 与 Qwen2 generate 的 RoPE/packing 组合不兼容 | HF 路径不纳入最终训练结论；保留 `tools/position_ids_probe.py` 与 patch 作为诊断材料 | `/home/agentopsd/runs/alfworld_approx_response256_dump/rollouts/1.jsonl`；`tools/position_ids_probe.py` |
| F-03 | vLLM cache fraction | fraction 0.2/0.3 在 KV cache 初始化时报 `No available memory for the cache blocks` | FSDP 参数、optimizer、vLLM cache 共存后，低 fraction 无法满足最小 cache block | 不再将 0.2/0.3 当作可运行配置；失败结果保留 | `/home/agentopsd/runs/alfworld_approx_vllm256_memsafe/`；`/home/agentopsd/runs/alfworld_approx_vllm256_gpu03/` |
| F-04 | 3B 资源边界 | fraction 0.4 可完成 1 update，但峰值 32,138/32,607 MiB，仅 50 MiB free | 单卡物理显存没有安全 headroom | 只作为工程 smoke；禁止启动长训或论文 batch | `evidence/alfworld_vllm_gpu04_manifest.json`；`/home/agentopsd/runs/alfworld_approx_vllm256_gpu04/` |
| F-05 | 16-row ALFWorld probe | 16×8 变体在 Ray worker/主机内存或调度阶段失败/停滞，未形成有效 benchmark update | 不是 GPU OOM，而是 Ray host-RAM 放大或 CPU reservation 与现有资源不匹配 | 四次 probe 全部标记为 failure evidence；不再扩大 batch | `/home/agentopsd/runs/alfworld_approx_1p5b_vllm_16x8_3updates/`、`..._safe_1update/`、`..._v2/`、`..._v3/` |
| F-06 | FSDP1 + CPU parameter offload | compute-device assertion | 旧 FSDP1/offload 路径与当前 Torch/CUDA 组合不兼容 | 不作为论文路径修补；切换到 FSDP2 | 交接 terminal log（WSL `/home/agentopsd/runs`） |
| F-07 | FSDP1 no offload | `CUDA illegal memory access` 出现在 `clip_grad_norm_` | 旧 FSDP1 flatten/gradient 路径的运行时错误 | 不作为论文路径修补；切换到 FSDP2 | 交接 terminal log（WSL `/home/agentopsd/runs`） |
| F-08 | optimizer offload probe | 可完成但 PyTorch high-water mark 仍约 36.354 GiB，且更慢 | optimizer offload 没有降低共同占用峰值 | 作为负向资源证据保留；不声称解决显存问题 | `/home/agentopsd/runs/alfworld_approx_vllm256_opt_offload/` |
| F-09 | ALFWorld reward | response-128 的 5-update/h8 分支 reward/success 全为 0 | 短 horizon/短 response 不能稳定产生完整动作序列 | 增加到 response-256 后只出现偶发成功；仍不构成稳定学习证据 | `evidence/alfworld_single_card_results.md` |
| F-10 | stochastic follow-up | n=4、5-update 的 1/16 成功没有在相同 n=4、10-update 复跑中重现 | 采样和极小数据造成高方差/稀疏奖励 | 将该现象降级为 reachability signal；不计算论文式均值或显著性 | `evidence/alfworld_response256_followups.json` |
| F-11 | checkpoint/evaluation | 所有短程 wrapper 使用 `trainer.save_freq=-1`；没有最终 checkpoint 可独立评估 | 任务目标是先验证循环与资源边界，未进入正式 eval 阶段 | 冻结包明确记录该缺口；若继续实验，必须新建 seed 和 checkpoint 目录 | `tools/run_alfworld_approx_*.sh` |
| F-12 | WebShop data | 只有 1k 商品子集和本地生成 index | 完整商品数据/论文资源没有在本机冻结 | 仅报告 reset/step interface gate，不报告成功率 | WSL `/home/agentopsd/runs/webshop_text_smoke.json` |
| F-13 | Search retriever | 只有 20 文档 CPU BM25；没有 Wikipedia-18 dense index | 单卡资源和 retriever 资产不足以运行论文服务 | 仅报告 environment smoke reward 1.0，不与论文 Search 表格比较 | WSL `/home/agentopsd/runs/search_tiny_env_smoke.json` |

## 失败判定原则

1. 进程能启动不等于结果有效：必须同时有完整 update、环境 reward、rollout 文件和
   telemetry。
2. HTTP/接口 smoke 只证明接口连通；没有完整数据和论文评估协议时，不升级为 benchmark。
3. 资源边界（如 50 MiB free）按失败/停止条件处理，即使该步没有抛出 OOM。
4. 任何未保存 checkpoint 的训练只能证明训练循环执行，不能证明可以独立复评。
