# AgentOPSD Repair Fork Handover

## 冻结状态

- Fork 路径：`/work/FCY/agentopsd-upstream-fork`
- 分支：`repair/fork-pilot`
- 冻结 commit：`46cf04924d7478a609bf59998ca093e257c7b060`
- 上游基线：`0c478b2d7cdc201d9b1f076ec5b3dec7e88a161b`
- 当前日期：2026-08-20
- 标签：`fork_engineering_pilot`

该冻结版本是工程修复 fork，不是官方未修改实现，也不是论文复现版本。

## 已修复的上游阻塞

公开上游的 `verl/trainer/ppo/opsd_ray_trainer.py` 无法完成 OPSD 更新：

1. `OPSDRayTrainer.fit()` 调用不存在的 `_compute_teacher_log_probs`。
2. 文件尾部包含未定义的 `actor_outpu`，实际 actor update 路径会失败。

本 fork 在 `verl/trainer/ppo/opsd_ray_trainer.py` 补齐 teacher log-prob 方法和 RLSD 对应的 actor-update/validation/checkpoint 尾部。该修改使训练路径在静态检查中完整，但没有把官方 Ray/FSDP 论文配置跑在本机。

## 单卡工程配置

- GPU：RTX 5070 Ti，16 GB
- 模型：本地 `Qwen/Qwen2.5-1.5B-Instruct`
- 模型 revision：`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- LoRA：rank 8，alpha 16，dropout 0
- 精度：BF16
- 离线约束：`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`local_files_only=True`
- prompt adapter：`qwen_chat_template_single_user_v1`
- chat template SHA-256：`cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`

`examples/agentopsd_trainer/fork_pilot_train.py` 不使用 Ray、FSDP 或 vLLM。它为每个 arm 重新加载相同 base checkpoint 和 LoRA 配置，使用确定性生成 seed；仅在第一个更新前检查 GRPO 与 AgentOPSD 的 completion token fingerprints 相同。

## 任务与比较契约

任务：`fsm-code-v1`

- 3 个 turn
- 4 条 rollout / prompt group
- 动作格式严格为 `ACTION A`、`ACTION B` 或 `ACTION C`
- 非法动作立即结束并获得 0 reward
- 稀疏终局 reward
- student rollout 只使用公开任务 view
- oracle skill 只进入 AgentOPSD 的 teacher log-prob 输入

runner 会拒绝单臂运行、复用已有输出目录、缺少本地模型和缺少 chat template。失败时仍写出 `manifest.json`、`status.json` 和 `terminal.jsonl`。

## 已验证的实验

先前 raw-text prompt smoke 全部产生非法自然语言续写，因此没有形成有效学习信号。该失败模式促成了 Qwen chat-template adapter 修复；不要把该原始 smoke 当作算法结果。

最终 exploratory cohort：

- seeds：17、29、43
- 每个 seed：GRPO 与 AgentOPSD 各 1 个 optimizer update
- 每个 arm：训练 1 个 group，加 4 个 eval task，每个 eval task 4 条 trajectory
- 这是短时工程实验，不是原计划的 120-update 学习曲线

描述性原始结果：

| Arm | Eval successes | Eval trials | Invalid actions | Peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| GRPO | 5 | 48 | 0 | 3409.531 MiB |
| AgentOPSD | 7 | 48 | 0 | 3409.531 MiB |

逐 seed 的 `AgentOPSD - GRPO` eval-success delta 为 `[+1, 0, +1]`。

这些数值只能说明修复 fork 在该合成任务上端到端运行，且该极小 exploratory 样本中 AgentOPSD 得到更多 eval successes。禁止从中推导：

- 论文复现成功
- 统计显著性
- AgentOPSD 普遍优于 GRPO
- 对 ALFWorld、Search、WebShop 或论文 3B/7B 多卡设置的结论

## 证据包

版本化 evidence 位于：

```text
docs/agentopsd/fork_engineering_pilot_evidence/
```

其中包含冻结 cohort manifest、closure summary、每个 seed 的 manifest/status/terminal/GRPO telemetry/AgentOPSD telemetry，以及非自指 `SHA256SUMS.txt`。

校验：

```bash
cd /work/FCY/agentopsd-upstream-fork/docs/agentopsd/fork_engineering_pilot_evidence
sha256sum --check --strict SHA256SUMS.txt
```

预期：18/18 `OK`。

## 本地验证命令

```bash
cd /work/FCY/agentopsd-upstream-fork
/work/FCY/.agentopsd-upstream-smoke/venv/bin/python -m unittest -v \
  tests.agentopsd.test_fork_pilot_fsm \
  tests.agentopsd.test_fork_pilot_train
```

冻结时结果：19 个 focused CPU tests 通过。

## 后续 Agent 约束

1. 不要修改 `/work/FCY/upstream-AgentOPSD`；它保留为只读官方基线。
2. 不要复用任何已有 `outputs/fork_engineering_pilot-*` 目录；runner 会 fail closed。
3. 不要杀死或假设拥有外部 GPU 进程。
4. 若扩大实验，先新建并 hash 一个 cohort manifest，再使用新的输出根。
5. 扩大到多 update 或更多 seed 后，仍需保持相同 checkpoint、task schema、chat-template hash、base rollout seed 表、模型哈希和离线约束。
6. 官方论文复现需要作者完整训练代码、论文规模资产及 2 至 8 GPU；当前机器条件不满足。

## GitHub 发布状态

本地 fork 已冻结并附带完整 handover/evidence。当前 `origin` 仍指向本地官方 checkout，不能直接推送。机器缺少有效 GitHub CLI/SSH/token 认证；GitHub 发布需要先配置一个独立 fork remote 和可用认证，随后推送 `repair/fork-pilot` 与 tag `fork-engineering-pilot-20260820`.
