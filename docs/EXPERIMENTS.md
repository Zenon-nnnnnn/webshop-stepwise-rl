# 实验操作手册

所有命令从仓库根目录执行。不要同时运行两个使用单状态环境服务的采集任务。先按 [安装](SETUP.md) 与 [数据](DATA.md) 完成资产、映射和划分。

## 1. 实验矩阵与公平比较

| ID | 方法 | 状态 | 用来回答什么 |
|---|---|---|---|
| B0 | 基础 instruct 模型，同一购物提示词 | 待目标机运行 | 未任务微调的起点 |
| B1 | 相同模型的 QLoRA SFT | 待目标机运行 | SFT 是否有效 |
| P0 | 线性可加进度预测 | CPU 合成例已验证；真实数据待运行 | 低成本终局预测基线 |
| R0 | SFT + terminal-reward RL | 上游代码保留，适配与验证待完成 | 稀疏奖励对照 |
| R1 | SFT + stepwise-reward RL | 上游代码保留，适配与验证待完成 | 逐步奖励效果 |

最新主实验矩阵见 [逐日计划](DAILY_PLAN.md)：终局/逐步 RL 两臂固定相同 grounding，分别记为 R-TG / R-PG。该 TG 奖励路径和新 RL 入口仍需开发验证，不能把现有 reward-export 的 terminal 输出当成已完成控制。

B0/B1 的基础模型、量化、提示词、上下文、动作上限和任务列表保持一致。R0/R1 还要控制采样预算、更新次数及初始化 checkpoint。不能将不同模型或不同训练预算混为奖励方法差异。

默认配置是 Qwen 1.5B 的资源受限实验。使用 Llama 3B 时应复制并修改一整套训练、评估和探索配置；不能只改 SFT 而忘记 rollout。

## 2. 先做环境与基础模型小检查

终端 1，CPU 环境：

```bash
conda activate webshop-env
webshop-lab env-server
```

终端 2，GPU 环境，首先确认 DATA 中的最终划分已经完成：

```bash
conda activate webshop-gpu
webshop-lab doctor --output artifacts/doctor-gpu.json
webshop-lab rollout --config configs/eval-base.json --ids data/processed/val_ids.json --limit 3 --output artifacts/b0-smoke.jsonl
webshop-lab summarize --input artifacts/b0-smoke.jsonl --output artifacts/b0-smoke-summary.json
```

预期产生 3 条 JSONL episode 及同名前缀的 `.meta.json`。即使全部失败也应有可检查的动作与反馈。网络失败、服务失败、上下文无法容纳当前观测属于运行错误，会中断，不能当作模型任务失败偷偷计入分母。

服务启动时的指令映射必须与 SFT 对齐。工具在 health 上检查 alignment 和 split_status；任何失败先按 [故障排查](TROUBLESHOOTING.md) 处理。

## 3. SFT 训练

先用独立输出目录做 2-step smoke。复制 `configs/sft-qwen15b.json` 为本地 `artifacts/sft-smoke-config.json`，将 `output_dir` 改为 `artifacts/sft-smoke-seed42`。然后：

```bash
webshop-lab sft --config artifacts/sft-smoke-config.json --max-steps 2
```

确认日志中 trainable 参数为 LoRA 参数、loss 有限、adapter 已保存。`max-steps=2` 指 2 次优化更新，存在梯度累积，因此不等于只读取 2 个样本；训练结束会计算验证 loss。

正式训练：

```bash
webshop-lab sft --config configs/sft-qwen15b.json
```

默认：1 epoch、micro-batch=1、梯度累积=8、上下文 2048、LoRA r=8、q/v 投影、学习率 2e-4、4-bit NF4 基础模型。它们是起始配置，不是最优参数承诺。

处理方式：去掉原始对话的固定 `OK` 回合，以统一任务系统提示构造对话；对每个 assistant 回合建立一个例子，仅在回复 token 上计算 loss。历史超长时删除最早完整交互，同时在 system 中保留购物要求。若当前观测与回复本身也超长，会记录 skipped 数量；不能静默忽略大量样本后继续称为全量训练。

产物：

```text
artifacts/sft-qwen15b-seed42/
  adapter/          # LoRA adapter + tokenizer
  checkpoint-*/    # Trainer checkpoint，最多保留两份
  config.json
  run.json         # 输入 SHA、样本数、skipped、GPU、显存峰值、Git commit、验证指标
```

新增工具没有实现训练断点自动恢复，重新运行应使用新的输出目录。它拒绝覆盖非空输出目录。

## 4. SFT 评估与探索

先检查 `configs/eval-sft.json` 的 `adapter` 指向正式训练产物：

```bash
webshop-lab rollout --config configs/eval-sft.json --ids data/processed/val_ids.json --limit 3 --output artifacts/b1-smoke.jsonl
webshop-lab rollout --config configs/eval-base.json --ids data/processed/val_ids.json --output artifacts/b0-val.jsonl
webshop-lab rollout --config configs/eval-sft.json --ids data/processed/val_ids.json --output artifacts/b1-val.jsonl
webshop-lab summarize --input artifacts/b0-val.jsonl --output artifacts/b0-val-summary.json
webshop-lab summarize --input artifacts/b1-val.jsonl --output artifacts/b1-val-summary.json
```

开发阶段只看 val。确认配置后，把服务返回的 `goal_fingerprint` 填入正式评估配置；这会阻止目标映射改变后继续混用结果。最终才将 `--ids` 改成 `data/processed/test_ids.json`，分别运行 B0/B1，且不再根据结果调参。

探索使用相同 SFT adapter、temperature=0.7，并明确任务集合：

```bash
webshop-lab rollout --config configs/explore-sft.json --ids data/processed/train_ids.json --limit 50 --repeats 3 --output artifacts/explore-train-small.jsonl
webshop-lab rollout --config configs/explore-sft.json --ids data/processed/val_ids.json --limit 20 --repeats 3 --output artifacts/explore-val-small.jsonl
```

50 / 20 / 3 只是流程起始规模，不保证足够训练奖励模型。先检查成功/失败分布；若全部 0，优先修复策略格式、模型加载和任务匹配，而不是不断拟合无信息数据。探索数据由 train tasks 采集，进度模型用独立 val tasks 验证；测试任务禁止进入训练或奖励拟合。

## 5. 指标与结果记录

- **Success rate**：`done=true` 且最终奖励 ≥ `1-1e-6` 的任务比例。
- **Mean final reward**：全部已完成记录的最终奖励均值，包括失败。注意 `final_reward` 对未购买的 episode 是页面匹配分，任务奖励要看 `mean_terminal_reward`（见 [统一评估](EVALUATION.md) 第 1.1 节）。
- **Mean steps**：每个 episode 实际交互轮数均值，非法动作也消耗步数。
- **Invalid action rate**：所有尝试动作中格式错误或当前不可执行的比例。
- **Grounding accuracy**：1 − invalid action rate。
- **95% Wilson interval**：每个任务仅一条 episode 时提供；有重复任务时不将轨迹视为独立样本，改用按任务簇的 bootstrap。

`done=true` 只表示购买结束，不等于成功：买错规格或超预算也会结束但奖励小于 1。购买结束数、满奖励成功数与两者之差必须分开记录，分母、区间与失败分诊规则见 [统一评估与失败分析](EVALUATION.md)。

greedy 测试不需要重复同一任务来扩大样本量。多种子训练与任务内重复采样是两种不同的不确定性来源。较小任务集上的少数成功差异不应被夸大。

每次运行把证据登记到 `results/experiment-register.csv`：run ID、状态、方法、模型、seed、split SHA、goal fingerprint、commit、配置、任务数、指标、显存和备注。空白表头不是结果。

正式汇总前核对 `.meta.json` 的 selected_ids 与 JSONL 完成列表一致；中断产生的部分文件不可冒充完整任务集。当前 summarize 不替代这项完整性审核，用 `webshop-lab check-run`（见 [统一评估](EVALUATION.md) 第 2 节）确认分母后再汇总。

## 6. 失败案例分析

至少选若干真实案例，保留任务、脱敏指令、每步动作及观测：

1. 输出格式错误或多动作；
2. 检索关键词不匹配；
3. 选对商品类型但规格错误；
4. 忽略预算；
5. 无效循环或步数耗尽；
6. 环境/模型加载错误——单独列为工程故障。

给每种方法用同一套分类。自动初筛用 `webshop-lab evaluate --cases` 生成案例工作表，分类规则、优先级与人工复核流程见 [统一评估](EVALUATION.md) 第 4 节；未人工确认的标签不能进报告。展示界面并非必须，完整日志与对照案例优先。

## 7. 奖励实验入口

见 [RL 与进度估计说明](RL.md)。`reward-export` 只生成数据，不会训练策略。不能把该命令执行成功写成“PPO 已跑通”。
