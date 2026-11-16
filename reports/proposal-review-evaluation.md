# Proposal 评估视角审阅 + 证据索引（E/QiuQiu-xuan）

审阅对象：[proposal 草案](proposal-draft.md)（2026-11-16 版），基线提交 `main@295ba6a` 加上本次评估工具改动。
职责范围：实验可比性、指标口径、数据划分、证据可追溯性。**不包括**叙事、引用格式与页面设计（B/Zqqq-qq 与 Yuxiao-Liu233 主责）。

本文件没有模型结果。所有数字要么是数据审计统计，要么是工具自检结果，文中逐条标明。

## 1. 结论：提交前必须改的 4 处

| # | 位置 | 问题 | 必须改成 |
|---|---|---|---|
| 1 | §4（第 27 行） | 写成 "terminal-reward 与 stepwise-reward" 对比，没有写两臂的 grounding 权重必须相同。当前 `reward-export --mode terminal` 不含 grounding，直接照这句话做实验会把"奖励分配"与"grounding 有无"混在一起 | 明确为 R-TG / R-PG 两臂（终局+grounding 权重 β / 逐步进度+**相同** β），并注明 TG 路径尚未实现、属计划工作 |
| 2 | §4（第 27 行） | "所有模型与上下文、提示、任务列表和训练预算保持可比"不可检查 | 换成可核对的字段清单（模型/量化/采样/上下文/`max_steps`/环境/fingerprint/ID 顺序/repeats/采样与更新预算），并写 `check-parity` 与 `preflight` 是执行手段 |
| 3 | §5（第 31 行） | 只写"满奖励任务成功率"，没写分母、"购买结束但不满分"怎么处理、区间方法 | 补：主指标分母 = 全部尝试 episode；购买结束数与满奖励成功数分开报；单条任务用 Wilson、重复采样用任务簇 bootstrap；每臂写清任务数与区间 |
| 4 | §3（第 19–21 行） | 只写"1,624 条 SFT 记录"与"有 12 个 ID 重叠"，没写实际划分规模与 test 状态 | 补：分组划分 1451 train / 160 val / 200 reserved test，排除 13 条，seed 42，ID 与指令哈希交叉为 0；环境 goal-map 核对仍未完成，不能先写"已隔离" |

## 2. 建议修改（不阻塞提交）

1. **§4 进度估计器命名**：现有线性可加基线不是原论文神经 SPA，草案里"进度估计器"容易被读成已复现论文方法。建议第一次出现就写"可加线性基线（新增）"，并说明神经 SPA 属资源允许时的进一步复现。
2. **§5 进度估计的验证**：写清在**独立 val 任务**上验证、报 `sum(progress)` 与终局奖励的 MSE/误差分布，并承认逐步分数的非唯一性（低 MSE 不等于每步分数有因果意义）。
3. **§6 贡献边界**：把"已完成"与"计划中"分成两小段。目前除数据审计、划分、工具链外，训练与 RL 都还没实机验证。
4. **§7 风险**：已写"若 RL 未通过则收敛范围"，建议补一句：届时明确标注 on-policy 策略梯度分支、不称 PPO、不报 RL 提升。
5. **§8 时间线**：与 [逐日计划](https://github.com/Zqqq-qq/webshop-stepwise-rl/blob/main/docs/DAILY_PLAN.md) 对齐——12/01 真实 RL 硬验收、12/07 冻结实验、12/08–10 最终测试；proposal 里"12 月初完成可行性验证"偏乐观。
6. **失败分析**：§5 已写"报告失败类别"，建议直接列五类（格式/检索/规格/预算/循环）+ 工程故障单独统计，并说明真实案例来自本组 rollout。

## 3. 一个容易误引的数字

`results/material-audit.json` 里 SFT 数据的奖励分布是 `1.0: 969 / 1624`（59.7%）。这是**上游收集数据里满奖励轨迹的占比**，不是本组模型的成功率，也不是 val/test 指标。任何地方引用它都必须写明"数据构成"，不能写成"成功率 59.7%"。

## 4. 证据索引（11/22 交接用）

文件哈希为 sha256 全值；改动任一文件后需重新计算。commit 指本次评审所在的仓库状态，`main@295ba6a` + PR #8 分支。

| 证据 | sha256 | 证明什么 | 不证明什么 |
|---|---|---|---|
| `results/material-audit.json` | `537d682e263be8f01aca85058217e3a899261d6203b8e1f4ac9f7c1132d980b6` | 原始 SFT 1624 行、1624 个 ID、1599 条不同指令、12 个 train/test ID 重叠、12 条 `reward`/`true_reward` 不一致 | 不是模型效果 |
| `results/provisional-split-validation.json` | `e700af7b9fc7daf494bddafc39a2e47144cd1f4e1a8d4ff655f36c32ccdd2bdb` | 1451/160/200 划分、排除 13 条、seed 42、ID 与指令哈希交叉为 0 | 离线划分，未与真实环境 goal-map 核对 |
| `results/dependency-availability.json` | `1d5c348b64c286de2059d1480442904f13d02682c62b68a4834d438742a14174` | 候选依赖版本在官方源存在 | 不证明互相兼容，也不证明能装上 50 系显卡 |
| `results/upstream-demo-data-check.json` | `006200ca2155699f33baa8151a1960a9aba1a4684d730b974a5c4b2e53dec913` | 上游演示轨迹 78 条、803/1170 个 assistant 回合不可解析、奖励只有 0/1、79/100 次购买前置观测含价格 | 证明它**不能**当本组失败案例或成功率 |
| `results/metrics-definition.json` | `02ed87c53359ca059a956333051b4b0b870a89eea1757a28ba1808b0da21ba67` | 指标 v1 的机器可读定义，与代码字段一致（有测试守住） | 不含任何结果 |
| `results/validation.md` | `06c77eb94f57c989d82735f2433c6493fde1ac9cbf62dda2d1f0363909107df0` | 本地 CPU 检查记录：43 项测试、合成 fixture 端到端、分诊修正 | 不含 GPU / 真实 rollout 证据 |
| `results/experiment-register.csv` | `13bd48654f4cc449e8820951062b93e695110f714128777454b8b873c727233f` | 登记表结构与校验规则；当前只有表头 | 表头不是结果 |
| `docs/EVALUATION.md` | `65eeb17704033ac111b6ba85a65ef66943d47fa81bf8cee1a7438d40d9c01f29` | 分母、区间、公平性、失败分诊与人工核对流程 | 不含模型结论 |

数据与模型侧：

- SFT 源文件 sha256：`721e46dba410cff900e187bccea91a456c38b105b6d14d068d7762afb7791de3`（来自 `material-audit.json` 的 `sft.sha256`，与划分文件记录的 `source_sha256` 一致）。
- 模型 revision：**未知**。仓库没有下载权重；首次实机运行必须在 run 记录里补 Hugging Face revision。
- 环境 goal fingerprint：**未知**。/health 尚未在目标机跑过；没有它就不能做最终评估。
- 测试集：`upstream/eval_agent/data/webshop/test_indices.json` 为保留 test（200 任务），按计划 12/08 才首次使用。

## 5. 未验证 / 阻塞

- 真实 B0/B1 val 结果：等 HowseaonLink 的目标机 rollout（需要 `.jsonl` + `.meta.json` + config + `*_ids.json`）。
- 五类真实失败案例与人工复核 3 条轨迹：同上；上游演示轨迹已确认不可替代。
- `configs/eval-sft.json` 指向的 adapter 目录不存在，`preflight` 会报错——SFT 产物落地后再跑一次即可。
- 两臂配对比较（`compare`）已实现并有测试，但还没有真实两臂数据可算。
