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

评审基线：`main@295ba6a` + PR #8 分支（11/17 上午）。数据与环境证据在 11/17 下午已有更新（Release `data-v1`、`results/env-2026-11-17/`），所以下表只列"证明什么 / 不证明什么"，**不写死哈希**——文件会随迭代变化，哈希请在目标 commit 上现算：

```bash
git cat-file blob <commit>:<path> | sha256sum     # 任一提交里的文件哈希
sha256sum webshop_sft.json                        # 实际输入数据，应为 721e46db…
```

| 证据 | 证明什么 | 不证明什么 |
|---|---|---|
| `results/material-audit.json` | 原始 SFT 1624 行、1624 个 ID、1599 条不同指令、12 个 train/test ID 重叠、8 条 `reward`/`true_reward` 不一致 | 不是模型效果 |
| `results/provisional-split-validation.json` | 离线划分 1451/160/200、排除 13 条、seed 42、ID 与指令哈希交叉为 0 | 不是环境冻结版（见下） |
| `results/offline-split-20261117-zqq/`（PR #11） | 用 Release 副本重跑 `audit` + 两次 `split`，ID 与 manifest 逐字节可复现 | 仍缺环境映射，标 `provisional_offline_only` |
| `results/env-2026-11-17/`（PR #10） | 目标机环境、`/health`、冻结划分、首批 3 条真实 rollout 与 meta | 不含 SFT/B1、无性能结论 |
| `results/dependency-availability.json` | 候选依赖版本在官方源存在 | 不证明互相兼容，也不证明能装上 50 系显卡 |
| `results/upstream-demo-data-check.json` | 上游演示轨迹 78 条、803/1170 个 assistant 回合不可解析、奖励只有 0/1、79/100 次购买前置观测含价格 | 证明它**不能**当本组失败案例或成功率 |
| `results/metrics-definition.json` | 指标 v1.1 的机器可读定义（含 `terminal_reward` 与页面分数的区分），与代码字段一致（有测试守住） | 不含任何结果 |
| `results/failure-cases-b0-smoke.md` | 3 条真实 B0 轨迹的人工复核（loop ×2、format+loop ×1） | n=3，不能推广 |
| `results/validation.md` | 本地 CPU 检查记录（日期化，不随迭代改写） | 不含 GPU / 真实 rollout 证据 |
| `results/experiment-register.csv` | 登记表结构与校验规则；当前只有表头 | 表头不是结果 |
| `docs/EVALUATION.md` | 分母、区间、公平性、失败分诊与人工核对流程 | 不含模型结论 |

数据与模型侧：

- SFT 源文件 sha256：`721e46dba410cff900e187bccea91a456c38b105b6d14d068d7762afb7791de3`（Release `data-v1` 附件 digest；与 `material-audit.json` 的 `sft.sha256`、划分文件的 `source_sha256` 一致）。
- 模型 revision：`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`（Qwen/Qwen2.5-1.5B-Instruct，来自 PR #10 的 `base-model-manifest.json`）。注意 `model.py::load()` 目前不透传 revision，登记表尚无该列，见 Issue #3。
- 环境 goal fingerprint：`e69c9ae2a0aad88977c44a7baf49f1f5ef7918ba8317923c4ae24dcc1f5638d8`（PR #10 的 `/health`，`n_goals=12087`，与冻结划分 manifest 一致）。
- 测试集：`upstream/eval_agent/data/webshop/test_indices.json` 为保留 test（200 任务），按计划 12/08 才首次使用。

## 5. 未验证 / 阻塞

- B0 真实结果：首批 3 条 val 轨迹已出并人工复核（`results/failure-cases-b0-smoke.md`），全量 160 任务待 HowseaonLink 按 11/17 确定的规格执行；B1 要等 SFT adapter。
- 五类真实失败案例：已有人工确认的首批（loop ×2、format+loop ×1），其余类别要等全量 val 与 B1。
- `configs/eval-sft.json` 指向的 adapter 目录不存在，`preflight` 会报错——SFT 产物落地后再跑一次即可。
- 两臂配对比较（`compare`）已实现并有测试，但还没有真实两臂数据可算。
- `model.py::load()` 不透传 `revision`，登记表也还没有 `model_revision` 列；在 Issue #3 有结论前，模型版本只能记在 notes 里。
