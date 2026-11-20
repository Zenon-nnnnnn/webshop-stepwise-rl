# Proposal 评估视角审阅 + 证据索引（E/QiuQiu-xuan）

审阅对象：[proposal 草案](proposal-draft.md)。11/19 快照基于 `main@21f3c83`（PR #8 已合并）；11/20 对照了 `main@aa09199` 的草案与 [PR #13 的待合并草案](https://github.com/Zqqq-qq/webshop-stepwise-rl/blob/8fc00a7cf2572fc78eafc8f17c8a649f435cc52b/reports/proposal-draft.md)。
职责范围：实验可比性、指标口径、数据划分、证据可追溯性。**不包括**叙事、引用格式与页面设计（Zqqq-qq 与 Yuxiao-Liu233 主责）。

第 6 节引用已复核的 full-160 候选验证结果，供 proposal 和 pre 按原口径引用。

## 0. 11/19 复核结论（先看这里）

`main` 上的草案仍待补齐下节四项。PR #13（`8fc00a7`）的待合并版本已更新 §4 的进度监督目标，以及 §5 的成功定义、平均终局任务奖励与页面分诊断口径。§4 已明确购买取购买评分、未购买取 0；§5 已明确购买完成且满奖励才计为成功。下节列出仍需整合的内容。
- 第 4 条（划分规模与 test 状态）的证据已在 **PR #15** 落地：真实 goal-map 重划后 1451/160/200、排除 13、真实指令零交叉；proposal §3 仍只写"1,624 条"和"12 个 ID 重叠"。
- 模型 revision 已由 **PR #14** 支持钉版，评估侧登记表也加了 `model_revision`；当前 `main` 的采集器尚未写入 `model_source`。

## 1. 结论：提交前必须改的 4 处（含 11/19 状态）

| # | 位置 | `main` 草案的缺口 | 11/22 前待补 | PR #13（`8fc00a7`）与相关证据 |
|---|---|---|---|---|
| 1 | §4 | 缺少两臂共享的辅助奖励规则 | 写明 R-TG（终局任务奖励）与 R-PG（逐步进度奖励）共用同一动作合法性奖励公式和权重。说明带共用辅助项的 TG 路径仍待实现，辅助项权重待冻结 | 草案已更新进度估计器的终局任务奖励目标；实验协议已列共同辅助项 |
| 2 | §4 | “保持可比”缺少可核对字段 | 列出共享的初始化、算法、模型/revision、量化、提示词、采样配置、上下文、`max_steps`、环境/fingerprint、任务 ID 顺序、repeats 和预算。评估配置用 `preflight` 和 `check-parity` 核对，训练预算另按运行日志核对 | 草案仍为通用表述；协议和 `configs/eval-*-frozen.json` 可供核对 |
| 3 | §5 | 成功分母、购买计数和区间方法待定义 | 完整运行以全部尝试的 episode 为分母，每臂报告任务数、episode 数、购买数、满分成功数及区间。每任务一次时报告单臂 Wilson 95% 区间，重复采样时按任务簇 bootstrap；两臂差值使用配对任务 bootstrap | 草案已补购买且满奖励才算成功、平均终局任务奖励及页面分仅供诊断；分母、购买单列和区间仍待补 |
| 4 | §3 | 缺少按真实环境映射验证后的划分规模 | 写明 1451 train / 160 val / 200 reserved test、排除 13、seed 42、ID 与指令跨集合零交叉；test 只在 12/07 冻结后使用 | §3 沿用 `main` 表述；PR #15 提供真实 goal-map 的验证证据 |

## 2. 建议修改（不阻塞提交，按 11/19 状态更新）

1. **§4 进度估计器命名**：线性可加基线不是原论文神经 SPA，建议第一次出现写"可加线性基线（新增）"，神经 SPA 标为资源允许时的进一步复现。**PR #13 已明确线性基线优先、神经 SPA 待评估**，可再注明线性基线为本组新增。
2. **§5 进度估计的验证**：PR #13 草案已写独立任务的总分 MSE；建议补充误差分布，并说明逐步归因的非唯一性。
3. **§6 贡献边界**：把"已完成"与"计划中"分开。**PR #13 部分改**（§7 风险段）。
4. **§7 风险**：明确 on-policy 策略梯度分支、不称 PPO、不报 RL 提升。**PR #13 已改**。
5. **§8 时间线**：与逐日计划对齐（12/01 硬验收、12/07 冻结、12/08–10 最终 test）。**PR #13 已改**。
6. **失败分析**：列出五类 + 工程故障单独统计，说明真实案例来自本组 rollout。**协议已列；正文未改**。

## 3. 一个容易误引的数字

`results/material-audit.json` 里 SFT 数据的奖励分布是 `1.0: 969 / 1624`（59.7%）。这是**上游收集数据里满奖励轨迹的占比**，不是本组模型的成功率，也不是 val/test 指标。引用时必须写"数据构成"。

## 4. 证据索引（11/22 交接用）

基线 `main@21f3c83`。哈希按下面的命令在目标 commit 上现算；Windows 原生检出会转 CRLF，以 git blob 为准：

```bash
git cat-file blob <commit>:<path> | sha256sum
git cat-file blob HEAD:results/eval-frozen-20261120/val_ids.json | sha256sum   # 45440d72…
```

| 证据 | 证明什么 | 不证明什么 |
|---|---|---|
| `results/material-audit.json` | 原始 SFT 1624 行、1624 个 ID、1599 条不同指令、12 个 train/test ID 重叠、8 条 `reward`/`true_reward` 不一致 | 不是模型效果 |
| `results/offline-split-20261117-zqq/`（PR #11） | Release 副本重跑 `audit` + 两次 `split`，ID 与 manifest 逐字节可复现 | 仍缺环境映射（当时） |
| `results/mapped-split-20261119-zqq/`（PR #15） | 用真实 `goal-instructions.json` 重划：1451/160/200、排除 13、真实指令零交叉；manifest 与环境侧一致 | 未启 live 环境；正式训练前仍需 `/health` |
| `results/env-2026-11-17/`（PR #10） | 目标机环境、`/health`、冻结划分、首批 3 条真实 rollout（含重生 summary） | 不含 SFT/B1、无性能结论 |
| `results/qwen-base-pilot-2026-11-19/`（PR #16） | Qwen2.5 与 Qwen3 在 8 个 val 任务上的选型试跑，两臂均 0 满分 | **不是受控方法对照**，不能当能力结论，未登记 |
| `results/eval-frozen-20261120/` | 160 任务 val 清单（sha `45440d72…`）、goal fingerprint、两份冻结配置哈希 | 不含 test、不含结果 |
| `results/metrics-definition.json` | 指标 v1.1（含 `terminal_reward` 与页面分数区分），有测试守住 | 不含结果 |
| `results/failure-cases-b0-smoke.md` | 3 条真实 B0 轨迹的人工复核（loop ×2、format+loop ×1） | n=3，不能推广 |
| `results/experiment-register.csv` | 登记表结构；当前只有表头 | 表头不是结果 |
| `docs/EVALUATION.md` | 分母、区间、公平性、失败分诊、登记硬门槛 | 不含模型结论 |

数据与模型侧：

- SFT 源文件 sha256：`721e46dba410cff900e187bccea91a456c38b105b6d14d068d7762afb7791de3`（Release `data-v1` digest；与 `material-audit.json`、划分 manifest 的 `source_sha256` 一致）。
- 模型 revision：`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。PR #14 已支持 config 透传 revision 并写 `meta.model_source`；使用旧采集器生成的 metadata 缺少该字段；旧运行的模型来源须另行核对，见 §6。
- 环境 goal fingerprint：`e69c9ae2a0aad88977c44a7baf49f1f5ef7918ba8317923c4ae24dcc1f5638d8`（`n_goals=12087`，与冻结划分 manifest 一致）。
- 冻结 val 清单：`results/eval-frozen-20261120/val_ids.json`，sha256 `45440d724b33e69e13916d0adbe243cc63ce3bd67677b8220feff9f30b866c36`。
- 测试集：`upstream/eval_agent/data/webshop/test_indices.json` 为保留 test（200 任务），12/08 才首次使用。

## 5. 未验证 / 阻塞（11/19 快照，11/20 更新见 §6）

- **全量 160 任务 B0 未跑**：目前只有 3 条 smoke 与 8 任务 pilot；真实 val 表、区间、B0/B1 对照都还没有。
- **B1 adapter 未产出**：`check-parity` 无法做两臂比对；`preflight` 在 B1 冻结配置上只报这一项。
- **五类真实失败案例只覆盖两类**：现有 3 条真实轨迹是 loop 与 format；budget / specification / retrieval 仍无本组真实案例。
- **PR #13 / #14 未合并**：协议的正文修改与 revision 透传还没进 main；在那之前 proposal 正文与 meta 都不完整。
- 两臂配对比较（`compare`）已实现并有测试，但还没有真实两臂数据可算。

## 6. 11/20 更新：候选结果已出，但正式登记仍未完成

11/19 的快照已被下面的事实部分取代，保留原文供追溯：

| 11/19 的说法 | 11/20 的实际状态 |
|---|---|
| 全量 160 任务 B0 未跑 | **已跑**：B0 与本地 1 epoch LoRA 在同一批 160 个冻结 val 任务上各 160 条；满分成功 19/160 与 50/160，配对差 +19.38pp，任务簇 95% CI +13.11 至 +26.25（[PR #18 的运行证据](https://github.com/Zqqq-qq/webshop-stepwise-rl/pull/18#issuecomment-7026403338)及[固定归档快照](https://github.com/Zqqq-qq/webshop-stepwise-rl/tree/ac95bded9b398e785452278e17b50f7803b85d8f/results/eval-20261119-full160)） |
| B1 adapter 未产出 | **本地候选 adapter 已产出**并重载一致（`93760afb…`）；但不是正式冻结 SFT 配置的产物 |
| `check-parity` 无法比对 | **已比对通过**：两臂同模型/revision/采样/环境指纹，仅 adapter 不同 |

**但正式登记仍未完成**，因此上述 19 vs 50 只能写成"候选验证结果"，不能写成正式 B1 成绩：

- 路径 A（沿用旧运行）：采样源码 `dd9797e` 已在私有仓库可查。须关联实际环境 wrapper、独立核对底模与 adapter 来源、采用实际执行配置及其哈希，再用原始 JSONL 完成 `check-run`、`check-parity` 和登记检查；
- 路径 B（重采）：在审定源码与冻结配置下重采 160 个任务，并完成同样的原始轨迹和登记检查；
- 两臂 meta 没有 `model_source`（PR #14 未合并）；执行配置与 `configs/eval-b0-frozen.json` 有四个字段差异（`model` 本地路径、`environment_url=8766`、`model_id`、`weights_sha256`）。

**proposal 可以怎么用**：作为"160 个冻结 val 任务上的候选对照"引用，并写明本地 1 epoch LoRA、非正式冻结、test 未用、未登记；沿用旧运行的登记以团队审定的方法和配置为准；正式 B1 结果还需满足正式训练的验收。

其余仍未变：五类真实失败案例只覆盖 loop / format（full-160 只提供脱敏 per-task 记录，没有动作级日志）。PR #13 / #14 仍待合并，proposal 技术整合可按 §1 的剩余清单推进。
