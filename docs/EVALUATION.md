# 统一评估、结果登记与失败分析

> 现状（2026-11-16）：评估工具已实现并通过 CPU 测试；**尚无真实 B0/B1 val 日志，本文不含任何模型成功率**。文中出现的数字只用于说明字段定义，全部来自 `tests/` 的合成 fixture，不得写进 proposal、pre 或 report。真实日志到位后按第 6 节执行，并把证据登记到 [结果登记表](../results/experiment-register.csv)。

配套文件：[实验手册](EXPERIMENTS.md) 第 5、6 节是操作入口；[数据指南](DATA.md) 定义划分与 SHA；[分工](TEAM.md) 定义谁在哪个节点交付什么。本文件只负责"怎样算一个可信的数字"。

## 0. 角色与时间点

| 时间 | 谁 | 做什么 | 产出 |
|---|---|---|---|
| 现在 | E（QiuQiu-xuan） | 指标定义、评估脚本、登记表结构、失败分诊规则 | 本文件 + `src/webshop_lab/evaluation.py` 等 |
| 11/25–26 | A/HowseaonLink 执行，C/Jun040504 提供 checkpoint | 用冻结配置在**同一 val 列表**上跑 B0/B1 | `.jsonl` + `.meta.json` 原样交回 |
| 收到日志当天 | E | `check-run` 核对分母，`evaluate` 出指标，人工核对案例 | `artifacts/<run>-evaluation.json`、案例表 |
| 12/10 前 | E + B/Zqqq-qq | 冻结配置后才跑 test；确认 B0/B1 只差 adapter | 冻结声明 + 最终登记行 |
| 12/11–14 | E | 从登记表生成图与表，写不确定性与限制 | report 第 6、7 节素材 |

## 1. 指标定义

### 1.1 购买结束 ≠ 成功

WebShop 在点击 `Buy Now` 时就返回 `done=True`，但最终奖励可以是 0 到 1 之间的任何值：买错规格、买超预算、买错商品类型都会"结束但不成功"。上游演示里已有把 `点击 Buy Now` 直接当成功的写法，本项目不采用。因此：

- `done=True` 只表示**购买结束**；
- **成功**必须是 `done=True` 且 `final_reward >= 1 - 1e-6`；
- 两者之差 `purchase_without_full_reward` 是独立报告的量，不能合并进成功率。

对应代码：`evaluation.is_success()` 与 `evaluation.summarize_episodes()`；`tests/test_evaluation.py::MetricTests` 固定了该行为。

### 1.2 字段与分母

| 字段 | 定义 | 分母 |
|---|---|---|
| `episodes` | 已完成的 episode 条数（= 任务数 × repeats） | — |
| `unique_tasks` | 不同 `task_id` 数 | — |
| `purchase_terminated` | `done=True` 的 episode 数 | episodes |
| `full_reward_successes`（别名 `successes`） | `done=True` 且 `final_reward >= 1-1e-6` | episodes |
| `purchase_without_full_reward` | `purchase_terminated - full_reward_successes` | episodes |
| `max_steps_terminated` | `episodes - purchase_terminated` | episodes |
| `success_rate` | **主指标**：`full_reward_successes / episodes`，失败也进分母 | episodes |
| `conditional_success_given_purchase` | `successes / purchase_terminated`，仅诊断"买了但没买对"占比 | purchase_terminated |
| `mean_final_reward` | 所有 episode 的 `final_reward` 均值，含失败 | episodes |
| `mean_steps` | 实际交互轮数均值，非法动作也消耗步数 | episodes |
| `invalid_action_rate` | `grounding=0` 的步数 / 总步数 | steps |
| `grounding_accuracy` | `1 - invalid_action_rate` | steps |
| `episodes_with_invalid_actions` | 至少出现一次非法动作的 episode 数 | episodes |
| `termination_counts` | `purchase` / `max_steps` 计数 | episodes |
| `success_rate_ci95` | 见 1.3 | — |

网络、服务、上下文溢出属于**运行错误**：`rollout` 会直接中断而不是写成失败 episode（见 [实验手册](EXPERIMENTS.md) 第 2 节）。若日志里出现 `engineering` 类案例，说明有人手工拼接或改过文件，必须回到原始日志重跑。

### 1.3 不确定性

- 每个任务只有 1 条 episode：报 95% Wilson 区间（`common.wilson`）。
- 同一任务重复采样（`--repeats > 1`）：轨迹不独立，**不报 Wilson**，改用按任务簇重采样的 bootstrap（2000 次，seed=42，`evaluation.cluster_bootstrap_ci`），并注明用的是簇 bootstrap。
- 规模感：n=160、p=0.5 时 Wilson 半宽约 ±7.7 个百分点；n=20 时约 ±20 个百分点；n=6 时约 ±31 个百分点。**少数任务上的成功差异不要写成"提升"**。
- 同一测试集上多次调参后的最优值不是无偏估计；冻结配置后只报告一次 test 结果。

## 2. 运行完整性核对

`rollout` 在采集前写入同名前缀的 `.meta.json`，其中 `selected_ids`、`repeats`、`config_sha256`、`ids_sha256`、`health.goal_fingerprint` 是核对依据。`summarize` 不会替你检查完整性，必须单独跑：

```bash
webshop-lab check-run --meta artifacts/b0-val.meta.json --jsonl artifacts/b0-val.jsonl \
  --config configs/eval-base.json --output artifacts/b0-val-audit.json
```

判定：

| 状态 | 含义 | 处理 |
|---|---|---|
| `complete` | 期望的 (task, repeat) 集合与 JSONL 完全一致 | 可以汇总 |
| `partial` | 缺少部分 episode（中断、手工删除） | 默认拒绝汇总；需 `--allow-partial` 且结果标"partial" |
| `mismatched` | 多了任务、重复了 (task, repeat)、`run_id` 或 fingerprint 不一致 | 视为无效，重跑 |

人工核对清单（每次收到日志都做一遍，约 5 分钟）：

1. `check-run` 的 `expected_episodes` 是否等于"任务数 × repeats"，且等于你被告知的任务数；
2. JSONL 首尾两行是否属于同一 `run_id`，`goal_fingerprint` 是否一致；
3. 抽 1 个成功、1 个"购买但非满奖励"、1 个 `max_steps`，打开原始 JSONL 逐步读动作与观测，确认与指标一致（不只看汇总数字）；
4. `config_sha256` 与给的 config 文件哈希一致（`evaluate --config` 会自动比对，被改过会给出 warning）。

## 3. B0/B1 公平性核对

只允许**一个**差异：`adapter`（B1 的 LoRA 产物）。其余字段必须完全相同：

| 字段 | 含义 |
|---|---|
| `model` | 基础模型名（含 revision 时以日志为准） |
| `load_in_4bit` | 量化方式 |
| `seed`、`temperature`、`top_p` | 采样设置（greedy 时 temperature=0） |
| `context_length`、`max_new_tokens` | 上下文与生成预算 |
| `max_steps` | 动作上限 |
| `environment_url` | 同一环境服务与 goal 顺序 |
| `goal_fingerprint` | 冻结的环境目标映射 |
| `selected_ids`（含顺序）、`repeats` | 同一 val 列表、同一次数 |

系统提示词与动作解析在代码里（`model.SYSTEM`、`common.parse_action`），所以两侧还必须来自**同一 commit**；不同 commit 的 B0/B1 不能对比。

```bash
webshop-lab check-parity --a-meta artifacts/b0-val.meta.json --b-meta artifacts/b1-val.meta.json \
  --require-frozen --output artifacts/b0-b1-parity.json
```

退出码非 0 或 `blocking_issues` 非空时，不得把这对结果写进报告。`goal_fingerprint` 为空只允许出现在开发阶段；正式评估前必须从 `/health` 填进配置与 meta。

## 4. 失败分类

`evaluate --cases` 会按下面的优先级给失败 episode 打标签，并保留该 episode 的全部动作、观测、奖励与错误信息。

| 优先级 | 类别 | 判定规则（自动初筛） |
|---|---|---|
| 1 | `engineering` | 步骤里出现连接/超时/服务错误，或 `termination` 不是 `purchase`/`max_steps` |
| 2 | `format` | 没有唯一可解析的 `Action:` 行（多动作、缺括号、截断），或动作解析成功但不在当前 `available` 列表 |
| 3 | `budget` | 已购买且 `0<reward<1`，购买页 `Price: $x` 超过指令的 `price lower than` 预算 |
| 4 | `specification` | 已购买、`0<reward<1`，价格在预算内 → 规格/选项不符 |
| 5 | `loop` | `max_steps` 结束且同一动作重复 ≥3 次，或同一"动作+观测"重复 ≥2 次 |
| 6 | `retrieval` | 非购买结束，且所有 search 关键词与指令内容词零交集 |
| 7 | `other` | 没有规则命中 → 必须人工看日志，不代表模型没问题 |

已知局限（写报告时必须承认）：

- 这是**规则初筛**，不是人工标注。`retrieval` 依赖分词，"换个近义词就搜到"的正确行为可能被误判；`budget` 依赖页面上的 `Price: $x` 文本，自定义渲染会失效。
- 一条 episode 可能同时具备多种问题，只按优先级记一个主类；案例表里保留了全部步骤，供人工改判。
- 成功 episode 不进入分类；`engineering` 不计入方法差异，单独列出。

人工核对流程：从 `--cases` 生成的工作表里每类挑 1–2 条，打开 JSONL 原文确认"动作确实长这样、观测确实是这个页面、奖励确实是这个数"，然后把确认过的案例抄进 `results/failure-cases-<method>.md` 并在报告里引用。没有核对过的自动标签不能直接进报告。

## 5. 结果登记与图表

登记表列（顺序固定，`check-register` 会校验表头）：

```text
run_id,status,method,model,seed,split_sha256,goal_fingerprint,git_commit,config_path,config_sha256,
episodes,successes,purchase_terminated,success_rate,success_rate_ci95,mean_reward,mean_steps,
invalid_action_rate,peak_vram_gb,evidence_path,notes
```

规则：

- 一行 = 一次真实 run。`status` 取 `planned/running/complete/failed`；只有 `complete` 行可进报告。
- `complete` 行必须同时具备：`split_sha256`（所用 `*_ids.json` 的 sha256）、`goal_fingerprint`、`git_commit`（本仓库存在的 commit）、`config_path` + `config_sha256`、`episodes/successes/purchase_terminated`、`success_rate` 与区间、`evidence_path`。
- `successes <= purchase_terminated <= episodes`，且 `success_rate` 必须等于 `successes/episodes`（1e-6 容差），区间必须包含该值。
- 目前只有表头，`check-register` 通过是正常状态；**表头不是结果**。

一条命令从评估直接生成登记行（避免手抄误差）：

```bash
webshop-lab evaluate --method B1 \
  --meta artifacts/b1-val.meta.json --jsonl artifacts/b1-val.jsonl --config configs/eval-sft.json \
  --split-ids data/processed/val_ids.json \
  --output artifacts/b1-val-evaluation.json --cases artifacts/b1-val-cases.md \
  --figures-dir artifacts/figures --register-row artifacts/b1-val-row.csv --peak-vram-gb 7.8
```

然后人工把 `artifacts/b1-val-row.csv` 的那一行贴进登记表，并跑：

```bash
webshop-lab check-register --register results/experiment-register.csv
webshop-lab figures --register results/experiment-register.csv --output results/success-by-method.svg
```

结果表模板（等真实数据填，不要预填）：

| method | 模型 | 任务数 | 购买结束 | 满奖励成功 | success_rate (95% CI) | invalid action rate | 备注 |
|---|---|---|---|---|---|---|---|
| B0 |  |  |  |  |  |  |  |
| B1 |  |  |  |  |  |  |  |

`figures` 只画 `status=complete` 的行；没有真实行时它拒绝出图，避免出现"占位图被当成结果"。

## 6. 命令清单

目标机（A/HowseaonLink 执行，按 [实验手册](EXPERIMENTS.md) 第 4 节）：

```bash
webshop-lab rollout --config configs/eval-base.json --ids data/processed/val_ids.json --output artifacts/b0-val.jsonl
webshop-lab rollout --config configs/eval-sft.json  --ids data/processed/val_ids.json --output artifacts/b1-val.jsonl
```

复核机（E，CPU 即可）：

```bash
webshop-lab check-run    --meta artifacts/b0-val.meta.json --jsonl artifacts/b0-val.jsonl --config configs/eval-base.json
webshop-lab check-parity --a-meta artifacts/b0-val.meta.json --b-meta artifacts/b1-val.meta.json --require-frozen
webshop-lab evaluate     --method B0 --meta artifacts/b0-val.meta.json --jsonl artifacts/b0-val.jsonl \
                         --config configs/eval-base.json --split-ids data/processed/val_ids.json \
                         --output artifacts/b0-val-evaluation.json --cases artifacts/b0-val-cases.md \
                         --figures-dir artifacts/figures --register-row artifacts/b0-val-row.csv
webshop-lab check-register --register results/experiment-register.csv
webshop-lab check-register --register artifacts/b0-val-row.csv   # 贴入前先校验单行
webshop-lab figures      --register results/experiment-register.csv --output results/success-by-method.svg
```

## 7. 当前缺口

- 没有真实 B0/B1 val 日志，因此没有成功率、没有区间、没有真实失败案例；本文第 1 节的数字只描述字段。
- `rollout` 与真实 WebShop 服务的联调、`--repeats` 采样预算、test 集一次性评估都依赖目标机，尚未执行。
- 图表目前是纯 SVG（无 matplotlib），方法对比图只有一个维度（success rate）；若后面要画奖励分布或步数分布，再扩展 `figures.py` 并补测试。
- 自动化只覆盖"能算对"：任务与指令是否对齐、模型是否真的加载了 adapter，仍要靠 `doctor`、环境 `/health` 与人工抽查。

## 8. 代码与测试入口

| 内容 | 位置 |
|---|---|
| 完整性核对、指标、失败分诊、公平性 | `src/webshop_lab/evaluation.py` |
| 登记表校验与登记行生成 | `src/webshop_lab/register.py` |
| SVG 图表 | `src/webshop_lab/figures.py` |
| 单元测试（32 项，含端到端 fixture） | `tests/test_core.py`、`tests/test_evaluation.py` |
