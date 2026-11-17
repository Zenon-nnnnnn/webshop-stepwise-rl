# 首批真实轨迹失败案例（B0 smoke，n=3）

> 这是 **B0 基线 smoke**，不是实验结果：3 个 val 任务、单种子、1 条轨迹/任务，未经 SFT。用于满足 11/19「人工复核 3 条真实轨迹及错误分类」，不能推广为模型能力结论，也不进入 `results/experiment-register.csv`。

## 来源与可核对信息

| 项 | 值 |
|---|---|
| 分支/PR | `env/setup-howseaonlink` / PR #10 |
| 目录 | `results/env-2026-11-17/` |
| 轨迹 | `rollout-3val.jsonl`（3 episode × 10 步，含每步 observation / available / action / grounding / error / environment_reward） |
| metadata | `rollout-3val.meta.json`：`run_id=1ed67684-a2b2-4fcf-b1a3-2385653ec7a6`、`selected_ids=[16,75,104]`、`repeats=1` |
| 配置 | `rollout-base-3val.json`（`configs/eval-base.json` + 填入 fingerprint），`temperature=0`、`max_steps=10` |
| 环境 | `goal_fingerprint=e69c9ae2…5638d8`；`alignment_checked=1624`、`mismatches=0`、`split_status.verified=true` |
| 划分 | 冻结 val 160 个任务，本次取前 3 个：16 / 75 / 104 |

复核方式：逐条读原始 JSONL 的动作与观测，再用 `webshop-lab evaluate` 复核计数，最后人工确认分类。两套结果一致；下列判读已人工核对，不是自动标签。

## 3 条轨迹的人工判读

| task | 指令要点 | 终止 | 步数 | 非法步 | `final_reward` | 主标签 | 次要信号 |
|---|---|---|---|---|---|---|---|
| 16 | 25+ 小时、tin voyager、<$50 的无铅香薰蜡烛 | max_steps | 10 | 0 | 0.75 | `loop` | — |
| 75 | 3×4 fl oz 绿色染发剂、<$60 | max_steps | 10 | 8 | 1.0 | `format` | `loop` |
| 104 | 白色、女款、万圣节 T 恤、包邮 | max_steps | 10 | 0 | 1.0 | `loop` | — |

三条共同点：**`buy now` 在每一步的 `available.clickables` 里都存在**（最后一次可见的 clickable 数分别是 19 / 50 / 27），但一次都没被点击；`done` 全为 false、`termination` 全为 `max_steps`。

### task 16：卡在选项页，重复同一动作

- 动作序列：`search[...]` → `click[b091d87qmm]` → `click[25+ hour]` → `click[sea change]` ×7。
- 页面匹配分：0.63 → 1.0 → 1.0 → 0.75（后 7 步稳定 0.75）。
- `final_reward=0.75` 是最后一步的页面分，不是任务完成分。
- 判读：`loop` 成立（同一动作连续 7 次、观测不变），且**不是环境不可行**——`buy now` 一直可用。

### task 75：8/10 步动作不可执行，且停止在旧页面分上

- 动作序列：`search[...]` → `click[b01ipzcpxq]` → `click[4 fl. oz (pack of 3)]` ×2 → `click[size: 4 fl. oz (pack of 3)]` ×6。
- 后 8 步全部 `grounding=0`、`error="Action is not in available actions"`：环境里的实际可点击值是 `4 fl oz (pack of 3)`，模型自行加了 `size:` 前缀，并把 `fl oz` 写成 `fl. oz`。
- 页面分：0.71 → 1.0 → 0（第 3 步起全 0）。
- **`final_reward=1.0`**：`rollout` 只在动作合法时更新 `reward`，非法步不覆盖，所以它保留了第 2 步的 1.0。这比"停在 0.75 的页面"更能说明问题——连续失败 8 步，记录下来的却是满分。
- 判读：主标签 `format`（8/10 步不可执行，是最主要的失败原因），次要信号 `loop`（同一非法动作重复 6 次）。

### task 104：页面分满分但从未购买

- 动作序列：`search[...]` → `click[b08cj2bw1g]` → `click[women]` → `click[white]` ×7。
- 页面匹配分从第 2 步起就是 1.0，`final_reward=1.0`，但 `done=false`。
- 判读：`loop`（同一动作连续 7 次）。这条最能说明"页面匹配分 ≠ 任务奖励"：属性都点对了，但没有下单。

## 和指标口径的关系

- 3 条全部 `max_steps`、购买 0 次：`success_rate=0`、`mean_terminal_reward=0.0`。
- 但 `mean_final_reward=0.9167`（0.75 + 1.0 + 1.0 再除以 3）——它是"最后记录到的页面分"，不是任务得分。评估侧已把 `terminal_reward` 与 `final_reward` 分开（`results/metrics-definition.json` v1.1，PR #8），登记表 `mean_reward` 用 terminal 值。
- 若把 `final_reward` 直接当终局回报（进度模型目标、terminal 奖励臂），模型会学到"停在高分页面"而不是"下单买对"；本案例就是反例。

## 局限与后续

- 3 个任务、1 条轨迹/任务、单种子、未做 SFT；只能说明"这 3 条没有购买"，不能说明 B0 的整体成功率。
- 自动分诊的 `loop` 阈值（同动作 ≥3 次或同动作+观测 ≥2 次）在 10 步轨迹上很容易触发，引用分类计数时要同时给步数上限。
- 等全量 160 个 val 任务的 B0 跑完，再用同样流程更新本文件与失败分类统计；B1 需要 SFT adapter，`check-parity` 才能比对。
