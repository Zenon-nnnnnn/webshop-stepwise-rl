# 每日交付：2026-11-17 / QiuQiu-xuan

对应 [逐日计划](https://github.com/Zqqq-qq/webshop-stepwise-rl/blob/main/docs/DAILY_PLAN.md) 11/17 的评估角色任务："手算成功率/平均奖励/非法率；交独立指标核对样例"。

## 今天要解决的问题

1. 用一份**手算**过的小样本，把"分母是谁、购买与成功怎么分、非法率按步还是按 episode"固定下来，并让两套代码都能复现。
2. 核实 issue #3 里 HowseaonLink 提的两个与评估/训练实现有关的问题：`generate` 有没有显式传 `attention_mask`；`generation_config.json` 的采样参数在 greedy 下被忽略会不会影响 `check-parity` 的口径。
3. 确认 11/19「人工复核 3 条真实轨迹」的依赖链是否仍然阻塞。

## 学到并能解释的内容

**WebShop 的奖励不是一个二值信号（L2）。** 买到商品（`done=True`）时，奖励由"商品类型/必需属性/可选属性/价格"的覆盖情况算出，可能是 1.0、0.6，也可能是 0。所以一条"买了但尺寸错"的轨迹是"购买结束、非成功"，必须在指标里同时体现：`purchase_terminated` 记它，`full_reward_successes` 不记它。手算样例 `results/hand-check-metrics.md` 里 task 102 就是这种情况。

**分母决定结论。** 4 条样本里 2 条购买、1 条成功：按"全部 episode"算成功率是 0.25，按"购买数"算是 0.5。后者会把"没买成"的任务悄悄从分母里删掉，正是要避免的高估。

**小样本区间为什么必须报。** n=4、k=1 的 Wilson 95% 区间是 `[0.0456, 0.6994]`，宽 0.65；160 个任务、p=0.5 时半宽约 ±7.7 个百分点。只写"成功率 25%"而不给任务数与区间，读者无法判断差异是否有意义。

**为什么 mask 必须显式传。** Qwen2.5 的 `pad_token_id` 等于 `eos_token_id`，transformers 无法从 input_ids 推断 padding 边界；训练里若把 pad 位置算进 loss 或 attention，会静默出错。仓库里两条路径其实都已显式传：`rollout.py` 第 82 行给单条序列传 `torch.ones_like(input_ids)`；`sft.py` 每条样本带 `[1]*len(ids)`，`collate` 再把 mask 按 0 补齐、`labels` 按 -100 补齐（第 43、54–55 行）。

**greedy 下采样参数被忽略是预期行为。** `generation_config.json` 自带 `temperature/top_p/top_k`，`do_sample=False` 时 transformers 会忽略它们并打印警告；`rollout` 用 `temperature` 决定 `do_sample`，所以 B0/B1 的 greedy 对照只要写清"temperature=0、seed 42"即可。若改用采样，必须把 `top_k` 也写进 config，否则 parity 只能比到 `temperature`/`top_p`（今天已把 `top_k` 加进比对字段）。

## 实际完成与证据

- 新增 `tests/fixtures/hand-check-episodes.jsonl` + `hand-check.meta.json`：4 条合成 episode（1 条满奖励、1 条规格错、2 条步数耗尽且各含 1 个非法步）。
- 新增 `results/hand-check-metrics.md`：逐条数据、手算过程、Wilson 代入、两套代码复算结果。
- 新增 `tests/test_hand_check.py`：把手算数字写成断言（0.25 / 0.4 / 2.25 / 2÷9 / 0.5 / Wilson 边界），口径变了测试会失败。
- `scripts/verify_metrics.py` 与 `evaluate` 在该 fixtures 上 10/10 字段一致；手算、`evaluate`、独立实现三者相同。
- `check-parity` 的比对字段加入 `top_k`；`docs/EVALUATION.md` 补了采样口径与 `generation_config` 继承的说明。
- 本地检查：`python -m unittest discover -s tests` → **45 项通过**；`compileall`、`check_repository.py`、`check-register` 均通过。
- 未验证：任何真实模型输出；4 条合成样本不能当结果。

## 观察、错误与判断

1. **手算与代码 0 差异**，但顺带发现 task 103 因为重复 `search[red running shoes]` 被记了 `loop` 次要信号（`any_counts=1`）。这说明在只有 3 步的短轨迹上"同动作重复 ≥3 或同动作+观测 ≥2"很容易触发；报告里引用 loop 计数时需要连同步数上限一起说。
2. **issue #3 的第 2 条担心不成立**：`rollout.py` 与 `sft.py` 都显式传了 `attention_mask`。HowseaonLink 看到的警告来自他加载 smoke 时的手写调用，不是这两个模块。但有一个真实隐患要留给他和 Zqqq-qq：`rollout.py` 现在用 `torch.ones_like`，**只在 batch=1 且无 padding 时成立**；以后若要批量 rollout，必须按 `input_ids != pad` 构造 mask。
3. **11/19 仍然阻塞**：依赖链是"资产包 → env-server → rollout 日志"，目前资产包未到（issue #1 第 3 条、PR #9），所以真实轨迹的人工复核无法开始。演示轨迹已证明不能替代（`results/upstream-demo-data-check.json`）。

## 交接

- 给 HowseaonLink / Zqqq-qq：mask 与采样口径的核实结论（带行号）已发在 issue #3；`goal_fingerprint` 产出后请同步给我，我跑 `check-parity --require-frozen` 并冻结评估配置。
- 给 Jun040504：`sft.py` 的 mask/label 补齐逻辑（第 43、54–55 行）可作为他 11/18–19「token/label mask 检查」的对照；`run.json` 里已记录 `skipped_*` 与 `train_examples`，可直接用于 skipped 数量验收。
- 给 Yuxiao-Liu233：指标口径的一页说明可直接用 `results/hand-check-metrics.md` 第 1–3 节，图注写"任务数 + 区间"。
- 给 Zqqq-qq：proposal 的指标段可以直接引用 `results/metrics-definition.json` 与这份手算样例。

## 补充：评审 PR #9 / #10 / #11 与真实日志的第一课

下午的评审请求来自三个 PR，我用独立复现而不是读描述的方式核对：

- **PR #11（Zqqq-qq 的离线审计与候选划分）：通过。** 我从私有 Release `data-v1` 下载 `webshop_sft.json`（9,312,587 字节，SHA-256 `721e46db…`）重跑 `audit` 与两次 `split --seed 42`：audit 输出与其 `audit.json` 完全一致；`train_ids` / `val_ids` / `test_ids` / `split-manifest` 的 **git blob 哈希与提交文件逐一相同**，两次划分也逐字节一致（1451 / 160 / 200，排除 13）。`1746` 与 `5413` 确实只差预算（380 vs 370 美元），但这两个字符串没有进仓库，建议补进 README 以便别人不拿原始文件也能核。
- **PR #10（HowseaonLink 的目标机环境与首批真实 rollout）：要求修改。** 正面项：`ids_sha256`/`config_sha256` 与 blob 哈希一致；健康检查 `alignment_checked=1624`、`mismatches=0`、`split_status.verified=true`；**冻结划分的 ID 与 PR #11 的候选划分逐字节相同**（说明 goal-map 重划没有改变成员）；GPU 钉版与 `requirements/gpu.txt` 一致。阻塞项两条：①`final_reward` 的语义（见下）；②README 写"65 秒"，而 JSONL 的 `seconds` 合计 52.4 秒。
- **PR #9（Jun040504 的数据阻塞说明）：要求修改。** 内容本身准确，但标题里的"缺少 `webshop_sft.json`"已被 Release `data-v1` 解决，直接合并会留下过期结论。

**真实日志带来的第一个重要修正：`final_reward` 不是任务奖励。** 三条真实 episode 全部 `max_steps`、购买数为 0，但 `mean_final_reward = 0.9167`（任务 75/104 的 `final_reward` 是 1.0）。查上游 `web_agent_text_env.py`：搜索页把 `self.reward` 设为 `search_max`（约 478 行）、商品页设为 `maxr`（约 558 行），只有 `done()` 才返回购买得分。所以非购买 episode 的 `final_reward` 是"最后停留页面的匹配分"。

这也解释了为什么不能直接用 `final_reward` 当终局回报：进度模型目标与 terminal 奖励都会学到"停在高分页面"而不是"买到对的东西"。已在评估侧修正（`results/metrics-definition.json` 升到 v1.1）：

- `evaluation.terminal_reward(row)`：购买结束记 `final_reward`，否则记 0；
- 汇总新增 `mean_terminal_reward`，登记表的 `mean_reward` 列改用它；
- 对真实日志重算：`mean_terminal_reward = 0.0`，`mean_final_reward = 0.9167`，购买 0、成功 0。

顺带修掉两个工具问题：①配置哈希在 Windows 检出（CRLF）下会误报"config 被改"，现在接受 LF 归一化后的匹配并给出警告；②我自己改这段代码时把 `expected`（期望 episode 列表）覆盖成了配置哈希，导致 `3 of 64` 的错误摘要——因为每次都用真实产物核数字才在提交前发现，已修复并加了断言。测试从 45 增到 **48 项**，全部通过。

## 补充二：11/19 的人工复核已完成

HowseaonLink 在 issue #5 同步了首批真实日志与 `goal_fingerprint`，阻塞解除。我把 3 条轨迹逐步读完并人工确认分类，产出 `results/failure-cases-b0-smoke.md`：

- task 16：连续 7 次 `click[sea change]`，页面分 0.75，未购买 → loop；
- task 75：8/10 步动作不可执行（`size:` 前缀 + `fl. oz` 与环境的 `fl oz` 不符），页面分从 1.0 掉到 0，但记录的 `final_reward` 仍是 1.0（非法步不覆盖 reward）→ 主标签 format、次要 loop；
- task 104：`click[white]` ×7，页面分已 1.0 但从未下单 → loop。

三条的共同点：`buy now` 每一步都在可用列表里却从未被点击——是策略问题，不是环境不可行。这也再次说明「页面分 1.0」不等于「任务完成」。

同时给 HowseaonLink 回了批量规格（B0：冻结 val 160 任务 × 1 repeat、greedy、`max_steps=10`、按 `val_ids` 顺序，预计 45–60 分钟），并明确这 3 条 smoke **不登记**到 experiment-register，等全量 B0 跑完再登记一行。

## 下一个学习问题

同一批 160 个 val 任务上，"每任务 1 条"的 Wilson 区间和"重复采样后"的任务簇 bootstrap 会差多少？报告里该报哪一个、怎么措辞才不夸大一个种子的差异？这直接关系到 12/03 两臂对比表的写法。
