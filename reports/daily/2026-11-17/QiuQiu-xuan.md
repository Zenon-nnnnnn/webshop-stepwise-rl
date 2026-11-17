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

## 下一个学习问题

同一批 160 个 val 任务上，"每任务 1 条"的 Wilson 区间和"重复采样后"的任务簇 bootstrap 会差多少？报告里该报哪一个、怎么措辞才不夸大一个种子的差异？这直接关系到 12/03 两臂对比表的写法。
