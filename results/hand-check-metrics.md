# 手算样例：成功率、平均奖励、非法动作率

对应 [逐日计划](https://github.com/Zqqq-qq/webshop-stepwise-rl/blob/main/docs/DAILY_PLAN.md) 11/17 里评估的角色任务："手算成功率/平均奖励/非法率；交独立指标核对样例"。

目的：先用手算固定口径（分母是谁、购买与成功怎么分、非法率按步还是按 episode），再用两套彼此独立的代码复算。**样本是 4 条合成 episode，不是模型结果**，不能写进 proposal 或 report；真实日志到位后按同一流程重做。

- 输入：`tests/fixtures/hand-check-episodes.jsonl`（4 条，含 2 条购买结束、2 条步数耗尽）
- metadata：`tests/fixtures/hand-check.meta.json`（`selected_ids` 覆盖 4 个任务，所以完成度应为 complete）
- 定义：`results/metrics-definition.json`；实现：`src/webshop_lab/evaluation.py`

## 1. 逐条数据

| task | 终止 | 步数 | 非法步（grounding=0） | final_reward | 备注 |
|---|---|---|---|---|---|
| 101 | purchase | 2 | 0 | 1.0 | 尺寸对、价格在预算内 → 满奖励 |
| 102 | purchase | 2 | 0 | 0.6 | 买了同款但尺寸错 → 部分奖励 |
| 103 | max_steps | 3 | 1 | 0.0 | 有一步没有唯一可解析的 `Action:` |
| 104 | max_steps | 2 | 1 | 0.0 | 点击了当前页面不存在的按钮 |

## 2. 手算

```text
episodes                 = 4
purchase_terminated      = #{101, 102}                      = 2
full_reward_successes    = #{101 : done 且 reward >= 1-1e-6} = 1
purchase_without_full_reward = 2 - 1                        = 1
max_steps_terminated     = 4 - 2                            = 2

success_rate             = 1 / 4                            = 0.25
mean_final_reward        = (1.0 + 0.6 + 0.0 + 0.0) / 4 = 1.6 / 4 = 0.4
mean_steps               = (2 + 2 + 3 + 2) / 4 = 9 / 4      = 2.25
steps                    = 2 + 2 + 3 + 2                    = 9
invalid steps            = 1 (task 103) + 1 (task 104)      = 2
invalid_action_rate      = 2 / 9                            = 0.222222...
grounding_accuracy       = 1 - 2/9 = 7/9                    = 0.777778...
episodes_with_invalid    = 2  (103 与 104 各出现一次)
conditional_success_given_purchase = 1 / 2                  = 0.5
```

注意分母：`success_rate` 的分母是**全部 4 条 episode**，不是"买了的 2 条"；如果按购买数做分母就会得到 0.5，正是需要避免的高估。

## 3. 区间（Wilson，1 条/任务）

n=4、k=1、p̂=0.25、z=1.9599639845：

```text
z²            = 3.84145882
den           = 1 + z²/n                    = 1.96036471
center        = (p̂ + z²/(2n)) / den         = 0.37247271
half-width    = z/den * sqrt(p̂(1-p̂)/n + z²/(4n²)) = 0.32688545
95% interval  = 0.37247271 ± 0.32688545     = [0.04558726, 0.69935816]
```

4 条样本的区间宽到 0.65，说明"1/4 成功"这种小样本数字几乎不能区分好坏；这正是报告里必须同时给任务数与区间的原因。

## 4. 两套代码复算

```bash
# 第一套：evaluate（仓库主实现）
PYTHONPATH=src python -m webshop_lab.cli evaluate --method B0 \
  --meta tests/fixtures/hand-check.meta.json --jsonl tests/fixtures/hand-check-episodes.jsonl \
  --output artifacts/hand-check-evaluation.json

# 第二套：scripts/verify_metrics.py（不 import 本包的独立实现）
python scripts/verify_metrics.py --jsonl tests/fixtures/hand-check-episodes.jsonl \
  --evaluation artifacts/hand-check-evaluation.json
```

实测输出（2026-11-17，Windows / Python 3.12.10）：

```text
field                                    recomputed           recorded  verdict
episodes                                          4                  4  PASS
purchase_terminated                               2                  2  PASS
full_reward_successes                             1                  1  PASS
purchase_without_full_reward                      1                  1  PASS
max_steps_terminated                              2                  2  PASS
success_rate                                   0.25               0.25  PASS
mean_final_reward                               0.4                0.4  PASS
mean_steps                                     2.25               2.25  PASS
invalid_action_rate              0.2222222222222222 0.2222222222222222  PASS
unique_tasks                                      4                  4  PASS
All 10 fields agree within 1e-09.
```

手算、`evaluate`、`verify_metrics.py` 三者一致。`tests/test_hand_check.py` 把这些数字写成断言，口径被改动时测试会失败。

## 5. 失败分诊顺带核对

| 分类 | 数量 | 说明 |
|---|---|---|
| `specification` | 1 | task 102：买了同款但规格错，奖励 0.6 |
| `format` | 2 | task 103 无唯一可解析动作；task 104 动作不可执行 |
| `loop`（仅 `any_counts`） | 1 | task 103 重复了同一个 `search[red running shoes]`，作为次要信号记录 |

## 6. 局限

- 4 条合成样本只能验证算术与口径，不能说明任何模型行为；真实日志到手后要用同样的"手算 + 两套复算"流程重做一遍。
- 104 的"动作不可执行"在这里由 `error` 字段直接给出；真实 `rollout` 记录里该字段由环境响应填充，若日志缺少 `available` 列表，这类错误只能靠 `error` 文本判断。
- 区间用的是"每任务 1 条"的 Wilson；一旦同一任务重复采样，必须换成任务簇 bootstrap，不能继续用这个区间。
