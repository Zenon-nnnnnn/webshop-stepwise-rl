# 测试 fixture：手算样例

`hand-check-episodes.jsonl` 是 4 条**合成** episode，用来把手算结果与代码对齐（见
`results/hand-check-metrics.md` 与 `tests/test_hand_check.py`）。它们不是模型输出，也不能填进
`results/experiment-register.csv`。

| task | 结果 | steps | 非法步 | final_reward | 分诊主标签 |
|---|---|---|---|---|---|
| 101 | 满奖励成功 | 2 | 0 | 1.0 | —（成功，不分类） |
| 102 | 购买但规格错 | 2 | 0 | 0.6 | specification |
| 103 | max_steps，含一个无法解析的动作 | 3 | 1 | 0.0 | format（次要：loop） |
| 104 | max_steps，含一个不可执行动作 | 2 | 1 | 0.0 | format |

`hand-check.meta.json` 是同名 metadata，字段结构与 `rollout` 写出的一致，`selected_ids` 覆盖
4 个任务，因此 `check-run` / `evaluate` 会判为 complete。
