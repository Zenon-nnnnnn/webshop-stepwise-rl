# 实验证据与结果登记

- `material-audit.json`：本地真实 SFT 与原包示例数据审计，不是模型效果。
- `provisional-split-validation.json`：离线分组划分检查，1451 train / 160 val / 200 reserved test，排除 13 条；仍需目标机完整环境 goal-map 核对。
- `dependency-availability.json`：官方 PyPI 元数据查验，仅证明候选包版本存在，不证明环境兼容。
- `validation.md`：本地检查记录；GitHub Actions 另在 Actions 页面查看。
- `metrics-definition.json`：指标定义 v1.1（机器可读），只定义口径，不含任何结果；`tests/test_analysis_tools.py` 会检查它与代码字段一致。
- `hand-check-metrics.md`：4 条合成 episode 的手算样例（成功率/平均奖励/非法率/区间），与两套代码复算结果逐项对照；`tests/test_hand_check.py` 把数字写成断言。
- `eval-frozen-20261120/`：开发 val 冻结件——160 个任务 ID（sha256 `45440d72…`）、goal fingerprint、划分哈希与两份冻结配置的哈希；不含 test、不含任何结果。
- `eval-20261119-full160/`：B0 与本地 LoRA 的 160 任务验证结果（19 vs 50 满分、配对差 +19.38pp），含独立核对 `verification.json`；**暂未登记**（采样 commit 未入仓库、执行配置与冻结配置不同），只作初步证据。
- `failure-cases-b0-smoke.md`：首批 3 条**真实** B0 轨迹（PR #10 的 `rollout-3val.jsonl`）的人工复核案例：2 条 loop、1 条 format+loop；不是结果，只是 11/19 要求的案例核对。
- `upstream-demo-data-check.json`：上游演示轨迹的可用性检查（78 条、803/1170 个回合不可解析、奖励只有 0/1），说明它只能校准解析、不能当失败案例或成功率。
- `experiment-register.csv`：真实训练/评估运行登记表；目前只有表头，没有预填性能。
- `summary.md`、`success-by-method.svg`：由 `webshop-lab results-table` / `figures` 从登记表生成，等真实 `complete` 行出现后才有内容。

登记表列与校验规则见 [统一评估](../docs/EVALUATION.md) 第 5 节：一行必须能追到真实 run、config（路径与哈希）、commit、划分 SHA、goal fingerprint 与 JSONL 证据路径；`webshop-lab check-register` 会在 CI 中检查表头与数值自洽。表头本身不是结果。

不要将测试中的合成轨迹、本地 fixture 生成的示例数字、原论文指标或上游演示数据填入正式实验结果。
