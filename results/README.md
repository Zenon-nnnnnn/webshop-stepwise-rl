# 实验证据与结果登记

- `material-audit.json`：本地真实 SFT 与原包示例数据审计，不是模型效果。
- `provisional-split-validation.json`：离线分组划分检查，1451 train / 160 val / 200 reserved test，排除 13 条；仍需目标机完整环境 goal-map 核对。
- `dependency-availability.json`：官方 PyPI 元数据查验，仅证明候选包版本存在，不证明环境兼容。
- `validation.md`：本地检查记录；GitHub Actions 另在 Actions 页面查看。
- `experiment-register.csv`：真实训练/评估运行登记表；目前只有表头，没有预填性能。

不要将测试中的两条合成轨迹、原论文指标或上游演示数据填入正式实验结果。
