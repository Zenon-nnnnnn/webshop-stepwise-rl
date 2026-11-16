# 验证记录

日期：2026-11-16。执行机器：macOS，Python 3.13.16；不是目标 RTX 5060 Ti 主机。

## 已运行

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/check_repository.py
python -m compileall -q src scripts
PYTHONPATH=src python -m webshop_lab.cli audit --sft /path/to/webshop_sft.json --output results/material-audit.json
PYTHONPATH=src python -m webshop_lab.cli split --sft /path/to/webshop_sft.json --output /temporary/split-validation
```

8 个测试通过：动作解析/可执行性、ZIP 路径穿越拒绝、历史保留、同 ID/指令/预算变体分组、奖励公式与非有限值拒绝、购买与成功区分、线性进度训练/标注/奖励导出端到端以及泄漏拒绝。

文档本地相对链接与 credential 模式检查通过。新代码语法编译通过。真实 SFT 离线划分：1451 train、160 val、200 reserved test，排除 13 条；ID 和 train/val 底层指令哈希交叉均为 0。完整环境的额外指令对齐仍待完成。

## 未运行

完整大包解压、Java/Lucene/WebShop 启动、5060 Ti CUDA kernel、量化加载、SFT 训练、真实 rollout、原神经 SPA、PPO。无真实成功率或性能提升结论。候选版本的兼容性以目标机 smoke 为准。

## GitHub Actions

首次完整提交 `e105f0e84fc47fd9f3454d56cd5a77933df75df6` 的 [CPU checks](https://github.com/Zqqq-qq/webshop-stepwise-rl/actions/runs/43848833958) 已通过：Ubuntu / Python 3.10 上安装项目、8 项测试、语法编译、链接/credential 检查和审计均成功。线上检查仍不包含 GPU 或真实模型训练。
